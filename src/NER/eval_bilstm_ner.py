from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from NER import ner


LABELS = ["O", "B-ARTICLE", "I-ARTICLE"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
ARTICLE_REFERENCE_PATTERN = re.compile(r"(?i)\bđiều\s+(?:\d+|[ivxlcdm]+)\b")
LEGAL_MARKERS = {"luật", "nghị", "định", "bộ"}


def load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8")) # Tiếng Việt
    if not isinstance(payload, list):
        raise ValueError(f"Dataset {path} must be a JSON list")
    for index, sample in enumerate(payload):
        if "tokens" not in sample or "labels" not in sample:
            raise ValueError(f"Sample #{index} in {path} is missing tokens/labels")
        tokens = sample["tokens"]
        labels = sample["labels"]
        if len(tokens) != len(labels):
            raise ValueError(f"Sample #{index} in {path} has mismatched token and label lengths")
        unknown_labels = [label for label in labels if label not in LABEL2ID]
        if unknown_labels:
            raise ValueError(f"Sample #{index} in {path} has unknown labels: {sorted(set(unknown_labels))}")
    return payload


def count_entities(labels: list[str]) -> int:
    count = 0
    in_entity = False
    for label in labels:
        if label == "B-ARTICLE":
            count += 1
            in_entity = True
            continue
        if label == "I-ARTICLE":
            if not in_entity:
                count += 1
                in_entity = True
            continue
        in_entity = False
    return count


def extract_entities_from_labels(labels: list[str]) -> set[tuple[int, int, str]]:
    entities: set[tuple[int, int, str]] = set()
    start = None
    entity_type = None

    for index, label in enumerate(labels):
        if label.startswith("B-"):
            if start is not None and entity_type is not None:
                entities.add((start, index, entity_type))
            start = index
            entity_type = label[2:]
            continue

        if label.startswith("I-"):
            current_type = label[2:]
            if start is None or entity_type != current_type:
                if start is not None and entity_type is not None:
                    entities.add((start, index, entity_type))
                start = index
                entity_type = current_type
            continue

        if start is not None and entity_type is not None:
            entities.add((start, index, entity_type))
        start = None
        entity_type = None

    if start is not None and entity_type is not None:
        entities.add((start, len(labels), entity_type))

    return entities


def compute_entity_and_token_metrics(
    predicted_label_sequences: list[list[str]],
    gold_label_sequences: list[list[str]],
) -> dict[str, float]:
    token_correct = 0
    token_total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for predicted_labels, gold_labels in zip(predicted_label_sequences, gold_label_sequences):
        compare_length = min(len(predicted_labels), len(gold_labels))
        for index in range(compare_length):
            token_total += 1
            if predicted_labels[index] == gold_labels[index]:
                token_correct += 1

        pred_entities = extract_entities_from_labels(predicted_labels)
        gold_entities = extract_entities_from_labels(gold_labels)
        true_positive += len(pred_entities & gold_entities)
        false_positive += len(pred_entities - gold_entities)
        false_negative += len(gold_entities - pred_entities)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    token_accuracy = token_correct / token_total if token_total else 0.0
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "token_accuracy": token_accuracy,
    }


def compute_false_positive_rate(
    predicted_label_sequences: list[list[str]],
    gold_label_sequences: list[list[str]],
) -> float:
    negative_indices = [
        index for index, labels in enumerate(gold_label_sequences) if count_entities(labels) == 0
    ]
    if not negative_indices:
        return 0.0

    false_positive_cases = 0
    for index in negative_indices:
        if count_entities(predicted_label_sequences[index]) > 0:
            false_positive_cases += 1
    return false_positive_cases / len(negative_indices)


def has_article_reference(tokens: list[str]) -> bool:
    text = " ".join(tokens)
    return bool(ARTICLE_REFERENCE_PATTERN.search(text))


def has_legal_marker(tokens: list[str]) -> bool:
    token_set = {token.lower() for token in tokens}
    return bool(token_set & LEGAL_MARKERS)


def build_slice_indices(samples: list[dict]) -> dict[str, list[int]]:
    slices: dict[str, list[int]] = {
        "single_article": [],
        "multi_article": [],
        "negative_with_dieu": [],
        "legal_generic_no_article": [],
    }

    for index, sample in enumerate(samples):
        tokens = sample["tokens"]
        labels = sample["labels"]
        entity_count = count_entities(labels)
        if entity_count == 1:
            slices["single_article"].append(index)
        elif entity_count >= 2:
            slices["multi_article"].append(index)

        if entity_count == 0:
            if any(token.lower() == "điều" for token in tokens):
                slices["negative_with_dieu"].append(index)
            if has_legal_marker(tokens) and not has_article_reference(tokens):
                slices["legal_generic_no_article"].append(index)

    return slices


def subset_by_indices(values: list, indices: list[int]) -> list:
    return [values[index] for index in indices]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def predict_label_sequences(
    *,
    samples: list[dict],
    model,
    token2id_map: dict[str, int],
    device: torch.device,
) -> list[list[str]]:
    predictions: list[list[str]] = []
    unk_id = token2id_map["<UNK>"]

    model.eval()
    with torch.no_grad():
        for sample in samples:
            tokens = sample["tokens"]
            token_ids = [token2id_map.get(token, unk_id) for token in tokens]
            inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(inputs)
            predicted_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            predictions.append([ID2LABEL[int(label_id)] for label_id in predicted_ids])

    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM NER baseline on Phase 1 test split.")
    parser.add_argument("--test", default="src/NER/processed/phase1_test.json")
    parser.add_argument("--model-path", default="src/NER/bilstm_ner.pt")
    parser.add_argument("--output", default="src/NER/reports/bilstm_phase1_eval.json")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test)
    model_path = Path(args.model_path)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"BiLSTM checkpoint not found: {model_path}")

    samples = load_samples(test_path)
    if args.max_test_samples > 0:
        samples = samples[: args.max_test_samples]
    if not samples:
        raise ValueError("Test dataset is empty")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    started_at = time.perf_counter()
    token2id_map, _ = ner.get_token_mappings()
    model = ner.load_model(model_path=str(model_path), device=device)

    gold_label_sequences = [sample["labels"] for sample in samples]
    predicted_label_sequences = predict_label_sequences(
        samples=samples,
        model=model,
        token2id_map=token2id_map,
        device=device,
    )

    overall_metrics = compute_entity_and_token_metrics(predicted_label_sequences, gold_label_sequences)
    overall_false_positive_rate = compute_false_positive_rate(predicted_label_sequences, gold_label_sequences)

    slice_indices = build_slice_indices(samples)
    slice_metrics = {}
    for slice_name, indices in slice_indices.items():
        slice_gold = subset_by_indices(gold_label_sequences, indices)
        slice_pred = subset_by_indices(predicted_label_sequences, indices)
        metrics = compute_entity_and_token_metrics(slice_pred, slice_gold)
        metrics["sample_count"] = len(indices)
        metrics["false_positive_rate"] = compute_false_positive_rate(slice_pred, slice_gold)
        slice_metrics[slice_name] = metrics

    total_seconds = time.perf_counter() - started_at
    report = {
        "config": {
            "model_path": str(model_path),
            "device": str(device),
            "labels": LABELS,
        },
        "dataset": {
            "test_path": str(test_path),
            "test_samples": len(samples),
        },
        "overall": {
            **overall_metrics,
            "false_positive_rate": overall_false_positive_rate,
        },
        "slices": slice_metrics,
        "runtime": {
            "total_seconds": round(total_seconds, 4),
            "samples_per_second": round(len(samples) / total_seconds, 4) if total_seconds else 0.0,
        },
    }
    write_json(output_path, report)

    print(f"Evaluation completed for {len(samples)} samples")
    print(f"Overall entity_f1: {overall_metrics['entity_f1']:.4f}")
    print(f"Overall false_positive_rate: {overall_false_positive_rate:.4f}")
    print(f"Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
