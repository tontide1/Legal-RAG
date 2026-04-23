from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ["O", "B-ARTICLE", "I-ARTICLE"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
ARTICLE_REFERENCE_PATTERN = re.compile(r"(?i)\bđiều\s+(?:\d+|[ivxlcdm]+)\b")
LEGAL_MARKERS = {"luật", "nghị", "định", "bộ"}


def load_samples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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


def has_article_reference(tokens: list[str]) -> bool:
    text = " ".join(tokens)
    return bool(ARTICLE_REFERENCE_PATTERN.search(text))


def has_legal_marker(tokens: list[str]) -> bool:
    token_set = {token.lower() for token in tokens}
    return bool(token_set & LEGAL_MARKERS)


def _token_to_subtoken_ids(tokenizer, token: str) -> list[int]:
    subtoken_ids = tokenizer.encode(token, add_special_tokens=False)
    if subtoken_ids:
        return subtoken_ids
    if tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer produced empty subtoken ids and has no unk_token_id fallback")
    return [tokenizer.unk_token_id]


def encode_tokens_for_manual_inference(*, tokens: list[str], tokenizer, max_length: int) -> dict:
    token_piece_ids: list[int] = []
    token_start_piece_positions: list[int] = []
    for token in tokens:
        token_start_piece_positions.append(len(token_piece_ids))
        token_piece_ids.extend(_token_to_subtoken_ids(tokenizer, token))

    input_ids = tokenizer.build_inputs_with_special_tokens(token_piece_ids)
    special_tokens_mask = tokenizer.get_special_tokens_mask(token_piece_ids, already_has_special_tokens=False)
    if len(input_ids) != len(special_tokens_mask):
        raise ValueError("special token mask length mismatch")

    sequence_position_for_piece_position: dict[int, int] = {}
    piece_cursor = 0
    for sequence_position, is_special in enumerate(special_tokens_mask):
        if is_special:
            continue
        sequence_position_for_piece_position[piece_cursor] = sequence_position
        piece_cursor += 1

    token_start_sequence_positions: list[int] = []
    for start_piece_position in token_start_piece_positions:
        sequence_position = sequence_position_for_piece_position.get(start_piece_position)
        if sequence_position is None:
            continue
        token_start_sequence_positions.append(sequence_position)

    attention_mask = [1] * len(input_ids)
    if max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_start_sequence_positions = [
            position for position in token_start_sequence_positions if position < max_length
        ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_start_positions": token_start_sequence_positions,
        "token_count": len(tokens),
    }


def predict_label_sequences(
    *,
    model,
    tokenizer,
    samples: list[dict],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[list[str]]:
    model.eval()
    predictions: list[list[str]] = []

    encoded_samples = [
        encode_tokens_for_manual_inference(
            tokens=sample["tokens"],
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for sample in samples
    ]

    for start_index in range(0, len(samples), batch_size):
        batch_features = encoded_samples[start_index : start_index + batch_size]
        batch_for_padding = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in batch_features
        ]
        padded_batch = tokenizer.pad(
            batch_for_padding,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = {name: tensor.to(device) for name, tensor in padded_batch.items()}

        with torch.no_grad():
            logits = model(**model_inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        for batch_index, feature in enumerate(batch_features):
            token_predictions: list[str] = []
            for token_start_position in feature["token_start_positions"]:
                label_id = int(predicted_ids[batch_index][token_start_position])
                token_predictions.append(ID2LABEL[label_id])

            token_count = feature["token_count"]
            if len(token_predictions) < token_count:
                token_predictions.extend(["O"] * (token_count - len(token_predictions)))
            elif len(token_predictions) > token_count:
                token_predictions = token_predictions[:token_count]

            predictions.append(token_predictions)

    return predictions


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 PhoBERT NER model.")
    parser.add_argument("--test", default="src/NER/processed/phase1_test.json")
    parser.add_argument("--checkpoint-dir", default="src/NER/checkpoints/phobert_article_ner")
    parser.add_argument("--output", default="src/NER/reports/phobert_phase1_eval.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_path = Path(args.test)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}. "
            "Run training first or provide a valid local path."
        )

    samples = load_samples(test_path)
    if args.max_test_samples > 0:
        samples = samples[: args.max_test_samples]
    if not samples:
        raise ValueError("Test dataset is empty")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)

    gold_label_sequences = [sample["labels"] for sample in samples]
    predicted_label_sequences = predict_label_sequences(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
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

    report = {
        "config": {
            "checkpoint_dir": str(checkpoint_dir),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
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
    }
    write_json(output_path, report)

    print(f"Evaluation completed for {len(samples)} samples")
    print(f"Overall entity_f1: {overall_metrics['entity_f1']:.4f}")
    print(f"Overall false_positive_rate: {overall_false_positive_rate:.4f}")
    print(f"Evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
