from __future__ import annotations

import argparse
import inspect
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


LABELS = ["O", "B-ARTICLE", "I-ARTICLE"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


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


class TokenClassificationDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples = samples
        self.features: list[dict[str, list[int]]] = []

        for sample in samples:
            tokens = sample["tokens"]
            labels = sample["labels"]
            feature = encode_tokens_with_manual_alignment(
                tokens=tokens,
                labels=labels,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        feature = self.features[index]
        return {name: torch.tensor(values, dtype=torch.long) for name, values in feature.items()}


def _token_to_subtoken_ids(tokenizer, token: str) -> list[int]:
    subtoken_ids = tokenizer.encode(token, add_special_tokens=False)
    if subtoken_ids:
        return subtoken_ids
    if tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer produced empty subtoken ids and has no unk_token_id fallback")
    return [tokenizer.unk_token_id]


def encode_tokens_with_manual_alignment(
    *,
    tokens: list[str],
    labels: list[str],
    tokenizer,
    max_length: int,
) -> dict[str, list[int]]:
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have identical length")

    token_piece_ids: list[int] = []
    token_piece_labels: list[int] = []
    for token, label in zip(tokens, labels):
        subtoken_ids = _token_to_subtoken_ids(tokenizer, token)
        token_piece_ids.extend(subtoken_ids)
        token_piece_labels.append(LABEL2ID[label])
        if len(subtoken_ids) > 1:
            token_piece_labels.extend([-100] * (len(subtoken_ids) - 1))

    input_ids = tokenizer.build_inputs_with_special_tokens(token_piece_ids)
    special_tokens_mask = tokenizer.get_special_tokens_mask(token_piece_ids, already_has_special_tokens=False)
    if len(input_ids) != len(special_tokens_mask):
        raise ValueError("special token mask length mismatch")

    aligned_labels: list[int] = []
    piece_cursor = 0
    for is_special in special_tokens_mask:
        if is_special:
            aligned_labels.append(-100)
            continue
        if piece_cursor >= len(token_piece_labels):
            raise ValueError("Label alignment cursor exceeded token piece labels")
        aligned_labels.append(token_piece_labels[piece_cursor])
        piece_cursor += 1

    if piece_cursor != len(token_piece_labels):
        raise ValueError("Label alignment cursor did not consume all token piece labels")

    attention_mask = [1] * len(input_ids)
    if max_length > 0 and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        aligned_labels = aligned_labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": aligned_labels,
    }


def decode_eval_predictions(eval_predictions) -> tuple[list[list[str]], list[list[str]]]:
    logits, label_ids = eval_predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_ids = np.argmax(logits, axis=-1)

    predicted_label_sequences: list[list[str]] = []
    gold_label_sequences: list[list[str]] = []
    for prediction_row, label_row in zip(predicted_ids, label_ids):
        predicted_labels = []
        gold_labels = []
        for predicted_label_id, gold_label_id in zip(prediction_row, label_row):
            if int(gold_label_id) == -100:
                continue
            predicted_labels.append(ID2LABEL[int(predicted_label_id)])
            gold_labels.append(ID2LABEL[int(gold_label_id)])
        predicted_label_sequences.append(predicted_labels)
        gold_label_sequences.append(gold_labels)

    return predicted_label_sequences, gold_label_sequences


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_training_arguments(*, args, output_dir: Path, with_eval: bool) -> TrainingArguments:
    init_parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": with_eval,
        "save_strategy": "epoch",
        "load_best_model_at_end": with_eval,
        "metric_for_best_model": "entity_f1" if with_eval else None,
        "greater_is_better": True,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": [],
    }

    if "evaluation_strategy" in init_parameters:
        training_kwargs["evaluation_strategy"] = "epoch" if with_eval else "no"
    elif "eval_strategy" in init_parameters:
        training_kwargs["eval_strategy"] = "epoch" if with_eval else "no"

    if "overwrite_output_dir" in init_parameters:
        training_kwargs["overwrite_output_dir"] = True

    filtered_kwargs = {key: value for key, value in training_kwargs.items() if key in init_parameters}
    return TrainingArguments(**filtered_kwargs)


def ensure_accelerate_available(min_version: str = "1.1.0") -> None:
    try:
        installed_version = metadata.version("accelerate")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Missing dependency 'accelerate'. Install it in your RAG env with: "
            "python3 -m pip install 'accelerate>=1.1.0'"
        ) from exc

    def to_parts(version: str) -> tuple[int, ...]:
        cleaned = version.split("+")[0]
        numeric = []
        for part in cleaned.split("."):
            digits = "".join(ch for ch in part if ch.isdigit())
            numeric.append(int(digits) if digits else 0)
        return tuple(numeric)

    if to_parts(installed_version) < to_parts(min_version):
        raise ImportError(
            f"accelerate>={min_version} is required, found {installed_version}. "
            "Upgrade with: python3 -m pip install -U 'accelerate>=1.1.0'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhoBERT token-classification model for Phase 1.")
    parser.add_argument("--train", default="src/NER/processed/phase1_train.json")
    parser.add_argument("--val", default="src/NER/processed/phase1_val.json")
    parser.add_argument("--output-dir", default="src/NER/checkpoints/phobert_article_ner")
    parser.add_argument("--report-output", default="src/NER/reports/phobert_phase1_train_report.json")
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=float, default=4.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_accelerate_available()

    train_path = Path(args.train)
    val_path = Path(args.val)
    output_dir = Path(args.output_dir)
    report_output_path = Path(args.report_output)

    train_samples = load_samples(train_path)
    val_samples = load_samples(val_path)

    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]

    if not train_samples:
        raise ValueError("Training dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = TokenClassificationDataset(train_samples, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = TokenClassificationDataset(val_samples, tokenizer=tokenizer, max_length=args.max_length)

    with_eval = len(val_dataset) > 0
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    training_args = build_training_arguments(args=args, output_dir=output_dir, with_eval=with_eval)

    def compute_metrics(eval_predictions) -> dict[str, float]:
        predicted_sequences, gold_sequences = decode_eval_predictions(eval_predictions)
        return compute_entity_and_token_metrics(predicted_sequences, gold_sequences)

    trainer_init_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset if with_eval else None,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if with_eval else None,
    }
    if "processing_class" in trainer_init_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate() if with_eval else {}

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    report = {
        "config": {
            "model_name": args.model_name,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "labels": LABELS,
        },
        "dataset": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        },
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "output_dir": str(output_dir),
    }
    write_json(report_output_path, report)

    print(f"Training completed. Model saved to: {output_dir}")
    print(f"Training report saved to: {report_output_path}")


if __name__ == "__main__":
    main()
