from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ["O", "B-ARTICLE", "I-ARTICLE"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

_CACHE: dict[str, object] = {
    "key": None,
    "tokenizer": None,
    "model": None,
    "device": None,
}


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _token_to_subtoken_ids(tokenizer, token: str) -> list[int]:
    subtoken_ids = tokenizer.encode(token, add_special_tokens=False)
    if subtoken_ids:
        return subtoken_ids
    if tokenizer.unk_token_id is None:
        raise ValueError("Tokenizer produced empty subtoken ids and has no unk_token_id fallback")
    return [tokenizer.unk_token_id]


def _encode_tokens_for_inference(*, tokens: list[str], tokenizer, max_length: int) -> dict:
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


def load_model(*, checkpoint_dir: str | Path, device: str | torch.device = "auto"):
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PhoBERT checkpoint does not exist: {checkpoint_path}")

    resolved_device = _resolve_device(device)
    cache_key = f"{checkpoint_path.resolve()}::{resolved_device}"
    if _CACHE["key"] == cache_key:
        return _CACHE["tokenizer"], _CACHE["model"], _CACHE["device"]

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_path))
    model.to(resolved_device)
    model.eval()

    _CACHE["key"] = cache_key
    _CACHE["tokenizer"] = tokenizer
    _CACHE["model"] = model
    _CACHE["device"] = resolved_device
    return tokenizer, model, resolved_device


def predict_labels(
    *,
    tokens: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 128,
) -> list[str]:
    encoded = _encode_tokens_for_inference(tokens=tokens, tokenizer=tokenizer, max_length=max_length)
    padded_batch = tokenizer.pad(
        [
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        ],
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {name: tensor.to(device) for name, tensor in padded_batch.items()}

    with torch.no_grad():
        logits = model(**model_inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()[0]

    token_predictions: list[str] = []
    for token_start_position in encoded["token_start_positions"]:
        label_id = int(predicted_ids[token_start_position])
        token_predictions.append(ID2LABEL.get(label_id, "O"))

    token_count = encoded["token_count"]
    if len(token_predictions) < token_count:
        token_predictions.extend(["O"] * (token_count - len(token_predictions)))
    elif len(token_predictions) > token_count:
        token_predictions = token_predictions[:token_count]

    return token_predictions


def extract_entities(tokens: list[str], labels: list[str]) -> list[str]:
    entities: list[str] = []
    current_entity: list[str] = []

    for token, label in zip(tokens, labels):
        if label == "B-ARTICLE":
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)
            continue
        if label == "I-ARTICLE":
            if not current_entity:
                current_entity.append(token)
            else:
                current_entity.append(token)
            continue
        if current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))
    return entities


def infer(
    query: str,
    *,
    checkpoint_dir: str | Path = "src/NER/checkpoints/phobert_article_ner",
    max_length: int = 128,
    device: str | torch.device = "auto",
) -> tuple[list[str], list[str], list[str]]:
    tokens = query.split()
    tokenizer, model, resolved_device = load_model(checkpoint_dir=checkpoint_dir, device=device)
    labels = predict_labels(
        tokens=tokens,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        max_length=max_length,
    )
    entities = extract_entities(tokens, labels)
    return tokens, labels, entities


__all__ = [
    "LABELS",
    "LABEL2ID",
    "ID2LABEL",
    "extract_entities",
    "infer",
    "load_model",
    "predict_labels",
]
