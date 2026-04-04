import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_NER_BACKEND = "phobert"
DEFAULT_PHOBERT_CHECKPOINT = PROJECT_ROOT / "NER" / "checkpoints" / "phobert_article_ner"
DEFAULT_DEDUP_DATASET_PATH = PROJECT_ROOT / "NER" / "processed" / "phase1_dedup.json"
DEFAULT_BILSTM_MODEL_PATH = PROJECT_ROOT / "NER" / "bilstm_ner.pt"


def _ner_dataset_path() -> Path:
    dataset_path_from_env = os.getenv("NER_DATASET_PATH")
    if dataset_path_from_env:
        resolved_path = Path(dataset_path_from_env)
    else:
        resolved_path = DEFAULT_DEDUP_DATASET_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"NER dataset not found at {resolved_path}. "
            "Generate processed dataset first or set NER_DATASET_PATH to a valid processed file."
        )
    return resolved_path


_EXAMPLES = None
_TOKEN2ID = None
_ID2TOKEN = None

token2id = {}
id2token = {}
label2id = {"O": 0, "B-ARTICLE": 1, "I-ARTICLE": 2}
id2label = {v: k for k, v in label2id.items()}


def load_examples():
    global _EXAMPLES
    if _EXAMPLES is None:
        with open(_ner_dataset_path(), "r", encoding="utf-8") as file:
            _EXAMPLES = json.load(file)
    return _EXAMPLES


def get_token_mappings():
    global _TOKEN2ID
    global _ID2TOKEN
    global token2id
    global id2token

    if _TOKEN2ID is None or _ID2TOKEN is None:
        examples = load_examples()
        token_set = set()
        for ex in examples:
            token_set.update(ex["tokens"])

        _TOKEN2ID = {"<PAD>": 0, "<UNK>": 1}
        for token in sorted(token_set):
            _TOKEN2ID[token] = len(_TOKEN2ID)

        _ID2TOKEN = {v: k for k, v in _TOKEN2ID.items()}
        token2id = _TOKEN2ID
        id2token = _ID2TOKEN

    return _TOKEN2ID, _ID2TOKEN

class NERDataset(Dataset):
    def __init__(self, examples, token2id, label2id):
        self.examples = examples
        self.token2id = token2id
        self.label2id = label2id

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        labels = ex["labels"]
        token_ids = [self.token2id.get(t, self.token2id["<UNK>"]) for t in tokens]
        label_ids = [self.label2id[l] for l in labels]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

def pad_collate(batch):
    tokens, labels = zip(*batch)
    lengths = [len(t) for t in tokens]
    max_len = max(lengths)
    padded_tokens = []
    padded_labels = []
    for t, l in zip(tokens, labels):
        pad_len = max_len - len(t)
        padded_tokens.append(torch.cat([t, torch.zeros(pad_len, dtype=torch.long)]))
        # Sử dụng -100 cho padding label để ignore khi tính loss
        padded_labels.append(torch.cat([l, torch.full((pad_len,), -100, dtype=torch.long)]))
    return torch.stack(padded_tokens), torch.stack(padded_labels), lengths

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.1):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=True, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

embedding_dim = 100
hidden_dim = 128
num_layers = 2
num_labels = len(label2id)


def resolve_ner_backend(backend: str | None = None) -> str:
    selected_backend = backend or os.getenv("NER_BACKEND", DEFAULT_NER_BACKEND)
    normalized_backend = selected_backend.strip().lower()
    if normalized_backend in {"bilstm", "phobert"}:
        return normalized_backend
    return DEFAULT_NER_BACKEND


def _resolve_device(device=None):
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_phobert_module():
    try:
        from NER import phobert_ner
    except ImportError:
        import phobert_ner
    return phobert_ner


def load_model(
    model_path: str | Path | None = None,
    device=None,
    *,
    backend: str | None = None,
    checkpoint_dir: str | Path | None = None,
):
    selected_backend = resolve_ner_backend(backend)
    resolved_device = _resolve_device(device)

    if selected_backend == "phobert":
        phobert_ner = _load_phobert_module()
        resolved_checkpoint = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_PHOBERT_CHECKPOINT
        tokenizer, model, phobert_device = phobert_ner.load_model(
            checkpoint_dir=resolved_checkpoint,
            device=resolved_device,
        )
        return {
            "backend": "phobert",
            "tokenizer": tokenizer,
            "model": model,
            "device": phobert_device,
            "checkpoint_dir": str(resolved_checkpoint),
        }

    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_BILSTM_MODEL_PATH
    token2id_map, _ = get_token_mappings()
    model = BiLSTM_NER(
        vocab_size=len(token2id_map),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_labels=num_labels,
        dropout=0.1
    ).to(resolved_device)
    model.load_state_dict(torch.load(str(resolved_model_path), map_location=resolved_device))
    model.eval()
    return model


def predict(
    query,
    model,
    token2id_map=None,
    device=None,
    *,
    backend: str | None = None,
    max_length: int = 128,
):
    selected_backend = resolve_ner_backend(backend)

    if isinstance(model, dict) and model.get("backend") == "phobert":
        selected_backend = "phobert"

    if selected_backend == "phobert":
        phobert_ner = _load_phobert_module()
        if not isinstance(model, dict) or model.get("backend") != "phobert":
            raise ValueError("PhoBERT backend requires a model bundle returned by load_model(..., backend='phobert')")

        query_tokens = query.split()
        predicted_labels = phobert_ner.predict_labels(
            tokens=query_tokens,
            tokenizer=model["tokenizer"],
            model=model["model"],
            device=model["device"],
            max_length=max_length,
        )
        predictions = [label2id.get(label, label2id["O"]) for label in predicted_labels]
        return query_tokens, predictions

    if token2id_map is None:
        token2id_map, _ = get_token_mappings()
    resolved_device = _resolve_device(device)
    query_tokens = query.split()
    query_token_ids = [token2id_map.get(tok, token2id_map["<UNK>"]) for tok in query_tokens]
    query_tensor = torch.tensor(query_token_ids, dtype=torch.long).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        logits = model(query_tensor)  # (1, seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    return query_tokens, predictions


def extract_entities(tokens, predictions, id2label_map=None):
    if id2label_map is None:
        id2label_map = id2label

    entities = []
    current_entity = []
    for token, pred in zip(tokens, predictions):
        if isinstance(pred, str):
            label = pred
        else:
            label = id2label_map[pred]
        if label == "B-ARTICLE":
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)
        elif label == "I-ARTICLE":
            if current_entity:
                current_entity.append(token)
            else:
                current_entity.append(token)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))
    return entities


def infer(
    query,
    model_path: str | Path | None = None,
    device=None,
    *,
    backend: str | None = None,
    checkpoint_dir: str | Path | None = None,
    max_length: int = 128,
):
    selected_backend = resolve_ner_backend(backend)
    model = load_model(
        model_path,
        device,
        backend=selected_backend,
        checkpoint_dir=checkpoint_dir,
    )
    tokens, predictions = predict(
        query,
        model,
        device=device,
        backend=selected_backend,
        max_length=max_length,
    )
    entities = extract_entities(tokens, predictions, id2label)
    return tokens, predictions, entities

if __name__ == '__main__':
    examples = load_examples()
    token2id_map, _ = get_token_mappings()

    full_dataset = NERDataset(examples, token2id, label2id)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị sử dụng:", device)

    learning_rate = 0.0001
    num_epochs = 10

    model = BiLSTM_NER(len(token2id_map), embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_tokens, batch_labels, lengths in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_tokens)
            logits = logits.view(-1, num_labels)
            batch_labels = batch_labels.view(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    model_save_path = str(PROJECT_ROOT / "NER" / "bilstm_ner.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model đã được lưu tại: {model_save_path}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_tokens, batch_labels, lengths in test_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_tokens)
            logits = logits.view(-1, num_labels)
            batch_labels = batch_labels.view(-1)
            loss = criterion(logits, batch_labels)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"\n[Test] Loss trung bình: {avg_test_loss:.4f}")

__all__ = [
    "token2id", "id2token", "label2id", "id2label",
    "embedding_dim", "hidden_dim", "num_layers", "num_labels",
    "DEFAULT_NER_BACKEND", "DEFAULT_PHOBERT_CHECKPOINT",
    "DEFAULT_DEDUP_DATASET_PATH", "DEFAULT_BILSTM_MODEL_PATH", "resolve_ner_backend",
    "BiLSTM_NER", "load_model", "predict", "extract_entities", "infer"
]
