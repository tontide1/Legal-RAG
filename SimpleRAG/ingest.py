import os
import re
import sys
import glob
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "legal_chunks"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
CHUNK_MAX_CHARS = 3000         
CHUNK_OVERLAP_CHARS = 200   

def collect_txt_files(root_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root_dir, "**", "*.txt"), recursive=True))


def _detect_doc_name(filepath: str) -> str:
    parts = Path(filepath).parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def chunk_by_dieu(text: str, doc_name: str) -> list[dict]:
    #Chia văn bản thành các chunk theo 'Điều <số>'.
    #Mỗi chunk chứa nội dung của 1 Điều.
    pattern = re.compile(r"(?:^|\n)(Điều\s+\d+[a-zđ]?\.?\s)", re.MULTILINE)
    matches = list(pattern.finditer(text))
    chunks: list[dict] = []
    if not matches:
        # Không tìm thấy Điều nào → chunk theo kích thước cố định
        for sub in _split_long(text):
            chunks.append({
                "text": sub.strip(),
                "doc_name": doc_name,
                "dieu": "N/A",
            })
        return chunks

    preamble = text[: matches[0].start()].strip()
    if preamble:
        for sub in _split_long(preamble):
            chunks.append({
                "text": sub.strip(),
                "doc_name": doc_name,
                "dieu": "Mở đầu",
            })
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()

        # Trích tên Điều
        dieu_match = re.match(r"(Điều\s+\d+[a-zđ]?)", section)
        dieu_label = dieu_match.group(1) if dieu_match else f"Điều ?({i})"

        for sub in _split_long(section):
            chunks.append({
                "text": sub.strip(),
                "doc_name": doc_name,
                "dieu": dieu_label,
            })

    return chunks


def _split_long(text: str) -> list[str]:
    #Chia text dài hơn CHUNK_MAX_CHARS thành các phần nhỏ với overlap.
    if len(text) <= CHUNK_MAX_CHARS:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_MAX_CHARS, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + (CHUNK_MAX_CHARS // 2), end)
            if nl > start:
                end = nl
        parts.append(text[start:end])
        if end >= len(text):
            break
        next_start = end - CHUNK_OVERLAP_CHARS
        start = max(next_start, start + 1)
    return parts


def chunk_id(doc_name: str, dieu: str, index: int) -> str:
    #Tạo ID ổn định cho chunk.
    raw = f"{doc_name}|{dieu}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()

def main() -> None:
    files = collect_txt_files(DATASET_DIR)
    if not files:
        print(f"[LOI] Khong tim thay file .txt trong {DATASET_DIR}")
        sys.exit(1)

    print(f"[INFO] Tim thay {len(files)} file .txt")
    for f in files:
        print(f"   - {_detect_doc_name(f)}")

    all_chunks: list[dict] = []
    for filepath in files:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        doc_name = _detect_doc_name(filepath)
        chunks = chunk_by_dieu(text, doc_name)
        all_chunks.extend(chunks)
        print(f"   [CHUNK] {doc_name}: {len(chunks)} chunk(s)")

    print(f"\n[TONG] {len(all_chunks)} chunks")

    all_chunks = [c for c in all_chunks if len(c["text"].strip()) > 20]
    print(f"   (sau lọc trống: {len(all_chunks)} chunks)")

    print(f"\n[MODEL] Dang tai model embedding: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print("[EMBED] Dang tao embedding ...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    print(f"[OK] Da tao {len(embeddings)} embedding (dim={embeddings.shape[1]})")

    print(f"\n[SAVE] Luu vao ChromaDB tai {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chuẩn bị batch
    ids: list[str] = []
    metadatas: list[dict] = []
    for i, c in enumerate(all_chunks):
        ids.append(chunk_id(c["doc_name"], c["dieu"], i))
        metadatas.append({
            "doc_name": c["doc_name"],
            "dieu": c["dieu"],
            "char_len": len(c["text"]),
        })

    BATCH = 500
    for start in range(0, len(all_chunks), BATCH):
        end = min(start + BATCH, len(all_chunks))
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"   [OK] Batch {start}-{end} upserted")

    print(f"\n[DONE] Hoan tat! Collection '{COLLECTION_NAME}' co {collection.count()} documents")


if __name__ == "__main__":
    main()
