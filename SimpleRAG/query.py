
import os
import sys
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "legal_chunks"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

SEMANTIC_TOP_K = 15       
BM25_TOP_K = 15            
FINAL_TOP_K = 7            

if not GOOGLE_API_KEY:
    print("[LOI] Thieu GOOGLE_API_KEY trong .env")
    sys.exit(1)


def simple_tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho BM25: lowercase + split whitespace."""
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
                  " ", text.lower())
    return text.split()


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:

    #Reciprocal Rank Fusion (RRF) để merge nhiều ranked list.
    #Mỗi ranked_list = [(doc_id, score), ...]
    #Trả về list doc_id sorted desc theo RRF score.
 
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class HybridRetriever:
    #BM25 + Semantic retrieval từ ChromaDB.

    def __init__(self) -> None:
        print("[LOAD] Dang tai ChromaDB collection ...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

        # Lấy toàn bộ documents + ids
        result = self.collection.get(include=["documents", "metadatas"])
        self.all_ids: list[str] = result["ids"]
        self.all_docs: list[str] = result["documents"]
        self.all_metas: list[dict] = result["metadatas"]

        print(f"   [DATA] {len(self.all_docs)} chunks loaded")

        # Tạo BM25 index
        print("[BM25] Xay dung BM25 index ...")
        tokenized = [simple_tokenize(doc) for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized)

        # Load embedding model
        print(f"[MODEL] Dang tai model: {EMBEDDING_MODEL} ...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[OK] Retriever san sang!\n")

    def retrieve(self, query: str) -> list[dict]:
       # Hybrid retrieval: BM25 + semantic → RRF merge → top FINAL_TOP_K.

        # --- BM25 ---
        query_tokens = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[:BM25_TOP_K]
        bm25_list = [(self.all_ids[i], score) for i, score in bm25_ranked]

        # --- Semantic (ChromaDB) ---
        query_emb = self.embed_model.encode([query])[0].tolist()
        sem_result = self.collection.query(
            query_embeddings=[query_emb],
            n_results=SEMANTIC_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        sem_list = [
            (sem_result["ids"][0][j], 1.0 - sem_result["distances"][0][j])
            for j in range(len(sem_result["ids"][0]))
        ]

        # --- RRF merge ---
        fused = reciprocal_rank_fusion([bm25_list, sem_list])
        top_ids = [doc_id for doc_id, _ in fused[:FINAL_TOP_K]]

        # Lấy nội dung
        id_to_idx = {did: idx for idx, did in enumerate(self.all_ids)}
        results: list[dict] = []
        for doc_id in top_ids:
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                results.append({
                    "text": self.all_docs[idx],
                    "doc_name": self.all_metas[idx].get("doc_name", ""),
                    "dieu": self.all_metas[idx].get("dieu", ""),
                })

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật Việt Nam. 
Dựa vào các trích đoạn văn bản pháp luật được cung cấp bên dưới, hãy trả lời câu hỏi của người dùng.

QUY TẮC:
1. Chỉ trả lời dựa trên thông tin trong các trích đoạn. Nếu không đủ thông tin, hãy nói rõ.
2. Trích dẫn cụ thể Điều, Khoản, Điểm khi có thể.
3. Trả lời bằng tiếng Việt, rõ ràng và chính xác.
4. Nếu câu hỏi không liên quan đến pháp luật trong ngữ cảnh, hãy thông báo.
"""


def build_context(chunks: list[dict]) -> str:
    #Tạo context từ các chunk đã retrieve.
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"--- Trích đoạn {i} ---\n"
            f"Nguồn: {c['doc_name']}\n"
            f"Vị trí: {c['dieu']}\n"
            f"Nội dung:\n{c['text']}\n"
        )
    return "\n".join(parts)


def answer_question(query: str, context: str) -> str:
    #Gọi Gemini API để trả lời câu hỏi.
    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CÁC TRÍCH ĐOẠN VĂN BẢN PHÁP LUẬT ===\n{context}\n\n"
        f"=== CÂU HỎI ===\n{query}\n\n"
        f"=== TRẢ LỜI ==="
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    return response.text


def main() -> None:
    print("=" * 60)
    print("  SIMPLE LEGAL RAG - Hoi dap phap luat Viet Nam")
    print("=" * 60)

    retriever = HybridRetriever()

    print("Nhập câu hỏi (gõ 'exit' hoặc 'q' để thoát):\n")

    while True:
        try:
            query = input("Cau hoi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet!")
            break

        if not query or query.lower() in ("exit", "q", "quit"):
            print("Tam biet!")
            break

        # Retrieve
        print("[SEARCH] Dang tim kiem ...")
        chunks = retriever.retrieve(query)

        if not chunks:
            print("[WARN] Khong tim thay thong tin lien quan.\n")
            continue

        # Show retrieved chunks summary
        print(f"[RESULT] Tim thay {len(chunks)} trich doan lien quan:")
        for i, c in enumerate(chunks, 1):
            preview = c["text"][:80].replace("\n", " ")
            print(f"   {i}. [{c['dieu']}] {c['doc_name']}: {preview}...")

        # Generate answer
        context = build_context(chunks)
        print("\n[LLM] Dang tao cau tra loi ...\n")
        answer = answer_question(query, context)

        print("=" * 60)
        print("TRA LOI:")
        print("-" * 60)
        print(answer)
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
