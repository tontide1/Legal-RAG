from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from pipeline_utils import get_configured_gemini_model

load_dotenv()

DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_TOP_K = 5
DEFAULT_ANSWER_TEMPLATE = (
    "Tôi không tìm thấy căn cứ pháp lý phù hợp cho câu hỏi này."
)


def load_documents(root_dir: Path) -> list[dict]:
    root_dir = root_dir.expanduser().resolve()
    documents: list[dict] = []
    if not root_dir.exists():
        return documents

    for path in sorted(root_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not text:
            continue
        documents.append(
            {
                "doc_id": str(path.relative_to(root_dir)),
                "title": path.stem,
                "source": str(path),
                "text": text,
            }
        )
    return documents


def normalize_scores(scores: Iterable[float]) -> list[float]:
    scores_list = list(scores)
    if not scores_list:
        return []
    minimum = min(scores_list)
    maximum = max(scores_list)
    if abs(maximum - minimum) < 1e-9:
        return [1.0] * len(scores_list)
    return [(score - minimum) / (maximum - minimum) for score in scores_list]


def build_context_for_documents(documents: list[dict]) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        snippet = doc["text"][:900].strip()
        if len(doc["text"]) > 900:
            snippet = f"{snippet}..."
        lines.append(
            f"Nguồn {idx} [{doc['doc_id']}]: {doc['title']}\n{snippet}"
        )
    return "\n\n".join(lines)


def create_answer_chain() -> ChatGoogleGenerativeAI:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. Please set it in .env file"
        )

    gemini_model = get_configured_gemini_model()
    answer_template = PromptTemplate.from_template(
        """Hãy trở thành chuyên gia tư vấn luật tại Việt Nam.
Câu hỏi của người dùng: {query}
Trả lời dựa vào các trích dẫn sau:
{context}
Yêu cầu:
1. Trả lời ngắn gọn, rõ ràng.
2. Nếu không có thông tin phù hợp, trả lời: \"Tôi không tìm thấy căn cứ pháp lý phù hợp cho câu hỏi này.\".
3. Kèm theo nguồn tham khảo và chú thích điều luật nếu có."""
    )
    return answer_template | ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=google_api_key,
        temperature=0.3,
        max_output_tokens=500,
    )


class TraditionalRAG:
    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        top_k: int = DEFAULT_TOP_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        embedding_model_name: str = "keepitreal/vietnamese-sbert",
        device: str | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.embedding_model_name = embedding_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.documents = load_documents(self.dataset_root)

        if not self.documents:
            raise ValueError(f"Không tìm thấy tài liệu văn bản trong '{self.dataset_root}'.")

        self.texts = [doc["text"] for doc in self.documents]
        self.tokenized_texts = [text.split() for text in self.texts]
        self.bm25_model = BM25Okapi(self.tokenized_texts)
        self.embed_model = SentenceTransformer(self.embedding_model_name)
        self.embed_model.to(self.device)
        self.document_embeddings = self.embed_model.encode(
            self.texts,
            device=self.device,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.document_norms = np.linalg.norm(self.document_embeddings, axis=1)

    def _query_embedding(self, query: str) -> np.ndarray:
        return self.embed_model.encode([query], device=self.device, convert_to_numpy=True)[0]

    def retrieve(self, query: str) -> list[dict]:
        tokens = query.split()
        bm25_scores = self.bm25_model.get_scores(tokens)
        query_embedding = self._query_embedding(query)

        cosine_scores = []
        query_norm = np.linalg.norm(query_embedding)
        for doc_emb, doc_norm in zip(self.document_embeddings, self.document_norms):
            denom = doc_norm * query_norm
            cosine_scores.append(
                float(np.dot(doc_emb, query_embedding) / (denom + 1e-10)) if denom > 0 else 0.0
            )

        bm25_norm = normalize_scores(bm25_scores)
        cosine_norm = normalize_scores([(score + 1.0) / 2.0 for score in cosine_scores])

        results = []
        for idx, document in enumerate(self.documents):
            combined_score = (
                self.bm25_weight * bm25_norm[idx]
                + (1.0 - self.bm25_weight) * cosine_norm[idx]
            )
            results.append(
                {
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "source": document["source"],
                    "text": document["text"],
                    "bm25_score": bm25_norm[idx],
                    "cosine_score": cosine_norm[idx],
                    "combined_score": combined_score,
                }
            )

        results.sort(key=lambda item: item["combined_score"], reverse=True)
        return results[: self.top_k]

    def run(self, query: str) -> dict:
        retrieved_documents = self.retrieve(query)
        context = build_context_for_documents(retrieved_documents)
        answer_text = DEFAULT_ANSWER_TEMPLATE

        if retrieved_documents:
            answer_chain = create_answer_chain()
            response = answer_chain.invoke({"query": query, "context": context})
            answer_text = response.content if hasattr(response, "content") else str(response)

        return {
            "query": query,
            "retrieved_documents": retrieved_documents,
            "context": context,
            "answer": answer_text,
            "scores": {
                "retrieved_count": len(retrieved_documents),
                "top_score": retrieved_documents[0]["combined_score"] if retrieved_documents else None,
            },
        }
