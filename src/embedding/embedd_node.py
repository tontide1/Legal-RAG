import numpy as np
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline_utils import build_text_payload

_MODEL = None


def get_embedding_model(device):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("keepitreal/vietnamese-sbert")
    _MODEL.to(device)
    return _MODEL


def delete_old_embeddings(session):
    query = """
    MATCH (n:LegalRAG)
    REMOVE n.embedding
    """
    session.run(query)
    print("Đã xóa tất cả các embedding trong db")

def get_entities_from_neo4j(session):
    query = """
    MATCH (n:LegalRAG)
    RETURN n.node_id AS node_id, n.ten AS ten, coalesce(n.Value, '') AS value
    """
    result = session.run(query)
    return [(record["node_id"], record["ten"], record["value"]) for record in result]

def create_embeddings(texts, device):
    model = get_embedding_model(device)
    embeddings = model.encode(texts, device=device)
    return embeddings

def save_embeddings_to_neo4j(session, entities, embeddings):
    for (node_id, _, _), embedding in zip(entities, embeddings):
        query = """
        MATCH (n:LegalRAG {node_id: $node_id})
        SET n.embedding = $embedding
        """
        session.run(query, node_id=node_id, embedding=embedding.tolist())
    print("Đã lưu embedding vào db")


__all__ = [
    "build_text_payload",
    "create_embeddings",
    "delete_old_embeddings",
    "get_entities_from_neo4j",
    "save_embeddings_to_neo4j",
]
