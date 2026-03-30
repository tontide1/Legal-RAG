import unittest

from src.legal_qa import (
    DEFAULT_ABSTAIN_ANSWER,
    extract_citations_from_nodes,
    run_legal_qa,
)


class LegalQAPipelineTest(unittest.TestCase):
    def test_extract_citations_from_nodes_uses_retrieved_metadata(self) -> None:
        citations = extract_citations_from_nodes(
            [
                {
                    "node_id": "Dieu::1",
                    "label": "Điều",
                    "name": "Điều 1: Phạm vi điều chỉnh",
                    "value": "Nội dung điều 1",
                    "final_score": 1.7,
                }
            ]
        )

        self.assertEqual(
            citations,
            [
                {
                    "node_id": "Dieu::1",
                    "label": "Điều",
                    "name": "Điều 1: Phạm vi điều chỉnh",
                    "score": 1.7,
                }
            ],
        )

    def test_run_legal_qa_returns_abstain_when_retrieval_is_empty(self) -> None:
        result = run_legal_qa(
            "Mức phạt là gì?",
            ner_infer_fn=lambda query: ["Điều 1"],
            retrieve_fn=lambda query, ner_entities: [],
            generate_answer_fn=lambda query, context_text: "không được gọi",
        )

        self.assertEqual(result["query"], "Mức phạt là gì?")
        self.assertEqual(result["ner_entities"], ["Điều 1"])
        self.assertEqual(result["retrieved_nodes"], [])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["answer_text"], DEFAULT_ABSTAIN_ANSWER)
        self.assertEqual(result["context_text"], "")
        self.assertEqual(result["errors"], [])
        self.assertIn("total_ms", result["timings"])

    def test_run_legal_qa_returns_structured_serializable_output(self) -> None:
        result = run_legal_qa(
            "Điều 1 nói gì?",
            ner_infer_fn=lambda query: ["Điều 1"],
            retrieve_fn=lambda query, ner_entities: [
                {
                    "node_id": "Dieu::1",
                    "label": "Điều",
                    "name": "Điều 1: Phạm vi điều chỉnh",
                    "value": "Nội dung điều 1",
                    "bm25": 0.8,
                    "cosine": 0.9,
                    "graph_sum": 1.0,
                    "final_score": 2.2,
                    "content_embedding": [0.1, 0.2],
                    "graph_embedding": [0.3, 0.4],
                }
            ],
            generate_answer_fn=lambda query, context_text: "Điều 1 quy định phạm vi điều chỉnh.",
        )

        self.assertEqual(result["answer_text"], "Điều 1 quy định phạm vi điều chỉnh.")
        self.assertEqual(result["ner_entities"], ["Điều 1"])
        self.assertEqual(
            result["retrieved_nodes"],
            [
                {
                    "node_id": "Dieu::1",
                    "label": "Điều",
                    "name": "Điều 1: Phạm vi điều chỉnh",
                    "value": "Nội dung điều 1",
                    "bm25": 0.8,
                    "cosine": 0.9,
                    "graph_sum": 1.0,
                    "final_score": 2.2,
                }
            ],
        )
        self.assertEqual(
            result["citations"],
            [
                {
                    "node_id": "Dieu::1",
                    "label": "Điều",
                    "name": "Điều 1: Phạm vi điều chỉnh",
                    "score": 2.2,
                }
            ],
        )
        self.assertIn("Điều 1: Phạm vi điều chỉnh", result["context_text"])
        self.assertEqual(result["errors"], [])


if __name__ == "__main__":
    unittest.main()
