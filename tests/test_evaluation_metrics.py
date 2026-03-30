import unittest

from evaluation.metrics import (
    abstention_precision,
    citation_scores,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_retrieval_metrics_use_binary_relevance(self) -> None:
        retrieved_ids = ["n1", "n2", "n3", "n4"]
        gold_ids = {"n2", "n4"}

        self.assertEqual(hit_at_k(retrieved_ids, gold_ids, 2), 1.0)
        self.assertEqual(recall_at_k(retrieved_ids, gold_ids, 2), 0.5)
        self.assertEqual(precision_at_k(retrieved_ids, gold_ids, 2), 0.5)

    def test_ndcg_at_k_rewards_better_ranking(self) -> None:
        better = ndcg_at_k(["n2", "n1", "n4"], {"n2", "n4"}, 3)
        worse = ndcg_at_k(["n1", "n2", "n4"], {"n2", "n4"}, 3)
        self.assertGreater(better, worse)

    def test_citation_scores_compare_normalized_sets(self) -> None:
        scores = citation_scores(
            found_citations=["Điều 1", "Luật Hải quan"],
            gold_citations=["Điều 1", "Nghị định 100"],
        )

        self.assertEqual(scores["citation_precision"], 0.5)
        self.assertEqual(scores["citation_recall"], 0.5)
        self.assertEqual(scores["citation_exact_match"], 0.0)

    def test_abstention_precision_uses_only_predicted_abstentions(self) -> None:
        score = abstention_precision(
            [
                {"answerable": False, "did_abstain": True},
                {"answerable": True, "did_abstain": True},
                {"answerable": False, "did_abstain": False},
            ]
        )

        self.assertEqual(score, 0.5)


if __name__ == "__main__":
    unittest.main()
