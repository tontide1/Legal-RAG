import unittest

from src.NER.prepare_phase1_dataset import (
    deduplicate_samples,
    generate_synthetic_negative_samples,
    has_article_reference,
    normalize_text_key_from_tokens,
    split_samples_without_text_leakage,
    summarize_samples,
)


class NERDatasetPreparationTest(unittest.TestCase):
    def test_deduplicate_samples_removes_exact_duplicates_only(self) -> None:
        samples = [
            {"tokens": ["Điều", "1"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "1"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "1"], "labels": ["O", "O"]},
        ]

        deduplicated, removed_count = deduplicate_samples(samples)

        self.assertEqual(removed_count, 1)
        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated[0]["labels"], ["B-ARTICLE", "I-ARTICLE"])
        self.assertEqual(deduplicated[1]["labels"], ["O", "O"])

    def test_summarize_samples_reports_positive_negative_and_entity_buckets(self) -> None:
        samples = [
            {"tokens": ["Điều", "1"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {
                "tokens": ["Điều", "2", "Điều", "3"],
                "labels": ["B-ARTICLE", "I-ARTICLE", "B-ARTICLE", "I-ARTICLE"],
            },
            {"tokens": ["Điều", "đó"], "labels": ["O", "O"]},
        ]

        summary = summarize_samples(samples)

        self.assertEqual(summary["total_samples"], 3)
        self.assertEqual(summary["positive_samples"], 2)
        self.assertEqual(summary["negative_samples"], 1)
        self.assertEqual(summary["entity_count_buckets"]["0"], 1)
        self.assertEqual(summary["entity_count_buckets"]["1"], 1)
        self.assertEqual(summary["entity_count_buckets"]["2+"], 1)

    def test_generate_synthetic_negative_samples_uses_o_labels_and_avoids_article_ref(self) -> None:
        existing_text_keys = {"điều này thực sự quan trọng không ?"}
        synthetic_samples, family_distribution = generate_synthetic_negative_samples(
            target_count=40,
            existing_text_keys=existing_text_keys,
            random_seed=123,
        )

        self.assertEqual(len(synthetic_samples), 40)
        self.assertTrue(family_distribution)

        normalized_text_keys = set()
        for sample in synthetic_samples:
            self.assertTrue(sample["tokens"])
            self.assertEqual(set(sample["labels"]), {"O"})
            text = " ".join(sample["tokens"])
            self.assertFalse(has_article_reference(text))

            normalized_key = normalize_text_key_from_tokens(sample["tokens"])
            self.assertNotIn(normalized_key, existing_text_keys)
            self.assertNotIn(normalized_key, normalized_text_keys)
            normalized_text_keys.add(normalized_key)

    def test_split_samples_without_text_leakage_keeps_text_keys_disjoint(self) -> None:
        samples = [
            {"tokens": ["Điều", "1"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "2"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "3"], "labels": ["B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "đó"], "labels": ["O", "O"]},
            {"tokens": ["Bạn", "đang", "lo", "điều", "gì", "?"], "labels": ["O", "O", "O", "O", "O", "O"]},
            {"tokens": ["Hồ", "sơ", "này", "cần", "gì", "?"], "labels": ["O", "O", "O", "O", "O", "O"]},
            {"tokens": ["Điều", "4", "Điều", "5"], "labels": ["B-ARTICLE", "I-ARTICLE", "B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Điều", "6", "Điều", "7"], "labels": ["B-ARTICLE", "I-ARTICLE", "B-ARTICLE", "I-ARTICLE"]},
            {"tokens": ["Nghị", "định", "này", "áp", "dụng", "ra", "sao", "?"], "labels": ["O", "O", "O", "O", "O", "O", "O", "O"]},
            {"tokens": ["Thời", "hạn", "30", "ngày", "có", "gia", "hạn", "không", "?"], "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]},
        ]

        splits, report = split_samples_without_text_leakage(
            samples,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            random_seed=42,
        )

        train_keys = {normalize_text_key_from_tokens(sample["tokens"]) for sample in splits["train"]}
        val_keys = {normalize_text_key_from_tokens(sample["tokens"]) for sample in splits["val"]}
        test_keys = {normalize_text_key_from_tokens(sample["tokens"]) for sample in splits["test"]}

        self.assertFalse(train_keys & val_keys)
        self.assertFalse(train_keys & test_keys)
        self.assertFalse(val_keys & test_keys)
        self.assertFalse(report["leakage"]["has_overlap"])

        total_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        self.assertEqual(total_samples, len(samples))


if __name__ == "__main__":
    unittest.main()
