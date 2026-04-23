import importlib.util
import unittest
from unittest.mock import patch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class NERInferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.NER.ner import (
            DEFAULT_DEDUP_DATASET_PATH,
            _ner_dataset_path,
            extract_entities,
            id2label,
            resolve_ner_backend,
        )

        cls.extract_entities_fn = staticmethod(extract_entities)
        cls.id2label = id2label
        cls.resolve_ner_backend_fn = staticmethod(resolve_ner_backend)
        cls.ner_dataset_path_fn = staticmethod(_ner_dataset_path)
        cls.default_dedup_dataset_path = DEFAULT_DEDUP_DATASET_PATH

    def test_resolve_ner_backend_defaults_to_phobert(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.resolve_ner_backend_fn(), "phobert")

    def test_resolve_ner_backend_accepts_phobert(self) -> None:
        with patch.dict("os.environ", {"NER_BACKEND": "phobert"}, clear=True):
            self.assertEqual(self.resolve_ner_backend_fn(), "phobert")

    def test_ner_dataset_path_accepts_env_override(self) -> None:
        with patch.dict("os.environ", {"NER_DATASET_PATH": "src/NER/processed/phase1_train.json"}, clear=True):
            self.assertEqual(str(self.ner_dataset_path_fn()), "src/NER/processed/phase1_train.json")

    def test_ner_dataset_path_defaults_to_processed_dedup(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(self.ner_dataset_path_fn(), self.default_dedup_dataset_path)

    def test_extract_entities_returns_empty_for_all_o_labels(self) -> None:
        tokens = ["Điều", "đó", "có", "hợp", "lý", "không", "?"]
        predictions = [0, 0, 0, 0, 0, 0, 0]

        entities = self.extract_entities_fn(tokens, predictions, self.id2label)
        self.assertEqual(entities, [])

    def test_extract_entities_joins_b_i_article_span(self) -> None:
        tokens = ["Điều", "15", "quy", "định", "gì", "?"]
        predictions = [1, 2, 0, 0, 0, 0]

        entities = self.extract_entities_fn(tokens, predictions, self.id2label)
        self.assertEqual(entities, ["Điều 15"])

    def test_extract_entities_supports_string_labels(self) -> None:
        tokens = ["Điều", "20", "được", "quy", "định"]
        predictions = ["B-ARTICLE", "I-ARTICLE", "O", "O", "O"]

        entities = self.extract_entities_fn(tokens, predictions, self.id2label)
        self.assertEqual(entities, ["Điều 20"])


if __name__ == "__main__":
    unittest.main()
