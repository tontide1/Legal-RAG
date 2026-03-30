import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this environment")
class NERInferenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.NER.ner import extract_entities, id2label

        cls.extract_entities_fn = staticmethod(extract_entities)
        cls.id2label = id2label

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


if __name__ == "__main__":
    unittest.main()
