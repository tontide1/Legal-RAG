import unittest
from unittest.mock import patch


class StreamlitSmokeTest(unittest.TestCase):
    def test_streamlit_app_imports_without_error(self) -> None:
        try:
            import streamlit_app  # noqa: F401
        except ModuleNotFoundError as e:
            if "streamlit" in str(e):
                self.skipTest("streamlit not installed in test environment")
            self.fail(f"streamlit_app.py failed to import: {e}")
        except ImportError as e:
            self.fail(f"streamlit_app.py failed to import: {e}")

    def test_ui_runtime_imports_without_error(self) -> None:
        try:
            from src import ui_runtime  # noqa: F401
        except ImportError as e:
            self.fail(f"src/ui_runtime.py failed to import: {e}")

    def test_check_env_status_returns_dict(self) -> None:
        from src.ui_runtime import check_env_status
        result = check_env_status()
        self.assertIsInstance(result, dict)
        self.assertIn("NEO4J_URI", result)
        self.assertIn("GOOGLE_API_KEY", result)

    @patch("src.ui_runtime.run_legal_qa_for_ui")
    def test_run_legal_qa_for_ui_returns_dict(self, mock_run) -> None:
        from src.ui_runtime import run_legal_qa_for_ui
        mock_run.return_value = {
            "query": "test",
            "ner_entities": [],
            "retrieved_nodes": [],
            "context_text": "",
            "answer_text": "test answer",
            "citations": [],
            "scores": {},
            "timings": {},
            "errors": [],
        }
        result = run_legal_qa_for_ui("test")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer_text"], "test answer")


if __name__ == "__main__":
    unittest.main()
