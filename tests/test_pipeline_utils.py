import unittest

from Code.pipeline_utils import build_text_payload, make_node_id


class PipelineUtilsTest(unittest.TestCase):
    def test_make_node_id_uses_label_and_name(self) -> None:
        node_id = make_node_id("Nghi Dinh 100/2019/ND-CP", "Dieu 1: Pham vi dieu chinh")
        self.assertEqual(node_id, "Nghi Dinh 100/2019/ND-CP::Dieu 1: Pham vi dieu chinh")

    def test_make_node_id_distinguishes_duplicate_names(self) -> None:
        left = make_node_id("Luat Hai quan", "Dieu 1: Pham vi dieu chinh")
        right = make_node_id("Nghi Dinh 100/2019/ND-CP", "Dieu 1: Pham vi dieu chinh")
        self.assertNotEqual(left, right)

    def test_build_text_payload_uses_empty_string_for_missing_value(self) -> None:
        self.assertEqual(build_text_payload("Chuong I", None), "Chuong I")
        self.assertEqual(build_text_payload("Chuong I", ""), "Chuong I")
        self.assertEqual(build_text_payload("Dieu 1", "Noi dung"), "Dieu 1 Noi dung")


if __name__ == "__main__":
    unittest.main()
