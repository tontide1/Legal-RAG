import unittest

from src.NER.legal_ref_parser import (
    extract_code_law_entities,
    extract_decree_entities,
    extract_law_entities,
    extract_legal_document_entities,
    merge_legal_entities,
    normalize_entity_text,
    normalize_whitespace,
)


class LegalRefParserTest(unittest.TestCase):

    def test_normalize_whitespace_collapses_spaces(self) -> None:
        self.assertEqual(normalize_whitespace("  Luật   Hải  quan  "), "Luật Hải quan")

    def test_normalize_entity_text_title_cases(self) -> None:
        self.assertEqual(normalize_entity_text("luật hải quan"), "Luật Hải Quan")

    def test_extract_decree_entities_finds_standard_code(self) -> None:
        result = extract_decree_entities("Theo Nghị định 100/2019/NĐ-CP thì sao?")
        self.assertEqual(result, ["Nghị định 100/2019/NĐ-CP"])

    def test_extract_decree_entities_normalizes_lowercase(self) -> None:
        result = extract_decree_entities("nghị định 100/2019/nđ-cp quy định gì")
        self.assertEqual(result, ["Nghị định 100/2019/NĐ-CP"])

    def test_extract_decree_entities_returns_empty_for_no_match(self) -> None:
        result = extract_decree_entities("Luật Hải quan quy định gì?")
        self.assertEqual(result, [])

    def test_extract_law_entities_finds_known_law(self) -> None:
        result = extract_law_entities("Luật Hải quan quy định gì?")
        self.assertEqual(result, ["Luật Hải Quan"])

    def test_extract_law_entities_finds_doanh_nghiep(self) -> None:
        result = extract_law_entities("Luật Doanh nghiệp nói gì?")
        self.assertEqual(result, ["Luật Doanh Nghiệp"])

    def test_extract_law_entities_finds_lao_dong(self) -> None:
        result = extract_law_entities("Theo Luật Lao động thì sao?")
        self.assertEqual(result, ["Luật Lao Động"])

    def test_extract_law_entities_does_not_match_luat_su(self) -> None:
        result = extract_law_entities("Tôi muốn hỏi luật sư về thủ tục")
        self.assertEqual(result, [])

    def test_extract_law_entities_stops_at_stop_token(self) -> None:
        result = extract_law_entities("Luật Hải quan là gì?")
        self.assertEqual(result, ["Luật Hải Quan"])

    def test_extract_code_law_entities_finds_known_code(self) -> None:
        result = extract_code_law_entities("Bộ luật Dân sự quy định thế nào?")
        self.assertEqual(result, ["Bộ Luật Dân Sự"])

    def test_extract_code_law_entities_finds_lao_dong(self) -> None:
        result = extract_code_law_entities("Theo Bộ luật Lao động thì sao?")
        self.assertEqual(result, ["Bộ Luật Lao Động"])

    def test_extract_code_law_entities_does_not_match_bo_phan(self) -> None:
        result = extract_code_law_entities("Bộ phận tiếp nhận hồ sơ ở đâu")
        self.assertEqual(result, [])

    def test_extract_legal_document_entities_combines_all_types(self) -> None:
        result = extract_legal_document_entities(
            "Theo Nghị định 100/2019/NĐ-CP và Luật Hải quan thì sao?"
        )
        self.assertIn("Nghị định 100/2019/NĐ-CP", result)
        self.assertIn("Luật Hải Quan", result)

    def test_extract_legal_document_entities_dedupes(self) -> None:
        result = extract_legal_document_entities(
            "Luật Hải quan và luật hải quan quy định gì?"
        )
        count = sum(1 for e in result if e.lower() == "luật hải quan")
        self.assertEqual(count, 1)

    def test_merge_legal_entities_preserves_order_and_dedupes(self) -> None:
        base = ["Điều 33", "Luật Hải Quan"]
        parsed = ["Luật Hải Quan", "Nghị định 100/2019/NĐ-CP"]
        result = merge_legal_entities(base, parsed)
        self.assertEqual(result, ["Điều 33", "Luật Hải Quan", "Nghị định 100/2019/NĐ-CP"])

    def test_merge_legal_entities_handles_empty_base(self) -> None:
        result = merge_legal_entities([], ["Luật Hải Quan"])
        self.assertEqual(result, ["Luật Hải Quan"])

    def test_merge_legal_entities_handles_empty_parsed(self) -> None:
        result = merge_legal_entities(["Điều 33"], [])
        self.assertEqual(result, ["Điều 33"])

    def test_negative_case_luat_su_not_matched(self) -> None:
        result = extract_legal_document_entities("Tôi muốn hỏi luật sư về thủ tục")
        self.assertEqual(result, [])

    def test_negative_case_dieu_kien_not_matched(self) -> None:
        result = extract_legal_document_entities("Điều kiện đăng ký là gì")
        self.assertEqual(result, [])

    def test_negative_case_bo_phan_not_matched(self) -> None:
        result = extract_legal_document_entities("Bộ phận tiếp nhận hồ sơ")
        self.assertEqual(result, [])

    def test_mixed_article_and_document(self) -> None:
        result = extract_legal_document_entities("Luật Hải quan và Điều 33 quy định gì?")
        self.assertIn("Luật Hải Quan", result)


if __name__ == "__main__":
    unittest.main()
