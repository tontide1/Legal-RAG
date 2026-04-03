from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_train.json"
DEFAULT_VAL_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_val.json"
DEFAULT_TEST_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_test.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_train_augmented.json"
DEFAULT_AUDIT_PATH = REPO_ROOT / "src" / "NER" / "reports" / "phase1_train_augmented_audit.json"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "src" / "NER" / "reports" / "phase1_train_augmented_manifest.json"

TOKEN_PATTERN = re.compile(r"\w+(?:[/-]\w+)*|[^\w\s]", flags=re.UNICODE)
ARTICLE_REGEX = re.compile(r"(?i)\bđiều\s+(\d+|[IVXLC]+)\b")
ROMAN_REGEX = re.compile(r"(?i)^[IVXLC]+$")
NUMBER_REGEX = re.compile(r"^\d+$")

LABEL_O = "O"
LABEL_B = "B-ARTICLE"
LABEL_I = "I-ARTICLE"

FAMILY_QUOTAS: dict[str, int] = {
    "negative_with_dieu_non_citation": 500,
    "negative_legal_generic": 400,
    "negative_numeric_non_article": 300,
    "negative_adversarial_near_miss": 300,
    "positive_single_article_paraphrase": 500,
    "positive_multi_article_paraphrase": 400,
    "positive_roman_numeral_article": 120,
    "positive_punctuation_and_case_variants": 100,
    "positive_article_with_doc_context": 80,
}

NEGATIVE_FAMILIES = {
    "negative_with_dieu_non_citation",
    "negative_legal_generic",
    "negative_numeric_non_article",
    "negative_adversarial_near_miss",
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    family: str
    template: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment phase1_train.json according to phase1 blueprint.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample-ratio", type=float, default=1.2)
    parser.add_argument("--template-max-ratio", type=float, default=0.15)
    return parser.parse_args()


def read_json_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list at {path}")
    return payload


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text_key(tokens: list[str]) -> str:
    return " ".join(" ".join(tokens).lower().split())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def count_entities(labels: list[str]) -> int:
    count = 0
    in_entity = False
    for label in labels:
        if label == LABEL_B:
            count += 1
            in_entity = True
            continue
        if label == LABEL_I:
            if not in_entity:
                count += 1
                in_entity = True
            continue
        in_entity = False
    return count


def summarize_samples(samples: list[dict]) -> dict:
    label_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    entity_bucket = Counter({"0": 0, "1": 0, "2+": 0})
    positive = 0

    for sample in samples:
        labels = sample["labels"]
        label_counter.update(labels)
        source_counter.update([sample.get("source", "original")])
        if sample.get("template_family"):
            family_counter.update([sample["template_family"]])

        entities = count_entities(labels)
        if entities == 0:
            entity_bucket["0"] += 1
        elif entities == 1:
            entity_bucket["1"] += 1
        else:
            entity_bucket["2+"] += 1
        if any(label != LABEL_O for label in labels):
            positive += 1

    total = len(samples)
    return {
        "total_samples": total,
        "positive_samples": positive,
        "negative_samples": total - positive,
        "entity_count_buckets": dict(entity_bucket),
        "label_distribution": dict(label_counter),
        "source_distribution": dict(source_counter),
        "template_family_distribution": dict(family_counter),
    }


def has_article_citation(text: str) -> bool:
    return bool(ARTICLE_REGEX.search(text))


def make_labels_for_article_mentions(tokens: list[str]) -> list[str]:
    labels = [LABEL_O] * len(tokens)
    for index in range(len(tokens) - 1):
        if tokens[index].lower() != "điều":
            continue
        next_token = tokens[index + 1]
        if NUMBER_REGEX.match(next_token) or ROMAN_REGEX.match(next_token):
            labels[index] = LABEL_B
            labels[index + 1] = LABEL_I
    return labels


def is_valid_positive(tokens: list[str], labels: list[str]) -> bool:
    if not tokens or len(tokens) != len(labels):
        return False
    return count_entities(labels) > 0


def is_valid_negative(text: str, tokens: list[str], labels: list[str]) -> bool:
    if not tokens or len(tokens) != len(labels):
        return False
    if any(label != LABEL_O for label in labels):
        return False
    return not has_article_citation(text)


def build_rules() -> list[Rule]:
    rules: list[Rule] = []

    negative_dieu_templates = [
        "Theo {audience}, điều này có hợp lý trong trường hợp {context} không?",
        "Tôi đang băn khoăn điều gì quan trọng nhất về {topic} khi {context}?",
        "Bạn nghĩ điều đó có cần xác nhận thêm cho {topic} tại {location} không?",
        "Điều này có ổn khi {context} liên quan đến {topic} trong {timeframe} không?",
        "Theo bạn điều đó có phù hợp trong tình huống {context} ở {location} không?",
        "Điều này có cần làm rõ thêm khi xử lý {topic} qua {procedure} không?",
        "Bạn có thấy điều đó quan trọng với {topic} khi {audience} xử lý hồ sơ không?",
        "Điều đó có nên áp dụng cho {topic} trong bối cảnh {context} tại {location} không?",
        "Trong {timeframe}, điều này có còn phù hợp cho {topic} không?",
        "Với {audience}, điều đó có giúp giải quyết {topic} khi {context} không?",
        "Điều này có gây vướng mắc cho {topic} ở {location} không?",
        "Theo kinh nghiệm của {audience}, điều đó có nên ưu tiên cho {topic} không?",
    ]
    for index, template in enumerate(negative_dieu_templates, start=1):
        rules.append(Rule(f"ndieu_{index}", "negative_with_dieu_non_citation", template))

    negative_legal_templates = [
        "Luật {law_name} quy định gì về {topic}?",
        "Nghị định {decree_id} áp dụng cho trường hợp nào?",
        "Bộ luật {law_name} có nguyên tắc nào cho {topic}?",
        "Luật {law_name} điều chỉnh vấn đề {topic} ra sao?",
        "Nghị định {decree_id} có phạm vi áp dụng thế nào với {topic}?",
        "Bộ luật {law_name} có nêu điều kiện cho {topic} không?",
        "Theo Luật {law_name}, thủ tục {topic} thực hiện thế nào?",
        "Nghị định {decree_id} hướng dẫn gì cho {topic}?",
    ]
    for index, template in enumerate(negative_legal_templates, start=1):
        rules.append(Rule(f"nlegal_{index}", "negative_legal_generic", template))

    negative_numeric_templates = [
        "Mức phạt {amount} đồng có bắt buộc nộp ngay không?",
        "Thời hạn {days} ngày có được gia hạn không?",
        "Sau {days} ngày chưa nộp thì xử lý thế nào?",
        "Khoản tiền {amount} đồng có thể nộp trực tuyến không?",
        "Quá {days} ngày thì hồ sơ có bị từ chối không?",
        "Mốc {days} ngày được tính từ thời điểm nào?",
        "Số tiền {amount} đồng có thể chia đợt nộp không?",
        "Nếu chậm {days} ngày thì có bị phạt thêm không?",
        "Với thủ tục {action}, mức {amount} đồng được nộp ở đâu?",
        "Trong hồ sơ {topic}, hạn {days} ngày có thể kéo dài không?",
        "Nếu quá hạn {days} ngày khi {context} thì xử lý ra sao?",
        "Khoản {amount} đồng cho {topic} có được miễn giảm không?",
        "Mức đóng {amount} đồng mỗi lần có đúng với {topic} không?",
        "Thời gian {days} ngày cho {action} có đủ để hoàn tất không?",
        "Nếu nộp chậm {days} ngày trong {topic} thì có bị khóa hồ sơ không?",
        "Khi {context}, khoản {amount} đồng có thể nộp thành nhiều lần không?",
    ]
    for index, template in enumerate(negative_numeric_templates, start=1):
        rules.append(Rule(f"nnum_{index}", "negative_numeric_non_article", template))

    negative_near_miss_templates = [
        "Điều kiện để {action} là gì?",
        "Điều đó có hiệu lực chưa?",
        "Điều này có cần công chứng không?",
        "Bạn muốn biết điều gì nữa về {topic}?",
        "Điều khoản này có hợp lý cho {topic} không?",
        "Điều kiện nào cần chuẩn bị để {action}?",
        "Điều đó có bắt buộc với {topic} không?",
        "Điều này có cần xác nhận thêm khi {action} không?",
        "Điều kiện để {action} tại {location} gồm những gì?",
        "Điều đó có còn phù hợp cho {topic} trong {timeframe} không?",
        "Điều này có phải làm lại khi {context} không?",
        "Điều khoản chung cho {topic} có cần bổ sung gì không?",
        "Bạn muốn biết điều gì về {topic} để hoàn tất {action}?",
        "Điều này có bắt buộc với {audience} khi {context} không?",
        "Điều kiện sơ bộ cho {action} có thay đổi theo {location} không?",
        "Điều đó có nên ưu tiên với {topic} trong giai đoạn {timeframe} không?",
        "Điều này có cần xác minh thêm với {audience} không?",
        "Điều kiện thực hiện {action} khi {context} có phức tạp không?",
    ]
    for index, template in enumerate(negative_near_miss_templates, start=1):
        rules.append(Rule(f"nmiss_{index}", "negative_adversarial_near_miss", template))

    positive_single_templates = [
        "Điều {n} quy định gì?",
        "Theo Điều {n} thì nội dung thế nào?",
        "Nội dung của Điều {n} là gì?",
        "Điều {n} áp dụng ra sao trong trường hợp này?",
        "Điều {n} có nêu rõ quy định không?",
        "Xin hỏi Điều {n} quy định cụ thể thế nào?",
        "Cho tôi biết Điều {n} nói gì?",
        "Điều {n} được hiểu như thế nào?",
    ]
    for index, template in enumerate(positive_single_templates, start=1):
        rules.append(Rule(f"psingle_{index}", "positive_single_article_paraphrase", template))

    positive_multi_templates = [
        "Điều {a} và Điều {b} quy định gì?",
        "Theo Điều {a}, Điều {b}, Điều {c} thì xử lý thế nào?",
        "Điều {a} khác gì Điều {b}?",
        "Điều {a}, Điều {b} có liên quan gì với nhau?",
        "So sánh Điều {a} với Điều {b} giúp tôi.",
        "Theo Điều {a} và Điều {b}, cần lưu ý điều gì?",
        "Điều {a}, Điều {b}, Điều {c} quy định ra sao?",
        "Điều {a} cùng Điều {b} áp dụng như thế nào?",
    ]
    for index, template in enumerate(positive_multi_templates, start=1):
        rules.append(Rule(f"pmulti_{index}", "positive_multi_article_paraphrase", template))

    roman_templates = [
        "Điều {roman} quy định gì?",
        "Theo Điều {roman} thì áp dụng như thế nào?",
        "Điều {roman} nói gì về trường hợp này?",
        "Nội dung Điều {roman} là gì?",
        "Điều {roman} có hiệu lực ra sao?",
        "Xin hỏi Điều {roman} quy định thế nào?",
        "Theo quy định tại Điều {roman}, xử lý ra sao?",
        "Điều {roman} được áp dụng khi nào?",
        "Theo Điều {roman} thì {topic} cần xử lý thế nào?",
        "Điều {roman} của Luật {law_name} quy định gì?",
        "Trong Nghị định {decree_id}, Điều {roman} áp dụng ra sao?",
        "Điều {roman} có liên quan gì đến {action}?",
        "Theo Điều {roman}, trường hợp {context} xử lý như thế nào?",
        "Điều {roman} tại {location} có áp dụng giống nhau không?",
    ]
    for index, template in enumerate(roman_templates, start=1):
        rules.append(Rule(f"proman_{index}", "positive_roman_numeral_article", template))

    punctuation_templates = [
        "điều {n} quy định gì?",
        "Điều {n}, quy định thế nào?",
        "Theo điều {n} thì sao?",
        "Điều {n}: nội dung ra sao?",
        "Cho tôi hỏi điều {n} quy định gì?",
        "Điều {n} - áp dụng như thế nào?",
        "điều {n}, có hiệu lực chưa?",
        "Theo điều {n}, cần làm gì?",
    ]
    for index, template in enumerate(punctuation_templates, start=1):
        rules.append(Rule(f"ppunc_{index}", "positive_punctuation_and_case_variants", template))

    doc_context_templates = [
        "Điều {n} của Luật {law_name} quy định gì?",
        "Theo Điều {n} trong Nghị định {decree_id} thì mức phạt thế nào?",
        "Nội dung Điều {n} của Luật {law_name} là gì?",
        "Theo Nghị định {decree_id}, Điều {n} áp dụng ra sao?",
        "Điều {n} trong Luật {law_name} có nêu rõ điều kiện không?",
        "Xin hỏi Điều {n} của Nghị định {decree_id} quy định gì?",
        "Điều {n} thuộc Luật {law_name} nói gì?",
        "Trong Nghị định {decree_id}, Điều {n} được hiểu thế nào?",
    ]
    for index, template in enumerate(doc_context_templates, start=1):
        rules.append(Rule(f"pdoc_{index}", "positive_article_with_doc_context", template))

    return rules


def slot_values() -> dict[str, list[str]]:
    return {
        "context": [
            "nộp hồ sơ trực tuyến",
            "thay đổi nơi cư trú",
            "chậm nộp hồ sơ",
            "cập nhật thông tin cá nhân",
            "xin cấp lại giấy tờ",
            "nộp phạt quá hạn",
            "xử lý hồ sơ bổ sung",
            "khiếu nại quyết định",
            "xác nhận tạm trú",
            "điều chỉnh thông tin doanh nghiệp",
            "nộp hồ sơ qua bưu điện",
            "bổ sung giấy tờ còn thiếu",
            "giải trình vi phạm",
            "đăng ký thay đổi ngành nghề",
            "chờ phản hồi từ cơ quan",
            "nộp lệ phí muộn",
            "đăng ký tài khoản dịch vụ công",
            "xác minh thông tin cư trú",
            "hoàn thiện hồ sơ bản giấy",
            "chuyển hồ sơ liên thông",
        ],
        "topic": [
            "xử lý vi phạm hành chính",
            "thủ tục cư trú",
            "thuế thu nhập cá nhân",
            "bảo hiểm xã hội",
            "đăng ký kinh doanh",
            "xin giấy phép xây dựng",
            "khiếu nại hành chính",
            "xử phạt giao thông",
            "điều kiện hành nghề",
            "đăng ký lao động",
            "quản lý hợp đồng",
            "đóng bảo hiểm bắt buộc",
            "xử lý hồ sơ doanh nghiệp",
            "phạt chậm khai báo",
            "hỗ trợ thủ tục công dân",
            "miễn giảm nghĩa vụ tài chính",
            "xác nhận tình trạng pháp lý",
            "tuân thủ quy định nội bộ",
            "thẩm định hồ sơ đầu tư",
            "đăng ký kinh doanh trực tuyến",
        ],
        "law_name": [
            "Đất đai",
            "Doanh nghiệp",
            "Dân sự",
            "Lao động",
            "Bảo hiểm xã hội",
            "An ninh mạng",
            "Thương mại",
            "Hôn nhân và gia đình",
        ],
        "decree_id": [
            "100/2019/NĐ-CP",
            "15/2020/NĐ-CP",
            "123/2021/NĐ-CP",
            "45/2022/NĐ-CP",
            "12/2023/NĐ-CP",
        ],
        "action": [
            "xin giấy phép",
            "đăng ký tạm trú",
            "nộp phạt",
            "khiếu nại quyết định xử phạt",
            "xác nhận cư trú",
            "đăng ký kinh doanh",
            "bổ sung hồ sơ",
            "xác nhận thông tin doanh nghiệp",
            "xin cấp lại giấy chứng nhận",
            "đăng ký thay đổi thông tin",
        ],
        "amount": [
            "250000",
            "500000",
            "750000",
            "1000000",
            "1500000",
            "2000000",
            "3000000",
            "5000000",
            "7000000",
            "10000000",
            "15000000",
            "20000000",
        ],
        "days": ["3", "5", "7", "10", "15", "20", "30", "45", "60", "90", "120"],
        "n": [str(value) for value in range(1, 251)],
        "roman": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"],
        "audience": [
            "cơ quan xử lý",
            "người dân",
            "doanh nghiệp",
            "cán bộ tiếp nhận",
            "bộ phận một cửa",
            "người nộp hồ sơ",
            "đơn vị quản lý",
            "cơ quan chuyên môn",
        ],
        "location": [
            "cấp quận",
            "cấp huyện",
            "cấp tỉnh",
            "trung tâm hành chính công",
            "cổng dịch vụ công",
            "địa phương",
            "đơn vị tiếp nhận",
            "khu vực cư trú",
        ],
        "timeframe": [
            "giai đoạn hiện tại",
            "thời điểm này",
            "quý này",
            "năm nay",
            "đợt xử lý mới",
            "kỳ tiếp nhận hồ sơ",
        ],
        "procedure": [
            "dịch vụ công trực tuyến",
            "quy trình một cửa",
            "hồ sơ bản điện tử",
            "nộp trực tiếp",
            "hệ thống liên thông",
            "quy trình tiếp nhận nhanh",
        ],
    }


def fill_template(template: str, values: dict[str, list[str]], rng: random.Random) -> str:
    result = template
    for slot_name, options in values.items():
        placeholder = "{" + slot_name + "}"
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(options), 1)

    if "{a}" in result or "{b}" in result or "{c}" in result:
        picks = rng.sample(values["n"], 3)
        result = result.replace("{a}", picks[0]).replace("{b}", picks[1]).replace("{c}", picks[2])

    return result


def quality_gate(
    *,
    sample: dict,
    is_negative: bool,
    global_text_keys: set[str],
) -> tuple[bool, str]:
    tokens = sample["tokens"]
    labels = sample["labels"]
    text = " ".join(tokens)

    if not tokens or len(tokens) < 4:
        return False, "too_short"
    if len(tokens) != len(labels):
        return False, "length_mismatch"

    text_key = normalize_text_key(tokens)
    if text_key in global_text_keys:
        return False, "duplicate"

    if is_negative:
        if not is_valid_negative(text, tokens, labels):
            return False, "invalid_negative"
    else:
        if not is_valid_positive(tokens, labels):
            return False, "invalid_positive"

    return True, "ok"


def generate_family_samples(
    *,
    family: str,
    quota: int,
    rules: list[Rule],
    values: dict[str, list[str]],
    rng: random.Random,
    global_text_keys: set[str],
    oversample_ratio: float,
    template_max_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    is_negative = family in NEGATIVE_FAMILIES
    target_raw = math.ceil(quota * oversample_ratio)
    max_per_rule = max(1, math.ceil(quota * template_max_ratio))

    selected: list[dict] = []
    manifest: list[dict] = []
    rejected_reasons: Counter[str] = Counter()
    rule_usage: Counter[str] = Counter()
    attempts = 0
    max_attempts = target_raw * 120

    while len(selected) < quota and attempts < max_attempts:
        attempts += 1
        rule = rng.choice(rules)
        if rule_usage[rule.rule_id] >= max_per_rule:
            rejected_reasons["rule_cap"] += 1
            continue

        text = fill_template(rule.template, values, rng)
        tokens = tokenize(text)

        if is_negative:
            labels = [LABEL_O] * len(tokens)
        else:
            labels = make_labels_for_article_mentions(tokens)

        sample = {
            "tokens": tokens,
            "labels": labels,
            "source": "augmentation_v1",
            "template_family": family,
            "generation_rule_id": rule.rule_id,
        }

        passed, reason = quality_gate(sample=sample, is_negative=is_negative, global_text_keys=global_text_keys)
        if not passed:
            rejected_reasons[reason] += 1
            continue

        selected.append(sample)
        rule_usage[rule.rule_id] += 1
        text_key = normalize_text_key(tokens)
        global_text_keys.add(text_key)

        manifest.append(
            {
                "sample_index": len(selected) - 1,
                "source": "augmentation_v1",
                "template_family": family,
                "generation_rule_id": rule.rule_id,
                "seed": seed,
                "text": " ".join(tokens),
            }
        )

    if len(selected) < quota:
        raise RuntimeError(
            f"Family {family} generated {len(selected)}/{quota} samples after {attempts} attempts."
        )

    generation_stats = {
        "family": family,
        "quota": quota,
        "target_raw": target_raw,
        "selected": len(selected),
        "attempts": attempts,
        "max_per_rule": max_per_rule,
        "rule_usage": dict(rule_usage),
        "rejected_reasons": dict(rejected_reasons),
    }
    return selected, manifest, generation_stats


def build_leakage_counts(
    *,
    generated_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
) -> dict[str, int]:
    generated_keys = {normalize_text_key(sample["tokens"]) for sample in generated_samples}
    val_keys = {normalize_text_key(sample["tokens"]) for sample in val_samples}
    test_keys = {normalize_text_key(sample["tokens"]) for sample in test_samples}
    return {
        "generated_vs_val": len(generated_keys & val_keys),
        "generated_vs_test": len(generated_keys & test_keys),
    }


def build_audit(
    *,
    train_samples: list[dict],
    generated_samples: list[dict],
    merged_samples: list[dict],
    generation_stats: list[dict],
    leakage_counts: dict[str, int],
    seed: int,
    oversample_ratio: float,
    template_max_ratio: float,
) -> dict:
    family_counter = Counter(sample.get("template_family", "") for sample in generated_samples)
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "seed": seed,
            "oversample_ratio": oversample_ratio,
            "template_max_ratio": template_max_ratio,
            "family_quotas": FAMILY_QUOTAS,
        },
        "summary": {
            "before_augment": summarize_samples(train_samples),
            "generated_only": summarize_samples(generated_samples),
            "after_augment": summarize_samples(merged_samples),
            "generated_count": len(generated_samples),
            "family_distribution": dict(family_counter),
            "leakage_counts": leakage_counts,
        },
        "generation_stats": generation_stats,
    }


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    output_path = Path(args.output)
    audit_output_path = Path(args.audit_output)
    manifest_output_path = Path(args.manifest_output)

    rng = random.Random(args.seed)

    train_samples = read_json_list(train_path)
    val_samples = read_json_list(val_path)
    test_samples = read_json_list(test_path)

    train_keys = {normalize_text_key(sample["tokens"]) for sample in train_samples}
    val_keys = {normalize_text_key(sample["tokens"]) for sample in val_samples}
    test_keys = {normalize_text_key(sample["tokens"]) for sample in test_samples}
    global_text_keys = set(train_keys | val_keys | test_keys)

    rules = build_rules()
    rules_by_family: dict[str, list[Rule]] = defaultdict(list)
    for rule in rules:
        rules_by_family[rule.family].append(rule)

    values = slot_values()

    generated_samples: list[dict] = []
    manifest_rows: list[dict] = []
    generation_stats: list[dict] = []

    for family, quota in FAMILY_QUOTAS.items():
        family_rules = rules_by_family[family]
        selected, manifest, stats = generate_family_samples(
            family=family,
            quota=quota,
            rules=family_rules,
            values=values,
            rng=rng,
            global_text_keys=global_text_keys,
            oversample_ratio=args.oversample_ratio,
            template_max_ratio=args.template_max_ratio,
            seed=args.seed,
        )
        generated_samples.extend(selected)
        manifest_rows.extend(manifest)
        generation_stats.append(stats)

    merged_samples = train_samples + generated_samples

    leakage_counts = build_leakage_counts(
        generated_samples=generated_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )
    if leakage_counts["generated_vs_val"] != 0 or leakage_counts["generated_vs_test"] != 0:
        raise RuntimeError(f"Leakage detected after generation: {leakage_counts}")

    audit = build_audit(
        train_samples=train_samples,
        generated_samples=generated_samples,
        merged_samples=merged_samples,
        generation_stats=generation_stats,
        leakage_counts=leakage_counts,
        seed=args.seed,
        oversample_ratio=args.oversample_ratio,
        template_max_ratio=args.template_max_ratio,
    )

    write_json(output_path, merged_samples)
    write_json(audit_output_path, audit)
    write_json(manifest_output_path, manifest_rows)

    print(f"Saved augmented train: {output_path}")
    print(f"Saved audit report: {audit_output_path}")
    print(f"Saved manifest: {manifest_output_path}")
    print(f"Generated samples: {len(generated_samples)}")
    print(f"Final train size: {len(merged_samples)}")


if __name__ == "__main__":
    main()
