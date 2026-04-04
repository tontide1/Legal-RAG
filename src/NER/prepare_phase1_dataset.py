from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from math import floor
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "src" / "NER" / "ner_data_8000.json"
DEFAULT_DEDUP_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_dedup.json"
DEFAULT_AUDIT_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "reports" / "phase1_dataset_audit.json"
DEFAULT_SYNTHETIC_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_synthetic_negatives.json"
DEFAULT_TRAIN_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_train.json"
DEFAULT_VAL_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_val.json"
DEFAULT_TEST_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "processed" / "phase1_test.json"
DEFAULT_SPLIT_REPORT_OUTPUT_PATH = REPO_ROOT / "src" / "NER" / "reports" / "phase1_split_report.json"

LABEL_O = "O"
LABEL_B_ARTICLE = "B-ARTICLE"
LABEL_I_ARTICLE = "I-ARTICLE"
ARTICLE_REFERENCE_PATTERN = re.compile(r"(?i)\bđiều\s+(?:\d+|[ivxlcdm]+)\b")
WORD_OR_PUNCT_PATTERN = re.compile(r"\w+(?:[/-]\w+)*|[^\w\s]", flags=re.UNICODE)


def load_dataset(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"NER dataset must be a list, got {type(payload)!r}")
    return payload


def validate_sample(sample: dict, index: int) -> None:
    if "tokens" not in sample or "labels" not in sample:
        raise ValueError(f"Sample #{index} is missing 'tokens' or 'labels'")

    tokens = sample["tokens"]
    labels = sample["labels"]
    if not isinstance(tokens, list) or not isinstance(labels, list):
        raise ValueError(f"Sample #{index} must use list values for tokens/labels")
    if len(tokens) != len(labels):
        raise ValueError(
            f"Sample #{index} has mismatched lengths: tokens={len(tokens)} labels={len(labels)}"
        )
    unknown_labels = [label for label in labels if label not in {LABEL_O, LABEL_B_ARTICLE, LABEL_I_ARTICLE}]
    if unknown_labels:
        raise ValueError(f"Sample #{index} contains unsupported labels: {sorted(set(unknown_labels))}")


def sample_key(sample: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(sample["tokens"]), tuple(sample["labels"])


def deduplicate_samples(samples: list[dict]) -> tuple[list[dict], int]:
    deduplicated: list[dict] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for index, sample in enumerate(samples):
        validate_sample(sample, index)
        key = sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(sample)

    removed_count = len(samples) - len(deduplicated)
    return deduplicated, removed_count


def count_entities(labels: list[str]) -> int:
    entity_count = 0
    in_entity = False
    for label in labels:
        if label == LABEL_B_ARTICLE:
            entity_count += 1
            in_entity = True
            continue
        if label == LABEL_I_ARTICLE:
            if not in_entity:
                entity_count += 1
                in_entity = True
            continue
        in_entity = False
    return entity_count


def summarize_samples(samples: list[dict]) -> dict:
    label_counter: Counter[str] = Counter()
    entity_bucket_counter: Counter[str] = Counter({"0": 0, "1": 0, "2+": 0})

    positive_count = 0
    for index, sample in enumerate(samples):
        validate_sample(sample, index)
        labels = sample["labels"]
        label_counter.update(labels)

        entity_count = count_entities(labels)
        if entity_count == 0:
            entity_bucket_counter["0"] += 1
        elif entity_count == 1:
            entity_bucket_counter["1"] += 1
        else:
            entity_bucket_counter["2+"] += 1

        if any(label != LABEL_O for label in labels):
            positive_count += 1

    total = len(samples)
    negative_count = total - positive_count
    return {
        "total_samples": total,
        "positive_samples": positive_count,
        "negative_samples": negative_count,
        "entity_count_buckets": {
            "0": entity_bucket_counter["0"],
            "1": entity_bucket_counter["1"],
            "2+": entity_bucket_counter["2+"],
        },
        "label_distribution": dict(label_counter),
    }


def normalize_text_key_from_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return " ".join(text.lower().split())


def tokenize_text(text: str) -> list[str]:
    return WORD_OR_PUNCT_PATTERN.findall(text)


def sample_text(sample: dict) -> str:
    return " ".join(sample["tokens"])


def has_article_reference(text: str) -> bool:
    return bool(ARTICLE_REFERENCE_PATTERN.search(text))


def build_synthetic_candidate_texts() -> dict[str, list[str]]:
    topics = [
        "thủ tục nộp phạt",
        "hồ sơ xin giấy phép",
        "điều kiện kinh doanh",
        "thời hiệu xử phạt",
        "trình tự giải quyết",
        "miễn giảm nghĩa vụ",
        "xử lý vi phạm hành chính",
        "quy trình khiếu nại",
        "đăng ký tạm trú",
        "thuế thu nhập cá nhân",
        "bảo hiểm xã hội",
        "an toàn giao thông",
    ]
    contexts = [
        "nộp hồ sơ trực tuyến",
        "làm việc với cơ quan nhà nước",
        "chậm nộp hồ sơ",
        "cập nhật thông tin cá nhân",
        "xin cấp lại giấy tờ",
        "thay đổi nơi cư trú",
        "kinh doanh hộ cá thể",
        "nộp phạt quá hạn",
        "thực hiện thủ tục mới",
        "áp dụng quy định cũ",
    ]
    actions = [
        "đăng ký kinh doanh",
        "xin cấp giấy phép xây dựng",
        "đăng ký tạm trú",
        "khai báo thuế",
        "nộp phạt vi phạm",
        "khiếu nại quyết định xử phạt",
        "đề nghị miễn giảm",
        "xác nhận cư trú",
    ]
    violations = [
        "chậm nộp thuế",
        "không đội mũ bảo hiểm",
        "xây dựng không phép",
        "kinh doanh không đăng ký",
        "khai báo sai thông tin",
        "không chấp hành quyết định xử phạt",
        "không nộp hồ sơ đúng hạn",
    ]
    law_names = [
        "Đất đai",
        "Doanh nghiệp",
        "Bảo hiểm xã hội",
        "An ninh mạng",
        "Giao thông đường bộ",
        "Hôn nhân và gia đình",
        "Dân sự",
        "Lao động",
    ]
    decree_ids = [
        "15/2020/NĐ-CP",
        "123/2021/NĐ-CP",
        "100/2019/NĐ-CP",
        "45/2022/NĐ-CP",
        "12/2023/NĐ-CP",
    ]
    durations = ["07", "10", "15", "30", "45", "60", "90"]
    money_values = [
        "500000",
        "1000000",
        "1500000",
        "2000000",
        "3000000",
        "5000000",
        "10000000",
    ]

    family_candidates: dict[str, list[str]] = defaultdict(list)

    for topic in topics:
        for context in contexts:
            family_candidates["doi-song-co-tu-dieu"].append(
                f"Điều này có phù hợp khi {context} liên quan đến {topic} không?"
            )
            family_candidates["doi-song-co-tu-dieu"].append(
                f"Tôi đang băn khoăn điều gì quan trọng nhất về {topic} khi {context}?"
            )

    for action in actions:
        family_candidates["phap-ly-tong-quat"].append(f"Thủ tục {action} hiện được thực hiện như thế nào?")
        family_candidates["phap-ly-tong-quat"].append(f"Chi phí khi {action} thường gồm những khoản nào?")
        for context in contexts:
            family_candidates["phap-ly-tong-quat"].append(
                f"Khi {context} thì quy trình {action} cần lưu ý những gì?"
            )

    for violation in violations:
        family_candidates["phap-ly-tong-quat"].append(
            f"Mức phạt đối với hành vi {violation} hiện nay là bao nhiêu?"
        )
        for context in contexts:
            family_candidates["phap-ly-tong-quat"].append(
                f"Trong trường hợp {context}, hành vi {violation} được xử lý ra sao?"
            )

    for law_name in law_names:
        for topic in topics:
            family_candidates["co-ten-van-ban-khong-co-dieu"].append(
                f"Luật {law_name} quy định gì về {topic}?"
            )
            family_candidates["co-ten-van-ban-khong-co-dieu"].append(
                f"Bộ luật {law_name} có nguyên tắc nào liên quan đến {topic}?"
            )

    for decree_id in decree_ids:
        for topic in topics:
            family_candidates["co-ten-van-ban-khong-co-dieu"].append(
                f"Nghị định {decree_id} áp dụng như thế nào đối với {topic}?"
            )

    for duration in durations:
        family_candidates["co-so-khong-phai-citation"].append(
            f"Thời hạn {duration} ngày có được gia hạn thêm không?"
        )
        family_candidates["co-so-khong-phai-citation"].append(
            f"Nếu quá {duration} ngày thì hồ sơ có bị hủy không?"
        )
        for action in actions:
            family_candidates["co-so-khong-phai-citation"].append(
                f"Với thủ tục {action}, mốc {duration} ngày được tính từ thời điểm nào?"
            )

    for amount in money_values:
        family_candidates["co-so-khong-phai-citation"].append(
            f"Mức phạt {amount} đồng có bắt buộc nộp ngay không?"
        )
        for violation in violations:
            family_candidates["co-so-khong-phai-citation"].append(
                f"Khoản phạt {amount} đồng đối với hành vi {violation} có thể nộp trực tuyến không?"
            )

    for topic in topics:
        family_candidates["hieu-luc-pham-vi-dieu-kien"].append(
            f"Quy định về {topic} có hiệu lực từ thời điểm nào?"
        )
        family_candidates["hieu-luc-pham-vi-dieu-kien"].append(
            f"Phạm vi áp dụng của quy định về {topic} được hiểu ra sao?"
        )
        family_candidates["hieu-luc-pham-vi-dieu-kien"].append(
            f"Điều kiện để được áp dụng quy định về {topic} gồm những gì?"
        )
        for context in contexts:
            family_candidates["hieu-luc-pham-vi-dieu-kien"].append(
                f"Trong bối cảnh {context}, quy định về {topic} có còn hiệu lực không?"
            )

    return {family: list(dict.fromkeys(candidates)) for family, candidates in family_candidates.items()}


def generate_synthetic_negative_samples(
    *,
    target_count: int,
    existing_text_keys: set[str],
    random_seed: int,
) -> tuple[list[dict], dict[str, int]]:
    if target_count <= 0:
        return [], {}

    candidate_by_family = build_synthetic_candidate_texts()
    rng = random.Random(random_seed)
    for candidates in candidate_by_family.values():
        rng.shuffle(candidates)

    generated_samples: list[dict] = []
    generated_text_keys = set(existing_text_keys)
    generated_family_counter: Counter[str] = Counter()
    family_names = sorted(candidate_by_family.keys())
    cursor_by_family = {family: 0 for family in family_names}

    while len(generated_samples) < target_count:
        progressed = False
        for family in family_names:
            candidates = candidate_by_family[family]
            cursor = cursor_by_family[family]

            while cursor < len(candidates):
                candidate_text = candidates[cursor]
                cursor += 1

                normalized_key = " ".join(candidate_text.lower().split())
                if normalized_key in generated_text_keys:
                    continue
                if has_article_reference(candidate_text):
                    continue

                tokens = tokenize_text(candidate_text)
                if not tokens:
                    continue

                generated_samples.append(
                    {
                        "tokens": tokens,
                        "labels": [LABEL_O] * len(tokens),
                        "source": "synthetic_hard_negative",
                        "template_family": family,
                    }
                )
                generated_family_counter[family] += 1
                generated_text_keys.add(normalized_key)
                progressed = True
                break

            cursor_by_family[family] = cursor
            if len(generated_samples) >= target_count:
                break

        if not progressed:
            break

    if len(generated_samples) < target_count:
        raise ValueError(
            "Unable to generate enough synthetic negatives. "
            f"Generated={len(generated_samples)} target={target_count}."
        )

    return generated_samples, dict(generated_family_counter)


def entity_bucket_key(entity_count: int) -> str:
    if entity_count <= 0:
        return "0"
    if entity_count == 1:
        return "1"
    return "2+"


def make_group_profile(samples_in_group: list[dict]) -> tuple[bool, str]:
    has_positive = any(any(label != LABEL_O for label in sample["labels"]) for sample in samples_in_group)
    max_entity_count = max(count_entities(sample["labels"]) for sample in samples_in_group)
    return has_positive, entity_bucket_key(max_entity_count)


def allocate_split_counts(total_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw_counts = [ratio * total_items for ratio in ratios]
    floor_counts = [floor(value) for value in raw_counts]
    remainder = total_items - sum(floor_counts)
    fractional_order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - floor_counts[index]),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        floor_counts[index] += 1
    return floor_counts[0], floor_counts[1], floor_counts[2]


def split_samples_without_text_leakage(
    samples: list[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    grouped_by_text: dict[str, list[dict]] = defaultdict(list)
    for index, sample in enumerate(samples):
        validate_sample(sample, index)
        text_key = normalize_text_key_from_tokens(sample["tokens"])
        grouped_by_text[text_key].append(sample)

    groups_by_stratum: dict[str, list[str]] = defaultdict(list)
    for text_key, group_samples in grouped_by_text.items():
        has_positive, entity_bucket = make_group_profile(group_samples)
        stratum_key = f"{'positive' if has_positive else 'negative'}::{entity_bucket}"
        groups_by_stratum[stratum_key].append(text_key)

    rng = random.Random(random_seed)
    train_text_keys: set[str] = set()
    val_text_keys: set[str] = set()
    test_text_keys: set[str] = set()
    stratum_allocation: dict[str, dict[str, int]] = {}

    for stratum_key, text_keys in groups_by_stratum.items():
        keys = list(text_keys)
        rng.shuffle(keys)
        train_count, val_count, test_count = allocate_split_counts(
            len(keys),
            (train_ratio, val_ratio, test_ratio),
        )

        train_part = keys[:train_count]
        val_part = keys[train_count : train_count + val_count]
        test_part = keys[train_count + val_count :]

        train_text_keys.update(train_part)
        val_text_keys.update(val_part)
        test_text_keys.update(test_part)

        stratum_allocation[stratum_key] = {
            "group_count": len(keys),
            "train_groups": len(train_part),
            "val_groups": len(val_part),
            "test_groups": len(test_part),
        }

    train_samples = [sample for key in train_text_keys for sample in grouped_by_text[key]]
    val_samples = [sample for key in val_text_keys for sample in grouped_by_text[key]]
    test_samples = [sample for key in test_text_keys for sample in grouped_by_text[key]]

    overlap_train_val = train_text_keys & val_text_keys
    overlap_train_test = train_text_keys & test_text_keys
    overlap_val_test = val_text_keys & test_text_keys
    leakage = bool(overlap_train_val or overlap_train_test or overlap_val_test)

    report = {
        "group_count": len(grouped_by_text),
        "stratum_allocation": stratum_allocation,
        "split_group_counts": {
            "train": len(train_text_keys),
            "val": len(val_text_keys),
            "test": len(test_text_keys),
        },
        "leakage": {
            "has_overlap": leakage,
            "overlap_counts": {
                "train_val": len(overlap_train_val),
                "train_test": len(overlap_train_test),
                "val_test": len(overlap_val_test),
            },
        },
    }

    if leakage:
        raise RuntimeError(f"Detected text leakage across splits: {report['leakage']}")

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }, report


def source_distribution(samples: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter(sample.get("source", "original") for sample in samples)
    return dict(counter)


def unique_text_count(samples: list[dict]) -> int:
    return len({normalize_text_key_from_tokens(sample["tokens"]) for sample in samples})


def build_split_report(
    *,
    synthetic_count_target: int,
    synthetic_samples: list[dict],
    synthetic_family_distribution: dict[str, int],
    merged_samples_count: int,
    merged_duplicates_removed: int,
    split_samples: dict[str, list[dict]],
    split_internal_report: dict,
    random_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "random_seed": random_seed,
            "synthetic_negative_target": synthetic_count_target,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "synthetic": {
            "generated_count": len(synthetic_samples),
            "family_distribution": synthetic_family_distribution,
        },
        "merged_dataset": {
            "total_samples": merged_samples_count,
            "duplicates_removed_after_merge": merged_duplicates_removed,
        },
        "splits": {
            split_name: {
                "summary": summarize_samples(samples),
                "source_distribution": source_distribution(samples),
                "unique_text_count": unique_text_count(samples),
            }
            for split_name, samples in split_samples.items()
        },
        "split_internal_report": split_internal_report,
    }


def build_audit_report(
    *,
    input_path: Path,
    dedup_output_path: Path,
    original_samples: list[dict],
    deduplicated_samples: list[dict],
    duplicates_removed: int,
) -> dict:
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_path": str(input_path),
        "dedup_output_path": str(dedup_output_path),
        "summary": {
            "before_dedup": summarize_samples(original_samples),
            "after_dedup": summarize_samples(deduplicated_samples),
            "duplicates_removed": duplicates_removed,
        },
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 1 NER dataset (deduplicate + synthetic negatives + leakage-safe split)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Path to source NER JSON dataset.")
    parser.add_argument(
        "--dedup-output",
        default=str(DEFAULT_DEDUP_OUTPUT_PATH),
        help="Path to write deduplicated dataset JSON.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_AUDIT_OUTPUT_PATH),
        help="Path to write dataset audit JSON.",
    )
    parser.add_argument(
        "--synthetic-output",
        default=str(DEFAULT_SYNTHETIC_OUTPUT_PATH),
        help="Path to write generated synthetic hard negatives JSON.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_TRAIN_OUTPUT_PATH),
        help="Path to write Phase 1 train split JSON.",
    )
    parser.add_argument(
        "--val-output",
        default=str(DEFAULT_VAL_OUTPUT_PATH),
        help="Path to write Phase 1 validation split JSON.",
    )
    parser.add_argument(
        "--test-output",
        default=str(DEFAULT_TEST_OUTPUT_PATH),
        help="Path to write Phase 1 test split JSON.",
    )
    parser.add_argument(
        "--split-report-output",
        default=str(DEFAULT_SPLIT_REPORT_OUTPUT_PATH),
        help="Path to write split report JSON.",
    )
    parser.add_argument(
        "--synthetic-negative-count",
        type=int,
        default=500,
        help="Number of synthetic hard negative samples to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation and splitting.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    dedup_output_path = Path(args.dedup_output)
    audit_output_path = Path(args.audit_output)
    synthetic_output_path = Path(args.synthetic_output)
    train_output_path = Path(args.train_output)
    val_output_path = Path(args.val_output)
    test_output_path = Path(args.test_output)
    split_report_output_path = Path(args.split_report_output)

    samples = load_dataset(input_path)
    deduplicated, removed_count = deduplicate_samples(samples)
    report = build_audit_report(
        input_path=input_path,
        dedup_output_path=dedup_output_path,
        original_samples=samples,
        deduplicated_samples=deduplicated,
        duplicates_removed=removed_count,
    )

    existing_text_keys = {normalize_text_key_from_tokens(sample["tokens"]) for sample in deduplicated}
    synthetic_samples, synthetic_family_distribution = generate_synthetic_negative_samples(
        target_count=args.synthetic_negative_count,
        existing_text_keys=existing_text_keys,
        random_seed=args.seed,
    )

    merged_samples = deduplicated + synthetic_samples
    merged_deduplicated_samples, merged_removed_count = deduplicate_samples(merged_samples)

    split_samples, split_internal_report = split_samples_without_text_leakage(
        merged_deduplicated_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    split_report = build_split_report(
        synthetic_count_target=args.synthetic_negative_count,
        synthetic_samples=synthetic_samples,
        synthetic_family_distribution=synthetic_family_distribution,
        merged_samples_count=len(merged_deduplicated_samples),
        merged_duplicates_removed=merged_removed_count,
        split_samples=split_samples,
        split_internal_report=split_internal_report,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    write_json(dedup_output_path, deduplicated)
    write_json(audit_output_path, report)
    write_json(synthetic_output_path, synthetic_samples)
    write_json(train_output_path, split_samples["train"])
    write_json(val_output_path, split_samples["val"])
    write_json(test_output_path, split_samples["test"])
    write_json(split_report_output_path, split_report)

    print(f"Saved deduplicated dataset: {dedup_output_path}")
    print(f"Saved audit report: {audit_output_path}")
    print(f"Saved synthetic hard negatives: {synthetic_output_path}")
    print(f"Saved train split: {train_output_path}")
    print(f"Saved val split: {val_output_path}")
    print(f"Saved test split: {test_output_path}")
    print(f"Saved split report: {split_report_output_path}")
    print(f"Removed duplicates: {removed_count}")
    print(f"Generated synthetic negatives: {len(synthetic_samples)}")
    print(f"Merged duplicates removed: {merged_removed_count}")


if __name__ == "__main__":
    main()
