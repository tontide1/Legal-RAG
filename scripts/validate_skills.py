from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


FRONTMATTER_BOUNDARY = "---"
REQUIRED_FRONTMATTER_KEYS = {"name", "description"}
SKILL_SECTION_PATTERNS = (
    "related skill",
    "related skills",
    "route to other skills",
    "default toolchain",
    "prefer these skills first",
    "in this repo, prefer",
)
HEADING_RE = re.compile(r"^(#+)\s+(.*)$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
PATH_LIKE_RE = re.compile(r"^[A-Za-z0-9_./-]+\.[A-Za-z0-9]+$")
SKILL_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass
class ValidationMessage:
    level: str
    path: Path
    text: str


def parse_frontmatter(skill_file: Path) -> tuple[dict[str, str], list[str]]:
    lines = skill_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[0].strip() != FRONTMATTER_BOUNDARY:
        raise ValueError("missing YAML frontmatter boundary")

    try:
        end_index = lines[1:].index(FRONTMATTER_BOUNDARY) + 1
    except ValueError as exc:
        raise ValueError("unterminated YAML frontmatter") from exc

    frontmatter_lines = lines[1:end_index]
    body_lines = lines[end_index + 1 :]
    data: dict[str, str] = {}
    for line in frontmatter_lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data, body_lines


def referenced_paths(skill_dir: Path, body_lines: list[str]) -> set[str]:
    refs: set[str] = set()
    for line in body_lines:
        for token in BACKTICK_RE.findall(line):
            if PATH_LIKE_RE.match(token):
                refs.add(token)
    return refs


def reference_path_exists(repo_root: Path, skill_dir: Path, rel_path: str) -> bool:
    local_candidate = skill_dir / rel_path
    repo_candidate = repo_root / rel_path
    if local_candidate.exists() or repo_candidate.exists():
        return True

    path_obj = Path(rel_path)
    if len(path_obj.parts) == 1:
        matches = list(repo_root.rglob(rel_path))
        if matches:
            return True

    return False


def referenced_skills(body_lines: list[str], known_skill_names: set[str]) -> set[str]:
    refs: set[str] = set()
    in_skill_section = False

    for line in body_lines:
        heading_match = HEADING_RE.match(line)
        if heading_match:
            heading_text = heading_match.group(2).strip().lower()
            in_skill_section = any(pattern in heading_text for pattern in SKILL_SECTION_PATTERNS)
            continue

        lowered = line.strip().lower()
        line_mentions_skills = "skill" in lowered or in_skill_section
        if not line_mentions_skills:
            continue

        for token in BACKTICK_RE.findall(line):
            if SKILL_NAME_RE.match(token) and token in known_skill_names:
                refs.add(token)
            elif SKILL_NAME_RE.match(token) and not any(ch.isdigit() for ch in token):
                refs.add(token)

    return refs


def collect_codex_skills(repo_root: Path) -> dict[str, Path]:
    skill_map: dict[str, Path] = {}
    for skill_file in sorted((repo_root / ".codex" / "skills").glob("*/SKILL.md")):
        frontmatter, _ = parse_frontmatter(skill_file)
        skill_name = frontmatter.get("name")
        if skill_name:
            skill_map[skill_name] = skill_file
    return skill_map


def validate_codex_skills(repo_root: Path) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    codex_root = repo_root / ".codex" / "skills"
    skill_files = sorted(codex_root.glob("*/SKILL.md"))
    known_skill_names = {path.parent.name for path in skill_files}

    for skill_file in skill_files:
        try:
            frontmatter, body_lines = parse_frontmatter(skill_file)
        except ValueError as exc:
            messages.append(ValidationMessage("error", skill_file, str(exc)))
            continue

        missing_keys = REQUIRED_FRONTMATTER_KEYS - set(frontmatter)
        if missing_keys:
            messages.append(
                ValidationMessage(
                    "error",
                    skill_file,
                    f"missing required frontmatter keys: {', '.join(sorted(missing_keys))}",
                )
            )

        skill_name = frontmatter.get("name")
        if skill_name and skill_name != skill_file.parent.name:
            messages.append(
                ValidationMessage(
                    "warning",
                    skill_file,
                    f"frontmatter name '{skill_name}' differs from directory name '{skill_file.parent.name}'",
                )
            )

        line_count = len(body_lines)
        if line_count > 500:
            messages.append(
                ValidationMessage(
                    "warning",
                    skill_file,
                    f"body is {line_count} lines; consider moving detail into references/",
                )
            )

        for rel_path in sorted(referenced_paths(skill_file.parent, body_lines)):
            if not reference_path_exists(repo_root, skill_file.parent, rel_path):
                messages.append(
                    ValidationMessage(
                        "error",
                        skill_file,
                        f"referenced path does not exist: {rel_path}",
                    )
                )

        for ref_name in sorted(referenced_skills(body_lines, known_skill_names)):
            if ref_name not in known_skill_names:
                messages.append(
                    ValidationMessage(
                        "error",
                        skill_file,
                        f"referenced skill does not exist in .codex/skills: {ref_name}",
                    )
                )

    for extra_file in sorted(codex_root.glob("*/CREATION-LOG.md")):
        messages.append(
            ValidationMessage(
                "warning",
                extra_file,
                "auxiliary creation log should not live in the skill bundle",
            )
        )

    return messages


def detect_agent_duplicates(repo_root: Path) -> list[ValidationMessage]:
    messages: list[ValidationMessage] = []
    codex_names = {
        path.parent.name
        for path in sorted((repo_root / ".codex" / "skills").glob("*/SKILL.md"))
    }
    for skill_file in sorted((repo_root / ".agents" / "skills").glob("*/SKILL.md")):
        skill_name = skill_file.parent.name
        if skill_name in codex_names:
            messages.append(
                ValidationMessage(
                    "warning",
                    skill_file,
                    f"duplicate skill also exists in .codex/skills: {skill_name}",
                )
            )
    return messages


def format_message(message: ValidationMessage) -> str:
    rel_path = message.path.as_posix()
    return f"[{message.level.upper()}] {rel_path}: {message.text}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local Codex skill bundles.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing .codex/skills and optionally .agents/skills",
    )
    parser.add_argument(
        "--fail-on-agent-duplicates",
        action="store_true",
        help="Treat duplicate .agents skill names as errors",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    messages = validate_codex_skills(repo_root)
    duplicate_messages = detect_agent_duplicates(repo_root)
    if args.fail_on_agent_duplicates:
        duplicate_messages = [
            ValidationMessage("error", message.path, message.text) for message in duplicate_messages
        ]
    messages.extend(duplicate_messages)

    for message in messages:
        print(format_message(message))

    error_count = sum(message.level == "error" for message in messages)
    if error_count:
        print(f"Skill validation failed with {error_count} error(s).", file=sys.stderr)
        return 1

    warning_count = sum(message.level == "warning" for message in messages)
    if warning_count:
        print(f"Skill validation passed with {warning_count} warning(s).")
    else:
        print("Skill validation passed with no warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
