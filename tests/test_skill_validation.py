import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SkillValidationTest(unittest.TestCase):
    def test_skill_validator_passes_for_codex_and_opencode_skills(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/validate_skills.py", "--repo-root", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_project_specific_skills_exist_in_codex_and_opencode(self) -> None:
        self.assertTrue((REPO_ROOT / ".codex/skills/legal-graph-rag/SKILL.md").exists())
        self.assertTrue((REPO_ROOT / ".codex/skills/vietnamese-legal-nlp/SKILL.md").exists())
        self.assertTrue((REPO_ROOT / ".opencode/skills/legal-graph-rag/SKILL.md").exists())
        self.assertTrue((REPO_ROOT / ".opencode/skills/vietnamese-legal-nlp/SKILL.md").exists())


if __name__ == "__main__":
    unittest.main()
