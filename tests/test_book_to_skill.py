from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from ragent.book_to_skill import find_skill, generate_skill, load_skill
from ragent.config import DEFAULT_CONFIG, BookCfg, Config
from ragent.errors import RagentError


_FAKE_HOST = r'''import os
import sys
from pathlib import Path

with Path(__file__).with_name("calls.txt").open("a", encoding="utf-8") as calls:
    calls.write("called\n")

slug = sys.argv[1].splitlines()[0].split()[-1]
print("fake agent done", flush=True)
if os.environ.get("FAKE_EXIT") == "3":
    sys.exit(3)

root = Path.cwd() / slug
root.mkdir(parents=True, exist_ok=True)
(root / "SKILL.md").write_text("# Generated skill\n", encoding="utf-8")
if os.environ.get("FAKE_PARTIAL") == "1":
    sys.exit(0)

(root / "chapters").mkdir(exist_ok=True)
(root / "chapters" / "ch01-intro.md").write_text(
    "# Introduction\n\n## Core Idea\nRead the source.\n", encoding="utf-8"
)
for name in ("glossary.md", "patterns.md", "cheatsheet.md"):
    (root / name).write_text("# " + name + "\n", encoding="utf-8")
'''


class BookToSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "08IJBAS31.pdf"
        self.source.write_bytes(b"fixture source")
        self.skills_root = self.root / "skills"
        self.host = self.root / "fake_host.py"
        self.host.write_text(_FAKE_HOST, encoding="utf-8")
        self.calls = self.root / "calls.txt"
        self.cfg = BookCfg(agent=[sys.executable, str(self.host), "{prompt}"])
        environment = patch.dict(os.environ, {"FAKE_EXIT": "0", "FAKE_PARTIAL": "0"})
        environment.start()
        self.addCleanup(environment.stop)

    def make_skill(self, name: str, chapters: tuple[str, ...]) -> Path:
        root = self.skills_root / name
        (root / "chapters").mkdir(parents=True)
        for filename in ("SKILL.md", "glossary.md", "patterns.md", "cheatsheet.md"):
            (root / filename).write_text("# Fixture\n", encoding="utf-8")
        for filename in chapters:
            (root / "chapters" / filename).write_text("# Chapter\n", encoding="utf-8")
        return root

    def test_subprocess_generates_complete_skill_and_streams_output(self) -> None:
        events = []
        artifacts = generate_skill(
            self.source, self.skills_root, self.cfg, on_event=events.append
        )
        target = self.skills_root / "08ijbas31"
        self.assertEqual(artifacts.root, target)
        self.assertEqual(artifacts.skill_file, target / "SKILL.md")
        self.assertEqual(artifacts.chapter_files, [target / "chapters" / "ch01-intro.md"])
        self.assertEqual(
            artifacts.supporting_files,
            [target / name for name in ("glossary.md", "patterns.md", "cheatsheet.md")],
        )
        for path in [artifacts.skill_file, *artifacts.chapter_files, *artifacts.supporting_files]:
            self.assertTrue(path.is_file(), str(path))
        self.assertIn({"event": "agent_output", "line": "fake agent done"}, events)
        self.assertEqual(
            events[-1],
            {"event": "skill_done", "path": str(target / "SKILL.md"), "chapters": 1, "cached": False},
        )

    def test_cached_rerun_skips_host_and_force_replaces_output(self) -> None:
        original = generate_skill(self.source, self.skills_root, self.cfg)
        stale = original.root / "stale.md"
        stale.write_text("old output", encoding="utf-8")
        events = []
        cached = generate_skill(
            self.source, self.skills_root, self.cfg, on_event=events.append
        )
        self.assertEqual(cached, original)
        self.assertEqual(self.calls.read_text(encoding="utf-8").splitlines(), ["called"])
        self.assertTrue(stale.exists())
        self.assertEqual(
            events,
            [{"event": "skill_done", "path": str(original.skill_file), "chapters": 1, "cached": True}],
        )
        events.clear()
        regenerated = generate_skill(
            self.source, self.skills_root, self.cfg, force=True, on_event=events.append
        )
        self.assertEqual(regenerated, original)
        self.assertEqual(
            self.calls.read_text(encoding="utf-8").splitlines(), ["called", "called"]
        )
        self.assertFalse(stale.exists())
        self.assertFalse(events[-1]["cached"])

    def test_nonzero_host_exit_reports_status(self) -> None:
        with patch.dict(os.environ, {"FAKE_EXIT": "3"}):
            with self.assertRaisesRegex(RagentError, "status 3"):
                generate_skill(self.source, self.skills_root, self.cfg)
        self.assertEqual(self.calls.read_text(encoding="utf-8").splitlines(), ["called"])

    def test_successful_host_with_partial_output_is_rejected(self) -> None:
        with patch.dict(os.environ, {"FAKE_PARTIAL": "1"}):
            with self.assertRaisesRegex(
                RagentError, "incomplete book-to-skill output.*glossary[.]md"
            ):
                generate_skill(self.source, self.skills_root, self.cfg)
        self.assertTrue((self.skills_root / "08ijbas31" / "SKILL.md").is_file())

    def test_missing_host_reports_path_error(self) -> None:
        cfg = BookCfg(agent=["ragent-no-such-agent-xyz", "{prompt}"])
        with self.assertRaisesRegex(RagentError, "not found on PATH"):
            generate_skill(self.source, self.skills_root, cfg)
        self.assertFalse(self.calls.exists())

    def test_load_skill_orders_chapters_numerically_and_ignores_notes(self) -> None:
        root = self.make_skill("ordered", ("ch10-c.md", "notes.md", "ch2-b.md", "ch1-a.md"))
        artifacts = load_skill(root)
        self.assertEqual(
            artifacts.chapter_files,
            [root / "chapters" / name for name in ("ch1-a.md", "ch2-b.md", "ch10-c.md")],
        )

    def test_find_skill_requires_exactly_one_candidate(self) -> None:
        with self.assertRaisesRegex(RagentError, "no generated skill.*--skill-dir"):
            find_skill(self.skills_root)
        first = self.make_skill("alpha", ("ch01-intro.md",))
        self.assertEqual(find_skill(self.skills_root), first)
        self.make_skill("beta", ("ch01-intro.md",))
        with self.assertRaisesRegex(RagentError, "multiple skills.*alpha, beta.*--skill-dir"):
            find_skill(self.skills_root)

    def test_config_rejects_agent_without_prompt_placeholder(self) -> None:
        with self.assertRaises(ValidationError):
            Config.model_validate({**DEFAULT_CONFIG, "book": {"agent": ["hermes"]}})


if __name__ == "__main__":
    unittest.main()
