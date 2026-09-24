from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ragent.config import DEFAULT_CONFIG, Config
from ragent.errors import ExportError
from ragent.executor.context import RunContext
from ragent.tools.obsidian_mcp import export_bundle
from ragent.trajectories.log import TraceLog


class BundleTests(unittest.TestCase):
    def make_context(
        self,
        root: Path,
        *,
        folder: str = "Research/Test",
        tags: list[str] | None = None,
    ) -> RunContext:
        workspace = root / "workspace"
        run_id = "20260924T000000Z-fixture"
        run_dir = workspace / "runs" / run_id
        (run_dir / "artifacts").mkdir(parents=True)
        vault = root / "vault"
        (vault / ".obsidian").mkdir(parents=True)
        (vault / "Related").mkdir()
        (vault / "Related" / "Existing.md").write_text("existing\n", encoding="utf-8")
        cfg = Config.model_validate(DEFAULT_CONFIG)
        cfg.workspace = workspace
        cfg.research.min_sources = 1
        cfg.obsidian.vault_path = vault
        cfg.obsidian.folder = folder
        cfg.obsidian.layout = "bundle"
        cfg.obsidian.tags = tags or ["Research", "Agent Systems"]
        cfg.obsidian.related_notes = ["Related/Existing"]
        evidence = "Exact source excerpt with [[Bad/Injected]] and ``` nested fence."
        evidence_path = run_dir / "sources" / "fixture.md"
        evidence_path.parent.mkdir()
        evidence_path.write_text(evidence, encoding="utf-8")
        url = "https://example.test/source"
        stages = {
            "goal": "A sufficiently explicit goal.",
            "prior_work": f"Prior evidence [{url}]({url}) with [[Invented/Note|bad link]].",
            "limitations": "- Limitation one\n- Limitation two\n- Limitation three",
            "gaps": "- Gap one\n- Gap two\n- Gap three",
            "feasibility": "Data and methods are available.",
            "quick_test": "A concrete quick test design.",
        }
        report = (
            "# Research Report\n\n"
            + "\n\n".join(
                f"## {heading}\n\n{stages[key]}"
                for heading, key in (
                    ("Goal", "goal"),
                    ("Prior Work", "prior_work"),
                    ("Limitations", "limitations"),
                    ("Gaps", "gaps"),
                    ("Feasibility", "feasibility"),
                    ("Quick Test", "quick_test"),
                )
            )
            + f"\n\n## References\n\n- [{url}]({url})\n"
        )
        report_path = run_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        ctx = RunContext(
            run_id=run_id,
            query='Malicious "title"\nvalue [[Outside/Target]]',
            workspace=workspace,
            artifacts={**stages, "report_path": str(report_path)},
            citations=[
                {
                    "url": url,
                    "original_url": url,
                    "title": 'Source "title" [[Injected/Target]]',
                    "fetched_at": "2026-09-24T00:00:00+00:00",
                    "evidence_path": "sources/fixture.md",
                    "text_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
                }
            ],
            cfg=cfg,
            trace=TraceLog(run_dir, run_id),
            source_aliases={url: url},
        )
        return ctx

    def test_bundle_is_connected_safe_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.make_context(Path(directory))
            manifest = Path(export_bundle(ctx))
            first = {
                path.relative_to(manifest.parent): path.read_bytes()
                for path in manifest.parent.rglob("*")
                if path.is_file()
            }
            self.assertTrue((manifest.parent / "index.md").is_file())
            self.assertIn("agent-systems", (manifest.parent / "index.md").read_text())
            prior = (manifest.parent / "stages" / "prior_work.md").read_text()
            self.assertNotIn("[[Invented/Note", prior)
            self.assertIn("bad link", prior)
            source = next((manifest.parent / "sources").glob("*.md")).read_text()
            self.assertIn("Exact source excerpt with [[Bad/Injected]]", source)
            self.assertEqual(export_bundle(ctx), str(manifest))
            second = {
                path.relative_to(manifest.parent): path.read_bytes()
                for path in manifest.parent.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first, second)
            self.assertEqual(
                (Path(directory) / "vault" / "Related" / "Existing.md").read_text(),
                "existing\n",
            )

    def test_edited_same_run_note_and_collision_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.make_context(Path(directory))
            manifest = Path(export_bundle(ctx))
            (manifest.parent / "index.md").write_text("edited", encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "modified|conflicts"):
                export_bundle(ctx)
        with tempfile.TemporaryDirectory() as directory:
            ctx = self.make_context(Path(directory))
            destination = (
                ctx.cfg.obsidian.vault_path / ctx.cfg.obsidian.folder / ctx.run_id
            )
            destination.mkdir(parents=True)
            (destination / "user.md").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "without a manifest"):
                export_bundle(ctx)
            self.assertEqual((destination / "user.md").read_text(), "preserve")

    def test_unsafe_folders_and_invalid_tags_fail_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("../escape", "/absolute"):
                ctx = self.make_context(root / folder.replace("/", "_"), folder=folder)
                with self.assertRaises(ExportError):
                    export_bundle(ctx)
            ctx = self.make_context(root / "tags", tags=["ok", "bad:tag"])
            with self.assertRaisesRegex(ExportError, "invalid Obsidian tag"):
                export_bundle(ctx)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.make_context(root, folder="linked/out")
            outside = root / "outside"
            outside.mkdir()
            linked = ctx.cfg.obsidian.vault_path / "linked"
            linked.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ExportError, "escapes"):
                export_bundle(ctx)
            self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
