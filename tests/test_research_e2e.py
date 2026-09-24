from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from ragent.config import DEFAULT_CONFIG, ROLE_NAMES, Config
from ragent.errors import ProviderError, RagentError
from ragent.executor.context import RunContext
from ragent.executor.preflight import preflight
from ragent.executor.runner import run
from ragent.executor.verification import verify_run
from ragent.graph_builder.schema import Edge, Graph, Metric, Node
from ragent.llm.base import LLM, Completion, ToolCall, Usage
from ragent.llm.usage import get_ledger
from ragent.tools import browser
from ragent.trajectories.log import TraceLog

URLS = [f"https://example.test/source-{index}" for index in range(6)]


class FixtureProvider:
    def __init__(self, role: str) -> None:
        self.role = role
        self.calls = 0

    def complete(self, **kwargs: object) -> Completion:
        self.calls += 1
        messages = kwargs["messages"]
        prompt = "\n".join(message.content for message in messages)
        usage = Usage(100, 50, 150, 0.001)
        if self.role == "judge":
            return Completion(
                '{"passed": true, "reason": "all gaps have verdicts"}', [], usage
            )
        if self.role == "report":
            return Completion(messages[-1].content, [], usage)
        if "Research prior work" in prompt:
            if self.calls == 1:
                calls = [
                    ToolCall(
                        "search",
                        "browser__search",
                        {"query": "fixture sources", "k": 6},
                    ),
                    *[
                        ToolCall(f"fetch-{index}", "browser__fetch", {"url": url})
                        for index, url in enumerate(URLS)
                    ],
                ]
                return Completion("", calls, usage)
            citations = "\n".join(
                f"- [Source {index}]({url})" for index, url in enumerate(URLS)
            )
            return Completion(
                "Architecture comparison and synthesis matrix based on fetched evidence.\n"
                + citations,
                [],
                usage,
            )
        if "concrete research goal" in prompt:
            return Completion(" ".join(["goal"] * 75), [], usage)
        if "identify at least three" in prompt:
            return Completion(
                "- Limitation one\n- Limitation two\n- Limitation three", [], usage
            )
        if "specific research gaps" in prompt:
            return Completion(
                "- Gap one matters\n- Gap two matters\n- Gap three matters", [], usage
            )
        if "Assess feasibility" in prompt:
            return Completion(
                "Gap one: data available, method available. Gap two: data available, method available. "
                "Gap three: data available, method available.",
                [],
                usage,
            )
        if "quick discriminating test" in prompt:
            return Completion(" ".join(["test"] * 140), [], usage)
        return Completion("publication complete", [], usage)


def graph() -> Graph:
    node_ids = [
        "start",
        "goal",
        "prior",
        "limitations",
        "gaps",
        "feasibility",
        "quick",
        "done",
    ]
    nodes = [
        Node(id=value, title=value.title(), terminal=value == "done")
        for value in node_ids
    ]
    return Graph(
        entry="start",
        nodes=nodes,
        edges=[
            Edge(
                id="frame_goal",
                source="start",
                target="goal",
                prompt_template="For {query}, write a concrete research goal. {artifacts}",
                termination_metric=Metric(kind="min_words", key="goal", n=60),
                produces="goal",
            ),
            Edge(
                id="review_prior_work",
                source="goal",
                target="prior",
                prompt_template="Research prior work for {query}. {artifacts}",
                tool_set=["browser.search", "browser.fetch"],
                termination_metric=Metric(kind="has_citations", key="prior_work", n=3),
                produces="prior_work",
            ),
            Edge(
                id="identify_limitations",
                source="prior",
                target="limitations",
                prompt_template="identify at least three limitations for {query}",
                termination_metric=Metric(kind="min_items", key="limitations", n=3),
                produces="limitations",
            ),
            Edge(
                id="derive_gaps",
                source="limitations",
                target="gaps",
                prompt_template="derive specific research gaps for {query}",
                termination_metric=Metric(kind="min_items", key="gaps", n=3),
                produces="gaps",
            ),
            Edge(
                id="assess_feasibility",
                source="gaps",
                target="feasibility",
                prompt_template="Assess feasibility for {query}",
                termination_metric=Metric(
                    kind="llm_rubric",
                    key="feasibility",
                    rubric="each gap has data/method availability verdict",
                ),
                produces="feasibility",
            ),
            Edge(
                id="design_quick_test",
                source="feasibility",
                target="quick",
                prompt_template="Design a quick discriminating test for {query}",
                termination_metric=Metric(kind="min_words", key="quick_test", n=120),
                produces="quick_test",
            ),
            Edge(
                id="publish_results",
                source="quick",
                target="done",
                prompt_template="Publish {query}",
                tool_set=["report.generate", "obsidian.note"],
                termination_metric=Metric(kind="artifact_exists", key="report_path"),
                produces="report_path",
            ),
        ],
    )


class ResearchE2ETests(unittest.TestCase):
    def make_config(self, root: Path, *, suffix: str = "one") -> Config:
        cfg = Config.model_validate(DEFAULT_CONFIG)
        cfg.workspace = root / f"workspace-{suffix}"
        for role in ROLE_NAMES:
            cfg.roles[role].provider = "openrouter"
            cfg.roles[role].model = "fixture/model"
            cfg.roles[role].max_tokens = 1000
        cfg.research.query = f"Configured query {suffix}"
        cfg.research.min_sources = 6
        cfg.research.max_steps = 12
        cfg.research.report = True
        cfg.research.obsidian = True
        vault = root / "vault"
        (vault / ".obsidian").mkdir(parents=True, exist_ok=True)
        related = vault / "Related" / f"{suffix}.md"
        related.parent.mkdir(parents=True, exist_ok=True)
        related.write_text(f"related {suffix}\n", encoding="utf-8")
        cfg.obsidian.vault_path = vault
        cfg.obsidian.folder = f"Research/{suffix}"
        cfg.obsidian.layout = "bundle"
        cfg.obsidian.tags = ["Research", f"Tag {suffix}"]
        cfg.obsidian.related_notes = [f"Related/{suffix}"]
        cfg.budget.persist = True
        cfg.budget.strict = False
        cfg.budget.cost_limit_usd = None
        cfg.budget.token_limit = None
        return cfg

    def fake_factory(self, base_cfg: Config):
        def factory(
            role: str, node_override: object = None, cfg: Config | None = None
        ) -> LLM:
            effective = cfg or base_cfg
            return LLM(
                provider=FixtureProvider(role),
                model="fixture/model",
                temperature=0,
                max_tokens=1000,
                role=role,
                provider_name="openrouter",
                ledger=get_ledger(effective),
            )

        return factory

    @staticmethod
    def fake_search(ctx, query: str, k: int | None = None):
        results = [
            {"title": f"Source {index}", "url": url} for index, url in enumerate(URLS)
        ]
        ctx.trace.append(
            {
                "event": "search",
                "query": query,
                "backend": "fixture",
                "result_count": len(results),
                "urls": URLS,
            }
        )
        return results

    @staticmethod
    def fake_fetch(ctx, url: str):
        index = URLS.index(url)
        text = f"Exact evidence excerpt {index}."
        sources = ctx.run_dir / "sources"
        sources.mkdir(exist_ok=True)
        filename = hashlib.sha256(url.encode()).hexdigest()[:16] + ".md"
        evidence = sources / filename
        evidence.write_text(text, encoding="utf-8")
        citation = {
            "url": url,
            "original_url": url,
            "title": f"Source {index}",
            "fetched_at": "2026-09-24T00:00:00+00:00",
            "evidence_path": evidence.relative_to(ctx.run_dir).as_posix(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        ctx.citations.append(citation)
        ctx.source_aliases[url] = url
        (ctx.run_dir / "sources.json").write_text(
            json.dumps(
                {"version": 1, "sources": ctx.citations, "aliases": ctx.source_aliases},
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"url": url, "title": f"Source {index}", "text": text}

    def run_fixture(self, cfg: Config, query: str):
        factory = self.fake_factory(cfg)
        with (
            patch("ragent.executor.runner.get_llm", side_effect=factory),
            patch("ragent.executor.metrics.get_llm", side_effect=factory),
            patch("ragent.tools.report_generator.get_llm", side_effect=factory),
            patch("ragent.tools.browser.search", side_effect=self.fake_search),
            patch("ragent.tools.browser.fetch", side_effect=self.fake_fetch),
        ):
            return run(graph(), query, cfg)

    def test_complete_graph_search_fetch_report_bundle_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.make_config(root)
            result = self.run_fixture(cfg, "First configured execution query")
            self.assertTrue(result.reached_done)
            self.assertTrue(result.verification and result.verification["passed"])
            report = Path(result.report_path or "")
            index = Path(result.obsidian_index_path or "")
            self.assertTrue(report.is_file())
            self.assertTrue(index.is_file())
            self.assertIn("First configured execution query", index.read_text())
            trace = (cfg.workspace / "runs" / result.run_id / "trace.jsonl").read_text()
            self.assertIn('"event": "search"', trace)
            self.assertIn('"role": "judge"', trace)
            self.assertIn('"role": "report"', trace)
            self.assertTrue(
                verify_run(cfg.workspace / "runs" / result.run_id)["passed"]
            )

            second = self.make_config(root, suffix="two")
            second_result = self.run_fixture(
                second, "Second query from changed configuration"
            )
            second_index = Path(second_result.obsidian_index_path or "")
            self.assertIn(
                "Second query from changed configuration", second_index.read_text()
            )
            self.assertIn("tag-two", second_index.read_text())
            self.assertIn("/Research/two/", str(second_index))

    def test_invented_urls_and_fake_paths_do_not_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.make_config(Path(directory))
            run_dir = cfg.workspace / "runs" / "fixture"
            (run_dir / "artifacts").mkdir(parents=True)
            from ragent.executor.context import RunContext
            from ragent.executor.metrics import check
            from ragent.trajectories.log import TraceLog

            ctx = RunContext(
                run_id="fixture",
                query="query",
                workspace=cfg.workspace,
                artifacts={
                    "prior_work": "https://fake.test/a https://fake.test/b https://fake.test/c",
                    "report_path": "saved",
                },
                citations=[],
                cfg=cfg,
                trace=TraceLog(run_dir, "fixture"),
            )
            self.assertFalse(
                check(Metric(kind="has_citations", key="prior_work", n=3), ctx).passed
            )
            self.assertFalse(
                check(Metric(kind="artifact_exists", key="report_path"), ctx).passed
            )

    def make_browser_context(self, root: Path) -> RunContext:
        cfg = self.make_config(root)
        cfg.research.min_sources = 1
        run_dir = cfg.workspace / "runs" / "browser"
        run_dir.mkdir(parents=True)
        return RunContext(
            run_id="browser",
            query="query",
            workspace=cfg.workspace,
            artifacts={},
            citations=[],
            cfg=cfg,
            trace=TraceLog(run_dir, "browser"),
        )

    @staticmethod
    def client_returning(responses):
        client = MagicMock()
        if isinstance(responses, list):
            client.get.side_effect = responses
        else:
            client.get.return_value = responses
        manager = MagicMock()
        manager.__enter__.return_value = client
        manager.__exit__.return_value = False
        return manager

    def test_retrieval_failures_never_become_citations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.make_browser_context(root)
            request = httpx.Request("GET", "https://html.duckduckgo.com/html/")
            for body in (
                "<html>captcha challenge-form</html>",
                "<html>no results</html>",
            ):
                response = httpx.Response(200, text=body, request=request)
                with patch.object(
                    browser.httpx,
                    "Client",
                    return_value=self.client_returning(response),
                ), self.assertRaises(ProviderError):
                    browser.search(ctx, "query")
            empty_html = httpx.Response(
                200,
                text="<html><title>empty</title></html>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", "https://example.test/empty"),
            )
            with (
                patch.object(
                    browser.httpx,
                    "Client",
                    return_value=self.client_returning(empty_html),
                ),
                patch.object(browser.trafilatura, "extract", return_value=""),
            ):
                with self.assertRaises(ProviderError):
                    browser.fetch(ctx, "https://example.test/empty")
            bad_pdf = httpx.Response(
                200,
                content=b"not a pdf",
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", "https://example.test/bad.pdf"),
            )
            with patch.object(
                browser.httpx,
                "Client",
                return_value=self.client_returning(bad_pdf),
            ), self.assertRaises(ProviderError):
                browser.fetch(ctx, "https://example.test/bad.pdf")
            self.assertEqual(ctx.citations, [])

    def test_plain_text_pdf_and_redirect_dedup_persist_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctx = self.make_browser_context(root)
            plain = httpx.Response(
                200,
                text="plain evidence",
                headers={"content-type": "text/plain"},
                request=httpx.Request("GET", "https://canonical.test/plain"),
            )
            with patch.object(
                browser.httpx,
                "Client",
                return_value=self.client_returning(plain),
            ):
                browser.fetch(ctx, "https://requested.test/plain")

            class Page:
                def extract_text(self) -> str:
                    return "pdf evidence"

            readable_pdf = httpx.Response(
                200,
                content=b"%PDF-fixture",
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", "https://canonical.test/paper.pdf"),
            )
            with (
                patch.object(
                    browser.httpx,
                    "Client",
                    return_value=self.client_returning(readable_pdf),
                ),
                patch.object(
                    browser, "PdfReader", return_value=MagicMock(pages=[Page()])
                ),
            ):
                browser.fetch(ctx, "https://requested.test/paper.pdf")
            self.assertEqual(len(ctx.citations), 2)
            self.assertEqual(
                ctx.source_aliases["https://requested.test/plain"],
                "https://canonical.test/plain",
            )
            index = json.loads((ctx.run_dir / "sources.json").read_text())
            self.assertEqual(index["version"], 1)
            for citation in ctx.citations:
                evidence = ctx.run_dir / citation["evidence_path"]
                self.assertEqual(
                    hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    citation["text_sha256"],
                )
            duplicate = httpx.Response(
                200,
                text="changed response must not replace snapshot",
                headers={"content-type": "text/plain"},
                request=httpx.Request("GET", "https://canonical.test/plain"),
            )
            with patch.object(
                browser.httpx,
                "Client",
                return_value=self.client_returning(duplicate),
            ):
                result = browser.fetch(ctx, "https://another.test/plain")
            self.assertEqual(len(ctx.citations), 2)
            self.assertEqual(result["text"], "plain evidence")

    def test_missing_preflight_inputs_create_no_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.make_config(root)
            cfg.research.skill_path = root / "missing-skill.md"
            with self.assertRaises((RagentError, OSError)):
                preflight(graph(), cfg)
            self.assertFalse((cfg.workspace / "runs").exists())

            cfg.research.skill_path = None
            cfg.research.obsidian = False
            cfg.budget.strict = True
            cfg.budget.cost_limit_usd = 3.5
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RagentError):
                    preflight(graph(), cfg)
            self.assertFalse((cfg.workspace / "runs").exists())

    def test_verifier_detects_and_recovers_from_output_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.make_config(Path(directory))
            result = self.run_fixture(cfg, "Tamper verification query")
            run_dir = cfg.workspace / "runs" / result.run_id
            manifest_path = Path(result.artifacts["obsidian_manifest_path"])
            bundle_report = manifest_path.parent / "report.md"
            bundle_bytes = bundle_report.read_bytes()
            bundle_report.unlink()
            failed = verify_run(run_dir)
            self.assertFalse(failed["passed"])
            self.assertTrue(
                any("bundle note report.md" in item for item in failed["errors"])
            )
            bundle_report.write_bytes(bundle_bytes)

            source = next((run_dir / "sources").glob("*.md"))
            source_bytes = source.read_bytes()
            source.write_text("tampered", encoding="utf-8")
            failed = verify_run(run_dir)
            self.assertFalse(failed["passed"])
            self.assertTrue(
                any(
                    "source evidence hash mismatch" in item for item in failed["errors"]
                )
            )
            source.write_bytes(source_bytes)

            report = Path(result.artifacts["report_path"])
            report_bytes = report.read_bytes()
            report.write_text(
                report.read_text(encoding="utf-8")
                + "\nUnfetched https://unfetched.test/source\n",
                encoding="utf-8",
            )
            failed = verify_run(run_dir)
            self.assertFalse(failed["passed"])
            self.assertTrue(
                any("report source provenance" in item for item in failed["errors"])
            )
            report.write_bytes(report_bytes)
            self.assertTrue(verify_run(run_dir)["passed"])

    def test_publication_exception_records_failure_without_terminal_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.make_config(Path(directory))
            factory = self.fake_factory(cfg)
            with (
                patch("ragent.executor.runner.get_llm", side_effect=factory),
                patch("ragent.executor.metrics.get_llm", side_effect=factory),
                patch("ragent.tools.browser.search", side_effect=self.fake_search),
                patch("ragent.tools.browser.fetch", side_effect=self.fake_fetch),
                patch(
                    "ragent.tools.report_generator.generate",
                    side_effect=RuntimeError("report fixture failure"),
                ),self.assertRaisesRegex(RuntimeError, "report fixture failure")
            ):
                run(graph(), "Failure query", cfg)
            run_dirs = list((cfg.workspace / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            metadata = json.loads((run_dirs[0] / "run.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            events = [
                json.loads(line)
                for line in (run_dirs[0] / "trace.jsonl").read_text().splitlines()
                if line
            ]
            self.assertFalse(
                any(
                    event.get("target") == "done"
                    and (event.get("metric") or {}).get("passed")
                    for event in events
                )
            )


if __name__ == "__main__":
    unittest.main()
