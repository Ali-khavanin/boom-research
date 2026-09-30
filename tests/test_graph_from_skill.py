from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ragent.errors import RagentError
from ragent.graph_builder import audit, build_graph, load_seed
from ragent.llm.base import Completion, LLM, Message


class RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[list[Message]] = []

    def complete(self, **kwargs) -> Completion:
        self.messages.append(kwargs["messages"])
        proposal = {
            "version": 1,
            "entry": "start",
            "nodes": [{"id": "screen_sources", "title": "Screen sources",
                       "provenance": {"chapter": "model-supplied.md", "cue": "screen"}}],
            "edges": [
                {"id": edge_id, "source": source, "target": target,
                 "prompt_template": "Screen {query}",
                 "termination_metric": {"kind": "min_words", "n": 10},
                 "provenance": {"chapter": "model-supplied.md", "cue": "screen"}}
                for edge_id, source, target in (
                    ("begin_screen", "what_has_been_done", "screen_sources"),
                    ("finish_screen", "screen_sources", "limitations"),
                )
            ],
        }
        return Completion(text=json.dumps(proposal), tool_calls=[])


class GraphFromSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "chapters").mkdir()
        for name, text in {
            "SKILL.md": "SKILL-SENTINEL",
            "glossary.md": "GLOSSARY-SENTINEL",
            "patterns.md": "# Patterns",
            "cheatsheet.md": "# Cheatsheet",
        }.items():
            (self.root / name).write_text(text, encoding="utf-8")
        for name in ("ch10-c.md", "ch2-b.md", "ch1-a.md"):
            (self.root / "chapters" / name).write_text(
                "# Sources\n\n## Core Idea\nscreen sources\n\n"
                "## Frameworks Introduced\nWhen to use: before synthesis\nHow: screen then compare\n",
                encoding="utf-8",
            )
        self.provider = RecordingProvider()
        self.llm = LLM(provider=self.provider, model="fake", temperature=0.0,
                       max_tokens=4096, role="graph_builder")

    def test_compiles_only_numeric_chapters_and_keeps_first_provenance(self) -> None:
        graph = build_graph(self.root, self.llm, load_seed())
        users = [next(message.content for message in messages if message.role == "user")
                 for messages in self.provider.messages]
        self.assertEqual(len(users), 3)
        for content, chapter in zip(users, ("ch1-a.md", "ch2-b.md", "ch10-c.md")):
            self.assertIn(f"Chapter: {chapter}", content)
        for messages in self.provider.messages:
            for message in messages:
                self.assertNotIn("SKILL-SENTINEL", message.content)
                self.assertNotIn("GLOSSARY-SENTINEL", message.content)
        added = [node for node in graph.nodes if node.id == "screen_sources"]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].provenance, {"chapter": "ch1-a.md", "cue": "screen"})
        edges = [edge for edge in graph.edges if edge.id in {"begin_screen", "finish_screen"}]
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(edge.provenance == {"chapter": "ch1-a.md", "cue": "screen"}
                            for edge in edges))
        self.assertTrue(audit(graph).ok)

    def test_incomplete_skill_fails_before_provider_call(self) -> None:
        (self.root / "SKILL.md").unlink()
        with self.assertRaisesRegex(RagentError, "incomplete book-to-skill output.*SKILL[.]md"):
            build_graph(self.root, self.llm)
        self.assertEqual(self.provider.messages, [])


if __name__ == "__main__":
    unittest.main()
