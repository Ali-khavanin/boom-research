from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import httpx

from ragent.errors import BudgetError, ProviderError
from ragent.llm.base import LLM, Completion, Message, Usage
from ragent.llm.openai_compat import OpenAICompatProvider
from ragent.llm.usage import UsageLedger


def ledger(path: Path, limit: float = 3.5) -> UsageLedger:
    return UsageLedger(
        prices={},
        token_limit=None,
        cost_limit_usd=limit,
        warn_fraction=0.8,
        store_path=path,
        strict=True,
    )


class BudgetTests(unittest.TestCase):
    def test_lifetime_limit_survives_reload_and_session_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            first = ledger(path)
            reservation = first.reserve(
                role="executor",
                label="openrouter/model",
                max_cost_usd=Decimal("3.49"),
            )
            first.settle(reservation, Usage(10, 10, 20, 3.49))
            first.reset_session()
            second = ledger(path)
            with self.assertRaises(BudgetError):
                second.reserve(
                    role="executor",
                    label="openrouter/model",
                    max_cost_usd=Decimal("0.02"),
                )
            self.assertEqual(second.snapshot()["lifetime"]["cost_usd"], 3.49)
            with self.assertRaisesRegex(BudgetError, "cannot be reopened non-strict"):
                UsageLedger(
                    prices={},
                    token_limit=None,
                    cost_limit_usd=None,
                    warn_fraction=0.8,
                    store_path=path,
                )

    def test_outstanding_reservation_blocks_second_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            first = ledger(path)
            first.reserve(
                role="executor",
                label="openrouter/model",
                max_cost_usd=Decimal("0.20"),
            )
            second = ledger(path)
            with self.assertRaisesRegex(BudgetError, "unresolved"):
                second.reserve(
                    role="report",
                    label="openrouter/model",
                    max_cost_usd=Decimal("0.20"),
                )

    def test_malformed_and_unwritable_strict_ledgers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(BudgetError):
                ledger(path)
            parent_file = Path(directory) / "not-a-directory"
            parent_file.write_text("x", encoding="utf-8")
            with self.assertRaises(BudgetError):
                ledger(parent_file / "usage.json")

    def test_settlement_replaces_reservation_and_cannot_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = ledger(Path(directory) / "usage.json")
            reservation = value.reserve(
                role="executor",
                label="openrouter/model",
                max_cost_usd=Decimal("0.20"),
            )
            value.settle(reservation, Usage(100, 25, 125, 0.05))
            snapshot = value.snapshot()
            self.assertEqual(snapshot["campaign_exposure_usd"], 0.05)
            self.assertEqual(snapshot["reserved_cost_usd"], 0)
            with self.assertRaisesRegex(BudgetError, "already settled"):
                value.settle(reservation, Usage(100, 25, 125, 0.05))

    def test_timeout_or_missing_usage_leaves_reservation_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                return httpx.Response(
                    200,
                    request=request,
                    json={"choices": [{"message": {"content": "ok"}}]},
                )

            provider = OpenAICompatProvider(
                "https://example.test/v1",
                "secret",
                report_cost=True,
                strict=True,
                provider_only=["fixture"],
                max_prompt_price_per_million="1",
                max_completion_price_per_million="1",
            )
            provider.client.close()
            provider.client = httpx.Client(
                base_url="https://example.test/v1",
                transport=httpx.MockTransport(handler),
            )
            value = ledger(Path(directory) / "usage.json")
            llm = LLM(
                provider=provider,
                model="model",
                temperature=0,
                max_tokens=100,
                role="executor",
                provider_name="openrouter",
                ledger=value,
                strict_max_cost_usd="0.20",
                strict_max_prompt_tokens=1000,
            )
            with self.assertRaises(BudgetError):
                llm.complete([Message("user", "hello")])
            self.assertEqual(calls, 1)
            self.assertEqual(value.snapshot()["outstanding_reservations"], 1)
            with self.assertRaisesRegex(BudgetError, "unresolved"):
                llm.complete([Message("user", "again")])
            self.assertEqual(calls, 1)
            provider.client.close()

    def test_strict_http_failure_is_one_post(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("fixture timeout", request=request)

        provider = OpenAICompatProvider(
            "https://example.test/v1",
            "secret",
            strict=True,
            report_cost=True,
            provider_only=["fixture"],
            max_prompt_price_per_million="1",
            max_completion_price_per_million="1",
        )
        provider.client.close()
        provider.client = httpx.Client(
            base_url="https://example.test/v1", transport=httpx.MockTransport(handler)
        )
        with self.assertRaises(ProviderError):
            provider.complete(
                model="model",
                messages=[Message("user", "hello")],
                temperature=0,
                max_tokens=10,
                json_schema={"type": "object"},
            )
        self.assertEqual(calls, 1)
        provider.client.close()

    def test_structured_repair_accounts_for_two_requests(self) -> None:
        class RepairProvider:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, **_: object) -> Completion:
                self.calls += 1
                text = "not json" if self.calls == 1 else '{"ok": true}'
                return Completion(text, [], Usage(10, 5, 15, 0.01))

        with tempfile.TemporaryDirectory() as directory:
            provider = RepairProvider()
            value = ledger(Path(directory) / "usage.json")
            llm = LLM(
                provider=provider,
                model="model",
                temperature=0,
                max_tokens=100,
                role="executor",
                provider_name="openrouter",
                ledger=value,
                strict_max_cost_usd="0.20",
                strict_max_prompt_tokens=1000,
            )
            result = llm.json(
                [Message("user", "json")],
                {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            )
            self.assertEqual(result, {"ok": True})
            self.assertEqual(provider.calls, 2)
            self.assertEqual(value.snapshot()["lifetime"]["calls"], 2)
            persisted = json.loads((Path(directory) / "usage.json").read_text())
            self.assertEqual(persisted["reservations"], {})


if __name__ == "__main__":
    unittest.main()
