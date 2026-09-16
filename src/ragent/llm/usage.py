from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ragent.config import Config
from ragent.errors import BudgetError

from .base import Usage

_UNSET = object()


@dataclass(slots=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


@dataclass(slots=True)
class Totals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0


def _fmt_int(value: int) -> str:
    return f"{value:,}"


class UsageLedger:
    def __init__(
        self,
        *,
        prices: dict[str, ModelPrice],
        token_limit: int | None,
        cost_limit_usd: float | None,
        warn_fraction: float,
        store_path: Path | None,
    ) -> None:
        self.prices = dict(prices)
        self.token_limit = token_limit
        self.cost_limit_usd = cost_limit_usd
        self.warn_fraction = warn_fraction
        self.store_path = store_path
        self.session = Totals()
        self.lifetime = Totals()
        self.by_model: dict[str, Totals] = {}
        self.unpriced_models: set[str] = set()
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if self.store_path is None:
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        try:
            self.lifetime = Totals(**raw.get("lifetime", {}))
            self.by_model = {
                key: Totals(**value) for key, value in raw.get("by_model", {}).items()
            }
        except TypeError:
            self.lifetime = Totals()
            self.by_model = {}

    def _persist(self) -> None:
        if self.store_path is None:
            return
        payload = {
            "version": 1,
            "lifetime": asdict(self.lifetime),
            "by_model": {key: asdict(value) for key, value in self.by_model.items()},
        }
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.store_path)
        except OSError:
            pass

    # -- pricing -----------------------------------------------------------

    def _resolve_price(self, label: str) -> ModelPrice | None:
        if label in self.prices:
            return self.prices[label]
        if "/" in label:
            bare = label.split("/", 1)[1]
            if bare in self.prices:
                return self.prices[bare]
        return None

    # -- accounting ----------------------------------------------------------

    def precheck(self, *, role: str, label: str) -> None:
        with self._lock:
            if self.token_limit is not None and self.session.total_tokens >= self.token_limit:
                raise BudgetError(
                    f"token budget reached: {self.session.total_tokens}/{self.token_limit} tokens "
                    f"spent; raise it with '/budget tokens N' or clear limits with '/budget off' "
                    f"(blocked role {role} on {label})"
                )
            if self.cost_limit_usd is not None and self.session.cost_usd >= self.cost_limit_usd:
                raise BudgetError(
                    f"cost budget reached: ${self.session.cost_usd:.4f}/${self.cost_limit_usd:.2f} "
                    f"spent; raise it with '/budget cost USD' or clear limits with '/budget off' "
                    f"(blocked role {role} on {label})"
                )

    def record(self, *, role: str, label: str, usage: Usage) -> dict[str, Any]:
        with self._lock:
            total_tokens = usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)
            cost_usd: float | None
            if usage.cost_usd is not None:
                cost_usd = usage.cost_usd
            else:
                price = self._resolve_price(label)
                if price is not None:
                    cost_usd = (
                        usage.prompt_tokens / 1_000_000 * price.input_per_mtok
                        + usage.completion_tokens / 1_000_000 * price.output_per_mtok
                    )
                else:
                    cost_usd = None

            for totals in (self.session, self.lifetime, self.by_model.setdefault(label, Totals())):
                totals.calls += 1
                totals.prompt_tokens += usage.prompt_tokens
                totals.completion_tokens += usage.completion_tokens
                totals.total_tokens += total_tokens
                if cost_usd is not None:
                    totals.cost_usd += cost_usd
                else:
                    totals.unpriced_calls += 1

            if cost_usd is None:
                self.unpriced_models.add(label)

            self._persist()

            warn = False
            if self.token_limit is not None and self.session.total_tokens >= self.token_limit * self.warn_fraction:
                warn = True
            if (
                self.cost_limit_usd is not None
                and self.session.cost_usd >= self.cost_limit_usd * self.warn_fraction
            ):
                warn = True

            event = {
                "event": "usage",
                "role": role,
                "model": label,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "warn": warn,
                **self._snapshot_locked(),
            }
        self._emit(event)
        return event

    # -- limits / pricing mutation ---------------------------------------

    def set_limits(
        self,
        *,
        token_limit: int | None | object = _UNSET,
        cost_limit_usd: float | None | object = _UNSET,
    ) -> None:
        with self._lock:
            if token_limit is not _UNSET:
                self.token_limit = token_limit  # type: ignore[assignment]
            if cost_limit_usd is not _UNSET:
                self.cost_limit_usd = cost_limit_usd  # type: ignore[assignment]
            event = {
                "event": "limits",
                "role": None,
                "model": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None,
                "warn": False,
                **self._snapshot_locked(),
            }
        self._emit(event)

    def set_price(self, key: str, price: ModelPrice) -> None:
        with self._lock:
            self.prices[key] = price
            event = {
                "event": "price",
                "role": None,
                "model": key,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None,
                "warn": False,
                **self._snapshot_locked(),
            }
        self._emit(event)

    def reset_session(self) -> None:
        with self._lock:
            self.session = Totals()
            event = {
                "event": "reset",
                "role": None,
                "model": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None,
                "warn": False,
                **self._snapshot_locked(),
            }
        self._emit(event)

    # -- subscriptions -----------------------------------------------------

    def subscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:
                pass

    # -- reporting -----------------------------------------------------------

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "session": asdict(self.session),
            "lifetime": asdict(self.lifetime),
            "by_model": {key: asdict(value) for key, value in self.by_model.items()},
            "token_limit": self.token_limit,
            "cost_limit_usd": self.cost_limit_usd,
            "unpriced": sorted(self.unpriced_models),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def status_line(self, markup: bool = True) -> str:
        with self._lock:
            session = self.session
            lifetime = self.lifetime
            token_limit = self.token_limit
            cost_limit_usd = self.cost_limit_usd
            warn_fraction = self.warn_fraction
            unpriced_calls = session.unpriced_calls

        parts = [
            f"tokens {_fmt_int(session.total_tokens)} "
            f"(in {_fmt_int(session.prompt_tokens)} / out {_fmt_int(session.completion_tokens)})",
            f"bill ${session.cost_usd:,.4f}",
        ]

        limit_clauses: list[str] = []
        if token_limit is not None:
            fraction = (session.total_tokens / token_limit) if token_limit else 0.0
            warn_now = fraction >= warn_fraction
            clause = f"limit {_fmt_int(token_limit)} tokens ({fraction * 100:.1f}%)"
            if markup:
                color = "red" if warn_now else "green"
                clause = f"[{color}]{clause}[/{color}]"
            limit_clauses.append(clause)
        if cost_limit_usd is not None:
            fraction = (session.cost_usd / cost_limit_usd) if cost_limit_usd else 0.0
            warn_now = fraction >= warn_fraction
            clause = f"limit ${cost_limit_usd:,.2f} ({fraction * 100:.1f}%)"
            if markup:
                color = "red" if warn_now else "green"
                clause = f"[{color}]{clause}[/{color}]"
            limit_clauses.append(clause)
        parts.extend(limit_clauses)

        parts.append(f"session {_fmt_int(session.calls)} calls")
        parts.append(f"lifetime {_fmt_int(lifetime.total_tokens)} tokens / ${lifetime.cost_usd:,.4f}")

        line = " · ".join(parts)
        if unpriced_calls > 0:
            line += f" · {unpriced_calls} calls unpriced (set /budget price)"
        return line


_LEDGERS: dict[str, UsageLedger] = {}
_LAST_LEDGER_KEY: str | None = None


def get_ledger(cfg: Config) -> UsageLedger:
    global _LAST_LEDGER_KEY
    key = str(Path(cfg.workspace).resolve())
    ledger = _LEDGERS.get(key)
    if ledger is None:
        prices = {
            model: ModelPrice(price.input_per_mtok, price.output_per_mtok)
            for model, price in cfg.prices.items()
        }
        store_path = (
            Path(cfg.workspace) / "usage.json" if cfg.budget.persist else None
        )
        ledger = UsageLedger(
            prices=prices,
            token_limit=cfg.budget.token_limit,
            cost_limit_usd=cfg.budget.cost_limit_usd,
            warn_fraction=cfg.budget.warn_fraction,
            store_path=store_path,
        )
        _LEDGERS[key] = ledger
    _LAST_LEDGER_KEY = key
    return ledger


def current_ledger() -> UsageLedger | None:
    if _LAST_LEDGER_KEY is None:
        return None
    return _LEDGERS.get(_LAST_LEDGER_KEY)
