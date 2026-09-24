from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from ragent.config import Config
from ragent.errors import BudgetError

from .base import Usage

try:
    import fcntl
except ImportError:  # pragma: no cover - strict mode is intentionally Unix-only
    fcntl = None  # type: ignore[assignment]

_UNSET = object()
_ZERO = Decimal(0)


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
    cost_usd: Decimal = _ZERO
    unpriced_calls: int = 0


@dataclass(slots=True)
class Reservation:
    role: str
    label: str
    max_cost_usd: Decimal


def _decimal(value: Any, *, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BudgetError(f"invalid {name}: {value!r}") from exc
    if not result.is_finite():
        raise BudgetError(f"invalid {name}: value must be finite")
    return result


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _totals_from_json(value: Any) -> Totals:
    if not isinstance(value, dict):
        raise BudgetError("usage totals must be an object")
    try:
        return Totals(
            calls=int(value.get("calls", 0)),
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            completion_tokens=int(value.get("completion_tokens", 0)),
            total_tokens=int(value.get("total_tokens", 0)),
            cost_usd=_decimal(value.get("cost_usd", "0"), name="persisted cost"),
            unpriced_calls=int(value.get("unpriced_calls", 0)),
        )
    except (TypeError, ValueError) as exc:
        raise BudgetError(f"invalid persisted usage totals: {exc}") from exc


def _totals_json(value: Totals) -> dict[str, Any]:
    return {
        "calls": value.calls,
        "prompt_tokens": value.prompt_tokens,
        "completion_tokens": value.completion_tokens,
        "total_tokens": value.total_tokens,
        "cost_usd": str(value.cost_usd),
        "unpriced_calls": value.unpriced_calls,
    }


def _totals_snapshot(value: Totals) -> dict[str, Any]:
    result = _totals_json(value)
    result["cost_usd"] = float(value.cost_usd)
    return result


class UsageLedger:
    def __init__(
        self,
        *,
        prices: dict[str, ModelPrice],
        token_limit: int | None,
        cost_limit_usd: float | None,
        warn_fraction: float,
        store_path: Path | None,
        strict: bool = False,
    ) -> None:
        self.prices = dict(prices)
        self.token_limit: int | None = token_limit
        self.cost_limit_usd: float | None = cost_limit_usd
        self.warn_fraction = warn_fraction
        self.store_path = store_path
        self.strict = strict
        self.strict_cost_limit_usd = (
            _decimal(cost_limit_usd, name="strict cost limit")
            if strict and cost_limit_usd is not None
            else None
        )
        self.session = Totals()
        self.lifetime = Totals()
        self.by_model: dict[str, Totals] = {}
        self.reservations: dict[str, Reservation] = {}
        self.settled_reservations: set[str] = set()
        self.unpriced_models: set[str] = set()
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        if strict:
            if fcntl is None:
                raise BudgetError("strict accounting requires fcntl advisory locking")
            if store_path is None or self.strict_cost_limit_usd is None:
                raise BudgetError(
                    "strict accounting requires persistent storage and a cost limit"
                )
            if self.strict_cost_limit_usd <= _ZERO:
                raise BudgetError("strict cost limit must be positive")
        self._load()

    @property
    def lock_path(self) -> Path | None:
        if self.store_path is None:
            return None
        return self.store_path.with_name(self.store_path.name + ".lock")

    @contextmanager
    def _file_lock(self) -> Generator[None, None, None]:
        if self.store_path is None:
            yield
            return
        if fcntl is None:
            if self.strict:
                raise BudgetError("strict accounting requires fcntl advisory locking")
            yield
            return
        lock_path = self.lock_path
        assert lock_path is not None
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            if self.strict:
                raise BudgetError(f"strict accounting lock failed: {exc}") from exc
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _load(self) -> None:
        with self._lock, self._file_lock():
            self._reload_locked()

    def _reload_locked(self) -> None:
        if self.store_path is None or not self.store_path.exists():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise BudgetError("usage ledger root must be an object")
            version = int(raw.get("version", 1))
            if version not in {1, 2}:
                raise BudgetError(f"unsupported usage ledger version: {version}")
            persisted_limit = raw.get("strict_cost_limit_usd")
            if persisted_limit is not None:
                persisted = _decimal(
                    persisted_limit, name="persisted strict cost limit"
                )
                if not self.strict:
                    raise BudgetError(
                        "workspace has a strict campaign and cannot be reopened non-strict"
                    )
                assert self.strict_cost_limit_usd is not None
                if self.strict_cost_limit_usd > persisted:
                    raise BudgetError(
                        f"strict campaign limit cannot increase from ${persisted} to ${self.strict_cost_limit_usd}"
                    )
                self.strict_cost_limit_usd = min(self.strict_cost_limit_usd, persisted)
                self.cost_limit_usd = float(self.strict_cost_limit_usd)
            self.lifetime = _totals_from_json(raw.get("lifetime", {}))
            by_model = raw.get("by_model", {})
            if not isinstance(by_model, dict):
                raise BudgetError("usage by_model must be an object")
            self.by_model = {
                str(key): _totals_from_json(value) for key, value in by_model.items()
            }
            self.reservations = {}
            reservations = raw.get("reservations", {}) if version == 2 else {}
            if not isinstance(reservations, dict):
                raise BudgetError("usage reservations must be an object")
            for key, value in reservations.items():
                if not isinstance(value, dict):
                    raise BudgetError("usage reservation must be an object")
                self.reservations[str(key)] = Reservation(
                    role=str(value["role"]),
                    label=str(value["label"]),
                    max_cost_usd=_decimal(
                        value["max_cost_usd"], name="reservation bound"
                    ),
                )
            settled = raw.get("settled_reservations", []) if version == 2 else []
            if not isinstance(settled, list):
                raise BudgetError("settled_reservations must be a list")
            self.settled_reservations = {str(item) for item in settled}
            self.unpriced_models = {
                label
                for label, totals in self.by_model.items()
                if totals.unpriced_calls
            }
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            BudgetError,
        ) as exc:
            if isinstance(exc, BudgetError) and "cannot be reopened non-strict" in str(exc):
                raise
            if self.strict:
                if isinstance(exc, BudgetError):
                    raise
                raise BudgetError(
                    f"strict accounting ledger is unreadable: {exc}"
                ) from exc
            self.lifetime = Totals()
            self.by_model = {}
            self.reservations = {}
            self.settled_reservations = set()

    def _payload_locked(self) -> dict[str, Any]:
        return {
            "version": 2,
            "strict_cost_limit_usd": (
                str(self.strict_cost_limit_usd)
                if self.strict_cost_limit_usd is not None
                else None
            ),
            "lifetime": _totals_json(self.lifetime),
            "by_model": {
                key: _totals_json(value) for key, value in self.by_model.items()
            },
            "reservations": {
                key: {
                    "role": value.role,
                    "label": value.label,
                    "max_cost_usd": str(value.max_cost_usd),
                }
                for key, value in self.reservations.items()
            },
            "settled_reservations": sorted(self.settled_reservations)[-1000:],
        }

    def _persist_locked(self) -> None:
        if self.store_path is None:
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=self.store_path.name + ".",
                suffix=".tmp",
                dir=self.store_path.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        self._payload_locked(), handle, ensure_ascii=False, indent=2
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, self.store_path)
                directory_fd = os.open(self.store_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass
        except OSError as exc:
            if self.strict:
                raise BudgetError(
                    f"strict accounting persistence failed: {exc}"
                ) from exc

    def validate_strict(self) -> None:
        if not self.strict:
            return
        with self._lock, self._file_lock():
            self._reload_locked()
            if self.lifetime.unpriced_calls:
                raise BudgetError("cannot enter strict mode with prior unpriced calls")
            exposure = self.lifetime.cost_usd + self._reserved_locked()
            assert self.strict_cost_limit_usd is not None
            if self.strict_cost_limit_usd < exposure:
                raise BudgetError(
                    f"strict limit ${self.strict_cost_limit_usd} is below campaign exposure ${exposure}"
                )

    def _resolve_price(self, label: str) -> ModelPrice | None:
        if label in self.prices:
            return self.prices[label]
        if "/" in label:
            bare = label.split("/", 1)[1]
            if bare in self.prices:
                return self.prices[bare]
        return None

    def _reserved_locked(self) -> Decimal:
        return sum((item.max_cost_usd for item in self.reservations.values()), _ZERO)

    def precheck(self, *, role: str, label: str) -> None:
        with self._lock:
            if self.strict:
                raise BudgetError(
                    "strict requests must reserve a priced maximum before inference"
                )
            if (
                self.token_limit is not None
                and self.session.total_tokens >= self.token_limit
            ):
                raise BudgetError(
                    f"token budget reached: {self.session.total_tokens}/{self.token_limit} tokens "
                    f"spent (blocked role {role} on {label})"
                )
            if self.cost_limit_usd is not None and self.session.cost_usd >= _decimal(
                self.cost_limit_usd, name="cost limit"
            ):
                raise BudgetError(
                    f"cost budget reached: ${self.session.cost_usd:.4f}/${self.cost_limit_usd:.2f} "
                    f"spent (blocked role {role} on {label})"
                )

    def reserve(self, *, role: str, label: str, max_cost_usd: Decimal) -> str:
        bound = _decimal(max_cost_usd, name="reservation bound")
        if bound <= _ZERO:
            raise BudgetError("reservation bound must be positive")
        if not self.strict:
            raise BudgetError("reservations require strict accounting")
        with self._lock, self._file_lock():
            self._reload_locked()
            if self.reservations:
                raise BudgetError(
                    "an unresolved strict reservation blocks all further paid requests"
                )
            if self.lifetime.unpriced_calls:
                raise BudgetError("strict accounting cannot admit prior unpriced usage")
            assert self.strict_cost_limit_usd is not None
            exposure = self.lifetime.cost_usd + self._reserved_locked()
            if exposure + bound > self.strict_cost_limit_usd:
                raise BudgetError(
                    f"strict budget rejects ${bound} request: campaign exposure "
                    f"${exposure}/${self.strict_cost_limit_usd}"
                )
            reservation_id = uuid4().hex
            self.reservations[reservation_id] = Reservation(role, label, bound)
            self._persist_locked()
            return reservation_id

    def settle(self, reservation_id: str, usage: Usage) -> dict[str, Any]:
        if not self.strict:
            raise BudgetError("reservation settlement requires strict accounting")
        total_tokens = usage.total_tokens or (
            usage.prompt_tokens + usage.completion_tokens
        )
        if total_tokens <= 0 or usage.prompt_tokens < 0 or usage.completion_tokens < 0:
            raise BudgetError(
                "strict response omitted meaningful nonnegative token usage"
            )
        if (
            usage.cost_usd is None
            or not math.isfinite(usage.cost_usd)
            or usage.cost_usd < 0
        ):
            raise BudgetError("strict response omitted a real nonnegative finite cost")
        actual = _decimal(usage.cost_usd, name="reported cost")
        with self._lock, self._file_lock():
            self._reload_locked()
            if reservation_id in self.settled_reservations:
                raise BudgetError(f"reservation was already settled: {reservation_id}")
            reservation = self.reservations.get(reservation_id)
            if reservation is None:
                raise BudgetError(f"unknown reservation: {reservation_id}")
            if actual > reservation.max_cost_usd:
                raise BudgetError(
                    f"reported cost ${actual} exceeds reserved bound ${reservation.max_cost_usd}"
                )
            event = self._record_locked(
                role=reservation.role,
                label=reservation.label,
                usage=usage,
                cost_usd=actual,
            )
            del self.reservations[reservation_id]
            self.settled_reservations.add(reservation_id)
            self._persist_locked()
        self._emit(event)
        return event

    def _record_locked(
        self,
        *,
        role: str,
        label: str,
        usage: Usage,
        cost_usd: Decimal | None,
    ) -> dict[str, Any]:
        total_tokens = usage.total_tokens or (
            usage.prompt_tokens + usage.completion_tokens
        )
        for totals in (
            self.session,
            self.lifetime,
            self.by_model.setdefault(label, Totals()),
        ):
            totals.calls += 1
            totals.prompt_tokens += usage.prompt_tokens
            totals.completion_tokens += usage.completion_tokens
            totals.total_tokens += total_tokens
            if cost_usd is None:
                totals.unpriced_calls += 1
            else:
                totals.cost_usd += cost_usd
        if cost_usd is None:
            self.unpriced_models.add(label)
        cost_limit = (
            _decimal(self.cost_limit_usd, name="cost limit")
            if self.cost_limit_usd
            else None
        )
        warn = bool(
            (
                self.token_limit is not None
                and self.session.total_tokens >= self.token_limit * self.warn_fraction
            )
            or (
                cost_limit is not None
                and self.session.cost_usd
                >= cost_limit * Decimal(str(self.warn_fraction))
            )
        )
        return {
            "event": "usage",
            "role": role,
            "model": label,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(cost_usd) if cost_usd is not None else None,
            "warn": warn,
            **self._snapshot_locked(),
        }

    def record(self, *, role: str, label: str, usage: Usage) -> dict[str, Any]:
        if self.strict:
            raise BudgetError("strict usage must settle its reservation")
        with self._lock:
            cost_usd: Decimal | None
            if usage.cost_usd is not None:
                cost_usd = _decimal(usage.cost_usd, name="reported cost")
            else:
                price = self._resolve_price(label)
                cost_usd = (
                    _decimal(usage.prompt_tokens, name="prompt tokens")
                    / Decimal(1_000_000)
                    * _decimal(price.input_per_mtok, name="input price")
                    + _decimal(usage.completion_tokens, name="completion tokens")
                    / Decimal(1_000_000)
                    * _decimal(price.output_per_mtok, name="output price")
                    if price is not None
                    else None
                )
            event = self._record_locked(
                role=role, label=label, usage=usage, cost_usd=cost_usd
            )
            with self._file_lock():
                self._persist_locked()
        self._emit(event)
        return event

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
                if self.strict:
                    if cost_limit_usd is None:
                        raise BudgetError(
                            "strict campaign cost limit cannot be disabled"
                        )
                    proposed = _decimal(cost_limit_usd, name="cost limit")
                    assert self.strict_cost_limit_usd is not None
                    if proposed > self.strict_cost_limit_usd:
                        raise BudgetError(
                            "strict campaign cost limit cannot be increased"
                        )
                    exposure = self.lifetime.cost_usd + self._reserved_locked()
                    if proposed < exposure:
                        raise BudgetError(
                            f"cost limit ${proposed} is below campaign exposure ${exposure}"
                        )
                    self.strict_cost_limit_usd = proposed
                self.cost_limit_usd = cost_limit_usd  # type: ignore[assignment]
                with self._file_lock():
                    self._persist_locked()
            event = {"event": "limits", **self._zero_event_locked()}
        self._emit(event)

    def set_price(self, key: str, price: ModelPrice) -> None:
        with self._lock:
            self.prices[key] = price
            event = {
                "event": "price",
                "model": key,
                **self._zero_event_locked(model=False),
            }
        self._emit(event)

    def reset_session(self) -> None:
        with self._lock:
            self.session = Totals()
            event = {"event": "reset", **self._zero_event_locked()}
        self._emit(event)

    def _zero_event_locked(self, *, model: bool = True) -> dict[str, Any]:
        return {
            "role": None,
            **({"model": None} if model else {}),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": None,
            "warn": False,
            **self._snapshot_locked(),
        }

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

    def _snapshot_locked(self) -> dict[str, Any]:
        reserved = self._reserved_locked()
        limit = (
            self.strict_cost_limit_usd
            if self.strict
            else (
                _decimal(self.cost_limit_usd, name="cost limit")
                if self.cost_limit_usd
                else None
            )
        )
        return {
            "session": _totals_snapshot(self.session),
            "lifetime": _totals_snapshot(self.lifetime),
            "by_model": {
                key: _totals_snapshot(value) for key, value in self.by_model.items()
            },
            "token_limit": self.token_limit,
            "cost_limit_usd": float(limit) if limit is not None else None,
            "strict": self.strict,
            "reserved_cost_usd": float(reserved),
            "campaign_exposure_usd": float(self.lifetime.cost_usd + reserved),
            "remaining_strict_usd": (
                float(limit - self.lifetime.cost_usd - reserved)
                if self.strict and limit is not None
                else None
            ),
            "outstanding_reservations": len(self.reservations),
            "unpriced": sorted(self.unpriced_models),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.strict:
                with self._file_lock():
                    self._reload_locked()
            return self._snapshot_locked()

    def status_line(self, markup: bool = True) -> str:
        snapshot = self.snapshot()
        session = snapshot["session"]
        lifetime = snapshot["lifetime"]
        parts = [
            f"tokens {_fmt_int(session['total_tokens'])} "
            f"(in {_fmt_int(session['prompt_tokens'])} / out {_fmt_int(session['completion_tokens'])})",
            f"bill ${session['cost_usd']:,.4f}",
        ]
        if self.strict:
            parts.extend(
                [
                    f"campaign actual ${lifetime['cost_usd']:,.4f}",
                    f"reserved ${snapshot['reserved_cost_usd']:,.4f}",
                    f"remaining ${snapshot['remaining_strict_usd']:,.4f}",
                ]
            )
        else:
            if self.token_limit is not None:
                parts.append(f"limit {_fmt_int(self.token_limit)} tokens")
            if self.cost_limit_usd is not None:
                parts.append(f"limit ${self.cost_limit_usd:,.2f}")
            parts.append(
                f"lifetime {_fmt_int(lifetime['total_tokens'])} tokens / ${lifetime['cost_usd']:,.4f}"
            )
        parts.append(f"session {_fmt_int(session['calls'])} calls")
        line = " · ".join(parts)
        if session["unpriced_calls"]:
            line += f" · {session['unpriced_calls']} calls unpriced (set /budget price)"
        return line


_LEDGERS: dict[str, UsageLedger] = {}
_LAST_LEDGER_KEY: str | None = None


def get_ledger(cfg: Config) -> UsageLedger:
    global _LAST_LEDGER_KEY
    key = str(Path(cfg.workspace).resolve())
    ledger = _LEDGERS.get(key)
    requested = (
        cfg.budget.strict,
        cfg.budget.persist,
        cfg.budget.cost_limit_usd,
        cfg.budget.token_limit,
    )
    if ledger is not None:
        existing = (
            ledger.strict,
            ledger.store_path is not None,
            ledger.cost_limit_usd,
            ledger.token_limit,
        )
        if requested != existing:
            raise BudgetError(
                f"cached ledger configuration conflicts for workspace {key}: "
                f"requested {requested}, active {existing}"
            )
    else:
        prices = {
            model: ModelPrice(price.input_per_mtok, price.output_per_mtok)
            for model, price in cfg.prices.items()
        }
        store_path = Path(cfg.workspace) / "usage.json" if cfg.budget.persist else None
        ledger = UsageLedger(
            prices=prices,
            token_limit=cfg.budget.token_limit,
            cost_limit_usd=cfg.budget.cost_limit_usd,
            warn_fraction=cfg.budget.warn_fraction,
            store_path=store_path,
            strict=cfg.budget.strict,
        )
        _LEDGERS[key] = ledger
    _LAST_LEDGER_KEY = key
    return ledger


def current_ledger() -> UsageLedger | None:
    if _LAST_LEDGER_KEY is None:
        return None
    return _LEDGERS.get(_LAST_LEDGER_KEY)
