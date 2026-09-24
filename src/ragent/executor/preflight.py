from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx

from ragent.config import ROLE_NAMES, Config, RoleName, require_api_key
from ragent.errors import BudgetError, PreflightError
from ragent.graph_builder import Graph, audit
from ragent.llm.usage import get_ledger

_TOOL_NAMES = {"browser.search", "browser.fetch", "report.generate", "obsidian.note"}
_REQUIRED_PARAMETERS = {"tools", "structured_outputs", "reasoning"}
_ADMISSION: dict[str, dict[str, Any]] = {}


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PreflightError(f"model metadata has invalid {label}: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise PreflightError(f"model metadata has nonfinite or negative {label}")
    return result


def _response_json(response: httpx.Response, label: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
        value = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PreflightError(f"OpenRouter {label} check failed: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"OpenRouter {label} returned a non-object response")
    return value


def _model_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", payload)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _route_preview(graph: Graph) -> tuple[list[str], list[str]]:
    route: list[str] = []
    optional: list[str] = []
    seen: set[str] = set()
    node = graph.node(graph.entry)
    while not node.terminal:
        if node.id in seen:
            raise PreflightError(f"default route loops at node {node.id}")
        seen.add(node.id)
        edges = graph.out_edges(node.id)
        if not edges:
            raise PreflightError(f"default route dead-ends at node {node.id}")
        edge = edges[0]
        route.append(edge.id)
        optional.extend(item.id for item in edges[1:])
        if not edge.prompt_template.strip():
            raise PreflightError(f"default route edge {edge.id} has an empty prompt")
        unknown = sorted(set(edge.tool_set) - _TOOL_NAMES)
        if unknown:
            raise PreflightError(
                f"default route edge {edge.id} has unknown tools: {', '.join(unknown)}"
            )
        node = graph.node(edge.target)
    return route, optional


def _resolved_models(graph: Graph, cfg: Config) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in ROLE_NAMES:
        value = cfg.roles[cast(RoleName, role)]
        result[role] = {
            "provider": value.provider,
            "model": value.model,
            "max_tokens": value.max_tokens,
            "reasoning_max_tokens": value.reasoning_max_tokens,
        }
    for node in graph.nodes:
        override = node.model
        if override is None:
            continue
        base = cfg.roles["executor"]
        provider = override.provider or base.provider
        model = override.model or base.model
        max_tokens = (
            override.max_tokens if override.max_tokens is not None else base.max_tokens
        )
        reasoning = (
            override.reasoning_max_tokens
            if override.reasoning_max_tokens is not None
            else base.reasoning_max_tokens
        )
        if max_tokens <= 0 or (
            reasoning is not None and (reasoning <= 0 or reasoning >= max_tokens)
        ):
            raise PreflightError(
                f"node {node.id} has an invalid effective token/reasoning cap"
            )
        result[f"executor@{node.id}"] = {
            "provider": provider,
            "model": model,
            "max_tokens": max_tokens,
            "reasoning_max_tokens": reasoning,
        }
    return result


def _eligible_metadata(
    *,
    client: httpx.Client,
    base_url: str,
    key: str,
    model: str,
    required_max_tokens: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}"}
    catalog_payload = _response_json(
        client.get(base_url + "/models", headers=headers), "model catalog"
    )
    matches = [item for item in _model_rows(catalog_payload) if item.get("id") == model]
    if len(matches) != 1:
        raise PreflightError(f"exact OpenRouter model is unavailable: {model}")
    catalog = matches[0]
    supported = set(catalog.get("supported_parameters") or [])
    missing = sorted(_REQUIRED_PARAMETERS - supported)
    if missing:
        raise PreflightError(
            f"model {model} lacks required parameters: {', '.join(missing)}"
        )
    pricing = catalog.get("pricing")
    if not isinstance(pricing, dict):
        raise PreflightError(f"model {model} has no pricing metadata")
    prompt_price = _decimal(pricing.get("prompt"), "catalog prompt price")
    completion_price = _decimal(pricing.get("completion"), "catalog completion price")
    endpoint_payload = _response_json(
        client.get(
            base_url + "/models/" + quote(model, safe="/") + "/endpoints",
            headers=headers,
        ),
        "endpoint catalog",
    )
    data = endpoint_payload.get("data", endpoint_payload)
    endpoints = data.get("endpoints", []) if isinstance(data, dict) else []
    if not isinstance(endpoints, list):
        endpoints = []
    eligible: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        parameters = set(endpoint.get("supported_parameters") or [])
        if not _REQUIRED_PARAMETERS.issubset(parameters):
            continue
        endpoint_pricing = endpoint.get("pricing")
        if not isinstance(endpoint_pricing, dict):
            continue
        try:
            endpoint_prompt = _decimal(
                endpoint_pricing.get("prompt"), "endpoint prompt price"
            )
            endpoint_completion = _decimal(
                endpoint_pricing.get("completion"), "endpoint completion price"
            )
            request_price = _decimal(
                endpoint_pricing.get("request", "0"), "request price"
            )
            extra_fee = sum(
                (
                    _decimal(endpoint_pricing.get(name, "0"), f"{name} price")
                    for name in ("image", "image_output", "web_search", "audio")
                ),
                Decimal(0),
            )
        except PreflightError:
            continue
        context = int(endpoint.get("context_length") or 0)
        output_limit = int(
            endpoint.get("max_completion_tokens")
            or endpoint.get("max_output_tokens")
            or context
            or 0
        )
        provider = str(
            endpoint.get("provider_name")
            or endpoint.get("provider")
            or endpoint.get("name")
            or ""
        ).strip()
        if (
            provider
            and context > 0
            and output_limit >= required_max_tokens
            and request_price == 0
            and extra_fee == 0
            and endpoint_prompt <= prompt_price
            and endpoint_completion <= completion_price
        ):
            eligible.append(
                {
                    "provider": provider,
                    "context_length": context,
                    "max_completion_tokens": output_limit,
                    "prompt_price_per_token": str(endpoint_prompt),
                    "completion_price_per_token": str(endpoint_completion),
                }
            )
    if not eligible:
        raise PreflightError(
            f"model {model} has no endpoint satisfying strict controls"
        )
    context_length = max(
        int(catalog.get("context_length") or 0),
        *(item["context_length"] for item in eligible),
    )
    if context_length <= 0:
        raise PreflightError(f"model {model} has no positive advertised context length")
    reservation = prompt_price * context_length + completion_price * required_max_tokens
    if not reservation.is_finite() or reservation <= 0:
        raise PreflightError(f"model {model} has an invalid reservation bound")
    return {
        "model": model,
        "context_length": context_length,
        "prompt_price_per_token": str(prompt_price),
        "completion_price_per_token": str(completion_price),
        "prompt_price_per_million": str(prompt_price * Decimal(1_000_000)),
        "completion_price_per_million": str(completion_price * Decimal(1_000_000)),
        "max_tokens": required_max_tokens,
        "max_cost_usd": str(reservation),
        "eligible_providers": sorted({item["provider"] for item in eligible}),
        "endpoints": eligible,
    }


def _check_obsidian(cfg: Config) -> dict[str, Any]:
    if cfg.obsidian.backend == "mcp":
        if not cfg.obsidian.mcp_command:
            raise PreflightError("Obsidian MCP backend requires obsidian.mcp_command")
        return {
            "backend": "mcp",
            "layout": cfg.obsidian.layout,
            "command_configured": True,
        }
    vault_value = cfg.obsidian.vault_path
    if vault_value is None:
        raise PreflightError("Obsidian vault path is required")
    vault = vault_value.expanduser().resolve(strict=True)
    if not vault.is_dir() or not (vault / ".obsidian").is_dir():
        raise PreflightError(
            f"configured path is not an existing Obsidian vault: {vault}"
        )
    folder = Path(cfg.obsidian.folder)
    if folder.is_absolute() or ".." in folder.parts:
        raise PreflightError("Obsidian folder must be a relative path without '..'")
    destination = (vault / folder).resolve(strict=False)
    if not destination.is_relative_to(vault):
        raise PreflightError("Obsidian folder escapes the configured vault")
    related: dict[str, str] = {}
    for note in cfg.obsidian.related_notes:
        relative = Path(note)
        if relative.is_absolute() or ".." in relative.parts:
            raise PreflightError(f"related note path is unsafe: {note}")
        relative_md = (
            relative if relative.suffix == ".md" else Path(str(relative) + ".md")
        )
        resolved = (vault / relative_md).resolve(strict=True)
        if not resolved.is_relative_to(vault) or not resolved.is_file():
            raise PreflightError(f"related note is unavailable: {note}")
        related[note] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "backend": "vault",
        "vault": str(vault),
        "folder": folder.as_posix(),
        "layout": cfg.obsidian.layout,
        "tags": list(cfg.obsidian.tags),
        "related_notes": related,
    }


def admission_for(cfg: Config, model: str) -> dict[str, Any] | None:
    snapshot = _ADMISSION.get(str(cfg.workspace.resolve()))
    if snapshot is None:
        return None
    value = snapshot.get("pricing", {}).get(model)
    return dict(value) if isinstance(value, dict) else None


def preflight(graph: Graph, cfg: Config) -> dict[str, Any]:
    result = audit(graph)
    if not result.ok:
        detail = "; ".join(
            item.detail for item in result.findings if item.level == "error"
        )
        raise PreflightError(f"graph audit failed: {detail}")
    route, optional = _route_preview(graph)
    skill: dict[str, Any] | None = None
    if cfg.research.skill_path is not None:
        try:
            path = cfg.research.skill_path.expanduser().resolve(strict=True)
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise PreflightError(
                f"configured skill file cannot be read: {cfg.research.skill_path}: {exc}"
            ) from exc
        if not text.strip():
            raise PreflightError(f"configured skill file is empty: {path}")
        skill = {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    models = _resolved_models(graph, cfg)
    pricing: dict[str, Any] = {}
    authenticated = False
    if cfg.budget.strict:
        model_groups: dict[str, list[int]] = {}
        for label, value in models.items():
            provider_name = str(value["provider"])
            provider = cfg.providers.get(provider_name)
            if provider is None or provider.kind != "openrouter":
                raise PreflightError(
                    f"strict effective model is not OpenRouter: {label}"
                )
            model_groups.setdefault(str(value["model"]), []).append(
                int(value["max_tokens"])
            )
        provider_name = str(next(iter(models.values()))["provider"])
        provider_cfg = cfg.providers[provider_name]
        key = require_api_key(provider_name, provider_cfg)
        base_url = (provider_cfg.base_url or "https://openrouter.ai/api/v1").rstrip("/")
        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            _response_json(
                client.get(
                    base_url + "/key", headers={"Authorization": f"Bearer {key}"}
                ),
                "credential",
            )
            authenticated = True
            for model, caps in model_groups.items():
                pricing[model] = _eligible_metadata(
                    client=client,
                    base_url=base_url,
                    key=key,
                    model=model,
                    required_max_tokens=max(caps),
                )
        ledger = get_ledger(cfg)
        ledger.validate_strict()
        for model, value in pricing.items():
            if Decimal(value["max_cost_usd"]) > Decimal(str(cfg.budget.cost_limit_usd)):
                raise BudgetError(
                    f"one conservative {model} reservation exceeds the strict campaign limit"
                )
    obsidian = _check_obsidian(cfg) if cfg.research.obsidian else None
    snapshot = {
        "passed": True,
        "checks": [
            "graph audit",
            "default route bindings and prompts",
            *(["explicit skill"] if skill else []),
            *(
                ["OpenRouter credential, model, endpoint, pricing, and ledger"]
                if cfg.budget.strict
                else []
            ),
            *(["Obsidian vault, folder, and related notes"] if obsidian else []),
        ],
        "route": route,
        "optional_edges_unexercised": optional,
        "models": models,
        "pricing": pricing,
        "budget": {
            "strict": cfg.budget.strict,
            "limit_usd": cfg.budget.cost_limit_usd,
            "ledger": get_ledger(cfg).snapshot(),
        },
        "skill": skill,
        "obsidian": obsidian,
        "authenticated": authenticated,
    }
    _ADMISSION[str(cfg.workspace.resolve())] = snapshot
    return snapshot
