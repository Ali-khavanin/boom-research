from __future__ import annotations

from typing import Any

from ragent.config import Config, load_config, require_api_key
from ragent.errors import ProviderError

from .base import LLM
from .google import GoogleProvider
from .openai_compat import OpenAICompatProvider
from .usage import get_ledger


def get_llm(role: str, node_override: Any = None, cfg: Config | None = None) -> LLM:
    config = cfg or load_config()
    if role not in config.roles:
        raise ProviderError(f"unknown LLM role: {role}")
    role_cfg = config.roles[role].model_copy()
    if node_override is not None:
        if getattr(node_override, "provider", None):
            role_cfg.provider = node_override.provider
        if getattr(node_override, "model", None):
            role_cfg.model = node_override.model
        if getattr(node_override, "temperature", None) is not None:
            role_cfg.temperature = node_override.temperature
        if getattr(node_override, "max_tokens", None) is not None:
            role_cfg.max_tokens = node_override.max_tokens
    provider_cfg = config.providers.get(role_cfg.provider)
    if provider_cfg is None:
        raise ProviderError(f"role '{role}' references unknown provider '{role_cfg.provider}'")
    api_key = require_api_key(role_cfg.provider, provider_cfg)
    if provider_cfg.kind == "google":
        provider = GoogleProvider(
            provider_cfg.base_url or "https://generativelanguage.googleapis.com/v1beta",
            api_key,
        )
    else:
        if provider_cfg.kind == "openrouter":
            base_url = provider_cfg.base_url or "https://openrouter.ai/api/v1"
            headers = {"HTTP-Referer": "https://local/ragent"}
        else:
            base_url = provider_cfg.base_url or "https://api.openai.com/v1"
            headers = {}
        provider = OpenAICompatProvider(
            base_url,
            api_key,
            headers,
            report_cost=provider_cfg.kind == "openrouter",
        )
    return LLM(
        provider=provider,
        model=role_cfg.model,
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        role=role,
        provider_name=role_cfg.provider,
        ledger=get_ledger(config),
    )


__all__ = ["get_llm", "get_ledger", "LLM"]
