from __future__ import annotations

import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from ragent.errors import ProviderError

RoleName: TypeAlias = Literal[
    "book_to_skill",
    "graph_builder",
    "executor",
    "report",
    "obsidian",
    "refiner",
    "judge",
]
ROLE_NAMES: tuple[str, ...] = (
    "book_to_skill",
    "graph_builder",
    "executor",
    "report",
    "obsidian",
    "refiner",
    "judge",
)


class ProviderCfg(BaseModel):
    kind: Literal["openai", "openrouter", "google"]
    base_url: str | None = None
    api_key_env: str


class RoleCfg(BaseModel):
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 8192


class SearchCfg(BaseModel):
    backend: Literal["tavily", "ddg"] = "ddg"
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = Field(default=6, ge=1, le=20)


class ObsidianCfg(BaseModel):
    backend: Literal["vault", "mcp"] = "vault"
    vault_path: Path | None = None
    folder: str = "Research"
    mcp_command: list[str] = Field(default_factory=list)


class ModelPriceCfg(BaseModel):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class BudgetCfg(BaseModel):
    token_limit: int | None = Field(default=None, ge=1)
    cost_limit_usd: float | None = Field(default=None, gt=0)
    warn_fraction: float = Field(default=0.8, gt=0, le=1)
    persist: bool = True


class Config(BaseModel):
    workspace: Path = Path(".ragent")
    providers: dict[str, ProviderCfg]
    roles: dict[RoleName, RoleCfg]
    search: SearchCfg = Field(default_factory=SearchCfg)
    obsidian: ObsidianCfg = Field(default_factory=ObsidianCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    prices: dict[str, ModelPriceCfg] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_role_providers(self) -> "Config":
        missing_roles = [name for name in ROLE_NAMES if name not in self.roles]
        if missing_roles:
            raise ValueError(f"missing role configuration: {', '.join(missing_roles)}")
        unknown = sorted({role.provider for role in self.roles.values()} - self.providers.keys())
        if unknown:
            raise ValueError(f"roles reference unknown providers: {', '.join(unknown)}")
        return self


DEFAULT_CONFIG: dict = {
    "workspace": ".ragent",
    "providers": {
        "openrouter": {
            "kind": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        "openai": {"kind": "openai", "api_key_env": "OPENAI_API_KEY"},
        "google": {"kind": "google", "api_key_env": "GOOGLE_API_KEY"},
    },
    "roles": {
        role: {
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4.5",
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        for role in ROLE_NAMES
    },
    "search": {"backend": "ddg", "api_key_env": "TAVILY_API_KEY", "max_results": 6},
    "obsidian": {"backend": "vault", "folder": "Research", "mcp_command": []},
    "budget": {"warn_fraction": 0.8, "persist": True},
    "prices": {},
}


def _merge(base: dict, overlay: dict) -> dict:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _environment_overlay() -> dict:
    overlay: dict = {}
    if workspace := os.getenv("RAGENT_WORKSPACE"):
        overlay["workspace"] = workspace
    if backend := os.getenv("RAGENT_SEARCH_BACKEND"):
        overlay.setdefault("search", {})["backend"] = backend
    if vault := os.getenv("RAGENT_OBSIDIAN_VAULT_PATH"):
        overlay.setdefault("obsidian", {})["vault_path"] = vault
    if folder := os.getenv("RAGENT_OBSIDIAN_FOLDER"):
        overlay.setdefault("obsidian", {})["folder"] = folder
    for role in ROLE_NAMES:
        if value := os.getenv(f"RAGENT_MODEL_{role.upper()}"):
            if "/" not in value:
                raise ValueError(f"RAGENT_MODEL_{role.upper()} must be provider/model")
            provider, model = value.split("/", 1)
            overlay.setdefault("roles", {}).setdefault(role, {}).update(
                {"provider": provider, "model": model}
            )
    if tokens_raw := os.getenv("RAGENT_BUDGET_TOKENS"):
        try:
            overlay.setdefault("budget", {})["token_limit"] = int(tokens_raw)
        except ValueError as exc:
            raise ValueError("RAGENT_BUDGET_TOKENS must be a number") from exc
    if cost_raw := os.getenv("RAGENT_BUDGET_COST_USD"):
        try:
            overlay.setdefault("budget", {})["cost_limit_usd"] = float(cost_raw)
        except ValueError as exc:
            raise ValueError("RAGENT_BUDGET_COST_USD must be a number") from exc
    return overlay


def parse_model_overrides(values: list[str] | None) -> dict:
    overlay: dict = {"roles": {}}
    for value in values or []:
        if "=" not in value or "/" not in value.split("=", 1)[1]:
            raise ValueError(f"model override must be role=provider/model: {value}")
        role, target = value.split("=", 1)
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role in model override: {role}")
        provider, model = target.split("/", 1)
        overlay["roles"].setdefault(role, {}).update(
            {"provider": provider, "model": model}
        )
    return overlay


def load_config(
    config_path: Path | None = None,
    *,
    workspace: Path | None = None,
    model_overrides: list[str] | None = None,
) -> Config:
    data = deepcopy(DEFAULT_CONFIG)
    path = config_path or Path("ragent.toml")
    if path.exists():
        with path.open("rb") as handle:
            data = _merge(data, tomllib.load(handle))
    data = _merge(data, _environment_overlay())
    data = _merge(data, parse_model_overrides(model_overrides))
    if workspace is not None:
        data["workspace"] = str(workspace)
    return Config.model_validate(data)


def require_api_key(provider_name: str, cfg: ProviderCfg) -> str:
    value = os.getenv(cfg.api_key_env)
    if not value:
        raise ProviderError(
            f"provider '{provider_name}' requires environment variable {cfg.api_key_env}"
        )
    return value
