from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_STAGE_KEYS = ("goal", "prior_work", "limitations", "gaps", "feasibility", "quick_test")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_FENCE = re.compile(r"(?ms)^(`{3,}|~{3,}).*?^\1\s*$")


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("root is not an object")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label} is missing or corrupt: {exc}")
        return None


def _normalize_url(value: str) -> str:
    parsed = urlsplit(value.strip().rstrip(".,;:"))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").lower()
    if parsed.port is not None:
        default = (parsed.scheme.lower() == "http" and parsed.port == 80) or (
            parsed.scheme.lower() == "https" and parsed.port == 443
        )
        if not default:
            host += f":{parsed.port}"
    return urlunsplit(
        (parsed.scheme.lower(), host, parsed.path or "/", parsed.query, "")
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    checks: list[str], errors: list[str], name: str, passed: bool, detail: str = ""
) -> None:
    if passed:
        checks.append(name)
    else:
        errors.append(f"{name} failed" + (f": {detail}" if detail else ""))


def verify_outputs(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    checks: list[str] = []
    errors: list[str] = []
    metadata = _load_json(run_dir / "run.json", "run metadata", errors)
    if metadata is None:
        return {"passed": False, "checks": checks, "errors": errors}
    required_keys = {
        "version",
        "run_id",
        "query",
        "status",
        "final_node",
        "requirements",
        "models",
        "graph",
        "skill",
        "pricing",
        "budget",
        "obsidian",
        "outputs",
    }
    _record(
        checks,
        errors,
        "run metadata contract",
        metadata.get("version") == 1 and required_keys.issubset(metadata),
        "version or required top-level keys differ",
    )
    if metadata.get("run_id") != run_dir.name:
        errors.append("run metadata identity failed: run_id does not match directory")
    for label in ("graph", "skill"):
        entry = metadata.get(label)
        if entry is None and label == "skill":
            continue
        if not isinstance(entry, dict):
            errors.append(f"{label} provenance failed: metadata is missing")
            continue
        path = run_dir / str(entry.get("path", ""))
        valid = path.is_file() and _sha(path) == entry.get("sha256")
        _record(checks, errors, f"{label} provenance", valid, str(path))
    artifacts = run_dir / "artifacts"
    for key in _STAGE_KEYS:
        path = artifacts / f"{key}.md"
        valid = path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        _record(checks, errors, f"stage {key}", valid, str(path))
    requirements_value = metadata.get("requirements")
    requirements: dict[str, Any] = (
        requirements_value if isinstance(requirements_value, dict) else {}
    )
    minimum = int(requirements.get("min_sources", 0) or 0)
    source_index = _load_json(run_dir / "sources.json", "source index", errors)
    canonical: set[str] = set()
    aliases: dict[str, str] = {}
    if source_index is not None:
        sources = source_index.get("sources")
        raw_aliases = source_index.get("aliases")
        if not isinstance(sources, list) or not isinstance(raw_aliases, dict):
            errors.append(
                "source index schema failed: expected sources list and aliases object"
            )
        else:
            aliases = {
                _normalize_url(str(key)): _normalize_url(str(value))
                for key, value in raw_aliases.items()
                if _normalize_url(str(key)) and _normalize_url(str(value))
            }
            for item in sources:
                if not isinstance(item, dict):
                    errors.append(
                        "source index schema failed: source record is not an object"
                    )
                    continue
                url = _normalize_url(str(item.get("url", "")))
                evidence = run_dir / str(item.get("evidence_path", ""))
                if not url:
                    errors.append("source provenance failed: invalid canonical URL")
                    continue
                if not evidence.is_file():
                    errors.append(f"source evidence missing: {url}")
                    continue
                if _sha(evidence) != item.get("text_sha256"):
                    errors.append(f"source evidence hash mismatch: {url}")
                    continue
                canonical.add(url)
            _record(
                checks,
                errors,
                "minimum fetched sources",
                len(canonical) >= minimum,
                f"{len(canonical)} found; {minimum} required",
            )
    outputs_value = metadata.get("outputs")
    outputs: dict[str, Any] = outputs_value if isinstance(outputs_value, dict) else {}
    report_required = bool(requirements.get("report"))
    report_value = outputs.get("report_path")
    report_path = (
        Path(str(report_value)).expanduser() if report_value else run_dir / "report.md"
    )
    if report_required:
        valid_report = report_path.is_file() and bool(
            report_path.read_text(encoding="utf-8").strip()
        )
        _record(checks, errors, "required report", valid_report, str(report_path))
        if valid_report:
            report = report_path.read_text(encoding="utf-8")
            required_headings = {
                "## Goal",
                "## Prior Work",
                "## Limitations",
                "## Gaps",
                "## Feasibility",
                "## Quick Test",
                "## References",
            }
            _record(
                checks,
                errors,
                "report sections",
                all(heading in report for heading in required_headings),
                "required heading missing",
            )
            report_urls = {
                aliases.get(_normalize_url(raw), _normalize_url(raw))
                for raw in _URL.findall(report)
                if _normalize_url(raw)
            }
            unknown = sorted(report_urls - canonical)
            _record(
                checks,
                errors,
                "report source provenance",
                not unknown and len(report_urls) >= minimum,
                f"unknown={unknown}; cited={len(report_urls)} required={minimum}",
            )
    if requirements.get("obsidian_bundle"):
        manifest_value = outputs.get("obsidian_manifest_path")
        manifest_path = (
            Path(str(manifest_value)).expanduser() if manifest_value else Path()
        )
        manifest = (
            _load_json(manifest_path, "Obsidian bundle manifest", errors)
            if manifest_value
            else None
        )
        if not manifest_value:
            errors.append(
                "Obsidian bundle manifest failed: required output path is null"
            )
        if manifest is not None:
            notes = manifest.get("notes")
            root = manifest_path.parent
            vault_value = (
                (metadata.get("obsidian") or {}).get("vault")
                if isinstance(metadata.get("obsidian"), dict)
                else None
            )
            vault = (
                Path(str(vault_value)).expanduser().resolve() if vault_value else None
            )
            allowed = (
                {
                    (root / relative).relative_to(vault).with_suffix("").as_posix()
                    for relative in notes
                    if isinstance(notes, dict)
                    and vault is not None
                    and (root / relative).is_relative_to(vault)
                }
                if isinstance(notes, dict)
                else set()
            )
            related = (
                set(manifest.get("related_notes", {}))
                if isinstance(manifest.get("related_notes"), dict)
                else set()
            )
            if not isinstance(notes, dict):
                errors.append("Obsidian manifest schema failed: notes is not an object")
            else:
                for relative, entry in notes.items():
                    path = root / relative
                    valid = (
                        isinstance(entry, dict)
                        and path.is_file()
                        and _sha(path) == entry.get("sha256")
                    )
                    _record(checks, errors, f"bundle note {relative}", valid, str(path))
                    if valid:
                        text = path.read_text(encoding="utf-8")
                        frontmatter = (
                            text.startswith("---\n")
                            and "\ntags: [" in text.split("---", 2)[1]
                        )
                        _record(
                            checks,
                            errors,
                            f"bundle frontmatter {relative}",
                            frontmatter,
                        )
                        links = {
                            match.group(1).strip().removesuffix(".md")
                            for match in _WIKILINK.finditer(_FENCE.sub("", text))
                        }
                        broken = sorted(links - allowed - related)
                        _record(
                            checks,
                            errors,
                            f"bundle links {relative}",
                            not broken,
                            ", ".join(broken),
                        )
                related_manifest = manifest.get("related_notes")
                if not isinstance(related_manifest, dict):
                    errors.append(
                        "Obsidian manifest schema failed: related_notes is not an object"
                    )
                    related_manifest = {}
                for target, digest in related_manifest.items():
                    if vault is None:
                        errors.append(
                            "related note verification failed: vault metadata is missing"
                        )
                        break
                    path = vault / (str(target) + ".md")
                    _record(
                        checks,
                        errors,
                        f"related note {target}",
                        path.is_file() and _sha(path) == digest,
                        str(path),
                    )
    budget_value = metadata.get("budget")
    budget: dict[str, Any] = budget_value if isinstance(budget_value, dict) else {}
    if budget.get("strict"):
        ledger_path = Path(str(budget.get("ledger_path", "")))
        ledger = _load_json(ledger_path, "strict campaign ledger", errors)
        if ledger is not None:
            try:
                lifetime = ledger.get("lifetime")
                reservations = ledger.get("reservations")
                if not isinstance(lifetime, dict) or not isinstance(reservations, dict):
                    raise TypeError("lifetime or reservations has invalid type")
                actual = Decimal(str(lifetime.get("cost_usd", "0")))
                reserved = sum(
                    (
                        Decimal(str(item["max_cost_usd"]))
                        for item in reservations.values()
                    ),
                    Decimal(0),
                )
                limit = Decimal(str(budget.get("limit_usd")))
                exposure_ok = actual + reserved <= limit
            except (InvalidOperation, TypeError, KeyError, AttributeError) as exc:
                errors.append(f"strict campaign accounting failed: {exc}")
            else:
                _record(
                    checks,
                    errors,
                    "strict campaign exposure",
                    exposure_ok and not reservations,
                    f"actual={actual} reserved={reserved} limit={limit}",
                )
                allowed_models = {
                    f"{item.get('provider')}/{item.get('model')}"
                    for item in (metadata.get("models") or {}).values()
                    if isinstance(item, dict)
                }
                used = set((ledger.get("by_model") or {}).keys())
                _record(
                    checks,
                    errors,
                    "strict model identity",
                    used.issubset(allowed_models),
                    f"unexpected={sorted(used - allowed_models)}",
                )
    return {"passed": not errors, "checks": checks, "errors": errors}


def verify_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    result = verify_outputs(run_dir)
    checks = list(result["checks"])
    errors = list(result["errors"])
    metadata = _load_json(run_dir / "run.json", "run metadata", errors)
    if metadata is not None:
        _record(
            checks,
            errors,
            "completed run state",
            metadata.get("status") == "completed",
            f"status={metadata.get('status')}",
        )
    transitions: list[dict[str, Any]] = []
    try:
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                event = json.loads(line)
                if isinstance(event, dict):
                    transitions.append(event)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"trace evidence is missing or corrupt: {exc}")
    else:
        terminal = any(
            event.get("target") == "done"
            and isinstance(event.get("metric"), dict)
            and event["metric"].get("passed") is True
            for event in transitions
        )
        complete = any(event.get("event") == "complete" for event in transitions)
        _record(checks, errors, "passed terminal transition", terminal)
        _record(checks, errors, "completion trace event", complete)
    final: dict[str, Any] = {
        "passed": not errors,
        "checks": checks,
        "errors": errors,
    }
    try:
        (run_dir / "verification.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        final["passed"] = False
        final["errors"].append(f"verification result could not be written: {exc}")
    return final
