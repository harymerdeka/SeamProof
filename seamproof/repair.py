"""Deterministic draft-verify-correct loop with the LLM proposing numbers only.

Each round drafts, verifies, and on failure sends the LLM just the failed
checks plus the tunable parameters with ranges. Every proposal is checked
deterministically (known and in-range parameters, redraft, edit locality,
re-verify) before acceptance. The LLM never draws geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from seamproof import llm as llm_module
from seamproof.draft import REGISTRY
from seamproof.locality import edit_locality
from seamproof.params import PARAMETERS, tunable_params
from seamproof.pattern import Pattern
from seamproof.verify import Report, verify_pattern

REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "changes": {"type": "object", "values": {"type": "number"}},
        "reason": {"type": "string"},
    },
    "required": ["changes", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You correct garment pattern parameters. You NEVER draw geometry or compute "
    "final numbers; you only choose parameter values inside their allowed ranges. "
    "Propose the smallest change that fixes the failed checks, using each "
    "parameter's description and affects list. Reply with a single raw JSON object: "
    '{"changes": {param: number}, "reason": "..."}.'
)

# Named bad starting points for demos; values must fail verification.
INJECTS: dict[str, dict[str, float]] = {
    "sleeve_cap_too_high": {"sleeve_cap_height": 160.0},
}


@dataclass
class RoundLog:
    """One repair round: proposal, deterministic gates and outcome."""

    round: int
    failed: list[str] = field(default_factory=list)
    proposed: dict[str, float] = field(default_factory=dict)
    decision: str = ""
    reason: str = ""
    locality_passed: bool | None = None
    outcome: str = ""


def _check_proposal(raw: Any, applicable: list[str]) -> tuple[dict[str, float], str, str]:
    """Validate a proposal shape; return (changes, reason, error)."""
    if not isinstance(raw, dict):
        return {}, "", "usulan bukan objek JSON"
    changes, reason = raw.get("changes"), raw.get("reason", "")
    if not isinstance(changes, dict) or not isinstance(reason, str):
        return {}, "", "usulan harus berisi changes (objek) dan reason (teks)"
    if not changes:
        return {}, reason, "usulan tanpa perubahan ditolak"
    for name in changes:
        if name not in applicable:
            return {}, reason, f"parameter {name} tidak dikenal atau tidak berlaku"
    clean: dict[str, float] = {}
    for name, value in changes.items():
        spec = PARAMETERS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {}, reason, f"{name} harus angka nyata"
        if not spec.min <= value <= spec.max:
            return {}, reason, f"{name}={value} di luar rentang {spec.min}–{spec.max}"
        clean[name] = float(value)
    return clean, reason, ""


def _payload(pattern: Pattern, report: Report, applicable: list[str]) -> str:
    """Serialize failed checks and tunable parameters for the LLM, no photos."""
    failed = [
        {"id": c.id, "name": c.name_id, "actual": c.actual, "expected": c.expected,
         "tolerance": c.tolerance, "unit": "mm", "hint": c.message_id}
        for c in report.failures()
    ]
    params = {
        name: {"value": pattern.params[name], "min": spec.min, "max": spec.max,
               "unit": spec.unit, "description": spec.description,
               "affects": [f"{panel}.{edge}" for panel, edge in spec.affects]}
        for name in applicable for spec in [PARAMETERS[name]]
    }
    return json.dumps({"garment": pattern.garment_type, "fabric": pattern.fabric,
                       "failed_checks": failed, "parameters": params})


def rounds_table_md(logs: list[RoundLog]) -> str:
    """Render the round log as a Markdown table."""
    lines = ["| Ronde | Gagal | Usulan | Keputusan | Lokalitas | Hasil |",
             "|-------|-------|--------|-----------|-----------|-------|"]
    for log in logs:
        proposed = ", ".join(f"{k}={v}" for k, v in log.proposed.items()) or "—"
        locality = "—" if log.locality_passed is None else ("lulus" if log.locality_passed else "gagal")
        lines.append(f"| {log.round} | {', '.join(log.failed) or '—'} | {proposed} | "
                     f"{log.decision}: {log.reason} | {locality} | {log.outcome} |")
    return "\n".join(lines) + "\n"


def repair_loop(measurements, fabric: str, params: Mapping[str, float] | None = None,
                max_rounds: int = 5, garment: str = "blouse",
                llm_fn: Callable[[str, str], dict[str, Any]] | None = None,
                ) -> tuple[Pattern, Report, list[RoundLog]]:
    """Draft, verify and correct up to max_rounds; return (pattern, report, logs)."""
    from seamproof.params import resolve_params

    if garment not in REGISTRY:
        raise ValueError(f"Unknown garment: {garment}")
    applicable = tunable_params(garment)
    base = resolve_params(params)
    draft = REGISTRY[garment]

    def build(overrides: Mapping[str, float]) -> Pattern:
        merged = dict(base)
        merged.update({k: float(v) for k, v in overrides.items()})
        return draft(measurements, fabric, merged)

    pattern = build({})
    report = verify_pattern(pattern)
    logs: list[RoundLog] = []
    if report.passed or not applicable:
        # Nothing the LLM may tune: report the failure honestly, no rounds.
        return pattern, report, logs
    for number in range(1, max_rounds + 1):
        failed = [c.id for c in report.failures()]
        log = RoundLog(round=number, failed=failed)
        try:
            if llm_fn is not None:
                raw = llm_fn(SYSTEM_PROMPT, _payload(pattern, report, applicable))
            else:
                raw = llm_module.call_json(SYSTEM_PROMPT, _payload(pattern, report, applicable),
                                           REPAIR_SCHEMA, f"repair-{garment}-round{number}")
        except Exception as exc:
            log.decision, log.reason = "rejected", f"LLM gagal: {type(exc).__name__}"
            log.outcome = f"tersisa: {', '.join(failed)}"
            logs.append(log)
            continue
        changes, reason, error = _check_proposal(raw, applicable)
        log.proposed = dict(raw.get("changes", {})) if isinstance(raw, dict) else {}
        if error:
            log.decision, log.reason = "rejected", error
            log.outcome = f"tersisa: {', '.join(failed)}"
            logs.append(log)
            continue
        try:
            candidate = build(changes)
        except ValueError as exc:
            log.decision, log.reason = "rejected", f"draft ulang gagal: {exc}"
            log.outcome = f"tersisa: {', '.join(failed)}"
            logs.append(log)
            continue
        locality = edit_locality(pattern, candidate, changes)
        log.locality_passed = locality.passed
        if not locality.passed:
            log.decision, log.reason = "rejected", locality.message
            log.outcome = f"tersisa: {', '.join(failed)}"
            logs.append(log)
            continue
        pattern = candidate
        report = verify_pattern(pattern)
        remaining = [c.id for c in report.failures()]
        log.decision, log.reason = "accepted", reason
        log.outcome = (f"{len(report.checks)}/{len(report.checks)} lulus" if report.passed
                       else f"tersisa: {', '.join(remaining)}")
        logs.append(log)
        if report.passed:
            break
    return pattern, report, logs
