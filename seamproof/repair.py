"""Repair agent: the LLM picks the next action, deterministic code does the work.

Each step sends the LLM the failed checks, the tunable parameters with ranges
and the full step history. It answers with one action: try_params (dry run),
apply_params (commit) or stop. Every proposal passes the same deterministic
gates (known and in-range parameters, redraft, edit locality, re-verify). The
LLM never draws geometry and cannot declare success: only the verifier's report
decides that. Steps are capped by max_rounds and every step is logged.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
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
        "action": {"type": "string"},
        "changes": {"type": "object", "values": {"type": "number"}},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
    "additionalProperties": False,
}

TRY, APPLY, STOP = "try_params", "apply_params", "stop"

SYSTEM_PROMPT = (
    "You correct garment pattern parameters. You NEVER draw geometry or compute "
    "final numbers; you only choose parameter values inside their allowed ranges. "
    "Use the smallest change that fixes the failed checks, using each parameter's "
    "description and affects list. Choose one action per step: "
    f"{TRY} tests changes without saving them (a change that passes every check is "
    f"saved automatically); {APPLY} saves changes that improve the pattern but do not "
    f"finish it, and is refused if it adds a new failed check; {STOP} ends the repair "
    "when no allowed change can help. Read history before acting: it lists every "
    "earlier step, why it was refused, and the checks it left failing (checks). Do not "
    "repeat a refused change. steps_left is how many steps remain. Reply with a single "
    'raw JSON object: {"action": "...", "changes": {param: number}, "reason": "..."}; '
    f"{STOP} takes no changes."
)

# Named bad starting points for demos; values must fail verification.
INJECTS: dict[str, dict[str, float]] = {
    "sleeve_cap_too_high": {"sleeve_cap_height": 160.0},
}


@dataclass
class RoundLog:
    """One agent step: action, proposal, deterministic gates and outcome.

    decision is accepted (saved), tried (dry run, not saved), rejected (refused
    by a gate) or stopped (the agent ended the repair).
    """

    round: int
    action: str = APPLY
    checks: list[dict[str, Any]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    proposed: dict[str, float] = field(default_factory=dict)
    decision: str = ""
    reason: str = ""
    locality_passed: bool | None = None
    outcome: str = ""


def _check_proposal(raw: Any, applicable: list[str]) -> tuple[dict[str, float], str, str]:
    """Validate a proposal shape; return (changes, reason, error)."""
    if not isinstance(raw, dict):
        return {}, "", "the proposal is not a JSON object"
    changes, reason = raw.get("changes"), raw.get("reason", "")
    if not isinstance(changes, dict) or not isinstance(reason, str):
        return {}, "", "the proposal must contain changes (object) and reason (text)"
    if not changes:
        return {}, reason, "a proposal without changes is refused"
    for name in changes:
        if name not in applicable:
            return {}, reason, f"parameter {name} is unknown or does not apply"
    clean: dict[str, float] = {}
    for name, value in changes.items():
        spec = PARAMETERS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return {}, reason, f"{name} must be a real number"
        if not spec.min <= value <= spec.max:
            return {}, reason, f"{name}={value} is outside the {spec.min}–{spec.max} range"
        clean[name] = float(value)
    return clean, reason, ""


def _failed_checks(report: Report) -> list[dict[str, Any]]:
    """List a report's failed checks with the verifier's own numbers."""
    return [
        {"id": c.id, "name": c.name_id, "actual": c.actual, "expected": c.expected,
         "tolerance": c.tolerance, "unit": "mm", "hint": c.message_id}
        for c in report.failures()
    ]


def _payload(pattern: Pattern, report: Report, applicable: list[str], logs: list[RoundLog],
             steps_left: int) -> str:
    """Serialize failed checks, tunable parameters and earlier steps for the LLM, no photos."""
    params = {
        name: {"value": pattern.params[name], "min": spec.min, "max": spec.max,
               "unit": spec.unit, "description": spec.description,
               "affects": [f"{panel}.{edge}" for panel, edge in spec.affects]}
        for name in applicable for spec in [PARAMETERS[name]]
    }
    return json.dumps({"garment": pattern.garment_type, "fabric": pattern.fabric,
                       "failed_checks": _failed_checks(report), "parameters": params,
                       "history": [asdict(log) for log in logs], "steps_left": steps_left})


def rounds_table_md(logs: list[RoundLog]) -> str:
    """Render the round log as a Markdown table."""
    lines = ["| Round | Failed | Proposed | Decision | Locality | Outcome |",
             "|-------|-------|--------|-----------|-----------|-------|"]
    for log in logs:
        proposed = ", ".join(f"{k}={v}" for k, v in log.proposed.items()) or "—"
        locality = "—" if log.locality_passed is None else ("passed" if log.locality_passed else "failed")
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
    current = dict(base)  # grows with every accepted change, so later rounds build on it

    def build(overrides: Mapping[str, float]) -> Pattern:
        merged = dict(current)
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
        logs.append(log)
        log.outcome = f"remaining: {', '.join(failed)}"
        payload = _payload(pattern, report, applicable, logs[:-1], max_rounds - number + 1)
        try:
            if llm_fn is not None:
                raw = llm_fn(SYSTEM_PROMPT, payload)
            else:
                raw = llm_module.call_json(SYSTEM_PROMPT, payload, REPAIR_SCHEMA,
                                           f"repair-{garment}-round{number}")
        except Exception as exc:
            log.decision, log.reason = "rejected", f"LLM failed: {type(exc).__name__}"
            continue
        # A reply without an action is an apply_params.
        action = raw.get("action", APPLY) if isinstance(raw, dict) else APPLY
        log.action = str(action)
        if action == STOP:
            reason = raw.get("reason", "")
            log.decision, log.reason = "stopped", reason if isinstance(reason, str) else ""
            break
        if action not in (TRY, APPLY):
            log.decision, log.reason = "rejected", f"unknown action {action}"
            continue
        changes, reason, error = _check_proposal(raw, applicable)
        proposed = raw.get("changes") if isinstance(raw, dict) else None
        log.proposed = dict(proposed) if isinstance(proposed, dict) else {}
        if error:
            log.decision, log.reason = "rejected", error
            continue
        try:
            candidate = build(changes)
        except ValueError as exc:
            log.decision, log.reason = "rejected", f"redraft failed: {exc}"
            continue
        locality = edit_locality(pattern, candidate, changes)
        log.locality_passed = locality.passed
        if not locality.passed:
            log.decision, log.reason = "rejected", locality.message
            continue
        result = verify_pattern(candidate)
        log.checks = _failed_checks(result)
        remaining = [c.id for c in result.failures()]
        if not result.passed and action == TRY:
            log.decision, log.reason = "tried", reason
            log.outcome = f"if applied, remaining: {', '.join(remaining)}"
            continue
        added = sorted(set(remaining) - set(failed))
        if added:
            log.decision, log.reason = "rejected", f"adds failed checks: {', '.join(added)}"
            continue
        current.update(changes)
        pattern, report = candidate, result
        log.decision, log.reason = "accepted", reason
        log.outcome = (f"{len(report.checks)}/{len(report.checks)} passed" if report.passed
                       else f"remaining: {', '.join(remaining)}")
        if report.passed:
            break
    return pattern, report, logs
