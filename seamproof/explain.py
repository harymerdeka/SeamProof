"""Tailor-language explanations for verifier reports, in Indonesian.

The LLM only turns verifier numbers into short practical instructions; it
never computes. Every number in the model text is extracted by regex and must
equal a number from the failed checks (or its mm<->cm conversion); any
mismatch discards the model text and falls back to a deterministic template.
Passing patterns get a template summary with no LLM call at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from seamproof import llm as llm_module
from seamproof.pattern import Pattern
from seamproof.verify import Report

EXPLAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "panel": {"type": "string"},
                    "edge": {"type": "string"},
                    "action": {"type": "string"},
                    "amount_mm": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["panel", "edge", "action", "amount_mm", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["instructions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Kamu menulis instruksi perbaikan pola untuk penjahit konveksi. "
    "Setiap instruksi: panel mana, tepi (edge) mana, titik/acuan mana, arah mana, berapa mm. "
    "Gunakan HANYA angka-angka yang diberikan (satuan mm, boleh ditulis dalam cm). "
    "Jangan menulis angka lain apa pun: tulis hitungan dengan kata (satu, dua). "
    "Jangan menghitung ulang atau mengubah angka. "
    "Pakai id panel dan id tepi persis seperti daftar yang diberikan. "
    "Balas hanya satu objek JSON mentah: "
    '{"instructions": [{"panel": "...", "edge": "...", "action": "...", "amount_mm": 0, "text": "..."}]}.'
)

# Assumption: Indonesian decimals use a comma ("9,5"); dots inside digit runs
# are thousand separators ("1.000" = 1000). Bare numbers mean mm.
_NUMBER_RE = re.compile(r"(\d[\d.]*(?:,\d+)?)\s*(cm|mm|sentimeter|milimeter|m\b)?",
                        re.IGNORECASE)


@dataclass
class Instruction:
    """One tailor instruction: where, what to do, how far, in words."""

    panel: str
    edge: str
    action: str
    amount_mm: float | None
    text: str


def _fmt(value: float | None) -> str:
    """Format a verifier number for template text."""
    return "—" if value is None else f"{value:.1f}"


def _numbers_in_mm(text: str) -> list[float]:
    """Extract every number from prose, converted to mm (cm x10)."""
    found = []
    for raw, unit in _NUMBER_RE.findall(text):
        cleaned = re.sub(r"(\d)\.(\d{3}\b)", r"\1\2", raw).replace(",", ".")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if (unit or "").lower().startswith(("cm", "sentimeter")):
            value *= 10.0
        found.append(round(value, 1))
    return found


def _allowed_numbers(report: Report) -> set[float]:
    """Collect verifier numbers plus their cm readings, rounded to 0.1 mm."""
    allowed = set()
    for check in report.failures():
        for value in (check.actual, check.expected, check.tolerance):
            if value is None:
                continue
            allowed.add(round(float(value), 1))
            allowed.add(round(float(value) / 10.0, 1))
    return allowed


def _template_instructions(report: Report) -> list[Instruction]:
    """Deterministic Indonesian fallback quoting only verifier numbers."""
    out = []
    for check in report.failures():
        if check.actual is not None and check.expected is not None:
            amount: float | None = abs(check.actual - check.expected)
        else:
            amount = check.actual if check.actual is not None else check.expected
        out.append(Instruction(
            panel="—", edge="—", action="perbaiki", amount_mm=amount,
            text=(f"{check.id} {check.name_id}: {check.message_id} "
                  f"(nilai {_fmt(check.actual)} mm, harapan {_fmt(check.expected)} mm, "
                  f"batas {_fmt(check.tolerance)} mm).")))
    return out


def _known_ids(pattern: Pattern) -> tuple[set[str], set[str]]:
    """Return known panel names and edge ids for validating proposals."""
    panels = {panel.name for panel in pattern.panels}
    edges = {edge.id for panel in pattern.panels for edge in panel.edges}
    return panels, edges


def _payload(report: Report, pattern: Pattern) -> str:
    """Serialize failed checks with numbers plus valid panel/edge ids."""
    failed = [
        {"id": c.id, "name": c.name_id, "pesan": c.message_id,
         "nilai_mm": c.actual, "harapan_mm": c.expected, "batas_mm": c.tolerance}
        for c in report.failures()
    ]
    panels = [{"panel": panel.name, "tepi": [edge.id for edge in panel.edges]}
              for panel in pattern.panels]
    return json.dumps({"pakaian": report.garment, "ukuran": report.size,
                       "gagal": failed, "panel": panels})


def explain(report: Report, pattern: Pattern, lang: str = "id",
            llm_fn: Callable[[str, str], dict[str, Any]] | None = None
            ) -> tuple[str, list[Instruction]]:
    """Explain a report in tailor Indonesian; return (summary, instructions).

    Passing reports use a template with no LLM call. For failures the LLM
    drafts instructions, but every number in its text (and amount_mm, panel
    and edge ids) is verified; anything unverifiable falls back to templates.
    """
    if lang != "id":
        raise ValueError("Penjelasan hanya tersedia dalam bahasa Indonesia (id).")
    total = len(report.checks)
    failed = report.failures()
    if not failed:
        return (f"Semua {total} pemeriksaan lulus — pola {report.garment} "
                f"{report.size} siap toile.", [])
    summary = (f"Ada {len(failed)} dari {total} pemeriksaan yang gagal "
               f"pada pola {report.garment} {report.size}: "
               + ", ".join(c.id for c in failed) + ".")
    try:
        if llm_fn is not None:
            raw = llm_fn(SYSTEM_PROMPT, _payload(report, pattern))
        else:
            raw = llm_module.call_json(SYSTEM_PROMPT, _payload(report, pattern),
                                       EXPLAIN_SCHEMA, f"explain-{report.garment}")
        drafted = _accept(raw, report, pattern)
    except Exception:
        drafted = None
    return summary, drafted if drafted is not None else _template_instructions(report)


def _accept(raw: Any, report: Report, pattern: Pattern) -> list[Instruction] | None:
    """Validate a model proposal; return None when it must be discarded."""
    if not isinstance(raw, dict) or not isinstance(raw.get("instructions"), list):
        return None
    panels, edges = _known_ids(pattern)
    allowed = _allowed_numbers(report)
    out = []
    for item in raw["instructions"]:
        if not isinstance(item, dict):
            return None
        try:
            panel, edge = str(item["panel"]), str(item["edge"])
            action, text = str(item["action"]), str(item["text"])
            amount = float(item["amount_mm"])
        except (KeyError, TypeError, ValueError):
            return None
        if panel not in panels or edge not in edges:
            return None
        if round(amount, 1) not in allowed:
            return None
        if any(number not in allowed for number in _numbers_in_mm(text)):
            return None
        out.append(Instruction(panel, edge, action, amount, text))
    return out if out else None
