"""Muse Spark writes the words around computed results; it never supplies a number.

Used for: the pattern story (verdict in tailor words, sewing order, toile
checklist), production notes (cutting-room steps, owner summary) and the
fabric suggestion. Each call gets a JSON of facts computed by code. Every
number in the model's text must equal one of those facts (or its mm->cm/m
reading) at the precision the model wrote it; one unknown number discards the
whole answer and a deterministic template is used instead. The fabric choice
must be one of the candidates code already verified.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from seamproof import llm as llm_module
from seamproof.pattern import Pattern
from seamproof.verify import Report

LLMFn = Callable[[str, str], dict[str, Any]]

PANEL_ID = {"front": "front bodice", "back": "back bodice", "sleeve": "sleeve", "facing": "neck facing",
            "collar": "collar", "cuff": "cuff", "pocket": "pocket", "neckband": "neckband",
            "skirt_front": "front skirt", "skirt_back": "back skirt", "hem_facing": "hem facing",
            "waist_facing": "waist facing", "waistband": "waistband",
            "armhole_bias": "armhole bias", "neck_bias": "neck bias"}
SKIRT_PANEL_ID = {"front": "front skirt", "back": "back skirt"}
EDGE_ID = {"shoulder": "shoulder", "side_upper": "upper side", "side_lower": "lower side", "side_low": "lower side",
           "side": "side", "armhole": "armhole", "cap": "sleeve cap", "neckline": "neckline",
           "neck": "neck", "underarm_front": "underarm seam", "underarm_back": "underarm seam",
           "waist_cf": "waist", "waist_side": "waist", "hem": "hem", "scallop": "scallop hem",
           "attach": "joining edge", "cuff_join": "sleeve hem", "edge": "front edge", "slit": "slit",
           "cb": "centre back", "collar_line": "collar line",
           "dart_leg1": "dart leg", "dart_leg2": "dart leg",
           "wdart_leg1": "waist dart leg", "wdart_leg2": "waist dart leg"}
# How a finish piece is sewn on, in words (no numbers), so Muse orders the steps right.
FINISH_ID = {
    "armhole_bias": ("Armhole bias: sew the shoulder and side first, join both bias ends into a ring, "
                     "then attach the bias folded double to the armhole."),
    "neck_bias": ("Neck bias: two bias strips joined at centre back, attached from the left front edge "
                  "to the right front edge, ends folded in at the front edge (not a ring)."),
    "facing": "Neck facing: attach after the shoulders are sewn, turn it inside, then understitch.",
}
ZIP_ID = "Concealed zip on the left side: leave an opening in the left side seam and set the zip before the hem."
CONSTRUCTION_ID = {"waist_seam": "bodice and skirt joined at the waist", "shift": "one piece with no waist seam"}
GARMENT_ID = {"tshirt": "T-shirt", "blouse": "blouse", "tunic": "tunic", "shirt": "shirt", "skirt": "skirt",
              "dress": "dress", "bodice": "bodice block"}
FABRIC_ID = {"knit_low_stretch": "low-stretch knit (combed cotton, pique)",
             "knit_high_stretch": "high-stretch knit (spandex, jersey)",
             "woven": "woven (cotton, linen, rayon, toyobo)"}

_RULES = ("Use ONLY numbers found in the facts data (mm may be written as cm or m). Do NOT compute, "
          "add up, re-round or guess numbers. Spell counts as words (two, three) and do not number "
          "the steps. Write plain English for garment workshop tailors; do not write technical ids "
          "(dotted or underscored words), use the plain names from the data.")

# Assumption, as in explain.py: a dot or comma before exactly three digits is a thousands
# separator ("1.500", "1,500"); otherwise either one is the decimal mark ("17,78", "17.5").
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _parse(token: str) -> tuple[float, int] | None:
    """Read one number token; return (value, decimals written)."""
    text = re.sub(r"[.,](?=\d{3}(?:\D|$))", "", token)
    if text.count(",") > 1 or (text.count(".") and text.count(",")):
        return None
    text = text.replace(",", ".")
    if text.count(".") > 1:
        return None
    decimals = len(text.split(".")[1]) if "." in text else 0
    try:
        return float(text), decimals
    except ValueError:
        return None


def fact_numbers(facts: Any, key: str = "") -> list[float]:
    """Every number in a facts JSON, plus cm and m readings of *_mm values."""
    out: list[float] = []
    if isinstance(facts, bool):
        return out
    if isinstance(facts, (int, float)):
        out.append(float(facts))
        if key.endswith("_mm"):
            out += [float(facts) / 10.0, float(facts) / 1000.0]
    elif isinstance(facts, dict):
        for name, value in facts.items():
            out += fact_numbers(value, str(name))
    elif isinstance(facts, (list, tuple)):
        for value in facts:
            out += fact_numbers(value, key)
    return out


def numbers_ok(texts: list[str], allowed: list[float]) -> bool:
    """True when every number written in the texts is a fact at that precision."""
    for text in texts:
        for token in _NUM.findall(text):
            parsed = _parse(token)
            if parsed is None:
                return False
            value, decimals = parsed
            if not any(math.isclose(round(a, decimals), value, abs_tol=10 ** -(decimals + 3)) for a in allowed):
                return False
    return True


def _call(system: str, facts: dict[str, Any], schema: dict[str, Any], purpose: str,
          llm_fn: LLMFn | None) -> dict[str, Any] | None:
    """Ask Muse Spark (or the injected fake); None on any failure."""
    user = json.dumps(facts, ensure_ascii=False)
    try:
        return llm_fn(system, user) if llm_fn is not None else llm_module.call_json(system, user, schema, purpose)
    except Exception:  # model down or invalid: the template takes over
        return None


def _strings(raw: Any, key: str, limit: int) -> list[str] | None:
    """A non-empty list of short strings from the model, or None."""
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items or not all(isinstance(i, str) and i.strip() for i in items):
        return None
    return [i.strip()[:300] for i in items[:limit]]


def _r(value: float | None) -> float | None:
    """Round a fact to 0.1 so the model sees tidy numbers."""
    return None if value is None else round(float(value), 1)


# ---------------------------------------------------------------- pattern story

@dataclass
class PatternStory:
    """Verdict in words, sewing order and toile checklist for one pattern."""

    summary: str
    sewing_steps: list[str] = field(default_factory=list)
    toile_checks: list[str] = field(default_factory=list)
    source: str = "template"


PATTERN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"},
                   "sewing_steps": {"type": "array", "items": {"type": "string"}},
                   "toile_checks": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "sewing_steps", "toile_checks"],
}

PATTERN_SYSTEM = (
    "You are the head tailor in a garment workshop. From the pattern facts (computed by code, already verified) write JSON: "
    '{"summary": "two or three sentences: what the verification result means for the tailor", '
    '"sewing_steps": ["sewing order, one sentence per step, six to twelve steps"], '
    '"toile_checks": ["four to eight things to LOOK AT on the toile when worn: how the fabric falls, '
    'pulling, puckering, dart tip position, hem balance; do not repeat check numbers"]}. '
    "The sewing order follows workshop habit: darts first, then shoulder, neck, sleeve, side, hem; "
    "use the seams list. When 'finishes' are present, their order and method MUST be followed "
    "(e.g. armhole bias goes on after the side is sewn). Use numbers only when needed (e.g. seam allowance). " + _RULES
)


def pattern_facts(pattern: Pattern, report: Report) -> dict[str, Any]:
    """Facts the pattern story may use, all computed by code."""
    names = dict(PANEL_ID, **(SKIRT_PANEL_ID if pattern.garment_type == "skirt" else {}))
    edges = {e.id: (panel.name, e) for panel in pattern.panels for e in panel.edges}
    seams = set()
    for edge_id, (panel, edge) in edges.items():
        partner = edge.seam_partner_id
        if partner and partner in edges and partner != edge_id:
            other = edges[partner][0]
            seams.add(tuple(sorted((_edge_name(names, panel, edge_id), _edge_name(names, other, partner)))))
    allowance = pattern.metadata.get("seam_allowance_mm") or {"default": 10, "hem": 25, "fold": 0}
    darts = {name: {k: _r(v) for k, v in d.items() if k.endswith("_mm")}
             for name, d in (pattern.metadata.get("darts") or {}).items() if isinstance(d, dict)}
    return {
        "garment": GARMENT_ID.get(pattern.garment_type, pattern.garment_type), "size": pattern.size_label,
        "fabric": FABRIC_ID.get(pattern.fabric, pattern.fabric), "passed": report.passed,
        "panels": [{"name": names.get(p.name, p.name), "id": p.name, "cut": p.cut_count,
                   "on_fold": p.cut_on_fold} for p in pattern.panels],
        "seams": sorted(list(pair) for pair in seams),
        "seam_allowance_mm": {"regular_mm": allowance.get("default", 10), "hem_mm": allowance.get("hem", 25)},
        "darts": darts,
        "gathers": {f"{k}_mm": round(float(v), 1) for k, v in (pattern.metadata.get("gathers_mm") or {}).items()
                    if float(v) > 0},
        "construction": CONSTRUCTION_ID.get(pattern.metadata.get("construction", ""), ""),
        "finishes": [FINISH_ID[p.name] for p in pattern.panels if p.name in FINISH_ID]
        + ([ZIP_ID] if pattern.garment_type in ("dress", "skirt") else []),
        "checks": [{"id": c.id, "name": c.name_id, "passed": c.passed, "actual_mm": _r(c.actual),
                         "expected_mm": _r(c.expected), "tolerance_mm": _r(c.tolerance)} for c in report.checks],
    }


def _edge_name(names: dict[str, str], panel: str, edge_id: str) -> str:
    """Plain 'panel — edge' name for an edge id like front.shoulder."""
    suffix = edge_id.split(".", 1)[-1]
    return f"{names.get(panel, panel)} — {EDGE_ID.get(suffix, suffix.replace('_', ' '))}"


def _pattern_template(pattern: Pattern, report: Report) -> PatternStory:
    """Deterministic story when Muse is unavailable or wrote a stray number."""
    total, failed = len(report.checks), report.failures()
    label = GARMENT_ID.get(pattern.garment_type, pattern.garment_type)
    summary = (f"All {total} checks passed — the {label} {pattern.size_label} pattern is ready for a toile."
               if not failed else f"{len(failed)} of {total} checks failed: "
               + ", ".join(f"{c.id} {c.name_id}" for c in failed) + ". Fix before the toile.")
    checks = [f"Check the {c.name_id.lower()} on the toile ({c.id})." for c in report.checks
              if c.id[0] in "SWDK" and c.passed][:6]
    return PatternStory(summary, [], checks, "template")


def narrate_pattern(pattern: Pattern, report: Report, llm_fn: LLMFn | None = None) -> PatternStory:
    """Muse writes the verdict, sewing order and toile checks; numbers are guarded."""
    facts = pattern_facts(pattern, report)
    raw = _call(PATTERN_SYSTEM, facts, PATTERN_SCHEMA, f"story-{pattern.garment_type}", llm_fn)
    if not isinstance(raw, dict) or not isinstance(raw.get("summary"), str):
        return _pattern_template(pattern, report)
    steps, checks = _strings(raw, "sewing_steps", 14), _strings(raw, "toile_checks", 10)
    summary = raw["summary"].strip()[:600]
    if not summary or steps is None or checks is None:
        return _pattern_template(pattern, report)
    if not numbers_ok([summary, *steps, *checks], fact_numbers(facts)):
        return _pattern_template(pattern, report)
    return PatternStory(summary, steps, checks, "muse")


# ------------------------------------------------------------ production notes

@dataclass
class ProductionStory:
    """Cutting-room steps and a short owner summary for one production run."""

    cutting_steps: list[str] = field(default_factory=list)
    owner_summary: str = ""
    source: str = "template"


PRODUCTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"cutting_steps": {"type": "array", "items": {"type": "string"}},
                   "owner_summary": {"type": "string"}},
    "required": ["cutting_steps", "owner_summary"],
}

PRODUCTION_SYSTEM = (
    "You are the head cutter in a garment workshop. From the production facts (computed by code) write JSON: "
    '{"cutting_steps": ["spreading and cutting steps per size: how many spreads, how many layers, '
    'marker length; one sentence per step"], '
    '"owner_summary": "two or three sentences for the owner: total fabric, cost, sizes not yet passed"}. '
    + _RULES
)


def production_facts(report) -> dict[str, Any]:
    """Facts for production notes; spreads per size are computed here, not by the model."""
    sizes = []
    for item in report.sizes:
        sizes.append({"size": item.size, "quantity": item.qty, "passed": item.verified,
                      "spreads": math.ceil(item.qty / report.layers), "layers_per_spread": report.layers,
                      "marker_length_mm": round(item.marker.used_length_mm),
                      "fabric_metres": round(item.metres, 2),
                      "utilisation_percent": round(item.marker.utilisation * 100, 1),
                      "pieces_per_marker": len(item.marker.placements)})
    return {"garment": GARMENT_ID.get(report.garment, report.garment),
            "fabric": FABRIC_ID.get(report.fabric, report.fabric),
            "fabric_width_mm": round(report.fabric_width_mm), "price_per_metre": round(report.price_per_m),
            "sizes": sizes, "total_metres": round(report.total_metres, 2),
            "total_cost_rupiah": round(report.total_cost),
            "total_units": sum(item.qty for item in report.sizes),
            "overall_utilisation_percent": round(report.overall_utilisation * 100, 1),
            "cut_list": dict(report.cut_list)}


def _production_template(report) -> ProductionStory:
    """Deterministic cutting steps quoting only computed numbers."""
    steps = []
    for item in report.sizes:
        if not item.verified:
            steps.append(f"Size {item.size}: not verified yet — do not cut it.")
            continue
        steps.append(f"Size {item.size}: {math.ceil(item.qty / report.layers)} spreads, "
                     f"{report.layers} layers per spread, marker {item.marker.used_length_mm / 1000:.2f} m.")
    return ProductionStory(steps, report.summary_id(), "template")


def narrate_production(report, llm_fn: LLMFn | None = None) -> ProductionStory:
    """Muse writes cutting steps and the owner summary; numbers are guarded."""
    facts = production_facts(report)
    raw = _call(PRODUCTION_SYSTEM, facts, PRODUCTION_SCHEMA, f"production-{report.garment}", llm_fn)
    steps = _strings(raw, "cutting_steps", 12)
    owner = raw.get("owner_summary") if isinstance(raw, dict) else None
    if steps is None or not isinstance(owner, str) or not owner.strip():
        return _production_template(report)
    owner = owner.strip()[:600]
    if not numbers_ok([owner, *steps], fact_numbers(facts)):
        return _production_template(report)
    return ProductionStory(steps, owner, "muse")


# ------------------------------------------------------------ fabric suggestion

FABRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"fabric": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["fabric", "reason"],
}

FABRIC_SYSTEM = (
    "You are the fabric adviser in a garment workshop. Choose ONE category from 'candidates' (all have passed "
    "pattern verification) that best suits this design, then give a short reason and example fabrics "
    'that are easy to buy. JSON: {"fabric": "<candidate id>", "reason": "one or two sentences"}. ' + _RULES
)


@dataclass
class FabricSuggestion:
    """One fabric category chosen among verified candidates, with a reason."""

    fabric: str
    reason: str
    source: str = "template"


def suggest_fabric(garment: str, candidates: list[str], features: dict[str, Any] | None = None,
                   brief: str = "", llm_fn: LLMFn | None = None) -> FabricSuggestion:
    """Muse picks among verified candidates; the first candidate is the fallback."""
    if not candidates:
        raise ValueError("No fabric category passes verification for this design.")
    facts = {"garment": GARMENT_ID.get(garment, garment), "style": features or {}, "brief": brief[:2000],
             "candidates": [{"id": c, "name": FABRIC_ID.get(c, c)} for c in candidates]}
    fallback = FabricSuggestion(candidates[0], f"{FABRIC_ID.get(candidates[0], candidates[0]).capitalize()} "
                                f"passes verification for this {GARMENT_ID.get(garment, garment)}.")
    raw = _call(FABRIC_SYSTEM, facts, FABRIC_SCHEMA, f"fabric-{garment}", llm_fn)
    if not isinstance(raw, dict) or raw.get("fabric") not in candidates or not isinstance(raw.get("reason"), str):
        return fallback
    reason = raw["reason"].strip()[:400]
    if not reason or not numbers_ok([reason], fact_numbers(facts)):
        return fallback
    return FabricSuggestion(raw["fabric"], reason, "muse")
