"""Measurement intake: files, size-table bases and Indonesian/English free text.

All values become millimetres. Missing values are never invented: parse
functions return {"measurements": {...mm...}, "missing": [...], "warnings":
[...]} and callers must stop when "missing" is non-empty. Free text is mapped
by the LLM to canonical fields; every range and cross-field rule below runs
deterministically afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from seamproof import llm as llm_module
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE

TSHIRT_FIELDS = ["chest", "waist", "hip", "shoulder_width", "back_length", "sleeve_length",
                 "armhole_depth", "neck_width", "bicep"]
WOMEN_FIELDS = ["bust", "high_bust", "waist", "hip", "shoulder_width", "shoulder_length",
                "front_length", "back_length", "apex_height", "apex_distance", "waist_to_hip",
                "armhole_depth", "sleeve_length", "bicep", "wrist"]
GARMENT_FIELDS = {"tshirt": TSHIRT_FIELDS, "bodice": WOMEN_FIELDS, "blouse": WOMEN_FIELDS,
                  "tunic": WOMEN_FIELDS, "shirt": WOMEN_FIELDS, "skirt": WOMEN_FIELDS,
                  "dress": WOMEN_FIELDS}

# Plausible adult body ranges in mm; starting values, tune with workshops.
RANGES: dict[str, tuple[float, float]] = {
    "chest": (700, 1400), "bust": (700, 1400), "high_bust": (650, 1350),
    "waist": (550, 1300), "hip": (700, 1450), "shoulder_width": (300, 550),
    "shoulder_length": (90, 170), "back_length": (300, 850), "front_length": (320, 550),
    "apex_height": (180, 350), "apex_distance": (120, 280), "waist_to_hip": (150, 280),
    "sleeve_length": (100, 700), "armhole_depth": (150, 320), "neck_width": (100, 250),
    "bicep": (200, 450), "wrist": (120, 250),
}

INTAKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "measurements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "required": ["field", "value", "unit"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["measurements"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You map tailor body-measurement notes to canonical fields. Canonical T-shirt fields: "
    "chest, waist, hip, shoulder_width, back_length, sleeve_length, armhole_depth, neck_width, bicep. "
    "Canonical women's fields: bust, high_bust, waist, hip, shoulder_width, shoulder_length, "
    "front_length, back_length, apex_height, apex_distance, waist_to_hip, armhole_depth, "
    "sleeve_length, bicep, wrist. Indonesian hints: lingkar dada = chest/bust, lingkar pinggang = "
    "waist, lingkar pinggul = hip, lebar bahu = shoulder_width, panjang baju = back_length (or "
    "front_length for women's front), panjang lengan = sleeve_length, kerung lengan = armhole_depth, "
    "lebar leher = neck_width, lingkar lengan = bicep, dada atas = high_bust. A trailing unit applies "
    "to every bare number; per-value units override it. Never invent values: list only what is stated. "
    "Reply with a single raw JSON object: "
    '{"measurements": [{"field": "...", "value": number, "unit": "cm"|"mm"}]}.'
)


def _build_result(items: list[tuple[str, float, str]], garment: str) -> dict[str, Any]:
    """Convert, range-check and cross-check raw items into the intake dict."""
    fields = GARMENT_FIELDS[garment]
    seen: dict[str, float] = {}
    missing: list[str] = []
    warnings: list[str] = []
    for field, value, unit in items:
        if field not in fields:
            warnings.append(f"Field '{field}' is unknown for {garment} — ignored.")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            missing.append(field)
            warnings.append(f"The {field} value is not a number — marked missing.")
            continue
        if unit not in ("cm", "mm"):
            if field not in missing:
                missing.append(field)
            warnings.append(f"The {field} unit must be cm or mm — marked missing.")
            continue
        mm = float(value) * 10 if unit == "cm" else float(value)
        low, high = RANGES[field]
        if mm <= 0 or not low <= mm <= high:
            if field not in missing:
                missing.append(field)
            warnings.append(f"{field} {mm:.0f} mm is outside {low:.0f}–{high:.0f} — marked missing.")
            continue
        seen[field] = mm
    for field in fields:
        if field not in seen and field not in missing:
            missing.append(field)
    warnings.extend(_cross_checks(seen, garment))
    return {"measurements": seen, "missing": missing, "warnings": warnings}


def _cross_checks(seen: dict[str, float], garment: str) -> list[str]:
    """Sanity rules between fields; values are kept, problems become warnings."""
    warnings = []
    chest = seen.get("bust", seen.get("chest", 0))
    if "waist" in seen and chest and seen["waist"] > chest + 150:
        warnings.append("Waist is much larger than chest/bust — check the numbers.")
    if "neck_width" in seen and "shoulder_width" in seen and seen["neck_width"] >= seen["shoulder_width"]:
        warnings.append("Neck is wider than the shoulders — check the numbers.")
    if "armhole_depth" in seen and "back_length" in seen and seen["armhole_depth"] >= seen["back_length"]:
        warnings.append("Armhole is deeper than the back length — check the numbers.")
    if "bust" in seen and "high_bust" in seen and seen["bust"] <= seen["high_bust"]:
        warnings.append("Bust must be larger than high bust — check the numbers.")
    if "front_length" in seen and "apex_height" in seen and seen["front_length"] <= seen["apex_height"]:
        warnings.append("Front length must pass the bust apex — check the numbers.")
    if "apex_distance" in seen and "shoulder_width" in seen and seen["apex_distance"] >= seen["shoulder_width"]:
        warnings.append("Apex distance exceeds the shoulder width — check the numbers.")
    return warnings


def parse_file(path: str | Path) -> dict[str, Any]:
    """Load JSON/YAML measurements with a required unit field and garment guess.

    File shape: {"garment": ..., "unit": "cm"|"mm", "measurements": {...}}.
    Garment defaults to tshirt when the fields fit, else blouse.
    """
    import yaml

    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw) if str(path).endswith(".json") else yaml.safe_load(raw)
    if not isinstance(data, dict) or not isinstance(data.get("measurements"), dict):
        raise ValueError("The file must contain unit and measurements.")
    unit = data.get("unit")
    if unit not in ("cm", "mm"):
        raise ValueError("The file must state a unit: cm or mm.")
    garment = data.get("garment") or _guess_garment(set(data["measurements"]))
    if garment not in GARMENT_FIELDS:
        raise ValueError(f"Unknown garment: {garment}.")
    items = [(str(field), value, unit) for field, value in data["measurements"].items()]
    return _build_result(items, garment)


def _guess_garment(names: set[str]) -> str:
    """Default to tshirt when every name fits it, else blouse."""
    if names and names <= set(TSHIRT_FIELDS):
        return "tshirt"
    return "blouse"


def parse_text(text: str, garment: str = "tshirt",
               llm_fn: Callable[[str, str], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Map free Indonesian/English text to millimetre measurements via the LLM."""
    if garment not in GARMENT_FIELDS:
        raise ValueError(f"Unknown garment: {garment}.")
    if llm_fn is not None:
        raw = llm_fn(SYSTEM_PROMPT, text)
    else:
        raw = llm_module.call_json(SYSTEM_PROMPT, text, INTAKE_SCHEMA, f"intake-{garment}")
    if not isinstance(raw, dict) or not isinstance(raw.get("measurements"), list):
        raise ValueError("The LLM did not return a measurement list.")
    items = []
    for entry in raw["measurements"]:
        if isinstance(entry, dict):
            items.append((entry.get("field"), entry.get("value"), entry.get("unit")))
    return _build_result(items, garment)


def from_size_table(garment: str, size: str, overrides_mm: Mapping[str, float]) -> dict[str, Any]:
    """Start from a size-table entry in mm and override the given fields."""
    table = SIZE_TABLE if garment == "tshirt" else WOMEN_SIZE_TABLE
    if garment not in GARMENT_FIELDS or size not in table:
        raise ValueError(f"Unknown size: {garment} {size}.")
    base = table[size].to_dict()
    extra = [name for name in overrides_mm if name not in base]
    merged = dict(base)
    merged.update({name: float(value) for name, value in overrides_mm.items() if name in base})
    result = _build_result([(name, value, "mm") for name, value in merged.items()], garment)
    for name in extra:
        result["warnings"].append(f"Field '{name}' is unknown for {garment} — ignored.")
    return result
