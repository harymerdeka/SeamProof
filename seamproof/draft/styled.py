"""Style finishes of the darted blouse and tunic: sleeve, sleeveless and gathered variants.

Assumptions to tune with samples (a pattern maker should review):
- Sleeveless and a shortened plain sleeve work as on the dress block.
- A cuffed or gathered sleeve (cuff_depth > 0) is the shirt sleeve, its cap
  solved against the darted block's armhole, with no silhouette ease added.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from seamproof.draft.dress import _arm_finish
from seamproof.draft.finish import SLEEVELESS_CUT_IN_MM, SLEEVELESS_RAISE_MM
from seamproof.draft.shirt import ELASTIC_UNDER_MM, _cuff, _sleeve, arm_girth_at, cuff_band_length, gathered
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.pattern import Panel, Pattern, edge_length


def styled_block(m: WomenMeasurements, p: Mapping[str, float]) -> WomenMeasurements:
    """Measurements to draft on: a sleeveless block has a higher, narrower armhole."""
    if p["sleeveless"] < 0.5:
        return m
    if p["cuff_depth"] > 0 or p["sleeve_hem_fullness"] > 0 or p["sleeve_cap_gather"] > 0:
        raise ValueError("A sleeveless top has no sleeve, cuff or gathers")
    return replace(m, armhole_depth=m.armhole_depth - SLEEVELESS_RAISE_MM,
                   shoulder_length=m.shoulder_length - SLEEVELESS_CUT_IN_MM)


def _cuffed_sleeve(pattern: Pattern, m: WomenMeasurements, p: Mapping[str, float]) -> None:
    """Swap the plain sleeve for a cuffed or gathered one and record its gathers."""
    edges = {e.id: e for panel in pattern.panels for e in panel.edges}
    armholes = edge_length(edges["front.armhole"]) + edge_length(edges["back.armhole"])
    low, high = FABRIC_EASE[pattern.fabric].sleeve_cap_ease_range_mm
    sleeve = _sleeve(m, {**p, "design_ease": 0.0}, armholes, (low + high) / 2)
    cap_height = next(e for e in sleeve.edges if e.id == "sleeve.underarm_back").curve.points[0].y
    band = cuff_band_length(m, p, cap_height) if gathered(p) else None
    cuff: Panel = _cuff(sleeve, p["cuff_depth"], band)
    pattern.panels = [sleeve if panel.name == "sleeve" else panel for panel in pattern.panels] + [cuff]
    hem = next(edge_length(e) for e in sleeve.edges if e.id == "sleeve.cuff_join")
    pattern.metadata["sleeve_cap_ease_mm"] = edge_length(sleeve.edges[0]) - armholes
    pattern.metadata["gathers_mm"] = {
        "sleeve_hem": hem - band if band is not None else 0.0, "sleeve_cap": p["sleeve_cap_gather"],
        "elastic_cut": (arm_girth_at(m, p["shirt_sleeve_length"], cap_height) - ELASTIC_UNDER_MM
                        if p["cuff_elastic"] >= 0.5 else 0.0)}


def finish_arms(pattern: Pattern, m: WomenMeasurements, p: Mapping[str, float]) -> None:
    """Apply the chosen arm finish in place; the pattern keeps the real body measurements."""
    sleeveless = p["sleeveless"] >= 0.5
    if not sleeveless and p["cuff_depth"] > 0:
        _cuffed_sleeve(pattern, m, p)
    else:
        length = 0.0 if sleeveless else min(p["blouse_sleeve_length"], m.sleeve_length)
        pattern.panels = _arm_finish(pattern.panels, length, sleeveless)
    pattern.size_label = next((name for name, values in WOMEN_SIZE_TABLE.items() if values == m), "custom")
    pattern.measurements = m.to_dict()
    pattern.metadata["sleeveless"] = sleeveless
    if sleeveless:
        pattern.metadata.pop("sleeve_cap_ease_mm", None)
