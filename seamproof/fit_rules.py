"""Deterministic fit-issue rules: each issue maps to bounded param deltas.

All amounts are assumptions to tune with real toiles, in mm unless noted.
Severity 1..3 interpolates linearly between the lo and hi bound. A rule with
no param entry is advice-only (e.g. ease changes need a redraft decision, not
a silent tweak); the model never chooses numbers, it only names issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleDelta:
    """Bounded change for one parameter; sign sets the direction."""

    param: str
    lo_mm: float
    hi_mm: float


@dataclass(frozen=True)
class FitRule:
    """One fittable issue: bounded deltas plus tailor advice."""

    key: str
    description_id: str
    deltas: tuple[RuleDelta, ...] = ()
    advice_id: str = ""
    assumptions: tuple[str, ...] = ()


# Assumption: amounts below are starting values from fitting practice, not
# measured standards. drag_lines uses dart rotation (deg) because dart intake
# is derived from bust-high_bust, not a tunable parameter. body tightness has
# no width parameter, so it stays advice-only (add ease / size up at redraft).
RULES: dict[str, FitRule] = {
    "pulling_at_underarm": FitRule(
        "pulling_at_underarm", "Pulling at the underarm",
        (RuleDelta("armhole_curve_depth", 5.0, 10.0),),
        "Loosen the armhole per the table."),
    "sleeve_cap_puckering": FitRule(
        "sleeve_cap_puckering", "Sleeve cap puckering",
        (RuleDelta("sleeve_cap_height", -8.0, -3.0),),
        "Lower the sleeve cap per the table."),
    "shoulder_seam_drops": FitRule(
        "shoulder_seam_drops", "Shoulder seam drops",
        (RuleDelta("shoulder_slope", 3.0, 8.0),),
        "Adjust the shoulder slope per the table.",
        ("Slope direction follows the shoulder_slope definition (drop from the neck).",)),
    "neckline_gapes": FitRule(
        "neckline_gapes", "Neckline gapes",
        (RuleDelta("neck_drop_front", -6.0, -3.0),),
        "Raise the front neckline per the table."),
    "body_too_tight_chest": FitRule(
        "body_too_tight_chest", "Body too tight at the chest",
        (),
        "Add ease within the fabric range or size up when redrafting.",
        ("There is no body width parameter; an ease change needs a redraft decision.",)),
    "drag_lines_toward_bust": FitRule(
        "drag_lines_toward_bust", "Drag lines toward the bust",
        (RuleDelta("bust_dart_angle", 1.0, 3.0),),
        "Rotate the dart per the table.",
        ("The sketch asks for mm of intake; intake is derived (bust-high_bust), "
         "so dart rotation in degrees is used.",)),
    "dart_point_too_pointed": FitRule(
        "dart_point_too_pointed", "Dart point too pointed",
        (RuleDelta("bust_dart_clearance", 5.0, 10.0),),
        "Move the dart tip away from the apex per the table."),
    "gaping_front_armhole": FitRule(
        "gaping_front_armhole", "Front armhole gapes",
        (RuleDelta("armhole_curve_depth", -6.0, -3.0),),
        "Tighten the front armhole per the table.",
        ("Moving it into a dart is not automatic; that needs a new pattern.",)),
    "hem_swings_forward": FitRule(
        "hem_swings_forward", "Hem swings forward",
        (RuleDelta("hem_curve", 2.0, 5.0),),
        "Adjust the hem curve per the table."),
    "tunic_flare_too_wide": FitRule(
        "tunic_flare_too_wide", "Tunic flare too wide",
        (RuleDelta("side_flare", -30.0, -10.0),),
        "Reduce the side flare per the table."),
}

LABEL_ID = "Suggested correction — confirm before cutting"


def delta_for(rule: FitRule, severity: int) -> dict[str, float]:
    """Interpolate each bounded delta by severity 1..3."""
    if severity not in (1, 2, 3):
        raise ValueError("Severity must be 1, 2 or 3.")
    out = {}
    for item in rule.deltas:
        out[item.param] = item.lo_mm + (item.hi_mm - item.lo_mm) * (severity - 1) / 2
    return out
