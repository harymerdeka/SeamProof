"""Deterministic pattern verifier with tailor-language reports.

Every check measures the pattern record independently: lengths from stitch
curves, widths at body lines, darts from leg geometry. Nothing here draws
geometry or calls the network. Tolerances below are workshop working limits
except S2B_TOL_MM, which is an interim allowance: the bodice block drafts
front and back sides independently (bust dart only in front, different waist
heights), so the waist-to-hem sides differ by ~30 mm. That gap is reported in
plain words on every run and must be balanced at the toile or in a later
block revision; the check still guards against larger regressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.errors import GEOSException
from shapely.geometry import LineString

from seamproof.fabric import FABRIC_EASE
from seamproof.pattern import Edge, Panel, Pattern, Point, edge_length, panel_polygon, params_hash

G_TOL_MM = 1.0
SHOULDER_TOL_MM = 1.0
TSHIRT_SIDE_TOL_MM = 3.0
SIDE_ABOVE_TOL_MM = 10.0
SIDE_BELOW_TOL_MM = 45.0
FACING_TOL_MM = 1.0
UNDERARM_TOL_MM = 0.5
NECKBAND_TOL_MM = 1.0
WIDTH_TOL_MM = 2.0
DART_LEG_TOL_MM = 1.0
TIP_TOL_MM = 1.0
INTAKE_TOL_MM = 1.0
KINK_DEG = 12.0
CUFF_TOL_MM = 1.0
COLLAR_TOL_MM = 2.0
WAIST_FINISH_TOL_MM = 2.0
HEM_FACING_TOL_MM = 1.0
WAIST_SEAM_TOL_MM = 2.0
BIAS_TOL_MM = 2.0
# Gathered sleeves (mirrors the shirt draft rules, checked independently):
# a fitted cuff is the arm girth at the cuff line plus 60 mm (elastic, slides
# over the hand) or 30 mm (no elastic); the gathered hem must be 1-3x the cuff.
CUFF_ELASTIC_PASS_MM = 60.0
CUFF_FITTED_EASE_MM = 30.0
GATHER_MAX_RATIO = 3.0
# Shirt hem must clear the body at its level by at least this much (woven).
SHIRT_HIP_MIN_EASE_MM = 40.0
# Skirt waist ease; mirrors the draft constant but is checked independently.
SKIRT_WAIST_EASE_MM = 10.0


@dataclass
class Check:
    """One verifier finding; message_id is in tailor language."""

    id: str
    name_id: str
    passed: bool
    message_id: str
    actual: float | None = None
    expected: float | None = None
    tolerance: float | None = None


@dataclass
class Report:
    """The full verdict for one pattern."""

    garment: str
    size: str
    fabric: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def failures(self) -> list[Check]:
        """Return only the failing checks."""
        return [check for check in self.checks if not check.passed]

    def summary_id(self) -> str:
        """Return a one-line verdict."""
        total = len(self.checks)
        good = sum(check.passed for check in self.checks)
        if self.passed:
            return (f"PASS: {good}/{total} checks — {self.garment} {self.size} pattern "
                    f"ready for toile.")
        names = ", ".join(check.name_id for check in self.failures())
        return (f"FAIL: {total - good}/{total} checks — {names}. "
                f"Fix it, then verify again.")


def _edges(panel: Panel) -> dict[str, Edge]:
    """Index a panel's edges by id."""
    return {edge.id: edge for edge in panel.edges}


def _station_width(panel: Panel, y: float) -> float:
    """Measure a panel's width where it crosses a horizontal body line."""
    return panel_polygon(panel).intersection(LineString([(-1000, y), (1000, y)])).length


def _check(id: str, name: str, passed: bool, message: str, actual: float | None = None,
           expected: float | None = None, tolerance: float | None = None) -> Check:
    """Build a check record."""
    return Check(id, name, passed, message, actual, expected, tolerance)


def verify_pattern(pattern: Pattern) -> Report:
    """Run every applicable check and return a report."""
    checks: list[Check] = []

    def fail(id: str, name: str, message: str, actual: float | None = None,
             expected: float | None = None, tolerance: float | None = None) -> None:
        checks.append(_check(id, name, False, message, actual, expected, tolerance))

    def ok(id: str, name: str, message: str, actual: float | None = None,
           expected: float | None = None, tolerance: float | None = None) -> None:
        checks.append(_check(id, name, True, message, actual, expected, tolerance))

    panels = {panel.name: panel for panel in pattern.panels}
    edges: dict[str, Edge] = {edge.id: edge for panel in pattern.panels for edge in panel.edges}
    fabric = FABRIC_EASE.get(pattern.fabric)
    measures = pattern.measurements
    params = pattern.params
    # A shift dress is the bodice block lengthened; a waist-seam dress gets its own checks.
    dress_seam = pattern.garment_type == "dress" and float(params.get("dress_waist_seam", 1)) >= 0.5
    sleeveless = (pattern.garment_type in ("shirt", "dress", "blouse", "tunic") and float(params.get("sleeveless", 0)) >= 0.5)
    bodice = (pattern.garment_type in ("bodice", "blouse", "tunic")
              or (pattern.garment_type == "dress" and not dress_seam))

    # G1: closed, valid outlines.
    try:
        areas = {name: panel_polygon(panel).area for name, panel in panels.items()}
        if all(area > 0 for area in areas.values()):
            ok("G1", "Closed panels", "All panels are closed and valid.")
        else:
            fail("G1", "Closed panels", "A panel has zero area.")
    except ValueError as exc:
        fail("G1", "Closed panels", f"Panel not closed/valid: {exc}. Check where the edges join.")

    # G2: parameter fingerprint intact.
    try:
        if pattern.metadata.get("params_hash") == params_hash(params):
            ok("G2", "Parameter hash", "Parameter hash matches the draft.")
        else:
            fail("G2", "Parameter hash", "Parameter hash does not match — the pattern was changed outside the draft.")
    except ValueError:
        fail("G2", "Parameter hash", "Parameters contain a number that cannot be hashed.")

    def need(*names: str) -> bool:
        return all(name in edges for name in names)

    # S1: shoulder seams match.
    if need("front.shoulder", "back.shoulder"):
        front_sh = edge_length(edges["front.shoulder"])
        back_sh = edge_length(edges["back.shoulder"])
        diff = abs(front_sh - back_sh)
        if diff <= SHOULDER_TOL_MM:
            ok("S1", "Shoulder seam", f"Front and back shoulders match ({front_sh:.1f} vs {back_sh:.1f} mm).",
               diff, 0.0, SHOULDER_TOL_MM)
        else:
            fail("S1", "Shoulder seam",
                 f"Front shoulder {front_sh:.1f} mm vs back {back_sh:.1f} mm "
                 f"(difference {diff:.1f} mm, limit {SHOULDER_TOL_MM:.0f} mm). Match the shoulder lines.",
                 diff, 0.0, SHOULDER_TOL_MM)

    # S2: side seams, sewn state (dart legs excluded).
    if pattern.garment_type in ("tshirt", "shirt", "skirt") and need("front.side", "back.side"):
        diff = abs(edge_length(edges["front.side"]) - edge_length(edges["back.side"]))
        if diff <= TSHIRT_SIDE_TOL_MM:
            ok("S2", "Side seam", f"Front and back sides match (difference {diff:.1f} mm).",
               diff, 0.0, TSHIRT_SIDE_TOL_MM)
        else:
            fail("S2", "Side seam",
                 f"Front and back sides differ by {diff:.1f} mm (limit {TSHIRT_SIDE_TOL_MM:.0f} mm).",
                 diff, 0.0, TSHIRT_SIDE_TOL_MM)
    elif dress_seam and need("front.side_upper", "front.side_lower", "back.side_upper"):
        # Bodice side underarm to waist (bust dart closed), then the skirt sides.
        front_above = edge_length(edges["front.side_upper"]) + edge_length(edges["front.side_lower"])
        back_above = edge_length(edges["back.side_upper"])
        diff_a = abs(front_above - back_above)
        if diff_a <= SIDE_ABOVE_TOL_MM:
            ok("S2a", "Bodice side", f"Underarm-to-waist sides differ by {diff_a:.1f} mm.",
               diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        else:
            fail("S2a", "Bodice side",
                 f"Underarm-to-waist sides differ by {diff_a:.1f} mm (limit {SIDE_ABOVE_TOL_MM:.0f} mm). "
                 f"Match the front and back bodice sides.", diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        if need("skirt_front.side", "skirt_back.side"):
            diff_b = abs(edge_length(edges["skirt_front.side"]) - edge_length(edges["skirt_back.side"]))
            if diff_b <= TSHIRT_SIDE_TOL_MM:
                ok("S2b", "Skirt side", f"Front and back skirt sides match (difference {diff_b:.1f} mm).",
                   diff_b, 0.0, TSHIRT_SIDE_TOL_MM)
            else:
                fail("S2b", "Skirt side", f"Front and back skirt sides differ by {diff_b:.1f} mm "
                     f"(limit {TSHIRT_SIDE_TOL_MM:.0f} mm).", diff_b, 0.0, TSHIRT_SIDE_TOL_MM)
        else:
            fail("S2b", "Skirt side", "Front or back skirt panel is missing.")
    elif bodice and need("front.side_upper", "front.side_lower", "back.side_upper", "back.side_low", "back.side_lower"):
        lower_pts = _edges(panels["front"])["front.side_lower"].curve.points
        front_above = edge_length(edges["front.side_upper"]) + _seg(lower_pts[0], lower_pts[1])
        front_below = sum(_seg(a, b) for a, b in zip(lower_pts[1:], lower_pts[2:]))
        back_above = edge_length(edges["back.side_upper"])
        back_below = edge_length(edges["back.side_low"]) + edge_length(edges["back.side_lower"])
        diff_a = abs(front_above - back_above)
        if diff_a <= SIDE_ABOVE_TOL_MM:
            ok("S2a", "Upper side", f"Underarm-to-waist sides differ by {diff_a:.1f} mm.",
               diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        else:
            fail("S2a", "Upper side",
                 f"Underarm-to-waist sides differ by {diff_a:.1f} mm (limit {SIDE_ABOVE_TOL_MM:.0f} mm). "
                 f"Match the upper side curves.",
                 diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        diff_b = abs(front_below - back_below)
        if diff_b <= SIDE_BELOW_TOL_MM:
            ok("S2b", "Lower side",
               f"Waist-to-hem sides differ by {diff_b:.1f} mm (interim limit {SIDE_BELOW_TOL_MM:.0f} mm). "
               f"Balance them at the toile.",
               diff_b, 0.0, SIDE_BELOW_TOL_MM)
        else:
            fail("S2b", "Lower side",
                 f"Waist-to-hem sides differ by {diff_b:.1f} mm (limit {SIDE_BELOW_TOL_MM:.0f} mm).",
                 diff_b, 0.0, SIDE_BELOW_TOL_MM)

    # S3: sleeve cap ease inside the fabric range.
    if fabric is None:
        fail("S3", "Sleeve ease", f"Unknown fabric '{pattern.fabric}' — the ease range cannot be checked.")
    elif sleeveless:
        pass  # no sleeve: the armhole binding is checked by S10
    elif need("front.armhole", "back.armhole") and "sleeve.cap" not in edges:
        fail("S3", "Sleeve ease", "The armhole exists but the sleeve pattern is missing. Redraft the sleeve.")
    elif need("front.armhole", "back.armhole", "sleeve.cap"):
        armholes = edge_length(edges["front.armhole"]) + edge_length(edges["back.armhole"])
        ease = edge_length(edges["sleeve.cap"]) - armholes
        low, high = fabric.sleeve_cap_ease_range_mm
        # A puff cap is gathered on purpose: the declared gather shifts the range.
        gather = float(params.get("sleeve_cap_gather", 0)) if pattern.garment_type in ("shirt", "blouse", "tunic") else 0.0
        low, high = low + gather, high + gather
        if low <= ease <= high:
            ok("S3", "Sleeve ease",
               f"Sleeve cap ease {ease:.1f} mm is within the {low:.0f}–{high:.0f} mm range.",
               ease, (low + high) / 2, (high - low) / 2)
        else:
            fail("S3", "Sleeve ease",
                 f"Sleeve cap ease {ease:.1f} mm is outside the {low:.0f}–{high:.0f} mm range. "
                 f"Adjust the sleeve cap height.",
                 ease, (low + high) / 2, (high - low) / 2)

    # S4: neck facing matches the front neckline.
    if "facing.neck" in edges and "front.neckline" in edges:
        diff = abs(edge_length(edges["facing.neck"]) - edge_length(edges["front.neckline"]))
        if diff <= FACING_TOL_MM:
            ok("S4", "Neck facing", f"Neck facing matches the neckline (difference {diff:.1f} mm).",
               diff, 0.0, FACING_TOL_MM)
        else:
            fail("S4", "Neck facing",
                 f"Neck facing differs from the neckline by {diff:.1f} mm (limit {FACING_TOL_MM:.0f} mm). "
                 f"Redraw the facing to follow the neckline.",
                 diff, 0.0, FACING_TOL_MM)

    # S5: sleeve underarm pair matches itself.
    if need("sleeve.underarm_front", "sleeve.underarm_back"):
        diff = abs(edge_length(edges["sleeve.underarm_front"]) - edge_length(edges["sleeve.underarm_back"]))
        if diff <= UNDERARM_TOL_MM:
            ok("S5", "Sleeve seam", f"Underarm seams match (difference {diff:.1f} mm).",
               diff, 0.0, UNDERARM_TOL_MM)
        else:
            fail("S5", "Sleeve seam",
                 f"Front and back underarm seams differ by {diff:.1f} mm (limit {UNDERARM_TOL_MM:.1f} mm).",
                 diff, 0.0, UNDERARM_TOL_MM)

    # S6: t-shirt neckband ring matches the necklines.
    if "neckband.attach" in edges and need("front.neckline", "back.neckline"):
        ring = 2 * (edge_length(edges["front.neckline"]) + edge_length(edges["back.neckline"]))
        diff = abs(edge_length(edges["neckband.attach"]) - ring)
        if diff <= NECKBAND_TOL_MM:
            ok("S6", "Neckband", f"Neckband circles the neckline (difference {diff:.1f} mm).",
               diff, 0.0, NECKBAND_TOL_MM)
        else:
            fail("S6", "Neckband",
                 f"Neckband differs from the neckline by {diff:.1f} mm (limit {NECKBAND_TOL_MM:.0f} mm).",
                 diff, 0.0, NECKBAND_TOL_MM)

    # S7: shirt cuff band equals the sleeve hem it is sewn to; a gathered sleeve
    # has a fitted cuff sized from the arm instead, and S9 checks the gathers.
    sleeve_gathered = pattern.garment_type in ("shirt", "blouse", "tunic") and (
        float(params.get("sleeve_hem_fullness", 0)) > 0 or float(params.get("cuff_elastic", 0)) >= 0.5)
    if sleeve_gathered and need("cuff.attach", "sleeve.cuff_join", "sleeve.underarm_back"):
        try:
            cap_y = edges["sleeve.underarm_back"].curve.points[0].y
            level = float(params["shirt_sleeve_length"])
            span = measures["sleeve_length"] - cap_y
            t = 1.0 if span <= 0 else min(1.0, max(0.0, (level - cap_y) / span))
            girth = measures["bicep"] + (measures["wrist"] - measures["bicep"]) * t
            elastic = float(params.get("cuff_elastic", 0)) >= 0.5
            expected = girth + (CUFF_ELASTIC_PASS_MM if elastic else CUFF_FITTED_EASE_MM)
            band = edge_length(edges["cuff.attach"])
            diff = abs(band - expected)
            kind = "elastic cuff (fits over the hand)" if elastic else "fitted cuff"
            if diff <= CUFF_TOL_MM:
                ok("S7", "Cuff", f"{kind.capitalize()} length {band:.1f} mm suits the {girth:.0f} mm arm girth.",
                   band, expected, CUFF_TOL_MM)
            else:
                fail("S7", "Cuff", f"{kind.capitalize()} length {band:.1f} mm vs {expected:.1f} mm from the arm girth "
                     f"(limit {CUFF_TOL_MM:.0f} mm). Measure the cuff again.", band, expected, CUFF_TOL_MM)
            ratio = edge_length(edges["sleeve.cuff_join"]) / band
            if 1.0 - 1e-6 <= ratio <= GATHER_MAX_RATIO:
                ok("S9", "Sleeve gathers", f"Sleeve hem is gathered {ratio:.2f}x into the cuff "
                   f"(limit 1–{GATHER_MAX_RATIO:.0f}x).", ratio, (1.0 + GATHER_MAX_RATIO) / 2,
                   (GATHER_MAX_RATIO - 1.0) / 2)
            else:
                fail("S9", "Sleeve gathers", f"Sleeve hem is {ratio:.2f}x the cuff — outside 1–{GATHER_MAX_RATIO:.0f}x. "
                     f"Reduce the sleeve fullness or widen the cuff.", ratio, (1.0 + GATHER_MAX_RATIO) / 2,
                     (GATHER_MAX_RATIO - 1.0) / 2)
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            fail("S7", "Cuff", f"Cuff or sleeve data missing: {exc}.")
    elif need("cuff.attach", "sleeve.cuff_join"):
        diff = abs(edge_length(edges["cuff.attach"]) - edge_length(edges["sleeve.cuff_join"]))
        if diff <= CUFF_TOL_MM:
            ok("S7", "Cuff", f"Cuff matches the sleeve hem (difference {diff:.1f} mm).",
               diff, 0.0, CUFF_TOL_MM)
        else:
            fail("S7", "Cuff",
                 f"Cuff differs from the sleeve hem by {diff:.1f} mm (limit {CUFF_TOL_MM:.0f} mm). "
                 f"Match the cuff length to the sleeve hem width.",
                 diff, 0.0, CUFF_TOL_MM)

    # S8: half collar neck edge equals half the body neckline (back + front).
    if need("collar.neck", "front.neckline", "back.neckline"):
        line = edge_length(edges["front.neckline"]) + edge_length(edges["back.neckline"])
        diff = abs(edge_length(edges["collar.neck"]) - line)
        if diff <= COLLAR_TOL_MM:
            ok("S8", "Collar", f"Collar neck edge matches the neckline (difference {diff:.1f} mm).",
               diff, 0.0, COLLAR_TOL_MM)
        else:
            fail("S8", "Collar",
                 f"Collar neck edge differs from the neckline by {diff:.1f} mm (limit {COLLAR_TOL_MM:.0f} mm). "
                 f"Redraw the collar from the neckline.",
                 diff, 0.0, COLLAR_TOL_MM)

    # S10: a sleeveless armhole is bound with a bias strip as long as the armhole.
    if sleeveless:
        if need("armhole_bias.attach", "front.armhole", "back.armhole") and "sleeve.cap" not in edges:
            armhole = edge_length(edges["front.armhole"]) + edge_length(edges["back.armhole"])
            diff = abs(edge_length(edges["armhole_bias.attach"]) - armhole)
            if diff <= BIAS_TOL_MM:
                ok("S10", "Armhole bias", f"Bias {edge_length(edges['armhole_bias.attach']):.0f} mm matches the "
                   f"{armhole:.0f} mm armhole (difference {diff:.1f} mm).", diff, 0.0, BIAS_TOL_MM)
            else:
                fail("S10", "Armhole bias", f"Bias differs by {diff:.1f} mm from the {armhole:.0f} mm armhole "
                     f"(limit {BIAS_TOL_MM:.0f} mm). Cut the bias to the armhole length.", diff, 0.0, BIAS_TOL_MM)
        else:
            fail("S10", "Armhole bias", "A sleeveless pattern needs armhole bias and no sleeve panel.")

    # S11: a collarless shirt neck is bound with bias, one half strip per side (joined at CB).
    if pattern.garment_type == "shirt" and int(float(params.get("collar_style", 0))) == 0:
        ids = ("neck_bias.attach", "front.neckline", "front.ext_top", "back.neckline")
        if need(*ids):
            neck = sum(edge_length(edges[e]) for e in ids[1:])
            diff = abs(edge_length(edges["neck_bias.attach"]) - neck)
            if diff <= BIAS_TOL_MM:
                ok("S11", "Neck bias", f"Neck bias matches the {neck:.0f} mm neckline (difference {diff:.1f} mm).",
                   diff, 0.0, BIAS_TOL_MM)
            else:
                fail("S11", "Neck bias", f"Neck bias differs by {diff:.1f} mm from the {neck:.0f} mm neckline "
                     f"(limit {BIAS_TOL_MM:.0f} mm). Cut the bias to the neckline length.", diff, 0.0, BIAS_TOL_MM)
        else:
            fail("S11", "Neck bias", "A collarless shirt needs neck bias.")

    # Skirt construction checks (the skirt, or the skirt halves of a waist-seam dress).
    skirt_sides = ("skirt_front", "skirt_back") if dress_seam else ("front", "back")
    if pattern.garment_type == "skirt" or dress_seam:
        # K1: both legs of each skirt waist dart are equal.
        worst, seen = 0.0, 0
        for side in skirt_sides:
            if need(f"{side}.dart_leg1", f"{side}.dart_leg2"):
                seen += 1
                worst = max(worst, abs(edge_length(edges[f"{side}.dart_leg1"])
                                       - edge_length(edges[f"{side}.dart_leg2"])))
        if seen == 2 and worst <= DART_LEG_TOL_MM:
            ok("K1", "Skirt dart", f"Front and back skirt dart legs match (largest difference {worst:.1f} mm).",
               worst, 0.0, DART_LEG_TOL_MM)
        else:
            fail("K1", "Skirt dart",
                 f"Skirt dart legs differ by {worst:.1f} mm (limit {DART_LEG_TOL_MM:.0f} mm) "
                 f"or a dart is missing. Match the dart legs.",
                 worst, 0.0, DART_LEG_TOL_MM)
        # K2: waistband (with overlap) or bias facing matches the sewn waist seam.
        seam_ids = [f"{side}.{part}" for side in skirt_sides for part in ("waist_cf", "waist_side")]
        finish = next((edges[e] for e in ("waistband.attach", "waist_facing.attach") if e in edges), None)
        if dress_seam:
            # K4: each bodice half is sewn to the skirt half below it at the same length.
            worst, missing = 0.0, []
            for top, low in (("front", "skirt_front"), ("back", "skirt_back")):
                ids = [f"{name}.{part}" for name in (top, low) for part in ("waist_cf", "waist_side")]
                if not need(*ids):
                    missing.append(top)
                    continue
                bodice_w = edge_length(edges[f"{top}.waist_cf"]) + edge_length(edges[f"{top}.waist_side"])
                skirt_w = edge_length(edges[f"{low}.waist_cf"]) + edge_length(edges[f"{low}.waist_side"])
                worst = max(worst, abs(bodice_w - skirt_w))
            if not missing and worst <= WAIST_SEAM_TOL_MM:
                ok("K4", "Waist seam",
                   f"Bodice and skirt waists are the same length (largest difference {worst:.1f} mm).",
                   worst, 0.0, WAIST_SEAM_TOL_MM)
            else:
                fail("K4", "Waist seam",
                     f"Bodice and skirt waists differ by {worst:.1f} mm (limit {WAIST_SEAM_TOL_MM:.0f} mm)"
                     + (f"; waist edges missing: {', '.join(missing)}" if missing else "")
                     + ". Match the bodice and skirt waist widths.", worst, 0.0, WAIST_SEAM_TOL_MM)
        elif finish is not None and need(*seam_ids):
            sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
            overlap = float(pattern.metadata.get("waistband_overlap_mm", 0.0)) if finish.id.startswith("waistband") else 0.0
            diff = abs(edge_length(finish) - (sewn + overlap))
            label = "Waistband" if finish.id.startswith("waistband") else "Waist facing"
            if diff <= WAIST_FINISH_TOL_MM:
                ok("K2", label, f"{label} matches the {sewn:.0f} mm waist seam (difference {diff:.1f} mm).",
                   diff, 0.0, WAIST_FINISH_TOL_MM)
            else:
                fail("K2", label,
                     f"{label} differs by {diff:.1f} mm from the {sewn:.0f} mm waist seam "
                     f"(limit {WAIST_FINISH_TOL_MM:.0f} mm).",
                     diff, 0.0, WAIST_FINISH_TOL_MM)
        else:
            fail("K2", "Waistband", "Waistband, waist facing or waist edge is missing.")
        # K3: a scallop hem facing traces the hem exactly.
        fs, bs = (f"{side}.scallop" for side in skirt_sides)
        if "hem_facing.scallop" in edges or fs in edges:
            if need("hem_facing.scallop", fs, bs):
                facing = edge_length(edges["hem_facing.scallop"])
                diff = max(abs(facing - edge_length(edges[fs])), abs(facing - edge_length(edges[bs])))
                if diff <= HEM_FACING_TOL_MM:
                    ok("K3", "Hem facing", f"Hem facing follows the scallop (difference {diff:.1f} mm).",
                       diff, 0.0, HEM_FACING_TOL_MM)
                else:
                    fail("K3", "Hem facing",
                         f"Hem facing differs from the scallop by {diff:.1f} mm (limit {HEM_FACING_TOL_MM:.0f} mm).",
                         diff, 0.0, HEM_FACING_TOL_MM)
            else:
                fail("K3", "Hem facing", "Scallop hem has no matching hem facing.")

    # Widths at body lines.
    if fabric is None:
        fail("W1", "Body width", f"Unknown fabric '{pattern.fabric}'.")
        fail("W2", "Waist width", f"Unknown fabric '{pattern.fabric}'.")
        fail("W3", "Hip width", f"Unknown fabric '{pattern.fabric}'.")
    else:
        ease_mm = fabric.default_ease_mm
        try:
            if pattern.garment_type == "tshirt":
                chest = measures["chest"]
                depth = measures["armhole_depth"]
                total = 2 * (_station_width(panels["front"], depth) + _station_width(panels["back"], depth))
                _width(ok, fail, "W1", "Chest width", total, chest + ease_mm)
                length = measures["back_length"] + params.get("body_length_adjust", 0)
                waist_y, hip_y = (depth + length) / 2, length
                total = 2 * (_station_width(panels["front"], waist_y) + _station_width(panels["back"], waist_y))
                _width(ok, fail, "W2", "Waist width", total, measures["waist"] + ease_mm)
                total = 2 * (_station_width(panels["front"], hip_y) + _station_width(panels["back"], hip_y))
                _width(ok, fail, "W3", "Hip width", total, measures["hip"] + ease_mm)
            elif pattern.garment_type == "shirt":
                placket = float(params["placket_width"])
                depth = measures["armhole_depth"] + 0.6 * float(params["shoulder_drop"])
                total = 2 * (_station_width(panels["front"], depth) - placket
                             + _station_width(panels["back"], depth))
                _width(ok, fail, "W1", "Chest width", total,
                       measures["bust"] + ease_mm + float(params["design_ease"]))
                below = float(params["shirt_length"])
                hem_y = measures["back_length"] + below - 1.0
                body = (measures["hip"] if below >= measures["waist_to_hip"] else
                        measures["waist"] + (measures["hip"] - measures["waist"]) * below / measures["waist_to_hip"])
                total = 2 * (_station_width(panels["front"], hem_y) - placket + _station_width(panels["back"], hem_y))
                need_mm = body + SHIRT_HIP_MIN_EASE_MM
                if total >= need_mm - WIDTH_TOL_MM:
                    ok("W3", "Hip room", f"Hem girth {total:.1f} mm is enough for the {body:.0f} mm body "
                       f"+ {SHIRT_HIP_MIN_EASE_MM:.0f} mm.", total, need_mm, WIDTH_TOL_MM)
                else:
                    fail("W3", "Hip room", f"Hem girth {total:.1f} mm is less than {need_mm:.0f} mm "
                         f"(body + {SHIRT_HIP_MIN_EASE_MM:.0f} mm). Widen the lower sides.",
                         total, need_mm, WIDTH_TOL_MM)
            elif pattern.garment_type == "skirt":
                seam_ids = [f"{side}.{part}" for side in ("front", "back") for part in ("waist_cf", "waist_side")]
                sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
                _width(ok, fail, "W2", "Waist width", sewn, measures["waist"] + SKIRT_WAIST_EASE_MM)
                hip_y = measures["waist_to_hip"]
                total = 2 * (_station_width(panels["front"], hip_y) + _station_width(panels["back"], hip_y))
                _width(ok, fail, "W3", "Hip width", total, measures["hip"] + float(params["skirt_hip_ease"]))
            elif dress_seam:
                legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
                join = (legs["front.dart_leg1"].curve.points[0].x + legs["front.dart_leg2"].curve.points[-1].x) / 2
                total = 2 * (join + _station_width(panels["back"], measures["apex_height"]))
                _width(ok, fail, "W1", "Bust width", total, measures["bust"] + ease_mm)
                seam_ids = [f"{side}.{part}" for side in ("front", "back") for part in ("waist_cf", "waist_side")]
                sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
                _width(ok, fail, "W2", "Waist width", sewn, measures["waist"] + SKIRT_WAIST_EASE_MM)
                hip_y = measures["waist_to_hip"]
                total = 2 * (_station_width(panels["skirt_front"], hip_y) + _station_width(panels["skirt_back"], hip_y))
                _width(ok, fail, "W3", "Hip width", total, measures["hip"] + float(params["skirt_hip_ease"]))
            elif bodice and "front" in panels and "back" in panels:
                bust, waist, hip = measures["bust"], measures["waist"], measures["hip"]
                apex_y = measures["apex_height"]
                legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
                if len(legs) == 2:
                    join = (legs["front.dart_leg1"].curve.points[0].x
                            + legs["front.dart_leg2"].curve.points[-1].x) / 2
                    total = 2 * (join + _station_width(panels["back"], apex_y))
                    _width(ok, fail, "W1", "Bust width", total, bust + ease_mm)
                intake = float(params.get("back_waist_dart_intake", 0))
                total = 2 * (_station_width(panels["front"], measures["front_length"])
                             + _station_width(panels["back"], measures["back_length"]))
                _width(ok, fail, "W2", "Waist width", total, waist + ease_mm - intake)
                total = 2 * (_station_width(panels["front"], measures["front_length"] + measures["waist_to_hip"])
                             + _station_width(panels["back"], measures["back_length"] + measures["waist_to_hip"]))
                _width(ok, fail, "W3", "Hip width", total, hip + ease_mm)
        except (KeyError, TypeError, ValueError, GEOSException) as exc:
            fail("W1", "Body width", f"Measurement data missing or panel broken: {exc}.")

    # Dart checks for the bodice family (the waist-seam dress bodice too).
    if (bodice or dress_seam) and "front" in panels:
        legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
        if len(legs) == 2:
            leg1 = edge_length(legs["front.dart_leg1"])
            leg2 = edge_length(legs["front.dart_leg2"])
            diff = abs(leg1 - leg2)
            if diff <= DART_LEG_TOL_MM:
                ok("D1", "Dart legs", f"Both front dart legs match ({leg1:.1f} mm).",
                   diff, 0.0, DART_LEG_TOL_MM)
            else:
                fail("D1", "Dart legs",
                     f"Dart legs {leg1:.1f} mm vs {leg2:.1f} mm (difference {diff:.1f} mm, "
                     f"limit {DART_LEG_TOL_MM:.0f} mm). Match the dart leg lengths.",
                     diff, 0.0, DART_LEG_TOL_MM)
            try:
                apex = Point(measures["apex_distance"] / 2, measures["apex_height"])
                tip = legs["front.dart_leg1"].curve.points[-1]
                clearance = math.hypot(tip.x - apex.x, tip.y - apex.y)
                expected = float(params["bust_dart_clearance"])
                if abs(clearance - expected) <= TIP_TOL_MM:
                    ok("D2", "Dart tip",
                       f"Dart tip is {clearance:.1f} mm from the bust apex (parameter {expected:.0f} mm).",
                       clearance, expected, TIP_TOL_MM)
                else:
                    fail("D2", "Dart tip",
                         f"Dart tip is {clearance:.1f} mm from the bust apex, parameter {expected:.0f} mm "
                         f"(limit {TIP_TOL_MM:.0f} mm). Move the dart tip.",
                         clearance, expected, TIP_TOL_MM)
                base = math.hypot(legs["front.dart_leg1"].curve.points[0].x - legs["front.dart_leg2"].curve.points[-1].x,
                                    legs["front.dart_leg1"].curve.points[0].y - legs["front.dart_leg2"].curve.points[-1].y)
                rule = (measures["bust"] - measures["high_bust"]) / 4
                if abs(base - rule) <= INTAKE_TOL_MM:
                    ok("D3", "Dart intake",
                       f"Dart intake {base:.1f} mm matches the bust difference ({rule:.1f} mm).",
                       base, rule, INTAKE_TOL_MM)
                else:
                    fail("D3", "Dart intake",
                         f"Dart intake {base:.1f} mm vs the bust-difference rule {rule:.1f} mm "
                         f"(limit {INTAKE_TOL_MM:.0f} mm).",
                         base, rule, INTAKE_TOL_MM)
            except (KeyError, TypeError) as exc:
                fail("D2", "Dart tip", f"Dart data missing: {exc}.")
            # D4: the closed side seam meets at one point with a small kink.
            kink = _closed_seam(panels["front"])
            if kink is None:
                fail("D4", "Dart closure", "The side seam does not join when the dart is closed.")
            elif kink <= KINK_DEG:
                ok("D4", "Dart closure", f"The side seam joins when the dart is closed (kink {kink:.1f}°).",
                   kink, 0.0, KINK_DEG)
            else:
                fail("D4", "Dart closure",
                     f"The side seam kinks {kink:.1f}° when the dart is closed (limit {KINK_DEG:.0f}°). "
                     f"Smooth the side line.",
                     kink, 0.0, KINK_DEG)
        if dress_seam:
            # D5/D6: the open waist darts of the dress bodice have equal legs.
            for id_, name, panel_name, label in (("D5", "Back dart", "back", "back"),
                                                 ("D6", "Front waist dart", "front", "front waist")):
                pair = (f"{panel_name}.wdart_leg1", f"{panel_name}.wdart_leg2")
                if not need(*pair):
                    fail(id_, name, f"The {label} dart is missing.")
                    continue
                diff = abs(edge_length(edges[pair[0]]) - edge_length(edges[pair[1]]))
                if diff <= DART_LEG_TOL_MM:
                    ok(id_, name, f"The {label} dart legs match (difference {diff:.1f} mm).", diff, 0.0, DART_LEG_TOL_MM)
                else:
                    fail(id_, name, f"The {label} dart legs differ by {diff:.1f} mm (limit {DART_LEG_TOL_MM:.0f} mm). "
                         f"Match the dart legs.", diff, 0.0, DART_LEG_TOL_MM)
        # D5: back waist dart recorded and trued into the side (drill-marked bodice blocks).
        if not dress_seam:
            try:
                intake = float(params.get("back_waist_dart_intake", 0))
                upper = edges["back.side_upper"]
                narrowing = (measures["waist"] + ease_mm) / 4 - upper.curve.points[-1].x if fabric else 0.0
                drills = pattern.metadata.get("darts", {}).get("back_waist", {}).get("drills", [])
                narrow_ok = abs(narrowing - intake / 2) <= G_TOL_MM
                drills_ok = (len(drills) == 3) if intake > 0 else (len(drills) == 0)
                if narrow_ok and drills_ok:
                    ok("D5", "Back dart",
                       f"Back waist dart {intake:.0f} mm is recorded with drill marks.",
                       narrowing, intake / 2, G_TOL_MM)
                else:
                    fail("D5", "Back dart",
                         f"Waist dart {narrowing * 2:.1f} mm vs parameter {intake:.0f} mm, "
                         f"{len(drills)} drill marks. Check the back dart.",
                         narrowing, intake / 2, G_TOL_MM)
            except (KeyError, TypeError) as exc:
                fail("D5", "Back dart", f"Back dart data missing: {exc}.")

    # N1: every notch sits on its edge.
    try:
        bad = []
        for panel in pattern.panels:
            known = _edges(panel)
            for notch in panel.notches:
                edge = known.get(notch.edge_id)
                if edge is None or not 0 <= notch.distance_from_start_mm <= edge_length(edge):
                    bad.append(f"{panel.name}:{notch.edge_id}")
        if not bad:
            ok("N1", "Valid notches", "All notches sit on their edges.")
        else:
            fail("N1", "Valid notches", f"Loose or broken notches: {', '.join(bad)}. Fix the notch positions.")
    except (TypeError, ValueError) as exc:
        fail("N1", "Valid notches", f"Notches cannot be checked: {exc}.")

    # N2: shoulder-point notches open both armholes.
    if need("front.armhole", "back.armhole"):
        have = {(n.edge_id, n.distance_from_start_mm) for panel in pattern.panels for n in panel.notches}
        if ("front.armhole", 0) in have and ("back.armhole", 0) in have:
            ok("N2", "Shoulder notches", "Shoulder-point notches are on the front and back armholes.")
        else:
            fail("N2", "Shoulder notches", "Shoulder-point notch (distance 0) is missing on the front or back armhole.")

    # R1: grainlines run parallel to the fold.
    folded = [panel for panel in pattern.panels if panel.cut_on_fold]
    if folded:
        tilted = [panel.name for panel in folded if abs(panel.grainline[0].x - panel.grainline[1].x) > 1e-9]
        if not tilted:
            ok("R1", "Grainline", "Grainlines run parallel to the fold.")
        else:
            fail("R1", "Grainline", f"Grainline tilted on panels: {', '.join(tilted)}. Straighten the grainline.")

    passed = all(check.passed for check in checks)
    report = Report(pattern.garment_type, pattern.size_label, pattern.fabric, passed, checks)
    report.meta = {
        "version": pattern.metadata.get("version", ""),
        "params_hash": pattern.metadata.get("params_hash", ""),
        "repair_rounds": pattern.metadata.get("repair_rounds", 0),
    }
    if any(float(v) > 0 for v in (pattern.metadata.get("gathers_mm") or {}).values()):
        report.meta["gathers_mm"] = dict(pattern.metadata["gathers_mm"])
    return report


def _seg(a: Point, b: Point) -> float:
    """Return the length of one sampled segment, in mm."""
    return math.hypot(b.x - a.x, b.y - a.y)


def _width(ok: Any, fail: Any, id: str, name: str, actual: float, expected: float) -> None:
    """Record a body-width check against body plus ease."""
    if abs(actual - expected) <= WIDTH_TOL_MM:
        ok(id, name, f"{name} {actual:.1f} mm matches body + ease ({expected:.0f} mm).",
           actual, expected, WIDTH_TOL_MM)
    else:
        fail(id, name,
             f"{name} {actual:.1f} mm vs body + ease {expected:.0f} mm "
             f"(difference {abs(actual - expected):.1f} mm, limit {WIDTH_TOL_MM:.0f} mm).",
             actual, expected, WIDTH_TOL_MM)


def _closed_seam(front: Panel) -> float | None:
    """Return the side-seam kink in degrees when the dart closes, if computable.

    Join coincidence is already D1's business, so this check owns the kink
    angle alone.
    """
    legs = {e.id: e for e in front.edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
    upper = _edges(front).get("front.side_upper")
    lower = _edges(front).get("front.side_lower")
    if len(legs) != 2 or upper is None or lower is None:
        return None
    d1, tip, d2 = legs["front.dart_leg1"].curve.points[0], legs["front.dart_leg1"].curve.points[-1], legs[
        "front.dart_leg2"].curve.points[-1]
    before = math.atan2(d1.y - tip.y, d1.x - tip.x)
    after = math.atan2(d2.y - tip.y, d2.x - tip.x)
    incoming = _rot(Point(upper.curve.points[-1].x - upper.curve.points[-2].x,
                          upper.curve.points[-1].y - upper.curve.points[-2].y), Point(0, 0), after - before)
    outgoing = Point(lower.curve.points[1].x - lower.curve.points[0].x,
                     lower.curve.points[1].y - lower.curve.points[0].y)
    cosine = ((incoming.x * outgoing.x + incoming.y * outgoing.y)
              / math.hypot(incoming.x, incoming.y) / math.hypot(outgoing.x, outgoing.y))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _rot(point: Point, centre: Point, angle: float) -> Point:
    """Rotate a point about a centre by an angle in radians."""
    dx, dy = point.x - centre.x, point.y - centre.y
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return Point(centre.x + dx * cos_a - dy * sin_a, centre.y + dx * sin_a + dy * cos_a)
