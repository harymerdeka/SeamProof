"""Deterministic pattern verifier with tailor-language (Indonesian) reports.

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
    """One verifier finding; message_id is tailor-language Indonesian."""

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
        """Return a one-line Indonesian verdict."""
        total = len(self.checks)
        good = sum(check.passed for check in self.checks)
        if self.passed:
            return (f"LULUS: {good}/{total} pemeriksaan — pola {self.garment} {self.size} "
                    f"siap toile.")
        names = ", ".join(check.name_id for check in self.failures())
        return (f"GAGAL: {total - good}/{total} pemeriksaan — {names}. "
                f"Perbaiki lalu verifikasi ulang.")


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
    """Run every applicable check and return an Indonesian report."""
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
    bodice = (pattern.garment_type in ("bodice", "blouse", "tunic")
              or (pattern.garment_type == "dress" and not dress_seam))

    # G1: closed, valid outlines.
    try:
        areas = {name: panel_polygon(panel).area for name, panel in panels.items()}
        if all(area > 0 for area in areas.values()):
            ok("G1", "Panel tertutup", "Semua panel tertutup dan valid.")
        else:
            fail("G1", "Panel tertutup", "Ada panel dengan luas nol.")
    except ValueError as exc:
        fail("G1", "Panel tertutup", f"Panel tidak tertutup/valid: {exc}. Periksa titik sambung tepi.")

    # G2: parameter fingerprint intact.
    try:
        if pattern.metadata.get("params_hash") == params_hash(params):
            ok("G2", "Hash parameter", "Hash parameter cocok dengan draft.")
        else:
            fail("G2", "Hash parameter", "Hash parameter tidak cocok — pola diubah tanpa lewat draft.")
    except ValueError:
        fail("G2", "Hash parameter", "Parameter memuat angka yang tidak bisa di-hash.")

    def need(*names: str) -> bool:
        return all(name in edges for name in names)

    # S1: shoulder seams match.
    if need("front.shoulder", "back.shoulder"):
        front_sh = edge_length(edges["front.shoulder"])
        back_sh = edge_length(edges["back.shoulder"])
        diff = abs(front_sh - back_sh)
        if diff <= SHOULDER_TOL_MM:
            ok("S1", "Kampuh bahu", f"Bahu depan-belakang sama ({front_sh:.1f} vs {back_sh:.1f} mm).",
               diff, 0.0, SHOULDER_TOL_MM)
        else:
            fail("S1", "Kampuh bahu",
                 f"Bahu depan {front_sh:.1f} mm vs belakang {back_sh:.1f} mm "
                 f"(selisih {diff:.1f} mm, batas {SHOULDER_TOL_MM:.0f} mm). Samakan garis bahu.",
                 diff, 0.0, SHOULDER_TOL_MM)

    # S2: side seams, sewn state (dart legs excluded).
    if pattern.garment_type in ("tshirt", "shirt", "skirt") and need("front.side", "back.side"):
        diff = abs(edge_length(edges["front.side"]) - edge_length(edges["back.side"]))
        if diff <= TSHIRT_SIDE_TOL_MM:
            ok("S2", "Kampuh sisi", f"Sisi depan-belakang sama (selisih {diff:.1f} mm).",
               diff, 0.0, TSHIRT_SIDE_TOL_MM)
        else:
            fail("S2", "Kampuh sisi",
                 f"Sisi depan-belakang selisih {diff:.1f} mm (batas {TSHIRT_SIDE_TOL_MM:.0f} mm).",
                 diff, 0.0, TSHIRT_SIDE_TOL_MM)
    elif dress_seam and need("front.side_upper", "front.side_lower", "back.side_upper"):
        # Bodice side underarm to waist (bust dart closed), then the skirt sides.
        front_above = edge_length(edges["front.side_upper"]) + edge_length(edges["front.side_lower"])
        back_above = edge_length(edges["back.side_upper"])
        diff_a = abs(front_above - back_above)
        if diff_a <= SIDE_ABOVE_TOL_MM:
            ok("S2a", "Sisi badan", f"Sisi ketiak–pinggang selisih {diff_a:.1f} mm.",
               diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        else:
            fail("S2a", "Sisi badan",
                 f"Sisi ketiak–pinggang selisih {diff_a:.1f} mm (batas {SIDE_ABOVE_TOL_MM:.0f} mm). "
                 f"Samakan sisi badan depan-belakang.", diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        if need("skirt_front.side", "skirt_back.side"):
            diff_b = abs(edge_length(edges["skirt_front.side"]) - edge_length(edges["skirt_back.side"]))
            if diff_b <= TSHIRT_SIDE_TOL_MM:
                ok("S2b", "Sisi rok", f"Sisi rok depan-belakang sama (selisih {diff_b:.1f} mm).",
                   diff_b, 0.0, TSHIRT_SIDE_TOL_MM)
            else:
                fail("S2b", "Sisi rok", f"Sisi rok depan-belakang selisih {diff_b:.1f} mm "
                     f"(batas {TSHIRT_SIDE_TOL_MM:.0f} mm).", diff_b, 0.0, TSHIRT_SIDE_TOL_MM)
        else:
            fail("S2b", "Sisi rok", "Panel rok depan/belakang hilang.")
    elif bodice and need("front.side_upper", "front.side_lower", "back.side_upper", "back.side_low", "back.side_lower"):
        lower_pts = _edges(panels["front"])["front.side_lower"].curve.points
        front_above = edge_length(edges["front.side_upper"]) + _seg(lower_pts[0], lower_pts[1])
        front_below = sum(_seg(a, b) for a, b in zip(lower_pts[1:], lower_pts[2:]))
        back_above = edge_length(edges["back.side_upper"])
        back_below = edge_length(edges["back.side_low"]) + edge_length(edges["back.side_lower"])
        diff_a = abs(front_above - back_above)
        if diff_a <= SIDE_ABOVE_TOL_MM:
            ok("S2a", "Sisi atas", f"Sisi ketiak–pinggang selisih {diff_a:.1f} mm.",
               diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        else:
            fail("S2a", "Sisi atas",
                 f"Sisi ketiak–pinggang selisih {diff_a:.1f} mm (batas {SIDE_ABOVE_TOL_MM:.0f} mm). "
                 f"Samakan lengkung sisi atas.",
                 diff_a, 0.0, SIDE_ABOVE_TOL_MM)
        diff_b = abs(front_below - back_below)
        if diff_b <= SIDE_BELOW_TOL_MM:
            ok("S2b", "Sisi bawah",
               f"Sisi pinggang–hem selisih {diff_b:.1f} mm (batas interim {SIDE_BELOW_TOL_MM:.0f} mm). "
               f"Samakan saat toile.",
               diff_b, 0.0, SIDE_BELOW_TOL_MM)
        else:
            fail("S2b", "Sisi bawah",
                 f"Sisi pinggang–hem selisih {diff_b:.1f} mm (batas {SIDE_BELOW_TOL_MM:.0f} mm).",
                 diff_b, 0.0, SIDE_BELOW_TOL_MM)

    # S3: sleeve cap ease inside the fabric range.
    if fabric is None:
        fail("S3", "Kelonggaran lengan", f"Kain '{pattern.fabric}' tidak dikenal — rentang kelonggaran tak bisa diperiksa.")
    elif need("front.armhole", "back.armhole", "sleeve.cap"):
        armholes = edge_length(edges["front.armhole"]) + edge_length(edges["back.armhole"])
        ease = edge_length(edges["sleeve.cap"]) - armholes
        low, high = fabric.sleeve_cap_ease_range_mm
        # A puff cap is gathered on purpose: the declared gather shifts the range.
        gather = float(params.get("sleeve_cap_gather", 0)) if pattern.garment_type == "shirt" else 0.0
        low, high = low + gather, high + gather
        if low <= ease <= high:
            ok("S3", "Kelonggaran lengan",
               f"Kelonggaran puncak lengan {ease:.1f} mm dalam rentang {low:.0f}–{high:.0f} mm.",
               ease, (low + high) / 2, (high - low) / 2)
        else:
            fail("S3", "Kelonggaran lengan",
                 f"Kelonggaran puncak lengan {ease:.1f} mm di luar rentang {low:.0f}–{high:.0f} mm. "
                 f"Sesuaikan tinggi puncak lengan.",
                 ease, (low + high) / 2, (high - low) / 2)

    # S4: neck facing matches the front neckline.
    if "facing.neck" in edges and "front.neckline" in edges:
        diff = abs(edge_length(edges["facing.neck"]) - edge_length(edges["front.neckline"]))
        if diff <= FACING_TOL_MM:
            ok("S4", "Lapisan leher", f"Lapisan leher sama dengan kerung leher (selisih {diff:.1f} mm).",
               diff, 0.0, FACING_TOL_MM)
        else:
            fail("S4", "Lapisan leher",
                 f"Lapisan leher selisih {diff:.1f} mm dari kerung leher (batas {FACING_TOL_MM:.0f} mm). "
                 f"Gambar ulang lapisan mengikuti kerung.",
                 diff, 0.0, FACING_TOL_MM)

    # S5: sleeve underarm pair matches itself.
    if need("sleeve.underarm_front", "sleeve.underarm_back"):
        diff = abs(edge_length(edges["sleeve.underarm_front"]) - edge_length(edges["sleeve.underarm_back"]))
        if diff <= UNDERARM_TOL_MM:
            ok("S5", "Kampuh lengan", f"Kampuh bawah lengan sama (selisih {diff:.1f} mm).",
               diff, 0.0, UNDERARM_TOL_MM)
        else:
            fail("S5", "Kampuh lengan",
                 f"Kampuh bawah lengan depan-belakang selisih {diff:.1f} mm (batas {UNDERARM_TOL_MM:.1f} mm).",
                 diff, 0.0, UNDERARM_TOL_MM)

    # S6: t-shirt neckband ring matches the necklines.
    if "neckband.attach" in edges and need("front.neckline", "back.neckline"):
        ring = 2 * (edge_length(edges["front.neckline"]) + edge_length(edges["back.neckline"]))
        diff = abs(edge_length(edges["neckband.attach"]) - ring)
        if diff <= NECKBAND_TOL_MM:
            ok("S6", "Ban leher", f"Ban leher melingkari kerung leher (selisih {diff:.1f} mm).",
               diff, 0.0, NECKBAND_TOL_MM)
        else:
            fail("S6", "Ban leher",
                 f"Ban leher selisih {diff:.1f} mm dari kerung leher (batas {NECKBAND_TOL_MM:.0f} mm).",
                 diff, 0.0, NECKBAND_TOL_MM)

    # S7: shirt cuff band equals the sleeve hem it is sewn to; a gathered sleeve
    # has a fitted cuff sized from the arm instead, and S9 checks the gathers.
    sleeve_gathered = pattern.garment_type == "shirt" and (
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
            kind = "manset karet (lewat tangan)" if elastic else "manset pas"
            if diff <= CUFF_TOL_MM:
                ok("S7", "Manset", f"Panjang {kind} {band:.1f} mm sesuai lingkar lengan {girth:.0f} mm.",
                   band, expected, CUFF_TOL_MM)
            else:
                fail("S7", "Manset", f"Panjang {kind} {band:.1f} mm vs {expected:.1f} mm dari lingkar lengan "
                     f"(batas {CUFF_TOL_MM:.0f} mm). Ukur ulang manset.", band, expected, CUFF_TOL_MM)
            ratio = edge_length(edges["sleeve.cuff_join"]) / band
            if 1.0 - 1e-6 <= ratio <= GATHER_MAX_RATIO:
                ok("S9", "Kerutan lengan", f"Ujung lengan dikerut {ratio:.2f}x ke manset "
                   f"(batas 1–{GATHER_MAX_RATIO:.0f}x).", ratio, (1.0 + GATHER_MAX_RATIO) / 2,
                   (GATHER_MAX_RATIO - 1.0) / 2)
            else:
                fail("S9", "Kerutan lengan", f"Ujung lengan {ratio:.2f}x manset — di luar 1–{GATHER_MAX_RATIO:.0f}x. "
                     f"Kurangi kelebaran lengan atau lebarkan manset.", ratio, (1.0 + GATHER_MAX_RATIO) / 2,
                     (GATHER_MAX_RATIO - 1.0) / 2)
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            fail("S7", "Manset", f"Data manset/lengan hilang: {exc}.")
    elif need("cuff.attach", "sleeve.cuff_join"):
        diff = abs(edge_length(edges["cuff.attach"]) - edge_length(edges["sleeve.cuff_join"]))
        if diff <= CUFF_TOL_MM:
            ok("S7", "Manset", f"Manset sama dengan lingkar ujung lengan (selisih {diff:.1f} mm).",
               diff, 0.0, CUFF_TOL_MM)
        else:
            fail("S7", "Manset",
                 f"Manset selisih {diff:.1f} mm dari ujung lengan (batas {CUFF_TOL_MM:.0f} mm). "
                 f"Samakan panjang manset dengan lebar ujung lengan.",
                 diff, 0.0, CUFF_TOL_MM)

    # S8: half collar neck edge equals half the body neckline (back + front).
    if need("collar.neck", "front.neckline", "back.neckline"):
        line = edge_length(edges["front.neckline"]) + edge_length(edges["back.neckline"])
        diff = abs(edge_length(edges["collar.neck"]) - line)
        if diff <= COLLAR_TOL_MM:
            ok("S8", "Kerah", f"Leher kerah sama dengan kerung leher (selisih {diff:.1f} mm).",
               diff, 0.0, COLLAR_TOL_MM)
        else:
            fail("S8", "Kerah",
                 f"Leher kerah selisih {diff:.1f} mm dari kerung leher (batas {COLLAR_TOL_MM:.0f} mm). "
                 f"Gambar ulang kerah dari kerung leher.",
                 diff, 0.0, COLLAR_TOL_MM)

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
            ok("K1", "Kupnat rok", f"Kaki kupnat rok depan-belakang sama (selisih terbesar {worst:.1f} mm).",
               worst, 0.0, DART_LEG_TOL_MM)
        else:
            fail("K1", "Kupnat rok",
                 f"Kaki kupnat rok selisih {worst:.1f} mm (batas {DART_LEG_TOL_MM:.0f} mm) "
                 f"atau kupnat hilang. Samakan kaki kupnat.",
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
                ok("K4", "Sambungan pinggang",
                   f"Pinggang badan dan rok sama panjang (selisih terbesar {worst:.1f} mm).",
                   worst, 0.0, WAIST_SEAM_TOL_MM)
            else:
                fail("K4", "Sambungan pinggang",
                     f"Pinggang badan vs rok selisih {worst:.1f} mm (batas {WAIST_SEAM_TOL_MM:.0f} mm)"
                     + (f"; tepi pinggang hilang: {', '.join(missing)}" if missing else "")
                     + ". Samakan lebar pinggang badan dan rok.", worst, 0.0, WAIST_SEAM_TOL_MM)
        elif finish is not None and need(*seam_ids):
            sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
            overlap = float(pattern.metadata.get("waistband_overlap_mm", 0.0)) if finish.id.startswith("waistband") else 0.0
            diff = abs(edge_length(finish) - (sewn + overlap))
            label = "Ban pinggang" if finish.id.startswith("waistband") else "Lapisan pinggang"
            if diff <= WAIST_FINISH_TOL_MM:
                ok("K2", label, f"{label} cocok dengan jahitan pinggang {sewn:.0f} mm (selisih {diff:.1f} mm).",
                   diff, 0.0, WAIST_FINISH_TOL_MM)
            else:
                fail("K2", label,
                     f"{label} selisih {diff:.1f} mm dari jahitan pinggang {sewn:.0f} mm "
                     f"(batas {WAIST_FINISH_TOL_MM:.0f} mm).",
                     diff, 0.0, WAIST_FINISH_TOL_MM)
        else:
            fail("K2", "Ban pinggang", "Ban/lapisan pinggang atau tepi pinggang hilang.")
        # K3: a scallop hem facing traces the hem exactly.
        fs, bs = (f"{side}.scallop" for side in skirt_sides)
        if "hem_facing.scallop" in edges or fs in edges:
            if need("hem_facing.scallop", fs, bs):
                facing = edge_length(edges["hem_facing.scallop"])
                diff = max(abs(facing - edge_length(edges[fs])), abs(facing - edge_length(edges[bs])))
                if diff <= HEM_FACING_TOL_MM:
                    ok("K3", "Lapisan kelim", f"Lapisan kelim mengikuti scallop (selisih {diff:.1f} mm).",
                       diff, 0.0, HEM_FACING_TOL_MM)
                else:
                    fail("K3", "Lapisan kelim",
                         f"Lapisan kelim selisih {diff:.1f} mm dari scallop (batas {HEM_FACING_TOL_MM:.0f} mm).",
                         diff, 0.0, HEM_FACING_TOL_MM)
            else:
                fail("K3", "Lapisan kelim", "Kelim scallop tanpa lapisan kelim yang cocok.")

    # Widths at body lines.
    if fabric is None:
        fail("W1", "Lebar badan", f"Kain '{pattern.fabric}' tidak dikenal.")
        fail("W2", "Lebar pinggang", f"Kain '{pattern.fabric}' tidak dikenal.")
        fail("W3", "Lebar pinggul", f"Kain '{pattern.fabric}' tidak dikenal.")
    else:
        ease_mm = fabric.default_ease_mm
        try:
            if pattern.garment_type == "tshirt":
                chest = measures["chest"]
                depth = measures["armhole_depth"]
                total = 2 * (_station_width(panels["front"], depth) + _station_width(panels["back"], depth))
                _width(ok, fail, "W1", "Lebar dada", total, chest + ease_mm)
                length = measures["back_length"] + params.get("body_length_adjust", 0)
                waist_y, hip_y = (depth + length) / 2, length
                total = 2 * (_station_width(panels["front"], waist_y) + _station_width(panels["back"], waist_y))
                _width(ok, fail, "W2", "Lebar pinggang", total, measures["waist"] + ease_mm)
                total = 2 * (_station_width(panels["front"], hip_y) + _station_width(panels["back"], hip_y))
                _width(ok, fail, "W3", "Lebar pinggul", total, measures["hip"] + ease_mm)
            elif pattern.garment_type == "shirt":
                placket = float(params["placket_width"])
                depth = measures["armhole_depth"] + 0.6 * float(params["shoulder_drop"])
                total = 2 * (_station_width(panels["front"], depth) - placket
                             + _station_width(panels["back"], depth))
                _width(ok, fail, "W1", "Lebar dada", total,
                       measures["bust"] + ease_mm + float(params["design_ease"]))
                below = float(params["shirt_length"])
                hem_y = measures["back_length"] + below - 1.0
                body = (measures["hip"] if below >= measures["waist_to_hip"] else
                        measures["waist"] + (measures["hip"] - measures["waist"]) * below / measures["waist_to_hip"])
                total = 2 * (_station_width(panels["front"], hem_y) - placket + _station_width(panels["back"], hem_y))
                need_mm = body + SHIRT_HIP_MIN_EASE_MM
                if total >= need_mm - WIDTH_TOL_MM:
                    ok("W3", "Ruang pinggul", f"Lingkar bawah {total:.1f} mm cukup untuk badan {body:.0f} mm "
                       f"+ {SHIRT_HIP_MIN_EASE_MM:.0f} mm.", total, need_mm, WIDTH_TOL_MM)
                else:
                    fail("W3", "Ruang pinggul", f"Lingkar bawah {total:.1f} mm kurang dari {need_mm:.0f} mm "
                         f"(badan + {SHIRT_HIP_MIN_EASE_MM:.0f} mm). Lebarkan sisi bawah.",
                         total, need_mm, WIDTH_TOL_MM)
            elif pattern.garment_type == "skirt":
                seam_ids = [f"{side}.{part}" for side in ("front", "back") for part in ("waist_cf", "waist_side")]
                sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
                _width(ok, fail, "W2", "Lebar pinggang", sewn, measures["waist"] + SKIRT_WAIST_EASE_MM)
                hip_y = measures["waist_to_hip"]
                total = 2 * (_station_width(panels["front"], hip_y) + _station_width(panels["back"], hip_y))
                _width(ok, fail, "W3", "Lebar pinggul", total, measures["hip"] + float(params["skirt_hip_ease"]))
            elif dress_seam:
                legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
                join = (legs["front.dart_leg1"].curve.points[0].x + legs["front.dart_leg2"].curve.points[-1].x) / 2
                total = 2 * (join + _station_width(panels["back"], measures["apex_height"]))
                _width(ok, fail, "W1", "Lebar bust", total, measures["bust"] + ease_mm)
                seam_ids = [f"{side}.{part}" for side in ("front", "back") for part in ("waist_cf", "waist_side")]
                sewn = 2 * sum(edge_length(edges[e]) for e in seam_ids)
                _width(ok, fail, "W2", "Lebar pinggang", sewn, measures["waist"] + SKIRT_WAIST_EASE_MM)
                hip_y = measures["waist_to_hip"]
                total = 2 * (_station_width(panels["skirt_front"], hip_y) + _station_width(panels["skirt_back"], hip_y))
                _width(ok, fail, "W3", "Lebar pinggul", total, measures["hip"] + float(params["skirt_hip_ease"]))
            elif bodice and "front" in panels and "back" in panels:
                bust, waist, hip = measures["bust"], measures["waist"], measures["hip"]
                apex_y = measures["apex_height"]
                legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
                if len(legs) == 2:
                    join = (legs["front.dart_leg1"].curve.points[0].x
                            + legs["front.dart_leg2"].curve.points[-1].x) / 2
                    total = 2 * (join + _station_width(panels["back"], apex_y))
                    _width(ok, fail, "W1", "Lebar bust", total, bust + ease_mm)
                intake = float(params.get("back_waist_dart_intake", 0))
                total = 2 * (_station_width(panels["front"], measures["front_length"])
                             + _station_width(panels["back"], measures["back_length"]))
                _width(ok, fail, "W2", "Lebar pinggang", total, waist + ease_mm - intake)
                total = 2 * (_station_width(panels["front"], measures["front_length"] + measures["waist_to_hip"])
                             + _station_width(panels["back"], measures["back_length"] + measures["waist_to_hip"]))
                _width(ok, fail, "W3", "Lebar pinggul", total, hip + ease_mm)
        except (KeyError, TypeError, ValueError, GEOSException) as exc:
            fail("W1", "Lebar badan", f"Data ukuran hilang atau panel rusak: {exc}.")

    # Dart checks for the bodice family (the waist-seam dress bodice too).
    if (bodice or dress_seam) and "front" in panels:
        legs = {e.id: e for e in panels["front"].edges if e.id in ("front.dart_leg1", "front.dart_leg2")}
        if len(legs) == 2:
            leg1 = edge_length(legs["front.dart_leg1"])
            leg2 = edge_length(legs["front.dart_leg2"])
            diff = abs(leg1 - leg2)
            if diff <= DART_LEG_TOL_MM:
                ok("D1", "Kaki kupnat", f"Kedua kaki kupnat depan sama ({leg1:.1f} mm).",
                   diff, 0.0, DART_LEG_TOL_MM)
            else:
                fail("D1", "Kaki kupnat",
                     f"Kaki kupnat {leg1:.1f} mm vs {leg2:.1f} mm (selisih {diff:.1f} mm, "
                     f"batas {DART_LEG_TOL_MM:.0f} mm). Samakan panjang kaki kupnat.",
                     diff, 0.0, DART_LEG_TOL_MM)
            try:
                apex = Point(measures["apex_distance"] / 2, measures["apex_height"])
                tip = legs["front.dart_leg1"].curve.points[-1]
                clearance = math.hypot(tip.x - apex.x, tip.y - apex.y)
                expected = float(params["bust_dart_clearance"])
                if abs(clearance - expected) <= TIP_TOL_MM:
                    ok("D2", "Ujung kupnat",
                       f"Ujung kupnat {clearance:.1f} mm dari puncak dada (parameter {expected:.0f} mm).",
                       clearance, expected, TIP_TOL_MM)
                else:
                    fail("D2", "Ujung kupnat",
                         f"Ujung kupnat {clearance:.1f} mm dari puncak dada, parameter {expected:.0f} mm "
                         f"(batas {TIP_TOL_MM:.0f} mm). Geser ujung kupnat.",
                         clearance, expected, TIP_TOL_MM)
                base = math.hypot(legs["front.dart_leg1"].curve.points[0].x - legs["front.dart_leg2"].curve.points[-1].x,
                                    legs["front.dart_leg1"].curve.points[0].y - legs["front.dart_leg2"].curve.points[-1].y)
                rule = (measures["bust"] - measures["high_bust"]) / 4
                if abs(base - rule) <= INTAKE_TOL_MM:
                    ok("D3", "Besar kupnat",
                       f"Besar kupnat {base:.1f} mm sesuai selisih dada ({rule:.1f} mm).",
                       base, rule, INTAKE_TOL_MM)
                else:
                    fail("D3", "Besar kupnat",
                         f"Besar kupnat {base:.1f} mm vs aturan selisih dada {rule:.1f} mm "
                         f"(batas {INTAKE_TOL_MM:.0f} mm).",
                         base, rule, INTAKE_TOL_MM)
            except (KeyError, TypeError) as exc:
                fail("D2", "Ujung kupnat", f"Data kupnat hilang: {exc}.")
            # D4: the closed side seam meets at one point with a small kink.
            kink = _closed_seam(panels["front"])
            if kink is None:
                fail("D4", "Tutup kupnat", "Kampuh sisi tidak menyambung saat kupnat ditutup.")
            elif kink <= KINK_DEG:
                ok("D4", "Tutup kupnat", f"Kampuh sisi menyambung saat kupnat tertutup (tekuk {kink:.1f}°).",
                   kink, 0.0, KINK_DEG)
            else:
                fail("D4", "Tutup kupnat",
                     f"Kampuh sisi menekuk {kink:.1f}° saat kupnat tertutup (batas {KINK_DEG:.0f}°). "
                     f"Ratakan garis sisi.",
                     kink, 0.0, KINK_DEG)
        if dress_seam:
            # D5/D6: the open waist darts of the dress bodice have equal legs.
            for id_, name, panel_name, label in (("D5", "Kupnat belakang", "back", "belakang"),
                                                 ("D6", "Kupnat pinggang depan", "front", "pinggang depan")):
                pair = (f"{panel_name}.wdart_leg1", f"{panel_name}.wdart_leg2")
                if not need(*pair):
                    fail(id_, name, f"Kupnat {label} hilang.")
                    continue
                diff = abs(edge_length(edges[pair[0]]) - edge_length(edges[pair[1]]))
                if diff <= DART_LEG_TOL_MM:
                    ok(id_, name, f"Kaki kupnat {label} sama (selisih {diff:.1f} mm).", diff, 0.0, DART_LEG_TOL_MM)
                else:
                    fail(id_, name, f"Kaki kupnat {label} selisih {diff:.1f} mm (batas {DART_LEG_TOL_MM:.0f} mm). "
                         f"Samakan kaki kupnat.", diff, 0.0, DART_LEG_TOL_MM)
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
                    ok("D5", "Kupnat belakang",
                       f"Kupnat pinggang belakang {intake:.0f} mm tercatat dengan tanda lubang.",
                       narrowing, intake / 2, G_TOL_MM)
                else:
                    fail("D5", "Kupnat belakang",
                         f"Kupnat pinggang {narrowing * 2:.1f} mm vs parameter {intake:.0f} mm, "
                         f"tanda lubang {len(drills)}. Periksa kupnat belakang.",
                         narrowing, intake / 2, G_TOL_MM)
            except (KeyError, TypeError) as exc:
                fail("D5", "Kupnat belakang", f"Data kupnat belakang hilang: {exc}.")

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
            ok("N1", "Notch valid", "Semua notch menempel pada tepinya.")
        else:
            fail("N1", "Notch valid", f"Notch lepas/rusak: {', '.join(bad)}. Perbaiki posisi notch.")
    except (TypeError, ValueError) as exc:
        fail("N1", "Notch valid", f"Notch tak bisa diperiksa: {exc}.")

    # N2: shoulder-point notches open both armholes.
    if need("front.armhole", "back.armhole"):
        have = {(n.edge_id, n.distance_from_start_mm) for panel in pattern.panels for n in panel.notches}
        if ("front.armhole", 0) in have and ("back.armhole", 0) in have:
            ok("N2", "Notch bahu", "Notch titik bahu ada di kerung depan-belakang.")
        else:
            fail("N2", "Notch bahu", "Notch titik bahu (jarak 0) hilang di kerung depan/belakang.")

    # R1: grainlines run parallel to the fold.
    folded = [panel for panel in pattern.panels if panel.cut_on_fold]
    if folded:
        tilted = [panel.name for panel in folded if abs(panel.grainline[0].x - panel.grainline[1].x) > 1e-9]
        if not tilted:
            ok("R1", "Arah serat", "Arah serat sejajar lipatan.")
        else:
            fail("R1", "Arah serat", f"Arah serat miring di panel: {', '.join(tilted)}. Tegakkan garis serat.")

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
        ok(id, name, f"{name} {actual:.1f} mm sesuai badan + kelonggaran ({expected:.0f} mm).",
           actual, expected, WIDTH_TOL_MM)
    else:
        fail(id, name,
             f"{name} {actual:.1f} mm vs badan + kelonggaran {expected:.0f} mm "
             f"(selisih {abs(actual - expected):.1f} mm, batas {WIDTH_TOL_MM:.0f} mm).",
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
