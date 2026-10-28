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
        "pulling_at_underarm", "Ketarik di ketiak",
        (RuleDelta("armhole_curve_depth", 5.0, 10.0),),
        "Longgarkan kerung lengan sesuai tabel."),
    "sleeve_cap_puckering": FitRule(
        "sleeve_cap_puckering", "Puncak lengan mengerut",
        (RuleDelta("sleeve_cap_height", -8.0, -3.0),),
        "Rendahkan puncak lengan sesuai tabel."),
    "shoulder_seam_drops": FitRule(
        "shoulder_seam_drops", "Kampuh bahu turun",
        (RuleDelta("shoulder_slope", 3.0, 8.0),),
        "Sesuaikan kemiringan bahu sesuai tabel.",
        ("Arah slope mengikuti definisi shoulder_slope (turun dari leher).",)),
    "neckline_gapes": FitRule(
        "neckline_gapes", "Kerung leher menganga",
        (RuleDelta("neck_drop_front", -6.0, -3.0),),
        "Naikkan kerung leher depan sesuai tabel."),
    "body_too_tight_chest": FitRule(
        "body_too_tight_chest", "Badan sempit di dada",
        (),
        "Tambah kelonggaran dalam rentang kain atau naikkan ukuran saat draft ulang.",
        ("Tidak ada parameter lebar badan; perubahan ease butuh keputusan redraft.",)),
    "drag_lines_toward_bust": FitRule(
        "drag_lines_toward_bust", "Garis tarik ke arah dada",
        (RuleDelta("bust_dart_angle", 1.0, 3.0),),
        "Putar kupnat mengikuti tabel.",
        ("Sketsa meminta mm untuk intake; intake turunan (bust-high_bust), "
         "maka dipakai rotasi kupnat dalam derajat.",)),
    "dart_point_too_pointed": FitRule(
        "dart_point_too_pointed", "Ujung kupnat terlalu runcing",
        (RuleDelta("bust_dart_clearance", 5.0, 10.0),),
        "Jauhkan ujung kupnat dari puncak sesuai tabel."),
    "gaping_front_armhole": FitRule(
        "gaping_front_armhole", "Kerung depan longgar",
        (RuleDelta("armhole_curve_depth", -6.0, -3.0),),
        "Kecilkan kerung depan sesuai tabel.",
        ("Alternatif memindah ke kupnat tidak otomatis; butuh pola baru.",)),
    "hem_swings_forward": FitRule(
        "hem_swings_forward", "Hem mengayun ke depan",
        (RuleDelta("hem_curve", 2.0, 5.0),),
        "Sesuaikan lengkung hem sesuai tabel."),
    "tunic_flare_too_wide": FitRule(
        "tunic_flare_too_wide", "Flare tunik terlalu lebar",
        (RuleDelta("side_flare", -30.0, -10.0),),
        "Kurangi flare sisi sesuai tabel."),
}

LABEL_ID = "Saran koreksi — konfirmasi sebelum memotong"


def delta_for(rule: FitRule, severity: int) -> dict[str, float]:
    """Interpolate each bounded delta by severity 1..3."""
    if severity not in (1, 2, 3):
        raise ValueError("Severity harus 1, 2 atau 3.")
    out = {}
    for item in rule.deltas:
        out[item.param] = item.lo_mm + (item.hi_mm - item.lo_mm) * (severity - 1) / 2
    return out
