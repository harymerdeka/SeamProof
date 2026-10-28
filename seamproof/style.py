"""Deterministic map from photo style categories to a garment and its parameters.

The vision model only names categories (seamproof.vision). Every number here
is a fixed table value, chosen for an average Indonesian women's M figure
and marked as an assumption to tune with workshop samples. Anything the
library cannot draft is reported in `notes`, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from seamproof.params import GARMENT_PARAMS, resolve_params

# Assumed mm values per category (tune with real samples).
FIT_EASE = {"fitted": 20.0, "regular": 60.0, "boxy": 140.0}
SHIRT_LENGTH = {"waist": 20.0, "high_hip": 110.0, "hip": 200.0, "thigh": 320.0, "knee": 450.0,
                "below_knee": 450.0}
SLEEVE_LENGTH = {"sleeveless": 200.0, "short": 200.0, "elbow": 300.0, "three_quarter": 430.0,
                 "long": 580.0}
COLLAR_STYLE = {"none": 0, "shirt_collar": 1, "peter_pan": 2, "band": 1, "other": 2}
SKIRT_FLARE = {"straight": 0.0, "a_line": 80.0, "flared": 180.0, "other": 80.0}
SKIRT_LENGTH = {"mini": 420.0, "above_knee": 500.0, "knee": 580.0, "midi": 750.0, "maxi": 950.0}
CUFF_DEPTH_MM = 35.0
# Bishop: hem fullness gathered into the cuff; puff adds a gathered cap (assumed, tune with samples).
SLEEVE_HEM_FULLNESS = {"straight": 0.0, "bishop": 120.0, "puff": 120.0}
SLEEVE_CAP_GATHER = {"straight": 0.0, "bishop": 0.0, "puff": 40.0}
SCALLOP_DEPTH_MM = 18.0
WAISTBAND_MM = 35.0
# Dress sleeves are measured below the crown; long is capped at the arm by the draft.
DRESS_SLEEVE = {"sleeveless": 200.0, "short": 200.0, "elbow": 300.0, "three_quarter": 430.0, "long": 570.0}

LABELS = {"shirt": "Kemeja", "skirt": "Rok", "tshirt": "Kaos", "blouse": "Blus", "tunic": "Tunik",
          "dress": "Dress"}


@dataclass
class StylePlan:
    """What SeamProof will draft for one garment piece, and what it cannot."""

    part: str
    garment: str | None
    params: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    kind: str = ""


def _shirt(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Shirt parameters from top features."""
    if f.get("sleeve_length") == "sleeveless":
        notes.append("Tanpa lengan belum didukung — dipakai lengan pendek.")
    if f.get("collar") == "band":
        notes.append("Kerah tegak (band) belum didukung — dipakai kerah rebah runcing.")
    if f.get("collar") == "other":
        notes.append("Bentuk kerah tidak dikenal — dipakai kerah bulat (peter pan).")
    if f.get("opening") != "button_front":
        notes.append("Kemeja selalu digambar dengan kancing depan.")
    if f.get("length") == "below_knee":
        notes.append("Panjang di bawah lutut dipotong ke batas kemeja (450 mm di bawah pinggang).")
    collar = COLLAR_STYLE.get(f.get("collar", "none"), 0)
    style = f.get("sleeve_style", "straight")
    sleeve = SLEEVE_LENGTH.get(f.get("sleeve_length", "short"), 200.0)
    cuff = bool(f.get("cuff")) or bool(f.get("elastic_cuff"))
    elastic = bool(f.get("elastic_cuff"))
    if style != "straight" and not cuff:
        notes.append("Lengan bishop/puff dikerut ke manset — manset ditambahkan.")
        cuff = True
    if cuff and style != "straight" and not elastic and sleeve >= SLEEVE_LENGTH["long"]:
        # A fitted wrist cuff without elastic needs a sleeve placket, which is not drafted.
        notes.append("Manset kancing dengan belahan lengan belum didukung — dipakai manset karet.")
        elastic = True
    if style == "puff":
        notes.append("Puncak lengan dikerut di bahu (puff) — kerut di antara notch puncak lengan.")
    if f.get("contrast_collar") or f.get("contrast_cuff"):
        parts = [name for key, name in (("contrast_collar", "kerah"), ("contrast_cuff", "manset")) if f.get(key)]
        notes.append(f"Kain kontras untuk {' & '.join(parts)} — denah produksi masih memakai satu kain.")
    return {
        "design_ease": FIT_EASE.get(f.get("fit", "regular"), 60.0),
        "shoulder_drop": 50.0 if f.get("shoulder") == "dropped" else 0.0,
        "shirt_length": SHIRT_LENGTH.get(f.get("length", "hip"), 200.0),
        "shirt_sleeve_length": SLEEVE_LENGTH.get(f.get("sleeve_length", "short"), 200.0),
        "collar_style": float(collar),
        "cuff_depth": CUFF_DEPTH_MM if cuff else 0.0,
        "sleeve_hem_fullness": SLEEVE_HEM_FULLNESS.get(style, 0.0),
        "sleeve_cap_gather": SLEEVE_CAP_GATHER.get(style, 0.0),
        "cuff_elastic": 1.0 if cuff and elastic else 0.0,
        "pocket_count": float(int(f.get("pockets", 0))),
        # A collar sits closer to the neck than an open neckline (assumption).
        "neckline_depth_front": 75.0 if collar else 90.0,
        "neck_drop_back": 20.0,
    }


def _skirt(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Skirt parameters from bottom features."""
    if f.get("silhouette") == "other":
        notes.append("Siluet rok tidak dikenal — dipakai A-line.")
    if f.get("hem") == "other":
        notes.append("Bentuk kelim tidak dikenal — dipakai kelim lurus.")
    notes.append("Rok memakai ritsleting jepang di sisi (asumsi).")
    return {
        "skirt_flare": SKIRT_FLARE.get(f.get("silhouette", "a_line"), 80.0),
        "skirt_length": SKIRT_LENGTH.get(f.get("length", "knee"), 580.0),
        "scallop_depth": SCALLOP_DEPTH_MM if f.get("hem") == "scalloped" else 0.0,
        "waistband_height": WAISTBAND_MM if f.get("waistband") else 0.0,
    }


def _dress(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Dress parameters from dress features; undraftable details become notes."""
    seam = bool(f.get("waist_seam"))
    if f.get("sleeve_length") == "sleeveless":
        notes.append("Tanpa lengan belum didukung — dipakai lengan pendek.")
    if f.get("collar", "none") != "none":
        notes.append("Kerah pada dress belum didukung — leher dirapikan dengan lapisan (facing).")
    if f.get("opening") == "button_front":
        notes.append("Kancing depan pada dress belum didukung — dipakai ritsleting jepang di sisi kiri.")
    if f.get("cuff") or f.get("elastic_cuff"):
        notes.append("Manset pada dress belum didukung — ujung lengan dikelim.")
    if f.get("sleeve_style", "straight") != "straight":
        notes.append("Lengan bishop/puff pada dress belum didukung — dipakai lengan biasa.")
    if int(f.get("pockets", 0)):
        notes.append("Saku pada dress belum didukung.")
    if f.get("contrast_collar") or f.get("contrast_cuff"):
        notes.append("Kain kontras — denah produksi masih memakai satu kain.")
    if f.get("shoulder") == "dropped":
        notes.append("Bahu turun pada dress belum didukung — dipakai bahu pas (set-in).")
    if f.get("fit", "regular") != "regular":
        notes.append("Siluet badan dress memakai blok blus standar (kelonggaran kain), bukan pas/longgar dari foto.")
    if f.get("silhouette") == "other":
        notes.append("Siluet rok tidak dikenal — dipakai A-line.")
    scallop = f.get("hem") == "scalloped"
    if f.get("hem") == "other":
        notes.append("Bentuk kelim tidak dikenal — dipakai kelim lurus.")
    if scallop and not seam:
        notes.append("Kelim scallop butuh dress bersambungan pinggang — dress lurus ini memakai kelim lurus.")
        scallop = False
    notes.append("Dress dengan sambungan pinggang: badan berkupnat disambung rok." if seam
                 else "Dress lurus (shift) tanpa sambungan pinggang.")
    notes.append("Dress memakai ritsleting jepang di sisi kiri (asumsi).")
    return {
        "dress_waist_seam": 1.0 if seam else 0.0,
        "dress_sleeve_length": DRESS_SLEEVE.get(f.get("sleeve_length", "short"), 200.0),
        "skirt_flare": SKIRT_FLARE.get(f.get("silhouette", "a_line"), 80.0),
        "skirt_length": SKIRT_LENGTH.get(f.get("length", "knee"), 580.0),
        "scallop_depth": SCALLOP_DEPTH_MM if scallop else 0.0,
    }


def plan_piece(part: str, kind: str, features: dict[str, Any]) -> StylePlan:
    """Choose a garment and parameters for one validated piece."""
    notes: list[str] = []
    params: dict[str, float] = {}
    garment: str | None
    if part == "top":
        shirt_like = (kind == "shirt" or features.get("opening") == "button_front"
                      or features.get("collar") in ("shirt_collar", "peter_pan", "band"))
        if shirt_like:
            garment, params = "shirt", _shirt(features, notes)
        elif kind in ("tshirt", "blouse", "tunic"):
            garment = kind
            notes.append(f"{LABELS[kind]} memakai blok standar; detail gaya dari foto tidak diubah.")
        else:
            garment = None
            notes.append("Jenis atasan ini belum ada di pustaka pola.")
    elif part == "dress":
        garment, params = "dress", _dress(features, notes)
    elif part == "bottom":
        if kind == "skirt":
            garment, params = "skirt", _skirt(features, notes)
        else:
            garment = None
            notes.append("Celana/bawahan ini belum ada di pustaka pola.")
    else:
        raise ValueError(f"Unknown part: {part}")
    if garment is not None:
        unknown = set(params) - set(GARMENT_PARAMS[garment])
        if unknown:
            raise ValueError(f"Style params not applicable to {garment}: {sorted(unknown)}")
        resolve_params(params)  # every table value must sit inside its declared range
    return StylePlan(part, garment, params, notes, dict(features), kind)
