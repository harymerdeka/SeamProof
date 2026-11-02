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
COLLAR_STYLE = {"none": 0, "shirt_collar": 1, "peter_pan": 2, "band": 3, "other": 2}
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
# Front V depth from the neck point (assumption: within the neckline_depth_front range; dress at its 140 mm cap).
V_NECK_DEPTH = {"shirt": 130.0, "dress": 140.0}
# Blouse and tunic sleeves; long is capped at the arm by the draft.
BLOUSE_SLEEVE = {"short": 200.0, "elbow": 300.0, "three_quarter": 430.0, "long": 650.0}
# Tunic hem below the front waist, and hem flare per side by fit (assumed, tune with samples).
TUNIC_LENGTH = {"waist": 350.0, "high_hip": 350.0, "hip": 350.0, "thigh": 420.0, "knee": 560.0, "below_knee": 650.0}
TUNIC_FLARE = {"fitted": 20.0, "regular": 60.0, "boxy": 110.0}
# Front drop and full width of the shaped necklines (assumed, tune with samples).
SCOOP_DEPTH_MM = 130.0
BOAT_DEPTH_MM, BOAT_WIDTH_MM = 60.0, 210.0
BOAT_CAP_HEIGHT_MM = 90.0  # the wider neck shortens the armhole, so the plain darted sleeve cap drops too

LABELS = {"shirt": "Shirt", "vest": "Vest", "skirt": "Skirt", "tshirt": "T-shirt", "blouse": "Blouse", "tunic": "Tunic",
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
    # Flare and length fitted to a sketch (seamproof.sketchfit); empty when the tables apply.
    proportions: dict[str, float] = field(default_factory=dict)


def apply_proportions(plan: StylePlan, proportions: Any) -> None:
    """Override table flare and length with fitted values, range-checked as parameters."""
    if not isinstance(proportions, dict) or not proportions:
        raise ValueError("proportions must be an object of fitted parameters.")
    if plan.garment not in ("skirt", "dress"):
        raise ValueError("Sketch proportions apply only to a skirt or dress.")
    if set(proportions) - {"skirt_flare", "skirt_length"}:
        raise ValueError("proportions may only set skirt_flare and skirt_length.")
    resolve_params(proportions)
    plan.proportions = {name: float(value) for name, value in proportions.items()}
    plan.params.update(plan.proportions)
    plan.notes.append("Hem width and length follow the sketch instead of the standard category values.")


def _shirt(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Shirt parameters from top features."""
    sleeveless = f.get("sleeve_length") == "sleeveless"
    if f.get("collar") == "other":
        notes.append("Unknown collar shape — using a round (peter pan) collar.")
    if f.get("opening") != "button_front":
        notes.append("The shirt is always drafted with a button front.")
    if f.get("length") == "below_knee":
        notes.append("Below-knee length is cut to the shirt limit (450 mm below the waist).")
    collar = COLLAR_STYLE.get(f.get("collar", "none"), 0)
    v_neck = _v_neck(f, bool(collar), notes)
    if not collar:
        notes.append("The collarless neck is finished with bias.")
    if sleeveless:
        return _vest(f, notes, float(collar), v_neck)
    sleeve_params = _cuffed_sleeve(f, notes)
    if f.get("contrast_collar") or f.get("contrast_cuff"):
        parts = [name for key, name in (("contrast_collar", "collar"), ("contrast_cuff", "cuff")) if f.get(key)]
        notes.append(f"Contrast fabric for {' & '.join(parts)} — the production marker still uses one fabric.")
    return sleeve_params | _shaped_neck(f, bool(collar), v_neck) | {
        "design_ease": FIT_EASE.get(f.get("fit", "regular"), 60.0),
        "shoulder_drop": 50.0 if f.get("shoulder") == "dropped" else 0.0,
        "shirt_length": SHIRT_LENGTH.get(f.get("length", "hip"), 200.0),
        "collar_style": float(collar),
        "pocket_count": float(int(f.get("pockets", 0))),
        "neck_drop_back": 20.0,
        "neck_v": 1.0 if v_neck else 0.0,
    }


def _cuffed_sleeve(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Sleeve length, cuff and gather parameters of a sleeve drafted on the shirt sleeve."""
    style = f.get("sleeve_style", "straight")
    sleeve = SLEEVE_LENGTH.get(f.get("sleeve_length", "short"), 200.0)
    cuff = bool(f.get("cuff")) or bool(f.get("elastic_cuff"))
    elastic = bool(f.get("elastic_cuff"))
    if style != "straight" and not cuff:
        notes.append("Bishop/puff sleeves gather into a cuff — a cuff was added.")
        cuff = True
    if cuff and style != "straight" and not elastic and sleeve >= SLEEVE_LENGTH["long"]:
        # A fitted wrist cuff without elastic needs a sleeve placket, which is not drafted.
        notes.append("A button cuff with a sleeve placket is not supported yet — using an elastic cuff.")
        elastic = True
    if style == "puff":
        notes.append("The sleeve cap is gathered at the shoulder (puff) — gather between the cap notches.")
    return {
        "shirt_sleeve_length": sleeve,
        "cuff_depth": CUFF_DEPTH_MM if cuff else 0.0,
        "sleeve_hem_fullness": SLEEVE_HEM_FULLNESS.get(style, 0.0),
        "sleeve_cap_gather": SLEEVE_CAP_GATHER.get(style, 0.0),
        "cuff_elastic": 1.0 if cuff and elastic else 0.0,
    }


def _shaped_neck(f: dict[str, Any], collar: bool, v_neck: bool) -> dict[str, float]:
    """Front neck drop and width for the neckline; a collar sits closer to the neck (assumption)."""
    if collar:
        return {"neckline_depth_front": 75.0}
    if v_neck:
        return {"neckline_depth_front": V_NECK_DEPTH["shirt"]}
    if f.get("neckline") == "scoop":
        return {"neckline_depth_front": SCOOP_DEPTH_MM}
    if f.get("neckline") == "boat":
        return {"neckline_depth_front": BOAT_DEPTH_MM, "neckline_width": BOAT_WIDTH_MM}
    return {"neckline_depth_front": 90.0}


def _darted(kind: str, f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Blouse or tunic parameters: the darted block with sleeve, neckline and tunic hem choices."""
    v_neck = _v_neck(f, False, notes)
    out: dict[str, float] = _shaped_neck(f, False, v_neck)
    out["neck_v"] = 1.0 if v_neck else 0.0
    length = f.get("sleeve_length")
    cuffed = f.get("cuff") or f.get("elastic_cuff") or f.get("sleeve_style", "straight") != "straight"
    if length == "sleeveless":
        notes.append("Sleeveless: underarm raised 15 mm, shoulder narrowed 10 mm, armholes finished with bias.")
        if cuffed:
            notes.append("Cuff and sleeve style are ignored on a sleeveless style.")
        out["sleeveless"] = 1.0
    elif cuffed:
        out |= _cuffed_sleeve(f, notes)
    else:
        if length in BLOUSE_SLEEVE:
            out["blouse_sleeve_length"] = BLOUSE_SLEEVE[length]
        if f.get("neckline") == "boat":
            out["sleeve_cap_height"] = BOAT_CAP_HEIGHT_MM
    if f.get("shoulder") == "dropped":
        notes.append(f"A dropped shoulder on a {kind} is not supported yet — using a set-in shoulder.")
    if int(f.get("pockets", 0)):
        notes.append(f"Pockets on a {kind} are not supported yet — choose a shirt for pockets.")
    if f.get("contrast_cuff"):
        notes.append("Contrast fabric for the cuff — the production marker still uses one fabric.")
    if kind == "tunic":
        if f.get("length") in ("waist", "high_hip"):
            notes.append("A tunic reaches at least the hip — cut to hip length.")
        if f.get("length") in TUNIC_LENGTH:
            out["tunic_length"] = TUNIC_LENGTH[f["length"]]
        if f.get("fit") in TUNIC_FLARE:
            out["side_flare"] = TUNIC_FLARE[f["fit"]]
    else:
        if f.get("length") not in (None, "hip"):
            notes.append("The blouse is cut at hip length — choose a tunic for longer or a shirt for shorter.")
        if f.get("fit") == "boxy":
            notes.append("The blouse keeps its darted fit — choose a shirt or tunic for a loose look.")
    return out


def _v_neck(f: dict[str, Any], collar: bool, notes: list[str]) -> bool:
    """A V neckline is drafted only without a collar."""
    if f.get("neckline") != "v":
        return False
    if collar:
        notes.append("A V neck with a collar is not supported yet — using a round collared neck.")
        return False
    return True


def _vest(f: dict[str, Any], notes: list[str], collar: float, v_neck: bool) -> dict[str, float]:
    """Sleeveless shirt (vest): no sleeve, cuff or dropped shoulder; armholes bound with bias."""
    notes.append("Sleeveless: underarm raised 15 mm, shoulder narrowed 10 mm, armholes finished with bias.")
    if f.get("shoulder") == "dropped":
        notes.append("A dropped shoulder needs a sleeve — the sleeveless style uses a set-in shoulder.")
    if f.get("cuff") or f.get("elastic_cuff") or f.get("sleeve_style", "straight") != "straight":
        notes.append("Cuff and sleeve style are ignored on a sleeveless style.")
    if f.get("contrast_collar"):
        notes.append("Contrast fabric for the collar — the production marker still uses one fabric.")
    return {
        "design_ease": FIT_EASE.get(f.get("fit", "regular"), 60.0),
        "shoulder_drop": 0.0,
        "shirt_length": SHIRT_LENGTH.get(f.get("length", "hip"), 200.0),
        "collar_style": collar,
        "pocket_count": float(int(f.get("pockets", 0))),
        **_shaped_neck(f, bool(collar), v_neck),
        "neck_drop_back": 20.0,
        "sleeveless": 1.0,
        "neck_v": 1.0 if v_neck else 0.0,
    }


def _skirt(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Skirt parameters from bottom features."""
    if f.get("silhouette") == "other":
        notes.append("Unknown skirt silhouette — using A-line.")
    if f.get("hem") == "other":
        notes.append("Unknown hem shape — using a straight hem.")
    notes.append("The skirt uses a concealed zip at the side (assumption).")
    return {
        "skirt_flare": SKIRT_FLARE.get(f.get("silhouette", "a_line"), 80.0),
        "skirt_length": SKIRT_LENGTH.get(f.get("length", "knee"), 580.0),
        "scallop_depth": SCALLOP_DEPTH_MM if f.get("hem") == "scalloped" else 0.0,
        "waistband_height": WAISTBAND_MM if f.get("waistband") else 0.0,
    }


def _dress(f: dict[str, Any], notes: list[str]) -> dict[str, float]:
    """Dress parameters from dress features; undraftable details become notes."""
    seam = bool(f.get("waist_seam"))
    sleeveless = f.get("sleeve_length") == "sleeveless"
    v_neck = _v_neck(f, False, notes)
    if sleeveless:
        notes.append("Sleeveless: underarm raised 15 mm, shoulder narrowed 10 mm, armholes finished with bias.")
    if v_neck:
        notes.append("V neck 14 cm deep from the neck point, finished with a facing.")
    if f.get("collar", "none") != "none":
        notes.append("A collar on a dress is not supported yet — the neck is finished with a facing.")
    if f.get("opening") == "button_front":
        notes.append("A button front on a dress is not supported yet — using a concealed zip on the left side.")
    if not sleeveless and (f.get("cuff") or f.get("elastic_cuff")):
        notes.append("Cuffs on a dress are not supported yet — the sleeve end is hemmed.")
    if not sleeveless and f.get("sleeve_style", "straight") != "straight":
        notes.append("Bishop/puff sleeves on a dress are not supported yet — using a plain sleeve.")
    if int(f.get("pockets", 0)):
        notes.append("Pockets on a dress are not supported yet.")
    if f.get("contrast_collar") or f.get("contrast_cuff"):
        notes.append("Contrast fabric — the production marker still uses one fabric.")
    if f.get("shoulder") == "dropped":
        notes.append("A dropped shoulder on a dress is not supported yet — using a set-in shoulder.")
    if f.get("fit", "regular") != "regular":
        notes.append("The dress bodice uses the standard blouse block (fabric ease), not the fitted/loose look from the photo.")
    if f.get("silhouette") == "other":
        notes.append("Unknown skirt silhouette — using A-line.")
    scallop = f.get("hem") == "scalloped"
    if f.get("hem") == "other":
        notes.append("Unknown hem shape — using a straight hem.")
    if scallop and not seam:
        notes.append("A scallop hem needs a waist-seam dress — this straight dress uses a straight hem.")
        scallop = False
    notes.append("Waist-seam dress: a darted bodice joined to a skirt." if seam
                 else "Straight (shift) dress with no waist seam.")
    notes.append("The dress uses a concealed zip on the left side (assumption).")
    out = {"sleeveless": 1.0} if sleeveless else {}
    if v_neck:
        out.update(neck_v=1.0, neckline_depth_front=V_NECK_DEPTH["dress"])
    elif f.get("neckline") in ("scoop", "boat"):
        out.update(_shaped_neck(f, False, False))
        if f["neckline"] == "boat" and not sleeveless:
            out["sleeve_cap_height"] = BOAT_CAP_HEIGHT_MM
    return out | {
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
        if kind == "vest":
            features = {**features, "sleeve_length": "sleeveless"}
        shirt_like = (kind in ("shirt", "vest") or features.get("opening") == "button_front"
                      or features.get("collar") in ("shirt_collar", "peter_pan", "band"))
        if shirt_like:
            garment, params = "shirt", _shirt(features, notes)
        elif kind in ("blouse", "tunic"):
            garment, params = kind, _darted(kind, features, notes)
        elif kind == "tshirt":
            garment = kind
            notes.append(f"{LABELS[kind]} uses the standard block; style details from the photo are not changed.")
        else:
            garment = None
            notes.append("This kind of top is not in the pattern library yet.")
    elif part == "dress":
        garment, params = "dress", _dress(features, notes)
    elif part == "bottom":
        if kind == "skirt":
            garment, params = "skirt", _skirt(features, notes)
        else:
            garment = None
            notes.append("These trousers or bottoms are not in the pattern library yet.")
    else:
        raise ValueError(f"Unknown part: {part}")
    if garment is not None:
        unknown = set(params) - set(GARMENT_PARAMS[garment])
        if unknown:
            raise ValueError(f"Style params not applicable to {garment}: {sorted(unknown)}")
        resolve_params(params)  # every table value must sit inside its declared range
    return StylePlan(part, garment, params, notes, dict(features), kind)
