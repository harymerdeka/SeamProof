"""Drafting parameter limits and explicit geometry dependencies, in mm."""

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ParameterSpec:
    """An allowed scalar change and every panel edge it may move."""

    default: float
    min: float
    max: float
    unit: str
    description: str
    affects: list[tuple[str, str]]
    # Whole-number style switches (e.g. collar type, pocket count).
    integer: bool = False
    # Style choices come from the photo or the user, never from the repair LLM.
    tunable: bool = True


def _body_edges(*names: str) -> list[tuple[str, str]]:
    """Expand common front/back edge dependencies."""
    return [(panel, f"{panel}.{name}") for panel in ("front", "back") for name in names]


def _facing_edges() -> list[tuple[str, str]]:
    """List every facing edge moved by a neckline change."""
    return [("facing", f"facing.{name}") for name in ("neck", "end_shoulder", "outer", "end_cf")]


def _panel(name: str, *edges: str) -> list[tuple[str, str]]:
    """List edges of one panel by short name."""
    return [(name, f"{name}.{edge}") for edge in edges]


SHIRT_SLEEVE = _panel("sleeve", "cap", "underarm_front", "underarm_back", "hem", "cuff_join")
SHIRT_CUFF = _panel("cuff", "attach", "end_r", "outer", "end_l")
SHIRT_COLLAR = _panel("collar", "neck", "end", "outer", "fold")
SKIRT_BODY = ("waist_cf", "dart_leg1", "dart_leg2", "waist_side", "side", "hem", "scallop")
HEM_FACING = _panel("hem_facing", "scallop", "side_r", "top", "fold")
WAIST_FINISH = (_panel("waistband", "attach", "end_r", "outer", "end_l")
                + _panel("waist_facing", "attach", "end_r", "outer", "end_l"))

DRESS_WAIST = _body_edges("waist_side", "wdart_leg1", "wdart_leg2", "waist_cf")
DRESS_SKIRT = [(p, f"{p}.{e}") for p in ("skirt_front", "skirt_back") for e in SKIRT_BODY + ("fold",)]
ARMHOLE_BIAS = _panel("armhole_bias", "attach", "end_r", "outer", "end_l")
NECK_BIAS = _panel("neck_bias", "attach", "end_r", "outer", "end_l")
BAND_EDGES = [("neckband", f"neckband.{name}") for name in ("attach", "end_right", "attach_inner", "end_left")]
# Assumptions: shoulder_slope is a vertical drop, not an angle; curve depth is
# a lateral control-point inset. Limits are provisional workshop tuning bounds.
PARAMETERS: dict[str, ParameterSpec] = {
    "armhole_curve_depth": ParameterSpec(15, 0, 45, "mm", "Lateral armhole scoop control-point inset", _body_edges("armhole")),
    "sleeve_cap_height": ParameterSpec(100, 40, 160, "mm", "Sleeve crown to bicep-line depth", [("sleeve", f"sleeve.{n}") for n in ("cap", "underarm_front", "underarm_back")]),
    "shoulder_slope": ParameterSpec(25, 5, 50, "mm", "Vertical drop from neck side to shoulder point", _body_edges("shoulder", "armhole")),
    "neck_drop_front": ParameterSpec(80, 40, 140, "mm", "Centre-front neck drop", [("front", "front.neckline"), ("front", "front.fold")] + BAND_EDGES),
    "neck_drop_back": ParameterSpec(25, 10, 60, "mm", "Centre-back neck drop", [("back", "back.neckline"), ("back", "back.fold")] + BAND_EDGES),
    "body_length_adjust": ParameterSpec(0, -80, 120, "mm", "Adjustment to measured back length", _body_edges("side", "hem", "fold")),
    "sleeve_width_adjust": ParameterSpec(0, -40, 80, "mm", "Adjustment to full bicep circumference including ease", [("sleeve", f"sleeve.{n}") for n in ("cap", "underarm_front", "underarm_back", "hem")]),
    "bust_dart_clearance": ParameterSpec(20, 10, 35, "mm", "Dart tip set back from the bust apex", [("front", "front.dart_leg1"), ("front", "front.dart_leg2"), ("front", "front.side_lower")]),
    "bust_dart_angle": ParameterSpec(0, -15, 2, "deg", "Dart centreline rotation from the apex direction, positive lifts the tip; higher tips sliver the dart", [("front", "front.dart_leg1"), ("front", "front.dart_leg2"), ("front", "front.side_lower")]),
    "back_waist_dart_intake": ParameterSpec(20, 0, 40, "mm", "Back waist dart intake per half-back, trued into the side seam; 0 sews no dart", [("back", "back.side_upper"), ("back", "back.side_low"), ("back", "back.armhole")]),
    "neckline_depth_front": ParameterSpec(90, 60, 140, "mm", "Centre-front neck drop", [("front", "front.neckline"), ("front", "front.fold")] + _facing_edges()),
    "neckline_width": ParameterSpec(180, 150, 220, "mm", "Full neckline width; moves the shoulder point and armhole", _body_edges("neckline", "shoulder", "armhole") + _facing_edges()),
    "tunic_length": ParameterSpec(560, 300, 700, "mm", "Tunic hem length measured down from the waist", [("front", "front.side_lower"), ("front", "front.hem"), ("front", "front.fold"), ("front", "front.slit"), ("back", "back.side_lower"), ("back", "back.hem"), ("back", "back.fold"), ("back", "back.slit")]),
    "side_flare": ParameterSpec(60, 0, 150, "mm", "Extra hem half-width per side for the tunic", [("front", "front.side_lower"), ("front", "front.hem"), ("front", "front.slit"), ("back", "back.side_lower"), ("back", "back.hem"), ("back", "back.slit")]),
    "hem_curve": ParameterSpec(10, 0, 30, "mm", "Tunic hem centre dip below the side hem", _body_edges("hem")),
    "slit_height": ParameterSpec(80, 0, 200, "mm", "Tunic side slit height from the hem; 0 sews no slit", [("front", "front.side_lower"), ("front", "front.slit"), ("front", "front.hem"), ("back", "back.side_lower"), ("back", "back.slit"), ("back", "back.hem")]),
    # Shirt style parameters. Values are mapped from photo categories by
    # seamproof.style; they are design choices, so the repair LLM never edits them.
    "design_ease": ParameterSpec(60, 0, 250, "mm", "Silhouette ease added to fabric ease at bust (fitted 0, regular 60, boxy 140)", _body_edges("side", "hem", "armhole") + SHIRT_SLEEVE + SHIRT_CUFF, tunable=False),
    "shoulder_drop": ParameterSpec(0, 0, 90, "mm", "Shoulder seam extension past the shoulder point (dropped shoulder)", _body_edges("shoulder", "armhole", "side") + SHIRT_SLEEVE, tunable=False),
    "shirt_length": ParameterSpec(180, 0, 450, "mm", "Shirt hem below the waist line", _body_edges("side", "hem") + [("front", "front.edge"), ("back", "back.fold")], tunable=False),
    "shirt_sleeve_length": ParameterSpec(250, 80, 650, "mm", "Finished shirt sleeve length from the armhole seam, cuff included", SHIRT_SLEEVE, tunable=False),
    "collar_style": ParameterSpec(2, 0, 3, "", "Collar type: 0 none, 1 flat pointed collar, 2 peter pan, 3 band", SHIRT_COLLAR, integer=True, tunable=False),
    "collar_width": ParameterSpec(65, 40, 100, "mm", "Finished collar width from the neck seam", SHIRT_COLLAR, tunable=False),
    "cuff_depth": ParameterSpec(0, 0, 60, "mm", "Finished turned-up cuff depth; 0 sews a plain hem", SHIRT_SLEEVE + SHIRT_CUFF, tunable=False),
    "sleeve_hem_fullness": ParameterSpec(0, 0, 300, "mm", "Extra sleeve hem width gathered into the cuff (bishop sleeve); 0 is straight", SHIRT_SLEEVE + SHIRT_CUFF, tunable=False),
    "sleeve_cap_gather": ParameterSpec(0, 0, 90, "mm", "Extra sleeve cap length gathered into the armhole (puff shoulder)", SHIRT_SLEEVE + SHIRT_CUFF, tunable=False),
    "cuff_elastic": ParameterSpec(0, 0, 1, "", "1 makes a fitted elastic casing cuff that slides over the hand", SHIRT_SLEEVE + SHIRT_CUFF, integer=True, tunable=False),
    "pocket_count": ParameterSpec(0, 0, 2, "", "Patch chest pockets on the front", _panel("pocket", "hem", "side_r", "bottom", "side_l"), integer=True, tunable=False),
    "placket_width": ParameterSpec(20, 15, 30, "mm", "Button extension past centre front", [("front", "front.edge"), ("front", "front.ext_top"), ("front", "front.hem")], tunable=False),
    # Skirt style parameters.
    "skirt_length": ParameterSpec(480, 300, 1000, "mm", "Skirt length from the waist seam", [(p, f"{p}.{e}") for p in ("front", "back") for e in ("side", "hem", "scallop", "fold")] + HEM_FACING, tunable=False),
    "skirt_flare": ParameterSpec(80, 0, 250, "mm", "Hem flare added per side seam (straight 0, A-line 80, flared 180)", [(p, f"{p}.{e}") for p in ("front", "back") for e in ("side", "hem", "scallop")] + HEM_FACING, tunable=False),
    "skirt_hip_ease": ParameterSpec(50, 20, 120, "mm", "Hip ease for the skirt", [(p, f"{p}.{e}") for p in ("front", "back") for e in SKIRT_BODY] + HEM_FACING, tunable=False),
    "scallop_depth": ParameterSpec(0, 0, 40, "mm", "Scallop depth at the hem; 0 sews a straight hem", [(p, f"{p}.{e}") for p in ("front", "back") for e in ("hem", "scallop")] + HEM_FACING, tunable=False),
    "scallop_width": ParameterSpec(55, 30, 90, "mm", "Target width of one hem scallop", [(p, f"{p}.{e}") for p in ("front", "back") for e in ("hem", "scallop")] + HEM_FACING, tunable=False),
    "waistband_height": ParameterSpec(35, 0, 60, "mm", "Finished waistband height; 0 finishes the waist with a bias facing", WAIST_FINISH, tunable=False),
    # Dress style switches, mapped from photo categories like the shirt ones.
    "dress_waist_seam": ParameterSpec(1, 0, 1, "", "1 joins bodice and skirt at the waist; 0 drafts a one-piece shift dress", _body_edges("side_lower", "side_low", "hem", "fold") + DRESS_WAIST + DRESS_SKIRT, integer=True, tunable=False),
    "dress_sleeve_length": ParameterSpec(570, 150, 650, "mm", "Dress sleeve length below the crown; capped at the measured arm length", _panel("sleeve", "underarm_front", "underarm_back", "hem"), tunable=False),
    "blouse_sleeve_length": ParameterSpec(650, 150, 650, "mm", "Blouse or tunic plain sleeve length below the crown; capped at the measured arm length", _panel("sleeve", "underarm_front", "underarm_back", "hem"), tunable=False),
    # Finishes shared by shirt and dress (see seamproof.draft.finish).
    "sleeveless": ParameterSpec(0, 0, 1, "", "1 drafts no sleeve: armhole raised, shoulder cut in, armholes bound with bias", _body_edges("shoulder", "armhole", "side", "side_upper", "side_lower", "side_low") + SHIRT_SLEEVE + SHIRT_CUFF + ARMHOLE_BIAS, integer=True, tunable=False),
    "neck_v": ParameterSpec(0, 0, 1, "", "1 makes the front neckline a straight V to centre front (no collar)", [("front", "front.neckline"), ("front", "front.fold"), ("front", "front.ext_top"), ("front", "front.edge")] + _facing_edges() + NECK_BIAS, integer=True, tunable=False),
}

# Extra edges a shared parameter moves in one garment only. Locality adds
# these for that garment, so other garments keep their tighter allowlists.
# Assumption: the shirt sleeve cap is solved against the armhole and the
# collar is traced from the necklines, so armhole/neck params reach them.
DARTED_AFFECTS: dict[str, list[tuple[str, str]]] = {
    "armhole_curve_depth": SHIRT_SLEEVE + SHIRT_CUFF + ARMHOLE_BIAS,
    "shoulder_slope": SHIRT_SLEEVE + SHIRT_CUFF + ARMHOLE_BIAS,
    "neckline_width": SHIRT_SLEEVE + SHIRT_CUFF + ARMHOLE_BIAS,
    "back_waist_dart_intake": SHIRT_SLEEVE + SHIRT_CUFF + ARMHOLE_BIAS,
    "sleeve_cap_height": _panel("sleeve", "hem"),
}
GARMENT_AFFECTS: dict[str, dict[str, list[tuple[str, str]]]] = {
    # Bias strips are cut to the edges they bind, so edge params reach them too.
    "shirt": {
        "armhole_curve_depth": SHIRT_SLEEVE + ARMHOLE_BIAS,
        "shoulder_slope": SHIRT_SLEEVE + SHIRT_COLLAR + ARMHOLE_BIAS,
        "neckline_width": SHIRT_SLEEVE + SHIRT_COLLAR + ARMHOLE_BIAS + NECK_BIAS,
        "neckline_depth_front": SHIRT_COLLAR + [("front", "front.ext_top"), ("front", "front.edge")] + NECK_BIAS,
        "neck_drop_back": SHIRT_COLLAR + NECK_BIAS,
        "design_ease": ARMHOLE_BIAS,
        "shoulder_drop": ARMHOLE_BIAS,
        "placket_width": NECK_BIAS,
        "collar_style": NECK_BIAS,
    },
    # Cuffed sleeves are solved against the armhole, so armhole params reach them.
    "blouse": DARTED_AFFECTS,
    "tunic": DARTED_AFFECTS,
    # Dress: skirt params move the renamed skirt halves; the back side truing
    # shifts the back waist dart; a new cap height moves the shortened hem.
    "dress": {
        # On a shift dress the skirt runs on in the body panels' lower side seam.
        "skirt_length": DRESS_SKIRT + _body_edges("side_lower"),
        "skirt_flare": DRESS_SKIRT + _body_edges("side_lower"),
        "skirt_hip_ease": DRESS_SKIRT,
        "scallop_depth": DRESS_SKIRT,
        "scallop_width": DRESS_SKIRT,
        "back_waist_dart_intake": DRESS_WAIST,
        "sleeve_cap_height": _panel("sleeve", "hem"),
        "armhole_curve_depth": ARMHOLE_BIAS,
        "shoulder_slope": ARMHOLE_BIAS,
        "neckline_width": ARMHOLE_BIAS,
    },
}


# Drafts honour only their listed subset; other known params are ignored.
# Assumption: tunic slits and the back waist dart may add or remove edges,
# so locality for those params compares edges by id rather than by order.
DARTED_STYLE = ["blouse_sleeve_length", "shirt_sleeve_length", "cuff_depth", "sleeve_hem_fullness",
                "sleeve_cap_gather", "cuff_elastic", "sleeveless", "neck_v"]
GARMENT_PARAMS: dict[str, list[str]] = {
    "tshirt": ["armhole_curve_depth", "sleeve_cap_height", "shoulder_slope", "neck_drop_front",
               "neck_drop_back", "body_length_adjust", "sleeve_width_adjust"],
    "blouse": ["armhole_curve_depth", "sleeve_cap_height", "shoulder_slope", "bust_dart_clearance",
               "bust_dart_angle", "back_waist_dart_intake", "neckline_depth_front", "neckline_width"] + DARTED_STYLE,
    "tunic": ["armhole_curve_depth", "sleeve_cap_height", "shoulder_slope", "bust_dart_clearance",
              "bust_dart_angle", "back_waist_dart_intake", "neckline_depth_front", "neckline_width",
              "tunic_length", "side_flare", "hem_curve", "slit_height"] + DARTED_STYLE,
    "shirt": ["armhole_curve_depth", "shoulder_slope", "neckline_depth_front", "neck_drop_back",
              "neckline_width", "design_ease", "shoulder_drop", "shirt_length", "shirt_sleeve_length",
              "collar_style", "collar_width", "cuff_depth", "pocket_count", "placket_width",
              "sleeve_hem_fullness", "sleeve_cap_gather", "cuff_elastic", "sleeveless", "neck_v"],
    "dress": ["armhole_curve_depth", "sleeve_cap_height", "shoulder_slope", "bust_dart_clearance",
              "bust_dart_angle", "back_waist_dart_intake", "neckline_depth_front", "neckline_width",
              "skirt_length", "skirt_flare", "skirt_hip_ease", "scallop_depth", "scallop_width",
              "dress_waist_seam", "dress_sleeve_length", "sleeveless", "neck_v"],
    "skirt": ["skirt_length", "skirt_flare", "skirt_hip_ease", "scallop_depth", "scallop_width",
              "waistband_height"],
}


def tunable_params(garment: str) -> list[str]:
    """Parameters of a garment the repair loop may hand to the LLM."""
    return [name for name in GARMENT_PARAMS.get(garment, list(PARAMETERS)) if PARAMETERS[name].tunable]


def resolve_params(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """Merge defaults and reject unknown, nonfinite or out-of-range values."""
    supplied = {} if overrides is None else dict(overrides)
    if supplied.keys() - PARAMETERS.keys():
        raise ValueError("Unknown drafting parameter")
    result = {name: float(spec.default) for name, spec in PARAMETERS.items()}
    for name, value in supplied.items():
        spec = PARAMETERS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number in {spec.unit}")
        if not math.isfinite(value) or not spec.min <= value <= spec.max:
            raise ValueError(f"{name} is outside its allowed range")
        if spec.integer and value != int(value):
            raise ValueError(f"{name} must be a whole number")
        result[name] = float(value)
    return result
