"""Deterministic 3D design preview: a small mesh spec the browser renders with three.js.

Same contract as seamproof.flat: the preview is built by code from the
*drafted* categories (flat.drafted_features), never by a model, so it never
shows a detail the pattern cannot make. It is a visual preview only: no
cloth physics, draping or fit simulation, and no millimetres; the 2D pattern
and its verifier stay the source of truth.

Coordinates reuse the flat illustration units (flat.py), mapped to 3D with
y up and z towards the viewer (front of the garment):
    x3 = (x_flat - CX) * S,  y3 = -(y_flat - SHOULDER_Y) * S,  z3 = depth.
Cross-sections are ellipses (rx wide, rz deep). Depth ratios below are
illustration assumptions, chosen to read as a body on a dress form.

Spec shape (all lists of plain floats so it serialises to JSON):
    lofts:   [{id, trim, rings: [{y, rx, rz, scallop?, depth?}]}]  ellipse rings, top to bottom
    tubes:   [{id, trim, path: [[x, y, z]], radii: [r]}]          sleeves and cuffs
    panels:  [{id, points: [[x, y, z]]}]                           collar wings (triangle fan)
    patches: [{id, points: [[x, y, z]]}]                           pockets on the front
    buttons: [[x, y, z]]
    lines:   [{id, points: [[x, y, z]]}]                           placket, waist seam
    seams:   [{id, points: [[x, y, z]], closed?}]                  construction stitching:
             shoulder/side seams, hem stitch, darts (only the darts the block drafts)
    form:    rings of the neutral dress form, plus floor_y; the form is kept inside the
             garment so it never shows through the cloth
"""

from __future__ import annotations

import math
from typing import Any

from seamproof.flat import (
    ARMPIT_Y,
    BODY_HALF,
    CX,
    DRESS_HEM_Y,
    DRESS_WAIST_Y,
    SHOULDER_Y,
    SKIRT_HEM_Y,
    SLEEVE_LEN,
    TOP_HEM_Y,
    color_hex,
    hem_size,
    view_features,
)
from seamproof.style import StylePlan

S = 0.25  # flat unit -> preview unit
SPEC_VERSION = 2
# A flat sleeve's half width is half its circumference; as a near-round tube its radius is
# about half width * 2/pi. 0.7 (slightly over 2/pi) keeps a little ease around the arm.
SLEEVE_ROUND = 0.7
FORM_INSET = 0.88  # the form is at most this share of the garment's radius at any height
# Blocks that draft a side bust dart (bodice block); shirt and tshirt carry the bust in ease.
BUST_DART_KINDS = {"blouse", "tunic", "dress"}
NECK_HALF = 17.0
# A skirt is drawn from its waist; this offset puts that waist on the form's waist.
SKIRT_OFFSET_Y = DRESS_WAIST_Y - 40.0
ARM_DIR = (0.55, 0.835)  # same arm angle as the flat sketch (about 57 degrees below horizontal)
# Neutral dress form under every garment (flat units: y, half width, half depth).
FORM_RINGS = [(24.0, 9.0, 8.0), (34.0, 14.0, 11.0), (42.0, 36.0, 15.0), (84.0, 39.0, 24.0),
              (DRESS_WAIST_Y, 31.0, 20.0), (180.0, 41.0, 27.0), (196.0, 36.0, 24.0)]


def _y(y_flat: float) -> float:
    """Flat y (down) to preview y (up), shoulder line at 0."""
    return round(-(y_flat - SHOULDER_Y) * S, 3)


def _r(value: float) -> float:
    """Flat length to preview length."""
    return round(value * S, 3)


def _ring(y_flat: float, half: float, depth_ratio: float, **extra: float) -> dict[str, float]:
    """One elliptical cross-section."""
    return {"y": _y(y_flat), "rx": _r(half), "rz": _r(half * depth_ratio), **extra}


def _surface_z(rings: list[dict[str, float]], x: float, y: float, lift: float = 0.25) -> float:
    """Front surface depth of a loft at (x, y), interpolating between rings."""
    for upper, lower in zip(rings, rings[1:]):
        if lower["y"] <= y <= upper["y"]:
            t = 0.0 if upper["y"] == lower["y"] else (upper["y"] - y) / (upper["y"] - lower["y"])
            rx = upper["rx"] + (lower["rx"] - upper["rx"]) * t
            rz = upper["rz"] + (lower["rz"] - upper["rz"]) * t
            break
    else:
        ring = rings[0] if y > rings[0]["y"] else rings[-1]
        rx, rz = ring["rx"], ring["rz"]
    u = min(abs(x) / rx, 0.98) if rx else 0.0
    return round(rz * math.sqrt(1 - u * u) + lift, 3)


def _radii_at(rings: list[dict[str, float]], y: float) -> tuple[float, float] | None:
    """Loft half width and depth at preview height y (linear between rings); None outside it."""
    for upper, lower in zip(rings, rings[1:]):
        if lower["y"] <= y <= upper["y"]:
            t = 0.0 if upper["y"] == lower["y"] else (upper["y"] - y) / (upper["y"] - lower["y"])
            return (upper["rx"] + (lower["rx"] - upper["rx"]) * t, upper["rz"] + (lower["rz"] - upper["rz"]) * t)
    return None


def _on_front(rings: list[dict[str, float]], points: list[tuple[float, float]], lift: float = 0.25) -> list[list[float]]:
    """Flat (x, y) points laid onto the front of a loft, `lift` in front of its surface."""
    out = []
    for x, y in points:
        x3, y3 = _r(x - CX), _y(y)
        out.append([x3, y3, _surface_z(rings, x3, y3, lift)])
    return out


def _mirror3(points: list[list[float]]) -> list[list[float]]:
    """Reflect 3D points across the centre plane."""
    return [[-x, y, z] for x, y, z in points]


def _sleeve_profile(f: dict[str, Any], steps: int) -> list[float]:
    """Radius factor along the sleeve (0 = armhole, 1 = hem) for its style."""
    style = f.get("sleeve_style", "straight")
    out = []
    for i in range(steps + 1):
        t = i / steps
        if style == "bishop":  # widens towards the hem, then gathers into the cuff
            factor = 1.0 + 0.45 * math.sin(t * math.pi * 0.82) if t < 0.92 else 0.62
        elif style == "puff":  # gathered cap stands up, then settles to the arm
            factor = 1.0 + 0.4 * math.exp(-((t - 0.12) / 0.16) ** 2) - 0.15 * t
        else:  # straight sleeve tapers slightly to the hem
            factor = 1.0 - 0.22 * t
        out.append(factor)
    if f.get("elastic_cuff") and style != "straight":
        out[-1] = 0.48
    return out


def _sleeves(f: dict[str, Any], half: float, shoulder_x: float, shoulder_y: float) -> list[dict[str, Any]]:
    """Both sleeves (and cuffs) as tubes along the arm; empty when sleeveless."""
    length = SLEEVE_LEN.get(f.get("sleeve_length", "short"), 34.0)
    if length <= 0:
        return []
    dx, dy = ARM_DIR
    r0 = ((ARMPIT_Y - shoulder_y) / 2 + 2) * SLEEVE_ROUND
    # Set-in sleeve: the cap's top edge starts at the shoulder point, so the tube's cut start
    # sits inside the body instead of standing out as a flap above the shoulder.
    start = (shoulder_x - dy * r0 - dx * 3, shoulder_y + dx * r0 - dy * 3)
    steps = 8
    profile = _sleeve_profile(f, steps)
    path = [[_r(start[0] + dx * length * i / steps - CX), _y(start[1] + dy * length * i / steps), 0.0]
            for i in range(steps + 1)]
    radii = [_r(r0 * k) for k in profile]
    tubes = [{"id": "sleeve_r", "trim": False, "path": path, "radii": radii}]
    if f.get("cuff"):
        band = 9.0
        end = (start[0] + dx * length, start[1] + dy * length)
        cuff_r = radii[-1] * (1.04 if f.get("elastic_cuff") else 1.06)
        tubes.append({"id": "cuff_r", "trim": True,
                      "path": [path[-1], [_r(end[0] + dx * band - CX), _y(end[1] + dy * band), 0.0]],
                      "radii": [cuff_r, cuff_r]})
    left = [{**t, "id": t["id"].replace("_r", "_l"), "path": _mirror3(t["path"])} for t in tubes]
    return tubes + left


def _collar(f: dict[str, Any], body: list[dict[str, float]], neck_y: float) -> tuple[list, list]:
    """Collar stand (as a loft) and wing panels on the chest."""
    collar = f.get("collar", "none")
    lofts: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    if collar == "none":
        return lofts, panels
    stand_top = neck_y - (9.0 if collar == "band" else 3.0)
    lofts.append({"id": "collar_stand", "trim": True,
                  "rings": [_ring(stand_top, NECK_HALF - 1, 0.86), _ring(neck_y + 1, NECK_HALF + 1, 0.86)]})
    if collar == "shirt_collar":
        wing = [(CX + NECK_HALF, neck_y), (CX + NECK_HALF + 7, neck_y + 16), (CX + 2, neck_y + 26), (CX + 1, neck_y + 12)]
    elif collar == "peter_pan":
        wing = [(CX + NECK_HALF, neck_y)] + [
            (CX + 1.5 + (NECK_HALF + 6) * math.sin(a), neck_y + 4 + 18 * (1 - math.cos(a)) * 0.6)
            for a in [k * math.pi / 8 for k in range(8, -1, -1)]]
    else:
        return lofts, panels
    right = _on_front(body, wing, lift=1.1)  # the viewer's smoothed loft bulges past linear rings here
    panels += [{"id": "collar_r", "points": right}, {"id": "collar_l", "points": _mirror3(right)}]
    return lofts, panels


def _top(f: dict[str, Any], kind: str) -> dict[str, Any]:
    """Body loft, sleeves, collar, placket, buttons and pockets of a top."""
    fit = f.get("fit", "regular")
    half = BODY_HALF.get(fit, 50.0)
    hem_y = TOP_HEM_Y.get(f.get("length", "hip"), 178.0)
    hem_half = half + (16.0 if kind == "tunic" else 4.0 if fit != "boxy" else 0.0)
    waist_half = half - (6.0 if fit == "fitted" else 1.0 if fit == "regular" else 0.0)
    dropped = f.get("shoulder") == "dropped"
    shoulder_x = CX + half - 6 + (12 if dropped else 0) - (5 if f.get("sleeve_length") == "sleeveless" else 0)
    shoulder_y = SHOULDER_Y + (8 if dropped else 0)
    neck_y = SHOULDER_Y - 10
    depth = 0.68 if fit == "boxy" else 0.62
    body = [_ring(neck_y, NECK_HALF, 0.86), _ring(shoulder_y, shoulder_x - CX, 0.44),
            _ring(ARMPIT_Y, half, depth), _ring((ARMPIT_Y + hem_y) / 2 - 8, waist_half, depth - 0.02),
            _ring(hem_y, hem_half, depth)]
    collar_lofts, panels = _collar(f, body, neck_y)
    if kind == "tshirt":
        collar_lofts.append(_neckband(body, neck_y))
    spec: dict[str, Any] = {"lofts": [{"id": "body", "trim": False, "rings": body}] + collar_lofts,
                            "tubes": _sleeves(f, half, shoulder_x, shoulder_y), "panels": panels,
                            "patches": [], "buttons": [], "lines": [],
                            "seams": _body_seams(body, hem_y, scalloped=False)}
    if kind in BUST_DART_KINDS:
        spec["seams"] += _bust_darts(body, half)
    v_top = _v_neck(f, body, neck_y)
    if v_top:
        spec["lines"].append(v_top)
    if f.get("opening") == "button_front":
        top = neck_y + (44 if v_top else 14)
        spec["lines"].append({"id": "placket", "points": _on_front(body, [(CX + 5, top + i * (hem_y - top) / 10) for i in range(11)])})
        y = top + 8
        while y < hem_y - 6:
            spec["buttons"].append(_on_front(body, [(CX, y)])[0])
            y += 18
    for i in range(int(f.get("pockets", 0) or 0)):
        x = CX + (-1 if i == 0 else 1) * half * 0.5
        spec["patches"].append({"id": f"pocket_{i + 1}", "points": _on_front(
            body, [(x - 10, ARMPIT_Y - 4), (x + 10, ARMPIT_Y - 4), (x + 10, ARMPIT_Y + 16), (x, ARMPIT_Y + 21), (x - 10, ARMPIT_Y + 16)])})
    return spec


def _neckband(body: list[dict[str, float]], neck_y: float) -> dict[str, Any]:
    """Knit T-shirt neckband: a narrow band lying on the shoulder slope round the neck."""
    lower_y = _y(neck_y + 3.0)
    rx, rz = _radii_at(body, lower_y) or (body[0]["rx"], body[0]["rz"])
    return {"id": "neckband", "trim": True,
            "rings": [{"y": round(body[0]["y"] + 0.2, 3), "rx": round(body[0]["rx"] + 0.15, 3),
                       "rz": round(body[0]["rz"] + 0.15, 3)},
                      {"y": lower_y, "rx": round(rx + 0.3, 3), "rz": round(rz + 0.3, 3)}]}


def _hem_stitch(rings: list[dict[str, float]], y_flat: float, lift: float = 0.1) -> dict[str, Any]:
    """Closed topstitch ring round a loft at flat height y_flat."""
    y = _y(y_flat)
    rx, rz = _radii_at(rings, y) or (rings[-1]["rx"], rings[-1]["rz"])
    return {"id": "hem_stitch", "closed": True, "points": [
        [round((rx + lift) * math.sin(a), 3), y, round((rz + lift) * math.cos(a), 3)]
        for a in [k * math.pi / 24 for k in range(48)]]}


def _side_line(rings: list[dict[str, float]], top_y: float, bottom_y: float, sign: float,
               lift: float = 0.08) -> list[list[float]]:
    """Points down the side (z = 0) of a loft between two preview heights."""
    ys = [top_y + (bottom_y - top_y) * i / 10 for i in range(11)]
    out = []
    for y in ys:
        rx, _ = _radii_at(rings, y) or (rings[-1]["rx"], 0.0)
        out.append([round(sign * (rx + lift), 3), round(y, 3), 0.0])
    return out


def _body_seams(body: list[dict[str, float]], hem_y: float, scalloped: bool) -> list[dict[str, Any]]:
    """Shoulder and side seams on both sides plus a hem stitch (a faced scallop has none)."""
    seams = []
    for sign, side in ((1.0, "r"), (-1.0, "l")):
        seams.append({"id": f"shoulder_seam_{side}", "points": _side_line(body, body[0]["y"], body[1]["y"], sign)})
        seams.append({"id": f"side_seam_{side}", "points": _side_line(body, _y(ARMPIT_Y), body[-1]["y"], sign)})
    if not scalloped:
        seams.append(_hem_stitch(body, hem_y - 6.0))
    return seams


def _bust_darts(body: list[dict[str, float]], half: float) -> list[dict[str, Any]]:
    """Side bust darts of the bodice block: from the side seam towards the bust point."""
    base = [(CX + half * 0.97, ARMPIT_Y + 18), (CX + half * 0.42, ARMPIT_Y + 10)]
    steps = [(base[0][0] + (base[1][0] - base[0][0]) * i / 6, base[0][1] + (base[1][1] - base[0][1]) * i / 6)
             for i in range(7)]
    right = _on_front(body, steps, lift=0.1)
    return [{"id": "bust_dart_r", "points": right}, {"id": "bust_dart_l", "points": _mirror3(right)}]


def _vertical_darts(rings: list[dict[str, float]], x_flat: float, top_flat: float, bottom_flat: float,
                    prefix: str) -> list[dict[str, Any]]:
    """Vertical waist darts, front and back, mirrored left and right."""
    pts = [(CX + x_flat, top_flat + (bottom_flat - top_flat) * i / 6) for i in range(7)]
    front = _on_front(rings, pts, lift=0.1)
    back = [[x, y, -z] for x, y, z in front]
    return [{"id": f"{prefix}_front_r", "points": front}, {"id": f"{prefix}_front_l", "points": _mirror3(front)},
            {"id": f"{prefix}_back_r", "points": back}, {"id": f"{prefix}_back_l", "points": _mirror3(back)}]


def _v_neck(f: dict[str, Any], body: list[dict[str, float]], neck_y: float) -> dict[str, Any] | None:
    """Shaped neckline on the front: straight V lines, or a scoop or boat curve."""
    shape = f.get("neckline")
    if shape not in ("v", "scoop", "boat") or f.get("collar", "none") != "none":
        return None
    if shape != "v":
        half, drop = (24.0, 7.0) if shape == "boat" else (NECK_HALF, 26.0)
        curve = [(CX + half * (i / 6 - 1), neck_y + drop * (1 - (i / 6 - 1) ** 2)) for i in range(7)]
        curve += [(2 * CX - x, y) for x, y in reversed(curve[:-1])]
        return {"id": "neck_shape", "points": _on_front(body, curve, lift=0.4)}
    steps = [(CX - NECK_HALF + NECK_HALF * i / 6, neck_y + 44 * i / 6) for i in range(7)]
    steps += [(2 * CX - x, y) for x, y in reversed(steps[:-1])]
    return {"id": "neck_v", "points": _on_front(body, steps, lift=0.4)}


def _skirt(f: dict[str, Any]) -> dict[str, Any]:
    """Skirt loft hanging from the form's waist, with optional waistband and scallops."""
    o = SKIRT_OFFSET_Y
    hem_half, hem_y = hem_size(f, SKIRT_HEM_Y, 195.0)
    hem_y += o
    hip_half = min(54.0, hem_half)
    scallop = {"scallop": float(max(10, round(hem_half / 4))), "depth": _r(8)} if f.get("hem") == "scalloped" else {}
    rings = [_ring(40 + o, 40, 0.7), _ring(92 + o, hip_half, 0.72), _ring(hem_y, hem_half, 0.78, **scallop)]
    lofts = [{"id": "skirt", "trim": False, "rings": rings}]
    if f.get("waistband"):
        lofts.append({"id": "waistband", "trim": True, "rings": [_ring(31 + o, 40.5, 0.71), _ring(41 + o, 40.5, 0.71)]})
    seams = [{"id": f"side_seam_{side}", "points": _side_line(rings, rings[0]["y"], rings[-1]["y"], sign)}
             for sign, side in ((1.0, "r"), (-1.0, "l"))]
    if not scallop:
        seams.append(_hem_stitch(rings, hem_y - 6.0))
    seams += _vertical_darts(rings, 20.0, 42 + o, 66 + o, "waist_dart")  # one dart per skirt half
    return {"lofts": lofts, "tubes": [], "panels": [], "patches": [], "buttons": [], "lines": [], "seams": seams}


def _dress(f: dict[str, Any]) -> dict[str, Any]:
    """One loft from neck to hem; a waist seam line when the dress has one."""
    seam = bool(f.get("waist_seam"))
    half, neck_y = 44.0, SHOULDER_Y - 10
    hem_half, hem_y = hem_size(f, DRESS_HEM_Y, 252.0)
    waist_half = 38.0 if seam else 45.0
    scallop = {"scallop": float(max(10, round(hem_half / 4))), "depth": _r(8)} if f.get("hem") == "scalloped" else {}
    body = [_ring(neck_y, NECK_HALF, 0.86), _ring(SHOULDER_Y, half - 6, 0.44), _ring(ARMPIT_Y, half, 0.62),
            _ring(DRESS_WAIST_Y, waist_half, 0.66), _ring(DRESS_WAIST_Y + 52, min(54.0, hem_half), 0.72),
            _ring(hem_y, hem_half, 0.78, **scallop)]
    shoulder_x = CX + half - 6 - (5 if f.get("sleeve_length") == "sleeveless" else 0)
    spec = {"lofts": [{"id": "body", "trim": False, "rings": body}], "tubes": _sleeves(f, half, shoulder_x, SHOULDER_Y),
            "panels": [], "patches": [], "buttons": [], "lines": [],
            "seams": _body_seams(body, hem_y, scalloped=bool(scallop)) + _bust_darts(body, half)}
    if seam:  # bodice waist darts up towards the bust, skirt darts down from the seam
        spec["seams"] += _vertical_darts(body, half * 0.42, DRESS_WAIST_Y, ARMPIT_Y + 22, "bodice_dart")
        spec["seams"] += _vertical_darts(body, half * 0.42, DRESS_WAIST_Y, DRESS_WAIST_Y + 24, "skirt_dart")
    v_top = _v_neck(f, body, neck_y)
    if v_top:
        spec["lines"].append(v_top)
    if seam:
        ring = body[3]
        spec["lines"].append({"id": "waist_seam", "points": [
            [round(ring["rx"] * math.cos(a), 3), ring["y"], round(ring["rz"] * math.sin(a) + 0.15, 3)]
            for a in [k * math.pi / 24 for k in range(25)]]})
    return spec


def preview_spec(part: str, kind: str, features: dict[str, Any], color: str | None = None) -> dict[str, Any]:
    """3D preview spec for already-drafted features of one garment piece."""
    spec = _skirt(features) if part == "bottom" else _dress(features) if part == "dress" else _top(features, kind)
    lowest = min(r["y"] for loft in spec["lofts"] for r in loft["rings"])
    spec.update(version=SPEC_VERSION, part=part, color=color_hex(color),
                form={"rings": _form_rings(spec["lofts"][0]["rings"]), "floor_y": round(lowest - 6.0, 3)})
    return spec


def _form_rings(garment: list[dict[str, float]]) -> list[dict[str, float]]:
    """Dress form rings, shrunk where the garment is close so the form never pokes through."""
    out = []
    for y, rx, rz in FORM_RINGS:
        ring = _ring(y, rx, rz / rx)
        cover = _radii_at(garment, ring["y"])
        if cover:
            ring["rx"] = round(min(ring["rx"], cover[0] * FORM_INSET), 3)
            ring["rz"] = round(min(ring["rz"], cover[1] * FORM_INSET), 3)
        out.append(ring)
    return out


def plan_preview(plan: StylePlan, color: str | None = None) -> dict[str, Any]:
    """Preview of what the plan will draft (raw features when the plan is undraftable)."""
    return preview_spec(plan.part, plan.garment or plan.kind, view_features(plan), color)
