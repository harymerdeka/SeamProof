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
    form:    rings of the neutral dress form, plus floor_y
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
    SKIRT_HEM_HALF,
    SKIRT_HEM_Y,
    SLEEVE_LEN,
    TOP_HEM_Y,
    color_hex,
    drafted_features,
)
from seamproof.style import StylePlan

S = 0.25  # flat unit -> preview unit
SPEC_VERSION = 1
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
    start = ((shoulder_x + CX + half) / 2 - 2, (shoulder_y + ARMPIT_Y) / 2)
    r0 = (ARMPIT_Y - shoulder_y) / 2 + 2
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
    shoulder_x = CX + half - 6 + (12 if dropped else 0)
    shoulder_y = SHOULDER_Y + (8 if dropped else 0)
    neck_y = SHOULDER_Y - 10
    depth = 0.68 if fit == "boxy" else 0.62
    body = [_ring(neck_y, NECK_HALF, 0.86), _ring(shoulder_y, shoulder_x - CX, 0.44),
            _ring(ARMPIT_Y, half, depth), _ring((ARMPIT_Y + hem_y) / 2 - 8, waist_half, depth - 0.02),
            _ring(hem_y, hem_half, depth)]
    collar_lofts, panels = _collar(f, body, neck_y)
    spec: dict[str, Any] = {"lofts": [{"id": "body", "trim": False, "rings": body}] + collar_lofts,
                            "tubes": _sleeves(f, half, shoulder_x, shoulder_y), "panels": panels,
                            "patches": [], "buttons": [], "lines": []}
    if f.get("opening") == "button_front":
        top = neck_y + 14
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


def _skirt(f: dict[str, Any]) -> dict[str, Any]:
    """Skirt loft hanging from the form's waist, with optional waistband and scallops."""
    o = SKIRT_OFFSET_Y
    hem_y = SKIRT_HEM_Y.get(f.get("length", "knee"), 195.0) + o
    hem_half = SKIRT_HEM_HALF.get(f.get("silhouette", "a_line"), 70.0)
    hip_half = min(54.0, hem_half)
    scallop = {"scallop": float(max(10, round(hem_half / 4))), "depth": _r(8)} if f.get("hem") == "scalloped" else {}
    rings = [_ring(40 + o, 40, 0.7), _ring(92 + o, hip_half, 0.72), _ring(hem_y, hem_half, 0.78, **scallop)]
    lofts = [{"id": "skirt", "trim": False, "rings": rings}]
    if f.get("waistband"):
        lofts.append({"id": "waistband", "trim": True, "rings": [_ring(31 + o, 40.5, 0.71), _ring(41 + o, 40.5, 0.71)]})
    return {"lofts": lofts, "tubes": [], "panels": [], "patches": [], "buttons": [], "lines": []}


def _dress(f: dict[str, Any]) -> dict[str, Any]:
    """One loft from neck to hem; a waist seam line when the dress has one."""
    seam = bool(f.get("waist_seam"))
    half, neck_y = 44.0, SHOULDER_Y - 10
    hem_y = DRESS_HEM_Y.get(f.get("length", "knee"), 252.0)
    hem_half = SKIRT_HEM_HALF.get(f.get("silhouette", "a_line"), 70.0)
    waist_half = 38.0 if seam else 45.0
    scallop = {"scallop": float(max(10, round(hem_half / 4))), "depth": _r(8)} if f.get("hem") == "scalloped" else {}
    body = [_ring(neck_y, NECK_HALF, 0.86), _ring(SHOULDER_Y, half - 6, 0.44), _ring(ARMPIT_Y, half, 0.62),
            _ring(DRESS_WAIST_Y, waist_half, 0.66), _ring(DRESS_WAIST_Y + 52, min(54.0, hem_half), 0.72),
            _ring(hem_y, hem_half, 0.78, **scallop)]
    spec = {"lofts": [{"id": "body", "trim": False, "rings": body}], "tubes": _sleeves(f, half, CX + half - 6, SHOULDER_Y),
            "panels": [], "patches": [], "buttons": [], "lines": []}
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
                form={"rings": [_ring(y, rx, rz / rx) for y, rx, rz in FORM_RINGS],
                      "floor_y": round(lowest - 6.0, 3)})
    return spec


def plan_preview(plan: StylePlan, color: str | None = None) -> dict[str, Any]:
    """Preview of what the plan will draft (raw features when the plan is undraftable)."""
    return preview_spec(plan.part, plan.garment or plan.kind, drafted_features(plan), color)
