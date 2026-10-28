"""Women's woven skirt: straight to A-line, one waist dart per half panel, optional scallop hem.

Assumptions to tune with real samples (a pattern maker should review):
- Front and back half panels are identical apart from dart length (front
  90 mm, back 120 mm); both are cut on the fold. Closure is an invisible
  side zip, so no centre-back seam is drafted.
- Waist ease 10 mm. Waist-to-hip difference per quarter is split: half into
  the dart (capped at 30 mm intake), the rest into the side-seam curve.
- The waist line is drafted flat (no side raise); the hem curves up toward
  the side by 15 % of the flare so the side meets the hem near square.
- A scalloped hem is faced with a hem facing traced from the hem (60 mm
  deep), so the scallop edge gets a normal 10 mm seam, not a turned hem.
- No waistband (waistband_height 0) finishes the waist with a 40 mm bias
  facing strip of the same sewn length.
No seam allowances are included here.
"""

from __future__ import annotations

import math
from typing import Mapping

from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length

WAIST_EASE_MM = 10.0
DART_CAP_MM = 30.0
DART_LENGTH_MM = {"front": 90.0, "back": 120.0}
HEM_FACING_DEPTH_MM = 60.0
BIAS_FACING_MM = 40.0
WAISTBAND_OVERLAP_MM = 30.0


def _line(edge_id: str, start: Point, end: Point, partner: str | None = None, fold: bool = False) -> Edge:
    """Create a straight stitch-line edge without allowance."""
    return Edge(edge_id, Curve((start, end)), partner, needs_allowance=not fold, is_fold=fold)


def _hem_base(x: float, side_x: float, length: float, rise: float) -> float:
    """Hem height at x: centre at full length, rising toward the side."""
    return length - rise * (x / side_x) ** 2


def _hem_points(side_x: float, length: float, rise: float, depth: float, width: float) -> list[Point]:
    """Hem from side to centre; scallops bulge downward between cusps."""
    if depth <= 0:
        steps = max(8, int(side_x / 10))
        return [Point(side_x * (1 - i / steps), _hem_base(side_x * (1 - i / steps), side_x, length, rise))
                for i in range(steps + 1)]
    count = max(2, round(side_x / width))
    span = side_x / count
    points: list[Point] = []
    for k in range(count):
        x0 = side_x - k * span
        for i in range(0 if k == 0 else 1, 13):
            t = i / 12
            x = x0 - span * t
            points.append(Point(x, _hem_base(x, side_x, length, rise) + depth * math.sin(math.pi * t)))
    return points


def _half(name: str, m: WomenMeasurements, p: dict[str, float]) -> tuple[Panel, dict[str, float]]:
    """Draft one half panel on the fold with its waist dart."""
    waist_q = (m.waist + WAIST_EASE_MM) / 4
    hip_q = (m.hip + p["skirt_hip_ease"]) / 4
    diff = hip_q - waist_q
    if diff < 0:
        raise ValueError("Waist larger than hip: this skirt block needs a different shape")
    intake = min(diff / 2, DART_CAP_MM)
    side_top = Point(waist_q + intake, 0)
    dart_x = side_top.x * 0.45
    dart_len = DART_LENGTH_MM[name]
    length, hip_y = p["skirt_length"], m.waist_to_hip
    if not dart_len < hip_y < length:
        raise ValueError("Skirt length must pass the hip line and the dart must end above it")
    flare = p["skirt_flare"]
    rise = flare * 0.15
    hem_side = Point(hip_q + flare, _hem_base(hip_q + flare, hip_q + flare, length, rise))
    hip_pt = Point(hip_q, hip_y)
    # Side seam: waist curve to the hip, then straight to the hem (A-line).
    # The lower control point lies on the flare line so the seam is smooth at the hip.
    run_x, run_y = hem_side.x - hip_q, hem_side.y - hip_y
    run = math.hypot(run_x, run_y)
    pull = hip_y * 0.3
    c2 = Point(hip_q - run_x / run * pull, hip_y - run_y / run * pull)
    # c1 sits between the waist and c2 horizontally: one convex hip curve, no S.
    c1 = Point(side_top.x + (c2.x - side_top.x) * 0.75, hip_y * 0.3)
    upper = Curve.from_cubic_bezier(side_top, c1, c2, hip_pt)
    # Then straight from the hip to the hem side (the A-line flare).
    side_pts = upper.points + (hem_side,)
    leg_a, tip, leg_b = Point(dart_x - intake / 2, 0), Point(dart_x, dart_len), Point(dart_x + intake / 2, 0)
    hem = _hem_points(hem_side.x, length, rise, p["scallop_depth"], p["scallop_width"])
    hem_id = f"{name}.scallop" if p["scallop_depth"] > 0 else f"{name}.hem"
    other = "back" if name == "front" else "front"
    edges = [
        _line(f"{name}.waist_cf", Point(0, 0), leg_a, "waist.finish"),
        _line(f"{name}.dart_leg1", leg_a, tip, f"{name}.dart_leg2"),
        _line(f"{name}.dart_leg2", tip, leg_b, f"{name}.dart_leg1"),
        _line(f"{name}.waist_side", leg_b, side_top, "waist.finish"),
        Edge(f"{name}.side", Curve(side_pts), f"{other}.side"),
        Edge(hem_id, Curve(tuple(hem)), "hem_facing.scallop" if p["scallop_depth"] > 0 else None),
        _line(f"{name}.fold", hem[-1], Point(0, 0), fold=True),
    ]
    side_len = edge_length(edges[4])
    upper_len = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(upper.points, upper.points[1:]))
    notches = [Notch(edges[4].id, upper_len),  # hip line on the side seam
               Notch(edges[1].id, 0), Notch(edges[2].id, edge_length(edges[2]))]
    if name == "back":
        notches.append(Notch(edges[4].id, upper_len + 3))  # double notch marks the back
    info = {"intake": intake, "dart_len": dart_len, "side_len": side_len,
            "sewn_waist": dart_x - intake / 2 + side_top.x - (dart_x + intake / 2)}
    return Panel(name, edges, (Point(30, dart_len + 20), Point(30, length - 40)), notches,
                 cut_on_fold=True), info


def _hem_facing(front: Panel) -> Panel:
    """Facing traced from the scalloped hem, HEM_FACING_DEPTH_MM deep, on the fold."""
    hem = next(e for e in front.edges if e.id == "front.scallop").curve.points
    side = next(e for e in front.edges if e.id == "front.side").curve.points
    top_y_side = hem[0].y - HEM_FACING_DEPTH_MM
    # Walk the side seam up to the facing depth (straight lower part).
    a, b = side[-2], side[-1]
    t = (b.y - top_y_side) / (b.y - a.y)
    side_top = Point(b.x + (a.x - b.x) * t, top_y_side)
    centre_top = Point(0, hem[-1].y - HEM_FACING_DEPTH_MM)
    edges = [
        Edge("hem_facing.scallop", Curve(tuple(hem)), "front.scallop"),
        _line("hem_facing.fold", hem[-1], centre_top, fold=True),
        _line("hem_facing.top", centre_top, side_top),
        _line("hem_facing.side_r", side_top, hem[0]),
    ]
    return Panel("hem_facing", edges, (Point(20, centre_top.y + 5), Point(20, hem[-1].y - 5)), [],
                 cut_count=2, cut_on_fold=True)


def _waist_finish(sewn_total: float, height: float) -> Panel:
    """Waistband (folded, with overlap) or a bias facing strip when height is 0."""
    if height > 0:
        name, length, depth = "waistband", sewn_total + WAISTBAND_OVERLAP_MM, 2 * height
    else:
        name, length, depth = "waist_facing", sewn_total, BIAS_FACING_MM
    a, b, c, d = Point(0, 0), Point(length, 0), Point(length, depth), Point(0, depth)
    edges = [_line(f"{name}.attach", a, b, "waist.finish"), _line(f"{name}.end_r", b, c),
             _line(f"{name}.outer", c, d), _line(f"{name}.end_l", d, a)]
    # Side, CF and side marks along the sewn length (CF at the middle).
    notches = [Notch(f"{name}.attach", sewn_total / 4), Notch(f"{name}.attach", sewn_total / 2),
               Notch(f"{name}.attach", sewn_total * 3 / 4)]
    return Panel(name, edges, (Point(10, depth / 2), Point(length - 10, depth / 2)), notches)


def draft_skirt(measurements: WomenMeasurements | Mapping[str, float], fabric: str,
                params: Mapping[str, float] | None = None) -> Pattern:
    """Draft front, back, waist finish and (for scallops) the hem facing."""
    m = measurements if isinstance(measurements, WomenMeasurements) else WomenMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    p = resolve_params(params)
    front, f_info = _half("front", m, p)
    back, b_info = _half("back", m, p)
    sewn_total = 2 * (f_info["sewn_waist"] + b_info["sewn_waist"])
    panels = [front, back, _waist_finish(sewn_total, p["waistband_height"])]
    if p["scallop_depth"] > 0:
        panels.append(_hem_facing(front))
    size = next((name for name, values in WOMEN_SIZE_TABLE.items() if values == m), "custom")
    return Pattern("skirt", size, fabric, m.to_dict(), p, panels, {
        "status": "draft_unverified",
        "ease_mm": {"waist": WAIST_EASE_MM, "hip": p["skirt_hip_ease"]},
        "darts": {"skirt": {"front_intake_mm": f_info["intake"], "back_intake_mm": b_info["intake"]}},
        "waist_seam_mm": sewn_total,
        "waistband_overlap_mm": WAISTBAND_OVERLAP_MM if p["waistband_height"] > 0 else 0.0,
        "assumptions": ("Identical front/back halves on the fold, side zip, flat waist, dart takes "
                        "half the waist-hip difference (max 30 mm), faced scallop hem."),
    })
