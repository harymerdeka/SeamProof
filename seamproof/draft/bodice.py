"""Provisional women's bodice block with bust and waist darts, in mm.

Assumptions to tune with samples: equal front/back width allocation; bodice
and blouse hems levelled to the front hip line; underarm hangs straight from
the bust point width; bust dart intake is (bust - high_bust) / 4 per half
front; back waist dart is vertical, 140 mm long, centred mid-half-back; sleeve
bicep ease 60 mm and wrist ease 20 mm; neck facing 40 mm wide; tunic hem is
levelled to front waist plus tunic_length. Dart legs are trued equal by
construction; closing a dart rotates the upper panel about the dart tip.
Seam-length compatibility is not certified by drafting; a verifier must gate
production use. No seam allowances are included here.
"""

import math
from typing import Mapping

from seamproof.draft.finish import v_line
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length

BACK_NECK_DROP = 20.0
FACING_WIDTH = 40.0
# Back truing stops once the back half is this close to a quarter bust (a
# higher sleeveless armhole converges slower than the sleeved block).
TRUE_PASSES = 8
TRUE_GAP_MM = 0.05
SLEEVE_BICEP_EASE = 60.0
SLEEVE_WRIST_EASE = 20.0
BACK_DART_LENGTH = 140.0
BLOUSE_HIP_EXTENSION = 20.0
BLOUSE_HEM_EXTRA = 10.0

# Edges above the front bust dart; closing the dart rotates these about its tip.
_UPPER_FRONT_IDS = {"front.neckline", "front.shoulder", "front.armhole", "front.side_upper"}
_FRONT_LEG_IDS = {"front.dart_leg1", "front.dart_leg2"}


def _line(edge_id: str, start: Point, end: Point, partner: str | None = None, fold: bool = False) -> Edge:
    """Create a straight stitch-line edge without allowance."""
    return Edge(edge_id, Curve((start, end)), partner, needs_allowance=not fold, is_fold=fold)


def _rotate(point: Point, centre: Point, angle: float) -> Point:
    """Rotate a point about a centre by an angle in radians."""
    dx, dy = point.x - centre.x, point.y - centre.y
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return Point(centre.x + dx * cos_a - dy * sin_a, centre.y + dx * sin_a + dy * cos_a)


def _shoulder_tip(snp_x: float, slope: float, length: float) -> Point:
    """Place the shoulder tip at an exact seam length from the neck point."""
    if length <= slope:
        raise ValueError("Shoulder length must exceed the shoulder slope")
    return Point(snp_x + math.sqrt(length * length - slope * slope), slope)


def _front_dart(joint: Point, apex: Point, half_intake: float, clearance: float, angle_deg: float) -> tuple[Point, Point, Point]:
    """Locate a side bust dart: base ends on the side line, tip near the apex.

    The base is vertical and symmetric about bust height; the tip sits on the
    apex ray (rotated by angle_deg, positive down) clearance mm short of the
    apex. The lower base end is trued so both legs match the upper leg.
    """
    upper = Point(joint.x, joint.y - half_intake)
    distance = math.hypot(apex.x - joint.x, apex.y - joint.y)
    if distance <= clearance:
        raise ValueError("Bust dart clearance exceeds the apex distance")
    theta = math.atan2(apex.y - joint.y, apex.x - joint.x) + math.radians(angle_deg)
    reach = distance - clearance
    tip = Point(joint.x + reach * math.cos(theta), joint.y + reach * math.sin(theta))
    upper_leg = math.hypot(tip.x - upper.x, tip.y - upper.y)
    lateral = abs(joint.x - tip.x)
    if upper_leg < lateral:
        raise ValueError("Bust dart geometry cannot close its legs")
    lower = Point(joint.x, tip.y + math.sqrt(upper_leg * upper_leg - lateral * lateral))
    return upper, tip, lower


def _torso_front(
    m: WomenMeasurements, ease: float, p: dict[str, float],
    hem_y: float, hem_half: float, slit_mm: float, hem_curve: float, facing: bool,
) -> tuple[Panel, dict[str, Point]]:
    """Draft a half-front with a side bust dart; hem and slit vary by garment."""
    depth = m.armhole_depth
    apex = Point(m.apex_distance / 2, m.apex_height)
    # The dart base sits vertically on the bust joint, so the sewn side join
    # lands exactly on the eased quarter bust without any correction pass.
    joint_x = (m.bust + ease) / 4
    joint = Point(joint_x, m.apex_height)
    intake = (m.bust - m.high_bust) / 4
    upper, tip, lower = _front_dart(joint, apex, intake / 2, p["bust_dart_clearance"], p["bust_dart_angle"])
    if not depth < upper.y < m.front_length or not depth < lower.y < m.front_length:
        raise ValueError("Bust dart must sit between armhole and waist")
    snp_x = p["neckline_width"] / 2
    neck = Point(0, p["neckline_depth_front"])
    snp = Point(snp_x, 0)
    shoulder = _shoulder_tip(snp_x, p["shoulder_slope"], m.shoulder_length)
    underarm = Point(joint_x, depth)
    if not 0 < neck.y < depth or not snp.x < shoulder.x < underarm.x:
        raise ValueError("Front measurements do not fit this bodice block")
    if shoulder.y >= depth:
        raise ValueError("Shoulder drop must be less than armhole depth")
    waist = Point((m.waist + ease) / 4, m.front_length)
    hip = Point((m.hip + ease) / 4, m.front_length + m.waist_to_hip)
    if not waist.y > lower.y or not waist.x < joint_x or not waist.x < hip.x:
        raise ValueError("Front waist shaping does not fit this bodice block")
    corner = Point(hem_half, hem_y)
    if not corner.y >= hip.y or not corner.x >= hip.x:
        raise ValueError("Hem must fall at or below the hip, at or outside its width")
    # neck_v: straight V to centre front.
    neck_curve = (v_line(neck, snp) if p["neck_v"] >= 0.5 else
                  Curve.from_cubic_bezier(neck, Point(snp.x * 0.55, neck.y), Point(snp.x, neck.y * 0.55), snp))
    neckline = Edge("front.neckline", neck_curve, "facing.neck" if facing else None)
    armhole = Edge("front.armhole", Curve.from_cubic_bezier(
        shoulder,
        Point(shoulder.x - p["armhole_curve_depth"], shoulder.y + (depth - shoulder.y) * 0.55),
        Point(underarm.x - p["armhole_curve_depth"], depth), underarm,
    ), "sleeve.cap")
    side_upper = Edge("front.side_upper", Curve((underarm, upper)), "back.side_upper")
    leg1 = Edge("front.dart_leg1", Curve((upper, tip)), "front.dart_leg2")
    leg2 = Edge("front.dart_leg2", Curve((tip, lower)), "front.dart_leg1")
    low_points = [lower, waist, hip]
    slit: Edge | None = None
    if slit_mm > 0:
        span = math.hypot(corner.x - hip.x, corner.y - hip.y)
        if slit_mm >= span:
            raise ValueError("Side slit must be shorter than the hip-to-hem seam")
        slit_top = Point(corner.x + (hip.x - corner.x) * slit_mm / span,
                         corner.y + (hip.y - corner.y) * slit_mm / span)
        low_points.append(slit_top)
        slit = Edge("front.slit", Curve((slit_top, corner)))
    elif corner != hip:
        low_points.append(corner)
    side_lower = Edge("front.side_lower", Curve(tuple(low_points)), "back.side_low")
    bottom = Point(0, hem_y)
    if hem_curve > 0:
        mid = Point(corner.x / 2, hem_y + hem_curve)
        hem = Edge("front.hem", Curve.from_cubic_bezier(corner, mid, mid, bottom))
    else:
        hem = Edge("front.hem", Curve((corner, bottom)))
    edges = [neckline, _line("front.shoulder", snp, shoulder, "back.shoulder"), armhole,
             side_upper, leg1, leg2, side_lower]
    if slit is not None:
        edges.append(slit)
    edges += [hem, _line("front.fold", bottom, neck, fold=True)]
    arm_length = edge_length(armhole)
    leg_length = edge_length(leg1)
    notches = [Notch(armhole.id, 0), Notch(armhole.id, arm_length * 0.6),
               Notch(leg1.id, 0), Notch(leg2.id, leg_length)]
    if slit is not None:
        notches.append(Notch(slit.id, 0))
    panel = Panel("front", edges, (Point(30, depth), Point(30, hem_y - 20)), notches, cut_on_fold=True)
    return panel, {"upper": upper, "tip": tip, "lower": lower, "leg_length": leg_length,
                   "armhole_length": arm_length, "intake": intake}


def _torso_back(
    m: WomenMeasurements, ease: float, p: dict[str, float],
    hem_y: float, hem_half: float, slit_mm: float, hem_curve: float, opening: str,
    joint_extra: float = 0.0,
) -> tuple[Panel, dict[str, float]]:
    """Draft a half-back with an optional vertical waist dart."""
    depth = m.armhole_depth
    snp_x = p["neckline_width"] / 2
    neck = Point(0, BACK_NECK_DROP)
    snp = Point(snp_x, 0)
    shoulder = _shoulder_tip(snp_x, p["shoulder_slope"], m.shoulder_length)
    underarm = Point((m.bust + ease) / 4 + joint_extra, depth)
    if not 0 < neck.y < depth or not snp.x < shoulder.x < underarm.x:
        raise ValueError("Back measurements do not fit this bodice block")
    if shoulder.y >= depth:
        raise ValueError("Shoulder drop must be less than armhole depth")
    waist_y = m.back_length
    dart_intake = p["back_waist_dart_intake"]
    corner = Point(hem_half, hem_y)
    hip = Point((m.hip + ease) / 4, waist_y + m.waist_to_hip)
    if not corner.y >= hip.y or not corner.x >= hip.x:
        raise ValueError("Hem must fall at or below the hip, at or outside its width")
    neckline = Edge("back.neckline", Curve.from_cubic_bezier(
        neck, Point(snp.x * 0.55, neck.y), Point(snp.x, neck.y * 0.55), snp,
    ))
    armhole = Edge("back.armhole", Curve.from_cubic_bezier(
        shoulder,
        Point(shoulder.x - p["armhole_curve_depth"], shoulder.y + (depth - shoulder.y) * 0.55),
        Point(underarm.x - p["armhole_curve_depth"], depth), underarm,
    ), "sleeve.cap")
    # The waist dart intake is trued out of the outline (a transverse wedge
    # cannot sit in a simple closed loop) and recorded as drill markings.
    side_waist = Point((m.waist + ease) / 4 - dart_intake / 2, waist_y)
    side_upper = Edge("back.side_upper", Curve((underarm, side_waist)), "front.side_upper")
    side_low = Edge("back.side_low", Curve((side_waist, hip)), "front.side_lower")
    dart_tip_y = waist_y - BACK_DART_LENGTH
    if dart_tip_y <= depth + 20:
        raise ValueError("Back waist dart runs into the armhole")
    dart_leg = math.hypot(dart_intake / 2, BACK_DART_LENGTH) if dart_intake > 0 else 0.0
    drills = ([[side_waist.x / 2 - dart_intake / 2, waist_y],
               [side_waist.x / 2 + dart_intake / 2, waist_y],
               [side_waist.x / 2, dart_tip_y]] if dart_intake > 0 else [])
    slit: Edge | None = None
    if slit_mm > 0:
        span = math.hypot(corner.x - hip.x, corner.y - hip.y)
        if slit_mm >= span:
            raise ValueError("Side slit must be shorter than the hip-to-hem seam")
        slit_top = Point(corner.x + (hip.x - corner.x) * slit_mm / span,
                         corner.y + (hip.y - corner.y) * slit_mm / span)
        slit = Edge("back.slit", Curve((slit_top, corner)))
        side_lower: Edge | None = Edge("back.side_lower", Curve((hip, slit_top)), "front.side_lower")
    elif corner != hip:
        side_lower = Edge("back.side_lower", Curve((hip, corner)), "front.side_lower")
    else:
        side_lower = None  # Bodice hem sits exactly on the hip line.
    bottom = Point(0, hem_y)
    if hem_curve > 0:
        mid = Point(corner.x / 2, hem_y + hem_curve)
        hem = Edge("back.hem", Curve.from_cubic_bezier(corner, mid, mid, bottom))
    else:
        hem = Edge("back.hem", Curve((corner, bottom)))
    if opening == "centre_back_zip":
        closer: Edge = Edge("back.cb", Curve((bottom, neck)), "back.cb")
        cut_count, on_fold = 2, False
    else:
        closer = _line("back.fold", bottom, neck, fold=True)
        cut_count, on_fold = 1, True
    edges = [neckline, _line("back.shoulder", snp, shoulder, "front.shoulder"), armhole,
             side_upper, side_low]
    if side_lower is not None:
        edges.append(side_lower)
    if slit is not None:
        edges.append(slit)
    edges += [hem, closer]
    arm_length = edge_length(armhole)
    notches = [Notch(armhole.id, 0)]
    notches.extend(Notch(armhole.id, arm_length * 0.55 + offset) for offset in (-3, 3))
    if slit is not None:
        notches.append(Notch(slit.id, 0))
    panel = Panel("back", edges, (Point(30, depth), Point(30, hem_y - 20)), notches,
                  cut_count=cut_count, cut_on_fold=on_fold)
    return panel, {"armhole_length": arm_length, "dart_intake": dart_intake,
                   "dart_leg_mm": dart_leg, "drills": drills}


def _sleeve(m: WomenMeasurements, p: dict[str, float], armhole_length: float) -> tuple[Panel, float]:
    """Draft a tapered one-piece sleeve; cap fullness comes from the block.

    Assumption: cap-curve control multipliers below are provisional; the cap
    height grades from armhole depth (1.4x - 197 tuned so default ease lands
    mid-range for woven across sizes) and the parameter adjusts around that
    base. Ease is measured against the bodice armhole and must land in the
    fabric range.
    """
    width = (m.bicep + SLEEVE_BICEP_EASE) / 2
    height = 1.4 * m.armhole_depth - 197 + p["sleeve_cap_height"] - 100
    length = m.sleeve_length
    wrist_half = (m.wrist + SLEEVE_WRIST_EASE) / 2
    if width <= 0 or length <= height or wrist_half <= 0:
        raise ValueError("Sleeve length must exceed cap height and widths must be positive")
    front, crown, back = Point(-width, height), Point(0, 0), Point(width, height)
    front_cap = Curve.from_cubic_bezier(front, Point(-width * 0.8, height * 0.5), Point(-width * 0.45, 0), crown)
    back_cap = Curve.from_cubic_bezier(crown, Point(width * 0.55, 0), Point(width * 0.85, height * 0.5), back)
    cap = Edge("sleeve.cap", Curve(front_cap.points + back_cap.points[1:]), "body.armholes")
    front_length = edge_length(Edge("front_cap", front_cap))
    back_length = edge_length(Edge("back_cap", back_cap))
    left_hem, right_hem = Point(-wrist_half, length), Point(wrist_half, length)
    edges = [cap, _line("sleeve.underarm_back", back, right_hem, "sleeve.underarm_front"),
             _line("sleeve.hem", right_hem, left_hem),
             _line("sleeve.underarm_front", left_hem, front, "sleeve.underarm_back")]
    notches = [Notch(cap.id, front_length), Notch(cap.id, front_length * 0.4)]
    notches.extend(Notch(cap.id, front_length + back_length * 0.55 + offset) for offset in (-3, 3))
    panel = Panel("sleeve", edges, (Point(0, 20), Point(0, length - 20)), notches, cut_count=2)
    return panel, edge_length(cap) - armhole_length


def _facing(front: Panel) -> Panel:
    """Draft a neck facing strip following the front neckline outward."""
    neck = next(edge for edge in front.edges if edge.id == "front.neckline")
    # Every stitch point is kept: sparser sampling shortens the facing neck
    # and kinks its outward offset on tight necklines.
    samples = neck.curve.points
    outer: list[Point] = []
    for before, point, after in zip((samples[0], *samples), samples, (*samples[1:], samples[-1])):
        tx, ty = after.x - before.x, after.y - before.y
        hypo = math.hypot(tx, ty) or 1.0
        # Outward is up-left of the neckline direction (y grows downward).
        outer.append(Point(point.x + ty / hypo * FACING_WIDTH, point.y - tx / hypo * FACING_WIDTH))
    # A V neck's offset crosses centre front: clip the outer edge back onto the CF line.
    while len(outer) > 2 and outer[1].x <= 0 < -outer[0].x - 1e-6:
        outer.pop(0)
    if outer[0].x < -1e-6:
        a, b = outer[0], outer[1]
        t = -a.x / (b.x - a.x)
        outer[0] = Point(0.0, a.y + (b.y - a.y) * t)
    neck_edge = Edge("facing.neck", Curve(tuple(samples)), "front.neckline")
    shoulder_end = Edge("facing.end_shoulder", Curve((samples[-1], outer[-1])))
    outer_edge = Edge("facing.outer", Curve(tuple(reversed(outer))))
    cf_end = Edge("facing.end_cf", Curve((outer[0], samples[0])))
    length = edge_length(neck_edge)
    mid_x = sum(point.x for point in (*samples, *outer)) / (2 * len(samples))
    return Panel("facing", [neck_edge, shoulder_end, outer_edge, cf_end],
                 (Point(mid_x, 20), Point(mid_x, 60)),
                 [Notch(neck_edge.id, 0), Notch(neck_edge.id, length)])


def close_dart(front: Panel, tip: Point) -> Panel:
    """Return the front with its bust dart sewn shut, for sewn-state checks.

    Edges above the dart rotate about the tip until the upper leg lands on
    the lower leg; both legs then leave the outline and the side seam meets
    at one point. The input panel is not modified: every kept edge is
    rebuilt so the fold-top fix cannot leak back into the draft.
    """
    legs = {edge.id: edge for edge in front.edges if edge.id in _FRONT_LEG_IDS}
    upper_end = legs["front.dart_leg1"].curve.points[0]
    lower_end = legs["front.dart_leg2"].curve.points[-1]
    before = math.atan2(upper_end.y - tip.y, upper_end.x - tip.x)
    after = math.atan2(lower_end.y - tip.y, lower_end.x - tip.x)
    rebuilt: list[Edge] = []
    for edge in front.edges:
        if edge.id in _FRONT_LEG_IDS:
            continue
        if edge.id in _UPPER_FRONT_IDS:
            # Notches above the dart keep arc positions; lengths barely change.
            # The dart base snaps exactly shut; float rotation alone would not.
            turned = Curve(tuple(lower_end if point == upper_end else _rotate(point, tip, after - before)
                                 for point in edge.curve.points))
            rebuilt.append(Edge(edge.id, turned, edge.seam_partner_id, edge.needs_allowance,
                                edge.allowance_mm, edge.is_fold))
        else:
            rebuilt.append(Edge(edge.id, Curve(tuple(edge.curve.points)), edge.seam_partner_id,
                                edge.needs_allowance, edge.allowance_mm, edge.is_fold))
    kept_notches = [n for n in front.notches
                    if n.edge_id not in _FRONT_LEG_IDS and n.edge_id in {e.id for e in rebuilt}]
    # The fold stays on the fabric but its top follows the rotated neckline.
    if rebuilt[-1].id == "front.fold":
        fold = rebuilt[-1]
        fold.curve = Curve((fold.curve.points[0], rebuilt[0].curve.points[0]))
    return Panel(front.name, rebuilt, front.grainline, kept_notches,
                 front.cut_count, front.cut_on_fold, front.cut_line)


def anchor_fold(panel: Panel, height_y: float) -> Panel:
    """Translate a closed front in x so its fold sits at zero at bust height.

    Closing the dart swings the fold top off grain; pressing the fold back to
    centre front removes that rigid shift from sewn-state measurements while
    preserving every shape the verifier checks.
    """
    fold = next(edge for edge in panel.edges if edge.id == "front.fold")
    (x0, y0), (x1, y1) = ((p.x, p.y) for p in (fold.curve.points[0], fold.curve.points[-1]))
    fold_x = x0 if y1 == y0 else x0 + (x1 - x0) * (height_y - y0) / (y1 - y0)
    shifted = [Edge(edge.id, Curve(tuple(Point(p.x - fold_x, p.y) for p in edge.curve.points)),
                    edge.seam_partner_id, edge.needs_allowance, edge.allowance_mm, edge.is_fold)
               for edge in panel.edges]
    return Panel(panel.name, shifted, panel.grainline, list(panel.notches),
                 panel.cut_count, panel.cut_on_fold, panel.cut_line)


def _assemble(
    garment: str, m: WomenMeasurements, fabric: str, p: dict[str, float],
    hem_y: float, hem_half: float, slit_mm: float, hem_curve: float,
    facing_piece: bool, opening: str,
) -> Pattern:
    """Draft front, back and sleeve over a shared torso and record metadata.

    The back bust width is trued in the sewn state: up to TRUE_PASSES passes
    widen the back underarm until the back half measures a quarter of the eased
    bust. The front needs no pass: its dart base sits vertically on the bust
    joint, so the sewn side join lands on the quarter bust by construction.
    """
    from shapely.geometry import LineString

    from seamproof.pattern import panel_polygon

    ease = FABRIC_EASE[fabric].default_ease_mm
    quarter = (m.bust + ease) / 4
    line = LineString([(-1000, m.apex_height), (1000, m.apex_height)])
    front, dart = _torso_front(m, ease, p, hem_y, hem_half, slit_mm, hem_curve, facing_piece)
    extra_b = 0.0
    for _ in range(TRUE_PASSES):
        back, back_info = _torso_back(m, ease, p, hem_y, hem_half, slit_mm, hem_curve,
                                      opening, extra_b)
        gap = quarter - panel_polygon(back).intersection(line).length
        if abs(gap) <= TRUE_GAP_MM:
            break
        extra_b += gap
    front_join = (dart["upper"].x + dart["lower"].x) / 2
    back_half = panel_polygon(back).intersection(line).length
    sleeve, cap_ease = _sleeve(m, p, dart["armhole_length"] + back_info["armhole_length"])
    panels = [front, back, sleeve]
    if facing_piece:
        panels.append(_facing(front))
    size = next((name for name, values in WOMEN_SIZE_TABLE.items() if values == m), "custom")
    ease = FABRIC_EASE[fabric].default_ease_mm
    return Pattern(garment, size, fabric, m.to_dict(), p, panels, {
        "status": "draft_unverified", "ease_mm": {name: ease for name in ("bust", "waist", "hip")},
        "bust_line_y_mm": m.apex_height,
        "seam_groups": {
            "body.armholes": [{"edge_id": "front.armhole", "reverse": True}, {"edge_id": "back.armhole", "reverse": False}],
        },
        "darts": {
            "front_bust": {"intake_mm": dart["intake"], "leg_mm": dart["leg_length"],
                           "tip": [dart["tip"].x, dart["tip"].y]},
            "back_waist": {"intake_mm": back_info["dart_intake"], "leg_mm": back_info["dart_leg_mm"],
                           "drills": back_info["drills"]},
        },
        "sleeve_cap_ease_mm": cap_ease,
        # Sewn-state bust: the front side join plus the measured back half.
        "sewn_bust_mm": {"front_join": front_join, "back_half": back_half,
                         "total": 2 * (front_join + back_half)},
        "assumptions": "Provisional block and ease; tune with real samples.",
    }), {"front": front, "tip": dart["tip"], "back": back}


def draft_bodice(
    measurements: WomenMeasurements | Mapping[str, float],
    fabric: str,
    params: Mapping[str, float] | None = None,
) -> Pattern:
    """Draft a hip-length bodice block: front, back and sleeve, no facing."""
    m = measurements if isinstance(measurements, WomenMeasurements) else WomenMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    p = resolve_params(params)
    hip_y = m.front_length + m.waist_to_hip
    hip_half = (m.hip + FABRIC_EASE[fabric].default_ease_mm) / 4
    pattern, _ = _assemble("bodice", m, fabric, p, hip_y, hip_half, 0, 0, False, "none")
    return pattern
