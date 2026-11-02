"""Women's boxy woven shirt: button front, dropped shoulder, collar, cuffs, patch pockets.

Assumptions to tune with real samples (a pattern maker should review):
- No bust dart: boxy shirts carry the bust in ease (design_ease). Fuller
  busts need a dart or an FBA that this block does not draft.
- Front and back share the same side length and hem level; no front dip.
- Button stand is grown on (front.edge is folded back as its own facing,
  normal allowance); buttons sit on centre front.
- The sleeve cap height is solved so cap ease lands mid fabric range; the
  sleeve is straight (no taper), so a plain turned-up cuff equals the sleeve hem.
- Bishop sleeve (sleeve_hem_fullness > 0): the underarm seams flare out so the
  hem gains that width, plus 30 mm of length that blouses over the cuff; the
  hem is gathered into a fitted cuff. Puff shoulder (sleeve_cap_gather > 0):
  the cap is solved for that much extra length, gathered into the armhole.
- A fitted cuff is sized from the arm girth at the cuff line, interpolated
  linearly from bicep (at the cap line) to wrist (at the measured arm length).
  An elastic casing cuff (cuff_elastic) is girth + 60 mm so it slides over the
  hand; its elastic is cut girth - 10 mm. A fitted cuff without elastic needs
  a sleeve placket at the wrist, which is not drafted, so it is refused there.
- Collars are flat collars traced from the joined necklines with a 12 mm
  shoulder overlap (gives a slight roll); style 1 pointed, style 2 rounded
  (peter pan). Style 3 is a straight band (mandarin) collar BAND_COLLAR_MM high.
  No collar stand is drafted on the flat styles. The overlap is taken 100 mm out
  along the shoulder so dropping the shoulder never reshapes the collar.
- Patch pocket size scales with bust (clamped 110-140 mm wide); placement is
  recorded in metadata, not drawn as drill holes.
- Sleeveless (vest): no sleeve or cuff; the armhole is raised and the shoulder
  cut in (seamproof.draft.finish) and both armholes are bound with bias. A
  dropped shoulder needs a sleeve, so it is refused.
- Without a collar the neckline (back neck, front neck and the top of the
  button stand) is bound with bias cut as two halves joined at centre back,
  outer ends turned in at the front edges.
- V neck: straight front neckline to centre front; the stand top continues
  the V line past centre front. A V neck has no collar here.
No seam allowances are included here.
"""

from __future__ import annotations

import math
from typing import Mapping

from shapely.geometry import LineString

from seamproof.draft.finish import SLEEVELESS_CUT_IN_MM, SLEEVELESS_RAISE_MM, bias_strip, v_line
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length

SHOULDER_OVERLAP_MM = 12.0
# Overlap is measured this far out along the shoulder, so a dropped shoulder
# (longer seam) does not change the collar (assumption).
SHOULDER_OVERLAP_REF_MM = 100.0
HIP_MIN_EASE_MM = 40.0
SLEEVE_BICEP_EASE_MM = 60.0
BISHOP_BLOUSE_MM = 30.0  # extra sleeve length that hangs over a gathered cuff
CUFF_ELASTIC_PASS_MM = 60.0  # elastic cuff band over the arm girth, so the hand slides through
CUFF_FITTED_EASE_MM = 30.0  # gathered non-elastic band over the arm girth
ELASTIC_UNDER_MM = 10.0  # elastic cut this much under the arm girth
WRIST_ZONE_MM = 60.0  # a cuff this close to the wrist must pass the hand
BAND_COLLAR_MM = 30.0  # finished height of a band collar (assumption)


def _line(edge_id: str, start: Point, end: Point, partner: str | None = None, fold: bool = False) -> Edge:
    """Create a straight stitch-line edge without allowance."""
    return Edge(edge_id, Curve((start, end)), partner, needs_allowance=not fold, is_fold=fold)


def _hip_at(m: WomenMeasurements, below_waist: float) -> float:
    """Body circumference at a level below the waist (linear waist->hip, then hip)."""
    if below_waist >= m.waist_to_hip:
        return m.hip
    return m.waist + (m.hip - m.waist) * max(0.0, below_waist) / m.waist_to_hip


def _geometry(m: WomenMeasurements, ease: float, p: dict[str, float]) -> dict[str, float | Point]:
    """Shared body points for front and back half panels, in mm."""
    neck_side = Point(p["neckline_width"] / 2, 0)
    tip = Point(m.shoulder_width / 2, p["shoulder_slope"])
    run = Point(tip.x - neck_side.x, tip.y - neck_side.y)
    length = math.hypot(run.x, run.y)
    if length <= 0:
        raise ValueError("Shoulder width must exceed the neckline width")
    drop = p["shoulder_drop"]
    sleeveless = p["sleeveless"] >= 0.5
    if sleeveless and drop > 0:
        raise ValueError("A dropped shoulder needs a sleeve")
    # Sleeveless: the shoulder is cut in along its own line (so the collar is unchanged).
    reach = -SLEEVELESS_CUT_IN_MM if sleeveless else drop
    shoulder = Point(tip.x + run.x / length * reach, tip.y + run.y / length * reach)
    depth = m.armhole_depth + 0.6 * drop - (SLEEVELESS_RAISE_MM if sleeveless else 0.0)
    chest_half = (m.bust + ease + p["design_ease"]) / 4
    hem_y = m.back_length + p["shirt_length"]
    hip_need = (_hip_at(m, p["shirt_length"]) + HIP_MIN_EASE_MM) / 4
    hem_half = max(chest_half, hip_need)
    if not neck_side.x < shoulder.x < chest_half or not shoulder.y < depth < hem_y:
        raise ValueError("Measurements do not fit this shirt block")
    return {"neck_side": neck_side, "shoulder": shoulder, "underarm": Point(chest_half, depth),
            "hem": Point(hem_half, hem_y), "depth": depth, "hem_y": hem_y}


def _armhole(name: str, g: dict, p: dict[str, float]) -> Edge:
    """Armhole from shoulder point to underarm, scoop set by armhole_curve_depth."""
    shoulder, underarm, depth = g["shoulder"], g["underarm"], g["depth"]
    return Edge(f"{name}.armhole", Curve.from_cubic_bezier(
        shoulder,
        Point(shoulder.x - p["armhole_curve_depth"] * 0.5, shoulder.y + (depth - shoulder.y) * 0.55),
        Point(underarm.x - p["armhole_curve_depth"], depth), underarm,
    ), "armhole_bias.attach" if p["sleeveless"] >= 0.5 else "sleeve.cap")


def _arm_notches(name: str, armhole: Edge) -> list[Notch]:
    """Shoulder point plus front single / back double balance notches."""
    length = edge_length(armhole)
    if name == "front":
        return [Notch(armhole.id, 0), Notch(armhole.id, length * 0.6)]
    return [Notch(armhole.id, 0)] + [Notch(armhole.id, length * 0.55 + o) for o in (-3, 3)]


def _back(m: WomenMeasurements, g: dict, p: dict[str, float]) -> Panel:
    """Half back on the fold."""
    drop = p["neck_drop_back"]
    neck, ns = Point(0, drop), g["neck_side"]
    neckline = Edge("back.neckline", Curve.from_cubic_bezier(
        neck, Point(ns.x * 0.55, drop), Point(ns.x, drop * 0.55), ns), _neck_partner(p))
    armhole = _armhole("back", g, p)
    bottom = Point(0, g["hem_y"])
    edges = [neckline, _line("back.shoulder", ns, g["shoulder"], "front.shoulder"), armhole,
             _line("back.side", g["underarm"], g["hem"], "front.side"),
             _line("back.hem", g["hem"], bottom), _line("back.fold", bottom, neck, fold=True)]
    return Panel("back", edges, (Point(25, g["depth"]), Point(25, g["hem_y"] - 25)),
                 _arm_notches("back", armhole), cut_on_fold=True)


def _front(m: WomenMeasurements, g: dict, p: dict[str, float]) -> Panel:
    """Half front with a grown-on button stand past centre front; cut as a pair."""
    drop, ext = p["neckline_depth_front"], p["placket_width"]
    neck, ns = Point(0, drop), g["neck_side"]
    if drop >= g["depth"]:
        raise ValueError("Front neck drop must stay above the armhole depth")
    if p["neck_v"] >= 0.5:
        # The stand top carries the V line on past centre front to the front edge.
        neckline = Edge("front.neckline", v_line(neck, ns), _neck_partner(p))
        edge_top = Point(-ext, drop + drop / ns.x * ext)
    else:
        neckline = Edge("front.neckline", Curve.from_cubic_bezier(
            neck, Point(ns.x * 0.55, drop), Point(ns.x, drop * 0.55), ns), _neck_partner(p))
        edge_top = Point(-ext, drop)
    armhole = _armhole("front", g, p)
    edge_bottom = Point(-ext, g["hem_y"])
    edges = [neckline, _line("front.shoulder", ns, g["shoulder"], "back.shoulder"), armhole,
             _line("front.side", g["underarm"], g["hem"], "back.side"),
             _line("front.hem", g["hem"], edge_bottom),
             _line("front.edge", edge_bottom, edge_top), _line("front.ext_top", edge_top, neck)]
    # Centre-front notches at neck and hem mark the button line.
    stand_top = math.hypot(edge_top.x - neck.x, edge_top.y - neck.y)
    notches = _arm_notches("front", armhole) + [Notch("front.ext_top", stand_top), Notch("front.hem", g["hem"].x)]
    return Panel("front", edges, (Point(25, g["depth"]), Point(25, g["hem_y"] - 25)), notches, cut_count=2)


def _neck_partner(p: Mapping[str, float]) -> str:
    """The neckline is sewn to the collar, or bound with bias when there is none."""
    return "collar.neck" if int(p["collar_style"]) else "neck_bias.attach"


def _cap(width: float, height: float) -> Curve:
    """Sleeve cap from front underarm over the crown to back underarm."""
    front, crown, back = Point(-width, height), Point(0, 0), Point(width, height)
    a = Curve.from_cubic_bezier(front, Point(-width * 0.75, height * 0.45), Point(-width * 0.4, 0), crown)
    b = Curve.from_cubic_bezier(crown, Point(width * 0.45, 0), Point(width * 0.8, height * 0.45), back)
    return Curve(a.points + b.points[1:])


def _cap_length(width: float, height: float) -> float:
    """Arc length of a sampled cap, in mm."""
    points = _cap(width, height).points
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def arm_girth_at(m: WomenMeasurements, level: float, cap_height: float) -> float:
    """Arm circumference at a distance below the crown: bicep at the cap line, wrist at full arm length."""
    span = m.sleeve_length - cap_height
    t = 1.0 if span <= 0 else min(1.0, max(0.0, (level - cap_height) / span))
    return m.bicep + (m.wrist - m.bicep) * t


def gathered(p: Mapping[str, float]) -> bool:
    """True when the sleeve hem is gathered into a fitted cuff."""
    return float(p["sleeve_hem_fullness"]) > 0 or float(p["cuff_elastic"]) >= 0.5


def cuff_band_length(m: WomenMeasurements, p: Mapping[str, float], cap_height: float) -> float:
    """Fitted cuff length from the arm girth at the finished sleeve length."""
    girth = arm_girth_at(m, p["shirt_sleeve_length"], cap_height)
    if float(p["cuff_elastic"]) >= 0.5:
        return girth + CUFF_ELASTIC_PASS_MM
    if p["shirt_sleeve_length"] >= m.sleeve_length - WRIST_ZONE_MM:
        raise ValueError("A fitted wrist cuff without elastic needs a sleeve placket — use an elastic cuff")
    return girth + CUFF_FITTED_EASE_MM


def _sleeve(m: WomenMeasurements, p: dict[str, float], armholes_mm: float, target_ease: float) -> Panel:
    """Sleeve whose cap height is solved for the target cap ease (+ puff gathers); bishop hems flare."""
    width = (m.bicep + SLEEVE_BICEP_EASE_MM + p["design_ease"] * 0.25) / 2
    goal = armholes_mm + target_ease + p["sleeve_cap_gather"]
    low, high = 5.0, 260.0
    if not _cap_length(width, low) <= goal <= _cap_length(width, high):
        raise ValueError("No sleeve cap height reaches the armhole length")
    for _ in range(40):
        mid = (low + high) / 2
        low, high = (mid, high) if _cap_length(width, mid) < goal else (low, mid)
    height = round((low + high) / 2, 2)
    fullness = p["sleeve_hem_fullness"]
    if gathered(p) and p["cuff_depth"] <= 0:
        raise ValueError("A gathered sleeve hem needs a cuff (cuff_depth > 0)")
    body = p["shirt_sleeve_length"] - p["cuff_depth"] + (BISHOP_BLOUSE_MM if fullness > 0 else 0.0)
    if body <= height + 10:
        raise ValueError("Sleeve too short for its cap")
    cap = Edge("sleeve.cap", _cap(width, height), "body.armholes")
    left, right = Point(-width - fullness / 2, body), Point(width + fullness / 2, body)
    hem_id = "sleeve.cuff_join" if p["cuff_depth"] > 0 else "sleeve.hem"
    edges = [cap, _line("sleeve.underarm_back", Point(width, height), right, "sleeve.underarm_front"),
             _line(hem_id, right, left, "cuff.attach" if p["cuff_depth"] > 0 else None),
             _line("sleeve.underarm_front", left, Point(-width, height), "sleeve.underarm_back")]
    half = edge_length(Edge("half", Curve(cap.curve.points[:len(cap.curve.points) // 2 + 1])))
    notches = [Notch(cap.id, half), Notch(cap.id, half * 0.45)]
    notches += [Notch(cap.id, half + (edge_length(cap) - half) * 0.55 + o) for o in (-3, 3)]
    if p["cuff_depth"] > 0:  # centre of the hem meets the cuff's centre notch
        notches.append(Notch(hem_id, width + fullness / 2))
    panel = Panel("sleeve", edges, (Point(0, height + 10), Point(0, body - 10)), notches, cut_count=2)
    panel.notches = [n for n in panel.notches if n.distance_from_start_mm <= edge_length(cap)]
    return panel


def _cuff(sleeve: Panel, depth: float, length: float | None = None) -> Panel:
    """Folded cuff band, height twice the depth; attach edge equals the sleeve hem unless fitted."""
    hem = edge_length(next(e for e in sleeve.edges if e.id == "sleeve.cuff_join"))
    length = hem if length is None else length
    if length > hem + 1e-6:
        raise ValueError("Fitted cuff is longer than the sleeve hem it gathers")
    a, b, c, d = Point(0, 0), Point(length, 0), Point(length, 2 * depth), Point(0, 2 * depth)
    edges = [_line("cuff.attach", a, b, "sleeve.cuff_join"), _line("cuff.end_r", b, c, "cuff.end_l"),
             _line("cuff.outer", c, d), _line("cuff.end_l", d, a, "cuff.end_r")]
    return Panel("cuff", edges, (Point(10, depth), Point(length - 10, depth)),
                 [Notch("cuff.attach", length / 2)], cut_count=2)


def _rotate(point: Point, centre: Point, angle: float) -> Point:
    """Rotate a point about a centre, radians."""
    dx, dy = point.x - centre.x, point.y - centre.y
    c, s = math.cos(angle), math.sin(angle)
    return Point(centre.x + dx * c - dy * s, centre.y + dx * s + dy * c)


def _collar(front: Panel, back: Panel, style: int, width: float) -> Panel:
    """Half flat collar on the CB fold, traced from back + front necklines."""
    back_neck = next(e for e in back.edges if e.id == "back.neckline").curve.points
    front_neck = next(e for e in front.edges if e.id == "front.neckline").curve.points
    back_sh = next(e for e in back.edges if e.id == "back.shoulder").curve.points
    front_sh = next(e for e in front.edges if e.id == "front.shoulder").curve.points
    ns = back_neck[-1]
    # Flip the front above the shoulder line, align shoulders, then overlap.
    flipped = [Point(p.x, -p.y) for p in front_neck]
    f_dir = math.atan2(-(front_sh[1].y - front_sh[0].y), front_sh[1].x - front_sh[0].x)
    b_dir = math.atan2(back_sh[1].y - back_sh[0].y, back_sh[1].x - back_sh[0].x)
    overlap = 2 * math.asin(SHOULDER_OVERLAP_MM / 2 / SHOULDER_OVERLAP_REF_MM)
    angle = (b_dir - f_dir) + overlap
    placed_front = [_rotate(p, ns, angle) for p in flipped]
    neck_pts = list(back_neck) + list(reversed(placed_front))[1:]
    neck_line = LineString([(p.x, p.y) for p in neck_pts])
    centre = Point(sum(p.x for p in neck_pts) / len(neck_pts), sum(p.y for p in neck_pts) / len(neck_pts))
    best = None
    for side in (width, -width):
        off = neck_line.offset_curve(side, quad_segs=8, join_style="round")
        if off.geom_type != "LineString" or off.is_empty:
            continue
        coords = list(off.coords)
        far = sum(math.hypot(x - centre.x, y - centre.y) for x, y in coords) / len(coords)
        if best is None or far > best[0]:
            best = (far, coords)
    if best is None:
        raise ValueError("Collar offset failed")
    outer = [Point(x, y) for x, y in best[1]]
    if math.hypot(outer[0].x - neck_pts[0].x, outer[0].y - neck_pts[0].y) > \
            math.hypot(outer[-1].x - neck_pts[0].x, outer[-1].y - neck_pts[0].y):
        outer.reverse()
    outer[0] = Point(0.0, outer[0].y)  # CB stays on the fold line.
    cf, end_out = neck_pts[-1], outer[-1]
    tx, ty = cf.x - neck_pts[-2].x, cf.y - neck_pts[-2].y
    tl = math.hypot(tx, ty)
    fwd = Point(tx / tl, ty / tl)
    if style == 2:
        # Round the CF corner inside the CF line: trim the outer edge back by
        # 0.6 x width, then sweep from the neck point out to the trimmed end.
        trim, walked = width * 0.6, 0.0
        while len(outer) > 2:
            a, b = outer[-1], outer[-2]
            step = math.hypot(a.x - b.x, a.y - b.y)
            if walked + step >= trim:
                break
            walked += step
            outer.pop()
        q, q_prev = outer[-1], outer[-2]
        ql = math.hypot(q.x - q_prev.x, q.y - q_prev.y)
        toward_cf = Point((q.x - q_prev.x) / ql, (q.y - q_prev.y) / ql)
        normal = Point((end_out.x - cf.x) / width, (end_out.y - cf.y) / width)
        end = Curve.from_cubic_bezier(
            cf, Point(cf.x + normal.x * width * 0.6, cf.y + normal.y * width * 0.6),
            Point(q.x + toward_cf.x * trim * 0.6, q.y + toward_cf.y * trim * 0.6), q)
        end_out = q
    else:
        tip = Point(end_out.x + fwd.x * width * 0.2, end_out.y + fwd.y * width * 0.2)
        end = Curve((cf, tip, end_out))
    edges = [Edge("collar.neck", Curve(tuple(neck_pts)), "body.collar_line"),
             Edge("collar.end", end),
             Edge("collar.outer", Curve(tuple(reversed(outer)))),
             _line("collar.fold", outer[0], neck_pts[0], fold=True)]
    back_len = edge_length(Edge("b", Curve(tuple(back_neck))))
    gx = 15.0
    gy0 = min(p.y for p in outer + neck_pts) + 5
    gy1 = max(p.y for p in outer + neck_pts) - 5
    return Panel("collar", edges, (Point(gx, gy0), Point(gx, gy1)),
                 [Notch("collar.neck", back_len)], cut_count=2, cut_on_fold=True)


def _band_collar(front: Panel, back: Panel) -> Panel:
    """Half straight band collar on the CB fold; its neck edge is half the joined neckline."""
    back_len = edge_length(next(e for e in back.edges if e.id == "back.neckline"))
    length = back_len + edge_length(next(e for e in front.edges if e.id == "front.neckline"))
    a, b, c, d = Point(0, 0), Point(length, 0), Point(length, BAND_COLLAR_MM), Point(0, BAND_COLLAR_MM)
    edges = [Edge("collar.neck", Curve((a, b)), "body.collar_line"), Edge("collar.end", Curve((b, c))),
             Edge("collar.outer", Curve((c, d))), _line("collar.fold", d, a, fold=True)]
    return Panel("collar", edges, (Point(15, 5), Point(15, BAND_COLLAR_MM - 5)),
                 [Notch("collar.neck", back_len)], cut_count=2, cut_on_fold=True)


def _pocket(m: WomenMeasurements, count: int) -> Panel:
    """Rectangular patch pocket; the top edge is a 25 mm turned facing."""
    w = min(140.0, max(110.0, m.bust * 0.13))
    h = round(w * 1.15, 1)
    a, b, c, d = Point(0, 0), Point(w, 0), Point(w, h), Point(0, h)
    edges = [_line("pocket.hem", a, b), _line("pocket.side_r", b, c),
             _line("pocket.bottom", c, d), _line("pocket.side_l", d, a)]
    return Panel("pocket", edges, (Point(w / 2, 10), Point(w / 2, h - 10)), [], cut_count=count)


def draft_shirt(measurements: WomenMeasurements | Mapping[str, float], fabric: str,
                params: Mapping[str, float] | None = None) -> Pattern:
    """Draft back, front pair, sleeves and the optional collar, cuffs and pockets."""
    m = measurements if isinstance(measurements, WomenMeasurements) else WomenMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    p = resolve_params(params)
    ease = FABRIC_EASE[fabric]
    g = _geometry(m, ease.default_ease_mm, p)
    style = int(p["collar_style"])
    if style and p["neck_v"] >= 0.5:
        raise ValueError("A V neck is drafted without a collar")
    back, front = _back(m, g, p), _front(m, g, p)
    armholes = sum(edge_length(e) for panel in (front, back) for e in panel.edges if e.id.endswith(".armhole"))
    sleeveless = p["sleeveless"] >= 0.5
    band, cap_height, hem_join = None, 0.0, 0.0
    if sleeveless:
        if p["cuff_depth"] > 0 or p["sleeve_hem_fullness"] > 0 or p["sleeve_cap_gather"] > 0:
            raise ValueError("A sleeveless shirt has no sleeve, cuff or gathers")
        panels = [front, back, bias_strip("armhole_bias", armholes, "body.armholes", 2)]
    else:
        low, high = ease.sleeve_cap_ease_range_mm
        sleeve = _sleeve(m, p, armholes, (low + high) / 2)
        panels = [front, back, sleeve]
        cap_height = next(e for e in sleeve.edges if e.id == "sleeve.underarm_back").curve.points[0].y
        band = cuff_band_length(m, p, cap_height) if p["cuff_depth"] > 0 and gathered(p) else None
    if style:
        panels.append(_band_collar(front, back) if style == 3 else _collar(front, back, style, p["collar_width"]))
    else:
        edge = {e.id: edge_length(e) for panel in (front, back) for e in panel.edges}
        # Cut as two halves joined at centre back: a shorter bias strip nests tighter.
        half_neck = edge["back.neckline"] + edge["front.neckline"] + edge["front.ext_top"]
        panels.append(bias_strip("neck_bias", half_neck, "body.neck", 2))
    if not sleeveless and p["cuff_depth"] > 0:
        panels.append(_cuff(sleeve, p["cuff_depth"], band))
        hem_join = next(edge_length(e) for e in sleeve.edges if e.id == "sleeve.cuff_join")
    if int(p["pocket_count"]):
        panels.append(_pocket(m, int(p["pocket_count"])))
    size = next((name for name, values in WOMEN_SIZE_TABLE.items() if values == m), "custom")
    pocket_x = m.apex_distance / 2
    return Pattern("shirt", size, fabric, m.to_dict(), p, panels, {
        "status": "draft_unverified",
        "ease_mm": {"bust": ease.default_ease_mm + p["design_ease"]},
        "chest_line_y_mm": g["depth"],
        "seam_groups": {
            "body.armholes": [{"edge_id": "front.armhole", "reverse": True},
                              {"edge_id": "back.armhole", "reverse": False}],
            "body.collar_line": [{"edge_id": "back.neckline", "reverse": False},
                                 {"edge_id": "front.neckline", "reverse": True}],
        },
        "pocket_placement_mm": {"centre_x_from_cf": pocket_x, "top_y": max(m.apex_height - 40, 120)},
        # Gathering instructions: hem gathered into the cuff, cap gathered between the cap notches.
        "gathers_mm": {"sleeve_hem": (hem_join - band) if band is not None else 0.0,
                       "sleeve_cap": p["sleeve_cap_gather"],
                       "elastic_cut": (arm_girth_at(m, p["shirt_sleeve_length"], cap_height) - ELASTIC_UNDER_MM
                                       if p["cuff_elastic"] >= 0.5 and p["cuff_depth"] > 0 and not sleeveless
                                       else 0.0)},
        "sleeveless": sleeveless,
        "assumptions": ("Boxy no-dart block; grown-on button stand; flat traced collar with "
                        "12 mm shoulder overlap; cap height solved for mid-range cap ease."),
    })
