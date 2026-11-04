"""Named pattern mutations backing `mutation-report`; offline and deterministic.

Each mutator breaks one valid draft in a targeted way so the verifier must
catch it. Draft-correction code would re-place derived notches; the mutators
do the same to stay targeted.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

from seamproof.draft.blouse import draft_blouse
from seamproof.draft.dress import draft_dress
from seamproof.draft.shirt import draft_shirt
from seamproof.draft.skirt import draft_skirt
from seamproof.draft.tshirt import draft_tshirt
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length
from seamproof.verify import verify_pattern


def _edge(panel: Panel, edge_id: str) -> Edge:
    """Fetch one panel edge by id."""
    return next(edge for edge in panel.edges if edge.id == edge_id)


def _swap(pattern: Pattern, panel_index: int, *new_edges: Edge) -> Pattern:
    """Return a copy with the named edges replaced on one panel."""
    clone = copy.deepcopy(pattern)
    panel = clone.panels[panel_index]
    table = {edge.id: edge for edge in new_edges}
    panel.edges = [table.get(edge.id, edge) for edge in panel.edges]
    return clone


def _sync_leg_notches(panel: Panel) -> None:
    """Re-place dart-seam notches after a leg edit, in place."""
    legs = {edge.id: edge for edge in panel.edges if edge.id in ("front.dart_leg1", "front.dart_leg2")}
    panel.notches = [n for n in panel.notches if n.edge_id not in legs]
    panel.notches.append(Notch("front.dart_leg1", 0))
    panel.notches.append(Notch("front.dart_leg2", edge_length(legs["front.dart_leg2"])))


def _blouse() -> Pattern:
    """Return the pristine M blouse every bodice mutation starts from."""
    return draft_blouse(WOMEN_SIZE_TABLE["M"], "woven")


def _tshirt() -> Pattern:
    """Return the pristine S t-shirt the neckband mutation starts from."""
    return draft_tshirt(SIZE_TABLE["S"], "knit_low_stretch")


# Style of notes/sample1.png: boxy dropped-shoulder shirt with peter pan collar,
# cuffs and two pockets; A-line mini skirt with a faced scallop hem.
SHIRT_STYLE = {"design_ease": 140, "shoulder_drop": 50, "shirt_length": 200, "shirt_sleeve_length": 300,
               "collar_style": 2, "cuff_depth": 35, "pocket_count": 2, "neckline_depth_front": 75,
               "neck_drop_back": 20}
SKIRT_STYLE = {"skirt_length": 420, "scallop_depth": 18, "waistband_height": 0}


def _shirt() -> Pattern:
    """Return the pristine M shirt in the sample-photo style."""
    return draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", SHIRT_STYLE)


# Brief style: long puff (bishop + gathered cap) sleeves on elastic cuffs, pointed collar.
BISHOP_STYLE = {"collar_style": 1, "shirt_sleeve_length": 580, "cuff_depth": 35, "sleeve_hem_fullness": 120,
                "sleeve_cap_gather": 40, "cuff_elastic": 1, "neckline_depth_front": 75}


def _bishop() -> Pattern:
    """Return the pristine M shirt with gathered puff sleeves."""
    return draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", BISHOP_STYLE)


def m_bishop_cuff(pattern: Pattern) -> Pattern:
    """Cut the elastic cuff band 4 mm long so it no longer fits the arm rule."""
    return _stretch_rect(pattern, "cuff", "cuff", 4)


def m_bishop_gathers(pattern: Pattern) -> Pattern:
    """Flare the sleeve hem 100 mm more each side: too much to gather into the cuff."""
    clone = _move_corner(pattern, "sleeve", "sleeve.underarm_back", "sleeve.cuff_join", Point(100, 0))
    return _move_corner(clone, "sleeve", "sleeve.cuff_join", "sleeve.underarm_front", Point(-100, 0))


# Vest style (photo of a navy vest): sleeveless, collarless V neck, button front, two pockets.
VEST_STYLE = {"sleeveless": 1, "collar_style": 0, "neck_v": 1, "neckline_depth_front": 130, "design_ease": 20,
              "pocket_count": 2}


def _vest() -> Pattern:
    """Return the pristine M sleeveless V-neck shirt (vest)."""
    return draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", VEST_STYLE)


def m_armhole_bias(pattern: Pattern) -> Pattern:
    """Cut the armhole bias 5 mm short of the armhole."""
    return _stretch_rect(pattern, "armhole_bias", "armhole_bias", -5)


def m_neck_bias(pattern: Pattern) -> Pattern:
    """Cut the neck bias 5 mm long."""
    return _stretch_rect(pattern, "neck_bias", "neck_bias", 5)


def m_lost_sleeve(pattern: Pattern) -> Pattern:
    """Drop the sleeve panel from a sleeved blouse."""
    clone = copy.deepcopy(pattern)
    clone.panels = [panel for panel in clone.panels if panel.name != "sleeve"]
    return clone


def _skirt() -> Pattern:
    """Return the pristine M skirt in the sample-photo style."""
    return draft_skirt(WOMEN_SIZE_TABLE["M"], "woven", SKIRT_STYLE)


# Waist-seam dress: A-line knee skirt, short sleeve, straight hem.
DRESS_STYLE = {"dress_waist_seam": 1, "skirt_length": 580, "skirt_flare": 80, "dress_sleeve_length": 200}


def _dress() -> Pattern:
    """Return the pristine M waist-seam dress."""
    return draft_dress(WOMEN_SIZE_TABLE["M"], "woven", DRESS_STYLE)


def _index(pattern: Pattern, name: str) -> int:
    """Position of a panel by name."""
    return next(i for i, panel in enumerate(pattern.panels) if panel.name == name)


def _move_corner(pattern: Pattern, panel: str, before_id: str, after_id: str, delta: Point) -> Pattern:
    """Move the shared point where edge before_id ends and after_id starts."""
    i = _index(pattern, panel)
    a, b = _edge(pattern.panels[i], before_id), _edge(pattern.panels[i], after_id)
    corner = a.curve.points[-1]
    moved = Point(corner.x + delta.x, corner.y + delta.y)
    return _swap(pattern, i, Edge(a.id, Curve(a.curve.points[:-1] + (moved,)), a.seam_partner_id,
                                  a.needs_allowance, a.allowance_mm, a.is_fold),
                 Edge(b.id, Curve((moved,) + b.curve.points[1:]), b.seam_partner_id,
                      b.needs_allowance, b.allowance_mm, b.is_fold))


def _stretch_rect(pattern: Pattern, panel: str, prefix: str, extra: float, names: tuple[str, str] = ("attach", "end_r")) -> Pattern:
    """Lengthen a rectangular band's attach edge (and its far end) by extra mm."""
    i = _index(pattern, panel)
    attach, end, outer = (_edge(pattern.panels[i], f"{prefix}.{n}") for n in (*names, "outer"))
    b, c = attach.curve.points[-1], end.curve.points[-1]
    nb, nc = Point(b.x + extra, b.y), Point(c.x + extra, c.y)
    return _swap(pattern, i, Edge(attach.id, Curve((attach.curve.points[0], nb)), attach.seam_partner_id),
                 Edge(end.id, Curve((nb, nc)), end.seam_partner_id),
                 Edge(outer.id, Curve((nc, outer.curve.points[-1])), outer.seam_partner_id))


def m_shirt_side(pattern: Pattern) -> Pattern:
    """Drop the back hem corner 6 mm, lengthening the back side seam."""
    return _move_corner(pattern, "back", "back.side", "back.hem", Point(0, 6))


def m_cuff(pattern: Pattern) -> Pattern:
    """Make the cuff band 4 mm longer than the sleeve hem."""
    return _stretch_rect(pattern, "cuff", "cuff", 4)


def m_collar(pattern: Pattern) -> Pattern:
    """Pull the collar's CF neck point 6 mm back along the neck edge."""
    i = _index(pattern, "collar")
    neck, end = _edge(pattern.panels[i], "collar.neck"), _edge(pattern.panels[i], "collar.end")
    pts = neck.curve.points
    walked, k = 0.0, len(pts) - 1
    while walked < 6 and k > 1:
        walked += math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y)
        k -= 1
    new_cf = pts[k]
    return _swap(pattern, i, Edge(neck.id, Curve(pts[:k + 1]), neck.seam_partner_id),
                 Edge(end.id, Curve((new_cf,) + end.curve.points[1:])))


def m_shirt_chest(pattern: Pattern) -> Pattern:
    """Push the back underarm out 5 mm, widening the chest."""
    return _move_corner(pattern, "back", "back.armhole", "back.side", Point(5, 0))


def m_shirt_hem(pattern: Pattern) -> Pattern:
    """Pull both hem corners in 80 mm so the hem no longer clears the hips."""
    clone = _move_corner(pattern, "back", "back.side", "back.hem", Point(-80, 0))
    clone = _move_corner(clone, "front", "front.side", "front.hem", Point(-80, 0))
    front = clone.panels[_index(clone, "front")]
    hem = _edge(front, "front.hem")
    front.notches = [n for n in front.notches if n.edge_id != "front.hem"]
    front.notches.append(Notch("front.hem", hem.curve.points[0].x))  # CF line stays marked
    return clone


def m_skirt_dart(pattern: Pattern) -> Pattern:
    """Slide the front skirt dart tip 10 mm sideways so its legs differ."""
    i = _index(pattern, "front")
    leg1, leg2 = _edge(pattern.panels[i], "front.dart_leg1"), _edge(pattern.panels[i], "front.dart_leg2")
    tip = leg1.curve.points[-1]
    moved = Point(tip.x + 10, tip.y)
    clone = _swap(pattern, i, Edge(leg1.id, Curve((leg1.curve.points[0], moved)), leg1.seam_partner_id),
                  Edge(leg2.id, Curve((moved, leg2.curve.points[-1])), leg2.seam_partner_id))
    front = clone.panels[i]
    front.notches = [n for n in front.notches if n.edge_id != "front.dart_leg2"]
    front.notches.append(Notch("front.dart_leg2", edge_length(_edge(front, "front.dart_leg2"))))
    return clone


def m_waist_finish(pattern: Pattern) -> Pattern:
    """Cut the waist facing strip 5 mm short."""
    return _stretch_rect(pattern, "waist_facing", "waist_facing", -5)


def m_hem_facing(pattern: Pattern) -> Pattern:
    """Bow one scallop of the hem facing 10 mm so it no longer traces the hem."""
    i = _index(pattern, "hem_facing")
    edge = _edge(pattern.panels[i], "hem_facing.scallop")
    pts = list(edge.curve.points)
    mid = len(pts) // 2
    pts[mid] = Point(pts[mid].x, pts[mid].y + 10)
    return _swap(pattern, i, Edge(edge.id, Curve(tuple(pts)), edge.seam_partner_id))


def m_skirt_waist(pattern: Pattern) -> Pattern:
    """Widen the front waist 3 mm and keep the facing in step (waist width only)."""
    clone = _move_corner(pattern, "front", "front.waist_side", "front.side", Point(3, 0))
    return _stretch_rect(clone, "waist_facing", "waist_facing", 6)


def m_skirt_hip(pattern: Pattern) -> Pattern:
    """Ease the front side out 5 mm around the hip line with a smooth bump."""
    i = _index(pattern, "front")
    side = _edge(pattern.panels[i], "front.side")
    hip_y = pattern.measurements["waist_to_hip"]
    inner = tuple(Point(p.x + 5 * math.exp(-((p.y - hip_y) / 40) ** 2), p.y) for p in side.curve.points[1:-1])
    pts = side.curve.points[:1] + inner + side.curve.points[-1:]  # endpoints stay joined
    return _swap(pattern, i, Edge(side.id, Curve(pts), side.seam_partner_id))


def _slide_tip(pattern: Pattern, panel: str, leg1_id: str, leg2_id: str, dx: float) -> Pattern:
    """Slide a dart tip sideways so its legs differ; the base (and waist) stay put."""
    i = _index(pattern, panel)
    leg1, leg2 = _edge(pattern.panels[i], leg1_id), _edge(pattern.panels[i], leg2_id)
    moved = Point(leg1.curve.points[-1].x + dx, leg1.curve.points[-1].y)
    clone = _swap(pattern, i, Edge(leg1.id, Curve((leg1.curve.points[0], moved)), leg1.seam_partner_id),
                  Edge(leg2.id, Curve((moved, leg2.curve.points[-1])), leg2.seam_partner_id))
    target = clone.panels[i]
    for notch in [n for n in target.notches if n.edge_id == leg2_id]:
        target.notches.remove(notch)
        target.notches.append(Notch(leg2_id, min(notch.distance_from_start_mm, edge_length(_edge(target, leg2_id)))))
    return clone


def m_dress_front_dart(pattern: Pattern) -> Pattern:
    """Slide the front waist dart tip 10 mm so its legs differ."""
    return _slide_tip(pattern, "front", "front.wdart_leg1", "front.wdart_leg2", 10)


def m_dress_back_dart(pattern: Pattern) -> Pattern:
    """Slide the long back waist dart tip 20 mm so its legs differ (10 mm stays in tolerance)."""
    return _slide_tip(pattern, "back", "back.wdart_leg1", "back.wdart_leg2", 20)


def m_dress_skirt_dart(pattern: Pattern) -> Pattern:
    """Slide the skirt front dart tip 10 mm so its legs differ."""
    return _slide_tip(pattern, "skirt_front", "skirt_front.dart_leg1", "skirt_front.dart_leg2", 10)


def m_dress_waist_seam(pattern: Pattern) -> Pattern:
    """Widen the skirt front waist 3 mm so it no longer matches the bodice."""
    return _move_corner(pattern, "skirt_front", "skirt_front.waist_side", "skirt_front.side", Point(3, 0))


def m_dress_waist(pattern: Pattern) -> Pattern:
    """Widen bodice and skirt front waists together 3 mm (seam still matches)."""
    clone = _move_corner(pattern, "front", "front.side_lower", "front.waist_side", Point(3, 0))
    return _move_corner(clone, "skirt_front", "skirt_front.waist_side", "skirt_front.side", Point(3, 0))


def m_dress_hip(pattern: Pattern) -> Pattern:
    """Ease the skirt front side out 5 mm around the hip line."""
    i = _index(pattern, "skirt_front")
    side = _edge(pattern.panels[i], "skirt_front.side")
    hip_y = pattern.measurements["waist_to_hip"]
    inner = tuple(Point(p.x + 5 * math.exp(-((p.y - hip_y) / 40) ** 2), p.y) for p in side.curve.points[1:-1])
    return _swap(pattern, i, Edge(side.id, Curve(side.curve.points[:1] + inner + side.curve.points[-1:]),
                                  side.seam_partner_id))


def m_dress_skirt_side(pattern: Pattern) -> Pattern:
    """Drop the skirt back hem corner 6 mm, lengthening its side seam."""
    return _move_corner(pattern, "skirt_back", "skirt_back.side", "skirt_back.hem", Point(0, 6))


def m_dress_bodice_side(pattern: Pattern) -> Pattern:
    """Drop the front side-waist corner 10 mm, lengthening the front bodice side."""
    return _move_corner(pattern, "front", "front.side_lower", "front.waist_side", Point(0, 10))


def m_closure(pattern: Pattern) -> Pattern:
    """Shift a facing end so its panel no longer closes."""
    facing = copy.deepcopy(pattern.panels[3])
    end = _edge(facing, "facing.end_cf")
    a, b = end.curve.points
    return _swap(pattern, 3, Edge(end.id, Curve((a, Point(b.x + 7, b.y)))))


def m_params(pattern: Pattern) -> Pattern:
    """Edit params without redrafting, breaking the fingerprint."""
    clone = copy.deepcopy(pattern)
    clone.params["unlogged_tweak"] = 1.0
    return clone


def m_shoulder(pattern: Pattern) -> Pattern:
    """Shorten the back shoulder 2 mm at the tip."""
    back = copy.deepcopy(pattern.panels[1])
    shoulder = _edge(back, "back.shoulder")
    armhole = _edge(back, "back.armhole")
    tip = Point(shoulder.curve.points[-1].x - 2, shoulder.curve.points[-1].y)
    return _swap(pattern, 1,
                 Edge(shoulder.id, Curve((shoulder.curve.points[0], tip)), shoulder.seam_partner_id),
                 Edge(armhole.id, Curve((tip, *armhole.curve.points[1:])), armhole.seam_partner_id))


def m_side_above(pattern: Pattern) -> Pattern:
    """Drop the front underarm 8 mm, lengthening the upper side."""
    front = copy.deepcopy(pattern.panels[0])
    upper = _edge(front, "front.side_upper")
    armhole = _edge(front, "front.armhole")
    low = Point(upper.curve.points[0].x, upper.curve.points[0].y - 8)
    return _swap(pattern, 0,
                 Edge(upper.id, Curve((low, upper.curve.points[-1])), upper.seam_partner_id),
                 Edge(armhole.id, Curve((*armhole.curve.points[:-1], low)), armhole.seam_partner_id))


def m_side_below(pattern: Pattern) -> Pattern:
    """Widen the back hem corner 35 mm, lengthening the lower side."""
    back = copy.deepcopy(pattern.panels[1])
    lower = _edge(back, "back.side_lower")
    hem = _edge(back, "back.hem")
    corner = Point(lower.curve.points[-1].x + 35, lower.curve.points[-1].y)
    return _swap(pattern, 1,
                 Edge(lower.id, Curve((lower.curve.points[0], corner)), lower.seam_partner_id),
                 Edge(hem.id, Curve((corner, hem.curve.points[-1]))))


def m_cap_ease(pattern: Pattern) -> Pattern:
    """Stretch both armholes sideways, starving the cap ease."""
    clone = copy.deepcopy(pattern)
    for index in (0, 1):
        panel = clone.panels[index]
        armhole = _edge(panel, panel.name + ".armhole")
        base = armhole.curve.points[-1]
        inner = [Point(base.x + (p.x - base.x) * 1.35, p.y) for p in armhole.curve.points[1:-1]]
        widened = Curve((armhole.curve.points[0], *inner, armhole.curve.points[-1]))
        panel.edges = [e if e.id != armhole.id else Edge(armhole.id, widened, armhole.seam_partner_id)
                       for e in panel.edges]
    return clone


def m_facing(pattern: Pattern) -> Pattern:
    """Bow the facing neck outward so it no longer matches."""
    facing = copy.deepcopy(pattern.panels[3])
    neck = _edge(facing, "facing.neck")
    points = [Point(p.x, p.y - 8) for p in neck.curve.points]
    points[0], points[-1] = neck.curve.points[0], neck.curve.points[-1]
    return _swap(pattern, 3, Edge(neck.id, Curve(tuple(points)), neck.seam_partner_id))


def m_underarm(pattern: Pattern) -> Pattern:
    """Drop one sleeve hem corner 5 mm, splitting the underarm pair."""
    sleeve = copy.deepcopy(pattern.panels[2])
    seam = _edge(sleeve, "sleeve.underarm_back")
    hem = _edge(sleeve, "sleeve.hem")
    low = Point(seam.curve.points[-1].x, seam.curve.points[-1].y + 5)
    return _swap(pattern, 2,
                 Edge(seam.id, Curve((seam.curve.points[0], low)), seam.seam_partner_id),
                 Edge(hem.id, Curve((low, hem.curve.points[-1]))))


def m_neckband(pattern: Pattern) -> Pattern:
    """Shorten the neckband ring 5 mm and re-place its end notch."""
    band = copy.deepcopy(pattern.panels[3])
    attach = _edge(band, "neckband.attach")
    end = _edge(band, "neckband.end_right")
    short = Point(attach.curve.points[-1].x - 5, attach.curve.points[-1].y)
    new_attach = Edge(attach.id, Curve((attach.curve.points[0], short)), attach.seam_partner_id)
    swapped = [new_attach,
               Edge(end.id, Curve((short, end.curve.points[-1])), end.seam_partner_id)]
    panel = _swap(pattern, 3, *swapped)
    length = edge_length(new_attach)
    panel.panels[3].notches = [
        n if n.edge_id != attach.id else Notch(n.edge_id, min(n.distance_from_start_mm, length))
        for n in panel.panels[3].notches]
    return panel


def _move_legs(pattern: Pattern, d1: Point, tip: Point, d2: Point) -> Pattern:
    """Rebuild the front dart and its neighbouring edges around new points."""
    front = copy.deepcopy(pattern.panels[0])
    upper = _edge(front, "front.side_upper")
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    lower = _edge(front, "front.side_lower")
    swapped = _swap(pattern, 0,
                    Edge(upper.id, Curve((upper.curve.points[0], d1)), upper.seam_partner_id),
                    Edge(leg1.id, Curve((d1, tip)), leg1.seam_partner_id),
                    Edge(leg2.id, Curve((tip, d2)), leg2.seam_partner_id),
                    Edge(lower.id, Curve((d2, *lower.curve.points[1:])), lower.seam_partner_id))
    _sync_leg_notches(swapped.panels[0])
    return swapped


def m_bust_join(pattern: Pattern) -> Pattern:
    """Narrow both dart base ends 5 mm, shrinking the sewn bust."""
    front = pattern.panels[0]
    leg1, leg2 = _edge(front, "front.dart_leg1"), _edge(front, "front.dart_leg2")
    d1 = Point(leg1.curve.points[0].x - 5, leg1.curve.points[0].y)
    d2 = Point(leg2.curve.points[-1].x - 5, leg2.curve.points[-1].y)
    return _move_legs(pattern, d1, leg1.curve.points[-1], d2)


def m_waist(pattern: Pattern) -> Pattern:
    """Push the front waist out 5 mm."""
    front = copy.deepcopy(pattern.panels[0])
    lower = _edge(front, "front.side_lower")
    points = [Point(p.x + 5 if i == 1 else p.x, p.y) for i, p in enumerate(lower.curve.points)]
    return _swap(pattern, 0, Edge(lower.id, Curve(tuple(points)), lower.seam_partner_id))


def m_hip(pattern: Pattern) -> Pattern:
    """Push the front hip out 5 mm."""
    front = copy.deepcopy(pattern.panels[0])
    lower = _edge(front, "front.side_lower")
    points = [Point(p.x + 5 if i == 2 else p.x, p.y) for i, p in enumerate(lower.curve.points)]
    return _swap(pattern, 0, Edge(lower.id, Curve(tuple(points)), lower.seam_partner_id))


def m_legs(pattern: Pattern) -> Pattern:
    """Lift the dart tip 15 mm so the legs split (clearance follows)."""
    front = pattern.panels[0]
    leg1 = _edge(front, "front.dart_leg1")
    tip = Point(leg1.curve.points[-1].x, leg1.curve.points[-1].y + 15)
    return _move_legs(pattern, leg1.curve.points[0], tip, _edge(front, "front.dart_leg2").curve.points[-1])


def m_tip(pattern: Pattern) -> Pattern:
    """Move the tip 5 mm toward the base along the centreline."""
    front = pattern.panels[0]
    leg1, leg2 = _edge(front, "front.dart_leg1"), _edge(front, "front.dart_leg2")
    tip = Point(leg1.curve.points[-1].x + 5, leg1.curve.points[-1].y)
    return _move_legs(pattern, leg1.curve.points[0], tip, leg2.curve.points[-1])


def m_intake(pattern: Pattern) -> Pattern:
    """Spread the dart base 3 mm each way, growing the intake."""
    front = pattern.panels[0]
    leg1, leg2 = _edge(front, "front.dart_leg1"), _edge(front, "front.dart_leg2")
    d1 = Point(leg1.curve.points[0].x, leg1.curve.points[0].y - 3)
    d2 = Point(leg2.curve.points[-1].x, leg2.curve.points[-1].y + 3)
    return _move_legs(pattern, d1, leg1.curve.points[-1], d2)


def m_kink(pattern: Pattern) -> Pattern:
    """Shove the tip 120 mm toward the base to kink the closed seam."""
    front = pattern.panels[0]
    leg1, leg2 = _edge(front, "front.dart_leg1"), _edge(front, "front.dart_leg2")
    tip = Point(leg1.curve.points[-1].x + 120, leg1.curve.points[-1].y)
    return _move_legs(pattern, leg1.curve.points[0], tip, leg2.curve.points[-1])


def m_drills(pattern: Pattern) -> Pattern:
    """Drop one back-dart drill from the record."""
    clone = copy.deepcopy(pattern)
    clone.metadata["darts"]["back_waist"]["drills"].pop()
    return clone


def m_notch(pattern: Pattern) -> Pattern:
    """Push a sleeve notch past its edge."""
    clone = copy.deepcopy(pattern)
    notch = clone.panels[2].notches[0]
    edge = next(e for e in clone.panels[2].edges if e.id == notch.edge_id)
    clone.panels[2].notches[0] = Notch(notch.edge_id, edge_length(edge) + 10)
    return clone


def m_shoulder_notch(pattern: Pattern) -> Pattern:
    """Remove the front zero notch from the armhole."""
    clone = copy.deepcopy(pattern)
    front = clone.panels[0]
    front.notches = [n for n in front.notches
                     if not (n.edge_id == "front.armhole" and n.distance_from_start_mm == 0)]
    return clone


def m_grainline(pattern: Pattern) -> Pattern:
    """Tilt the front grainline off the fold."""
    clone = copy.deepcopy(pattern)
    a, b = clone.panels[0].grainline
    clone.panels[0].grainline = (Point(a.x + 5, a.y), b)
    return clone


def _yoke() -> Pattern:
    """Return the pristine M shirt with a back yoke (sample-photo style)."""
    return draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", dict(SHIRT_STYLE, back_yoke=1))


def m_yoke_seam(pattern: Pattern) -> Pattern:
    """Push the lower back's armhole end of the yoke seam out 3 mm, so it outgrows the yoke."""
    return _move_corner(pattern, "back", "back.yoke_seam", "back.armhole", Point(3, 0))


def _pleat() -> Pattern:
    """Return the pristine M yoked shirt with a centre-back box pleat."""
    return draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", dict(SHIRT_STYLE, back_yoke=1, back_pleat=1))


def m_pleat_notch(pattern: Pattern) -> Pattern:
    """Drop the pleat notch on the yoke seam, so the pleat line is no longer marked."""
    clone = copy.deepcopy(pattern)
    back = clone.panels[_index(clone, "back")]
    back.notches = [n for n in back.notches if n.edge_id != "back.yoke_seam"]
    return clone


Mutator = Callable[[Pattern], Pattern]

MUTATIONS: list[tuple[str, str, str, Mutator]] = [
    ("G1", "blouse", "Facing end moved 7 mm", m_closure),
    ("G2", "blouse", "Parameter changed without a redraft", m_params),
    ("S1", "blouse", "Back shoulder shortened 2 mm", m_shoulder),
    ("S2a", "blouse", "Front underarm lowered 8 mm", m_side_above),
    ("S2b", "blouse", "Back hem corner widened 35 mm", m_side_below),
    ("S3", "blouse", "Both armholes widened", m_cap_ease),
    ("S4", "blouse", "Facing neck bowed outward", m_facing),
    ("S5", "blouse", "One sleeve hem corner lowered 5 mm", m_underarm),
    ("S6", "tshirt", "Neckband shortened 5 mm", m_neckband),
    ("W1", "blouse", "Both dart ends narrowed 5 mm", m_bust_join),
    ("W2", "blouse", "Front waist widened 5 mm", m_waist),
    ("W3", "blouse", "Front hip widened 5 mm", m_hip),
    ("D1", "blouse", "Dart tip lifted 15 mm", m_legs),
    ("D2", "blouse", "Dart tip moved 5 mm inward", m_tip),
    ("D3", "blouse", "Dart base widened 3 mm", m_intake),
    ("D4", "blouse", "Dart tip pushed 120 mm", m_kink),
    ("D5", "blouse", "One back drill mark removed", m_drills),
    ("N1", "blouse", "One notch moved off its edge", m_notch),
    ("N2", "blouse", "Front zero notch removed", m_shoulder_notch),
    ("R1", "blouse", "Grainline tilted", m_grainline),
    ("S2", "shirt", "Shirt back hem corner lowered 6 mm", m_shirt_side),
    ("S7", "shirt", "Cuff lengthened 4 mm", m_cuff),
    ("S8", "shirt", "Collar neck end pulled 6 mm", m_collar),
    ("W1", "shirt", "Shirt back underarm widened 5 mm", m_shirt_chest),
    ("W3", "shirt", "Shirt hem narrowed 80 mm per side", m_shirt_hem),
    ("K1", "skirt", "Skirt dart tip moved 10 mm", m_skirt_dart),
    ("K2", "skirt", "Waist facing shortened 5 mm", m_waist_finish),
    ("K3", "skirt", "One hem facing scallop bowed 10 mm", m_hem_facing),
    ("W2", "skirt", "Skirt waist widened 3 mm (facing follows)", m_skirt_waist),
    ("W3", "skirt", "Skirt hip widened 5 mm", m_skirt_hip),
    ("S7", "bishop", "Elastic cuff lengthened 4 mm", m_bishop_cuff),
    ("S9", "bishop", "Bishop sleeve hem widened 100 mm per side", m_bishop_gathers),
    ("K4", "dress", "Dress skirt waist widened 3 mm", m_dress_waist_seam),
    ("D6", "dress", "Front waist dart tip moved 10 mm", m_dress_front_dart),
    ("D5", "dress", "Dress back dart tip moved 20 mm", m_dress_back_dart),
    ("K1", "dress", "Dress skirt dart tip moved 10 mm", m_dress_skirt_dart),
    ("W2", "dress", "Front bodice + skirt waist widened 3 mm", m_dress_waist),
    ("W3", "dress", "Dress skirt hip widened 5 mm", m_dress_hip),
    ("S2b", "dress", "Back skirt hem corner lowered 6 mm", m_dress_skirt_side),
    ("S2a", "dress", "Dress front waist corner lowered 10 mm", m_dress_bodice_side),
    ("S10", "vest", "Armhole bias shortened 5 mm", m_armhole_bias),
    ("S11", "vest", "Neck bias lengthened 5 mm", m_neck_bias),
    ("S3", "blouse", "Blouse sleeve panel removed", m_lost_sleeve),
    ("Y1", "yoke", "Lower back yoke seam widened 3 mm", m_yoke_seam),
    ("Y2", "pleat", "Back pleat notch removed from the yoke seam", m_pleat_notch),
]

BASES: dict[str, Callable[[], Pattern]] = {"blouse": _blouse, "tshirt": _tshirt, "shirt": _shirt, "skirt": _skirt,
                                           "dress": _dress, "bishop": _bishop,
                                           "vest": _vest, "yoke": _yoke, "pleat": _pleat}


def run_mutation_report() -> list[tuple[str, str, set[str]]]:
    """Mutate pristine drafts and return (check, description, failed ids)."""
    rows = []
    for check_id, base, description, mutate in MUTATIONS:
        report = verify_pattern(mutate(BASES[base]()))
        rows.append((check_id, description, {check.id for check in report.failures()}))
    return rows
