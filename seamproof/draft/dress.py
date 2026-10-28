"""Women's woven dress: the blouse bodice block joined to the skirt block, or a one-piece shift.

Two constructions, chosen by dress_waist_seam (assumptions to tune with
samples; a pattern maker should review):
- Waist seam (1): the bodice block is cut at the waist. Front keeps its side
  bust dart and gains a waist dart under the apex, its tip 25 mm below the
  apex; back gains a waist dart at mid half-back, 140 mm long but
  ending at least 15 mm below the bust line. Both darts take the
  bodice from fabric ease down to the skirt's 10 mm waist ease, so the bodice
  and skirt waists are sewn to the same length. The skirt halves are the
  skirt block (one dart each, flare, optional faced scallop hem) renamed
  skirt_front/skirt_back. No waistband or waist facing: the bodice is the
  waist finish.
- Shift (0): the bodice block lengthened to skirt_length below the front
  waist with skirt_flare per side at the hem, like the tunic without slits.
  A scallop hem needs the waist-seam construction.
Closure for both is an invisible zip in the left side seam, so front and back
stay cut on the fold. The sleeve is the blouse sleeve shortened to
dress_sleeve_length along its underarm lines (never longer than the arm).
No collar, cuff, pocket or front opening is drafted; seamproof.style reports
those as notes. No seam allowances are included here.
"""

from __future__ import annotations

from typing import Mapping

from seamproof.draft.bodice import BACK_DART_LENGTH, _assemble
from seamproof.draft.skirt import WAIST_EASE_MM, _half, _hem_facing
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WomenMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point

FRONT_WAIST_DART_GAP_MM = 25.0  # front waist dart tip below the bust apex
MIN_SLEEVE_BELOW_CAP_MM = 20.0
BACK_DART_BELOW_BUST_MM = 15.0  # back waist dart tip at least this far below the bust line


def _line(edge_id: str, start: Point, end: Point, partner: str | None = None, fold: bool = False) -> Edge:
    """Create a straight stitch-line edge without allowance."""
    return Edge(edge_id, Curve((start, end)), partner, needs_allowance=not fold, is_fold=fold)


def _edge(panel: Panel, edge_id: str) -> Edge:
    """Fetch one panel edge by id."""
    return next(edge for edge in panel.edges if edge.id == edge_id)


def _waist_edges(name: str, side: Point, centre_y: float, dart_x: float, intake: float,
                 tip_y: float) -> list[Edge]:
    """Waist edge from the side to centre with an open waist dart (bodice bottom)."""
    if intake <= 0:
        raise ValueError(f"{name}: bodice waist already at or below the skirt waist — no dart room")
    near, far = Point(dart_x + intake / 2, centre_y), Point(dart_x - intake / 2, centre_y)
    if not 0 < far.x < near.x < side.x:
        raise ValueError(f"{name}: waist dart does not fit between centre and side")
    tip = Point(dart_x, tip_y)
    return [_line(f"{name}.waist_side", side, near, "dress.waist"),
            _line(f"{name}.wdart_leg1", near, tip, f"{name}.wdart_leg2"),
            _line(f"{name}.wdart_leg2", tip, far, f"{name}.wdart_leg1"),
            _line(f"{name}.waist_cf", far, Point(0, centre_y), "dress.waist")]


def _cut_front(front: Panel, m: WomenMeasurements, target: float) -> tuple[Panel, float]:
    """Cut the hip-length bodice front at the waist and add the front waist dart."""
    lower_side = _edge(front, "front.side_lower").curve.points
    lower, waist = lower_side[0], lower_side[1]
    neck = _edge(front, "front.fold").curve.points[-1]
    intake = waist.x - target
    waist_edges = _waist_edges("front", waist, waist.y, m.apex_distance / 2, intake,
                               m.apex_height + FRONT_WAIST_DART_GAP_MM)
    edges = []
    for edge in front.edges:
        if edge.id == "front.side_lower":
            edges.append(Edge(edge.id, Curve((lower, waist)), edge.seam_partner_id))
            edges += waist_edges
        elif edge.id not in ("front.hem", "front.fold"):
            edges.append(edge)
    edges.append(_line("front.fold", Point(0, waist.y), neck, fold=True))
    notches = list(front.notches) + [Notch("front.waist_side", 0)]
    grain = (front.grainline[0], Point(front.grainline[0].x, waist.y - 20))
    return Panel("front", edges, grain, notches, front.cut_count, front.cut_on_fold), intake


def _cut_back(back: Panel, m: WomenMeasurements, target: float) -> tuple[Panel, float]:
    """Cut the hip-length bodice back at the waist and open its waist dart there."""
    side_waist = _edge(back, "back.side_upper").curve.points[-1]
    neck = _edge(back, "back.fold").curve.points[-1]
    intake = side_waist.x - target
    # The tip stays below the bust line so the back keeps its full bust width there.
    tip_y = max(side_waist.y - BACK_DART_LENGTH, m.apex_height + BACK_DART_BELOW_BUST_MM)
    waist_edges = _waist_edges("back", side_waist, side_waist.y, side_waist.x / 2, intake, tip_y)
    edges = []
    for edge in back.edges:
        if edge.id == "back.side_upper":
            edges.append(edge)
            edges += waist_edges
        elif edge.id in ("back.neckline", "back.shoulder", "back.armhole"):
            edges.append(edge)
    edges.append(_line("back.fold", Point(0, side_waist.y), neck, fold=True))
    notches = list(back.notches) + [Notch("back.waist_side", 0)]
    grain = (back.grainline[0], Point(back.grainline[0].x, side_waist.y - 20))
    return Panel("back", edges, grain, notches, back.cut_count, back.cut_on_fold), intake


def _shorten_sleeve(sleeve: Panel, length: float) -> Panel:
    """Cut the sleeve at length below the crown along its tapering underarm lines."""
    back_ua, hem, front_ua = (_edge(sleeve, f"sleeve.{n}") for n in ("underarm_back", "hem", "underarm_front"))
    (bx, by), (rx, ry) = ((p.x, p.y) for p in back_ua.curve.points)
    full = ry
    if length >= full:
        return sleeve
    if length < by + MIN_SLEEVE_BELOW_CAP_MM:
        raise ValueError("Dress sleeve is shorter than its cap")
    t = (length - by) / (ry - by)
    right = Point(bx + (rx - bx) * t, length)
    left = Point(-right.x, length)
    front = front_ua.curve.points[-1]
    back = back_ua.curve.points[0]
    edges = []
    for edge in sleeve.edges:
        if edge.id == "sleeve.underarm_back":
            edges.append(Edge(edge.id, Curve((back, right)), edge.seam_partner_id))
        elif edge.id == "sleeve.hem":
            edges.append(Edge(edge.id, Curve((right, left)), edge.seam_partner_id))
        elif edge.id == "sleeve.underarm_front":
            edges.append(Edge(edge.id, Curve((left, front)), edge.seam_partner_id))
        else:
            edges.append(edge)
    grain = (sleeve.grainline[0], Point(0, length - 20))
    return Panel("sleeve", edges, grain, list(sleeve.notches), sleeve.cut_count, sleeve.cut_on_fold)


def _rename(panel: Panel, old: str, new: str, partners: Mapping[str, str]) -> Panel:
    """Rename a skirt panel and its edge ids; remap seam partners."""
    def rid(edge_id: str | None) -> str | None:
        if edge_id is None:
            return None
        if edge_id in partners:
            return partners[edge_id]
        for prefix in ("front.", "back."):
            if edge_id.startswith(prefix):
                return "skirt_" + edge_id
        return edge_id

    edges = [Edge(rid(e.id) if e.id.startswith(old + ".") else e.id, e.curve, rid(e.seam_partner_id),
                  e.needs_allowance, e.allowance_mm, e.is_fold) for e in panel.edges]
    notches = [Notch(rid(n.edge_id) if n.edge_id.startswith(old + ".") else n.edge_id, n.distance_from_start_mm)
               for n in panel.notches]
    return Panel(new if panel.name == old else panel.name, edges, panel.grainline, notches,
                 panel.cut_count, panel.cut_on_fold)


def draft_dress(measurements: WomenMeasurements | Mapping[str, float], fabric: str,
                params: Mapping[str, float] | None = None) -> Pattern:
    """Draft a waist-seam dress (bodice + skirt) or a one-piece shift dress."""
    m = measurements if isinstance(measurements, WomenMeasurements) else WomenMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    p = resolve_params(params)
    ease = FABRIC_EASE[fabric].default_ease_mm
    hip_half = (m.hip + ease) / 4
    sleeve_length = min(p["dress_sleeve_length"], m.sleeve_length)
    if p["dress_waist_seam"] < 0.5:
        if p["scallop_depth"] > 0:
            raise ValueError("A scallop hem needs the waist-seam dress")
        pattern, _ = _assemble("dress", m, fabric, p, m.front_length + p["skirt_length"],
                               hip_half + p["skirt_flare"], 0, 0, True, "none")
        pattern.panels = [_shorten_sleeve(panel, sleeve_length) if panel.name == "sleeve" else panel
                          for panel in pattern.panels]
        pattern.metadata["construction"] = "shift"
        pattern.metadata["sleeve_length_mm"] = sleeve_length
        pattern.metadata["assumptions"] += (" Shift dress: bodice block lengthened to skirt_length below the "
                                            "front waist; left side invisible zip; no collar or opening.")
        return pattern

    hip_y = m.front_length + m.waist_to_hip
    bodice, parts = _assemble("dress", m, fabric, p, hip_y, hip_half, 0, 0, True, "none")
    target = (m.waist + WAIST_EASE_MM) / 4
    front, front_intake = _cut_front(parts["front"], m, target)
    back, back_intake = _cut_back(parts["back"], m, target)
    by_name = {panel.name: panel for panel in bodice.panels}
    sleeve = _shorten_sleeve(by_name["sleeve"], sleeve_length)
    skirt_front, f_info = _half("front", m, p)
    skirt_back, b_info = _half("back", m, p)
    extra = []
    if p["scallop_depth"] > 0:
        extra.append(_rename(_hem_facing(skirt_front), "hem_facing", "hem_facing",
                             {"front.scallop": "skirt_front.scallop"}))
    panels = [front, back, sleeve, by_name["facing"],
              _rename(skirt_front, "front", "skirt_front", {}),
              _rename(skirt_back, "back", "skirt_back", {})] + extra
    for panel in panels[4:6]:
        for edge in panel.edges:
            if edge.seam_partner_id == "waist.finish":
                edge.seam_partner_id = "dress.waist"
    metadata = dict(bodice.metadata)
    metadata["darts"] = {
        "front_bust": bodice.metadata["darts"]["front_bust"],
        "front_waist": {"intake_mm": front_intake},
        "back_waist": {"intake_mm": back_intake, "drills": []},
        "skirt": {"front_intake_mm": f_info["intake"], "back_intake_mm": b_info["intake"]},
    }
    metadata.update({
        "construction": "waist_seam",
        "ease_mm": {"bust": ease, "waist": WAIST_EASE_MM, "hip": p["skirt_hip_ease"]},
        "waist_seam_mm": 4 * target,
        "sleeve_length_mm": sleeve_length,
        "assumptions": ("Bodice block cut at the waist with front/back waist darts down to the skirt waist "
                        "(10 mm ease); skirt block halves; left side invisible zip; no collar or opening."),
    })
    return Pattern("dress", bodice.size_label, fabric, m.to_dict(), p, panels, metadata)
