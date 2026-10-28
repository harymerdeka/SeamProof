"""Provisional set-in-sleeve T-shirt block, using deterministic mm geometry.

Assumptions to tune with samples: equal front/back circumference allocation;
waist halfway between chest and hem; sleeve bicep ease 40 mm; straight sleeve
underarms; a 40 mm unfolded neckband at 1:1 neckline length. Woven is a draft
option only: head access may require a closure not supplied by this block.
Seam-length compatibility is not certified by drafting; a verifier must gate
production use. No seam allowances are included here.
"""

from typing import Mapping

from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import SIZE_TABLE, TShirtMeasurements
from seamproof.params import resolve_params
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length


def _line(edge_id: str, start: Point, end: Point, partner: str | None = None, fold: bool = False) -> Edge:
    """Create a straight stitch-line edge without allowance."""
    return Edge(edge_id, Curve((start, end)), partner, needs_allowance=not fold, is_fold=fold)


def _body(name: str, m: TShirtMeasurements, ease: float, p: dict[str, float]) -> Panel:
    """Draft a half-body with exact shared endpoints and an explicit fold."""
    length = m.back_length + p["body_length_adjust"]
    depth = m.armhole_depth
    drop = p[f"neck_drop_{name}"]
    neck = Point(0, drop)
    neck_side = Point(m.neck_width / 2, 0)
    shoulder = Point(m.shoulder_width / 2, p["shoulder_slope"])
    underarm = Point((m.chest + ease) / 4, depth)
    waist = Point((m.waist + ease) / 4, (depth + length) / 2)
    hip = Point((m.hip + ease) / 4, length)
    bottom = Point(0, length)
    if not 0 < drop < depth < length or not 0 < neck_side.x < shoulder.x < underarm.x:
        raise ValueError("Body measurements do not fit this T-shirt block")
    if shoulder.y >= depth:
        raise ValueError("Shoulder drop must be less than armhole depth")
    other = "back" if name == "front" else "front"
    neckline = Edge(f"{name}.neckline", Curve.from_cubic_bezier(
        neck, Point(neck_side.x * 0.55, drop), Point(neck_side.x, drop * 0.55), neck_side,
    ), "neckband.attach")
    armhole = Edge(f"{name}.armhole", Curve.from_cubic_bezier(
        shoulder,
        Point(shoulder.x - p["armhole_curve_depth"], shoulder.y + (depth - shoulder.y) * 0.55),
        Point(underarm.x - p["armhole_curve_depth"], depth), underarm,
    ), "sleeve.cap")
    edges = [
        neckline, _line(f"{name}.shoulder", neck_side, shoulder, f"{other}.shoulder"),
        armhole,
        Edge(f"{name}.side", Curve((underarm, waist, hip)), f"{other}.side"),
        _line(f"{name}.hem", hip, bottom), _line(f"{name}.fold", bottom, neck, fold=True),
    ]
    arm_length = edge_length(armhole)
    notches = [Notch(armhole.id, 0)]  # Shoulder point at start of armhole.
    if name == "front":
        notches.append(Notch(armhole.id, arm_length * 0.6))
    else:
        notches.extend(Notch(armhole.id, arm_length * 0.55 + offset) for offset in (-3, 3))
    return Panel(name, edges, (Point(20, depth), Point(20, length - 20)), notches, cut_on_fold=True)


def _sleeve(m: TShirtMeasurements, p: dict[str, float]) -> Panel:
    """Draft a full sleeve; cap order runs front underarm to back underarm."""
    width = (m.bicep + 40 + p["sleeve_width_adjust"]) / 2
    height = p["sleeve_cap_height"]
    length = m.sleeve_length
    if width <= 0 or length <= height:
        raise ValueError("Sleeve length must exceed cap height and width must be positive")
    front, crown, back = Point(-width, height), Point(0, 0), Point(width, height)
    front_cap = Curve.from_cubic_bezier(front, Point(-width * 0.8, height * 0.5), Point(-width * 0.45, 0), crown)
    back_cap = Curve.from_cubic_bezier(crown, Point(width * 0.55, 0), Point(width * 0.85, height * 0.5), back)
    cap = Edge("sleeve.cap", Curve(front_cap.points + back_cap.points[1:]), "body.armholes")
    front_length = edge_length(Edge("front_cap", front_cap))
    back_length = edge_length(Edge("back_cap", back_cap))
    left_hem, right_hem = Point(-width, length), Point(width, length)
    edges = [cap, _line("sleeve.underarm_back", back, right_hem, "sleeve.underarm_front"),
             _line("sleeve.hem", right_hem, left_hem),
             _line("sleeve.underarm_front", left_hem, front, "sleeve.underarm_back")]
    notches = [Notch(cap.id, front_length), Notch(cap.id, front_length * 0.4)]
    notches.extend(Notch(cap.id, front_length + back_length * 0.55 + offset) for offset in (-3, 3))
    return Panel("sleeve", edges, (Point(0, 20), Point(0, length - 20)), notches, cut_count=2)


def _neckband(front: Panel, back: Panel) -> Panel:
    """Draft an unfolded doubled strip; its two long raw edges join together."""
    length = 2 * (edge_length(front.edges[0]) + edge_length(back.edges[0]))
    a, b, c, d = Point(0, 0), Point(length, 0), Point(length, 40), Point(0, 40)
    edges = [_line("neckband.attach", a, b, "body.neckline"),
             _line("neckband.end_right", b, c, "neckband.end_left"),
             _line("neckband.attach_inner", c, d, "body.neckline"),
             _line("neckband.end_left", d, a, "neckband.end_right")]
    # The ring join is centre back (0/length); the halfway point is centre front.
    notches = [Notch(edge.id, distance) for edge in (edges[0], edges[2]) for distance in (0, length / 2, length)]
    return Panel("neckband", edges, (Point(10, 20), Point(length - 10, 20)), notches)


def draft_tshirt(
    measurements: TShirtMeasurements | Mapping[str, float],
    fabric: str,
    params: Mapping[str, float] | None = None,
) -> Pattern:
    """Draft four stitch-line panels; no LLM, allowance or verifier is invoked."""
    m = measurements if isinstance(measurements, TShirtMeasurements) else TShirtMeasurements(**measurements)
    if fabric not in FABRIC_EASE:
        raise ValueError("Unknown fabric category")
    p = resolve_params(params)
    ease = FABRIC_EASE[fabric].default_ease_mm
    front, back = (_body(name, m, ease, p) for name in ("front", "back"))
    sleeve, band = _sleeve(m, p), _neckband(front, back)
    size = next((name for name, values in SIZE_TABLE.items() if values == m), "custom")
    return Pattern("tshirt", size, fabric, m.to_dict(), p, [front, back, sleeve, band], {
        "status": "draft_unverified", "ease_mm": {name: ease for name in ("chest", "waist", "hip")},
        "chest_line_y_mm": m.armhole_depth,
        # Composite order follows the sleeve cap: front underarm -> shoulder -> back underarm.
        "seam_groups": {
            "body.armholes": [{"edge_id": "front.armhole", "reverse": True}, {"edge_id": "back.armhole", "reverse": False}],
            # Half-body necklines occur twice after unfolding; ring join is centre back.
            "body.neckline": [{"edge_id": "back.neckline", "reverse": False}, {"edge_id": "front.neckline", "reverse": True},
                              {"edge_id": "front.neckline", "reverse": False}, {"edge_id": "back.neckline", "reverse": True}],
        },
        "neckband_fold_line_mm": [[0, 20], [edge_length(band.edges[0]), 20]],
        "assumptions": "Provisional block and ease; tune with real samples. Neckband ratio 1:1; bicep ease 40 mm.",
    })
