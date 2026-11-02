"""Fit skirt and dress proportions to a sketch, with a deterministic match score.

The model only draws an SVG. Code reads its silhouette, turns it into target
parameters and redrafts; every step passes the usual gates (parameter range,
edit locality, verifier). The correction is computed here and never comes from
the LLM, because skirt_flare and skirt_length are not repair-tunable.

Assumptions to tune with real samples (a pattern maker should review):
- A sketch has no scale, so one known width sets mm per sketch unit: the sewn
  waist, or the bust at the side seam for a shift dress drawn with no waist.
- Only closed filled shapes crossing the centre line form the body; a separate
  sleeve outline is ignored. A sleeve drawn as one outline with the body and
  touching it inflates the widths.
- Dress height runs from the shoulder-neck point; the sketch ignores the extra
  front length taken up by the bust. Flats are often stretched: a visible dress
  waist corrects that, but a skirt or waistless shift has no vertical reference.
- The score compares waist to hem only. Bodice, sleeves and details are not
  compared, and the design size M is used for the fit.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from xml.etree import ElementTree

from shapely.errors import ShapelyError
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from seamproof.draft import REGISTRY
from seamproof.locality import edit_locality
from seamproof.params import PARAMETERS, resolve_params
from seamproof.pattern import Pattern
from seamproof.verify import verify_pattern

FIT_GARMENTS = ("skirt", "dress")
FIT_PARAMS = ("skirt_flare", "skirt_length")
RESIDUAL_TOL_MM = 2.0
MAX_ROUNDS = 5
SCORE_ROWS = 120
CURVE_STEPS = 16
MIN_VIEW_HEIGHT_RATIO = 0.1
MAX_HEIGHT_TO_WAIST = 12.0
LOW_SCORE = 70.0
WAIST_ROW_MM = 0.5
STRAIGHT_RATIO = 0.95  # a shift sketch this close to its bust width at the waist counts as straight

Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}
_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
Pt = tuple[float, float]


class SketchUnreadable(ValueError):
    """The sketch has no silhouette code can measure."""


@dataclass
class Silhouette:
    """Front-view body of a sketch, in sketch units with y growing downward."""

    shape: BaseGeometry
    cx: float
    top: float
    bottom: float


@dataclass
class SketchFit:
    """Fitted proportions, the match before and after, and every logged round."""

    params: dict[str, float]
    score_before: float
    score_after: float
    hem_half_mm: float
    length_mm: float
    residual_mm: dict[str, float]
    verified: bool
    low_confidence: bool
    rounds: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cubic(a: Pt, b: Pt, c: Pt, d: Pt) -> list[Pt]:
    """Flatten a cubic Bezier, excluding its start point."""
    out = []
    for i in range(1, CURVE_STEPS + 1):
        t = i / CURVE_STEPS
        u = 1 - t
        out.append((u**3 * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t**3 * d[0],
                    u**3 * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t**3 * d[1]))
    return out


def _quad(a: Pt, b: Pt, c: Pt) -> list[Pt]:
    """Flatten a quadratic Bezier, excluding its start point."""
    return _cubic(a, (a[0] + 2 / 3 * (b[0] - a[0]), a[1] + 2 / 3 * (b[1] - a[1])),
                  (c[0] + 2 / 3 * (b[0] - c[0]), c[1] + 2 / 3 * (b[1] - c[1])), c)


def _path_subpaths(d: str) -> list[tuple[list[Pt], bool]]:
    """Flatten path data into (points, closed) subpaths.

    Assumption: an arc is taken as a straight chord; the sketch format asks for
    Bezier curves, so arcs should only be small corners.
    """
    tokens = _TOKEN.findall(d)
    out: list[tuple[list[Pt], bool]] = []
    points: list[Pt] = []
    pos: Pt = (0.0, 0.0)
    last_ctrl: Pt | None = None
    last_cmd = ""
    i, cmd = 0, ""
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        elif not cmd or cmd in "Zz":
            raise SketchUnreadable("The sketch has broken path data.")
        upper, rel = cmd.upper(), cmd.islower()
        if upper == "Z":
            if points:
                out.append((points, True))
                pos = points[0]
            points, last_ctrl = [], None
            last_cmd = "Z"
            continue
        args = tokens[i:i + _ARITY[upper]]
        if len(args) < _ARITY[upper] or any(a.isalpha() for a in args):
            raise SketchUnreadable("The sketch has broken path data.")
        v = [float(a) for a in args]
        i += _ARITY[upper]
        ox, oy = pos if rel else (0.0, 0.0)
        if upper == "M":
            if len(points) > 1:
                out.append((points, False))
            pos = (v[0] + ox, v[1] + oy)
            points = [pos]
            cmd = "l" if rel else "L"  # later coordinate pairs are implicit lines
        elif upper in ("L", "T", "A"):
            end = (v[-2] + ox, v[-1] + oy)
            if upper == "T":
                ctrl = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl and last_cmd in "QT" else pos
                points.extend(_quad(pos, ctrl, end))
                last_ctrl = ctrl
            else:
                points.append(end)
            pos = end
        elif upper == "H":
            pos = (v[0] + ox, pos[1])
            points.append(pos)
        elif upper == "V":
            pos = (pos[0], v[0] + oy)
            points.append(pos)
        elif upper == "C":
            c1, c2, end = (v[0] + ox, v[1] + oy), (v[2] + ox, v[3] + oy), (v[4] + ox, v[5] + oy)
            points.extend(_cubic(pos, c1, c2, end))
            pos, last_ctrl = end, c2
        elif upper == "S":
            c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl and last_cmd in "CS" else pos
            c2, end = (v[0] + ox, v[1] + oy), (v[2] + ox, v[3] + oy)
            points.extend(_cubic(pos, c1, c2, end))
            pos, last_ctrl = end, c2
        elif upper == "Q":
            ctrl, end = (v[0] + ox, v[1] + oy), (v[2] + ox, v[3] + oy)
            points.extend(_quad(pos, ctrl, end))
            pos, last_ctrl = end, ctrl
        last_cmd = upper
    if len(points) > 1:
        out.append((points, False))
    return out


def _multiply(a: Matrix, b: Matrix) -> Matrix:
    """Compose two SVG matrices: b is applied first."""
    return (a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
            a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5])


def _transform(text: str | None) -> Matrix:
    """Parse an SVG transform list; an unsupported function makes the sketch unreadable."""
    result = IDENTITY
    for name, body in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        v = [float(n) for n in _NUMBER.findall(body)]
        if name == "matrix" and len(v) == 6:
            m: Matrix = tuple(v)  # type: ignore[assignment]
        elif name == "translate" and v:
            m = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            m = (v[0], 0.0, 0.0, v[1] if len(v) > 1 else v[0], 0.0, 0.0)
        elif name == "rotate" and len(v) == 1:
            c, s = math.cos(math.radians(v[0])), math.sin(math.radians(v[0]))
            m = (c, s, -s, c, 0.0, 0.0)
        else:
            raise SketchUnreadable("The sketch uses a transform code cannot read.")
        result = _multiply(result, m)
    return result


def _shape_subpaths(node: ElementTree.Element, name: str) -> list[tuple[list[Pt], bool]]:
    """Flatten one drawing element into (points, closed) subpaths."""
    num = lambda key: float(_NUMBER.match(node.get(key, "0")).group())  # noqa: E731
    if name == "path":
        return _path_subpaths(node.get("d", ""))
    if name in ("polygon", "polyline"):
        v = [float(n) for n in _NUMBER.findall(node.get("points", ""))]
        return [(list(zip(v[::2], v[1::2])), name == "polygon")]
    if name == "rect":
        x, y, w, h = num("x"), num("y"), num("width"), num("height")
        return [([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], True)]
    if name in ("circle", "ellipse"):
        rx, ry = (num("r"), num("r")) if name == "circle" else (num("rx"), num("ry"))
        return [([(num("cx") + rx * math.cos(k * math.pi / 18), num("cy") + ry * math.sin(k * math.pi / 18))
                  for k in range(36)], True)]
    return []


def _collect(node: ElementTree.Element, matrix: Matrix, fill: str | None,
             out: list[tuple[list[Pt], bool, str | None]]) -> None:
    """Walk the SVG tree, gathering transformed subpaths with their inherited fill."""
    matrix = _multiply(matrix, _transform(node.get("transform")))
    fill = node.get("fill", fill)
    name = node.tag.rsplit("}", 1)[-1]
    a, b, c, d, e, f = matrix
    for points, closed in _shape_subpaths(node, name):
        moved = [(a * x + c * y + e, b * x + d * y + f) for x, y in points]
        if not all(math.isfinite(v) for point in moved for v in point):
            raise SketchUnreadable("The sketch has coordinates code cannot read.")
        out.append((moved, closed, fill))
    for child in node:
        _collect(child, matrix, fill, out)


def _view_box(root: ElementTree.Element, shapes: list[Polygon]) -> tuple[float, float, float, float]:
    """The drawing area from viewBox, or the bounds of the shapes without one."""
    numbers = [float(n) for n in _NUMBER.findall(root.get("viewBox", ""))]
    if len(numbers) == 4 and numbers[2] > 0 and numbers[3] > 0:
        return numbers[0], numbers[1], numbers[2], numbers[3]
    x0, y0, x1, y1 = unary_union(shapes).bounds
    return x0, y0, x1 - x0, y1 - y0


def read_silhouette(svg: str) -> Silhouette:
    """Extract the front-view body silhouette from a sanitised sketch SVG."""
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        raise SketchUnreadable("The sketch is not a usable SVG.") from exc
    raw: list[tuple[list[Pt], bool, str | None]] = []
    try:
        _collect(root, IDENTITY, None, raw)
    except (ValueError, AttributeError, ShapelyError) as exc:
        raise SketchUnreadable("The sketch has shapes code cannot read.") from exc
    shapes, unfilled = [], []
    for points, closed, fill in raw:
        if len(points) < 3:
            continue
        gap = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1])
        x0, y0, x1, y1 = LineString(points).bounds
        if not closed and gap > 0.03 * max(x1 - x0, y1 - y0):
            continue
        polygon = Polygon(points).buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        (unfilled if fill == "none" else shapes).append(polygon)
    shapes = shapes or unfilled
    if not shapes:
        raise SketchUnreadable("The sketch has no closed garment outline.")
    vx, vy, vw, vh = _view_box(root, shapes)
    shapes = [s for s in shapes if s.area < 0.9 * vw * vh]  # drop a background rectangle
    mid = vx + vw / 2
    # Assumption: two views sit side by side, the front on the left. A large
    # shape over the middle of the drawing means there is only one view.
    crossing = any(s.bounds[0] < mid < s.bounds[2] and s.bounds[2] - s.bounds[0] > 0.25 * vw for s in shapes)
    front = shapes if crossing else [s for s in shapes if s.centroid.x < mid]
    if not front:
        raise SketchUnreadable("The sketch has no front view.")
    x0, _, x1, _ = unary_union(front).bounds
    cx = (x0 + x1) / 2
    body = [s for s in front if s.bounds[0] < cx < s.bounds[2]]
    if not body:
        raise SketchUnreadable("No garment outline crosses the centre of the sketch.")
    union = unary_union(body)
    _, top, _, bottom = union.bounds
    if bottom - top < MIN_VIEW_HEIGHT_RATIO * vh:
        raise SketchUnreadable("The garment outline in the sketch is too small to measure.")
    return Silhouette(union, cx, top, bottom)


def half_width(shape: BaseGeometry, cx: float, y: float) -> float:
    """Half the width of the interval around the centre line at height y; 0 when outside."""
    x0, _, x1, _ = shape.bounds
    cut = shape.intersection(LineString([(x0 - 1, y), (x1 + 1, y)]))
    parts = getattr(cut, "geoms", [cut])
    for part in parts:
        if part.is_empty or part.geom_type != "LineString":
            continue
        a, _, b, _ = part.bounds
        if a <= cx <= b:
            return (b - a) / 2
    return 0.0


def _panel(pattern: Pattern, name: str):
    """Fetch a panel by name, or None."""
    return next((panel for panel in pattern.panels if panel.name == name), None)


def _edge_points(panel, suffix: str):
    """Points of the panel edge whose id ends in suffix, or None."""
    edge = next((e for e in panel.edges if e.id.endswith("." + suffix)), None)
    return None if edge is None else edge.curve.points


def _lower(pattern: Pattern) -> tuple[Any, float, bool]:
    """The panel holding waist-to-hem, its waist height, and whether it is a separate skirt."""
    skirt = _panel(pattern, "front") if pattern.garment_type == "skirt" else _panel(pattern, "skirt_front")
    if skirt is not None:
        return skirt, 0.0, True
    return _panel(pattern, "front"), float(pattern.measurements["front_length"]), False


def _hem(panel) -> Any:
    """Hem points from side to centre, straight or scalloped."""
    return _edge_points(panel, "scallop") or _edge_points(panel, "hem")


def pattern_silhouette(pattern: Pattern) -> BaseGeometry:
    """Sewn waist-to-hem silhouette in mm, centred on x=0, with the waist at y=0.

    A skirt dart is closed by moving the side in, tapering out at the dart tip.
    """
    panel, waist_y, separate = _lower(pattern)
    side = _edge_points(panel, "side") if separate else _edge_points(panel, "side_lower")
    points = [(p.x, p.y - waist_y) for p in side]
    if separate:
        leg1, leg2 = _edge_points(panel, "dart_leg1"), _edge_points(panel, "dart_leg2")
        intake, tip = leg2[-1].x - leg1[0].x, leg1[-1].y
        points = [(x - intake * max(0.0, 1 - y / tip), y) for x, y in points]
    hem = [(p.x, p.y - waist_y) for p in _hem(panel)]
    half = [(0.0, points[0][1])] + points + hem[1:]
    outline = half + [(-x, y) for x, y in reversed(half[1:-1])]
    shape = Polygon(outline).buffer(0)
    return shape.intersection(box(-1e5, 0.0, 1e5, 1e5))


def pattern_hem(pattern: Pattern) -> tuple[float, float]:
    """(hem half width, centre length below the waist) of a pattern, in mm."""
    panel, waist_y, _separate = _lower(pattern)
    hem = _hem(panel)
    return max(p.x for p in hem), hem[-1].y - waist_y


def _waist_half(pattern: Pattern) -> float:
    """Sewn half width at the waist; constant under flare and length."""
    return half_width(pattern_silhouette(pattern), 0.0, WAIST_ROW_MM)


def _bust_half(pattern: Pattern) -> float:
    """Half width of the front at the bust side seam, for a dress."""
    return max(p.x for p in _edge_points(_panel(pattern, "front"), "side_upper"))


@dataclass
class _Target:
    """What the sketch asks for in mm, plus how to map sketch rows onto the pattern."""

    hem_half: float
    length: float
    scale: float  # mm per sketch unit across
    scale_y: float  # mm per sketch unit down
    waist_y: float


def _narrowest(silhouette: Silhouette, start: float, stop: float) -> float:
    """Smallest non-zero half width between two height fractions."""
    height = silhouette.bottom - silhouette.top
    widths = [half_width(silhouette.shape, silhouette.cx, silhouette.top + height * (start + (stop - start) * i / 30))
              for i in range(31)]
    widths = [w for w in widths if w > 0]
    if not widths:
        raise SketchUnreadable("The sketch has no measurable waist.")
    return min(widths)


def _narrowest_y(silhouette: Silhouette, start: float, stop: float) -> float:
    """Height of the narrowest row between two height fractions."""
    height = silhouette.bottom - silhouette.top
    rows = [silhouette.top + height * (start + (stop - start) * i / 30) for i in range(31)]
    widths = [(half_width(silhouette.shape, silhouette.cx, y), y) for y in rows]
    return min((w, y) for w, y in widths if w > 0)[1]


def _top_half(silhouette: Silhouette) -> float:
    """Half width at the top edge, extrapolated from just below it to skip rounded corners."""
    height = silhouette.bottom - silhouette.top
    near = half_width(silhouette.shape, silhouette.cx, silhouette.top + 0.02 * height)
    far = half_width(silhouette.shape, silhouette.cx, silhouette.top + 0.06 * height)
    if near <= 0 or far <= 0:
        raise SketchUnreadable("The sketch has no measurable waist.")
    return max(near - (far - near) / 2, near / 2)


def read_target(silhouette: Silhouette, pattern: Pattern) -> _Target:
    """Convert sketch proportions to mm using the pattern's own known width."""
    height = silhouette.bottom - silhouette.top
    is_dress = pattern.garment_type == "dress"
    reference = _narrowest(silhouette, 0.25, 0.60) if is_dress else _top_half(silhouette)
    known = _waist_half(pattern)
    if is_dress and pattern.metadata.get("construction") == "shift":
        # A straight shift sketch has no waist to measure: use its width at the bust instead.
        upper = half_width(silhouette.shape, silhouette.cx, silhouette.top + 0.25 * height)
        if reference >= STRAIGHT_RATIO * upper:
            reference, known = upper, _bust_half(pattern)
    if height / (2 * reference) > MAX_HEIGHT_TO_WAIST:
        raise SketchUnreadable("The sketch is too narrow to be this garment.")
    scale = known / reference
    band = pattern.params["waistband_height"] if _panel(pattern, "waistband") is not None else 0.0
    scale_y, waist_y = scale, silhouette.top + band / scale
    if is_dress:
        # Flats are often drawn taller than life. A visible waist gives the dress its own
        # vertical scale; otherwise the across scale is the only one there is.
        drawn_waist = _narrowest_y(silhouette, 0.25, 0.60)
        if reference < STRAIGHT_RATIO * half_width(silhouette.shape, silhouette.cx, silhouette.top + 0.25 * height):
            scale_y = float(pattern.measurements["front_length"]) / (drawn_waist - silhouette.top)
            waist_y = drawn_waist
        else:
            waist_y = silhouette.top + float(pattern.measurements["front_length"]) / scale
    if waist_y >= silhouette.bottom:
        raise SketchUnreadable("The sketch ends above the waist, so it is not this dress.")
    hem_half = max(half_width(silhouette.shape, silhouette.cx, silhouette.bottom - height * 0.1 * i / 10 - 1e-6)
                   for i in range(11)) * scale
    return _Target(hem_half, (silhouette.bottom - waist_y) * scale_y, scale, scale_y, waist_y)


def match_score(silhouette: Silhouette, target: _Target, pattern: Pattern) -> float:
    """Overlap of sketch and pattern from waist to hem, 0..100 (intersection over union of rows)."""
    shape = pattern_silhouette(pattern)
    depth = max(target.length, shape.bounds[3])
    common, total = 0.0, 0.0
    for i in range(SCORE_ROWS):
        y = depth * (i + 0.5) / SCORE_ROWS
        drawn = half_width(silhouette.shape, silhouette.cx, target.waist_y + y / target.scale_y) * target.scale
        made = half_width(shape, 0.0, y)
        common += min(drawn, made)
        total += max(drawn, made)
    return round(100 * common / total, 1) if total else 0.0


def _bounded(name: str, value: float) -> float:
    """Clamp a value to its parameter range, in whole millimetres."""
    spec = PARAMETERS[name]
    return float(round(min(spec.max, max(spec.min, value))))


def fit_to_sketch(svg: str, garment: str, measurements: Any, fabric: str,
                  params: Mapping[str, float] | None = None, max_rounds: int = MAX_ROUNDS) -> SketchFit:
    """Move flare and length toward the sketch, passing every gate each round.

    A refused step is halved and retried. Only a step that raises the score is
    kept, so the result is never worse than the categories it started from.
    """
    if garment not in FIT_GARMENTS:
        raise ValueError(f"Sketch proportions are not supported for {garment}.")
    silhouette = read_silhouette(svg)
    draft = REGISTRY[garment]
    current = resolve_params(params)
    pattern = draft(measurements, fabric, current)
    report = verify_pattern(pattern)
    target = read_target(silhouette, pattern)
    notes: list[str] = []
    rounds: list[dict[str, Any]] = []
    step = 1.0
    # A length past the limits means the drawing is stretched, not that the garment is that long.
    spec = PARAMETERS["skirt_length"]
    ignore_length = not spec.min <= target.length <= spec.max
    if ignore_length:
        notes.append(f"The sketch implies a {target.length:.0f} mm length, outside the {spec.min:.0f}-{spec.max:.0f} mm "
                     "limit, so it looks stretched: the length stays at the category value.")
        # Compare shapes as if the drawing were the current length tall.
        target.length = current["skirt_length"]
        target.scale_y = target.length / (silhouette.bottom - target.waist_y)
    before = score = match_score(silhouette, target, pattern)

    def residual(p: Pattern) -> tuple[float, float]:
        hem_half, length = pattern_hem(p)
        return target.hem_half - hem_half, target.length - length

    for number in range(1, max_rounds + 1):
        hem_gap, length_gap = residual(pattern)
        if abs(hem_gap) <= RESIDUAL_TOL_MM and abs(length_gap) <= RESIDUAL_TOL_MM:
            break
        proposed = {"skirt_flare": _bounded("skirt_flare", current["skirt_flare"] + step * hem_gap),
                    "skirt_length": _bounded("skirt_length", current["skirt_length"] + step * length_gap)}
        changes = {k: v for k, v in proposed.items() if v != current[k]}
        log: dict[str, Any] = {"round": number, "proposed": changes, "decision": "", "reason": "",
                               "score": score, "locality_passed": None}
        rounds.append(log)
        if not changes:
            log.update(decision="stopped", reason="The parameter limits or whole-mm steps leave nothing to change.")
            break
        refused = ""
        try:
            candidate = draft(measurements, fabric, {**current, **changes})
        except ValueError as exc:
            refused = f"redraft failed: {exc}"
        else:
            locality = edit_locality(pattern, candidate, changes)
            log["locality_passed"] = locality.passed
            result = verify_pattern(candidate)
            added = sorted({c.id for c in result.failures()} - {c.id for c in report.failures()})
            new_score = match_score(silhouette, target, candidate)
            if not locality.passed:
                refused = locality.message
            elif added:
                refused = f"adds failed checks: {', '.join(added)}"
            elif new_score <= score:
                refused = f"score would not improve ({new_score} after {score})"
        if refused:
            log.update(decision="rejected", reason=refused)
            step /= 2
            continue
        current.update(changes)
        pattern, report, score = candidate, result, new_score
        log.update(decision="accepted", reason="closer to the sketch", score=score)
    hem_gap, length_gap = residual(pattern)
    for name, gap in (("skirt_flare", hem_gap), ("skirt_length", length_gap)):
        asked = current[name] + gap
        spec = PARAMETERS[name]
        if abs(gap) > RESIDUAL_TOL_MM and not spec.min <= asked <= spec.max:
            notes.append(f"The sketch asks for {name} {asked:.0f} mm, outside the {spec.min:.0f}-{spec.max:.0f} mm "
                         f"limit: stopped at {current[name]:.0f} mm.")
        elif abs(gap) > RESIDUAL_TOL_MM:
            notes.append(f"{name} stayed {abs(gap):.0f} mm from the sketch: a closer value failed a check.")
    low = score < LOW_SCORE
    if low:
        notes.append("Low match: the sketch may show a different garment or an unusual shape. The pattern was not changed.")
    return SketchFit({name: current[name] for name in FIT_PARAMS}, before, score, target.hem_half, target.length,
                     {"hem_half": round(hem_gap, 1), "length": round(length_gap, 1)}, report.passed, low, rounds, notes)
