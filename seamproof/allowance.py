"""Deterministic outward seam allowances while retaining stitch-line edges."""

import copy
import math

from shapely.geometry import Polygon

from seamproof.pattern import Curve, Panel, Pattern, Point, panel_polygon


def _cross(a: Point, b: Point) -> float:
    """Return the two-dimensional cross product."""
    return a.x * b.y - a.y * b.x


def _offset_join(vertex: Point, before: Point, after: Point, previous_mm: float,
                 next_mm: float, orientation: float) -> list[Point]:
    """Intersect adjacent outward offset lines; bevel excessive mitres.

    Assumption: a mitre limit of five times the larger allowance is suitable
    for this block. Parallel unequal offsets use a bevel transition.
    """
    u = Point(vertex.x - before.x, vertex.y - before.y)
    v = Point(after.x - vertex.x, after.y - vertex.y)
    ul, vl = math.hypot(u.x, u.y), math.hypot(v.x, v.y)
    u, v = Point(u.x / ul, u.y / ul), Point(v.x / vl, v.y / vl)
    a = Point(vertex.x + orientation * u.y * previous_mm, vertex.y - orientation * u.x * previous_mm)
    b = Point(vertex.x + orientation * v.y * next_mm, vertex.y - orientation * v.x * next_mm)
    denominator = _cross(u, v)
    if abs(denominator) < 1e-10:
        return [a] if a == b else [a, b]
    delta = Point(b.x - a.x, b.y - a.y)
    distance = _cross(delta, v) / denominator
    join = Point(a.x + distance * u.x, a.y + distance * u.y)
    limit = 5 * max(previous_mm, next_mm)
    if math.hypot(join.x - vertex.x, join.y - vertex.y) > limit + 1e-8:
        return [a, b]
    return [join]


def _cut_outline(panel: Panel) -> Curve:
    """Offset sampled boundary segments with per-edge widths, in mm."""
    polygon = panel_polygon(panel)
    if not polygon.is_valid or polygon.area == 0:
        raise ValueError("Allowance requires a valid stitch-line polygon")
    vertices: list[Point] = []
    widths: list[float] = []
    for edge in panel.edges:
        for a, b in zip(edge.curve.points, edge.curve.points[1:]):
            if a != b:
                vertices.append(a)
                widths.append(edge.allowance_mm)
    orientation = 1.0 if polygon.exterior.is_ccw else -1.0
    outline: list[Point] = []
    for i, vertex in enumerate(vertices):
        outline.extend(_offset_join(vertex, vertices[i - 1], vertices[(i + 1) % len(vertices)],
                                    widths[i - 1], widths[i], orientation))
    # Concave offsets can overlap when allowance exceeds local curvature radius.
    # A zero-distance buffer trims those overlapping offset loops deterministically;
    # it never repairs the input stitch polygon (validated above). Reject splits,
    # holes or lost stitch area instead of choosing an arbitrary component.
    cut = Polygon([(p.x, p.y) for p in outline]).buffer(0)
    if (cut.geom_type != "Polygon" or not cut.is_valid or cut.area == 0
            or len(cut.interiors) or not cut.buffer(1e-7).covers(polygon)):
        raise ValueError("Allowance offset is invalid; reduce allowance or revise geometry")
    return Curve(tuple(Point(float(x), float(y)) for x, y in cut.exterior.coords))


def add_seam_allowance(pattern: Pattern, default_mm: float = 10, hem_mm: float = 25) -> Pattern:
    """Return a copy with cut lines, assigning hems 25 mm and folds zero.

    Edges ending in '.hem' receive hem_mm. Fold edges and edges explicitly
    marked needs_allowance=False receive zero. Other edges receive default_mm.
    Repeated calls regenerate from stitch lines; allowances never accumulate.
    """
    if any(not math.isfinite(value) or value < 0 for value in (default_mm, hem_mm)):
        raise ValueError("Allowances must be nonnegative finite millimetres")
    result = copy.deepcopy(pattern)
    for panel in result.panels:
        for edge in panel.edges:
            if edge.is_fold or not edge.needs_allowance:
                edge.allowance_mm = 0
            else:
                edge.allowance_mm = hem_mm if edge.id.endswith(".hem") else default_mm
        panel.cut_line = _cut_outline(panel)
    result.metadata["seam_allowance_mm"] = {"default": default_mm, "hem": hem_mm, "fold": 0}
    return result


def ensure_allowance(pattern: Pattern) -> Pattern:
    """Return the pattern with cut lines, adding default allowances if absent.

    Drafts stay stitch-line only; every output that is printed, cut or
    packed passes through here so no piece leaves SeamProof without
    seam allowance. Patterns that already carry cut lines are returned as-is.
    """
    if all(panel.cut_line is not None for panel in pattern.panels):
        return pattern
    return add_seam_allowance(pattern)
