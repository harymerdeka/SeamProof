"""Allowance geometry, fold boundaries and stitch-line preservation tests."""

import json

import pytest
from shapely.geometry import Polygon

from seamproof.allowance import add_seam_allowance
from seamproof.draft.tshirt import draft_tshirt
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import SIZE_TABLE
from seamproof.params import PARAMETERS
from seamproof.pattern import Curve, Pattern, Point, panel_polygon


def _cut_polygon(pattern: Pattern, index: int) -> Polygon:
    """Return one generated cut polygon for assertions."""
    cut = pattern.panels[index].cut_line
    assert cut is not None and cut.points[0] == cut.points[-1]
    return Polygon([(point.x, point.y) for point in cut.points])


@pytest.mark.parametrize("fabric", FABRIC_EASE)
@pytest.mark.parametrize("size", SIZE_TABLE)
def test_allowance_preserves_stitch_and_creates_closed_cut(size: str, fabric: str) -> None:
    """Stitch lines survive; each cut outline is closed, larger and covering."""
    original = draft_tshirt(SIZE_TABLE[size], fabric)
    before = original.to_dict()
    result = add_seam_allowance(original)
    assert original.to_dict() == before
    for i, (old_panel, panel) in enumerate(zip(original.panels, result.panels)):
        assert panel.grainline == old_panel.grainline
        assert panel.notches == old_panel.notches
        for old_edge, edge in zip(old_panel.edges, panel.edges):
            assert old_edge.curve == edge.curve
            expected = 0 if edge.is_fold else 25 if edge.id.endswith(".hem") else 10
            assert edge.allowance_mm == expected
        cut = _cut_polygon(result, i)
        assert cut.is_valid
        assert cut.area > panel_polygon(panel).area
        assert cut.buffer(1e-7).covers(panel_polygon(panel))
        if panel.cut_on_fold:
            assert cut.bounds[0] == pytest.approx(0)
            assert cut.bounds[3] == pytest.approx(SIZE_TABLE[size].back_length + 25)
    assert Pattern.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    assert add_seam_allowance(result).to_dict() == result.to_dict()


def test_rectangle_exact_allowance_and_no_allowance_flag(pattern: Pattern) -> None:
    """A rectangle offsets exactly; opted-out edges stay on the stitch line."""
    result = add_seam_allowance(pattern)
    # Fixture fold is x=0; all other edges have default allowance.
    assert _cut_polygon(result, 0).bounds == pytest.approx((0, -10, 110, 210))
    pattern.panels[0].edges[0].needs_allowance = False
    result = add_seam_allowance(pattern, default_mm=7, hem_mm=20)
    assert _cut_polygon(result, 0).bounds == pytest.approx((0, 0, 107, 207))


def test_clockwise_outline(pattern: Pattern) -> None:
    """Reversed winding still offsets outward, not inward."""
    panel = pattern.panels[0]
    panel.edges.reverse()
    for edge in panel.edges:
        edge.curve = Curve(tuple(reversed(edge.curve.points)))
    result = add_seam_allowance(pattern)
    assert _cut_polygon(result, 0).bounds == pytest.approx((0, -10, 110, 210))


def test_zero_allowance(pattern: Pattern) -> None:
    """Zero widths reproduce the stitch outline exactly."""
    result = add_seam_allowance(pattern, 0, 0)
    assert _cut_polygon(result, 0).equals(panel_polygon(pattern.panels[0]))


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
@pytest.mark.parametrize("argument", ["default_mm", "hem_mm"])
def test_invalid_allowance(pattern: Pattern, value: float, argument: str) -> None:
    """Negative or non-finite widths are rejected."""
    with pytest.raises(ValueError, match="nonnegative finite"):
        add_seam_allowance(pattern, **{argument: value})


def test_invalid_stitch_polygon_rejected(pattern: Pattern) -> None:
    """A self-intersecting stitch outline cannot produce a cut line."""
    panel = pattern.panels[0]
    # Mutate a closed rectangle to a closed bow tie.
    points = [Point(0, 0), Point(100, 200), Point(100, 0), Point(0, 200)]
    for i, edge in enumerate(panel.edges):
        edge.curve = Curve((points[i], points[(i + 1) % 4]))
    with pytest.raises(ValueError, match="valid stitch-line"):
        add_seam_allowance(pattern)


@pytest.mark.parametrize("name", PARAMETERS)
@pytest.mark.parametrize("bound", ["min", "max"])
def test_parameter_limits_have_valid_cut_lines(name: str, bound: str) -> None:
    """Every parameter limit still yields clean, hole-free cut outlines."""
    draft = draft_tshirt(SIZE_TABLE["M"], "knit_low_stretch", {name: getattr(PARAMETERS[name], bound)})
    result = add_seam_allowance(draft)
    for i, panel in enumerate(result.panels):
        cut = _cut_polygon(result, i)
        assert cut.is_valid and not cut.interiors
        assert cut.buffer(1e-7).covers(panel_polygon(panel))
