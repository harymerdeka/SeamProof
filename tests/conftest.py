"""Shared deterministic pattern fixture, with no LLM or network dependency."""

import pytest

from seamproof.measurements import SIZE_TABLE
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point


@pytest.fixture
def pattern() -> Pattern:
    """Return a rectangular synthetic panel, not a drafted garment."""
    points = [Point(0, 0), Point(100, 0), Point(100, 200), Point(0, 200)]
    edges = [
        Edge(f"front.e{i}", Curve((start, points[(i + 1) % 4])))
        for i, start in enumerate(points)
    ]
    edges[0].seam_partner_id = "back.e0"
    edges[0].allowance_mm = 10
    edges[3].needs_allowance = False
    edges[3].is_fold = True
    panel = Panel(
        "front", edges, (Point(50, 20), Point(50, 180)),
        [Notch("front.e1", 50)], cut_count=1, cut_on_fold=True,
    )
    return Pattern(
        "tshirt", "M", "cotton jersey", SIZE_TABLE["M"].to_dict(),
        {"ease_mm": 40, "nested": {"b": 2, "a": 1}}, [panel],
        {"source": "test", "version": "0.1.0"},
    )
