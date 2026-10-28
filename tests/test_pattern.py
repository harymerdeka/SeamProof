"""Data model, geometry and serialization tests."""

import copy
import hashlib
import json

import pytest

from seamproof.pattern import Curve, Edge, Pattern, Point, edge_length, panel_polygon, params_hash


def test_edge_length() -> None:
    """A 3-4-5 triangle chain measures 8 mm."""
    edge = Edge("test", Curve((Point(0, 0), Point(3, 4), Point(6, 4))))
    assert edge_length(edge) == 8


def test_straight_bezier_density_and_endpoints() -> None:
    """A straight 30 mm cubic samples at one point per mm with exact ends."""
    curve = Curve.from_cubic_bezier(Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0))
    assert len(curve.points) == 31
    assert curve.points[0] == Point(0, 0)
    assert curve.points[-1] == Point(30, 0)
    assert edge_length(Edge("line", curve)) == pytest.approx(30)
    assert all(b.x - a.x == pytest.approx(1) for a, b in zip(curve.points, curve.points[1:]))


def test_curved_bezier_length_and_determinism() -> None:
    """A curved cubic is deterministic and close to its known length."""
    controls = (Point(0, 0), Point(0, 100), Point(100, 100), Point(100, 0))
    curve = Curve.from_cubic_bezier(*controls)
    assert curve == Curve.from_cubic_bezier(*controls)
    assert edge_length(Edge("cubic", curve)) == pytest.approx(200, abs=0.01)
    assert len(Curve.from_cubic_bezier(*controls, samples_per_mm=2).points) > len(curve.points)


def test_zero_length_bezier() -> None:
    """Coincident controls collapse to the shared endpoints."""
    point = Point(1, 2)
    assert Curve.from_cubic_bezier(point, point, point, point).points == (point, point)


@pytest.mark.parametrize("density", [0, -1, float("nan"), float("inf")])
def test_invalid_sample_density(density: float) -> None:
    """Non-positive or non-finite density is rejected."""
    with pytest.raises(ValueError, match="density"):
        Curve.from_cubic_bezier(*(Point(0, 0),) * 4, samples_per_mm=density)


def test_curve_requires_finite_points() -> None:
    """NaN coordinates and single-point curves are rejected."""
    with pytest.raises(ValueError, match="finite"):
        Curve((Point(0, 0), Point(float("nan"), 0)))
    with pytest.raises(ValueError, match="two points"):
        Curve((Point(0, 0),))


def test_polygon_closure(pattern: Pattern) -> None:
    """Four ordered edges produce the expected 100x200 rectangle."""
    polygon = panel_polygon(pattern.panels[0])
    assert polygon.exterior.is_closed
    assert polygon.is_valid
    assert polygon.area == 20000
    assert polygon.bounds == (0, 0, 100, 200)


@pytest.mark.parametrize("edge_index", [1, 3])
def test_mutated_gap_or_open_outline_is_rejected(pattern: Pattern, edge_index: int) -> None:
    """Moving one edge endpoint breaks continuity and must fail loudly."""
    panel = pattern.panels[0]
    assert panel_polygon(panel).is_valid
    edge = panel.edges[edge_index]
    edge.curve = Curve((edge.curve.points[0], Point(1, 1)))
    with pytest.raises(ValueError, match="continuous closed outline"):
        panel_polygon(panel)


def test_missing_or_degenerate_outline(pattern: Pattern) -> None:
    """Empty and point-only outlines are rejected."""
    panel = pattern.panels[0]
    panel.edges = []
    with pytest.raises(ValueError, match="boundary edges"):
        panel_polygon(panel)
    panel.edges = [Edge("zero", Curve((Point(0, 0), Point(0, 0))))]
    with pytest.raises(ValueError, match="three distinct points"):
        panel_polygon(panel)


def test_json_round_trip(pattern: Pattern) -> None:
    """Serialization preserves types and survives a JSON round trip."""
    serialized = json.loads(json.dumps(pattern.to_dict()))
    original = copy.deepcopy(serialized)
    restored = Pattern.from_dict(serialized)
    assert restored == pattern
    assert restored.to_dict() == serialized
    assert serialized == original
    assert isinstance(restored.panels[0].edges[0].curve.points[0], Point)
    assert restored.metadata["params_hash"] == params_hash(restored.params)


def test_serialized_data_is_independent_and_hash_is_current(pattern: Pattern) -> None:
    """to_dict returns a copy and refreshes the hash after param edits."""
    data = pattern.to_dict()
    data["params"]["nested"]["a"] = 99
    assert pattern.params["nested"]["a"] == 1
    previous = pattern.metadata["params_hash"]
    pattern.params["ease_mm"] = 50
    assert pattern.to_dict()["metadata"]["params_hash"] != previous


def test_default_metadata(pattern: Pattern) -> None:
    """Patterns without metadata gain the version and current hash."""
    data = pattern.to_dict()
    data.pop("metadata")
    restored = Pattern.from_dict(data)
    assert restored.metadata["version"] == "0.1.0"
    assert restored.metadata["params_hash"] == params_hash(pattern.params)


def test_params_hash_stability() -> None:
    """Key order does not change the hash; value changes do."""
    first = {"z": 4, "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "z": 4}
    assert params_hash(first) == params_hash(second)
    canonical = b'{"nested":{"a":1,"b":2},"z":4}'
    assert params_hash(first) == hashlib.sha256(canonical).hexdigest()
    assert params_hash(first) != params_hash({**first, "z": 5})


def test_hash_rejects_non_json_numbers() -> None:
    """NaN params cannot be hashed into a canonical fingerprint."""
    with pytest.raises(ValueError):
        params_hash({"ease_mm": float("nan")})
