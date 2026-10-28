"""Pattern data and deterministic geometry helpers, in millimetres only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from seamproof import __version__

SAMPLES_PER_MM = 1.0


@dataclass(frozen=True)
class Point:
    """A Cartesian position in millimetres."""

    x: float
    y: float


def _distance(start: Point, end: Point) -> float:
    """Return Euclidean distance in millimetres."""
    return math.hypot(end.x - start.x, end.y - start.y)


@dataclass(frozen=True)
class Curve:
    """An ordered polyline whose endpoints belong to the owning edge."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        """Normalize point storage and reject unusable coordinates."""
        object.__setattr__(self, "points", tuple(self.points))
        if len(self.points) < 2:
            raise ValueError("A curve requires at least two points")
        if any(not math.isfinite(v) for p in self.points for v in (p.x, p.y)):
            raise ValueError("Curve coordinates must be finite")

    @classmethod
    def from_cubic_bezier(
        cls,
        start: Point,
        control1: Point,
        control2: Point,
        end: Point,
        samples_per_mm: float = SAMPLES_PER_MM,
    ) -> Curve:
        """Sample a cubic at approximately equal arc-length intervals.

        Assumption: one segment per mm is the default precision. Arc length
        is approximated on a deterministic grid at four times that density,
        using control-polygon length as an upper bound. The final segment
        count is ceil(estimated length * density); endpoints are exact.
        """
        if not math.isfinite(samples_per_mm) or samples_per_mm <= 0:
            raise ValueError("Sample density must be positive and finite")
        controls = (start, control1, control2, end)
        cls(controls)  # Validate coordinates before allocating the grid.
        bound = sum(_distance(a, b) for a, b in zip(controls, controls[1:]))
        if bound == 0:
            return cls((start, end))

        def evaluate(t: np.ndarray) -> np.ndarray:
            """Evaluate the cubic at the supplied parameter positions."""
            t = t[:, None]
            a, b, c, d = np.array([(p.x, p.y) for p in controls], dtype=float)
            return (1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t**2 * c + t**3 * d

        grid = np.linspace(0, 1, max(16, math.ceil(bound * samples_per_mm * 4)) + 1)
        positions = evaluate(grid)
        lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))))
        segments = max(1, math.ceil(float(lengths[-1]) * samples_per_mm))
        targets = np.linspace(0, lengths[-1], segments + 1)
        sampled = evaluate(np.interp(targets, lengths, grid))
        points = [Point(float(x), float(y)) for x, y in sampled]
        points[0], points[-1] = start, end
        return cls(tuple(points))


@dataclass(frozen=True)
class Notch:
    """A notch positioned along an edge's sampled arc length, in mm."""

    edge_id: str
    distance_from_start_mm: float


@dataclass
class Edge:
    """A directed panel boundary with seam and allowance information."""

    id: str
    curve: Curve
    seam_partner_id: str | None = None
    needs_allowance: bool = True
    allowance_mm: float = 0.0
    is_fold: bool = False


@dataclass
class Panel:
    """A cut piece with ordered boundary edges and a two-point grainline."""

    name: str
    edges: list[Edge]
    grainline: tuple[Point, Point]
    notches: list[Notch] = field(default_factory=list)
    cut_count: int = 1
    cut_on_fold: bool = False
    cut_line: Curve | None = None


def params_hash(params: dict[str, Any]) -> str:
    """Hash canonical JSON parameters, rejecting non-JSON numeric values."""
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Pattern:
    """A serializable draft; construction alone does not certify verification."""

    garment_type: str
    size_label: str
    fabric: str
    measurements: dict[str, float]
    params: dict[str, Any]
    panels: list[Panel]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Attach a format version and the current parameter fingerprint."""
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("version", __version__)
        self.metadata["params_hash"] = params_hash(self.params)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data with a fresh params hash."""
        self.metadata["params_hash"] = params_hash(self.params)
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pattern:
        """Reconstruct nested types from JSON-compatible pattern data."""
        copied = json.loads(json.dumps(data, allow_nan=False))
        panels = []
        for panel in copied["panels"]:
            edges = []
            for edge in panel["edges"]:
                edge["curve"] = Curve(tuple(Point(**p) for p in edge["curve"]["points"]))
                edges.append(Edge(**edge))
            panel["edges"] = edges
            panel["grainline"] = tuple(Point(**p) for p in panel["grainline"])
            panel["notches"] = [Notch(**n) for n in panel["notches"]]
            if panel.get("cut_line") is not None:
                panel["cut_line"] = Curve(tuple(Point(**p) for p in panel["cut_line"]["points"]))
            panels.append(Panel(**panel))
        copied["panels"] = panels
        return cls(**copied)


def edge_length(edge: Edge) -> float:
    """Sum sampled polyline segment lengths in millimetres."""
    return sum(_distance(a, b) for a, b in zip(edge.curve.points, edge.curve.points[1:]))


def panel_polygon(panel: Panel) -> Polygon:
    """Build an outline polygon; reject gaps instead of silently closing them.

    Assumption: deterministic drafting shares exact endpoint coordinates.
    This helper checks closure only; it is not the production verifier.
    """
    if not panel.edges:
        raise ValueError("Panel must have boundary edges")
    for edge, following in zip(panel.edges, panel.edges[1:] + panel.edges[:1]):
        if edge.curve.points[-1] != following.curve.points[0]:
            raise ValueError("Panel edges must form a continuous closed outline")
    points = [p for edge in panel.edges for p in edge.curve.points[:-1]]
    if len(set(points)) < 3:
        raise ValueError("Panel outline requires at least three distinct points")
    return Polygon([(p.x, p.y) for p in points])
