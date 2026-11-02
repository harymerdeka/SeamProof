"""Edit-locality lock: parameter changes may only move their declared edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from shapely.geometry import LineString

from seamproof.params import GARMENT_AFFECTS, PARAMETERS
from seamproof.pattern import Pattern

MOVE_TOL_MM = 0.5


@dataclass
class CheckResult:
    """The locality verdict for one proposed parameter change."""

    passed: bool
    moved: list[tuple[str, str, float]] = field(default_factory=list)
    message: str = ""


def _hausdorff(a: LineString, b: LineString) -> float:
    """Return the symmetric Hausdorff distance between two sampled edges."""
    return a.hausdorff_distance(b)


def edit_locality(before: Pattern, after: Pattern, changed_params: Mapping[str, float]) -> CheckResult:
    """Fail when an edge outside the declared affects moved over 0.5 mm.

    changed_params names the proposed parameters (values are not inspected).
    Added or removed edge ids always count as moved.
    """
    unknown = [name for name in changed_params if name not in PARAMETERS]
    if unknown:
        return CheckResult(False, [], f"Unknown parameter: {', '.join(sorted(unknown))}.")
    allowed = {edge for name in changed_params for edge in PARAMETERS[name].affects}
    extra = GARMENT_AFFECTS.get(after.garment_type, {})
    allowed |= {edge for name in changed_params for edge in extra.get(name, [])}
    before_edges = {(panel.name, edge.id): edge for panel in before.panels for edge in panel.edges}
    after_edges = {(panel.name, edge.id): edge for panel in after.panels for edge in panel.edges}
    moved: list[tuple[str, str, float]] = []
    for key in sorted(before_edges.keys() | after_edges.keys()):
        old, new = before_edges.get(key), after_edges.get(key)
        if old is None or new is None:
            moved.append((key[0], key[1], float("inf")))
            continue
        if old.curve == new.curve:
            continue
        distance = _hausdorff(
            LineString([(p.x, p.y) for p in old.curve.points]),
            LineString([(p.x, p.y) for p in new.curve.points]))
        if distance > MOVE_TOL_MM:
            moved.append((key[0], key[1], distance))
    undeclared = [(panel, edge, dist) for panel, edge, dist in moved if (panel, edge) not in allowed]
    if undeclared:
        names = ", ".join(f"{panel}.{edge}" for panel, edge, _dist in undeclared)
        return CheckResult(False, undeclared,
                           f"The change touches edges outside its declaration: {names}. "
                           f"Refuse this change.")
    return CheckResult(True, moved, "All movement is within the declared edges.")
