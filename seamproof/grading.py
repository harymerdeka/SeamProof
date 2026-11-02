"""Size grading with per-size verification and grade-consistency checks.

grade() drafts every requested size with the same draft function, fabric and
params. grade_and_verify() additionally runs the full verifier on every size
and then grade_consistency(), which requires consecutive sizes to progress
monotonically with each step inside an expected range. The smallest and
largest sizes must always be included, because grading usually breaks at the
ends.

Step ranges below are assumptions from the current size tables (labelled
"starting values" there), not a validated sizing standard. Geometry step
bounds are generous sanity limits; the monotonicity requirement is the real
grading signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from seamproof.draft import REGISTRY
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE
from seamproof.pattern import Pattern, edge_length
from seamproof.verify import Check, Report, _check, verify_pattern

CANONICAL_SIZES = ["S", "M", "L", "XL"]

# Assumption: expected body-measurement step between consecutive sizes, in mm.
BODY_STEP_MM: dict[str, tuple[float, float]] = {
    "chest": (40, 120), "high_bust": (40, 120), "bust": (40, 120),
    "waist": (40, 120), "hip": (40, 120),
    "shoulder_width": (5, 40), "shoulder_length": (0, 20),
    "back_length": (5, 40), "front_length": (5, 40),
    "sleeve_length": (0, 30), "armhole_depth": (0, 30),
    "neck_width": (0, 30), "bicep": (5, 40), "wrist": (0, 20),
    "apex_height": (0, 30), "apex_distance": (0, 30), "waist_to_hip": (0, 30),
}

# Key body measurements tracked per garment family.
TSHIRT_KEYS = ("chest", "waist", "hip", "shoulder_width", "back_length",
               "sleeve_length", "armhole_depth", "bicep")
BODICE_KEYS = ("bust", "waist", "hip", "front_length", "back_length",
               "sleeve_length", "armhole_depth", "apex_height", "apex_distance")

# Assumption: generous sanity bounds for drafted-geometry steps, in mm.
GEOM_STEP_MM = (0.0, 100.0)
DART_INTAKE_STEP_MM = (0.0, 20.0)
APEX_STEP_MM = (0.0, 50.0)
NOISE_MM = 1e-6


def _table(garment: str):
    """Return the size table for a garment family."""
    if garment == "tshirt":
        return SIZE_TABLE
    return WOMEN_SIZE_TABLE


def _order(sizes: list[str], table: Mapping[str, Any]) -> list[str]:
    """Order requested sizes canonically and require the extreme sizes."""
    unknown = [size for size in sizes if size not in table]
    if unknown:
        raise ValueError(f"Unknown size: {', '.join(unknown)}")
    names = list(table.keys())
    if names[0] not in sizes or names[-1] not in sizes:
        raise ValueError(
            f"Grading must include the smallest ({names[0]}) and largest "
            f"({names[-1]}) sizes; requested: {', '.join(sizes)}.")
    return [size for size in CANONICAL_SIZES if size in sizes]


def grade(garment: str, sizes: list[str], fabric: str,
          params: Mapping[str, float] | None = None) -> dict[str, Pattern]:
    """Draft one pattern per size with the same draft function and params."""
    if garment not in REGISTRY or garment == "bodice":
        raise ValueError(f"Unknown gradable garment: {garment}")
    if fabric not in FABRIC_EASE:
        raise ValueError(f"Unknown fabric category: {fabric}")
    table = _table(garment)
    ordered = _order(list(sizes), table)
    draft = REGISTRY[garment]
    return {size: draft(table[size], fabric, dict(params or {})) for size in ordered}


def _edge_len(pattern: Pattern, panel_name: str, edge_id: str) -> float | None:
    """Return a stitch edge length, or None when the edge is absent."""
    for panel in pattern.panels:
        if panel.name == panel_name:
            for edge in panel.edges:
                if edge.id == edge_id:
                    return edge_length(edge)
    return None


def _step_check(id: str, name: str, values: list[float], lo: float, hi: float,
                unit: str = "mm") -> Check:
    """Require a size series to progress monotonically with steps in range."""
    steps = [after - before for before, after in zip(values, values[1:])]
    bad = [step for step in steps if step < lo - NOISE_MM or step > hi + NOISE_MM]
    rendered = ", ".join(f"{step:+.1f}" for step in steps)
    if not bad:
        return _check(id, name, True,
                      f"{name} grows in steps ({rendered} {unit}, limit {lo:.0f}–{hi:.0f}).",
                      min(steps), (lo + hi) / 2, (hi - lo) / 2)
    return _check(id, name, False,
                  f"{name} does not grade smoothly ({rendered} {unit}, limit {lo:.0f}–{hi:.0f}). "
                  f"Check the draft at the end sizes.",
                  min(steps), (lo + hi) / 2, (hi - lo) / 2)


def grade_consistency(patterns: dict[str, Pattern]) -> list[Check]:
    """Check monotonic, in-range progression across consecutive sizes."""
    checks: list[Check] = []
    sizes = list(patterns.keys())
    if len(sizes) < 2:
        return [_check("GR1", "Size series", False,
                       "At least two sizes are needed to check grading.")]
    first = patterns[sizes[0]]
    bodice = first.garment_type in ("blouse", "tunic")
    keys = BODICE_KEYS if bodice else TSHIRT_KEYS

    names = {size: {panel.name for panel in pattern.panels} for size, pattern in patterns.items()}
    if any(set(panel_names) != names[sizes[0]] for panel_names in names.values()):
        checks.append(_check("GR1", "Consistent panels", False,
                             f"Panels differ between sizes: {names}. The draft must be the same for all sizes."))
    else:
        checks.append(_check("GR1", "Consistent panels", True,
                             "All sizes use the same panels."))

    for key in keys:
        try:
            values = [float(patterns[size].measurements[key]) for size in sizes]
        except (KeyError, TypeError, ValueError):
            checks.append(_check(f"GR3-{key}", f"Langkah {key}", False,
                                 f"Size {key} is missing from the pattern."))
            continue
        lo, hi = BODY_STEP_MM[key]
        checks.append(_step_check(f"GR3-{key}", f"Langkah {key}", values, lo, hi))

    series: list[tuple[str, str, list[float | None]]] = [
        ("GR3-armhole", "Armhole curve",
         [_armhole_total(patterns[size]) for size in sizes]),
        ("GR3-cap", "Sleeve cap",
         [_edge_len(patterns[size], "sleeve", "sleeve.cap") for size in sizes]),
        ("GR3-shoulder", "Shoulder line",
         [_edge_len(patterns[size], "front", "front.shoulder") for size in sizes]),
    ]
    if not bodice:
        series.append(("GR3-side", "Bodice side",
                       [_edge_len(patterns[size], "front", "front.side") for size in sizes]))
    for id, name, values in series:
        if any(value is None for value in values):
            checks.append(_check(id, name, False, f"Edge {name} is missing from the pattern."))
        else:
            checks.append(_step_check(id, name, [float(v) for v in values], *GEOM_STEP_MM))

    if bodice:
        intake: list[float | None] = []
        tip_x: list[float | None] = []
        tip_y: list[float | None] = []
        for size in sizes:
            dart = patterns[size].metadata.get("darts", {}).get("front_bust", {})
            intake.append(dart.get("intake_mm"))
            tip = dart.get("tip")
            tip_x.append(tip[0] if isinstance(tip, (list, tuple)) and len(tip) == 2 else None)
            tip_y.append(tip[1] if isinstance(tip, (list, tuple)) and len(tip) == 2 else None)
        for id, name, values, bounds in (
                ("GR4-intake", "Dart intake", intake, DART_INTAKE_STEP_MM),
                ("GR4-apex-x", "Bust apex (x)", tip_x, APEX_STEP_MM),
                ("GR4-apex-y", "Bust apex (y)", tip_y, APEX_STEP_MM)):
            if any(value is None for value in values):
                checks.append(_check(id, name, False, f"Dart data {name} is missing from the metadata."))
            else:
                checks.append(_step_check(id, name, [float(v) for v in values], *bounds))
    return checks


def _armhole_total(pattern: Pattern) -> float | None:
    """Return the combined front+back armhole length, or None when absent."""
    front = _edge_len(pattern, "front", "front.armhole")
    back = _edge_len(pattern, "back", "back.armhole")
    if front is None or back is None:
        return None
    return front + back


@dataclass
class GradeReport:
    """Per-size verdicts plus grade-level checks for one grading run."""

    garment: str
    fabric: str
    sizes: list[str] = field(default_factory=list)
    patterns: dict[str, Pattern] = field(default_factory=dict)
    reports: dict[str, Report] = field(default_factory=dict)
    grade_checks: list[Check] = field(default_factory=list)
    passed: bool = False

    def summary_id(self) -> str:
        """Return a one-line verdict for the whole grade."""
        good = sum(report.passed for report in self.reports.values())
        grades = sum(check.passed for check in self.grade_checks)
        if self.passed:
            return (f"PASS grading: {good}/{len(self.reports)} sizes passed, "
                    f"{grades}/{len(self.grade_checks)} grading checks passed.")
        return (f"FAIL grading: {good}/{len(self.reports)} sizes passed, "
                f"{grades}/{len(self.grade_checks)} grading checks passed. "
                f"Fix it, then verify again.")


def grade_and_verify(garment: str, sizes: list[str], fabric: str,
                     params: Mapping[str, float] | None = None) -> GradeReport:
    """Draft, verify every size, then check grade consistency."""
    patterns = grade(garment, sizes, fabric, params)
    ordered = list(patterns.keys())
    reports = {size: verify_pattern(patterns[size]) for size in ordered}
    per_size_ok = _check(
        "GR2", "All sizes passed",
        all(reports[size].passed for size in ordered),
        ("All sizes passed verification."
         if all(reports[size].passed for size in ordered)
         else "Failed sizes: " + ", ".join(size for size in ordered if not reports[size].passed)
         + ". Fix those sizes first."),
        sum(not reports[size].passed for size in ordered), 0.0, 0.0)
    # GR2 first: per-size verdicts read before the progression detail.
    grade_checks = [per_size_ok] + grade_consistency(patterns)
    passed = all(report.passed for report in reports.values())
    passed = passed and all(check.passed for check in grade_checks)
    return GradeReport(garment, fabric, ordered, patterns, reports, grade_checks, passed)


def grade_report_md(report: GradeReport, explanations=None) -> str:
    """Render the per-size proof table, grade checks and tailor instructions."""
    lines = [f"# SeamProof Grade Report — {report.garment} ({report.fabric})", "",
             "| Size | Status | Checks | Parameter hash |",
             "|--------|--------|-------------|----------------|"]
    for size in report.sizes:
        single = report.reports[size]
        pattern = report.patterns[size]
        lines.append(f"| {size} | {'PASS' if single.passed else 'FAIL'} | "
                     f"{sum(c.passed for c in single.checks)}/{len(single.checks)} | "
                     f"`{pattern.metadata.get('params_hash', '')[:8]}` |")
    lines += ["", "| ID | Grading check | Value | Expected | Tolerance | Result |",
              "|----|-------------------|-------|---------|-----------|-------|"]
    for check in report.grade_checks:
        actual = "—" if check.actual is None else f"{check.actual:.1f}"
        expected = "—" if check.expected is None else f"{check.expected:.1f}"
        tolerance = "—" if check.tolerance is None else f"{check.tolerance:.1f}"
        lines.append(f"| {check.id} | {check.name_id} | {actual} | {expected} | {tolerance} | "
                     f"{'PASS' if check.passed else 'FAIL'} |")
    lines += ["", report.summary_id(), ""]
    if explanations:
        lines += ["## Tailor explanation", ""]
        for size in report.sizes:
            if size not in explanations:
                continue
            summary, instructions = explanations[size]
            lines += [f"### Size {size}", "", summary, ""]
            for item in instructions:
                amount = "—" if item.amount_mm is None else f"{item.amount_mm:.1f} mm"
                lines.append(f"- **{item.panel} / {item.edge}** ({item.action}, {amount}): {item.text}")
            lines.append("")
    return "\n".join(lines)
