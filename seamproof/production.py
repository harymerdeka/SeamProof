"""Production packs: cut lists, fabric metres and cost per size run.

One marker is packed per size with the same draft function used everywhere
else; utilisation is measured from the packing, never targeted. Layers do not
change fabric use: a spread of L plies consumes L marker lengths, so the
total is always marker length x quantity; layers only set how many
garments one spread cuts. Markers pack cut lines (with seam allowance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from seamproof.draft import REGISTRY
from seamproof.fabric import FABRIC_EASE
from seamproof.marker import Marker, pack
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE
from seamproof.pattern import Pattern
from seamproof.verify import verify_pattern


@dataclass
class SizeProduction:
    """Marker, verdict and fabric use for one size in the run."""

    size: str
    qty: int
    pattern: Pattern
    verified: bool
    marker: Marker
    metres: float


@dataclass
class ProductionReport:
    """Whole-run cut list, metres, cost and measured utilisation."""

    garment: str
    fabric: str
    fabric_width_mm: float
    price_per_m: float
    layers: int
    sizes: list[SizeProduction] = field(default_factory=list)
    cut_list: dict[str, int] = field(default_factory=dict)
    total_metres: float = 0.0
    total_cost: float = 0.0
    overall_utilisation: float = 0.0

    def summary_id(self) -> str:
        """Return a one-line run summary."""
        units = sum(item.qty for item in self.sizes)
        return (f"Production pack {self.garment}: {units} units, "
                f"{self.total_metres:.2f} m of fabric, Rp{self.total_cost:,.0f}, "
                f"utilisation {self.overall_utilisation * 100:.1f}%.")


def _table(garment: str):
    """Return the size table for a garment family."""
    return SIZE_TABLE if garment == "tshirt" else WOMEN_SIZE_TABLE


def _parse_run(spec: str, table) -> dict[str, int]:
    """Parse 'S=20,M=50' into checked quantities."""
    run: dict[str, int] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"Jatah salah bentuk: '{part.strip()}' (contoh S=20).")
        size, qty = part.split("=", 1)
        size, qty = size.strip(), qty.strip()
        if size not in table:
            raise ValueError(f"Unknown size: {size}.")
        try:
            count = int(qty)
        except ValueError:
            raise ValueError(f"Quantity for {size} must be a whole number: '{qty}'.") from None
        if count <= 0:
            raise ValueError(f"Quantity for {size} must be positive.")
        run[size] = count
    if not run:
        raise ValueError("Jatah kosong.")
    return run


def produce(garment: str, run_spec: str, fabric: str, fabric_width_mm: float,
            price_per_m: float, layers: int = 1,
            params: Mapping[str, float] | None = None) -> ProductionReport:
    """Draft, verify and pack every size in the run; total metres and cost.

    params (e.g. a photo style plan) apply to every size, as in grading.
    """
    if garment not in REGISTRY or garment == "bodice":
        raise ValueError(f"Unknown gradable garment: {garment}")
    if fabric not in FABRIC_EASE:
        raise ValueError(f"Unknown fabric category: {fabric}")
    if not fabric_width_mm > 0:
        raise ValueError("Fabric width must be positive.")
    if not price_per_m >= 0:
        raise ValueError("Price per metre must not be negative.")
    if layers < 1:
        raise ValueError("Layers must be at least 1.")
    table = _table(garment)
    run = _parse_run(run_spec, table)
    draft = REGISTRY[garment]
    report = ProductionReport(garment, fabric, fabric_width_mm, price_per_m, layers)
    piece_area, fabric_area = 0.0, 0.0
    for size in [s for s in ("S", "M", "L", "XL") if s in run]:
        qty = run[size]
        pattern = draft(table[size], fabric, dict(params or {}))
        marker = pack(pattern, fabric_width_mm)
        metres = marker.used_length_mm * qty / 1000.0
        report.sizes.append(SizeProduction(size, qty, pattern,
                                           verify_pattern(pattern).passed, marker, metres))
        for panel in pattern.panels:
            key = f"{size} {panel.name}"
            report.cut_list[key] = report.cut_list.get(key, 0) + panel.cut_count * qty
        report.total_metres += metres
        piece_area += marker.piece_area_mm2 * qty
        fabric_area += fabric_width_mm * marker.used_length_mm * qty
    report.total_cost = report.total_metres * price_per_m
    report.overall_utilisation = piece_area / fabric_area if fabric_area > 0 else 0.0
    return report


def production_md(report: ProductionReport) -> str:
    """Render the cut list, markers and totals as Markdown."""
    lines = [f"# Production Pack — {report.garment} ({report.fabric})", "",
             f"- Fabric width: {report.fabric_width_mm:.0f} mm",
             f"- Price: Rp{report.price_per_m:,.0f}/m — Layers: {report.layers}",
             f"- Total: {report.total_metres:.2f} m — Rp{report.total_cost:,.0f}",
             f"- Utilisation: {report.overall_utilisation * 100:.1f}% (measured, not a target)",
             "",
             "| Size | Quantity | Verified | Length (mm) | Metres | Utilisation |",
             "|--------|-------|---------------|---------------|-------|-----------|"]
    for item in report.sizes:
        lines.append(f"| {item.size} | {item.qty} | "
                     f"{'PASS' if item.verified else 'FAIL'} | "
                     f"{item.marker.used_length_mm:.0f} | {item.metres:.2f} | "
                     f"{item.marker.utilisation * 100:.1f}% |")
    lines += ["", "| Cut (size panel) | Count |",
              "|--------|--------|"]
    for key in sorted(report.cut_list):
        lines.append(f"| {key} | {report.cut_list[key]} |")
    lines += ["", report.summary_id(), ""]
    return "\n".join(lines)
