"""One-to-one PDF export; millimetres become points only in this module.

Scale: 1 mm = 2.8346 pt exactly. Each panel gets its own page sized to its
cut or stitch bounding box plus margin, so printing at 100% yields true
size. A 100 mm ruler on every page lets the workshop confirm the scale.
"""

from __future__ import annotations

import math
from datetime import date
from typing import BinaryIO
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from seamproof import __version__
from seamproof.allowance import ensure_allowance
from seamproof.pattern import Curve, Panel, Pattern, Point

MM_TO_PT = 2.8346
MARGIN_MM = 10.0
RULER_MM = 100.0


def mm_to_pt(mm: float) -> float:
    """Convert millimetres to PDF points at the 1:1 scale."""
    return mm * MM_TO_PT


def panel_bounds(panel: Panel) -> tuple[float, float, float, float]:
    """Return the stitch bounding box plus cut line, as (min_x, min_y, max_x, max_y) in mm."""
    curves = [edge.curve for edge in panel.edges]
    if panel.cut_line is not None:
        curves.append(panel.cut_line)
    xs = [point.x for curve in curves for point in curve.points]
    ys = [point.y for curve in curves for point in curve.points]
    return min(xs), min(ys), max(xs), max(ys)


def panel_page_size(panel: Panel, margin_mm: float = MARGIN_MM) -> tuple[float, float]:
    """Return the 1:1 page size for a panel, in PDF points."""
    min_x, min_y, max_x, max_y = panel_bounds(panel)
    return (mm_to_pt(max_x - min_x + 2 * margin_mm), mm_to_pt(max_y - min_y + 2 * margin_mm))


def place_point(point: Point, bounds: tuple[float, float, float, float],
                margin_mm: float = MARGIN_MM) -> tuple[float, float]:
    """Map a pattern point to its 1:1 page position in points (y grows up)."""
    min_x, _min_y, _max_x, max_y = bounds
    return (mm_to_pt(point.x - min_x + margin_mm), mm_to_pt(max_y - point.y + margin_mm))


def place_curve(curve: Curve, bounds: tuple[float, float, float, float],
                margin_mm: float = MARGIN_MM) -> list[tuple[float, float]]:
    """Map every sampled point of a curve to page positions in points."""
    return [place_point(point, bounds, margin_mm) for point in curve.points]


def _stroke(canvas: Canvas, placed: list[tuple[float, float]]) -> None:
    """Stroke one placed polyline on the canvas."""
    path = canvas.beginPath()
    path.moveTo(*placed[0])
    for x, y in placed[1:]:
        path.lineTo(x, y)
    canvas.drawPath(path, stroke=1, fill=0)


def draw_panel_page(canvas: Canvas, panel: Panel, pattern: Pattern,
                    margin_mm: float = MARGIN_MM) -> None:
    """Draw one panel at 1:1 with grainline, notches, ruler and labels."""
    bounds = panel_bounds(panel)
    canvas.setPageSize(panel_page_size(panel, margin_mm))
    canvas.setLineWidth(1.0)
    for edge in panel.edges:
        _stroke(canvas, place_curve(edge.curve, bounds, margin_mm))
    if panel.cut_line is not None:
        canvas.saveState()
        canvas.setDash(4, 3)
        _stroke(canvas, place_curve(panel.cut_line, bounds, margin_mm))
        canvas.restoreState()
    # Grainline with arrowheads.
    canvas.setLineWidth(1.5)
    grain = place_curve(Curve(panel.grainline), bounds, margin_mm)
    _stroke(canvas, grain)
    (gx0, gy0), (gx1, gy1) = grain
    canvas.drawString(gx1 + 4, gy1 + 4, "SG")
    # Notches as small filled squares.
    known = {edge.id: edge for edge in panel.edges}
    for notch in panel.notches:
        edge = known.get(notch.edge_id)
        if edge is None:
            continue
        target, walked, total = notch.distance_from_start_mm, 0.0, _curve_length(edge.curve)
        spot = edge.curve.points[0]
        for a, b in zip(edge.curve.points, edge.curve.points[1:]):
            step = _seg(a, b)
            if walked + step >= target:
                ratio = 0.0 if step == 0 else (target - walked) / step
                spot = Point(a.x + (b.x - a.x) * ratio, a.y + (b.y - a.y) * ratio)
                break
            walked += step
        x, y = place_point(spot, bounds, margin_mm)
        canvas.rect(x - 2, y - 2, 4, 4, stroke=1, fill=0)
    # 100 mm ruler for scale confirmation.
    rx, ry = mm_to_pt(margin_mm), mm_to_pt(margin_mm)
    canvas.setLineWidth(1.0)
    canvas.line(rx, ry, rx + mm_to_pt(RULER_MM), ry)
    for i in range(11):
        tick = rx + mm_to_pt(i * 10)
        canvas.line(tick, ry, tick, ry - (6 if i % 5 == 0 else 3))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(rx, ry + 4, f"{RULER_MM:.0f} mm (scale 1:1)")
    # Labels.
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(rx, ry + 20, f"{panel.name} — {pattern.garment_type} {pattern.size_label}")
    canvas.setFont("Helvetica", 9)
    cut_text = f"cut {panel.cut_count}x" + (" (on fold)" if panel.cut_on_fold else "")
    canvas.drawString(rx, ry + 32, f"{pattern.fabric} — {cut_text}")
    canvas.drawString(rx, ry + 44, f"SeamProof {__version__} — {pattern.metadata.get('params_hash', '')[:8]}")
    canvas.showPage()


def draw_cover_page(canvas: Canvas, pattern: Pattern) -> None:
    """Draw an A4 cover with the pattern's identity and panel list."""
    canvas.setPageSize(A4)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(50, 780, f"SeamProof — {pattern.garment_type} {pattern.size_label}")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(50, 755, f"Fabric: {pattern.fabric} — date: {date.today().isoformat()}")
    canvas.drawString(50, 735, f"Status: {pattern.metadata.get('status', '')}")
    y = 700
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(50, y, "Panels (print the next pages at 100%, check the 100 mm ruler):")
    canvas.setFont("Helvetica", 12)
    for panel in pattern.panels:
        y -= 20
        cut_text = f"cut {panel.cut_count}x" + (" (on fold)" if panel.cut_on_fold else "")
        canvas.drawString(70, y, f"{panel.name}: {len(panel.edges)} edges — {cut_text}")
    canvas.showPage()


def draw_pattern_pdf(pattern: Pattern, target: str | BinaryIO) -> None:
    """Write a cover plus one 1:1 page per panel to a path or binary stream."""
    canvas = Canvas(target, pagesize=A4, pageCompression=0)
    canvas.setTitle(f"SeamProof {pattern.garment_type} {pattern.size_label}")
    canvas.setAuthor(f"SeamProof {__version__}")
    draw_cover_page(canvas, pattern)
    for panel in pattern.panels:
        draw_panel_page(canvas, panel, pattern)
    canvas.save()


def _seg(a: Point, b: Point) -> float:
    """Return one segment length in mm."""
    return ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5


def _curve_length(curve: Curve) -> float:
    """Sum sampled segment lengths in mm."""
    return sum(_seg(a, b) for a, b in zip(curve.points, curve.points[1:]))


TILE_MARGIN_MM = 10.0
TILE_OVERLAP_MM = 15.0
TILE_USABLE_W_MM = 210.0 - 2 * TILE_MARGIN_MM
TILE_USABLE_H_MM = 297.0 - 2 * TILE_MARGIN_MM
TILE_STEP_X_MM = TILE_USABLE_W_MM - TILE_OVERLAP_MM
TILE_STEP_Y_MM = TILE_USABLE_H_MM - TILE_OVERLAP_MM
A4_PT = (MM_TO_PT * 210.0, MM_TO_PT * 297.0)
NOT_VERIFIED = "NOT VERIFIED"
TEST_SQUARE_LABEL = "Measure: must be 10 cm"


def tile_grid(min_x: float, min_y: float, max_x: float, max_y: float,
              margin_mm: float = TILE_MARGIN_MM,
              overlap_mm: float = TILE_OVERLAP_MM) -> tuple[int, int, float, float]:
    """Cover a bounding box with overlapping A4 tiles; return (cols, rows, x0, y0) in mm."""
    usable_w = 210.0 - 2 * margin_mm
    usable_h = 297.0 - 2 * margin_mm
    step_x, step_y = usable_w - overlap_mm, usable_h - overlap_mm
    span_x = (max_x + margin_mm) - (min_x - margin_mm)
    span_y = (max_y + margin_mm) - (min_y - margin_mm)
    cols = 1 if span_x <= usable_w else 1 + math.ceil((span_x - usable_w) / step_x)
    rows = 1 if span_y <= usable_h else 1 + math.ceil((span_y - usable_h) / step_y)
    return cols, rows, min_x - margin_mm, min_y - margin_mm


def tile_place(x_mm: float, y_mm: float, tile_x0: float, tile_y0: float,
               margin_mm: float = TILE_MARGIN_MM) -> tuple[float, float]:
    """Map a pattern point onto its A4 tile in points at exactly 1:1 (y grows up)."""
    return (mm_to_pt(margin_mm + x_mm - tile_x0),
            A4_PT[1] - mm_to_pt(margin_mm + y_mm - tile_y0))


def _tile_rect_pt(tile_x0: float, tile_y0: float,
                  margin_mm: float = TILE_MARGIN_MM) -> tuple[float, float, float, float]:
    """Return the tile content rectangle on the page in points."""
    usable_w = 210.0 - 2 * margin_mm
    usable_h = 297.0 - 2 * margin_mm
    x0, y0 = mm_to_pt(margin_mm), A4_PT[1] - mm_to_pt(margin_mm + usable_h)
    return x0, y0, x0 + mm_to_pt(usable_w), y0 + mm_to_pt(usable_h)


def _stroke_placed(canvas: Canvas, placed: list[tuple[float, float]]) -> None:
    """Stroke one placed polyline on the canvas."""
    path = canvas.beginPath()
    path.moveTo(*placed[0])
    for x, y in placed[1:]:
        path.lineTo(x, y)
    canvas.drawPath(path, stroke=1, fill=0)


def _placed_curve_capped(corners: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return placed points, guarding against degenerate input."""
    return [(float(x), float(y)) for x, y in corners]


def draw_test_square(canvas: Canvas, x_pt: float, y_pt: float) -> tuple[float, float]:
    """Draw the 100x100 mm scale square; return its size in points."""
    side = mm_to_pt(RULER_MM)
    canvas.setLineWidth(1.5)
    canvas.rect(x_pt, y_pt, side, side, stroke=1, fill=0)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(x_pt, y_pt - 12, TEST_SQUARE_LABEL)
    return side, side


def _fmt(value: float | None) -> str:
    """Format a check number for the proof table."""
    return "—" if value is None else f"{value:.1f}"


def draw_proof_page(canvas: Canvas, pattern: Pattern, report, explanation=None) -> None:
    """Draw page 1: identity, verdict, check table, tailor lines, test square."""
    from seamproof.verify import Report

    assert isinstance(report, Report)
    canvas.setPageSize(A4_PT)
    if not report.passed:
        _stamp_not_verified(canvas)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(50, 800, f"SeamProof — {pattern.garment_type} {pattern.size_label}")
    canvas.setFont("Helvetica", 11)
    meta = report.meta
    canvas.drawString(50, 778, f"Fabric: {pattern.fabric} — version: {meta.get('version', '')}")
    canvas.drawString(50, 762, f"Parameter hash: {meta.get('params_hash', '')[:16]}…")
    seam = pattern.metadata.get("seam_allowance_mm") or {}
    seam_text = (f" — seam allowance {seam.get('default', 0):g} mm, hem {seam.get('hem', 0):g} mm, fold 0"
                 if seam else " — NO seam allowance")
    canvas.drawString(50, 746, f"Repair rounds: {meta.get('repair_rounds', 0)}{seam_text}")
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(50, 722, "PASS — ready for toile." if report.passed else "FAIL — fix it first.")
    signoff = meta.get("signoff")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(300, 722, f"Checked by pattern maker: {signoff['name']} ({signoff['at'][:10]})"
                      if signoff else "Not yet checked by a pattern maker.")
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(50, 700, "ID  Check                  Value   Expected  Limit  Result")
    canvas.setFont("Helvetica", 9)
    y = 686
    for check in report.checks:
        result = "PASS" if check.passed else "FAIL"
        canvas.drawString(50, y, f"{check.id:<4}{check.name_id[:22]:<23}{_fmt(check.actual):>7} "
                                f"{_fmt(check.expected):>8} {_fmt(check.tolerance):>6}  {result}")
        y -= 13
    if explanation is not None:
        _summary, instructions = explanation
        y -= 6
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(50, y, "Tailor explanation (details in report.md):")
        canvas.setFont("Helvetica", 9)
        for item in instructions[:4]:
            y -= 13
            canvas.drawString(60, y, f"- {item.text[:95]}")
        if len(instructions) > 4:
            y -= 13
            canvas.drawString(60, y, f"+{len(instructions) - 4} more instructions, see report.md.")
            y -= 13
    draw_test_square(canvas, 50, y - 130)
    canvas.showPage()


def _stamp_not_verified(canvas: Canvas) -> None:
    """Stamp a forced page so nobody mistakes it for a verified pattern."""
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(50, A4_PT[1] - 40, NOT_VERIFIED)
    canvas.restoreState()


def draw_tile_page(canvas: Canvas, panel: Panel, pattern: Pattern, col: int, row: int,
                   tile_x0: float, tile_y0: float, forced: bool,
                   margin_mm: float = TILE_MARGIN_MM) -> None:
    """Draw one A4 tile of a panel at exactly 1:1 with overlap marks."""
    canvas.setPageSize(A4_PT)
    if forced:
        _stamp_not_verified(canvas)
    x0, y0, x1, y1 = _tile_rect_pt(tile_x0, tile_y0, margin_mm)
    canvas.saveState()
    clip = canvas.beginPath()
    clip.rect(x0, y0, x1 - x0, y1 - y0)
    canvas.clipPath(clip, stroke=0, fill=0)
    canvas.setLineWidth(0.8)
    canvas.setDash(5, 4)
    for edge in panel.edges:
        placed = _placed_curve_capped([tile_place(p.x, p.y, tile_x0, tile_y0, margin_mm)
                                       for p in edge.curve.points])
        if len(placed) > 1:
            _stroke_placed(canvas, placed)
    canvas.setDash()
    if panel.cut_line is not None:
        canvas.setLineWidth(1.2)
        placed = _placed_curve_capped([tile_place(p.x, p.y, tile_x0, tile_y0, margin_mm)
                                       for p in panel.cut_line.points])
        if len(placed) > 1:
            _stroke_placed(canvas, placed)
    canvas.setLineWidth(1.5)
    grain = _placed_curve_capped([tile_place(p.x, p.y, tile_x0, tile_y0, margin_mm)
                                  for p in panel.grainline])
    _stroke_placed(canvas, grain)
    _arrow(canvas, grain[0], grain[1])
    known = {edge.id: edge for edge in panel.edges}
    for notch in panel.notches:
        edge = known.get(notch.edge_id)
        if edge is None:
            continue
        spot = _notch_point(edge.curve, notch.distance_from_start_mm)
        if spot is None:
            continue
        x, y = tile_place(spot.x, spot.y, tile_x0, tile_y0, margin_mm)
        canvas.rect(x - 2, y - 2, 4, 4, stroke=1, fill=0)
    canvas.restoreState()
    # Overlap registration marks at the content corners.
    canvas.setLineWidth(1.0)
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        canvas.line(cx - 6, cy, cx + 6, cy)
        canvas.line(cx, cy - 6, cx, cy + 6)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(x0, y1 + 4, f"{panel.name} — R{row + 1}C{col + 1}")
    cut_text = f"cut {panel.cut_count}x" + (" (on fold)" if panel.cut_on_fold else "")
    canvas.drawString(x0, y0 - 12, f"{pattern.garment_type} {pattern.size_label} — {cut_text}")
    canvas.showPage()


def _arrow(canvas: Canvas, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw an arrowhead at the end of a placed grainline."""
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for delta in (25, -25):
        radians = math.radians(delta)
        canvas.line(end[0], end[1], end[0] - 10 * math.cos(angle + radians),
                    end[1] - 10 * math.sin(angle + radians))


def _notch_point(curve: Curve, distance_mm: float) -> Point | None:
    """Walk a curve to the notch position, or None when out of range."""
    walked = 0.0
    for a, b in zip(curve.points, curve.points[1:]):
        step = _seg(a, b)
        if walked + step >= distance_mm:
            ratio = 0.0 if step == 0 else (distance_mm - walked) / step
            return Point(a.x + (b.x - a.x) * ratio, a.y + (b.y - a.y) * ratio)
        walked += step
    return None


def export_pdf_tiled(pattern: Pattern, report, path: str | BinaryIO, force: bool = False,
                     explanation=None) -> None:
    """Write the proof report plus A4 tiles at exact 1:1 scale.

    Refuses failing reports unless forced; forced pages carry NOT VERIFIED.
    """
    if not report.passed and not force:
        raise ValueError("The pattern failed verification — PDF refused. Use --force to print with a NOT VERIFIED stamp.")
    pattern = ensure_allowance(pattern)
    canvas = Canvas(path, pagesize=A4_PT, pageCompression=0)
    canvas.setTitle(f"SeamProof {pattern.garment_type} {pattern.size_label}")
    canvas.setAuthor(f"SeamProof {__version__}")
    draw_proof_page(canvas, pattern, report, explanation)
    for panel in pattern.panels:
        min_x, min_y, max_x, max_y = panel_bounds(panel)
        cols, rows, x0, y0 = tile_grid(min_x, min_y, max_x, max_y)
        for row in range(rows):
            for col in range(cols):
                draw_tile_page(canvas, panel, pattern, col, row, x0 + col * TILE_STEP_X_MM,
                               y0 + row * TILE_STEP_Y_MM, forced=not report.passed)
    canvas.save()


def export_svg(pattern: Pattern, path: str | BinaryIO) -> None:
    """Write a flat preview SVG with millimetre units, one row of panels.

    Cut line solid, stitch line dashed, matching the printed tiles.
    """
    pattern = ensure_allowance(pattern)
    gap = 20.0
    boxes = [panel_bounds(panel) for panel in pattern.panels]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    total_w = sum(widths) + gap * (len(widths) - 1)
    total_h = max(heights)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}mm" '
             f'height="{total_h:.1f}mm" viewBox="0 0 {total_w:.1f} {total_h:.1f}">']
    offset = 0.0
    for panel, (min_x, min_y, _max_x, _max_y), width in zip(pattern.panels, boxes, widths):
        lines.append(f'<g id="{panel.name}">')
        for edge in panel.edges:
            points = " ".join(f"{p.x - min_x + offset:.1f},{p.y - min_y:.1f}" for p in edge.curve.points)
            lines.append(f'<polyline id="{edge.id}" points="{points}" fill="none" stroke="black" '
                         f'stroke-width="0.4" stroke-dasharray="3 2" class="stitch"/>')
        if panel.cut_line is not None:
            points = " ".join(f"{p.x - min_x + offset:.1f},{p.y - min_y:.1f}" for p in panel.cut_line.points)
            lines.append(f'<polyline id="{panel.name}.cut" class="cut" points="{points}" fill="none" '
                         f'stroke="black" stroke-width="0.7"/>')
        a, b = panel.grainline
        lines.append(f'<line x1="{a.x - min_x + offset:.1f}" y1="{a.y - min_y:.1f}" '
                     f'x2="{b.x - min_x + offset:.1f}" y2="{b.y - min_y:.1f}" stroke="blue" stroke-width="0.5" class="grain"/>')
        for notch in panel.notches:
            spot = None
            known = edge_by_id(panel, notch.edge_id)
            if known is not None:
                spot = _notch_point(known.curve, notch.distance_from_start_mm)
            if spot is not None:
                lines.append(f'<rect x="{spot.x - min_x + offset - 1:.1f}" y="{spot.y - min_y - 1:.1f}" '
                             f'width="2" height="2" fill="none" stroke="red" stroke-width="0.5" class="notch"/>')
        lines.append(f'<text x="{offset:.1f}" y="{total_h - 2:.1f}" font-size="5" class="label">{panel.name} '
                     f'(cut {panel.cut_count}x{" on fold" if panel.cut_on_fold else ""})</text>')
        lines.append("</g>")
        offset += width + gap
    lines.append("</svg>")
    content = "\n".join(lines) + "\n"
    if isinstance(path, str):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write(content.encode("utf-8"))


DXF_LAYERS = {"CUT": 7, "STITCH": 8, "GRAIN": 5, "NOTCH": 1, "TEXT": 3, "FOLD": 4}
DXF_NOTCH_MM = 6.0


def _outward_normal(panel: Panel, curve: Curve, at: Point) -> tuple[float, float]:
    """Unit normal at a point on an edge, pointing out of the stitch polygon."""
    from shapely.geometry import Point as ShapelyPoint

    from seamproof.pattern import panel_polygon

    best = min(zip(curve.points, curve.points[1:]),
               key=lambda ab: math.hypot((ab[0].x + ab[1].x) / 2 - at.x, (ab[0].y + ab[1].y) / 2 - at.y))
    dx, dy = best[1].x - best[0].x, best[1].y - best[0].y
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    if panel_polygon(panel).contains(ShapelyPoint(at.x + nx, at.y + ny)):
        nx, ny = -nx, -ny
    return nx, ny


def export_dxf(pattern: Pattern, report, path: str | BinaryIO, force: bool = False) -> None:
    """Write a cutter/CAD DXF in millimetres, one row of panels like the SVG.

    Layers: CUT (closed cut line), STITCH (stitch edges), FOLD (fold edges),
    GRAIN, NOTCH (6 mm slits from the stitch line outward) and TEXT (panel,
    size, cut count). Fold pieces stay as halves marked LIPAT, as in the PDF.
    Assumption: a generic R2010 DXF with $INSUNITS = 4 (mm); confirm the
    import settings once with the workshop's cutter software. Refuses failing
    reports unless forced.
    """
    import io as _io

    import ezdxf

    if not report.passed and not force:
        raise ValueError("The pattern failed verification — DXF refused.")
    pattern = ensure_allowance(pattern)
    doc = ezdxf.new("R2010", setup=False)
    doc.header["$INSUNITS"] = 4
    for name, color in DXF_LAYERS.items():
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    offset, gap = 0.0, 20.0
    for panel in pattern.panels:
        min_x, min_y, max_x, max_y = panel_bounds(panel)

        def place(p: Point) -> tuple[float, float]:
            return (p.x - min_x + offset, -(p.y - min_y))  # DXF y runs up

        if panel.cut_line is not None:
            msp.add_lwpolyline([place(p) for p in panel.cut_line.points], close=True, dxfattribs={"layer": "CUT"})
        for edge in panel.edges:
            msp.add_lwpolyline([place(p) for p in edge.curve.points],
                               dxfattribs={"layer": "FOLD" if edge.is_fold else "STITCH"})
        a, b = panel.grainline
        msp.add_line(place(a), place(b), dxfattribs={"layer": "GRAIN"})
        for notch in panel.notches:
            edge = edge_by_id(panel, notch.edge_id)
            spot = None if edge is None else _notch_point(edge.curve, notch.distance_from_start_mm)
            if spot is None:
                continue
            nx, ny = _outward_normal(panel, edge.curve, spot)
            tip = Point(spot.x + nx * DXF_NOTCH_MM, spot.y + ny * DXF_NOTCH_MM)
            msp.add_line(place(spot), place(tip), dxfattribs={"layer": "NOTCH"})
        label = (f"{pattern.garment_type} {pattern.size_label} {panel.name} cut {panel.cut_count}x"
                 + (" FOLD" if panel.cut_on_fold else ""))
        msp.add_text(label, height=6, dxfattribs={"layer": "TEXT"}).set_placement(
            (offset, -(max_y - min_y) - 12))
        offset += (max_x - min_x) + gap
    if isinstance(path, str):
        doc.saveas(path)
    else:
        text = _io.StringIO()
        doc.write(text)
        path.write(text.getvalue().encode("utf-8"))


def edge_by_id(panel: Panel, edge_id: str):
    """Return one panel edge by id, or None."""
    for edge in panel.edges:
        if edge.id == edge_id:
            return edge
    return None


def draw_production_summary(canvas: Canvas, report) -> None:
    """Draw the production totals, cut list and per-size table on A4."""
    canvas.setPageSize(A4_PT)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(50, 800, f"Production summary — {report.garment} ({report.fabric})")
    canvas.setFont("Helvetica", 11)
    facts = [f"Fabric width: {report.fabric_width_mm:.0f} mm — Layers: {report.layers}",
             f"Price: Rp{report.price_per_m:,.0f}/m",
             f"Total: {report.total_metres:.2f} m — Rp{report.total_cost:,.0f}",
             f"Utilisation: {report.overall_utilisation * 100:.1f}% (measured, not a target)"]
    y = 776
    for fact in facts:
        canvas.drawString(50, y, fact)
        y -= 16
    y -= 8
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(50, y, "Size   Qty    Verified    Length   Metres Utilisation")
    canvas.setFont("Helvetica", 9)
    for item in report.sizes:
        y -= 13
        canvas.drawString(50, y, f"{item.size:<7}{item.qty:<7}{'PASS' if item.verified else 'FAIL':<12}"
                                f"{item.marker.used_length_mm:<9.0f}{item.metres:<7.2f}"
                                f"{item.marker.utilisation * 100:.1f}%")
    y -= 19
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(50, y, "Cut list (size panel: count)")
    canvas.setFont("Helvetica", 9)
    for key in sorted(report.cut_list):
        y -= 13
        if y < 60:
            break
        canvas.drawString(60, y, f"{key}: {report.cut_list[key]}")
    canvas.showPage()


MARKER_PAD_MM = 10.0
MARKER_PLOT_MAX_MM = 4800.0  # longest 1:1 page; PDF viewers stop near 5080 mm (200 in)
MARKER_PLOT_OVERLAP_MM = 50.0
# Title block and test square sit after the marker end, so the page is only
# 20 mm wider than the fabric and fits a 1.6 m plotter roll for 1.5 m fabric.
MARKER_TAIL_MM = 185.0


def _marker_view(marker, x: float, y: float) -> tuple[float, float]:
    """Marker mm -> display mm: fabric length runs right, width runs down.

    A quarter turn (not a mirror), so pieces keep their face-up shape.
    """
    return y, marker.fabric_width_mm - x


def _piece_caption(place, size: str) -> tuple[str, str]:
    """Two-line piece label: name, then size, copy and how it is laid."""
    extra = []
    if place.copies > 1:
        extra.append(f"{place.copy}/{place.copies}")
    if place.mirrored:
        extra.append("mirrored")
    if place.opened_fold:
        extra.append("fold opened")
    return place.panel, " · ".join([size] + extra) if size else " · ".join(extra)


def export_marker_svg(marker, title: str, path: str | BinaryIO, size: str = "") -> None:
    """Write a marker as SVG in mm: real cut shapes, grain, notches, fold lines.

    Display is turned a quarter: fabric length runs left to right, the two
    selvedges are the top and bottom edges. viewBox units are mm (1:1).
    """
    length, width = marker.used_length_mm, marker.fabric_width_mm
    view = lambda x, y: _marker_view(marker, x, y)  # noqa: E731
    pts = lambda seq: " ".join(f"{a:.1f},{b:.1f}" for a, b in (view(x, y) for x, y in seq))  # noqa: E731
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{length + 40:.0f}mm" height="{width + 70:.0f}mm" '
             f'viewBox="-20 -50 {length + 40:.1f} {width + 70:.1f}" font-family="Helvetica, Arial, sans-serif">',
             f'<text x="0" y="-30" font-size="14" font-weight="bold" class="mk-title">{escape(title)}</text>',
             f'<text x="0" y="-12" font-size="10" class="mk-sub">width {width:.0f} mm · length {length:.0f} mm · '
             f'utilisation {marker.utilisation * 100:.1f}% · fabric length direction →</text>',
             f'<rect x="0" y="0" width="{length:.1f}" height="{width:.1f}" fill="#fff" stroke="black" '
             f'stroke-width="1" class="mk-fabric"/>']
    for tick in range(0, int(length) + 1, 100):
        big = tick % 500 == 0
        lines.append(f'<line x1="{tick}" y1="0" x2="{tick}" y2="{12 if big else 6}" stroke="black" '
                     f'stroke-width="0.5" class="mk-tick"/>')
        if big and tick:
            lines.append(f'<text x="{tick + 2}" y="20" font-size="9" class="mk-tick-l">{tick / 1000:.1f} m</text>')
    lines.append(f'<text x="4" y="{width - 4:.1f}" font-size="8" class="mk-sel">FABRIC EDGE</text>')
    for place in marker.placements:
        name, sub = _piece_caption(place, size)
        lines.append(f'<g data-name="{escape(name + " · " + sub, {chr(34): "&quot;"})}" class="mk-piece">')
        lines.append(f'<polygon points="{pts(place.cut)}" fill="#f4f1e4" stroke="none" class="piece"/>')
        for seq in place.stitch:
            lines.append(f'<polyline points="{pts(seq)}" fill="none" stroke="#777" stroke-width="0.4" '
                         f'stroke-dasharray="3 2" class="stitch"/>')
        if place.fold:
            lines.append(f'<polyline points="{pts(place.fold)}" fill="none" stroke="#2c5a8c" stroke-width="0.5" '
                         f'stroke-dasharray="8 2 2 2" class="fold"/>')
        lines.append(f'<polyline points="{pts(place.cut)}" fill="none" stroke="black" stroke-width="0.7" class="cut"/>')
        (gx0, gy0), (gx1, gy1) = view(*place.grain[0]), view(*place.grain[1])
        lines.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" stroke="#2c5a8c" '
                     f'stroke-width="0.6" class="grain"/>')
        angle = math.atan2(gy1 - gy0, gx1 - gx0)
        for delta in (0.45, -0.45):
            hx, hy = gx1 - 10 * math.cos(angle + delta), gy1 - 10 * math.sin(angle + delta)
            lines.append(f'<line x1="{gx1:.1f}" y1="{gy1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="#2c5a8c" '
                         f'stroke-width="0.6" class="grain"/>')
        for seq in place.notches:
            lines.append(f'<polyline points="{pts(seq)}" fill="none" stroke="#a3352a" stroke-width="0.8" class="notch"/>')
        lx, ly = view(*place.label_at)
        lines.append(f'<text x="{lx:.1f}" y="{ly - 2:.1f}" font-size="16" font-weight="bold" text-anchor="middle" '
                     f'class="mlabel">{escape(name)}</text>')
        lines.append(f'<text x="{lx:.1f}" y="{ly + 14:.1f}" font-size="10" text-anchor="middle" '
                     f'class="mlabel-sub">{escape(sub)}</text>')
        lines.append("</g>")
    lines.append("</svg>")
    content = "\n".join(lines) + "\n"
    if isinstance(path, str):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write(content.encode("utf-8"))


def _draw_marker(canvas: Canvas, marker, size: str, to_pt, scale: float,
                 label_pt: float = 9.0, window: tuple[float, float, float, float] | None = None) -> None:
    """Draw fabric, pieces, grain, notches and labels through a mm->pt map.

    to_pt maps display mm (length right, width down) to page points; scale
    is points per display mm, used only for arrowheads and dash lengths.
    window (display x0, y0, x1, y1) skips pieces that cannot show on a tile.
    """
    length, width = marker.used_length_mm, marker.fabric_width_mm

    def placed(seq):
        return [to_pt(*_marker_view(marker, x, y)) for x, y in seq]

    canvas.setStrokeColor(black)
    canvas.setLineWidth(1.0)
    _stroke_placed(canvas, [to_pt(0, 0), to_pt(length, 0), to_pt(length, width), to_pt(0, width), to_pt(0, 0)])
    canvas.setLineWidth(0.5)
    canvas.setFont("Helvetica", 7)
    for tick in range(0, int(length) + 1, 100):
        big = tick % 500 == 0
        canvas.line(*to_pt(tick, 0), *to_pt(tick, 12 if big else 6))
        if big and tick:
            canvas.drawString(*to_pt(tick + 2, 22), f"{tick / 1000:.1f} m")
    canvas.drawString(*to_pt(4, width - 4), "FABRIC EDGE")
    for place in marker.placements:
        if window is not None:
            shown = [_marker_view(marker, x, y) for x, y in place.cut]
            if (max(x for x, _ in shown) < window[0] or min(x for x, _ in shown) > window[2]
                    or max(y for _, y in shown) < window[1] or min(y for _, y in shown) > window[3]):
                continue
        canvas.setStrokeColor(HexColor("#777777"))
        canvas.setLineWidth(0.4)
        canvas.setDash(max(1.0, 3 * scale), max(1.0, 2 * scale))
        for seq in place.stitch:
            _stroke_placed(canvas, placed(seq))
        if place.fold:
            canvas.setStrokeColor(HexColor("#2c5a8c"))
            canvas.setDash(max(2.0, 8 * scale), max(1.0, 2 * scale))
            _stroke_placed(canvas, placed(place.fold))
        canvas.setDash()
        canvas.setStrokeColor(black)
        canvas.setLineWidth(1.0)
        _stroke_placed(canvas, placed(place.cut))
        canvas.setStrokeColor(HexColor("#2c5a8c"))
        canvas.setLineWidth(0.9)
        grain = placed(place.grain)
        _stroke_placed(canvas, grain)
        _arrow(canvas, grain[0], grain[1])
        canvas.setStrokeColor(HexColor("#a3352a"))
        canvas.setLineWidth(1.2)
        for seq in place.notches:
            _stroke_placed(canvas, placed(seq))
        canvas.setFillColor(black)
        name, sub = _piece_caption(place, size)
        lx, ly = to_pt(*_marker_view(marker, *place.label_at))
        canvas.setFont("Helvetica-Bold", label_pt)
        canvas.drawCentredString(lx, ly + 2, name)
        canvas.setFont("Helvetica", label_pt * 0.75)
        canvas.drawCentredString(lx, ly - label_pt * 0.8, sub)
    canvas.setStrokeColor(black)


def _marker_header(canvas: Canvas, marker, title: str, x_pt: float, y_pt: float, one_to_one: bool,
                   k: float = 1.0, arrow: bool = True) -> None:
    """Title, size facts and the scale warning above a marker drawing; k scales the type.

    arrow=False drops the "length runs right" hint for turned title blocks.
    """
    canvas.setFillColor(black)
    canvas.setFont("Helvetica-Bold", 14 * k)
    canvas.drawString(x_pt, y_pt, title)
    canvas.setFont("Helvetica", 9 * k)
    scale = ("SCALE 1:1 — print at 100% / Actual size, not 'Fit'. Measure the 10 cm test square before cutting."
             if one_to_one else "Reduced drawing — not for tracing. Use the plotter or A4 1:1 PDF.")
    canvas.drawString(x_pt, y_pt - 13 * k, f"Fabric width {marker.fabric_width_mm:.0f} mm · marker length "
                                          f"{marker.used_length_mm:.0f} mm · utilisation {marker.utilisation * 100:.1f}% · "
                                          f"{len(marker.placements)} pieces"
                                          + (" · fabric length direction →" if arrow else ""))
    canvas.drawString(x_pt, y_pt - 25 * k, scale)


def draw_marker_page(canvas: Canvas, marker, title: str, size: str = "") -> None:
    """Draw one marker fitted to A4 landscape (overview, not 1:1)."""
    page_w, page_h = A4_PT[1], A4_PT[0]
    canvas.setPageSize((page_w, page_h))
    box_w, box_h = 277.0, 160.0
    scale_mm = min(box_w / marker.used_length_mm, box_h / marker.fabric_width_mm)
    s = mm_to_pt(scale_mm)
    left, top = mm_to_pt(10), page_h - mm_to_pt(36)
    _marker_header(canvas, marker, title, left, page_h - mm_to_pt(12), one_to_one=False)
    _draw_marker(canvas, marker, size, lambda x, y: (left + x * s, top - y * s), s, label_pt=6.0)
    canvas.showPage()


def _windows(total: float, window: float, overlap: float) -> list[float]:
    """Start offsets of overlapping windows that cover [0, total]."""
    if total <= window:
        return [0.0]
    step = window - overlap
    return [i * step for i in range(1 + math.ceil((total - window) / step))]


def _check_marker_print(verified: bool, force: bool) -> None:
    """Refuse 1:1 cutting prints of a pattern that failed verification."""
    if not verified and not force:
        raise ValueError("This size failed verification — 1:1 marker refused.")


def marker_plot_page_mm(marker) -> list[float]:
    """Plotter page (length, height) in mm for a single-page plot; longer markers split."""
    return [round(min(marker.used_length_mm + MARKER_TAIL_MM, MARKER_PLOT_MAX_MM) + 2 * MARKER_PAD_MM, 1),
            round(marker.fabric_width_mm + 2 * MARKER_PAD_MM, 1)]


def export_marker_plot_pdf(marker, title: str, path: str | BinaryIO, size: str = "",
                           verified: bool = True, force: bool = False) -> None:
    """Write the marker at exactly 1:1 for a wide-format plotter (kertas marker).

    The page is (marker length + 185 mm tail) x (fabric width + 20 mm): the
    roll only needs to be 20 mm wider than the fabric. The title block runs
    along the width in the tail, next to a 10 cm test square. Markers longer
    than 4.8 m (PDF viewers stop near 5 m) continue on further pages with
    50 mm overlap and dashed join lines.
    """
    _check_marker_print(verified, force)
    length, width = marker.used_length_mm, marker.fabric_width_mm
    total = length + MARKER_TAIL_MM
    starts = _windows(total, MARKER_PLOT_MAX_MM, MARKER_PLOT_OVERLAP_MM)
    canvas = Canvas(path, pageCompression=1)
    canvas.setTitle(f"SeamProof marker 1:1 {title}")
    canvas.setAuthor(f"SeamProof {__version__}")
    for index, start in enumerate(starts):
        span = min(MARKER_PLOT_MAX_MM, total - start)
        page_w = mm_to_pt(span + 2 * MARKER_PAD_MM)
        page_h = mm_to_pt(width + 2 * MARKER_PAD_MM)
        canvas.setPageSize((page_w, page_h))
        top, left = page_h - mm_to_pt(MARKER_PAD_MM), mm_to_pt(MARKER_PAD_MM)

        def to_pt(x: float, y: float, start=start) -> tuple[float, float]:
            return left + mm_to_pt(x - start), top - mm_to_pt(y)

        canvas.saveState()
        clip = canvas.beginPath()
        clip.rect(left, top - mm_to_pt(width), mm_to_pt(min(span, length - start)), mm_to_pt(width))
        canvas.clipPath(clip, stroke=0, fill=0)
        _draw_marker(canvas, marker, size, to_pt, MM_TO_PT, label_pt=28.0)
        canvas.restoreState()
        part = f" — part {index + 1}/{len(starts)}" if len(starts) > 1 else ""
        canvas.setFont("Helvetica", 9)
        canvas.drawString(left, mm_to_pt(3), f"{title}{part} — scale 1:1")
        for edge in (start, start + span):
            if 0 < edge < length:  # join lines where pages meet
                x, _ = to_pt(edge, 0)
                canvas.setDash(6, 4)
                canvas.line(x, top + 8, x, top - mm_to_pt(width) - 8)
                canvas.setDash()
        if start + span >= total - 1e-6:
            tail_x, _ = to_pt(length + 12, 0)
            canvas.saveState()
            canvas.translate(tail_x, top)
            # Turned -90 deg: text runs down the fabric width; local +y is page right.
            canvas.rotate(-90)
            _marker_header(canvas, marker, title + part, mm_to_pt(10), mm_to_pt(32), one_to_one=True, k=2.5,
                           arrow=False)
            if not verified:
                canvas.setFont("Helvetica-Bold", 24)
                canvas.drawString(mm_to_pt(10), mm_to_pt(40), NOT_VERIFIED)
            canvas.restoreState()
            sq_x, _ = to_pt(length + 70, 0)
            draw_test_square(canvas, sq_x, top - mm_to_pt(RULER_MM + 10))
        canvas.showPage()
    canvas.save()


def _marker_tiles(marker) -> tuple[list[tuple[int, int, float, float, bool]], int, int]:
    """A4 windows over the display marker: (row, col, x0, y0, has_content), rows, cols."""
    from shapely.geometry import Polygon as _Polygon
    from shapely.geometry import box as _box

    length, width = marker.used_length_mm, marker.fabric_width_mm
    xs = _windows(length, TILE_USABLE_W_MM, TILE_OVERLAP_MM)
    ys = _windows(width, TILE_USABLE_H_MM, TILE_OVERLAP_MM)
    shapes = [_Polygon([_marker_view(marker, x, y) for x, y in p.cut]) for p in marker.placements]
    edges = _box(0, 0, length, width).exterior
    tiles = []
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            window = _box(x0, y0, x0 + TILE_USABLE_W_MM, y0 + TILE_USABLE_H_MM)
            used = window.intersects(edges) or any(window.intersects(s) for s in shapes)
            tiles.append((row, col, x0, y0, used))
    return tiles, len(ys), len(xs)


def marker_tile_count(marker) -> int:
    """A4 sheets the tiled marker prints (sheet map included)."""
    tiles, _rows, _cols = _marker_tiles(marker)
    return 1 + sum(1 for tile in tiles if tile[4])


def export_marker_tiles_pdf(marker, title: str, path: str | BinaryIO, size: str = "",
                            verified: bool = True, force: bool = False) -> None:
    """Write the marker at exactly 1:1 on overlapping A4 sheets to tape together.

    Page 1 is the sheet map with a 10 cm test square; tiles with no fabric
    or piece on them are skipped and greyed on the map. Tiles overlap 15 mm
    and carry corner registration crosses, as the pattern tiles do.
    """
    _check_marker_print(verified, force)
    usable_w, usable_h = TILE_USABLE_W_MM, TILE_USABLE_H_MM
    tiles, n_rows, n_cols = _marker_tiles(marker)
    xs = sorted({t[2] for t in tiles})
    ys = sorted({t[3] for t in tiles})
    canvas = Canvas(path, pagesize=A4_PT, pageCompression=1)
    canvas.setTitle(f"SeamProof marker A4 {title}")
    canvas.setAuthor(f"SeamProof {__version__}")
    # Sheet map.
    if not verified:
        _stamp_not_verified(canvas)
    _marker_header(canvas, marker, title, mm_to_pt(10), A4_PT[1] - mm_to_pt(15), one_to_one=True)
    printed = sum(1 for t in tiles if t[4])
    canvas.setFont("Helvetica", 9)
    canvas.drawString(mm_to_pt(10), A4_PT[1] - mm_to_pt(25) - 14,
                      f"{printed} A4 sheets (of {len(tiles)}; blank sheets skipped). Tape them in order, "
                      f"overlap {TILE_OVERLAP_MM:.0f} mm, + mark meets +.")
    map_w = 190.0
    s = mm_to_pt(min(map_w / (xs[-1] + usable_w), 120.0 / (ys[-1] + usable_h)))
    left, top = mm_to_pt(10), A4_PT[1] - mm_to_pt(40)
    to_map = lambda x, y: (left + x * s, top - y * s)  # noqa: E731
    _draw_marker(canvas, marker, size, to_map, s, label_pt=4.0)
    canvas.setFont("Helvetica-Bold", 6)
    for row, col, x0, y0, used in tiles:
        canvas.setStrokeColor(HexColor("#2c5a8c") if used else HexColor("#bbbbbb"))
        canvas.setLineWidth(0.5)
        mx, my = to_map(x0, y0 + usable_h)
        canvas.rect(mx, my, usable_w * s, usable_h * s, stroke=1, fill=0)
        if used:
            canvas.setFillColor(HexColor("#2c5a8c"))
            canvas.drawString(mx + 2, my + usable_h * s - 8, f"R{row + 1}C{col + 1}")
    canvas.setFillColor(black)
    canvas.setStrokeColor(black)
    draw_test_square(canvas, mm_to_pt(10), mm_to_pt(25))
    canvas.showPage()
    for row, col, x0, y0, used in tiles:
        if not used:
            continue
        canvas.setPageSize(A4_PT)
        if not verified:
            _stamp_not_verified(canvas)
        cx0, cy0, cx1, cy1 = _tile_rect_pt(0, 0)

        def to_pt(x: float, y: float, x0=x0, y0=y0) -> tuple[float, float]:
            return cx0 + mm_to_pt(x - x0), cy1 - mm_to_pt(y - y0)

        canvas.saveState()
        clip = canvas.beginPath()
        clip.rect(cx0, cy0, cx1 - cx0, cy1 - cy0)
        canvas.clipPath(clip, stroke=0, fill=0)
        _draw_marker(canvas, marker, size, to_pt, MM_TO_PT, label_pt=16.0,
                     window=(x0, y0, x0 + usable_w, y0 + usable_h))
        canvas.restoreState()
        canvas.setLineWidth(1.0)
        for px, py in ((cx0, cy0), (cx1, cy0), (cx0, cy1), (cx1, cy1)):
            canvas.line(px - 6, py, px + 6, py)
            canvas.line(px, py - 6, px, py + 6)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(cx0, cy1 + 4, f"{title} R{row + 1}C{col + 1} (row {row + 1}/{n_rows}, "
                                        f"column {col + 1}/{n_cols})")
        canvas.drawString(cx0, cy0 - 12, "Scale 1:1 — print at 100%. The next column is to the right, the next row is below.")
        canvas.showPage()
    canvas.save()


def export_fit_overlay_svg(v1: Pattern, v2: Pattern, path: str | BinaryIO) -> None:
    """Overlay v1 (black) and v2 (green dashed) outlines, one row per panel, in mm."""
    gap = 20.0
    names = [panel.name for panel in v1.panels]
    by_name_2 = {panel.name: panel for panel in v2.panels}
    rows = []
    for name in names:
        if name not in by_name_2:
            continue
        first = next(p for p in v1.panels if p.name == name)
        second = by_name_2[name]
        boxes = [panel_bounds(first), panel_bounds(second)]
        rows.append((name, first, second, boxes))
    total_h = sum(max(b[3] - b[1] for b in boxes) for _n, _f, _s, boxes in rows) + gap * len(rows)
    total_w = max(max(b[2] - b[0] for b in boxes) for _n, _f, _s, boxes in rows)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.1f}mm" '
             f'height="{total_h:.1f}mm" viewBox="0 0 {total_w:.1f} {total_h:.1f}">',
             '<text x="2" y="10" font-size="6">black=v1, green dashed=v2 (mm)</text>']
    y_off = 14.0
    for name, first, second, boxes in rows:
        (a0, a1, _a2, _a3), (b0, b1, _b2, _b3) = boxes
        for panel, boxes_min, color, dash in ((first, (a0, a1), "black", ""),
                                             (second, (b0, b1), "green", ' stroke-dasharray="3,2"')):
            min_x, min_y = boxes_min
            for edge in panel.edges:
                points = " ".join(f"{p.x - min_x:.1f},{p.y - min_y + y_off:.1f}"
                                  for p in edge.curve.points)
                lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                             f'stroke-width="0.5"{dash}/>')
        lines.append(f'<text x="2" y="{y_off - 2:.1f}" font-size="6">{name}</text>')
        y_off += max(b[3] - b[1] for b in boxes) + gap
    lines.append("</svg>")
    content = "\n".join(lines) + "\n"
    if isinstance(path, str):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write(content.encode("utf-8"))


def export_production_pdf(report, path: str | BinaryIO) -> None:
    """Write the production summary plus one denah page per size."""
    canvas = Canvas(path, pagesize=A4_PT, pageCompression=0)
    canvas.setTitle(f"SeamProof production {report.garment}")
    canvas.setAuthor(f"SeamProof {__version__}")
    draw_production_summary(canvas, report)
    for item in report.sizes:
        draw_marker_page(canvas, item.marker, f"{report.garment} {item.size} x{item.qty}", item.size)
    canvas.save()


def export_report_md(report, path: str | BinaryIO, explanation=None) -> None:
    """Write the proof report as Markdown, plus tailor instructions."""
    meta = report.meta
    lines = [f"# SeamProof Proof Report — {report.garment} {report.size}",
             "",
             f"- Fabric: {report.fabric}",
             f"- Version: {meta.get('version', '')}",
             f"- Parameter hash: `{meta.get('params_hash', '')}`",
             f"- Repair rounds: {meta.get('repair_rounds', 0)}",
             (f"- Checked by pattern maker: {meta['signoff']['name']} ({meta['signoff']['at']})"
              + (f" — {meta['signoff']['note']}" if meta['signoff'].get('note') else "")
              if meta.get("signoff") else "- Checked by pattern maker: not yet"),
             f"- Status: **{'PASS' if report.passed else 'FAIL'}** "
             f"({sum(c.passed for c in report.checks)}/{len(report.checks)})",
             "",
             "| ID | Check | Value | Expected | Tolerance | Result |",
             "|----|-------------|-------|---------|-----------|-------|"]
    for check in report.checks:
        lines.append(f"| {check.id} | {check.name_id} | {_fmt(check.actual)} | {_fmt(check.expected)} | "
                     f"{_fmt(check.tolerance)} | {'PASS' if check.passed else 'FAIL'} |")
    lines += ["", report.summary_id(), ""]
    gathers = meta.get("gathers_mm") or {}
    if gathers:
        lines += ["## Gathers (from the pattern, computed by code)", ""]
        if gathers.get("sleeve_hem"):
            lines.append(f"- Sleeve hem: gather {gathers['sleeve_hem']:.0f} mm of excess into the cuff (match the centre notch).")
        if gathers.get("sleeve_cap"):
            lines.append(f"- Sleeve cap: gather {gathers['sleeve_cap']:.0f} mm between the front and back cap notches.")
        if gathers.get("elastic_cut"):
            lines.append(f"- Cuff elastic: cut {gathers['elastic_cut']:.0f} mm per cuff (before joining).")
        lines.append("")
    if explanation is not None:
        summary, instructions = explanation
        lines += ["## Tailor explanation", "", summary, ""]
        for item in instructions:
            amount = "—" if item.amount_mm is None else f"{item.amount_mm:.1f} mm"
            lines.append(f"- **{item.panel} / {item.edge}** ({item.action}, {amount}): {item.text}")
        lines.append("")
    content = "\n".join(lines)
    if isinstance(path, str):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
    else:
        path.write(content.encode("utf-8"))


# Nested grading PDF: every size overlaid at 1:1, one page per panel name.
# (color, dash pattern, style name) per position in the size list.
GRADE_STYLES = [
    (black, [], "solid black"),
    (HexColor("#1a56db"), [5, 4], "dashed blue"),
    (HexColor("#c81e1e"), [2, 2], "dotted red"),
    (HexColor("#057a55"), [8, 3], "long-dash green"),
]


def _grade_style(index: int) -> tuple:
    """Return the line style for the i-th graded size."""
    return GRADE_STYLES[index % len(GRADE_STYLES)]


def draw_graded_proof_page(canvas: Canvas, report, explanations=None) -> None:
    """Draw page 1: per-size proof table, grade checks, legend, test square."""
    from seamproof.grading import GradeReport

    assert isinstance(report, GradeReport)
    canvas.setPageSize(A4_PT)
    if not report.passed:
        _stamp_not_verified(canvas)
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(50, 800, f"SeamProof grading — {report.garment} ({report.fabric})")
    canvas.setFont("Helvetica", 11)
    canvas.drawString(50, 778, f"Sizes: {', '.join(report.sizes)} — scale 1:1, check the 10 cm square.")
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(50, 756, "Proof per size")
    canvas.setFont("Helvetica", 9)
    y = 740
    for size in report.sizes:
        single = report.reports[size]
        good = sum(c.passed for c in single.checks)
        canvas.drawString(60, y, f"{size}: {'PASS' if single.passed else 'FAIL'} "
                                f"{good}/{len(single.checks)} — "
                                f"{report.patterns[size].metadata.get('params_hash', '')[:8]}")
        y -= 13
    y -= 8
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(50, y, "Grading checks")
    canvas.setFont("Helvetica", 9)
    y -= 16
    for check in report.grade_checks:
        canvas.drawString(60, y, f"{check.id} {check.name_id[:30]}: "
                                f"{'PASS' if check.passed else 'FAIL'}")
        y -= 13
    y -= 8
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(50, y, "Legend (pattern pages stack all sizes at 1:1)")
    canvas.setFont("Helvetica", 9)
    y -= 16
    for index, size in enumerate(report.sizes):
        _color, _dash, style = _grade_style(index)
        canvas.drawString(60, y, f"{size} = {style}")
        y -= 13
    canvas.drawString(60, y, "Cut lines: see the per-size PDF (draft).")
    if explanations:
        y -= 19
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(50, y, "Tailor explanation (details in grade_report.md):")
        canvas.setFont("Helvetica", 9)
        shown = 0
        for size in report.sizes:
            if shown >= 4 or size not in explanations:
                continue
            _summary, instructions = explanations[size]
            if instructions:
                y -= 13
                canvas.drawString(60, y, f"{size}: {instructions[0].text[:85]}")
                shown += 1
    draw_test_square(canvas, 50, 60)
    canvas.showPage()


def draw_nested_panel_page(canvas: Canvas, panel_name: str, report) -> None:
    """Draw one panel with every size overlaid at exactly 1:1."""
    from seamproof.grading import GradeReport

    assert isinstance(report, GradeReport)
    ordered = [(size, report.patterns[size]) for size in report.sizes]
    boxes = [panel_bounds(next(p for p in pattern.panels if p.name == panel_name))
             for _size, pattern in ordered]
    bounds = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes))
    canvas.setPageSize((mm_to_pt(bounds[2] - bounds[0] + 2 * MARGIN_MM),
                        mm_to_pt(bounds[3] - bounds[1] + 2 * MARGIN_MM)))
    if not report.passed:
        _stamp_not_verified(canvas)
    for index, (_size, pattern) in enumerate(ordered):
        color, dash, _style = _grade_style(index)
        panel = next(p for p in pattern.panels if p.name == panel_name)
        canvas.saveState()
        canvas.setStrokeColor(color)
        canvas.setLineWidth(0.8)
        if dash:
            canvas.setDash(*dash)
        for edge in panel.edges:
            placed = _placed_curve_capped(place_curve(edge.curve, bounds))
            if len(placed) > 1:
                _stroke_placed(canvas, placed)
        canvas.restoreState()
    # Grainline and notches from the smallest size only, to avoid clutter.
    base = next(p for p in ordered[0][1].panels if p.name == panel_name)
    canvas.setStrokeColor(black)
    canvas.setLineWidth(1.5)
    grain = _placed_curve_capped(place_curve(Curve(base.grainline), bounds))
    _stroke_placed(canvas, grain)
    (gx0, gy0), (gx1, gy1) = grain
    canvas.drawString(gx1 + 4, gy1 + 4, "SG")
    known = {edge.id: edge for edge in base.edges}
    canvas.setLineWidth(1.0)
    for notch in base.notches:
        edge = known.get(notch.edge_id)
        if edge is None:
            continue
        spot = _notch_point(edge.curve, notch.distance_from_start_mm)
        if spot is None:
            continue
        x, y = place_point(spot, bounds)
        canvas.rect(x - 2, y - 2, 4, 4, stroke=1, fill=0)
    canvas.setFont("Helvetica-Bold", 12)
    canvas.drawString(mm_to_pt(MARGIN_MM), mm_to_pt(MARGIN_MM) + 34,
                      f"{panel_name} — {report.garment} {', '.join(report.sizes)}")
    canvas.setFont("Helvetica", 9)
    legend = "   ".join(f"{size}={_grade_style(i)[2]}" for i, size in enumerate(report.sizes))
    canvas.drawString(mm_to_pt(MARGIN_MM), mm_to_pt(MARGIN_MM) + 20, legend)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(mm_to_pt(MARGIN_MM), mm_to_pt(MARGIN_MM) + 8,
                      f"{RULER_MM:.0f} mm at 100% print (check the ruler on the proof page)")
    canvas.showPage()


def export_pdf_graded(report, path: str | BinaryIO, force: bool = False,
                      explanations=None) -> None:
    """Write the per-size proof table plus nested 1:1 panel overlays.

    Refuses failing grades unless forced; forced pages carry NOT VERIFIED.
    Stitch lines nest for every size; cut lines stay in the per-size PDFs.
    """
    from seamproof.grading import GradeReport

    assert isinstance(report, GradeReport)
    if not report.passed and not force:
        raise ValueError("Grading failed verification — PDF refused. Use --force to print with a NOT VERIFIED stamp.")
    canvas = Canvas(path, pagesize=A4_PT, pageCompression=0)
    canvas.setTitle(f"SeamProof grading {report.garment} {','.join(report.sizes)}")
    canvas.setAuthor(f"SeamProof {__version__}")
    draw_graded_proof_page(canvas, report, explanations)
    names = [panel.name for panel in report.patterns[report.sizes[0]].panels]
    for name in names:
        draw_nested_panel_page(canvas, name, report)
    canvas.save()
