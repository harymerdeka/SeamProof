"""1:1 PDF export tests: scale math and document structure, offline."""

import pytest

from seamproof.draft.blouse import draft_blouse
from seamproof.draft.tunic import draft_tunic
from seamproof.export import (
    MM_TO_PT,
    draw_pattern_pdf,
    panel_bounds,
    panel_page_size,
    place_curve,
    place_point,
)
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.pattern import Curve, Edge, Panel, Pattern, Point


@pytest.fixture
def blouse_m() -> Pattern:
    """Return a size M woven blouse."""
    return draft_blouse(WOMEN_SIZE_TABLE["M"], "woven")


def test_scale_factor_is_exact() -> None:
    """One millimetre is exactly 2.8346 points; 100 mm is 283.46 pt."""
    assert MM_TO_PT == 2.8346
    assert MM_TO_PT * 100 == pytest.approx(283.46)


def test_page_size_and_placement() -> None:
    """A 100x200 panel maps onto points with margin and a flipped y axis."""
    panel = Panel("test", [
        Edge("t.a", Curve((Point(0, 0), Point(100, 0)))),
        Edge("t.b", Curve((Point(100, 0), Point(100, 200)))),
        Edge("t.c", Curve((Point(100, 200), Point(0, 200)))),
        Edge("t.d", Curve((Point(0, 200), Point(0, 0)))),
    ], (Point(50, 20), Point(50, 180)))
    assert panel_bounds(panel) == (0, 0, 100, 200)
    assert panel_page_size(panel) == pytest.approx((120 * MM_TO_PT, 220 * MM_TO_PT))
    low = place_point(Point(0, 0), (0, 0, 100, 200))
    high = place_point(Point(100, 200), (0, 0, 100, 200))
    assert low == pytest.approx((10 * MM_TO_PT, 210 * MM_TO_PT))
    assert high == pytest.approx((110 * MM_TO_PT, 10 * MM_TO_PT))


def test_100mm_edge_measures_283pt() -> None:
    """A 100 mm stitch edge spans exactly 283.46 pt on the page."""
    import math

    curve = Curve((Point(0, 50), Point(100, 50)))
    placed = place_curve(curve, (0, 0, 100, 200))
    length = math.hypot(placed[1][0] - placed[0][0], placed[1][1] - placed[0][1])
    assert length == pytest.approx(283.46)


def _page_count(data: bytes) -> int:
    """Read the authoritative page count from the PDF page tree root."""
    import re

    counts = re.findall(rb"/Count (\d+)", data)
    assert counts
    return max(int(value) for value in counts)


def test_pdf_structure_has_cover_and_panels(blouse_m: Pattern) -> None:
    """The PDF opens with %PDF, names every panel and paginates cover plus panels."""
    from io import BytesIO

    stream = BytesIO()
    draw_pattern_pdf(blouse_m, stream)
    data = stream.getvalue()
    assert data.startswith(b"%PDF")
    for panel in blouse_m.panels:
        assert panel.name.encode() in data
    assert b"100 mm" in data and b"scale 1:1" in data
    assert _page_count(data) == len(blouse_m.panels) + 1


def test_pdf_tunic(blouse_m: Pattern) -> None:
    """The tunic exports with the same structure."""
    from io import BytesIO

    pattern = draft_tunic(WOMEN_SIZE_TABLE["M"], "woven")
    stream = BytesIO()
    draw_pattern_pdf(pattern, stream)
    assert _page_count(stream.getvalue()) == len(pattern.panels) + 1


def test_square_is_283pt() -> None:
    """The proof test square draws exactly 283.46 x 283.46 pt."""
    from io import BytesIO

    from reportlab.pdfgen.canvas import Canvas

    from seamproof.export import draw_test_square

    stream = BytesIO()
    canvas = Canvas(stream, pageCompression=0)
    assert draw_test_square(canvas, 50, 50) == pytest.approx((283.46, 283.46))
    canvas.save()
    assert "Measure: must be 10 cm".encode() in stream.getvalue()


def test_tiled_pdf_paginates(blouse_m: Pattern) -> None:
    """The tiled PDF holds the proof page plus one A4 page per tile."""
    from io import BytesIO

    from seamproof.export import export_pdf_tiled, panel_bounds, tile_grid
    from seamproof.verify import verify_pattern

    report = verify_pattern(blouse_m)
    assert report.passed
    stream = BytesIO()
    export_pdf_tiled(blouse_m, report, stream)
    data = stream.getvalue()
    assert data.startswith(b"%PDF")
    expected = 1 + sum(tile_grid(*panel_bounds(panel))[:2][0] * tile_grid(*panel_bounds(panel))[1]
                       for panel in blouse_m.panels)
    assert _page_count(data) == expected
    assert b"R1C1" in data and "cut".encode() in data


def test_tiled_pdf_refuses_failing_pattern(blouse_m: Pattern) -> None:
    """A failing report is refused unless forced, and forced pages are stamped."""
    import copy

    from io import BytesIO

    from seamproof.export import NOT_VERIFIED, export_pdf_tiled
    from seamproof.verify import verify_pattern

    broken = copy.deepcopy(blouse_m)
    broken.params["unlogged_tweak"] = 1.0
    report = verify_pattern(broken)
    assert not report.passed
    with pytest.raises(ValueError, match="refused"):
        export_pdf_tiled(broken, report, BytesIO())
    stream = BytesIO()
    export_pdf_tiled(broken, report, stream, force=True)
    assert NOT_VERIFIED.encode() in stream.getvalue()


def test_svg_preview_is_valid_xml(blouse_m: Pattern, tmp_path) -> None:
    """The SVG preview parses, uses millimetres and names every panel."""
    import xml.etree.ElementTree as ET

    from seamproof.export import export_svg

    target = tmp_path / "pattern.svg"
    export_svg(blouse_m, str(target))
    root = ET.fromstring(target.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    assert root.attrib["width"].endswith("mm")
    ids = {element.attrib.get("id") for element in root.iter()}
    for panel in blouse_m.panels:
        assert panel.name in ids


def test_report_md_lists_every_check(blouse_m: Pattern, tmp_path) -> None:
    """The Markdown report tables every check with its verdict."""
    from seamproof.export import export_report_md
    from seamproof.verify import verify_pattern

    report = verify_pattern(blouse_m)
    target = tmp_path / "report.md"
    export_report_md(report, str(target))
    text = target.read_text(encoding="utf-8")
    assert "PASS" in text and blouse_m.metadata["params_hash"] in text
    for check in report.checks:
        assert f"| {check.id} |" in text


def test_notch_point_walks_past_first_segment() -> None:
    """A notch beyond the first segment lands on the curve, not dropped."""
    from seamproof.export import _notch_point

    curve = Curve((Point(0, 0), Point(10, 0), Point(10, 10), Point(10, 30)))
    assert _notch_point(curve, 5) == Point(5, 0)
    assert _notch_point(curve, 25) == Point(10, 15)
    assert _notch_point(curve, 41) is None


def test_svg_draws_every_blouse_notch() -> None:
    """Every in-range notch of the blouse is drawn in the SVG preview."""
    import io

    from seamproof.export import export_svg

    pattern = draft_blouse(WOMEN_SIZE_TABLE["M"], "woven", {})
    stream = io.BytesIO()
    export_svg(pattern, stream)
    drawn = stream.getvalue().decode("utf-8").count('stroke="red"')
    assert drawn == sum(len(panel.notches) for panel in pattern.panels)


def test_exports_always_carry_seam_allowance() -> None:
    """SVG and PDF from a stitch-only draft include the cut line and allowance note."""
    import io

    from seamproof.export import export_pdf_tiled, export_svg
    from seamproof.verify import verify_pattern as verify

    pattern = draft_blouse(WOMEN_SIZE_TABLE["M"], "woven", {})
    assert all(panel.cut_line is None for panel in pattern.panels)
    svg = io.BytesIO()
    export_svg(pattern, svg)
    assert svg.getvalue().decode("utf-8").count('class="cut"') == len(pattern.panels)
    pdf = io.BytesIO()
    export_pdf_tiled(pattern, verify(pattern), pdf)
    assert b"seam allowance 10 mm, hem 25 mm" in pdf.getvalue()


def test_dxf_refuses_failing_pattern_and_is_in_mm() -> None:
    """DXF export mirrors the PDF gate and declares millimetre units."""
    import io

    import ezdxf

    from seamproof.export import export_dxf
    from seamproof.verify import Report, verify_pattern

    pattern = draft_blouse(WOMEN_SIZE_TABLE["M"], "woven", {})
    with pytest.raises(ValueError):
        export_dxf(pattern, Report("blouse", "M", "woven", False, []), io.BytesIO())
    stream = io.BytesIO()
    export_dxf(pattern, verify_pattern(pattern), stream)
    doc = ezdxf.read(io.StringIO(stream.getvalue().decode("utf-8")))
    assert doc.header["$INSUNITS"] == 4
    cuts = [e for e in doc.modelspace() if e.dxf.layer == "CUT"]
    assert len(cuts) == len(pattern.panels) and all(e.closed for e in cuts)
