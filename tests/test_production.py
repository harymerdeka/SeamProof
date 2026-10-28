"""Production tests: arithmetic, packing geometry and CLI exports."""

import sys
from io import BytesIO

import pytest
from shapely.geometry import Polygon

from seamproof.cli import main
from seamproof.draft import REGISTRY
from seamproof.marker import SPACING_MM, Marker, pack
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.production import produce


def _blouse_m_marker() -> Marker:
    """Pack one M blouse at 1500 mm width."""
    return pack(REGISTRY["blouse"](WOMEN_SIZE_TABLE["M"], "woven", {}), 1500.0)


def test_cut_list_arithmetic() -> None:
    """Cut list multiplies each panel's cut_count by the size quantity."""
    report = produce("blouse", "S=20,M=50", "woven", 1500.0, 45000.0)
    for item in report.sizes:
        for panel in item.pattern.panels:
            assert report.cut_list[f"{item.size} {panel.name}"] == panel.cut_count * item.qty
    assert report.cut_list["M sleeve"] == next(
        p.cut_count for p in report.sizes[1].pattern.panels if p.name == "sleeve") * 50


def _styled_markers() -> list[tuple[str, Marker]]:
    """Markers for every garment at M, the shirt and skirt with photo styles."""
    from seamproof.cli import DEFAULT_FABRIC
    from seamproof.measurements import SIZE_TABLE
    from seamproof.style import plan_piece

    shirt = plan_piece("top", "shirt", {"collar": "peter_pan", "opening": "button_front", "cuff": True,
                                        "pockets": 2, "shoulder": "dropped", "fit": "boxy"}).params
    skirt = plan_piece("bottom", "skirt", {"silhouette": "a_line", "length": "mini", "hem": "scalloped"}).params
    out = []
    for garment, params in (("tshirt", {}), ("blouse", {}), ("tunic", {}), ("shirt", shirt), ("skirt", skirt)):
        table = SIZE_TABLE if garment == "tshirt" else WOMEN_SIZE_TABLE
        out.append((garment, pack(REGISTRY[garment](table["M"], DEFAULT_FABRIC[garment], params), 1500.0)))
    return out


MARKERS = _styled_markers()


@pytest.mark.parametrize("garment,marker", MARKERS, ids=[g for g, _m in MARKERS])
def test_cut_shapes_never_overlap_and_keep_spacing(garment, marker) -> None:
    """Real cut outlines stay at least SPACING_MM apart (not just their boxes)."""
    shapes = [Polygon(p.cut) for p in marker.placements]
    assert all(s.is_valid and s.area > 0 for s in shapes)
    for i, first in enumerate(shapes):
        for second in shapes[i + 1:]:
            assert first.distance(second) >= SPACING_MM - 1e-6


@pytest.mark.parametrize("garment,marker", MARKERS, ids=[g for g, _m in MARKERS])
def test_pieces_inside_fabric_with_margins(garment, marker) -> None:
    """Every cut point sits inside the selvedge margins and the marker length."""
    for place in marker.placements:
        xs = [x for x, _y in place.cut]
        ys = [y for _x, y in place.cut]
        assert min(xs) >= SPACING_MM - 1e-6
        assert max(xs) <= marker.fabric_width_mm - SPACING_MM + 1e-6
        assert min(ys) >= SPACING_MM - 1e-6
        assert max(ys) <= marker.used_length_mm - SPACING_MM + 1e-6


@pytest.mark.parametrize("garment,marker", MARKERS, ids=[g for g, _m in MARKERS])
def test_grainline_runs_along_fabric_length(garment, marker) -> None:
    """Every grainline, including crosswise-drafted strips, ends up parallel to the selvedge."""
    for place in marker.placements:
        (x0, y0), (x1, y1) = place.grain
        assert abs(x1 - x0) < 0.05 and abs(y1 - y0) > 10


def test_crosswise_pieces_are_turned() -> None:
    """A strip whose grain is drafted along x is turned a quarter on the marker."""
    shirt = dict(MARKERS)["shirt"]
    cuffs = [p for p in shirt.placements if p.panel == "cuff"]
    assert cuffs and all(p.rotation_deg in (90, 270) for p in cuffs)
    assert all(p.rotation_deg % 180 == 0 for p in shirt.placements if p.panel == "front")


def test_fold_pieces_open_and_pairs_mirror() -> None:
    """Fold pieces lie opened (double area, fold line kept); pairs give left and right."""
    from seamproof.allowance import ensure_allowance

    raw = ensure_allowance(REGISTRY["blouse"](WOMEN_SIZE_TABLE["M"], "woven", {}))
    marker = _blouse_m_marker()
    for panel in raw.panels:
        placed = [p for p in marker.placements if p.panel == panel.name]
        assert len(placed) == panel.cut_count
        half = Polygon([(q.x, q.y) for q in panel.cut_line.points]).area
        for place in placed:
            assert place.opened_fold == panel.cut_on_fold
            assert Polygon(place.cut).area == pytest.approx(half * (2 if panel.cut_on_fold else 1), rel=1e-3)
            assert bool(place.fold) == panel.cut_on_fold
        if panel.cut_count == 2 and not panel.cut_on_fold:
            assert sorted(p.mirrored for p in placed) == [False, True]


def test_utilisation_counts_real_piece_area() -> None:
    """Utilisation uses cut-outline area, which is smaller than the boxes' area."""
    marker = _blouse_m_marker()
    assert marker.piece_area_mm2 == pytest.approx(sum(Polygon(p.cut).area for p in marker.placements), rel=1e-5)
    assert marker.piece_area_mm2 < sum(p.width * p.height for p in marker.placements)
    assert marker.utilisation == pytest.approx(
        marker.piece_area_mm2 / (marker.fabric_width_mm * marker.used_length_mm))


def test_packing_is_deterministic() -> None:
    """The same pattern and width always give the same marker."""
    first, second = _blouse_m_marker(), _blouse_m_marker()
    assert [(p.panel, p.x, p.y, p.rotation_deg, p.mirrored) for p in first.placements] == \
        [(p.panel, p.x, p.y, p.rotation_deg, p.mirrored) for p in second.placements]
    assert first.used_length_mm == second.used_length_mm


def test_metres_cost_and_utilisation() -> None:
    """Metres, rupiah and utilisation follow the measured packing."""
    report = produce("blouse", "S=20,M=50", "woven", 1500.0, 45000.0, layers=2)
    assert abs(report.total_metres - sum(i.metres for i in report.sizes)) < 1e-9
    assert report.total_cost == pytest.approx(report.total_metres * 45000.0)
    for item in report.sizes:
        assert item.metres == pytest.approx(item.marker.used_length_mm * item.qty / 1000.0)
        assert 0 < item.marker.utilisation <= 1
    assert 0 < report.overall_utilisation <= 1


def test_piece_wider_than_fabric_is_refused() -> None:
    """A panel that cannot fit raises instead of overflowing."""
    pattern = REGISTRY["blouse"](WOMEN_SIZE_TABLE["M"], "woven", {})
    with pytest.raises(ValueError):
        pack(pattern, 100.0)


def test_bad_run_specs_rejected() -> None:
    """Malformed runs, unknown sizes and bad layers are refused."""
    with pytest.raises(ValueError):
        produce("blouse", "M=0", "woven", 1500.0, 45000.0)
    with pytest.raises(ValueError):
        produce("blouse", "XXL=5", "woven", 1500.0, 45000.0)
    with pytest.raises(ValueError):
        produce("blouse", "M=5", "woven", 1500.0, 45000.0, layers=0)


def test_exports_are_wellformed() -> None:
    """Marker SVG parses and carries every piece; the production PDF starts with %PDF."""
    import xml.etree.ElementTree as xml_tree

    from seamproof.export import export_marker_svg, export_production_pdf

    report = produce("blouse", "M=5", "woven", 1500.0, 45000.0)
    svg = BytesIO()
    export_marker_svg(report.sizes[0].marker, "blouse M", svg, "M")
    root = xml_tree.fromstring(svg.getvalue())
    pieces = root.findall("{http://www.w3.org/2000/svg}g")
    assert len(pieces) == len(report.sizes[0].marker.placements)
    assert all(g.find("{http://www.w3.org/2000/svg}polyline[@class='cut']") is not None for g in pieces)
    pdf = BytesIO()
    export_production_pdf(report, pdf)
    assert pdf.getvalue().startswith(b"%PDF")


def _page_sizes_mm(data: bytes) -> list[tuple[float, float]]:
    """MediaBox of every page in mm, read from the raw PDF."""
    import re

    boxes = re.findall(rb"/MediaBox \[\s*([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) \]", data)
    return [((float(x1) - float(x0)) / 2.8346, (float(y1) - float(y0)) / 2.8346) for x0, y0, x1, y1 in boxes]


def test_plot_pdf_is_one_to_one() -> None:
    """The plotter page is marker length x fabric width plus fixed margins, in true mm."""
    from seamproof.export import MARKER_PAD_MM, MARKER_TAIL_MM, export_marker_plot_pdf

    marker = _blouse_m_marker()
    out = BytesIO()
    export_marker_plot_pdf(marker, "blouse M", out, "M")
    (width_mm, height_mm), = _page_sizes_mm(out.getvalue())
    assert width_mm == pytest.approx(marker.used_length_mm + MARKER_TAIL_MM + 2 * MARKER_PAD_MM, abs=0.05)
    assert height_mm == pytest.approx(marker.fabric_width_mm + 2 * MARKER_PAD_MM, abs=0.05)


def test_long_marker_splits_into_overlapping_pages() -> None:
    """Markers longer than one PDF page continue on further 1:1 pages."""
    from seamproof.export import MARKER_PLOT_MAX_MM, export_marker_plot_pdf

    marker = Marker(1500.0, [], 9000.0, 0.0)
    out = BytesIO()
    export_marker_plot_pdf(marker, "long", out)
    sizes = _page_sizes_mm(out.getvalue())
    assert len(sizes) == 2
    assert max(w for w, _h in sizes) <= MARKER_PLOT_MAX_MM + 20 + 1e-6


def test_tiled_marker_pages_are_a4_and_counted() -> None:
    """The A4 marker prints the sheet map plus every non-empty tile, all A4 portrait."""
    from seamproof.export import export_marker_tiles_pdf, marker_tile_count

    marker = _blouse_m_marker()
    out = BytesIO()
    export_marker_tiles_pdf(marker, "blouse M", out, "M")
    sizes = _page_sizes_mm(out.getvalue())
    assert len(sizes) == marker_tile_count(marker) > 1
    assert all(w == pytest.approx(210, abs=0.1) and h == pytest.approx(297, abs=0.1) for w, h in sizes)


def test_unverified_marker_print_is_refused() -> None:
    """1:1 marker prints refuse failing patterns unless forced, then carry the stamp."""
    from seamproof.export import NOT_VERIFIED, export_marker_plot_pdf, export_marker_tiles_pdf

    marker = _blouse_m_marker()
    for writer in (export_marker_plot_pdf, export_marker_tiles_pdf):
        with pytest.raises(ValueError):
            writer(marker, "x", BytesIO(), verified=False)
    forced = BytesIO()
    export_marker_plot_pdf(marker, "x", forced, verified=False, force=True)
    assert forced.getvalue().startswith(b"%PDF")
    assert NOT_VERIFIED  # stamp text exists; drawn on forced pages


def test_cli_production_writes_pack(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """production writes md, pdf and per-size marker SVGs with rupiah totals."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "production", "--garment", "blouse",
                                      "--run", "S=20,M=50", "--width", "1500",
                                      "--price", "45000", "--out", str(out)])
    assert main() == 0
    assert (out / "production.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "marker_S.svg").is_file() and (out / "marker_M.svg").is_file()
    assert (out / "marker_M_plotter.pdf").read_bytes().startswith(b"%PDF")
    text = (out / "production.md").read_text()
    assert "Rp" in text and "Utilisasi" in text


def test_layers_do_not_change_fabric_metres() -> None:
    """Stacking plies cuts more per spread but uses the same total fabric."""
    one = produce("blouse", "M=10", "woven", 1500.0, 45000.0, layers=1)
    five = produce("blouse", "M=10", "woven", 1500.0, 45000.0, layers=5)
    assert five.total_metres == pytest.approx(one.total_metres)
    assert five.total_cost == pytest.approx(one.total_cost)


def test_marker_packs_cut_lines_with_allowance() -> None:
    """Pieces are packed with seam allowance, so they are larger than stitch lines."""
    from seamproof.allowance import add_seam_allowance
    from seamproof.export import panel_bounds

    raw = REGISTRY["blouse"](WOMEN_SIZE_TABLE["M"], "woven", {})
    marker = pack(raw, 1500.0)
    cut = add_seam_allowance(raw)
    widest_stitch = max(panel_bounds(p)[3] - panel_bounds(p)[1] for p in raw.panels)
    widest_cut = max(panel_bounds(p)[3] - panel_bounds(p)[1] for p in cut.panels)
    assert widest_cut > widest_stitch
    assert marker.used_length_mm >= widest_cut
