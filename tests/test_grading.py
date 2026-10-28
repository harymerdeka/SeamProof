"""Grading tests: per-size verification, consistency and the nested PDF."""

import re
import sys
from io import BytesIO

import pytest

from seamproof.cli import main
from seamproof.grading import grade, grade_and_verify, grade_consistency

SIZES = ["S", "M", "L", "XL"]


def _swap_curve(pattern, panel_name, edge_id, curve) -> None:
    """Point one pattern edge at another pattern's curve, in place."""
    for panel in pattern.panels:
        if panel.name == panel_name:
            for edge in panel.edges:
                if edge.id == edge_id:
                    edge.curve = curve


def _shrink_xl_armhole(patterns) -> None:
    """Mutation: give XL the shorter S armholes (XL shrinking below L)."""
    curves = {}
    for panel in patterns["S"].panels:
        for edge in panel.edges:
            if edge.id in ("front.armhole", "back.armhole"):
                curves[edge.id] = edge.curve
    panel_of = {"front.armhole": "front", "back.armhole": "back"}
    for edge_id, curve in curves.items():
        _swap_curve(patterns["XL"], panel_of[edge_id], edge_id, curve)


def test_grade_runs_verifier_on_every_size() -> None:
    """Every size gets a full independent verifier report."""
    report = grade_and_verify("blouse", SIZES, "woven")
    assert report.sizes == SIZES
    assert set(report.patterns) == set(SIZES) == set(report.reports)
    for size in SIZES:
        assert len(report.reports[size].checks) > 0
        assert report.patterns[size].size_label == size


def test_all_sizes_pass_for_blouse_and_tunic() -> None:
    """Blouse and tunic pass every check on every size with woven ease."""
    for garment in ("blouse", "tunic"):
        report = grade_and_verify(garment, SIZES, "woven")
        assert report.passed, report.summary_id()
        assert all(check.passed for check in report.grade_checks)


def test_consistency_passes_on_clean_grades() -> None:
    """Monotonic progression holds for t-shirt and blouse geometries."""
    for garment, fabric in (("tshirt", "knit_low_stretch"), ("blouse", "woven")):
        patterns = grade(garment, SIZES, fabric)
        checks = [c for c in grade_consistency(patterns) if c.id != "GR2"]
        assert all(c.passed for c in checks), [(c.id, c.message_id) for c in checks if not c.passed]


def test_consistency_catches_shrinking_xl_armhole() -> None:
    """MUTATION: an XL armhole shorter than L fails exactly GR3-armhole."""
    patterns = grade("blouse", SIZES, "woven")
    _shrink_xl_armhole(patterns)
    failed = {c.id for c in grade_consistency(patterns) if not c.passed}
    assert failed == {"GR3-armhole"}


def test_extreme_sizes_required() -> None:
    """Grading without the smallest or largest size is refused."""
    with pytest.raises(ValueError):
        grade("blouse", ["M", "L", "XL"], "woven")
    with pytest.raises(ValueError):
        grade("blouse", ["S", "M", "L"], "woven")
    report = grade_and_verify("blouse", ["S", "XL"], "woven")
    assert report.sizes == ["S", "XL"]


def test_dart_intake_and_apex_progress_monotonically() -> None:
    """Bust dart intake grows and the apex tracks outward across sizes."""
    patterns = grade("blouse", SIZES, "woven")
    intakes = [patterns[size].metadata["darts"]["front_bust"]["intake_mm"] for size in SIZES]
    assert intakes == sorted(intakes) and len(set(intakes)) == len(intakes)
    tips = [patterns[size].metadata["darts"]["front_bust"]["tip"] for size in SIZES]
    assert [t[0] for t in tips] == sorted(t[0] for t in tips)
    assert [t[1] for t in tips] == sorted(t[1] for t in tips)


def _page_count(data: bytes) -> int:
    """Read the authoritative page count from the PDF page tree root."""
    counts = re.findall(rb"/Count (\d+)", data)
    assert counts
    return max(int(value) for value in counts)


def test_graded_pdf_nests_all_sizes() -> None:
    """The graded PDF holds the proof page plus one overlay page per panel."""
    from reportlab.pdfgen.canvas import Canvas

    from seamproof.export import draw_test_square, export_pdf_graded

    report = grade_and_verify("blouse", SIZES, "woven")
    assert report.passed
    stream = BytesIO()
    export_pdf_graded(report, stream)
    data = stream.getvalue()
    assert data.startswith(b"%PDF")
    assert _page_count(data) == 1 + len(report.patterns["S"].panels)
    for size in SIZES:
        assert size.encode() in data
    assert "Ukur: harus 10 cm".encode() in data
    scratch = BytesIO()
    canvas = Canvas(scratch, pageCompression=0)
    assert draw_test_square(canvas, 50, 50) == pytest.approx((283.46, 283.46))
    canvas.save()


def test_graded_pdf_refuses_failing_grade() -> None:
    """A failing grade is refused, or stamped NOT VERIFIED when forced."""
    from seamproof.export import export_pdf_graded

    report = grade_and_verify("blouse", SIZES, "woven")
    _shrink_xl_armhole(report.patterns)
    report.grade_checks = grade_consistency(report.patterns)
    report.passed = (all(r.passed for r in report.reports.values())
                     and all(c.passed for c in report.grade_checks))
    assert not report.passed
    with pytest.raises(ValueError):
        export_pdf_graded(report, BytesIO())
    forced = BytesIO()
    export_pdf_graded(report, forced, force=True)
    assert b"NOT VERIFIED" in forced.getvalue()


def test_cli_grade_writes_pack(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """grade writes per-size JSON, the nested PDF and the grade report."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "grade", "--garment", "blouse",
                                      "--sizes", "S,M,L,XL", "--fabric", "woven",
                                      "--out", str(out)])
    assert main() == 0
    for size in SIZES:
        assert (out / f"pattern_{size}.json").is_file()
    assert (out / "graded.pdf").read_bytes().startswith(b"%PDF")
    assert "LULUS" in (out / "grade_report.md").read_text()


def test_cli_grade_rejects_missing_extreme(monkeypatch: pytest.MonkeyPatch, tmp_path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    """grade exits 2 when the smallest or largest size is missing."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "grade", "--garment", "blouse",
                                      "--sizes", "M,L,XL", "--fabric", "woven",
                                      "--out", str(out)])
    assert main() == 2
    assert "terkecil" in capsys.readouterr().err
