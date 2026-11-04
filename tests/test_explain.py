"""Explanation tests with a scripted stand-in LLM; no network calls."""

import sys
from io import BytesIO

import pytest
from reportlab.pdfgen.canvas import Canvas

from seamproof.cli import main
from seamproof.draft import REGISTRY
from seamproof.explain import _allowed_numbers, _numbers_in_mm, explain
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.repair import INJECTS
from seamproof.verify import verify_pattern

M = WOMEN_SIZE_TABLE["M"]


def _failing():
    """Blouse M with an injected high cap; S3 must fail."""
    pattern = REGISTRY["blouse"](M, "woven", dict(INJECTS["sleeve_cap_too_high"]))
    report = verify_pattern(pattern)
    assert not report.passed
    return pattern, report


def test_passing_summary_needs_no_llm() -> None:
    """A clean report gets a template summary without calling the model."""

    def fail_if_called(system, user):
        raise AssertionError("LLM must not be called")

    pattern = REGISTRY["blouse"](M, "woven", {})
    report = verify_pattern(pattern)
    assert report.passed
    summary, instructions = explain(report, pattern, llm_fn=fail_if_called)
    assert summary == (f"All {len(report.checks)} checks passed — "
                       f"blouse M pattern ready for toile.")
    assert instructions == []


def test_good_instruction_is_accepted() -> None:
    """Model text echoing verifier numbers is kept verbatim."""
    pattern, report = _failing()
    s3 = next(c for c in report.failures() if c.id == "S3")
    text = (f"Pada panel sleeve, turunkan garis sleeve.cap sebesar {s3.actual:.1f} mm "
            f"menuju {s3.expected:.1f} mm, batas {s3.tolerance:.1f} mm.")
    llm = lambda system, user: {"instructions": [
        {"panel": "sleeve", "edge": "sleeve.cap", "action": "turunkan",
         "amount_mm": round(s3.actual, 1), "text": text}]}
    summary, instructions = explain(report, pattern, llm_fn=llm)
    assert len(instructions) == 1
    assert instructions[0].text == text
    assert "failed" in summary


def test_injected_failure_numbers_match_verifier() -> None:
    """Done-when: every instruction number equals a verifier number or cm form."""
    pattern, report = _failing()
    s3 = next(c for c in report.failures() if c.id == "S3")
    text = (f"Pada panel sleeve, turunkan garis sleeve.cap sebesar {s3.actual:.1f} mm "
            f"menuju {s3.expected:.1f} mm, batas {s3.tolerance:.1f} mm.")
    llm = lambda system, user: {"instructions": [
        {"panel": "sleeve", "edge": "sleeve.cap", "action": "turunkan",
         "amount_mm": round(s3.actual, 1), "text": text}]}
    _summary, instructions = explain(report, pattern, llm_fn=llm)
    allowed = _allowed_numbers(report)
    assert instructions
    for item in instructions:
        assert _numbers_in_mm(item.text), "the instruction must quote a verifier number"
        assert all(n in allowed for n in _numbers_in_mm(item.text))
        assert round(item.amount_mm, 1) in allowed


def test_hallucinated_number_falls_back_to_template() -> None:
    """A number outside the report discards the model text."""
    pattern, report = _failing()
    llm = lambda system, user: {"instructions": [
        {"panel": "sleeve", "edge": "sleeve.cap", "action": "turunkan",
         "amount_mm": 999.0, "text": "Turunkan sleeve.cap sebesar 999 mm."}]}
    _summary, instructions = explain(report, pattern, llm_fn=llm)
    assert instructions
    assert all("999" not in item.text for item in instructions)
    assert any(item.text.startswith("S3") for item in instructions)


def test_unknown_edge_falls_back_to_template() -> None:
    """Invented panel/edge ids discard the model text."""
    pattern, report = _failing()
    s3 = next(c for c in report.failures() if c.id == "S3")
    llm = lambda system, user: {"instructions": [
        {"panel": "lengan", "edge": "lengan.khayalan", "action": "turunkan",
         "amount_mm": round(s3.actual, 1), "text": f"Ubah {s3.actual:.1f} mm."}]}
    _summary, instructions = explain(report, pattern, llm_fn=llm)
    assert any(item.panel == "—" for item in instructions)


def test_llm_error_falls_back_to_template() -> None:
    """A backend failure still yields deterministic template instructions."""

    def boom(system, user):
        raise RuntimeError("jaringan putus")

    pattern, report = _failing()
    summary, instructions = explain(report, pattern, llm_fn=boom)
    assert len(instructions) == len(report.failures())
    assert "failed" in summary


def test_other_language_is_rejected() -> None:
    """Only English explanations are supported."""
    pattern = REGISTRY["blouse"](M, "woven", {})
    report = verify_pattern(pattern)
    with pytest.raises(ValueError):
        explain(report, pattern, lang="xx")


def test_report_md_and_proof_page_carry_explanation() -> None:
    """The instruction text lands in report.md and on the proof page."""
    from seamproof.export import draw_proof_page, export_report_md

    pattern = REGISTRY["blouse"](M, "woven", {})
    report = verify_pattern(pattern)
    explanation = ("All checks passed.", [])
    stream = BytesIO()
    export_report_md(report, stream, explanation)
    assert "Tailor explanation".encode() in stream.getvalue()
    pdf = BytesIO()
    canvas = Canvas(pdf, pageCompression=0)
    draw_proof_page(canvas, pattern, report, explanation)
    canvas.save()
    assert "Tailor explanation".encode() in pdf.getvalue()


def test_cli_explain_writes_instructions(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """draft --explain adds the tailor section without network (template path)."""
    import seamproof.explain as explain_module

    monkeypatch.setattr(explain_module.llm_module, "call_json",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", "blouse",
                                      "--size", "M", "--fabric", "woven",
                                      "--out", str(out), "--explain", "--force"])
    assert main() == 0
    assert "Tailor explanation" in (out / "report.md").read_text()
    assert "Tailor explanation".encode() in (out / "pattern.pdf").read_bytes()


def test_numbers_in_mm_read_english_units_and_separators() -> None:
    """Spelled-out English units convert to mm; a comma before three digits is thousands."""
    assert _numbers_in_mm("lower it 1.5 centimetres") == [15.0]
    assert _numbers_in_mm("a 1,500 mm marker") == [1500.0]
