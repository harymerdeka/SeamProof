"""Fit-correction tests; vision is a scripted stand-in, no network calls."""

import json
import sys

import pytest

from seamproof.cli import main
from seamproof.draft import REGISTRY
from seamproof.fit_rules import RULES, delta_for
from seamproof.fitcorrect import Issue, fit_correct, fit_md, notes_to_issues
from seamproof.measurements import WOMEN_SIZE_TABLE

M = WOMEN_SIZE_TABLE["M"]


def _blouse_v1():
    """Pristine M blouse v1."""
    return REGISTRY["blouse"](M, "woven", {})


def test_notes_map_to_rules_with_severity() -> None:
    """Keywords classify; intensifiers set severity, numbers stay evidence."""
    issues = notes_to_issues(["ketiak sangat sempit 2 cm", "ujung kupnat agak runcing"])
    assert [(i.key, i.severity) for i in issues] == [
        ("pulling_at_underarm", 3), ("dart_point_too_pointed", 1)]
    assert "2 cm" in issues[0].evidence


def test_rule_amounts_stay_bounded() -> None:
    """Severity interpolates inside the table bounds."""
    assert delta_for(RULES["pulling_at_underarm"], 1) == {"armhole_curve_depth": 5.0}
    assert delta_for(RULES["pulling_at_underarm"], 2) == {"armhole_curve_depth": 7.5}
    assert delta_for(RULES["pulling_at_underarm"], 3) == {"armhole_curve_depth": 10.0}
    with pytest.raises(ValueError):
        delta_for(RULES["pulling_at_underarm"], 4)


def test_notes_produce_verified_v2() -> None:
    """A dart-clearance note yields a locality-clean, verified v2."""
    proposal, v2 = fit_correct(_blouse_v1(), notes=["ujung kupnat runcing"])
    assert proposal.changes == {"bust_dart_clearance": 27.5}
    assert proposal.locality_passed is True
    assert proposal.verified and v2 is not None
    assert v2.params["bust_dart_clearance"] == 27.5


def test_deltas_clamp_to_param_ranges() -> None:
    """Repeated severe flare notes clamp side_flare at 0 with a note."""
    v1 = REGISTRY["tunic"](M, "woven", {})
    proposal, v2 = fit_correct(v1, notes=["flare sangat lebar"] * 10)
    assert proposal.changes == {"side_flare": 0.0}
    assert any("dijepit" in note for note in proposal.clamped)
    assert v2 is not None


def test_advice_only_issue_changes_nothing() -> None:
    """Tight chest has no width param: advice recorded, geometry identical."""
    proposal, v2 = fit_correct(_blouse_v1(), notes=["dada sempit"])
    assert proposal.changes == {}
    assert proposal.advice and v2 is not None
    assert proposal.locality_passed is True


def test_photos_refused_without_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Photos need face blurring; without cv2 only notes are accepted."""
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "cv2", None)  # simulate OpenCV not installed
    with pytest.raises(ValueError, match="OpenCV"):
        fit_correct(_blouse_v1(), photos=["foto.jpg"], notes=["ketiak sempit"])


def test_mocked_vision_fn_supplies_issues(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scripted vision function drives the same rule pipeline."""
    import sys as _sys
    import types

    monkeypatch.setitem(_sys.modules, "cv2", types.ModuleType("cv2"))
    import seamproof.fitcorrect as fit_module

    monkeypatch.setattr(fit_module, "_blur_faces", lambda data: data)
    photo = tmp_path / "depan.jpg"
    photo.write_bytes(b"palsu")
    seen = []

    def fake_vision(images):
        seen.extend(images)
        return [Issue("neckline_gapes", "depan", 2, "vision: menganga")]

    proposal, v2 = fit_correct(_blouse_v1(), photos=[str(photo)], vision_fn=fake_vision)
    # The t-shirt neck rule lands on the blouse's own centre-front neck drop.
    assert seen and proposal.changes == {"neckline_depth_front": 85.5}
    assert v2 is not None


def test_fit_md_and_overlay_carry_diff(tmp_path) -> None:
    """fit.md holds the label, issues and v1->v2 rows; overlay nests both."""
    from seamproof.export import export_fit_overlay_svg

    v1 = _blouse_v1()
    proposal, v2 = fit_correct(v1, notes=["ujung kupnat runcing"])
    text = fit_md(proposal, v1, "v2")
    assert "konfirmasi sebelum memotong" in text
    assert "| bust_dart_clearance | 20.0 | 27.5 |" in text
    overlay = tmp_path / "overlay.svg"
    export_fit_overlay_svg(v1, v2, str(overlay))
    content = overlay.read_text()
    assert "hitam=v1" in content and 'stroke="green"' in content


def test_cli_fitcorrect_writes_v2(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """fitcorrect writes fit.md, overlay and a verified v2 pack."""
    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(_blouse_v1().to_dict()), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "fitcorrect", "--pattern", str(v1_path),
                                      "--notes", "ujung kupnat runcing", "--out", str(out)])
    assert main() == 0
    assert (out / "v2_pattern.pdf").read_bytes().startswith(b"%PDF")
    assert "konfirmasi sebelum memotong" in (out / "fit.md").read_text()
