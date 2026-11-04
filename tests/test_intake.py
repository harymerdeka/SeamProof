"""Intake tests with a stubbed LLM mapper; no network calls."""

import json
import sys

import pytest

import seamproof.llm as llm_module
from seamproof.cli import main
from seamproof.intake import from_size_table, parse_file, parse_text


def _mapper(entries):
    """Return an llm_fn serving canned field entries, recording prompts."""

    def fn(system, user):
        fn.users.append(user)
        return {"measurements": entries}

    fn.users = []
    return fn


FULL_TSHIRT = [
    {"field": "chest", "value": 88, "unit": "cm"},
    {"field": "waist", "value": 76, "unit": "cm"},
    {"field": "hip", "value": 90, "unit": "cm"},
    {"field": "shoulder_width", "value": 40, "unit": "cm"},
    {"field": "back_length", "value": 64, "unit": "cm"},
    {"field": "sleeve_length", "value": 20, "unit": "cm"},
    {"field": "armhole_depth", "value": 21, "unit": "cm"},
    {"field": "neck_width", "value": 15, "unit": "cm"},
    {"field": "bicep", "value": 28, "unit": "cm"},
]


def test_indonesian_text_converts_units() -> None:
    """Centimetre notes become millimetres with nothing invented."""
    result = parse_text("lingkar dada 88, ...", "tshirt", llm_fn=_mapper(FULL_TSHIRT))
    assert result["missing"] == [] and result["warnings"] == []
    assert result["measurements"]["chest"] == 880
    assert result["measurements"]["back_length"] == 640
    assert len(result["measurements"]) == 9


def test_missing_fields_reported_not_invented() -> None:
    """Three stated fields leave six missing and no guesses."""
    entries = FULL_TSHIRT[:3]
    result = parse_text("dada 96 pinggang 84 pinggul 98", "tshirt", llm_fn=_mapper(entries))
    assert result["missing"] == ["shoulder_width", "back_length", "sleeve_length",
                                 "armhole_depth", "neck_width", "bicep"]
    assert len(result["measurements"]) == 3


def test_out_of_range_dropped() -> None:
    """A 200 cm waist is dropped to missing with a warning."""
    entries = [dict(e) for e in FULL_TSHIRT]
    entries[1] = {"field": "waist", "value": 200, "unit": "cm"}
    result = parse_text("x", "tshirt", llm_fn=_mapper(entries))
    assert "waist" in result["missing"]
    assert any("waist" in w for w in result["warnings"])
    assert "waist" not in result["measurements"]


def test_cross_field_warning_kept() -> None:
    """A waist far above the chest warns but keeps both values."""
    entries = [dict(e) for e in FULL_TSHIRT]
    entries[1] = {"field": "waist", "value": 110, "unit": "cm"}
    result = parse_text("x", "tshirt", llm_fn=_mapper(entries))
    assert result["missing"] == []
    assert any("Waist" in w for w in result["warnings"])


def test_unknown_field_ignored() -> None:
    """Fields outside the garment warn and drop out."""
    entries = FULL_TSHIRT + [{"field": "sayap", "value": 10, "unit": "cm"}]
    result = parse_text("x", "tshirt", llm_fn=_mapper(entries))
    assert "sayap" not in result["measurements"]
    assert any("sayap" in w for w in result["warnings"])


def test_file_mm_and_cm(tmp_path) -> None:
    """JSON mm and YAML cm files both land in millimetres."""
    mm = tmp_path / "m.json"
    mm.write_text(json.dumps({"garment": "tshirt", "unit": "mm",
                              "measurements": {"chest": 960, "waist": 840}}), encoding="utf-8")
    result = parse_file(mm)
    assert result["measurements"] == {"chest": 960, "waist": 840}
    assert len(result["missing"]) == 7
    cm = tmp_path / "m.yaml"
    cm.write_text("garment: tshirt\nunit: cm\nmeasurements:\n  chest: 96\n  waist: 84\n", encoding="utf-8")
    assert parse_file(cm)["measurements"] == {"chest": 960, "waist": 840}


def test_file_requires_unit(tmp_path) -> None:
    """A file without an explicit unit is rejected."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"measurements": {"chest": 960}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unit"):
        parse_file(bad)


def test_from_size_table_overrides() -> None:
    """Table bases fill the gaps; overrides win in millimetres."""
    result = from_size_table("tshirt", "M", {"chest": 970.0})
    assert result["missing"] == [] and result["measurements"]["chest"] == 970.0
    assert result["measurements"]["waist"] == 840.0


def _mock_llm(monkeypatch: pytest.MonkeyPatch, entries) -> None:
    """Route the real call_json to canned intake entries."""

    def fake_call_json(system, user, schema, purpose):
        assert "measurements" in schema["properties"]
        return {"measurements": entries}

    monkeypatch.setattr(llm_module, "call_json", fake_call_json)


def test_cli_intake_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Full Indonesian text drafts a verified PDF with the LLM mocked."""
    _mock_llm(monkeypatch, FULL_TSHIRT)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "intake",
                                      "--text", "lingkar dada 96, panjang baju 70, ...",
                                      "--garment", "tshirt", "--fabric", "knit_low_stretch",
                                      "--out", str(out)])
    assert main() == 0
    for name in ("pattern.json", "intake.json", "pattern.svg", "pattern.pdf", "report.md", "rounds.md"):
        assert (out / name).is_file()
    assert "PASS" in (out / "report.md").read_text()


def test_cli_intake_stops_on_missing(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    """Half a note stops before drafting anything."""
    _mock_llm(monkeypatch, FULL_TSHIRT[:3])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "intake", "--text", "dada 96",
                                      "--garment", "tshirt", "--out", str(out)])
    assert main() == 2
    assert "Missing" in capsys.readouterr().err
    assert not out.exists()
