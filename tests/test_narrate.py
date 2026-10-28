"""Model-written words: number guard, fallbacks, fabric choice and measurement photos (all mocked)."""

import pytest

from seamproof.draft import REGISTRY
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.narrate import (fact_numbers, narrate_pattern, narrate_production, numbers_ok,
                               pattern_facts, production_facts, suggest_fabric)
from seamproof.production import produce
from seamproof.verify import verify_pattern
from seamproof.vision import PreparedPhoto, read_measurement_photo


def _blouse():
    """Pristine M blouse and its report."""
    pattern = REGISTRY["blouse"](WOMEN_SIZE_TABLE["M"], "woven", {})
    return pattern, verify_pattern(pattern)


def _offline(*args, **kwargs):
    """A model that is never reachable."""
    raise RuntimeError("offline")


def test_number_guard_reads_indonesian_formats() -> None:
    """Thousands dots, decimal commas, cm and m readings of mm facts all match; others fail."""
    allowed = fact_numbers({"total_biaya_rupiah": 533286, "meter_kain": 17.78, "panjang_marker_mm": 1180,
                            "kampuh_mm": 10})
    assert numbers_ok(["Biaya Rp533.286 untuk 17,78 m kain, marker 1,18 m, kampuh 1 cm."], allowed)
    assert numbers_ok(["Tanpa angka sama sekali."], allowed)
    assert not numbers_ok(["Biaya sekitar Rp540.000."], allowed)
    assert not numbers_ok(["Marker 1,2 m dan 3 lapis."], allowed)


def test_pattern_story_accepts_guarded_model_text() -> None:
    """Muse text that only quotes facts is used as written."""
    pattern, report = _blouse()
    facts = pattern_facts(pattern, report)
    assert any(p["nama_id"] == "badan depan" for p in facts["panel"]) and facts["sambungan"]

    def muse(system, user):
        assert "JANGAN" in system.upper() and "badan depan" in user
        return {"summary": "Semua pemeriksaan lulus, pola siap dibuat toile.",
                "sewing_steps": ["Jahit kupnat dada pada badan depan.", "Sambung bahu depan dan belakang, kampuh 1 cm."],
                "toile_checks": ["Lihat apakah ujung kupnat jatuh rapi di bawah puncak dada."]}

    story = narrate_pattern(pattern, report, llm_fn=muse)
    assert story.source == "muse" and len(story.sewing_steps) == 2


def test_pattern_story_rejects_invented_numbers_and_outages() -> None:
    """One stray number, a bad shape or an outage falls back to the code template."""
    pattern, report = _blouse()
    stray = lambda s, u: {"summary": "Lulus.", "sewing_steps": ["Jahit bahu 7 cm."], "toile_checks": ["Cek."]}  # noqa: E731
    for fn in (stray, lambda s, u: {"summary": 3}, _offline):
        story = narrate_pattern(pattern, report, llm_fn=fn)
        assert story.source == "template" and "lulus" in story.summary
        assert story.toile_checks


def test_production_story_quotes_only_computed_numbers() -> None:
    """Cutting steps may quote spreads and lengths computed by code; template otherwise."""
    report = produce("blouse", "M=10", "woven", 1500.0, 45000.0, layers=5)
    facts = production_facts(report)
    assert facts["ukuran"][0]["gelar"] == 2
    meters = f"{facts['total_meter']:.2f}".replace(".", ",")

    def muse(system, user):
        return {"cutting_steps": ["Ukuran M: gelar dua kali, lima lapis per gelar."],
                "owner_summary": f"Total kain {meters} m."}

    story = narrate_production(report, llm_fn=muse)
    assert story.source == "muse"
    fallback = narrate_production(report, llm_fn=lambda s, u: {"cutting_steps": ["Gelar 4 kali."],
                                                               "owner_summary": "x"})
    assert fallback.source == "template" and "gelar 2 kali" in fallback.cutting_steps[0]


def test_fabric_suggestion_stays_among_verified_candidates() -> None:
    """Muse may only pick a candidate code verified; anything else uses the first candidate."""
    good = suggest_fabric("dress", ["woven"], {"silhouette": "a_line"}, "dress pesta",
                          llm_fn=lambda s, u: {"fabric": "woven", "reason": "Tenun menahan bentuk rok A-line."})
    assert (good.fabric, good.source) == ("woven", "muse")
    bad = suggest_fabric("dress", ["woven"], llm_fn=lambda s, u: {"fabric": "knit_high_stretch", "reason": "melar"})
    assert (bad.fabric, bad.source) == ("woven", "template")
    with pytest.raises(ValueError):
        suggest_fabric("dress", [], llm_fn=_offline)


def test_measurement_photo_goes_through_intake_rules() -> None:
    """Wonder only transcribes; ranges and missing fields are the intake's deterministic rules."""
    photo = PreparedPhoto(b"\xff\xd8", 10, 10, 0)

    def wonder(system, text, urls, schema, purpose):
        assert urls[0].startswith("data:image/jpeg;base64,") and "measurement note" in text
        return {"measurements": [{"field": "bust", "value": 92, "unit": "cm"},
                                 {"field": "waist", "value": 7400, "unit": "mm"}]}

    result = read_measurement_photo(photo, "dress", vision_fn=wonder)
    assert result["measurements"] == {"bust": 920.0}
    assert "waist" in result["missing"] and any("waist" in w for w in result["warnings"])
