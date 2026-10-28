"""Bishop and puff shirt sleeves with fitted (elastic) cuffs: plan, draft, verifier and flat."""

import pytest

from seamproof.design import design_from_brief
from seamproof.draft import REGISTRY
from seamproof.draft.shirt import CUFF_ELASTIC_PASS_MM, arm_girth_at
from seamproof.flat import flat_svg
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import BISHOP_STYLE
from seamproof.pattern import edge_length
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern
from seamproof.vision import validate_reading

BRIEF_FEATURES = {"collar": "shirt_collar", "opening": "button_front", "sleeve_length": "long",
                  "sleeve_style": "puff", "fit": "regular", "length": "hip", "cuff": True, "elastic_cuff": True}


def _edges(pattern):
    """Every edge by id."""
    return {e.id: e for panel in pattern.panels for e in panel.edges}


def test_reading_keeps_sleeve_style_and_elastic_cuff() -> None:
    """The new categories survive validation; unknown values fall back to straight."""
    piece = validate_reading({"pieces": [{"part": "top", "kind": "shirt", "features": BRIEF_FEATURES}]}).pieces[0]
    assert piece.features["sleeve_style"] == "puff" and piece.features["elastic_cuff"] is True
    odd = validate_reading({"pieces": [{"part": "top", "kind": "shirt", "features": {"sleeve_style": "lantern"}}]})
    assert odd.pieces[0].features["sleeve_style"] == "straight" and odd.warnings


def test_puff_plan_gathers_hem_and_cap() -> None:
    """Puff = hem fullness + cap gather + elastic cuff; a bishop without a cuff gets one."""
    plan = plan_piece("top", "shirt", dict(BRIEF_FEATURES))
    assert plan.params["sleeve_hem_fullness"] > 0 and plan.params["sleeve_cap_gather"] > 0
    assert plan.params["cuff_elastic"] == 1.0 and plan.params["cuff_depth"] > 0
    bishop = plan_piece("top", "shirt", dict(BRIEF_FEATURES, sleeve_style="bishop", cuff=False, elastic_cuff=False))
    assert bishop.params["cuff_elastic"] == 1.0 and bishop.params["sleeve_cap_gather"] == 0
    assert any("karet" in n for n in bishop.notes)


@pytest.mark.parametrize("size", ["S", "M", "L", "XL"])
@pytest.mark.parametrize("style", [
    BISHOP_STYLE,
    dict(BISHOP_STYLE, sleeve_cap_gather=0),
    dict(BISHOP_STYLE, shirt_sleeve_length=300, cuff_elastic=0),
    dict(BISHOP_STYLE, shirt_sleeve_length=200, sleeve_hem_fullness=200, sleeve_cap_gather=60, design_ease=140,
         shoulder_drop=50),
], ids=["puff-long", "bishop-long", "puff-elbow-band", "puff-short-boxy"])
def test_gathered_sleeves_pass_verifier(size, style) -> None:
    """Gathered sleeves verify across sizes, including the S9 gather check."""
    report = verify_pattern(REGISTRY["shirt"](WOMEN_SIZE_TABLE[size], "woven", style))
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]
    assert "S9" in {c.id for c in report.checks}


def test_elastic_cuff_fits_the_arm_and_gathers_the_hem() -> None:
    """The elastic cuff is arm girth + 60 mm; the flared hem is longer and gathered into it."""
    m = WOMEN_SIZE_TABLE["M"]
    pattern = REGISTRY["shirt"](m, "woven", BISHOP_STYLE)
    edges = _edges(pattern)
    cap_y = edges["sleeve.underarm_back"].curve.points[0].y
    band = edge_length(edges["cuff.attach"])
    assert band == pytest.approx(arm_girth_at(m, 580, cap_y) + CUFF_ELASTIC_PASS_MM, abs=0.01)
    assert edge_length(edges["sleeve.cuff_join"]) > 2 * band
    gathers = pattern.metadata["gathers_mm"]
    assert gathers["sleeve_cap"] == 40 and gathers["elastic_cut"] < band


def test_puff_cap_needs_its_declared_gather() -> None:
    """The same tall cap fails S3 when the gather is not declared, so gathers are never silent."""
    pattern = REGISTRY["shirt"](WOMEN_SIZE_TABLE["M"], "woven", BISHOP_STYLE)
    pattern.params["sleeve_cap_gather"] = 0.0
    pattern.metadata["params_hash"] = __import__("seamproof.pattern", fromlist=["params_hash"]).params_hash(pattern.params)
    assert "S3" in {c.id for c in verify_pattern(pattern).failures()}


def test_fitted_wrist_cuff_without_elastic_is_refused() -> None:
    """A tight wrist cuff with no elastic needs a placket the draft lacks."""
    with pytest.raises(ValueError):
        REGISTRY["shirt"](WOMEN_SIZE_TABLE["M"], "woven", dict(BISHOP_STYLE, cuff_elastic=0))


def test_flat_draws_billow_and_gather_marks() -> None:
    """The sketch shows gather marks and the elastic line only for gathered sleeves."""
    puff = flat_svg("top", "shirt", BRIEF_FEATURES, "dusty_pink")
    plain = flat_svg("top", "shirt", dict(BRIEF_FEATURES, sleeve_style="straight", elastic_cuff=False), "dusty_pink")
    assert puff.count('class="detail"') > plain.count('class="detail"')
    assert "stroke-dasharray" in puff


def test_brief_with_bishop_sleeves_is_now_drafted() -> None:
    """A Muse reply using the new categories becomes a gathered-sleeve shirt with nothing missing but trousers."""
    reply = {"options": [{"name": "Blus Puff Dusty Pink", "part": "top", "kind": "shirt", "features": BRIEF_FEATURES,
                          "color": "dusty_pink", "fabric": "woven", "reason": "Kemeja lengan puff manset karet.",
                          "unsupported": ["celana bahan"]}]}
    option = design_from_brief("Blus dusty pink lengan puff, manset karet, kerah kemeja.", lambda *a: reply)[0]
    assert option.plan.garment == "shirt" and option.plan.params["sleeve_hem_fullness"] > 0
    assert option.unsupported == ["celana bahan"] and not option.warnings
    report = verify_pattern(REGISTRY["shirt"](WOMEN_SIZE_TABLE["M"], "woven", option.plan.params))
    assert report.passed


def test_report_md_lists_gathers_for_the_tailor() -> None:
    """report.md carries the code-computed gathering amounts and elastic length."""
    import io

    from seamproof.export import export_report_md

    report = verify_pattern(REGISTRY["shirt"](WOMEN_SIZE_TABLE["M"], "woven", BISHOP_STYLE))
    out = io.BytesIO()
    export_report_md(report, out)
    text = out.getvalue().decode()
    assert "## Kerutan" in text and "Karet manset" in text and "Puncak lengan" in text
    plain = verify_pattern(REGISTRY["shirt"](WOMEN_SIZE_TABLE["M"], "woven", {}))
    assert "gathers_mm" not in plain.meta
