"""Dress: photo reading as one piece, style plan, both constructions and their checks."""

import pytest

from seamproof.draft import REGISTRY
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import DRESS_STYLE
from seamproof.params import resolve_params
from seamproof.pattern import edge_length
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern
from seamproof.vision import USER_PROMPT, PreparedPhoto, describe_garments, validate_reading

DRESS_READING = {"part": "dress", "kind": "dress", "evidence": "one-piece dress",
                 "features": {"collar": "none", "opening": "none", "sleeve_length": "short", "shoulder": "set_in",
                              "fit": "regular", "silhouette": "a_line", "length": "knee", "hem": "straight",
                              "waist_seam": True, "pockets": 0}}


def _edges(pattern):
    """Every edge by id."""
    return {e.id: e for panel in pattern.panels for e in panel.edges}


def test_prompt_asks_for_one_dress_piece() -> None:
    """The vision prompt offers the dress part and forbids splitting it into top + skirt."""
    assert "top|bottom|dress" in USER_PROMPT
    assert "never split it into a top plus a skirt" in USER_PROMPT
    assert "waist_seam" in USER_PROMPT


def test_dress_reading_is_validated_as_one_piece() -> None:
    """A dress piece keeps its bodice and skirt categories and coerces unknown values."""
    raw = {"pieces": [dict(DRESS_READING, features={**DRESS_READING["features"], "length": "ankle"})]}
    reading = validate_reading(raw)
    assert len(reading.pieces) == 1
    piece = reading.pieces[0]
    assert (piece.part, piece.kind) == ("dress", "dress")
    assert piece.features["waist_seam"] is True and piece.features["silhouette"] == "a_line"
    assert piece.features["length"] == "mini" and any("ankle" in w for w in reading.warnings)


def test_describe_garments_reads_dress_with_mocked_wonder() -> None:
    """The model call is mocked; a dress reply becomes a dress plan, not tunic + skirt."""
    photo = PreparedPhoto(b"\xff\xd8", 10, 10, 0)
    reading = describe_garments(photo, lambda *args: {"photo_view": "worn", "pieces": [DRESS_READING]})
    piece = reading.pieces[0]
    plan = plan_piece(piece.part, piece.kind, piece.features)
    assert plan.garment == "dress"
    assert plan.params["dress_waist_seam"] == 1.0 and plan.params["skirt_length"] == 580.0


def test_dress_plan_reports_what_it_cannot_draft() -> None:
    """Collar, buttons, pockets and a shift scallop become notes; numbers stay in range."""
    features = {**DRESS_READING["features"], "collar": "peter_pan", "opening": "button_front", "pockets": 2,
                "hem": "scalloped", "waist_seam": False, "length": "maxi", "sleeve_length": "long"}
    plan = plan_piece("dress", "dress", validate_reading({"pieces": [{"part": "dress", "kind": "dress",
                                                                     "features": features}]}).pieces[0].features)
    text = " ".join(plan.notes)
    assert "Kerah" in text and "Kancing" in text and "Saku" in text and "scallop" in text
    assert plan.params["scallop_depth"] == 0.0 and plan.params["dress_waist_seam"] == 0.0
    resolve_params(plan.params)


@pytest.mark.parametrize("size", ["S", "M", "L", "XL"])
@pytest.mark.parametrize("style", [
    {"dress_waist_seam": 1, "skirt_length": 580, "dress_sleeve_length": 200},
    {"dress_waist_seam": 1, "skirt_length": 950, "skirt_flare": 180, "scallop_depth": 18, "dress_sleeve_length": 570},
    {"dress_waist_seam": 0, "skirt_length": 420, "skirt_flare": 0, "dress_sleeve_length": 300},
    {"dress_waist_seam": 0, "skirt_length": 750, "skirt_flare": 180, "dress_sleeve_length": 430},
], ids=["seam-knee", "seam-maxi-scallop", "shift-mini", "shift-midi"])
def test_dress_sizes_pass_verifier(size, style) -> None:
    """Both constructions pass every check across the women's size table."""
    report = verify_pattern(REGISTRY["dress"](WOMEN_SIZE_TABLE[size], "woven", style))
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]


def test_waist_seam_dress_joins_bodice_to_skirt() -> None:
    """Bodice and skirt halves sew to the same waist, which is the body waist + 10 mm."""
    m = WOMEN_SIZE_TABLE["M"]
    pattern = REGISTRY["dress"](m, "woven", DRESS_STYLE)
    edges = _edges(pattern)
    assert [p.name for p in pattern.panels] == ["front", "back", "sleeve", "facing", "skirt_front", "skirt_back"]
    for top, low in (("front", "skirt_front"), ("back", "skirt_back")):
        bodice = edge_length(edges[f"{top}.waist_cf"]) + edge_length(edges[f"{top}.waist_side"])
        skirt = edge_length(edges[f"{low}.waist_cf"]) + edge_length(edges[f"{low}.waist_side"])
        assert bodice == pytest.approx(skirt, abs=0.01)
        assert bodice == pytest.approx((m.waist + 10) / 4, abs=0.01)
    assert edges["front.waist_cf"].seam_partner_id == "dress.waist"
    assert edges["skirt_front.side"].seam_partner_id == "skirt_back.side"
    checks = {c.id for c in verify_pattern(pattern).checks}
    assert {"K4", "D5", "D6", "K1", "S2a", "S2b"} <= checks and "K2" not in checks


def test_shift_dress_is_one_piece_bodice_block() -> None:
    """The shift dress keeps front/back full length, with the tunic-style checks."""
    m = WOMEN_SIZE_TABLE["M"]
    pattern = REGISTRY["dress"](m, "woven", {"dress_waist_seam": 0, "skirt_length": 580})
    assert [p.name for p in pattern.panels] == ["front", "back", "sleeve", "facing"]
    hem = _edges(pattern)["front.hem"].curve.points
    assert hem[-1].y == pytest.approx(m.front_length + 580)
    with pytest.raises(ValueError):
        REGISTRY["dress"](m, "woven", {"dress_waist_seam": 0, "scallop_depth": 18})


def test_sleeve_is_shortened_but_never_longer_than_the_arm() -> None:
    """Short sleeves are cut on the underarm lines; long is capped at the arm length."""
    m = WOMEN_SIZE_TABLE["S"]
    short = _edges(REGISTRY["dress"](m, "woven", {"dress_sleeve_length": 200}))
    assert short["sleeve.hem"].curve.points[0].y == pytest.approx(200)
    long = _edges(REGISTRY["dress"](m, "woven", {"dress_sleeve_length": 650}))
    assert long["sleeve.hem"].curve.points[0].y == pytest.approx(m.sleeve_length)


def test_dress_production_packs_every_piece() -> None:
    """The production pack nests all dress pieces, opened folds included."""
    from seamproof.production import produce

    report = produce("dress", "M=10", "woven", 1500.0, 45000.0, params=DRESS_STYLE)
    item = report.sizes[0]
    assert item.verified
    names = {p.panel for p in item.marker.placements}
    assert names == {"front", "back", "sleeve", "facing", "skirt_front", "skirt_back"}
    assert report.cut_list["M skirt_front"] == 10
