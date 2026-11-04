"""Sleeveless shirts/dresses and V necklines: bias finishes, refusals and their checks."""

import pytest

from seamproof.draft import REGISTRY
from seamproof.draft.finish import SLEEVELESS_RAISE_MM
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import VEST_STYLE, m_armhole_bias, m_lost_sleeve, m_neck_bias
from seamproof.pattern import edge_length
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern
from seamproof.vision import USER_PROMPT, validate_reading

M = WOMEN_SIZE_TABLE["M"]
SHIFT_SLEEVELESS = {"dress_waist_seam": 0, "sleeveless": 1, "skirt_length": 950}
SEAM_SLEEVELESS_V = {"dress_waist_seam": 1, "sleeveless": 1, "neck_v": 1, "neckline_depth_front": 140,
                     "skirt_length": 950, "skirt_flare": 180}


def _edges(pattern):
    """Every edge by id."""
    return {e.id: e for panel in pattern.panels for e in panel.edges}


@pytest.mark.parametrize("size", list(WOMEN_SIZE_TABLE))
@pytest.mark.parametrize("garment,style", [
    ("shirt", VEST_STYLE), ("shirt", {"sleeveless": 1, "collar_style": 0}), ("shirt", {"collar_style": 0}),
    ("dress", SHIFT_SLEEVELESS), ("dress", SEAM_SLEEVELESS_V),
    ("dress", {"dress_waist_seam": 0, "neck_v": 1, "neckline_depth_front": 120}),
], ids=["vest-v", "vest-round", "collarless", "shift-sleeveless", "seam-sleeveless-v", "shift-v"])
def test_finishes_pass_verifier(size, garment, style) -> None:
    """Every new finish passes every check across the women's size table."""
    report = verify_pattern(REGISTRY[garment](WOMEN_SIZE_TABLE[size], "woven", style))
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]


def test_vest_has_bias_instead_of_sleeve() -> None:
    """No sleeve or cuff; armhole and neck bias are as long as the edges they bind."""
    pattern = REGISTRY["shirt"](M, "woven", VEST_STYLE)
    names = {p.name for p in pattern.panels}
    assert names == {"front", "back", "armhole_bias", "neck_bias", "pocket"}
    e = _edges(pattern)
    armhole = edge_length(e["front.armhole"]) + edge_length(e["back.armhole"])
    assert edge_length(e["armhole_bias.attach"]) == pytest.approx(armhole)
    neck = sum(edge_length(e[i]) for i in ("front.neckline", "front.ext_top", "back.neckline"))
    assert edge_length(e["neck_bias.attach"]) == pytest.approx(neck)
    assert e["front.armhole"].curve.points[-1].y == pytest.approx(M.armhole_depth - SLEEVELESS_RAISE_MM)
    checks = {c.id for c in verify_pattern(pattern).checks}
    assert {"S10", "S11"} <= checks and not {"S3", "S5", "S7"} & checks


def test_v_neck_is_straight_and_stand_continues_it() -> None:
    """The V runs straight to centre front; the stand top stays on that line."""
    e = _edges(REGISTRY["shirt"](M, "woven", VEST_STYLE))
    pts = e["front.neckline"].curve.points
    assert (pts[0].x, pts[0].y) == (0, 130)
    a, b = e["front.ext_top"].curve.points
    slope_v = (pts[-1].y - pts[0].y) / (pts[-1].x - pts[0].x)
    assert (b.y - a.y) / (b.x - a.x) == pytest.approx(slope_v)


def test_dress_keeps_real_measurements_and_facing_stays_on_cf() -> None:
    """The block is drafted higher, but the pattern records the body; the V facing stops at CF."""
    pattern = REGISTRY["dress"](M, "woven", SEAM_SLEEVELESS_V)
    assert pattern.size_label == "M" and pattern.measurements["armhole_depth"] == M.armhole_depth
    assert "sleeve" not in {p.name for p in pattern.panels}
    facing = next(p for p in pattern.panels if p.name == "facing")
    assert min(pt.x for edge in facing.edges for pt in edge.curve.points) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("garment,style", [
    ("shirt", {"sleeveless": 1, "shoulder_drop": 50}),
    ("shirt", {"sleeveless": 1, "cuff_depth": 35}),
    ("shirt", {"neck_v": 1, "collar_style": 1}),
])
def test_impossible_combinations_are_refused(garment, style) -> None:
    """A dropped shoulder or cuff needs a sleeve; a V neck has no collar."""
    with pytest.raises(ValueError):
        REGISTRY[garment](M, "woven", style)


def test_mutations_fail_exactly_their_check() -> None:
    """MUTATION: short armhole bias -> S10, long neck bias -> S11, lost sleeve -> S3."""
    vest = REGISTRY["shirt"](M, "woven", VEST_STYLE)
    assert {c.id for c in verify_pattern(m_armhole_bias(vest)).failures()} == {"S10"}
    assert {c.id for c in verify_pattern(m_neck_bias(vest)).failures()} == {"S11"}
    blouse = REGISTRY["blouse"](M, "woven", {})
    assert {c.id for c in verify_pattern(m_lost_sleeve(blouse)).failures()} == {"S3"}
    dress = REGISTRY["dress"](M, "woven", SHIFT_SLEEVELESS)
    dress.panels = [p for p in dress.panels if p.name != "armhole_bias"]
    assert {c.id for c in verify_pattern(dress).failures()} == {"S10"}


def test_photo_categories_map_to_sleeveless_v_plans() -> None:
    """The vision prompt offers neckline; a sleeveless V vest and gown plan the new params."""
    assert "neckline: round|v" in USER_PROMPT
    vest = validate_reading({"pieces": [{"part": "top", "kind": "other", "features": {
        "collar": "none", "neckline": "v", "opening": "button_front", "sleeve_length": "sleeveless",
        "shoulder": "dropped", "cuff": True}}]}).pieces[0]
    plan = plan_piece(vest.part, vest.kind, vest.features)
    assert plan.garment == "shirt" and plan.params["sleeveless"] == 1 and plan.params["neck_v"] == 1
    assert plan.params["shoulder_drop"] == 0 and "cuff_depth" not in plan.params
    assert any("dropped shoulder" in n.lower() for n in plan.notes)
    gown = plan_piece("dress", "dress", {"sleeve_length": "sleeveless", "neckline": "v", "waist_seam": True,
                                         "silhouette": "a_line", "length": "maxi", "hem": "straight"})
    assert gown.params["sleeveless"] == 1 and gown.params["neckline_depth_front"] == 140
    assert verify_pattern(REGISTRY["dress"](M, "woven", gown.params)).passed
