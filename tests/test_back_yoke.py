"""Shirt back yoke (horizontal seam across the upper back), from a front + back photo pair. Offline."""

from __future__ import annotations

import io

import pytest
from shapely.geometry import Polygon

from seamproof.design import merge_photos
from seamproof.draft.shirt import BACK_YOKE_MM, draft_shirt
from seamproof.export import export_dxf, export_svg
from seamproof.flat import plan_flat
from seamproof.locality import edit_locality
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import BASES, MUTATIONS, SHIRT_STYLE
from seamproof.params import GARMENT_PARAMS, PARAMETERS
from seamproof.pattern import edge_length, panel_polygon
from seamproof.preview3d import plan_preview
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern
from seamproof.vision import validate_reading

YOKED = dict(SHIRT_STYLE, back_yoke=1)
SHIRT = {"collar": "shirt_collar", "opening": "button_front", "sleeve_length": "long", "shoulder": "dropped",
         "fit": "boxy", "length": "thigh", "pockets": 1, "cuff": True}
# What the vision model reads from the two demo photos: a front view and a back view of the same shirt.
FRONT = {"part": "top", "kind": "shirt", "features": dict(SHIRT, back_yoke=False)}
BACK = {"part": "top", "kind": "shirt", "features": dict(SHIRT, opening="none", pockets=0, back_yoke=True)}


def _panels(pattern):
    return {panel.name: panel for panel in pattern.panels}


@pytest.mark.parametrize("size", list(WOMEN_SIZE_TABLE))
@pytest.mark.parametrize("style", [{}, {"collar_style": 0}, {"sleeveless": 1, "collar_style": 0},
                                   {"collar_style": 1, "cuff_depth": 35, "shoulder_drop": 50}])
def test_yoked_shirt_passes_every_check(size: str, style: dict) -> None:
    """Yoke + lower back replace the one-piece back; the verifier passes, Y1 included."""
    pattern = draft_shirt(WOMEN_SIZE_TABLE[size], "woven", dict(style, back_yoke=1))
    report = verify_pattern(pattern)
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]
    assert "Y1" in {c.id for c in report.checks}
    assert {"yoke", "back"} <= set(_panels(pattern))


def test_yoke_is_a_horizontal_cut_of_the_same_back() -> None:
    """Yoke + lower back cover the unyoked back exactly; the armhole and sleeve do not change."""
    m = WOMEN_SIZE_TABLE["M"]
    plain, yoked = _panels(draft_shirt(m, "woven", SHIRT_STYLE)), _panels(draft_shirt(m, "woven", YOKED))
    whole = panel_polygon(plain["back"]).area
    parts = panel_polygon(yoked["yoke"]).area + panel_polygon(yoked["back"]).area
    assert parts == pytest.approx(whole, rel=1e-3)
    seam = next(e for e in yoked["yoke"].edges if e.id == "yoke.seam").curve.points
    assert seam[0].y == seam[1].y == pytest.approx(BACK_YOKE_MM)
    arm = lambda ps, ids: sum(edge_length(e) for p in ps for e in p.edges if e.id in ids)  # noqa: E731
    assert arm([yoked["yoke"], yoked["back"]], {"yoke.armhole", "back.armhole"}) == pytest.approx(
        arm([plain["back"]], {"back.armhole"}), abs=0.01)
    assert plain["sleeve"].edges == yoked["sleeve"].edges and plain["collar"].edges == yoked["collar"].edges
    assert yoked["yoke"].cut_on_fold and yoked["yoke"].cut_count == 2


def test_yoke_mutation_fails_exactly_y1() -> None:
    """Every verifier check has a mutation: widening the lower back's yoke seam fails only Y1."""
    rows = [m for m in MUTATIONS if m[0] == "Y1"]
    assert rows
    for check_id, base, description, mutate in rows:
        assert {c.id for c in verify_pattern(mutate(BASES[base]())).failures()} == {check_id}, description


def test_missing_yoke_panel_fails_y1() -> None:
    """A yoked parameter set without its yoke panel is refused."""
    pattern = BASES["yoke"]()
    pattern.panels = [p for p in pattern.panels if p.name != "yoke"]
    assert "Y1" in {c.id for c in verify_pattern(pattern).failures()}


def test_every_param_stays_local_on_a_yoked_shirt() -> None:
    """Locality holds for each shirt parameter at both range ends with the yoke on."""
    m = WOMEN_SIZE_TABLE["M"]
    before = draft_shirt(m, "woven", YOKED)
    checked = 0
    for name in GARMENT_PARAMS["shirt"]:
        spec = PARAMETERS[name]
        for value in (spec.min, spec.max):
            try:
                after = draft_shirt(m, "woven", dict(YOKED, **{name: value}))
            except ValueError:
                continue
            result = edit_locality(before, after, {name: 0.0})
            assert result.passed, (name, value, result.message)
            checked += 1
    assert checked >= len(GARMENT_PARAMS["shirt"])


def test_style_maps_back_yoke_only_on_shirts() -> None:
    """A read yoke drafts on a shirt or vest; a blouse explains it cannot."""
    assert plan_piece("top", "shirt", dict(SHIRT, back_yoke=True)).params["back_yoke"] == 1.0
    assert plan_piece("top", "shirt", SHIRT).params["back_yoke"] == 0.0
    assert plan_piece("top", "vest", dict(SHIRT, back_yoke=True)).params["back_yoke"] == 1.0
    blouse = plan_piece("top", "blouse", dict(SHIRT, collar="none", opening="none", back_yoke=True))
    assert "back_yoke" not in blouse.params and any("yoke" in n for n in blouse.notes)


def test_vision_reads_back_yoke_and_photo_side() -> None:
    """back_yoke is a top flag; photo_side is front|back|other."""
    reading = validate_reading({"photo_side": "back", "pieces": [BACK]})
    assert reading.photo_side == "back" and reading.pieces[0].features["back_yoke"] is True
    assert validate_reading({"photo_side": "sideways", "pieces": [FRONT]}).photo_side == "other"
    assert validate_reading({"pieces": [{"part": "top", "kind": "shirt", "features": {}}]}).pieces[0].features[
        "back_yoke"] is False


def test_front_and_back_photos_merge_into_one_yoked_shirt() -> None:
    """Base = front photo; the back part (yoke) comes from the back photo, everything else from the front."""
    merged = merge_photos("top", 1, {1: FRONT, 2: BACK}, {"kuk": 2})
    f = merged.piece.features
    assert f["back_yoke"] is True and f["pockets"] == 1 and f["opening"] == "button_front"
    assert merged.provenance["kuk"] == 2 and merged.provenance["badan"] == 1
    assert merged.plan.garment == "shirt" and merged.plan.params["back_yoke"] == 1.0
    pattern = draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", merged.plan.params)
    assert verify_pattern(pattern).passed and "yoke" in _panels(pattern)
    # Without the back photo's part, the shirt has a one-piece back.
    assert merge_photos("top", 1, {1: FRONT, 2: BACK}, {}).piece.features["back_yoke"] is False


def test_previews_show_the_yoke_only_when_drafted() -> None:
    """Flat back view and 3D preview draw the yoke seam for a yoked shirt, not for a blouse."""
    shirt = plan_piece("top", "shirt", dict(SHIRT, back_yoke=True))
    plain = plan_piece("top", "shirt", SHIRT)
    assert plan_flat(shirt).count('class="detail"') == plan_flat(plain).count('class="detail"') + 1
    seams = lambda plan: {s["id"] for s in plan_preview(plan)["seams"]}  # noqa: E731
    assert "yoke_seam" in seams(shirt) and "yoke_seam" not in seams(plain)
    blouse = plan_piece("top", "blouse", dict(SHIRT, collar="none", opening="none", back_yoke=True))
    assert "yoke_seam" not in seams(blouse)


def test_yoked_shirt_exports() -> None:
    """The yoke panel goes through SVG and DXF export like any other panel."""
    pattern = BASES["yoke"]()
    report = verify_pattern(pattern)
    svg, dxf = io.BytesIO(), io.BytesIO()
    export_svg(pattern, svg)
    export_dxf(pattern, report, dxf)
    assert b'id="yoke"' in svg.getvalue() and b"yoke" in dxf.getvalue()


def test_yoked_shirt_grades_consistently() -> None:
    """A yoked shirt grades S-XL like the plain shirt (yoke armhole counted in GR checks).

    GR3-chest already fails for every women's shirt (no "chest" in women's sizes);
    that is not a yoke issue, so the yoke must just not add failures.
    """
    from seamproof.grading import grade, grade_consistency

    failed = lambda style: {c.id for c in grade_consistency(grade("shirt", ["S", "M", "L", "XL"], "woven", style))  # noqa: E731
                            if not c.passed}
    assert failed(YOKED) == failed(SHIRT_STYLE)


PLEATED = dict(YOKED, back_pleat=1)


@pytest.mark.parametrize("size", list(WOMEN_SIZE_TABLE))
def test_back_pleat_passes_and_is_marked(size: str) -> None:
    """The pleat extends the lower back past CB; Y1 (pleat closed) and Y2 (CB notches) pass."""
    m = WOMEN_SIZE_TABLE[size]
    pleated, plain = _panels(draft_shirt(m, "woven", PLEATED)), _panels(draft_shirt(m, "woven", YOKED))
    report = verify_pattern(draft_shirt(m, "woven", PLEATED))
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]
    assert {"Y1", "Y2"} <= {c.id for c in report.checks}
    seam = lambda ps: edge_length(next(e for e in ps["back"].edges if e.id == "back.yoke_seam"))  # noqa: E731
    from seamproof.draft.shirt import BACK_PLEAT_MM
    assert seam(pleated) == pytest.approx(seam(plain) + BACK_PLEAT_MM)
    assert pleated["yoke"].edges == plain["yoke"].edges


def test_pleat_without_yoke_is_refused_by_the_draft() -> None:
    """A pleat hangs from the yoke seam; the draft refuses one without it."""
    with pytest.raises(ValueError):
        draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", dict(SHIRT_STYLE, back_pleat=1))


def test_pleat_mutation_fails_exactly_y2() -> None:
    """Removing the pleat notch on the yoke seam fails only Y2."""
    rows = [m for m in MUTATIONS if m[0] == "Y2"]
    assert rows
    for check_id, base, description, mutate in rows:
        assert {c.id for c in verify_pattern(mutate(BASES[base]())).failures()} == {check_id}, description


def test_style_adds_the_yoke_a_pleat_needs() -> None:
    """A pleat read without a yoke still drafts: the yoke is added and explained."""
    plan = plan_piece("top", "shirt", dict(SHIRT, back_pleat=True))
    assert plan.params["back_yoke"] == plan.params["back_pleat"] == 1.0
    assert any("yoke was added" in n for n in plan.notes)


def test_merge_groups_are_split_front_back_and_whole() -> None:
    """The web shows front, back and whole-garment parts; the back parts come from the back photo."""
    back = dict(BACK, features=dict(BACK["features"], back_pleat=True))
    merged = merge_photos("top", 1, {1: FRONT, 2: back}, {"kuk": 2, "lipit": 2})
    sides = {g["id"]: g["side"] for g in merged.groups}
    assert sides == {"siluet": "both", "lengan": "both", "badan": "front", "kerah": "front",
                     "kuk": "back", "lipit": "back"}
    assert merged.plan.params["back_pleat"] == 1.0 and merged.piece.features["pockets"] == 1
    pattern = draft_shirt(WOMEN_SIZE_TABLE["M"], "woven", merged.plan.params)
    assert verify_pattern(pattern).passed


def test_previews_show_the_pleat() -> None:
    """Flat back view and 3D preview draw the pleat folds under the yoke."""
    pleat = plan_piece("top", "shirt", dict(SHIRT, back_yoke=True, back_pleat=True))
    yoke = plan_piece("top", "shirt", dict(SHIRT, back_yoke=True))
    assert plan_flat(pleat).count('class="detail"') == plan_flat(yoke).count('class="detail"') + 2
    assert {"back_pleat_r", "back_pleat_l"} <= {s["id"] for s in plan_preview(pleat)["seams"]}


def test_web_page_has_front_and_back_part_sections() -> None:
    """The mix-parts panel names a back section and takes back parts from a back photo."""
    from pathlib import Path

    html = (Path(__file__).parent.parent / "seamproof" / "static" / "index.html").read_text(encoding="utf-8")
    assert "Back of the garment" in html and "Front of the garment" in html
    assert 'const BACK_GROUPS = ["kuk", "lipit"]' in html
