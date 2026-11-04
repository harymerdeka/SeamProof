"""Design from scratch: brief -> options, refine diff, flats. Muse is mocked (no network)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.design import design_from_brief, diff_design, refine_design
from seamproof.flat import COLORS, DEFAULT_COLOR, drafted_features, flat_svg, plan_flat
from seamproof.server import app
from seamproof.style import plan_piece
from seamproof.vision import GarmentPiece

SHIRT = {"collar": "shirt_collar", "opening": "button_front", "sleeve_length": "short",
         "shoulder": "set_in", "fit": "regular", "length": "hip", "pockets": 1, "cuff": False,
         "contrast_collar": False, "contrast_cuff": False}
# Shaped like a Muse Spark reply: one good shirt, one skirt with an unknown hem and colour.
BRIEF_REPLY = {"options": [
    {"name": "Kemeja linen sage", "part": "top", "kind": "shirt", "features": SHIRT,
     "color": "sage", "fabric": "woven", "reason": "Rapi untuk kantor."},
    {"name": "Rok A-line", "part": "bottom", "kind": "skirt",
     "features": {"silhouette": "a_line", "length": "knee", "hem": "zigzag", "waistband": True},
     "color": "ungu tua", "fabric": "denim"},
    {"name": "Opsi ketiga", "part": "top", "kind": "blouse", "features": {}},
]}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every real backend call fails; tests pass their own fake call_fn or patch call_json."""
    def unreachable(*args, **kwargs):
        raise llm_module.LLMError("offline")

    monkeypatch.setattr(llm_module, "call_json", unreachable)


def _fake(reply: dict, seen: list | None = None):
    """A call_fn returning a fixed reply and recording the prompt."""
    def call(system, user, schema, purpose):
        if seen is not None:
            seen.append((purpose, user))
        return reply
    return call


def test_brief_gives_two_validated_options() -> None:
    """At most two options; unknown categories are coerced with warnings; fabric validated."""
    seen: list = []
    options = design_from_brief("kemeja kantor adem, warna kalem", _fake(BRIEF_REPLY, seen))
    assert len(options) == 2
    shirt, skirt = options
    assert shirt.plan.garment == "shirt" and shirt.color == "sage" and shirt.fabric == "woven"
    assert shirt.plan.params["pocket_count"] == 1
    assert skirt.piece.features["hem"] == "straight"
    assert skirt.color == DEFAULT_COLOR and skirt.fabric is None
    assert any("zigzag" in w for w in skirt.warnings) and any("ungu tua" in w for w in skirt.warnings)
    assert seen[0][0] == "design_brief" and "kemeja kantor adem" in seen[0][1]


def test_brief_without_usable_options_is_an_llm_error() -> None:
    """Nothing draftable from the model -> LLMValidationError (server maps to 502)."""
    with pytest.raises(llm_module.LLMValidationError):
        design_from_brief("baju", _fake({"options": [{"name": "x", "part": "shoes", "kind": "x", "features": {}}]}))
    with pytest.raises(ValueError):
        design_from_brief("  ", _fake(BRIEF_REPLY))


def test_refine_diff_is_computed_by_code_and_part_is_locked() -> None:
    """The model may not move the part; the diff lists exactly what changed."""
    current = {"part": "top", "kind": "shirt", "features": SHIRT, "color": "sage"}
    reply = {"kind": "shirt", "features": {"sleeve_length": "long", "part": "bottom"}, "color": "sage",
             "reason": "Lengan panjang."}
    result = refine_design(current, "buat lengan panjang", _fake(reply))
    assert result.piece.part == "top"
    assert result.changes == [{"key": "sleeve_length", "from": "short", "to": "long"}]
    assert result.plan.params["shirt_sleeve_length"] > 200
    assert result.piece.features["pockets"] == 1  # untouched features are kept


def test_refine_warns_on_no_change_and_on_many_changes() -> None:
    """Zero or more than two changes are flagged for the user."""
    current = {"part": "top", "kind": "shirt", "features": SHIRT, "color": "sage"}
    same = refine_design(current, "ubah jadi kain sutra", _fake({"kind": "shirt", "features": {}}))
    assert not same.changes and any("No category" in w for w in same.warnings)
    many = refine_design(current, "ganti semuanya", _fake({"kind": "shirt", "color": "navy", "features": {
        "fit": "boxy", "length": "thigh", "pockets": 2}}))
    assert len(many.changes) == 4 and any("4 things" in w for w in many.warnings)


def test_diff_design_orders_kind_features_colour() -> None:
    """Stable order so the version list reads the same every time."""
    a = GarmentPiece("top", "shirt", dict(SHIRT))
    b = GarmentPiece("top", "blouse", dict(SHIRT, fit="boxy"))
    assert [c["key"] for c in diff_design(a, "sage", b, "navy")] == ["kind", "fit", "color"]


def test_flat_is_valid_svg_and_follows_features() -> None:
    """Every colour renders; sleeves, pockets and buttons appear only when asked."""
    for color in COLORS:
        ET.fromstring(flat_svg("top", "shirt", SHIRT, color))
    plain = flat_svg("top", "shirt", dict(SHIRT, pockets=0, opening="none", sleeve_length="sleeveless"))
    rich = flat_svg("top", "shirt", dict(SHIRT, pockets=2, cuff=True))
    assert plain.count('class="button"') == 0 and rich.count('class="button"') > 3
    assert rich.count("h20 v20") == 2 and plain.count("h20 v20") == 0
    assert plain.count('<polygon class="fill"') == 2  # body front + back, no sleeves
    skirt = flat_svg("bottom", "skirt", {"silhouette": "flared", "length": "midi", "hem": "scalloped", "waistband": True})
    ET.fromstring(skirt)
    assert "Q" in skirt and skirt.count("trim") >= 2


def test_flat_shows_what_is_drafted_not_what_was_asked() -> None:
    """Unsupported choices are drawn as the pattern will be cut."""
    asked = dict(SHIRT, collar="band", sleeve_length="sleeveless", opening="none")
    drawn = drafted_features(plan_piece("top", "shirt", asked))
    assert drawn["collar"] == "band" and drawn["sleeve_length"] == "sleeveless"
    assert drawn["opening"] == "button_front"
    # A V neck is drawn only when the pattern drafts it: never with a collar.
    assert drafted_features(plan_piece("top", "shirt", dict(asked, neckline="v")))["neckline"] == "round"
    vest = plan_piece("top", "shirt", dict(SHIRT, collar="none", neckline="v", sleeve_length="sleeveless"))
    assert drafted_features(vest)["neckline"] == "v" and vest.params["neck_v"] == 1.0
    blouse = drafted_features(plan_piece("top", "blouse", dict(SHIRT, collar="none", opening="none", pockets=2)))
    assert blouse["pockets"] == 0 and blouse["collar"] == "none"
    ET.fromstring(plan_flat(plan_piece("top", "other", SHIRT)))  # undraftable still sketches


client = TestClient(app)


def test_design_endpoint_returns_options_with_flats(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /api/design -> two options, each with plan, flat and notes."""
    monkeypatch.setattr(llm_module, "call_json", _fake(BRIEF_REPLY))
    body = client.post("/api/design", json={"brief": "kemeja kantor"}).json()
    assert [o["plan"]["garment"] for o in body["options"]] == ["shirt", "skirt"]
    assert body["options"][0]["plan"]["flat"].startswith("<svg")
    assert any("zigzag" in n for n in body["options"][1]["plan"]["notes"])


def test_design_endpoint_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short brief is 422; an unreachable model is 502, never 500."""
    assert client.post("/api/design", json={"brief": "x"}).status_code == 422
    assert client.post("/api/design", json={"brief": "kemeja"}).status_code == 502


def test_refine_endpoint_and_drafting_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refined design drafts and verifies through the unchanged /api/pack path."""
    monkeypatch.setattr(llm_module, "call_json", _fake({"kind": "shirt", "features": {"pockets": 2}}))
    current = {"part": "top", "kind": "shirt", "features": SHIRT, "color": "sage"}
    body = client.post("/api/design/refine", json={"design": current, "instruction": "dua saku"}).json()
    assert body["changes"] == [{"key": "pockets", "from": 1, "to": 2}]
    style = {"part": body["part"], "kind": body["kind"], "features": body["features"]}
    pack = client.post("/api/pack", json={"garment": "shirt", "mode": "size", "size": "M", "style": style}).json()
    assert pack["passed"] and pack["style"]["params"]["pocket_count"] == 2


def test_catalog_and_style_plan_flat() -> None:
    """Catalogue lists colours; manual edits get a flat without any model call."""
    catalog = client.get("/api/design/catalog").json()
    assert set(catalog["colors"]) == set(COLORS) and "collar" in catalog["top"]
    body = client.post("/api/style-plan", json={"part": "top", "kind": "shirt", "features": SHIRT,
                                                "color": "navy"}).json()
    assert COLORS["navy"][0] in body["plan"]["flat"] and body["kind"] == "shirt"


DRESS = {"collar": "peter_pan", "opening": "button_front", "sleeve_length": "sleeveless", "shoulder": "dropped",
         "fit": "boxy", "pockets": 2, "cuff": True, "contrast_collar": False, "contrast_cuff": False,
         "silhouette": "a_line", "length": "knee", "hem": "scalloped", "waist_seam": False}


def test_dress_flat_draws_what_the_dress_draft_makes() -> None:
    """Dress details the draft lacks are not sketched; scallop needs a waist seam."""
    shift = drafted_features(plan_piece("dress", "dress", DRESS))
    assert shift["collar"] == "none" and shift["opening"] == "none" and shift["pockets"] == 0
    assert shift["sleeve_length"] == DRESS["sleeve_length"] and shift["hem"] == "straight"
    seamed = drafted_features(plan_piece("dress", "dress", dict(DRESS, waist_seam=True)))
    assert seamed["hem"] == "scalloped"
    shift_svg = plan_flat(plan_piece("dress", "dress", DRESS))
    seamed_svg = plan_flat(plan_piece("dress", "dress", dict(DRESS, waist_seam=True)))
    ET.fromstring(shift_svg)
    assert 'class="button"' not in shift_svg
    # The waist seam adds one line plus two bodice darts per view.
    assert seamed_svg.count('class="detail"') == shift_svg.count('class="detail"') + 6


def test_dress_in_catalog_and_design_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalogue lists dress categories and a Muse dress option drafts as a dress."""
    assert "waist_seam" not in client.get("/api/design/catalog").json()["dress"]  # flags are not lists
    assert "silhouette" in client.get("/api/design/catalog").json()["dress"]
    reply = {"options": [{"name": "Dress A-line", "part": "dress", "kind": "dress",
                          "features": dict(DRESS, waist_seam=True), "color": "terracotta"}]}
    monkeypatch.setattr(llm_module, "call_json", _fake(reply))
    option = client.post("/api/design", json={"brief": "dress kondangan"}).json()["options"][0]
    assert option["plan"]["garment"] == "dress" and option["plan"]["drafted"]["hem"] == "scalloped"
    style = {"part": "dress", "kind": "dress", "features": option["features"]}
    pack = client.post("/api/pack", json={"garment": "dress", "mode": "size", "size": "M", "style": style}).json()
    assert pack["passed"], pack["summary"]


# Three recorded-style photo readings: a boxy shirt, a fitted blouse-like top with
# long cuffed sleeves and a peter pan collar, and a dress.
PHOTO_1 = {"part": "top", "kind": "shirt", "features": dict(SHIRT, fit="boxy", pockets=2)}
PHOTO_2 = {"part": "top", "kind": "blouse", "features": dict(SHIRT, collar="peter_pan", opening="none",
                                                              sleeve_length="long", cuff=True, pockets=0, fit="fitted")}
PHOTO_3 = {"part": "dress", "kind": "dress", "features": dict(DRESS, waist_seam=True, sleeve_length="elbow")}


def test_merge_takes_each_part_from_its_photo() -> None:
    """Body from photo 1, sleeves and collar from photo 2; the diff is pure code."""
    from seamproof.design import merge_photos

    merged = merge_photos("top", 1, {1: PHOTO_1, 2: PHOTO_2}, {"lengan": 2, "kerah": 2})
    f = merged.piece.features
    assert merged.piece.kind == "shirt" and f["fit"] == "boxy" and f["pockets"] == 2
    assert f["sleeve_length"] == "long" and f["cuff"] is True and f["collar"] == "peter_pan"
    assert merged.provenance == {"siluet": 1, "badan": 1, "lengan": 2, "kerah": 2, "kuk": 1, "lipit": 1}
    assert merged.plan.garment == "shirt" and merged.plan.params["collar_style"] == 2


def test_merge_only_moves_parts_that_mean_the_same() -> None:
    """Dress sleeves fit a top; a dress length or kind never moves into a top."""
    from seamproof.design import compatible, merge_photos

    assert compatible(("sleeve_length", "cuff"), "dress", "top")
    assert not compatible(("kind", "fit", "length"), "dress", "top")
    assert not compatible(("waist_seam",), "bottom", "dress")
    merged = merge_photos("top", 1, {1: PHOTO_1, 3: PHOTO_3}, {"lengan": 3})
    assert merged.piece.features["sleeve_length"] == "elbow"
    groups = {g["id"]: g["sources"] for g in merged.groups}
    assert groups["siluet"] == [1] and groups["lengan"] == [1, 3]
    with pytest.raises(ValueError):
        merge_photos("top", 1, {1: PHOTO_1, 3: PHOTO_3}, {"siluet": 3})
    with pytest.raises(ValueError):  # the base photo must hold the target garment
        merge_photos("dress", 1, {1: PHOTO_1, 3: PHOTO_3}, {})


def test_merge_endpoint_without_notes_needs_no_model() -> None:
    """Live preview: merging calls no model (call_json is unreachable here) and returns a flat."""
    body = client.post("/api/design/merge", json={"part": "top", "base": 1, "sources": {"1": PHOTO_1, "2": PHOTO_2},
                                                  "assign": {"kerah": 2}, "color": "navy"}).json()
    assert body["features"]["collar"] == "peter_pan" and body["provenance"]["kerah"] == 2
    assert COLORS["navy"][0] in body["plan"]["flat"] and body["changes"] == []
    bad = client.post("/api/design/merge", json={"part": "top", "base": 1, "sources": {"1": PHOTO_1},
                                                 "assign": {"kerah": 2}})
    assert bad.status_code == 422
    offline = client.post("/api/design/merge", json={"part": "top", "base": 1, "sources": {"1": PHOTO_1},
                                                     "notes": "warna lebih gelap"})
    assert offline.status_code == 502


def test_merge_endpoint_applies_notes_with_muse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Notes go through the refine step after the code merge; its diff is reported."""
    monkeypatch.setattr(llm_module, "call_json", _fake({"kind": "shirt", "features": {"length": "thigh"}}))
    body = client.post("/api/design/merge", json={"part": "top", "base": 1, "sources": {"1": PHOTO_1, "2": PHOTO_2},
                                                  "assign": {"lengan": 2}, "notes": "lebih panjang"}).json()
    assert body["changes"] == [{"key": "length", "from": "hip", "to": "thigh"}]
    assert body["features"]["sleeve_length"] == "long"


def test_merge_uses_the_matching_piece_of_a_two_piece_photo() -> None:
    """A photo with a shirt and a skirt lends its skirt to a dress and its shirt sleeves to a top."""
    from seamproof.design import merge_photos

    skirt = {"part": "bottom", "kind": "skirt",
             "features": {"silhouette": "flared", "length": "midi", "hem": "straight", "waistband": True}}
    merged = merge_photos("dress", 3, {3: PHOTO_3, 1: [PHOTO_1, skirt]}, {"rok": 1, "lengan": 1})
    f = merged.piece.features
    assert f["silhouette"] == "flared" and f["length"] == "midi" and f["sleeve_length"] == "short"
    lenders = {g["id"]: g["lenders"] for g in merged.groups}
    assert lenders["rok"]["1"] == "bottom" and lenders["lengan"]["1"] == "top"
    assert "1" not in lenders["pinggang"]  # a skirt has no waist seam to lend


def test_brief_details_the_pattern_cannot_show_are_reported() -> None:
    """Balloon sleeves or trousers in a brief come back as unsupported, never silently lost."""
    reply = {"options": [{"name": "Kemeja lengan balon", "part": "top", "kind": "shirt",
                          "features": dict(SHIRT, sleeve_length="long", cuff=True, sleeve_ruffle="layered"),
                          "color": "dusty_pink",
                          "unsupported": ["lengan balon dikerut", "manset karet", "lengan balon dikerut", " ",
                                          "celana bahan krem"]}]}
    long_brief = "Kemeja wanita dusty pink, lengan balon. " * 40  # ~1600 characters, like a ChatGPT description
    seen: list = []
    option = design_from_brief(long_brief, _fake(reply, seen))[0]
    assert option.unsupported == ["lengan balon dikerut", "manset karet", "celana bahan krem"]
    assert any("sleeve_ruffle" in w for w in option.warnings)  # an invented feature is reported
    assert long_brief.strip()[-30:] in seen[0][1]  # the whole brief reaches Muse, not the first 600 chars
    assert client.post("/api/design", json={"brief": "x" * 1500}).status_code != 422


def test_refine_without_change_says_what_is_missing() -> None:
    """The no-change note names the detail the pattern lacks instead of a generic message."""
    current = {"part": "top", "kind": "shirt", "features": SHIRT, "color": "sage"}
    result = refine_design(current, "buat lengan balon", _fake({"kind": "shirt", "features": {},
                                                                 "unsupported": ["lengan balon"]}))
    assert not result.changes and result.unsupported == ["lengan balon"]
    assert any("lengan balon" in w for w in result.warnings)


def test_flat_draws_contrast_trim_and_visible_lines_on_dark_fabric() -> None:
    """Contrast collar/cuffs get the contrast fill; detail lines turn light on black fabric."""
    shirt = plan_piece("top", "shirt", dict(SHIRT, cuff=True, contrast_collar=True, contrast_cuff=True))
    dark, light = plan_flat(shirt, "hitam"), plan_flat(shirt, "putih")
    assert 'trim contrast' in dark and ".trim.contrast{fill:#F6F5F0" in dark
    assert ".trim.contrast{fill:#2A2A2A" in light
    assert "stroke:#d9dedb" in dark and "stroke:#d9dedb" not in light
    assert "trim contrast" not in plan_flat(plan_piece("top", "shirt", SHIRT), "hitam")
