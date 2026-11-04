"""Photo style path, shirt and skirt drafts: offline (vision mocked), deterministic."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.allowance import ensure_allowance
from seamproof.draft import REGISTRY
from seamproof.locality import edit_locality
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import BASES, MUTATIONS, SHIRT_STYLE, SKIRT_STYLE
from seamproof.params import GARMENT_PARAMS, PARAMETERS, resolve_params, tunable_params
from seamproof.production import produce
from seamproof.server import app
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern
from seamproof.vision import describe_garments, prepare_photo, validate_reading

# What Muse Wonder returned for the sample shirt-and-skirt photo (recorded once, live).
SAMPLE_READING = {
    "photo_view": "worn",
    "pieces": [
        {"part": "top", "kind": "shirt", "evidence": "button-front top, white collar and cuffs",
         "features": {"collar": "peter_pan", "opening": "button_front", "sleeve_length": "elbow",
                      "shoulder": "dropped", "fit": "boxy", "length": "hip", "cuff": True,
                      "contrast_collar": True, "contrast_cuff": True, "pockets": 2}},
        {"part": "bottom", "kind": "skirt", "evidence": "skirt widening to a scalloped hem",
         "features": {"silhouette": "a_line", "length": "mini", "hem": "scalloped", "waistband": False}},
    ],
}


def _jpeg() -> bytes:
    """A small synthetic photo (no people) encoded as JPEG."""
    import cv2
    import numpy as np

    image = np.full((400, 300, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (220, 300), (20, 20, 20), -1)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return bytes(data)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network: text LLM unreachable, vision returns the recorded sample reading."""
    def unreachable(*args, **kwargs):
        raise llm_module.LLMError("offline")

    monkeypatch.setattr(llm_module, "call_json", unreachable)
    monkeypatch.setattr(llm_module, "call_vision_json", lambda *a, **k: SAMPLE_READING)


def test_prepare_photo_downscales_and_reencodes() -> None:
    """Photos become JPEG within the size cap; garbage is refused."""
    prepared = prepare_photo(_jpeg())
    assert prepared.jpeg[:2] == b"\xff\xd8" and max(prepared.width, prepared.height) <= 1280
    assert prepared.faces_blurred == 0
    with pytest.raises(ValueError):
        prepare_photo(b"not an image")


def test_reading_coerces_unknown_values_with_warnings() -> None:
    """Unknown categories fall back and are reported, never passed through."""
    reading = validate_reading({"pieces": [
        {"part": "top", "kind": "jacket", "features": {"collar": "sailor", "pockets": 7}},
        {"part": "shoes", "kind": "boot", "features": {}},
    ]})
    assert len(reading.pieces) == 1
    top = reading.pieces[0]
    assert top.kind == "other" and top.features["collar"] == "none" and top.features["pockets"] == 0
    assert len(reading.warnings) == 4


@pytest.mark.parametrize("collar,opening,expected", [("shirt_collar", "button_front", "round"),
                                                     ("none", "button_front", "v"),
                                                     ("shirt_collar", "none", "v")])
def test_open_shirt_collar_is_not_read_as_a_v_neck(collar: str, opening: str, expected: str) -> None:
    """A collared button-front shirt worn open is a collared neck; other V readings stay V."""
    reading = validate_reading({"pieces": [{"part": "top", "kind": "shirt", "features": {
        "collar": collar, "opening": opening, "neckline": "v"}}]})
    piece = reading.pieces[0]
    assert piece.features["neckline"] == expected and not reading.warnings
    if expected == "round":
        plan = plan_piece(piece.part, piece.kind, piece.features)
        assert not any("V neck" in note for note in plan.notes)


def test_sample_reading_maps_to_shirt_and_skirt() -> None:
    """The sample photo plans a boxy shirt with collar/cuffs/pockets and a scallop skirt."""
    reading = describe_garments(prepare_photo(_jpeg()))
    plans = [plan_piece(p.part, p.kind, p.features) for p in reading.pieces]
    shirt, skirt = plans
    assert shirt.garment == "shirt" and skirt.garment == "skirt"
    assert shirt.params["collar_style"] == 2 and shirt.params["pocket_count"] == 2
    assert shirt.params["cuff_depth"] > 0 and shirt.params["shoulder_drop"] > 0
    assert skirt.params["scallop_depth"] > 0 and skirt.params["waistband_height"] == 0
    assert any("Contrast" in note for note in shirt.notes)


def test_unsupported_pieces_are_reported() -> None:
    """Trousers have no block: the plan says so instead of guessing."""
    plan = plan_piece("bottom", "trousers", {})
    assert plan.garment is None and plan.notes


@pytest.mark.parametrize("size", WOMEN_SIZE_TABLE)
@pytest.mark.parametrize("garment,style", [("shirt", SHIRT_STYLE), ("shirt", {}),
                                           ("skirt", SKIRT_STYLE), ("skirt", {})])
def test_shirt_and_skirt_pass_every_check(garment: str, style: dict, size: str) -> None:
    """Every size, with and without the photo style, passes the verifier and takes allowance."""
    pattern = REGISTRY[garment](WOMEN_SIZE_TABLE[size], "woven", style)
    report = verify_pattern(pattern)
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]
    assert all(panel.cut_line is not None for panel in ensure_allowance(pattern).panels)


def test_style_switches_add_and_remove_panels() -> None:
    """Collar, cuffs, pockets, scallop facing and waistband follow their params."""
    m = WOMEN_SIZE_TABLE["M"]
    names = lambda p: {panel.name for panel in p.panels}  # noqa: E731
    assert names(REGISTRY["shirt"](m, "woven", SHIRT_STYLE)) == {"front", "back", "sleeve", "collar", "cuff", "pocket"}
    assert names(REGISTRY["shirt"](m, "woven", {"collar_style": 0})) == {"front", "back", "sleeve", "neck_bias"}
    assert names(REGISTRY["skirt"](m, "woven", SKIRT_STYLE)) == {"front", "back", "waist_facing", "hem_facing"}
    assert names(REGISTRY["skirt"](m, "woven", {})) == {"front", "back", "waistband"}
    with pytest.raises(ValueError):
        resolve_params({"pocket_count": 1.5})


@pytest.mark.parametrize("check_id,base,description,mutate",
                         [m for m in MUTATIONS if m[1] in ("shirt", "skirt")])
def test_new_mutations_fail_exactly_their_check(check_id, base, description, mutate) -> None:
    """Each shirt/skirt mutation breaks only the check it targets."""
    failed = {c.id for c in verify_pattern(mutate(BASES[base]())).failures()}
    assert failed == {check_id}, description


def _sweep(garment: str, style: dict):
    """Yield (name, before, after) for min/max of every applicable param."""
    m = WOMEN_SIZE_TABLE["M"]
    before = REGISTRY[garment](m, "woven", style)
    for name in GARMENT_PARAMS[garment]:
        spec = PARAMETERS[name]
        for value in (spec.min, spec.max):
            changed = dict(style, **{name: value})
            try:
                after = REGISTRY[garment](m, "woven", changed)
            except ValueError:
                continue  # an extreme the block refuses is not a locality question
            yield name, before, after


@pytest.mark.parametrize("garment,style", [("shirt", SHIRT_STYLE), ("skirt", SKIRT_STYLE)])
def test_every_param_moves_only_declared_edges(garment: str, style: dict) -> None:
    """Edit locality holds for every shirt/skirt parameter at both range ends."""
    checked = 0
    for name, before, after in _sweep(garment, style):
        result = edit_locality(before, after, {name: 0.0})
        assert result.passed, (name, result.message)
        checked += 1
    assert checked >= len(GARMENT_PARAMS[garment])


def test_locality_catches_a_leaky_shirt_change() -> None:
    """A collar change that also moves the back hem is refused."""
    m = WOMEN_SIZE_TABLE["M"]
    before = REGISTRY["shirt"](m, "woven", SHIRT_STYLE)
    after = REGISTRY["shirt"](m, "woven", dict(SHIRT_STYLE, collar_width=80, shirt_length=260))
    assert not edit_locality(before, after, {"collar_width": 80}).passed


def test_style_params_are_never_offered_to_the_llm() -> None:
    """Only block params are tunable; skirt has none, so repair runs no rounds."""
    assert set(tunable_params("shirt")) == {"armhole_curve_depth", "shoulder_slope", "neckline_depth_front",
                                            "neck_drop_back", "neckline_width"}
    assert tunable_params("skirt") == []


@pytest.mark.parametrize("garment,style", [("shirt", SHIRT_STYLE), ("skirt", SKIRT_STYLE)])
def test_production_pack_for_new_garments(garment: str, style: dict) -> None:
    """A size run packs and costs shirts and skirts like the other garments."""
    report = produce(garment, "S=5,M=10", "woven", 1500.0, 45000.0)
    assert report.total_metres > 0 and all(item.verified for item in report.sizes)


client = TestClient(app)


def test_photo_endpoint_returns_blurred_preview_and_plans() -> None:
    """Upload -> blurred preview + categorical reading + deterministic plans."""
    body = client.post("/api/photo", files={"file": ("sample.jpg", _jpeg(), "image/jpeg")}).json()
    assert body["preview"].startswith("data:image/jpeg;base64,")
    assert [p["plan"]["garment"] for p in body["pieces"]] == ["shirt", "skirt"]
    assert "collar" in body["options"]["top"] and "hem" in body["options"]["bottom"]


def test_pack_with_style_drafts_the_photo_garment() -> None:
    """A pack with style features drafts and verifies the styled shirt."""
    piece = SAMPLE_READING["pieces"][0]
    pack = client.post("/api/pack", json={"garment": "shirt", "fabric": "woven", "mode": "size",
                                          "size": "M", "style": piece}).json()
    assert pack["passed"] and pack["files"]["pdf"]
    assert pack["style"]["params"]["collar_style"] == 2
    mismatch = client.post("/api/pack", json={"garment": "skirt", "mode": "size", "size": "M", "style": piece})
    assert mismatch.status_code == 422


@pytest.mark.parametrize("sleeve_length", [None, "unknown", "short", "long", "sleeveless"])
def test_shirt_pack_only_omits_sleeves_when_explicit(sleeve_length: str | None) -> None:
    """Missing/unknown length keeps sleeves in the design, JSON, SVG and PDF."""
    import xml.etree.ElementTree as ET

    features = {"collar": "shirt_collar", "opening": "button_front"}
    if sleeve_length is not None:
        features["sleeve_length"] = sleeve_length
    style = {"part": "top", "kind": "shirt", "features": features}
    plan = client.post("/api/style-plan", json=style).json()
    expected = sleeve_length if sleeve_length in ("long", "sleeveless") else "short"
    assert plan["features"]["sleeve_length"] == expected
    assert plan["plan"]["drafted"]["sleeve_length"] == expected
    assert bool(plan["plan"]["preview3d"]["tubes"]) == (expected != "sleeveless")

    response = client.post("/api/pack", json={"garment": "shirt", "fabric": "woven",
                                               "mode": "size", "size": "M", "style": style})
    assert response.status_code == 200
    pack = response.json()
    assert pack["passed"] and pack["files"]["pdf"]
    pattern = client.get(pack["files"]["json"]).json()
    panels = {panel["name"]: panel for panel in pattern["panels"]}
    svg = ET.fromstring(client.get(pack["files"]["svg"]).content)
    ids = {element.get("id") for element in svg.iter()}
    pdf = client.get(pack["files"]["pdf"]).content
    if expected == "sleeveless":
        assert "sleeve" not in panels and "sleeve" not in ids
        assert b"sleeve " not in pdf and "armhole_bias" in panels
    else:
        assert panels["sleeve"]["cut_count"] == 2
        assert {"sleeve", "sleeve.cap", "sleeve.cut"} <= ids
        assert b"sleeve " in pdf


def test_style_plan_endpoint_remaps_edits() -> None:
    """Editing a category re-plans without any model call."""
    piece = dict(SAMPLE_READING["pieces"][1])
    piece["features"] = dict(piece["features"], hem="straight", waistband=True)
    plan = client.post("/api/style-plan", json=piece).json()["plan"]
    assert plan["params"]["scallop_depth"] == 0 and plan["params"]["waistband_height"] > 0


def test_production_uses_the_photo_style() -> None:
    """Production with a style packs the styled shirt (collar, cuffs, pockets in the cut list)."""
    piece = SAMPLE_READING["pieces"][0]
    body = client.post("/api/production", json={"garment": "shirt", "fabric": "woven", "run": {"M": 10},
                                                "width": 1500, "price": 45000, "style": piece}).json()
    assert {"M collar", "M cuff", "M pocket"} <= set(body["cut_list"])
    assert body["cut_list"]["M pocket"] == 20
