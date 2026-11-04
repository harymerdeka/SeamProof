"""Women's top models: vest, band collar, and blouse/tunic sleeve, neckline and length choices."""

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from seamproof.draft import REGISTRY
from seamproof.flat import drafted_features, plan_flat
from seamproof.locality import edit_locality
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.mutations import _stretch_rect, m_lost_sleeve
from seamproof.pattern import edge_length
from seamproof.preview3d import plan_preview
from seamproof.server import _plan_dict, app
from seamproof.style import plan_piece
from seamproof.verify import verify_pattern

M = WOMEN_SIZE_TABLE["M"]
TOP = {"collar": "none", "opening": "none", "neckline": "round", "sleeve_length": "short",
       "sleeve_style": "straight", "shoulder": "set_in", "fit": "regular", "length": "hip", "pockets": 0}
BUTTONS = dict(TOP, opening="button_front")
MODELS = [
    ("blouse", dict(TOP, sleeve_length="sleeveless")),
    ("blouse", dict(TOP, neckline="v", sleeve_length="elbow")),
    ("blouse", dict(TOP, neckline="scoop", sleeve_length="three_quarter")),
    ("blouse", dict(TOP, neckline="boat", sleeve_length="long")),
    ("blouse", dict(TOP, sleeve_length="short", sleeve_style="puff")),
    ("blouse", dict(TOP, neckline="v", sleeve_length="long", sleeve_style="bishop")),
    ("blouse", dict(TOP, sleeve_length="three_quarter", cuff=True)),
    ("tunic", dict(TOP, length="hip", fit="fitted", sleeve_length="sleeveless")),
    ("tunic", dict(TOP, length="thigh", neckline="v", sleeve_length="short")),
    ("tunic", dict(TOP, length="knee", fit="boxy", neckline="boat", sleeve_length="elbow")),
    ("tunic", dict(TOP, length="below_knee", sleeve_length="long", sleeve_style="puff")),
    ("shirt", dict(BUTTONS, collar="band", sleeve_length="long", sleeve_style="bishop")),
    ("shirt", dict(BUTTONS, collar="band", sleeve_length="short")),
    ("shirt", dict(BUTTONS, neckline="scoop", sleeve_length="elbow")),
    ("shirt", dict(BUTTONS, neckline="boat", sleeve_length="short")),
    ("vest", dict(BUTTONS, neckline="v", fit="fitted")),
    ("vest", dict(BUTTONS, collar="band")),
    ("vest", dict(BUTTONS, neckline="scoop", length="thigh", pockets=2)),
]


def _draft(kind, features, size="M"):
    """Plan a top from its categories and draft it."""
    plan = plan_piece("top", kind, features)
    return REGISTRY[plan.garment](WOMEN_SIZE_TABLE[size], "woven", plan.params)


def _edges(pattern):
    """Every edge by id."""
    return {e.id: e for panel in pattern.panels for e in panel.edges}


@pytest.mark.parametrize("size", list(WOMEN_SIZE_TABLE))
@pytest.mark.parametrize("kind,features", MODELS)
def test_every_model_passes_the_verifier(size, kind, features) -> None:
    """Each model passes every check in every size."""
    report = verify_pattern(_draft(kind, features, size))
    assert report.passed, [(c.id, c.message_id) for c in report.failures()]


@pytest.mark.parametrize("kind,features", MODELS)
def test_every_model_draws_flat_and_3d(kind, features) -> None:
    """Flat and 3D previews render from the same plan."""
    plan = plan_piece("top", kind, features)
    ET.fromstring(plan_flat(plan))
    assert plan_preview(plan)["lofts"]


def test_default_blouse_and_tunic_are_unchanged() -> None:
    """With no style choice the blocks still draft their standard panels."""
    for garment in ("blouse", "tunic"):
        assert {p.name for p in REGISTRY[garment](M, "woven").panels} == {"front", "back", "sleeve", "facing"}


def test_sleeve_choices_change_the_blouse_pattern() -> None:
    """Short, long, sleeveless and puff each cut a different sleeve, and keep the real body."""
    short = _edges(_draft("blouse", dict(TOP, sleeve_length="short")))
    long_ = _edges(_draft("blouse", dict(TOP, sleeve_length="long")))
    assert short["sleeve.hem"].curve.points[0].y == 200
    assert long_["sleeve.hem"].curve.points[0].y == M.sleeve_length
    bare = _draft("blouse", dict(TOP, sleeve_length="sleeveless"))
    assert "sleeve.cap" not in _edges(bare) and "armhole_bias.attach" in _edges(bare)
    assert bare.size_label == "M" and bare.measurements == M.to_dict()
    puff = _draft("blouse", dict(TOP, sleeve_style="puff"))
    assert "cuff.attach" in _edges(puff) and puff.metadata["gathers_mm"]["sleeve_cap"] == 40
    assert "front.dart_leg1" in _edges(puff)  # still the darted block


def test_necklines_change_the_front() -> None:
    """Scoop is deeper, boat wider and shallower than round; V is straight."""
    def neck(shape):
        points = _edges(_draft("blouse", dict(TOP, neckline=shape)))["front.neckline"].curve.points
        return points[0].y, points[-1].x
    assert neck("scoop")[0] > neck("round")[0] > neck("boat")[0]
    assert neck("boat")[1] > neck("round")[1]
    assert _draft("blouse", dict(TOP, neckline="v")).params["neck_v"] == 1


def test_tunic_length_and_fit_change_the_hem() -> None:
    """A longer choice lowers the hem, a looser fit flares it."""
    def hem(**extra):
        return _edges(_draft("tunic", dict(TOP, **extra)))["front.hem"].curve.points[0]
    assert hem(length="knee").y > hem(length="thigh").y > hem(length="hip").y
    assert hem(fit="boxy").x > hem(fit="fitted").x


def test_band_collar_is_its_own_panel() -> None:
    """A band collar is a straight strip as long as half the neckline, not a pointed collar."""
    pattern = _draft("shirt", dict(BUTTONS, collar="band"))
    edges = _edges(pattern)
    assert pattern.params["collar_style"] == 3
    assert len(edges["collar.neck"].curve.points) == 2
    assert edge_length(edges["collar.neck"]) == pytest.approx(
        edge_length(edges["front.neckline"]) + edge_length(edges["back.neckline"]))
    plan = plan_piece("top", "shirt", dict(BUTTONS, collar="band"))
    assert drafted_features(plan)["collar"] == "band" and not any("band" in n.lower() for n in plan.notes)


def test_vest_is_always_a_sleeveless_shirt() -> None:
    """Kind vest removes the sleeve whatever was asked, and is labelled Vest."""
    plan = plan_piece("top", "vest", dict(BUTTONS, sleeve_length="long", sleeve_style="puff"))
    assert plan.garment == "shirt" and plan.params["sleeveless"] == 1
    assert drafted_features(plan)["sleeve_length"] == "sleeveless"
    assert _plan_dict(plan)["label"] == "Vest"


def test_undraftable_choices_are_reported_and_drawn_as_cut() -> None:
    """Dropped shoulder, pockets and a short blouse become notes, and the flat shows the real cut."""
    plan = plan_piece("top", "blouse", dict(TOP, shoulder="dropped", pockets=2, length="waist", fit="boxy"))
    text = " ".join(plan.notes)
    assert all(word in text for word in ("dropped shoulder", "Pockets", "hip length", "darted fit"))
    drawn = drafted_features(plan)
    assert (drawn["shoulder"], drawn["pockets"], drawn["length"], drawn["fit"]) == ("set_in", 0, "hip", "fitted")


def test_collar_or_buttons_still_move_a_blouse_to_the_shirt() -> None:
    """The darted block has no collar or front opening yet, so the plan says shirt."""
    assert plan_piece("top", "blouse", dict(TOP, collar="peter_pan")).garment == "shirt"
    assert plan_piece("top", "tunic", dict(TOP, opening="button_front")).garment == "shirt"


@pytest.mark.parametrize("param", ["armhole_curve_depth", "shoulder_slope", "neckline_width",
                                   "back_waist_dart_intake", "sleeve_cap_height"])
@pytest.mark.parametrize("features", [dict(TOP, sleeve_style="puff"), dict(TOP, sleeve_length="sleeveless"),
                                      dict(TOP, sleeve_length="short")], ids=["puff", "sleeveless", "short"])
def test_repair_parameters_stay_local_on_styled_blouses(param, features) -> None:
    """Parameters the repair loop may tune move only their declared edges on every style."""
    plan = plan_piece("top", "blouse", features)
    draft = REGISTRY["blouse"]
    change = {"armhole_curve_depth": 25, "shoulder_slope": 30, "neckline_width": 190,
              "back_waist_dart_intake": 25, "sleeve_cap_height": 105}[param]
    before = draft(M, "woven", plan.params)
    after = draft(M, "woven", {**plan.params, param: change})
    result = edit_locality(before, after, {param: change})
    assert result.passed, result.message


def test_mutations_fail_exactly_their_check() -> None:
    """MUTATION: band collar -> S8, blouse bias -> S10, blouse cuff -> S7, lost tunic sleeve -> S3."""
    band = _draft("shirt", dict(BUTTONS, collar="band"))
    assert {c.id for c in verify_pattern(_stretch_rect(band, "collar", "collar", 5, ("neck", "end"))).failures()} == {"S8"}
    bare = _draft("blouse", dict(TOP, sleeve_length="sleeveless"))
    assert {c.id for c in verify_pattern(_stretch_rect(bare, "armhole_bias", "armhole_bias", -5)).failures()} == {"S10"}
    puff = _draft("blouse", dict(TOP, sleeve_style="puff"))
    assert {c.id for c in verify_pattern(_stretch_rect(puff, "cuff", "cuff", 4)).failures()} == {"S7"}
    tunic = _draft("tunic", dict(TOP, length="knee"))
    assert {c.id for c in verify_pattern(m_lost_sleeve(tunic)).failures()} == {"S3"}


@pytest.mark.parametrize("garment,kind,features", [
    ("blouse", "blouse", dict(TOP, sleeve_style="puff")),
    ("tunic", "tunic", dict(TOP, length="knee", neckline="v", sleeve_length="sleeveless")),
    ("shirt", "vest", dict(BUTTONS, collar="band")),
], ids=["puff-blouse", "sleeveless-tunic", "band-vest"])
def test_new_models_reach_the_production_pack(garment, kind, features) -> None:
    """A styled top goes through the real site flow and every size is verified and packed."""
    pack = TestClient(app).post("/api/production", json={
        "garment": garment, "run": {"S": 5, "XL": 5}, "style": {"part": "top", "kind": kind, "features": features}})
    assert pack.status_code == 200, pack.text
    assert all(item["verified"] and item["metres"] > 0 for item in pack.json()["sizes"])
