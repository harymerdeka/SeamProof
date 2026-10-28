"""3D design preview spec: deterministic, follows the drafted categories, serialisable."""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from seamproof.flat import COLORS
from seamproof.preview3d import plan_preview, preview_spec
from seamproof.server import app
from seamproof.style import plan_piece

SHIRT = {"collar": "shirt_collar", "opening": "button_front", "sleeve_length": "long", "shoulder": "set_in",
         "fit": "regular", "length": "hip", "pockets": 1, "cuff": True, "contrast_collar": False,
         "contrast_cuff": False}
SKIRT = {"silhouette": "a_line", "length": "knee", "hem": "straight", "waistband": False}
DRESS = {"collar": "none", "opening": "none", "sleeve_length": "elbow", "shoulder": "set_in", "fit": "regular",
         "pockets": 0, "cuff": False, "contrast_collar": False, "contrast_cuff": False,
         "silhouette": "flared", "length": "midi", "hem": "scalloped", "waist_seam": True}


def _numbers(value):
    """Every number anywhere in a nested spec."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numbers(v)
    elif isinstance(value, list):
        for v in value:
            yield from _numbers(v)


def _tube(spec, tube_id):
    return next(t for t in spec["tubes"] if t["id"] == tube_id)


def _length(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


@pytest.mark.parametrize("part,kind,features", [("top", "shirt", SHIRT), ("bottom", "skirt", SKIRT),
                                                ("dress", "dress", DRESS), ("top", "blouse", {})])
def test_spec_is_deterministic_finite_json(part, kind, features) -> None:
    """Same plan, same spec; every number finite; it survives a JSON round trip."""
    a = plan_preview(plan_piece(part, kind, features), "sage")
    b = plan_preview(plan_piece(part, kind, features), "sage")
    assert a == b and json.loads(json.dumps(a)) == a
    assert all(math.isfinite(n) for n in _numbers(a))
    assert a["color"] == COLORS["sage"][0] and a["version"] == 1
    for loft in a["lofts"]:
        ys = [r["y"] for r in loft["rings"]]
        assert ys == sorted(ys, reverse=True) and all(r["rx"] > 0 and r["rz"] > 0 for r in loft["rings"])
    assert a["form"]["floor_y"] < min(r["y"] for loft in a["lofts"] for r in loft["rings"])


def test_sleeves_follow_length_style_and_cuff() -> None:
    """Longer sleeves are longer tubes; bishop billows then gathers; sleeveless has none."""
    short = preview_spec("top", "shirt", dict(SHIRT, sleeve_length="short", cuff=False))
    long_ = preview_spec("top", "shirt", dict(SHIRT, cuff=False))
    assert _length(_tube(long_, "sleeve_r")["path"]) > 2.5 * _length(_tube(short, "sleeve_r")["path"])
    straight = _tube(long_, "sleeve_r")["radii"]
    assert max(straight) == straight[0]  # a straight sleeve only tapers
    bishop = _tube(preview_spec("top", "shirt", dict(SHIRT, sleeve_style="bishop", elastic_cuff=True)), "sleeve_r")["radii"]
    assert max(bishop) > 1.3 * bishop[0] and bishop[-1] < 0.55 * bishop[0]
    assert {t["id"] for t in long_["tubes"]} == {"sleeve_r", "sleeve_l"}
    assert {t["id"] for t in preview_spec("top", "shirt", SHIRT)["tubes"]} >= {"cuff_r", "cuff_l"}
    assert preview_spec("top", "shirt", dict(SHIRT, sleeve_length="sleeveless"))["tubes"] == []
    left, right = _tube(long_, "sleeve_l")["path"], _tube(long_, "sleeve_r")["path"]
    assert all(l[0] == -r[0] and l[1:] == r[1:] for l, r in zip(left, right))  # mirrored


def test_front_details_appear_only_when_drafted() -> None:
    """Buttons, placket, pockets and collar wings come from the categories."""
    plain = preview_spec("top", "shirt", dict(SHIRT, opening="none", pockets=0, collar="none"))
    assert not plain["buttons"] and not plain["patches"] and not plain["panels"] and not plain["lines"]
    rich = preview_spec("top", "shirt", dict(SHIRT, pockets=2))
    assert len(rich["buttons"]) >= 5 and len(rich["patches"]) == 2
    assert {p["id"] for p in rich["panels"]} == {"collar_r", "collar_l"}
    assert all(b[2] > 0 for b in rich["buttons"])  # on the front surface


def test_preview_shows_what_is_drafted_not_what_was_asked() -> None:
    """A blouse block ignores pocket/collar requests; a shift dress gets no seam or scallop."""
    blouse = plan_preview(plan_piece("top", "blouse", dict(SHIRT, collar="none", opening="none", pockets=2)))
    assert not blouse["patches"] and not blouse["buttons"]
    shift = plan_preview(plan_piece("dress", "dress", dict(DRESS, waist_seam=False)))
    assert not shift["lines"] and "scallop" not in shift["lofts"][0]["rings"][-1]
    seamed = plan_preview(plan_piece("dress", "dress", DRESS))
    assert seamed["lines"][0]["id"] == "waist_seam" and seamed["lofts"][0]["rings"][-1]["scallop"] > 0


def test_skirt_hangs_from_the_form_waist() -> None:
    """The skirt waist sits at the dress form's waist; flare widens the hem."""
    straight = preview_spec("bottom", "skirt", dict(SKIRT, silhouette="straight"))
    flared = preview_spec("bottom", "skirt", dict(SKIRT, silhouette="flared", waistband=True))
    form_waist = min(straight["form"]["rings"], key=lambda r: r["rx"] if r["y"] < -10 else 99)
    assert abs(straight["lofts"][0]["rings"][0]["y"] - form_waist["y"]) < 1.0
    assert flared["lofts"][0]["rings"][-1]["rx"] > straight["lofts"][0]["rings"][-1]["rx"]
    assert [l["id"] for l in flared["lofts"]] == ["skirt", "waistband"]


def test_garment_sits_outside_the_form() -> None:
    """At bust level the garment is wider and deeper than the dress form under it."""
    for fit in ("fitted", "regular", "boxy"):
        spec = preview_spec("top", "shirt", dict(SHIRT, fit=fit))
        bust = spec["lofts"][0]["rings"][2]
        form = min(spec["form"]["rings"], key=lambda r: abs(r["y"] - bust["y"]))
        assert bust["rx"] > form["rx"] and bust["rz"] > form["rz"]


def test_api_plans_carry_the_preview() -> None:
    """Every serialised plan (style-plan, design, photo, merge) includes preview3d."""
    body = TestClient(app).post("/api/style-plan", json={"part": "top", "kind": "shirt", "features": SHIRT,
                                                         "color": "navy"}).json()
    spec = body["plan"]["preview3d"]
    assert spec["color"] == COLORS["navy"][0] and spec["part"] == "top" and spec["tubes"]
