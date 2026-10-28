"""Bodice block, blouse and tunic draft tests, all deterministic and offline."""

import json
import math
import sys

import pytest
from shapely.geometry import LineString

from seamproof.allowance import add_seam_allowance
from seamproof.cli import main
from seamproof.draft.blouse import draft_blouse
from seamproof.draft.bodice import close_dart, draft_bodice
from seamproof.draft.tunic import draft_tunic
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.params import GARMENT_PARAMS, PARAMETERS
from seamproof.pattern import Pattern, Point, edge_length, panel_polygon

BLOUSE = {"garment": "blouse", "draft": draft_blouse}
TUNIC = {"garment": "tunic", "draft": draft_tunic}


@pytest.fixture(params=[BLOUSE, TUNIC], ids=["blouse", "tunic"])
def garment(request) -> dict:
    """Return each draped garment's name and draft function."""
    return request.param


@pytest.fixture
def blouse_m() -> Pattern:
    """Return a size M woven blouse with default parameters."""
    return draft_blouse(WOMEN_SIZE_TABLE["M"], "woven")


def _closed_bust(pattern: Pattern, m: WomenMeasurements) -> float:
    """Sewn bust from the panel: side-join position plus measured back half."""
    front = pattern.panels[0]
    legs = {edge.id: edge for edge in front.edges if edge.id in ("front.dart_leg1", "front.dart_leg2")}
    join = (legs["front.dart_leg1"].curve.points[0].x + legs["front.dart_leg2"].curve.points[-1].x) / 2
    line = LineString([(-1000, m.apex_height), (1000, m.apex_height)])
    back_half = panel_polygon(pattern.panels[1]).intersection(line).length
    return 2 * (join + back_half)


@pytest.mark.parametrize("size", WOMEN_SIZE_TABLE)
def test_sizes_build_closed_panels(garment: dict, size: str) -> None:
    """Every size builds four closed, valid panels with no allowances yet."""
    m = WOMEN_SIZE_TABLE[size]
    pattern = garment["draft"](m, "woven")
    assert pattern.garment_type == garment["garment"]
    assert pattern.size_label == size
    assert [p.name for p in pattern.panels] == ["front", "back", "sleeve", "facing"]
    assert [p.cut_count for p in pattern.panels] == [1, 1, 2, 1]
    assert [p.cut_on_fold for p in pattern.panels] == [True, True, False, False]
    for panel in pattern.panels:
        polygon = panel_polygon(panel)
        assert polygon.is_valid and polygon.area > 0
        assert panel.edges[-1].curve.points[-1] == panel.edges[0].curve.points[0]
        assert panel.cut_line is None
        assert all(edge.allowance_mm == 0 for edge in panel.edges)
    assert pattern.metadata["status"] == "draft_unverified"


@pytest.mark.parametrize("size", WOMEN_SIZE_TABLE)
def test_closed_bust_matches_eased_body(garment: dict, size: str) -> None:
    """Sewn-state bust equals body bust plus woven ease within 2 mm."""
    m = WOMEN_SIZE_TABLE[size]
    pattern = garment["draft"](m, "woven")
    assert _closed_bust(pattern, m) == pytest.approx(m.bust + FABRIC_EASE["woven"].default_ease_mm, abs=2)


def test_dart_legs_equal_within_1mm(blouse_m: Pattern) -> None:
    """Front dart legs match after truing; intake follows the bust rule."""
    m = WOMEN_SIZE_TABLE["M"]
    front = blouse_m.panels[0]
    legs = [edge for edge in front.edges if edge.id in ("front.dart_leg1", "front.dart_leg2")]
    assert len(legs) == 2
    assert abs(edge_length(legs[0]) - edge_length(legs[1])) <= 1
    assert blouse_m.metadata["darts"]["front_bust"]["intake_mm"] == pytest.approx((m.bust - m.high_bust) / 4)
    back_dart = blouse_m.metadata["darts"]["back_waist"]
    assert back_dart["intake_mm"] == pytest.approx(20)
    assert back_dart["leg_mm"] > 0 and len(back_dart["drills"]) == 3


@pytest.mark.parametrize("size", WOMEN_SIZE_TABLE)
def test_sleeve_cap_ease_in_fabric_range(garment: dict, size: str) -> None:
    """Set-in sleeve ease lands inside the woven cap-ease range."""
    pattern = garment["draft"](WOMEN_SIZE_TABLE[size], "woven")
    low, high = FABRIC_EASE["woven"].sleeve_cap_ease_range_mm
    assert low <= pattern.metadata["sleeve_cap_ease_mm"] <= high
    edges = {edge.id: edge for edge in pattern.panels[2].edges}
    assert edge_length(edges["sleeve.underarm_front"]) == edge_length(edges["sleeve.underarm_back"])


def test_seam_references_and_facing(blouse_m: Pattern) -> None:
    """Partners resolve; facing follows the front neckline."""
    edges = {edge.id: edge for panel in blouse_m.panels for edge in panel.edges}
    groups = blouse_m.metadata["seam_groups"]
    for edge in edges.values():
        partner = edge.seam_partner_id
        assert partner is None or partner in edges or partner in groups or partner == edge.id
    for group in groups.values():
        assert all(member["edge_id"] in edges for member in group)
    assert edges["front.side_upper"].seam_partner_id == "back.side_upper"
    assert edges["front.side_lower"].seam_partner_id == "back.side_low"
    assert edges["front.neckline"].seam_partner_id == "facing.neck"
    assert edges["facing.neck"].seam_partner_id == "front.neckline"
    assert edges["front.armhole"].seam_partner_id == edges["back.armhole"].seam_partner_id == "sleeve.cap"
    facing = blouse_m.panels[3]
    assert panel_polygon(facing).is_valid
    assert edge_length(edges["facing.neck"]) == pytest.approx(edge_length(edges["front.neckline"]))


def test_notches_and_grainlines(blouse_m: Pattern) -> None:
    """Shoulder, single/double and dart-seam notches exist; grainlines run vertical."""
    front, back, sleeve, facing = blouse_m.panels
    assert [len(panel.notches) for panel in blouse_m.panels] == [4, 3, 4, 2]
    assert front.notches[0].distance_from_start_mm == back.notches[0].distance_from_start_mm == 0
    assert back.notches[2].distance_from_start_mm - back.notches[1].distance_from_start_mm == pytest.approx(6)
    leg1 = next(edge for edge in front.edges if edge.id == "front.dart_leg1")
    leg2 = next(edge for edge in front.edges if edge.id == "front.dart_leg2")
    assert (leg1.id, 0) in [(n.edge_id, n.distance_from_start_mm) for n in front.notches]
    assert (leg2.id, edge_length(leg2)) in [(n.edge_id, n.distance_from_start_mm) for n in front.notches]
    for panel in blouse_m.panels:
        edges = {edge.id: edge for edge in panel.edges}
        for notch in panel.notches:
            assert notch.edge_id in edges
            assert 0 <= notch.distance_from_start_mm <= edge_length(edges[notch.edge_id])
    for panel in (front, back):
        assert panel.grainline[0].x == panel.grainline[1].x
    assert sleeve.grainline[0].x == sleeve.grainline[1].x == 0


def test_closed_side_seam_is_nearly_continuous(blouse_m: Pattern) -> None:
    """Closing the dart leaves the side seam meeting at a kink under 12 degrees."""
    m = WOMEN_SIZE_TABLE["M"]
    tip = Point(*blouse_m.metadata["darts"]["front_bust"]["tip"])
    closed = close_dart(blouse_m.panels[0], tip)
    assert panel_polygon(closed).is_valid
    edges = {edge.id: edge for edge in closed.edges}
    upper = edges["front.side_upper"].curve.points
    lower = edges["front.side_lower"].curve.points
    assert upper[-1] == lower[0]
    incoming = Point(upper[-1].x - upper[-2].x, upper[-1].y - upper[-2].y)
    outgoing = Point(lower[1].x - lower[0].x, lower[1].y - lower[0].y)
    cosine = ((incoming.x * outgoing.x + incoming.y * outgoing.y)
              / math.hypot(incoming.x, incoming.y) / math.hypot(outgoing.x, outgoing.y))
    assert math.degrees(math.acos(max(-1, min(1, cosine)))) < 12


def test_back_opening_variants() -> None:
    """A centre-back zip splits the back; anything else is rejected."""
    m = WOMEN_SIZE_TABLE["M"]
    zipped = draft_blouse(m, "woven", back_opening="centre_back_zip")
    assert zipped.panels[1].cut_count == 2
    assert not zipped.panels[1].cut_on_fold
    closer = next(edge for edge in zipped.panels[1].edges if edge.id == "back.cb")
    assert closer.seam_partner_id == "back.cb"
    assert panel_polygon(zipped.panels[1]).is_valid
    with pytest.raises(ValueError, match="Back opening"):
        draft_blouse(m, "woven", back_opening="side-seam")


def test_tunic_slit_variants() -> None:
    """Slit 0 omits slit edges; an overlong slit is rejected."""
    m = WOMEN_SIZE_TABLE["M"]
    plain = draft_tunic(m, "woven", {"slit_height": 0})
    assert all("slit" not in edge.id for panel in plain.panels for edge in panel.edges)
    for panel in plain.panels:
        assert panel_polygon(panel).is_valid
    with pytest.raises(ValueError, match="slit"):
        draft_tunic(m, "woven", {"slit_height": 200, "tunic_length": 300})


def test_no_back_dart_builds() -> None:
    """Zero back intake sews no dart and still closes cleanly."""
    pattern = draft_blouse(WOMEN_SIZE_TABLE["M"], "woven", {"back_waist_dart_intake": 0})
    assert pattern.metadata["darts"]["back_waist"]["drills"] == []
    for panel in pattern.panels:
        assert panel_polygon(panel).is_valid


@pytest.mark.parametrize("garment_name", ["blouse", "tunic"])
@pytest.mark.parametrize("name", PARAMETERS)
@pytest.mark.parametrize("bound", ["min", "max"])
def test_parameter_geometry_locality(garment_name: str, name: str, bound: str) -> None:
    """Each applicable parameter moves only its declared panel edges."""
    if name not in GARMENT_PARAMS[garment_name]:
        pytest.skip("Parameter does not apply to this garment")
    from seamproof.draft import REGISTRY

    draft = REGISTRY[garment_name]
    m = WOMEN_SIZE_TABLE["M"]
    spec = PARAMETERS[name]
    base = draft(m, "woven")
    changed = draft(m, "woven", {name: getattr(spec, bound)})
    allowed = set(spec.affects)
    actual = set()
    for old_panel, new_panel in zip(base.panels, changed.panels):
        assert panel_polygon(new_panel).is_valid
        old_edges = {edge.id: edge for edge in old_panel.edges}
        new_edges = {edge.id: edge for edge in new_panel.edges}
        for edge_id in old_edges.keys() | new_edges.keys():
            if edge_id not in old_edges or edge_id not in new_edges:
                actual.add((old_panel.name, edge_id))
            elif old_edges[edge_id].curve != new_edges[edge_id].curve:
                actual.add((old_panel.name, edge_id))
    assert actual
    assert actual <= allowed


def test_bodice_block_builds() -> None:
    """The bare block drafts three panels with the same dart contract."""
    m = WOMEN_SIZE_TABLE["M"]
    pattern = draft_bodice(m, "woven")
    assert [panel.name for panel in pattern.panels] == ["front", "back", "sleeve"]
    assert _closed_bust(pattern, m) == pytest.approx(m.bust + FABRIC_EASE["woven"].default_ease_mm, abs=2)


def test_from_cm_and_custom() -> None:
    """Centimetre inputs store millimetres; off-table drafts read custom."""
    assert WomenMeasurements.from_cm({"bust": 92, "high_bust": 84, "waist": 74, "hip": 98,
                                       "shoulder_width": 39, "shoulder_length": 12.5, "front_length": 43,
                                       "back_length": 40, "apex_height": 26, "apex_distance": 19,
                                       "waist_to_hip": 21, "armhole_depth": 22, "sleeve_length": 57,
                                       "bicep": 29, "wrist": 16}).bust == 920
    measurements = {**WOMEN_SIZE_TABLE["M"].to_dict(), "bust": 930}
    assert draft_blouse(measurements, "woven").size_label == "custom"


@pytest.mark.parametrize("change", [{"bust": 840}, {"front_length": 200}])
def test_reject_impossible_measurements(change: dict) -> None:
    """A missing bust difference or short front is rejected."""
    with pytest.raises(ValueError):
        WomenMeasurements(**{**WOMEN_SIZE_TABLE["M"].to_dict(), **change})


def test_reject_bad_fabric_or_opening() -> None:
    """Unknown fabrics fail before any geometry is built."""
    with pytest.raises(ValueError, match="fabric"):
        draft_tunic(WOMEN_SIZE_TABLE["M"], "denim")


def test_draft_determinism_and_json(blouse_m: Pattern) -> None:
    """Same inputs redraft identically and survive a JSON round trip."""
    again = draft_blouse(WOMEN_SIZE_TABLE["M"].to_dict(), blouse_m.fabric, blouse_m.params)
    assert again.to_dict() == blouse_m.to_dict()
    assert Pattern.from_dict(json.loads(json.dumps(blouse_m.to_dict()))) == blouse_m


def test_allowance_on_darted_blouse(blouse_m: Pattern) -> None:
    """Allowances generate covering cut lines around darts, hems and folds."""
    result = add_seam_allowance(blouse_m)
    for i, panel in enumerate(result.panels):
        cut = panel.cut_line
        assert cut is not None and cut.points[0] == cut.points[-1]
        assert panel_polygon(panel).is_valid
        from shapely.geometry import Polygon

        assert Polygon([(point.x, point.y) for point in cut.points]).buffer(1e-7).covers(panel_polygon(panel))
        for edge in panel.edges:
            if edge.is_fold:
                assert edge.allowance_mm == 0
            elif edge.id.endswith(".hem"):
                assert edge.allowance_mm == 25


@pytest.mark.parametrize("garment", ["tshirt", "blouse", "tunic"])
def test_cli_drafts_garment(monkeypatch: pytest.MonkeyPatch, tmp_path, garment: str) -> None:
    """The registry backs draft --garment for every production garment."""
    out = tmp_path / garment
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", garment,
                                      "--size", "M", "--out", str(out), "--force"])
    assert main() == 0
    assert (out / "pattern.json").is_file()
