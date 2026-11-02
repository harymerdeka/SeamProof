"""Draft structure, geometry, seam references and locality regression tests."""

import json

import pytest
from shapely.geometry import LineString

from seamproof.draft.tshirt import draft_tshirt
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import SIZE_TABLE
from seamproof.params import GARMENT_PARAMS, PARAMETERS, resolve_params
from seamproof.pattern import Pattern, edge_length, panel_polygon


@pytest.fixture
def tshirt() -> Pattern:
    """Return a size M stitch-line draft."""
    return draft_tshirt(SIZE_TABLE["M"], "knit_low_stretch", {})


@pytest.mark.parametrize("fabric", FABRIC_EASE)
@pytest.mark.parametrize("size", SIZE_TABLE)
def test_sizes_build_closed_panels_and_chest(size: str, fabric: str) -> None:
    """Every size and fabric builds closed panels with the eased chest width."""
    m = SIZE_TABLE[size]
    pattern = draft_tshirt(m, fabric, {})
    assert pattern.size_label == size
    assert [p.name for p in pattern.panels] == ["front", "back", "sleeve", "neckband"]
    assert [p.cut_count for p in pattern.panels] == [1, 1, 2, 1]
    assert [p.cut_on_fold for p in pattern.panels] == [True, True, False, False]
    for panel in pattern.panels:
        polygon = panel_polygon(panel)
        assert polygon.is_valid and polygon.area > 0
        assert polygon.exterior.is_closed
        assert panel.edges[-1].curve.points[-1] == panel.edges[0].curve.points[0]
        assert panel.cut_line is None
        assert all(edge.allowance_mm == 0 for edge in panel.edges)
    chest_line = LineString([(-1000, m.armhole_depth), (1000, m.armhole_depth)])
    chest = 2 * sum(panel_polygon(panel).intersection(chest_line).length for panel in pattern.panels[:2])
    assert chest == pytest.approx(m.chest + FABRIC_EASE[fabric].default_ease_mm, abs=2)


def test_seam_references_and_composites(tshirt: Pattern) -> None:
    """Every seam partner resolves and composite groups cover armhole/neckline."""
    edges = {e.id: e for panel in tshirt.panels for e in panel.edges}
    groups = tshirt.metadata["seam_groups"]
    assert len(edges) == sum(len(panel.edges) for panel in tshirt.panels)
    for edge in edges.values():
        if edge.seam_partner_id:
            assert edge.seam_partner_id in edges or edge.seam_partner_id in groups
    for group in groups.values():
        assert all(member["edge_id"] in edges for member in group)
    assert edges["front.side"].seam_partner_id == "back.side"
    assert edges["back.side"].seam_partner_id == "front.side"
    assert edges["front.shoulder"].seam_partner_id == "back.shoulder"
    assert edges["back.shoulder"].seam_partner_id == "front.shoulder"
    assert edges["sleeve.underarm_front"].seam_partner_id == "sleeve.underarm_back"
    assert edges["sleeve.underarm_back"].seam_partner_id == "sleeve.underarm_front"
    assert edge_length(edges["sleeve.underarm_front"]) == edge_length(edges["sleeve.underarm_back"])
    assert edges["front.armhole"].seam_partner_id == edges["back.armhole"].seam_partner_id == "sleeve.cap"
    assert [item["edge_id"] for item in groups["body.armholes"]] == ["front.armhole", "back.armhole"]
    neck_length = 2 * sum(edge_length(edges[f"{name}.neckline"]) for name in ("front", "back"))
    assert edge_length(edges["neckband.attach"]) == pytest.approx(neck_length)


def test_notches_and_grainlines(tshirt: Pattern) -> None:
    """Notch counts, double-notch spacing and grainline directions hold."""
    front, back, sleeve, band = tshirt.panels
    assert [len(panel.notches) for panel in tshirt.panels] == [2, 3, 4, 6]
    assert front.notches[0].distance_from_start_mm == back.notches[0].distance_from_start_mm == 0
    assert back.notches[2].distance_from_start_mm - back.notches[1].distance_from_start_mm == pytest.approx(6)
    assert sleeve.notches[3].distance_from_start_mm - sleeve.notches[2].distance_from_start_mm == pytest.approx(6)
    for panel in tshirt.panels:
        edges = {edge.id: edge for edge in panel.edges}
        for notch in panel.notches:
            assert notch.edge_id in edges
            assert 0 <= notch.distance_from_start_mm <= edge_length(edges[notch.edge_id])
    for panel in (front, back):
        a, b = panel.grainline
        fold = next(e for e in panel.edges if e.is_fold)
        assert a.x == b.x
        assert fold.curve.points[0].x == fold.curve.points[-1].x == 0
    assert sleeve.grainline[0].x == sleeve.grainline[1].x == 0
    assert band.notches[1].distance_from_start_mm == edge_length(band.edges[0]) / 2


@pytest.mark.parametrize("name", GARMENT_PARAMS["tshirt"])
@pytest.mark.parametrize("bound", ["min", "max"])
def test_parameter_geometry_locality(name: str, bound: str, tshirt: Pattern) -> None:
    """Each parameter at its limits moves only its declared panel edges."""
    spec = PARAMETERS[name]
    changed = draft_tshirt(SIZE_TABLE["M"], tshirt.fabric, {name: getattr(spec, bound)})
    allowed = set(spec.affects)
    actual = set()
    for old_panel, new_panel in zip(tshirt.panels, changed.panels):
        assert panel_polygon(new_panel).is_valid
        for old_edge, new_edge in zip(old_panel.edges, new_panel.edges):
            key = (old_panel.name, old_edge.id)
            if old_edge.curve != new_edge.curve:
                actual.add(key)
    assert actual
    assert actual <= allowed


def test_draft_determinism_and_json(tshirt: Pattern) -> None:
    """Same inputs redraft identically and survive a JSON round trip."""
    again = draft_tshirt(SIZE_TABLE["M"].to_dict(), tshirt.fabric, tshirt.params)
    assert again.to_dict() == tshirt.to_dict()
    assert Pattern.from_dict(json.loads(json.dumps(tshirt.to_dict()))) == tshirt
    assert tshirt.metadata["status"] == "draft_unverified"


def test_custom_measurements() -> None:
    """Off-table measurements draft with a custom size label."""
    measurements = {**SIZE_TABLE["M"].to_dict(), "chest": 970}
    assert draft_tshirt(measurements, "knit_low_stretch").size_label == "custom"


@pytest.mark.parametrize("params", [{"unknown": 1}, {"neck_drop_front": 0}, {"shoulder_slope": float("nan")}, {"sleeve_cap_height": True}, {"sleeve_cap_height": "100"}])
def test_reject_invalid_params(params: dict) -> None:
    """Unknown, out-of-range or non-numeric params are rejected."""
    with pytest.raises(ValueError):
        draft_tshirt(SIZE_TABLE["M"], "knit_low_stretch", params)


def test_parameter_specs_and_fabric_ranges() -> None:
    """Specs with valid bounds; defaults sit inside fabric ease ranges."""
    assert len(PARAMETERS) == 39  # 16 block + 12 shirt + 6 skirt + 2 dress + 1 blouse + 2 finish style params
    assert resolve_params() == {name: spec.default for name, spec in PARAMETERS.items()}
    for spec in PARAMETERS.values():
        assert spec.min <= spec.default <= spec.max
        assert spec.description and spec.affects
    covered = {name for names in GARMENT_PARAMS.values() for name in names}
    assert covered == set(PARAMETERS)
    for fabric in FABRIC_EASE.values():
        for low, high in (fabric.chest_ease_range_mm, fabric.waist_ease_range_mm, fabric.hip_ease_range_mm):
            assert low <= fabric.default_ease_mm <= high
        assert fabric.sleeve_cap_ease_range_mm[0] <= fabric.sleeve_cap_ease_range_mm[1]


def test_invalid_fabric_or_proportions() -> None:
    """Unknown fabrics and impossible proportions fail with a clear error."""
    m = SIZE_TABLE["M"].to_dict()
    with pytest.raises(ValueError, match="fabric"):
        draft_tshirt(m, "unknown")
    with pytest.raises(ValueError, match="Body measurements"):
        draft_tshirt({**m, "armhole_depth": 800}, "knit_low_stretch")
    with pytest.raises(ValueError, match="Shoulder drop"):
        draft_tshirt({**m, "armhole_depth": 45}, "knit_low_stretch", {"neck_drop_front": 40, "shoulder_slope": 50})
    with pytest.raises(ValueError, match="Sleeve length"):
        draft_tshirt({**m, "sleeve_length": 90}, "knit_low_stretch")
