"""Verifier baselines, known failures and one mutation test per check."""

import copy

import pytest

from seamproof.draft.blouse import draft_blouse
from seamproof.draft.tshirt import draft_tshirt
from seamproof.draft.tunic import draft_tunic
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE
from seamproof.pattern import Curve, Edge, Notch, Panel, Pattern, Point, edge_length
from seamproof.verify import Report, verify_pattern

BLOUSE_IDS = {"G1", "G2", "S1", "S2a", "S2b", "S3", "S4", "S5", "W1", "W2", "W3",
              "D1", "D2", "D3", "D4", "D5", "N1", "N2", "R1"}
TSHIRT_IDS = {"G1", "G2", "S1", "S2", "S3", "S5", "S6", "W1", "W2", "W3", "N1", "N2", "R1"}


@pytest.fixture
def blouse_m() -> Pattern:
    """Return a pristine size M woven blouse."""
    return draft_blouse(WOMEN_SIZE_TABLE["M"], "woven")


def _edge(panel: Panel, edge_id: str) -> Edge:
    return next(edge for edge in panel.edges if edge.id == edge_id)


def _swap(panel: Panel, *new_edges: Edge) -> Panel:
    """Return a panel with the named edges replaced, preserving the rest."""
    table = {edge.id: edge for edge in new_edges}
    rebuilt = [table.get(edge.id, edge) for edge in panel.edges]
    return Panel(panel.name, rebuilt, panel.grainline, list(panel.notches),
                 panel.cut_count, panel.cut_on_fold, panel.cut_line)


def _swap_pattern(pattern: Pattern, index: int, panel: Panel) -> Pattern:
    """Return a pattern copy with one panel replaced."""
    clone = copy.deepcopy(pattern)
    clone.panels[index] = panel
    return clone


def _only(report: Report, check_id: str) -> None:
    """Assert exactly the named check failed."""
    assert not report.passed
    assert {check.id for check in report.failures()} == {check_id}


def _sync_leg_notches(panel: Panel) -> Panel:
    """Re-place dart-seam notches after a leg mutation, as a drafter would."""
    legs = {edge.id: edge for edge in panel.edges if edge.id in ("front.dart_leg1", "front.dart_leg2")}
    notches = [n for n in panel.notches if n.edge_id not in legs]
    notches.append(Notch("front.dart_leg1", 0))
    notches.append(Notch("front.dart_leg2", edge_length(legs["front.dart_leg2"])))
    return Panel(panel.name, list(panel.edges), panel.grainline, notches,
                 panel.cut_count, panel.cut_on_fold, panel.cut_line)


@pytest.mark.parametrize("size", WOMEN_SIZE_TABLE)
@pytest.mark.parametrize("draft", [draft_blouse, draft_tunic])
def test_defaults_pass_every_check(draft, size: str) -> None:
    """Every size of blouse and tunic passes the full check list."""
    report = verify_pattern(draft(WOMEN_SIZE_TABLE[size], "woven"))
    assert report.passed
    assert {check.id for check in report.checks} == BLOUSE_IDS
    assert report.summary_id().startswith("LULUS")


@pytest.mark.parametrize("fabric", ["knit_low_stretch", "knit_high_stretch"])
def test_tshirt_baselines(fabric: str) -> None:
    """Passing t-shirt combos by size; the check list has no dart checks."""
    size = "S" if fabric == "knit_low_stretch" else "M"
    report = verify_pattern(draft_tshirt(SIZE_TABLE[size], fabric))
    assert report.passed
    assert {check.id for check in report.checks} == TSHIRT_IDS


def test_tshirt_negative_ease_is_reported() -> None:
    """Size M knit_low has -1.3 mm cap ease, so exactly S3 must fail."""
    report = verify_pattern(draft_tshirt(SIZE_TABLE["M"], "knit_low_stretch"))
    _only(report, "S3")
    assert "di luar rentang" in report.failures()[0].message_id


def test_variants_pass() -> None:
    """Zip back, missing back dart and slit-free tunic all verify clean."""
    m = WOMEN_SIZE_TABLE["M"]
    assert verify_pattern(draft_blouse(m, "woven", back_opening="centre_back_zip")).passed
    assert verify_pattern(draft_blouse(m, "woven", {"back_waist_dart_intake": 0})).passed
    assert verify_pattern(draft_tunic(m, "woven", {"slit_height": 0})).passed


def test_mutation_breaks_closure(blouse_m: Pattern) -> None:
    """Shifting a facing end breaks exactly G1."""
    facing = blouse_m.panels[3]
    end = _edge(facing, "facing.end_cf")
    a, b = end.curve.points
    moved = Edge(end.id, Curve((a, Point(b.x + 7, b.y))))
    report = verify_pattern(_swap_pattern(blouse_m, 3, _swap(facing, moved)))
    _only(report, "G1")


def test_mutation_tampers_params(blouse_m: Pattern) -> None:
    """Editing params without redrafting breaks exactly G2."""
    clone = copy.deepcopy(blouse_m)
    clone.params["bust_dart_clearance"] = 21.0
    _only(verify_pattern(clone), "G2")


def test_mutation_shortens_shoulder(blouse_m: Pattern) -> None:
    """A 2 mm shorter back shoulder breaks exactly S1."""
    back = blouse_m.panels[1]
    shoulder = _edge(back, "back.shoulder")
    armhole = _edge(back, "back.armhole")
    tip = Point(shoulder.curve.points[-1].x - 2, shoulder.curve.points[-1].y)
    new_shoulder = Edge(shoulder.id, Curve((shoulder.curve.points[0], tip)), shoulder.seam_partner_id)
    new_armhole = Edge(armhole.id, Curve((tip, *armhole.curve.points[1:])), armhole.seam_partner_id)
    report = verify_pattern(_swap_pattern(blouse_m, 1, _swap(back, new_shoulder, new_armhole)))
    _only(report, "S1")


def test_mutation_lengthens_side_above(blouse_m: Pattern) -> None:
    """Dropping the underarm 8 mm breaks exactly S2a."""
    front = blouse_m.panels[0]
    upper = _edge(front, "front.side_upper")
    armhole = _edge(front, "front.armhole")
    low = Point(upper.curve.points[0].x, upper.curve.points[0].y - 8)
    new_upper = Edge(upper.id, Curve((low, upper.curve.points[-1])), upper.seam_partner_id)
    new_armhole = Edge(armhole.id, Curve((*armhole.curve.points[:-1], low)), armhole.seam_partner_id)
    report = verify_pattern(_swap_pattern(blouse_m, 0, _swap(front, new_upper, new_armhole)))
    _only(report, "S2a")


def test_mutation_widens_hem(blouse_m: Pattern) -> None:
    """Widening the back hem corner 35 mm breaks exactly S2b."""
    back = blouse_m.panels[1]
    lower = _edge(back, "back.side_lower")
    hem = _edge(back, "back.hem")
    corner = Point(lower.curve.points[-1].x + 35, lower.curve.points[-1].y)
    new_lower = Edge(lower.id, Curve((lower.curve.points[0], corner)), lower.seam_partner_id)
    new_hem = Edge(hem.id, Curve((corner, hem.curve.points[-1])))
    report = verify_pattern(_swap_pattern(blouse_m, 1, _swap(back, new_lower, new_hem)))
    _only(report, "S2b")


def test_mutation_flattens_cap(blouse_m: Pattern) -> None:
    """Stretching the armholes sideways starves cap ease: exactly S3 fails."""
    panels = []
    for index in (0, 1):
        panel = blouse_m.panels[index]
        armhole = _edge(panel, panel.name + ".armhole")
        base = armhole.curve.points[-1]
        inner = [Point(base.x + (p.x - base.x) * 1.35, p.y) for p in armhole.curve.points[1:-1]]
        widened = Curve((armhole.curve.points[0], *inner, armhole.curve.points[-1]))
        panels.append(_swap(panel, Edge(armhole.id, widened, armhole.seam_partner_id)))
    clone = copy.deepcopy(blouse_m)
    clone.panels[0], clone.panels[1] = panels
    _only(verify_pattern(clone), "S3")


def test_mutation_stretches_facing(blouse_m: Pattern) -> None:
    """Bowing the facing neck outward breaks exactly S4."""
    facing = blouse_m.panels[3]
    neck = _edge(facing, "facing.neck")
    mid = len(neck.curve.points) // 2
    points = [Point(p.x, p.y - 8) for p in neck.curve.points]
    points[0], points[-1] = neck.curve.points[0], neck.curve.points[-1]
    assert mid not in (0, len(points) - 1)
    report = verify_pattern(_swap_pattern(blouse_m, 3, _swap(facing, Edge(neck.id, Curve(tuple(points)), neck.seam_partner_id))))
    _only(report, "S4")


def test_mutation_lengthens_underarm(blouse_m: Pattern) -> None:
    """Dropping one sleeve hem corner 5 mm breaks exactly S5."""
    sleeve = blouse_m.panels[2]
    back_seam = _edge(sleeve, "sleeve.underarm_back")
    hem = _edge(sleeve, "sleeve.hem")
    low = Point(back_seam.curve.points[-1].x, back_seam.curve.points[-1].y + 5)
    new_seam = Edge(back_seam.id, Curve((back_seam.curve.points[0], low)), back_seam.seam_partner_id)
    new_hem = Edge(hem.id, Curve((low, hem.curve.points[-1])))
    report = verify_pattern(_swap_pattern(blouse_m, 2, _swap(sleeve, new_seam, new_hem)))
    _only(report, "S5")


def test_mutation_shortens_neckband() -> None:
    """A 5 mm shorter neckband ring breaks exactly S6 on the t-shirt."""
    pattern = draft_tshirt(SIZE_TABLE["S"], "knit_low_stretch")
    band = pattern.panels[3]
    attach = _edge(band, "neckband.attach")
    end = _edge(band, "neckband.end_right")
    short = Point(attach.curve.points[-1].x - 5, attach.curve.points[-1].y)
    new_attach = Edge(attach.id, Curve((attach.curve.points[0], short)), attach.seam_partner_id)
    new_end = Edge(end.id, Curve((short, end.curve.points[-1])), end.seam_partner_id)
    swapped = _swap(band, new_attach, new_end)
    length = edge_length(new_attach)
    swapped.notches = [n if n.edge_id != attach.id else Notch(n.edge_id, min(n.distance_from_start_mm, length))
                       for n in swapped.notches]
    _only(verify_pattern(_swap_pattern(pattern, 3, swapped)), "S6")


def test_mutation_narrows_bust_join(blouse_m: Pattern) -> None:
    """Narrowing both dart base ends 5 mm breaks exactly W1."""
    front = blouse_m.panels[0]
    upper = _edge(front, "front.side_upper")
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    lower = _edge(front, "front.side_lower")
    d1 = Point(leg1.curve.points[0].x - 5, leg1.curve.points[0].y)
    d2 = Point(leg2.curve.points[-1].x - 5, leg2.curve.points[-1].y)
    tip = leg1.curve.points[-1]
    new_edges = [
        Edge(upper.id, Curve((upper.curve.points[0], d1)), upper.seam_partner_id),
        Edge(leg1.id, Curve((d1, tip)), leg1.seam_partner_id),
        Edge(leg2.id, Curve((tip, d2)), leg2.seam_partner_id),
        Edge(lower.id, Curve((d2, *lower.curve.points[1:])), lower.seam_partner_id),
    ]
    _only(verify_pattern(_swap_pattern(blouse_m, 0, _sync_leg_notches(_swap(front, *new_edges)))), "W1")


def test_mutation_widens_waist(blouse_m: Pattern) -> None:
    """Pushing the front waist out 5 mm breaks exactly W2."""
    front = blouse_m.panels[0]
    lower = _edge(front, "front.side_lower")
    points = [Point(p.x + 5 if i == 1 else p.x, p.y) for i, p in enumerate(lower.curve.points)]
    _only(verify_pattern(_swap_pattern(blouse_m, 0, _swap(front, Edge(lower.id, Curve(tuple(points)), lower.seam_partner_id)))), "W2")


def test_mutation_widens_hip(blouse_m: Pattern) -> None:
    """Pushing the front hip out 5 mm breaks exactly W3."""
    front = blouse_m.panels[0]
    lower = _edge(front, "front.side_lower")
    points = [Point(p.x + 5 if i == 2 else p.x, p.y) for i, p in enumerate(lower.curve.points)]
    _only(verify_pattern(_swap_pattern(blouse_m, 0, _swap(front, Edge(lower.id, Curve(tuple(points)), lower.seam_partner_id)))), "W3")


def test_mutation_lifts_dart_tip(blouse_m: Pattern) -> None:
    """Lifting the tip 15 mm splits the legs, so D1 fails with D2 in tow."""
    front = blouse_m.panels[0]
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    tip = Point(leg1.curve.points[-1].x, leg1.curve.points[-1].y + 15)
    new_edges = [
        Edge(leg1.id, Curve((leg1.curve.points[0], tip)), leg1.seam_partner_id),
        Edge(leg2.id, Curve((tip, leg2.curve.points[-1])), leg2.seam_partner_id),
    ]
    report = verify_pattern(_swap_pattern(blouse_m, 0, _sync_leg_notches(_swap(front, *new_edges))))
    assert not report.passed
    # An asymmetric lift also moves the tip off its clearance circle.
    assert {check.id for check in report.failures()} == {"D1", "D2"}


def test_mutation_shifts_tip_inward(blouse_m: Pattern) -> None:
    """Moving the tip 5 mm toward the base breaks exactly D2."""
    front = blouse_m.panels[0]
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    tip = Point(leg1.curve.points[-1].x + 5, leg1.curve.points[-1].y)
    new_edges = [
        Edge(leg1.id, Curve((leg1.curve.points[0], tip)), leg1.seam_partner_id),
        Edge(leg2.id, Curve((tip, leg2.curve.points[-1])), leg2.seam_partner_id),
    ]
    _only(verify_pattern(_swap_pattern(blouse_m, 0, _sync_leg_notches(_swap(front, *new_edges)))), "D2")


def test_mutation_widens_dart_base(blouse_m: Pattern) -> None:
    """Spreading the dart base 3 mm each way breaks exactly D3."""
    front = blouse_m.panels[0]
    upper = _edge(front, "front.side_upper")
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    lower = _edge(front, "front.side_lower")
    d1 = Point(leg1.curve.points[0].x, leg1.curve.points[0].y - 3)
    d2 = Point(leg2.curve.points[-1].x, leg2.curve.points[-1].y + 3)
    tip = leg1.curve.points[-1]
    new_edges = [
        Edge(upper.id, Curve((upper.curve.points[0], d1)), upper.seam_partner_id),
        Edge(leg1.id, Curve((d1, tip)), leg1.seam_partner_id),
        Edge(leg2.id, Curve((tip, d2)), leg2.seam_partner_id),
        Edge(lower.id, Curve((d2, *lower.curve.points[1:])), lower.seam_partner_id),
    ]
    _only(verify_pattern(_swap_pattern(blouse_m, 0, _swap(front, *new_edges))), "D3")


def test_mutation_kinks_closed_seam(blouse_m: Pattern) -> None:
    """Shoving the tip 120 mm toward the base kinks the closed seam: D4 fails."""
    front = blouse_m.panels[0]
    leg1 = _edge(front, "front.dart_leg1")
    leg2 = _edge(front, "front.dart_leg2")
    tip = Point(leg1.curve.points[-1].x + 120, leg1.curve.points[-1].y)
    new_edges = [
        Edge(leg1.id, Curve((leg1.curve.points[0], tip)), leg1.seam_partner_id),
        Edge(leg2.id, Curve((tip, leg2.curve.points[-1])), leg2.seam_partner_id),
    ]
    report = verify_pattern(_swap_pattern(blouse_m, 0, _sync_leg_notches(_swap(front, *new_edges))))
    assert not report.passed
    # The tip move also breaks tip clearance; the kink finding must fire too.
    assert {c.id for c in report.failures()} == {"D2", "D4"}


def test_mutation_loses_drill(blouse_m: Pattern) -> None:
    """Dropping a back-dart drill breaks exactly D5."""
    clone = copy.deepcopy(blouse_m)
    clone.metadata["darts"]["back_waist"]["drills"].pop()
    _only(verify_pattern(clone), "D5")


def test_mutation_moves_notch(blouse_m: Pattern) -> None:
    """Pushing a notch past its edge breaks exactly N1."""
    clone = copy.deepcopy(blouse_m)
    notch = clone.panels[2].notches[0]
    edge = next(e for e in clone.panels[2].edges if e.id == notch.edge_id)
    clone.panels[2].notches[0] = Notch(notch.edge_id, edge_length(edge) + 10)
    _only(verify_pattern(clone), "N1")


def test_mutation_removes_shoulder_notch(blouse_m: Pattern) -> None:
    """Removing the front zero notch breaks exactly N2."""
    clone = copy.deepcopy(blouse_m)
    front = clone.panels[0]
    front.notches = [n for n in front.notches if not (n.edge_id == "front.armhole" and n.distance_from_start_mm == 0)]
    _only(verify_pattern(clone), "N2")


def test_mutation_tilts_grainline(blouse_m: Pattern) -> None:
    """Tilting the front grainline breaks exactly R1."""
    clone = copy.deepcopy(blouse_m)
    a, b = clone.panels[0].grainline
    clone.panels[0].grainline = (Point(a.x + 5, a.y), b)
    _only(verify_pattern(clone), "R1")


def test_report_summary_and_ids(blouse_m: Pattern) -> None:
    """Summaries speak Indonesian; failures name their checks."""
    good = verify_pattern(blouse_m)
    assert good.passed and good.summary_id().startswith("LULUS")
    clone = copy.deepcopy(blouse_m)
    clone.params["x"] = 1
    bad = verify_pattern(clone)
    assert not bad.passed and bad.summary_id().startswith("GAGAL")
    assert bad.failures()[0].id == "G2"
