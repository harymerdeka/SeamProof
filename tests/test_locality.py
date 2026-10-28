"""Edit-locality tests: declared moves pass, leaks and strangers fail."""

import copy
from dataclasses import replace

import pytest

from seamproof.draft.blouse import draft_blouse
from seamproof.locality import MOVE_TOL_MM, edit_locality
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.params import PARAMETERS
from seamproof.pattern import Curve, Point


@pytest.fixture
def pair():
    """Return identical before/after drafts of the M blouse."""
    m = WOMEN_SIZE_TABLE["M"]
    return draft_blouse(m, "woven"), draft_blouse(m, "woven")


def test_identical_patterns_pass(pair) -> None:
    """No changes means locality trivially holds."""
    before, after = pair
    result = edit_locality(before, after, {"sleeve_cap_height": 100.0})
    assert result.passed and result.moved == []


def test_declared_param_change_passes() -> None:
    """A real clearance change moves only declared edges."""
    m = WOMEN_SIZE_TABLE["M"]
    before = draft_blouse(m, "woven")
    after = draft_blouse(m, "woven", {"bust_dart_clearance": 25.0})
    result = edit_locality(before, after, {"bust_dart_clearance": 25.0})
    assert result.passed
    allowed = set(PARAMETERS["bust_dart_clearance"].affects)
    assert all((panel, edge) in allowed for panel, edge, _dist in result.moved)
    assert result.moved, "the change must move something"


def test_submillimetre_drift_passes(pair) -> None:
    """A 0.3 mm nudge under the threshold is not a move."""
    before, after = pair
    fold = next(e for e in after.panels[0].edges if e.id == "front.fold")
    fold.curve = Curve(tuple(Point(p.x + 0.3, p.y) for p in fold.curve.points))
    assert edit_locality(before, after, {"sleeve_cap_height": 100.0}).passed


def test_larger_drift_fails(pair) -> None:
    """A 0.6 mm nudge on an undeclared edge names that edge."""
    before, after = pair
    fold = next(e for e in after.panels[0].edges if e.id == "front.fold")
    fold.curve = Curve(tuple(Point(p.x + 0.6, p.y) for p in fold.curve.points))
    result = edit_locality(before, after, {"sleeve_cap_height": 100.0})
    assert not result.passed
    assert ("front", "front.fold", pytest.approx(0.6)) in result.moved
    assert "front.fold" in result.message


def test_unknown_param_fails(pair) -> None:
    """Parameters outside the registry are rejected outright."""
    before, after = pair
    result = edit_locality(before, after, {"袖子": 1.0})
    assert not result.passed


def test_leaking_implementation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """A param whose code moves an undeclared edge must be caught.

    Simulates the leak by shrinking the declared affects: raising the cap
    still moves the back underarm, which the narrowed declaration forbids.
    """
    m = WOMEN_SIZE_TABLE["M"]
    before = draft_blouse(m, "woven")
    after = draft_blouse(m, "woven", {"sleeve_cap_height": 120.0})
    spec = PARAMETERS["sleeve_cap_height"]
    narrowed = [e for e in spec.affects if e != ("sleeve", "sleeve.underarm_back")]
    assert len(narrowed) < len(spec.affects)
    monkeypatch.setitem(PARAMETERS, "sleeve_cap_height", replace(spec, affects=narrowed))
    result = edit_locality(before, after, {"sleeve_cap_height": 120.0})
    assert not result.passed
    assert ("sleeve", "sleeve.underarm_back") in [(p, e) for p, e, _d in result.moved]
    assert MOVE_TOL_MM == 0.5
    assert copy.deepcopy(before).to_dict() == before.to_dict()
