"""Repair-loop tests with a scripted stand-in LLM; no network calls."""

import json
import sys
from dataclasses import replace

import pytest

import seamproof.llm as llm_module
from seamproof.cli import main
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.params import PARAMETERS
from seamproof.repair import INJECTS, repair_loop, rounds_table_md

M = WOMEN_SIZE_TABLE["M"]


def _script(answers):
    """Return an llm_fn serving canned proposals in order, repeating the last."""
    calls = []

    def fn(system, user):
        calls.append((system, user))
        return answers[min(len(calls) - 1, len(answers) - 1)]

    fn.calls = calls
    return fn


def test_stops_when_passing() -> None:
    """A clean draft returns with no rounds and no LLM call."""

    def fail_if_called(system, user):
        raise AssertionError("LLM must not be called")

    pattern, report, logs = repair_loop(M, "woven", {}, llm_fn=fail_if_called)
    assert report.passed and logs == []
    assert pattern.metadata["status"] == "draft_unverified"


def test_repairs_injected_defect() -> None:
    """One good proposal fixes the injected cap and passes locality."""
    llm = _script([{"changes": {"sleeve_cap_height": 100.0}, "reason": "turunkan puncak"}])
    pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]), llm_fn=llm)
    assert report.passed and len(logs) == 1
    assert logs[0].decision == "accepted" and logs[0].locality_passed is True
    assert pattern.params["sleeve_cap_height"] == 100.0
    assert "sleeve_cap_height=100.0" in rounds_table_md(logs)


def test_rejects_out_of_range_then_recovers() -> None:
    """999 is rejected for range; the next proposal is accepted."""
    llm = _script([{"changes": {"sleeve_cap_height": 999.0}, "reason": "terlalu tinggi"},
                   {"changes": {"sleeve_cap_height": 100.0}, "reason": "betulkan"}])
    _pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=llm, max_rounds=3)
    assert report.passed and len(logs) == 2
    assert logs[0].decision == "rejected" and "range" in logs[0].reason
    assert logs[0].locality_passed is None
    assert logs[1].decision == "accepted"


def test_rejects_unknown_param() -> None:
    """Proposing outside the registry is rejected without redrafting."""
    llm = _script([{"changes": {"sayap_naga": 1.0}, "reason": "ngawur"},
                   {"changes": {"sleeve_cap_height": 100.0}, "reason": "betulkan"}])
    _pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=llm, max_rounds=3)
    assert report.passed
    assert logs[0].decision == "rejected" and "unknown" in logs[0].reason


def test_rejects_malformed_proposal() -> None:
    """A payload without changes/reason is rejected, not crashed on."""
    llm = _script([{"changes": {}, "reason": "kosong"},
                   {"changes": {"sleeve_cap_height": 100.0}, "reason": "betulkan"}])
    _pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=llm, max_rounds=3)
    assert report.passed and logs[0].decision == "rejected"


def test_rejects_locality_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A correct-looking value is rejected when its code moves undeclared edges."""
    spec = PARAMETERS["sleeve_cap_height"]
    monkeypatch.setitem(PARAMETERS, "sleeve_cap_height", replace(spec, affects=[]))
    llm = _script([{"changes": {"sleeve_cap_height": 100.0}, "reason": "coba"}])
    _pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=llm, max_rounds=2)
    assert not report.passed and len(logs) == 2
    assert all(log.decision == "rejected" for log in logs)
    assert all(log.locality_passed is False for log in logs)
    assert "declaration" in logs[0].reason


def test_respects_max_rounds() -> None:
    """Endless bad proposals stop at max_rounds with the draft intact."""
    llm = _script([{"changes": {"sleeve_cap_height": 999.0}, "reason": "ngotot"}])
    pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                        llm_fn=llm, max_rounds=3)
    assert len(logs) == 3 and not report.passed
    assert pattern.params["sleeve_cap_height"] == 160.0
    assert [c.id for c in report.failures()] == ["S3"]


def test_llm_error_becomes_rejected_round() -> None:
    """A crashing LLM consumes a round instead of aborting the loop."""
    calls = []

    def flaky(system, user):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("jaringan putus")
        return {"changes": {"sleeve_cap_height": 100.0}, "reason": "coba lagi"}

    _pattern, report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=flaky, max_rounds=3)
    assert report.passed and logs[0].decision == "rejected" and "LLM failed" in logs[0].reason


def test_cli_repair_with_mocked_llm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The repair command writes the pack and exits zero on a scripted fix."""
    def fake_call_json(system, user, schema, purpose):
        assert "sleeve_cap_height" in user
        return {"changes": {"sleeve_cap_height": 100.0}, "reason": "demo"}

    monkeypatch.setattr(llm_module, "call_json", fake_call_json)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "repair", "--size", "M", "--fabric", "woven",
                                      "--inject", "sleeve_cap_too_high", "--out", str(out)])
    assert main() == 0
    for name in ("pattern.json", "pattern.svg", "pattern.pdf", "report.md", "rounds.md"):
        assert (out / name).is_file()
    assert "sleeve_cap_height=100.0" in (out / "rounds.md").read_text()


def test_accepted_change_survives_later_rounds() -> None:
    """Round 2 builds on round 1: an accepted value is not silently reverted."""
    llm = _script([{"changes": {"armhole_curve_depth": 22.5}, "reason": "pertama"},
                   {"changes": {"bust_dart_clearance": 22.5}, "reason": "kedua"}])
    pattern, _report, logs = repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]),
                                         llm_fn=llm, max_rounds=2)
    assert [log.decision for log in logs] == ["accepted", "accepted"]
    assert pattern.params["armhole_curve_depth"] == 22.5
    assert pattern.params["bust_dart_clearance"] == 22.5


def test_model_sees_why_earlier_rounds_were_rejected() -> None:
    """Each round's payload carries the earlier proposals and their gate verdicts."""
    llm = _script([{"changes": {"sleeve_cap_height": 999.0}, "reason": "ngotot"},
                   {"changes": {"sleeve_cap_height": 100.0}, "reason": "betulkan"}])
    repair_loop(M, "woven", dict(INJECTS["sleeve_cap_too_high"]), llm_fn=llm, max_rounds=2)
    first, second = (json.loads(user) for _system, user in llm.calls)
    assert first["history"] == []
    earlier = second["history"][0]
    assert earlier["proposed"] == {"sleeve_cap_height": 999.0}
    assert earlier["decision"] == "rejected" and "range" in earlier["reason"]


def _bad_cap():
    """Return the injected sleeve-cap defect, which fails only S3."""
    return dict(INJECTS["sleeve_cap_too_high"])


def test_try_params_does_not_save_a_partial_fix() -> None:
    """A dry run reports what would remain and leaves the pattern alone."""
    llm = _script([{"action": "try_params", "changes": {"bust_dart_angle": -15.0}, "reason": "coba"}])
    pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=1)
    assert logs[0].decision == "tried" and "S2a" in logs[0].outcome
    assert {c["id"] for c in logs[0].checks} == {"S2a", "S3", "D2", "D3"}
    assert pattern.params["bust_dart_angle"] != -15.0
    assert [c.id for c in report.failures()] == ["S3"]


def test_try_params_result_reaches_the_next_step() -> None:
    """The verifier's numbers for a dry run are in the next step's history."""
    llm = _script([{"action": "try_params", "changes": {"sleeve_cap_height": 150.0}, "reason": "coba"},
                   {"action": "try_params", "changes": {"sleeve_cap_height": 100.0}, "reason": "lagi"}])
    repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=2)
    first, second = (json.loads(user) for _system, user in llm.calls)
    assert (first["steps_left"], second["steps_left"]) == (2, 1)
    assert second["history"][0]["checks"][0]["id"] == "S3"
    assert second["history"][0]["checks"][0]["actual"] is not None


def test_try_params_that_passes_is_saved() -> None:
    """A dry run passing every check ends the repair with that pattern."""
    llm = _script([{"action": "try_params", "changes": {"sleeve_cap_height": 100.0}, "reason": "coba"}])
    pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm)
    assert report.passed and len(logs) == 1 and logs[0].decision == "accepted"
    assert pattern.params["sleeve_cap_height"] == 100.0


def test_apply_params_refuses_new_failed_checks() -> None:
    """A commit adding failed checks is rejected; the pattern stays as it was."""
    llm = _script([{"action": "apply_params", "changes": {"bust_dart_angle": -15.0}, "reason": "paksa"}])
    pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=1)
    assert logs[0].decision == "rejected" and "S2a" in logs[0].reason
    assert pattern.params["bust_dart_angle"] != -15.0
    assert [c.id for c in report.failures()] == ["S3"]


def test_try_params_goes_through_the_gates() -> None:
    """Dry runs are range-checked like commits."""
    llm = _script([{"action": "try_params", "changes": {"sleeve_cap_height": 999.0}, "reason": "coba"}])
    _pattern, _report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=1)
    assert logs[0].decision == "rejected" and "range" in logs[0].reason


def test_stop_ends_early_and_cannot_claim_success() -> None:
    """Stop spends no more steps; the verifier's failure stands whatever the reason says."""
    llm = _script([{"action": "stop", "reason": "semua sudah lulus"}])
    _pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=5)
    assert len(llm.calls) == 1 and len(logs) == 1
    assert logs[0].decision == "stopped" and logs[0].outcome == "remaining: S3"
    assert not report.passed


def test_unknown_action_is_rejected() -> None:
    """An invented action consumes a step and changes nothing."""
    llm = _script([{"action": "edit_geometry", "changes": {"sleeve_cap_height": 100.0}, "reason": "x"}])
    _pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=llm, max_rounds=1)
    assert logs[0].decision == "rejected" and "edit_geometry" in logs[0].reason
    assert not report.passed


@pytest.mark.parametrize("raw", [{"changes": 5, "reason": "x"}, ["bukan", "objek"], "teks"])
def test_malformed_replies_do_not_crash(raw) -> None:
    """Odd reply shapes become rejected steps."""
    _pattern, report, logs = repair_loop(M, "woven", _bad_cap(), llm_fn=_script([raw]), max_rounds=1)
    assert logs[0].decision == "rejected" and logs[0].proposed == {} and not report.passed
