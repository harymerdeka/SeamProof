"""Web endpoints that add Muse to each step; every model call is mocked (no network)."""

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.server import app

client = TestClient(app)


def _jpeg() -> bytes:
    """A small valid JPEG with no face."""
    import cv2
    import numpy as np

    ok, data = cv2.imencode(".jpg", np.full((60, 80, 3), 200, dtype=np.uint8))
    assert ok
    return bytes(data)


@pytest.fixture()
def muse(monkeypatch: pytest.MonkeyPatch):
    """Scripted Muse Spark: answers by purpose; records every call."""
    calls = []

    def fake(system, user, schema, purpose):
        calls.append(purpose)
        if purpose.startswith("fabric"):
            return {"fabric": "woven", "reason": "Tenun menahan bentuk dan mudah dijahit."}
        if purpose.startswith("story"):
            return {"summary": "Semua pemeriksaan lulus.", "sewing_steps": ["Jahit kupnat badan depan.", "Sambung bahu."],
                    "toile_checks": ["Lihat letak ujung kupnat."]}
        if purpose.startswith("production"):
            return {"cutting_steps": ["Gelar kain lalu potong sesuai marker."], "owner_summary": "Semua ukuran lulus."}
        raise llm_module.LLMError("unexpected purpose")

    monkeypatch.setattr(llm_module, "call_json", fake)
    return calls


def _pack(garment: str = "blouse") -> dict:
    """A verified M pack."""
    response = client.post("/api/pack", json={"garment": garment, "fabric": "woven", "mode": "size", "size": "M"})
    assert response.status_code == 200 and response.json()["passed"]
    return response.json()


def test_fabric_suggest_picks_a_verified_candidate(muse) -> None:
    """Code lists fabrics that verify; Muse picks one of them and gives a reason."""
    body = client.post("/api/fabric-suggest", json={"garment": "dress", "brief": "dress pesta"}).json()
    assert body["fabric"] in body["candidates"] and body["source"] == "muse"
    assert "woven" in body["candidates"] and muse == ["fabric-dress"]


def test_fabric_suggest_falls_back_when_muse_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the model the first verified candidate is returned, marked as template."""
    monkeypatch.setattr(llm_module, "call_json", lambda *a: (_ for _ in ()).throw(llm_module.LLMError("off")))
    body = client.post("/api/fabric-suggest", json={"garment": "tshirt"}).json()
    assert body["source"] == "template" and body["fabric"] == body["candidates"][0]


def test_measure_photo_fills_fields_in_cm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wonder transcribes a note; the answer comes back in cm after intake range checks."""
    monkeypatch.setattr(llm_module, "call_vision_json", lambda *a: {"measurements": [
        {"field": "bust", "value": 92, "unit": "cm"}, {"field": "hip", "value": 5, "unit": "cm"}]})
    body = client.post("/api/measure-photo", data={"garment": "blouse"},
                       files={"file": ("catatan.jpg", _jpeg(), "image/jpeg")}).json()
    assert body["measurements_cm"] == {"bust": 92.0}
    assert "hip" in body["missing"] and body["warnings"]


def test_pattern_story_is_written_once_per_pack(muse) -> None:
    """The story endpoint calls Muse once and caches; unknown jobs are 404."""
    job = _pack()["job"]
    first = client.get(f"/api/story/{job}").json()
    second = client.get(f"/api/story/{job}").json()
    assert first == second and first["source"] == "muse" and len(first["sewing_steps"]) == 2
    assert muse.count("story-blouse") == 1
    assert client.get("/api/story/nope").status_code == 404


def test_production_story_uses_muse(muse) -> None:
    """Production notes come from Muse when its numbers check out."""
    run = client.post("/api/production", json={"garment": "blouse", "fabric": "woven", "run": {"M": 5}}).json()
    story = client.get(f"/api/production-story/{run['job']}").json()
    assert story["source"] == "muse" and story["cutting_steps"]
    assert client.get("/api/production-story/nope").status_code == 404


def test_toile_notes_make_a_verified_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed toile notes become rule changes; v2 is redrafted, verified and stored as a new pack."""
    monkeypatch.setattr(llm_module, "call_json", lambda *a: (_ for _ in ()).throw(llm_module.LLMError("off")))
    job = _pack()["job"]
    body = client.post(f"/api/toile/{job}", data={"notes": "ketiak menarik"}).json()
    assert [c["param"] for c in body["changes"]] == ["armhole_curve_depth"]
    assert body["locality"] and body["verified"]
    assert body["pack"]["job"] != job and body["pack"]["passed"]
    assert client.post(f"/api/toile/{job}", data={"notes": ""}).status_code == 422


def test_toile_photo_goes_through_wonder(monkeypatch: pytest.MonkeyPatch) -> None:
    """A toile photo is face-blurred, then the (mocked) vision step names the issue."""
    import seamproof.fitcorrect as fit_module
    from seamproof.fitcorrect import Issue

    seen = []
    monkeypatch.setattr(fit_module, "_vision_issues",
                        lambda images: seen.extend(images) or [Issue("neckline_gapes", "depan", 2, "menganga")])
    job = _pack()["job"]
    body = client.post(f"/api/toile/{job}", files=[("files", ("toile.jpg", _jpeg(), "image/jpeg"))]).json()
    assert seen and [c["param"] for c in body["changes"]] == ["neckline_depth_front"]
    assert body["issues"][0]["name"]


def test_toile_notes_are_read_by_muse_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Muse maps free notes to rule keys; the rule table still sizes the change."""
    def fake(system, user, schema, purpose):
        if purpose == "toile-notes":
            assert "ketiaknya ketarik" in user
            return {"issues": [{"issue": "pulling_at_underarm", "location": "ketiak", "severity": 3,
                                "evidence": "ketiaknya ketarik banget"}]}
        raise llm_module.LLMError("off")

    monkeypatch.setattr(llm_module, "call_json", fake)
    job = _pack()["job"]
    body = client.post(f"/api/toile/{job}", data={"notes": "ketiaknya ketarik banget"}).json()
    assert body["read_by"] == {"photos": "", "notes": "muse"}
    assert body["issues"][0]["severity"] == 3 and [c["param"] for c in body["changes"]] == ["armhole_curve_depth"]


def test_toile_notes_fall_back_to_keywords() -> None:
    """Without Muse the deterministic keyword table reads the notes."""
    from seamproof.fitcorrect import notes_to_issues_muse

    issues, source = notes_to_issues_muse(["leher menganga sedikit"], llm_fn=lambda s, u: {"issues": "bad"})
    assert source == "keywords" and issues[0].key == "neckline_gapes" and issues[0].severity == 1


def test_long_design_brief_is_accepted_everywhere(muse) -> None:
    """A long brief (up to the design cap) also works for the fabric suggestion."""
    brief = "dress kerja wanita, kain adem, warna kalem. " * 40
    assert 600 < len(brief) <= 2000
    response = client.post("/api/fabric-suggest", json={"garment": "dress", "brief": brief})
    assert response.status_code == 200 and response.json()["source"] == "muse"
