"""AI reference sketch: sanitising model SVG and the endpoint; no network calls."""

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.server import app
from seamproof.vision import PreparedPhoto, merge_sketch, sanitize_sketch, sketch_from_photo

GOOD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500"><g><path d="M0 0L9 9" stroke="black"/></g></svg>'


def test_sanitize_keeps_the_drawing() -> None:
    """Plain drawing elements and attributes survive."""
    out = sanitize_sketch(GOOD)
    assert 'd="M0 0L9 9"' in out and 'viewBox="0 0 800 500"' in out


def test_sanitize_strips_active_content() -> None:
    """Scripts, handlers, links, foreign content and url() references are removed."""
    out = sanitize_sketch('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script>'
                          '<foreignObject><p>x</p></foreignObject><image href="http://x/y.png"/>'
                          '<path d="M0 0L9 9" onclick="alert(1)" fill="url(http://x/#a)" stroke="black"/></svg>')
    for banned in ("script", "foreignObject", "image", "onclick", "url(", "http://x"):
        assert banned not in out
    assert 'stroke="black"' in out


@pytest.mark.parametrize("bad", ["not xml", "<html/>", '<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>',
                                 '<!DOCTYPE svg [<!ENTITY a "a">]><svg><path d="M0 0"/></svg>',
                                 "<svg>" + "<path/>" * 40000 + "</svg>"])
def test_sanitize_rejects_unusable_svg(bad: str) -> None:
    """Broken, empty, entity-bearing or oversized replies are refused."""
    with pytest.raises(ValueError):
        sanitize_sketch(bad)


def test_sketch_from_photo_sends_the_prepared_photo() -> None:
    """The model receives the prepared JPEG as a data URL."""
    seen = {}

    def fake(system, text, urls, schema, purpose):
        seen.update(urls=urls, purpose=purpose)
        return {"svg": GOOD}

    assert "path" in sketch_from_photo(PreparedPhoto(b"jpg", 1, 1, 0), vision_fn=fake)
    assert seen["urls"] == ["data:image/jpeg;base64,anBn"] and seen["purpose"] == "photo_sketch"


def _jpeg() -> bytes:
    """Return a small JPEG, or skip without OpenCV."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    return cv2.imencode(".jpg", numpy.full((60, 60, 3), 200, dtype=numpy.uint8))[1].tobytes()


def test_endpoint_returns_clean_svg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route blurs, calls the model and returns sanitised SVG."""
    reply = {"svg": GOOD.replace("<g>", "<script>x</script><g>")}
    monkeypatch.setattr(llm_module, "call_vision_json", lambda *args: reply)
    response = TestClient(app).post("/api/photo/sketch", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == 200
    assert "path" in response.json()["svg"] and "script" not in response.json()["svg"]


@pytest.mark.parametrize("error,status", [(llm_module.LLMError("down"), 502), (None, 422)])
def test_endpoint_reports_failures(monkeypatch: pytest.MonkeyPatch, error, status: int) -> None:
    """A model failure is 502 and an unusable drawing is 422, never 500."""
    def fake(*args):
        if error:
            raise error
        return {"svg": "not xml"}

    monkeypatch.setattr(llm_module, "call_vision_json", fake)
    response = TestClient(app).post("/api/photo/sketch", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert response.status_code == status


def test_merge_sketch_sends_all_photos_and_each_part_source() -> None:
    """The model gets every photo plus which image gives which part."""
    seen = {}

    def fake(system, text, urls, schema, purpose):
        seen.update(text=text, urls=urls, purpose=purpose)
        return {"svg": GOOD}

    photos = [PreparedPhoto(b"one", 1, 1, 0), PreparedPhoto(b"two", 1, 1, 0)]
    assert "path" in merge_sketch(photos, "Shirt", [("Collar", 1), ("Sleeves", 2)], 1, "add a pocket", vision_fn=fake)
    assert len(seen["urls"]) == 2 and seen["purpose"] == "photo_sketch_merge"
    for wanted in ("Shirt", "Collar: from image 1", "Sleeves: from image 2", "follows image 1", "add a pocket"):
        assert wanted in seen["text"]


@pytest.mark.parametrize("parts,base", [([("Collar", 3)], 1), ([("Collar", 0)], 1), ([("Collar", 1)], 2)])
def test_merge_sketch_rejects_unknown_photo_numbers(parts, base) -> None:
    """A part pointing at a photo that was not sent never reaches the model."""
    def fail(*args):
        raise AssertionError("model must not be called")

    with pytest.raises(ValueError):
        merge_sketch([PreparedPhoto(b"one", 1, 1, 0)], "Shirt", parts, base, vision_fn=fail)


def test_merge_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route accepts several photos; bad part lists are 422."""
    monkeypatch.setattr(llm_module, "call_vision_json", lambda *args: {"svg": GOOD})
    client = TestClient(app)
    photo = ("files", ("a.jpg", _jpeg(), "image/jpeg"))
    form = {"garment": "Shirt", "base": "1", "parts": '[{"label": "Collar", "photo": 2}]'}
    response = client.post("/api/photo/sketch-merge", files=[photo, photo], data=form)
    assert response.status_code == 200 and "path" in response.json()["svg"]
    for bad in ("not json", "[]", '[{"label": "Collar"}]', '[{"label": "Collar", "photo": 9}]'):
        response = client.post("/api/photo/sketch-merge", files=[photo], data={**form, "parts": bad})
        assert response.status_code == 422, bad
