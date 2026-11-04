"""Web UI tests via the in-process app; pack math stays deterministic."""

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in tests: the backend is always unreachable here."""

    def unreachable(*args, **kwargs):
        raise llm_module.LLMError("offline")

    monkeypatch.setattr(llm_module, "call_json", unreachable)


def test_index_serves_single_page() -> None:
    """GET / returns the stepped studio page, design step first."""
    response = client.get("/")
    assert response.status_code == 200
    for step in ("desain", "sempurnakan", "ukuran", "bukti", "cek", "produksi"):
        assert f'data-step="{step}"' in response.text
    assert "/api/design" in response.text


def test_fields_lists_garment_options() -> None:
    """Field, size and fabric options follow the garment."""
    data = client.get("/api/fields", params={"garment": "blouse"}).json()
    assert "bust" in data["fields"] and data["sizes"] == ["S", "M", "L", "XL"]
    assert client.get("/api/fields", params={"garment": "naga"}).status_code == 422


def test_fields_size_values_are_cm_of_size_table() -> None:
    """Prefill values are the size table converted mm -> cm, nothing invented."""
    from seamproof.measurements import WOMEN_SIZE_TABLE

    data = client.get("/api/fields", params={"garment": "blouse"}).json()
    assert data["size_values"]["M"]["bust"] == round(WOMEN_SIZE_TABLE["M"].bust / 10, 1)
    assert set(data["size_values"]["M"]) == set(data["fields"])


def test_pack_size_mode_returns_verified_downloads() -> None:
    """A passing pack yields SVG, proof table and a downloadable PDF."""
    pack = client.post("/api/pack", json={"garment": "blouse", "fabric": "woven",
                                          "mode": "size", "size": "M"}).json()
    assert pack["passed"] and pack["job"]
    assert any(c["id"] == "S3" and c["passed"] for c in pack["checks"])
    assert pack["explanation"]["summary"].startswith("All")
    pdf = client.get(pack["files"]["pdf"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    md = client.get(pack["files"]["md"])
    assert "Tailor explanation" in md.text


def test_pack_failing_pattern_has_no_pdf() -> None:
    """A failing pack still downloads JSON/md, but no PDF."""
    pack = client.post("/api/pack", json={"garment": "tshirt", "fabric": "knit_low_stretch",
                                          "mode": "size", "size": "M"}).json()
    assert not pack["passed"]
    assert pack["files"]["pdf"] is None
    assert client.get(pack["files"]["json"]).status_code == 200


def test_pack_measures_mode_validates_missing() -> None:
    """Incomplete manual measurements are reported, never invented."""
    response = client.post("/api/pack", json={"garment": "blouse", "fabric": "woven",
                                              "mode": "measures", "unit": "cm",
                                              "measures": {"bust": 92}})
    assert response.status_code == 422
    assert "waist" in response.json()["detail"]["missing"]


def test_production_endpoint_totals_run() -> None:
    """The production endpoint totals metres, cost and markers."""
    pack = client.post("/api/production", json={"garment": "blouse", "fabric": "woven",
                                                "run": {"S": 20, "M": 50},
                                                "width": 1500, "price": 45000,
                                                "layers": 1}).json()
    assert abs(pack["total_cost"] - pack["total_metres"] * 45000) < 0.01
    assert 0 < pack["overall_utilisation"] <= 1
    assert set(pack["markers"]) == {"S", "M"}
    assert "Rp" in pack["md"]
    assert "<polyline" in pack["markers"]["M"] and 'class="cut"' in pack["markers"]["M"]
    links = pack["files"]["markers"]["M"]
    for kind in ("plot", "a4"):
        got = client.get(links[kind])
        assert got.status_code == 200 and got.content.startswith(b"%PDF")
    assert client.get(links["svg"]).text.startswith("<svg")
    assert client.get(pack["files"]["pdf"]).content.startswith(b"%PDF")
    assert "Rp" in client.get(pack["files"]["md"]).text
    assert pack["sizes"][0]["a4_sheets"] > 1
    assert client.get(f"/api/production/{pack['job']}/marker/XL/plot").status_code == 404
    assert client.get("/api/production/nope/pdf").status_code == 404
    bad = client.post("/api/production", json={"garment": "blouse", "fabric": "woven",
                                               "run": {"XXL": 5}, "width": 1500,
                                               "price": 45000})
    assert bad.status_code == 422


def test_unverified_size_marker_is_409(monkeypatch: pytest.MonkeyPatch) -> None:
    """A size that failed verification gets no 1:1 marker link and a 409 on request."""
    import seamproof.production as production_module

    class Failing:
        passed = False

    monkeypatch.setattr(production_module, "verify_pattern", lambda pattern: Failing())
    pack = client.post("/api/production", json={"garment": "blouse", "fabric": "woven",
                                                "run": {"M": 5}, "width": 1500}).json()
    links = pack["files"]["markers"]["M"]
    assert links["plot"] is None and links["a4"] is None
    assert client.get(f"/api/production/{pack['job']}/marker/M/plot").status_code == 409


def test_photo_rejects_non_image() -> None:
    """Undecodable uploads are a 422 before any model call."""
    response = client.post("/api/photo", files={"file": ("flat.jpg", b"\xff\xd8fake", "image/jpeg")})
    assert response.status_code == 422


@pytest.mark.parametrize("body", [
    {"garment": "blouse", "run": {"M": "abc"}},
    {"garment": "blouse", "run": ["M", 5]},
    {"garment": "blouse", "run": {"M": 5}, "width": None},
    {"garment": "blouse", "run": {"M": 5}, "price": "1e999"},
    {"garment": "blouse", "run": {"M": 5}, "layers": 2.7},
    {"garment": "blouse", "run": {"M=1,S": 5}},
    {"garment": "blouse", "run": {}},
])
def test_production_bad_input_is_422(body) -> None:
    """Malformed production payloads are refused cleanly, never a 500."""
    assert client.post("/api/production", json=body).status_code == 422


def test_pack_measures_must_be_object() -> None:
    """A list instead of {field: value} is a 422."""
    response = client.post("/api/pack", json={"garment": "blouse", "mode": "measures",
                                              "measures": ["bust", 92]})
    assert response.status_code == 422


def test_pack_undrawable_measurements_is_422() -> None:
    """In-range values the draft refuses return 422 instead of crashing."""
    from dataclasses import asdict

    from seamproof.measurements import WOMEN_SIZE_TABLE

    measures = {k: v / 10 for k, v in asdict(WOMEN_SIZE_TABLE["M"]).items()}
    measures["waist"] = 95.0
    response = client.post("/api/pack", json={"garment": "blouse", "mode": "measures",
                                              "unit": "cm", "measures": measures})
    assert response.status_code == 422


def test_dxf_locked_until_pattern_maker_signs_off() -> None:
    """DXF needs a sign-off; the sign-off is printed in report.md and unlocks DXF."""
    import io

    import ezdxf

    pack = client.post("/api/pack", json={"garment": "skirt", "fabric": "woven",
                                          "mode": "size", "size": "M"}).json()
    assert pack["passed"] and pack["files"]["dxf"] is None
    assert client.get(f"/api/download/{pack['job']}/dxf").status_code == 409
    assert client.post(f"/api/check/{pack['job']}", json={"name": "x"}).status_code == 422
    done = client.post(f"/api/check/{pack['job']}", json={"name": "Bu Sari", "note": "toile oke"}).json()
    assert done["signoff"]["name"] == "Bu Sari" and done["files"]["dxf"]
    dxf = client.get(done["files"]["dxf"])
    doc = ezdxf.read(io.StringIO(dxf.text))
    assert doc.header["$INSUNITS"] == 4
    assert {"CUT", "STITCH", "NOTCH", "GRAIN"} <= {e.dxf.layer for e in doc.modelspace()}
    assert "Bu Sari" in client.get(done["files"]["md"]).text


def test_failing_pack_cannot_be_signed_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern that fails verification is refused at the check gate."""
    import seamproof.server as server_module
    from seamproof.verify import Report

    pack = client.post("/api/pack", json={"garment": "skirt", "mode": "size", "size": "M"}).json()
    server_module.JOBS[pack["job"]]["report"] = Report("skirt", "M", "woven", False, [])
    assert client.post(f"/api/check/{pack['job']}", json={"name": "Bu Sari"}).status_code == 409
