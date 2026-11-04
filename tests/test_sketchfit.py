"""Sketch proportion fit: silhouette reading, the gated loop, API and previews; no network calls."""

import pytest
from fastapi.testclient import TestClient

from seamproof.draft import REGISTRY
from seamproof.flat import SKIRT_HEM_HALF, plan_flat, view_features
from seamproof.locality import edit_locality
from seamproof.measurements import WOMEN_SIZE_TABLE
from seamproof.preview3d import plan_preview
from seamproof.server import app
from seamproof.sketchfit import (
    SketchUnreadable,
    fit_to_sketch,
    half_width,
    match_score,
    pattern_hem,
    pattern_silhouette,
    read_silhouette,
    read_target,
)
from seamproof.style import apply_proportions, plan_piece
from seamproof.verify import verify_pattern

M = WOMEN_SIZE_TABLE["M"]
SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">{}</svg>'
SKIRT = {"silhouette": "a_line", "length": "knee", "hem": "straight", "waistband": False}
DRESS = {"silhouette": "a_line", "length": "knee", "hem": "straight", "collar": "none", "opening": "none",
         "sleeve_length": "short", "shoulder": "set_in", "fit": "regular", "neckline": "round"}


def outline(half_widths: list[tuple[float, float]], cx: float = 200.0, extra: str = "") -> str:
    """A closed symmetrical sketch outline with Bezier-free edges, front at cx, back mirrored at x+400."""
    right = [(cx + w, y) for w, y in half_widths]
    left = [(cx - w, y) for w, y in reversed(half_widths)]
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in right + left) + " Z"
    return (f'<path d="{d}" fill="white" stroke="black"/>'
            f'<g transform="translate(400,0)"><path d="{d}" fill="white"/></g>' + extra)


def sketch_of(garment: str, params: dict[str, float], scale: float = 0.4) -> str:
    """Draw the sketch a pattern would produce, so the fit has a known right answer."""
    pattern = REGISTRY[garment](M, "woven", params)
    shape = pattern_silhouette(pattern)
    depth = shape.bounds[3]
    rows = [(half_width(shape, 0.0, depth * i / 40 * 0.9999 + 1e-3) * scale, 20 + depth * i / 40 * scale) for i in range(41)]
    return SVG.format(outline(rows))


def test_reads_front_view_and_ignores_back_and_separate_sleeves() -> None:
    """Back view and a sleeve beside the body are outside the silhouette."""
    sleeve = '<path d="M250 20 L330 20 L330 200 L250 200 Z" fill="white"/><path d="M150 20 L70 20 L70 200 L150 200 Z" fill="white"/>'
    silhouette = read_silhouette(SVG.format(outline([(50, 20), (100, 400)], extra=sleeve)))
    assert silhouette.cx == pytest.approx(200) and silhouette.top == 20 and silhouette.bottom == 400
    assert half_width(silhouette.shape, silhouette.cx, 399.9) == pytest.approx(100, abs=0.1)


def test_reads_curves_relative_commands_and_transforms() -> None:
    """Relative commands, Beziers and a scaled group produce real coordinates."""
    body = ('<g transform="scale(2)"><path d="m50 10 h100 c0 50 25 100 25 180 s-100 0 -150 0 q0 -90 25 -180 z" '
            'fill="white"/></g>')
    silhouette = read_silhouette('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">' + body + "</svg>")
    assert silhouette.top == pytest.approx(20)
    # Worked by hand: the reflected S control bulges the seam to y=225.56, doubled by the scale.
    assert silhouette.bottom == pytest.approx(451.1, abs=0.5)


@pytest.mark.parametrize("bad", [
    SVG.format('<path d="M0 0 L9 9" stroke="black"/>'),  # no closed shape
    SVG.format('<rect x="0" y="0" width="800" height="500" fill="white"/>'),  # only a background
    SVG.format('<path d="M 10 10 L 20 oops" fill="white"/>'),
    SVG.format('<path d="M190 200 L210 200 L210 210 L190 210 Z" fill="white"/>'),  # tiny
    "not xml",
])
def test_unusable_sketches_are_refused(bad: str) -> None:
    """A sketch with no measurable outline raises instead of inventing numbers."""
    with pytest.raises(SketchUnreadable):
        read_silhouette(bad)


@pytest.mark.parametrize("garment,base,wanted", [
    ("skirt", {"skirt_flare": 80.0, "skirt_length": 580.0, "waistband_height": 0.0}, {"skirt_flare": 150.0, "skirt_length": 700.0}),
    ("skirt", {"skirt_flare": 180.0, "skirt_length": 950.0, "waistband_height": 0.0}, {"skirt_flare": 20.0, "skirt_length": 450.0}),
    ("dress", {"dress_waist_seam": 1.0, "skirt_flare": 80.0, "skirt_length": 580.0},
     {"dress_waist_seam": 1.0, "skirt_flare": 140.0, "skirt_length": 680.0}),
])
def test_fit_recovers_known_proportions(garment: str, base: dict, wanted: dict) -> None:
    """A sketch made from known parameters is fitted back to them, and the pattern passes."""
    if garment == "dress":  # the dress sketch starts at the shoulder-neck point
        pattern = REGISTRY[garment](M, "woven", wanted)
        shape = pattern_silhouette(pattern)
        depth, scale = shape.bounds[3], 0.25
        top = 10.0
        waist = top + M.front_length * scale
        rows = [(half_width(shape, 0.0, 0.001) * scale, top)]
        rows += [(half_width(shape, 0.0, depth * i / 40 * 0.9999 + 1e-3) * scale, waist + depth * i / 40 * scale)
                 for i in range(41)]
        svg = SVG.format(outline(rows))
    else:
        svg = sketch_of(garment, wanted)
    fit = fit_to_sketch(svg, garment, M, "woven", base)
    assert fit.params["skirt_flare"] == pytest.approx(wanted["skirt_flare"], abs=8)
    assert fit.params["skirt_length"] == pytest.approx(wanted["skirt_length"], abs=8)
    assert fit.score_after > fit.score_before and fit.score_after > 95
    assert fit.verified and [r for r in fit.rounds if r["decision"] == "accepted"]
    assert all(r["locality_passed"] for r in fit.rounds if r["decision"] == "accepted")


def test_shift_dress_change_stays_local_and_fits() -> None:
    """Flare and length on a shift dress move only declared edges, and the fit runs."""
    base = {"dress_waist_seam": 0.0, "skirt_flare": 80.0, "skirt_length": 580.0}
    change = {"skirt_flare": 150.0, "skirt_length": 700.0}
    before, after = REGISTRY["dress"](M, "woven", base), REGISTRY["dress"](M, "woven", {**base, **change})
    assert edit_locality(before, after, change).passed
    bust = max(p.x for e in before.panels[0].edges if e.id == "front.side_upper" for p in e.curve.points)
    shape = pattern_silhouette(after)
    depth, scale = shape.bounds[3], 0.25
    rows = [(bust * scale, 10.0), (half_width(shape, 0.0, 0.5) * scale, 10.0 + M.front_length * scale)]
    rows += [(half_width(shape, 0.0, depth * i / 20 * 0.9999 + 1e-3) * scale, 10.0 + (M.front_length + depth * i / 20) * scale)
             for i in range(1, 21)]
    fit = fit_to_sketch(SVG.format(outline(rows)), "dress", M, "woven", base)
    assert fit.params["skirt_length"] == pytest.approx(700, abs=8) and fit.verified


def test_straight_shift_sketch_is_scaled_from_the_bust() -> None:
    """A box-shaped shift has no waist, so its width stands for the bust, not the waist."""
    base = {"dress_waist_seam": 0.0, "skirt_flare": 0.0, "skirt_length": 580.0}
    pattern = REGISTRY["dress"](M, "woven", base)
    bust = max(p.x for e in pattern.panels[0].edges if e.id == "front.side_upper" for p in e.curve.points)
    silhouette = read_silhouette(SVG.format(outline([(bust * 0.25, 10.0), (bust * 0.25, 10.0 + 1000 * 0.25)])))
    assert read_target(silhouette, pattern).scale == pytest.approx(4.0)


def test_out_of_range_sketch_stops_at_the_limit_and_says_so() -> None:
    """A very wide hem clamps to the declared maximum, with a note."""
    fit = fit_to_sketch(SVG.format(outline([(30, 20), (45, 90), (190, 300)])), "skirt", M, "woven", {"waistband_height": 0.0})
    assert fit.params["skirt_flare"] == 250
    assert any("skirt_flare" in note and "limit" in note for note in fit.notes)


def test_stretched_skirt_keeps_its_category_length() -> None:
    """A skirt drawn far too tall still fits its width, but the length stays and a note says why."""
    drawing = SVG.format(outline([(50, 40), (140, 450)]))
    fit = fit_to_sketch(drawing, "skirt", M, "woven", {"skirt_flare": 80.0, "skirt_length": 580.0, "waistband_height": 0.0})
    assert fit.params["skirt_length"] == 580 and fit.params["skirt_flare"] != 80
    assert any("stretched" in note for note in fit.notes)


def test_stretched_dress_uses_its_visible_waist_for_length() -> None:
    """Drawing the dress twice as tall gives the same length, read from the waist position."""
    base = {"dress_waist_seam": 1.0, "skirt_flare": 80.0, "skirt_length": 580.0}
    pattern = REGISTRY["dress"](M, "woven", base)
    rows = lambda k: [(55, 10.0), (46.9, 10 + 430 * 0.25 * k), (70, 10 + (430 + 680) * 0.25 * k)]  # noqa: E731
    normal = read_target(read_silhouette(SVG.format(outline(rows(1.0)))), pattern)
    tall = read_target(read_silhouette(SVG.format(outline(rows(1.7)))), pattern)
    assert normal.length == pytest.approx(680, abs=15) and tall.length == pytest.approx(680, abs=15)


@pytest.mark.parametrize("bad", ['<path d="M1e999 0 L10 10 L20 0 Z" fill="white"/>', '<rect x="1e400" width="5" height="5"/>'])
def test_non_finite_coordinates_are_refused(bad: str) -> None:
    """Coordinates overflowing to infinity are a readable refusal, not a crash."""
    with pytest.raises(SketchUnreadable):
        read_silhouette(SVG.format(bad))


def test_score_is_100_for_the_same_shape_and_lower_when_different() -> None:
    """The score measures overlap: identical is 100, a wider pattern is lower."""
    params = {"skirt_flare": 100.0, "skirt_length": 600.0, "waistband_height": 0.0}
    pattern = REGISTRY["skirt"](M, "woven", params)
    silhouette = read_silhouette(sketch_of("skirt", params))
    target = read_target(silhouette, pattern)
    assert match_score(silhouette, target, pattern) > 98
    wider = REGISTRY["skirt"](M, "woven", {**params, "skirt_flare": 250.0})
    assert match_score(silhouette, target, wider) < 90


def test_rejected_steps_are_logged_and_the_pattern_never_gets_worse(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the verifier refuses every step, the original parameters stay and rounds show why."""
    from seamproof import sketchfit

    real, calls = verify_pattern, []

    def fails_after_first(pattern):
        report = real(pattern)
        calls.append(1)
        if len(calls) > 1:
            report.checks[0].passed = False
        return report

    monkeypatch.setattr(sketchfit, "verify_pattern", fails_after_first)
    base = {"skirt_flare": 80.0, "skirt_length": 580.0, "waistband_height": 0.0}
    fit = fit_to_sketch(sketch_of("skirt", {"skirt_flare": 200.0, "skirt_length": 800.0}), "skirt", M, "woven", base)
    assert fit.params == {k: base[k] for k in fit.params} and fit.score_after == fit.score_before
    assert fit.rounds and all(r["decision"] == "rejected" and "adds failed" in r["reason"] for r in fit.rounds)


def test_other_garments_are_refused() -> None:
    """Only skirt and dress have fitted proportions."""
    with pytest.raises(ValueError):
        fit_to_sketch(sketch_of("skirt", {}), "shirt", M, "woven")


def test_previews_follow_fitted_values_and_categories_stay_identical() -> None:
    """A category's own value draws unchanged; a different fitted hem changes flat and 3D."""
    plan = plan_piece("bottom", "skirt", dict(SKIRT))
    flat, spec = plan_flat(plan), plan_preview(plan)
    same = plan_piece("bottom", "skirt", dict(SKIRT))
    apply_proportions(same, {"skirt_flare": 80.0, "skirt_length": 580.0})
    assert plan_flat(same) == flat and plan_preview(same) == spec
    wide = plan_piece("bottom", "skirt", dict(SKIRT))
    apply_proportions(wide, {"skirt_flare": 250.0, "skirt_length": 1000.0})
    assert plan_flat(wide) != flat and plan_preview(wide) != spec
    assert view_features(wide)["hem_half"] > SKIRT_HEM_HALF["flared"]


@pytest.mark.parametrize("bad", [{"skirt_flare": 9999.0}, {"design_ease": 10.0}, {"skirt_flare": "wide"}, {}, "x"])
def test_apply_proportions_rejects_bad_values(bad) -> None:
    """Out-of-range, foreign and malformed proportions never reach a draft."""
    with pytest.raises(ValueError):
        apply_proportions(plan_piece("bottom", "skirt", dict(SKIRT)), bad)


def test_apply_proportions_rejects_other_garments() -> None:
    """A shirt has no flare to fit."""
    plan = plan_piece("top", "shirt", {"collar": "shirt_collar", "opening": "button_front"})
    with pytest.raises(ValueError):
        apply_proportions(plan, {"skirt_flare": 50.0})


def test_api_fit_then_pack_uses_the_fitted_values() -> None:
    """The endpoint returns the fit; the pack with those proportions drafts and verifies them."""
    client = TestClient(app)
    body = {"part": "bottom", "kind": "skirt", "features": SKIRT, "color": "navy",
            "svg": sketch_of("skirt", {"skirt_flare": 150.0, "skirt_length": 700.0})}
    reply = client.post("/api/sketch-fit", json=body)
    assert reply.status_code == 200
    data = reply.json()
    assert data["fit"]["score_after"] > data["fit"]["score_before"]
    assert data["before"] == {"skirt_flare": 80.0, "skirt_length": 580.0}
    fitted = data["plan"]["proportions"]
    style = {"part": "bottom", "kind": "skirt", "features": SKIRT, "proportions": fitted}
    pack = client.post("/api/pack", json={"garment": "skirt", "mode": "size", "size": "M", "style": style})
    assert pack.status_code == 200 and pack.json()["passed"]
    assert pack.json()["style"]["params"]["skirt_length"] == fitted["skirt_length"]
    again = client.post("/api/style-plan", json={**style, "color": "navy"})
    assert again.json()["plan"]["proportions"] == fitted


def test_api_refuses_bad_input() -> None:
    """A top, an unreadable sketch and bad proportions are client errors."""
    client = TestClient(app)
    top = {"part": "top", "kind": "shirt", "features": {"collar": "shirt_collar"}, "svg": sketch_of("skirt", {})}
    assert client.post("/api/sketch-fit", json=top).status_code == 422
    flat_line = {"part": "bottom", "kind": "skirt", "features": SKIRT, "svg": SVG.format('<path d="M0 0L9 9" stroke="black"/>')}
    assert client.post("/api/sketch-fit", json=flat_line).status_code == 422
    style = {"part": "bottom", "kind": "skirt", "features": SKIRT, "proportions": {"skirt_flare": 9999}}
    assert client.post("/api/pack", json={"garment": "skirt", "mode": "size", "size": "M", "style": style}).status_code == 422
