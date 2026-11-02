"""Offline browser checks for generated designs; requires Playwright Chromium."""

from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

import seamproof.llm as llm_module
from seamproof.server import app

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def browser_page(monkeypatch: pytest.MonkeyPatch):
    """Route all app requests in process and abort every external request."""
    reply = {"options": [
        {"name": "Sage shirt", "part": "top", "kind": "shirt", "features": {
            "collar": "shirt_collar", "opening": "button_front", "sleeve_length": "long",
            "fit": "regular", "length": "hip", "pockets": 1, "cuff": True}, "color": "sage"},
        {"name": "A-line skirt", "part": "bottom", "kind": "skirt", "features": {
            "silhouette": "a_line", "length": "knee", "waistband": True}, "color": "navy"},
    ]}
    monkeypatch.setattr(llm_module, "call_json", lambda *args, **kwargs: reply)
    with playwright.sync_playwright() as driver, TestClient(app) as client:
        try:
            browser = driver.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                                   "--enable-unsafe-swiftshader"])
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Install the test browser: python -m playwright install chromium")
            if "error while loading shared libraries" in str(exc):
                pytest.skip("Chromium system libraries are unavailable on this server")
            raise
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def serve(route):
            """Serve only this app; test execution never accesses the network."""
            request = route.request
            url = urlsplit(request.url)
            if url.hostname != "seamproof.test":
                route.abort()
                return
            response = client.request(request.method, url.path, content=request.post_data_buffer,
                                      headers={"content-type": request.headers.get("content-type", "")})
            route.fulfill(status=response.status_code, body=response.content,
                          content_type=response.headers.get("content-type", ""))

        page.route("**/*", serve)
        yield page, errors
        browser.close()


def test_landing_page_is_first_and_opens_the_studio(browser_page) -> None:
    """The root shows the landing page with live engine data; its button opens the studio."""
    page, errors = browser_page
    page.goto("http://seamproof.test/")
    page.get_by_role("heading", name="Meet your AI pattern agent.").wait_for()
    assert not page.locator("#d-brief").is_visible()
    page.locator("#lp-art svg.pattern").wait_for()
    assert page.locator("#lp-cap").inner_text().startswith("Pattern ready: 19 of 19")
    # Real garment photos, served from this app, with no sketches in the strip.
    assert page.evaluate("[...document.querySelectorAll('#lp-gallery img')].slice(0, 8).every(i => i.src.endsWith('.jpg'))")
    page.wait_for_function("document.querySelector('.lp-photo img').naturalWidth > 0")
    page.wait_for_function("document.querySelectorAll('#lp-steps li.on, #lp-steps li.done').length > 0")
    assert page.locator(".lp-ai-word").first.is_visible()
    assert page.get_by_text("Code", exact=True).count() == 0
    page.get_by_role("link", name="Start a design").first.click()
    page.locator("#d-brief").wait_for()
    assert not page.locator("#landing").is_visible()
    page.go_back()
    page.get_by_role("heading", name="Meet your AI pattern agent.").wait_for()
    assert not errors


def test_studio_link_skips_the_landing_page(browser_page) -> None:
    """A direct #studio link opens the app."""
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-brief").wait_for()
    assert not page.locator("#landing").is_visible()
    assert not errors


def test_generated_design_defaults_to_2d_and_can_switch_to_3d(browser_page) -> None:
    """Options and the chosen design open in 2D; 3D stays available and rotatable."""
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-brief").fill("Sage office shirt")
    page.locator("#d-go").click()
    page.locator(".opt img").first.wait_for()
    assert page.locator("#mat canvas").count() == 0
    page.get_by_role("button", name="3D", exact=True).click()
    page.wait_for_function("document.querySelectorAll('.opt .v3d.ready canvas').length === 2")
    page.get_by_role("button", name="2D", exact=True).click()
    assert page.locator("#mat canvas").count() == 0
    page.locator(".opt[data-opt='0']").click()
    page.get_by_alt_text("Front and back flat sketch").wait_for()
    assert page.locator(".hero canvas").count() == 0
    page.get_by_role("button", name="3D", exact=True).click()
    page.locator(".hero .v3d.ready canvas").wait_for()
    for name in ("Front", "Side", "Back"):
        page.get_by_role("button", name=name, exact=True).click()
    canvas = page.locator(".hero canvas")
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 + 80, box["y"] + box["height"] / 2 + 20, steps=8)
    page.mouse.up()
    assert page.locator(".hero .v3d.orbit.ready canvas").count() == 1
    page.get_by_role("button", name="2D", exact=True).click()
    assert page.locator(".hero canvas").count() == 0
    assert not errors


def test_webgl_unavailable_explains_flat_fallback(browser_page) -> None:
    """A browser without WebGL shows the flat and explains why 3D is absent."""
    page, errors = browser_page
    page.add_init_script("""const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(type, ...args) {
        return type.includes('webgl') ? null : original.call(this, type, ...args);
      };""")
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-brief").fill("Sage office shirt")
    page.locator("#d-go").click()
    page.locator(".opt img").first.wait_for()
    page.get_by_role("button", name="3D", exact=True).click()
    page.get_by_text("3D is not available in this browser. Showing the flat sketch.").first.wait_for()
    assert page.locator(".opt .v3d.ready").count() == 0
    assert page.locator(".opt .v3d-fallback").first.is_visible()
    assert not errors


def test_photo_shows_ai_sketch_as_reference(browser_page, monkeypatch: pytest.MonkeyPatch) -> None:
    """A photo gets Muse Wonder's sketch, labelled reference, also on the design it becomes."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    replies = {"photo_style": {"photo_view": "flat_lay", "pieces": [{"part": "top", "kind": "shirt", "features": {}}]},
               "photo_sketch": {"svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">'
                                       '<path d="M10 10L90 90" stroke="black"/></svg>'}}
    monkeypatch.setattr(llm_module, "call_vision_json", lambda s, t, u, schema, purpose: replies[purpose])
    jpeg = cv2.imencode(".jpg", numpy.full((60, 60, 3), 200, dtype=numpy.uint8))[1].tobytes()
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-mode button[data-mode='photo']").click()
    page.locator("#f-file").set_input_files({"name": "a.jpg", "mimeType": "image/jpeg", "buffer": jpeg})
    page.locator("#f-go").wait_for()
    assert page.locator(".hero img[alt='Preview of your design']").is_visible()
    assert not page.get_by_alt_text("AI sketch of photo 1").is_visible()
    page.locator("#f-go").click()
    page.get_by_text("Compare with the AI's drawing").click()
    page.locator(".hero").get_by_alt_text("AI sketch of photo 1").wait_for()
    assert "Reference only" in page.locator(".hero .ai-sketch").inner_text()
    assert not errors


def test_design_follows_the_ai_sketch_proportions(browser_page, monkeypatch: pytest.MonkeyPatch) -> None:
    """A skirt fits to its sketch as a new version; the old flat is kept and the pack gets the values."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    outline = "M150 20 L250 20 L320 400 L80 400 Z"
    replies = {"photo_style": {"photo_view": "flat_lay", "pieces": [{"part": "bottom", "kind": "skirt", "features": {
                   "silhouette": "a_line", "length": "knee", "hem": "straight", "waistband": False}}]},
               "photo_sketch": {"svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">'
                                       f'<path d="{outline}" fill="white" stroke="black"/></svg>'}}
    monkeypatch.setattr(llm_module, "call_vision_json", lambda s, t, u, schema, purpose: replies[purpose])
    jpeg = cv2.imencode(".jpg", numpy.full((60, 60, 3), 200, dtype=numpy.uint8))[1].tobytes()
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-mode button[data-mode='photo']").click()
    page.locator("#f-file").set_input_files({"name": "a.jpg", "mimeType": "image/jpeg", "buffer": jpeg})
    page.locator("#f-go").click()
    page.get_by_text("Compare with the AI's drawing").click()
    page.locator("#fit-sketch").click()
    page.get_by_text("Match with the sketch").wait_for()
    assert page.locator(".vthumb").count() == 2
    assert page.locator("#fit-sketch").count() == 0
    assert "Hem flare" in page.locator(".vthumb").nth(1).inner_text() or "Length" in page.locator(".vthumb").nth(1).inner_text()
    assert not errors


def test_merged_photos_show_one_ai_sketch_of_the_merge(browser_page, monkeypatch: pytest.MonkeyPatch) -> None:
    """Taking parts from two photos asks the model for one drawing from both, shown as the result."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    calls = []
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500"><path d="M10 10L90 90" stroke="black"/></svg>'
    reading = {"photo_view": "flat_lay", "pieces": [{"part": "top", "kind": "shirt", "features": {}}]}

    def fake(system, text, urls, schema, purpose):
        calls.append((purpose, text, len(urls)))
        if purpose != "photo_style":
            return {"svg": svg}
        # The two photos differ only in the collar, so that is the only part asked about.
        reads = sum(c[0] == "photo_style" for c in calls)
        collar = "shirt_collar" if reads == 1 else "peter_pan"
        return {**reading, "pieces": [{"part": "top", "kind": "shirt", "features": {"collar": collar}}]}

    monkeypatch.setattr(llm_module, "call_vision_json", fake)
    jpeg = cv2.imencode(".jpg", numpy.full((60, 60, 3), 200, dtype=numpy.uint8))[1].tobytes()
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    page.locator("#d-mode button[data-mode='photo']").click()
    page.locator("#f-file").set_input_files([{"name": f"{n}.jpg", "mimeType": "image/jpeg", "buffer": jpeg} for n in "ab"])
    page.locator("button.pick[data-n='2']").first.wait_for()
    page.wait_for_function("document.querySelectorAll('.part-row').length === 1")
    page.locator("button.pick[data-n='2']").click()
    page.get_by_text("Compare with the AI's drawing").click()
    page.locator(".hero").get_by_alt_text("AI sketch of the merged design").wait_for()
    merged = [c for c in calls if c[0] == "photo_sketch_merge"]
    assert len(merged) == 1 and merged[0][2] == 2 and "from image 2" in merged[0][1]
    assert page.locator(".hero").get_by_text("exactly what the pattern will make").count() == 1
    page.locator("#f-go").click()
    page.locator(".hero").get_by_alt_text("AI sketch of the merged design").wait_for()
    assert page.locator("#mat .hero").get_by_alt_text("Front and back flat sketch").count() == 1
    assert len([c for c in calls if c[0] == "photo_sketch_merge"]) == 1
    assert not errors


def test_first_time_user_is_told_the_next_action(browser_page, monkeypatch: pytest.MonkeyPatch) -> None:
    """One photo asks no merge questions; the guide line leads from the upload to Refine."""
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    replies = {"photo_style": {"photo_view": "flat_lay", "pieces": [{"part": "top", "kind": "shirt", "features": {}}]},
               "photo_sketch": {"svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500"/>'}}
    monkeypatch.setattr(llm_module, "call_vision_json", lambda s, t, u, schema, purpose: replies[purpose])
    jpeg = cv2.imencode(".jpg", numpy.full((60, 60, 3), 200, dtype=numpy.uint8))[1].tobytes()
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    assert "Describe the garment" in page.locator("#guide").inner_text()
    page.locator("#d-mode button[data-mode='photo']").click()
    assert "Add a photo" in page.locator("#guide").inner_text()
    page.locator("#f-file").set_input_files({"name": "a.jpg", "mimeType": "image/jpeg", "buffer": jpeg})
    page.locator("#f-go").wait_for()
    assert "Use this design" in page.locator("#guide").inner_text()
    assert page.locator("button.pick").count() == 0 and page.locator("button.target").count() == 0
    assert page.get_by_text("Merged").count() == 0
    page.locator("#f-go").click()
    assert "press Continue" in page.locator("#guide").inner_text()
    assert page.locator("#r-more").count() == 1 and not page.locator("#r-more .chip").first.is_visible()
    assert not errors


def test_shirt_sleeve_can_be_focused_in_pattern_view(browser_page) -> None:
    """Create a shirt pack, focus its sleeve outline, then fit all panels again."""
    page, errors = browser_page
    page.goto("http://seamproof.test/#studio")
    # Use the base shirt: this flow needs no model or WebGL.
    page.locator("#d-mode button[data-mode='manual']").click()
    page.locator("label[for='gc-shirt']").click()
    page.locator("#r-next").click()
    page.locator("#u-go").click()
    page.locator(".mat-panels button[data-panel='sleeve']").wait_for()
    home = page.locator("svg.pattern").get_attribute("viewBox")
    page.get_by_role("button", name="Sleeve", exact=True).click()
    assert page.locator("#sleeve.panel-selected polyline.cut").count() == 1
    assert page.locator(".mat-panels button[data-panel='sleeve']").get_attribute("aria-pressed") == "true"
    assert page.locator("svg.pattern").get_attribute("viewBox") != home
    assert page.evaluate("""() => {
      const svg = document.querySelector('svg.pattern'), v = svg.viewBox.baseVal;
      const b = svg.querySelector('#sleeve polyline.cut').getBBox();
      return b.x >= v.x && b.y >= v.y && b.x + b.width <= v.x + v.width && b.y + b.height <= v.y + v.height;
    }""")
    page.get_by_role("button", name="All panels", exact=True).click()
    assert page.locator("svg.pattern").get_attribute("viewBox") == home
    assert page.locator(".panel-selected").count() == 0
    assert not errors
