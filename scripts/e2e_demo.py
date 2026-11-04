"""End-to-end run of the demo flow through the real web UI, against live Meta models.

This is the exact flow shown in the demo video: landing -> photo mode with a
front and a back photo (Muse Wonder) -> mix parts -> refine (Muse Spark) -> 3D
-> fabric suggestion -> draft + 15 verifier checks + explanation (Muse Spark
Open when BASETEN_TASKS=explain) -> toile photo (Muse Wonder) -> verified v2
-> sign-off -> production pack + marker.

It only drives the browser; every number on screen comes from the Python
engine. Start the server first (python -m seamproof server), then:

    python scripts/e2e_demo.py OUT_DIR [BASE_URL]

Needs the dev extras (playwright) and `playwright install chromium`.
Writes a screenshot per stage plus the panel text of each model step.
"""
from __future__ import annotations

import asyncio
import base64
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

INPUTS = Path(__file__).resolve().parent.parent / "demo" / "inputs"
REFINE = "Make it in a warm khaki beige like our sample, keep the oversized boxy fit"
T0 = time.time()


def log(stage: str) -> None:
    """Print one stage with its elapsed time."""
    print(f"{time.time() - T0:7.1f}s {stage}", flush=True)


async def run(out: Path, base: str) -> None:
    """Walk the whole demo flow once and save evidence into out."""
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl",
                                                "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080}, accept_downloads=True)
        page = await ctx.new_page()
        page.on("pageerror", lambda e: print("PAGEERROR", e, flush=True))

        async def shot(name: str) -> None:
            await page.screenshot(path=out / f"{name}.png"); log(name)

        async def click(sel: str) -> None:
            loc = page.locator(sel).first
            await loc.wait_for(state="visible", timeout=240000)
            await loc.scroll_into_view_if_needed(); await loc.click(); await asyncio.sleep(0.3)

        async def until(js: str, timeout: int = 300000) -> None:
            await page.wait_for_function(js, timeout=timeout)

        async def save_text(name: str, sel: str) -> None:
            (out / f"{name}.txt").write_text(await page.locator(sel).inner_text())

        await page.goto(base + "/", wait_until="networkidle"); await shot("00_landing")
        await click(".lp-cta a.lp-btn.ai"); await asyncio.sleep(1); await shot("01_studio")

        # Muse Wonder reads the front and the back photo.
        await click('#d-mode [data-mode="photo"]')
        await page.set_input_files("#f-file", [str(INPUTS / "front.jpeg"), str(INPUTS / "back.png")])
        await until("!document.querySelector('#f-slots').innerText.includes('Reading')")
        await page.wait_for_selector("#f-go:not([disabled])", timeout=240000)
        await asyncio.sleep(1); await shot("02_photos_read"); await save_text("photos", "aside.panel")

        # Muse Spark applies one design change.
        await click("#f-go"); await page.wait_for_selector("#r-next", state="visible", timeout=240000)
        await click("#r-text"); await page.keyboard.type(REFINE); await click("#r-go")
        await until("document.querySelector('#r-go') && !document.querySelector('#r-go').disabled"
                    " && !document.querySelector('#r-go .spinner')", 180000)
        await asyncio.sleep(1); await shot("03_refined"); await save_text("refine", "aside.panel")
        await click('[data-pvmode="3d"]'); await asyncio.sleep(3); await shot("04_3d")

        # Size, fabric suggestion (Muse Spark), then the deterministic draft + verifier.
        await click("#r-next"); await click("#u-fab-go")
        await until("document.querySelector('#u-fab-out').innerText.trim().length>0", 120000)
        await save_text("fabric", "#u-fab-out")
        await click("#u-go"); await page.wait_for_selector("#b-tabs", timeout=240000)
        try:
            await until("!document.querySelector('aside.panel').innerText.includes('Muse is writing')", 170000)
        except Exception:
            log("story timeout (continuing)")
        await shot("05_checks"); await save_text("checks", "aside.panel")
        await click('#b-tabs [data-sub="explain"]'); await asyncio.sleep(2); await shot("06_explain")
        await click('#b-tabs [data-sub="checks"]')

        # Toile photo -> Muse Wonder notes -> rule table -> verified v2.
        await click("#to-cek")
        await page.set_input_files("#t-files", str(INPUTS / "toile.png")); await click("#t-go")
        await until("document.querySelector('#t-go') && !document.querySelector('#t-go').disabled"
                    " && document.querySelector('#t-out').innerText.trim().length>0")
        await shot("07_toile"); await save_text("toile", "#t-out")
        if await page.locator("#t-use").count():
            await click("#t-use"); await asyncio.sleep(2); await shot("08_v2"); await click("#to-cek")

        # Sign-off, pattern PDF, production pack and marker.
        await click("#s-name"); await page.keyboard.type("Pattern maker"); await click("#s-go")
        await asyncio.sleep(2); await shot("09_signed")
        pdf = await page.evaluate("""async()=>{const a=[...document.querySelectorAll('.downloads a')]
            .find(x=>x.href.includes('/pdf'));if(!a)return null;const r=await fetch(a.href);
            const b=new Uint8Array(await r.arrayBuffer());let s='';for(const c of b)s+=String.fromCharCode(c);
            return btoa(s)}""")
        if pdf:
            (out / "pattern.pdf").write_bytes(base64.b64decode(pdf)); log("pattern.pdf")
        await click('.step[data-step="produksi"]'); await click("#p-go")
        await until("document.querySelector('#p-go') && !document.querySelector('#p-go').disabled", 240000)
        await asyncio.sleep(1); await shot("10_production"); await save_text("production", "aside.panel")
        await click('[data-view="marker"]'); await asyncio.sleep(2); await shot("11_marker")
        await browser.close()


def main() -> int:
    """CLI entry: OUT_DIR [BASE_URL]."""
    if len(sys.argv) < 2:
        print(__doc__); return 2
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    asyncio.run(run(Path(sys.argv[1]), base))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
