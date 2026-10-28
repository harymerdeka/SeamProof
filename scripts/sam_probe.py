#!/usr/bin/env python3
"""Step 0 probe: segment one test image with sam-3.1 via the Meta Model API.

Official request format only, taken from the meta-models/meta-model-cookbook
recipe ``07_sam/01_api_basics`` (``python/sam_api.py``) and
https://dev.meta.ai/docs/sam/segmenting:

- ``POST /v1/responses`` with ``stream=True``, ``model="sam-3.1"``.
- ``input`` is a single user message whose content is a short noun phrase
  (``input_text``) plus the photo as a data URL (``input_image``).
- Still images use a data URL; the Files API (``purpose="user_data"``) is only
  used for video uploads, so this probe does not use it.
- Streamed events are decoded with the official ``meta_sam_parser`` package
  (``pip install meta_sam_parser``); this script never parses the model's
  text grammar itself.

Saves to ``out/``: ``probe_mask.png`` (largest mask), ``probe_mask.svg``,
``probe_summary.json``. Prints the raw response structure (record counts,
boxes, mask dims, diagnostics) without keys or image bytes.

Requires ``MODEL_API_KEY`` in the environment (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import struct
import sys
import zlib
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is a project dependency; keep a clear error.
    print("python-dotenv is required: pip install python-dotenv", file=sys.stderr)
    raise

try:
    from meta_sam_parser import (
        ResponsesEventLike,
        decode_mask_to_raster,
        decode_mask_to_svg_path,
        image_segmentation_format,
        parse_responses_stream,
    )
except ImportError:
    print(
        "meta_sam_parser is required: pip install meta_sam_parser "
        "(official SAM output parser, see dev.meta.ai/docs/sam/client-libraries)",
        file=sys.stderr,
    )
    raise

try:
    from openai import AsyncOpenAI
except ImportError:
    print("openai SDK is required: pip install openai", file=sys.stderr)
    raise

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


def media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is None:
        raise ValueError(f"Could not infer a media type for {path}.")
    return guessed


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type(path)};base64,{encoded}"


async def stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[ResponsesEventLike]:
    """Mirror of the cookbook's lane finalizer: the API streams
    ``response.content_part.done`` but not ``response.output_text.done``, which
    the parser uses to close a lane, so synthesize it from the content part."""
    completed_lanes: set[tuple[object, object, object]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (
            data.get("item_id"),
            data.get("output_index"),
            data.get("content_index"),
        )
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


def write_png_grayscale(path: Path, width: int, height: int, pixels: list[int]) -> None:
    """Minimal stdlib PNG writer (8-bit grayscale). Avoids new dependencies."""
    if len(pixels) != width * height:
        raise ValueError(
            f"Pixel count {len(pixels)} does not match {width}x{height}."
        )
    raw = b"".join(
        b"\x00" + struct.pack(f">{width}B", *[min(255, max(0, v)) for v in pixels[y * width : (y + 1) * width]])
        for y in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        out = tag + data
        return struct.pack(">I", len(data)) + out + struct.pack(">I", zlib.crc32(out) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="flat-lay garment photo (garments only, no people)")
    parser.add_argument("--concept", default="shirt", help="short noun phrase, e.g. 'shirt'")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    load_dotenv()
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY before calling the SAM API.")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    media = {"type": "input_image", "image_url": image_data_url(args.image)}

    async with AsyncOpenAI(base_url=BASE_URL, api_key=key) as api:
        stream = await api.responses.create(
            model=MODEL,
            stream=True,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": args.concept.strip()}, media],
                }
            ],
        )
        async with stream, aclosing(stream_events(stream)) as events:
            result = await parse_responses_stream(events, image_segmentation_format()).final_result()

    boxes = [r for r in result.records if r.kind == "box"]
    masks = [r for r in result.records if r.kind == "mask"]
    print(f"outcome: {result.outcome.status}")
    print(f"records: {len(result.records)} (boxes={len(boxes)}, masks={len(masks)})")
    for b in boxes:
        print(f"  box object={b.object_id} frame={b.frame} "
              f"left={b.left} top={b.top} right={b.right} bottom={b.bottom}")
    scored = []
    for m in masks:
        fg = sum(decode_mask_to_raster(m.mask))
        scored.append((fg, m))
        print(f"  mask object={m.object_id} encoding={m.mask.encoding} "
              f"size={m.mask.width}x{m.mask.height} foreground_pixels={fg}")
    for d in result.diagnostics:
        print(f"  diagnostic {d.severity} line={d.line} code={d.code}: {d.message}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": MODEL,
        "concept": args.concept,
        "outcome": result.outcome.status,
        "boxes": len(boxes),
        "masks": len(masks),
    }
    if result.outcome.status != "completed":
        print("Segmentation incomplete; no mask saved.", file=sys.stderr)
    elif not masks:
        print("No masks returned for this concept; no mask saved.", file=sys.stderr)
    else:
        fg_best, best = max(scored, key=lambda t: t[0])
        raster = [255 if v else 0 for v in decode_mask_to_raster(best.mask)]
        write_png_grayscale(args.out / "probe_mask.png", best.mask.width, best.mask.height, raster)
        (args.out / "probe_mask.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {best.mask.width} {best.mask.height}">\n'
            f'  <path d="{decode_mask_to_svg_path(best.mask)}" fill="black"/>\n'
            "</svg>\n",
            encoding="utf-8",
        )
        summary["saved_object_id"] = best.object_id
        summary["saved_foreground_pixels"] = fg_best
        print(f"Saved largest mask to {args.out / 'probe_mask.png'}")
    (args.out / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
