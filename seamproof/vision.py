"""Photo intake (style path): blur faces, then read garment *style* with Muse Wonder.

Division of labour (Prompt 7, adapted for worn photos without an A4 sheet):
- OpenCV (deterministic): decode, downscale, detect and blur faces before any
  API call. No measurement in mm is ever taken from a worn photo.
- Muse Wonder: categorical style features only (garment kind, collar,
  opening, pockets, sleeve length, silhouette, hem shape...). It never
  returns millimetres.
- seamproof.style (deterministic): maps categories to drafting parameters;
  body millimetres come from the size table or the user's own measurements.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Callable
from xml.etree import ElementTree

from seamproof import llm as llm_module

MAX_SIDE_PX = 1280

# Allowed values per feature; anything else becomes the listed fallback.
TOP_FEATURES: dict[str, tuple[str, ...]] = {
    "collar": ("none", "shirt_collar", "peter_pan", "band", "other"),
    # Front neckline shape when there is no collar (v = straight V to centre front,
    # scoop = deep round, boat = wide and shallow).
    "neckline": ("round", "v", "scoop", "boat"),
    "opening": ("none", "button_front", "other"),
    # Assumption: an unspecified sleeve is short; sleeveless must be explicit.
    # validate_reading uses the first allowed value for missing/unknown input.
    "sleeve_length": ("short", "elbow", "three_quarter", "long", "sleeveless"),
    # straight; bishop = full at the hem, gathered into a cuff; puff = bishop plus a gathered cap.
    "sleeve_style": ("straight", "bishop", "puff"),
    "shoulder": ("set_in", "dropped"),
    "fit": ("fitted", "regular", "boxy"),
    "length": ("waist", "high_hip", "hip", "thigh", "knee", "below_knee"),
}
BOTTOM_FEATURES: dict[str, tuple[str, ...]] = {
    "silhouette": ("straight", "a_line", "flared", "other"),
    "length": ("mini", "above_knee", "knee", "midi", "maxi"),
    "hem": ("straight", "scalloped", "other"),
}
# A dress is one piece from shoulder to hem: bodice features of a top plus
# skirt features, and whether a seam joins bodice and skirt at the waist.
DRESS_FEATURES: dict[str, tuple[str, ...]] = {
    **{key: values for key, values in TOP_FEATURES.items() if key != "length"},
    **BOTTOM_FEATURES,
}
TOP_KINDS = ("shirt", "vest", "tshirt", "blouse", "tunic", "other")
BOTTOM_KINDS = ("skirt", "trousers", "other")
DRESS_KINDS = ("dress", "other")
PARTS = ("top", "bottom", "dress")

STYLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "photo_view": {"type": "string"},
        "pieces": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "part": {"type": "string"},
                "kind": {"type": "string"},
                "features": {"type": "object"},
                "evidence": {"type": "string"},
            },
            "required": ["part", "kind", "features"],
        }},
    },
    "required": ["pieces"],
}


def _options(table: dict[str, tuple[str, ...]]) -> str:
    """Render allowed values for the prompt."""
    return "; ".join(f"{key}: {'|'.join(values)}" for key, values in table.items())


SYSTEM_PROMPT = (
    "You describe garments in a photo for a pattern maker. Faces are blurred. "
    "Report ONLY categorical style features from the allowed values; never estimate "
    "measurements, centimetres or millimetres."
)
USER_PROMPT = (
    "List every garment piece visible. JSON: {\"photo_view\": \"worn|flat_lay|other\", "
    "\"pieces\": [{\"part\": \"top|bottom|dress\", \"kind\": ..., \"features\": {...}, "
    "\"evidence\": \"short visual reason in English\"}]}.\n"
    "A dress (one garment from shoulder to hem, including shirt dresses) is ONE piece with "
    "part dress — never split it into a top plus a skirt. Use top + bottom only for two separate garments.\n"
    f"Top kinds: {'|'.join(TOP_KINDS)}. Top features — {_options(TOP_FEATURES)}; "
    "pockets: 0|1|2 (visible chest/patch pockets); cuff: true|false (turned-up or band cuff); "
    "elastic_cuff: true|false (cuff gathered on elastic / ruched); "
    "contrast_collar: true|false; contrast_cuff: true|false.\n"
    f"Bottom kinds: {'|'.join(BOTTOM_KINDS)}. Bottom features — {_options(BOTTOM_FEATURES)}; "
    "waistband: true|false.\n"
    f"Dress kinds: {'|'.join(DRESS_KINDS)}. Dress features — {_options(DRESS_FEATURES)}; "
    "waist_seam: true|false (a seam joins bodice and skirt at the waist); pockets: 0|1|2; "
    "cuff: true|false; elastic_cuff: true|false; contrast_collar: true|false; contrast_cuff: true|false.\n"
    "Length of a top is where its hem falls on the body; length of a skirt or dress is where its hem falls."
)


@dataclass
class PreparedPhoto:
    """A privacy-safe JPEG ready to send, plus what was done to it."""

    jpeg: bytes
    width: int
    height: int
    faces_blurred: int


@dataclass
class GarmentPiece:
    """One garment read from a photo; features are validated categories."""

    part: str
    kind: str
    features: dict[str, Any]
    evidence: str = ""


@dataclass
class StyleReading:
    """The validated vision result with any values that were coerced."""

    photo_view: str
    pieces: list[GarmentPiece] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = "wonder"


def prepare_photo(data: bytes) -> PreparedPhoto:
    """Decode, downscale to MAX_SIDE_PX and blur every detected face.

    Assumption: Haar cascades (frontal + profile) are good enough for
    catalogue photos; faces are blurred with a box expanded by 30 % so hair
    and chin edges are covered too.
    """
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The photo cannot be read — send a JPG or PNG.")
    height, width = image.shape[:2]
    scale = min(1.0, MAX_SIDE_PX / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    boxes: list[tuple[int, int, int, int]] = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        min_side = max(24, min(height, width) // 20)
        boxes.extend(tuple(int(v) for v in box)
                     for box in cascade.detectMultiScale(gray, 1.1, 4, minSize=(min_side, min_side)))
    for x, y, w, h in boxes:
        pad_w, pad_h = int(w * 0.3), int(h * 0.3)
        x0, y0 = max(0, x - pad_w), max(0, y - pad_h)
        x1, y1 = min(width, x + w + pad_w), min(height, y + h + pad_h)
        region = image[y0:y1, x0:x1]
        # Collapse to a 4x4 mosaic, then smooth: identity is not recoverable.
        small = cv2.resize(region, (4, 4), interpolation=cv2.INTER_AREA)
        image[y0:y1, x0:x1] = cv2.GaussianBlur(
            cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (51, 51), 0)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("The photo could not be processed.")
    return PreparedPhoto(bytes(encoded), width, height, len(boxes))


def _bool(value: Any) -> bool | None:
    """Accept JSON booleans and common string spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "yes", "no", "ya", "tidak"):
        return value.lower() in ("true", "yes", "ya")
    return None


def validate_reading(raw: dict[str, Any]) -> StyleReading:
    """Coerce the model JSON to known categories; record every coercion."""
    warnings: list[str] = []
    view = str(raw.get("photo_view", "other"))
    if view not in ("worn", "flat_lay", "other"):
        view = "other"
    pieces: list[GarmentPiece] = []
    for item in raw.get("pieces", []):
        part = str(item.get("part", ""))
        if part not in PARTS:
            warnings.append(f"Bagian '{part}' diabaikan (bukan atasan/bawahan/dress).")
            continue
        kinds, table = {"top": (TOP_KINDS, TOP_FEATURES), "bottom": (BOTTOM_KINDS, BOTTOM_FEATURES),
                        "dress": (DRESS_KINDS, DRESS_FEATURES)}[part]
        kind = str(item.get("kind", "other"))
        if kind not in kinds:
            warnings.append(f"Unknown kind '{kind}' — marked 'other'.")
            kind = "other"
        given = item.get("features") or {}
        features: dict[str, Any] = {}
        for key, allowed in table.items():
            value = str(given.get(key, allowed[0]))
            if value not in allowed:
                warnings.append(f"Unknown {part}.{key}='{value}' — using '{allowed[0]}'.")
                value = allowed[0]
            features[key] = value
        flags = {"top": ("cuff", "elastic_cuff", "contrast_collar", "contrast_cuff"), "bottom": ("waistband",),
                 "dress": ("waist_seam", "cuff", "elastic_cuff", "contrast_collar", "contrast_cuff")}[part]
        for key in flags:
            features[key] = bool(_bool(given.get(key)))
        if part in ("top", "dress"):
            pockets = given.get("pockets", 0)
            if isinstance(pockets, bool) or not isinstance(pockets, (int, float)) or pockets not in (0, 1, 2):
                warnings.append(f"Invalid pocket count '{pockets}' — using 0.")
                pockets = 0
            features["pockets"] = int(pockets)
        pieces.append(GarmentPiece(part, kind, features, str(item.get("evidence", ""))[:200]))
    return StyleReading(view, pieces, warnings)


def describe_garments(photo: PreparedPhoto,
                      vision_fn: Callable[..., dict[str, Any]] | None = None) -> StyleReading:
    """Ask Muse Wonder for categorical style features of a face-blurred photo."""
    url = "data:image/jpeg;base64," + base64.b64encode(photo.jpeg).decode("ascii")
    call = vision_fn or llm_module.call_vision_json
    raw = call(SYSTEM_PROMPT, USER_PROMPT, [url], STYLE_SCHEMA, "photo_style")
    return validate_reading(raw)


MEASURE_PROMPT = (
    "The photo shows a tailor's measurement note (handwritten or printed: order book, form, chat "
    "screenshot). Read ONLY the numbers that are written there and map each to a canonical field. "
    "Do not estimate anything from bodies or garments in the photo; skip values you cannot read. "
)


def read_measurement_photo(photo: PreparedPhoto, garment: str,
                           vision_fn: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """Muse Wonder transcribes a measurement note; intake rules check every value.

    The model only copies written numbers into fields (interpreting input);
    ranges, units and cross-field rules are the same deterministic intake
    checks as typed text, and missing fields stay missing.
    """
    from seamproof.intake import GARMENT_FIELDS, INTAKE_SCHEMA, SYSTEM_PROMPT, _build_result

    if garment not in GARMENT_FIELDS:
        raise ValueError(f"Unknown garment: {garment}.")
    url = "data:image/jpeg;base64," + base64.b64encode(photo.jpeg).decode("ascii")
    call = vision_fn or llm_module.call_vision_json
    raw = call(SYSTEM_PROMPT, MEASURE_PROMPT + f"Garment: {garment}.", [url], INTAKE_SCHEMA,
               f"measure-photo-{garment}")
    if not isinstance(raw, dict) or not isinstance(raw.get("measurements"), list):
        raise ValueError("The model did not return a measurement list.")
    items = [(e.get("field"), e.get("value"), e.get("unit")) for e in raw["measurements"] if isinstance(e, dict)]
    return _build_result(items, garment)



SKETCH_FORMAT = (
    "Copy the real silhouette and every construction detail you can see: neckline, collar, placket, buttons, darts, tucks, pleats, gathers, yokes, seams, sleeve "
    "shape, cuffs, hem, pockets. Draw the garment laid flat and symmetrical, with no person and no "
    "background. Output one SVG: viewBox '0 0 800 500', front view centred at x=200, back view centred at "
    "x=600 (infer the back if hidden). Only <path>, <line>, <circle>, <g> elements; closed outlines with "
    "fill white, stroke black, stroke-width 2.5; construction seams stroke-width 1.2; topstitching "
    "stroke-dasharray '6 4'. Smooth bezier curves, no text, no shading, no script, no images. "
    'JSON: {"svg": "<svg ...>...</svg>"}.'
)
SKETCH_SYSTEM = ("You are a senior fashion technical designer. Draw a professional technical flat sketch of the "
                 "main garment in the photo, as used in a tech pack. " + SKETCH_FORMAT)
MERGE_SKETCH_SYSTEM = (
    "You are a senior fashion technical designer. You receive numbered garment photos and a list saying which "
    "part of the new garment comes from which image. Draw ONE new garment that really combines those parts, as "
    "a professional technical flat sketch for a tech pack: each listed part must look like it does in its own "
    "image, joined into one believable garment. Never copy a part from an image other than the one listed. "
    + SKETCH_FORMAT
)
SKETCH_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"svg": {"type": "string"}}, "required": ["svg"]}
SVG_NS = "http://www.w3.org/2000/svg"
SKETCH_MAX_CHARS = 200_000
SKETCH_SHAPES = {"path", "line", "circle", "ellipse", "rect", "polyline", "polygon"}
SKETCH_ELEMENTS = SKETCH_SHAPES | {"svg", "g"}
SKETCH_ATTRS = {"viewBox", "d", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "width", "height",
                "points", "fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap",
                "stroke-linejoin", "transform"}


def sanitize_sketch(svg: str) -> str:
    """Keep only plain drawing elements and attributes of a model-written SVG.

    Raises ValueError when it is not a usable drawing. The UI also shows the
    result through <img>, so nothing in it can run even if this missed something.
    """
    text = svg.strip()
    if len(text) > SKETCH_MAX_CHARS or "<!" in text:  # "<!" blocks DOCTYPE entity expansion
        raise ValueError("The AI sketch is not a usable SVG.")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise ValueError("The AI sketch is not a usable SVG.") from exc

    def name(node: ElementTree.Element) -> str:
        return node.tag.rsplit("}", 1)[-1]

    if name(root) != "svg":
        raise ValueError("The AI sketch is not a usable SVG.")

    def clean(node: ElementTree.Element) -> ElementTree.Element:
        out = ElementTree.Element(f"{{{SVG_NS}}}{name(node)}")
        for key, value in node.attrib.items():
            if key in SKETCH_ATTRS and "url(" not in value.lower():
                out.set(key, value)
        out.extend(clean(child) for child in node if name(child) in SKETCH_ELEMENTS and name(child) != "svg")
        return out

    result = clean(root)
    if not any(name(node) in SKETCH_SHAPES for node in result.iter()):
        raise ValueError("The AI sketch has no drawing.")
    ElementTree.register_namespace("", SVG_NS)
    return ElementTree.tostring(result, encoding="unicode")


def sketch_from_photo(photo: PreparedPhoto, vision_fn: Callable[..., dict[str, Any]] | None = None) -> str:
    """Ask Muse Wonder to draw a reference flat of the photographed garment.

    It can show details no category can draft, so it is labelled as reference
    next to the coded sketch. Only seamproof.sketchfit reads it, and only for
    skirt and dress proportions, through the usual range and verifier gates.
    """
    url = "data:image/jpeg;base64," + base64.b64encode(photo.jpeg).decode("ascii")
    call = vision_fn or llm_module.call_vision_json
    raw = call(SKETCH_SYSTEM, "Draw the technical flat of this garment.", [url], SKETCH_SCHEMA, "photo_sketch")
    return sanitize_sketch(raw["svg"])


def merge_sketch(photos: list[PreparedPhoto], garment: str, parts: list[tuple[str, int]], base: int,
                 notes: str = "", vision_fn: Callable[..., dict[str, Any]] | None = None) -> str:
    """Ask Muse Wonder to draw one reference flat combining parts of several photos.

    parts pairs a part label with a 1-based position in photos; base is the
    position that supplies anything not listed. Like sketch_from_photo, the
    result is a reference that only seamproof.sketchfit reads.
    """
    for _label, number in parts:
        if not 1 <= number <= len(photos):
            raise ValueError(f"No photo {number} was sent.")
    if not 1 <= base <= len(photos):
        raise ValueError(f"No photo {base} was sent.")
    lines = "\n".join(f"- {label}: from image {number}" for label, number in parts)
    text = (f"Images are numbered in the order sent, 1 to {len(photos)}. New garment: {garment}.\n{lines}\n"
            f"Anything not listed, and the fabric colour, follows image {base}.")
    if notes:
        text += f"\nExtra request from the workshop: {notes}"
    urls = ["data:image/jpeg;base64," + base64.b64encode(photo.jpeg).decode("ascii") for photo in photos]
    call = vision_fn or llm_module.call_vision_json
    return sanitize_sketch(call(MERGE_SKETCH_SYSTEM, text, urls, SKETCH_SCHEMA, "photo_sketch_merge")["svg"])
