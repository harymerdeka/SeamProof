"""Design from scratch: a written brief becomes two draftable design options.

Model choice: Muse Spark (MUSE_MODEL through llm.call_json, so LLM_BACKEND
can still switch to Baseten). The brief and the refine step are text-only;
reference photos keep going to Muse Wonder (seamproof.vision); SAM is not
needed because no mask is taken from images. Models never report measurements;
seamproof.sketchfit measures a drawn sketch's proportions in code.

Division of labour, same as the photo path:
- Muse only picks *categories* (garment kind, collar, sleeve, colour key...)
  from the allowed lists, and writes the short name and reason.
- seamproof.vision.validate_reading coerces every category, seamproof.style
  maps them to drafting parameters, seamproof.flat draws the sketch.
- The diff of a refine step is computed here, never taken from the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from seamproof import llm as llm_module
from seamproof.fabric import FABRIC_EASE
from seamproof.flat import COLORS, DEFAULT_COLOR
from seamproof.style import StylePlan, plan_piece
from seamproof.vision import (
    BOTTOM_FEATURES,
    BOTTOM_KINDS,
    DRESS_FEATURES,
    DRESS_KINDS,
    TOP_FEATURES,
    TOP_KINDS,
    GarmentPiece,
    _options,
    validate_reading,
)

# A ChatGPT-style garment description runs 600-1500 characters; keep all of it.
MAX_BRIEF_CHARS = 2000
MAX_UNSUPPORTED = 8
MAX_OPTIONS = 2

DESIGN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "options": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "part": {"type": "string"},
                "kind": {"type": "string"},
                "features": {"type": "object"},
                "color": {"type": "string"},
                "fabric": {"type": "string"},
                "reason": {"type": "string"},
                "unsupported": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "part", "kind", "features"],
        }},
    },
    "required": ["options"],
}
REFINE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "features": {"type": "object"},
        "color": {"type": "string"},
        "reason": {"type": "string"},
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "features"],
}

_CATALOGUE = (
    f"Top kinds: {'|'.join(TOP_KINDS)}. Top features — {_options(TOP_FEATURES)}; pockets: 0|1|2; "
    "cuff: true|false; elastic_cuff: true|false; contrast_collar: true|false; contrast_cuff: true|false.\n"
    "sleeve_style: bishop = sleeve full at the hem, gathered into a cuff (balloon/bishop); puff = bishop "
    "plus a gathered, puffed shoulder/cap; elastic_cuff = cuff gathered on elastic (ruched, shirred).\n"
    f"Bottom kinds: {'|'.join(BOTTOM_KINDS)}. Bottom features — {_options(BOTTOM_FEATURES)}; "
    "waistband: true|false.\n"
    f"Dress kinds: {'|'.join(DRESS_KINDS)}. Dress features — {_options(DRESS_FEATURES)}; "
    "waist_seam: true|false; pockets: 0|1|2; cuff: true|false; elastic_cuff: true|false.\n"
    f"Colours: {'|'.join(COLORS)}. Fabrics: {'|'.join(sorted(FABRIC_EASE))}.\n"
    "The pattern library drafts: shirt (button front, collar none/shirt_collar/peter_pan/band, any sleeve "
    "length, neckline round, v, scoop or boat when there is no collar, "
    "sleeve_style straight/bishop/puff, plain or elastic cuffs), vest (a sleeveless button-front shirt), "
    "blouse and tunic (darted, no collar or button front, any sleeve length including sleeveless, those necklines, "
    "those sleeve styles and cuffs; a tunic has length hip to below_knee and its fit sets the hem flare), tshirt, skirt, "
    "and dress (one piece, no collar, side zip, sleeveless or sleeved, neckline round or v; "
    "scalloped hem only with waist_seam). Prefer these. Spaghetti straps, a V deeper than about 14 cm "
    "and wrap/crossover fronts are not drafted."
)
DESIGN_SYSTEM = (
    "You are a fashion designer helping a small Indonesian garment workshop. "
    "Turn the customer's brief into garment designs a pattern maker can draft. "
    "Use ONLY the allowed category values; never give measurements, centimetres or millimetres. "
    "Write name, reason and unsupported in English, short and plain. "
    "Be honest about fidelity: the categories are all the pattern can draw. Every detail of the brief "
    "that no category captures (e.g. ruffles, pleats, prints, embroidery, puff sleeves on a dress, or "
    "trousers or any other garment not designed) MUST be listed in unsupported, and reason must only "
    "describe what the chosen categories show. Do NOT list as unsupported: things the chosen values "
    "already cover (bishop/puff sleeves and elastic cuffs on a shirt are covered); fabric weight, drape "
    "or finish (covered by the fabric category; mention them in reason); button or trim colours "
    "(a buying note, not pattern); mood or occasion words such as elegant, modern, office."
)
DESIGN_USER = (
    "Customer brief (may be in Indonesian or English):\n{brief}\n\n"
    "Give exactly two options for the main garment of the brief. Option 1 follows the brief as "
    "closely as the categories allow (closest value when nothing matches exactly). Option 2 changes "
    "exactly one style choice and keeps everything else of option 1. "
    "JSON: {{\"options\": [{{\"name\": \"model name, at most 5 words\", \"part\": \"top|bottom|dress\", "
    "\"kind\": ..., \"features\": {{...}}, \"color\": ..., \"fabric\": ..., "
    "\"reason\": \"one sentence: what the categories show\", "
    "\"unsupported\": [\"brief details the pattern cannot make yet\"]}}]}}.\n" + _CATALOGUE
)
REFINE_SYSTEM = (
    "You edit one garment design for a pattern maker. Change ONLY what the instruction asks; "
    "copy every other value unchanged. Use only the allowed category values; never give "
    "measurements. Write reason in English, one short sentence. If the instruction asks for "
    "something no category can show (e.g. ruffles, pleats, prints), keep the values and list it in "
    "unsupported instead of pretending a category covers it. Bishop/puff sleeves and elastic cuffs on "
    "a shirt ARE categories (sleeve_style, elastic_cuff)."
)
REFINE_USER = (
    "Current design (JSON): {current}\nRequested change: {instruction}\n\n"
    "JSON: {{\"kind\": ..., \"features\": {{...all features...}}, \"color\": ..., \"reason\": ..., "
    "\"unsupported\": [...]}}. "
    "The part (top/bottom/dress) cannot change.\n" + _CATALOGUE
)

CallFn = Callable[..., dict[str, Any]]


@dataclass
class DesignOption:
    """One validated design: categories, colour, fabric and its drafting plan."""

    name: str
    piece: GarmentPiece
    color: str
    fabric: str | None
    reason: str
    plan: StylePlan
    warnings: list[str] = field(default_factory=list)
    # Brief details the categories (and so the pattern) cannot show, as Muse reported them.
    unsupported: list[str] = field(default_factory=list)


@dataclass
class Refinement:
    """A validated edit with the code-computed diff against the previous design."""

    piece: GarmentPiece
    color: str
    reason: str
    plan: StylePlan
    changes: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _unsupported(value: Any) -> list[str]:
    """Clean the model's list of details the pattern cannot show."""
    if not isinstance(value, list):
        return []
    items = [str(v).strip()[:120] for v in value if str(v).strip()]
    return list(dict.fromkeys(items))[:MAX_UNSUPPORTED]


def _color(value: Any, warnings: list[str]) -> str:
    """Coerce a colour key, recording unknown ones."""
    key = str(value or DEFAULT_COLOR).strip().lower().replace(" ", "_")
    if key not in COLORS:
        warnings.append(f"Unknown colour '{value}' — using {COLORS[DEFAULT_COLOR][1]}.")
        key = DEFAULT_COLOR
    return key


def _fabric(value: Any, warnings: list[str]) -> str | None:
    """Coerce a fabric category; unknown ones leave the garment's default fabric."""
    fabric = str(value or "")
    if fabric not in FABRIC_EASE:
        if value:
            warnings.append(f"Unknown fabric '{value}' — using the garment's suggested fabric.")
        return None
    return fabric


def _piece(raw: dict[str, Any], warnings: list[str]) -> GarmentPiece | None:
    """Validate one option's categories through the shared vision validator.

    Feature names the part does not have (e.g. a model inventing
    "sleeve_style") are reported, never silently dropped.
    """
    part, given = raw.get("part"), raw.get("features") or {}
    if part in MERGE_GROUPS and isinstance(given, dict):
        known = [k for k in given if k != "part" and _allowed(str(part), k) is None]
        if known:
            warnings.append(f"Features {', '.join(map(str, known))} are not in the pattern library — ignored.")
    reading = validate_reading({"pieces": [{"part": raw.get("part"), "kind": raw.get("kind"),
                                            "features": raw.get("features") or {}}]})
    warnings.extend(reading.warnings)
    return reading.pieces[0] if reading.pieces else None


def design_from_brief(brief: str, call_fn: CallFn | None = None) -> list[DesignOption]:
    """Ask Muse for two options and validate them; raise LLMValidationError if none survive."""
    brief = brief.strip()
    if len(brief) < 3:
        raise ValueError("Write a design brief first (at least one sentence).")
    call = call_fn or llm_module.call_json
    raw = call(DESIGN_SYSTEM, DESIGN_USER.format(brief=brief[:MAX_BRIEF_CHARS]), DESIGN_SCHEMA, "design_brief")
    options: list[DesignOption] = []
    for item in raw.get("options", [])[:MAX_OPTIONS]:
        warnings: list[str] = []
        piece = _piece(item, warnings)
        if piece is None:
            continue
        plan = plan_piece(piece.part, piece.kind, piece.features)
        color = _color(item.get("color"), warnings)
        options.append(DesignOption(str(item.get("name") or "New design")[:60], piece, color,
                                    _fabric(item.get("fabric"), warnings),
                                    str(item.get("reason") or "")[:240], plan, warnings,
                                    _unsupported(item.get("unsupported"))))
    if not options:
        raise llm_module.LLMValidationError("The model gave no usable design option.")
    return options


def diff_design(before: GarmentPiece, before_color: str, after: GarmentPiece, after_color: str) -> list[dict[str, Any]]:
    """Every category that differs, in a stable order (kind, features, colour)."""
    changes = []
    if before.kind != after.kind:
        changes.append({"key": "kind", "from": before.kind, "to": after.kind})
    for key in after.features:
        if before.features.get(key) != after.features[key]:
            changes.append({"key": key, "from": before.features.get(key), "to": after.features[key]})
    if before_color != after_color:
        changes.append({"key": "color", "from": before_color, "to": after_color})
    return changes


def refine_design(current: dict[str, Any], instruction: str, call_fn: CallFn | None = None) -> Refinement:
    """Apply one free-text change via Muse; the part is locked and the diff is ours."""
    instruction = instruction.strip()
    if len(instruction) < 3:
        raise ValueError("Write the change you want.")
    warnings: list[str] = []
    before = _piece(current, warnings)
    if before is None:
        raise ValueError("The current design is invalid (part must be top, bottom or dress).")
    before_color = _color(current.get("color"), warnings)
    state = {"part": before.part, "kind": before.kind, "features": before.features, "color": before_color}
    call = call_fn or llm_module.call_json
    raw = call(REFINE_SYSTEM, REFINE_USER.format(current=json.dumps(state), instruction=instruction[:300]),
               REFINE_SCHEMA, "design_refine")
    # Missing features keep their current value; the part never changes.
    merged = {**before.features, **(raw.get("features") or {})}
    after = _piece({"part": before.part, "kind": raw.get("kind"), "features": merged}, warnings)
    if after is None:  # unreachable: part comes from the validated current design
        raise llm_module.LLMValidationError("The model changed the garment part.")
    color = _color(raw.get("color") or before_color, warnings)
    changes = diff_design(before, before_color, after, color)
    unsupported = _unsupported(raw.get("unsupported"))
    if not changes:
        warnings.append("No category changed — " + (
            "the pattern cannot make this detail yet: " + "; ".join(unsupported) + "." if unsupported
            else "the pattern may not support this change yet."))
    elif len(changes) > 2:
        warnings.append(f"{len(changes)} things changed at once — check them one by one before continuing.")
    plan = plan_piece(after.part, after.kind, after.features)
    return Refinement(after, color, str(raw.get("reason") or "")[:240], plan, changes, warnings, unsupported)


# Parts of a garment the user can take from different photos (pjstudio-style
# "front from photo 1, sleeves from photo 2"). Assumption: photo categories
# carry no front/back split, so "front bodice" groups the front-only details.
MERGE_GROUPS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "top": [("siluet", "Kind, silhouette & length", ("kind", "fit", "length")),
            ("badan", "Front bodice (opening, pockets)", ("opening", "pockets")),
            ("lengan", "Sleeves & shoulder", ("sleeve_length", "sleeve_style", "shoulder", "cuff", "elastic_cuff",
                                         "contrast_cuff")),
            ("kerah", "Collar & neckline", ("collar", "neckline", "contrast_collar"))],
    "bottom": [("siluet", "Kind & silhouette", ("kind", "silhouette")),
               ("panjang", "Length", ("length",)),
               ("kelim", "Hem", ("hem",)),
               ("pinggang", "Waistband", ("waistband",))],
    "dress": [("badan", "Bodice (silhouette, opening, pockets)", ("fit", "opening", "pockets")),
              ("lengan", "Sleeves & shoulder", ("sleeve_length", "shoulder", "cuff", "contrast_cuff")),
              ("kerah", "Collar & neckline", ("collar", "neckline", "contrast_collar")),
              ("rok", "Skirt (silhouette, length, hem)", ("silhouette", "length", "hem")),
              ("pinggang", "Waist seam", ("waist_seam",))],
}
_TABLES = {"top": (TOP_KINDS, TOP_FEATURES), "bottom": (BOTTOM_KINDS, BOTTOM_FEATURES),
           "dress": (DRESS_KINDS, DRESS_FEATURES)}
_FLAGS = {"top": ("cuff", "elastic_cuff", "contrast_collar", "contrast_cuff"), "bottom": ("waistband",),
          "dress": ("waist_seam", "cuff", "elastic_cuff", "contrast_collar", "contrast_cuff")}


def _allowed(part: str, key: str) -> tuple[Any, ...] | None:
    """Values a feature may take on a part; None if the part has no such feature."""
    kinds, table = _TABLES[part]
    if key == "kind":
        return (part,) + kinds  # kinds only ever transfer within the same part
    if key in table:
        return table[key]
    if key in _FLAGS[part]:
        return (False, True)
    if key == "pockets" and part in ("top", "dress"):
        return (0, 1, 2)
    return None


def compatible(keys: tuple[str, ...], source_part: str, target_part: str) -> bool:
    """A source piece can give these features only if they mean the same on both parts.

    E.g. a dress's sleeves fit a top, but a dress length (mini..maxi) is not
    a top length (waist..knee), so length never moves between them.
    """
    return all(_allowed(source_part, k) is not None and _allowed(source_part, k) == _allowed(target_part, k)
               for k in keys)


@dataclass
class Merge:
    """A design merged from several photo readings, with where each part came from."""

    piece: GarmentPiece
    provenance: dict[str, int]
    groups: list[dict[str, Any]]
    plan: StylePlan
    warnings: list[str] = field(default_factory=list)


def merge_photos(target: str, base: int, sources: dict[int, Any], assign: dict[str, int]) -> Merge:
    """Merge feature groups from photo pieces into one design; pure code, no model.

    `sources` maps photo number -> its pieces ({part, kind, features}, or a
    list of them: one photo may show a top and a skirt). The base photo must
    hold a piece of the target part; it supplies every group the user did
    not assign elsewhere. From another photo, each group uses that photo's
    first piece that can lend it (same part first).
    """
    if target not in MERGE_GROUPS:
        raise ValueError(f"Unknown merge type: {target} (top|bottom|dress).")
    warnings: list[str] = []
    pieces: dict[int, list[GarmentPiece]] = {}
    for number, raw in sources.items():
        valid = [p for p in (_piece(item, warnings) for item in (raw if isinstance(raw, list) else [raw])) if p]
        # Same-part pieces first so a photo lends its own top/skirt before a look-alike.
        pieces[int(number)] = sorted(valid, key=lambda p: p.part != target)
    base_piece = next((p for p in pieces.get(base, []) if p.part == target), None)
    if base_piece is None:
        raise ValueError(f"Base photo {base} has no readable {target}.")
    kind, features = base_piece.kind, dict(base_piece.features)
    provenance: dict[str, int] = {}
    groups: list[dict[str, Any]] = []
    for group_id, label, keys in MERGE_GROUPS[target]:
        lenders = {n: next((p for p in ([base_piece] if n == base else ps) if compatible(keys, p.part, target)), None)
                   for n, ps in pieces.items()}
        options = sorted(n for n, p in lenders.items() if p is not None)
        source = int(assign.get(group_id, base))
        if source not in options:
            raise ValueError(f"Photo {source} has no '{label}' part usable for {target}.")
        lender = lenders[source]
        for key in keys:
            if key == "kind":
                kind = lender.kind
            else:
                features[key] = lender.features[key]
        provenance[group_id] = source
        groups.append({"id": group_id, "label": label, "keys": list(keys), "sources": options, "chosen": source,
                       "lenders": {str(n): p.part for n, p in lenders.items() if p is not None}})
    merged = _piece({"part": target, "kind": kind, "features": features}, warnings)
    assert merged is not None  # target is a valid part, checked above
    return Merge(merged, provenance, groups, plan_piece(merged.part, merged.kind, merged.features), warnings)
