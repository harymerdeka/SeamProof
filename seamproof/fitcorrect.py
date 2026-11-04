"""Toile fit correction: issues (typed notes or vision) become bounded params.

The rule table decides every number; the vision model only names issues with
severity and evidence. Photos require OpenCV for face blurring first: without
it, photos are refused and only typed notes are accepted. v2 always passes
through edit-locality, repair_loop and the verifier before export.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from seamproof.draft import REGISTRY
from seamproof.fit_rules import LABEL_ID, RULES, delta_for
from seamproof.locality import edit_locality
from seamproof.params import GARMENT_PARAMS, PARAMETERS
from seamproof.pattern import Pattern
from seamproof.repair import repair_loop
from seamproof.verify import Report, verify_pattern

# Deterministic Indonesian keyword mapping for typed notes, in match order.
NOTES_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pulling_at_underarm", ("ketiak", "underarm", "armpit")),
    ("sleeve_cap_puckering", ("puncak lengan", "cap mengerut", "mengerut", "sleeve cap", "pucker")),
    ("shoulder_seam_drops", ("bahu turun", "bahu jatuh", "shoulder")),
    ("neckline_gapes", ("leher menganga", "leher longgar", "gapes", "neckline gap", "neck gap")),
    ("body_too_tight_chest", ("dada sempit", "dada ketat", "bust sempit", "sempit di dada", "tight at the chest", "tight chest",
                              "chest tight", "chest too tight", "tight at the bust", "bust tight")),
    ("drag_lines_toward_bust", ("garis tarik", "drag lines", "drag line", "tarik ke dada")),
    ("dart_point_too_pointed", ("kupnat runcing", "ujung kupnat", "pointed", "dart tip", "dart point")),
    ("gaping_front_armhole", ("kerung depan longgar", "kerung longgar", "armhole gap", "gaping armhole", "loose armhole")),
    ("hem_swings_forward", ("hem mengayun", "hem ke depan", "swings", "hem swing")),
    ("tunic_flare_too_wide", ("flare", "terlalu lebar", "kelebaran", "too wide")),
)

WONDER_MODEL = "muse-wonder-1.0"
# Same body change, named per block family (t-shirt -> women's bodice).
PARAM_ALIASES = {"neck_drop_front": "neckline_depth_front"}


@dataclass
class Issue:
    """One fit issue: rule key (or 'other'), location, severity 1-3, evidence."""

    key: str
    location: str
    severity: int
    evidence: str


@dataclass
class FitProposal:
    """The full v1->v2 correction record for tailor confirmation."""

    label: str = LABEL_ID
    issues: list[Issue] = field(default_factory=list)
    changes: dict[str, float] = field(default_factory=dict)
    clamped: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    locality_passed: bool | None = None
    v1_failures: list[str] = field(default_factory=list)
    v2_report: Report | None = None
    v2_logs: list[Any] = field(default_factory=list)
    verified: bool = False


def _severity_of(note: str) -> int:
    """Read severity from Indonesian or English intensifiers; user numbers stay evidence only."""
    lowered = note.lower()
    if any(word in lowered for word in ("sangat", "sekali", "banget", "very ", "severe", "extremely", "badly")):
        return 3
    if any(word in lowered for word in ("sedikit", "agak", "kurang", "slightly", "a little", "a bit")):
        return 1
    return 2


def notes_to_issues(notes: list[str]) -> list[Issue]:
    """Map typed Indonesian or English notes to rule issues deterministically."""
    issues = []
    for note in notes:
        lowered = note.lower()
        hit = next((key for key, words in NOTES_KEYWORDS if any(w in lowered for w in words)), "other")
        issues.append(Issue(hit, note.strip()[:80], _severity_of(note), note.strip()[:200]))
    return issues


NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"issues": {"type": "array", "items": {
        "type": "object",
        "properties": {"issue": {"type": "string"}, "location": {"type": "string"},
                       "severity": {"type": "integer"}, "evidence": {"type": "string"}},
        "required": ["issue", "location", "severity", "evidence"]}}},
    "required": ["issues"],
}


def notes_to_issues_muse(notes: list[str],
                         llm_fn: Callable[[str, str], dict[str, Any]] | None = None) -> tuple[list[Issue], str]:
    """Muse Spark reads free toile notes into rule keys; keyword mapping is the fallback.

    The model only names issues and a 1-3 severity (interpreting input); the
    rule table still sizes every change. Returns (issues, "muse"|"keywords").
    """
    from seamproof import llm as llm_module

    keys = ", ".join(f"{key} ({rule.description_id})" for key, rule in sorted(RULES.items()))
    system = ("You read toile notes from a tailor (Indonesian or English). Map each problem to ONE key from this list, "
              f"or 'other': {keys}. severity 1 (slight), 2 (moderate), 3 (severe), judged from the words. "
              "Do not write correction amounts. Write location in English; evidence quotes the note. JSON: "
              '{"issues": [{"issue": "...", "location": "...", "severity": 2, "evidence": "quote from the note"}]}.')
    user = json.dumps({"notes": notes}, ensure_ascii=False)
    try:
        raw = llm_fn(system, user) if llm_fn is not None else llm_module.call_json(system, user, NOTES_SCHEMA,
                                                                                   "toile-notes")
        items = raw["issues"]
        issues = []
        for item in items:
            key = str(item.get("issue", "other"))
            severity = item.get("severity", 2)
            if isinstance(severity, bool) or not isinstance(severity, int):
                severity = 2
            issues.append(Issue(key if key in RULES else "other", str(item.get("location", ""))[:80],
                                min(3, max(1, severity)), str(item.get("evidence", ""))[:200]))
        if not issues:
            raise ValueError("no issues")
        return issues, "muse"
    except Exception:  # model down or unusable: deterministic keywords
        return notes_to_issues(notes), "keywords"


def _blur_faces(image_bytes: bytes) -> bytes:
    """Blur detected faces in place; raise when no face module is available."""
    import cv2
    import numpy as np

    data = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError("The photo cannot be read — check the file.")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for (x, y, w, h) in cascade.detectMultiScale(data, 1.1, 4):
        face = data[y:y + h, x:x + w]
        data[y:y + h, x:x + w] = cv2.GaussianBlur(face, (51, 51), 0)
    ok, encoded = cv2.imencode(".jpg", data)
    if not ok:
        raise ValueError("The photo could not be processed.")
    return bytes(encoded)


def _vision_request(model: str, images_b64: list[str]) -> list[Issue]:
    """Ask the vision model for issues; validate shape, retry handled by caller."""
    from openai import OpenAI

    from seamproof.config import load_settings

    settings = load_settings()
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=settings.model_api_key)
    content: list[dict[str, Any]] = [
        {"type": "input_text",
         "text": ("Look at the toile photos (faces are already blurred). Reply with JSON only: "
                  '{"issues": [{"issue": "<one of: ' + ", ".join(sorted(RULES))
                  + ', other>", "location": "...", "severity": 1|2|3, "evidence": "..."}]}. '
                  "Do not compute numbers; severity comes from how severe it looks. Write location and evidence in English.")}]
    content.extend({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                   for b64 in images_b64)
    response = client.responses.create(model=model, input=[{"type": "message", "role": "user",
                                                            "content": content}])
    data = json.loads(response.output_text.strip().strip("`"))
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        raise ValueError("The vision response is not issues JSON.")
    issues = []
    for item in data["issues"]:
        key = str(item.get("issue", "other"))
        severity = int(item.get("severity", 2))
        if key not in RULES:
            key = "other"
        issues.append(Issue(key, str(item.get("location", ""))[:80],
                            min(3, max(1, severity)), str(item.get("evidence", ""))[:200]))
    return issues


def _vision_issues(images_b64: list[str]) -> list[Issue]:
    """Try Wonder, fall back to MUSE_MODEL once; raise when both fail."""
    from seamproof.config import load_settings

    try:
        return _vision_request(WONDER_MODEL, images_b64)
    except Exception:
        fallback = load_settings().muse_model or ""
        if not fallback or fallback == WONDER_MODEL:
            raise
        return _vision_request(fallback, images_b64)


def _apply(issues: list[Issue], base: Mapping[str, float]) -> tuple[dict[str, float], list[str], list[str]]:
    """Sum rule deltas, clamp to param bounds; return (changes, clamped, advice)."""
    total: dict[str, float] = {}
    advice: list[str] = []
    for issue in issues:
        rule = RULES.get(issue.key)
        if rule is None:
            advice.append(f"{issue.location or issue.key}: no rule — check by hand.")
            continue
        if rule.advice_id:
            advice.append(f"{rule.description_id}: {rule.advice_id}")
        for param, step in delta_for(rule, issue.severity).items():
            total[param] = total.get(param, 0.0) + step
    changes, clamped = {}, []
    for param, step in total.items():
        spec = PARAMETERS[param]
        value = float(base.get(param, spec.default)) + step
        if value < spec.min or value > spec.max:
            clamped.append(f"{param} clamped to the {spec.min}–{spec.max} range")
            value = min(spec.max, max(spec.min, value))
        changes[param] = value
    return changes, clamped, advice


def fit_correct(pattern: Pattern, photos: list[str | Path] | None = None,
                notes: list[str] | None = None,
                vision_fn: Callable[[list[str]], list[Issue]] | None = None,
                blurred_b64: list[str] | None = None,
                note_issues: list[Issue] | None = None) -> tuple[FitProposal, Pattern | None]:
    """Correct v1 into a verified v2; return (proposal, v2-or-None).

    Photos need OpenCV for face blurring and are refused without it.
    blurred_b64 are JPEGs already face-blurred by vision.prepare_photo (web UI).
    note_issues, when given, replace the keyword reading of notes (Muse read them).
    """
    proposal = FitProposal()
    issues = list(note_issues) if note_issues is not None else notes_to_issues(list(notes or []))
    if photos:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ValueError("Photos need OpenCV for face blurring — it is not installed. "
                             "Send written notes only, without faces.")
        blurred = [base64.b64encode(_blur_faces(Path(p).read_bytes())).decode("ascii")
                   for p in photos]
        issues.extend(vision_fn(blurred) if vision_fn is not None else _vision_issues(blurred))
    if blurred_b64:
        issues.extend(vision_fn(blurred_b64) if vision_fn is not None else _vision_issues(blurred_b64))
    proposal.issues = issues
    changes, clamped, advice = _apply(issues, pattern.params)
    # A rule may name a block parameter this garment does not draft; it would be a no-op.
    # The centre-front neck drop has one name per block family, so it is renamed first.
    drafted = set(GARMENT_PARAMS.get(pattern.garment_type, PARAMETERS))
    for param, alias in PARAM_ALIASES.items():
        if param in changes and param not in drafted and alias in drafted:
            step = changes.pop(param) - PARAMETERS[param].default
            spec = PARAMETERS[alias]
            changes[alias] = min(spec.max, max(spec.min, float(pattern.params.get(alias, spec.default)) + step))
    for param in sorted(set(changes) - drafted):
        advice.append(f"{param} is not used by the {pattern.garment_type} pattern — fix it by hand on the toile.")
        del changes[param]
    proposal.changes, proposal.clamped, proposal.advice = changes, clamped, advice
    garment = pattern.garment_type
    if garment not in REGISTRY:
        raise ValueError(f"Unknown garment: {garment}")
    merged = dict(pattern.params)
    merged.update(changes)

    def reject(system: str, user: str) -> dict[str, Any]:
        raise RuntimeError("fitcorrect without LLM: v2 as it is")

    candidate = REGISTRY[garment](pattern.measurements, pattern.fabric, merged)
    locality = edit_locality(pattern, candidate, changes)
    proposal.locality_passed = locality.passed
    if not locality.passed:
        proposal.v1_failures = [c.id for c in verify_pattern(pattern).failures()]
        return proposal, None
    _v2, v2_report, logs = repair_loop(pattern.measurements, pattern.fabric, merged,
                                       garment=garment, llm_fn=reject)
    proposal.v2_report, proposal.v2_logs = v2_report, logs
    proposal.verified = v2_report.passed
    proposal.v1_failures = [c.id for c in verify_pattern(pattern).failures()]
    return proposal, _v2


def fit_md(proposal: FitProposal, v1: Pattern, v2_label: str) -> str:
    """Render issues, the param diff and the v2 verdict as Markdown."""
    lines = [f"# Fit Correction — {v1.garment_type} {v1.size_label} → {v2_label}", "",
             proposal.label, "",
             "| Issue | Location | Severity | Evidence |",
             "|---------|--------|-------|-------|"]
    for issue in proposal.issues:
        name = RULES[issue.key].description_id if issue.key in RULES else issue.key
        lines.append(f"| {name} | {issue.location} | {issue.severity} | {issue.evidence} |")
    lines += ["", "| Parameter | v1 | v2 |", "|-----------|----|----|"]
    for param in sorted(proposal.changes):
        lines.append(f"| {param} | {float(v1.params.get(param, 0.0)):.1f} | "
                     f"{proposal.changes[param]:.1f} |")
    for note in proposal.clamped + proposal.advice:
        lines.append(f"- {note}")
    lines.append(f"- Edit locality: {'passed' if proposal.locality_passed else 'FAILED'}")
    if proposal.v2_report is not None:
        lines.append(f"- v2: {proposal.v2_report.summary_id()}")
    lines += [""]
    return "\n".join(lines)
