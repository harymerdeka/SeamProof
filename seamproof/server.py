"""Minimal local web UI backend: one page, JSON API, downloads.

Binds to 127.0.0.1 only; the user reaches it through VS Code port
forwarding. No login, no database, no cloud storage: finished packs live in
a small in-memory job store (latest 50) that vanishes on restart.
"""

from __future__ import annotations

import io
import json
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StrictInt

from seamproof.cli import DEFAULT_FABRIC
from seamproof.allowance import ensure_allowance
from seamproof.explain import explain
from seamproof.export import export_pdf_tiled, export_report_md, export_svg
from seamproof.fabric import FABRIC_EASE
from seamproof.intake import GARMENT_FIELDS, _build_result, parse_text
from seamproof.measurements import SIZE_TABLE, TShirtMeasurements, WOMEN_SIZE_TABLE, WomenMeasurements
from seamproof.repair import repair_loop, rounds_table_md
from seamproof.style import LABELS, StylePlan, apply_proportions, plan_piece

GARMENTS = ["tshirt", "blouse", "tunic", "shirt", "skirt", "dress"]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_JOBS = 50

app = FastAPI(title="SeamProof")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
# Production runs (report + markdown); marker PDFs are rendered on download.
PRODUCTIONS: dict[str, dict[str, Any]] = {}


def _table(garment: str):
    """Return the size table for a garment."""
    return SIZE_TABLE if garment == "tshirt" else WOMEN_SIZE_TABLE


def _check_garment(garment: str) -> None:
    """Reject unknown garments listing what exists."""
    if garment not in GARMENTS:
        raise HTTPException(422, f"Unknown garment: {garment} (choose {', '.join(GARMENTS)}).")


def _check_fabric(fabric: str) -> None:
    """Reject unknown fabric categories."""
    if fabric not in FABRIC_EASE:
        raise HTTPException(422, f"Unknown fabric: {fabric}.")


def _store(pdf: bytes | None, data: dict[str, Any], svg: str, md: str,
           pattern=None, report=None, explanation=None) -> str:
    """Keep one finished pack in memory and return its job id."""
    job = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job] = {"pdf": pdf, "json": data, "md": md, "svg": svg, "dxf": None,
                     "pattern": pattern, "report": report, "explanation": explanation, "signoff": None}
        while len(JOBS) > MAX_JOBS:
            JOBS.pop(next(iter(JOBS)))
    return job


def _render_docs(pattern, report, explanation) -> tuple[bytes | None, str]:
    """PDF (only when passed) and report.md; both carry the sign-off if any."""
    md_stream = io.BytesIO()
    export_report_md(report, md_stream, explanation)
    pdf: bytes | None = None
    if report.passed:
        pdf_stream = io.BytesIO()
        export_pdf_tiled(pattern, report, pdf_stream, explanation=explanation)
        pdf = pdf_stream.getvalue()
    return pdf, md_stream.getvalue().decode("utf-8")


def _files(job: str, entry: dict[str, Any]) -> dict[str, str | None]:
    """Download links; DXF only after a pattern maker signed the pack off."""
    return {"pdf": f"/api/download/{job}/pdf" if entry["pdf"] else None,
            "dxf": f"/api/download/{job}/dxf" if entry["dxf"] else None,
            "json": f"/api/download/{job}/json",
            "md": f"/api/download/{job}/md",
            "svg": f"/api/download/{job}/svg"}


def _pack_response(pattern, report, logs) -> dict[str, Any]:
    """Serialize a finished pack and store its downloads."""
    svg_stream = io.BytesIO()
    export_svg(pattern, svg_stream)
    summary, instructions = explain(report, pattern)
    pdf, md = _render_docs(pattern, report, (summary, instructions))
    job = _store(pdf, json.loads(json.dumps(ensure_allowance(pattern).to_dict())),
                 svg_stream.getvalue().decode("utf-8"), md,
                 pattern, report, (summary, instructions))
    return {
        "job": job,
        "passed": report.passed,
        "summary": report.summary_id(),
        "checks": [{"id": c.id, "name": c.name_id, "passed": c.passed, "actual": c.actual,
                    "expected": c.expected, "tolerance": c.tolerance, "message": c.message_id}
                   for c in report.checks],
        "rounds": [asdict(log) for log in logs],
        "rounds_md": rounds_table_md(logs),
        "explanation": {"summary": summary,
                        "instructions": [asdict(item) for item in instructions]},
        "files": _files(job, JOBS[job]),
        "signoff": None,
    }


class SignOff(BaseModel):
    """A pattern maker's check of one verified pack."""

    name: str = Field(min_length=2, max_length=60)
    note: str = Field("", max_length=300)


@app.post("/api/check/{job}")
def check(job: str, payload: SignOff) -> dict[str, Any]:
    """Record the pattern maker's check, re-render the proof docs and unlock DXF.

    Only verified packs can be signed off. Kept in memory with the pack (no
    database); the name and date are printed on the proof page and report.md.
    """
    from datetime import datetime, timezone

    from seamproof.export import export_dxf

    entry = JOBS.get(job)
    if entry is None:
        raise HTTPException(404, "Pack not found (was the server just restarted?).")
    report = entry["report"]
    if not report.passed:
        raise HTTPException(409, "Only a pattern that passed verification can be checked by a pattern maker.")
    signoff = {"name": payload.name.strip(), "note": payload.note.strip(),
               "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    report.meta["signoff"] = signoff
    pdf, md = _render_docs(entry["pattern"], report, entry["explanation"])
    dxf_stream = io.BytesIO()
    export_dxf(entry["pattern"], report, dxf_stream)
    with JOBS_LOCK:
        entry.update(pdf=pdf, md=md, dxf=dxf_stream.getvalue(), signoff=signoff)
    return {"signoff": signoff, "files": _files(job, entry)}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/fields")
def fields(garment: str) -> dict[str, Any]:
    """List measurement fields, sizes and fabrics for a garment."""
    _check_garment(garment)
    table = _table(garment)
    # Size-table values in cm so the form can start from a size and override.
    size_values = {size: {k: round(v / 10, 1) for k, v in asdict(m).items()}
                   for size, m in table.items()}
    return {"fields": GARMENT_FIELDS[garment], "sizes": list(table),
            "size_values": size_values,
            "fabrics": sorted(FABRIC_EASE), "default_fabric": DEFAULT_FABRIC[garment]}


@app.post("/api/pack")
def pack(payload: dict[str, Any]) -> dict[str, Any]:
    """Draft, verify, correct and explain one pack; store its downloads."""
    garment = str(payload.get("garment", ""))
    _check_garment(garment)
    fabric = str(payload.get("fabric") or DEFAULT_FABRIC[garment])
    _check_fabric(fabric)
    mode = str(payload.get("mode", "size"))
    if mode == "size":
        size = str(payload.get("size", "M"))
        if size not in _table(garment):
            raise HTTPException(422, f"Unknown size: {size}.")
        measurements = _table(garment)[size]
    elif mode == "text":
        try:
            result = parse_text(str(payload.get("text", "")), garment)
        except Exception as exc:
            raise HTTPException(502, f"Text mapping failed ({type(exc).__name__}); enter the measurements by hand.") from exc
        if result["missing"]:
            raise HTTPException(422, {"missing": result["missing"], "warnings": result["warnings"]})
        cls = TShirtMeasurements if garment == "tshirt" else WomenMeasurements
        try:
            measurements = cls(**result["measurements"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, f"Invalid measurements: {exc}.") from exc
    elif mode == "measures":
        unit = str(payload.get("unit", "cm"))
        values = payload.get("measures") or {}
        if not isinstance(values, dict):
            raise HTTPException(422, "measures must be an object {field: value}.")
        result = _build_result([(k, v, unit) for k, v in values.items()], garment)
        if result["missing"]:
            raise HTTPException(422, {"missing": result["missing"], "warnings": result["warnings"]})
        cls = TShirtMeasurements if garment == "tshirt" else WomenMeasurements
        try:
            measurements = cls(**result["measurements"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, f"Invalid measurements: {exc}.") from exc
    else:
        raise HTTPException(422, f"Unknown mode: {mode} (size|text|measures).")
    plan = _style_plan(payload.get("style"), garment)
    try:
        pattern, report, logs = repair_loop(measurements, fabric, plan.params if plan else {},
                                            garment=garment)
    except ValueError as exc:
        # The draft itself refuses some in-range combinations (e.g. waist vs bust).
        raise HTTPException(422, f"These measurements cannot be drafted: {exc}") from exc
    response = _pack_response(pattern, report, logs)
    if plan is not None:
        response["style"] = {"params": plan.params, "notes": plan.notes}
    return response


def _style_plan(raw: Any, garment: str) -> StylePlan | None:
    """Map photo style categories to params; the garment must match the plan."""
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("features"), dict):
        raise HTTPException(422, "style must be {part, kind, features}.")
    from seamproof.vision import validate_reading

    reading = validate_reading({"pieces": [{"part": raw.get("part"), "kind": raw.get("kind"),
                                            "features": raw["features"]}]})
    if not reading.pieces:
        raise HTTPException(422, "Unknown style part (top|bottom|dress).")
    piece = reading.pieces[0]
    plan = plan_piece(piece.part, piece.kind, piece.features)
    if plan.garment != garment:
        raise HTTPException(422, f"This style maps to {plan.garment}, not {garment}.")
    plan.notes = reading.warnings + plan.notes
    _apply_proportions(plan, raw.get("proportions"))
    return plan


def _apply_proportions(plan: StylePlan, proportions: Any) -> None:
    """Apply fitted proportions when sent; invalid ones are a client error."""
    if proportions is None:
        return
    try:
        apply_proportions(plan, proportions)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _plan_dict(plan: StylePlan, color: str | None = None) -> dict[str, Any]:
    """Serialize a style plan for the UI, with the flat and 3D preview of what will be drafted."""
    from seamproof.flat import drafted_features, plan_flat
    from seamproof.preview3d import plan_preview

    return {"garment": plan.garment, "label": LABELS.get("vest" if plan.kind == "vest" else plan.garment or "", "Not supported yet"),
            "params": plan.params, "proportions": plan.proportions, "notes": plan.notes,
            "drafted": drafted_features(plan),
            "flat": plan_flat(plan, color), "preview3d": plan_preview(plan, color)}


@app.post("/api/style-plan")
def style_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Re-map edited categories (no model call) so the UI shows what will be drafted."""
    from seamproof.vision import validate_reading

    reading = validate_reading({"pieces": [payload]})
    if not reading.pieces:
        raise HTTPException(422, "Unknown style part (top|bottom|dress).")
    piece = reading.pieces[0]
    plan = plan_piece(piece.part, piece.kind, piece.features)
    plan.notes = reading.warnings + plan.notes
    _apply_proportions(plan, payload.get("proportions"))
    color = payload.get("color") if isinstance(payload.get("color"), str) else None
    return {"kind": piece.kind, "features": piece.features, "plan": _plan_dict(plan, color)}


class SketchFitRequest(BaseModel):
    """A design's categories and the sketch its proportions should follow."""

    part: str
    kind: str
    features: dict[str, Any]
    svg: str = Field(min_length=10, max_length=200_000)
    color: str | None = None


@app.post("/api/sketch-fit")
def sketch_fit(payload: SketchFitRequest) -> dict[str, Any]:
    """Fit skirt or dress flare and length to a sketch (code only, no model call)."""
    from seamproof.sketchfit import FIT_GARMENTS, fit_to_sketch
    from seamproof.vision import sanitize_sketch, validate_reading

    reading = validate_reading({"pieces": [{"part": payload.part, "kind": payload.kind, "features": payload.features}]})
    if not reading.pieces:
        raise HTTPException(422, "Unknown style part (top|bottom|dress).")
    piece = reading.pieces[0]
    plan = plan_piece(piece.part, piece.kind, piece.features)
    if plan.garment not in FIT_GARMENTS:
        raise HTTPException(422, "Sketch proportions are available for a skirt or dress only.")
    plan.notes = reading.warnings + plan.notes
    try:
        # Size M, like the category tables; the verifier re-checks the chosen size at drafting.
        fit = fit_to_sketch(sanitize_sketch(payload.svg), plan.garment, _table(plan.garment)["M"],
                            DEFAULT_FABRIC[plan.garment], plan.params)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    before = {name: plan.params.get(name) for name in fit.params}
    apply_proportions(plan, fit.params)
    return {"kind": piece.kind, "features": piece.features, "before": before, "fit": asdict(fit),
            "plan": _plan_dict(plan, payload.color)}


@app.get("/api/design/catalog")
def design_catalog() -> dict[str, Any]:
    """Allowed categories and colours for the design editor (no model call)."""
    from seamproof.flat import COLORS
    from seamproof.vision import BOTTOM_FEATURES, BOTTOM_KINDS, DRESS_FEATURES, DRESS_KINDS, TOP_FEATURES, TOP_KINDS

    return {"top": {"kinds": list(TOP_KINDS), **{k: list(v) for k, v in TOP_FEATURES.items()}},
            "bottom": {"kinds": list(BOTTOM_KINDS), **{k: list(v) for k, v in BOTTOM_FEATURES.items()}},
            "dress": {"kinds": list(DRESS_KINDS), **{k: list(v) for k, v in DRESS_FEATURES.items()}},
            "colors": {k: {"hex": hex_, "name": name} for k, (hex_, name) in COLORS.items()}}


class DesignBrief(BaseModel):
    """A written design brief from the customer or workshop."""

    brief: str = Field(min_length=3, max_length=2000)


@app.post("/api/design")
def design(payload: DesignBrief) -> dict[str, Any]:
    """Brief -> two validated, draftable design options (Muse Spark picks categories only)."""
    from seamproof import llm as llm_module
    from seamproof.design import design_from_brief

    try:
        options = design_from_brief(payload.brief)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"Muse could not create a design ({exc}). Try again or choose a style yourself.") from exc
    out = []
    for option in options:
        option.plan.notes = option.warnings + option.plan.notes
        out.append({"name": option.name, "part": option.piece.part, "kind": option.piece.kind,
                    "features": option.piece.features, "color": option.color, "fabric": option.fabric,
                    "reason": option.reason, "unsupported": option.unsupported,
                    "plan": _plan_dict(option.plan, option.color)})
    return {"options": out}


class PhotoMerge(BaseModel):
    """Which photo each garment part comes from; notes are optional (Muse Spark)."""

    part: str
    base: StrictInt
    sources: dict[str, list[dict[str, Any]] | dict[str, Any]] = Field(min_length=1, max_length=3)
    assign: dict[str, StrictInt] = Field(default_factory=dict)
    color: str | None = None
    notes: str = Field("", max_length=300)


@app.post("/api/design/merge")
def design_merge(payload: PhotoMerge) -> dict[str, Any]:
    """Merge parts of several photo readings into one design (code), then apply notes (Muse)."""
    from seamproof import llm as llm_module
    from seamproof.design import merge_photos, refine_design

    try:
        sources = {int(k): v for k, v in payload.sources.items()}
        merged = merge_photos(payload.part, payload.base, sources, payload.assign)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    piece, plan, color = merged.piece, merged.plan, payload.color
    notes, changes, reason, unsupported = list(merged.warnings), [], "", []
    if payload.notes.strip():
        try:
            result = refine_design({"part": piece.part, "kind": piece.kind, "features": piece.features,
                                    "color": color}, payload.notes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except llm_module.LLMError as exc:
            raise HTTPException(502, f"Muse could not apply the notes ({exc}). Merge without notes.") from exc
        piece, plan, color = result.piece, result.plan, result.color
        notes += result.warnings
        changes, reason, unsupported = result.changes, result.reason, result.unsupported
    plan.notes = notes + plan.notes
    return {"part": piece.part, "kind": piece.kind, "features": piece.features, "color": color,
            "provenance": merged.provenance, "groups": merged.groups, "changes": changes, "reason": reason,
            "unsupported": unsupported,
            "plan": _plan_dict(plan, color)}


class DesignEdit(BaseModel):
    """One free-text change to the current design."""

    design: dict[str, Any]
    instruction: str = Field(min_length=3, max_length=300)


@app.post("/api/design/refine")
def design_refine(payload: DesignEdit) -> dict[str, Any]:
    """Apply one change via Muse; the diff and the plan are computed by code."""
    from seamproof import llm as llm_module
    from seamproof.design import refine_design

    try:
        result = refine_design(payload.design, payload.instruction)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"Muse could not apply the change ({exc}). Use the quick buttons.") from exc
    result.plan.notes = result.warnings + result.plan.notes
    return {"part": result.piece.part, "kind": result.piece.kind, "features": result.piece.features,
            "color": result.color, "reason": result.reason, "changes": result.changes,
            "unsupported": result.unsupported, "plan": _plan_dict(result.plan, result.color)}


@app.post("/api/photo")
def photo(file: UploadFile = File(...)) -> dict[str, Any]:
    """Read garment style from a photo: blur faces, ask Muse Wonder, map to a plan.

    Body millimetres never come from this photo (no scale reference on a worn
    photo); the UI asks for a size or own measurements next.
    """
    from seamproof import llm as llm_module
    from seamproof.vision import (BOTTOM_FEATURES, BOTTOM_KINDS, DRESS_FEATURES, DRESS_KINDS, TOP_FEATURES,
                                  TOP_KINDS, describe_garments, prepare_photo)

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The photo is over 15 MB — make it smaller first.")
    try:
        prepared = prepare_photo(data)
    except ImportError as exc:
        raise HTTPException(503, "OpenCV is not installed: pip install opencv-python-headless") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        reading = describe_garments(prepared)
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The vision model failed ({exc}). Try again or fill in the style by hand.") from exc
    import base64

    return {
        "filename": file.filename or "photo",
        "faces_blurred": prepared.faces_blurred,
        "preview": "data:image/jpeg;base64," + base64.b64encode(prepared.jpeg).decode("ascii"),
        "photo_view": reading.photo_view,
        "photo_side": reading.photo_side,
        "warnings": reading.warnings,
        "pieces": [{"part": piece.part, "kind": piece.kind, "features": piece.features,
                    "evidence": piece.evidence,
                    "plan": _plan_dict(plan_piece(piece.part, piece.kind, piece.features))}
                   for piece in reading.pieces],
        "options": {"top": {"kinds": list(TOP_KINDS), **{k: list(v) for k, v in TOP_FEATURES.items()}},
                    "bottom": {"kinds": list(BOTTOM_KINDS), **{k: list(v) for k, v in BOTTOM_FEATURES.items()}},
                    "dress": {"kinds": list(DRESS_KINDS), **{k: list(v) for k, v in DRESS_FEATURES.items()}}},
    }


@app.post("/api/photo/sketch")
def photo_sketch(file: UploadFile = File(...)) -> dict[str, str]:
    """Muse Wonder draws a reference flat from a photo; it never feeds the pattern."""
    from seamproof import llm as llm_module
    from seamproof.vision import sketch_from_photo

    prepared = _prepared_upload(file)
    try:
        return {"svg": sketch_from_photo(prepared)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The vision model could not draw the sketch ({exc}).") from exc


@app.post("/api/photo/sketch-merge")
def photo_sketch_merge(files: list[UploadFile] = File(...), garment: str = Form(...), parts: str = Form(...),
                       base: int = Form(...), notes: str = Form("")) -> dict[str, str]:
    """Muse Wonder draws one reference flat from the chosen part of each photo."""
    import json

    from seamproof import llm as llm_module
    from seamproof.vision import merge_sketch

    if not 1 <= len(files) <= 3:
        raise HTTPException(422, "Send 1 to 3 photos.")
    try:
        chosen = [(str(item["label"])[:80], int(item["photo"])) for item in json.loads(parts)]
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, "The list of parts is not valid.") from exc
    if not chosen or len(chosen) > 20:
        raise HTTPException(422, "The list of parts is not valid.")
    photos = [_prepared_upload(file) for file in files]
    try:
        return {"svg": merge_sketch(photos, garment[:80], chosen, base, notes[:300])}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The vision model could not draw the merged sketch ({exc}).") from exc


class ProductionRequest(BaseModel):
    """Validated production input; bad values become 422, never 500."""

    garment: str
    fabric: str | None = None
    run: dict[str, StrictInt] = Field(min_length=1)
    width: float = Field(1500, gt=0, le=10000, allow_inf_nan=False)
    price: float = Field(0, ge=0, le=1e9, allow_inf_nan=False)
    layers: StrictInt = Field(1, ge=1, le=500)
    style: dict[str, Any] | None = None


@app.post("/api/production")
def production(payload: ProductionRequest) -> dict[str, Any]:
    """Pack a size run; return totals, markers and the markdown pack."""
    from seamproof.export import export_marker_svg, marker_plot_page_mm, marker_tile_count
    from seamproof.production import produce, production_md

    garment = payload.garment
    _check_garment(garment)
    fabric = payload.fabric or DEFAULT_FABRIC[garment]
    _check_fabric(fabric)
    for size in payload.run:
        if size not in _table(garment):
            raise HTTPException(422, f"Unknown size: {size}.")
    spec = ",".join(f"{s}={q}" for s, q in payload.run.items())
    plan = _style_plan(payload.style, garment)
    try:
        report = produce(garment, spec, fabric, payload.width, payload.price,
                         layers=payload.layers, params=plan.params if plan else None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    label = LABELS.get(garment, garment) + (" (photo)" if plan else "")
    md = production_md(report)
    pid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        PRODUCTIONS[pid] = {"report": report, "md": md, "label": label}
        while len(PRODUCTIONS) > MAX_JOBS:
            PRODUCTIONS.pop(next(iter(PRODUCTIONS)))
    svgs = {}
    for item in report.sizes:
        stream = io.BytesIO()
        export_marker_svg(item.marker, f"{label} {item.size} — cutting marker", stream, item.size)
        svgs[item.size] = stream.getvalue().decode("utf-8")
    base = f"/api/production/{pid}"
    return {"job": pid, "summary": report.summary_id(),
            "sizes": [{"size": i.size, "qty": i.qty, "verified": i.verified,
                       "length_mm": round(i.marker.used_length_mm, 1), "metres": i.metres,
                       "utilisation": i.marker.utilisation, "pieces": len(i.marker.placements),
                       "a4_sheets": marker_tile_count(i.marker),
                       "plot_page_mm": marker_plot_page_mm(i.marker)} for i in report.sizes],
            "cut_list": report.cut_list, "total_metres": report.total_metres,
            "total_cost": report.total_cost,
            "overall_utilisation": report.overall_utilisation,
            "md": md, "markers": svgs,
            "files": {"pdf": f"{base}/pdf", "md": f"{base}/md",
                      "markers": {i.size: {"plot": f"{base}/marker/{i.size}/plot" if i.verified else None,
                                           "a4": f"{base}/marker/{i.size}/a4" if i.verified else None,
                                           "svg": f"{base}/marker/{i.size}/svg"}
                                  for i in report.sizes}}}


@app.get("/api/production/{pid}/{kind}")
def production_file(pid: str, kind: str) -> Response:
    """Production summary PDF (totals + A4 overview per size) or markdown."""
    from seamproof.export import export_production_pdf

    entry = PRODUCTIONS.get(pid)
    if entry is None or kind not in ("pdf", "md"):
        raise HTTPException(404, "File not found (was the server just restarted?).")
    if kind == "md":
        return Response(entry["md"].encode("utf-8"), media_type="text/markdown",
                        headers={"Content-Disposition": "attachment; filename=production.md"})
    stream = io.BytesIO()
    export_production_pdf(entry["report"], stream)
    return Response(stream.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=production.pdf"})


@app.get("/api/production/{pid}/marker/{size}/{fmt}")
def marker_file(pid: str, size: str, fmt: str) -> Response:
    """One size's marker: plot (1:1 plotter PDF), a4 (1:1 tiled PDF) or svg."""
    from seamproof.export import export_marker_plot_pdf, export_marker_svg, export_marker_tiles_pdf

    entry = PRODUCTIONS.get(pid)
    item = None if entry is None else next((i for i in entry["report"].sizes if i.size == size), None)
    if item is None or fmt not in ("plot", "a4", "svg"):
        raise HTTPException(404, "File not found (was the server just restarted?).")
    title = f"{entry['label']} {size} — marker"
    stream = io.BytesIO()
    if fmt == "svg":
        export_marker_svg(item.marker, title, stream, size)
        return Response(stream.getvalue(), media_type="image/svg+xml",
                        headers={"Content-Disposition": f"attachment; filename=marker_{size}.svg"})
    if not item.verified:
        raise HTTPException(409, f"Size {size} has not passed verification — the 1:1 marker is not printed.")
    writer = export_marker_plot_pdf if fmt == "plot" else export_marker_tiles_pdf
    writer(item.marker, title, stream, size)
    name = f"marker_{size}_{'plotter' if fmt == 'plot' else 'A4'}.pdf"
    return Response(stream.getvalue(), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


def _prepared_upload(file: UploadFile):
    """Read one upload, enforce the size cap and blur faces (before any model call)."""
    from seamproof.vision import prepare_photo

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The photo is over 15 MB — make it smaller first.")
    try:
        return prepare_photo(data)
    except ImportError as exc:
        raise HTTPException(503, "OpenCV is not installed: pip install opencv-python-headless") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class FabricAsk(BaseModel):
    """Ask Muse to pick a fabric among categories the verifier accepts for this design."""

    garment: str
    style: dict[str, Any] | None = None
    brief: str = Field("", max_length=2000)  # same cap as the design brief (design.MAX_BRIEF_CHARS)


@app.post("/api/fabric-suggest")
def fabric_suggest(payload: FabricAsk) -> dict[str, Any]:
    """Code verifies an M draft per fabric; Muse Spark picks one of those and says why."""
    from seamproof.draft import REGISTRY
    from seamproof.narrate import suggest_fabric
    from seamproof.verify import verify_pattern

    garment = payload.garment
    _check_garment(garment)
    plan = _style_plan(payload.style, garment)
    params = plan.params if plan else {}
    candidates = []
    for fabric in FABRIC_EASE:
        try:
            if verify_pattern(REGISTRY[garment](_table(garment)["M"], fabric, dict(params))).passed:
                candidates.append(fabric)
        except ValueError:
            continue  # this block cannot be drafted in that fabric at all
    if not candidates:
        raise HTTPException(422, "No fabric category passes verification for this design.")
    pick = suggest_fabric(garment, candidates, plan.features if plan else None, payload.brief)
    return {"fabric": pick.fabric, "reason": pick.reason, "source": pick.source, "candidates": candidates}


@app.post("/api/measure-photo")
def measure_photo(file: UploadFile = File(...), garment: str = Form(...)) -> dict[str, Any]:
    """Muse Wonder transcribes a photographed measurement note; intake rules check it."""
    from seamproof import llm as llm_module
    from seamproof.vision import read_measurement_photo

    _check_garment(garment)
    prepared = _prepared_upload(file)
    try:
        result = read_measurement_photo(prepared, garment)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except llm_module.LLMError as exc:
        raise HTTPException(502, f"The vision model could not read the note ({exc}). Type the measurements by hand.") from exc
    return {"measurements_cm": {k: round(v / 10, 1) for k, v in result["measurements"].items()},
            "missing": result["missing"], "warnings": result["warnings"],
            "faces_blurred": prepared.faces_blurred}


@app.get("/api/story/{job}")
def pattern_story(job: str) -> dict[str, Any]:
    """Muse Spark's verdict, sewing order and toile checks for a stored pack (cached)."""
    from seamproof.narrate import narrate_pattern

    entry = JOBS.get(job)
    if entry is None or entry.get("pattern") is None:
        raise HTTPException(404, "Pattern not found (was the server just restarted?).")
    if entry.get("story") is None:
        entry["story"] = asdict(narrate_pattern(entry["pattern"], entry["report"]))
    return entry["story"]


@app.get("/api/production-story/{pid}")
def production_story(pid: str) -> dict[str, Any]:
    """Muse Spark's cutting-room steps and owner summary for a production run (cached)."""
    from seamproof.narrate import narrate_production

    entry = PRODUCTIONS.get(pid)
    if entry is None:
        raise HTTPException(404, "Production pack not found (was the server just restarted?).")
    if entry.get("story") is None:
        entry["story"] = asdict(narrate_production(entry["report"]))
    return entry["story"]


@app.post("/api/toile/{job}")
def toile(job: str, notes: str = Form(""), files: list[UploadFile] | None = File(None)) -> dict[str, Any]:
    """Toile correction: Wonder names fit issues, rules size the change, v2 is redrafted and verified."""
    import base64

    from seamproof.fit_rules import RULES
    from seamproof.fitcorrect import fit_correct, notes_to_issues_muse

    entry = JOBS.get(job)
    if entry is None or entry.get("pattern") is None:
        raise HTTPException(404, "Pattern not found (was the server just restarted?).")
    files = [f for f in (files or []) if f.filename]
    lines = [line.strip() for line in notes.replace(";", "\n").splitlines() if line.strip()][:10]
    if not files and not lines:
        raise HTTPException(422, "Send a toile photo or write a note about the problem.")
    if len(files) > 3:
        raise HTTPException(422, "At most 3 toile photos.")
    blurred = [base64.b64encode(_prepared_upload(f).jpeg).decode("ascii") for f in files]
    v1 = entry["pattern"]
    try:
        note_issues, notes_source = notes_to_issues_muse(lines) if lines else ([], "")
        proposal, v2 = fit_correct(v1, blurred_b64=blurred, note_issues=note_issues)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # the vision model failed after its fallback
        raise HTTPException(502, f"The vision model could not read the toile ({type(exc).__name__}). "
                                 "Write a note about the problem instead.") from exc
    response: dict[str, Any] = {
        "issues": [{"key": i.key, "name": RULES[i.key].description_id if i.key in RULES else "Other",
                    "location": i.location, "severity": i.severity, "evidence": i.evidence}
                   for i in proposal.issues],
        "changes": [{"param": p, "v1": float(v1.params.get(p, 0.0)), "v2": v} for p, v in sorted(proposal.changes.items())],
        "clamped": proposal.clamped, "advice": proposal.advice,
        "locality": proposal.locality_passed, "verified": proposal.verified, "pack": None,
        "read_by": {"photos": "muse-wonder" if blurred else "", "notes": notes_source},
    }
    if v2 is not None and proposal.v2_report is not None and proposal.changes:
        response["pack"] = _pack_response(v2, proposal.v2_report, proposal.v2_logs)
    return response


@app.get("/api/download/{job}/{kind}")
def download(job: str, kind: str) -> Response:
    """Download one stored artifact (pdf, dxf, json, md or svg)."""
    if job not in JOBS or kind not in ("pdf", "json", "md", "svg", "dxf"):
        raise HTTPException(404, "File not found (was the server just restarted?).")
    blob = JOBS[job][kind]
    if blob is None:
        raise HTTPException(409, "The DXF opens after a pattern maker has checked the pattern." if kind == "dxf"
                            else "The PDF is only available for a pattern that passed.")
    media = {"pdf": "application/pdf", "json": "application/json", "dxf": "application/dxf",
             "md": "text/markdown", "svg": "image/svg+xml"}[kind]
    body = blob.encode("utf-8") if isinstance(blob, str) else blob
    if kind == "json":
        body = json.dumps(blob, indent=2).encode("utf-8")
    name = {"pdf": "pattern.pdf", "json": "pattern.json", "dxf": "pattern.dxf",
            "md": "report.md", "svg": "pattern.svg"}[kind]
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f"attachment; filename={name}"})
