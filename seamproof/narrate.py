"""Muse Spark writes the words around computed results; it never supplies a number.

Used for: the pattern story (verdict in tailor words, sewing order, toile
checklist), production notes (cutting-room steps, owner summary) and the
fabric suggestion. Each call gets a JSON of facts computed by code. Every
number in the model's text must equal one of those facts (or its mm->cm/m
reading) at the precision the model wrote it; one unknown number discards the
whole answer and a deterministic template is used instead. The fabric choice
must be one of the candidates code already verified.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from seamproof import llm as llm_module
from seamproof.pattern import Pattern
from seamproof.verify import Report

LLMFn = Callable[[str, str], dict[str, Any]]

PANEL_ID = {"front": "badan depan", "back": "badan belakang", "sleeve": "lengan", "facing": "lapisan leher",
            "collar": "kerah", "cuff": "manset", "pocket": "saku", "neckband": "ban leher",
            "skirt_front": "rok depan", "skirt_back": "rok belakang", "hem_facing": "lapisan kelim",
            "waist_facing": "lapisan pinggang", "waistband": "ban pinggang"}
SKIRT_PANEL_ID = {"front": "rok depan", "back": "rok belakang"}
EDGE_ID = {"shoulder": "bahu", "side_upper": "sisi atas", "side_lower": "sisi bawah", "side_low": "sisi bawah",
           "side": "sisi", "armhole": "kerung lengan", "cap": "puncak lengan", "neckline": "kerung leher",
           "neck": "leher", "underarm_front": "jahitan bawah lengan", "underarm_back": "jahitan bawah lengan",
           "waist_cf": "pinggang", "waist_side": "pinggang", "hem": "kelim", "scallop": "kelim scallop",
           "attach": "tepi sambung", "cuff_join": "ujung lengan", "edge": "tepi depan", "slit": "belahan",
           "cb": "tengah belakang", "collar_line": "garis kerah",
           "dart_leg1": "kaki kupnat", "dart_leg2": "kaki kupnat",
           "wdart_leg1": "kaki kupnat pinggang", "wdart_leg2": "kaki kupnat pinggang"}
CONSTRUCTION_ID = {"waist_seam": "badan dan rok disambung di pinggang", "shift": "satu potong tanpa sambungan pinggang"}
GARMENT_ID = {"tshirt": "kaos", "blouse": "blus", "tunic": "tunik", "shirt": "kemeja", "skirt": "rok",
              "dress": "dress", "bodice": "blok badan"}
FABRIC_ID = {"knit_low_stretch": "rajut sedikit melar (kaos combed, pique)",
             "knit_high_stretch": "rajut sangat melar (spandex, jersey)",
             "woven": "tenun (katun, linen, rayon, toyobo)"}

_RULES = ("Pakai HANYA angka yang ada di data fakta (boleh mm ditulis cm atau m). Jangan menghitung, "
          "menjumlah, membulatkan ulang atau menebak angka. Tulis hitungan dengan kata (dua, tiga), "
          "jangan memberi nomor urut pada langkah. Bahasa Indonesia sederhana untuk penjahit konveksi; "
          "jangan menulis id teknis (kata bertitik atau bergaris bawah), pakai nama Indonesia dari data.")

# Assumption, as in explain.py: a dot before exactly three digits is a thousands
# separator ("1.500"), a comma is the decimal mark ("17,78"); "17.5" also reads as decimal.
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _parse(token: str) -> tuple[float, int] | None:
    """Read one number token; return (value, decimals written)."""
    text = re.sub(r"\.(?=\d{3}(?:\D|$))", "", token)
    if text.count(",") > 1 or (text.count(".") and text.count(",")):
        return None
    text = text.replace(",", ".")
    if text.count(".") > 1:
        return None
    decimals = len(text.split(".")[1]) if "." in text else 0
    try:
        return float(text), decimals
    except ValueError:
        return None


def fact_numbers(facts: Any, key: str = "") -> list[float]:
    """Every number in a facts JSON, plus cm and m readings of *_mm values."""
    out: list[float] = []
    if isinstance(facts, bool):
        return out
    if isinstance(facts, (int, float)):
        out.append(float(facts))
        if key.endswith("_mm"):
            out += [float(facts) / 10.0, float(facts) / 1000.0]
    elif isinstance(facts, dict):
        for name, value in facts.items():
            out += fact_numbers(value, str(name))
    elif isinstance(facts, (list, tuple)):
        for value in facts:
            out += fact_numbers(value, key)
    return out


def numbers_ok(texts: list[str], allowed: list[float]) -> bool:
    """True when every number written in the texts is a fact at that precision."""
    for text in texts:
        for token in _NUM.findall(text):
            parsed = _parse(token)
            if parsed is None:
                return False
            value, decimals = parsed
            if not any(math.isclose(round(a, decimals), value, abs_tol=10 ** -(decimals + 3)) for a in allowed):
                return False
    return True


def _call(system: str, facts: dict[str, Any], schema: dict[str, Any], purpose: str,
          llm_fn: LLMFn | None) -> dict[str, Any] | None:
    """Ask Muse Spark (or the injected fake); None on any failure."""
    user = json.dumps(facts, ensure_ascii=False)
    try:
        return llm_fn(system, user) if llm_fn is not None else llm_module.call_json(system, user, schema, purpose)
    except Exception:  # model down or invalid: the template takes over
        return None


def _strings(raw: Any, key: str, limit: int) -> list[str] | None:
    """A non-empty list of short strings from the model, or None."""
    items = raw.get(key) if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items or not all(isinstance(i, str) and i.strip() for i in items):
        return None
    return [i.strip()[:300] for i in items[:limit]]


def _r(value: float | None) -> float | None:
    """Round a fact to 0.1 so the model sees tidy numbers."""
    return None if value is None else round(float(value), 1)


# ---------------------------------------------------------------- pattern story

@dataclass
class PatternStory:
    """Verdict in words, sewing order and toile checklist for one pattern."""

    summary: str
    sewing_steps: list[str] = field(default_factory=list)
    toile_checks: list[str] = field(default_factory=list)
    source: str = "template"


PATTERN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"},
                   "sewing_steps": {"type": "array", "items": {"type": "string"}},
                   "toile_checks": {"type": "array", "items": {"type": "string"}}},
    "required": ["summary", "sewing_steps", "toile_checks"],
}

PATTERN_SYSTEM = (
    "Kamu kepala jahit di konveksi. Dari data fakta pola (hasil kode, sudah diverifikasi) tulis JSON: "
    '{"summary": "dua-tiga kalimat: arti hasil verifikasi untuk penjahit", '
    '"sewing_steps": ["urutan jahit, satu kalimat per langkah, enam sampai dua belas langkah"], '
    '"toile_checks": ["empat sampai delapan hal yang DILIHAT pada toile saat dipakai badan: jatuhnya kain, '
    'tarikan, kerutan, letak ujung kupnat, keseimbangan kelim; jangan mengulang angka pemeriksaan"]}. '
    "Urutan jahit mengikuti kebiasaan konveksi: kupnat dulu, lalu bahu, leher, lengan, sisi, kelim; "
    "pakai daftar sambungan. Angka hanya bila perlu (mis. lebar kampuh). " + _RULES
)


def pattern_facts(pattern: Pattern, report: Report) -> dict[str, Any]:
    """Facts the pattern story may use, all computed by code."""
    names = dict(PANEL_ID, **(SKIRT_PANEL_ID if pattern.garment_type == "skirt" else {}))
    edges = {e.id: (panel.name, e) for panel in pattern.panels for e in panel.edges}
    seams = set()
    for edge_id, (panel, edge) in edges.items():
        partner = edge.seam_partner_id
        if partner and partner in edges and partner != edge_id:
            other = edges[partner][0]
            seams.add(tuple(sorted((_edge_name(names, panel, edge_id), _edge_name(names, other, partner)))))
    allowance = pattern.metadata.get("seam_allowance_mm") or {"default": 10, "hem": 25, "fold": 0}
    darts = {name: {k: _r(v) for k, v in d.items() if k.endswith("_mm")}
             for name, d in (pattern.metadata.get("darts") or {}).items() if isinstance(d, dict)}
    return {
        "garmen": GARMENT_ID.get(pattern.garment_type, pattern.garment_type), "ukuran": pattern.size_label,
        "kain": FABRIC_ID.get(pattern.fabric, pattern.fabric), "lulus": report.passed,
        "panel": [{"nama_id": names.get(p.name, p.name), "id": p.name, "potong": p.cut_count,
                   "lipatan": p.cut_on_fold} for p in pattern.panels],
        "sambungan": sorted(list(pair) for pair in seams),
        "kampuh_mm": {"biasa_mm": allowance.get("default", 10), "kelim_mm": allowance.get("hem", 25)},
        "kupnat": darts,
        "kerutan": {f"{k}_mm": round(float(v), 1) for k, v in (pattern.metadata.get("gathers_mm") or {}).items()
                    if float(v) > 0},
        "konstruksi": CONSTRUCTION_ID.get(pattern.metadata.get("construction", ""), ""),
        "pemeriksaan": [{"id": c.id, "nama": c.name_id, "lulus": c.passed, "nilai_mm": _r(c.actual),
                         "harapan_mm": _r(c.expected), "batas_mm": _r(c.tolerance)} for c in report.checks],
    }


def _edge_name(names: dict[str, str], panel: str, edge_id: str) -> str:
    """Indonesian 'panel tepi' name for an edge id like front.shoulder."""
    suffix = edge_id.split(".", 1)[-1]
    return f"{names.get(panel, panel)} — {EDGE_ID.get(suffix, suffix.replace('_', ' '))}"


def _pattern_template(pattern: Pattern, report: Report) -> PatternStory:
    """Deterministic story when Muse is unavailable or wrote a stray number."""
    total, failed = len(report.checks), report.failures()
    label = GARMENT_ID.get(pattern.garment_type, pattern.garment_type)
    summary = (f"Semua {total} pemeriksaan lulus — pola {label} {pattern.size_label} siap dibuat toile."
               if not failed else f"{len(failed)} dari {total} pemeriksaan gagal: "
               + ", ".join(f"{c.id} {c.name_id}" for c in failed) + ". Perbaiki sebelum toile.")
    checks = [f"Cek {c.name_id.lower()} pada toile ({c.id})." for c in report.checks
              if c.id[0] in "SWDK" and c.passed][:6]
    return PatternStory(summary, [], checks, "template")


def narrate_pattern(pattern: Pattern, report: Report, llm_fn: LLMFn | None = None) -> PatternStory:
    """Muse writes the verdict, sewing order and toile checks; numbers are guarded."""
    facts = pattern_facts(pattern, report)
    raw = _call(PATTERN_SYSTEM, facts, PATTERN_SCHEMA, f"story-{pattern.garment_type}", llm_fn)
    if not isinstance(raw, dict) or not isinstance(raw.get("summary"), str):
        return _pattern_template(pattern, report)
    steps, checks = _strings(raw, "sewing_steps", 14), _strings(raw, "toile_checks", 10)
    summary = raw["summary"].strip()[:600]
    if not summary or steps is None or checks is None:
        return _pattern_template(pattern, report)
    if not numbers_ok([summary, *steps, *checks], fact_numbers(facts)):
        return _pattern_template(pattern, report)
    return PatternStory(summary, steps, checks, "muse")


# ------------------------------------------------------------ production notes

@dataclass
class ProductionStory:
    """Cutting-room steps and a short owner summary for one production run."""

    cutting_steps: list[str] = field(default_factory=list)
    owner_summary: str = ""
    source: str = "template"


PRODUCTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"cutting_steps": {"type": "array", "items": {"type": "string"}},
                   "owner_summary": {"type": "string"}},
    "required": ["cutting_steps", "owner_summary"],
}

PRODUCTION_SYSTEM = (
    "Kamu kepala potong di konveksi. Dari data fakta produksi (hasil kode) tulis JSON: "
    '{"cutting_steps": ["langkah gelar dan potong per ukuran: berapa kali gelar, berapa lapis, '
    'panjang marker; satu kalimat per langkah"], '
    '"owner_summary": "dua-tiga kalimat untuk pemilik: total kain, biaya, ukuran yang belum lulus"}. '
    + _RULES
)


def production_facts(report) -> dict[str, Any]:
    """Facts for production notes; spreads per size are computed here, not by the model."""
    sizes = []
    for item in report.sizes:
        sizes.append({"ukuran": item.size, "jatah": item.qty, "lulus": item.verified,
                      "gelar": math.ceil(item.qty / report.layers), "lapis_per_gelar": report.layers,
                      "panjang_marker_mm": round(item.marker.used_length_mm),
                      "meter_kain": round(item.metres, 2),
                      "utilisasi_persen": round(item.marker.utilisation * 100, 1),
                      "potongan_per_marker": len(item.marker.placements)})
    return {"garmen": GARMENT_ID.get(report.garment, report.garment),
            "kain": FABRIC_ID.get(report.fabric, report.fabric),
            "lebar_kain_mm": round(report.fabric_width_mm), "harga_per_meter": round(report.price_per_m),
            "ukuran": sizes, "total_meter": round(report.total_metres, 2),
            "total_biaya_rupiah": round(report.total_cost),
            "total_potong": sum(item.qty for item in report.sizes),
            "utilisasi_total_persen": round(report.overall_utilisation * 100, 1),
            "cut_list": dict(report.cut_list)}


def _production_template(report) -> ProductionStory:
    """Deterministic cutting steps quoting only computed numbers."""
    steps = []
    for item in report.sizes:
        if not item.verified:
            steps.append(f"Ukuran {item.size}: belum lulus verifikasi — jangan dipotong dulu.")
            continue
        steps.append(f"Ukuran {item.size}: gelar {math.ceil(item.qty / report.layers)} kali, "
                     f"{report.layers} lapis per gelar, marker {item.marker.used_length_mm / 1000:.2f} m.")
    return ProductionStory(steps, report.summary_id(), "template")


def narrate_production(report, llm_fn: LLMFn | None = None) -> ProductionStory:
    """Muse writes cutting steps and the owner summary; numbers are guarded."""
    facts = production_facts(report)
    raw = _call(PRODUCTION_SYSTEM, facts, PRODUCTION_SCHEMA, f"production-{report.garment}", llm_fn)
    steps = _strings(raw, "cutting_steps", 12)
    owner = raw.get("owner_summary") if isinstance(raw, dict) else None
    if steps is None or not isinstance(owner, str) or not owner.strip():
        return _production_template(report)
    owner = owner.strip()[:600]
    if not numbers_ok([owner, *steps], fact_numbers(facts)):
        return _production_template(report)
    return ProductionStory(steps, owner, "muse")


# ------------------------------------------------------------ fabric suggestion

FABRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"fabric": {"type": "string"}, "reason": {"type": "string"}},
    "required": ["fabric", "reason"],
}

FABRIC_SYSTEM = (
    "Kamu penasihat kain di konveksi. Pilih SATU kategori dari 'kandidat' (semuanya sudah lulus "
    "verifikasi pola) yang paling cocok untuk desain ini, lalu beri alasan singkat dan contoh kain "
    'yang mudah dibeli. JSON: {"fabric": "<id kandidat>", "reason": "satu-dua kalimat"}. ' + _RULES
)


@dataclass
class FabricSuggestion:
    """One fabric category chosen among verified candidates, with a reason."""

    fabric: str
    reason: str
    source: str = "template"


def suggest_fabric(garment: str, candidates: list[str], features: dict[str, Any] | None = None,
                   brief: str = "", llm_fn: LLMFn | None = None) -> FabricSuggestion:
    """Muse picks among verified candidates; the first candidate is the fallback."""
    if not candidates:
        raise ValueError("Tidak ada kategori kain yang lulus verifikasi untuk desain ini.")
    facts = {"garmen": GARMENT_ID.get(garment, garment), "gaya": features or {}, "brief": brief[:2000],
             "kandidat": [{"id": c, "nama": FABRIC_ID.get(c, c)} for c in candidates]}
    fallback = FabricSuggestion(candidates[0], f"{FABRIC_ID.get(candidates[0], candidates[0]).capitalize()} "
                                f"lulus verifikasi untuk {GARMENT_ID.get(garment, garment)} ini.")
    raw = _call(FABRIC_SYSTEM, facts, FABRIC_SCHEMA, f"fabric-{garment}", llm_fn)
    if not isinstance(raw, dict) or raw.get("fabric") not in candidates or not isinstance(raw.get("reason"), str):
        return fallback
    reason = raw["reason"].strip()[:400]
    if not reason or not numbers_ok([reason], fact_numbers(facts)):
        return fallback
    return FabricSuggestion(raw["fabric"], reason, "muse")
