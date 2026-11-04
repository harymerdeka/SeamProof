# SeamProof

**From design to pattern to cut, end to end, for small garment workshops in Indonesia.**

A customer sends a small workshop 2–3 photos ("make this, but with the back of that
one"), or just describes what they want. Today a tailor guesses the pattern, sews,
finds the mistake and cuts again. SeamProof uses Meta's Muse models across the whole
job: it **designs** the garment (from a written prompt or by merging parts of several
photos), turns that design into a **pattern** that passes a deterministic verifier and
is explained in tailor language and corrected from a photo of the first sewn sample
(toile), and prepares the **cut** (marker, cut list, fabric metres, cost, notes for the
cutting room).

Built for the Meta Global AI Developer Hackathon (Oct 27 – Nov 4, 2026).

> **What existed before the hackathon:** only the workshop (@pintar.jahit) and one
> sewn sample shirt. All code in this repo was written during the hackathon.

## Where each Meta model is used

The rule of the project: **models read, propose and explain; Python draws and computes.**
No model ever draws a pattern line or produces a number in a report.

| Model | Where | What it does | Code |
|---|---|---|---|
| **Muse Wonder** (`muse-wonder-1.0`) | Design (photos), toile review | Reads each garment photo (front/back, collar, pocket, yoke, pleat, sleeve…) into structured JSON categories; draws a reference sketch of the merged design to compare against the code-drawn flat (it never feeds the pattern); reads the toile photo into fit notes ("shoulder seam drops"). Faces are blurred before a photo leaves the machine. | `seamproof/vision.py`, `seamproof/llm.py` (`call_vision_json`), `seamproof/fitcorrect.py` |
| **Muse Spark** (`MUSE_MODEL`, e.g. `muse-spark-1.3`) | Design (prompt + refine), pattern, cut | Turns a written prompt into two design options; applies one design change per refine step ("warm khaki beige, keep the boxy fit") as category choices; suggests fabric; proposes parameter changes when a check fails; writes the pattern story and the notes for the cutting room. | `seamproof/design.py`, `seamproof/repair.py`, `seamproof/narrate.py`, `seamproof/llm.py` (`call_json`) |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`, on Baseten) | Explanation tab | Explains the verifier result in plain tailor language. Routed with `BASETEN_TASKS=explain`. | `seamproof/explain.py`, `seamproof/llm.py` (`backend_for`) |
| **SAM 3.1** | Not in the product yet | Probe only (`scripts/sam_probe.py`); see backlog. | — |
| **Python (no model)** | Everywhere numbers matter | Pattern geometry, 15 verifier checks, edit-locality, grading, toile rule table, fabric metres, cost, marker, PDF/DXF export, 3D preview and flat sketch. | `pattern.py`, `draft/`, `verify.py`, `locality.py`, `fit_rules.py`, `production.py`, `marker.py`, `export.py`, `preview3d.py`, `flat.py` |

## The hardest part: two photos, one pattern

Customers rarely send one perfect photo. They send a front of one shirt and a back of
another. Muse Wonder reads each photo separately; SeamProof then merges the parts into
one design (collar and pocket from photo 1, back yoke and box pleat from photo 2) and
the merged design is still drafted only from categories the engine can make. So the
3D preview, the flat sketch and the pattern can never disagree, and the result still
has to pass all verifier checks.

## End to end: design → pattern → cut

| Step | What happens | Muse | Code |
|---|---|---|---|
| **1 · Design** | Start **from a prompt** ("oversized boxy shirt in warm khaki, back yoke with a box pleat…") or **from 1–3 photos** (front of one shirt, back of another). Pick which photo gives each part, then refine one change at a time; every change is a version you can go back to. | Muse Spark designs 2 options from the prompt · Muse Wonder reads the photos and sketches the merged design · Muse Spark applies each plain-English edit | Maps categories to drafting parameters, merges the parts, draws the flat sketch and the 3D preview |
| **2 · Pattern** | The chosen design becomes the pattern: fabric and size, then drafting on a 1:1 mat, 15 checks, a repair loop if a check fails, and correction from a toile photo into a verified v2. A pattern maker signs off. | Muse Spark suggests fabric and proposes repairs · Muse Spark Open explains the result · Muse Wonder reads the toile | Drafts every line, runs the verifier, range and edit-locality checks, sizes toile corrections from a rule table, exports PDF/DXF |
| **3 · Cut** | The signed pattern becomes a production pack: quantities per size, fabric metres, cost, cut list and a marker laid across the fabric width, ready to print 1:1 for the cutting table. | Muse Spark writes the notes for the cutting room and the owner | Computes every metre, rupiah and marker layout |

The design, the 3D preview and the pattern all come from the same categories, so the
preview never shows a detail the pattern cannot make.

```
prompt ─► Muse Spark (2 options) ─┐
photos ─► Muse Wonder (read) ─► merge parts (+ Muse Wonder sketch) ─┤
                                                                     ▼
   DESIGN   Muse Spark refine (one versioned edit at a time) ─► 3D preview (code)
   PATTERN  Python draft ─► verifier (15 checks) ─► fail? Muse Spark proposes a parameter
            ─► range + edit-locality check ─► redraft … (max N rounds, every round logged)
            ─► Muse Spark Open explains ─► toile photo ─► Muse Wonder notes ─► rule table
            ─► verified v2 ─► pattern maker sign-off ─► PDF 1:1 / DXF
   CUT      production pack (metres, cost, cut list) ─► marker ─► Muse Spark cutting-room notes
```

Garments: `tshirt`, `bodice`, `blouse`, `tunic`, `shirt`, `skirt`, `dress`, all from the
same engine and verifier.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # fill in your own keys; never commit .env
python -m seamproof server  # http://127.0.0.1:8000
```

`.env` (names only; values are yours):

```
LLM_BACKEND=meta
MODEL_API_KEY=...           # Meta Model API
MUSE_MODEL=muse-spark-1.3
BASETEN_API_KEY=...
BASETEN_BASE_URL=...
BASETEN_MODEL=meta-models/muse-spark-open-1026
BASETEN_TASKS=explain       # send the explanation to Muse Spark Open
```

### End-to-end demo (the flow in the video)

With the server running:

```bash
playwright install chromium
python scripts/e2e_demo.py out/e2e            # default http://127.0.0.1:8000
```

It drives the real UI with the inputs in `demo/inputs/` (`front.jpeg`, `back.png`,
`toile.png`) against the live models, through design (two photos), pattern and cut, and
saves a screenshot and the panel text of every stage, plus the signed pattern PDF and
the marker. To try the prompt path, pick **Describe it** in the studio and write a
brief; Muse Spark returns two design options and the rest of the flow is the same.

### CLI and tests

```bash
python -m seamproof draft --help      # draft, verify, repair, grade, production, fitcorrect…
python -m seamproof mutation-report   # what each mutation breaks
pytest -q                             # no network: the LLM is mocked
```

Every verifier check has a mutation test: a valid pattern is broken on purpose and
exactly that check must fail.

## Limitations

- Model calls take 10–40 s each; the demo video shortens these waits (marked on screen).
- Shirt back yoke/pleat and the listed garments only; no trousers yet.
- The toile rule table (`fit_rules.py`) covers 10 named fit faults (dropped shoulder,
  underarm pulling, gaping neckline, dart point, swinging hem…); anything else is
  reported but not auto-corrected.
- 3D is a visual preview built from code, not a fit or drape simulation; the 2D pattern
  and its verifier stay the source of truth.
- Market numbers in the video are third-party estimates.

## Backlog (models)

- **SAM 3.1**: segment the garment from a busy WhatsApp photo before Muse Wonder reads it.
- **Muse Wonder on video**: read a short toile try-on video instead of one photo.
- **Muse Spark** for Indonesian voice notes: the owner describes the change by voice.

## Credits

Landing photos: CC0 (see `seamproof/static/landing/CREDITS.txt`). three.js: MIT
(`seamproof/static/three-LICENSE.txt`).

## License

Open source under the [MIT License](LICENSE). Bundled third-party files keep their own
licenses: three.js (MIT) and the CC0 landing photos listed in `CREDITS.txt`.
