# SeamProof — Agent Context (Prompt 0 + updates)

You are helping build SeamProof during the Meta Global AI Developer Hackathon
(Oct 27 – Nov 4, 2026). All code must be written during the hackathon in this repo.

## PRODUCT

SeamProof turns body measurements (Path B) or a sample garment photo (Path A)
into a verified, production-ready sewing pattern for small garment workshops
(konveksi) in Indonesia. It explains problems in tailor language (Indonesian),
grades sizes, produces a production pack (cut list, fabric metres, cost), and
corrects the pattern from a photo of the first sewn sample (toile).

The scope has grown: SeamProof now also designs the garment before the pattern.
A design starts from a written brief, up to 3 reference photos (parts can be
mixed: body from photo 1, sleeves from photo 2) or a base model, is refined one
change at a time (versioned), and is shown as a 3D preview plus a technical
flat. The chosen design is then drafted, verified and produced as above.

## Garment library (built in this order, all from the same engine and verifier)

1. tshirt — unisex knit T-shirt (already built; pipeline proof, no darts)
2. blouse — women's woven blouse from a bodice block with one bust dart per
   side and a set-in sleeve
3. tunic — the same women's block extended in length with side flare (tunik)

New garments must reuse pattern.py, verify.py, export.py, repair.py,
locality.py. Only add draft modules, measurements and params.

## POSITIONING

Existing AI pattern tools focus on generating patterns. SeamProof is a
verification and correction layer with proof on real fabric.

## CORE PRINCIPLES (never break these)

1. Pattern geometry is ALWAYS produced by deterministic Python code from parameters.
2. The LLM NEVER draws pattern lines and NEVER computes or changes numbers in
   reports. It only (a) interprets inputs into structured JSON, (b) proposes
   parameter changes, (c) explains computed results in words.
3. Every pattern must pass a deterministic verifier. Loop: draft -> verify ->
   (if fail) LLM proposes parameter change -> range check -> edit-locality
   check -> redraft -> verify. Max N rounds, every round logged.
4. Every parameter change must touch only the panels/edges it declares (edit locality).
5. Design previews (3D model and flat sketch) are ALSO produced by deterministic
   Python code, from the same drafted categories the pattern uses, so the
   preview never shows a detail the pattern cannot make. The LLM only picks
   categories; it never draws a preview or a pattern.

## ENVIRONMENT

- Linux server accessed over SSH from VS Code. Python 3.10+, virtualenv at
  .venv. Give bash commands.
- Config comes from .env via python-dotenv: LLM_BACKEND (meta|baseten),
  MODEL_API_KEY, MUSE_MODEL, BASETEN_API_KEY, BASETEN_BASE_URL, BASETEN_MODEL.
- LLM backend is now LLM_BACKEND=meta with MUSE_MODEL from .env; Baseten stays
  as a second backend.
- Meta Model API: OpenAI SDK, base_url "https://api.meta.ai/v1",
  client.responses.create(model=MUSE_MODEL, input=...). Other models:
  muse-wonder-1.0, sam-3.1.
- Baseten: OpenAI SDK, base_url=BASETEN_BASE_URL,
  client.chat.completions.create(model=BASETEN_MODEL, ...). Model:
  meta-models/muse-spark-open-1026 (open weights).
- Never hardcode, print or log API keys.
- This repo is independent. Never read, modify or import anything outside this
  repo folder. Ignore the notes/ folder.

## ENGINEERING RULES

- Units: millimetres internally. PDF points only inside export (1 mm = 2.8346 pt).
- Small, typed functions with docstrings. No network calls in tests (mock the LLM).
- Every feature comes with pytest tests. Every verifier check has a MUTATION
  test (deliberately break a valid pattern; exactly that check must fail).
- Do not add features I did not ask for. No databases, cloud storage,
  model training, login, or payments.
- 3D is allowed as a visual design preview only (a rotatable garment model on
  the design table, built from code). No cloth physics, draping or 3D fit
  simulation; the 2D pattern and its verifier stay the source of truth.
- When unsure about garment construction or an API format, write the assumption
  in a comment or stop and ask. Never guess an API request format.
- End every task with: files changed, how to run, test results, known limitations.
