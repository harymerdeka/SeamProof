"""SeamProof command line: draft, verify and report on garments."""

import argparse
import json
import sys
from pathlib import Path

from seamproof import __version__
from seamproof.allowance import ensure_allowance
from seamproof.draft import REGISTRY
from seamproof.fabric import FABRIC_EASE
from seamproof.measurements import SIZE_TABLE, WOMEN_SIZE_TABLE

DEFAULT_FABRIC = {"tshirt": "knit_low_stretch", "bodice": "woven", "blouse": "woven", "tunic": "woven",
                  "shirt": "woven", "skirt": "woven", "dress": "woven"}


def _draft(args: argparse.Namespace) -> int:
    """Draft, verify and export a production pack into the output directory."""
    from seamproof.export import export_pdf_tiled, export_report_md, export_svg
    from seamproof.verify import verify_pattern

    table = SIZE_TABLE if args.garment == "tshirt" else WOMEN_SIZE_TABLE
    if args.size not in table:
        print(f"Unknown size for {args.garment}: {args.size}", file=sys.stderr)
        return 2
    fabric = args.fabric or DEFAULT_FABRIC[args.garment]
    if fabric not in FABRIC_EASE:
        print(f"Unknown fabric category: {fabric}", file=sys.stderr)
        return 2
    pattern = REGISTRY[args.garment](table[args.size], fabric, {})
    report = verify_pattern(pattern)
    explanation = _maybe_explain(args, report, pattern)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pattern.json").write_text(json.dumps(ensure_allowance(pattern).to_dict(), indent=2), encoding="utf-8")
    export_svg(pattern, str(out / "pattern.svg"))
    export_report_md(report, str(out / "report.md"), explanation)
    print(report.summary_id())
    try:
        export_pdf_tiled(pattern, report, str(out / "pattern.pdf"), force=args.force,
                           explanation=explanation)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(f"Wrote pattern.json, pattern.svg, report.md to {out} (pattern.pdf refused).")
        return 1
    print(f"Wrote pattern.json, pattern.svg, pattern.pdf, report.md to {out}.")
    return 0


def _maybe_explain(args: argparse.Namespace, report, pattern):
    """Explain a report when --explain is passed, else None (LLM or template)."""
    if not getattr(args, "explain", False):
        return None
    from seamproof.explain import explain

    return explain(report, pattern)


def _verify(args: argparse.Namespace) -> int:
    """Verify a stored pattern file and print the Indonesian verdict."""
    from seamproof.pattern import Pattern
    from seamproof.verify import verify_pattern

    pattern = Pattern.from_dict(json.loads(Path(args.pattern).read_text(encoding="utf-8")))
    report = verify_pattern(pattern)
    print(report.summary_id())
    for check in report.checks:
        print(("OK " if check.passed else "FAIL ") + f"{check.id} {check.name_id}: {check.message_id}")
    return 0 if report.passed else 1


def _repair(args: argparse.Namespace) -> int:
    """Run the draft-verify-correct loop with an optional injected defect."""
    from seamproof.export import export_pdf_tiled, export_report_md, export_svg
    from seamproof.repair import INJECTS, repair_loop, rounds_table_md

    table = SIZE_TABLE if args.garment == "tshirt" else WOMEN_SIZE_TABLE
    if args.size not in table:
        print(f"Unknown size for {args.garment}: {args.size}", file=sys.stderr)
        return 2
    if args.fabric not in FABRIC_EASE:
        print(f"Unknown fabric category: {args.fabric}", file=sys.stderr)
        return 2
    if args.inject is not None and args.inject not in INJECTS:
        print(f"Unknown inject preset: {args.inject} (choose {', '.join(INJECTS)})", file=sys.stderr)
        return 2
    base = dict(INJECTS[args.inject]) if args.inject else {}
    pattern, report, logs = repair_loop(table[args.size], args.fabric, base,
                                        max_rounds=args.max_rounds, garment=args.garment)
    print(rounds_table_md(logs), end="")
    print(report.summary_id())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pattern.json").write_text(json.dumps(ensure_allowance(pattern).to_dict(), indent=2), encoding="utf-8")
    export_svg(pattern, str(out / "pattern.svg"))
    repair_expl = _maybe_explain(args, report, pattern)
    export_report_md(report, str(out / "report.md"), repair_expl)
    (out / "rounds.md").write_text("# Repair rounds\n\n" + rounds_table_md(logs), encoding="utf-8")
    try:
        export_pdf_tiled(pattern, report, str(out / "pattern.pdf"), force=args.force,
                           explanation=repair_expl)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote pattern.json, pattern.svg, pattern.pdf, report.md, rounds.md to {out}.")
    return 0 if report.passed else 1


def _intake(args: argparse.Namespace) -> int:
    """Take measurements from text or file, then repair and export the pack."""
    from seamproof.export import export_pdf_tiled, export_report_md, export_svg
    from seamproof.intake import GARMENT_FIELDS, from_size_table, parse_file, parse_text
    from seamproof.measurements import TShirtMeasurements, WomenMeasurements
    from seamproof.repair import repair_loop, rounds_table_md

    if bool(args.text) == bool(args.file):
        print("Give exactly one of --text or --file.", file=sys.stderr)
        return 2
    garment = args.garment
    if garment is None:
        fabric_guess = args.fabric or ""
        garment = "tshirt" if fabric_guess.startswith("knit") else "blouse"
    if garment not in GARMENT_FIELDS:
        print(f"Unknown garment: {garment}", file=sys.stderr)
        return 2
    fabric = args.fabric or DEFAULT_FABRIC[garment]
    if fabric not in FABRIC_EASE:
        print(f"Unknown fabric category: {fabric}", file=sys.stderr)
        return 2
    if args.text:
        result = parse_text(args.text, garment)
    else:
        result = parse_file(args.file)
    if args.from_size:
        result = from_size_table(garment, args.from_size, result["measurements"])
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    if result["missing"]:
        print(f"Missing measurements: {', '.join(result['missing'])} — complete them first; nothing is invented.",
              file=sys.stderr)
        return 2
    cls = TShirtMeasurements if garment == "tshirt" else WomenMeasurements
    try:
        measurements = cls(**result["measurements"])
    except (TypeError, ValueError) as exc:
        print(f"Invalid measurements: {exc}", file=sys.stderr)
        return 2
    pattern, report, logs = repair_loop(measurements, fabric, {}, max_rounds=args.max_rounds,
                                        garment=garment)
    print(rounds_table_md(logs), end="")
    print(report.summary_id())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pattern.json").write_text(json.dumps(ensure_allowance(pattern).to_dict(), indent=2), encoding="utf-8")
    (out / "intake.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_svg(pattern, str(out / "pattern.svg"))
    intake_expl = _maybe_explain(args, report, pattern)
    export_report_md(report, str(out / "report.md"), intake_expl)
    (out / "rounds.md").write_text("# Repair rounds\n\n" + rounds_table_md(logs), encoding="utf-8")
    try:
        export_pdf_tiled(pattern, report, str(out / "pattern.pdf"), force=args.force,
                           explanation=intake_expl)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote pattern.json, intake.json, pattern.svg, pattern.pdf, report.md, rounds.md to {out}.")
    return 0 if report.passed else 1


def _grade(args: argparse.Namespace) -> int:
    """Grade all sizes, verify each, check consistency and export nested PDF."""
    from seamproof.export import export_pdf_graded
    from seamproof.grading import grade_and_verify, grade_report_md

    sizes = [part.strip() for part in args.sizes.split(",") if part.strip()]
    fabric = args.fabric or DEFAULT_FABRIC[args.garment]
    if fabric not in FABRIC_EASE:
        print(f"Unknown fabric category: {fabric}", file=sys.stderr)
        return 2
    try:
        report = grade_and_verify(args.garment, sizes, fabric)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report.summary_id())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for size, pattern in report.patterns.items():
        (out / f"pattern_{size}.json").write_text(json.dumps(ensure_allowance(pattern).to_dict(), indent=2),
                                                 encoding="utf-8")
    explanations = None
    if getattr(args, "explain", False):
        from seamproof.explain import explain

        explanations = {size: explain(report.reports[size], pattern)
                        for size, pattern in report.patterns.items()
                        if not report.reports[size].passed}
    (out / "grade_report.md").write_text(grade_report_md(report, explanations), encoding="utf-8")
    try:
        export_pdf_graded(report, str(out / "graded.pdf"), force=args.force,
                            explanations=explanations)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(f"Wrote pattern_<size>.json and grade_report.md to {out} (graded.pdf refused).")
        return 1
    print(f"Wrote pattern_<size>.json, graded.pdf, grade_report.md to {out}.")
    return 0 if report.passed else 1


def _production(args: argparse.Namespace) -> int:
    """Pack a size run into cut list, metres, cost and marker exports."""
    from seamproof.export import export_marker_plot_pdf, export_marker_svg, export_production_pdf
    from seamproof.production import produce, production_md

    fabric = args.fabric or DEFAULT_FABRIC[args.garment]
    try:
        report = produce(args.garment, args.run, fabric, args.width, args.price,
                         layers=args.layers)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report.summary_id())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "production.md").write_text(production_md(report), encoding="utf-8")
    export_production_pdf(report, str(out / "production.pdf"))
    for item in report.sizes:
        export_marker_svg(item.marker, f"{args.garment} {item.size}",
                          str(out / f"marker_{item.size}.svg"), item.size)
        if item.verified:  # 1:1 cutting prints only for verified sizes
            export_marker_plot_pdf(item.marker, f"{args.garment} {item.size}",
                                   str(out / f"marker_{item.size}_plotter.pdf"), item.size)
    print(f"Wrote production.md, production.pdf, marker_<size>.svg and "
          f"marker_<size>_plotter.pdf (verified sizes) to {out}.")
    return 0


def _fitcorrect(args: argparse.Namespace) -> int:
    """Correct a v1 pattern from toile notes/photos into a verified v2."""
    from seamproof.export import (export_fit_overlay_svg, export_pdf_tiled, export_report_md,
                                  export_svg)
    from seamproof.fitcorrect import fit_correct, fit_md
    from seamproof.pattern import Pattern
    from seamproof.verify import verify_pattern

    try:
        v1 = Pattern.from_dict(json.loads(Path(args.pattern).read_text(encoding="utf-8")))
    except (ValueError, KeyError) as exc:
        print(f"Invalid v1 pattern: {exc}", file=sys.stderr)
        return 2
    notes = [n.strip() for n in (args.notes or "").split(";") if n.strip()]
    photos = [p.strip() for p in (args.photos or "").split(",") if p.strip()]
    if not notes and not photos:
        print("Give --notes and/or --photos.", file=sys.stderr)
        return 2
    try:
        proposal, v2 = fit_correct(v1, photos or None, notes or None)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(proposal.label)
    for issue in proposal.issues:
        print(f"- {issue.key} [{issue.severity}]: {issue.evidence}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit.md").write_text(fit_md(proposal, v1, "v2"), encoding="utf-8")
    if v2 is None:
        print("Edit locality FAILED — v2 refused, not exported.", file=sys.stderr)
        return 1
    v2_report = verify_pattern(v2)
    (out / "v2_pattern.json").write_text(json.dumps(ensure_allowance(v2).to_dict(), indent=2), encoding="utf-8")
    export_svg(v2, str(out / "v2_pattern.svg"))
    export_fit_overlay_svg(v1, v2, str(out / "overlay.svg"))
    export_report_md(v2_report, str(out / "v2_report.md"))
    print(v2_report.summary_id())
    try:
        export_pdf_tiled(v2, v2_report, str(out / "v2_pattern.pdf"), force=args.force)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote fit.md, overlay.svg, v2_pattern.json, v2_report.md, v2_pattern.pdf to {out}.")
    return 0 if proposal.verified else 1


def _mutation_report(_args: argparse.Namespace) -> int:
    """Break pristine drafts once per check and show what the verifier catches."""
    from seamproof.mutations import MUTATIONS, run_mutation_report

    print("| Check | Mutation | Caught by |")
    print("|-------|--------|-----------------|")
    caught = 0
    for (check_id, _base, description, _mutate), (_rid, _desc, failed) in zip(MUTATIONS, run_mutation_report()):
        assert check_id == _rid
        hit = ", ".join(sorted(failed)) if failed else "LOLOS (bahaya!)"
        if check_id in failed:
            caught += 1
        print(f"| {check_id} | {description} | {hit} |")
    print(f"Caught: {caught}/{len(MUTATIONS)}.")
    return 0 if caught == len(MUTATIONS) else 1


def _server(args: argparse.Namespace) -> int:
    """Start the local web UI; binds loopback only for SSH port forwarding."""
    import uvicorn

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Only a loopback bind is allowed (127.0.0.1).", file=sys.stderr)
        return 2
    print(f"SeamProof UI at http://127.0.0.1:{args.port} — forward the port via VS Code, then open it in a browser.")
    uvicorn.run("seamproof.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch draft, verify and mutation-report subcommands."""
    parser = argparse.ArgumentParser(description="SeamProof sewing pattern tools")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    draft = sub.add_parser("draft", help="draft, verify and export a production pack")
    draft.add_argument("--garment", choices=sorted(REGISTRY), required=True)
    draft.add_argument("--size", default="M")
    draft.add_argument("--fabric", default=None)
    draft.add_argument("--out", required=True, help="output directory")
    draft.add_argument("--force", action="store_true", help="export PDF despite failures")
    draft.add_argument("--explain", action="store_true", help="add tailor-language instructions")
    draft.set_defaults(func=_draft)
    verify = sub.add_parser("verify", help="verify a stored pattern.json file")
    verify.add_argument("pattern", help="path to pattern.json")
    verify.set_defaults(func=_verify)
    mutation = sub.add_parser("mutation-report", help="show what each mutation breaks")
    mutation.set_defaults(func=_mutation_report)
    repair = sub.add_parser("repair", help="draft, verify and LLM-correct a pattern")
    repair.add_argument("--garment", default="blouse", choices=sorted(REGISTRY))
    repair.add_argument("--size", default="M")
    repair.add_argument("--fabric", default="woven")
    repair.add_argument("--inject", default=None, help="named bad starting point for demos")
    repair.add_argument("--max-rounds", type=int, default=5)
    repair.add_argument("--out", required=True, help="output directory")
    repair.add_argument("--force", action="store_true", help="export PDF despite failures")
    repair.add_argument("--explain", action="store_true", help="add tailor-language instructions")
    repair.set_defaults(func=_repair)
    intake = sub.add_parser("intake", help="take measurements, then repair and export")
    intake.add_argument("--text", default=None, help="free Indonesian/English measurement notes")
    intake.add_argument("--file", default=None, help="JSON/YAML measurements with a unit field")
    intake.add_argument("--garment", default=None, choices=sorted(REGISTRY))
    intake.add_argument("--fabric", default=None)
    intake.add_argument("--from-size", default=None, help="size-table base to override, e.g. M")
    intake.add_argument("--max-rounds", type=int, default=5)
    intake.add_argument("--out", required=True, help="output directory")
    intake.add_argument("--force", action="store_true", help="export PDF despite failures")
    intake.add_argument("--explain", action="store_true", help="add tailor-language instructions")
    intake.set_defaults(func=_intake)
    grade = sub.add_parser("grade", help="grade sizes, verify each and export nested PDF")
    grade.add_argument("--garment", default="blouse", choices=["tshirt", "blouse", "tunic", "shirt", "skirt", "dress"])
    grade.add_argument("--sizes", default="S,M,L,XL", help="comma sizes, must include extremes")
    grade.add_argument("--fabric", default=None)
    grade.add_argument("--out", required=True, help="output directory")
    grade.add_argument("--force", action="store_true", help="export PDF despite failures")
    grade.add_argument("--explain", action="store_true", help="add tailor-language instructions")
    grade.set_defaults(func=_grade)
    production = sub.add_parser("production", help="pack a size run into metres and cost")
    production.add_argument("--garment", default="blouse", choices=["tshirt", "blouse", "tunic", "shirt", "skirt", "dress"])
    production.add_argument("--run", required=True, help="size run, e.g. S=20,M=50,L=30")
    production.add_argument("--width", type=float, required=True, help="fabric width in mm")
    production.add_argument("--price", type=float, required=True, help="price per metre in Rp")
    production.add_argument("--fabric", default=None)
    production.add_argument("--layers", type=int, default=1, help="layers per spread")
    production.add_argument("--out", required=True, help="output directory")
    production.set_defaults(func=_production)
    fit = sub.add_parser("fitcorrect", help="correct a toile into a verified v2")
    fit.add_argument("--pattern", required=True, help="path to v1 pattern.json")
    fit.add_argument("--notes", default="", help="semicolon-separated fit notes")
    fit.add_argument("--photos", default="", help="comma-separated photo paths")
    fit.add_argument("--out", required=True, help="output directory")
    fit.add_argument("--force", action="store_true", help="export PDF despite failures")
    fit.set_defaults(func=_fitcorrect)
    server = sub.add_parser("server", help="start the local web UI (127.0.0.1 only)")
    server.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1)")
    server.add_argument("--port", type=int, default=8000)
    server.set_defaults(func=_server)
    arguments = parser.parse_args(argv)
    if getattr(arguments, "func", None) is None:
        parser.print_help()
        return 0
    return arguments.func(arguments)


if __name__ == "__main__":
    sys.exit(main())
