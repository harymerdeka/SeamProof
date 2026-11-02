"""Command-line entry point tests."""

import json
import sys

import pytest

from seamproof import __version__
from seamproof.cli import main


def test_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Bare invocation prints help text."""
    monkeypatch.setattr(sys, "argv", ["seamproof"])
    assert main() == 0
    assert "SeamProof" in capsys.readouterr().out


def test_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--version prints the package version."""
    monkeypatch.setattr(sys, "argv", ["seamproof", "--version"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_draft_writes_production_pack(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """draft writes pattern.json, pattern.svg, pattern.pdf and report.md."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", "blouse",
                                      "--size", "M", "--fabric", "woven", "--out", str(out)])
    assert main() == 0
    for name in ("pattern.json", "pattern.svg", "pattern.pdf", "report.md"):
        assert (out / name).is_file()
    assert json.loads((out / "pattern.json").read_text())["garment_type"] == "blouse"
    assert (out / "pattern.pdf").read_bytes().startswith(b"%PDF")
    assert "PASS" in (out / "report.md").read_text()


def test_draft_refuses_failing_pattern(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """draft exits nonzero without a PDF when verification fails unforced."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", "tshirt",
                                      "--size", "M", "--fabric", "knit_low_stretch", "--out", str(out)])
    assert main() == 1
    assert (out / "pattern.json").is_file()
    assert not (out / "pattern.pdf").exists()


def test_draft_force_writes_stamped_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """--force writes the PDF stamped NOT VERIFIED."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", "tshirt",
                                      "--size", "M", "--fabric", "knit_low_stretch",
                                      "--out", str(out), "--force"])
    assert main() == 0
    assert b"NOT VERIFIED" in (out / "pattern.pdf").read_bytes()


def test_verify_command(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """verify reads a stored pattern file and reports its verdict."""
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["seamproof", "draft", "--garment", "tunic",
                                      "--size", "M", "--fabric", "woven", "--out", str(out)])
    assert main() == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["seamproof", "verify", str(out / "pattern.json")])
    assert main() == 0
    assert "PASS" in capsys.readouterr().out


def test_mutation_report_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """mutation-report catches every mutation with its own check."""
    monkeypatch.setattr(sys, "argv", ["seamproof", "mutation-report"])
    assert main() == 0
    out = capsys.readouterr().out
    assert "Caught: 43/43" in out
    assert "| G1 |" in out and "| R1 |" in out
