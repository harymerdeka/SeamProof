"""Configuration loading tests using fake credentials only."""

from pathlib import Path

import pytest

from seamproof.config import load_settings

ENV_NAMES = (
    "LLM_BACKEND", "MODEL_API_KEY", "MUSE_MODEL", "BASETEN_API_KEY",
    "BASETEN_BASE_URL", "BASETEN_MODEL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove real environment values so tests use only fake credentials."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_load_env_and_hide_credentials(tmp_path: Path) -> None:
    """Values load from .env but never appear in repr or str."""
    path = tmp_path / ".env"
    path.write_text(
        "LLM_BACKEND=baseten\nMODEL_API_KEY=fake-meta-secret\n"
        "MUSE_MODEL=test-meta-model\nBASETEN_API_KEY=fake-baseten-secret\n"
        "BASETEN_BASE_URL=https://example.invalid/v1\nBASETEN_MODEL=test-model\n"
    )
    settings = load_settings(path)
    assert settings.llm_backend == "baseten"
    assert settings.model_api_key == "fake-meta-secret"
    assert settings.baseten_api_key == "fake-baseten-secret"
    assert settings.muse_model == "test-meta-model"
    assert settings.baseten_model == "test-model"
    assert settings.baseten_base_url == "https://example.invalid/v1"
    for display in (repr(settings), str(settings)):
        assert "fake-meta-secret" not in display
        assert "fake-baseten-secret" not in display


def test_process_environment_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Process env keeps precedence because .env never overrides it."""
    path = tmp_path / ".env"
    path.write_text("LLM_BACKEND=baseten\nMODEL_API_KEY=file-secret\n")
    monkeypatch.setenv("LLM_BACKEND", "meta")
    monkeypatch.setenv("MODEL_API_KEY", "environment-secret")
    settings = load_settings(path)
    assert settings.llm_backend == "meta"
    assert settings.model_api_key == "environment-secret"


def test_missing_env_is_safe(tmp_path: Path) -> None:
    """A missing file yields safe defaults, not an exception."""
    settings = load_settings(tmp_path / "missing")
    assert settings.llm_backend == "meta"
    assert settings.model_api_key == settings.baseten_api_key == ""


def test_invalid_backend_does_not_echo_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid backend errors without repeating the bad value."""
    monkeypatch.setenv("LLM_BACKEND", "secret-invalid-value")
    with pytest.raises(ValueError) as exc:
        load_settings(tmp_path / "missing")
    assert "secret-invalid-value" not in str(exc.value)
