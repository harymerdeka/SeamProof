"""LLM wrapper tests with stubbed backends; no network calls."""

import logging
from types import SimpleNamespace

import pytest

import seamproof.llm as llm_module
from seamproof.llm import LLMError, LLMValidationError, _validate, call_json

SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {"type": "object", "values": {"type": "number"}},
        "reason": {"type": "string"},
    },
    "required": ["changes", "reason"],
    "additionalProperties": False,
}
GOOD = '{"changes": {"sleeve_cap_height": 100}, "reason": "kembalikan puncak"}'


class FakeUsage:
    """Token counts shaped like the SDK usage object."""

    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class FakeResponses:
    """ scripted responses.create endpoint recording its calls."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []
        self.count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.texts[min(self.count, len(self.texts) - 1)]
        self.count += 1
        return SimpleNamespace(output_text=text, usage=FakeUsage())


class FakeCompletions:
    """Scripted chat.completions endpoint recording its calls."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []
        self.count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.texts[min(self.count, len(self.texts) - 1)]
        self.count += 1
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=FakeUsage())


class FakeClient:
    """Stub OpenAI client serving one backend namespace."""

    def __init__(self, texts, backend):
        if backend == "meta":
            self.responses = FakeResponses(texts)
        else:
            self.chat = SimpleNamespace(completions=FakeCompletions(texts))


def _settings(backend="meta"):
    """Fake settings with a decoy secret that must never appear in logs."""
    return SimpleNamespace(llm_backend=backend, model_api_key="fake-secret-123",
                           muse_model="test-muse", baseten_api_key="fake-secret-123",
                           baseten_base_url="https://example.invalid/v1",
                           baseten_model="test-baseten")


def _install(monkeypatch: pytest.MonkeyPatch, texts, backend="meta"):
    """Point the wrapper at a stub backend; return the stub endpoint."""
    monkeypatch.setattr(llm_module, "load_settings", lambda: _settings(backend))
    client = FakeClient(texts, backend)
    created = {}

    def factory(**kwargs):
        created.update(kwargs)
        return client

    monkeypatch.setattr(llm_module, "OpenAI", factory)
    endpoint = client.responses if backend == "meta" else client.chat.completions
    endpoint.factory_kwargs = created
    return endpoint


def test_meta_valid_first_try(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A good payload returns parsed on the first attempt with safe logs."""
    endpoint = _install(monkeypatch, [GOOD])
    with caplog.at_level(logging.INFO, logger="seamproof.llm"):
        assert call_json("sys", "user", SCHEMA, "test-purpose") == {
            "changes": {"sleeve_cap_height": 100}, "reason": "kembalikan puncak"}
    assert endpoint.count == 1
    assert endpoint.calls[0]["model"] == "test-muse"
    assert "instructions" in endpoint.calls[0] and "input" in endpoint.calls[0]
    assert endpoint.factory_kwargs["base_url"] == "https://api.meta.ai/v1"
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "test-purpose" in text and "test-muse" in text and "usage" in text
    assert "fake-secret-123" not in text


def test_retry_after_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Garbage then good JSON succeeds on the second attempt."""
    endpoint = _install(monkeypatch, ["bukan json", GOOD])
    assert call_json("sys", "user", SCHEMA, "retry")["reason"] == "kembalikan puncak"
    assert endpoint.count == 2


def test_invalid_twice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two bad payloads raise instead of looping forever."""
    _install(monkeypatch, ["bukan json", "[1, 2]"])
    with pytest.raises(LLMValidationError):
        call_json("sys", "user", SCHEMA, "retry")


def test_schema_violation_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Right JSON shape, wrong types: retry, then accept the fix."""
    endpoint = _install(monkeypatch, ['{"changes": {"a": "x"}, "reason": 1}', GOOD])
    assert call_json("sys", "user", SCHEMA, "schema")["reason"] == "kembalikan puncak"
    assert endpoint.count == 2


def test_baseten_chat_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseten backend uses chat.completions with system and user roles."""
    endpoint = _install(monkeypatch, [GOOD], backend="baseten")
    assert call_json("sys", "user", SCHEMA, "baseten")["reason"] == "kembalikan puncak"
    assert endpoint.calls[0]["model"] == "test-baseten"
    assert [m["role"] for m in endpoint.calls[0]["messages"]] == ["system", "user"]


def test_missing_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty key fails before any client is built."""
    settings = _settings()
    settings.model_api_key = ""
    monkeypatch.setattr(llm_module, "load_settings", lambda: settings)
    with pytest.raises(LLMError, match="not configured"):
        call_json("sys", "user", SCHEMA, "unconfigured")


def test_array_schema_validation() -> None:
    """Array items validate per element with indexed paths."""
    schema = {"type": "array", "items": {"type": "object", "properties": {"v": {"type": "number"}},
                                         "required": ["v"], "additionalProperties": False}}
    assert _validate([{"v": 1}, {"v": 2.5}], schema) == []
    assert _validate({"v": 1}, schema) == ["$: expected array"]
    errors = _validate([{"v": 1}, {"v": "x"}, {}], schema)
    assert errors == ["$[1].v: expected number", "$[2]: missing 'v'"]


def test_api_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend errors surface without leaking credentials."""
    monkeypatch.setattr(llm_module, "load_settings", lambda: _settings())

    class Broken:
        def create(self, **kwargs):
            raise RuntimeError("jaringan putus")

    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: SimpleNamespace(responses=Broken()))
    with pytest.raises(LLMError, match="call failed"):
        call_json("sys", "user", SCHEMA, "broken")
