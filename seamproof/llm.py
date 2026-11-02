"""LLM backend wrapper: JSON-only calls with validation, one retry and safe logging.

Backend from LLM_BACKEND: "meta" uses the responses API with MUSE_MODEL,
"baseten" uses chat.completions with BASETEN_MODEL. BASETEN_TASKS (comma-separated
purpose prefixes) sends those tasks to Baseten whatever LLM_BACKEND says. Call shapes follow the
task brief and were checked against the installed openai SDK signatures
(responses.create accepts model/input/instructions; chat.completions.create
accepts model/messages). Backends, models and URLs all come from .env via
config; nothing is hardcoded. Logs never carry keys or user photos, only
purpose/backend/model/latency/usage.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from seamproof.config import Settings, load_settings

# Documented Meta endpoint; deployments override it with META_BASE_URL in .env.
META_BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")

logger = logging.getLogger("seamproof.llm")


class LLMError(Exception):
    """The backend call itself failed or is not configured."""


class LLMValidationError(LLMError):
    """The backend kept returning unusable JSON after a retry."""


def _validate(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Check data against a small schema subset; return error messages.

    Supports type object/array/string/number/integer/boolean, properties,
    required, additionalProperties, items and values (per-key object schema).
    """
    errors: list[str] = []
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(data, dict):
            return [f"{path}: expected object"]
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}: missing '{key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errors.extend(_validate(data[key], subschema, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in data:
                if key not in schema.get("properties", {}):
                    errors.append(f"{path}: unexpected '{key}'")
        values = schema.get("values")
        if values is not None:
            for key, value in data.items():
                errors.extend(_validate(value, values, f"{path}.{key}"))
    elif kind == "array":
        if not isinstance(data, list):
            return [f"{path}: expected array"]
        for index, value in enumerate(data):
            errors.extend(_validate(value, schema.get("items", {}), f"{path}[{index}]"))
    elif kind == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string")
    elif kind == "number":
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number")
    elif kind == "integer":
        if isinstance(data, bool) or not isinstance(data, int):
            errors.append(f"{path}: expected integer")
    elif kind == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean")
    return errors


def _usage_dict(usage: Any) -> dict[str, int]:
    """Extract token counts when the backend reports them."""
    if usage is None:
        return {}
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            result[key] = value
    return result


def _strip_fences(text: str) -> str:
    """Remove one Markdown code fence so raw and fenced JSON both parse."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _log(purpose: str, backend: str, model: str, latency_ms: float,
         usage: dict[str, int], ok: bool, attempt: int) -> None:
    """Log one attempt without keys, prompts or photos."""
    logger.info("purpose=%s backend=%s model=%s latency_ms=%.0f usage=%s ok=%s attempt=%d",
                purpose, backend, model, latency_ms, usage, ok, attempt)


def call_vision_json(system: str, text: str, image_data_urls: list[str], schema: dict[str, Any],
                     purpose: str) -> dict[str, Any]:
    """Ask the Meta vision model (WONDER_MODEL) for JSON about images, retrying once.

    Request shape (responses API, one user message with input_text plus
    input_image data URLs) is the one already used by fitcorrect.py.
    Assumption: only the meta backend serves images; Baseten is text-only
    here. Images are never logged.
    """
    settings = load_settings()
    model = os.environ.get("WONDER_MODEL", "muse-wonder-1.0")
    if not settings.model_api_key:
        raise LLMError("meta vision is not configured (missing MODEL_API_KEY)")
    client = OpenAI(base_url=META_BASE_URL, api_key=settings.model_api_key)
    instruction = system + "\nReply with a single raw JSON object only, no other text."
    prompt, last_error = text, ""
    for attempt in (1, 2):
        if attempt == 2:
            prompt = text + f"\nInvalid response: {last_error}. Send only the correct JSON again."
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": url} for url in image_data_urls)
        started = time.perf_counter()
        try:
            response = client.responses.create(
                model=model, instructions=instruction,
                input=[{"type": "message", "role": "user", "content": content}])
            reply, usage = response.output_text, _usage_dict(response.usage)
        except Exception as exc:
            _log(purpose, "meta", model, (time.perf_counter() - started) * 1000, {}, False, attempt)
            raise LLMError(f"meta vision call failed: {type(exc).__name__}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            data = json.loads(_strip_fences(reply))
        except json.JSONDecodeError as exc:
            last_error = f"broken JSON ({exc.msg})"
            _log(purpose, "meta", model, latency_ms, usage, False, attempt)
            continue
        errors = _validate(data, schema) if isinstance(data, dict) else ["$: expected object"]
        if errors:
            last_error = "; ".join(errors)
            _log(purpose, "meta", model, latency_ms, usage, False, attempt)
            continue
        _log(purpose, "meta", model, latency_ms, usage, True, attempt)
        return data
    raise LLMValidationError(f"Vision model kept returning invalid JSON: {last_error}")


def backend_for(settings: Settings, purpose: str) -> str:
    """Pick the backend for one call: a BASETEN_TASKS match wins over LLM_BACKEND."""
    if any(purpose == t or purpose.startswith(t + "-") for t in settings.baseten_tasks):
        return "baseten"
    return settings.llm_backend


def call_json(system: str, user: str, schema: dict[str, Any], purpose: str) -> dict[str, Any]:
    """Call the configured backend, demanding JSON, retrying once on bad output.

    Returns the validated JSON object. Raises LLMValidationError after two
    bad payloads, LLMError when the backend errors or is not configured.
    """
    settings = load_settings()
    backend = backend_for(settings, purpose)
    if backend == "meta":
        key, model, base_url = settings.model_api_key, settings.muse_model, META_BASE_URL
        missing = [name for name, value in (("MODEL_API_KEY", key), ("MUSE_MODEL", model)) if not value]
    else:
        key, model, base_url = settings.baseten_api_key, settings.baseten_model, settings.baseten_base_url
        missing = [name for name, value in (("BASETEN_API_KEY", key), ("BASETEN_MODEL", model),
                                            ("BASETEN_BASE_URL", base_url)) if not value]
    if missing:
        raise LLMError(f"{backend} backend is not configured (missing {', '.join(missing)})")
    client = OpenAI(base_url=base_url, api_key=key)
    instruction = system + "\nReply with a single raw JSON object only, no other text."
    payload = user
    last_error = ""
    for attempt in (1, 2):
        if attempt == 2:
            payload = user + f"\nInvalid response: {last_error}. Send only the correct JSON again."
        started = time.perf_counter()
        try:
            if backend == "meta":
                response = client.responses.create(model=model, instructions=instruction, input=payload)
                text = response.output_text
                usage = _usage_dict(response.usage)
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": instruction},
                              {"role": "user", "content": payload}])
                text = response.choices[0].message.content or ""
                usage = _usage_dict(response.usage)
        except Exception as exc:
            _log(purpose, backend, model, (time.perf_counter() - started) * 1000, {}, False, attempt)
            raise LLMError(f"{backend} call failed: {type(exc).__name__}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            data = json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            last_error = f"broken JSON ({exc.msg})"
            _log(purpose, backend, model, latency_ms, usage, False, attempt)
            continue
        errors = _validate(data, schema) if isinstance(data, dict) else ["$: expected object"]
        if errors:
            last_error = "; ".join(errors)
            _log(purpose, backend, model, latency_ms, usage, False, attempt)
            continue
        _log(purpose, backend, model, latency_ms, usage, True, attempt)
        return data
    raise LLMValidationError(f"LLM kept returning invalid JSON: {last_error}")
