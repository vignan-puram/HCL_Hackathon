"""LLM client: one JSON-mode call, retry, model fallback chain, disk cache.

Plain HTTP via requests -- no provider SDKs. Providers:
  gemini  -> Google Generative Language REST API (generateContent)
  openai  -> any OpenAI-compatible /chat/completions endpoint (Groq, etc.)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import requests

import config


class LLMError(Exception):
    pass


_cooldown_until: dict[str, float] = {}  # model -> time it may be tried again


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool


def models() -> list[str]:
    if config.LLM_PROVIDER == "gemini":
        chain = [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS
        return list(dict.fromkeys(chain))  # dedupe, keep order
    return [config.OPENAI_MODEL]


def is_configured() -> bool:
    key = config.GEMINI_API_KEY if config.LLM_PROVIDER == "gemini" else config.OPENAI_API_KEY
    return bool(key)


def _call_gemini(model: str, system: str, user: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    resp = requests.post(
        url, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=config.LLM_TIMEOUT_S
    )
    if resp.status_code != 200:
        raise LLMError(f"{model} HTTP {resp.status_code}: {resp.text[:200]}")
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _call_openai(model: str, system: str, user: str) -> str:
    resp = requests.post(
        f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
        timeout=config.LLM_TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise LLMError(f"{model} HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


def _cache_path(system: str, user: str):
    # keyed on the prompt only, so a cached answer survives a fallback to another model
    key = hashlib.sha256(f"{config.LLM_PROVIDER}\n{system}\n{user}".encode()).hexdigest()
    return config.LLM_CACHE_DIR / f"{key}.json"


def complete_json(system: str, user: str, use_cache: bool = True) -> LLMResponse:
    """Return the model's JSON text. Raises LLMError when every model fails."""
    if not is_configured():
        raise LLMError(f"no API key configured for provider '{config.LLM_PROVIDER}'")
    cache_file = _cache_path(system, user)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return LLMResponse(cached["text"], cached["model"], cached=True)

    call = _call_gemini if config.LLM_PROVIDER == "gemini" else _call_openai
    errors: list[str] = []
    chain = [m for m in models() if _cooldown_until.get(m, 0) <= time.time()] or models()
    for model in chain:
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                text = call(model, system, user)
            except (requests.RequestException, LLMError, KeyError, IndexError) as exc:
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:120]}")
                if attempt < config.LLM_RETRIES and not isinstance(exc, requests.Timeout):
                    time.sleep(1.5)
                else:
                    _cooldown_until[model] = time.time() + config.LLM_COOLDOWN_S
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError:
                return LLMResponse(text, model, cached=False)  # generator attempts a repair
            config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": model, "text": text}), encoding="utf-8")
            return LLMResponse(text, model, cached=False)
    raise LLMError("; ".join(errors[-3:]))
