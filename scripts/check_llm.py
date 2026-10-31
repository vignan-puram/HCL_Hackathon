"""One tiny JSON-mode call per configured provider. Owner: Member 1.

Run from repo root: python -m scripts.check_llm
"""

import json
import time

import config

PROMPT = 'Reply with JSON only: {"ok": true}'


def check_gemini() -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=PROMPT,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return resp.text


def check_groq() -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def main() -> None:
    providers = [
        ("gemini", config.GEMINI_MODEL, config.GEMINI_API_KEY, check_gemini),
        ("groq", config.GROQ_MODEL, config.GROQ_API_KEY, check_groq),
    ]
    for name, model, key, fn in providers:
        if not key:
            print(f"[SKIP] {name}: no API key in .env")
            continue
        start = time.perf_counter()
        try:
            data = json.loads(fn())
            ms = int((time.perf_counter() - start) * 1000)
            print(f"[OK]   {name} {model} {ms} ms -> {data}")
        except Exception as exc:
            print(f"[FAIL] {name} {model}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
