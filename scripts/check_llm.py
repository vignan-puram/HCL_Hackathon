"""One tiny JSON-mode call to the configured provider (walks the fallback chain).

Run from repo root: python -m scripts.check_llm
"""

import json
import time

import config
from rag import llm


def main() -> None:
    print(f"provider={config.LLM_PROVIDER} models={llm.models()}")
    start = time.perf_counter()
    try:
        resp = llm.complete_json("You reply with JSON only.", 'Return {"ok": true}', use_cache=False)
        ms = int((time.perf_counter() - start) * 1000)
        print(f"[OK]   {resp.model} {ms} ms -> {json.loads(resp.text)}")
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
