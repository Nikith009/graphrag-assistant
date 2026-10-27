from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class Claude:
    """Anthropic client with an on-disk response cache.

    Indexing makes one call per chunk plus one per community, so caching
    makes re-indexing after a small change nearly free.
    """

    def __init__(self, model: str, cache_dir: Path | None = None, client=None):
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            client = anthropic.Anthropic(max_retries=5)
        self.client = client
        self.model = model
        self._cache_path = Path(cache_dir) / "llm_cache.jsonl" if cache_dir else None
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if self._cache_path and self._cache_path.exists():
            for line in self._cache_path.read_text("utf-8").splitlines():
                row = json.loads(line)
                self._cache[row["k"]] = row["v"]

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        key = hashlib.sha256(
            json.dumps([self.model, system, prompt, max_tokens]).encode()
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        with self._lock:
            self._cache[key] = text
            if self._cache_path:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": text}) + "\n")
        return text


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    match = _JSON.search(text)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def ask_json(llm: LLM, system: str, prompt: str, max_tokens: int = 2048) -> dict:
    reply = llm.complete(system, prompt, max_tokens)
    try:
        return parse_json(reply)
    except ValueError:
        reply = llm.complete(system, prompt + "\n\nReturn only the JSON object.", max_tokens)
        return parse_json(reply)
