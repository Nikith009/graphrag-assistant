"""Replay saved extraction results instead of calling a model.

Lets the whole index -> communities -> local search pipeline run offline from a
JSON file mapping document id -> {"entities": [...], "relationships": [...]}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DOC = re.compile(r"Passage \(([^)]+)\)")


class ReplayExtractor:
    def __init__(self, path: Path):
        self.by_doc: dict[str, dict] = json.loads(Path(path).read_text("utf-8"))

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        match = _DOC.search(prompt)
        data = self.by_doc.get(match.group(1), {}) if match else {}
        return json.dumps({"entities": [], "relationships": [], **data})
