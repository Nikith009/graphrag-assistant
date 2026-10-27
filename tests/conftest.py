import json
import re
from pathlib import Path

import pytest

from kgrag.config import Config
from kgrag.indexer import build_index

ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = json.loads((ROOT / "data" / "sample-extractions.json").read_text())


class FakeLLM:
    """Answers each prompt type with canned JSON so indexing and search run offline."""

    def __init__(self):
        self.calls = []

    def complete(self, system, prompt, max_tokens=1024):
        self.calls.append((system, prompt))
        if system.startswith("You build a knowledge graph"):
            doc = re.search(r"Passage \(([^)]+)\)", prompt).group(1)
            return json.dumps(EXTRACTIONS[doc])
        if system.startswith("You write a short report"):
            names = re.findall(r"^- ([^(\n]+) \(", prompt, flags=re.M)
            return json.dumps(
                {
                    "title": " / ".join(names[:2]),
                    "summary": "Entities: " + ", ".join(names),
                    "findings": [f"{n} is part of this group" for n in names[:3]],
                    "rating": len(names),
                }
            )
        if system.startswith("You are given community reports"):
            ids = re.findall(r"\[community (\d+)\]", prompt)
            return json.dumps(
                {
                    "points": [
                        {"text": f"point from {i}", "score": 50, "community": int(i)} for i in ids
                    ]
                }
            )
        return "ANSWER"


@pytest.fixture
def config(tmp_path):
    return Config(
        corpus=ROOT / "data" / "sample",
        index_path=tmp_path / "graph.json",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def graph(config):
    llm = FakeLLM()
    return build_index(config, llm, llm, workers=2)
