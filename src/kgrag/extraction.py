from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .documents import TextChunk
from .llm import LLM, ask_json

log = logging.getLogger(__name__)

ENTITY_TYPES = ["ORGANIZATION", "PERSON", "PRODUCT", "LOCATION", "EVENT", "REGULATION", "CONCEPT"]

EXTRACT_SYSTEM = f"""You build a knowledge graph from text.

From the passage, extract:
1. Entities: named things the passage says something about. Type is one of {", ".join(ENTITY_TYPES)}.
   Use the most complete name that appears (e.g. "Halvorsen Motors", not "the company").
   The description is one sentence about the entity based only on this passage.
2. Relationships between those entities that the passage states or clearly implies.
   `type` is a short verb phrase in snake_case (supplies, acquired, located_in, recalled,
   regulated_by, works_for, partnered_with). `strength` is 1-10 for how central and explicit it is.

Do not invent facts. Skip generic nouns that are not specific entities.
Return only JSON:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
 "relationships": [{{"source": "...", "target": "...", "type": "...", "description": "...", "strength": 7}}]}}"""


def extract_chunk(llm: LLM, chunk: TextChunk) -> dict:
    try:
        data = ask_json(llm, EXTRACT_SYSTEM, f"Passage ({chunk.doc}):\n{chunk.text}")
    except (ValueError, KeyError) as e:
        log.warning("extraction failed for %s: %s", chunk.id, e)
        return {"entities": [], "relationships": []}
    data.setdefault("entities", [])
    data.setdefault("relationships", [])
    return data


def extract_all(llm: LLM, chunks: list[TextChunk], workers: int = 4) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: extract_chunk(llm, c), chunks))
