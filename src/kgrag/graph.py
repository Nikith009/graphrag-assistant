"""The knowledge graph: entities, relations, source chunks and communities."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from .documents import TextChunk

_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "llc",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "group",
    "holdings",
}


def entity_key(name: str) -> str:
    """Normalize a surface name so 'Halvorsen Motors, Inc.' and 'halvorsen motors' merge."""
    s = name.casefold().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^\w&\s-]", " ", s)
    words = s.split()
    if words and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words = words[:-1]
    return " ".join(words)


@dataclass
class Entity:
    id: str
    names: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    descriptions: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.names.most_common(1)[0][0] if self.names else self.id

    @property
    def type(self) -> str:
        return self.types.most_common(1)[0][0] if self.types else "UNKNOWN"

    @property
    def description(self) -> str:
        return " ".join(self.descriptions[:3])

    def text(self) -> str:
        return f"{self.name} ({self.type}): {self.description}"


@dataclass
class Relation:
    source: str
    target: str
    type: str
    weight: float = 0.0
    descriptions: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return " ".join(self.descriptions[:2])


@dataclass
class Community:
    id: int
    entities: list[str]
    title: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    rating: float = 0.0

    def report(self) -> str:
        bullets = "\n".join(f"- {f}" for f in self.findings)
        return f"# {self.title}\n{self.summary}\n{bullets}".strip()


class KnowledgeGraph:
    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str, str], Relation] = {}
        self.chunks: dict[str, TextChunk] = {}
        self.communities: list[Community] = []
        self.entity_community: dict[str, int] = {}

    # --- building --------------------------------------------------------

    def add_entity(self, name: str, type_: str, description: str, chunk_id: str) -> str | None:
        key = entity_key(name)
        if not key:
            return None
        ent = self.entities.setdefault(key, Entity(id=key))
        ent.names[name.strip()] += 1
        if type_:  # relation endpoints come without a type; don't let them outvote real ones
            ent.types[type_.upper()] += 1
        if description and description not in ent.descriptions:
            ent.descriptions.append(description.strip())
        if chunk_id not in ent.chunks:
            ent.chunks.append(chunk_id)
        return key

    def add_relation(
        self, source: str, target: str, type_: str, description: str, weight: float, chunk_id: str
    ) -> None:
        src, tgt = self._endpoint(source, chunk_id), self._endpoint(target, chunk_id)
        if not src or not tgt or src == tgt:
            return
        rtype = re.sub(r"\W+", "_", (type_ or "related_to").strip().upper()).strip("_")
        rel = self.relations.setdefault((src, tgt, rtype), Relation(src, tgt, rtype))
        rel.weight += max(weight, 0.1)
        if description and description not in rel.descriptions:
            rel.descriptions.append(description.strip())
        if chunk_id not in rel.chunks:
            rel.chunks.append(chunk_id)

    def _endpoint(self, name: str, chunk_id: str) -> str | None:
        """Resolve a relation endpoint; only create the entity if extraction missed it."""
        key = entity_key(name)
        if not key:
            return None
        if key not in self.entities:
            return self.add_entity(name, "", "", chunk_id)
        return key

    def add_extraction(self, chunk: TextChunk, data: dict) -> None:
        self.chunks[chunk.id] = chunk
        for e in data.get("entities", []):
            if e.get("name"):
                self.add_entity(e["name"], e.get("type", ""), e.get("description", ""), chunk.id)
        for r in data.get("relationships", []):
            if r.get("source") and r.get("target"):
                self.add_relation(
                    r["source"],
                    r["target"],
                    r.get("type", "related_to"),
                    r.get("description", ""),
                    float(r.get("strength", 5)) / 10,
                    chunk.id,
                )

    # --- queries ---------------------------------------------------------

    def to_networkx(self) -> nx.Graph:
        """Undirected weighted graph; parallel relations between a pair are summed."""
        g = nx.Graph()
        g.add_nodes_from(self.entities)
        for rel in self.relations.values():
            if g.has_edge(rel.source, rel.target):
                g[rel.source][rel.target]["weight"] += rel.weight
            else:
                g.add_edge(rel.source, rel.target, weight=rel.weight)
        return g

    def relations_of(self, entity_id: str) -> list[Relation]:
        return [r for r in self.relations.values() if entity_id in (r.source, r.target)]

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "communities": len(self.communities),
        }

    # --- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [asdict(c) for c in self.chunks.values()],
            "entities": [
                {**asdict(e), "names": dict(e.names), "types": dict(e.types)}
                for e in self.entities.values()
            ],
            "relations": [asdict(r) for r in self.relations.values()],
            "communities": [asdict(c) for c in self.communities],
        }
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        data = json.loads(Path(path).read_text("utf-8"))
        g = cls()
        for c in data["chunks"]:
            g.chunks[c["id"]] = TextChunk(**c)
        for e in data["entities"]:
            g.entities[e["id"]] = Entity(
                id=e["id"],
                names=Counter(e["names"]),
                types=Counter(e["types"]),
                descriptions=e["descriptions"],
                chunks=e["chunks"],
            )
        for r in data["relations"]:
            g.relations[(r["source"], r["target"], r["type"])] = Relation(**r)
        g.communities = [Community(**c) for c in data["communities"]]
        g.entity_community = {e: c.id for c in g.communities for e in c.entities}
        return g
