"""Query time: local (entity-centric, multi-hop) and global (community map-reduce) search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .config import LocalSearchConfig
from .graph import KnowledgeGraph, Relation, entity_key
from .llm import LLM, ask_json

ANSWER_SYSTEM = """Answer the question using only the knowledge graph context provided.
Connect facts across entities and relationships when the question needs several hops.
Cite source passages in square brackets with their ids, e.g. [recall-2024#1].
If the context does not contain the answer, say so plainly instead of guessing.
Keep it under 150 words."""

MAP_SYSTEM = """You are given community reports from a knowledge graph and a question.
List the points from these reports that help answer the question, each with an importance
score from 0 (useless) to 100 (essential), and the id of the report it came from.
Return only JSON: {"points": [{"text": "...", "score": 80, "community": 3}]}
Return {"points": []} if nothing is relevant."""

REDUCE_SYSTEM = """Write the answer to the question from the analyst points below, which come
from different parts of a knowledge graph. Merge overlapping points, keep the specific facts,
and cite community ids in square brackets, e.g. [community 3]. Under 200 words.
If the points do not answer the question, say so."""


@dataclass
class SearchResult:
    answer: str
    entities: list[tuple[str, float]] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    communities: list[int] = field(default_factory=list)
    context: str = ""


class LocalSearch:
    """Start from the entities a question mentions and walk the graph outwards.

    The walk keeps, for each reachable entity, its best score over paths of up
    to `hops` edges: score(next) = score(current) * hop_decay * edge_factor.
    Only the top-scoring entities, the relations between them and the chunks
    that mention them go into the prompt, which keeps context small and on topic.
    """

    def __init__(self, graph: KnowledgeGraph, embedder, llm: LLM | None, cfg: LocalSearchConfig):
        self.graph = graph
        self.llm = llm
        self.cfg = cfg
        self.embedder = embedder
        self.ids = list(graph.entities)
        self.vectors = (
            embedder.embed([graph.entities[i].text() for i in self.ids])
            if self.ids
            else np.zeros((0, 1), dtype=np.float32)
        )
        self.nx = graph.to_networkx()
        weights = [d["weight"] for _, _, d in self.nx.edges(data=True)]
        self.max_weight = max(weights) if weights else 1.0

    def seed_entities(self, question: str) -> dict[str, float]:
        if not self.ids:
            return {}
        sims = self.vectors @ self.embedder.embed([question])[0]
        qkey = f" {entity_key(question)} "
        scores = {}
        for eid, sim in zip(self.ids, sims):
            score = float(sim)
            # an entity named outright in the question is almost always a seed
            if len(eid) >= 3 and re.search(rf"\b{re.escape(eid)}\b", qkey):
                score += 0.5
            scores[eid] = score
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: self.cfg.seed_entities]
        return {eid: s for eid, s in top if s > 0.05}

    def expand(self, seeds: dict[str, float]) -> dict[str, float]:
        best = dict(seeds)
        frontier = dict(seeds)
        for _ in range(self.cfg.hops):
            nxt: dict[str, float] = {}
            for node, score in frontier.items():
                for nbr, data in self.nx[node].items():
                    edge_factor = 0.5 + 0.5 * min(1.0, data["weight"] / self.max_weight)
                    cand = score * self.cfg.hop_decay * edge_factor
                    if cand > best.get(nbr, 0.0) and cand > nxt.get(nbr, 0.0):
                        nxt[nbr] = cand
            for node, score in nxt.items():
                best[node] = max(best.get(node, 0.0), score)
            frontier = nxt
            if not frontier:
                break
        return best

    def gather(self, question: str) -> SearchResult:
        cfg = self.cfg
        seeds = self.seed_entities(question)
        scores = self.expand(seeds)
        top = sorted(scores.items(), key=lambda kv: -kv[1])[: cfg.max_entities]
        keep = dict(top)

        rels: list[Relation] = [
            r for r in self.graph.relations.values() if r.source in keep and r.target in keep
        ]
        rels.sort(key=lambda r: -(keep[r.source] + keep[r.target]) * r.weight)
        rels = rels[: cfg.max_relations]

        chunk_scores: dict[str, float] = {}
        for eid, score in keep.items():
            for cid in self.graph.entities[eid].chunks:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + score
        for r in rels:
            for cid in r.chunks:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 0.5 * keep[r.source]
        chunk_ids = [c for c, _ in sorted(chunk_scores.items(), key=lambda kv: -kv[1])]
        chunk_ids = chunk_ids[: cfg.max_chunks]

        comm_scores: dict[int, float] = {}
        for eid, score in seeds.items():
            cid = self.graph.entity_community.get(eid)
            if cid is not None:
                comm_scores[cid] = comm_scores.get(cid, 0.0) + score
        comm_ids = [
            c
            for c, _ in sorted(comm_scores.items(), key=lambda kv: -kv[1])
            if self.graph.communities[c].summary
        ][: cfg.max_communities]

        rel_lines = [
            f"{self.graph.entities[r.source].name} -[{r.type}]-> "
            f"{self.graph.entities[r.target].name}: {r.description}"
            for r in rels
        ]
        sections = ["## Entities"]
        sections += [f"- {self.graph.entities[e].text()}" for e in keep]
        sections += ["", "## Relationships", *[f"- {line}" for line in rel_lines]]
        if comm_ids:
            sections += ["", "## Community reports"]
            sections += [self.graph.communities[c].report() for c in comm_ids]
        sections += ["", "## Source passages"]
        sections += [f"[{c}] {self.graph.chunks[c].text}" for c in chunk_ids]

        return SearchResult(
            answer="",
            entities=[(self.graph.entities[e].name, round(s, 3)) for e, s in top],
            relations=rel_lines,
            chunks=chunk_ids,
            communities=comm_ids,
            context="\n".join(sections),
        )

    def search(self, question: str) -> SearchResult:
        result = self.gather(question)
        if self.llm is None:
            raise RuntimeError("LocalSearch needs an LLM to answer; use gather() for context only")
        prompt = f"{result.context}\n\nQuestion: {question}"
        result.answer = self.llm.complete(ANSWER_SYSTEM, prompt, 600).strip()
        return result


class GlobalSearch:
    """Map-reduce over community reports, for questions about the corpus as a whole."""

    def __init__(
        self, graph: KnowledgeGraph, llm: LLM, batch_chars: int = 6000, max_points: int = 20
    ):
        self.graph = graph
        self.llm = llm
        self.batch_chars = batch_chars
        self.max_points = max_points

    def _batches(self) -> list[str]:
        reports = [c for c in self.graph.communities if c.summary]
        reports.sort(key=lambda c: (-c.rating, -len(c.entities)))
        batches, current = [], ""
        for c in reports:
            block = f"[community {c.id}]\n{c.report()}\n\n"
            if current and len(current) + len(block) > self.batch_chars:
                batches.append(current)
                current = ""
            current += block
        if current:
            batches.append(current)
        return batches

    def search(self, question: str) -> SearchResult:
        points = []
        for batch in self._batches():
            data = ask_json(self.llm, MAP_SYSTEM, f"Reports:\n{batch}\nQuestion: {question}")
            points.extend(p for p in data.get("points", []) if float(p.get("score", 0)) > 0)

        if not points:
            return SearchResult(answer="The indexed documents do not cover this question.")

        points.sort(key=lambda p: -float(p["score"]))
        points = points[: self.max_points]
        listing = "\n".join(
            f"- (score {p['score']}, community {p.get('community', '?')}) {p['text']}"
            for p in points
        )
        answer = self.llm.complete(
            REDUCE_SYSTEM, f"Points:\n{listing}\n\nQuestion: {question}", 800
        )
        used = sorted(
            {int(p["community"]) for p in points if str(p.get("community", "")).isdigit()}
        )
        return SearchResult(answer=answer.strip(), communities=used, context=listing)
