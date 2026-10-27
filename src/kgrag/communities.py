"""Community detection and community reports.

Leiden (Traag et al., 2019) improves on Louvain by adding a refinement step
that guarantees every community is internally connected; Louvain can produce
communities that are split in two. Both optimize modularity here, and
`resolution` trades fewer, larger communities (< 1) for more, smaller ones (> 1).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import networkx as nx

from .config import CommunityConfig
from .graph import Community, KnowledgeGraph
from .llm import LLM, ask_json

log = logging.getLogger(__name__)

REPORT_SYSTEM = """You write a short report about one community of related entities in a
knowledge graph, so that it can answer broad questions later.
Use only the entities and relationships given.
Return only JSON:
{"title": "short name for the group", "summary": "2-4 sentences on what ties it together",
 "findings": ["3-5 specific, factual statements"], "rating": <0-10 importance>}"""


def detect_communities(graph: KnowledgeGraph, cfg: CommunityConfig) -> list[list[str]]:
    g = graph.to_networkx()
    if g.number_of_nodes() == 0:
        return []
    if cfg.algorithm == "leiden":
        groups = nx.community.leiden_communities(
            g, weight="weight", resolution=cfg.resolution, metric="modularity", seed=cfg.seed
        )
    else:
        groups = nx.community.louvain_communities(
            g, weight="weight", resolution=cfg.resolution, seed=cfg.seed
        )
    # largest first, ids stable for a given graph + seed
    return [sorted(c) for c in sorted(groups, key=lambda c: (-len(c), min(c)))]


def community_context(graph: KnowledgeGraph, members: list[str], max_chars: int = 8000) -> str:
    member_set = set(members)
    lines = ["Entities:"]
    for eid in members:
        lines.append(f"- {graph.entities[eid].text()}")
    lines.append("Relationships:")
    rels = [
        r for r in graph.relations.values() if r.source in member_set and r.target in member_set
    ]
    for r in sorted(rels, key=lambda r: -r.weight):
        src, tgt = graph.entities[r.source].name, graph.entities[r.target].name
        lines.append(f"- {src} {r.type} {tgt}: {r.description}")
    return "\n".join(lines)[:max_chars]


def build_communities(
    graph: KnowledgeGraph, cfg: CommunityConfig, llm: LLM | None, workers: int = 4
) -> None:
    groups = detect_communities(graph, cfg)
    communities = [Community(id=i, entities=members) for i, members in enumerate(groups)]
    graph.communities = communities
    graph.entity_community = {e: c.id for c in communities for e in c.entities}

    to_report = [c for c in communities if len(c.entities) >= cfg.min_size]
    log.info("%d communities, %d with reports", len(communities), len(to_report))
    if llm is None:
        return

    def report(c: Community) -> None:
        try:
            data = ask_json(llm, REPORT_SYSTEM, community_context(graph, c.entities))
        except ValueError as e:
            log.warning("community %d report failed: %s", c.id, e)
            return
        c.title = str(data.get("title", "")).strip()
        c.summary = str(data.get("summary", "")).strip()
        c.findings = [str(f) for f in data.get("findings", [])][:6]
        c.rating = float(data.get("rating", 0) or 0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(report, to_report))
