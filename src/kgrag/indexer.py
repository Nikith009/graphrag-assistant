from __future__ import annotations

import logging

from .communities import build_communities
from .config import Config
from .documents import chunk_text, load_documents
from .extraction import extract_all
from .graph import KnowledgeGraph
from .llm import LLM

log = logging.getLogger(__name__)


def build_index(
    cfg: Config, extract_llm: LLM, report_llm: LLM | None, workers: int = 4
) -> KnowledgeGraph:
    """documents -> chunks -> entities/relations -> communities -> reports."""
    docs = load_documents(cfg.corpus)
    chunks = [
        chunk
        for doc, text in docs.items()
        for chunk in chunk_text(doc, text, cfg.chunking.size, cfg.chunking.overlap)
    ]
    log.info("%d documents -> %d chunks", len(docs), len(chunks))

    graph = KnowledgeGraph()
    for chunk, data in zip(chunks, extract_all(extract_llm, chunks, workers)):
        graph.add_extraction(chunk, data)
    log.info("extracted %d entities, %d relations", len(graph.entities), len(graph.relations))

    build_communities(graph, cfg.communities, report_llm, workers)
    return graph
