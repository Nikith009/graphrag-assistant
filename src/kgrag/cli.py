"""kgrag: build and query a GraphRAG index.

kgrag index -c configs/sample.yaml [--neo4j]
kgrag query -c configs/sample.yaml "question" [--mode local|global] [--show-context]
kgrag stats -c configs/sample.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .embeddings import build_embedder
from .graph import KnowledgeGraph
from .indexer import build_index
from .llm import Claude
from .search import GlobalSearch, LocalSearch


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _load_graph(cfg: Config, from_neo4j: bool) -> KnowledgeGraph:
    if from_neo4j:
        from .neo4j_store import Neo4jStore

        store = Neo4jStore(cfg.neo4j)
        try:
            return store.read()
        finally:
            store.close()
    if not cfg.index_path.exists():
        raise SystemExit(f"no index at {cfg.index_path}; run `kgrag index` first")
    return KnowledgeGraph.load(cfg.index_path)


def cmd_index(cfg: Config, args) -> int:
    if args.replay:
        # offline: saved extractions, no community reports
        from .replay import ReplayExtractor

        extract_llm, report_llm = ReplayExtractor(args.replay), None
    else:
        extract_llm = Claude(cfg.models.extraction, cfg.cache_dir)
        report_llm = Claude(cfg.models.summaries, cfg.cache_dir)
    graph = build_index(cfg, extract_llm, report_llm, workers=args.workers)
    graph.save(cfg.index_path)
    print(f"index written to {cfg.index_path}: {graph.stats()}")
    if args.neo4j:
        from .neo4j_store import Neo4jStore

        store = Neo4jStore(cfg.neo4j)
        try:
            store.write(graph)
        finally:
            store.close()
        print(f"graph written to Neo4j at {cfg.neo4j.uri}")
    return 0


def cmd_query(cfg: Config, args) -> int:
    graph = _load_graph(cfg, args.neo4j)
    if args.context_only:
        if args.mode == "global":
            raise SystemExit("global search needs a model; drop --context-only")
        searcher = LocalSearch(graph, build_embedder(cfg.embedder), None, cfg.local_search)
        result = searcher.gather(args.question)
    else:
        llm = Claude(cfg.models.answers, cfg.cache_dir)
        if args.mode == "global":
            result = GlobalSearch(graph, llm).search(args.question)
        else:
            searcher = LocalSearch(graph, build_embedder(cfg.embedder), llm, cfg.local_search)
            result = searcher.search(args.question)

    if result.answer:
        print(result.answer)
    if args.show_context or args.context_only:
        print("\n--- entities (score) ---")
        for name, score in result.entities:
            print(f"{score:6.3f}  {name}")
        print("\n--- relations ---")
        print("\n".join(result.relations) or "(none)")
        print("\n--- chunks ---")
        print(", ".join(result.chunks) or "(none)")
        print("\n--- communities ---")
        print(", ".join(str(c) for c in result.communities) or "(none)")
    return 0


def cmd_stats(cfg: Config, args) -> int:
    graph = _load_graph(cfg, args.neo4j)
    print(graph.stats())
    top = sorted(graph.communities, key=lambda c: -len(c.entities))[:10]
    for c in top:
        names = ", ".join(graph.entities[e].name for e in c.entities[:6])
        print(f"community {c.id} ({len(c.entities)}): {c.title or '-'} | {names}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="kgrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build the graph from the corpus")
    p_index.add_argument("--workers", type=int, default=4)
    p_index.add_argument(
        "--replay", metavar="JSON", help="use saved extractions instead of Claude (offline)"
    )

    p_query = sub.add_parser("query", help="ask a question")
    p_query.add_argument("question")
    p_query.add_argument("--mode", choices=["local", "global"], default="local")
    p_query.add_argument("--show-context", action="store_true")
    p_query.add_argument(
        "--context-only",
        action="store_true",
        help="print the retrieved graph context without calling a model",
    )

    sub.add_parser("stats", help="show graph size and largest communities")

    for p in sub.choices.values():
        p.add_argument("-c", "--config", default="configs/sample.yaml")
        p.add_argument("--neo4j", action="store_true", help="write to / read from Neo4j")
        p.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    cfg = Config.from_yaml(args.config)
    return {"index": cmd_index, "query": cmd_query, "stats": cmd_stats}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
