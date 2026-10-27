"""Write the knowledge graph to Neo4j, and read it back.

Schema:
    (:Entity {id, name, type, description})
    (:Chunk {id, doc, text})
    (:Community {id, title, summary, rating})
    (:Entity)-[:RELATED {type, weight, description}]->(:Entity)
    (:Entity)-[:MENTIONED_IN]->(:Chunk)
    (:Entity)-[:IN_COMMUNITY]->(:Community)
"""

from __future__ import annotations

import os
from collections import Counter

from .config import Neo4jConfig
from .documents import TextChunk
from .graph import Community, Entity, KnowledgeGraph, Relation

BATCH = 500

SCHEMA = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
]


def _batches(rows: list[dict]):
    for i in range(0, len(rows), BATCH):
        yield rows[i : i + BATCH]


class Neo4jStore:
    def __init__(self, cfg: Neo4jConfig):
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError("install the driver: pip install 'graphrag-assistant[neo4j]'") from e
        password = os.environ.get(cfg.password_env)
        if not password:
            raise RuntimeError(f"{cfg.password_env} is not set")
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, password))
        self.database = cfg.database

    def close(self) -> None:
        self.driver.close()

    def _run(self, query: str, **params):
        records, _, _ = self.driver.execute_query(query, database_=self.database, **params)
        return records

    def write(self, graph: KnowledgeGraph, clear: bool = True) -> None:
        if clear:
            self._run("MATCH (n) WHERE n:Entity OR n:Chunk OR n:Community DETACH DELETE n")
        for stmt in SCHEMA:
            self._run(stmt)

        chunks = [{"id": c.id, "doc": c.doc, "text": c.text} for c in graph.chunks.values()]
        for rows in _batches(chunks):
            self._run(
                "UNWIND $rows AS r MERGE (c:Chunk {id: r.id}) SET c.doc = r.doc, c.text = r.text",
                rows=rows,
            )

        entities = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "names": list(e.names),
                "descriptions": e.descriptions,
                "chunks": e.chunks,
            }
            for e in graph.entities.values()
        ]
        for rows in _batches(entities):
            self._run(
                """
                UNWIND $rows AS r
                MERGE (e:Entity {id: r.id})
                SET e.name = r.name, e.type = r.type, e.description = r.description,
                    e.names = r.names, e.descriptions = r.descriptions
                WITH e, r
                UNWIND r.chunks AS cid
                MATCH (c:Chunk {id: cid})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                rows=rows,
            )

        relations = [
            {
                "source": r.source,
                "target": r.target,
                "type": r.type,
                "weight": r.weight,
                "description": r.description,
                "descriptions": r.descriptions,
                "chunks": r.chunks,
            }
            for r in graph.relations.values()
        ]
        for rows in _batches(relations):
            self._run(
                """
                UNWIND $rows AS r
                MATCH (a:Entity {id: r.source}), (b:Entity {id: r.target})
                MERGE (a)-[rel:RELATED {type: r.type}]->(b)
                SET rel.weight = r.weight, rel.description = r.description,
                    rel.descriptions = r.descriptions, rel.chunks = r.chunks
                """,
                rows=rows,
            )

        communities = [
            {
                "id": c.id,
                "title": c.title,
                "summary": c.summary,
                "findings": c.findings,
                "rating": c.rating,
                "entities": c.entities,
            }
            for c in graph.communities
        ]
        for rows in _batches(communities):
            self._run(
                """
                UNWIND $rows AS r
                MERGE (c:Community {id: r.id})
                SET c.title = r.title, c.summary = r.summary, c.findings = r.findings,
                    c.rating = r.rating
                WITH c, r
                UNWIND r.entities AS eid
                MATCH (e:Entity {id: eid})
                MERGE (e)-[:IN_COMMUNITY]->(c)
                """,
                rows=rows,
            )

    def read(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        for rec in self._run("MATCH (c:Chunk) RETURN c.id AS id, c.doc AS doc, c.text AS text"):
            g.chunks[rec["id"]] = TextChunk(rec["id"], rec["doc"], rec["text"])

        for rec in self._run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN e.id AS id, e.names AS names, e.type AS type,
                   e.descriptions AS descriptions, collect(c.id) AS chunks
            """
        ):
            g.entities[rec["id"]] = Entity(
                id=rec["id"],
                names=Counter(rec["names"] or [rec["id"]]),
                types=Counter([rec["type"] or "UNKNOWN"]),
                descriptions=rec["descriptions"] or [],
                chunks=rec["chunks"],
            )

        for rec in self._run(
            """
            MATCH (a:Entity)-[r:RELATED]->(b:Entity)
            RETURN a.id AS source, b.id AS target, r.type AS type, r.weight AS weight,
                   r.descriptions AS descriptions, r.chunks AS chunks
            """
        ):
            rel = Relation(
                rec["source"],
                rec["target"],
                rec["type"],
                rec["weight"] or 0.0,
                rec["descriptions"] or [],
                rec["chunks"] or [],
            )
            g.relations[(rel.source, rel.target, rel.type)] = rel

        for rec in self._run(
            """
            MATCH (c:Community)
            OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
            RETURN c.id AS id, c.title AS title, c.summary AS summary, c.findings AS findings,
                   c.rating AS rating, collect(e.id) AS entities
            ORDER BY id
            """
        ):
            g.communities.append(
                Community(
                    id=rec["id"],
                    entities=sorted(rec["entities"]),
                    title=rec["title"] or "",
                    summary=rec["summary"] or "",
                    findings=rec["findings"] or [],
                    rating=rec["rating"] or 0.0,
                )
            )
        g.entity_community = {e: c.id for c in g.communities for e in c.entities}
        return g
