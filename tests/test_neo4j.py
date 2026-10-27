"""Round-trip the sample graph through a real Neo4j instance.

Skipped unless NEO4J_URI is set; CI runs it against a Neo4j service container.
"""

import os

import pytest

from kgrag.config import LocalSearchConfig, Neo4jConfig
from kgrag.embeddings import HashingEmbedder
from kgrag.search import LocalSearch


@pytest.fixture
def store():
    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI not set")
    from kgrag.neo4j_store import Neo4jStore

    s = Neo4jStore(Neo4jConfig(uri=os.environ["NEO4J_URI"]))
    yield s
    s.close()


def test_write_then_read_gives_the_same_graph(graph, store):
    store.write(graph)
    loaded = store.read()

    assert loaded.stats() == graph.stats()
    assert set(loaded.relations) == set(graph.relations)
    assert loaded.entity_community == graph.entity_community
    for eid, ent in graph.entities.items():
        assert loaded.entities[eid].name == ent.name
        assert loaded.entities[eid].type == ent.type
        assert set(loaded.entities[eid].chunks) == set(ent.chunks)
    for a, b in zip(graph.communities, loaded.communities):
        assert (a.id, a.entities, a.summary) == (b.id, b.entities, b.summary)


def test_search_on_neo4j_graph_matches_json_graph(graph, store):
    store.write(graph)
    loaded = store.read()
    cfg = LocalSearchConfig()
    question = "Which ferry companies could be affected by the Fjord EV recall?"
    from_json = LocalSearch(graph, HashingEmbedder(512), None, cfg).gather(question)
    from_neo4j = LocalSearch(loaded, HashingEmbedder(512), None, cfg).gather(question)
    assert dict(from_neo4j.entities) == pytest.approx(dict(from_json.entities))
    assert set(from_neo4j.chunks) == set(from_json.chunks)


def test_multi_hop_cypher_query(graph, store):
    store.write(graph)
    records = store._run(
        """
        MATCH (:Entity {id: 'fjord ev recall'})-[:RELATED*1..2]-(c:Entity {type: 'ORGANIZATION'})
        RETURN DISTINCT c.name AS name
        """
    )
    names = {r["name"] for r in records}
    assert {"Kestrel Energy", "Torvik Marine", "Ardent Cells"} <= names
