from kgrag.documents import TextChunk, chunk_text
from kgrag.graph import KnowledgeGraph, entity_key


def test_entity_key_merges_surface_forms():
    assert entity_key("Halvorsen Motors, Inc.") == "halvorsen motors"
    assert entity_key("The Kestrel Energy Corporation") == "kestrel energy"
    assert entity_key("Ardent Cells's") == "ardent cells"
    assert entity_key("Park Ji-ho") == "park ji-ho"
    assert entity_key("Co") == "co"  # never strip a name down to nothing


def test_chunk_text_keeps_sentences_whole():
    text = " ".join(f"Sentence {i} has five words." for i in range(20))
    chunks = chunk_text("doc", text, size=22, overlap=5)
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)
    assert chunks[0].id == "doc#0" and chunks[1].id == "doc#1"
    # one sentence of overlap between neighbours
    assert chunks[1].text.startswith(chunks[0].text.split(". ")[-1].rstrip("."))


def test_add_extraction_merges_entities_and_relations():
    g = KnowledgeGraph()
    g.add_extraction(
        TextChunk("a#0", "a", "..."),
        {
            "entities": [{"name": "Kestrel Energy", "type": "ORGANIZATION", "description": "x"}],
            "relationships": [
                {
                    "source": "Kestrel Energy",
                    "target": "Halvorsen Motors",
                    "type": "supplies",
                    "description": "packs",
                    "strength": 8,
                }
            ],
        },
    )
    g.add_extraction(
        TextChunk("b#0", "b", "..."),
        {
            "entities": [
                {"name": "Halvorsen Motors, Inc.", "type": "organization", "description": "y"}
            ],
            "relationships": [
                {
                    "source": "kestrel energy",
                    "target": "Halvorsen Motors, Inc.",
                    "type": "Supplies",
                    "description": "packs again",
                    "strength": 6,
                }
            ],
        },
    )
    assert set(g.entities) == {"kestrel energy", "halvorsen motors"}
    rel = g.relations[("kestrel energy", "halvorsen motors", "SUPPLIES")]
    assert rel.weight == 0.8 + 0.6
    assert rel.chunks == ["a#0", "b#0"]
    assert g.entities["halvorsen motors"].type == "ORGANIZATION"
    assert g.entities["kestrel energy"].name == "Kestrel Energy"


def test_self_relations_are_ignored():
    g = KnowledgeGraph()
    g.add_relation("Ardent Cells", "Ardent Cells, Ltd.", "same_as", "", 1.0, "c#0")
    assert g.relations == {}


def test_index_roundtrip(graph, tmp_path):
    path = tmp_path / "g.json"
    graph.save(path)
    loaded = KnowledgeGraph.load(path)
    assert loaded.stats() == graph.stats()
    assert loaded.entity_community == graph.entity_community
    assert loaded.entities["kestrel energy"].name == "Kestrel Energy"
