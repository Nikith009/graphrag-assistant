from conftest import FakeLLM

from kgrag.communities import detect_communities
from kgrag.config import CommunityConfig, LocalSearchConfig
from kgrag.embeddings import HashingEmbedder
from kgrag.search import GlobalSearch, LocalSearch


def test_index_builds_merged_graph(graph):
    stats = graph.stats()
    assert stats["chunks"] == 6
    # "Halvorsen Motors" and "Halvorsen Motors, Inc." are one entity
    assert "halvorsen motors" in graph.entities
    assert not any("inc" in key for key in graph.entities)
    kestrel = graph.entities["kestrel energy"]
    assert len(kestrel.chunks) == 4  # mentioned in four documents


def test_communities_split_supply_chain_from_robotics(graph):
    by_id = graph.entity_community
    assert by_id["ardent cells"] == by_id["lumen polymers"]
    assert by_id["kestrel energy"] == by_id["torvik marine"]
    assert by_id["fjord ev recall"] == by_id["european vehicle safety agency"]
    assert by_id["nordlys logistics"] == by_id["polaris vision"]
    assert by_id["ardent cells"] != by_id["polaris vision"]
    assert all(c.summary for c in graph.communities if len(c.entities) >= 2)


def test_louvain_and_leiden_both_run(graph):
    for algo in ("leiden", "louvain"):
        groups = detect_communities(graph, CommunityConfig(algorithm=algo))
        assert sum(len(g) for g in groups) == len(graph.entities)
        assert groups == sorted(groups, key=lambda g: (-len(g), min(g)))


def local(graph, **overrides):
    cfg = LocalSearchConfig(**overrides)
    return LocalSearch(graph, HashingEmbedder(512), FakeLLM(), cfg)


def test_multi_hop_reaches_supplier_two_hops_away(graph):
    """Torvik Marine is never mentioned in the recall document, only linked through Kestrel."""
    result = local(graph).gather("Which ferry companies could be affected by the Fjord EV recall?")
    names = [n for n, _ in result.entities]
    assert "Fjord EV Recall" in names
    assert "Kestrel Energy" in names
    assert "Torvik Marine" in names
    assert "torvik-marine#0" in result.chunks
    assert any("Kestrel Energy -[SUPPLIES]-> Torvik Marine" in r for r in result.relations)


def test_hops_limit_the_walk(graph):
    near = local(graph, hops=0, seed_entities=1).gather("Lumen Polymers")
    assert [n for n, _ in near.entities] == ["Lumen Polymers"]

    far = local(graph, hops=2, seed_entities=1).gather("Lumen Polymers")
    names = {n for n, _ in far.entities}
    assert {"Ardent Cells", "Kestrel Energy"} <= names  # one and two hops out
    assert "Torvik Marine" not in names  # three hops


def test_scores_decay_with_distance(graph):
    result = local(graph, seed_entities=1).gather("Lumen Polymers")
    scores = dict(result.entities)
    assert scores["Lumen Polymers"] > scores["Ardent Cells"] > scores["Kestrel Energy"]


def test_context_is_bounded(graph):
    result = local(graph, max_entities=4, max_chunks=2).gather("Kestrel Energy suppliers")
    assert len(result.entities) == 4
    assert len(result.chunks) == 2
    assert result.context.count("[") >= 2


def test_local_search_calls_llm_with_context(graph):
    searcher = local(graph)
    result = searcher.search("Who traced the cause of the recall?")
    system, prompt = searcher.llm.calls[-1]
    assert result.answer == "ANSWER"
    assert "Ines Kovac" in prompt and prompt.endswith("Who traced the cause of the recall?")


def test_global_search_map_reduce(graph):
    llm = FakeLLM()
    result = GlobalSearch(graph, llm, batch_chars=400).search("What are the main themes?")
    map_calls = [c for c in llm.calls if c[0].startswith("You are given community reports")]
    assert len(map_calls) > 1  # small batches -> several map calls
    assert result.answer == "ANSWER"
    assert result.communities
