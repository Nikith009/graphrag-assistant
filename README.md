# graphrag-assistant

Question answering over a document collection using a knowledge graph. Plain vector RAG retrieves passages that look like the question, which breaks down when the answer is spread across documents that never mention each other. This project extracts entities and relationships into a graph, groups them into communities with the Leiden algorithm, and answers questions by walking the graph across several hops before anything reaches the model.

The sample corpus shows the difference. The recall report names Halvorsen Motors, Kestrel Energy and Ardent Cells, but never Torvik Marine. *"Which ferry companies could be affected by the Fjord EV recall?"* needs the hop recall → Kestrel Energy → Torvik Marine, which lives in a different document. Similarity search over chunks does not make that connection; graph traversal does.

## Pipeline

```
documents ─► sentence-aligned chunks ─► entity + relation extraction (Claude)
          ─► entity resolution + merge ─► knowledge graph ─► Leiden communities
          ─► community reports (Claude) ─► JSON index and/or Neo4j
```

**Extraction.** One call per chunk returns entities (name, type, description) and relationships (source, target, type, description, strength 1–10) as JSON.

**Entity resolution.** Names are normalized before merging: case, punctuation, possessives, a leading "the" and corporate suffixes are stripped, so `Halvorsen Motors, Inc.` and `halvorsen motors` become one node. Descriptions, source chunks and relation weights accumulate across mentions; the display name and type are the most frequent ones seen.

**Communities.** The graph is clustered with Leiden on modularity (`networkx.community.leiden_communities`). Leiden's refinement step guarantees connected communities, which Louvain does not (Louvain is available as `algorithm: louvain` for comparison). Every community with at least two members gets a report: title, summary, key findings and an importance rating.

## Retrieval

**Local search** (specific questions):

1. **Seed** entities by embedding similarity between the question and each entity's name + description, with a boost when the question names the entity outright.
2. **Expand** up to `hops` edges out. Each reachable entity keeps its best score over all paths:
   `score(next) = score(current) × hop_decay × edge_factor`, where `edge_factor` grows with relation weight. Strong, close connections win; weak, distant ones fade out.
3. **Prune** to the top `max_entities`, keep the relations between them, and rank source chunks by the summed score of the entities they mention.
4. Add the reports of the seeds' communities and send the assembled context to Claude, which answers with chunk citations like `[torvik-marine#0]`.

Pruning before generation is what keeps answers grounded: the model sees around 15 entities, their relations and a handful of passages instead of every chunk that happens to share words with the question.

**Global search** (questions about the whole corpus, such as *"What are the main themes?"*): a map-reduce over community reports. Reports are batched; each batch is asked for relevant points with importance scores; the top points are merged into one answer citing community ids.

## Usage

```bash
pip install -e ".[neo4j,embeddings,dev]"
cp .env.example .env               # ANTHROPIC_API_KEY, NEO4J_PASSWORD

kgrag index -c configs/sample.yaml -v
kgrag stats -c configs/sample.yaml
kgrag query -c configs/sample.yaml "Which ferry companies could be affected by the Fjord EV recall?" --show-context
kgrag query -c configs/sample.yaml "What are the main themes across these documents?" --mode global
```

`--show-context` prints the scored entities, relations, chunks and communities that went into the prompt, which is the easiest way to see the multi-hop walk.

LLM responses are cached in `.cache/`, so re-indexing after a small change only pays for new prompts.

### Neo4j

```bash
docker compose up -d                      # Neo4j at http://localhost:7474
kgrag index -c configs/sample.yaml --neo4j
kgrag query -c configs/sample.yaml "..." --neo4j
```

Schema: `(:Entity)-[:RELATED {type, weight}]->(:Entity)`, `(:Entity)-[:MENTIONED_IN]->(:Chunk)`, `(:Entity)-[:IN_COMMUNITY]->(:Community)`. A multi-hop question in Cypher:

```cypher
MATCH path = (r:Entity {id: "fjord ev recall"})-[:RELATED*1..2]-(c:Entity)
WHERE c.type = "ORGANIZATION"
RETURN c.name, [n IN nodes(path) | n.name] AS via
```

## Configuration

See [`configs/sample.yaml`](configs/sample.yaml). The settings that matter most:

| Setting | Effect |
|---|---|
| `chunking.size` | Words per chunk. Smaller chunks mean more extraction calls but finer-grained mentions. |
| `communities.resolution` | Above 1 gives more, smaller communities; below 1 gives fewer, larger ones. |
| `local_search.hops` | How far the walk goes. Two is usually enough; three adds noise. |
| `local_search.hop_decay` | How fast scores drop per hop. |
| `embedder.provider` | `sentence-transformers` for real use, `hashing` for offline tests. |

## Development

```bash
pytest -q
ruff check . && ruff format --check .
```

Tests run offline. A fake LLM returns fixed extractions for the sample corpus, so they check the real merge, community, traversal and pruning logic, including that the walk reaches Torvik Marine from the recall in two hops and stops at the hop limit.

## Layout

```
src/kgrag/
  documents.py      loading and sentence-aligned chunking
  extraction.py     entity / relation extraction prompt
  graph.py          knowledge graph, entity resolution, JSON persistence
  communities.py    Leiden / Louvain and community reports
  search.py         local (multi-hop) and global (map-reduce) search
  neo4j_store.py    write to / read from Neo4j
  indexer.py        indexing pipeline
  llm.py            Claude client with response cache
  cli.py            `kgrag` command
```
