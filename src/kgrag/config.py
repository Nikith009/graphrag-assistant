from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    size: int = 250  # words
    overlap: int = 40


class ModelConfig(BaseModel):
    extraction: str = "claude-haiku-5-5"
    summaries: str = "claude-haiku-5-5"
    answers: str = "claude-sonnet-5-5"


class CommunityConfig(BaseModel):
    algorithm: Literal["leiden", "louvain"] = "leiden"
    resolution: float = 1.0
    min_size: int = 2  # communities smaller than this get no report
    seed: int = 42


class EmbedderConfig(BaseModel):
    provider: Literal["hashing", "sentence-transformers"] = "hashing"
    model: str = "all-MiniLM-L6-v2"
    dim: int = 512


class LocalSearchConfig(BaseModel):
    seed_entities: int = 5
    hops: int = 2
    hop_decay: float = 0.6
    max_entities: int = 15
    max_relations: int = 25
    max_chunks: int = 6
    max_communities: int = 2


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    database: str = "neo4j"


class Config(BaseModel):
    corpus: Path
    index_path: Path = Path("index/graph.json")
    cache_dir: Path = Path(".cache")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    communities: CommunityConfig = Field(default_factory=CommunityConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    local_search: LocalSearchConfig = Field(default_factory=LocalSearchConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
