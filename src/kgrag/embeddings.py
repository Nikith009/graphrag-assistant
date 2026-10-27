from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import EmbedderConfig

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    """a an the and or of to in on for is are was were be by with as at it that this from what
    how which who whom does do did can if its their they we our has have had""".split()
)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (v / n).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words vectors; no downloads, good enough for tests and demos."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]
            for feat in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
                h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(np.sign(out) * np.log1p(np.abs(out)))


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(self._model.encode(texts, batch_size=32), dtype=np.float32))


def build_embedder(cfg: EmbedderConfig):
    if cfg.provider == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model)
    return HashingEmbedder(cfg.dim)
