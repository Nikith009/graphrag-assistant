from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    id: str  # "<doc>#<n>"
    doc: str
    text: str


def load_documents(folder: Path) -> dict[str, str]:
    folder = Path(folder)
    docs = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            docs[path.relative_to(folder).with_suffix("").as_posix()] = path.read_text("utf-8")
    if not docs:
        raise ValueError(f"no .md or .txt files in {folder}")
    return docs


def chunk_text(doc: str, text: str, size: int, overlap: int) -> list[TextChunk]:
    """Sentence-aligned chunks of roughly `size` words with ~`overlap` words of overlap.

    Entity extraction works much better when sentences are not cut in half.
    """
    sentences = [s.strip() for s in SENTENCE_END.split(" ".join(text.split())) if s.strip()]
    chunks: list[TextChunk] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > size:
            chunks.append(TextChunk(f"{doc}#{len(chunks)}", doc, " ".join(current)))
            tail: list[str] = []
            tail_n = 0
            for prev in reversed(current):
                tail_n += len(prev.split())
                if tail_n > overlap:
                    break
                tail.insert(0, prev)
            current, count = tail, sum(len(s.split()) for s in tail)
        current.append(sent)
        count += n
    if current:
        chunks.append(TextChunk(f"{doc}#{len(chunks)}", doc, " ".join(current)))
    return chunks
