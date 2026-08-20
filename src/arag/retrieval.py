"""In-memory dense retrieval over embedded chunks.

Uses cosine similarity via a single normalized-matrix multiply (numpy). Not
FAISS — for corpora of thousands of chunks this is O(ms) and avoids a heavy
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import Chunk, embed


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    """Dense retriever over a fixed set of chunks."""

    def __init__(self, chunks: Sequence[Chunk], *, model: str | None = None):
        self.chunks: list[Chunk] = list(chunks)
        self._model = model
        if self.chunks:
            texts = [c.text for c in self.chunks]
            self._matrix = embed(texts, model=model) if model else embed(texts)
        else:
            self._matrix = np.zeros((0, 384), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def query(self, text: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        q = embed([text], model=self._model) if self._model else embed([text])
        # Both matrix rows and q rows are already L2-normalised (embed does it),
        # so dot product == cosine similarity.
        sims = (self._matrix @ q[0]).astype(float)
        top_k = min(top_k, len(self.chunks))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(sims[i])) for i in idx]

    def query_documents(self, text: str, *, top_k: int = 5) -> list[tuple[str, float]]:
        """Return top documents (best-scoring chunk per doc)."""
        hits = self.query(text, top_k=top_k * 4)  # over-fetch to allow dedup
        best: dict[str, float] = {}
        for h in hits:
            prev = best.get(h.chunk.doc_id)
            if prev is None or h.score > prev:
                best[h.chunk.doc_id] = h.score
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]
