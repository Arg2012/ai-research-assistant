"""Semantic embeddings.

Uses sentence-transformers locally (no external API key needed). Loading the
model is expensive, so the encoder is cached module-wide.

Also exposes a `chunk_text` helper — long PDFs are chunked into overlapping
windows so retrieval can score at paragraph-level granularity rather than only
per-paper.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


DEFAULT_MODEL = os.environ.get(
    "ARAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@lru_cache(maxsize=4)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: Iterable[str], *, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Encode texts to L2-normalised embeddings. Returns shape (n, dim)."""
    texts_list = list(texts)
    if not texts_list:
        return np.zeros((0, 384), dtype=np.float32)
    enc = _encoder(model)
    vecs = enc.encode(texts_list, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(text: str, *, max_words: int = 200, overlap: int = 40) -> list[str]:
    """Naive fixed-window word chunking with overlap.

    Good enough for retrieval-grade granularity on arXiv abstracts / body text.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), step)
        if i < len(words)
    ]


def chunk_documents(docs, *, max_words: int = 200, overlap: int = 40) -> list[Chunk]:
    """Explode documents into chunks. Doc id = arxiv_id."""
    chunks: list[Chunk] = []
    for d in docs:
        for i, piece in enumerate(chunk_text(d.text or d.abstract, max_words=max_words, overlap=overlap)):
            chunks.append(Chunk(doc_id=d.arxiv_id, chunk_id=i, text=piece))
    return chunks
