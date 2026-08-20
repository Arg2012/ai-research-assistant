"""End-to-end pipeline: ingest → embed → retrieve → summarise.

The single object you compose to answer a research question over an arXiv corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .embeddings import Chunk, chunk_documents
from .ingestion import Document, ingest_arxiv, documents_from_jsonl, documents_to_jsonl
from .llm import LLM, default_llm
from .retrieval import Retriever
from .summarisation import Summary, summarise


@dataclass
class Pipeline:
    """A composed research pipeline over a fixed document corpus."""

    documents: list[Document]
    llm: LLM = field(default_factory=default_llm)
    embedding_model: str | None = None
    max_words: int = 200
    overlap: int = 40

    def __post_init__(self) -> None:
        self._chunks: list[Chunk] = chunk_documents(
            self.documents, max_words=self.max_words, overlap=self.overlap
        )
        self._retriever = Retriever(self._chunks, model=self.embedding_model)

    @classmethod
    def from_arxiv_query(
        cls,
        query: str,
        *,
        max_results: int = 20,
        download_pdfs: bool = True,
        cache_dir: Path | None = None,
        llm: LLM | None = None,
    ) -> "Pipeline":
        docs = ingest_arxiv(
            query, max_results=max_results, download_pdfs=download_pdfs, cache_dir=cache_dir
        )
        return cls(documents=docs, llm=llm or default_llm())

    @classmethod
    def from_jsonl(cls, path: Path, *, llm: LLM | None = None) -> "Pipeline":
        return cls(documents=documents_from_jsonl(path), llm=llm or default_llm())

    def save_corpus(self, path: Path) -> None:
        documents_to_jsonl(self.documents, path)

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    def ask(self, query: str, *, top_k: int = 5, max_tokens: int = 800) -> Summary:
        hits = self._retriever.query(query, top_k=top_k)
        return summarise(query, hits, llm=self.llm, max_tokens=max_tokens)

    def retrieve_documents(self, query: str, *, top_k: int = 5) -> list[tuple[Document, float]]:
        ranked = self._retriever.query_documents(query, top_k=top_k)
        by_id = {d.arxiv_id: d for d in self.documents}
        return [(by_id[doc_id], score) for doc_id, score in ranked if doc_id in by_id]

    def iter_chunks(self) -> Iterable[Chunk]:
        return iter(self._chunks)
