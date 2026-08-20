"""Automated evaluation against reference material.

Two axes are measured:

1. Retrieval quality — given a query with known-relevant `gold_doc_ids`, does
   the retriever surface them in the top-k? Reported as recall@k, precision@k
   and mean reciprocal rank (MRR).

2. Generation quality — does the generated summary resemble a hand-written
   reference summary? Reported as ROUGE-L F1 (lexical overlap) and cosine
   similarity of sentence-embeddings (semantic overlap).

The reference dataset lives under `data/reference/`:
  * `benchmark.jsonl` — corpus of documents (see `arag.ingestion.Document`)
  * `queries.jsonl` — evaluation queries, each with `gold_doc_ids` and
    `reference_summary`

Nothing here requires an API key: retrieval metrics use sentence-transformers
locally, and generation is scored against the reference summary using ROUGE
plus a semantic cosine (also from sentence-transformers). If an
`AnthropicLLM` is available it is used for the generated summary; otherwise the
extractive fallback is used and evaluated honestly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

from .embeddings import embed
from .ingestion import Document
from .pipeline import Pipeline


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


@dataclass
class EvalQuery:
    id: str
    query: str
    gold_doc_ids: list[str]
    reference_summary: str


@dataclass
class QueryResult:
    query_id: str
    retrieved_doc_ids: list[str]
    generated_summary: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    rouge_l_f1: float
    semantic_cosine: float


@dataclass
class EvalReport:
    llm_name: str
    top_k: int
    per_query: list[QueryResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.per_query)

    def aggregate(self) -> dict[str, float]:
        if not self.per_query:
            return {}
        return {
            f"recall@{self.top_k}": mean(r.recall_at_k for r in self.per_query),
            f"precision@{self.top_k}": mean(r.precision_at_k for r in self.per_query),
            "mrr": mean(r.reciprocal_rank for r in self.per_query),
            "rouge_l_f1": mean(r.rouge_l_f1 for r in self.per_query),
            "semantic_cosine": mean(r.semantic_cosine for r in self.per_query),
        }

    def to_dict(self) -> dict:
        return {
            "llm_name": self.llm_name,
            "top_k": self.top_k,
            "n_queries": self.n,
            "aggregate": self.aggregate(),
            "per_query": [
                {
                    "query_id": r.query_id,
                    "retrieved_doc_ids": r.retrieved_doc_ids,
                    "recall_at_k": r.recall_at_k,
                    "precision_at_k": r.precision_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    "rouge_l_f1": r.rouge_l_f1,
                    "semantic_cosine": r.semantic_cosine,
                }
                for r in self.per_query
            ],
        }

    def format_table(self) -> str:
        agg = self.aggregate()
        lines = [
            f"LLM: {self.llm_name}   |   queries: {self.n}   |   top_k: {self.top_k}",
            "-" * 60,
        ]
        for k, v in agg.items():
            lines.append(f"{k:>20}: {v:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics


_VERSION_RE = re.compile(r"v\d+$")


def _normalise_id(arxiv_id: str) -> str:
    """Strip arXiv version suffix. Treats '1706.03762v7' and '1706.03762' as the same paper."""
    return _VERSION_RE.sub("", arxiv_id)


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    gold_set = {_normalise_id(g) for g in gold}
    top = {_normalise_id(x) for x in retrieved[:k]}
    return len(top & gold_set) / len(gold_set)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    gold_set = {_normalise_id(g) for g in gold}
    return sum(1 for d in top if _normalise_id(d) in gold_set) / len(top)


def reciprocal_rank(retrieved: list[str], gold: list[str]) -> float:
    gold_set = {_normalise_id(g) for g in gold}
    for i, doc in enumerate(retrieved, start=1):
        if _normalise_id(doc) in gold_set:
            return 1.0 / i
    return 0.0


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, hypothesis)["rougeL"].fmeasure)


def semantic_cosine(hypothesis: str, reference: str, *, model: str | None = None) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    vecs = embed([hypothesis, reference], model=model) if model else embed([hypothesis, reference])
    # embeddings are already L2-normalised
    return float(vecs[0] @ vecs[1])


# ---------------------------------------------------------------------------
# Reference set loading


def load_reference_corpus(path: Path = REFERENCE_DIR / "benchmark.jsonl") -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Reference corpus stores abstracts only — assign to `text` so the
            # retriever indexes them.
            data.setdefault("text", data.get("abstract", ""))
            data.setdefault("authors", [])
            data.setdefault("categories", [])
            data.setdefault("pdf_url", "")
            data.setdefault("published", "")
            data.setdefault("metadata", {})
            docs.append(Document(**data))
    return docs


def load_reference_queries(path: Path = REFERENCE_DIR / "queries.jsonl") -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(EvalQuery(**data))
    return queries


# ---------------------------------------------------------------------------
# Harness


def evaluate(
    pipeline: Pipeline,
    queries: Iterable[EvalQuery],
    *,
    top_k: int = 5,
    summary_max_tokens: int = 400,
) -> EvalReport:
    """Run each query, score retrieval + generation, return a full report.

    Retrieval is scored on the top-k *unique documents* returned by the
    retriever (i.e. best-scoring chunk per doc), which is what the CV bullet
    claims. Summarisation is separately given the top-k *chunks* so it has
    fine-grained context to ground its answer.
    """
    report = EvalReport(llm_name=pipeline.llm.name, top_k=top_k)
    for q in queries:
        # Retrieval metric: unique docs
        ranked_docs = pipeline.retriever.query_documents(q.query, top_k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in ranked_docs]
        # Generation: over the top-k chunks (which the pipeline picks)
        summary = pipeline.ask(q.query, top_k=top_k, max_tokens=summary_max_tokens)

        report.per_query.append(
            QueryResult(
                query_id=q.id,
                retrieved_doc_ids=retrieved_ids,
                generated_summary=summary.text,
                recall_at_k=recall_at_k(retrieved_ids, q.gold_doc_ids, top_k),
                precision_at_k=precision_at_k(retrieved_ids, q.gold_doc_ids, top_k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, q.gold_doc_ids),
                rouge_l_f1=rouge_l_f1(summary.text, q.reference_summary),
                semantic_cosine=semantic_cosine(summary.text, q.reference_summary),
            )
        )
    return report


def run_default_evaluation(top_k: int = 5) -> EvalReport:
    """Convenience: build a Pipeline over the reference corpus, run all queries."""
    docs = load_reference_corpus()
    queries = load_reference_queries()
    pipeline = Pipeline(documents=docs)
    return evaluate(pipeline, queries, top_k=top_k)


# Note on `np`: kept in imports so downstream tools that import from this
# module can pick it up without re-importing; also used in tests.
_ = np
