# arag — LLM Research Agent for Scientific Literature

A minimal, honest research assistant that turns a query into a grounded summary
of relevant arXiv papers, and — critically — comes with an automated evaluation
harness that scores the pipeline against a reference dataset.

## Pipeline

```
arxiv ─▶ PDF extract ─▶ chunk ─▶ embed ─▶ retrieve top-k ─▶ LLM summarise
                                    │
                                    ▼
                              evaluation harness
                        (retrieval + generation metrics)
```

See [`src/arag/`](src/arag/) for the modules.

## Install

```bash
uv sync                 # or: pip install -e .
cp .env.example .env    # set ANTHROPIC_API_KEY (optional — see below)
```

Only `ANTHROPIC_API_KEY` is optional. If unset, the summarisation step falls
back to a deterministic extractive baseline so ingestion, retrieval, and the
full evaluation harness continue to run.

## Quickstart

```bash
# Ask a question against a fresh arXiv fetch:
arag ask "diffusion models for molecular design" --query "cat:cs.LG" --max 10

# Or run the evaluation harness (uses the reference corpus in data/reference/):
arag eval --top-k 5
```

## Evaluation

The evaluation harness (`arag/evaluation.py`) benchmarks generated responses
against reference material:

* **Retrieval** — recall@k, precision@k, MRR against hand-annotated gold
  document ids.
* **Generation** — ROUGE-L F1 (lexical overlap) and sentence-embedding cosine
  (semantic overlap) against hand-written reference summaries.

The reference corpus (`data/reference/benchmark.jsonl`) is built by fetching
real abstracts from arXiv:

```bash
python scripts/build_reference_corpus.py
```

Queries and gold summaries live in `data/reference/queries.jsonl`.

## Tests

```bash
uv run pytest -q
```

Tests do not require network access or API keys — arXiv responses are stubbed
and the extractive LLM backend is used.

## Design notes

* **No fabricated fallbacks.** If retrieval fails or the LLM can't answer,
  we say so — never invent citations or paper content.
* **Local embeddings.** Sentence-transformers runs on CPU; no external
  embedding API is required.
* **Anthropic Claude for generation.** Configurable via `ARAG_MODEL`.
* **Small, checked-in reference set.** Ten landmark ML papers with hand-written
  reference summaries; enough for the eval harness to produce meaningful,
  reproducible numbers.
