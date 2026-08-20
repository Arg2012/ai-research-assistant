"""arag CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .evaluation import (
    evaluate,
    load_reference_corpus,
    load_reference_queries,
)
from .pipeline import Pipeline


def _cmd_ask(args: argparse.Namespace) -> int:
    load_dotenv()
    cache = Path(args.cache_dir) if args.cache_dir else None
    pipe = Pipeline.from_arxiv_query(
        args.query,
        max_results=args.max,
        download_pdfs=args.download_pdfs,
        cache_dir=cache,
    )
    if not pipe.documents:
        print(f"No papers returned for query: {args.query!r}", file=sys.stderr)
        return 1
    print(f"Indexed {len(pipe.documents)} papers "
          f"({sum(1 for c in pipe.iter_chunks())} chunks).")
    summary = pipe.ask(args.question, top_k=args.top_k, max_tokens=args.max_tokens)
    print(f"\n=== Summary (LLM: {summary.llm_name}) ===\n{summary.text}\n")
    print("=== Sources ===")
    for rc in summary.used_chunks:
        print(f"  [{rc.chunk.doc_id}]  score={rc.score:.3f}  chunk={rc.chunk.chunk_id}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    load_dotenv()
    docs = load_reference_corpus(Path(args.corpus)) if args.corpus else load_reference_corpus()
    queries = (
        load_reference_queries(Path(args.queries)) if args.queries else load_reference_queries()
    )
    pipe = Pipeline(documents=docs)
    report = evaluate(pipe, queries, top_k=args.top_k, summary_max_tokens=args.max_tokens)
    print(report.format_table())
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nWrote full report to {out}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    load_dotenv()
    cache = Path(args.cache_dir) if args.cache_dir else None
    pipe = Pipeline.from_arxiv_query(
        args.query,
        max_results=args.max,
        download_pdfs=args.download_pdfs,
        cache_dir=cache,
    )
    out = Path(args.output)
    pipe.save_corpus(out)
    print(f"Wrote {len(pipe.documents)} documents to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arag", description="LLM research agent for scientific literature")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Fetch arXiv papers then answer a research question")
    ask.add_argument("question", help="Natural-language research question")
    ask.add_argument("--query", required=True, help="arXiv search query, e.g. 'cat:cs.CL'")
    ask.add_argument("--max", type=int, default=10, help="Max papers to fetch")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--max-tokens", type=int, default=800)
    ask.add_argument("--cache-dir", default="runtime/cache/pdfs")
    ask.add_argument("--no-pdfs", dest="download_pdfs", action="store_false",
                     help="Skip PDF download; index abstracts only")
    ask.set_defaults(download_pdfs=True, func=_cmd_ask)

    ev = sub.add_parser("eval", help="Run the automated evaluation harness")
    ev.add_argument("--top-k", type=int, default=5)
    ev.add_argument("--max-tokens", type=int, default=400)
    ev.add_argument("--corpus", help="Override path to benchmark.jsonl")
    ev.add_argument("--queries", help="Override path to queries.jsonl")
    ev.add_argument("--output", help="Write full JSON report to this path")
    ev.set_defaults(func=_cmd_eval)

    ing = sub.add_parser("ingest", help="Fetch + PDF-extract an arXiv query into a JSONL corpus")
    ing.add_argument("--query", required=True)
    ing.add_argument("--max", type=int, default=20)
    ing.add_argument("--output", required=True, help="Destination JSONL path")
    ing.add_argument("--cache-dir", default="runtime/cache/pdfs")
    ing.add_argument("--no-pdfs", dest="download_pdfs", action="store_false")
    ing.set_defaults(download_pdfs=True, func=_cmd_ingest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
