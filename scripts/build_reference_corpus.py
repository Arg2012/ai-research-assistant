#!/usr/bin/env python3
"""Build data/reference/benchmark.jsonl by fetching real abstracts from arXiv.

The reference corpus is a fixed set of well-known ML papers used by the
evaluation harness. We fetch abstracts from the arXiv API at build time so the
committed reference file matches the real record — never hand-transcribed
(which risks fabrication).

Idempotent: re-running produces byte-identical output for a given ID list.
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx


PAPERS: list[str] = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3 (Language Models are Few-Shot Learners)
    "2103.00020",  # CLIP (Learning Transferable Visual Models From Natural Language Supervision)
    "2106.09685",  # LoRA (Low-Rank Adaptation of Large Language Models)
    "2201.11903",  # Chain-of-Thought Prompting
    "2203.02155",  # InstructGPT (Training language models to follow instructions with human feedback)
    "2204.02311",  # PaLM
    "2212.04356",  # Whisper
    "2302.13971",  # LLaMA
]

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OUT = Path(__file__).resolve().parents[1] / "data" / "reference" / "benchmark.jsonl"


def fetch(arxiv_id: str, client: httpx.Client) -> dict:
    resp = client.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", NS)
    if entry is None:
        raise RuntimeError(f"No entry returned for {arxiv_id}")

    def text_of(path: str) -> str:
        node = entry.find(path, NS)
        return " ".join((node.text or "").split()) if node is not None else ""

    authors: list[str] = []
    for author in entry.findall("atom:author", NS):
        n = author.find("atom:name", NS)
        if n is not None and n.text:
            authors.append(n.text.strip())

    categories: list[str] = []
    primary = entry.find("arxiv:primary_category", NS)
    if primary is not None and primary.get("term"):
        categories.append(primary.get("term"))
    for cat in entry.findall("atom:category", NS):
        term = cat.get("term")
        if term and term not in categories:
            categories.append(term)

    id_url = text_of("atom:id")
    returned_id = id_url.rsplit("/", 1)[-1] if id_url else arxiv_id

    pdf_url = ""
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break

    return {
        "arxiv_id": returned_id,
        "title": text_of("atom:title"),
        "authors": authors,
        "abstract": text_of("atom:summary"),
        "categories": categories,
        "published": text_of("atom:published"),
        "pdf_url": pdf_url,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for i, arxiv_id in enumerate(PAPERS):
            print(f"[{i+1}/{len(PAPERS)}] {arxiv_id}")
            records.append(fetch(arxiv_id, client))
            time.sleep(3)  # arXiv rate-limit courtesy

    with OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} papers to {OUT.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
