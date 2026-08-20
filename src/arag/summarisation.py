"""LLM-driven summarisation over retrieved context."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM, default_llm
from .retrieval import RetrievedChunk


SYSTEM_PROMPT = """You are a research analyst summarising scientific literature.
Be terse, technical, and grounded in the provided excerpts. Do not invent
citations, author names, institutions, or numeric results — if the excerpts do
not support a claim, say so.

Format the response as:
- 1 paragraph high-level summary
- 3-5 bullet points of specific technical findings

Cite each bullet with the arXiv id it draws from, e.g. [2401.12345]."""


@dataclass
class Summary:
    query: str
    text: str
    used_chunks: list[RetrievedChunk]
    llm_name: str


def summarise(
    query: str,
    retrieved: list[RetrievedChunk],
    *,
    llm: LLM | None = None,
    max_tokens: int = 800,
) -> Summary:
    """Summarise `retrieved` chunks in the context of `query`."""
    llm = llm or default_llm()

    if not retrieved:
        return Summary(
            query=query,
            text="No relevant excerpts retrieved.",
            used_chunks=[],
            llm_name=llm.name,
        )

    excerpts_block = "\n\n".join(
        f"[{c.chunk.doc_id}] (score={c.score:.3f})\n{c.chunk.text}" for c in retrieved
    )
    prompt = f"""{SYSTEM_PROMPT}

QUERY:
{query}

CONTENT:
{excerpts_block}
"""
    text = llm.complete(prompt, max_tokens=max_tokens)
    return Summary(query=query, text=text, used_chunks=retrieved, llm_name=llm.name)
