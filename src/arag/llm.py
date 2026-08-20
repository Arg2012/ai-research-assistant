"""LLM abstraction.

Real backend: Anthropic Claude via the `anthropic` SDK. Requires ANTHROPIC_API_KEY.
Fallback backend: a deterministic extractive summariser that concatenates leading
sentences. Lets the ingestion / retrieval / evaluation pipeline run — and be
tested — without any API key.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str: ...


@dataclass
class AnthropicLLM:
    """Thin wrapper around the Anthropic Messages API."""

    model: str = "claude-sonnet-4-6"
    api_key: str | None = None

    def __post_init__(self) -> None:
        import anthropic

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Either export it or use ExtractiveLLM()."
            )
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        return "".join(parts).strip()


@dataclass
class ExtractiveLLM:
    """Deterministic no-API fallback.

    Extractive summarisation is a legitimate baseline in the literature — it's what
    the evaluation harness compares LLM output against and is safe to use when no
    API key is present. Not a mock: real output, just not neural.
    """

    max_sentences: int = 5

    @property
    def name(self) -> str:
        return f"extractive:top-{self.max_sentences}"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        # Prompts in this project embed the paper text after a marker like
        # "PAPER:" or "CONTENT:". Extract the last block and take leading sentences.
        text = prompt
        for marker in ("CONTENT:", "PAPER:", "ABSTRACT:", "TEXT:"):
            if marker in text:
                text = text.rsplit(marker, 1)[1]
        text = text.strip()
        sentences = _split_sentences(text)
        return " ".join(sentences[: self.max_sentences]).strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def default_llm() -> LLM:
    """Return AnthropicLLM if a key is configured, else ExtractiveLLM."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        model = os.environ.get("ARAG_MODEL", "claude-sonnet-4-6")
        return AnthropicLLM(model=model)
    return ExtractiveLLM()
