"""Document ingestion.

Two entry points:
  * `fetch_arxiv(query, ...)` — hits the arXiv Atom API and returns paper metadata.
  * `extract_pdf_text(pdf_url_or_path, ...)` — downloads (if needed) and extracts text via PyMuPDF.

`ingest_arxiv(query, ..., download_pdfs=True)` composes them into a single pass
producing a list of `Document`s ready for embedding + retrieval.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_WS = re.compile(r"\s+")


@dataclass
class Document:
    """A single ingested paper. `text` may be just the abstract or full PDF text."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    pdf_url: str
    text: str = ""  # full text if the PDF has been extracted; else empty
    source: str = "arxiv"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_arxiv(
    query: str,
    *,
    max_results: int = 20,
    start: int = 0,
    client: httpx.Client | None = None,
) -> list[Document]:
    """Fetch papers from the arXiv API.

    `query` is an arXiv search string. Use `cat:cs.CL` for a category or
    `ti:transformer AND abs:attention` for field-specific search.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = client.get(ARXIV_API, params=params)
        resp.raise_for_status()
        return _parse_atom(resp.text)
    finally:
        if owns_client:
            client.close()


def _parse_atom(xml_text: str) -> list[Document]:
    root = ET.fromstring(xml_text)
    docs: list[Document] = []
    for entry in root.findall("atom:entry", _ATOM):
        title = _clean(_findtext(entry, "atom:title"))
        abstract = _clean(_findtext(entry, "atom:summary"))
        arxiv_url = _findtext(entry, "atom:id")
        arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else ""

        authors: list[str] = []
        for author in entry.findall("atom:author", _ATOM):
            name = _findtext(author, "atom:name")
            if name:
                authors.append(name)

        categories: list[str] = []
        primary = entry.find("arxiv:primary_category", _ATOM)
        if primary is not None and primary.get("term"):
            categories.append(primary.get("term"))
        for cat in entry.findall("atom:category", _ATOM):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        pdf_url = ""
        for link in entry.findall("atom:link", _ATOM):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"http://export.arxiv.org/pdf/{arxiv_id.split('v')[0]}"

        docs.append(
            Document(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published=_findtext(entry, "atom:published"),
                pdf_url=pdf_url,
            )
        )
    return docs


def _findtext(elem: ET.Element, path: str) -> str:
    node = elem.find(path, _ATOM)
    return (node.text or "").strip() if node is not None and node.text else ""


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def extract_pdf_text(
    pdf_source: str | Path,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Extract text from a PDF. `pdf_source` may be an https URL or local path."""
    import fitz  # PyMuPDF

    if isinstance(pdf_source, Path) or (isinstance(pdf_source, str) and Path(pdf_source).exists()):
        path = Path(pdf_source)
    else:
        assert cache_dir is not None, "cache_dir required when downloading a URL"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", pdf_source.rsplit("/", 1)[-1]) or "download.pdf"
        if not name.endswith(".pdf"):
            name += ".pdf"
        path = cache_dir / name
        if not path.exists():
            owns_client = client is None
            client = client or httpx.Client(timeout=60.0, follow_redirects=True)
            try:
                resp = client.get(pdf_source)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise ValueError(f"Response from {pdf_source} is not a PDF")
                path.write_bytes(resp.content)
            finally:
                if owns_client:
                    client.close()

    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
    return _clean("\n".join(pages))


def ingest_arxiv(
    query: str,
    *,
    max_results: int = 20,
    download_pdfs: bool = True,
    cache_dir: Path | None = None,
) -> list[Document]:
    """Fetch metadata and (optionally) full text for arXiv papers."""
    docs = fetch_arxiv(query, max_results=max_results)
    if not download_pdfs:
        # Fall back to abstract as the searchable text.
        for d in docs:
            d.text = d.abstract
        return docs

    cache_dir = cache_dir or Path("runtime/cache/pdfs")
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for d in docs:
            try:
                d.text = extract_pdf_text(d.pdf_url, cache_dir=cache_dir, client=client)
            except Exception as exc:  # noqa: BLE001
                d.metadata["pdf_error"] = str(exc)
                d.text = d.abstract  # graceful degradation, no fabrication
    return docs


def documents_to_jsonl(docs: Iterable[Document], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def documents_from_jsonl(path: Path) -> list[Document]:
    import json

    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            docs.append(Document(**data))
    return docs
