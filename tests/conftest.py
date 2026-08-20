import pytest

from arag.ingestion import Document


@pytest.fixture
def toy_docs() -> list[Document]:
    return [
        Document(
            arxiv_id="0001",
            title="Attention with Recurrence Removed",
            authors=["A. Author"],
            abstract=(
                "We introduce a sequence model based only on self-attention, "
                "eliminating recurrence. Trains faster and matches quality."
            ),
            categories=["cs.CL"],
            published="2020-01-01",
            pdf_url="",
            text=(
                "We introduce a sequence model based only on self-attention, "
                "eliminating recurrence. Trains faster and matches quality on translation."
            ),
        ),
        Document(
            arxiv_id="0002",
            title="Low-rank fine-tuning of transformers",
            authors=["B. Author"],
            abstract=(
                "We add low-rank adapters to frozen transformer weights, "
                "cutting trainable parameters by 10x."
            ),
            categories=["cs.LG"],
            published="2020-02-01",
            pdf_url="",
            text=(
                "We add low-rank adapters to frozen transformer weights, "
                "cutting trainable parameters by 10x while retaining quality."
            ),
        ),
        Document(
            arxiv_id="0003",
            title="Diffusion models for images",
            authors=["C. Author"],
            abstract="A denoising diffusion probabilistic model for image generation.",
            categories=["cs.CV"],
            published="2020-03-01",
            pdf_url="",
            text="A denoising diffusion probabilistic model for image generation.",
        ),
    ]


ARXIV_XML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.56789v1</id>
    <title>Testing the arXiv parser</title>
    <summary>A short abstract for parser testing.</summary>
    <author><name>Test Author</name></author>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <link title="pdf" type="application/pdf" href="http://arxiv.org/pdf/1234.56789v1"/>
    <published>2024-01-01T00:00:00Z</published>
  </entry>
</feed>
"""


@pytest.fixture
def arxiv_xml() -> str:
    return ARXIV_XML_FIXTURE
