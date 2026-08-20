from arag.llm import ExtractiveLLM
from arag.pipeline import Pipeline


def test_pipeline_ask_returns_grounded_summary(toy_docs):
    pipe = Pipeline(documents=toy_docs, llm=ExtractiveLLM(max_sentences=3))
    summary = pipe.ask("attention without recurrence", top_k=2)
    assert summary.text.strip()
    assert summary.used_chunks
    assert summary.llm_name.startswith("extractive:")


def test_pipeline_retrieve_documents_returns_ranked_docs(toy_docs):
    pipe = Pipeline(documents=toy_docs, llm=ExtractiveLLM())
    ranked = pipe.retrieve_documents("low-rank fine tuning", top_k=2)
    assert ranked[0][0].arxiv_id == "0002"
    # scores strictly decreasing
    for a, b in zip(ranked, ranked[1:]):
        assert a[1] >= b[1]


def test_pipeline_ask_empty_corpus():
    pipe = Pipeline(documents=[], llm=ExtractiveLLM())
    summary = pipe.ask("anything", top_k=5)
    assert "No relevant excerpts" in summary.text
    assert summary.used_chunks == []
