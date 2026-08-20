from arag.embeddings import chunk_documents
from arag.retrieval import Retriever


def test_retrieval_ranks_semantically_relevant_doc_first(toy_docs):
    chunks = chunk_documents(toy_docs, max_words=200)
    r = Retriever(chunks)
    hits = r.query("removing recurrence from sequence models", top_k=3)
    assert hits, "retriever returned nothing"
    assert hits[0].chunk.doc_id == "0001"


def test_retrieval_distinguishes_topics(toy_docs):
    r = Retriever(chunk_documents(toy_docs))
    top_docs = r.query_documents("parameter-efficient adaptation of a pretrained model", top_k=1)
    assert top_docs[0][0] == "0002"


def test_retrieval_empty_corpus_returns_empty():
    r = Retriever([])
    assert r.query("anything", top_k=5) == []
