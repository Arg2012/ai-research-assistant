import numpy as np

from arag.embeddings import chunk_text, embed


def test_chunk_text_short_returns_single_chunk():
    assert chunk_text("a b c d", max_words=100) == ["a b c d"]


def test_chunk_text_overlap_windows():
    words = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(words, max_words=20, overlap=5)
    # each chunk should have <= max_words words
    for c in chunks:
        assert len(c.split()) <= 20
    # union of chunks should contain every original word
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(words.split())


def test_embed_normalises_and_shape():
    vecs = embed(["hello world", "goodbye world"])
    assert vecs.shape[0] == 2
    norms = np.linalg.norm(vecs, axis=1)
    # sentence-transformers with normalize_embeddings=True -> unit vectors
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_empty_returns_zeros():
    vecs = embed([])
    assert vecs.shape[0] == 0
