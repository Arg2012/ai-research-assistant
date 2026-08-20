from arag.evaluation import (
    EvalQuery,
    evaluate,
    load_reference_corpus,
    load_reference_queries,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l_f1,
    run_default_evaluation,
    semantic_cosine,
)
from arag.llm import ExtractiveLLM
from arag.pipeline import Pipeline


def test_recall_precision_mrr_math():
    retrieved = ["a", "b", "c", "d", "e"]
    gold = ["b", "d"]
    assert recall_at_k(retrieved, gold, 5) == 1.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert precision_at_k(retrieved, gold, 5) == 2 / 5
    assert precision_at_k(retrieved, gold, 2) == 0.5
    # First gold appears at rank 2 -> 1/2
    assert reciprocal_rank(retrieved, gold) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_rouge_and_cosine_reward_similar_text():
    ref = "The Transformer uses multi-head self-attention and has no recurrence."
    close = "The Transformer relies on multi-head self-attention with no recurrence."
    far = "Convolutional networks classify images from pixel arrays."
    assert rouge_l_f1(close, ref) > rouge_l_f1(far, ref)
    assert semantic_cosine(close, ref) > semantic_cosine(far, ref)


def test_reference_corpus_loads_all_expected_papers():
    docs = load_reference_corpus()
    ids = {d.arxiv_id for d in docs}
    for base in [
        "1706.03762",
        "1810.04805",
        "2005.14165",
        "2103.00020",
        "2106.09685",
        "2201.11903",
        "2203.02155",
        "2204.02311",
        "2212.04356",
        "2302.13971",
    ]:
        assert any(i.startswith(base) for i in ids), f"missing {base}"


def test_reference_queries_have_gold_ids_in_corpus():
    docs = load_reference_corpus()
    corpus_ids = {d.arxiv_id for d in docs}
    for q in load_reference_queries():
        assert any(g in corpus_ids for g in q.gold_doc_ids), (
            f"{q.id}: none of {q.gold_doc_ids} in reference corpus"
        )


def test_evaluate_end_to_end_produces_all_metrics():
    docs = load_reference_corpus()
    queries = load_reference_queries()[:3]  # small slice for CI speed
    pipe = Pipeline(documents=docs, llm=ExtractiveLLM())
    report = evaluate(pipe, queries, top_k=3)
    agg = report.aggregate()
    for key in ("recall@3", "precision@3", "mrr", "rouge_l_f1", "semantic_cosine"):
        assert key in agg
    # Retrieval should be non-trivial on this well-known corpus
    assert agg["recall@3"] > 0.5


def test_run_default_evaluation_smoke():
    report = run_default_evaluation(top_k=3)
    assert report.n == 10
