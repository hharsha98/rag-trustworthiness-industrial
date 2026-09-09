"""Tests for experiments/08_beir_ablation.py's pure functions and the loader's
offline error path. Fast, no network, no real model downloads -- anything that
needs the actual BEIR download or real model weights is marked @pytest.mark.slow
(skipped by default; see pyproject.toml's `addopts = "-m \"not slow\""`).

`experiments/08_beir_ablation.py` is not an importable package module (its
filename starts with a digit), so it is loaded here via importlib -- the same
way any standalone script would be loaded for testing.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import pytest

from conftest import FakeEmbedder

_MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "08_beir_ablation.py"
_spec = importlib.util.spec_from_file_location("beir_ablation", _MODULE_PATH)
beir_ablation = importlib.util.module_from_spec(_spec)
sys.modules["beir_ablation"] = beir_ablation
_spec.loader.exec_module(beir_ablation)


# --------------------------------------------------------------------------- qrels_to_lookup


def test_qrels_to_lookup_groups_by_query_and_stringifies_ids():
    df = pd.DataFrame({
        "query-id": [1, 1, 3],
        "corpus-id": [100, 200, 300],
        "score": [1, 1, 1],
    })
    lookup = beir_ablation.qrels_to_lookup(df)
    assert lookup == {"1": ["100", "200"], "3": ["300"]}


def test_qrels_to_lookup_single_row_per_query():
    df = pd.DataFrame({"query-id": [5], "corpus-id": [50], "score": [1]})
    lookup = beir_ablation.qrels_to_lookup(df)
    assert lookup == {"5": ["50"]}


# --------------------------------------------------------------------------- binary nDCG
# (reused directly from ragtrust.metrics.relevance -- its {0,1} binary-gain
# semantics already match SciFact's judgments, so experiment 08 does not
# reimplement it; these tests exercise it through the loader module's import.)


def test_ndcg_at_k_hand_computed():
    # ranked = [a, b, c]; only "b" (0-indexed rank 1) is relevant.
    # DCG  = 1 / log2(1 + 2) = 1 / log2(3)
    # IDCG = 1 relevant doc, k=3 -> 1 / log2(0 + 2) = 1 / log2(2) = 1.0
    ranked = ["a", "b", "c"]
    relevant = ["b"]
    expected = (1.0 / math.log2(3)) / 1.0
    assert beir_ablation.ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = ["a", "b", "c"]
    relevant = ["a", "b"]
    assert beir_ablation.ndcg_at_k(ranked, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_no_relevant_docs_retrieved_is_zero():
    ranked = ["x", "y", "z"]
    relevant = ["not_present"]
    assert beir_ablation.ndcg_at_k(ranked, relevant, k=3) == 0.0


# --------------------------------------------------------------------------- Recall@k (reused)


def test_recall_at_k_partial_and_full():
    ranked = ["a", "b", "c", "d"]
    assert beir_ablation.recall_at_k(ranked, ["a", "z"], k=4) == pytest.approx(0.5)
    assert beir_ablation.recall_at_k(ranked, ["a", "b"], k=2) == pytest.approx(1.0)
    assert beir_ablation.recall_at_k(ranked, ["a"], k=0) == 0.0


# --------------------------------------------------------------------------- MRR@k (local wrapper)


def test_mrr_at_k_finds_first_relevant_within_k():
    ranked = ["x", "y", "z", "w"]
    assert beir_ablation.mrr_at_k(ranked, ["z"], k=4) == pytest.approx(1 / 3)


def test_mrr_at_k_zero_when_relevant_doc_outside_k():
    ranked = ["x", "y", "z", "w"]
    assert beir_ablation.mrr_at_k(ranked, ["w"], k=2) == 0.0


def test_mrr_at_k_zero_when_no_relevant_doc_present():
    ranked = ["x", "y"]
    assert beir_ablation.mrr_at_k(ranked, ["nonexistent"], k=2) == 0.0


# --------------------------------------------------------------------------- misc pure helpers


def test_candidate_corpus_ratio():
    assert beir_ablation.candidate_corpus_ratio(20, 5183) == pytest.approx(20 / 5183)


def test_is_mrr_saturated_true_when_all_near_one():
    assert beir_ablation.is_mrr_saturated([1.0, 0.99, 0.985])


def test_is_mrr_saturated_false_when_one_config_far_from_one():
    assert not beir_ablation.is_mrr_saturated([1.0, 0.5, 0.99])


def test_is_mrr_saturated_false_on_empty_input():
    assert not beir_ablation.is_mrr_saturated([])


def test_embedding_cache_path_sanitizes_slashes():
    path = beir_ablation.embedding_cache_path(
        Path("/tmp/cache"), "sentence-transformers/msmarco-distilbert-base-v4"
    )
    assert path.name == "scifact_corpus_embeddings__sentence-transformers__msmarco-distilbert-base-v4.npy"
    assert "/" not in path.name


# --------------------------------------------------------------------------- CachedRetriever
# (fake embedder, real faiss -- fast, no network, no real model weights)


def test_cached_retriever_writes_then_reuses_cache(tmp_path):
    cache_path = tmp_path / "cache.npy"
    passages = ["alpha beta", "gamma delta", "epsilon zeta"]

    embedder = FakeEmbedder(dim=8)
    encode_calls = []
    original_encode = embedder.encode

    def counting_encode(texts, **kwargs):
        encode_calls.append(texts)
        return original_encode(texts, **kwargs)

    embedder.encode = counting_encode

    r1 = beir_ablation.CachedRetriever(embedder, cache_path, normalize=True)
    r1.build(passages)
    assert cache_path.exists()
    assert not r1.cache_hit
    assert len(encode_calls) == 1  # corpus encoded exactly once

    r2 = beir_ablation.CachedRetriever(embedder, cache_path, normalize=True)
    r2.build(passages)
    assert r2.cache_hit
    assert len(encode_calls) == 1  # no second corpus encode -- cache was reused

    results = r2.search("alpha", k=1)
    assert len(results) == 1
    assert len(encode_calls) == 2  # only the query got encoded just now


def test_cached_retriever_recomputes_on_corpus_size_mismatch(tmp_path):
    cache_path = tmp_path / "cache.npy"
    embedder = FakeEmbedder(dim=8)

    beir_ablation.CachedRetriever(embedder, cache_path, normalize=True).build(["a", "b"])
    r2 = beir_ablation.CachedRetriever(embedder, cache_path, normalize=True)
    r2.build(["a", "b", "c"])  # different corpus size -- must not silently reuse stale cache
    assert not r2.cache_hit
    results = r2.search("a", k=3)
    assert len(results) == 3


# --------------------------------------------------------------------------- load_scifact
# offline error path. local_files_only=True makes huggingface_hub short-circuit
# before any network request, so this is genuinely no-network, not just fast.


def test_load_scifact_raises_clear_error_when_cache_absent_and_downloads_disabled(
    tmp_path, monkeypatch
):
    # Point the Hugging Face cache at an empty temp directory (nothing cached
    # there) and disable downloads via local_files_only -- the loader must
    # fail with a clear, readable RuntimeError, not a raw
    # huggingface_hub/requests traceback.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))

    with pytest.raises(RuntimeError, match="not in the local Hugging Face cache"):
        beir_ablation.load_scifact(local_files_only=True)


# --------------------------------------------------------------------------- slow / network integration check


@pytest.mark.slow
def test_load_scifact_real_download_matches_documented_shape():
    corpus_df, queries_df, qrels_df = beir_ablation.load_scifact()
    assert len(corpus_df) == 5183
    lookup = beir_ablation.qrels_to_lookup(qrels_df)
    assert len(lookup) == 300
