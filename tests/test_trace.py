"""Tests for ragtrust.trace: the in-process request-trace ring buffer used by
GET /traces (tests/test_service.py covers the route itself; this file covers
the buffer/stats logic in isolation). No model downloads, no network.
"""
import json
import threading

import pytest

from ragtrust.trace import StageTiming, Trace, TraceBuffer, hash_question


def _trace(total_ms=10.0, abstained=False, trust_geometric=0.5, corpus_id=None) -> Trace:
    return Trace(
        question_hash=hash_question("irrelevant for these tests"),
        corpus_id=corpus_id,
        abstained=abstained,
        trust_geometric=trust_geometric,
        stages=[StageTiming(stage="retrieve", ms=1.0), StageTiming(stage="gate", ms=2.0)],
        total_ms=total_ms,
    )


# --------------------------------------------------------------------- hash_question


def test_hash_question_is_12_hex_chars():
    h = hash_question("What is the capital of France?")
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_question_is_deterministic_and_distinguishes_input():
    assert hash_question("same question") == hash_question("same question")
    assert hash_question("question A") != hash_question("question B")


# --------------------------------------------------------------------- capacity / eviction


def test_buffer_never_exceeds_capacity():
    buf = TraceBuffer(capacity=3)
    for i in range(10):
        buf.add(_trace(total_ms=float(i)))
    assert buf.stats()["count"] == 3


def test_buffer_evicts_oldest_first():
    buf = TraceBuffer(capacity=2)
    buf.add(_trace(total_ms=1.0))
    buf.add(_trace(total_ms=2.0))
    buf.add(_trace(total_ms=3.0))  # evicts the total_ms=1.0 entry
    remaining = sorted(t.total_ms for t in buf.recent(limit=10))
    assert remaining == [2.0, 3.0]


def test_recent_returns_most_recent_first_and_respects_limit():
    buf = TraceBuffer(capacity=10)
    for i in range(5):
        buf.add(_trace(total_ms=float(i)))
    recent = buf.recent(limit=2)
    assert [t.total_ms for t in recent] == [4.0, 3.0]


# --------------------------------------------------------------------- thread safety


def test_add_is_thread_safe_under_concurrent_writers():
    # uvicorn runs sync route handlers (like /answer) in a threadpool, so
    # TraceBuffer.add() is called from many threads concurrently in production.
    # This drives that same access pattern directly: 20 threads x 50 adds each,
    # well past the buffer's capacity, and asserts the buffer ends up in a
    # consistent state -- exactly `capacity` entries, no crash, no corrupted
    # deque -- rather than trying to assert which specific traces survived
    # (that's a race by design once eviction is involved).
    capacity = 25
    buf = TraceBuffer(capacity=capacity)
    n_threads = 20
    per_thread = 50

    def worker(worker_id: int):
        for i in range(per_thread):
            buf.add(_trace(total_ms=float(worker_id * per_thread + i)))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = buf.stats()
    assert stats["count"] == capacity
    assert len(buf.recent(limit=1000)) == capacity


# --------------------------------------------------------------------- stats()


def test_stats_on_empty_buffer_returns_none_not_zero():
    buf = TraceBuffer(capacity=10)
    stats = buf.stats()
    assert stats == {
        "count": 0, "abstention_rate": None, "p50_ms": None, "p95_ms": None, "mean_trust": None,
    }


def test_stats_with_a_single_entry_does_not_raise():
    # statistics.quantiles raises StatisticsError below n=2 -- this is exactly
    # the case that must not propagate out of stats().
    buf = TraceBuffer(capacity=10)
    buf.add(_trace(total_ms=42.0, abstained=False, trust_geometric=0.75))
    stats = buf.stats()
    assert stats["count"] == 1
    assert stats["abstention_rate"] == 0.0
    assert stats["p50_ms"] == 42.0
    assert stats["p95_ms"] == 42.0
    assert stats["mean_trust"] == pytest.approx(0.75)


def test_stats_abstention_rate_and_mean_trust_over_many_entries():
    buf = TraceBuffer(capacity=10)
    buf.add(_trace(total_ms=10.0, abstained=False, trust_geometric=0.8))
    buf.add(_trace(total_ms=20.0, abstained=False, trust_geometric=0.4))
    buf.add(_trace(total_ms=30.0, abstained=True, trust_geometric=None))
    buf.add(_trace(total_ms=40.0, abstained=True, trust_geometric=None))
    stats = buf.stats()
    assert stats["count"] == 4
    assert stats["abstention_rate"] == pytest.approx(0.5)
    # mean_trust averages only the two answered entries -- an abstention's
    # trust is undefined (see TraceBuffer.stats's docstring/comment), not 0.0,
    # so it must not drag the mean down.
    assert stats["mean_trust"] == pytest.approx((0.8 + 0.4) / 2)


def test_stats_mean_trust_is_none_when_every_entry_abstained():
    buf = TraceBuffer(capacity=10)
    buf.add(_trace(abstained=True, trust_geometric=None))
    buf.add(_trace(abstained=True, trust_geometric=None))
    assert buf.stats()["mean_trust"] is None


def test_stats_p50_p95_over_many_entries():
    buf = TraceBuffer(capacity=100)
    for ms in range(1, 101):  # 1.0 .. 100.0
        buf.add(_trace(total_ms=float(ms)))
    stats = buf.stats()
    # Nearest-rank percentile over 100 sorted values 1..100: p50 -> index 49
    # (value 50), p95 -> index 94 (value 95).
    assert stats["p50_ms"] == 50.0
    assert stats["p95_ms"] == 95.0


# --------------------------------------------------------------------- privacy: no raw question text


def test_to_dict_never_contains_the_raw_question_text():
    secret_question = "Does the Q3 acquisition NDA cover subcontractor Example Corp?"
    trace = Trace(
        question_hash=hash_question(secret_question),
        corpus_id="some-corpus-id",
        abstained=False,
        trust_geometric=0.9,
        stages=[StageTiming(stage="retrieve", ms=1.5)],
        total_ms=12.3,
    )
    payload = trace.to_dict()
    serialized = json.dumps(payload)
    assert "Example Corp" not in serialized
    assert "NDA" not in serialized
    assert "subcontractor" not in serialized
    assert secret_question not in serialized
    # The hash is present (that's the point -- it's what lets repeated
    # questions be correlated) but it is not, and cannot be, the question.
    assert payload["question_hash"] == hash_question(secret_question)


def test_stage_timings_round_trip_in_to_dict():
    trace = _trace()
    payload = trace.to_dict()
    assert payload["stages"] == [
        {"stage": "retrieve", "ms": 1.0}, {"stage": "gate", "ms": 2.0},
    ]
