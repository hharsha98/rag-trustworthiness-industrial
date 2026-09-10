"""In-process request tracing for the /answer path.

A small ring buffer of recent requests, kept in memory (never persisted) so an
operator can watch `/traces` and see whether the service is abstaining more
than usual or getting slower, without wiring up an external observability
stack for a project this size. Deliberately NOT a general logging/metrics
system: it exists to make `pipeline.py`'s per-stage timings (see
`AnswerResult.stage_timings`) visible at runtime, matching the stage names the
dashboard already animates in `dashboard/index.html`'s `.steps` row (retrieve,
gate, generate, decompose, entail, score).
"""
from __future__ import annotations

import hashlib
import math
import threading
from collections import deque
from dataclasses import dataclass


def hash_question(question: str) -> str:
    """First 12 hex chars of the question's sha256.

    Never store the question itself -- see `Trace.question_hash`'s docstring
    for why. 12 hex chars (48 bits) is far more than enough to notice "this
    is the same question as three requests ago" in a buffer capped at a few
    hundred entries; it is not, and is not meant to be, collision-resistant
    against a determined adversary trying to invert it.
    """
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class StageTiming:
    stage: str  # retrieve | gate | generate | decompose | entail | score
    ms: float


@dataclass(frozen=True)
class Trace:
    """One recorded `/answer` request.

    `question_hash` is a sha256 prefix, NEVER the question text -- a document
    a user uploaded may be private (a contract, a medical record, an
    unreleased design doc), and a question about it can leak exactly what is
    in it ("does the NDA cover subcontractors?" already says a lot on its
    own). A trace buffer is precisely the kind of debugging convenience that
    quietly becomes a disclosure channel if it holds the question verbatim --
    an operator with `/traces` access would otherwise be able to read every
    question ever asked of every uploaded corpus. The hash still lets an
    operator notice "this exact question came through 6 times in the last
    hour" (repeated hashes correlate), which is everything the buffer is for;
    it just can't reconstruct what the question said.
    """

    question_hash: str
    corpus_id: str | None
    abstained: bool
    trust_geometric: float | None
    stages: list  # list[StageTiming]
    total_ms: float

    def to_dict(self) -> dict:
        return {
            "question_hash": self.question_hash,
            "corpus_id": self.corpus_id,
            "abstained": self.abstained,
            "trust_geometric": self.trust_geometric,
            "stages": [{"stage": s.stage, "ms": round(s.ms, 3)} for s in self.stages],
            "total_ms": round(self.total_ms, 3),
        }


def _percentile(sorted_values: list, pct: float):
    """Nearest-rank percentile over an already-sorted ascending list.

    `pct` in [0, 1]. Uses an explicit sorted index (`ceil(pct * n) - 1`,
    clamped into range) rather than `statistics.quantiles`, which raises
    `StatisticsError` for fewer than two data points -- exactly the case
    `TraceBuffer.stats()` must handle cleanly, since the very first `/answer`
    request leaves the buffer with only one entry. Nearest-rank is a coarser
    interpolation than a linear percentile method, but for a p50/p95 latency
    readout the difference is invisible past a handful of samples, and the
    buffer is capped at `capacity` (default 100) regardless.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    idx = max(0, min(n - 1, math.ceil(pct * n) - 1))
    return sorted_values[idx]


class TraceBuffer:
    """Fixed-capacity ring buffer of recent `Trace`s.

    Backed by `collections.deque(maxlen=capacity)`, so once full, adding a
    new trace silently evicts the oldest one -- no unbounded growth, no
    eviction policy to get wrong. Guarded by a `threading.Lock` because
    uvicorn runs synchronous route handlers (like `/answer`) in a threadpool,
    so concurrent requests call `add()` from different threads at once; a
    plain deque's `append` is not documented as thread-safe against
    concurrent `list(...)` reads from `stats()`/`recent()`.
    """

    def __init__(self, capacity: int = 100):
        self._buf: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, trace: Trace) -> None:
        with self._lock:
            self._buf.append(trace)

    def recent(self, limit: int = 20) -> list:
        """Most-recently-added traces first, capped at `limit`."""
        with self._lock:
            items = list(self._buf)
        items.reverse()
        return items[:limit]

    def stats(self) -> dict:
        with self._lock:
            items = list(self._buf)

        count = len(items)
        if count == 0:
            # Nothing recorded yet (a freshly started service, or capacity=0)
            # -- every derived statistic is undefined, not zero, so this
            # reports None throughout rather than a misleading 0.0 that would
            # read as "abstains on everything" or "instant responses".
            return {"count": 0, "abstention_rate": None, "p50_ms": None, "p95_ms": None, "mean_trust": None}

        abstained = sum(1 for t in items if t.abstained)
        durations = sorted(t.total_ms for t in items)
        # Trust is undefined for an abstention (AnswerResult.trust is a fixed
        # 0.0 placeholder there, not a measurement -- pipeline.py's
        # `_declined`), so only answered requests contribute to mean_trust;
        # otherwise a service that abstains often would show a falsely low
        # mean_trust that actually reflects abstention rate, not grounding
        # quality on the requests it did answer.
        trusts = [t.trust_geometric for t in items if t.trust_geometric is not None]

        return {
            "count": count,
            "abstention_rate": abstained / count,
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "mean_trust": (sum(trusts) / len(trusts)) if trusts else None,
        }
