"""Sliding-window rate limiting for `POST /corpora`.

The service is about to be exposed publicly with no authentication, and an
upload spends two things an anonymous caller does not pay for: disk (the raw
file, kept under CorpusStore's root) and CPU (chunking + embedding, the
expensive part of `CorpusStore.create`). Both need a hard ceiling before that
happens. See `service.py::create_app` for how the limiters built here are
wired into the route, and the module docstring on `client_key` below for why
identifying "one caller" is the part of this that is usually gotten wrong.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Tuple


class SlidingWindowLimiter:
    """Allows at most `max_events` events per key in any trailing
    `window_seconds` window.

    `max_events <= 0` means "deny everything" -- a deliberate policy (e.g. an
    operator setting RAGTRUST_UPLOAD_RATE_LIMIT=0 to shut uploads off without a
    code change or restart-free flag flip), not an error. `window_seconds <= 0`
    IS an error: there is no sensible window of zero or negative length, and
    silently accepting one would make every request either always-allowed or
    always-denied depending on float rounding rather than failing loudly at
    construction time.
    """

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds!r}")
        self.max_events = max_events
        self.window_seconds = window_seconds
        # Injectable so tests can advance time deterministically instead of
        # sleeping a real hour to exercise window expiry.
        self._clock = clock
        # uvicorn runs sync route handlers in a threadpool, so two uploads from
        # different clients (or the same one, racing) can call check()
        # concurrently. Without this lock, two threads could both observe
        # len(dq) < max_events and both append, letting max_events+1 events
        # through -- the exact bypass this class exists to prevent.
        self._lock = threading.Lock()
        self._events: Dict[str, Deque[float]] = {}

    def _prune_locked(self, now: float) -> None:
        """Drop expired timestamps from every key, and drop keys left empty.

        Called on every check(), for every key, not just the one being
        checked. The key space here is attacker-controlled: `client_key`
        below turns one key per source address, and a public endpoint lets
        any visitor mint a "new" address (a fresh IPv6 host, or -- when
        trust_proxy is misconfigured -- a fresh forged X-Forwarded-For value
        per request). A dict that only ever grows as distinct keys show up is
        itself a memory-exhaustion vector, which would make the rate limiter
        the thing an attacker uses to grow this process's memory rather than
        the thing that stops them. Pruning here keeps this map's size bounded
        by the number of DISTINCT keys with an event in the last window, not
        the number of distinct keys ever seen. Must be called with `_lock`
        held.
        """
        cutoff = now - self.window_seconds
        empty_keys = []
        for key, events in self._events.items():
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                empty_keys.append(key)
        for key in empty_keys:
            del self._events[key]

    def check(self, key: str) -> Tuple[bool, int]:
        """Record an event for `key` if under the limit.

        Returns `(allowed, retry_after)`. `retry_after` is 0 when allowed, and
        otherwise the whole-second ceiling of the time until the oldest
        in-window event ages out (i.e. the earliest moment a retry could
        succeed) -- rounded up, and floored at 1, so a caller that waits
        exactly `retry_after` seconds and retries never arrives a fraction of
        a second early and gets denied again.
        """
        with self._lock:
            now = self._clock()
            self._prune_locked(now)

            if self.max_events <= 0:
                # Nothing is ever recorded for this policy, so there is no
                # "oldest event" to measure against -- the window length
                # itself is the only honest upper bound to report.
                return False, math.ceil(self.window_seconds)

            events = self._events.setdefault(key, deque())
            if len(events) < self.max_events:
                events.append(now)
                return True, 0

            retry_after = math.ceil(events[0] + self.window_seconds - now)
            return False, max(1, retry_after)

    def reset(self) -> None:
        """Forget every key. Tests use this to start each case from a clean
        limiter without rebuilding the whole FastAPI app."""
        with self._lock:
            self._events.clear()


def client_key(request, trust_proxy: bool) -> str:
    """The identity a rate limiter should bucket `request` under.

    `X-Forwarded-For` is a request header, and therefore entirely
    client-controlled. Trusting it unconditionally makes any per-client rate
    limit trivially bypassable: an attacker sends a different fake value on
    every request and every request looks like a brand-new client, forever.
    It is only meaningful when a trusted reverse proxy sits in front of this
    service and is known to overwrite the header on every request before it
    reaches uvicorn -- deploy/Caddyfile.snippet's Caddy instance does exactly
    that -- which is precisely what `trust_proxy=True` asserts is the case.

    Behind such a proxy, the opposite failure applies: with `trust_proxy`
    left False, every request arrives from the proxy's own loopback address,
    so `request.client.host` is 127.0.0.1 for every caller and all real users
    share a single bucket. The flag exists because exactly one of these two
    failure modes is live depending on deployment topology, and getting it
    backwards silences the rate limit instead of merely weakening it.
    """
    if trust_proxy:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded.strip():
            # Leftmost entry: the proxy chain appends onto the right (RFC
            # 7239-adjacent convention followed by Caddy), so the leftmost
            # value is the one the proxy received from the original client --
            # everything a well-behaved proxy appends after it describes hops
            # inside our own infrastructure, not the caller.
            return forwarded.split(",")[0].strip()
        # Header absent or empty even though we were told to trust it (e.g. a
        # health check hitting uvicorn directly, bypassing Caddy) -- fall back
        # rather than bucket every such request under an empty-string key.
    client = request.client
    return client.host if client is not None else "unknown"
