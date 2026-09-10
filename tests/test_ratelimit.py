"""Tests for ragtrust.ratelimit: the sliding-window limiter and client_key,
both used to bound POST /corpora (see tests/test_service.py for the
service-level integration tests -- rate limiting wired into the actual route,
token auth, and the IPv6-style global-limit scenario)."""
import pytest

from ragtrust.ratelimit import SlidingWindowLimiter, client_key


class FakeClock:
    """A controllable clock so tests can jump an hour forward instantly
    instead of sleeping one, while still exercising the real prune/expiry
    logic (which only ever calls `self._clock()`, never `time.monotonic`
    directly)."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# --------------------------------------------------------- SlidingWindowLimiter


def test_allows_exactly_max_events_then_denies():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(3, 10, clock=clock)
    for _ in range(3):
        allowed, retry_after = limiter.check("a")
        assert allowed is True
        assert retry_after == 0
    allowed, retry_after = limiter.check("a")
    assert allowed is False
    assert 0 < retry_after <= 10


def test_window_expiry_allows_again():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 5, clock=clock)
    assert limiter.check("a")[0] is True
    assert limiter.check("a")[0] is False  # still within the window

    clock.advance(5.001)  # oldest (only) event is now outside the window
    allowed, retry_after = limiter.check("a")
    assert allowed is True
    assert retry_after == 0


def test_keys_are_independent():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 10, clock=clock)
    assert limiter.check("client-a")[0] is True
    assert limiter.check("client-b")[0] is True  # a's usage does not touch b
    assert limiter.check("client-a")[0] is False
    assert limiter.check("client-b")[0] is False


def test_pruning_empties_internal_map_after_all_events_age_out():
    # Covers the memory-exhaustion guard directly: the key space behind this
    # limiter is attacker-controlled (one key per source address), so a map
    # that only grows would itself be the vulnerability. This asserts the
    # internal map's size, not just observable allow/deny behaviour.
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, 5, clock=clock)
    limiter.check("a")
    limiter.check("b")
    assert len(limiter._events) == 2

    clock.advance(5.001)
    with limiter._lock:
        limiter._prune_locked(clock())
    assert len(limiter._events) == 0


def test_max_events_zero_denies_immediately():
    limiter = SlidingWindowLimiter(0, 10)
    allowed, retry_after = limiter.check("anyone")
    assert allowed is False
    assert retry_after > 0
    # Nothing was ever recorded for a deny-everything policy.
    assert len(limiter._events) == 0


@pytest.mark.parametrize("window_seconds", [0, -1, -0.5])
def test_non_positive_window_raises_value_error(window_seconds):
    with pytest.raises(ValueError):
        SlidingWindowLimiter(5, window_seconds)


def test_reset_clears_all_keys():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 10, clock=clock)
    limiter.check("a")
    assert limiter.check("a")[0] is False
    limiter.reset()
    assert limiter.check("a")[0] is True


# ------------------------------------------------------------------ client_key


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    """Enough of a Starlette Request to exercise client_key: `.client.host`
    and a `.headers.get(name, default)` lookup."""

    def __init__(self, client_host="203.0.113.5", headers=None):
        self.client = _FakeClient(client_host) if client_host is not None else None
        self.headers = headers or {}


def test_client_key_ignores_forged_forwarded_header_when_trust_proxy_false():
    # The header a hostile client controls (X-Forwarded-For) must NOT affect
    # identity when the deployment has not told us to trust it.
    req = _FakeRequest(client_host="203.0.113.5", headers={"X-Forwarded-For": "6.6.6.6"})
    assert client_key(req, trust_proxy=False) == "203.0.113.5"


def test_client_key_uses_leftmost_forwarded_entry_when_trust_proxy_true():
    req = _FakeRequest(client_host="127.0.0.1", headers={"X-Forwarded-For": "9.9.9.9, 127.0.0.1"})
    assert client_key(req, trust_proxy=True) == "9.9.9.9"


def test_client_key_falls_back_to_client_host_when_header_absent():
    req = _FakeRequest(client_host="203.0.113.5", headers={})
    assert client_key(req, trust_proxy=True) == "203.0.113.5"


def test_client_key_falls_back_when_forwarded_header_is_empty_string():
    req = _FakeRequest(client_host="203.0.113.5", headers={"X-Forwarded-For": "   "})
    assert client_key(req, trust_proxy=True) == "203.0.113.5"
