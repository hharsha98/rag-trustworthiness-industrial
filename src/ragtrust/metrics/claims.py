"""Claim decomposition: an answer is split into atomic claims for per-claim
faithfulness and attribution scoring (METRICS.md Part II.1-2)."""
from ..ingest.loader import segment_sentences


def split_claims(answer: str) -> list:
    return segment_sentences(answer)
