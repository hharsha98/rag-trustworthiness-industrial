"""Perturbation operators -- METRICS.md Part III, validation protocol.

Ground truth for metric validation is built by construction: start from a
grounded, real answer and apply an operator whose effect on faithfulness is
known analytically. Label-*flipping* operators (`hallucinated=True`) inject a
genuine, verifiable defect; label-*preserving* controls (`hallucinated=False`)
change the text without changing what it asserts, or change something other
than faithfulness (conciseness, relevance) so a metric that merely reacts to
*any* edit can be told apart from one that reacts to *hallucination*.

Every operator has the signature ``(answer: str, passages: list[str], rng:
numpy.random.Generator) -> Perturbation | None``, returning ``None`` when it
cannot apply to this particular answer (e.g. `number_corruption` on an answer
with no digits) so callers can skip it.

`off_topic_padding` is the one operator whose semantics ("a *non-retrieved*
corpus passage") need a passage pool wider than the single answer's own
retrieved passages. Rather than break the shared signature, `apply_all`
accepts an optional `distractor_pool` and, only for that operator, passes it
instead of the answer's own `passages` -- the operator function itself still
just receives "a list of passage strings to draw from" like every other
operator. If no distractor pool is supplied, it falls back to drawing from
`passages` itself (documented in the returned note).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingest.loader import segment_sentences

_CACHE_PATH = Path(__file__).resolve().parent / "_paraphrase_cache.json"
_OLLAMA_URL = "http://localhost:11434/api/generate"
_PARAPHRASE_MODEL = "phi4:latest"


@dataclass
class Perturbation:
    text: str
    operator: str
    hallucinated: bool
    note: str = ""


# ---------------------------------------------------------------------------
# Label-flipping operators (hallucinated=True)
# ---------------------------------------------------------------------------

# Sequences of 2-4 capitalised words ("Markov Decision Process") or a bare
# technical acronym ("LSTM", "CNN", "MDP").
_CAP_TERM_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _extract_terms(text: str) -> list:
    terms = {m.group() for m in _CAP_TERM_RE.finditer(text)}
    terms |= {m.group() for m in _ACRONYM_RE.finditer(text)}
    return sorted(terms)


def entity_swap(answer: str, passages: list, rng) -> "Perturbation | None":
    """Replace a capitalised multi-word term or technical acronym in the
    answer with a different one drawn from the corpus passages."""
    candidates = _extract_terms(answer)
    if not candidates:
        return None

    corpus_terms = _extract_terms(" ".join(passages))
    chosen = candidates[int(rng.integers(len(candidates)))]
    chosen_lower = chosen.lower()
    # Exclude terms that are the same (case-insensitively) or a substring
    # relationship of the chosen term (e.g. "Learning" inside "Deep Learning")
    # so the swap is a genuine, unambiguous meaning change.
    replacements = [
        t for t in corpus_terms
        if t.lower() != chosen_lower
        and chosen_lower not in t.lower()
        and t.lower() not in chosen_lower
    ]
    if not replacements:
        return None

    replacement = replacements[int(rng.integers(len(replacements)))]
    new_text = re.sub(r"\b" + re.escape(chosen) + r"\b", replacement, answer, count=1)
    if new_text == answer:
        return None
    return Perturbation(
        text=new_text, operator="entity_swap", hallucinated=True,
        note=f"replaced {chosen!r} with {replacement!r} (drawn from corpus passages)",
    )


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def number_corruption(answer: str, passages: list, rng) -> "Perturbation | None":
    """Perturb a number to a clearly different value -- change of magnitude
    or digits, never a +-1 nudge."""
    matches = list(_NUMBER_RE.finditer(answer))
    if not matches:
        return None

    m = matches[int(rng.integers(len(matches)))]
    original_str = m.group()
    original_val = float(original_str)

    factor = float(rng.choice([0.1, 0.01, 10.0, 100.0]))
    if original_val == 0:
        new_val = float(rng.integers(50, 999))
    else:
        new_val = original_val * factor
        # Guard the rare case a tiny value survives the scaling close to its
        # original size (e.g. 0.04 * 10 = 0.4, only +0.36): force a jump.
        if abs(new_val - original_val) < 2:
            sign = 1 if rng.random() < 0.5 else -1
            new_val = original_val + sign * max(50.0, abs(original_val) * 5 + 10)

    if "." in original_str:
        decimals = len(original_str.split(".")[1])
        new_str = f"{new_val:.{decimals}f}"
    else:
        new_str = str(int(round(new_val)))
    if new_str == original_str:
        new_str = str(int(round(new_val)) + 137)  # last-resort guaranteed difference

    new_text = answer[: m.start()] + new_str + answer[m.end():]
    return Perturbation(
        text=new_text, operator="number_corruption", hallucinated=True,
        note=f"changed {original_str!r} to {new_str!r}",
    )


# Ordered so already-negated forms are matched before their positive
# counterpart (otherwise "does not improve" would itself get re-flipped).
_NEGATION_PATTERNS = [
    (re.compile(r"\bdoes not improve\b", re.I), "improves"),
    (re.compile(r"\bdo not improve\b", re.I), "improve"),
    (re.compile(r"\bimproves\b", re.I), "does not improve"),
    (re.compile(r"\bimprove\b", re.I), "does not improve"),
    (re.compile(r"\bis not\b", re.I), "is"),
    (re.compile(r"\bis\b", re.I), "is not"),
    (re.compile(r"\bare not\b", re.I), "are"),
    (re.compile(r"\bare\b", re.I), "are not"),
    (re.compile(r"\bcannot\b", re.I), "can"),
    (re.compile(r"\bcan\b", re.I), "cannot"),
    (re.compile(r"\bdoes not allow\b", re.I), "allows"),
    (re.compile(r"\ballows\b", re.I), "does not allow"),
    (re.compile(r"\bdoes not retain\b", re.I), "retains"),
    (re.compile(r"\bretains\b", re.I), "does not retain"),
    (re.compile(r"\bdoes not reduce\b", re.I), "reduces"),
    (re.compile(r"\breduces\b", re.I), "does not reduce"),
    (re.compile(r"\bdoes not support\b", re.I), "supports"),
    (re.compile(r"\bsupports\b", re.I), "does not support"),
    (re.compile(r"\bdoes not include\b", re.I), "includes"),
    (re.compile(r"\bincludes\b", re.I), "does not include"),
    (re.compile(r"\bdoes not have\b", re.I), "has"),
    (re.compile(r"\bhas\b", re.I), "does not have"),
    (re.compile(r"\bconsists of\b", re.I), "does not consist of"),
]


def negation(answer: str, passages: list, rng) -> "Perturbation | None":
    """Insert a negation that reverses one claim in the answer."""
    hits = [(pat, repl) for pat, repl in _NEGATION_PATTERNS if pat.search(answer)]
    if not hits:
        return None

    pat, repl = hits[int(rng.integers(len(hits)))]
    match = pat.search(answer)
    matched_text = match.group()
    new_text = answer[: match.start()] + repl + answer[match.end():]
    return Perturbation(
        text=new_text, operator="negation", hallucinated=True,
        note=f"flipped {matched_text!r} to {repl!r}",
    )


_UNSUPPORTED_POOL = [
    "This finding was independently confirmed by three separate industrial deployments in 2023.",
    "The technique originated at a robotics lab in Munich before being adopted worldwide.",
    "A follow-up study found that this approach reduces energy consumption by over 40 percent.",
    "This is now considered standard practice across all major cloud providers.",
    "The method was first proposed in a 1998 paper that went largely unnoticed for a decade.",
    "Industry benchmarks show this consistently outperforms all competing approaches by a wide margin.",
    "Regulatory guidelines in the EU now require this technique for safety-critical systems.",
    "A 2021 survey of practitioners found this to be the single most requested feature.",
]


def unsupported_addition(answer: str, passages: list, rng) -> "Perturbation | None":
    """Append a fluent, plausible sentence not supported by any passage."""
    if not answer or not answer.strip():
        return None
    sentence = _UNSUPPORTED_POOL[int(rng.integers(len(_UNSUPPORTED_POOL)))]
    base = answer.rstrip()
    if not base.endswith((".", "!", "?")):
        base += "."
    new_text = base + " " + sentence
    return Perturbation(
        text=new_text, operator="unsupported_addition", hallucinated=True,
        note=f"appended unsupported sentence: {sentence!r}",
    )


# ---------------------------------------------------------------------------
# Label-preserving controls (hallucinated=False)
# ---------------------------------------------------------------------------

# Word/phrase substitutions used only when Ollama is unreachable. Chosen to
# be meaning-preserving (near-synonyms in this technical-answer register),
# not a semantic change.
_SYNONYM_MAP = [
    (re.compile(r"\bshows\b", re.I), "demonstrates"),
    (re.compile(r"\bshow\b", re.I), "demonstrate"),
    (re.compile(r"\buses\b", re.I), "utilizes"),
    (re.compile(r"\buse\b", re.I), "utilize"),
    (re.compile(r"\bmethod\b", re.I), "technique"),
    (re.compile(r"\bmethods\b", re.I), "techniques"),
    (re.compile(r"\ballows\b", re.I), "enables"),
    (re.compile(r"\bhelps\b", re.I), "assists"),
    (re.compile(r"\bimportant\b", re.I), "significant"),
    (re.compile(r"\bdetermines\b", re.I), "establishes"),
    (re.compile(r"\bdetermine\b", re.I), "establish"),
    (re.compile(r"\bconsists of\b", re.I), "is composed of"),
    (re.compile(r"\bincludes\b", re.I), "comprises"),
    (re.compile(r"\bpurpose\b", re.I), "function"),
    (re.compile(r"\breduce\b", re.I), "decrease"),
    (re.compile(r"\breduces\b", re.I), "decreases"),
    (re.compile(r"\bretain\b", re.I), "preserve"),
    (re.compile(r"\bretains\b", re.I), "preserves"),
    (re.compile(r"\bprevious\b", re.I), "prior"),
    (re.compile(r"\bability\b", re.I), "capacity"),
    (re.compile(r"\bserve to\b", re.I), "function to"),
    (re.compile(r"\baccording to\b", re.I), "based on"),
]


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _llm_paraphrase(answer: str) -> "str | None":
    """Try an Ollama phi4:latest paraphrase at temperature 0, seed 0. Returns
    None (never raises) if Ollama is unreachable or the call fails, so the
    rule-based fallback always has a clean path."""
    try:
        import requests

        resp = requests.post(
            _OLLAMA_URL,
            json={
                "model": _PARAPHRASE_MODEL,
                "prompt": (
                    "Rewrite the following text so it means exactly the same thing, "
                    "using different wording. Do not add, remove, or change any fact, "
                    "number, or name. Reply with only the rewritten text, nothing else.\n\n"
                    + answer
                ),
                "stream": False,
                "options": {"temperature": 0.0, "seed": 0},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return text or None
    except Exception:
        return None


def _rule_based_paraphrase(answer: str) -> str:
    text = answer
    changed = False

    def _make_sub(repl):
        def _sub(m):
            nonlocal changed
            changed = True
            matched = m.group()
            if matched[:1].isupper():
                return repl[:1].upper() + repl[1:]
            return repl
        return _sub

    for pattern, repl in _SYNONYM_MAP:
        text = pattern.sub(_make_sub(repl), text, count=1)

    if not changed:
        stripped = text.strip()
        if stripped:
            text = "In other words, " + stripped[:1].lower() + stripped[1:]
    return text


def paraphrase(answer: str, passages: list, rng) -> "Perturbation | None":
    """Meaning-preserving rewrite: LLM paraphrase (Ollama phi4:latest, temp 0,
    seed 0) cached to `_paraphrase_cache.json`, falling back to a
    deterministic rule-based rewrite when Ollama is unreachable."""
    if not answer or not answer.strip():
        return None

    cache = _load_cache()
    if answer in cache:
        text = cache[answer]
        note = "paraphrase (cached LLM output, phi4:latest temp=0 seed=0)"
    else:
        text = _llm_paraphrase(answer)
        if text:
            cache[answer] = text
            _save_cache(cache)
            note = "paraphrase (phi4:latest, temp=0, seed=0)"
        else:
            text = _rule_based_paraphrase(answer)
            note = "paraphrase (rule-based fallback -- Ollama unreachable)"

    if text.strip() == answer.strip():
        return None
    return Perturbation(text=text, operator="paraphrase", hallucinated=False, note=note)


def duplication(answer: str, passages: list, rng) -> "Perturbation | None":
    """Repeat one sentence of the answer verbatim, immediately after itself."""
    sentences = segment_sentences(answer)
    if not sentences:
        return None

    target = sentences[int(rng.integers(len(sentences)))]
    pos = answer.find(target)
    if pos == -1:
        base = answer.rstrip()
        if not base.endswith((".", "!", "?")):
            base += "."
        new_text = base + " " + target
    else:
        insert_at = pos + len(target)
        new_text = answer[:insert_at] + " " + target + answer[insert_at:]
    return Perturbation(
        text=new_text, operator="duplication", hallucinated=False,
        note=f"duplicated sentence: {target!r}",
    )


def off_topic_padding(answer: str, passages: list, rng) -> "Perturbation | None":
    """Append a sentence taken verbatim from a passage in `passages` -- when
    called via `apply_all` with a `distractor_pool`, that pool (real corpus
    text not retrieved for this question) is passed here instead of the
    answer's own retrieved passages, making the padding genuinely off-topic
    yet still corpus-grounded."""
    candidates = [p for p in passages if p and p.strip()]
    if not answer or not answer.strip() or not candidates:
        return None

    passage_text = candidates[int(rng.integers(len(candidates)))]
    sentences = segment_sentences(passage_text)
    sentence = (sentences[0] if sentences else passage_text).strip()
    if not sentence:
        return None

    base = answer.rstrip()
    if not base.endswith((".", "!", "?")):
        base += "."
    new_text = base + " " + sentence
    return Perturbation(
        text=new_text, operator="off_topic_padding", hallucinated=False,
        note=f"appended verbatim off-topic passage text: {sentence!r}",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_LABEL_FLIPPING = (entity_swap, number_corruption, negation, unsupported_addition)
_LABEL_PRESERVING = (paraphrase, duplication, off_topic_padding)
_ALL_OPERATORS = _LABEL_FLIPPING + _LABEL_PRESERVING


def apply_all(answer: str, passages: list, seeds=(0, 1, 2), distractor_pool=None) -> list:
    """Every applicable operator at each seed, plus the unmodified original.

    `distractor_pool`, if given, is used only for `off_topic_padding` in
    place of `passages` (see module docstring) so its padding is drawn from
    passages that were not retrieved for this question.
    """
    out = [Perturbation(text=answer, operator="original", hallucinated=False,
                         note="unmodified answer")]
    for op_idx, op in enumerate(_ALL_OPERATORS):
        source = distractor_pool if (op is off_topic_padding and distractor_pool) else passages
        for seed in seeds:
            rng = np.random.default_rng([seed, op_idx])
            result = op(answer, source, rng)
            if result is not None:
                out.append(result)
    return out
