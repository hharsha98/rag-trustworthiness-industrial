"""PDF text extraction and sentence/claim segmentation.

`segment_sentences` does real sentence-boundary segmentation instead of a naive
newline split.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(path: str) -> str:
    return "\n".join(extract_pages_from_pdf(path))


def load_corpus(path: str) -> list:
    """Load a corpus as a list of "pages", dispatching on file extension.

    `.md`/`.txt` corpora are split on top-level `## ` headings; PDFs are split per page.
    Both feed `chunk_passages` identically, so the rest of the pipeline is unchanged.

    The default corpus is `data/demo_corpus.md`, original text written for this
    repository. See `data/README.md`.
    """
    lowered = str(path).lower()
    if lowered.endswith(".pdf"):
        return extract_pages_from_pdf(path)
    if lowered.endswith((".md", ".txt")):
        return extract_sections_from_markdown(path)
    raise ValueError(f"Unsupported corpus type: {path!r} (expected .pdf, .md or .txt)")


def extract_sections_from_markdown(path: str) -> list:
    """Split a Markdown/plain-text corpus into sections on `## ` headings.

    Each section keeps its heading as its first line, so `chunk_passages` windows a
    heading together with the prose beneath it -- a naive line-by-line split would
    lose that pairing.
    """
    import re

    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    sections = []
    for part in parts[1:] if len(parts) > 1 else parts:
        cleaned = part.strip()
        if cleaned:
            sections.append(cleaned)
    return sections


def extract_pages_from_pdf(path: str) -> list:
    """Per-page text, so chunking can respect page boundaries.

    `extract_text_from_pdf` joins these with a newline. Joining without a separator
    would merge the last line of each page into the first line of the next, corrupting
    the corpus.
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


# ---------------------------------------------------------------------------
# Sentence segmentation -- real sentence boundaries, no nltk/spacy dependency
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "fig.", "eq.", "eqs.", "al.", "no.", "nos.", "vol.",
    "pp.", "p.", "st.", "approx.", "cf.", "inc.", "ltd.", "co.", "corp.",
    "a.m.", "p.m.", "u.s.", "u.k.",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def segment_sentences(text: str, min_chars: int = 20) -> list:
    """Split `text` into sentences using sentence-final punctuation, with
    guards against common abbreviations, then merge any fragment shorter
    than `min_chars` into the preceding sentence."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    raw_pieces = [p for p in _SENTENCE_SPLIT_RE.split(normalized) if p.strip()]

    # Re-merge splits that were only caused by an abbreviation before the
    # sentence-final punctuation (e.g. "Dr. Smith" should not end a sentence).
    unmerged: list = []
    for piece in raw_pieces:
        piece = piece.strip()
        if unmerged:
            prev = unmerged[-1]
            last_word = re.search(r"(\S+)$", prev)
            ends_in_abbrev = bool(
                last_word and last_word.group(1).lower() in _ABBREVIATIONS
            )
            if ends_in_abbrev:
                unmerged[-1] = prev + " " + piece
                continue
        unmerged.append(piece)

    # Merge fragments shorter than min_chars into the previous sentence so
    # that short trailing junk (headers, page numbers, stray punctuation)
    # doesn't become its own "claim".
    merged: list = []
    for s in unmerged:
        if merged and len(s) < min_chars:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# v2 retrieval chunking
# ---------------------------------------------------------------------------


def chunk_passages(pages: list, window: int = 8, stride: int = 4,
                   min_chars: int = 30, min_line_chars: int = 3) -> list:
    """Group each page's lines into overlapping windows to form retrievable passages.

    Why this exists. A slide-based or otherwise layout-heavy corpus, chunked naively
    by keeping each PDF layout line as its own "passage", yields short fragments
    (a handful of words each) that mostly read as section headings rather than
    content. That is a poor unit for retrieval: an NLI model cannot return
    ENTAILMENT against a five-word heading, and a generator correctly instructed to
    answer only from its context has nothing substantive to answer from.

    Overlapping windows keep a heading together with the content beneath it. On the
    bundled corpus this yields 192 passages averaging 54 words, after which context
    relevance separates cleanly between in-corpus questions (0.55-0.65) and
    out-of-corpus ones (peak 0.10) -- which is what makes the abstention threshold
    principled rather than arbitrary.

    Args:
        pages: per-page text, from `extract_pages_from_pdf`.
        window: lines per passage.
        stride: line offset between consecutive windows; `stride < window` overlaps.
        min_chars: drop assembled passages shorter than this.
        min_line_chars: drop source lines shorter than this before windowing.

    Returns:
        Dicts of ``{"page": 1-based page number, "text": passage}``, de-duplicated
        while preserving order.
    """
    if stride < 1 or window < 1:
        raise ValueError("window and stride must both be >= 1")

    out: list = []
    for page_no, page_text in enumerate(pages, start=1):
        lines = [ln.strip() for ln in (page_text or "").split("\n")
                 if len(ln.strip()) > min_line_chars]
        if not lines:
            continue
        for start in range(0, max(1, len(lines)), stride):
            text = " ".join(lines[start:start + window])
            if len(text) >= min_chars:
                out.append({"page": page_no, "text": text})
            if start + window >= len(lines):
                break

    seen: set = set()
    deduped: list = []
    for chunk in out:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            deduped.append(chunk)
    return deduped
