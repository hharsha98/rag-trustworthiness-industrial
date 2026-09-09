from pathlib import Path

import pytest

from ragtrust.ingest.loader import (
    chunk_passages,
    extract_text_from_pdf,
    load_corpus,
    segment_sentences,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_CORPUS = ROOT / "data" / "demo_corpus.md"

# PDF ingestion is a real code path, but no PDF ships with this repository -- the
# bundled corpus is Markdown, which keeps a fresh clone small and self-contained.
# Drop any PDF at `data/sample.pdf` to exercise these tests; without one they skip,
# and every other ingestion test still runs against the demo corpus.
PDF_PATH = ROOT / "data" / "sample.pdf"
requires_pdf = pytest.mark.skipif(
    not PDF_PATH.exists(),
    reason="no PDF at data/sample.pdf - drop one there to exercise PDF ingestion",
)


@requires_pdf
def test_segment_sentences_produces_punctuation_terminated_units():
    text = extract_text_from_pdf(str(PDF_PATH))
    segments = segment_sentences(text)

    assert len(segments) > 0
    terminated = sum(1 for s in segments if s.rstrip().endswith((".", "!", "?")))
    assert terminated / len(segments) > 0.5


def test_segment_sentences_merges_short_fragments():
    text = "This is a normal sentence. Ok. This is another normal sentence here."
    result = segment_sentences(text, min_chars=20)
    # "Ok." (3 chars) must not survive as its own unit -- it gets merged
    # into a neighbouring sentence.
    assert all(len(s) >= 20 for s in result) or len(result) == 1
    assert "Ok." not in result


def test_segment_sentences_guards_common_abbreviations():
    text = "Dr. Smith works with robots. The lab published results in 2024."
    result = segment_sentences(text, min_chars=5)
    # "Dr." must not have split the sentence in two.
    assert not any(s.strip() == "Dr." for s in result)
    assert any("Dr. Smith" in s for s in result)


def test_segment_sentences_empty_input():
    assert segment_sentences("") == []
    assert segment_sentences("   \n\n  ") == []


# ---------------------------------------------------------------------------
# Corpus loading and chunking -- these run on the bundled demo corpus, so a
# fresh clone without the optional PDF still gets real coverage.
# ---------------------------------------------------------------------------


def test_load_corpus_reads_markdown_sections():
    sections = load_corpus(str(DEMO_CORPUS))
    assert len(sections) > 5
    assert all(s.strip() for s in sections)
    # Each section keeps its heading as the first line, so chunking can window a
    # heading together with the prose beneath it.
    assert any("Max-Pooling" in s for s in sections)


def test_load_corpus_rejects_unknown_extension(tmp_path):
    bogus = tmp_path / "corpus.docx"
    bogus.write_text("x")
    with pytest.raises(ValueError, match="Unsupported corpus type"):
        load_corpus(str(bogus))


def test_chunk_passages_produces_substantive_overlapping_passages():
    chunks = chunk_passages(load_corpus(str(DEMO_CORPUS)))
    assert len(chunks) > 10
    assert all(set(c) == {"page", "text"} for c in chunks)

    # What this guards against: naive line-based splitting yields ~9-word
    # fragments -- mostly slide headings -- which an NLI model cannot entail
    # and a generator cannot answer from. Chunks must be substantive.
    words = [len(c["text"].split()) for c in chunks]
    assert sum(words) / len(words) > 30

    # De-duplicated, and page provenance preserved for citation.
    assert len({c["text"] for c in chunks}) == len(chunks)
    assert all(c["page"] >= 1 for c in chunks)


def test_chunk_passages_overlaps_when_stride_less_than_window():
    pages = ["\n".join(f"line number {i} with enough text to survive" for i in range(12))]
    wide = chunk_passages(pages, window=6, stride=6)
    overlapped = chunk_passages(pages, window=6, stride=3)
    assert len(overlapped) > len(wide)


def test_chunk_passages_edge_cases():
    assert chunk_passages([]) == []
    assert chunk_passages(["", "   "]) == []
    with pytest.raises(ValueError):
        chunk_passages(["some text here that is long enough"], stride=0)
