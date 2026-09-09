# Data

## What is here

| File | Contents |
|---|---|
| `demo_corpus.md` | A short original primer on AI in robotics, written for this repository. The default corpus for the demo, the pipeline and the tests. |
| `cached_answers.json` | Answers produced by actually running the pipeline — retrieval plus a local open-weight model via Ollama — over `demo_corpus.md`. Backs the hosted demo so it works without an inference token. |
| `benchmarks/` | Cached scores for the third-party validation benchmarks. Untracked (`.gitignore`); regenerated on demand by `experiments/10`–`14`. |

## Why the corpus is bundled

The demo and the test suite need *some* corpus to be meaningful to anyone who clones this
repository, and a repository that only works once you supply your own documents is hard to
evaluate. `demo_corpus.md` is original text written for that purpose, covering the same topics
as the preset questions in `cached_answers.json` so those questions stay answerable.

It also turns out to be a good demonstration corpus. Run over it, the pipeline answers
**10 of 10** in-corpus questions and abstains on **10 of 10** out-of-corpus ones, with mean top
retrieval scores of 0.722 against 0.082 — a clean separation that makes the abstention
threshold meaningful rather than arbitrary.

## Using your own documents

Point the indexer at any file or directory:

```bash
ragtrust index /path/to/your/documents --out ./index
```

Markdown and plain text work directly. PDF ingestion (`pdfplumber`) is supported too — drop a
PDF at `data/sample.pdf` to additionally exercise the PDF path in `tests/test_ingest.py`, which
is skipped when no PDF is present.
