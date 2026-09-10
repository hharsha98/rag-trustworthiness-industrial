"""Pre-indexing checks for uploaded documents.

Ordered cheapest-first: extension, emptiness, and size are string/int
comparisons, magic-byte and encoding checks touch the bytes once, and the PDF
page count is the only check that has to open the file at all. Anything that
fails here is rejected before `RAGTrustPipeline.index_corpus` ever runs, which
matters because chunking and embedding are the expensive part of a corpus
upload -- there is no reason to pay for that on a file that was never going
to produce a usable index.
"""
import io
import os
from pathlib import Path

from ..pipeline import CORPUS_SUFFIXES


class UploadRejected(ValueError):
    """An uploaded file failed a pre-indexing check. Carries a message safe to
    return to an HTTP client."""


# Read from the environment at call time, not at import time -- tests
# monkeypatch RAGTRUST_MAX_UPLOAD_MB / RAGTRUST_MAX_PDF_PAGES per-test, and a
# module-level constant would freeze whatever value was set (or unset) the
# moment this module was first imported, which for a service process is
# effectively "once, forever".
def max_upload_bytes() -> int:
    return int(os.environ.get("RAGTRUST_MAX_UPLOAD_MB", "10")) * 1024 * 1024


def max_pdf_pages() -> int:
    return int(os.environ.get("RAGTRUST_MAX_PDF_PAGES", "100"))


def safe_display_name(filename: str) -> str:
    """The uploaded filename, reduced to something safe to store and render.

    This value is NEVER used to build a path (corpus ids are uuid4 -- see
    corpora.py), so this is not a traversal guard. It exists because the name is
    client-controlled text that ends up in two places where raw input is a
    liability: passages.json, where it becomes the `source` shown beside every
    citation, and the dashboard, which renders it into HTML. Stripping control
    characters and capping the length means a hostile filename cannot smuggle
    newlines into stored metadata or wallpaper the UI, and the frontend's own
    escaping is then a second layer rather than the only one.
    """
    name = Path(filename or "").name
    name = "".join(ch for ch in name if ch.isprintable())
    name = " ".join(name.split()).strip()
    return name[:128] or "document"


def validate_upload(filename: str, data: bytes) -> str:
    """Reject an upload before it reaches the indexer. Returns the normalised
    lowercase suffix (".pdf", ".md" or ".txt") on success."""
    suffix = Path(filename).suffix.lower()
    if suffix not in CORPUS_SUFFIXES:
        raise UploadRejected(
            f"Unsupported file type {suffix or '(none)'!r}; expected one of "
            f"{', '.join(CORPUS_SUFFIXES)}."
        )

    if len(data) == 0:
        raise UploadRejected("Uploaded file is empty.")

    if len(data) > max_upload_bytes():
        # Backstop only: the route enforces this while streaming so an oversized
        # body never has to be fully buffered first. This check exists for any
        # other caller of validate_upload (e.g. a future CLI import path) that
        # reads the whole file into memory before validating it.
        raise UploadRejected(
            f"File is {len(data)} bytes, exceeding the {max_upload_bytes()}-byte limit."
        )

    if suffix == ".pdf":
        # The extension is a claim made by the uploader; the header is evidence.
        # A renamed .exe or a truncated download both pass the extension check
        # and both fail this one.
        if not data.startswith(b"%PDF-"):
            raise UploadRejected("File has a .pdf extension but is not a valid PDF (missing %PDF- header).")
    else:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            # extract_sections_from_markdown() calls Path.read_text(encoding="utf-8"),
            # which would raise this same UnicodeDecodeError deep inside ingestion --
            # uncaught there, it surfaces as a 500 instead of a rejected upload.
            raise UploadRejected(f"File is not valid UTF-8 text: {exc}") from exc

    if suffix == ".pdf":
        # Page count is checked before extraction because extraction is unbounded
        # work: a PDF with a huge page count (or a decompression-bomb-style crafted
        # file) would otherwise let an attacker spend arbitrary CPU on the server
        # for the cost of one upload -- a free DoS. Rejecting on page count alone,
        # before any text is pulled out of a single page, bounds that cost.
        import pdfplumber

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                page_count = len(pdf.pages)
        except Exception as exc:
            raise UploadRejected(f"Could not open file as a PDF: {exc}") from exc
        if page_count > max_pdf_pages():
            raise UploadRejected(
                f"PDF has {page_count} pages, exceeding the {max_pdf_pages()}-page limit."
            )

    return suffix
