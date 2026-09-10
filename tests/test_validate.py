"""Tests for pre-indexing upload checks. No model downloads, no network --
these only touch bytes and pdfplumber."""
import pytest

from ragtrust.ingest.validate import UploadRejected, max_upload_bytes, validate_upload

# A hand-built, structurally minimal one-page PDF. pdfminer (which pdfplumber
# wraps) recovers a usable page tree from this even though the xref table is a
# stub -- real-world PDFs from various producers are routinely this loose, so
# the test fixture matches that rather than a pristine file only a PDF library
# itself would ever produce.
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 50 100 Td (Hello World) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f
trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF
"""


def test_accepts_markdown():
    assert validate_upload("notes.md", b"## Heading\n\nSome body text.") == ".md"


def test_accepts_txt():
    assert validate_upload("notes.txt", b"Plain text content.") == ".txt"


def test_accepts_pdf():
    assert validate_upload("doc.pdf", _MINIMAL_PDF) == ".pdf"


def test_rejects_unsupported_extension():
    with pytest.raises(UploadRejected):
        validate_upload("report.docx", b"whatever bytes")


def test_rejects_empty_file():
    with pytest.raises(UploadRejected):
        validate_upload("notes.txt", b"")


def test_rejects_oversized_upload(monkeypatch):
    monkeypatch.setenv("RAGTRUST_MAX_UPLOAD_MB", "1")
    assert max_upload_bytes() == 1024 * 1024
    oversized = b"x" * (1024 * 1024 + 1)
    with pytest.raises(UploadRejected):
        validate_upload("notes.txt", oversized)


def test_rejects_pdf_without_magic_bytes():
    with pytest.raises(UploadRejected):
        validate_upload("doc.pdf", b"This is not a PDF, just text pretending to be one.")


def test_rejects_non_utf8_txt():
    # 0xff 0xfe is not valid UTF-8 (it is a UTF-16 BOM), so this must fail the
    # encoding check rather than reach the indexer's read_text(encoding="utf-8").
    with pytest.raises(UploadRejected):
        validate_upload("notes.txt", b"\xff\xfe\x00\x01garbage")


def test_rejects_pdf_over_page_limit(monkeypatch):
    monkeypatch.setenv("RAGTRUST_MAX_PDF_PAGES", "0")
    with pytest.raises(UploadRejected):
        validate_upload("doc.pdf", _MINIMAL_PDF)
