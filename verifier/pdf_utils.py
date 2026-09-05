"""PDF / text extraction helpers.

Two extractors are used on purpose:

* ``pdfplumber`` gives layout-aware text (keeps bullet lines separate, which
  matters a lot when we quote a bullet back to the user).
* ``pypdf`` is the fallback and is also the most reliable way to count pages
  when a PDF has odd/streamed page trees.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Everything downstream code needs to know about one uploaded file."""

    name: str
    text: str
    page_count: int
    extractor: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        # < 200 chars of text from a resume almost always means a scanned/image
        # PDF that would need OCR - worth telling the user rather than silently
        # producing a "no issues found" verdict on an empty string.
        return len(self.text.strip()) < 200


def _clean(text: str) -> str:
    """Normalise whitespace without destroying line structure."""
    text = text.replace(" ", " ").replace("﻿", "")
    # Ligatures & unicode bullets that PDF extraction commonly emits.
    for bad, good in (("ﬁ", "fi"), ("ﬂ", "fl"), ("–", "-"),
                      ("—", "-"), ("’", "'"), ("“", '"'),
                      ("”", '"'), ("•", "* "), ("●", "* ")):
        text = text.replace(bad, good)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _extract_with_pdfplumber(data: bytes) -> tuple[str, int]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        count = len(pdf.pages)
    return "\n".join(pages), count


def _extract_with_pypdf(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages), len(reader.pages)


def parse_upload(name: str, data: bytes) -> ParsedDocument:
    """Turn raw uploaded bytes into a :class:`ParsedDocument`.

    Accepts PDF, plain text and markdown. Non-PDF files are reported as a
    single page (the page-count check only meaningfully applies to PDFs).
    """
    lowered = name.lower()

    if lowered.endswith((".txt", ".md", ".markdown")):
        for encoding in ("utf-8", "latin-1"):
            try:
                return ParsedDocument(name, _clean(data.decode(encoding)), 1, "plaintext")
            except UnicodeDecodeError:
                continue
        return ParsedDocument(name, "", 0, "plaintext", ["Could not decode text file."])

    warnings: list[str] = []
    text, count, extractor = "", 0, "none"

    try:
        text, count = _extract_with_pdfplumber(data)
        extractor = "pdfplumber"
    except Exception as exc:  # noqa: BLE001 - any parse failure falls through
        warnings.append(f"pdfplumber failed ({exc.__class__.__name__}); trying pypdf.")

    # Fall back if pdfplumber is missing, errored, or returned almost nothing.
    if len(text.strip()) < 200:
        try:
            alt_text, alt_count = _extract_with_pypdf(data)
            if len(alt_text.strip()) > len(text.strip()):
                text, count, extractor = alt_text, alt_count, "pypdf"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pypdf failed ({exc.__class__.__name__}).")

    doc = ParsedDocument(name, _clean(text), count, extractor, warnings)
    if doc.is_empty and count > 0:
        doc.warnings.append(
            "Very little text extracted - this looks like a scanned/image PDF. "
            "Run OCR on it before trusting the verification result."
        )
    return doc
