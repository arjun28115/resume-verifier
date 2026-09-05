"""Line-level subset matching against the master resume.

Ported from the standalone `resume_processor` project, with one deliberate
change in policy.

That tool flags any line scoring below its match threshold - including a
reworded one - as a problem. This tool explicitly permits rephrasing, so a
low fuzzy score here is treated as *evidence*, never as a verdict:

    >= MATCH_THRESHOLD    the line is present in the master, near-verbatim
    >= REPHRASE_THRESHOLD  plausibly the same claim, reworded  -> informational
    <  REPHRASE_THRESHOLD  no textual basis found in the master -> real signal

Only the bottom band is treated as a finding, and even then the LLM adjudicates
it - a heavy but honest rewrite can score low. What this buys is a capability
the tool did not have before: without any API key it can now say "this entire
bullet has no counterpart in the master", not just "this number is unsupported".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MATCH_THRESHOLD = 88
REPHRASE_THRESHOLD = 65

# Lines shorter than this are section headers and page furniture - matching
# them tells us nothing.
MIN_LINE_CHARS = 12


@dataclass
class LineResult:
    raw: str
    score: float
    level: str            # "matched" | "rephrased" | "unsupported"
    page: int = 0
    rect: tuple[float, float, float, float] | None = None


def normalize(text: str) -> str:
    """Fold away cosmetic differences (bullets, dashes, quotes, case, spacing)."""
    text = unicodedata.normalize("NFKD", text)
    for bad, good in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(bad, good)
    text = re.sub(r"[‐-―−–—]", "-", text)
    text = re.sub(r"[•·▪◦●○►▶|*]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9@+#./%&'$ -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def dehyphenate(text: str) -> str:
    """Re-join words split across a line break by PDF hyphenation.

    Resume PDFs with column layouts break words mid-token: the master here
    contains "Materials char-\nacterization" and "Electronic Devices and
    Charac-". Without repair, a tailored resume that legitimately says
    "Electronic Devices and Characterization" has no match in the master and is
    reported as invented content.
    """
    return re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)


def build_master_corpus(master_text: str) -> tuple[str, list[str]]:
    """(full normalised text, per-line chunks) for the master resume."""
    repaired = dehyphenate(master_text)
    chunks = [normalize(ln) for ln in repaired.splitlines()]
    chunks = [c for c in chunks if c]
    # The joined text also lets a chunk match across the master's own line
    # breaks, which column layouts introduce constantly.
    return " ".join(chunks), chunks


def extract_pdf_lines(data: bytes) -> list[tuple[str, int, tuple]]:
    """(text, page_index, rect) for every visual line, in reading order.

    Returns an empty list if PyMuPDF is unavailable or the file is not a PDF;
    callers fall back to plain text lines (no annotation, matching still works).
    """
    try:
        import fitz
    except ImportError:
        return []

    lines: list[tuple[str, int, tuple]] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                for block in page.get_text("dict")["blocks"]:
                    if block.get("type") != 0:      # skip images
                        continue
                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"])
                        if text.strip():
                            lines.append((text, page.number, tuple(line["bbox"])))
    except Exception:   # noqa: BLE001 - a broken PDF just means no annotations
        return []
    return lines


def _score(norm_line: str, master_full: str, master_chunks: list[str]) -> float:
    from rapidfuzz import fuzz

    # partial_ratio finds the best window inside the master, which absorbs
    # line-wrap differences between the two documents.
    best = fuzz.partial_ratio(norm_line, master_full)
    if best >= MATCH_THRESHOLD:
        return best
    # token_set_ratio tolerates reordering within a bullet.
    for chunk in master_chunks:
        score = fuzz.token_set_ratio(norm_line, chunk)
        if score > best:
            best = score
            if best >= MATCH_THRESHOLD:
                break
    return best


def match_lines(tailored_text: str, master_text: str,
                pdf_bytes: bytes | None = None) -> list[LineResult]:
    """Score every line of the tailored resume against the master."""
    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        return []       # optional dependency - the rest of the tool still works

    master_full, master_chunks = build_master_corpus(master_text)
    if len(master_full) < 50:
        return []

    entries: list[tuple[str, int, tuple | None]] = []
    if pdf_bytes:
        entries = [(t, p, r) for t, p, r in extract_pdf_lines(pdf_bytes)]
    if not entries:
        entries = [(ln, 0, None) for ln in tailored_text.splitlines() if ln.strip()]

    results: list[LineResult] = []
    for raw, page, rect in entries:
        norm = normalize(raw)
        if len(norm) < MIN_LINE_CHARS:
            continue
        score = _score(norm, master_full, master_chunks)
        if score >= MATCH_THRESHOLD:
            level = "matched"
        elif score >= REPHRASE_THRESHOLD:
            level = "rephrased"
        else:
            level = "unsupported"
        results.append(LineResult(raw.strip(), round(score, 1), level, page, rect))
    return results


def unsupported(results: list[LineResult]) -> list[LineResult]:
    return [r for r in results if r.level == "unsupported"]


def coverage(results: list[LineResult]) -> float:
    """Fraction of content (length-weighted) with a counterpart in the master.

    Rephrased lines count in full - rewording is allowed, so it must not drag
    coverage down. Only lines with no textual basis at all are excluded.
    """
    if not results:
        return 1.0
    total = sum(len(r.raw) for r in results)
    found = sum(len(r.raw) for r in results if r.level != "unsupported")
    return found / total if total else 1.0
