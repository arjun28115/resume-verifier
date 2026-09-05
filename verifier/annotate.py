"""Draw the findings onto a copy of the tailored resume PDF.

A list of flagged bullets tells you *what* is wrong; a highlighted page tells
you *where*, which is far faster to act on when you are working through a batch.
Ported from the standalone `resume_processor` project.

Colour meaning here follows this tool's rules, not a generic diff:
    red     - no textual basis in the master, or a strict exclusion
    orange  - a metric with no counterpart in the master
    yellow  - reworded (informational; rephrasing is allowed)
"""

from __future__ import annotations

RED = (1.0, 0.45, 0.45)
ORANGE = (1.0, 0.7, 0.35)
YELLOW = (1.0, 0.9, 0.45)

_COLOURS = {"unsupported": RED, "metric": ORANGE, "rephrased": YELLOW}


def annotate_pdf(pdf_bytes: bytes, line_results, metric_lines: set[str] | None = None,
                 exclusion_lines: set[str] | None = None,
                 include_rephrased: bool = False) -> bytes | None:
    """Return an annotated copy of the PDF, or None if it cannot be produced.

    ``metric_lines`` are raw line texts known to carry an unsupported metric;
    they are highlighted even when the line itself matched the master, because
    a changed number barely moves a fuzzy score.
    """
    try:
        import fitz
    except ImportError:
        return None

    metric_lines = metric_lines or set()
    exclusion_lines = exclusion_lines or set()
    drawable = [
        r for r in line_results
        if r.rect and (
            r.level == "unsupported"
            or r.raw in exclusion_lines
            or r.raw in metric_lines
            or (include_rephrased and r.level == "rephrased")
        )
    ]
    if not drawable:
        return None

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for result in drawable:
                if result.page >= doc.page_count:
                    continue
                page = doc[result.page]
                annot = page.add_highlight_annot(fitz.Rect(result.rect))

                if result.raw in exclusion_lines:
                    # Highlighted even though it is kept out of the score's
                    # content findings - the point of the marked-up PDF is to
                    # show a reviewer exactly where every problem sits.
                    kind, note = "unsupported", (
                        "Strict exclusion: this line carries a phone number or a "
                        "JEE rank, neither of which may appear on the resume.")
                elif result.level == "unsupported":
                    kind, note = "unsupported", (
                        f"No close textual match in the master resume "
                        f"(best match {result.score}%). This compares wording, "
                        f"so a heavily reworded but honest bullet also lands "
                        f"here - check it against the master before acting.")
                elif result.raw in metric_lines:
                    kind, note = "metric", (
                        "Contains a metric with no match in the master resume.")
                else:
                    kind, note = "rephrased", (
                        f"Reworded (best match {result.score}%). Rephrasing is "
                        f"allowed - shown for context.")

                annot.set_colors(stroke=_COLOURS[kind])
                annot.set_info(title="resume-verifier", content=note)
                annot.update()
            return doc.tobytes(garbage=3, deflate=True)
    except Exception:   # noqa: BLE001 - annotation is a bonus, never fatal
        return None
