"""Match Score - how much of a tailored resume is verified against the master.

Design decisions worth knowing
------------------------------
1. **Rephrasing must never cost points.** The obvious implementation - lexical
   overlap between the two documents - would punish exactly the behaviour the
   tool explicitly permits. A candidate who rewrites every bullet truthfully
   would score worse than one who copy-pastes. So the score is computed from
   *checkable facts* (metrics, findings, exclusions), never from wording.

2. **No double counting.** When the LLM runs, an unsupported metric already
   becomes a finding, so the raw metric ratio is not also charged. In rule-only
   mode there are no findings, so the metric ratio stands in as the proxy - and
   the score is marked provisional.

3. **Deterministic and explainable.** Every deduction is itemised, so a number
   can always be traced to the lines that produced it. Asking the model for a
   score would be neither reproducible nor auditable.
"""

from __future__ import annotations

from .schema import ScoreLine, ResumeReport

# Deduction weights.
PENALTY_PHONE = 35
PENALTY_BANNED_EXAM = 35
PENALTY_PER_EXTRA_PAGE = 25   # per page OVER the configured limit
PENALTY_HIGH = 12
PENALTY_MEDIUM = 5
PENALTY_LOW = 2
# Rule-only mode: scaled by the unmatched ratio. Weighted heavily enough
# that a resume with half its metrics unsupported cannot look 'nearly fine'.
PENALTY_UNSUPPORTED_METRICS = 50

# NOTE: line-level coverage is deliberately NOT scored.
# Fuzzy line matching measures WORDING, and this tool's premise is that
# wording may change freely. Measured: an honest rewrite of three bullets
# ('Built a limit order book simulator in C++' -> 'Engineered a C++ LOB
# simulator') drops coverage to 75%, which would have cost 27 points for
# changing nothing but words. Coverage is reported as evidence and fed to
# the LLM, which can judge meaning; it never moves the number.

# Labels are descriptive, not editorial. The tool observes what a document says
# against the master; it does not rule on intent or on whether a resume should
# be sent - that is the reviewer's call, and a word like "fabrication" asserts a
# motive this check cannot see.
BANDS = [
    (90, "Verified"),
    (75, "Minor issues"),
    (50, "Needs review"),
    (0,  "Major issues"),
]

_GENERIC = {
    "Verified": "Every checkable claim traces back to the master resume.",
    "Minor issues": "Broadly supported, with a few points to tighten.",
    "Needs review": "Several claims need checking against the master resume.",
    "Major issues": "Substantial parts are unsupported, or an exclusion applies.",
}


def band_for(score: int) -> tuple[str, str]:
    """Return (label, generic description) for a score.

    Kept for callers that only have a number. Prefer :func:`describe_score`,
    which reports the deductions that actually applied.
    """
    for threshold, label in BANDS:
        if score >= threshold:
            return label, _GENERIC[label]
    last = BANDS[-1][1]
    return last, _GENERIC[last]


def band_for_report(report: ResumeReport) -> str:
    """Band label for a whole report, not just its number.

    The score can be a clean 100 while lines still await review, and printing
    "Verified" above "1 line needs review" contradicts itself. The band follows
    the verdict in that case.
    """
    label = band_for(report.match_score)[0]
    if label == "Verified" and report.unsupported_lines and not report.llm_used:
        return "Needs review"
    return label


def describe_score(score: int, lines: list[ScoreLine]) -> str:
    """Describe a score using the deductions that actually applied.

    A fixed per-band sentence states a cause that may not hold: a resume can
    fall below 50 purely by accumulating findings, with no exclusion violation
    at all. Building the text from the real deductions keeps it true for every
    input, and keeps the tool reporting observations rather than motives.
    """
    applied = [line.label for line in lines if line.delta]
    if not applied:
        # No deductions, but there may still be something outstanding.
        notes = [line.label for line in lines
                 if not line.delta and line.label != "No deductions"]
        if notes:
            joined = "; ".join(notes)
            return joined[0].upper() + joined[1:]
        return _GENERIC[band_for(score)[0]]
    if len(applied) > 3:
        applied = applied[:3] + [f"and {len(applied) - 3} more"]
    joined = "; ".join(applied)
    return joined[0].upper() + joined[1:]


def compute_match_score(report: ResumeReport) -> tuple[int, list[ScoreLine], bool]:
    """Return (score 0-100, itemised deductions, provisional?).

    ``provisional`` is True when no LLM cross-reference ran, because the
    deterministic layer alone cannot see hallucinated skills or altered wording.
    """
    lines: list[ScoreLine] = []
    score = 100.0

    if report.error and not report.llm_used:
        # The audit did not complete - a score would be misleading.
        return 0, [ScoreLine(label="Verification did not complete", delta=0)], True

    # --- strict exclusions ------------------------------------------------
    if report.has_phone:
        score -= PENALTY_PHONE
        lines.append(ScoreLine(label="Contains a phone number", delta=-PENALTY_PHONE))
    if report.has_banned_exam:
        score -= PENALTY_BANNED_EXAM
        named = ", ".join(sorted({h.split(":")[0] for h in report.banned_exam_hits
                                  if ":" in h})) or "an excluded exam"
        lines.append(ScoreLine(label=f"Mentions {named}", delta=-PENALTY_BANNED_EXAM))
    if not report.within_page_limit and report.page_count > report.page_limit:
        extra = report.page_count - report.page_limit
        cost = PENALTY_PER_EXTRA_PAGE * extra
        score -= cost
        lines.append(ScoreLine(
            label=f"{report.page_count} pages (limit {report.page_limit})",
            delta=-cost))

    # --- content findings -------------------------------------------------
    # Deterministic findings are already charged above as exclusions; counting
    # them again here would penalise the same violation twice.
    content = [f for f in report.findings if f.category != "OTHER"]
    weights = {"high": PENALTY_HIGH, "medium": PENALTY_MEDIUM, "low": PENALTY_LOW}
    for severity, weight in weights.items():
        count = sum(1 for f in content if f.severity == severity)
        if count:
            cost = weight * count
            score -= cost
            lines.append(ScoreLine(
                label=f"{count} {severity}-severity finding{'s' if count > 1 else ''}",
                delta=-cost))

    # --- metric support (rule-only mode only) -----------------------------
    provisional = not report.llm_used
    if provisional and report.total_metrics:
        ratio = len(report.unverified_metric_candidates) / report.total_metrics
        if ratio:
            cost = round(PENALTY_UNSUPPORTED_METRICS * ratio)
            score -= cost
            lines.append(ScoreLine(
                label=f"{len(report.unverified_metric_candidates)} of "
                      f"{report.total_metrics} metrics unmatched in the master",
                delta=-cost))

    # Not scored - fuzzy matching compares wording, so this band contains
    # honest rewrites and PDF-extraction artifacts as well as real inventions.
    # But it must not vanish either: reporting flagged lines while the verdict
    # says "everything traces back" is incoherent, so it is carried as a
    # zero-delta note that drives the status to WARNING instead.
    if provisional and report.unsupported_lines:
        n = len(report.unsupported_lines)
        lines.append(ScoreLine(
            label=f"{n} line{'s' if n > 1 else ''} with no match in the master "
                  f"- not scored, needs review",
            delta=0))

    final = int(max(0, min(100, round(score))))
    if not lines:
        lines.append(ScoreLine(label="No deductions", delta=0))
    return final, lines, provisional
