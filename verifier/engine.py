"""Orchestration: parse -> deterministic scan -> LLM audit -> merge -> verdict."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .annotate import annotate_pdf
from .identity import compare_identities, extract_identity
from .linematch import coverage as line_coverage
from .linematch import match_lines, unsupported
from .llm import BaseVerifier
from .pdf_utils import ParsedDocument, parse_upload
from .rules import (
    Metric,
    extract_metrics,
    strip_contact_noise,
    word_number_metrics,
    find_exam_references,
    find_phone_numbers,
    find_rank_mentions,
    DEFAULT_EXCLUDED_EXAMS,
    unmatched_metrics,
)
from .schema import Finding, ResumeReport, ScoreLine, Status
from .scoring import band_for_report, compute_match_score


@dataclass
class VerificationSettings:
    """Toggles exposed in the sidebar."""

    # A "single-pager" is the short-resume format, not literally one sheet -
    # plenty of them run to two pages. The limit is configurable rather than
    # hardcoded so a stricter or looser house rule is a setting, not an edit.
    max_pages: int = 2
    strict_page_limit: bool = True   # over the limit is a FAIL, not a warning
    check_calendar_years: bool = False  # also verify dates like "2024"
    max_workers: int = 4
    # Entrance exams barred from a tailored resume. See rules.EXAM_RULES
    # for each exam's mode (banned outright vs only with a rank/score).
    excluded_exams: tuple = DEFAULT_EXCLUDED_EXAMS


def analyse_master(master_doc: ParsedDocument) -> list[Metric]:
    """Harvest the master's metrics once and reuse for the whole batch."""
    clean = strip_contact_noise(master_doc.text)
    # Spelled-out numbers count as support: "a team of five" backs "5".
    return extract_metrics(clean) + word_number_metrics(clean)


def verify_one(
    name: str,
    data: bytes,
    master_text: str,
    master_metrics: list[Metric],
    verifier: BaseVerifier | None,
    settings: VerificationSettings,
) -> ResumeReport:
    """Run every check against a single tailored resume."""
    doc = parse_upload(name, data)
    report = ResumeReport(
        filename=name,
        page_count=doc.page_count,
        parse_warnings=list(doc.warnings),
    )

    if doc.is_empty:
        report.status = Status.FAIL
        report.error = (
            "No usable text could be extracted from this file. "
            "If it is a scanned PDF, OCR it first."
        )
        return report

    # ---------------- Identity gate ---------------------------------------
    # Runs first: if these are two different candidates, every downstream
    # check is meaningless (and the LLM call would just burn money describing
    # a stranger's resume as "unsupported").
    verdict = compare_identities(extract_identity(master_text), extract_identity(doc.text))
    report.identity_match = verdict.same_person
    report.identity_reason = verdict.reason

    if verdict.same_person is False:
        report.status = Status.FAIL
        report.findings.append(Finding(
            category="OTHER", severity="high",
            quote=verdict.tailored.name or (sorted(verdict.tailored.emails) or [""])[0],
            issue=f"This resume does not belong to the same candidate as the "
                  f"master resume. {verdict.reason}",
            master_evidence=verdict.master.name or (sorted(verdict.master.emails) or [""])[0],
            suggested_fix="Verify you uploaded the right master resume for this "
                          "candidate, then re-run.",
        ))
        report.match_score = 0
        report.score_lines = [ScoreLine(label="Different candidate to the master resume",
                                        delta=-100)]
        report.score_band = "Major issues"
        report.score_provisional = False
        return report

    # ---------------- Deterministic checks (never delegated to the LLM) ----
    report.page_limit = settings.max_pages
    report.within_page_limit = (
        doc.page_count <= settings.max_pages if doc.page_count else True)

    phone_hits = find_phone_numbers(doc.text)
    exam_hits = find_exam_references(doc.text, settings.excluded_exams)
    # Legitimate olympiad/contest ranks - reported for human review only.
    rank_mentions = find_rank_mentions(doc.text, settings.excluded_exams)
    report.has_phone = bool(phone_hits)
    report.has_banned_exam = bool(exam_hits)
    report.phone_hits = [h.match for h in phone_hits]
    report.banned_exam_hits = [f"{h.pattern}: {h.match}" if h.pattern not in
                               ("rank_with_exam_context",) else h.match
                               for h in exam_hits]
    report.rank_mentions = [f"{h.match} — {h.context}" for h in rank_mentions]

    for hit in phone_hits:
        report.findings.append(Finding(
            category="OTHER", severity="high", quote=hit.match,
            issue="Strict exclusion violated: the tailored resume contains a "
                  "phone number.",
            master_evidence="N/A - deterministic rule, not a master-resume check.",
            suggested_fix="Remove the phone number from the contact line. "
                          f"Context: ...{hit.context}...",
        ))
    for hit in exam_hits:
        report.findings.append(Finding(
            category="OTHER", severity="high", quote=hit.match,
            issue=f"Strict exclusion violated: the tailored resume references "
                  f"an excluded entrance exam ({hit.pattern}).",
            master_evidence="N/A - deterministic rule, not a master-resume check.",
            suggested_fix="Delete the mention. "
                          f"Context: ...{hit.context}...",
        ))
    if not report.within_page_limit:
        limit = settings.max_pages
        report.findings.append(Finding(
            category="OTHER",
            severity="high" if settings.strict_page_limit else "medium",
            quote=f"{doc.page_count}-page document",
            issue=f"This resume is {doc.page_count} pages; the limit is "
                  f"{limit} page{'s' if limit != 1 else ''}.",
            master_evidence="N/A - page-count check.",
            suggested_fix=f"Trim content until the PDF fits in {limit} "
                          f"page{'s' if limit != 1 else ''}.",
        ))

    # ---------------- Metric pre-scan (evidence for the LLM) --------------
    # Phone/JEE digits are stripped first - they are already hard failures and
    # would otherwise show up a second time as bogus 'unverified metrics'.
    tailored_metrics = extract_metrics(strip_contact_noise(doc.text))
    unmatched = unmatched_metrics(
        tailored_metrics, master_metrics, include_years=settings.check_calendar_years
    )
    report.unverified_metric_candidates = [m.raw for m in unmatched]
    report.total_metrics = len(tailored_metrics)

    # ---------------- Line-level subset match (works with no API key) ------
    line_results = match_lines(doc.text, master_text,
                               data if name.lower().endswith(".pdf") else None)
    # Drop lines already charged as a strict exclusion. A phone-number line has
    # no counterpart in the master by definition; reporting it again here would
    # double-report the same violation and double-charge it in the score.
    excluded_text = [h.match for h in phone_hits] + [h.match for h in exam_hits]
    exclusion_lines = {
        r.raw for r in line_results
        if any(bad and bad in r.raw for bad in excluded_text)
    }
    # Scoring and the "unsupported lines" list drop these; the annotated PDF
    # keeps them, because a reviewer needs to see where the violation is.
    reportable = [r for r in line_results if r.raw not in exclusion_lines]
    missing = unsupported(reportable)
    report.unsupported_lines = [r.raw for r in missing]
    report.rephrased_lines = sum(1 for r in reportable if r.level == "rephrased")
    report.coverage = round(line_coverage(reportable), 4)

    # Highlight the problems on a copy of the PDF itself.
    metric_lines = {
        r.raw for r in line_results
        if any(m.raw in r.raw for m in unmatched)
    }
    annotated = annotate_pdf(data, line_results, metric_lines, exclusion_lines)
    report.has_highlights = annotated is not None
    # Fall back to the untouched PDF so the viewer always has something to show
    # - a clean resume has nothing to highlight but still needs to be readable.
    if name.lower().endswith(".pdf"):
        report.annotated_pdf = annotated or data

    # ---------------- LLM cross-reference ---------------------------------
    if verifier is not None:
        try:
            llm_report = verifier.verify(doc.text, unmatched, name,
                                         report.unsupported_lines)
            report.findings.extend(llm_report.findings)
            report.verified_ok = llm_report.verified_ok
            report.overall_assessment = llm_report.overall_assessment
            report.llm_used = True
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never swallowed
            report.error = f"{exc.__class__.__name__}: {exc}"
    else:
        report.overall_assessment = (
            "Rule-only mode: deterministic checks ran, but no LLM cross-reference "
            "was performed. Metric and hallucination checks are incomplete."
        )

    report.status = _decide_status(report, settings)
    report.match_score, report.score_lines, report.score_provisional = (
        compute_match_score(report))
    report.score_band = band_for_report(report)
    return report


def _decide_status(report: ResumeReport, settings: VerificationSettings) -> Status:
    """Auto-fail conditions first, then severity-based grading."""
    if report.has_phone or report.has_banned_exam:
        return Status.FAIL
    if not report.within_page_limit and settings.strict_page_limit:
        return Status.FAIL
    if report.blocking_count:
        return Status.FAIL
    if report.error:
        # LLM leg failed - deterministic checks passed, but the audit is partial.
        return Status.WARNING
    if report.warning_count:
        return Status.WARNING
    # Content the deterministic pass could not match and no LLM has judged.
    # Not a failure - it is usually a rewrite or an extraction artifact - but
    # showing flagged lines under a green PASS is incoherent, so it asks for a
    # human look instead.
    if report.unsupported_lines and not report.llm_used:
        return Status.WARNING
    # Rule-only mode used to force WARNING here so a partial audit could never
    # read as a clean pass. In practice it made EVERY resume a warning, so the
    # column carried no information and trained people to ignore it. The
    # partial-audit caveat is already carried by `score_provisional` (the "*"
    # on the score, plus a caption saying what was not checked), so it does not
    # need encoding a second time at the cost of the whole signal.
    return Status.PASS


def verify_batch(
    files: list[tuple[str, bytes]],
    master_doc: ParsedDocument,
    verifier: BaseVerifier | None,
    settings: VerificationSettings,
    progress_cb=None,
) -> list[ResumeReport]:
    """Verify many resumes concurrently.

    Note: workers must not touch Streamlit APIs - results are rendered on the
    main thread after the pool drains.
    """
    master_metrics = analyse_master(master_doc)
    reports: list[ResumeReport] = []

    def _task(item: tuple[str, bytes]) -> ResumeReport:
        name, data = item
        return verify_one(name, data, master_doc.text, master_metrics, verifier, settings)

    with ThreadPoolExecutor(max_workers=max(1, settings.max_workers)) as pool:
        for i, result in enumerate(pool.map(_task, files), start=1):
            reports.append(result)
            if progress_cb:
                progress_cb(i, len(files), result)

    return reports
