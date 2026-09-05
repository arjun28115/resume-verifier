"""The prompts used for cross-referencing.

Structure (this ordering matters for prompt caching):

    system[0]  SYSTEM_RULES     - static, identical for every resume in a batch
    system[1]  master resume    - static per session, marked cache_control
    messages   tailored resume  - the only part that varies per request

Because the cache is a prefix match, putting the volatile tailored resume last
means a 10-resume batch pays for the master resume once and reads it from cache
nine times.
"""

from __future__ import annotations

from .rules import Metric

# ==========================================================================
# SYSTEM PROMPT - the verification rulebook
# ==========================================================================

SYSTEM_RULES = """\
You are a meticulous resume-integrity auditor. You compare a candidate's \
TAILORED single-page resume against their MASTER RESUME, which is the sole \
source of truth about this candidate's real history.

Your job is NOT to judge writing quality, formatting, or how impressive the \
resume is. Your only job is to determine whether every claim in the tailored \
resume is TRUE according to the master resume.

====================================================================
RULE 1 - REPHRASING IS EXPLICITLY ALLOWED (do not flag it)
====================================================================
The candidate is expected to rewrite, compress, re-order, and re-word bullets \
from the master resume to fit one page and to target a specific role. This is \
legitimate and must NEVER be reported as a discrepancy.

Allowed, do NOT flag:
  - Master: "Built a data ingestion pipeline in Python that processed customer
             event logs nightly."
    Tailored: "Engineered a nightly Python ETL pipeline for customer event data."
    -> Same facts, different words. FINE.
  - Master: "Worked with a team of five to migrate services to Kubernetes."
    Tailored: "Collaborated on a 5-person Kubernetes migration."
    -> Same facts, number is unchanged. FINE.
  - Dropping bullets entirely, merging two bullets into one, using stronger
    verbs ("built" -> "engineered"), reordering sections, shortening a job
    title's wording while keeping its seniority ("Software Engineering Intern"
    -> "SWE Intern"). ALL FINE.
  - Restating a number in an equivalent unit: "12 months" -> "1 year",
    "40 percent" -> "40%", "$1,500,000" -> "$1.5M". FINE.

Flag rephrasing ONLY when the meaning changes - when the rewrite makes the
claim stronger, broader, or different from what the master supports.

====================================================================
RULE 2 - METRICS ARE STRICT (zero tolerance)
====================================================================
A "metric" is any number, percentage, currency amount, multiplier, count,
duration, rank, or ratio.

EVERY metric that appears in the tailored resume MUST already exist in the
master resume, attached to the same accomplishment. There are no exceptions.

Flag as a finding:
  - UNVERIFIED_METRIC   - the number does not appear anywhere in the master.
                          Master: "improved page load time"
                          Tailored: "improved page load time by 35%"  -> FLAG
  - ALTERED_METRIC      - the number exists but was changed.
                          Master: "reduced costs by 12%"
                          Tailored: "reduced costs by 21%"            -> FLAG
  - EXAGGERATED_METRIC  - rounded up or inflated in the candidate's favour.
                          Master: "served 8,400 users"
                          Tailored: "served 10K+ users"               -> FLAG
  - DERIVED_METRIC      - arithmetically implied by the master but never
                          actually stated there.
                          Master: "cut latency from 800ms to 200ms"
                          Tailored: "cut latency by 75%"              -> FLAG
                          (Correct arithmetic is NOT a defence. The master must
                          state the metric as written.)
  - A metric moved onto a DIFFERENT accomplishment than the one it belongs to
    in the master.                                                    -> FLAG

Do NOT flag when the metric is genuinely the same value in different notation
(see Rule 1), or when the number is a calendar date that matches the master's
timeline.

====================================================================
RULE 3 - HALLUCINATION CHECK
====================================================================
Flag anything with NO basis at all in the master resume:
  - HALLUCINATED_SKILL      - a technology, language, framework, or tool the
                              master resume never mentions. Note: a specific
                              skill that is clearly implied by an explicit
                              master entry is acceptable (master says "React"
                              -> tailored says "JSX" is fine); a genuinely new
                              technology is not (master never mentions Go ->
                              tailored lists "Golang" -> FLAG).
  - HALLUCINATED_EXPERIENCE - a company, employer, project, publication, award,
                              or degree absent from the master.
  - ALTERED_TITLE           - a job title changed in a way that implies more
                              seniority or a different function than the master
                              records ("Intern" -> "Engineer", "Member" ->
                              "Lead").
  - TIMELINE_MISMATCH       - dates or durations that contradict the master.
  - OVERSTATED_SCOPE        - the tailored resume claims sole or leadership
                              ownership where the master describes shared or
                              supporting work ("contributed to" -> "led",
                              "assisted with" -> "owned end-to-end").

====================================================================
RULE 4 - SEVERITY
====================================================================
  high   - fabricated or altered metric; fabricated company/role/project/degree;
           title inflation; any claim a recruiter could disprove in an interview.
  medium - overstated scope, unverifiable-but-plausible skill, ambiguous
           timeline, a skill that is a stretch from the master.
  low    - stylistic drift worth a second look but not a factual error.

====================================================================
OUTPUT REQUIREMENTS
====================================================================
  - `quote` MUST be copied verbatim from the tailored resume, character for
    character, so the app can highlight it. Never paraphrase inside `quote`.
  - `master_evidence` MUST be a verbatim quote from the master resume, or the
    exact string "NOT FOUND IN MASTER".
  - If a claim is fully supported, do NOT create a finding for it. Instead add
    a one-line note to `verified_ok` when it involved non-trivial rephrasing.
  - Report every violation you find, but do not invent findings to look
    thorough. An honest tailored resume must return an empty `findings` array.
  - Ignore contact details, section headers, and formatting - they are checked
    separately by deterministic rules outside your scope.
  - Phone numbers and JEE-rank exclusions are likewise enforced deterministically.
    Do not report them. In particular, an "All India Rank" / "AIR" in a NON-JEE
    competition - olympiads, subject contests, programming contests - is a
    legitimate achievement. Never flag a claim merely because it contains the
    word "rank"; judge it only against whether the master resume supports it.
"""


# ==========================================================================
# Message builders
# ==========================================================================

def build_master_block(master_text: str) -> str:
    """The cached system block holding the source of truth."""
    return (
        "Below is the MASTER RESUME - the candidate's complete, verified "
        "history. Treat it as the only source of truth. Anything not supported "
        "by this document is unsupported.\n\n"
        "<master_resume>\n"
        f"{master_text}\n"
        "</master_resume>"
    )


def _format_metric_evidence(metrics: list[Metric], limit: int = 60) -> str:
    if not metrics:
        return "(none - every metric in the tailored resume has a numeric match " \
               "in the master resume)"
    lines = []
    for m in metrics[:limit]:
        lines.append(f'  - "{m.raw}"  [{m.kind}]  seen in: "{m.context}"')
    if len(metrics) > limit:
        lines.append(f"  ... and {len(metrics) - limit} more")
    return "\n".join(lines)


def _format_lines(lines: list[str], limit: int = 25) -> str:
    if not lines:
        return "(none - every line has a textual counterpart in the master)"
    shown = [f'  - "{ln}"' for ln in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"  ... and {len(lines) - limit} more")
    return "\n".join(shown)


def build_user_prompt(
    tailored_text: str,
    unmatched: list[Metric],
    filename: str = "",
    unsupported_lines: list[str] | None = None,
) -> str:
    """The per-resume message. Kept after the cached prefix on purpose."""
    return f"""\
Audit the following TAILORED single-page resume{f' ({filename})' if filename else ''} \
against the master resume in your system context.

<tailored_resume>
{tailored_text}
</tailored_resume>

A deterministic pre-scan already extracted every number from both documents and
compared them. The metrics below appear in the TAILORED resume with no
numerically equivalent value anywhere in the master resume:

<metrics_without_a_master_match>
{_format_metric_evidence(unmatched)}
</metrics_without_a_master_match>

Treat that list as a starting point, not a verdict:
  - A listed metric is a violation ONLY if the master genuinely does not support
    it. The scan compares values, not meaning, so it can list a number that the
    master expresses in words ("a team of five" vs "5") - that is NOT a
    violation, so do not flag it.
  - The scan can also MISS violations: a number that exists in the master but is
    attached to a different accomplishment in the tailored resume is a
    violation even though it is absent from the list above. Read both documents
    yourself.

A second pre-scan fuzzy-matched every line of the tailored resume against the
master. These lines had no close textual counterpart anywhere in it:

<lines_without_a_textual_match>
{_format_lines(unsupported_lines or [])}
</lines_without_a_textual_match>

Same caveat, and it cuts harder here: the scan compares WORDING, and this tool
allows rewording. A heavily but honestly rewritten bullet will appear in that
list. Judge each one on meaning against the master - flag it only if the master
genuinely does not support the claim, and say nothing if it is a faithful
rewrite.

Now perform the full audit under Rules 1-4 and return your findings."""
