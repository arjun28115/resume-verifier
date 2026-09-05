"""Deterministic (regex) checks.

Design note
-----------
The hard exclusions - phone number and JEE rank - are *never* delegated to the
LLM. They are objective, cheap to detect, and an auto-fail; a regex is both
more reliable and more auditable than a model verdict for these.

Metric harvesting is different. We extract every metric from both documents and
pre-compute which tailored-resume metrics have no counterpart in the master.
That list is handed to the LLM as *evidence*, not as a verdict - the model still
adjudicates, because "5 members" in the tailored resume may legitimately match
"a team of five" in the master, and because a metric can be present-but-altered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Hard exclusion 1: mobile / phone numbers
# --------------------------------------------------------------------------

_PHONE_PATTERNS: list[tuple[str, str]] = [
    # Explicitly labelled ("Mobile: 98765 43210", "Ph - +1 555 0100")
    ("labelled", r"(?i)\b(?:mobile|phone|tel(?:ephone)?|ph|cell|mob|contact\s*no)\b"
                 r"\s*(?:no\.?|number|#)?\s*[:\-]?\s*[+(]?\d[\d\s().\-]{6,}\d"),
    # International format: +91 98765 43210 / +1 (555) 123-4567
    ("international", r"\+\d{1,3}[\s.\-]?(?:\(?\d{2,5}\)?[\s.\-]?){1,4}\d{2,5}"),
    # North-American style: (555) 123-4567 / 555.123.4567
    ("na_format", r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"),
    # Indian 10-digit mobile, bare
    ("bare_10_digit", r"(?<![\d\-+/.])[6-9]\d{9}(?![\d\-/])"),
    # 5+5 split, common on Indian resumes: 98765 43210
    ("split_5_5", r"(?<![\d\-+/.])[6-9]\d{4}[\s.\-]\d{5}(?![\d\-/])"),
]

# --------------------------------------------------------------------------
# Hard exclusion 2: JEE rank / All India Rank
# --------------------------------------------------------------------------

# Entrance-exam exclusions.
#
# JEE and GATE are both banned on any mention, per the placement office's rule.
# Note the consequence: a legitimate M.Tech credential line such as "Qualified
# Graduate Aptitude Test in Engineering (GATE)" fails too, because the rule is
# about the exam appearing at all, not about a rank being quoted. That is a
# deliberate choice - flip an exam to "rank_only" below if it should instead be
# barred only when a rank/score/percentile is attached.
EXAM_RULES: dict[str, tuple[str, str]] = {
    "JEE": (r"(?i)\b(?:JEE|IIT[\s\-]?JEE|Joint\s+Entrance\s+Exam(?:ination)?)\b", "any"),
    "GATE": (r"(?i)\b(?:GATE|Graduate\s+Aptitude\s+Test\s+in\s+Engineering)\b", "any"),
    "CAT": (r"(?i)\b(?:CAT|Common\s+Admission\s+Test)\b", "rank_only"),
    "NEET": (r"(?i)\bNEET\b", "rank_only"),
}

DEFAULT_EXCLUDED_EXAMS = ("JEE", "GATE")

# Words that turn an exam mention into a rank/score claim (for "rank_only").
_SCORE_CONTEXT = (r"(?i)\b(?:rank|air|score|percentile|marks|topper|secured|"
                  r"qualified\s+with)\b")

# "AIR 10" / "All India Rank 3" on their own are NOT exam ranks. Candidates
# legitimately hold All India Ranks in olympiads and subject contests, and
# failing a resume for those was a false positive. A rank phrase counts only
# when an excluded exam is named in the SAME bullet/line.
_RANK_PHRASE = (r"(?i)\b(?:A\.?\s?I\.?\s?R\.?|All[\s\-]?India\s+Rank)"
                r"\s*[:\-#]?\s*\d[\d,]*")

# Extra context that implies JEE without naming it ("AIR 993 in Mains 2024").
# Deliberately narrow - a bare "Advanced" appears in plenty of non-JEE
# competition names.
_EXTRA_EXAM_CONTEXT = r"(?i)\bMains\b"


@dataclass
class RuleHit:
    """One deterministic detection, with enough context to show the user."""

    kind: str          # "phone" | "jee"
    pattern: str       # which sub-pattern matched
    match: str         # the exact matched substring
    context: str       # surrounding text, for the UI


def _token_around(text: str, start: int, end: int) -> str:
    """Return the whitespace-delimited token containing the span."""
    left = text.rfind(" ", 0, start) + 1
    left = max(left, text.rfind("\n", 0, start) + 1)
    right = min(
        x for x in (text.find(" ", end), text.find("\n", end), len(text)) if x != -1
    )
    return text[left:right]


def _context(text: str, start: int, end: int, width: int = 45) -> str:
    snippet = text[max(0, start - width): min(len(text), end + width)]
    return " ".join(snippet.split())


def find_phone_numbers(text: str) -> list[RuleHit]:
    """Detect phone numbers, with guards against common false positives.

    Guards applied:
      * total digit count must be 10-15 (kills years, "2021-2024", ZIP codes)
      * the containing token must not look like an email or a URL
      * overlapping matches from different patterns are de-duplicated
    """
    hits: list[RuleHit] = []
    claimed: set[int] = set()

    for pattern_name, pattern in _PHONE_PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.span()
            if any(i in claimed for i in range(start, end)):
                continue

            matched = m.group(0)
            digits = re.sub(r"\D", "", matched)

            # A labelled match is trusted at 7+ digits ("Ph: 555-0100");
            # anything unlabelled needs a full 10-15 digit phone shape.
            min_digits = 7 if pattern_name == "labelled" else 10
            if not (min_digits <= len(digits) <= 15):
                continue

            token = _token_around(text, start, end)
            if "@" in token or "://" in token or ".com" in token.lower():
                continue  # part of an email address or URL

            claimed.update(range(start, end))
            hits.append(RuleHit("phone", pattern_name, matched.strip(),
                                _context(text, start, end)))
    return hits


def _segment_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The bullet/line containing the span.

    Scoping to one bullet matters: a resume can carry
    "* AIR 993 in JEE Mains 2024" directly above
    "* AIR 10, Srinivasa Ramanujan Mathematics Competition".
    A naive character window would leak "JEE" from the first bullet into the
    second and fail the resume for an unrelated olympiad rank.
    """
    left = max(text.rfind("\n", 0, start) + 1, text.rfind("* ", 0, start) + 1)
    candidates = [i for i in (text.find("\n", end), text.find("* ", end)) if i != -1]
    right = min(candidates) if candidates else len(text)
    return left, right


def _exam_context_re(exams) -> "re.Pattern":
    """A regex matching any enabled exam's name, for rank-proximity checks.

    The per-exam patterns carry an inline ``(?i)``; a global flag is only legal
    at the start of an expression, so it is stripped before the alternation is
    built and applied as a compile flag instead.
    """
    parts = [EXAM_RULES[e][0] for e in exams if e in EXAM_RULES]
    parts.append(_EXTRA_EXAM_CONTEXT)
    stripped = [p.replace("(?i)", "") for p in parts]
    return re.compile("|".join(f"(?:{p})" for p in stripped), re.IGNORECASE)


def find_exam_references(text: str, exams=DEFAULT_EXCLUDED_EXAMS) -> list[RuleHit]:
    """Detect references to excluded entrance exams.

    Three tiers:
      1. an exam whose mode is "any", mentioned anywhere;
      2. an exam whose mode is "rank_only", when its own bullet also carries a
         rank/score word;
      3. an AIR / All India Rank phrase whose bullet names an excluded exam
         (catches "AIR 993 in Mains 2024" without the word JEE).

    A rank phrase with no exam context is NOT returned here - see
    :func:`find_rank_mentions`.
    """
    hits: list[RuleHit] = []
    claimed: set[int] = set()

    for exam in exams:
        rule = EXAM_RULES.get(exam)
        if not rule:
            continue
        pattern, mode = rule
        for m in re.finditer(pattern, text):
            start, end = m.span()
            if any(i in claimed for i in range(start, end)):
                continue
            if mode == "rank_only":
                left, right = _segment_bounds(text, start, end)
                if not re.search(_SCORE_CONTEXT, text[left:right]):
                    continue    # a bare qualification mention is allowed
            claimed.update(range(start, end))
            hits.append(RuleHit("exam", exam, m.group(0).strip(),
                                _context(text, start, end)))

    context_re = _exam_context_re(exams)
    for m in re.finditer(_RANK_PHRASE, text):
        start, end = m.span()
        if any(i in claimed for i in range(start, end)):
            continue
        left, right = _segment_bounds(text, start, end)
        if context_re.search(text[left:right]):
            claimed.update(range(start, end))
            hits.append(RuleHit("exam", "rank_with_exam_context",
                                m.group(0).strip(), _context(text, start, end)))
    return hits


def find_jee_references(text: str) -> list[RuleHit]:
    """Backwards-compatible alias: JEE only."""
    return find_exam_references(text, ("JEE",))


def find_rank_mentions(text: str, exams=DEFAULT_EXCLUDED_EXAMS) -> list[RuleHit]:
    """AIR / All India Rank claims with NO excluded-exam context in their bullet.

    Informational only. These are legitimate achievements (olympiads, subject
    contests) and must never fail a resume, but they are surfaced so a human
    reviewer can eyeball anything the exam rule deliberately let through.
    """
    context_re = _exam_context_re(exams)
    mentions: list[RuleHit] = []
    for m in re.finditer(_RANK_PHRASE, text):
        start, end = m.span()
        left, right = _segment_bounds(text, start, end)
        if context_re.search(text[left:right]):
            continue
        mentions.append(RuleHit("rank_mention", "non_exam_rank", m.group(0).strip(),
                                _context(text, start, end)))
    return mentions


# --------------------------------------------------------------------------
# Metric harvesting
# --------------------------------------------------------------------------

# Ordered: earlier patterns win, so "25%" is a percentage and not a bare number.
_METRIC_PATTERNS: list[tuple[str, str]] = [
    ("percentage", r"[~≈<>]?\s?\d{1,3}(?:\.\d+)?\s?(?:%|percent(?:age)?\b)"),
    ("currency",   r"(?:[$₹€£]|\b(?:USD|INR|EUR|GBP|Rs)\.?\s?)\s?\d[\d,]*(?:\.\d+)?"
                   r"\s?(?:[KkMmBb]\b|lakhs?\b|crores?\b)?"),
    ("multiplier", r"\b\d+(?:\.\d+)?\s?[x×]\b"),
    ("duration",   r"\b\d+(?:\.\d+)?\+?\s?(?:hrs?|hours?|days?|weeks?|months?|mos?"
                   r"|years?|yrs?)\b"),
    ("scale",      r"\b\d[\d,]*(?:\.\d+)?\s?(?:[KkMmBb]|lakhs?|crores?)\+?(?![\w])"),
    ("rank",       r"(?i)\b(?:top|rank(?:ed)?|position|#)\s*#?\s*\d+(?:st|nd|rd|th)?\b"),
    ("ratio",      r"\b\d+(?:\.\d+)?\s?/\s?\d+(?:\.\d+)?\b"),
    # Unit-bearing measurements. Without this, "800ms" is invisible to the
    # scan because the "plain" pattern refuses to match digits glued to letters.
    ("measurement", r"\b\d+(?:\.\d+)?\s?(?:ms|us|ns|sec(?:onds?)?|min(?:utes?)?|[KMGT]B|[KMGT]iB|dB|fps|[qrQR]ps|QPS|RPS|LOC|bps|[kMG]?Hz|pp)\b"),
    ("plain",      r"(?<![\w.$₹€£])\d[\d,]*(?:\.\d+)?\+?(?![\w%])"),
]

_SCALE_WORDS = {
    "k": 1e3, "m": 1e6, "b": 1e9,
    "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
}

# Base unit is the MONTH, not the day, so that "12 months" and "1 year" compare
# exactly equal. Using days would make them 360 vs 365 and flag honest reunit-ing
# of the same duration as an unverified metric.
_DURATION_TO_MONTHS = {
    "hr": 1 / 720, "hrs": 1 / 720, "hour": 1 / 720, "hours": 1 / 720,
    "day": 1 / 30, "days": 1 / 30, "week": 7 / 30, "weeks": 7 / 30,
    "mo": 1, "mos": 1, "month": 1, "months": 1,
    "yr": 12, "yrs": 12, "year": 12, "years": 12,
}


@dataclass
class Metric:
    """A numeric claim lifted out of a resume."""

    raw: str            # exactly as it appears in the document
    kind: str           # percentage | currency | duration | ...
    value: float | None # normalised numeric value (for cross-document matching)
    unit: str           # normalised unit class, used as part of the match key
    context: str        # the sentence/bullet it came from

    @property
    def key(self) -> tuple[str, float | None]:
        return (self.unit, None if self.value is None else round(self.value, 6))

    @property
    def is_calendar_year(self) -> bool:
        return (self.kind == "plain" and self.value is not None
                and self.value.is_integer() and 1900 <= self.value <= 2099)


def _to_float(token: str) -> float | None:
    token = token.replace(",", "").strip()
    m = re.search(r"\d+(?:\.\d+)?", token)
    return float(m.group(0)) if m else None


def _normalise(raw: str, kind: str) -> tuple[float | None, str]:
    """Map a raw metric string to (value, unit-class) for comparison.

    Normalising means "40%" matches "40 percent", "$1.5M" matches "1500000 USD",
    and "12 months" matches "1 year" - so honest rewording is never flagged.
    """
    low = raw.lower().strip()
    number = _to_float(low)
    if number is None:
        return None, kind

    if kind == "percentage":
        return number, "percent"

    if kind == "currency":
        for word, mult in _SCALE_WORDS.items():
            if re.search(rf"{word}\b", low) or (word in "kmb" and re.search(rf"\d\s?{word}\b", low)):
                number *= mult
                break
        return number, "currency"

    if kind == "duration":
        unit_match = re.search(r"(hrs?|hours?|days?|weeks?|months?|mos?|years?|yrs?)", low)
        if unit_match:
            number *= _DURATION_TO_MONTHS.get(unit_match.group(1), 1)
        return number, "duration_months"

    if kind == "scale":
        for word, mult in _SCALE_WORDS.items():
            if re.search(rf"\d\s?{word}\b", low):
                number *= mult
                break
        return number, "count"

    if kind == "measurement":
        unit = re.sub(r"[\d.,\s]", "", low) or "unit"
        return number, f"measure_{unit}"
    if kind == "multiplier":
        return number, "multiplier"
    if kind == "rank":
        return number, "rank"
    if kind == "ratio":
        return None, "ratio"  # compared by literal string instead
    return number, "count"


def strip_contact_noise(text: str) -> str:
    """Blank out phone numbers and JEE/AIR ranks before metric harvesting.

    Those digits are already handled as hard exclusions. Leaving them in would
    (a) pad the master's match pool, letting a fabricated tailored metric match
    a phone fragment, and (b) re-report the phone number as an "unverified
    metric", duplicating a finding the user already has.
    """
    redacted = text
    for hit in find_phone_numbers(text) + find_exam_references(text):
        redacted = redacted.replace(hit.match, " " * len(hit.match))
    return redacted


def extract_metrics(text: str) -> list[Metric]:
    """Harvest every numeric claim in the document, longest-pattern-first."""
    metrics: list[Metric] = []
    claimed: set[int] = set()

    for kind, pattern in _METRIC_PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.span()
            if any(i in claimed for i in range(start, end)):
                continue
            raw = m.group(0).strip()
            if not any(ch.isdigit() for ch in raw):
                continue
            claimed.update(range(start, end))
            value, unit = _normalise(raw, kind)
            metrics.append(Metric(raw, kind, value, unit, _context(text, start, end, 70)))
    return metrics


def unmatched_metrics(
    tailored: list[Metric],
    master: list[Metric],
    include_years: bool = False,
) -> list[Metric]:
    """Tailored metrics with no numerically-equivalent counterpart in the master.

    Deliberately *conservative*: anything that matches on (unit, value) is
    dropped from the list, so the LLM only spends attention on real candidates.
    Calendar years are excluded by default - they are dates, not achievement
    metrics, and flagging every "2024" buries the signal in noise.
    """
    master_keys = {m.key for m in master if m.value is not None}
    master_literals = {re.sub(r"[\s,+~]", "", m.raw.lower()) for m in master}

    out: list[Metric] = []
    for metric in tailored:
        if metric.is_calendar_year and not include_years:
            continue
        if metric.value is not None and metric.key in master_keys:
            continue
        if re.sub(r"[\s,+~]", "", metric.raw.lower()) in master_literals:
            continue
        out.append(metric)
    return out

# --------------------------------------------------------------------------
# Spelled-out numbers
# --------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1_000,
    "million": 1_000_000, "billion": 1_000_000_000, "dozen": 12,
}

_NUMBER_WORD_RE = re.compile(
    r"(?i)\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b")


def word_number_metrics(text: str) -> list[Metric]:
    """Metrics written as words - "a team of five", "three internships".

    Added to the MASTER's matching pool only, never harvested as a claim from
    a tailored resume. It can only ever remove a false positive: without it,
    rewriting the master's "a team of five" as "5-person team" is reported as
    an unverified metric, which penalises exactly the rephrasing this tool
    permits. Harvesting these from the tailored side too would create the
    opposite problem, since "one of the" would become a numeric claim.
    """
    metrics: list[Metric] = []
    for m in _NUMBER_WORD_RE.finditer(text):
        word = m.group(1).lower()
        metrics.append(Metric(
            raw=m.group(1), kind="wordnum", value=float(_NUMBER_WORDS[word]),
            unit="count", context=_context(text, m.start(), m.end(), 70)))
    return metrics
