"""The contract between the app and the LLM.

``REPORT_JSON_SCHEMA`` is hand-written and fully inlined (no ``$ref`` / ``$defs``)
because that is what the Messages API's ``output_config.format`` and OpenAI's
``json_schema`` response format both accept without surprises. The Pydantic
models below validate whatever comes back, so a malformed response fails loudly
instead of silently producing an empty "all clear".
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

CATEGORIES = [
    "UNVERIFIED_METRIC",      # a number in the tailored resume that isn't in the master
    "ALTERED_METRIC",         # exists in the master but the value was changed
    "EXAGGERATED_METRIC",     # rounded up / inflated relative to the master
    "DERIVED_METRIC",         # computed from master data but never stated there
    "HALLUCINATED_SKILL",     # skill with no basis in the master
    "HALLUCINATED_EXPERIENCE",# company / project / role not in the master
    "ALTERED_TITLE",          # job title upgraded or changed
    "TIMELINE_MISMATCH",      # dates/durations contradict the master
    "OVERSTATED_SCOPE",       # "led" vs "contributed to", sole vs team ownership
    "OTHER",
]

SEVERITIES = ["high", "medium", "low"]


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


# --------------------------------------------------------------------------
# Pydantic models (validation + typed access in the app)
# --------------------------------------------------------------------------


class Finding(BaseModel):
    category: str = Field(description="One of CATEGORIES")
    severity: str = Field(description="high | medium | low")
    quote: str = Field(description="Exact text copied from the tailored resume")
    issue: str = Field(description="What is wrong, in one sentence")
    master_evidence: str = Field(
        description="What the master resume actually says, or NOT FOUND IN MASTER"
    )
    suggested_fix: str = Field(description="Concrete rewrite or removal advice")

    @property
    def is_blocking(self) -> bool:
        return self.severity == "high"


class ScoreLine(BaseModel):
    """One itemised deduction from the match score."""

    label: str
    delta: int = 0


class LLMReport(BaseModel):
    """Exactly what the model is asked to return."""

    overall_assessment: str = ""
    findings: list[Finding] = Field(default_factory=list)
    verified_ok: list[str] = Field(
        default_factory=list,
        description="Claims that were checked and confirmed against the master",
    )


class ResumeReport(BaseModel):
    """The merged result: deterministic checks + LLM findings + verdict."""

    filename: str
    status: Status = Status.PASS
    identity_match: bool | None = None
    identity_reason: str = ""
    match_score: int = 100
    score_band: str = ""
    score_lines: list[ScoreLine] = Field(default_factory=list)
    score_provisional: bool = False
    total_metrics: int = 0
    page_count: int = 0
    within_page_limit: bool = True
    page_limit: int = 2
    has_phone: bool = False
    has_jee: bool = False
    phone_hits: list[str] = Field(default_factory=list)
    jee_hits: list[str] = Field(default_factory=list)
    rank_mentions: list[str] = Field(
        default_factory=list,
        description="Non-JEE All India Rank claims - informational, never a failure",
    )
    findings: list[Finding] = Field(default_factory=list)
    verified_ok: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
    unverified_metric_candidates: list[str] = Field(default_factory=list)
    unsupported_lines: list[str] = Field(
        default_factory=list,
        description="Lines with no textual counterpart in the master resume",
    )
    rephrased_lines: int = 0
    coverage: float = 1.0
    has_highlights: bool = False
    # Annotated PDF bytes - excluded from JSON export, shown in the viewer.
    annotated_pdf: bytes | None = Field(default=None, exclude=True, repr=False)
    parse_warnings: list[str] = Field(default_factory=list)
    error: str = ""
    llm_used: bool = False

    @property
    def blocking_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity in ("medium", "low"))

    def headline(self) -> str:
        bits: list[str] = []
        if self.has_phone:
            bits.append("phone number present")
        if self.has_jee:
            bits.append("JEE rank mentioned")
        if not self.within_page_limit:
            bits.append(f"{self.page_count} pages (limit {self.page_limit})")
        if self.blocking_count:
            bits.append(f"{self.blocking_count} critical content issue(s)")
        if self.warning_count:
            bits.append(f"{self.warning_count} warning(s)")
        if self.unsupported_lines and not self.llm_used:
            bits.append(f"{len(self.unsupported_lines)} line(s) need review")
        return "; ".join(bits) if bits else "No issues detected"


# --------------------------------------------------------------------------
# Raw JSON schema handed to the API
# --------------------------------------------------------------------------

REPORT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "overall_assessment": {
            "type": "string",
            "description": "2-3 sentence summary of how faithful the tailored "
                           "resume is to the master resume.",
        },
        "findings": {
            "type": "array",
            "description": "Every discrepancy found. Empty array if the tailored "
                           "resume is fully supported by the master resume.",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "quote": {
                        "type": "string",
                        "description": "Exact substring copied from the tailored "
                                       "resume - never paraphrased.",
                    },
                    "issue": {"type": "string"},
                    "master_evidence": {
                        "type": "string",
                        "description": "Quote from the master resume that supports "
                                       "or contradicts the claim, or the literal "
                                       "string 'NOT FOUND IN MASTER'.",
                    },
                    "suggested_fix": {"type": "string"},
                },
                "required": ["category", "severity", "quote", "issue",
                             "master_evidence", "suggested_fix"],
                "additionalProperties": False,
            },
        },
        "verified_ok": {
            "type": "array",
            "description": "Short notes on rephrased claims that were checked and "
                           "confirmed as faithful to the master resume.",
            "items": {"type": "string"},
        },
    },
    "required": ["overall_assessment", "findings", "verified_ok"],
    "additionalProperties": False,
}
