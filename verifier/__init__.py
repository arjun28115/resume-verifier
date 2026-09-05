"""Smart Resume Verification Tool - core package.

Modules
-------
pdf_utils : text + page-count extraction from PDF/TXT uploads
rules     : deterministic (regex) scanners - phone, JEE rank, metric harvesting
prompts   : the LLM system/user prompts used for cross-referencing
schema    : the JSON contract the LLM must return, plus Pydantic validation
llm       : provider adapters (Anthropic / OpenAI) returning a validated report
scoring   : the 0-100 Match Score and its itemised deductions
engine    : orchestration - parse -> rule scan -> LLM -> merge -> status
"""

from .schema import Finding, ResumeReport, ScoreLine, Status  # noqa: F401
from .scoring import compute_match_score  # noqa: F401
