"""Provider adapters. Each returns a validated :class:`LLMReport`.

WHERE TO PUT YOUR API KEY
-------------------------
Any one of these works (checked in this order by ``app.py``):

  1. Type it into the sidebar field in the running app (never persisted).
  2. Export it in your shell:      export ANTHROPIC_API_KEY=sk-ant-...
  3. Put it in .streamlit/secrets.toml:
         ANTHROPIC_API_KEY = "sk-ant-..."

Nothing in this repo writes a key to disk.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .prompts import SYSTEM_RULES, build_master_block, build_user_prompt
from .rules import Metric
from .schema import REPORT_JSON_SCHEMA, LLMReport

# Model choices surfaced in the UI. Opus 5 is the default: this task is a
# careful multi-constraint comparison where accuracy matters more than latency.
ANTHROPIC_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5",
]

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]

EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating stray prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON object in model response.")


class BaseVerifier(ABC):
    """Common interface so app.py never branches on provider."""

    def __init__(self, api_key: str, model: str, master_text: str, effort: str = "high"):
        self.api_key = api_key
        self.model = model
        self.master_block = build_master_block(master_text)
        self.effort = effort

    @abstractmethod
    def verify(self, tailored_text: str, unmatched: list[Metric], filename: str,
               unsupported_lines: list[str] | None = None) -> LLMReport:
        ...


class AnthropicVerifier(BaseVerifier):
    """Claude via the Messages API.

    Uses three things that matter for this workload:
      * adaptive thinking - the comparison is multi-constraint reasoning
      * output_config.format - guarantees schema-valid JSON, no repair parsing
      * cache_control on the master resume - the master is re-sent for every
        resume in the batch, so caching it is a large, free saving
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key or None)

    def _system_blocks(self) -> list[dict]:
        return [
            {"type": "text", "text": SYSTEM_RULES},
            {
                "type": "text",
                "text": self.master_block,
                # Everything before this point is identical across the batch,
                # so every resume after the first reads it from cache.
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def verify(self, tailored_text: str, unmatched: list[Metric], filename: str,
               unsupported_lines: list[str] | None = None) -> LLMReport:
        user_prompt = build_user_prompt(tailored_text, unmatched, filename,
                                        unsupported_lines)

        # Progressive degradation: each variant drops one optional feature, so
        # an older SDK or a model that rejects a parameter still produces a
        # usable audit instead of an error card.
        variants: list[dict] = [
            {
                "thinking": {"type": "adaptive"},
                "output_config": {
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": REPORT_JSON_SCHEMA},
                },
            },
            {
                "output_config": {
                    "format": {"type": "json_schema", "schema": REPORT_JSON_SCHEMA},
                },
            },
            {},  # plain call - JSON contract enforced by the prompt alone
        ]

        last_error: Exception | None = None
        for i, extra in enumerate(variants):
            prompt = user_prompt
            if "output_config" not in extra:
                prompt += (
                    "\n\nReturn ONLY a JSON object matching this schema, with no "
                    "surrounding prose:\n"
                    + json.dumps(REPORT_JSON_SCHEMA, indent=2)
                )
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=self._system_blocks(),
                    messages=[{"role": "user", "content": prompt}],
                    **extra,
                )
                if getattr(response, "stop_reason", None) == "refusal":
                    raise RuntimeError("Model declined to process this document.")

                texts = [b.text for b in response.content if b.type == "text"]
                if not texts:
                    raise ValueError("Model returned no text content.")
                return LLMReport.model_validate(_extract_json(texts[-1]))

            except self._anthropic.BadRequestError as exc:
                # Almost always an unsupported parameter for this model/SDK -
                # fall through to the next, simpler variant.
                last_error = exc
                if i == len(variants) - 1:
                    raise
            except (ValueError, TypeError) as exc:
                last_error = exc
                if i == len(variants) - 1:
                    raise

        raise last_error or RuntimeError("Verification failed.")


class OpenAIVerifier(BaseVerifier):
    """OpenAI chat-completions alternative, kept deliberately simple."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key or None)

    def verify(self, tailored_text: str, unmatched: list[Metric], filename: str,
               unsupported_lines: list[str] | None = None) -> LLMReport:
        prompt = build_user_prompt(tailored_text, unmatched, filename,
                                   unsupported_lines) + (
            "\n\nReturn ONLY a JSON object matching this schema:\n"
            + json.dumps(REPORT_JSON_SCHEMA, indent=2)
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_RULES + "\n\n" + self.master_block},
                {"role": "user", "content": prompt},
            ],
        )
        return LLMReport.model_validate(_extract_json(response.choices[0].message.content))


def build_verifier(provider: str, api_key: str, model: str,
                   master_text: str, effort: str = "high") -> BaseVerifier:
    if provider == "Anthropic (Claude)":
        return AnthropicVerifier(api_key, model, master_text, effort)
    if provider == "OpenAI":
        return OpenAIVerifier(api_key, model, master_text, effort)
    raise ValueError(f"Unknown provider: {provider}")
