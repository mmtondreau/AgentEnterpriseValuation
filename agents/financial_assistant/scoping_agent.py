"""Scoping & Clarification Agent for valuation workflow."""

from google.genai import types
from .agent_validator import AgentValidator, ExtraValidatorSpec

# Retry configuration for Gemini API
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Model selection
FLASH_MODEL = "gemini-2.5-flash"

# Scoping semantic validator
scoping_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. ALLOWED ENUMS: valuation_target must be "enterprise_value" or "equity_per_share"; control_perspective must be "control" or "minority".
2. DATE FORMAT: as_of_date must be "today" or ISO date format (YYYY-MM-DD).
3. CURRENCY FORMAT: currency must be a valid 3-letter currency code (e.g., USD, EUR, GBP).
""",
)

# For agents that need strict JSON output, create a separate model instance
# with JSON mode enabled (agents with tools cannot use JSON mode)
from google.adk.models import Gemini

json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

scoping_agent = AgentValidator(
    name="scoping",
    model=json_model,
    tools=[],
    extra_validators=[scoping_semantic],
    instruction="""
You are the Scoping & Clarification Agent in a valuation workflow.

Goal: Turn the user's natural language request into a compact scoping object. Do not call tools.

INPUTS:
- user_prompt: original user request.
- valuation_state: JSON of current state (may be empty).

STEPS (be deterministic, no commentary):
1. company_identifier
   - If a ticker is clearly given, use it (e.g. "AAPL.US").
   - Else use the company name string as given.

2. valuation_target
   - "enterprise_value" for value of operations.
   - "equity_per_share" for per-share equity value.
   - Default: "equity_per_share" if unclear.

3. as_of_date and currency
   - If a date is mentioned, use ISO "YYYY-MM-DD".
   - Else set "today" (exact date will be filled later).
   - Currency: use if specified (e.g. "USD"), else "USD".

4. context
   - control_perspective: "control" if user speaks as acquirer/buyer; else "minority".
   - holding_period_or_style: short phrase if horizon/style is explicit (e.g. "long-term investor"); else null.
   - additional_context_notes: brief free-text for any explicit constraints or preferences (e.g. conservative, downside focus).

OUTPUT:
Return ONLY a JSON object with top-level key "scoping_result" and fields:

{
  "scoping_result": {
    "company_identifier": "<string>",
    "valuation_target": "enterprise_value" or "equity_per_share",
    "as_of_date": "<string>",            # "today" or "YYYY-MM-DD"
    "currency": "<string>",              # e.g. "USD"
    "control_perspective": "minority" or "control",
    "holding_period_or_style": "<string or null>",
    "additional_context_notes": "<string>"
  }
}
""",
    output_key="scoping_result",
)
