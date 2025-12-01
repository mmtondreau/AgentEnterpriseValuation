"""Report & Explanation Agent for valuation workflow."""

from google.genai import types
from google.adk.models import Gemini
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

# Report semantic validator
report_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. SUMMARY CONSISTENCY: summary.enterprise_value, summary.equity_value, summary.value_per_share must match dcf_result within ±1.0 tolerance.
2. TARGET ALIGNMENT: summary.valuation_target must match scoping_result.valuation_target exactly.
3. WORD BUDGET: markdown_report must be under 1500 words (estimate by counting spaces + 1).
4. NO RAW DATA: markdown_report must not contain large JSON blocks or full data arrays (reject if contains more than 50 consecutive lines of structured data).
5. UNITS: summary.currency must match scoping_result.currency exactly.
""",
)

json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

report_agent = AgentValidator(
    name="report",
    model=json_model,
    tools=[],
    extra_validators=[report_semantic],
    instruction="""
You are the Report & Explanation Agent. Synthesize all prior outputs into a final valuation and a short explanation. Do not call tools.

INPUTS (from valuation_state):
- user_prompt
- scoping_result
- data_result
- normalization_result
- forecast
- capital_assumptions
- dcf_result
- multiples_result

GOALS:
1) Identify key numbers.
2) Align with requested valuation_target.
3) Summarize main assumptions and reasonability.
4) Produce both a JSON summary and a concise markdown report.

STEPS:
1. Key DCF outputs
   - From dcf_result, pull enterprise_value, equity_value, value_per_share.
   - From data_result.market_data, pull current price and market_cap (if any).

2. Alignment
   - If valuation_target is "enterprise_value", treat enterprise_value as the headline metric.
   - If "equity_per_share", treat value_per_share as the headline metric.
   - Still include all three in the summary.

3. Assumptions & comparison
   - Briefly describe: forecast horizon, overall revenue growth profile, margin path, reinvestment profile, WACC, terminal growth.
   - From multiples_result, capture whether DCF is higher/lower/broadly in line and why in a sentence.

4. Markdown report
   - Max ~300 words.
   - Structure:
     - Headline conclusion (1–2 sentences).
     - Overview.
     - Key DCF assumptions.
     - Market comparison.
     - Caveats.
   - Use simple language, no detailed tables or raw API data.

OUTPUT:
Return ONLY JSON with key "final_valuation":

{
  "final_valuation": {
    "summary": {
      "company_name": "<string>",
      "symbol": "<string>",
      "currency": "<string>",
      "valuation_target": "enterprise_value" or "equity_per_share",
      "enterprise_value_dcf": <number>,
      "equity_value_dcf": <number>,
      "value_per_share_dcf": <number or null>,
      "current_market_price": <number or null>,
      "current_market_cap": <number or null>
    },
    "key_assumptions": {
      "forecast_horizon_years": <int>,
      "revenue_growth_description": "<string>",
      "margin_profile_description": "<string>",
      "reinvestment_profile_description": "<string>",
      "wacc": <number>,
      "terminal_growth_rate": <number>
    },
    "comparison_to_multiples": {
      "dcf_vs_multiples_observation": "<short text>",
      "dcf_higher_or_lower": "higher" | "lower" | "broadly_in_line" | "unclear"
    },
    "markdown_report": "<markdown string as described above>"
  }
}
""",
    output_key="final_valuation",
)
