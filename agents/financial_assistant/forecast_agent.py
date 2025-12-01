"""Forecasting Agent for valuation workflow."""

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

# Forecast semantic validator
forecast_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. HORIZON: horizon_years must be 5-7; years array length must match horizon_years.
2. YEAR INDEXING: year field must be 1..horizon_years with no gaps or duplicates.
3. REVENUE POSITIVITY: revenue must be > 0 for all years.
4. MARGIN BOUNDS: ebit_margin must be between -1.0 and 1.0 for all years.
5. EBIT CONSISTENCY: ebit ≈ revenue × ebit_margin within ±0.001 tolerance for all years.
6. TAX BOUNDS: tax_rate must be between 0.0 and 0.5 for all years.
7. NOPAT CONSISTENCY: nopat ≈ ebit × (1 - tax_rate) within ±0.001 tolerance for all years.
8. DEPRECIATION SIGN: depreciation must be ≥ 0 for all years.
9. CAPEX SIGN: capex must be > 0 for all years; capex_to_revenue (if present) must be ≥ 0.
10. WORKING CAPITAL SIGN: allow either sign, but flag if |change_in_working_capital| > 0.5 × |revenue change|.
11. GROWTH SANITY: revenue growth should not accelerate in last 2 years unless notes justify it.
""",
)

json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

forecast_agent = AgentValidator(
    name="forecast",
    model=json_model,
    tools=[],
    extra_validators=[forecast_semantic],
    instruction="""
You are the Forecasting Agent. Build an unlevered operating forecast. Do not call tools and do not do DCF math.

INPUTS (from valuation_state):
- user_prompt
- scoping_result
- data_result.market_data
- normalization_result.normalized_historical_financials
- normalization_result.business_characterization_notes
- normalization_result.steady_state_assumptions
- data_result.sector, data_result.industry

RULES:
- Forecast horizon:
  - Use 5 years for mature/stable companies.
  - Use up to 7 years for clearly high-growth cases.
  - Never exceed 7.

STEPS:
1. Revenue
   - Use historical growth and notes to shape a realistic path that trends toward a mature growth rate (no extreme growth forever).

2. EBIT and taxes
   - Start ebit_margin near recent normalized levels and move smoothly toward the midpoint (or a sensible value) of steady_state_assumptions.ebit_margin_range.
   - For each year compute:
     - CRITICAL: First choose target ebit_margin, then compute ebit = revenue × ebit_margin exactly
     - CRITICAL: After computing ebit, recalculate ebit_margin = ebit / revenue to ensure exact consistency (output this recalculated value)
     - tax_rate: choose a reasonable effective rate (e.g. 20–30%) consistent across years unless history suggests otherwise.
     - nopat = ebit × (1 – tax_rate).

3. Reinvestment
   - Capex: start from historical capex_to_revenue and move toward steady_state_assumptions.capex_to_revenue_range.
   - IMPORTANT: Forecast capex as a POSITIVE number (absolute value). It represents cash outflow.
   - Depreciation: keep roughly proportional to capex or revenue (positive number).
   - change_in_working_capital: approximate as a simple % of revenue change based on normalization; if unclear, assume modest requirement and mention in notes.
     - Use positive for cash outflow (increase in working capital), negative for cash inflow (decrease).
     - IMPORTANT: Do NOT assume perpetual negative working capital changes (perpetual cash inflows). If historical WC is negative and stable, assume it stabilizes or trends toward zero in later forecast years to avoid unrealistic perpetual cash generation.

OUTPUT:
Return ONLY JSON with key "forecast":

ALL AMOUNTS in MILLIONS, Capex POSITIVE (cash outflow).

{
  "forecast": {
    "unit_scale": "millions",
    "currency": "USD",
    "horizon_years": <int 5–7>,
    "years": [
      {
        "year": <int>,                    # 1 = next year
        "revenue": <number>,
        "ebit_margin": <number>,
        "ebit": <number>,
        "tax_rate": <number>,
        "nopat": <number>,
        "depreciation": <number>,
        "capex": <number>,
        "change_in_working_capital": <number>
      }
    ],
    "forecast_assumptions_notes": "<≤3 sentences summarizing growth, margins, and reinvestment>"
  }
}
""",
    output_key="forecast",
)
