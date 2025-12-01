"""WACC & Capital Structure Agent for valuation workflow."""

from google.genai import types
from google.adk.models import Gemini
from .agent_validator import AgentValidator, ExtraValidatorSpec
from .eodhd_mcp import eodHistoricalData

# Retry configuration for Gemini API
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Model selection
FLASH_MODEL = "gemini-2.5-flash"

# WACC semantic validator
wacc_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. BOUNDS: cost_of_equity, cost_of_debt, wacc must each be between 0.0 and 0.5.
2. TERMINAL GROWTH BOUNDS: terminal_growth_rate must be between 0.0 and 0.06.
3. CRITICAL INEQUALITY: wacc must be > terminal_growth_rate by at least 0.005.
4. WEIGHTS VALIDITY: If equity_weight and debt_weight present, each must be 0.0-1.0 and sum to 1.0 within ±0.01.
5. WACC CONSISTENCY: If weights present, verify wacc ≈ equity_weight × cost_of_equity + debt_weight × cost_of_debt × (1 - tax_rate) within ±0.005.
6. CURRENCY SCALE: Must include unit_scale and currency fields.
""",
)

model = Gemini(model=FLASH_MODEL, retry_options=retry_config)

wacc_agent = AgentValidator(
    name="wacc",
    model=model,
    tools=[eodHistoricalData],
    extra_validators=[wacc_semantic],
    instruction="""
You are the WACC & Capital Structure Agent. Use tools only to fetch missing data (macro indicators, price, fundamentals). Do not do full valuation here.

TOOLS (via eodHistoricalData):
- get_macro_indicator
- get_live_price_data or get_us_live_extended_quotes
- get_fundamentals_data

INPUTS (from valuation_state):
- scoping_result
- data_result.market_data
- normalization_result.normalized_historical_financials
- forecast

STEPS:
1. Capital structure
   - Equity value: use market_data.market_cap if available; otherwise approximate from price × shares_outstanding.
   - Debt: use latest total_debt (book value is acceptable).
   - Cash: last available cash_and_equivalents.
   - Compute equity_weight and debt_weight where possible; if leverage is clearly low, note that and treat WACC ≈ cost of equity.

2. Cost of equity (r_e)
   - Use a CAPM-like approach conceptually: risk_free_rate + equity_risk_premium × typical beta.
   - Use get_macro_indicator for risk-free or inflation if needed.
   - If concrete inputs missing, choose a reasonable estimate based on company profile:
     - Mega-cap, stable, low-leverage (e.g., Apple, Microsoft): 7-9%
     - Large-cap, moderate risk: 9-11%
     - High-growth or higher risk: 11-14%
   - Be mindful that cost of equity directly impacts valuation: higher r_e → lower value.
   - Explain your choice briefly.

3. Cost of debt (r_d)
   - Infer from history (interest expense vs debt) if visible; else from company risk and macro rates.
   - Pick a plausible rate (e.g. 3–8%) and explain briefly.

4. WACC and terminal growth
   - Use a tax_rate consistent with forecast (e.g. its average).
   - WACC = E/(D+E) * r_e + D/(D+E) * r_d * (1 – tax_rate); if D very small, WACC ≈ r_e.
   - Choose a long-term terminal_growth_rate below long-run nominal GDP growth:
     - Typical range: 2-3% nominal (reflects inflation ~2% plus modest real growth ~0-1%)
     - For mature mega-caps, often 2.0-2.5%; for slower-growth, may be lower
     - IMPORTANT: State whether this is in nominal or real terms, and be consistent with WACC (which should be nominal)
     - Justify in 1–2 sentences.

OUTPUT REQUIREMENTS:
CRITICAL: Your response MUST be ONLY the JSON object below. Do NOT include any markdown formatting, explanations, or text before or after the JSON. Do NOT write things like "Here are the capital assumptions..." or "Based on the analysis...". ONLY output the raw JSON structure.

{
  "capital_assumptions": {
    "unit_scale": "millions",
    "currency": "USD",
    "cost_of_equity": <number>,        # e.g. 0.09
    "cost_of_debt": <number>,          # e.g. 0.04
    "equity_weight": <number or null>,
    "debt_weight": <number or null>,
    "wacc": <number>,
    "terminal_growth_rate": <number>,
    "capital_assumptions_notes": "<≤3 sentences summarizing data gaps and choices>"
  }
}
""",
    output_key="capital_assumptions",
)
