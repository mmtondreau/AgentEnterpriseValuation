"""Normalization & Business Understanding Agent for valuation workflow."""

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

# Normalization semantic validator spec
normalization_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. CAPEX SIGN: capex must be positive and capex_to_revenue must be non-negative.
2. MARGIN CONSISTENCY: ebit_margin must equal ebit divided by revenue within tolerance (±0.001).
3. RATIO CONSISTENCY: capex_to_revenue must equal capex divided by revenue within tolerance (±0.001).
5. UNIT SCALE: Must include "unit_scale": "millions" and "currency": "USD".
""",
)

json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)

normalization_agent = AgentValidator(
    name="normalization",
    model=json_model,
    tools=[],
    extra_validators=[normalization_semantic],
    instruction="""
    You are the Normalization & Business Understanding Agent. Do not call tools.

    INPUTS (from valuation_state):
    - user_prompt
    - scoping_result
    - data_result.historical_financials_normalized
    - data_result.market_data
    - data_result.sector, data_result.industry

    GOALS:
    1) Compute simple derived metrics.
    2) Describe business trends briefly.
    3) Propose rough steady-state margin/reinvestment ranges.

    STEPS:
    1. For each historical year (last 3–5 years):
    - Compute when possible:
        - revenue_growth vs prior year,
        - ebit_margin = ebit / revenue,
        - net_margin = net_income / revenue,
        - cfo_margin = cfo / revenue,
        - capex_to_revenue = abs(capex) / revenue (MUST be positive).
    - IMPORTANT: Ensure capex and capex_to_revenue are POSITIVE numbers. If capex is negative, take its absolute value.
    - Do NOT invent missing numbers; leave null if unavailable.

    2. Business characterization
    - From the time series infer if revenue is growing/stable/shrinking, margins expanding/stable/compressing, capex high/medium/low vs revenue, and whether volatility is high.
    - Be precise about trends: if a metric changes direction (e.g., working capital becomes less negative in the last year), note that explicitly.
    - Summarize in ≤ 2 sentences.

    3. Steady-state assumptions
    - Propose approximate ranges (low, high) for:
        - ebit_margin_range: where high > low (e.g. [0.28, 0.32])
        - capex_to_revenue_range: POSITIVE ratios where high > low (e.g. [0.02, 0.03])
    - Use history as anchor; if unclear, use null and explain briefly.
    - Add a short note on working_capital_intensity based on working_capital and revenue, or state that it is unclear.

    OUTPUT:
    Return ONLY JSON with key "normalization_result":

    ALL AMOUNTS in MILLIONS, Capex POSITIVE.

    {
    "normalization_result": {
        "unit_scale": "millions",
        "currency": "USD",
        "normalized_historical_financials": {
        "years": [
            {
            "year": <int>,
            "revenue": <number>,
            "revenue_growth": <number or null>,
            "ebit": <number or null>,
            "ebit_margin": <number or null>,
            "net_income": <number or null>,
            "net_margin": <number or null>,
            "cfo": <number or null>,
            "cfo_margin": <number or null>,
            "capex": <number or null>,
            "capex_to_revenue": <number or null>,
            "depreciation": <number or null>,
            "total_debt": <number or null>,
            "cash_and_equivalents": <number or null>,
            "working_capital": <number or null>
            }
        ]
        },
        "business_characterization_notes": "<≤2 sentences>",
        "steady_state_assumptions": {
        "ebit_margin_range": [<low or null>, <high or null>],
        "capex_to_revenue_range": [<low or null>, <high or null>],
        "working_capital_intensity_notes": "<string>"
        }
    }
    }
    """,
    output_key="normalized_result",
)
