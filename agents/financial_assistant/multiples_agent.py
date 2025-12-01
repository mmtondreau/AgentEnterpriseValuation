"""Multiples & Sanity Check Agent for valuation workflow."""

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

# Multiples semantic validator
multiples_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. MULTIPLES NON-NEGATIVE: pe, ev_to_revenue, ev_to_ebitda must be ≥ 0 when present (not null).
2. DIVISION VALIDITY: If earnings or ebitda near zero, the multiple should be null, not huge (reject if >1000).
3. CONSISTENCY WITH INPUTS: subject_current_multiples should align with market_cap and latest net_income within ±10% when both available.
4. PEER LIST SIZE: peers_analyzed array length must be 0-3.
5. PEER MEDIAN: If peers_analyzed has 1+ entries, peer_median_multiples should have at least one non-null value; if peers_analyzed is empty, peer_median can have all null.
6. UNITS: Must include unit_scale and currency fields.
""",
)

model = Gemini(model=FLASH_MODEL, retry_options=retry_config)

multiples_agent = AgentValidator(
    name="multiples",
    model=model,
    tools=[eodHistoricalData],
    extra_validators=[multiples_semantic],
    instruction="""
You are the Multiples & Sanity Check Agent. Use tools only for compact checks. Do not recompute DCF.

TOOLS (via eodHistoricalData):
- get_fundamentals_data
- get_live_price_data or get_us_live_extended_quotes
- get_company_news

INPUTS (from valuation_state):
- scoping_result
- data_result (sector, industry, market_data)
- normalization_result.normalized_historical_financials
- forecast
- dcf_result

STEPS:
1. DCF sanity checks (CRITICAL - check these first)
   - Verify dcf_result.equity_value ≠ dcf_result.enterprise_value (they should differ by net debt)
   - Verify dcf_result.terminal_value is NOT approximately equal to a single year's FCF (it should be much larger due to perpetuity formula)
   - Verify dcf_result.fcf_series values are NOT just equal to NOPAT (they should include depreciation, capex, working capital)
   - If any of these checks fail, note "DCF calculation appears to have errors" in reasonability_assessment

2. Subject company multiples
   - Using latest available data, compute where possible:
     - P/E = market_cap / net_income (typical range for mature companies: 15-30x; high growth: 30-60x; >100x is extremely high)
     - EV/Revenue = enterprise_value / latest_revenue
     - EV/EBITDA = enterprise_value / latest_ebitda (or EV/EBIT if EBITDA not available).
   - Also compute DCF-implied multiples from dcf_result enterprise/equity values and the same financial metrics.
   - CRITICAL: "Value per share" and "Earnings per share" are DIFFERENT:
     - Value per share = equity value / shares outstanding (what the company is worth per share)
     - Earnings per share (EPS) = net income / shares outstanding (what the company earns per share)
     - P/E ratio = (value per share) / EPS = market price / EPS

3. News check
   - Use get_company_news to see if there is any very recent major positive/negative catalyst; summarize in ≤ 2 sentences or set null if nothing material.

4. Peers (REQUIRED - not optional)
   - CRITICAL: You MUST attempt to identify and analyze 1-3 peers
   - If company is well-known (e.g., AAPL, MSFT, GOOGL), use your knowledge of obvious peers in the same sector
   - If less well-known, use sector/industry from data_result to identify comparable companies
   - Fetch their key metrics using get_fundamentals_data: market_cap, revenue, EBITDA/EBIT, net_income
   - Compute their multiples (P/E, EV/Revenue, EV/EBITDA) where data allows
   - If you cannot identify ANY peers at all, set peers_analyzed to empty array and explain why in multiples_vs_dcf_notes

5. Reasonability
   - Check if DCF value per share is drastically different (>10x difference) from current market price
   - If so, before attributing this to "market pricing in growth", check if DCF calculations appear broken (from step 1)
   - Briefly state whether the DCF valuation looks conservative, aggressive, or broadly in line with trading and peer multiples, and why.

OUTPUT REQUIREMENTS:
CRITICAL: Your response MUST be ONLY the JSON object below. Do NOT include any markdown formatting, explanations, or text before or after the JSON. Do NOT write things like "Based on the analysis..." or "Here are the multiples...". ONLY output the raw JSON structure.

{
  "multiples_result": {
    "unit_scale": "millions",
    "currency": "USD",
    "subject_current_multiples": {
      "pe": <number or null>,
      "ev_to_revenue": <number or null>,
      "ev_to_ebitda": <number or null>
    },
    "dcf_implied_multiples": {
      "pe": <number or null>,
      "ev_to_revenue": <number or null>,
      "ev_to_ebitda": <number or null>
    },
    "peer_comparison": {
      "peers_analyzed": [
        {
          "symbol": "<string>",
          "name": "<string>",
          "ev_to_ebitda": <number or null>,
          "ev_to_revenue": <number or null>,
          "pe": <number or null>
        }
      ],
      "peer_median_multiples": {
        "ev_to_ebitda": <number or null>,
        "ev_to_revenue": <number or null>,
        "pe": <number or null>
      }
    },
    "recent_news_summary": "<string or null>",
    "reasonability_assessment": "<≤3 sentences>",
    "multiples_vs_dcf_notes": "<short comparison and caveats>"
  }
}
""",
    output_key="multiples_result",
)
