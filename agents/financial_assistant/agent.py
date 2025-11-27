"""Financial Assistant Agent using EODHD MCP Server for market data."""

from google.adk.agents import SequentialAgent
from google.adk.models.google_llm import Gemini
from .eodhd_mcp import eodHistoricalData
from .agent_validator import AgentValidator, ExtraValidatorSpec
from google.genai import types

# Retry configuration for Gemini API
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Model selection: Flash has 2M context window vs Flash Lite's 1M
# Use Flash to handle the large context accumulation in SequentialAgent
FLASH_MODEL = "gemini-2.5-flash"
LITE_MODEL = "gemini-2.5-flash-lite"  # Kept for reference


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    if callback_context._invocation_context.memory_service:
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )


# # Define the root agent for ADK Web UI
# root_agent = Agent(
#     name="financial_assistant",
#     model=Gemini(model=FLASH_MODEL, retry_options=retry_config),
#     instruction="""
#     You are a financial assistant. Based on the user prompt use the eodHistoricalData get_fundamentals_data
#     tool to gather information about fundamentals. IMPORTANT: When calling get_fundamentals_data, you MUST use the from_date
#     parameter to limit data to only the last 2 years to avoid exceeding token limits. For example, use from_date="2023-01-01".
#     Also gather company news using get_company_news and current price data. Provide concise and accurate information to help
#     users make informed financial decisions. Make sure to include the current timestamp in your analysis that this report was
#     generated if it was just generated (e.g. not from memory).

#     You have access to load_memory tool to retrieve relevant past analysis from long-term memory. If you see the analysis is
#     recently done (within last 24 hours), you can reference it in your final response. Make sure to indicate the time of the
#     original analysis. We should be checking against the original analysis timestamp not the timestamp of the session in which it is
#     stored.
#     """,
#     tools=[eodHistoricalData, load_memory],
#     output_key="final_response",
#     after_agent_callback=auto_save_to_memory,
# )


from google.adk.models import Gemini

# Use Gemini 2.5 Flash (2M context) instead of Flash Lite (1M context)
# to handle large context accumulation in SequentialAgent with financial data
model = Gemini(model=FLASH_MODEL, retry_options=retry_config)

# For agents that need strict JSON output, create a separate model instance
# with JSON mode enabled (agents with tools cannot use JSON mode)
json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
)


# ============================================================================
# Extra Validator Specifications for Each Agent Stage
# ============================================================================

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

# Data semantic validator
data_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. SYMBOL PRESENT: resolved_symbol must be non-empty and plausible (not null or "").
2. MARKET CAP CONSISTENCY: If market_cap, price, and shares_outstanding all present, verify market_cap ≈ price × shares_outstanding within ±10% tolerance.
3. FINANCIAL YEAR ORDERING: years array must have strictly increasing year values and length 3-5.
4. MARGIN CONSISTENCY: If ebit_margin present in any year, verify ebit_margin ≈ ebit / revenue within ±0.001 tolerance.
5. UNITS: Must include "unit_scale": "millions" and "currency" field.
""",
)

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

# DCF semantic validator
dcf_semantic = ExtraValidatorSpec(
    suffix="semantic",
    validation_scope="semantic consistency",
    extra_checks_instruction="""
1. FCF CONSISTENCY: For each year, fcf ≈ nopat + depreciation - capex - change_in_working_capital within ±0.1 tolerance.
2. DISCOUNTING CONSISTENCY: pv_fcf ≈ fcf / (1 + wacc)^year within ±0.1 tolerance.
3. TERMINAL VALUE CONSISTENCY: terminal_value ≈ (last_fcf × (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate) within ±1.0 tolerance.
4. PV TERMINAL CONSISTENCY: pv_terminal_value ≈ terminal_value / (1 + wacc)^horizon within ±1.0 tolerance.
5. EV CONSISTENCY: enterprise_value ≈ sum(pv_fcf) + pv_terminal_value within ±1.0 tolerance.
6. EQUITY BRIDGE CONSISTENCY: If debt and cash available, equity_value ≈ enterprise_value - total_debt + cash_and_equivalents within ±1.0.
7. PER SHARE CONSISTENCY: If shares_outstanding available, value_per_share ≈ equity_value / shares_outstanding within ±0.01.
8. MONOTONIC DISCOUNTING: |pv_fcf| should generally decline with year; warn if it increases significantly.
9. UNITS: Must include "unit_scale": "millions" and "currency" fields.
""",
)

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

# ============================================================================
# Agent Definitions
# ============================================================================

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

data_agent = AgentValidator(
    name="data",
    model=model,
    tools=[eodHistoricalData],
    extra_validators=[data_semantic],
    instruction="""
You are the Data Collection Agent. Use ONLY the eodHistoricalData tools to gather compact inputs for valuation. Do not perform valuation math. Do not return raw API responses.

TOOLS (via eodHistoricalData MCP):
- get_stocks_from_search
- get_live_price_data
- get_us_live_extended_quotes
- get_fundamentals_data
- get_historical_market_cap
- get_earnings_trends
- get_company_news   # for context only, not detailed output

INPUTS:
- valuation_state.scoping_result.company_identifier
- valuation_state.scoping_result.as_of_date
- valuation_state.scoping_result.currency

STEPS:
1. Symbol resolution
   - If company_identifier is not clearly a ticker, call get_stocks_from_search and pick the best common-equity listing on a major exchange.
   - Store:
     - resolved_symbol (e.g. "AAPL.US")
     - resolved_name (e.g. "Apple Inc").

2. Market data
   - Use get_us_live_extended_quotes or get_live_price_data for:
     - last price, currency, volume, 52-week high/low if available.
   - Use get_historical_market_cap for recent market cap if available.
   - If market cap missing but shares_outstanding available from fundamentals, approximate market_cap = price × shares_outstanding; else null.

3. Fundamentals (last ~3 years)
   - CRITICAL: Call get_fundamentals_data with period=Annual (or equivalent parameter to ensure ANNUAL data, not quarterly).
   - Use 'from_date' = exactly 3 years before today.
   - From the last 3 ANNUAL fiscal years extract ONLY these specific fields:
     - income statement: revenue, EBIT (operating income), net_income.
     - balance sheet: total_debt, cash_and_equivalents, working_capital.
     - cash flow: operatingCashFlow (CFO), capitalExpenditures (capex), depreciation.
   - SANITY CHECK: For well-known mega-cap companies (e.g., AAPL, MSFT, GOOGL), annual revenue should be >$100B. If you get values like $80-90B for Apple, you likely pulled quarterly data by mistake - retry with explicit annual period.
   - CRITICAL: Extract ONLY the minimal required fields. Do NOT include the full API response or extra fields.
   - IMPORTANT: Store capex as a POSITIVE number (absolute value). If the API returns negative capex, negate it to make it positive.
   - Build a small normalized time series with ONLY the fields listed in the output schema below.

4. Earnings trends & sector
   - Optionally call get_earnings_trends and summarize only what is needed later (no raw payload).
   - From fundamentals, extract sector and industry strings.

OUTPUT REQUIREMENTS:
CRITICAL: Your response MUST be ONLY the JSON object below. Do NOT include any natural language text, summaries, explanations, or commentary before or after the JSON. Do NOT say things like "The current price is..." or "Here is the data...". ONLY output the raw JSON structure.

ALL FINANCIAL AMOUNTS:
- MUST be expressed in MILLIONS (e.g., Apple revenue of $383B = 383000 in millions)
- MUST include "unit_scale": "millions" and "currency": "USD" (or appropriate currency) at top level
- Capex MUST be stored as POSITIVE number representing cash outflow

{
  "data_result": {
    "resolved_symbol": "<string>",
    "resolved_name": "<string>",
    "unit_scale": "millions",
    "currency": "USD",
    "market_data": {
      "price": <number or null>,
      "currency": "<string or null>",
      "market_cap": <number or null>,
      "shares_outstanding": <number or null>
    },
    "historical_financials_normalized": {
      "years": [
        {
          "year": <int>,
          "revenue": <number>,
          "ebit": <number or null>,
          "net_income": <number or null>,
          "ebit_margin": <number or null>,
          "cfo": <number or null>,
          "capex": <number or null>,
          "depreciation": <number or null>,
          "total_debt": <number or null>,
          "cash_and_equivalents": <number or null>,
          "working_capital": <number or null>
        }
      ]
    },
    "sector": "<string or null>",
    "industry": "<string or null>"
  }
}
""",
    output_key="data_result",
)

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

dcf_agent = AgentValidator(
    name="dcf",
    model=json_model,
    tools=[],
    extra_validators=[dcf_semantic],
    instruction="""
You are the DCF Valuation Agent. Do not call tools.

INPUTS (from valuation_state):
- scoping_result
- data_result.market_data
- normalization_result.normalized_historical_financials
- forecast
- capital_assumptions

GOAL:
Compute unlevered FCFs, discount them, and derive enterprise value, equity value, and per-share value.

STEPS:
1. FCF series
   - For each forecast year t:
     - Extract: nopat, depreciation, capex, change_in_working_capital from forecast.years[t].
     - IMPORTANT: Capex and depreciation should be POSITIVE numbers in the forecast.
     - CRITICAL: Compute FCF for EACH year using this exact formula:
       FCF_t = nopat + depreciation – capex – change_in_working_capital
     - Example: If nopat=32.175B, dep=2.8B, capex=2.5B, ΔWC=-1.0B, then:
       FCF = 32.175 + 2.8 - 2.5 - (-1.0) = 33.475B
     - Do NOT return FCF = NOPAT. You MUST include all four components.

2. Terminal value (Gordon Growth formula)
   - Let n = last forecast year index.
   - FCF_(n+1) = FCF_n × (1 + terminal_growth_rate).
   - CRITICAL: Apply the perpetuity formula with the divisor:
     TerminalValue = FCF_(n+1) / (wacc – terminal_growth_rate)
   - Example: If FCF_(n+1)=37.4B, wacc=0.0889, g=0.025, then:
     TV = 37.4B / (0.0889 - 0.025) = 37.4B / 0.0639 ≈ 585B
   - Do NOT set TV = FCF_(n+1). You MUST divide by (wacc - g).

3. Discounting
   - CRITICAL: Use the exact wacc value from capital_assumptions (do not round prematurely)
   - For each t in 1..n:
     - Discount factor = (1 + wacc)^t
     - PV_FCF_t = FCF_t / discount_factor
     - Example: If FCF_1 = 89,429 million, wacc = 0.0831, then:
       discount_factor = (1.0831)^1 = 1.0831
       PV_FCF_1 = 89,429 / 1.0831 = 82,548.81 million
   - PV_TerminalValue = TerminalValue / (1 + wacc)^n
   - EnterpriseValue = sum(all PV_FCF_t) + PV_TerminalValue
   - IMPORTANT: Use sufficient precision (at least 2 decimal places for all intermediate calculations)

4. Equity value and per-share
   - Use latest total_debt and cash_and_equivalents from normalization_result or data_result.
   - CRITICAL: Compute net debt adjustment:
     EquityValue = EnterpriseValue – total_debt + cash_and_equivalents
   - Example: If EV=585B, debt=95B, cash=41B, then:
     Equity = 585 - 95 + 41 = 531B
   - Do NOT set EquityValue = EnterpriseValue. You MUST subtract debt and add cash.
   - If shares_outstanding present, ValuePerShare = EquityValue / shares_outstanding; else null.
   - Always compute all three: EnterpriseValue, EquityValue, ValuePerShare (if possible).
   - Align later with scoping_result.valuation_target (but still return all values).

OUTPUT:
Return ONLY JSON with key "dcf_result":

ALL AMOUNTS in MILLIONS.

{
  "dcf_result": {
    "unit_scale": "millions",
    "currency": "USD",
    "discount_rate_wacc": <number>,
    "terminal_growth_rate": <number>,
    "fcf_series": [
      { "year": <int>, "fcf": <number>, "pv_fcf": <number> }
    ],
    "terminal_value": <number>,
    "pv_terminal_value": <number>,
    "enterprise_value": <number>,
    "equity_value": <number>,
    "value_per_share": <number or null>,
    "dcf_notes": "<≤3 sentences on approximations or missing inputs>"
  }
}
""",
    output_key="dcf_result",
)

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

# Create the valuation workflow with validated agents
valuation_workflow = SequentialAgent(
    name="valuation_workflow",
    sub_agents=[
        scoping_agent,
        data_agent,
        normalization_agent,
        forecast_agent,
        wacc_agent,
        dcf_agent,
        multiples_agent,
        report_agent,
    ],
    after_agent_callback=auto_save_to_memory,
)

root_agent = valuation_workflow
# For backward compatibility
agent = root_agent
app_name = "financial_assistant"
