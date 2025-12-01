"""Data Collection Agent for valuation workflow."""

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

model = Gemini(model=FLASH_MODEL, retry_options=retry_config)

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
