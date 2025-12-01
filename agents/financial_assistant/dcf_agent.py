"""DCF Valuation Agent for valuation workflow."""

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

json_model = Gemini(
    model=FLASH_MODEL,
    retry_options=retry_config,
    generation_config=types.GenerateContentConfig(
        response_mime_type="application/json"
    ),
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
