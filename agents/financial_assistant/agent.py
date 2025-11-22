"""Financial Assistant Agent using EODHD MCP Server for market data."""

import os
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import load_memory
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.genai import types

# Retry configuration for Gemini API
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

LITE_MODEL = "gemini-2.5-flash-lite"


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    if callback_context._invocation_context.memory_service:
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
        )


# Initialize MCP toolset for financial data
eodHistoricalData = McpToolset(
    connection_params=StreamableHTTPServerParams(
        url="http://127.0.0.1:8000/mcp",
        timeout=60,
    ),
    tool_filter=[
        "get_historical_stock_prices",
        "get_fundamentals_data",
        "get_company_news",
        "get_sentiment_data",
        "get_macro_indicator",
        "get_economic_events",
        "get_upcoming_earnings",
    ],
)

# Define the root agent for ADK Web UI
root_agent = Agent(
    name="financial_assistant",
    model=Gemini(model=LITE_MODEL, retry_options=retry_config),
    instruction="""
    You are a financial assistant. Based on the user prompt use the eodHistoricalData get_fundamentals_data
    tool to gather information about fundamentals. IMPORTANT: When calling get_fundamentals_data, you MUST use the from_date
    parameter to limit data to only the last 2 years to avoid exceeding token limits. For example, use from_date="2023-01-01".
    Also gather company news using get_company_news and current price data. Provide concise and accurate information to help
    users make informed financial decisions. Make sure to include the current timestamp in your analysis that this report was
    generated if it was just generated (e.g. not from memory).

    You have access to load_memory tool to retrieve relevant past analysis from long-term memory. If you see the analysis is
    recently done (within last 24 hours), you can reference it in your final response. Make sure to indicate the time of the
    original analysis. We should be checking against the original analysis timestamp not the timestamp of the session in which it is
    stored.
    """,
    tools=[eodHistoricalData, load_memory],
    output_key="final_response",
    after_agent_callback=auto_save_to_memory,
)

# For backward compatibility
agent = root_agent
app_name = "financial_assistant"
