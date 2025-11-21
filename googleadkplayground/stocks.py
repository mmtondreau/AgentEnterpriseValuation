import os
from google.adk.agents import Agent, ParallelAgent, LoopAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.genai import types
from urllib.parse import urlencode

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StreamableHTTPServerParams,
)
from google.adk.sessions import DatabaseSessionService
from mcp import StdioServerParameters

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,  # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

LITE_MODEL = "gemini-2.5-flash-lite"

user_id = "debug_user"
session_id = "debug_session"


async def run():
    # Initialize MCP toolset within async context
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

    try:
        # Initialize agent with the toolset
        root_agent = Agent(
            name="root_agent",
            model=Gemini(model=LITE_MODEL, retry_options=retry_config),
            instruction="""You are a financial assistant. Based on the user prompt use the eodHistoricalData get_fundamentals_data tool to gather information about fundamentals. IMPORTANT: When calling get_fundamentals_data, you MUST use the from_date parameter to limit data to only the last 2 years to avoid exceeding token limits. For example, use from_date="2023-01-01". Also gather company news using get_company_news and current price data. Provide concise and accurate information to help users make informed financial decisions.""",
            tools=[eodHistoricalData],
            output_key="final_response",
        )

        db_url = "postgresql://postgres@localhost:5432/agent_state"  # Local PostgreSQL database
        session_service = DatabaseSessionService(db_url=db_url)

        # Initialize runner with the agent and database session service
        runner = Runner(
            agent=root_agent,
            app_name="financial_assistant",
            session_service=session_service
        )

        # Delete existing session if it exists, then create fresh session
        try:
            await runner.session_service.delete_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        except:
            pass  # Session doesn't exist, continue

        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        await runner.run_debug(
            "Get the latest fundamental data on AAPL and provide a summary of its recent performance.",
            user_id=user_id,
            session_id=session_id,
        )
        session = await runner.session_service.get_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        response = session.state.get("final_response")
        print(response)
    finally:
        # Properly close the MCP connection
        await eodHistoricalData.close()
