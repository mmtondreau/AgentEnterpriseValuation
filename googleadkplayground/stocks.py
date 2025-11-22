"""Legacy CLI runner for the financial assistant agent.

This module provides a command-line interface to run the financial assistant agent.
For the web UI, use: ./run_web_ui.sh

The agent itself is defined in agents/financial_assistant/agent.py and is shared
between this CLI runner and the Web UI.
"""

import os
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from services.postgres_memory_service import PostgresMemoryService

# Import the shared agent from agents directory
from agents.financial_assistant import root_agent, eodHistoricalData

user_id = "debug_user"
session_id = "debug_session"


async def run():
    """Run the financial assistant agent in CLI mode."""
    try:

        db_url = "postgresql://postgres@localhost:5432/agent_state"  # Local PostgreSQL database
        session_service = DatabaseSessionService(db_url=db_url)

        # Use async URL for memory service
        async_db_url = "postgresql+asyncpg://postgres@localhost:5432/agent_state"
        memory_service = PostgresMemoryService(db_url=async_db_url)
        # Initialize runner with the agent and database session service
        runner = Runner(
            agent=root_agent,
            app_name="financial_assistant",
            session_service=session_service,
            memory_service=memory_service,
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
