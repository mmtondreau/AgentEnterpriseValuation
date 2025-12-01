"""Financial Assistant Agent using EODHD MCP Server for market data."""

from google.adk.agents import SequentialAgent
from .scoping_agent import scoping_agent
from .data_agent import data_agent
from .normalization_agent import normalization_agent
from .forecast_agent import forecast_agent
from .wacc_agent import wacc_agent
from .dcf_agent import dcf_agent
from .multiples_agent import multiples_agent
from .report_agent import report_agent


async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    if callback_context._invocation_context.memory_service:
        await callback_context._invocation_context.memory_service.add_session_to_memory(
            callback_context._invocation_context.session
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
