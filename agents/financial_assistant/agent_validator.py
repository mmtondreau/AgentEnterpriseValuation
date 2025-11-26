import os
from google.adk.agents import Agent, LoopAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import load_memory
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


LITE_MODEL = "gemini-2.5-flash-lite"

from google.adk.tools import FunctionTool, ToolContext
import json


def validate_json(json_string: str) -> dict:
    """
    Validate a JSON string.

    Args:
        json_string: The JSON to validate.

    Returns:
        A dict with whether it's valid and any error message.
        Use this whenever you need to check or repair JSON.
    """
    try:
        obj = json.loads(json_string)
        return {
            "valid": True,
            "error": None,
            "parsed_type": type(obj).__name__,
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "parsed_type": None,
        }


def exit_loop():
    """Call this function ONLY when the response is 'APPROVED', indicating the response correct, valid, complete and no more changes are needed."""
    return {
        "status": "approved",
        "message": "Response approved. Exiting refinement loop.",
    }


class AgentValidator(SequentialAgent):
    """Validates that an agent is correctly configured."""

    def __init__(
        self,
        instruction: str,
        tools: list,
        output_key: str,
        name: str,
        model=None,
        **kwargs  # Accept any additional LlmAgent parameters
    ):
        # Use provided model or default to Flash Lite
        agent_model = model if model is not None else Gemini(model=LITE_MODEL, retry_options=retry_config)

        initial_agent = Agent(
            name=f"{name}_initial_agent",
            model=agent_model,
            instruction=instruction,
            tools=tools,
            output_key=f"{name}_current_output",
        )
        format_validator_agent = Agent(
            name=f"{name}_format_validator_agent",
            model=Gemini(model=LITE_MODEL, retry_options=retry_config, generation_config=types.GenerateContentConfig(temperature=0.0)),
            output_key=f"{name}_format_validation_feedback",
            instruction=f"""
You validate ONLY formatting and schema. You are NOT an assistant. You do NOT answer user questions. You do NOT generate content.

INPUT: The immediately previous agent's output from conversation history.

EXPECTED FORMAT (from agent instructions):
{instruction}

OUTPUT FORMAT (you must follow this EXACTLY):
- If format is correct: Output ONLY the word "APPROVED"
- If format is wrong: Output "REJECTED: " followed by ONE line describing the issue

VALIDATION CHECKS:
1. Is it valid JSON (if JSON expected)?
2. Are all required fields present with correct names?
3. Are values the correct types?
4. Is there NO extra text/markdown/explanation around the output?
5. UNIT SCALE: Must include "unit_scale": "millions" and "currency": "USD" (or appropriate) if financial amounts present
6. CAPEX CONVENTION: All capex values must be POSITIVE numbers (representing cash outflow)

FORBIDDEN:
- Do NOT ask questions
- Do NOT say "I need more information"
- Do NOT generate alternative outputs
- Do NOT explain what the agent should do
- Do NOT use code execution
- Do NOT provide multi-line feedback beyond "REJECTED: <issue>"

REMEMBER: Output is ONLY "APPROVED" or "REJECTED: <one-line issue>". Nothing else.
            """,
        )

        correctness_validator_agent = Agent(
            name=f"{name}_correctness_validator_agent",
            model=Gemini(model=LITE_MODEL, retry_options=retry_config, generation_config=types.GenerateContentConfig(temperature=0.0)),
            output_key=f"{name}_correctness_validation_feedback",
            instruction=f"""
You validate ONLY logical/numerical correctness. You are NOT an assistant. You do NOT answer user questions. You do NOT generate content.

INPUT: The immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{instruction}

OUTPUT FORMAT (you must follow this EXACTLY):
- If logically correct: Output ONLY the word "APPROVED"
- If logically incorrect: Output "REJECTED: " followed by ONE line describing the error

VALIDATION CHECKS (based on what's IN the output, not what's missing):
1. Are numbers internally consistent?
2. Are there logical contradictions?
3. Do calculations appear correct (spot check obvious ones)?
4. Does data match what was provided earlier in conversation?
5. SANITY CHECK: For mega-cap companies (AAPL, MSFT, GOOGL, AMZN), if annual revenue is <$100B, likely quarterly data was pulled - REJECT

FORBIDDEN:
- Do NOT ask "what valuation method?" or "what assumptions?"
- Do NOT say "I need more information to validate"
- Do NOT generate alternative outputs
- Do NOT perform full recalculations (only spot check if obvious)
- Do NOT use code execution unless absolutely necessary for simple spot check
- Do NOT provide multi-line feedback beyond "REJECTED: <issue>"

REMEMBER: Output is ONLY "APPROVED" or "REJECTED: <one-line issue>". Nothing else.
            """,
        )

        spec_validator_agent = Agent(
            name=f"{name}_spec_validator_agent",
            model=Gemini(model=LITE_MODEL, retry_options=retry_config, generation_config=types.GenerateContentConfig(temperature=0.0)),
            output_key=f"{name}_spec_validation_feedback",
            instruction=f"""
You validate ONLY requirement coverage. You are NOT an assistant. You do NOT answer user questions. You do NOT generate content.

INPUT: The immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{instruction}

OUTPUT FORMAT (you must follow this EXACTLY):
- If requirements met: Output ONLY the word "APPROVED"
- If requirements not met: Output "REJECTED: " followed by ONE line describing what's missing

VALIDATION CHECKS (based on what agent was supposed to produce):
1. Did it produce the required output structure?
2. Are required fields/sections present?
3. Did it follow the output type (JSON/markdown/etc.)?

FORBIDDEN:
- Do NOT ask "what else should be included?"
- Do NOT say "the user needs to specify..."
- Do NOT generate missing content
- Do NOT use code execution
- Do NOT provide multi-line feedback beyond "REJECTED: <issue>"

REMEMBER: Output is ONLY "APPROVED" or "REJECTED: <one-line issue>". Nothing else.
            """,
        )
        refiner_agent = Agent(
            name=f"{name}_refiner_agent",
            model=Gemini(model=LITE_MODEL, retry_options=retry_config, generation_config=types.GenerateContentConfig(temperature=0.3)),
            instruction=f"""
You are a refiner. Your job is to fix output based on validator feedback. You are NOT an assistant. You do NOT ask user questions.

INPUTS (from recent conversation history):
1. Original agent output
2. Three validator feedbacks: format, correctness, specification

ORIGINAL AGENT TASK:
{instruction}

DECISION LOGIC (follow EXACTLY):
1. Check the three most recent validator outputs
2. Count how many say EXACTLY "APPROVED" (not "approved", not "APPROVED with notes", EXACTLY "APPROVED")
3. IF all three validators said EXACTLY "APPROVED":
   - Call the exit_loop function
   - Output NOTHING else (no text, no explanation)
4. ELSE (at least one validator rejected):
   - Read the original output
   - Read the rejection reasons
   - Output a CORRECTED version in the EXACT SAME FORMAT
   - Do NOT add explanations or apologies
   - Do NOT ask questions
   - ONLY output the fixed content

EXAMPLE (if original was JSON and spec said "missing field X"):
Output ONLY this (no explanation before or after):
```json
{{
  "corrected": "json",
  "with_field_x": "added"
}}
```

CRITICAL: Do NOT write "The original output is missing..." or "Here is the corrected version...". ONLY output the corrected content.

FORBIDDEN:
- Do NOT say "I need clarification on..."
- Do NOT say "Please specify the valuation method..."
- Do NOT ask user questions
- Do NOT generate plans or explanations
- Do NOT output anything if all validators approved (just call exit_loop)

REMEMBER: If APPROVED APPROVED APPROVED → call exit_loop(). Else → output corrected content only.
            """,
            output_key=output_key,
            tools=[FunctionTool(exit_loop)],
        )
        parallel_critique_team = ParallelAgent(
            name="ParallelCritiqueTeam",
            sub_agents=[
                spec_validator_agent,
                format_validator_agent,
                correctness_validator_agent,
            ],
        )

        editing_loop_agent = LoopAgent(
            name="EditingLoopAgent",
            sub_agents=[
                parallel_critique_team,
                refiner_agent,
            ],
            max_iterations=2,
        )

        super().__init__(
            name=f"{name}_agent_validator",
            sub_agents=[
                initial_agent,
                editing_loop_agent,
            ],
        )
