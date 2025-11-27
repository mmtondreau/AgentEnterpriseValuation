import os
from dataclasses import dataclass
from typing import Optional, List
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

import json


@dataclass
class ExtraValidatorSpec:
    """Specification for a stage-specific validator."""
    suffix: str
    validation_scope: str
    extra_checks_instruction: str
    tools: Optional[list] = None


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


# Note: exit_loop function removed - LoopAgent cannot be exited early via tools.
# Instead, refiner outputs the original unchanged content when all approve.


class AgentValidator(SequentialAgent):
    """Validates that an agent is correctly configured."""

    @staticmethod
    def _extra_validator_prompt(
        base_instruction: str,
        scope_label: str,
        extra_checks: str
    ) -> str:
        """Generate prompt for an extra validator."""
        return f"""
CRITICAL: You are a {scope_label.upper()} VALIDATOR, NOT a content generator. Your ONLY job is to output a single word or a short rejection line.

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The word "APPROVED" (if {scope_label} is valid)
2. "REJECTED: <one-line issue>" (if {scope_label} has problems)

DO NOT EVER output JSON, code, explanations, or anything else. ONLY "APPROVED" or "REJECTED: <issue>".

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{base_instruction}

VALIDATION CHECKS:
{extra_checks}

ABSOLUTELY FORBIDDEN:
- Do NOT generate corrected JSON
- Do NOT generate alternative outputs
- Do NOT write code blocks
- Do NOT ask questions
- Do NOT explain anything
- Do NOT use tools (unless explicitly provided and absolutely necessary)

EXAMPLES OF CORRECT VALIDATOR OUTPUT:
- "APPROVED"
- "REJECTED: capex is negative instead of positive"
- "REJECTED: wacc <= terminal_growth_rate"

EXAMPLES OF INCORRECT VALIDATOR OUTPUT (NEVER DO THIS):
- ```json{{"corrected": "output"}}``` ← WRONG! This is generating content!
- "The values should be..." ← WRONG! This is explaining!
- Any output longer than one line ← WRONG!

YOUR OUTPUT RIGHT NOW:
        """

    @staticmethod
    def _refiner_prompt(
        base_instruction: str,
        validator_count: int
    ) -> str:
        """Generate prompt for the refiner with dynamic validator count."""
        return f"""
CRITICAL: You are a REFINER. You fix JSON/content based on validator rejections.

YOUR ONLY ALLOWED OUTPUT:
- [Corrected JSON/content] - Output the corrected content, or the ORIGINAL UNCHANGED content if all validators approved

DO NOT EVER OUTPUT:
- ❌ "REJECTED: ..." - This is a validator output, NOT a refiner output
- ❌ "The issue is..." - This is an explanation, NOT a refiner output
- ❌ "APPROVED" - This is a validator output, NOT a refiner output
- ❌ Any text explanation - This is NOT a refiner output

STEP 1: COUNT APPROVALS
Look at the {validator_count} most recent validator outputs.
Count how many say EXACTLY "APPROVED".

STEP 2: DECISION
IF approval_count == {validator_count}:
  → Output the ORIGINAL content EXACTLY as it was (no changes)
  → This signals validation passed

ELSE (some validators rejected):
  → Find the ORIGINAL output (from before validation)
  → Read validator rejection reasons
  → Fix the original output to address ALL rejections
  → Output ONLY the corrected JSON/content (pure JSON, no markdown code blocks)
  → Use the SAME format as original (if JSON, output JSON; if markdown, output markdown)
  → NO "REJECTED: ..." messages
  → NO explanations or text before or after the JSON
  → NO markdown formatting like ```json...```

ORIGINAL TASK (for fixing):
{base_instruction}

CORRECT REFINER OUTPUTS:

✅ Example 1 - All approved:
Input: Validators say "APPROVED", "APPROVED", "APPROVED", "APPROVED"
Original: {{"a": 1, "b": 2}}
Output: {{"a": 1, "b": 2}}
(Exact copy of original - no changes)

✅ Example 2 - Missing field:
Input: Validators say "APPROVED", "REJECTED: Missing field X", "APPROVED", "APPROVED"
Original: {{"a": 1, "b": 2}}
Output: {{"a": 1, "b": 2, "X": null}}
(Just the corrected JSON, NO other text)

✅ Example 3 - Invalid value:
Input: Validators say "REJECTED: capex must be positive", "APPROVED"
Original: {{"forecast": {{"capex": -100}}}}
Output: {{"forecast": {{"capex": 100}}}}
(Just the corrected JSON, NO other text)

✅ Example 4 - Missing data, call tools:
Input: Validators say "REJECTED: Missing historical financial data"
Original: {{"data_result": {{"years": []}}}}
Output: [Calls get_fundamentals_data tool, then outputs corrected JSON with populated data]

WRONG REFINER OUTPUTS:

❌ "REJECTED: Historical data missing" - This is pretending to be a validator!
❌ "The original output is missing X" - This is explaining, not fixing!
❌ "I cannot fix this because..." - This is refusing to work!
❌ ```json{{"fixed": "data"}}``` - This has markdown code blocks!
❌ "Here is the corrected output: {{"data": ...}}" - This has explanatory text!

REMEMBER: You are a FIXER, not a validator. Output corrected content (or original unchanged if all approved). NEVER output "REJECTED: ..."
        """

    def __init__(
        self,
        instruction: str,
        tools: list,
        output_key: str,
        name: str,
        model=None,
        extra_validators: Optional[List[ExtraValidatorSpec]] = None,
        **kwargs  # Accept any additional LlmAgent parameters
    ):
        # Use provided model or default to Flash Lite
        agent_model = model if model is not None else Gemini(model=LITE_MODEL, retry_options=retry_config)

        # Default to empty list if no extra validators
        extra_validators = extra_validators or []

        initial_agent = Agent(
            name=f"{name}_initial_agent",
            model=agent_model,
            instruction=instruction,
            tools=tools,
            output_key=output_key,
        )
        format_validator_agent = Agent(
            name=f"{name}_format_validator_agent",
            model=agent_model,  # Use same model as initial agent to handle large context
            tools=[],  # Validators must have NO tools
            output_key=f"{name}_format_validation_feedback",
            instruction=f"""
CRITICAL: You are a FORMAT VALIDATOR, NOT a content generator. Your ONLY job is to output a single word or a short rejection line.

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The word "APPROVED" (if format is correct)
2. "REJECTED: <one-line issue>" (if format has problems)

DO NOT EVER output JSON, code, explanations, or anything else. ONLY "APPROVED" or "REJECTED: <issue>".

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED FORMAT (from agent instructions):
{instruction}

VALIDATION CHECKS:
1. Is it valid JSON (if JSON expected)?
2. Are all required fields present with correct names?
3. Are values the correct types?
4. Is there NO extra text/markdown/explanation around the output?
5. UNIT SCALE: Must include "unit_scale": "millions" and "currency": "USD" (or appropriate) if financial amounts present
6. CAPEX CONVENTION: All capex values must be POSITIVE numbers (representing cash outflow)

ABSOLUTELY FORBIDDEN:
- Do NOT generate corrected JSON
- Do NOT generate alternative outputs
- Do NOT write code blocks
- Do NOT ask questions
- Do NOT explain anything
- Do NOT use tools

EXAMPLES OF CORRECT VALIDATOR OUTPUT:
- "APPROVED"
- "REJECTED: Missing required field 'unit_scale'"
- "REJECTED: capex values are negative instead of positive"

EXAMPLES OF INCORRECT VALIDATOR OUTPUT (NEVER DO THIS):
- ```json{{"corrected": "output"}}``` ← WRONG! This is generating content!
- "Here is the corrected version..." ← WRONG! This is explaining!
- Any output longer than one line ← WRONG!

YOUR OUTPUT RIGHT NOW:
            """,
        )

        correctness_validator_agent = Agent(
            name=f"{name}_correctness_validator_agent",
            model=agent_model,  # Use same model as initial agent to handle large context
            tools=[],  # Validators must have NO tools
            output_key=f"{name}_correctness_validation_feedback",
            instruction=f"""
CRITICAL: You are a CORRECTNESS VALIDATOR, NOT a content generator. Your ONLY job is to output a single word or a short rejection line.

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The word "APPROVED" (if logically correct)
2. "REJECTED: <one-line issue>" (if logic has problems)

DO NOT EVER output JSON, code, explanations, or anything else. ONLY "APPROVED" or "REJECTED: <issue>".

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{instruction}

VALIDATION CHECKS (based on what's IN the output):
1. Are numbers internally consistent?
2. Are there logical contradictions?
3. Do calculations appear correct (spot check obvious ones)?
4. Does data match what was provided earlier in conversation?
5. SANITY CHECK: For mega-cap companies (AAPL, MSFT, GOOGL, AMZN), if annual revenue is <$100B, likely quarterly data was pulled - REJECT

ABSOLUTELY FORBIDDEN:
- Do NOT generate corrected JSON
- Do NOT generate alternative outputs
- Do NOT write code blocks
- Do NOT ask questions
- Do NOT explain anything
- Do NOT use tools (unless absolutely necessary for simple spot check)

EXAMPLES OF CORRECT VALIDATOR OUTPUT:
- "APPROVED"
- "REJECTED: Revenue inconsistent with prior period"
- "REJECTED: Apple annual revenue <$100B suggests quarterly data"

EXAMPLES OF INCORRECT VALIDATOR OUTPUT (NEVER DO THIS):
- ```json{{"corrected": "output"}}``` ← WRONG! This is generating content!
- "The DCF valuation implies..." ← WRONG! This is explaining!
- Any output longer than one line ← WRONG!

YOUR OUTPUT RIGHT NOW:
            """,
        )

        spec_validator_agent = Agent(
            name=f"{name}_spec_validator_agent",
            model=agent_model,  # Use same model as initial agent to handle large context
            tools=[],  # Validators must have NO tools
            output_key=f"{name}_spec_validation_feedback",
            instruction=f"""
CRITICAL: You are a SPEC VALIDATOR, NOT a content generator. Your ONLY job is to output a single word or a short rejection line.

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The word "APPROVED" (if requirements met)
2. "REJECTED: <one-line issue>" (if requirements not met)

DO NOT EVER output JSON, code, explanations, or anything else. ONLY "APPROVED" or "REJECTED: <issue>".

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{instruction}

VALIDATION CHECKS (based on what agent was supposed to produce):
1. Did it produce the required output structure?
2. Are required fields/sections present?
3. Did it follow the output type (JSON/markdown/etc.)?

ABSOLUTELY FORBIDDEN:
- Do NOT generate corrected JSON
- Do NOT generate alternative outputs
- Do NOT write code blocks
- Do NOT ask questions
- Do NOT explain anything
- Do NOT use tools

EXAMPLES OF CORRECT VALIDATOR OUTPUT:
- "APPROVED"
- "REJECTED: Missing required field 'dcf_notes'"
- "REJECTED: Output is markdown instead of JSON"

EXAMPLES OF INCORRECT VALIDATOR OUTPUT (NEVER DO THIS):
- ```json{{"corrected": "output"}}``` ← WRONG! This is generating content!
- "The output should include..." ← WRONG! This is explaining!
- Any output longer than one line ← WRONG!

YOUR OUTPUT RIGHT NOW:
            """,
        )

        # Create extra validator agents from specs
        extra_validator_agents = []
        for ev in extra_validators:
            extra_validator_agents.append(
                Agent(
                    name=f"{name}_{ev.suffix}_validator_agent",
                    model=agent_model,  # Use same model as initial agent to handle large context
                    tools=ev.tools or [],
                    output_key=f"{name}_{ev.suffix}_validation_feedback",
                    instruction=AgentValidator._extra_validator_prompt(
                        base_instruction=instruction,
                        scope_label=ev.validation_scope,
                        extra_checks=ev.extra_checks_instruction,
                    ),
                )
            )

        # Combine all validators
        validator_agents = [
            spec_validator_agent,
            format_validator_agent,
            correctness_validator_agent,
            *extra_validator_agents,
        ]
        validator_count = len(validator_agents)

        # Create refiner with dynamic validator count
        # Refiner needs access to same tools as initial agent
        refiner_agent = Agent(
            name=f"{name}_refiner_agent",
            model=agent_model,  # Use same model as initial agent
            instruction=AgentValidator._refiner_prompt(
                base_instruction=instruction,
                validator_count=validator_count,
            ),
            output_key=output_key,
            tools=tools,  # Same tools as initial agent
        )

        parallel_critique_team = ParallelAgent(
            name="ParallelCritiqueTeam",
            sub_agents=validator_agents,
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
