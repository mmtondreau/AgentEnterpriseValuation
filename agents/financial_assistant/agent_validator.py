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

from google.adk.tools import FunctionTool, ToolContext
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


def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the response is 'APPROVED', indicating the response is correct, valid, complete and no more changes are needed.

    This will set the escalate flag to exit the validation loop early.
    """
    tool_context.actions.escalate = True
    return "Validation approved. Exiting refinement loop."


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
═══════════════════════════════════════════════════════════
CRITICAL: YOU ARE A {scope_label.upper()} VALIDATOR - NOT A CONTENT GENERATOR
═══════════════════════════════════════════════════════════

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The single word: APPROVED
2. The text: REJECTED: <one-line issue>

═══════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - NEVER EVER DO THIS:
═══════════════════════════════════════════════════════════
❌ Do NOT output JSON (```json{{...}}```)
❌ Do NOT output code blocks
❌ Do NOT output explanations or sentences
❌ Do NOT output anything longer than one line
❌ Do NOT generate corrected versions
❌ Do NOT use tools (unless explicitly provided and absolutely necessary)

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{base_instruction}

VALIDATION CHECKS:
{extra_checks}

═══════════════════════════════════════════════════════════
CORRECT VALIDATOR OUTPUTS:
═══════════════════════════════════════════════════════════
✅ "APPROVED"
✅ "REJECTED: capex is negative instead of positive"
✅ "REJECTED: wacc <= terminal_growth_rate"

═══════════════════════════════════════════════════════════
WRONG VALIDATOR OUTPUTS - NEVER DO THIS:
═══════════════════════════════════════════════════════════
❌ ```json{{"corrected": "output"}}``` ← Generating content!
❌ "The values should be..." ← Explaining!
❌ Any multi-line output ← Too long!
❌ {{"fixed": "data"}} ← Generating JSON!

YOUR OUTPUT RIGHT NOW (must be ONLY "APPROVED" or "REJECTED: ..."):
        """

    @staticmethod
    def _refiner_prompt(
        base_instruction: str,
        validator_count: int
    ) -> str:
        """Generate prompt for the refiner with dynamic validator count."""
        return f"""
===========================================
CRITICAL INSTRUCTION - READ THIS FIRST
===========================================
You are a REFINER agent. Your ONLY job is:
1. Call exit_loop() if all {validator_count} validators said "APPROVED"
2. Output corrected JSON/content if any validator rejected

NEVER EVER OUTPUT TEXT EXPLANATIONS OR "REJECTED: ..." MESSAGES!
===========================================

STEP 1: COUNT APPROVALS
Look at the {validator_count} most recent validator outputs ONLY.
Count how many say the EXACT word "APPROVED" (nothing else).

STEP 2: DECISION

IF you counted {validator_count} validators that said "APPROVED":
  ╔══════════════════════════════════════════════════════╗
  ║  ACTION: Call the exit_loop() function tool         ║
  ║  OUTPUT: Nothing else - do NOT add any text         ║
  ╚══════════════════════════════════════════════════════╝

ELSE (at least one validator said "REJECTED: ..."):
  ╔══════════════════════════════════════════════════════╗
  ║  1. Find the ORIGINAL output (before validators)     ║
  ║  2. Read each "REJECTED: ..." reason                 ║
  ║  3. Fix the original to address ALL rejections       ║
  ║  4. Output ONLY pure JSON (no markdown, no text)     ║
  ║  5. If data missing, call tools to fetch it          ║
  ╚══════════════════════════════════════════════════════╝

ORIGINAL TASK (for reference when fixing):
{base_instruction}

═══════════════════════════════════════════════════════════
EXAMPLES OF CORRECT REFINER BEHAVIOR
═══════════════════════════════════════════════════════════

✅ CORRECT Example 1 - All approved:
  Validators: "APPROVED", "APPROVED", "APPROVED"
  Refiner action: Call exit_loop() tool
  Refiner output: [function call only, no text]

✅ CORRECT Example 2 - Missing field:
  Validators: "APPROVED", "REJECTED: Missing field X", "APPROVED"
  Original: {{"a": 1, "b": 2}}
  Refiner output: {{"a": 1, "b": 2, "X": null}}

✅ CORRECT Example 3 - Invalid value:
  Validators: "REJECTED: capex must be positive"
  Original: {{"forecast": {{"capex": -100}}}}
  Refiner output: {{"forecast": {{"capex": 100}}}}

═══════════════════════════════════════════════════════════
EXAMPLES OF WRONG REFINER BEHAVIOR - NEVER DO THIS!
═══════════════════════════════════════════════════════════

❌ WRONG: "REJECTED: Historical data missing"
   → This is pretending to be a validator! You must FIX, not reject!

❌ WRONG: "The original output is missing X field"
   → This is explaining! You must output CORRECTED JSON, not text!

❌ WRONG: "I cannot fix this because..."
   → You must always try to fix! Call tools if needed!

❌ WRONG: ```json{{"fixed": "data"}}```
   → NO markdown blocks! Output pure JSON only!

❌ WRONG: "Here is the corrected output: {{"data": 123}}"
   → NO explanatory text! Output pure JSON only!

❌ WRONG: Outputting JSON when all validators approved
   → Call exit_loop() instead! Don't waste iterations!

═══════════════════════════════════════════════════════════
FINAL REMINDER
═══════════════════════════════════════════════════════════
- If {validator_count}/{validator_count} said "APPROVED" → Call exit_loop()
- If any validator rejected → Output corrected JSON (no text, no markdown)
- NEVER output "REJECTED: ..." or explanations
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
═══════════════════════════════════════════════════════════
CRITICAL: YOU ARE A FORMAT VALIDATOR - NOT A CONTENT GENERATOR
═══════════════════════════════════════════════════════════

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The single word: APPROVED
2. The text: REJECTED: <one-line issue>

═══════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - NEVER EVER DO THIS:
═══════════════════════════════════════════════════════════
❌ Do NOT output JSON (```json{{...}}```)
❌ Do NOT output code blocks
❌ Do NOT output explanations or sentences
❌ Do NOT output anything longer than one line
❌ Do NOT generate corrected versions
❌ Do NOT use tools

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

═══════════════════════════════════════════════════════════
CORRECT VALIDATOR OUTPUTS:
═══════════════════════════════════════════════════════════
✅ "APPROVED"
✅ "REJECTED: Missing required field 'unit_scale'"
✅ "REJECTED: capex values are negative instead of positive"

═══════════════════════════════════════════════════════════
WRONG VALIDATOR OUTPUTS - NEVER DO THIS:
═══════════════════════════════════════════════════════════
❌ ```json{{"corrected": "output"}}``` ← Generating content!
❌ "Here is the corrected version..." ← Explaining!
❌ Any multi-line output ← Too long!
❌ {{"fixed": "data"}} ← Generating JSON!

YOUR OUTPUT RIGHT NOW (must be ONLY "APPROVED" or "REJECTED: ..."):
            """,
        )

        correctness_validator_agent = Agent(
            name=f"{name}_correctness_validator_agent",
            model=agent_model,  # Use same model as initial agent to handle large context
            tools=[],  # Validators must have NO tools
            output_key=f"{name}_correctness_validation_feedback",
            instruction=f"""
═══════════════════════════════════════════════════════════
CRITICAL: YOU ARE A CORRECTNESS VALIDATOR - NOT A CONTENT GENERATOR
═══════════════════════════════════════════════════════════

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The single word: APPROVED
2. The text: REJECTED: <one-line issue>

═══════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - NEVER EVER DO THIS:
═══════════════════════════════════════════════════════════
❌ Do NOT output JSON (```json{{...}}```)
❌ Do NOT output code blocks
❌ Do NOT output explanations or sentences
❌ Do NOT output anything longer than one line
❌ Do NOT generate corrected versions
❌ Do NOT use tools (unless absolutely necessary)

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

═══════════════════════════════════════════════════════════
CORRECT VALIDATOR OUTPUTS:
═══════════════════════════════════════════════════════════
✅ "APPROVED"
✅ "REJECTED: Revenue inconsistent with prior period"
✅ "REJECTED: Apple annual revenue <$100B suggests quarterly data"

═══════════════════════════════════════════════════════════
WRONG VALIDATOR OUTPUTS - NEVER DO THIS:
═══════════════════════════════════════════════════════════
❌ ```json{{"corrected": "output"}}``` ← Generating content!
❌ "The DCF valuation implies..." ← Explaining!
❌ Any multi-line output ← Too long!
❌ {{"fixed": "data"}} ← Generating JSON!

YOUR OUTPUT RIGHT NOW (must be ONLY "APPROVED" or "REJECTED: ..."):
            """,
        )

        spec_validator_agent = Agent(
            name=f"{name}_spec_validator_agent",
            model=agent_model,  # Use same model as initial agent to handle large context
            tools=[],  # Validators must have NO tools
            output_key=f"{name}_spec_validation_feedback",
            instruction=f"""
═══════════════════════════════════════════════════════════
CRITICAL: YOU ARE A SPEC VALIDATOR - NOT A CONTENT GENERATOR
═══════════════════════════════════════════════════════════

YOUR ONLY TWO ALLOWED OUTPUTS:
1. The single word: APPROVED
2. The text: REJECTED: <one-line issue>

═══════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN - NEVER EVER DO THIS:
═══════════════════════════════════════════════════════════
❌ Do NOT output JSON (```json{{...}}```)
❌ Do NOT output code blocks
❌ Do NOT output explanations or sentences
❌ Do NOT output anything longer than one line
❌ Do NOT generate corrected versions
❌ Do NOT use tools

WHAT TO VALIDATE:
Review the immediately previous agent's output from conversation history.

EXPECTED TASK (from agent instructions):
{instruction}

VALIDATION CHECKS (based on what agent was supposed to produce):
1. Did it produce the required output structure?
2. Are required fields/sections present?
3. Did it follow the output type (JSON/markdown/etc.)?

═══════════════════════════════════════════════════════════
CORRECT VALIDATOR OUTPUTS:
═══════════════════════════════════════════════════════════
✅ "APPROVED"
✅ "REJECTED: Missing required field 'dcf_notes'"
✅ "REJECTED: Output is markdown instead of JSON"

═══════════════════════════════════════════════════════════
WRONG VALIDATOR OUTPUTS - NEVER DO THIS:
═══════════════════════════════════════════════════════════
❌ ```json{{"corrected": "output"}}``` ← Generating content!
❌ "The output should include..." ← Explaining!
❌ Any multi-line output ← Too long!
❌ {{"fixed": "data"}} ← Generating JSON!

YOUR OUTPUT RIGHT NOW (must be ONLY "APPROVED" or "REJECTED: ..."):
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
        # Refiner needs access to same tools as initial agent, plus exit_loop
        refiner_tools = tools + [FunctionTool(exit_loop)]
        refiner_agent = Agent(
            name=f"{name}_refiner_agent",
            model=agent_model,  # Use same model as initial agent
            instruction=AgentValidator._refiner_prompt(
                base_instruction=instruction,
                validator_count=validator_count,
            ),
            output_key=output_key,
            tools=refiner_tools,
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
            max_iterations=5,
        )

        super().__init__(
            name=f"{name}_agent_validator",
            sub_agents=[
                initial_agent,
                editing_loop_agent,
            ],
        )
