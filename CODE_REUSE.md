# Code Reuse Architecture

This document explains how the agent code is shared between the Web UI and CLI runner.

## Single Source of Truth

The financial assistant agent is defined **once** in:
```
agents/financial_assistant/agent.py
```

This agent definition is then reused by:
1. **ADK Web UI** (`./run_web_ui.sh`)
2. **Legacy CLI Runner** (`python -m googleadkplayground`)

## Architecture

```
agents/financial_assistant/
├── __init__.py          # Exports: root_agent, agent, eodHistoricalData
├── agent.py             # ✨ Single agent definition
└── README.md

googleadkplayground/
└── stocks.py            # CLI runner - imports agent from agents/
```

## How It Works

### Agent Definition (agents/financial_assistant/agent.py)

```python
# Define the agent ONCE
root_agent = Agent(
    name="financial_assistant",
    model=Gemini(model=LITE_MODEL),
    instruction="...",
    tools=[eodHistoricalData, load_memory],
    after_agent_callback=auto_save_to_memory,
)

# Export for reuse
agent = root_agent
app_name = "financial_assistant"
```

### Web UI Usage

The Web UI automatically discovers and loads the agent:

```bash
./run_web_ui.sh
# Looks for: agents/financial_assistant/agent.py -> root_agent
```

### CLI Usage (googleadkplayground/stocks.py)

The CLI imports and reuses the same agent:

```python
from agents.financial_assistant import root_agent, eodHistoricalData

async def run():
    runner = Runner(
        agent=root_agent,  # Reuse the shared agent
        app_name="financial_assistant",
        session_service=session_service,
        memory_service=memory_service,
    )
    # ... run the agent
```

## Benefits

### ✅ Single Source of Truth
- Agent definition exists in exactly one place
- No duplication of instructions, tools, or callbacks
- Changes to the agent automatically apply to both Web UI and CLI

### ✅ Consistency
- Both interfaces use identical agent behavior
- Same tools, same instructions, same memory handling
- Results are consistent regardless of how you run it

### ✅ Maintainability
- Update agent instructions in one place: `agents/financial_assistant/agent.py`
- Add/remove tools once - affects both Web UI and CLI
- Easier to test - one agent to test, not two

### ✅ Scalability
- Easy to add new agents to `agents/` directory
- Each agent can have its own CLI runner if needed
- Web UI automatically discovers new agents

## Making Changes

### To Update Agent Behavior

Edit: `agents/financial_assistant/agent.py`

```python
# Update instructions
instruction="""Your new instructions here..."""

# Add/remove tools
tools=[eodHistoricalData, load_memory, new_tool]

# Modify callbacks
after_agent_callback=your_new_callback
```

Changes automatically apply to:
- ✅ Web UI (restart web server)
- ✅ CLI Runner (next run)

### To Add a New Agent

1. Create new directory: `agents/my_new_agent/`
2. Add `__init__.py` and `agent.py`
3. Define `root_agent` in `agent.py`
4. Web UI automatically discovers it
5. Optionally create a CLI runner in `googleadkplayground/my_new_runner.py`

## Shared Components

### Agent Module Exports

From `agents/financial_assistant/__init__.py`:
```python
__all__ = ["root_agent", "agent", "app_name", "eodHistoricalData"]
```

- **root_agent**: Primary agent (used by Web UI)
- **agent**: Alias for root_agent (backward compatibility)
- **app_name**: Application name string
- **eodHistoricalData**: MCP toolset (for proper cleanup in CLI)

### Services

Shared services in `services/`:
```python
from services import PostgresMemoryService
```

Used by both Web UI and CLI for long-term memory.

## Testing Both Modes

### Test Web UI
```bash
./run_web_ui.sh
# Visit: http://localhost:8080
```

### Test CLI
```bash
python -m googleadkplayground
```

Both should produce equivalent results for the same queries!

## Migration Path

If you have custom code in `stocks.py`:

1. **Extract agent logic** to `agents/your_agent/agent.py`
2. **Keep runner logic** in `stocks.py` (session management, input/output)
3. **Import the agent** from `agents/your_agent`
4. **Test both modes** to ensure consistency

## Example: Adding a Tool

**Before** (duplicated):
```python
# agents/financial_assistant/agent.py
tools=[eodHistoricalData, load_memory]

# googleadkplayground/stocks.py
tools=[eodHistoricalData, load_memory]  # ❌ Duplicate!
```

**After** (shared):
```python
# agents/financial_assistant/agent.py
tools=[eodHistoricalData, load_memory, new_tool]  # ✅ Single definition

# googleadkplayground/stocks.py
from agents.financial_assistant import root_agent  # ✅ Reuses agent
```

Add the tool once, it works everywhere!

## Questions?

- Agent not loading in Web UI? Check `agents/*/agent.py` exports `root_agent`
- CLI not finding agent? Verify import path: `from agents.financial_assistant import root_agent`
- Changes not reflected? Web UI: restart server. CLI: run again.

---

**Remember**: One agent definition to rule them all! 🎯
