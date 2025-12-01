# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an AI agent for enterprise valuation using Google's Agent Development Kit (ADK) and integrating with the EODHD (EOD Historical Data) Market Context Protocol (MCP) server. The agent performs comprehensive DCF (Discounted Cash Flow) valuations with multiple validation stages including data collection, normalization, forecasting, WACC calculation, and multiples analysis.

## Development Setup

### Environment Setup

```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell

# Set required environment variables in .env
GOOGLE_API_KEY=your_gemini_api_key
EODHD_API_KEY=your_eodhd_api_key
```

### Running the Application

#### Laumch the EODHD MCP Server:

```bash
./run-mcp.sh
```

#### Run the agent service:

```bash
# Run the main application (currently configured to run stocks.py)
python -m agententerpriseval
```

## Code Architecture

### Main Components

**agententerpriseval/stocks.py** - Financial analysis agent that:

- Connects to EODHD MCP server via stdio connection
- Uses McpToolset to expose financial data tools (fundamentals, news, prices, etc.)
- Runs a single Agent with Gemini 2.5 Flash Lite model
- Currently hardcoded to analyze AAPL with path `/Users/matthewtondreau/Workplace/GoogleADKPlayground/EODHD_MCP_server/server.py`

**EODHD_MCP_server/** - Cloned MCP server providing financial data tools:

- Run standalone: `python EODHD_MCP_server/server.py --http`
- HTTP mode: defaults to http://127.0.0.1:8000/mcp
- Requires EODHD_API_KEY environment variable

### Key Patterns

**Agent State Management**: All agents use InMemoryRunner with session service. State is passed between agents via output_key/input variables in instruction templates using `{variable_name}` syntax.

**Retry Configuration**: All agents use consistent retry config for Gemini API:

```python
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)
```

**MCP Integration**: Uses google.adk.tools.mcp_tool.McpToolset with StdioConnectionParams to connect external MCP servers as tool providers.

## Development Commands

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy agententerpriseval/
```

## Important Notes

- The stocks.py agent has a hardcoded absolute path to the EODHD MCP server - update this when running on different machines
- The valuation workflow uses Gemini 2.5 Flash for handling large context accumulation
- All agents use `run_debug()` or `run()` methods with hardcoded user_id="debug_user" and session_id="debug_session"
- The conf/local.yml file is a Hydra configuration for Google API key management (currently minimal)
- Agent definitions are now modularized in separate files under `agents/financial_assistant/`
