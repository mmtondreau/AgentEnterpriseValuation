# Agent Enterprise Valuation

An AI agent for enterprise valuation using Google's Agent Development Kit (ADK) integrated with the EODHD (EOD Historical Data) Market Context Protocol (MCP) server. Performs comprehensive DCF (Discounted Cash Flow) valuations with multiple validation stages using Google's Gemini models.

## Features

- Financial analysis agents with real-time market data access via EODHD MCP server
- PostgreSQL database for persistent agent state and session management
- Dockerized infrastructure for easy deployment
- Multiple agent architectures (sequential, parallel, loop) for various use cases

## Prerequisites

- Python 3.11+
- Poetry (Python package manager)
- Docker and Docker Compose
- API Keys:
  - Google API Key (for Gemini models)
  - EODHD API Key (for financial data)

## Quick Start

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd AgentEnterpriseValuation

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# GOOGLE_API_KEY=your_gemini_api_key_here
# EODHD_API_KEY=your_eodhd_api_key_here
```

### 3. Start Infrastructure Services

Start the PostgreSQL database and EODHD MCP server using Docker Compose:

```bash
# Build and start both services
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f
```

This will start:
- **PostgreSQL** on `localhost:5432` (database: `agent_state`, user: `postgres`, no password)
- **EODHD MCP Server** on `localhost:8000`

### 4. Run the Agents

You have two options to run the agents:

#### Option A: ADK Web UI (Recommended)

Start the interactive web interface:

```bash
# Make the script executable (first time only)
chmod +x run_web_ui.sh

# Start the web UI
./run_web_ui.sh
```

Then open your browser to `http://localhost:8080` to interact with the financial assistant agent.

#### Option B: CLI Mode

Run the legacy CLI version:

```bash
# Run the main application (currently runs stocks.py)
python -m agententerpriseval
```

The agent will:
- Connect to the PostgreSQL database at `localhost:5432`
- Connect to the EODHD MCP server at `localhost:8000`
- Execute the financial valuation workflow

## Project Structure

```
AgentEnterpriseValuation/
├── agents/                       # ADK agents directory
│   └── financial_assistant/      # Financial valuation agent
│       ├── __init__.py           # Agent module initialization
│       ├── agent.py              # Main workflow orchestrator
│       ├── scoping_agent.py      # Scoping & clarification
│       ├── data_agent.py         # Data collection
│       ├── normalization_agent.py # Business normalization
│       ├── forecast_agent.py     # Financial forecasting
│       ├── wacc_agent.py         # WACC calculation
│       ├── dcf_agent.py          # DCF valuation
│       ├── multiples_agent.py    # Multiples analysis
│       ├── report_agent.py       # Report generation
│       ├── agent_validator.py    # Validation framework
│       ├── eodhd_mcp.py          # EODHD MCP integration
│       └── README.md             # Agent documentation
├── services/                     # Shared services
│   ├── __init__.py
│   └── postgres_memory_service.py # PostgreSQL memory service
├── agententerpriseval/           # CLI application code
│   ├── stocks.py                 # Agent runner
│   └── __main__.py              # Entry point for CLI mode
├── EODHD_MCP_server/            # EODHD MCP server (cloned submodule)
├── conf/                        # Hydra configuration files
│   └── local.yml               # Local config (API keys)
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile.postgres          # PostgreSQL database image
├── Dockerfile.eodmcp           # EODHD MCP server image
├── init-db.sql                 # Database schema initialization
├── agents_config.py             # Agent services configuration
├── run_web_ui.sh               # Start ADK Web UI
├── .env.example                # Environment variable template
├── pyproject.toml              # Poetry dependencies
└── README.md                   # This file
```

## Available Services

### PostgreSQL Database

The database stores agent state, sessions, and conversation history.

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `agent_state`
- User: `postgres`
- Password: (none - trust authentication)
- Connection String: `postgresql://postgres@localhost:5432/agent_state`

**Pre-configured Tables:**
- `sessions` - User session tracking
- `agent_state` - Agent state snapshots
- `conversation_history` - Message history with timestamps

### EODHD MCP Server

Provides financial data tools including fundamentals, news, prices, technical indicators, and more.

**Connection Details:**
- HTTP Mode: `http://localhost:8000`
- Available Tools: 40+ financial data endpoints

## Managing Services

```bash
# Stop all services
docker-compose down

# Restart a specific service
docker-compose restart postgres
docker-compose restart mcp

# View service logs
docker-compose logs -f postgres
docker-compose logs -f mcp

# Rebuild after changes
docker-compose up -d --build

# Stop and remove all data (WARNING: deletes database)
docker-compose down -v
```

## Development

### ADK Web UI Features

The ADK Web UI provides:
- **Interactive Chat Interface**: Chat with agents in real-time through a web browser
- **Session Management**: Persistent sessions stored in PostgreSQL
- **Long-term Memory**: Agents remember past conversations via the PostgreSQL memory service
- **Hot Reload**: Changes to agent code are automatically reloaded (when using `--reload_agents` flag)
- **Multi-agent Support**: Run multiple agents simultaneously from the `agents/` directory

### Creating New Agents

To create a new agent:

1. Create a new directory under `agents/`:
   ```bash
   mkdir -p agents/my_new_agent
   ```

2. Create `agents/my_new_agent/__init__.py`:
   ```python
   from .agent import root_agent, app_name
   __all__ = ["root_agent", "app_name"]
   ```

3. Create `agents/my_new_agent/agent.py`:
   ```python
   from google.adk.agents import Agent
   from google.adk.models.google_llm import Gemini

   # Define the root agent for ADK Web UI
   root_agent = Agent(
       name="my_new_agent",
       model=Gemini(model="gemini-2.5-flash-lite"),
       instruction="Your agent instructions here...",
       tools=[],  # Add tools here
   )

   # For ADK Web UI
   app_name = "my_new_agent"
   ```

4. Restart the web UI to see your new agent

### Running Different Agents

The main entry point is configured in `agententerpriseval/__main__.py`:

```bash
# Currently runs stocks.py (CLI mode)
python -m agententerpriseval
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy agententerpriseval/
```

### Agent Configuration

All agents use:
- **Model**: Gemini 2.5 Flash Lite (configurable)
- **State Management**: InMemoryRunner (can be switched to PostgreSQL-backed runner)
- **Retry Config**: 5 attempts with exponential backoff for API calls

## Examples

### Financial Valuation Agent (stocks.py)

Performs comprehensive enterprise valuations using EODHD data:

```python
# The agent connects to:
# 1. PostgreSQL for state persistence
# 2. EODHD MCP server for financial data tools
# 3. Gemini API for AI processing

python -m agententerpriseval
```

The agent performs:
- Company data collection and normalization
- Financial forecasting with multiple scenarios
- WACC (Weighted Average Cost of Capital) calculation
- DCF (Discounted Cash Flow) valuation
- Multiples analysis and peer comparison
- Comprehensive valuation report generation

## Troubleshooting

### Port Conflicts

If ports 5432 or 8000 are already in use:

```bash
# Check what's using the port
lsof -i :5432
lsof -i :8000

# Modify ports in docker-compose.yml if needed
```

### Database Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U postgres -d agent_state -c "SELECT version();"

# Check service health
docker-compose ps
docker-compose logs postgres
```

### MCP Server Issues

```bash
# Check MCP server logs
docker-compose logs mcp

# Verify EODHD_API_KEY is set
echo $EODHD_API_KEY

# Test MCP server endpoint
curl http://localhost:8000/
```

### Agent Connection Issues

Make sure:
1. Docker services are running: `docker-compose ps`
2. Environment variables are set in `.env`
3. Virtual environment is activated: `poetry shell`

## Additional Documentation

- **[WEB_UI_QUICKSTART.md](./WEB_UI_QUICKSTART.md)** - Quick start guide for the ADK Web UI (start here!)
- **[CODE_REUSE.md](./CODE_REUSE.md)** - How agent code is shared between Web UI and CLI
- **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - Understanding the new ADK structure
- [agents/financial_assistant/README.md](./agents/financial_assistant/README.md) - Financial assistant agent documentation
- [CLAUDE.md](./CLAUDE.md) - Project guidance for Claude Code
- [DOCKER_README.md](./DOCKER_README.md) - Detailed Docker setup documentation

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]