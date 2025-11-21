# Google ADK Playground

A playground for experimenting with Google's Agent Development Kit (ADK) integrated with the EODHD (EOD Historical Data) Market Context Protocol (MCP) server. Build AI agents using Google's Gemini models for financial data analysis and other tasks.

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
cd GoogleADKPlayground

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

### 4. Run the Agent Service

With the infrastructure running, start your agent:

```bash
# Run the main application (currently runs stocks.py)
python -m googleadkplayground
```

The agent will:
- Connect to the PostgreSQL database at `localhost:5432`
- Connect to the EODHD MCP server at `localhost:8000`
- Execute the financial analysis workflow

## Project Structure

```
GoogleADKPlayground/
├── googleadkplayground/          # Main application code
│   ├── stocks.py                 # Financial analysis agent
│   └── __main__.py              # Entry point
├── EODHD_MCP_server/            # EODHD MCP server (cloned submodule)
├── conf/                        # Hydra configuration files
│   └── local.yml               # Local config (API keys)
├── docker-compose.yml           # Multi-service orchestration
├── Dockerfile.postgres          # PostgreSQL database image
├── Dockerfile.eodmcp           # EODHD MCP server image
├── init-db.sql                 # Database schema initialization
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

### Running Different Agents

The main entry point is configured in `googleadkplayground/__main__.py`:

```bash
# Currently runs stocks.py
python -m googleadkplayground
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy googleadkplayground/
```

### Agent Configuration

All agents use:
- **Model**: Gemini 2.5 Flash Lite (configurable)
- **State Management**: InMemoryRunner (can be switched to PostgreSQL-backed runner)
- **Retry Config**: 5 attempts with exponential backoff for API calls

## Examples

### Financial Analysis Agent (stocks.py)

Analyzes stocks using EODHD data:

```python
# The agent connects to:
# 1. PostgreSQL for state persistence
# 2. EODHD MCP server for financial data tools
# 3. Gemini API for AI processing

python -m googleadkplayground
```

The agent can:
- Fetch company fundamentals
- Analyze historical prices
- Get real-time news
- Calculate technical indicators
- Generate investment insights

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

- [CLAUDE.md](./CLAUDE.md) - Project guidance for Claude Code
- [DOCKER_README.md](./DOCKER_README.md) - Detailed Docker setup documentation

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]