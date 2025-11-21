# Docker Setup for ADK Playground

This docker-compose configuration runs both the PostgreSQL database and EODHD MCP server together.

## Prerequisites

1. Docker and Docker Compose installed
2. `.env` file with required API keys (see `.env.example`)

## Quick Start

### 1. Set up environment variables

```bash
# Copy the example and fill in your API keys
cp .env.example .env
# Edit .env and add your EODHD_API_KEY and GOOGLE_API_KEY
```

### 2. Start all services

```bash
# Build and start both PostgreSQL and MCP server
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f mcp
```

### 3. Verify services are running

```bash
# Check service status
docker-compose ps

# Test PostgreSQL connection from host
psql -h localhost -U postgres -d agent_state -c "SELECT version();"

# Test MCP server (once health check endpoint is available)
curl http://localhost:8000/
```

## Service Details

### PostgreSQL Database
- **Host**: `localhost` (from host machine) or `postgres` (from other containers)
- **Port**: `5432`
- **Database**: `agent_state`
- **User**: `postgres`
- **Password**: None (trust authentication)
- **Connection String**: `postgresql://postgres@localhost:5432/agent_state`

The database is accessible from:
- Your local Python agents via `localhost:5432`
- Other Docker containers via `postgres:5432`

### EODHD MCP Server
- **Host**: `localhost` (from host machine) or `mcp` (from other containers)
- **Port**: `8000`
- **HTTP Mode**: Enabled by default

## Managing Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes database data)
docker-compose down -v

# Restart a specific service
docker-compose restart postgres
docker-compose restart mcp

# Rebuild after code changes
docker-compose up -d --build

# View container resource usage
docker stats adk_postgres adk_mcp
```

## Connecting from Your Python Agent

### PostgreSQL Connection

```python
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="agent_state",
    user="postgres"
    # no password needed
)
```

Or using SQLAlchemy:

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres@localhost:5432/agent_state")
```

### MCP Server Connection

The MCP server is accessible at `http://localhost:8000` and can be connected via stdio or HTTP as shown in `stocks.py`.

## Database Schema

The PostgreSQL database comes pre-initialized with tables for:
- `sessions` - User session tracking
- `agent_state` - Agent state snapshots
- `conversation_history` - Message history with timestamps

See `init-db.sql` for complete schema details.

## Troubleshooting

### Port conflicts
If ports 5432 or 8000 are already in use:
```bash
# Check what's using the port
lsof -i :5432
lsof -i :8000

# Modify docker-compose.yml to use different host ports
# Change "5432:5432" to "5433:5432" for example
```

### MCP health check failing
The health check uses curl to test the `/health` endpoint. If your MCP server doesn't have this endpoint, you can remove or modify the health check in `docker-compose.yml`.

### Database connection refused
Make sure the PostgreSQL service is healthy:
```bash
docker-compose ps
docker-compose logs postgres
```

### Environment variable issues
Ensure your `.env` file is in the same directory as `docker-compose.yml` and contains `EODHD_API_KEY`.

## Development Workflow

1. Start services: `docker-compose up -d`
2. Run your Python agent from the host: `python -m googleadkplayground`
3. Agent connects to:
   - PostgreSQL at `localhost:5432`
   - MCP server at `localhost:8000`
4. Monitor logs: `docker-compose logs -f`
5. Stop services when done: `docker-compose down`
