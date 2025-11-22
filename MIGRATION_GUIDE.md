# Migration Guide: ADK Web UI Structure

This guide explains the restructuring of the Google ADK Playground to support the ADK Web UI.

## What Changed

### New Directory Structure

The project has been reorganized to follow Google ADK's recommended structure:

```
Before:                              After:
googleadkplayground/                 agents/
├── stocks.py                        └── financial_assistant/
├── postgres_memory_service.py           ├── __init__.py
└── __main__.py                          ├── agent.py
                                         └── README.md

                                     services/
                                     ├── __init__.py
                                     └── postgres_memory_service.py

                                     googleadkplayground/
                                     ├── stocks.py (legacy)
                                     └── __main__.py
```

### Key Changes

1. **Agents Directory** (`agents/`)
   - New top-level directory for ADK agents
   - Each agent is a subdirectory with `__init__.py` and `agent.py`
   - Enables ADK Web UI to discover and load agents

2. **Services Directory** (`services/`)
   - Shared services moved here
   - `PostgresMemoryService` relocated from `googleadkplayground/`
   - Can be imported by multiple agents

3. **Agent Definition** (`agents/financial_assistant/agent.py`)
   - Agent code extracted from `stocks.py`
   - Exports `agent` and `app_name` for ADK Web UI
   - Maintains same functionality (MCP tools, memory, etc.)

4. **Web UI Launcher** (`run_web_ui.sh`)
   - New script to start ADK Web UI
   - Configures PostgreSQL session and memory services
   - Runs on http://localhost:8080

5. **Configuration** (`agents_config.py`)
   - Centralized configuration for services
   - Memory service factory function
   - Environment variable management

## Migration Steps

If you're updating from the old structure:

### 1. Update Imports

If you have code importing from the old locations:

```python
# Old
from googleadkplayground.postgres_memory_service import PostgresMemoryService

# New
from services import PostgresMemoryService
```

### 2. Run the Web UI

Instead of running the CLI:

```bash
# Old
python -m googleadkplayground

# New - Web UI (recommended)
./run_web_ui.sh

# Or run manually
python -m google.adk.cli web agents \
  --session_service_uri="postgresql://postgres@localhost:5432/agent_state" \
  --port 8080
```

### 3. Access the Web Interface

Open your browser to: http://localhost:8080

## Benefits of New Structure

1. **Web UI Support**: Native integration with Google ADK Web UI
2. **Multi-agent**: Easily add multiple agents to the `agents/` directory
3. **Hot Reload**: Changes to agents reload automatically
4. **Better Organization**: Clear separation of agents, services, and legacy code
5. **Scalability**: Standard structure makes it easier to add features

## Backward Compatibility

The old CLI mode still works:

```bash
python -m googleadkplayground
```

However, we recommend using the Web UI for the best experience.

## Database & Services

No changes to infrastructure:
- PostgreSQL still runs on `localhost:5432`
- EODHD MCP server still on `localhost:8000`
- Same `docker-compose up -d` command

Session and memory storage remain in PostgreSQL as before.

## Next Steps

1. **Try the Web UI**: Run `./run_web_ui.sh` and explore the interface
2. **Create New Agents**: Follow the guide in README.md
3. **Migrate Custom Code**: Move any custom agents to the `agents/` structure
4. **Remove Legacy Code**: Once migrated, you can remove `googleadkplayground/stocks.py`

## Troubleshooting

### Web UI doesn't start
- Ensure PostgreSQL is running: `docker-compose ps`
- Check the database URL in `run_web_ui.sh`
- Verify port 8080 is available

### Agent not appearing
- Check `agents/` directory structure
- Ensure `__init__.py` and `agent.py` exist
- Verify `agent` and `app` are exported in `__init__.py`

### Memory service errors
- Confirm PostgreSQL has `memory_entries` table
- Check async database URL uses `postgresql+asyncpg://`

## Questions?

Refer to:
- Main README.md for setup instructions
- agents/financial_assistant/README.md for agent examples
- Google ADK documentation for advanced features
