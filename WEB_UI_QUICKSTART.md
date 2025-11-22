# ADK Web UI Quick Start Guide

Get started with the Google ADK Web UI in under 5 minutes!

## Prerequisites

Make sure you have:
- ✅ PostgreSQL and MCP server running (`docker-compose up -d`)
- ✅ Environment variables configured (`.env` file)
- ✅ Dependencies installed (`poetry install`)

## Start the Web UI

```bash
./run_web_ui.sh
```

You should see:
```
Starting Google ADK Web UI...
Make sure PostgreSQL and MCP servers are running (docker-compose up -d)

INFO:     Started server process
ADK Web Server started
For local testing, access at http://0.0.0.0:8080.
```

## Access the Interface

Open your browser to: **http://localhost:8080**

## Using the Web UI

### 1. Select an Agent

On the home page, you'll see available agents:
- **financial_assistant** - Analyzes stocks using EODHD data

Click on an agent to start a conversation.

### 2. Start a Conversation

The chat interface allows you to:
- Type messages to the agent
- View agent responses in real-time
- See tool calls and function executions
- Access conversation history

### 3. Try Sample Queries

For the financial assistant:

```
Get the latest fundamental data on AAPL and provide a summary of its recent performance.
```

```
What's the current sentiment on Tesla stock?
```

```
Compare the fundamentals of MSFT and GOOGL.
```

### 4. Session Management

- Sessions are automatically saved to PostgreSQL
- Resume previous conversations by session ID
- Long-term memory preserves context across sessions

## Web UI Features

### Interactive Chat
- Real-time streaming responses
- Markdown rendering
- Code syntax highlighting

### Tool Execution Visibility
- See which tools the agent calls
- View tool parameters and responses
- Debug agent reasoning

### Session Persistence
- All conversations saved to database
- Resume from any point
- Cross-session memory

### Hot Reload (Development)
- Edit agent code in `agents/financial_assistant/agent.py`
- Changes reload automatically (with `--reload_agents` flag)
- No need to restart server

## Configuration

### Change Port

Edit `run_web_ui.sh`:
```bash
--port 8080  # Change to your desired port
```

### Database URL

The web UI uses:
- **Sessions**: `postgresql://postgres@localhost:5432/agent_state`
- **Memory**: Configured via the agent's memory service

### Add Custom Logo

Edit `run_web_ui.sh` and uncomment both logo flags:
```bash
# Both are required if you want custom branding
--logo-text "My Company" \
--logo-image-url "https://example.com/logo.png"
```

**Note**: You must provide BOTH flags or neither. Using only one will cause an error.

## Troubleshooting

### Web UI won't start

**Check PostgreSQL:**
```bash
docker-compose ps
# Should show adk_postgres as "Up"
```

**Check database connection:**
```bash
psql -h localhost -U postgres -d agent_state -c "SELECT 1;"
```

### Agent not appearing

**Verify agent structure:**
```bash
ls agents/financial_assistant/
# Should show: __init__.py  agent.py  README.md
```

**Check agent exports:**
```bash
python -c "from agents.financial_assistant import root_agent; print(root_agent.name)"
# Should print: financial_assistant
```

**Note**: The ADK Web UI looks for `root_agent`, not just `agent`.

### MCP server errors

**Ensure MCP server is running:**
```bash
curl http://localhost:8000/
# Should return MCP server response
```

**Check MCP connection in agent:**
```python
# In agents/financial_assistant/agent.py
# URL should be: http://127.0.0.1:8000/mcp
```

### Memory not persisting

**Check memory_entries table:**
```bash
psql -h localhost -U postgres -d agent_state -c "SELECT COUNT(*) FROM memory_entries;"
```

**Verify async database URL:**
Should use `postgresql+asyncpg://` protocol for memory service.

## Next Steps

1. **Explore the agent** - Try different financial queries
2. **Create a new agent** - Follow the guide in README.md
3. **Customize the UI** - Adjust logo, colors, and branding
4. **Add more tools** - Extend agent capabilities
5. **Deploy to production** - Use the ADK deployment guide

## Stopping the Web UI

Press `Ctrl+C` in the terminal to stop the server.

Or if running in background:
```bash
pkill -f "google.adk.cli web"
```

## Additional Resources

- [Main README](README.md) - Full project documentation
- [Migration Guide](MIGRATION_GUIDE.md) - Understand the new structure
- [Agent README](agents/financial_assistant/README.md) - Agent-specific docs
- [Google ADK Docs](https://cloud.google.com/vertex-ai/docs/generative-ai/agent-development-kit) - Official documentation

---

**Happy building! 🚀**
