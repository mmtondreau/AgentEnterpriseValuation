#!/bin/bash
set -e

# Start ADK Web UI with PostgreSQL session and memory services

echo "Starting Google ADK Web UI..."
echo "Make sure PostgreSQL and MCP servers are running (docker-compose up -d)"
echo ""

# Database URLs
SESSION_DB_URL="postgresql://postgres@localhost:5432/agent_state"
MEMORY_SERVICE_URI="postgresql+asyncpg://postgres@localhost:5432/agent_state"

# Start the web UI
# To add custom branding, uncomment and set both logo flags:
#   --logo-text "Your Company Name" \
#   --logo-image-url "https://example.com/logo.png" \
python -m google.adk.cli web agents \
  --host 0.0.0.0 \
  --port 8080 \
  --session_service_uri "${SESSION_DB_URL}" \
  --reload \
  --reload_agents

echo ""
echo "Web UI stopped."
