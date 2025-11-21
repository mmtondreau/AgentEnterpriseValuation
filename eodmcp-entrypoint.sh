#!/usr/bin/env sh
set -e

# Ensure the API key is provided at runtime
if [ -z "$EODHD_API_KEY" ]; then
  echo "ERROR: EODHD_API_KEY environment variable is required." >&2
  exit 1
fi

# Generate .env for the MCP server from runtime env vars
cat > .env <<EOF
EODHD_API_KEY=${EODHD_API_KEY}
MCP_HOST=${MCP_HOST:-0.0.0.0}
MCP_PORT=${MCP_PORT:-8000}
EOF

echo "Starting EODHD MCP server on ${MCP_HOST:-0.0.0.0}:${MCP_PORT:-8000}..."

exec python server.py --http
