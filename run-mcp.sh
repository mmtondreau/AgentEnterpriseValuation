set -eu
# Build image
docker build -f Dockerfile.eodmcp -t eodhd-mcp .

# Run container, passing secrets at runtime
docker run --rm \
  --name eodhd-mcp-server \
  -e EODHD_API_KEY=${EODHD_API_KEY} \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  -p 8000:8000 \
  eodhd-mcp