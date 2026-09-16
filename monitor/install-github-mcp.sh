#!/usr/bin/env bash
set -Eeuo pipefail

NAME="ai6-github-mcp"
IMAGE="ghcr.io/github/github-mcp-server:latest"
HOST_PORT="${AI6_GITHUB_MCP_PORT:-8108}"
CONTAINER_PORT="8082"
TOOLSETS="${AI6_GITHUB_MCP_TOOLSETS:-repos,issues,pull_requests}"

command -v docker >/dev/null 2>&1 || {
  echo "FAIL: docker not found"
  exit 1
}

echo "Pulling $IMAGE ..."
docker pull "$IMAGE"

echo "Replacing $NAME ..."
docker rm -f "$NAME" >/dev/null 2>&1 || true

# Start the official GitHub MCP server in Streamable HTTP mode.
# Authentication is supplied per request by the MCP client (Open WebUI) via
# Authorization: Bearer <GitHub token>, so no GitHub token is stored here.
# Read-only + lockdown are intentional for the first stage of the coding agent.
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e GITHUB_READ_ONLY=1 \
  -e GITHUB_LOCKDOWN_MODE=1 \
  -e "GITHUB_TOOLSETS=${TOOLSETS}" \
  "$IMAGE" http >/dev/null

sleep 2

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "FAIL: $NAME did not start"
  docker logs --tail 100 "$NAME" || true
  exit 1
fi

echo
echo "PASS: GitHub MCP is running"
echo "Container: $NAME"
echo "Host URL:  http://127.0.0.1:${HOST_PORT}"
echo "Open WebUI URL: http://host.docker.internal:${HOST_PORT}"
echo "Mode: read-only + lockdown"
echo "Toolsets: ${TOOLSETS}"
echo
echo "Open WebUI: Admin Settings -> Integrations -> External Tool Servers -> Add"
echo "Type: MCP (Streamable HTTP)"
echo "URL:  http://host.docker.internal:${HOST_PORT}"
echo "Auth header: Authorization: Bearer <YOUR_GITHUB_TOKEN>"
echo
echo "Do NOT put the token in this repository or shell history if you can avoid it."
echo "Use a fine-grained token scoped only to the repositories you want the agent to read."
