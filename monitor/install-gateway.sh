#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MON_DIR="$ROOT_DIR/monitor"
VENV="$MON_DIR/.venv"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
SERVICE=/etc/systemd/system/ai6-model-gateway.service

[[ -x "$VENV/bin/python" ]] || { echo "Monitor venv missing: run ./monitor/install.sh first"; exit 1; }

"$VENV/bin/pip" install -r "$MON_DIR/requirements.txt"

# Web tools + read-only GitHub MCP tools used by the local coding agent.
# The GitHub MCP container itself is also started in read-only mode, so this
# allow-list is defense in depth and prevents unrelated Open WebUI tools such
# as ask_user from reaching llama.cpp.
ALLOWED_TOOLS="search_web,fetch_url,web_search,fetch_webpage,get_file_contents,search_code,search_repositories,list_branches,list_commits,get_commit,issue_read,get_issue,list_issues,get_issue_comments,search_issues,pull_request_read,list_pull_requests,search_pull_requests,get_latest_release,get_release_by_tag,list_releases,list_tags,get_tag,github_get_file_contents,github_search_code,github_search_repositories,github_list_branches,github_list_commits,github_get_commit,github_issue_read,github_get_issue,github_list_issues,github_get_issue_comments,github_search_issues,github_pull_request_read,github_list_pull_requests,github_search_pull_requests,github_get_latest_release,github_get_release_by_tag,github_list_releases,github_list_tags,github_get_tag,run_command,get_process_status"

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
cat >"$TMP" <<EOF
[Unit]
Description=AI6 OpenAI-compatible Model Gateway
After=network-online.target ai6-monitor.service
Wants=network-online.target
Requires=ai6-monitor.service

[Service]
Type=simple
User=${USER_NAME}
Group=${GROUP_NAME}
WorkingDirectory=${MON_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=AI6_MONITOR_URL=http://127.0.0.1:8090
Environment="AI6_GATEWAY_ALLOWED_TOOLS=${ALLOWED_TOOLS}"
ExecStart=${VENV}/bin/uvicorn ai6_model_gateway:app --host 0.0.0.0 --port 8099
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$TMP" "$SERVICE"
sudo systemctl daemon-reload
sudo systemctl enable --now ai6-model-gateway
sudo systemctl restart ai6-model-gateway

sleep 2
if ! curl -fsS http://127.0.0.1:8099/health; then
  echo
  echo "FAIL: gateway health check failed"
  sudo journalctl -u ai6-model-gateway -n 80 --no-pager
  exit 1
fi

echo
echo "PASS: AI6 model gateway is running"
echo "OpenAI base URL: http://$(hostname -I | awk '{print $1}'):8099/v1"
echo "Models:"
curl -fsS http://127.0.0.1:8099/v1/models | "$VENV/bin/python" -m json.tool
