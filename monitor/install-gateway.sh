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
