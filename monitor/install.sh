#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run this script as your normal user; it will use sudo when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MON_DIR="$ROOT_DIR/monitor"
VENV="$MON_DIR/.venv"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
SERVICE_DEST=/etc/systemd/system/ai6-monitor.service
CSV_DIR=/var/lib/ai6-monitor

for cmd in python3 nvidia-smi systemctl curl sudo; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "FAIL: required command not found: $cmd"
    exit 1
  fi
done

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "Detected NVIDIA GPUs: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "FAIL: no NVIDIA GPUs detected"
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "FAIL: python3-venv is missing. Install it with:"
  echo "  sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$MON_DIR/requirements.txt"

sudo install -d -o "$USER_NAME" -g "$GROUP_NAME" -m 0755 "$CSV_DIR"

TMP_SERVICE=$(mktemp)
trap 'rm -f "$TMP_SERVICE"' EXIT
cat > "$TMP_SERVICE" <<EOF
[Unit]
Description=AI6 Host Resource Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
WorkingDirectory=$MON_DIR
Environment=PYTHONUNBUFFERED=1
Environment=AI6_MONITOR_CSV=$CSV_DIR/psu-test.csv
ExecStart=$VENV/bin/uvicorn ai6_monitor:app --host 0.0.0.0 --port 8090
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$TMP_SERVICE" "$SERVICE_DEST"
sudo systemctl daemon-reload
sudo systemctl enable --now ai6-monitor

sleep 2
if ! systemctl is-active --quiet ai6-monitor; then
  echo "FAIL: ai6-monitor did not start"
  sudo systemctl --no-pager --full status ai6-monitor || true
  exit 1
fi

if ! curl -fsS http://127.0.0.1:8090/health >/dev/null; then
  echo "FAIL: health endpoint did not respond"
  sudo journalctl -u ai6-monitor -n 50 --no-pager
  exit 1
fi

LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -n "$LAN_IP" ]] || LAN_IP="SERVER_IP"

echo
echo "PASS: AI6 Host Monitor is running"
echo "Dashboard: http://${LAN_IP}:8090/"
echo "Swagger:   http://${LAN_IP}:8090/docs"
echo "JSON API:  http://${LAN_IP}:8090/api/stats"
echo "PSU CSV:   $CSV_DIR/psu-test.csv"
echo
echo "GPU check:"
curl -fsS http://127.0.0.1:8090/api/stats | python3 -m json.tool || true
