#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run this script as your normal user; it will use sudo when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR/monitor/ai6-monitor.py"
SERVICE_SRC="$ROOT_DIR/monitor/ai6-monitor.service"
DEST_DIR=/opt/ai6-monitor
SERVICE_DEST=/etc/systemd/system/ai6-monitor.service

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

sudo install -d -m 0755 "$DEST_DIR"
sudo install -m 0755 "$SRC" "$DEST_DIR/ai6-monitor.py"
sudo install -m 0644 "$SERVICE_SRC" "$SERVICE_DEST"

sudo systemctl daemon-reload
sudo systemctl enable --now ai6-monitor

sleep 1
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
echo "JSON API:  http://${LAN_IP}:8090/api/stats"
echo
curl -fsS http://127.0.0.1:8090/api/stats | python3 -m json.tool || true
