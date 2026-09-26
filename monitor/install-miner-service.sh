#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
USER_HOME="$HOME"
RUNNER=/usr/local/lib/ai6-monitor/ai6-miner-run.py
CTL=/usr/local/sbin/ai6-minerctl
UNIT=/etc/systemd/system/ai6-miner.service
SUDOERS=/etc/sudoers.d/ai6-monitor-miner

STATE_DIR="$USER_HOME/.local/state/ai6-monitor"
LOG_FILE="$STATE_DIR/forge-miner.log"

mkdir -p "$STATE_DIR"
sudo install -d -m 0755 /usr/local/lib/ai6-monitor
sudo install -m 0755 "$ROOT_DIR/monitor/ai6-miner-run.py" "$RUNNER"
sudo install -m 0755 "$ROOT_DIR/monitor/ai6-minerctl" "$CTL"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
cat >"$tmp" <<EOF
[Unit]
Description=AI6 ForgeMiner
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$USER_NAME
Environment=HOME=$USER_HOME
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 $RUNNER
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
KillMode=control-group
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$tmp" "$UNIT"
{
  printf '%s ALL=(root) NOPASSWD: %s start\n' "$USER_NAME" "$CTL"
  printf '%s ALL=(root) NOPASSWD: %s stop\n' "$USER_NAME" "$CTL"
  printf '%s ALL=(root) NOPASSWD: %s restart\n' "$USER_NAME" "$CTL"
  printf '%s ALL=(root) NOPASSWD: %s status\n' "$USER_NAME" "$CTL"
} | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS" >/dev/null

sudo systemctl daemon-reload

echo "PASS: separate AI6 miner service installed"
echo "Service: ai6-miner.service"
echo "Log:     $LOG_FILE"
echo
echo "The service is intentionally NOT enabled at boot."
echo "Start/stop it from the AI6 Monitor Miner tab."
