#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

USER_NAME="$(id -un)"
HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
PORT="${AI6_WEBTERM_PORT:-8091}"
ENV_FILE="/etc/ai6-web-terminal.env"
SERVICE_FILE="/etc/systemd/system/ai6-web-terminal.service"

echo "Installing ttyd..."
sudo apt update
sudo apt install -y ttyd

if ! command -v ttyd >/dev/null 2>&1; then
  echo "FAIL: ttyd not found after installation"
  exit 1
fi

if [[ -n "${AI6_WEBTERM_PASSWORD:-}" ]]; then
  PASS="$AI6_WEBTERM_PASSWORD"
else
  PASS="$(openssl rand -hex 12)"
fi

sudo bash -c "umask 077; cat > '$ENV_FILE'" <<EOF
AI6_WEBTERM_USER=$USER_NAME
AI6_WEBTERM_PASSWORD=$PASS
AI6_WEBTERM_PORT=$PORT
EOF

sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=AI6 Web Terminal (ttyd)
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$HOME_DIR
EnvironmentFile=$ENV_FILE
ExecStart=/usr/bin/ttyd --interface 0.0.0.0 --port \${AI6_WEBTERM_PORT} --writable --credential \${AI6_WEBTERM_USER}:\${AI6_WEBTERM_PASSWORD} /bin/bash -l
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now ai6-web-terminal

sleep 1
if ! systemctl is-active --quiet ai6-web-terminal; then
  echo "FAIL: ai6-web-terminal did not start"
  sudo systemctl --no-pager --full status ai6-web-terminal || true
  exit 1
fi

LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
[[ -n "$LAN_IP" ]] || LAN_IP="$(hostname -I | awk '{print $1}')"

echo
echo "PASS: AI6 Web Terminal is running"
echo "URL:      http://${LAN_IP}:${PORT}/"
echo "User:     $USER_NAME"
echo "Password: $PASS"
echo
echo "Credentials are also stored in $ENV_FILE (root-readable)."
echo "Do not expose this port directly to the Internet."
