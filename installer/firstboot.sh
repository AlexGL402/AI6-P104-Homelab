#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/var/log/ai6-firstboot.log
exec > >(tee -a "$LOG") 2>&1

echo "=== AI6 first boot: $(date -Is) ==="

TARGET_USER="$(awk -F: '$3 >= 1000 && $3 < 65534 && $1 != "nobody" {print $1; exit}' /etc/passwd)"
if [[ -z "$TARGET_USER" ]]; then
  echo "ERROR: no normal user found"
  exit 1
fi
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl ca-certificates python3 python3-venv python3-pip pciutils jq sudo ubuntu-drivers-common docker.io docker-compose-v2
systemctl enable --now docker
usermod -aG docker "$TARGET_USER" || true

STATE_DIR=/var/lib/ai6-installer
mkdir -p "$STATE_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  if lspci -nn 2>/dev/null | grep -qiE 'NVIDIA.*CMP|3D controller.*10de:1e09'; then
    echo "CMP GPU detected."
    echo "Refusing Ubuntu auto-driver selection: AI6 CMP baseline is NVIDIA 610.43.03 with MIT/GPL Open Kernel Modules."
    echo "Install the pinned 610.43.03 driver using installer/README_RU.md, reboot, then rerun this script."
    exit 20
  fi

  echo "No CMP GPU detected; installing recommended Ubuntu NVIDIA compute driver..."
  ubuntu-drivers install --gpgpu || ubuntu-drivers install || true
  touch "$STATE_DIR/driver-installed"
  if systemctl cat ai6-firstboot.service >/dev/null 2>&1; then
    systemctl enable ai6-firstboot.service
  fi
  systemctl reboot
  exit 0
fi

nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader || true

REPO="$TARGET_HOME/AI6-P104-Homelab"
BRANCH="feature/ai6-installer"

if [[ ! -d "$REPO/.git" ]]; then
  sudo -u "$TARGET_USER" git clone --branch "$BRANCH" --single-branch https://github.com/AlexGL402/AI6-P104-Homelab.git "$REPO"
else
  sudo -u "$TARGET_USER" git -C "$REPO" fetch origin "$BRANCH"
  sudo -u "$TARGET_USER" git -C "$REPO" switch "$BRANCH"
  sudo -u "$TARGET_USER" git -C "$REPO" pull --ff-only
fi
chown -R "$TARGET_USER:$TARGET_USER" "$REPO"

MON="$REPO/monitor"
VENV="$MON/.venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$MON/requirements.txt"
install -d -o "$TARGET_USER" -g "$TARGET_USER" -m 0755 /var/lib/ai6-monitor

cat >/etc/systemd/system/ai6-monitor.service <<EOF
[Unit]
Description=AI6 Host Resource Monitor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$TARGET_USER
Group=$TARGET_USER
WorkingDirectory=$MON
Environment=PYTHONUNBUFFERED=1
Environment=AI6_MONITOR_CSV=/var/lib/ai6-monitor/psu-test.csv
ExecStart=$VENV/bin/uvicorn ai6_monitor_auto:app --host 0.0.0.0 --port 8090
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

install -m 0755 "$MON/ai6-gpuctl" /usr/local/sbin/ai6-gpuctl
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ai6-gpuctl power-limit *\n' "$TARGET_USER" >/etc/sudoers.d/ai6-monitor-gpuctl
chmod 0440 /etc/sudoers.d/ai6-monitor-gpuctl
visudo -cf /etc/sudoers.d/ai6-monitor-gpuctl

install -m 0755 "$MON/ai6-hostctl" /usr/local/sbin/ai6-hostctl
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ai6-hostctl reboot, /usr/local/sbin/ai6-hostctl poweroff\n' "$TARGET_USER" >/etc/sudoers.d/ai6-monitor-hostctl
chmod 0440 /etc/sudoers.d/ai6-monitor-hostctl
visudo -cf /etc/sudoers.d/ai6-monitor-hostctl

if [[ -x /usr/local/sbin/cmp-tune ]]; then
  install -m 0755 "$MON/ai6-cmptune" /usr/local/sbin/ai6-cmptune
  {
    printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ai6-cmptune apply *\n' "$TARGET_USER"
    printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/ai6-cmptune reset\n' "$TARGET_USER"
  } >/etc/sudoers.d/ai6-monitor-cmptune
  chmod 0440 /etc/sudoers.d/ai6-monitor-cmptune
  visudo -cf /etc/sudoers.d/ai6-monitor-cmptune
fi

systemctl daemon-reload
systemctl enable --now ai6-monitor.service
sleep 3

if ! curl -fsS http://127.0.0.1:8090/health >/dev/null; then
  echo "AI6 monitor health check failed"
  journalctl -u ai6-monitor -n 100 --no-pager || true
  exit 1
fi

# Start repo-managed Open WebUI + Open Terminal stack.
DEPLOY_DIR="$REPO/deploy"
ENV_FILE="$DEPLOY_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  WEBUI_SECRET_KEY="$(openssl rand -hex 32)"
  OPEN_TERMINAL_API_KEY="$(openssl rand -hex 32)"
  cat >"$ENV_FILE" <<EOF
WEBUI_SECRET_KEY=$WEBUI_SECRET_KEY
OPEN_TERMINAL_API_KEY=$OPEN_TERMINAL_API_KEY
EOF
  chown "$TARGET_USER:$TARGET_USER" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi

(
  cd "$DEPLOY_DIR"
  docker compose up -d
)

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "$LAN_IP" ]]; then LAN_IP=HOST_IP; fi

echo "========================================="
echo " AI6 READY"
echo " Monitor:       http://$LAN_IP:8090"
echo " Open WebUI:    http://$LAN_IP:3000"
echo " Open Terminal: http://$LAN_IP:8000"
echo " SSH:           ssh $TARGET_USER@$LAN_IP"
echo "========================================="

touch "$STATE_DIR/complete"
if systemctl cat ai6-firstboot.service >/dev/null 2>&1; then
  systemctl disable ai6-firstboot.service
fi
