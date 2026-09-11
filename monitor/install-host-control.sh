#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
HELPER=/usr/local/sbin/ai6-hostctl
SUDOERS=/etc/sudoers.d/ai6-monitor-hostctl

sudo install -m 0755 "$ROOT_DIR/monitor/ai6-hostctl" "$HELPER"
printf '%s ALL=(root) NOPASSWD: %s reboot, %s poweroff\n' "$USER_NAME" "$HELPER" "$HELPER" | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS" >/dev/null

echo "PASS: AI6 monitor host controls installed"
echo "Allowed commands only:"
echo "  sudo $HELPER reboot"
echo "  sudo $HELPER poweroff"
