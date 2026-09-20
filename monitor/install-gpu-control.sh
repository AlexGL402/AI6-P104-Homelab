#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
HELPER=/usr/local/sbin/ai6-gpuctl
SUDOERS=/etc/sudoers.d/ai6-monitor-gpuctl

sudo install -m 0755 "$ROOT_DIR/monitor/ai6-gpuctl" "$HELPER"
printf '%s ALL=(root) NOPASSWD: %s power-limit *\n' "$USER_NAME" "$HELPER" | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS" >/dev/null

echo "PASS: restricted AI6 GPU power-limit control installed"
echo "Allowed pattern:"
echo "  sudo $HELPER power-limit <gpu-index> <watts>"
