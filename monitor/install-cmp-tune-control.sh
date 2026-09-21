#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
HELPER=/usr/local/sbin/ai6-cmptune
CMP_TUNE=/usr/local/sbin/cmp-tune
SUDOERS=/etc/sudoers.d/ai6-monitor-cmptune

if [[ ! -x "$CMP_TUNE" ]]; then
  echo "cmp-tune is not installed at $CMP_TUNE"
  echo "Install cmp-tune first, then rerun this script."
  exit 2
fi

sudo install -m 0755 "$ROOT_DIR/monitor/ai6-cmptune" "$HELPER"
{
  printf '%s ALL=(root) NOPASSWD: %s apply *\n' "$USER_NAME" "$HELPER"
  printf '%s ALL=(root) NOPASSWD: %s reset\n' "$USER_NAME" "$HELPER"
} | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS" >/dev/null

echo "PASS: restricted AI6 cmp-tune control installed"
echo "Allowed:"
echo "  sudo $HELPER apply <profile>"
echo "  sudo $HELPER reset"
