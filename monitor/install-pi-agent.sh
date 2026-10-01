#!/usr/bin/env bash
set -Eeuo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/systemd/ai6-pi-agent.service"
DST="/etc/systemd/system/ai6-pi-agent.service"

sudo install -m 0644 "$SRC" "$DST"
sudo systemctl daemon-reload
sudo systemctl enable --now ai6-pi-agent

systemctl --no-pager --full status ai6-pi-agent
