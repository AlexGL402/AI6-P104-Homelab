#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  echo "Run as your normal user; sudo will be used when needed."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
ENV_FILE=/etc/ai6-worker-profiles.env
HELPER=/usr/local/sbin/ai6-workerctl
SUDOERS=/etc/sudoers.d/ai6-monitor-workerctl

[[ -x "$LLAMA_BIN" ]] || { echo "FAIL: llama-server not found: $LLAMA_BIN"; exit 1; }

sudo install -m 0755 "$ROOT_DIR/monitor/ai6-workerctl" "$HELPER"

if [[ ! -f "$ENV_FILE" ]]; then
  sudo install -m 0644 "$ROOT_DIR/configs/ai6-worker-profiles.env.example" "$ENV_FILE"
fi

make_unit() {
  local name="$1" gpus="$2" port="$3" alias="$4"
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
[Unit]
Description=llama.cpp 2-GPU worker ${gpus} on ${port}
After=network.target

[Service]
User=${USER_NAME}
Group=${GROUP_NAME}
WorkingDirectory=${HOME}/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=${gpus}"
Environment="NCCL_DEBUG=WARN"
EnvironmentFile=${ENV_FILE}
ExecStart=/bin/bash -lc 'exec "${LLAMA_BIN}" -m "\$MODEL_2GPU" -ngl all -sm tensor -ts 1,1 -c "\${CTX_2GPU:-32768}" --alias ${alias} --host 0.0.0.0 --port ${port}'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
  sudo install -m 0644 "$tmp" "/etc/systemd/system/${name}.service"
  rm -f "$tmp"
}

make_unit llama2-8081 "0,1" 8081 "qwen-2gpu-01"
make_unit llama2-8082 "2,3" 8082 "qwen-2gpu-23"
make_unit llama2-8083 "4,5" 8083 "qwen-2gpu-45"

printf '%s ALL=(root) NOPASSWD: %s *\n' "$USER_NAME" "$HELPER" | sudo tee "$SUDOERS" >/dev/null
sudo chmod 0440 "$SUDOERS"
sudo visudo -cf "$SUDOERS" >/dev/null

sudo systemctl daemon-reload
sudo systemctl disable llama2-8081 llama2-8082 llama2-8083 >/dev/null 2>&1 || true

echo
echo "PASS: worker control installed"
echo "Current 3+3 profile remains active."
echo "2+2+2 profile units are installed but NOT started."
echo
echo "To enable 2+2+2 later:"
echo "  sudo nano $ENV_FILE"
echo "Set MODEL_2GPU to a smaller GGUF that fits in 2x8 GB."
echo
echo "Control helper:"
echo "  sudo $HELPER status"
echo "  sudo $HELPER profile 33"
echo "  sudo $HELPER profile 222"
