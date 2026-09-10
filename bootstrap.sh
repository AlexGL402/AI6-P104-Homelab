#!/usr/bin/env bash
set -Eeuo pipefail

# AI6-P104-Homelab bootstrap helper
# Safe scope: validates prerequisites, installs repo-managed systemd services,
# prepares Docker secrets, starts the Docker stack, and prints PASS/FAIL checks.
# It intentionally does NOT install NVIDIA drivers, CUDA, NCCL, or model files.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="${SUDO_USER:-$USER}"
USER_HOME="$(getent passwd "$CURRENT_USER" | cut -d: -f6)"
LLAMA_DIR="${LLAMA_DIR:-$USER_HOME/llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/var/lib/ollama-models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a}"
DEPLOY_DIR="$REPO_DIR/deploy"
ENV_FILE="$DEPLOY_DIR/.env"
EXPECTED_GPUS="${EXPECTED_GPUS:-6}"

PASS=0
FAIL=0
WARN=0

ok()   { printf '\033[32m[PASS]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN+1)); }
info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }

usage() {
  cat <<'EOF'
AI6-P104-Homelab bootstrap helper

Usage:
  ./bootstrap.sh check
  ./bootstrap.sh install-services
  ./bootstrap.sh docker
  ./bootstrap.sh all

Optional environment overrides:
  MODEL_PATH=/path/to/model.gguf
  LLAMA_DIR=/home/user/llama.cpp
  LLAMA_SERVER=/path/to/llama-server
  EXPECTED_GPUS=6

What it does NOT install:
  - NVIDIA driver
  - CUDA 12.8
  - NCCL
  - model blobs

Those remain explicit prerequisites because GPU/package state varies by host.
EOF
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

check_prereqs() {
  info "Checking host prerequisites"

  if command_exists nvidia-smi; then
    ok "nvidia-smi found"
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$gpu_count" -eq "$EXPECTED_GPUS" ]]; then
      ok "Detected $gpu_count GPUs (expected $EXPECTED_GPUS)"
    else
      bad "Detected $gpu_count GPUs; expected $EXPECTED_GPUS"
    fi
    nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader || true
  else
    bad "nvidia-smi not found"
  fi

  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    ok "CUDA 12.8 nvcc found"
    /usr/local/cuda-12.8/bin/nvcc --version | tail -n 4 || true
  else
    bad "CUDA 12.8 nvcc not found at /usr/local/cuda-12.8/bin/nvcc"
  fi

  if ldconfig -p 2>/dev/null | grep -q 'libnccl.so.2'; then
    ok "NCCL runtime found"
    dpkg -l 2>/dev/null | grep -E 'libnccl|nccl' || true
  else
    bad "NCCL runtime libnccl.so.2 not found"
  fi

  if [[ -x "$LLAMA_SERVER" ]]; then
    ok "llama-server found: $LLAMA_SERVER"
  else
    bad "llama-server not found: $LLAMA_SERVER"
  fi

  if [[ -f "$MODEL_PATH" ]]; then
    ok "Model found: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
  else
    bad "Model not found: $MODEL_PATH"
  fi

  if command_exists docker; then
    ok "Docker found"
    docker --version || true
  else
    bad "Docker not found"
  fi

  if docker compose version >/dev/null 2>&1; then
    ok "Docker Compose plugin found"
  else
    bad "docker compose plugin not found"
  fi

  if command_exists curl; then ok "curl found"; else bad "curl not found"; fi
  if command_exists openssl; then ok "openssl found"; else bad "openssl not found"; fi

  if [[ -r /proc/meminfo ]]; then
    local mem_kb
    mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
    info "RAM: $((mem_kb/1024)) MiB"
  fi

  if command_exists nvidia-smi; then
    info "GPU topology"
    nvidia-smi topo -m || true
  fi
}

render_service() {
  local src="$1" dst="$2" port="$3" devices="$4" alias="$5"
  cat > "$dst" <<EOF
[Unit]
Description=llama.cpp server GPU $devices
After=network.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$LLAMA_DIR
Environment="CUDA_VISIBLE_DEVICES=$devices"
Environment="NCCL_DEBUG=WARN"
ExecStart=$LLAMA_SERVER \\
  -m $MODEL_PATH \\
  -ngl all \\
  -sm tensor \\
  -ts 1,1,1 \\
  -c 32768 \\
  --alias $alias \\
  --host 0.0.0.0 \\
  --port $port
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

install_services() {
  info "Installing llama.cpp systemd workers"

  [[ -x "$LLAMA_SERVER" ]] || { bad "Cannot install services: llama-server missing"; return 1; }
  [[ -f "$MODEL_PATH" ]] || { bad "Cannot install services: model missing"; return 1; }

  local tmp1 tmp2
  tmp1="$(mktemp)"
  tmp2="$(mktemp)"
  trap 'rm -f "$tmp1" "$tmp2"' RETURN

  render_service "$REPO_DIR/configs/llama-8081.service" "$tmp1" 8081 "0,1,2" "qwen3-coder-30b-gpu012"
  render_service "$REPO_DIR/configs/llama-8082.service" "$tmp2" 8082 "3,4,5" "qwen3-coder-30b-gpu345"

  info "Service user: $CURRENT_USER"
  info "llama.cpp dir: $LLAMA_DIR"
  info "model: $MODEL_PATH"

  sudo install -m 0644 "$tmp1" /etc/systemd/system/llama-8081.service
  sudo install -m 0644 "$tmp2" /etc/systemd/system/llama-8082.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now llama-8081 llama-8082

  ok "Installed and enabled llama-8081 + llama-8082"
}

prepare_env() {
  mkdir -p "$DEPLOY_DIR"
  if [[ -f "$ENV_FILE" ]]; then
    warn "$ENV_FILE already exists; keeping existing secrets"
    return 0
  fi

  local webui_secret terminal_key
  webui_secret="$(openssl rand -hex 32)"
  terminal_key="$(openssl rand -hex 32)"

  cat > "$ENV_FILE" <<EOF
WEBUI_SECRET_KEY=$webui_secret
OPEN_TERMINAL_API_KEY=$terminal_key
EOF
  chmod 600 "$ENV_FILE"
  ok "Created deploy/.env with fresh local secrets (not printed)"
}

start_docker() {
  info "Starting Open WebUI + Open Terminal"
  command_exists docker || { bad "Docker missing"; return 1; }
  docker compose version >/dev/null 2>&1 || { bad "Docker Compose plugin missing"; return 1; }

  prepare_env
  (cd "$DEPLOY_DIR" && docker compose up -d)
  ok "Docker stack started"
}

wait_health() {
  local url="$1" label="$2" tries="${3:-90}"
  local i code
  for ((i=1; i<=tries; i++)); do
    code="$(curl -s -o /tmp/ai6-health.$$ -w '%{http_code}' "$url" || true)"
    if [[ "$code" == "200" ]]; then
      ok "$label healthy ($url)"
      rm -f /tmp/ai6-health.$$
      return 0
    fi
    if (( i % 10 == 0 )); then info "$label still starting (HTTP $code, attempt $i/$tries)"; fi
    sleep 2
  done
  rm -f /tmp/ai6-health.$$
  bad "$label health check failed: $url"
  return 1
}

final_checks() {
  info "Running final service checks"

  if systemctl is-active --quiet llama-8081; then ok "llama-8081 active"; else bad "llama-8081 not active"; fi
  if systemctl is-active --quiet llama-8082; then ok "llama-8082 active"; else bad "llama-8082 not active"; fi

  wait_health http://127.0.0.1:8081/health "Worker GPU0-2" 90 || true
  wait_health http://127.0.0.1:8082/health "Worker GPU3-5" 90 || true

  if curl -fsS http://127.0.0.1:8000/openapi.json >/dev/null 2>&1; then
    ok "Open Terminal API reachable on :8000"
  else
    bad "Open Terminal API not reachable on :8000"
  fi

  if curl -fsS http://127.0.0.1:3000 >/dev/null 2>&1; then
    ok "Open WebUI reachable on :3000"
  else
    warn "Open WebUI not ready yet on :3000 (may still be starting)"
  fi

  info "Docker containers"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
}

summary() {
  echo
  echo "================ AI6 bootstrap summary ================"
  echo "PASS: $PASS"
  echo "WARN: $WARN"
  echo "FAIL: $FAIL"
  echo "======================================================="
  echo
  echo "LAN URLs after setup:"
  echo "  Open WebUI:   http://HOST_IP:3000"
  echo "  Open Terminal http://HOST_IP:8000"
  echo "  Agent apps:   http://HOST_IP:8001 ... :8010"
  echo
  if (( FAIL > 0 )); then
    echo "Result: FAIL — fix failed prerequisites/checks above."
    return 1
  fi
  echo "Result: PASS"
}

main() {
  local action="${1:-check}"
  case "$action" in
    check)
      check_prereqs
      ;;
    install-services)
      check_prereqs
      install_services
      final_checks
      ;;
    docker)
      check_prereqs
      start_docker
      final_checks
      ;;
    all)
      check_prereqs
      install_services
      start_docker
      final_checks
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  summary
}

main "$@"
