#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${AI6_MONITOR_URL:-http://127.0.0.1:8090}"
GPU="${GPU_INDEX:-0}"
COMMENT="${COMMENT:-CMP50_2}"
MODE="${1:-safe}"

if [[ "$MODE" != "safe" && "$MODE" != "full" ]]; then
  echo "Usage: $0 [safe|full]"
  echo "  safe: CMP40-comparable points at 130/150 W"
  echo "  full: safe + 184 W points"
  exit 64
fi

json_post() {
  local url="$1" body="$2"
  curl -fsS --max-time 900 -H 'Content-Type: application/json' -d "$body" "$url"
}

echo "== AI6 CMP50 vLLM comparison sweep =="
echo "Monitor: $BASE"
echo "GPU:     $GPU"
echo "Comment: $COMMENT"
echo "Mode:    $MODE"
echo

status="$(curl -fsS "$BASE/api/vllm/status")"
python3 - "$status" <<'PY'
import json,sys
d=json.loads(sys.argv[1])
if not d.get("ready"):
    raise SystemExit("ERROR: vLLM is not READY. Start Qwen3-4B-AWQ first.")
print("vLLM READY:", d.get("model") or d.get("model_path"))
print("Mode:", "eager" if d.get("enforce_eager") else "compiled/graphs")
print("Context:", d.get("max_model_len"))
PY

gpu_name="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | xargs)"
echo "GPU name: $gpu_name"
if [[ "$gpu_name" != *"CMP 50HX"* ]]; then
  echo "WARNING: expected CMP 50HX, got: $gpu_name"
fi
echo

set_pl() {
  local watts="$1"
  echo "---- Set PL $watts W ----"
  json_post "$BASE/api/gpu/power-limit" "{\"watts\":$watts,\"gpu\":$GPU}" >/dev/null
  sleep 2
  nvidia-smi -i "$GPU" --query-gpu=power.limit,pcie.link.gen.current,pcie.link.width.current,temperature.gpu --format=csv,noheader
}

run_one() {
  local watts="$1" out="$2"
  echo
  echo ">>> PL=$watts W | medium | conc=12 | out=$out"
  local resp run_id
  resp="$(json_post "$BASE/api/vllm/benchmark" "{\"concurrency\":12,\"max_tokens\":$out,\"temperature\":0,\"prompt_profile\":\"medium\",\"enable_thinking\":false}")"
  run_id="$(python3 - "$resp" <<'PY'
import json,sys
d=json.loads(sys.argv[1])["result"]
print(d["run_id"])
PY
)"
  python3 - "$resp" <<'PY'
import json,sys
r=json.loads(sys.argv[1])["result"]
pcie=(r.get("pcie") or [{}])[0]
print(
    f"RESULT  agg={r.get('aggregate_tok_s')} tok/s | "
    f"per={r.get('per_request_min_tok_s')}-{r.get('per_request_max_tok_s')} | "
    f"TTFT={r.get('ttft_avg_s')}/{r.get('ttft_max_s')} s | "
    f"wall={r.get('wall_s')} s | "
    f"power={r.get('gpu_power_avg_w')}/{r.get('gpu_power_peak_w')} W | "
    f"PCIe=G{pcie.get('gen_current','?')}x{pcie.get('width_current','?')}"
)
PY
  json_post "$BASE/api/vllm/benchmark/meta" "{\"run_id\":\"$run_id\",\"comment\":\"$COMMENT\",\"selected\":true}" >/dev/null
  sleep 2
}

# Same medium/conc12 comparison points used for the CMP40 baseline.
set_pl 130
for out in 128 512 1024 2048; do run_one 130 "$out"; done

set_pl 150
for out in 512 1024 2048; do run_one 150 "$out"; done

if [[ "$MODE" == "full" ]]; then
  set_pl 184
  for out in 1024 2048; do run_one 184 "$out"; done
fi

echo
echo "== Sweep complete =="
echo "All new rows are selected and tagged: $COMMENT"
echo "Open: $BASE/api/vllm/report/selected"
echo
echo "Current GPU:"
nvidia-smi -i "$GPU" --query-gpu=name,power.limit,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv
