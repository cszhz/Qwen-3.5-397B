#!/usr/bin/env bash
# JAX backend (port 8000) same-harness comparison matrix. Raw JSON -> results/jax/.
set -u

PROJ="${REPO:?set REPO to your checkout root}"
VENV="$PROJ/.venv"
MODEL_DIR="${MODEL_DIR:-$PROJ/models/Qwen3.5-397B-A17B-FP8}"
SERVED="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-8000}"
OUT="${RESULT_DIR:-$PROJ/results/jax}"
mkdir -p "$OUT"

run_load() {
  local IN=$1 O=$2; shift 2
  local CONCS=("$@")
  for c in "${CONCS[@]}"; do
    local np=$(( c * 2 )); (( np < 16 )) && np=16; (( np > 128 )) && np=128
    local label="in${IN}_out${O}_c${c}"
    local jf="$OUT/${label}.json"
    if [[ -f "$jf" ]]; then echo "[skip] $label already exists"; continue; fi
    echo "===== run $label (num-prompts=$np) $(date +%H:%M:%S) ====="
    "$VENV/bin/vllm" bench serve \
      --backend openai --host 127.0.0.1 --port "$PORT" \
      --endpoint /v1/completions \
      --model "$SERVED" --tokenizer "$MODEL_DIR" \
      --dataset-name random \
      --random-input-len "$IN" --random-output-len "$O" --random-range-ratio 0 \
      --num-prompts "$np" --request-rate inf --max-concurrency "$c" \
      --ignore-eos --temperature 0 --seed 42 \
      --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$OUT" --result-filename "${label}.json" \
      --label "$label" 2>&1 | grep -E "Total token throughput|Output token throughput|Request throughput|Mean TTFT|Mean TPOT|Successful requests|Median|Error|error" || true
  done
}

echo "########## Workload D: 1024/1 (pure prefill) ##########"
run_load 1024 1 1 2 4 8 16 32 64
echo "########## Workload E: 8192/1 (pure prefill, repo headline) ##########"
run_load 8192 1 1 2 4 8 16 32 64
echo "########## Workload A: 1024/1024 (balanced) ##########"
run_load 1024 1024 1 2 4 8 16 32 64
echo "########## Workload C: 8192/1024 (prefill-heavy) ##########"
run_load 8192 1024 1 2 4 8 16 32 64
echo "########## Workload B: 1024/8192 (decode-heavy) ##########"
run_load 1024 8192 4 8 16 32 64

echo "########## JAX matrix complete $(date +%H:%M:%S) ##########"
