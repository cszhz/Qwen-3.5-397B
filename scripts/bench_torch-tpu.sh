#!/usr/bin/env bash
# torch-tpu DP8 independent benchmark matrix: 5 workloads (A-E) x concurrency sweep.
# Calls `vllm bench serve` directly; raw JSON is written to results/torch-tpu/.
set -u

PROJ="${REPO:?set REPO to your checkout root}"
VENV=$PROJ/.venv
MODEL_DIR="${MODEL_DIR:-$PROJ/models/Qwen3.5-397B-A17B-FP8}"
SERVED="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-18100}"
OUT="${RESULT_DIR:-$PROJ/results/torch-tpu}"
mkdir -p "$OUT"

# Workload definition: "input output concurrency-list"
run_load() {
  local IN=$1 O=$2; shift 2
  local CONCS=("$@")
  for c in "${CONCS[@]}"; do
    # Scale num-prompts with concurrency (~2 steady-state rounds) while capping
    # the cost of low-concurrency / decode-heavy points.
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
      --label "$label" 2>&1 | grep -E "Total token throughput|Output token throughput|Request throughput|Mean TTFT|Mean TPOT|Successful requests|Median" || true
  done
}

echo "########## Workload A: 1024/1024 (balanced) ##########"
run_load 1024 1024 1 2 4 8 16 32 64
echo "########## Workload B: 1024/8192 (decode-heavy) ##########"
run_load 1024 8192 4 8 16 32 64
echo "########## Workload C: 8192/1024 (prefill-heavy, comparable to JAX) ##########"
run_load 8192 1024 1 2 4 8 16 32 64
echo "########## Workload D: 1024/1 (pure prefill, short input) ##########"
run_load 1024 1 1 2 4 8 16 32 64
echo "########## Workload E: 8192/1 (pure prefill, repo headline) ##########"
run_load 8192 1 1 2 4 8 16 32 64

echo "########## matrix complete $(date +%H:%M:%S) ##########"
