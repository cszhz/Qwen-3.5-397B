#!/usr/bin/env bash
# JAX 后端(端口8000)同 harness 对照矩阵。结果存 matrix_results_jax。
set -u

VENV="${REPO:?set REPO to your checkout root}/.venv"
MODEL_DIR=/data/red_poc/models/Qwen3.5-397B-A17B-FP8
SERVED="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
OUT=/data/red_poc/matrix_results_jax
mkdir -p "$OUT"

run_load() {
  local IN=$1 O=$2; shift 2
  local CONCS=("$@")
  for c in "${CONCS[@]}"; do
    local np=$(( c * 2 )); (( np < 16 )) && np=16; (( np > 128 )) && np=128
    local label="in${IN}_out${O}_c${c}"
    local jf="$OUT/${label}.json"
    if [[ -f "$jf" ]]; then echo "[skip] $label 已存在"; continue; fi
    echo "===== 跑 $label (num-prompts=$np) $(date +%H:%M:%S) ====="
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

echo "########## D: 1024/1 (纯 prefill) ##########"
run_load 1024 1 1 2 4 8 16 32 64
echo "########## E: 8192/1 (纯 prefill, 仓库口径) ##########"
run_load 8192 1 1 2 4 8 16 32 64
echo "########## A: 1024/1024 (均衡) ##########"
run_load 1024 1024 1 2 4 8 16 32 64
echo "########## C: 8192/1024 (prefill 密集) ##########"
run_load 8192 1024 1 2 4 8 16 32 64
echo "########## B: 1024/8192 (decode 密集) ##########"
run_load 1024 8192 4 8 16 32 64

echo "########## JAX 矩阵完成 $(date +%H:%M:%S) ##########"
