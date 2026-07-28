#!/usr/bin/env bash
# torch-tpu DP8 独立测试矩阵:3 种负载 × 并发扫描
# 直接调用 vllm bench serve,结果存 /data/red_poc/matrix_results,不覆盖复现报告。
set -u

PROJ="${REPO:?set REPO to your checkout root}"
VENV=$PROJ/.venv
MODEL_DIR=$PROJ/models/Qwen3.5-397B-A17B-FP8
SERVED=Qwen3.5-397B-A17B-FP8
PORT=18100
OUT=/data/red_poc/matrix_results
mkdir -p "$OUT"

# 负载定义: "输入 输出 并发列表"
run_load() {
  local IN=$1 O=$2; shift 2
  local CONCS=("$@")
  for c in "${CONCS[@]}"; do
    # num-prompts 随并发缩放,保证约 2 轮稳态,同时限制低并发/decode 密集耗时
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
      --label "$label" 2>&1 | grep -E "Total token throughput|Output token throughput|Request throughput|Mean TTFT|Mean TPOT|Successful requests|Median" || true
  done
}

echo "########## 负载 A: 1024/1024 (均衡) ##########"
run_load 1024 1024 1 2 4 8 16 32 64
echo "########## 负载 B: 1024/8192 (decode 密集) ##########"
run_load 1024 8192 4 8 16 32 64
echo "########## 负载 C: 8192/1024 (prefill 密集, 可比 JAX) ##########"
run_load 8192 1024 1 2 4 8 16 32 64
echo "########## 负载 D: 1024/1 (纯 prefill, 短输入) ##########"
run_load 1024 1 1 2 4 8 16 32 64
echo "########## 负载 E: 8192/1 (纯 prefill, 仓库口径) ##########"
run_load 8192 1 1 2 4 8 16 32 64

echo "########## 矩阵完成 $(date +%H:%M:%S) ##########"
