#!/usr/bin/env bash
# sglang-jax backend (port 30000) — SAME matrix/knobs as bench_jax.sh.
# Raw per-point JSON -> results/sglang-jax/. Uses sgl_jax.bench_serving (the
# native equivalent of `vllm bench serve`): random fixed in/out lens, greedy,
# ignore-eos (default on), max-concurrency sweep, num-prompts=clamp(2c,16,128).
set -u
VENV_PY="${VENV_PY:-/data/red_poc/.venv/bin/python}"
PROJ="${REPO:-/data/red_poc/Qwen-3.5-397B}"
MODEL_DIR="${MODEL_DIR:-/data/red_poc/models/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-30000}"
OUT="${RESULT_DIR:-$PROJ/results/sglang-jax}"
mkdir -p "$OUT"

run_load() {
  local IN=$1 O=$2; shift 2
  for c in "$@"; do
    local np=$(( c * 2 )); (( np < 16 )) && np=16; (( np > 128 )) && np=128
    local label="in${IN}_out${O}_c${c}"; local jf="$OUT/${label}.json"
    if [[ -f "$jf" ]]; then echo "[skip] $label"; continue; fi
    echo "===== $label (num-prompts=$np) $(date +%H:%M:%S) ====="
    "$VENV_PY" -m sgl_jax.bench_serving \
      --backend sgl-jax --host 127.0.0.1 --port "$PORT" \
      --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
      --dataset-name random \
      --random-input-len "$IN" --random-output-len "$O" --random-range-ratio 0 \
      --num-prompts "$np" --request-rate inf --max-concurrency "$c" --seed 42 \
      --output-file "$jf" --output-details 2>&1 \
      | grep -viE "oneDNN|InitializeLog|TF_ENABLE|it/s\]" \
      | grep -E "throughput|TTFT|TPOT|E2E|Successful|Concurrency|====|Error|error" || true
  done
}

echo "########## D: 1024/1 (pure prefill) ##########";      run_load 1024 1    1 2 4 8 16 32 64
echo "########## E: 8192/1 (pure prefill, headline) ##########"; run_load 8192 1 1 2 4 8 16 32 64
echo "########## A: 1024/1024 (balanced) ##########";       run_load 1024 1024 1 2 4 8 16 32 64
echo "########## C: 8192/1024 (prefill-heavy) ##########";  run_load 8192 1024 1 2 4 8 16 32 64
echo "########## B: 1024/8192 (decode-heavy) ##########";   run_load 1024 8192 4 8 16 32 64
echo "########## sglang-jax matrix complete $(date +%H:%M:%S) ##########"
