#!/usr/bin/env bash
# Self-contained throughput benchmark for the torch-tpu DP8 server.
# Runs `vllm bench serve` across a concurrency sweep at a fixed input/output length
# (default 8192/1 = pure prefill, the headline metric) and writes a summary.json.
# No external repo / report-publishing machinery required.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-18100}"
INPUT_LEN="${INPUT_LEN:-8192}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-dp8}"

RUN_DIR="${1:-$PROJECT_ROOT/runs/manual-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)

RESULT_PREFIX="vllm_${BENCHMARK_CONFIG}_tp1_len${INPUT_LEN}"
RESULT_DIR="$RUN_DIR/results/$BENCHMARK_CONFIG"
mkdir -p "$RESULT_DIR"

if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: vLLM CLI is missing: $VENV_DIR/bin/vllm" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: local tokenizer metadata is missing: $MODEL_DIR" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

read -r -a concurrencies <<< "${CONCURRENCIES:-1 2 4 8 16 32 64}"

for concurrency in "${concurrencies[@]}"; do
  echo "===== benchmark concurrency=$concurrency ($INPUT_LEN/$OUTPUT_LEN) ====="
  "$VENV_DIR/bin/vllm" bench serve \
    --backend openai \
    --host "$HOST" \
    --port "$PORT" \
    --endpoint /v1/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts 128 \
    --request-rate inf \
    --max-concurrency "$concurrency" \
    --ignore-eos \
    --temperature 0 \
    --seed 42 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "${RESULT_PREFIX}_c${concurrency}.json" \
    --label "${RESULT_PREFIX}_c${concurrency}" \
    "$@"
done

# Aggregate the per-concurrency JSONs into a single summary.json.
"$VENV_DIR/bin/python" - \
  "$RESULT_DIR" "$INPUT_LEN" "$OUTPUT_LEN" "$SERVED_MODEL_NAME" "$BENCHMARK_CONFIG" "$RESULT_PREFIX" <<'PY'
import glob, json, os, sys

result_dir, input_length, output_length, model, cfg, prefix = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
records = []
for path in sorted(glob.glob(os.path.join(result_dir, f"{prefix}_c*.json"))):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    records.append({
        "file": os.path.basename(path),
        "concurrency": int(d["max_concurrency"]),
        "completed": int(d["completed"]),
        "failed": int(d["failed"]),
        "request_throughput": float(d["request_throughput"]),
        "total_token_throughput": float(d["total_token_throughput"]),
        "mean_ttft_ms": float(d["mean_ttft_ms"]),
        "p99_ttft_ms": float(d["p99_ttft_ms"]),
    })
if not records:
    raise SystemExit("No benchmark result JSON files were produced.")
records.sort(key=lambda r: r["concurrency"])
best = max(records, key=lambda r: r["total_token_throughput"])
summary = {
    "benchmark": {"input_length": input_length, "output_length": output_length,
                  "model": model, "benchmark_config": cfg},
    "best": best, "results": records,
}
with open(os.path.join(result_dir, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"Highest total token throughput: {best['total_token_throughput']:.2f} tok/s "
      f"(concurrency={best['concurrency']})")
failed = sum(r["failed"] for r in records)
if failed:
    raise SystemExit(f"Benchmark completed with {failed} failed requests.")
PY

echo "Summary written to $RESULT_DIR/summary.json"
