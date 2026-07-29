#!/usr/bin/env bash
# Self-contained launcher for the torch-tpu (native PyTorch PrivateUse1) backend,
# DP8 config (DP=8 / TP=1 / EP), serving Qwen3.5-397B-A17B-FP8 on 4x Ironwood (v7x).
# No external repo checkout required: paths self-locate to this repo, all overridable via env.
#
# Prereqs (see torch-tpu.md §3): a Python venv with vLLM 0.22.1+tpu and the torch-tpu
# wheel installed, the vllm_torchtpu plugin (editable install), and the model metadata
# under models/Qwen3.5-397B-A17B-FP8.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCHTPU_DIR="${TORCHTPU_DIR:-$PROJECT_ROOT/third_party/torchtpu-vllm}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-69632}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
COMPILE_SIZES="${COMPILE_SIZES:-2,4,8,16,$MAX_NUM_BATCHED_TOKENS}"

if [[ ! -x "$VENV_DIR/bin/vllm" ]]; then
  echo "ERROR: vLLM is not installed in the venv: $VENV_DIR (see torch-tpu.md §3.3-3.4)" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "ERROR: local model metadata is incomplete: $MODEL_DIR" >&2
  exit 1
fi

# The vllm_torchtpu plugin is loaded via VLLM_PLUGINS. If a source tree is present,
# prepend it to PYTHONPATH; otherwise rely on the editable/site install.
if [[ -d "$TORCHTPU_DIR/src/vllm_torchtpu" ]]; then
  export PYTHONPATH="$TORCHTPU_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

export PJRT_DEVICE=TPU
export VLLM_TARGET_DEVICE=tpu
export VLLM_PLUGINS=torchtpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export SKIP_JAX_PRECOMPILE=1
# TorchTPU's split compiler artifact is not currently serializable; disable both
# vLLM's compile cache and PyTorch's AOTAutograd cache.
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
export RAY_memory_monitor_refresh_ms=0
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export VLLM_XLA_CHECK_RECOMPILATION=0
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY="${VLLM_MOE_ROUTING_SIMULATION_STRATEGY:-uniform_random}"
export PYTHONUNBUFFERED=1

COMPILATION_CONFIG=$(printf \
  '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[%s],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  "$COMPILE_SIZES")

echo "Starting $SERVED_MODEL_NAME (torch-tpu DP8) from $MODEL_DIR"
echo "parallelism: DP=8, TP=1 | compile sizes: $COMPILE_SIZES | port: $PORT"

exec "$VENV_DIR/bin/python" \
  -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --load-format dummy \
  --generation-config vllm \
  --seed 42 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size 8 \
  --attention-backend CUSTOM \
  --block-size 256 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-enable-prefix-caching \
  --prefill-context-parallel-size 1 \
  --cp-kv-cache-interleave-size 256 \
  --no-disable-hybrid-kv-cache-manager \
  --tensor-parallel-size 1 \
  --return-tokens-as-token-ids \
  --compilation-config "$COMPILATION_CONFIG" \
  "$@"
