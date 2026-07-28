#!/usr/bin/env bash
# Launch vLLM serving Qwen3.5-397B-A17B-FP8 on this VM's 4x Ironwood (TPU v7x) chips.
# Mirrors AI-Hypercomputer/tpu-recipes qwen3_5-server.yaml, adapted from GKE -> plain Docker.
set -euo pipefail

IMAGE="vllm/vllm-tpu:nightly-20260626-c539adc-cc79815"
CONTAINER="vllm-qwen35"
HOST_ROOT="/data/red_poc"                       # mounted as /data inside the container
MODEL_PATH="/data/models/Qwen3.5-397B-A17B-FP8" # path *inside* the container

export HF_TOKEN="${HF_TOKEN:-<YOUR_HF_TOKEN>}"

# remove any previous run
sudo docker rm -f "${CONTAINER}" 2>/dev/null || true

sudo docker run -d --name "${CONTAINER}" \
  --privileged \
  --net=host \
  --shm-size=64g \
  -v /dev/vfio:/dev/vfio \
  -v "${HOST_ROOT}:/data" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/data/hf_home \
  -e VLLM_TORCH_COMPILE_CACHE_DIR=/data/torch_compile_cache \
  -e XLA_CACHE_DIR=/data/xla_cache \
  -e USE_MOE_EP_KERNEL=0 \
  -e MODEL_IMPL_TYPE=vllm \
  -e ATTN_BUCKETIZED_NUM_REQS=true \
  -e ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 \
  -e RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_p_recurrent_kernel_d \
  -e ONEHOT_MOE_PERMUTE_THRESHOLD=32768 \
  -e NEW_MODEL_DESIGN=1 \
  -e VLLM_MOE_CHUNK_SIZE=256 \
  -e LIBTPU_INIT_ARGS="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false" \
  "${IMAGE}" \
  vllm serve "${MODEL_PATH}" \
    --served-model-name Qwen/Qwen3.5-397B-A17B-FP8 \
    --tokenizer "${MODEL_PATH}" \
    --host=0.0.0.0 --port=8000 \
    --tensor-parallel-size=8 \
    --kv-cache-dtype=fp8 \
    --max-model-len=9216 \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=64 \
    --max-num-batched-tokens=1024 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder \
    --reasoning-parser=qwen3 \
    --limit-mm-per-prompt='{"image": 0, "video": 0}' \
    --block-size=256 \
    --additional-config='{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}'

echo "Started container ${CONTAINER}. Follow logs with:"
echo "  sudo docker logs -f ${CONTAINER}"
