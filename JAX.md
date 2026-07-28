# Qwen3.5-397B-A17B-FP8 on Ironwood TPU — JAX Backend Deployment

- **Report date**: 2026-07-23
- **Goal**: On a single TPU VM without GKE, deploy and run `Qwen/Qwen3.5-397B-A17B-FP8` with vLLM, following [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3.5-397B).
- **Verdict**: ✅ Deployment succeeded; the OpenAI-compatible API returns inference correctly.

---

## 1. Environment / Hardware

| Item | Value |
|------|-------|
| Machine type | GCE `tpu7x-standard-4t` (Ironwood / TPU v7x) |
| Accelerator | 4 chips × 2 cores = **8 TPU cores**, VFIO passthrough (`/dev/vfio/0..7`) |
| Per-core HBM | 94.75 GiB (total ~758 GiB) |
| CPU / RAM | 224 vCPU / 944 GiB |
| Disk | `/data` 2.0 TB (NVMe), 1.6 TB free before deployment |
| OS | Ubuntu 24.04.4 LTS |
| Docker | 29.1.3 |

> Note: this machine type is exactly the hardware the recipe requires (`tpu7x-standard-4t`), so the recipe's GKE/K8s flow can be replaced with **direct Docker + vfio passthrough** — no GKE cluster needed.

---

## 2. Model

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen3.5-397B-A17B-FP8` |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (hybrid MoE) |
| Precision | FP8 |
| Disk footprint | 379 GiB (94 safetensors shards + config/tokenizer) |
| Local path | `/data/red_poc/models/Qwen3.5-397B-A17B-FP8` |
| Layer structure | 60 layers = 15 full-attention + 45 linear-attention (mamba-like) |
| MoE | 512 experts, top-10 routing per token, shared expert; 397B total / ~17B active (A17B) |

Download method: HF CLI (installed in venv `/data/red_poc/.venv`), `hf_transfer`/Xet fast transfer, authenticated account `cszhzleo`.

---

## 3. Deployment Steps

1. **Install Docker Engine** (`apt install docker.io`, Ubuntu 24.04)
2. **Pull the image** `vllm/vllm-tpu:nightly-20260626-c539adc-cc79815` (10.2 GB)
3. **Create persistent compile-cache dirs**: `/data/red_poc/{xla_cache,torch_compile_cache}` (avoids recompiling on every restart)
4. **Start the container** (script `/data/red_poc/run_vllm.sh`):
   - TPU access: `--privileged` + `-v /dev/vfio:/dev/vfio` (replaces GKE's `google.com/tpu:4`)
   - `--net=host` (port 8000), `--shm-size=64g`
   - Mount the local model: `-v /data/red_poc:/data`, `--model=/data/models/Qwen3.5-397B-A17B-FP8` (skips network download)
   - flags / env vars match the recipe's `qwen3_5-server.yaml`
5. **Wait for the first XLA compile to finish** → log shows `Application startup complete.`
6. **Smoke test** `/v1/chat/completions`

---

## 4. GitHub Recipe Defaults (adopted for this deployment)

### 4.1 Parallelism / sharding
| Parameter | Recommended |
|-----------|-------------|
| `--tensor-parallel-size` | 8 |
| `--enable-expert-parallel` | on |
| `--additional-config` | `{"sharding":{"sharding_strategy":{"enable_dp_attention":true}}}` |
| pipeline / data parallel | not set (=1) |

### 4.2 Memory / batching / lengths
| Parameter | Recommended |
|-----------|-------------|
| `--gpu-memory-utilization` | 0.9 |
| `--max-model-len` | 9216 |
| `--max-num-seqs` | 64 |
| `--max-num-batched-tokens` | 1024 |
| `--block-size` | 256 |
| `--kv-cache-dtype` | fp8 |
| `--no-enable-prefix-caching` | on |

### 4.3 Model / features
| Parameter | Recommended |
|-----------|-------------|
| `--language-model-only` | on |
| `--limit-mm-per-prompt` | `{"image":0,"video":0}` |
| `--enable-auto-tool-choice` | on |
| `--tool-call-parser` | `qwen3_coder` |
| `--reasoning-parser` | `qwen3` |

### 4.4 Environment variables
| Variable | Value |
|----------|-------|
| `USE_MOE_EP_KERNEL` | 0 |
| `MODEL_IMPL_TYPE` | vllm |
| `NEW_MODEL_DESIGN` | 1 |
| `VLLM_MOE_CHUNK_SIZE` | 256 |
| `ATTN_BUCKETIZED_NUM_REQS` | true |
| `ATTN_CUSTOM_NUM_REQS_BUCKETS` | 8,16,32,64 |
| `RAGGED_GATED_DELTA_RULE_IMPL` | chunked_kernel_p_recurrent_kernel_d |
| `ONEHOT_MOE_PERMUTE_THRESHOLD` | 32768 |
| `LIBTPU_INIT_ARGS` | `--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false` |

### 4.5 Infrastructure
| Item | Recommended | This deployment |
|------|-------------|-----------------|
| Image | `vllm/vllm-tpu:nightly-20260626-c539adc-cc79815` | same |
| TPU resource | `google.com/tpu: 4` | `--privileged` + `/dev/vfio` passthrough |
| Storage | 1500Gi hyperdisk → `/data` | local NVMe `/data/red_poc` → `/data` |
| `/dev/shm` | memory-backed | `--shm-size=64g` |

> Only two differences from the recipe: `--model` uses a local path; TPU access uses Docker vfio passthrough instead of the K8s device plugin. All numeric parameters follow the recommended defaults.

---

## 5. Actual Parallelism Assignment (runtime-verified)

The command line requests `TP=8 + EP` + `enable_dp_attention`; the TPU backend (`ShardingConfigManager`, 8-core mesh) actually orchestrates as follows:

| Model part | Actual parallelism | Evidence |
|------------|--------------------|----------|
| Attention + linear-attn/mamba layers | **Data parallel DP=8** (`attention_data_parallelism=8`); KV cache split into 8 shards by request | mesh `attn_dp:8`; KV spec `P(('data','attn_dp',...))` |
| MoE expert layers | **Expert parallel EP=8**, 512 experts ÷ 8 = 64 experts per core | sharded weight shape `(64, 4096, 1024)` |
| Layer dimension | **PP=1**, all 60 layers on every core | `pipeline_parallel_size=1` |

Runtime report:
```
ShardingConfigManager(total_devices=8,
  ShardingStrategy(tensor_parallelism=1, expert_parallelism=1,
    attention_data_parallelism=8, ...))
```

**Key point**: it is not "all weights uniformly TP-split across 8"; rather it is **attention DP=8 + MoE experts EP=8** sharing these 8 cores; PP is not used.
- MoE expert weights dominate the parameter count → spread via EP, so per-core HBM is only **~85 / 94.75 GiB**.
- Attention uses DP rather than TP → avoids KV-cache all-gather/all-reduce during decode, giving better throughput.

---

## 6. Resources and Startup Time

| Metric | Value |
|--------|-------|
| Weight load (disk → TPU) | 178.3 s |
| First XLA compile | 4161.5 s (~69 min, one-time, cached) |
| Total init engine | 4382.6 s |
| Per-core HBM usage | ~85 GiB / 94.75 GiB (weights) |
| HBM total cap / usage | 682.2 GiB (cap) / 424.5 GiB (weights) |
| KV cache capacity | 10,620,612 tokens |
| Max concurrency for a 9216-token request | 1152.4x |

> Compiled artifacts are cached in `/data/red_poc/{xla_cache,torch_compile_cache}`; subsequent restarts skip recompilation.

---

## 7. Functional Test

- **Endpoint**: `http://localhost:8000` (OpenAI-compatible)
- **Model name**: `Qwen/Qwen3.5-397B-A17B-FP8`

`GET /v1/models` → returns `['Qwen/Qwen3.5-397B-A17B-FP8']` ✅

`POST /v1/chat/completions` (prompt: "Introduce yourself in one sentence and state what hardware you run on"):
- Response: the model correctly introduces itself as Qwen3.5 (Alibaba's large language model) ✅
- usage: `prompt_tokens=24, completion_tokens=200, total_tokens=224` ✅

**Conclusion: the inference chain (routing → forward → decode → return) is fully working.**

---

## 8. Operations Commands

```bash
sudo docker logs -f vllm-qwen35        # logs
sudo docker restart vllm-qwen35        # restart (uses cache, fast)
sudo docker stop vllm-qwen35           # stop
cd /data/red_poc && ./run_vllm.sh      # (re)start / restart after changing params
```

Example call:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-397B-A17B-FP8",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":256}'
```

---

## 9. Performance Benchmarks (vllm bench serve)

- **Tool**: `vllm bench serve` inside the container (same source as the recipe's benchmark_serving), `--dataset-name random`
- **Target**: the real-weights service deployed here (`localhost:8000`, concurrency=64=`max-num-seqs`, `--ignore-eos`, `--request-rate inf`, `--seed 42`)
- **Result JSON**: raw per-workload JSON in `results/jax/` (produced by `scripts/bench_matrix_jax.sh`)

### 9.1 Measured results

| Workload | Output throughput | Total throughput | TTFT (mean/P99) | TPOT (mean) | Requests / failures |
|----------|-------------------|------------------|-----------------|-------------|---------------------|
| 1k in / 1k out | 2205 tok/s | 4410 tok/s | 2.6 s / 5.6 s | 26.2 ms | 128 / 0 |
| 1k in / 8k out (decode-heavy) | 2535 tok/s | 2852 tok/s | 1.2 s / 1.8 s | 25.1 ms | 64 / 0 |
| 8k in / 1k out (prefill-heavy) | 2204 tok/s | 19836 tok/s | 5.5 s / 12.8 s | 23.4 ms | 128 / 0 |

### 9.2 Comparison against the official recipe (output throughput, normalized ÷4 physical chips)

| Workload | This run tok/s (/chip) | Official tok/s (/chip) |
|----------|------------------------|------------------------|
| 1k/8k | 2535 (634/chip) | 5172 (1293/chip) |
| 8k/1k | 2204 (551/chip) | 2281 (570/chip) |

### 9.3 Conclusions

- **8k/1k (prefill-heavy) essentially matches the official** (551 vs 570 tok/s/chip).
- **1k/8k (decode-heavy) is about half the official**: decode is limited by TPOT≈25ms/token/stream, so total throughput ≈ number of concurrent streams × ~40 tok/s. This run used concurrency=64 (capped by `--max-num-seqs=64`), whereas the official benchmark uses `--max-concurrency=128` + 640 prompts to fill the batch, giving ~2× the throughput.
- All three workloads had **0 failures**, with stable TPOT/ITL (P99 ≈ mean) and healthy latency.
- **Improvement direction**: raise `--max-num-seqs` to 128 and re-test at concurrency 128 to approach the official decode throughput (note: changing this parameter triggers a one-time ~70 min recompile due to the new num_reqs bucket).

> Note: the torch-tpu DP8 harness (`scripts/start_dp_server.sh` + `scripts/bench_all.sh`, dummy weights, port 18100) is a separate path and cannot be used for this Docker/JAX deployment. The results above were measured against the real-weights JAX service using vLLM's built-in `bench serve`. A full controlled A/B between this JAX backend and the torch-tpu backend (same harness, all 5 workloads) is documented in `torch-tpu.md` §9.5.

---

## 10. Optional Follow-ups

- Run the recipe's built-in benchmark (1k/8k and 8k/1k workloads) for throughput
- Tuning: raise `--max-num-seqs` / `--max-model-len` to trade for throughput or long context
- Comparison: measure the memory/throughput impact of disabling `enable_dp_attention` (falling back to traditional TP attention)
- Configure systemd for auto-start on boot
