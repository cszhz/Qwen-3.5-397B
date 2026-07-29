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

### 9.1 Measurement setup

- **Tool**: `vllm bench serve` inside the container (same source as the recipe's benchmark_serving), `--dataset-name random`, `--endpoint /v1/completions`, `--random-range-ratio 0` (fixed length), `--ignore-eos --temperature 0 --seed 42`, `--request-rate inf`.
- **Harness**: `scripts/bench_jax.sh` (JAX) and `scripts/bench_torch-tpu.sh` (torch-tpu) use **identical parameters** — `num-prompts = c×2` clamped to [16,128]. Because both backends share the harness, §9.2 here and the cross-backend A/B in §9.4 are directly comparable with `torch-tpu.md` §9.2.
- **Raw JSON**: `results/jax/` (this backend), `results/torch-tpu/` (torch-tpu). Filenames: `in<INPUT>_out<OUTPUT>_c<CONCURRENCY>.json`.
- **Workloads**:

| Load | input / output | Character |
|------|----------------|-----------|
| A | 1024 / 1024 | balanced |
| B | 1024 / 8192 | decode-heavy |
| C | 8192 / 1024 | prefill-heavy (mixed) |
| D | 1024 / 1 | short pure-prefill |
| E | 8192 / 1 | long pure-prefill (the repo's headline metric) |

### 9.2 Measured results (JAX TP8)

Real-weights `tpu_inference` service (`scripts/run_vllm.sh`, TP=8 + attention DP + MoE EP, port 8000), all five workloads swept over concurrency:

**Workload A — 1024 / 1024 (balanced)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 1 | 105 | 53 | 0.05 | 272 | 18.7 |
| 2 | 252 | 126 | 0.12 | 300 | 15.6 |
| 4 | 484 | 242 | 0.24 | 346 | 16.2 |
| 8 | 920 | 460 | 0.45 | 434 | 17.0 |
| 16 | 1,133 | 566 | 0.55 | 4,293 | 18.3 |
| 32 | 2,069 | 1,035 | 1.01 | 3,171 | 20.5 |
| **64** | **3,498** | 1,749 | 1.71 | 1,945 | 25.5 |

**Workload B — 1024 / 8192 (decode-heavy)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 4 | 277 | 246 | 0.03 | 344 | 16.2 |
| 8 | 527 | 468 | 0.06 | 433 | 17.0 |
| 16 | 823 | 731 | 0.09 | 12,341 | 18.3 |
| 32 | 1,515 | 1,347 | 0.16 | 6,630 | 20.4 |
| **64** | **2,811** | 2,499 | 0.31 | 2,585 | 24.2 |

**Workload C — 8192 / 1024 (prefill-heavy, mixed)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 1 | 444 | 49 | 0.05 | 1,595 | 18.7 |
| 2 | 1,035 | 115 | 0.11 | 1,764 | 15.7 |
| 4 | 1,984 | 220 | 0.22 | 1,838 | 16.4 |
| 8 | 3,811 | 423 | 0.41 | 1,825 | 17.1 |
| 16 | 4,812 | 535 | 0.52 | 4,982 | 18.6 |
| 32 | 11,089 | 1,232 | 1.20 | 3,771 | 22.2 |
| **64** | **20,020** | 2,224 | 2.17 | 5,522 | 23.2 |

**Workload D — 1024 / 1 (pure prefill, short input)**

| Concurrency | Total tok/s | Req/s | Mean TTFT (ms) | p99 TTFT (ms) |
|-------------|------------|-------|----------------|---------------|
| 1 | 3,496 | 3.41 | 293 | 1,414 |
| 2 | 5,280 | 5.15 | 376 | 395 |
| 4 | 7,995 | 7.80 | 467 | 556 |
| 8 | 22,134 | 21.59 | 358 | 382 |
| 16 | 35,759 | 34.89 | 410 | 538 |
| 32 | 37,610 | 36.69 | 679 | 894 |
| **64** | **41,448** | 40.44 | 1,194 | 1,623 |

**Workload E — 8192 / 1 (pure prefill, long input; repo headline)**

| Concurrency | Total tok/s | Req/s | Mean TTFT (ms) | p99 TTFT (ms) |
|-------------|------------|-------|----------------|---------------|
| 1 | 5,973 | 0.73 | 1,371 | 1,377 |
| 2 | 9,759 | 1.19 | 1,658 | 1,720 |
| 4 | 19,116 | 2.33 | 1,692 | 1,768 |
| 8 | 35,255 | 4.30 | 1,783 | 1,953 |
| 16 | 38,827 | 4.74 | 2,780 | 3,573 |
| 32 | 43,395 | 5.30 | 4,814 | 6,141 |
| **64** | **44,651** | 5.45 | 9,083 | 11,787 |

**Observations**

1. **0 failures** across every workload, with stable TPOT/ITL (p99 ≈ mean) and healthy latency.
2. **8192/1024 (prefill+decode) is the strongest mixed workload**, reaching 20,020 tok/s @ c64; the pure-prefill workloads (D/E) reach 41k–45k @ c64.
3. **Decode-heavy (1024/8192) is TPOT-bound** at ~24 ms/token/stream, so total throughput ≈ concurrent streams × ~40 tok/s and tops out at 2,811 tok/s @ c64.

### 9.3 Comparison against the official recipe (output throughput, ÷4 physical chips)

| Workload | This run tok/s (/chip) | Official tok/s (/chip) |
|----------|------------------------|------------------------|
| 1k/8k | 2,499 (625/chip) | 5,172 (1,293/chip) |
| 8k/1k | 2,224 (556/chip) | 2,281 (570/chip) |

- **8k/1k (prefill-heavy) essentially matches the official** (556 vs 570 tok/s/chip).
- **1k/8k (decode-heavy) is about half the official**: decode is limited by TPOT ≈ 24 ms/token/stream, so total throughput ≈ concurrent streams × ~40 tok/s. This run was capped at concurrency=64 (`--max-num-seqs=64`), whereas the official benchmark uses `--max-concurrency=128` + 640 prompts to fill the batch, giving ~2× the throughput.
- **Improvement direction**: raise `--max-num-seqs` to 128 and re-test at concurrency 128 to approach the official decode throughput (note: changing this triggers a one-time ~70 min recompile for the new num_reqs bucket).

### 9.4 Cross-backend A/B vs torch-tpu (same harness)

Both backends drive the same 4 chips, so only one runs at a time; the A/B was measured by stopping one and bringing up the other under the identical harness (§9.1). Peak point (c64) per workload:

| Workload (in/out) | Type | torch-tpu | JAX | c64 winner |
|-------------------|------|-----------|-----|------------|
| 1024 / 1 | pure prefill (short) | 25,983 | **41,448** | JAX ↑ ~60% |
| 8192 / 1 | pure prefill (long) | **49,934** | 44,651 | **torch-tpu ↑ ~12%** |
| 1024 / 1024 | balanced | **3,607** | 3,498 | ~tie (torch-tpu ↑ ~3%) |
| 1024 / 8192 | decode-heavy | 2,488 | **2,811** | JAX ↑ ~13% |
| 8192 / 1024 | prefill+decode mixed | 14,147 | **20,020** | JAX ↑ ~42% |

**Pure-prefill, per concurrency (total tok/s)**

| Concurrency | 1024/1 torch-tpu | 1024/1 JAX | 8192/1 torch-tpu | 8192/1 JAX |
|-------------|------------------|------------|------------------|------------|
| 1 | 1,197 | 3,496 | 4,567 | 5,973 |
| 2 | 1,486 | 5,280 | 9,167 | 9,759 |
| 4 | 4,817 | 7,995 | 12,972 | 19,116 |
| 8 | 3,960 | 22,134 | 29,858 | 35,255 |
| 16 | 6,740 | 35,759 | 45,126 | 38,827 |
| 32 | 13,780 | 37,610 | 45,602 | 43,395 |
| 64 | 25,983 | 41,448 | 49,934 | 44,651 |

**Conclusions (c64)**

- **JAX leads in every scenario except long-input pure prefill**: short-input pure prefill (1024/1, +60% over torch-tpu), decode-heavy (1024/8192, +13%), and prefill+decode mixed (8192/1024, +42%).
- **torch-tpu wins only on long-input pure prefill** (8192/1, +12%); the balanced workload (1024/1024) is essentially a tie (+3% for torch-tpu).
- Any cross-backend conclusion must be anchored to "input length + whether decode is involved"; a single pure-prefill headline point is not representative on its own.

---

## 10. Gotchas

1. **Actual parallelism ≠ the CLI request** — the command line asks for `TP=8 + EP` + `enable_dp_attention`, but the TPU backend actually runs **attention DP=8 + MoE EP=8, TP=1, PP=1** (see §5). Check the `ShardingConfigManager` line in the logs to confirm what really got sharded.
2. **First compile is ~69 min (one-time)** — the initial XLA compile dominates startup (§6); artifacts are cached under `/data/red_poc/{xla_cache,torch_compile_cache}`, so subsequent restarts (`docker restart`) skip it.
3. **Changing batch/length params forces a full recompile** — raising `--max-num-seqs` or `--max-model-len` invalidates the cache and triggers a fresh ~70 min compile for the new bucket (§9.3). Plan tuning runs accordingly.
4. **Decode throughput is capped by `--max-num-seqs=64`** — decode-heavy workloads are TPOT-bound at ~40 tok/s per stream, so total ≈ concurrent streams × ~40; the batch cap (not the hardware) is the limiter (§9.3).
5. **Credentials** — the HF token is a live credential; keep it in the `HF_TOKEN` env var only, never in commits/logs/reports.

---

## 11. Optional Follow-ups

- Run the recipe's built-in benchmark (1k/8k and 8k/1k workloads) for throughput
- Tuning: raise `--max-num-seqs` / `--max-model-len` to trade for throughput or long context
- Comparison: measure the memory/throughput impact of disabling `enable_dp_attention` (falling back to traditional TP attention)
- Configure systemd for auto-start on boot
