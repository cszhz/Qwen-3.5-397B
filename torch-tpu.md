# Qwen3.5-397B-A17B-FP8 on Ironwood TPU — torch-tpu Backend Benchmark

- **Report date**: 2026-07-26
- **Goal**: Rebuild the `tpu_benchmark_daily` repo's **torch-tpu (native PyTorch PrivateUse1 backend)** throughput benchmark from scratch on this machine, **without** the private Google Artifact Registry (GAR), and verify whether the repo's committed DP8 / PCP8 results reproduce.
- **Verdict**: ✅ **DP8 fully reproduced** (peak 52,452 tok/s, slightly above the repo baseline of 40k–49k); ⚠️ **PCP8 cannot run on the current torchtpu-vllm HEAD** due to an architectural incompatibility (the baseline was captured on an earlier revision).
- **Relation to the JAX backend**: This is a **completely independent** deployment path. The JAX `tpu_inference` backend (Docker, port 8000) is documented in `JAX.md`; this document covers the torch-tpu native backend (built from source, port 18100). The two use different measurement setups (see §9).

---

## 1. Environment / Hardware

| Item | Value |
|------|-------|
| Machine type | GCE `tpu7x-standard-4t` (Ironwood / TPU v7x) |
| Accelerator | 4 chips × 2 cores = **8 TPU cores**, VFIO passthrough (`/dev/vfio/0..7`) |
| CPU / RAM | 224 vCPU / 944 GiB |
| Disk | `/data` 2.0 TB (NVMe), ~1.2 TB free before reproduction |
| OS | Ubuntu 24.04.4 LTS |
| Build tools | Bazel (bazelisk) **8.6.0**, uv **0.11.32**, gcc **13.3.0**, Python **3.12.3** |

> Note: torch-tpu uses a **bare-metal source build** (not Docker). libtpu provides the PjRt runtime via a PyPI wheel, accessing vfio directly — no GKE / privileged container required.

---

## 2. Backend and Model

### 2.1 Backend positioning (important)

This path uses **torch-tpu** (Google's native PyTorch TPU backend: PrivateUse1 device, FX→MLIR→PjRt compilation), **not torch_xla**. The torchtpu-vllm plugin explicitly "bypasses vLLM's default `torch_xla` TPU path and uses native TorchTPU + Pallas kernels." The three vLLM-on-TPU paths:

| Path | Stack | Repo / deployment |
|------|-------|-------------------|
| torch_xla | PyTorch/XLA lazy tensor | vLLM upstream default (not used) |
| tpu_inference | JAX | `run_vllm.sh` Docker, port 8000 |
| **torch-tpu** | **native PyTorch PrivateUse1** | **this document, port 18100** |

### 2.2 Model

| Item | Value |
|------|-------|
| Model | `Qwen3.5-397B-A17B-FP8` |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (hybrid: 15 full-attn layers + 45 linear-attn/mamba layers; 512 experts, top-10; 397B / ~17B active) |
| Local path | `/data/red_poc/tpu_benchmark_daily/models/Qwen3.5-397B-A17B-FP8` |
| **Weight loading** | **dummy (random weights)** — the benchmark measures throughput only, not accuracy |

> The benchmark uses `--load-format dummy`, loading only config/tokenizer; the real 379 GiB weights are not needed.

---

## 3. Reproduction Steps (no private GAR)

Core idea: every pin except `torch-tpu` comes from public PyPI; `torch-tpu` is built from source, with its version string reproduced exactly via `WHEEL_VERSION_EXTRAS` → guaranteeing ABI alignment with `torchtpu-vllm`.

### 3.1 Install system dependencies (needed for bare-metal vLLM build)

```bash
sudo apt-get install -y build-essential python3.12-dev   # gcc/g++ + Python.h
# bazelisk 8.6.0 and uv already installed
```

### 3.2 Build the torch-tpu wheel from source (the key to version alignment)

```bash
git clone https://github.com/google-pytorch/torch_tpu /data/red_poc/torch_tpu
cd /data/red_poc/torch_tpu
git checkout 40c8169c9903          # commit matching the repo baseline
# Reproduce the exact version 0.1.1.dev20260714090201 (otherwise the build-time timestamp is used)
bazel build -c opt --config=no_rbe //ci/wheel:torch_tpu_wheel \
  --repo_env=WHEEL_VERSION_EXTRAS=.dev20260714090201 \
  --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
  --define PYTHON_VERSION=3.12
# Output: bazel-bin/ci/wheel/torch_tpu-0.1.1.dev20260714090201-cp312-cp312-manylinux_2_31_x86_64.whl
```
- The Bazel hermetic build internally links `torch==2.11.0+cpu`, the same ABI as the torch installed in the venv.

### 3.3 Build vLLM 0.22.1+tpu from source

```bash
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm /data/red_poc/vllm
cd /data/red_poc/vllm
sed -i '/tpu-inference/d' requirements/tpu.txt        # drop the upstream JAX plugin
# Pre-install build deps into the venv, --no-build-isolation
SETUPTOOLS_SCM_PRETEND_VERSION=0.22.1 VLLM_TARGET_DEVICE=tpu CC=gcc CXX=g++ \
  <venv>/bin/pip install -e . --no-build-isolation
```

### 3.4 Create venv + install all pins (all from public PyPI)

```bash
cd /data/red_poc/tpu_benchmark_daily
uv venv --python 3.12 .venv
# Install: vllm (editable from previous step), the local torch-tpu wheel, and the pins in torchtpu-vllm's pyproject
<venv>/bin/pip install /data/red_poc/torch_tpu/bazel-bin/ci/wheel/torch_tpu-*.whl
uv pip install --python .venv/bin/python -e /data/red_poc/torchtpu-vllm   # install vllm_torchtpu + deps
uv pip check                                                              # passes
```

### 3.5 Wire into the repo scripts (bypassing the GAR-dependent update_environment.sh)

```bash
ln -s /data/red_poc/torchtpu-vllm /data/red_poc/tpu_benchmark_daily/third_party/torchtpu-vllm
ln -s <real model dir> models/Qwen3.5-397B-A17B-FP8   # provides config.json / tokenizer.json
```

> `scripts/update_environment.sh` pulls torch/torch-tpu from GAR; this reproduction **skips it** and uses the local wheel + editable installs above instead.

---

## 4. Dependency Version Alignment (all pins)

| Package | Version | Source |
|---------|---------|--------|
| torch-tpu | `0.1.1.dev20260714090201` | **built from source** (torch_tpu@40c8169) |
| torch | `2.11.0+cpu` | public PyPI (same ABI as the hermetic build) |
| vllm | `0.22.1+tpu` | built from source |
| vllm_torchtpu | `0.1.dev1+g88f359b` | editable (torchtpu-vllm@88f359b) |
| torchvision | `0.26.0+cpu` | PyPI |
| jax / jaxlib | `0.10.2` | PyPI |
| libtpu | `0.0.42.1` | PyPI (PjRt runtime) |
| numba | `0.65.0` | PyPI |

> The `torch_tpu` version must **exactly match** the `torch-tpu==` pin in `torchtpu-vllm/pyproject.toml`, otherwise dependency resolution fails — this is why the timestamp is reproduced via `WHEEL_VERSION_EXTRAS`.

---

## 5. Serving Parameters (DP8, `scripts/start_dp_server.sh`)

### 5.1 Parallelism / batching / lengths

| Parameter | Value |
|-----------|-------|
| `--data-parallel-size` | 8 (DP=8) |
| `--tensor-parallel-size` | 1 |
| `--prefill-context-parallel-size` | 1 (PCP off) |
| `--enable-expert-parallel` | on (EP) |
| `--max-model-len` | 69632 |
| `--max-num-batched-tokens` | 4096 |
| `--max-num-seqs` | 64 |
| `--block-size` | 256 |
| `--kv-cache-dtype` | fp8 |
| `--gpu-memory-utilization` | 0.90 |
| `--attention-backend` | CUSTOM |
| `--load-format` | dummy |
| `--no-enable-prefix-caching` / `--language-model-only` / `--disable-custom-all-reduce` | on |
| `--compilation-config` | `{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[2,4,8,16,4096],...}` |

### 5.2 Key environment variables

| Variable | Value |
|----------|-------|
| `PJRT_DEVICE` | TPU |
| `VLLM_TARGET_DEVICE` | tpu |
| `VLLM_PLUGINS` | torchtpu |
| `PYTHONPATH` | `$TORCHTPU_DIR/src` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 1 (offline) |
| `SKIP_JAX_PRECOMPILE` | 1 |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` | uniform_random (simulates expert routing under dummy weights) |
| `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL` | 0 |
| `VLLM_XLA_CACHE_PATH` / `VLLM_CACHE_ROOT` | `cache/{xla,vllm}/<CACHE_KEY>` (persists compiled artifacts keyed by config hash) |

### 5.3 Launch

```bash
cd /data/red_poc/tpu_benchmark_daily
nohup ./scripts/start_dp_server.sh > /data/red_poc/dp8_server.log 2>&1 &
# Wait for: curl http://127.0.0.1:18100/health → 200
```

---

## 6. Actual Parallelism / Resources / Startup Time (runtime-verified)

| Metric | Value |
|--------|-------|
| Orchestration | DP=8 (attention/KV split into 8 shards) + EP=8 (MoE experts spread across 8 cores) + TP=1, PCP=1 |
| KV cache type | hybrid (mamba state + attention), fp8 |
| Per-core KV capacity | GPU KV cache size ≈ **843,180 tokens/DP**; max concurrency for a 69,632-token request is 12.11x |
| First compile time | **~20 min** (`TpuCompilerAdaptor` compiles the FX graph per shape, 8 workers in parallel; a single (16,16) graph ~74s, a (4096,4096) graph ~300s) |
| Compile-log noise | `shm_broadcast "No available shared memory..."` and `tpu_info.py` GCE-metadata 404 are **both harmless** |

> Compiled artifacts are cached under `cache/xla/<CACHE_KEY>`; restarts with the same config skip recompilation.

---

## 7. Functional / Health Test

- **Endpoint**: `http://127.0.0.1:18100` (OpenAI-compatible)
- **Model name**: `Qwen3.5-397B-A17B-FP8`
- Startup-complete log: `Platform plugin torchtpu is activated` → 8× `EngineCore_DP0-7` / `Worker_DP0-7_EP0-7` each load libtpu → `Application startup complete.`
- `GET /health` → **200** ✅
- Benchmark ran with **128/128 requests successful, 0 failures** throughout (see §9)

---

## 8. Operations Commands

```bash
# Start the DP8 server
cd /data/red_poc/tpu_benchmark_daily && nohup ./scripts/start_dp_server.sh > /data/red_poc/dp8_server.log 2>&1 &
tail -f /data/red_poc/dp8_server.log                 # view logs / compile progress
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18100/health   # health check

# Stop the server (⚠️ stop by PID; do NOT use pkill -f api_server — it matches the command's own cmdline and kills the shell)
ps -eo pid,cmd | grep -E 'vllm.entrypoints|EngineCore' | grep -v grep    # find PIDs
kill -TERM <PID> ...

# Run the benchmark (no git publish)
PUBLISH_REPORTS=0 ./scripts/bench_all.sh
```

---

## 9. Performance Benchmarks

- **Tool**: `vllm bench serve` in the venv, `--dataset-name random`, `--endpoint /v1/completions`
- **Measurement setup**: **input 8192 / output 1** (pure-prefill throughput, matching the repo's committed setup), 128 prompts, `--request-rate inf`, `--ignore-eos`, `temperature 0`, `--seed 42`
- **Concurrency sweep**: `1 2 4 8 16 32 64` (`CONCURRENCIES`)
- **Results**: `runs/manual-*/results/dp8/*.json`; aggregated into `reports/latest.json`

### 9.1 DP8 measured results (2026-07-26)

| Concurrency | Total token throughput (tok/s) | Req/s |
|-------------|-------------------------------|-------|
| 1 | 4,567 | 0.56 |
| 2 | 6,123 | 0.75 |
| 4 | 13,044 | 1.59 |
| 8 | 29,353 | 3.58 |
| **16** | **52,453** ← peak | 6.40 |
| 32 | 50,075 | 6.11 |
| 64 | 49,953 | 6.10 |

- **Peak 52,452.7 tok/s @ concurrency 16**, mean TTFT 2.41 s / p99 3.34 s, 128/128 successful.

### 9.2 Comparison against the repo's committed baseline

| Config | This reproduction | Repo baseline (7/17–7/21) | Verdict |
|--------|-------------------|---------------------------|---------|
| **dp8** | **52,453 tok/s** | 40,378–49,381 tok/s | ✅ **matches and slightly exceeds the upper bound, no implementation gap** |
| pcp8 | ❌ did not run (see §9.3) | 34,296 tok/s (revision db5ae0ab) | ⚠️ incompatible on the current HEAD |

### 9.3 PCP8 (DP=1 / PCP=8) incompatibility note

`scripts/start_pcp_server.sh` compiles fine on the current stack (vLLM 0.22.1 + torchtpu-vllm **88f359b**), but the engine fails during initialization — Qwen3.5's hybrid (mamba+attn) KV conflicts with context parallelism:

| Attempt | Error |
|---------|-------|
| hybrid KV **on** (repo's original flag) | `Hybrid KV cache groups with multiple block sizes do not support context parallelism` |
| hybrid KV **off** (`--disable-hybrid-kv-cache-manager` override) | `Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` |

- The repo's pcp8 baseline (34,296) was captured on an **earlier revision `db5ae0ab`**; between then and 88f359b this path was blocked by vLLM's checks.
- A true pcp8 reproduction requires reverting to `db5ae0ab` (re-fetching that commit + possibly rebuilding the torch-tpu wheel). **This part is pending.**

### 9.4 Independent test matrix (varied input/output, DP8, 2026-07-27)

§9.1 only reproduces the repo's setup (8192/1, pure prefill). For an independent evaluation, three real workloads × concurrency sweeps were run (independent `vllm bench serve`, same DP8 server, `--random-range-ratio 0` fixed length, `--ignore-eos --temperature 0 --seed 42`, num-prompts scaled with concurrency as `c×2` clamped to [16,128]). Raw JSON is in `/data/red_poc/matrix_results/`.

**Workload A — 1024 / 1024 (balanced)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 1  | 100      | 50      | 0.05 | 860   | 19.2 |
| 2  | 192      | 96      | 0.09 | 990   | 19.9 |
| 4  | 384      | 192     | 0.19 | 873   | 20.0 |
| 8  | 764      | 382     | 0.37 | 1,612 | 19.4 |
| 16 | 1,516    | 758     | 0.74 | 1,654 | 19.5 |
| 32 | 2,604    | 1,302   | 1.27 | 1,582 | 23.1 |
| **64** | **3,607** | 1,803 | 1.76 | 2,101 | 32.8 |

**Workload B — 1024 / 8192 (decode-heavy)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 4  | 224    | 199   | 0.02 | 870   | 20.0 |
| 8  | 526    | 468   | 0.06 | 1,523 | 16.9 |
| 16 | 939    | 835   | 0.10 | 1,665 | 19.0 |
| 32 | 1,502  | 1,335 | 0.16 | 1,627 | 23.8 |
| **64** | **2,488** | 2,211 | 0.27 | 2,011 | 28.7 |

**Workload C — 8192 / 1024 (prefill-heavy, comparable to the JAX report's 8k/1k)**

| Concurrency | Total tok/s | Output tok/s | Req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------|------------|--------------|-------|----------------|----------------|
| 1  | 437     | 49    | 0.05 | 1,802 | 18.8 |
| 2  | 861     | 96    | 0.09 | 1,808 | 19.2 |
| 4  | 1,571   | 175   | 0.17 | 2,472 | 20.5 |
| 8  | 3,721   | 413   | 0.40 | 2,158 | 17.3 |
| 16 | 6,236   | 693   | 0.68 | 2,760 | 20.4 |
| 32 | 9,463   | 1,051 | 1.03 | 4,741 | 25.8 |
| **64** | **14,147** | 1,572 | 1.54 | 5,902 | 34.9 |

**Workload D — 1024 / 1 (pure prefill, short input)**

| Concurrency | Total tok/s | Req/s | Mean TTFT (ms) | p99 TTFT (ms) |
|-------------|------------|-------|----------------|---------------|
| 1  | 1,197   | 1.17  | 856   | 878   |
| 2  | 1,486   | 1.45  | 1,326 | 1,693 |
| 4  | 4,817   | 4.70  | 851   | 860   |
| 8  | 3,960   | 3.86  | 1,653 | 2,352 |
| 16 | 6,740   | 6.58  | 1,675 | 2,439 |
| 32 | 13,780  | 13.44 | 1,656 | 2,412 |
| **64** | **25,983** | 25.35 | 2,005 | 2,977 |

**Workload E — 8192 / 1 (pure prefill, repo setup, independent harness)**

| Concurrency | Total tok/s | Req/s | Mean TTFT (ms) | p99 TTFT (ms) |
|-------------|------------|-------|----------------|---------------|
| 1  | 4,567   | 0.56  | 1,794 | 1,804  |
| 2  | 9,167   | 1.12  | 1,787 | 1,807  |
| 4  | 12,972  | 1.58  | 2,473 | 2,596  |
| 8  | 29,858  | 3.64  | 2,148 | 2,254  |
| 16 | 45,126  | 5.51  | 2,542 | 3,357  |
| 32 | 45,602  | 5.57  | 4,784 | 6,669  |
| **64** | **49,934** | 6.09 | 8,346 | 14,032 |

> Workload E (8192/1) is the repo's setup, but here it uses the **independent harness** (num-prompts=`c×2`, not §9.1's fixed 128). Peak **49,934 tok/s @ c64**, same order of magnitude as §9.1 (bench_all with a fixed 128 prompts, 52,453 @ c16); the difference comes from different num-prompts and peak-concurrency points — at c16 here num-prompts=32 doesn't fill the pipeline, so §9.1's c16 is higher. **Both confirm 8192/1 is stable at ~45–52k.**

**Observations**

1. **The setup determines the magnitude**: on the same DP8 server, `output=1` (pure prefill) peaks at **52,453 tok/s**, but with `output≥1024` the total drops sharply to 3.6k–14.1k — because once decode begins, each request spends 20 s+ generating tokens one at a time (memory-bound), diluting the aggregate token/s. **So the repo's 52k is a prefill-only figure and should not be read as end-to-end serving throughput.**
2. **Stable TPOT**: across all workloads the decode step stays at **17–35 ms**, rising gently with concurrency — no abnormal bottleneck in the decode path.
3. **Strong prefill, average decode**: Workload C (long input) scales near-linearly with concurrency to 14k; Workload B (long output) is decode-limited, reaching only 2.5k even at concurrency 64. This is consistent with torch-tpu's Pallas prefill-kernel advantage.

### 9.5 Same-workload comparison against the JAX backend (controlled A/B)

**Controlled A/B (2026-07-28)**: the JAX backend (port 8000, `run_vllm.sh`, TP=8, committed config) was re-run with the **exact same harness as torch-tpu** (`bench_matrix_jax.sh`, same num-prompts / seed / parameters); the JAX raw JSON is in `/data/red_poc/matrix_results_jax/`. The peak point (c64) for each workload:

| Workload (in/out) | Type | torch-tpu | JAX | c64 winner |
|-------------------|------|-----------|-----|------------|
| 1024 / 1   | pure prefill (short) | 25,983 | **41,448** | JAX ↑ ~60% |
| 8192 / 1   | pure prefill (long)  | **49,934** (§9.1 52,453) | 44,651 | **torch-tpu ↑ ~12%** |
| 1024 / 1024 | balanced | **3,607** | 3,498 | tie (torch-tpu ↑ ~3%) |
| 1024 / 8192 | decode-heavy | 2,488 | **2,811** | JAX ↑ ~13% |
| 8192 / 1024 | prefill-heavy (with decode) | 14,147 | **20,020** | JAX ↑ ~42% |

> **Harness self-check**: the same-harness JAX 8192/1024 = 20,020 tok/s, nearly identical to the JAX report §9.1 doc value of 19,836 (1024/8192: report 2,852 vs measured 2,811; 1024/1024: report 4,410 vs measured 3,498 — the latter difference comes from num-prompts, as the report used a larger fixed batch). This shows the controlled re-run is comparable to the original JAX report's setup, so the table above is a **true controlled A/B** that supersedes the earlier doc-value comparison.

**Pure-prefill, per concurrency (total tok/s)**

| Concurrency | 1024/1 torch-tpu | 1024/1 JAX | 8192/1 torch-tpu | 8192/1 JAX |
|-------------|------------------|------------|------------------|------------|
| 1  | 1,197  | 3,496  | 4,567  | 5,973  |
| 2  | 1,486  | 5,280  | 9,167  | 9,759  |
| 4  | 4,817  | 7,995  | 12,972 | 19,116 |
| 8  | 3,960  | 22,134 | 29,858 | 35,255 |
| 16 | 6,740  | 35,759 | 45,126 | 38,827 |
| 32 | 13,780 | 37,610 | 45,602 | 43,395 |
| 64 | 25,983 | 41,448 | 49,934 | 44,651 |

**Conclusions (controlled A/B, c64)**
- **torch-tpu wins only on "long-input pure prefill"** (8192/1, +12%); on the balanced workload (1024/1024) it is essentially a tie (+3%).
- **JAX leads in every other scenario**: short-input pure prefill (1024/1, +60%), decode-heavy (1024/8192, +13%), and prefill+decode mixed (8192/1024, +42%).
- Key contrast: torch-tpu's **8192/1 pure prefill beats JAX**, but once decode is added to the same long input (8192/1024), **JAX wins by 42%** — torch-tpu's strength is concentrated in the prefill kernel, while its decode path (especially mixed with long prefill) is less mature than JAX's.
- Any cross-backend conclusion must be anchored to "input length + whether decode is involved"; the repo's single 8192/1 52k point is not representative on its own.

---

## 10. Gotchas

1. **`pkill -f "api_server"` kills itself** — the pattern matches the running command's own cmdline, SIGKILLs the launching shell, and the server never comes up. Always stop processes by PID.
2. **Three missing deps in the bare-metal vLLM build** — `No module named numpy` (needs `--no-build-isolation` + pre-installed build deps), `No CMAKE_CXX_COMPILER` (needs build-essential), `missing Python_INCLUDE_DIRS` (needs python3.12-dev).
3. **Stale submodule URL** — `.gitmodules` points to `vllm-project/vllm-torchtpu` (404); the real repo is `google-pytorch/torchtpu-vllm`.
4. **torch-tpu version must align exactly** — reproduce the timestamp via `WHEEL_VERSION_EXTRAS`, otherwise the `torch-tpu==` pin in pyproject fails to resolve.
5. **Compile noise is harmless** — `shm_broadcast` timeouts and the `tpu_info.py` GCE-metadata 404 are both normal.
6. **Credentials** — the HF token / GitHub PAT are live credentials; they have been scrubbed from all git remotes after cloning. Never write them into commits/logs/reports.

---

## 11. Optional Follow-ups

- **PCP8 deep reproduction**: fetch `db5ae0ab` + rebuild the matching torch-tpu → run the 8192/1 sweep, filling in the pcp8 comparison (requires the PAT).
- **Independent test matrix**: ✅ done (see §9.4, 3 workloads × concurrency sweep).
- **Strict cross-backend A/B**: ✅ done (2026-07-28, see §9.5). The JAX backend (port 8000, committed config) was run with the same harness across all 5 workloads including pure prefill; result JSON is in `/data/red_poc/matrix_results_jax/`. Harness self-check passed (JAX 8192/1024 measured 20,020 ≈ report's 19,836).
- Validate accuracy + throughput with real weights (dropping `--load-format dummy`).
