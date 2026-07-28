# Qwen3.5-397B-A17B-FP8 on Ironwood TPU (v7x)

Deployment and throughput-benchmark notes for serving **Qwen3.5-397B-A17B-FP8**
on a single Ironwood TPU v7x host (`tpu7x-standard-4t`, 4 chips / 8 cores) with
vLLM, across two independent TPU backends.

## Documents

| Doc | Backend | Contents |
|-----|---------|----------|
| [`torch-tpu.md`](torch-tpu.md) | **torch-tpu** — native PyTorch PrivateUse1 (source-built, port 18100, DP8) | Build-from-source (GAR-free), serving, full independent benchmark matrix, and a controlled A/B vs the JAX backend |
| [`JAX.md`](JAX.md) | **tpu_inference** — JAX (Docker, port 8000, TP=8 + attention DP + MoE EP) | Docker deployment, serving config, and benchmarks |

Both backends drive the same 4 TPU chips, so only one can run at a time; the A/B
in `torch-tpu.md` §9.5 was measured by stopping one and bringing up the other
under an identical harness.

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run_vllm.sh`](scripts/run_vllm.sh) | Launch the JAX (`tpu_inference`) Docker backend. Set `HF_TOKEN` before running. |
| [`scripts/bench_matrix.sh`](scripts/bench_matrix.sh) | Independent benchmark matrix (5 workloads × concurrency sweep) against the torch-tpu server (port 18100). |
| [`scripts/bench_matrix_jax.sh`](scripts/bench_matrix_jax.sh) | Same harness against the JAX server (port 8000). |

## Raw results

`results/torch-tpu/` and `results/jax/` hold the raw `vllm bench serve` JSON for
every `input/output/concurrency` point behind the tables in the docs. Filenames
follow `in<INPUT>_out<OUTPUT>_c<CONCURRENCY>.json`.

## Workloads

| Load | input / output | Character |
|------|----------------|-----------|
| A | 1024 / 1024 | balanced |
| B | 1024 / 8192 | decode-heavy |
| C | 8192 / 1024 | prefill-heavy (mixed) |
| D | 1024 / 1 | short pure-prefill |
| E | 8192 / 1 | long pure-prefill (repo's headline metric) |

> Note: `scripts/run_vllm.sh` ships with a placeholder `<YOUR_HF_TOKEN>`; supply
> your own Hugging Face token via the `HF_TOKEN` environment variable.
