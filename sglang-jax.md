# Qwen3.5-397B-A17B-FP8 on sglang-jax (Ironwood TPU v7x)

Experimental branch: getting **[sglang-jax](https://github.com/sgl-project/sglang-jax)**
to serve the FP8 checkpoint on a single v7x-4 host (8 devices), plus a same-harness
benchmark vs the vLLM `tpu_inference` (JAX) and torch-tpu backends.

> **Status: functional but NOT competitive.** sglang-jax now *loads and serves*
> the model, but throughput is far below the vLLM backends (see table). This is a
> first-working, mixed-precision port — kept on a branch for reference, **not**
> promoted to `main`.

## What was needed

sglang-jax ships a `qwen3_5` model class (hybrid GDN linear-attn + MoE) but it
(a) assumed a **pre-fused** expert checkpoint and (b) had **no FP8 path**. This
HF release is **per-expert unfused, FP8 block-quant (128×128)** with
`weight_scale_inv` sidecars.

- `sglang-jax/qwen3_5-fp8.patch` — patch against `sgl-project/sglang-jax`
  (`python/sgl_jax/srt/models/qwen3_5.py` + new
  `python/sgl_jax/srt/utils/quantization/configs/fp8_qwen3_5.yaml`). Adds a
  static-FP8 load path: routed experts via `create_moe_weights_mapping` +
  `weight_scale_inv` sidecars (mirrors `qwen3_moe`), full-attn q/o_proj in FP8,
  and manual fp8→bf16 dequant for k/v, shared-expert, and GDN projections
  (block-fp8 kernel narrow-N guard + GQA scale replication at TP=8).
- `sglang-jax/fp8_qwen3_5.yaml` — the quant config (also inside the patch).

Precision: the **512 routed experts (the bulk) stay FP8**; k/v + shared-expert +
GDN are dequantized to bf16. So it's mixed precision, unlike the fully-FP8 vLLM
runs — a caveat for the comparison below.

## Working launch (sglang-jax)

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
  --model-path /path/to/Qwen3.5-397B-A17B-FP8 --trust-remote-code \
  --tp-size 8 --device tpu --dtype bfloat16 \
  --quantization-config-path fp8_qwen3_5.yaml \
  --mem-fraction-static 0.8 --page-size 128 --skip-server-warmup \
  --disable-radix-cache \
  --host 0.0.0.0 --port 30000
```
`--disable-radix-cache` is required (GDN hybrid-recurrent). Verified with
`/generate` → "I am a student." → "Je suis étudiant." (coherent).

## Benchmark (same matrix/knobs as `bench_jax.sh`)

`scripts/bench_sglang-jax.sh` — same 5 workloads × concurrency sweep,
`num-prompts=clamp(2c,16,128)`, greedy, ignore-eos, seed 42. Uses
`sgl_jax.bench_serving` (native equivalent of `vllm bench serve`; same knobs/
metrics, different client binary). Raw per-point JSON in `results/sglang-jax/`.

### TOTAL token throughput (tok/s), c64

| Workload | sglang-jax | jax (vLLM) | torch-tpu |
|----------|-----------:|-----------:|----------:|
| E 8192/1 (prefill headline) | 1,018 | 44,651 | 49,934 |
| A 1024/1024 (balanced)      | 746   | 3,498  | 3,607  |
| C 8192/1024 (prefill-heavy) | 919   | 20,020 | 14,147 |
| B 1024/8192 (decode-heavy)  | 674   | 2,811  | 2,488  |

### Prefill scaling (E 8192/1) — sglang-jax barely batches

| conc | sglang-jax | jax | torch-tpu |
|-----:|-----------:|----:|----------:|
| 1  | 916   | 5,973  | 4,567  |
| 8  | 1,017 | 35,255 | 29,858 |
| 64 | 1,018 | 44,651 | 49,934 |

## Why it's slow (honest)

- **Prefill ~40–50× slower**: sglang-jax plateaus at ~1000 tok/s and does not
  scale with concurrency — the bf16-dequantized k/v + GDN + shared path and the
  unoptimized GDN/megablox kernels dominate. Decode (A/B) gap is ~4–5×.
- Not a hardware limit — it reflects sglang-jax's current `qwen3_5` + FP8
  maturity vs the production `tpu_inference` path.

## MTP: blocked (framework level)

The checkpoint's `mtp.*` head is a full-attn MoE block (same FP8 layout handled
here), but sglang-jax has no `qwen3_5` MTP/nextn draft class, and — the hard
blocker — its EAGLE/NEXTN/multi-layer draft workers have **no GDN recurrent-state
handling** (only DFLASH does copy-on-write/rollback, and it needs a dedicated
draft checkpoint, not the in-checkpoint head). `server_args.py` states recurrent
buffers "does not support speculative decoding yet." Enabling MTP needs
recurrent-aware speculative runtime work + a new draft model — out of scope for a
model-file change.
