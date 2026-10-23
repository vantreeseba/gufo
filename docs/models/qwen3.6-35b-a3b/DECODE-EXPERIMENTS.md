# Qwen3.6 35B-A3B decode experiments

Temporary switches for A/B testing decode speedups on gfx1151. Each one either
becomes the default (and loses its switch) or is reverted; cleanup is tracked
in [vantreeseba/gufo#1](https://github.com/vantreeseba/gufo/issues/1).

Every switch is an environment variable read once when the process starts.
Captured decode graphs record whichever kernels were active, so restart
`gufo` to change a switch. When any switch is on, `gufo` logs one line to
stderr listing the active settings.

| Variable | Effect | Falls back when |
| --- | --- | --- |
| `GUFO_EXP_MOE_Q6K_MMVQ=1` | Q6_K experts use the MMQ int8 vector kernels (Q8_1 activations) instead of the per-slot FP32 GEMV. Gate/up uses it for single-token decode; down uses it for any batch. | The tensor is not Q6_K, or K is not a multiple of 256 |
| `GUFO_EXP_MOE_WAVE64=1` | Routed expert gate/up (fused SwiGLU) and down run on wave64 kernels that decode Q8_0/Q4_K/Q5_K/Q6_K with the format fixed at compile time and dot with FP32 activations. Each row gets a 32-lane half of a wave64 (16 lanes for short rows); reductions never cross the two halves. No activation quantize launch. Takes precedence over the MMQ route. | Other formats, or gate and up in different formats |
| `GUFO_EXP_MOE_FUSE=router` | Router logits, the shared-expert gate logit and top-k in one launch (3 → 1 per MoE layer). | Router or shared-gate format not decoded by the warp row dot |
| `GUFO_EXP_MOE_FUSE=combine` | The shared expert's down projection is fused into the MoE combine (2 → 1 per MoE layer). | Shared-down format not decoded by the warp row dot |
| `GUFO_EXP_MOE_FUSE=router,combine` or `all` | Both fusions. | |

The switches apply to single-token decode, batched verification (DFlash2
and concurrent sessions) and the MTP head's MoE layer. Prefill is unchanged.

## HRX runtime (no code change)

[HRX](https://github.com/ROCm/hrx-system) is AMD's early-access alternative
HIP runtime. Its compatibility layer installs a `libamdhip64.so`, so the same
binary can run on it:

```sh
LD_PRELOAD=/path/to/hrx/lib/libamdhip64.so HRX_GPU_DRIVER=amdgpu \
  ./result/bin/gufo prompt --model "$MODEL" --prompt "Explain virtual memory." --temperature 0
```

Check correctness before timing. HIP graph support is undocumented in HRX
and gufo relies on graphs, so a failure at graph capture is the most likely
blocker.

## Testing

1. Correctness of each kernel against the route it replaces (runs regardless
   of the switches): `qwen35ba3b_moe_ops_test`. It prints a relative error
   per check; wave64 and the fused router should agree within 1e-4, and the
   Q6_K MMQ route within the int8 activation rounding (2e-2).
2. End to end, with each switch set: `qwen35ba3b_target_test`.
3. Speed: tg128 per switch versus all switches off, same binary, on
   UD-Q4_K_XL, UD-Q6_K_XL and UD-Q8_K_XL, plus tg128 at 16K and 32K depth,
   DFlash2 decode, and `gufo serve` with 1-4 sessions.

```sh
for flags in "" GUFO_EXP_MOE_Q6K_MMVQ=1 GUFO_EXP_MOE_WAVE64=1 \
             GUFO_EXP_MOE_FUSE=router GUFO_EXP_MOE_FUSE=combine \
             GUFO_EXP_MOE_FUSE=all; do
  env $flags ./result/bin/gufo ...   # the usual tg128 benchmark
done
```

Expectations: Q6K_MMVQ mainly affects UD-Q6_K_XL, the quant whose decode
trails llama.cpp Vulkan. WAVE64 affects every quant, since it replaces the
MMQ route as well. The fusions remove three launches per MoE layer per token. On gfx1151, HIP graph replay costs about 1.8 µs per node
(ROCm/rocm-systems#10834), so expect a gain of roughly 1-2%.
