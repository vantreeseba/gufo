# Qwen3.6 35B-A3B quality

These are execution-consistency checks: the GPU executor against the CPU
reference forward pass, plus GPU prefill/decode self-consistency. Original-target
and GGUF-conversion parity are not qualified here.

## What the target test checks

`qwen35ba3b_target_test` (gated by `GUFO_QWEN35BA3B_MODEL`, skipped when unset)
requires the loaded config to report an MoE architecture, then runs:

| Check | Method | Asserted contract |
| --- | --- | --- |
| GPU/CPU greedy parity, position 0–11 | Two fixed text fixtures; the GPU re-prefills each prefix while the CPU reference steps the same tokens (teacher forcing). Compares the final-position logits. | GPU logits must be **finite** and the greedy (argmax) next token must **equal the CPU reference** at every position. `CompareLogits` is called with `atol = rtol = 5e-2`, but the elementwise `match` that drives is not asserted; per-position max-abs and RMSE are printed for reference only. |
| Prefill/decode self-consistency | Same full prompt: one `ForwardPromptBatch` prefill versus a one-token-at-a-time `ForwardToken` decode loop. | The two next-token IDs must be **exactly equal**. |
| Concurrent verification | Three sessions verify 8, 8 and 7 rows together through `ForwardVerificationBatch` (23 stacked rows, more than the 8-row Q8_0 gated vector kernel takes per launch); each session's own `ForwardVerificationChunk` is the reference. | Logits **finite** and the greedy token of **every row equal** to the per-session reference (`CompareLogits`, `atol = rtol = 5e-2`). |

Executor context is capped at 4096 tokens for both checks.

On the non-UD `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` the GPU/CPU parity check
fails at position 10 of the first fixture (greedy tokens differ at a near-tie;
logit 22 is 0.0921 from the reference against a 0.0915 tolerance), identically
on `8c985f93` and with the November 2026 routed kernels, which those 10-12-row
prefills do not reach. UD-Q4_K_XL passes. The parity fixtures
must tokenize to at least 12 / 16 tokens respectively, or the test fails rather
than skips.

## MoE operator test

`qwen35ba3b_moe_ops_test` is a synthetic, model-free check of the MoE GEMM
families (gate/up/down, routed + shared) against a CPU reference. It qualifies
the kernels in isolation; it does not cover the full model or the quantized
GGUF path. It also checks the grouped BF16 GEMM against the per-slot GEMV
for 1, 7, 100 and 512 tokens on both projection shapes, requiring every
slot written, relative error < 1e-2, and byte-identical output across two
runs. The op test also checks the dense mode of the BF16 WMMA GEMM against
the FP32-activation GEMV for 1, 100 and 512 rows (relative error < 1e-2,
every row written). It also checks the routed F16 expert GEMM (Q8_0, Q6_K,
Q5_K and Q4_K; F32 rows over 16- and 48-row tiles, F16 rows over 48-, 64- and
128-row tiles) and the paired gate/up GEMM (the same formats, 64- and 128-row
tiles) against per-slot GEMV references, requiring every output written and
relative error below 1e-2 (routed) and 5e-3 (paired). The F16 rows of the 64-
and 128-row tiles, which take the register-resident kernels, must be
byte-identical to the 48-row tiles.

## MTP head

`qwen_mtp_gpu_test`, run with the 35B GGUF as both base and draft, compares
the GPU MTP layer with the CPU reference (hidden cosine > 0.999, selected
logits within 2.0) and checks reset determinism. Measured on UD-Q8_K_XL:
hidden cosine 0.999992 (RMSE 0.0108, max abs 0.0414), selected-logit max abs
error 0.0102; UD-Q6_K_XL gives the same figures. For `gufo serve` it also
checks the batched prompt prime and the batched append of verified rows
against the serial per-token path (logit cosine > 0.99999), chunk-size
invariance, and that the memory estimate matches the allocation.

## DFlash2 draft

`qwen_dflash_gpu_test`, run with this model as the target and the converted
Q8_0 DFlash2 draft (`GUFO_QWEN27B_MODEL`, `GUFO_QWEN27B_DFLASH_MODEL`),
passes its draft-state, snapshot and batched-proposal checks.

For this MoE target, prompt context of at least 128 rows is projected into
the draft's KV cache with BF16-rounded activations on the WMMA GEMM; dense
targets keep the exact FP32-activation route. This changes only the draft's
context, never the verified output distribution. On three 2.4K-3.7K-token
prompts decoded greedily for 768 tokens, outputs and draft/accepted counts
were identical before and after the change. The concurrent draft cap only
shortens proposals, so it does not change the verified output distribution
either; greedy output can still flip at near-ties because the verification
width changes the MoE numeric route (see [README](README.md)).

## Scope of the target test for prefill changes

The parity fixtures are 12-16 tokens, so the target test never reaches the
routes that only large chunks take (the paired routed tile map). Changes there
are qualified by comparing greedy completions on 6K and 12K prompts against the
previous build in server mode; the October 1, 2026 prefill changes keep them
byte-identical on UD-Q6_K_XL and UD-Q8_K_XL, and both artifacts pass the target
test and the MoE operator test.

## BF16 router and GDN controls in prefill

Prefill chunks of at least 128 tokens round the F32 router and the GDN
alpha/beta projections to BF16 (see [Experiments](EXPERIMENTS.md)), so their
output is not byte-identical to the FP32 route. The change was qualified on
UD-Q6_K_XL, greedy, plain decoding, against the same source without it:

| Check | FP32 route | BF16 route |
| --- | ---: | ---: |
| 5-shot MMLU, 969 questions over 57 subjects, mean prompt 651 tokens | 741 (76.47%) | 739 (76.26%) |
| Aggregation over 40/120/300 order records, 90 questions, up to 5.5K tokens | 40 | 40 |
| Recall of one or two access codes among three, 54 prompts at 2K-12K tokens | 54 | 54 |
| Recall of three codes among sixteen similar names, 30 prompts at 6K-24K tokens | 90 of 90 | 90 of 90 |

On MMLU the two routes pick the same letter on 97.0% of questions; 13 are right
only with FP32 and 11 only with BF16 (exact McNemar p = 0.84). `--validate-prefill`
keeps the sequential reference's top-1 token at 512, 2048 and 4096 tokens, with
an RMSE of 0.087 / 0.564 / 1.85 against 0.086 / 0.523 / 2.10 for the FP32 route.
Both artifacts pass the target test. This bounds a loss to about one MMLU point;
it does not measure long generations, tool calling or speculative decoding.

## Vision

The F16 `mmproj` widens to BF16 exactly (all 443,510,784 weights round-trip).
[`tools/qwen27b/vision_reference.py`](../../../tools/qwen27b/vision_reference.py)
compares Gufo's encoder trace with the pinned Transformers `qwen3_5_moe`
vision operators on the same weights, and both with an FP32 run of those
operators. Its pass condition: every stage within 5% relative L2 of the BF16
reference, each re-run layer within 1.5%, and the final embedding no further
from FP32 than 1.25x the reference's own BF16 error plus 0.001. Synthetic
screenshots of 1280x704, 1024x1024 and 2560x1440 resized pixels, October 26,
2026; relative L2 of the final embedding against FP32, and the largest
re-run layer difference:

| Image | Encoder | Embedding vs FP32 | Bound | Re-run layer | Pass |
| --- | --- | ---: | ---: | ---: | --- |
| 35B-A3B 1280x704 | before fused attention | 0.0747 | 0.0844 | 0.00154 | no (embedding 5.12% from BF16) |
| 35B-A3B 1280x704 | fused attention | 0.0621 | 0.0844 | 0.00148 | yes |
| 35B-A3B 1024x1024 | before | 0.0910 | 0.0943 | 0.00162 | no (6.93%) |
| 35B-A3B 1024x1024 | fused | 0.0562 | 0.0943 | 0.00149 | no (5.55%) |
| 35B-A3B 2560x1440 | before | 0.0696 | 0.0749 | 0.00188 | yes |
| 35B-A3B 2560x1440 | fused | 0.0607 | 0.0749 | 0.00159 | yes |
| 27B 1280x704 | before | 0.0622 | 0.0742 | 0.00154 | yes |
| 27B 1280x704 | fused | 0.0611 | 0.0742 | 0.00140 | yes |
| 27B 1024x1024 | before | 0.0413 | 0.0390 | 0.00155 | no |
| 27B 1024x1024 | fused | 0.0344 | 0.0390 | 0.00139 | yes |
| 27B 2560x1440 | before | 0.0345 | 0.0477 | 0.00183 | yes |
| 27B 2560x1440 | fused | 0.0366 | 0.0477 | 0.00150 | yes |

The reference's own BF16 run is 6-7.5% from FP32 on this projector, so the
5% stage limit, set on the 27B, is not met by the earlier encoder either on two
of three images; it is left unchanged. Served answers about the 2560x1440 image
match in content before and after.

### Functional image suites

The `qwen36-35b-q4/off` and `qwen36-35b-q4/mtp` profiles run the HTTP image
suites from [tests/functional](../../../tests/functional/README.md) on
UD-Q4_K_XL with `mmproj-BF16.gguf`, 32K context and two sessions. Run on
November 11, 2026 at `3c03cce1` (openai SDK 2.53.0):

| Suite | AR | MTP |
| --- | --- | --- |
| `discovery` (`text,image`) | pass | pass |
| `messages-image-counts` | pass | pass |
| `image-inputs` | pass | pass |
| `image-count`, including cancel and disk cancel | pass | pass |
| `image-detail` | pass | pass |
| `tool-images` | fail | fail |

Both `tool-images` failures are the same case, `tool_images_two_calls`: two
function results, each holding one image, with the request "give their
dominant colors in that order, separated by a comma and no spaces" and a
32-token cap. The 35B describes each image before giving the answer, so it runs
out of tokens. With a 256-token cap it names both colors correctly in both
orders and ends with exactly `red,blue` (or `blue,red` with the images
swapped). Image order and grouped tool-result rendering are therefore correct;
the model does not follow the answer format. Every earlier `tool-images` case
passed; the invalid-input cases after it did not run. The case is left
unchanged.

## Reproduce

Build the `gpu-test` preset and run the binaries inside the ROCm container:

```sh
cmake --build --preset gpu-test --target qwen35ba3b_target_test qwen35ba3b_moe_ops_test qwen_mtp_gpu_test --parallel 16
export GUFO_QWEN35BA3B_MODEL=path/to/Qwen3.6-35B-A3B-MTP-UD-Q8_K_XL.gguf
./build/gpu-test/bin/qwen35ba3b_target_test
./build/gpu-test/bin/qwen35ba3b_moe_ops_test
./build/gpu-test/bin/qwen_mtp_gpu_test "$GUFO_QWEN35BA3B_MODEL" "$GUFO_QWEN35BA3B_MODEL"
```

A missing-weight skip is not a pass.

## Meaning of exact

Matching the CPU reference's greedy token tests GPU/CPU execution consistency,
not original-model accuracy; a quantized GPU path and a BF16 CPU path can
disagree within the tolerance above. Keep the reference, weights and fixtures
fixed before interpreting any numerical difference.
