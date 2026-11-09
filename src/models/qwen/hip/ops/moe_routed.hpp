#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_ROUTED_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_ROUTED_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/// Routed-expert F16 WMMA GEMMs for Qwen3.6-35B-A3B (qwen35moe) prefill.
///
/// These began as Flash-Next's routed expert GEMMs and keep that layout. The
/// 35B needs Q6_K experts, paired Q6_K/Q8_0 gate/up launches and 64-row paired
/// tiles, which Flash-Next's kernels do not provide, so the Qwen model owns
/// this copy and its numerics; Flash-Next's own kernels evolve separately.
namespace gufo::hip::moe_routed {

/// GGUF type ids the routed GEMMs accept.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kBF16 = 30,
};

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, 48-row token tile) as expert | tile << 16,
/// so empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result.
/// `tile_rows` is the token rows per tile the map was built with: 16 or 48,
/// or 64 / 128 (Q4_K and Q5_K: F16 output without SwiGLU only). K-quant
/// tiles of 64 / 128 rows with that output take register-resident kernels
/// whose values equal the narrow tiles'.
/// Q4_K, Q5_K and Q6_K need k % 256 == 0, Q5_1 and Q8_0 k % 64 == 0; other
/// types and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Whether RoutedGatedF16Gemm decodes `type`.
bool RoutedGatedF16Supported(WeightType type);

/// Paired Q4_K/Q5_K/Q6_K/Q8_0 gate/up GEMM with SwiGLU and F16 output. Uses
/// the same routing layout as RoutedF16Gemm, with 64 or 128 token rows per
/// tile. The two projections share one launch and never materialize the gate
/// output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::hip::moe_routed

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_ROUTED_HPP_
