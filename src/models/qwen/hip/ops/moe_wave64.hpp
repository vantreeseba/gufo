#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_WAVE64_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_WAVE64_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

// Experimental (GUFO_EXP_MOE_WAVE64): routed expert GEMVs for decode and
// batched verification, compiled as wave64. Weights are decoded per 16-element
// sub-block with the format fixed at compile time, and dotted with the FP32
// activations directly (no Q8_1 quantize launch). Each launcher returns false,
// launching nothing, when it does not cover the format or shape; callers then
// keep the default route.

/// Whether the wave64 expert GEMVs decode `type` over `k` inputs.
[[nodiscard]] bool IsMoeWave64Type(core::GgmlType type, std::size_t k);

/// act[s, :] = SiLU(gate_e * x) * (up_e * x) for e = ids[s], where x is row
/// s / x_token_divisor. gate and up must share a format.
[[nodiscard]] bool LaunchMoeWave64SwigluGemv(
    const void* gate_w, core::GgmlType gate_type, const void* up_w,
    core::GgmlType up_type, const float* x, const std::int32_t* ids,
    float* act, std::size_t expert_ff, std::size_t hidden, std::uint32_t slots,
    std::uint32_t x_token_divisor, hipStream_t stream);

/// out[s, :] = W_e * x[s, :] for e = ids[s] (per-slot input rows).
[[nodiscard]] bool LaunchMoeWave64Gemv(const void* w, core::GgmlType type,
                                       const float* x, const std::int32_t* ids,
                                       float* out, std::size_t m,
                                       std::size_t k, std::uint32_t slots,
                                       hipStream_t stream);

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_WAVE64_HPP_
