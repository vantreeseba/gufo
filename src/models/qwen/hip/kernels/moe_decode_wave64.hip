#if defined(ENGINE_ENABLE_HIP)
// Experimental (GUFO_EXP_MOE_WAVE64). Compiled with -mwavefrontsize64.
//
// llama.cpp's Vulkan backend decodes Qwen3.6-35B-A3B faster than the HIP
// route on gfx1151; its mat-vec shaders dequantize the weights and dot them
// with float activations in wide subgroups. This unit tries the same shape:
// one wave64 (or a 32/16-lane slice of one, for short rows) per output row,
// every lane owning whole 16-element sub-blocks, FP32 activations read
// straight from the token row.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/moe_wave64.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kWave = 64;
constexpr std::uint32_t kThreads = 256;

/// Bytes of one row of `k` elements; the formats this unit decodes.
template<core::GgmlType Type>
__host__ __device__ constexpr std::size_t RowBytes(std::size_t k) {
  if constexpr (Type == core::GgmlType::kQ8_0) {
    return (k / 32) * 34;
  } else if constexpr (Type == core::GgmlType::kQ4_K) {
    return (k / 256) * 144;
  } else if constexpr (Type == core::GgmlType::kQ5_K) {
    return (k / 256) * 176;
  } else {
    static_assert(Type == core::GgmlType::kQ6_K);
    return (k / 256) * 210;
  }
}

/// Dot of one quantized row with x over `Lanes` lanes. The sum is valid in
/// every lane of the slice.
template<core::GgmlType Type, std::uint32_t Lanes>
__device__ inline float RowDot(const std::uint8_t* __restrict__ row,
                               const float* __restrict__ x, std::size_t k,
                               std::uint32_t lane) {
  const std::size_t subs = k / 16;
  float sum = 0.0F;
  QuantSub16 sub;
  for (std::size_t sb = lane; sb < subs; sb += Lanes) {
    DecodeQuantSub16(Type, row, sb, sub);
    const float4* xb = reinterpret_cast<const float4*>(x + (sb * 16));
    float dot = 0.0F;
    float xsum = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 v = xb[i];
      dot += static_cast<float>(sub.q[(4 * i) + 0]) * v.x;
      dot += static_cast<float>(sub.q[(4 * i) + 1]) * v.y;
      dot += static_cast<float>(sub.q[(4 * i) + 2]) * v.z;
      dot += static_cast<float>(sub.q[(4 * i) + 3]) * v.w;
      if constexpr (Type == core::GgmlType::kQ4_K ||
                    Type == core::GgmlType::kQ5_K) {
        xsum += (v.x + v.y) + (v.z + v.w);
      }
    }
    sum += (sub.scale * dot) - (sub.offset * xsum);
  }
  // XOR offsets below Lanes stay inside the lane's slice of the wave.
#pragma unroll
  for (std::uint32_t offset = Lanes / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor(sum, static_cast<int>(offset));
  }
  return sum;
}

template<core::GgmlType Type, std::uint32_t Lanes>
__launch_bounds__(kThreads) __global__
    void MoeWave64SwigluKernel(const std::uint8_t* __restrict__ gate_w,
                               const std::uint8_t* __restrict__ up_w,
                               const float* __restrict__ x,
                               const std::int32_t* __restrict__ ids,
                               float* __restrict__ act, std::uint32_t expert_ff,
                               std::uint32_t hidden, std::uint32_t x_divisor) {
  constexpr std::uint32_t kRowsPerBlock = kThreads / Lanes;
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % Lanes;
  const std::uint32_t row = (blockIdx.x * kRowsPerBlock) + (threadIdx.x / Lanes);
  if (row >= expert_ff) {
    return;
  }
  const std::size_t row_bytes = RowBytes<Type>(hidden);
  const std::size_t expert =
      static_cast<std::uint32_t>(ids[slot]);
  const std::size_t offset =
      (expert * expert_ff + row) * row_bytes;
  const float* x_row =
      x + static_cast<std::size_t>(x_divisor == 0 ? slot : slot / x_divisor) *
              hidden;
  const float g = RowDot<Type, Lanes>(gate_w + offset, x_row, hidden, lane);
  const float u = RowDot<Type, Lanes>(up_w + offset, x_row, hidden, lane);
  if (lane == 0) {
    act[static_cast<std::size_t>(slot) * expert_ff + row] =
        (g / (1.0F + __expf(-g))) * u;
  }
}

template<core::GgmlType Type, std::uint32_t Lanes>
__launch_bounds__(kThreads) __global__
    void MoeWave64GemvKernel(const std::uint8_t* __restrict__ w,
                             const float* __restrict__ x,
                             const std::int32_t* __restrict__ ids,
                             float* __restrict__ out, std::uint32_t m,
                             std::uint32_t k) {
  constexpr std::uint32_t kRowsPerBlock = kThreads / Lanes;
  const std::uint32_t slot = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % Lanes;
  const std::uint32_t row = (blockIdx.x * kRowsPerBlock) + (threadIdx.x / Lanes);
  if (row >= m) {
    return;
  }
  const std::size_t expert = static_cast<std::uint32_t>(ids[slot]);
  const std::size_t offset = (expert * m + row) * RowBytes<Type>(k);
  const float sum = RowDot<Type, Lanes>(
      w + offset, x + static_cast<std::size_t>(slot) * k, k, lane);
  if (lane == 0) {
    out[static_cast<std::size_t>(slot) * m + row] = sum;
  }
}

/// Lanes per row: a whole wave64 once every lane gets two or more
/// sub-blocks, narrower slices (more rows per wave) for short rows.
std::uint32_t LanesFor(std::size_t k) {
  const std::size_t subs = k / 16;
  if (subs >= 2 * kWave) {
    return 64;
  }
  if (subs >= 32) {
    return 32;
  }
  return 16;
}

template<typename Launch>
bool DispatchType(core::GgmlType type, std::size_t k, Launch&& launch) {
  const auto by_lanes = [&]<core::GgmlType Type>() {
    switch (LanesFor(k)) {
      case 64:
        launch.template operator()<Type, 64>();
        break;
      case 32:
        launch.template operator()<Type, 32>();
        break;
      default:
        launch.template operator()<Type, 16>();
        break;
    }
    return true;
  };
  switch (type) {
    case core::GgmlType::kQ8_0:
      return by_lanes.template operator()<core::GgmlType::kQ8_0>();
    case core::GgmlType::kQ4_K:
      return by_lanes.template operator()<core::GgmlType::kQ4_K>();
    case core::GgmlType::kQ5_K:
      return by_lanes.template operator()<core::GgmlType::kQ5_K>();
    case core::GgmlType::kQ6_K:
      return by_lanes.template operator()<core::GgmlType::kQ6_K>();
    default:
      return false;
  }
}

}  // namespace

bool IsMoeWave64Type(core::GgmlType type, std::size_t k) {
  switch (type) {
    case core::GgmlType::kQ8_0:
      return k % 32 == 0 && k > 0;
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
      return k % 256 == 0 && k > 0;
    default:
      return false;
  }
}

bool LaunchMoeWave64SwigluGemv(const void* gate_w, core::GgmlType gate_type,
                               const void* up_w, core::GgmlType up_type,
                               const float* x, const std::int32_t* ids,
                               float* act, std::size_t expert_ff,
                               std::size_t hidden, std::uint32_t slots,
                               std::uint32_t x_token_divisor,
                               hipStream_t stream) {
  if (gate_type != up_type || !IsMoeWave64Type(gate_type, hidden) ||
      slots == 0 || expert_ff == 0) {
    return false;
  }
  return DispatchType(
      gate_type, hidden, [&]<core::GgmlType Type, std::uint32_t Lanes>() {
        constexpr std::uint32_t kRowsPerBlock = kThreads / Lanes;
        const dim3 grid(static_cast<std::uint32_t>(
                            (expert_ff + kRowsPerBlock - 1) / kRowsPerBlock),
                        slots);
        MoeWave64SwigluKernel<Type, Lanes><<<grid, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(gate_w),
            static_cast<const std::uint8_t*>(up_w), x, ids, act,
            static_cast<std::uint32_t>(expert_ff),
            static_cast<std::uint32_t>(hidden), x_token_divisor);
      });
}

bool LaunchMoeWave64Gemv(const void* w, core::GgmlType type, const float* x,
                         const std::int32_t* ids, float* out, std::size_t m,
                         std::size_t k, std::uint32_t slots,
                         hipStream_t stream) {
  if (!IsMoeWave64Type(type, k) || slots == 0 || m == 0) {
    return false;
  }
  return DispatchType(type, k, [&]<core::GgmlType Type, std::uint32_t Lanes>() {
    constexpr std::uint32_t kRowsPerBlock = kThreads / Lanes;
    const dim3 grid(
        static_cast<std::uint32_t>((m + kRowsPerBlock - 1) / kRowsPerBlock),
        slots);
    MoeWave64GemvKernel<Type, Lanes><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w), x, ids, out,
        static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k));
  });
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
