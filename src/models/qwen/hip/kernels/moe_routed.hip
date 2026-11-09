// Routed-expert F16 WMMA GEMMs for Qwen3.6-35B-A3B; see ops/moe_routed.hpp.
// Taken from Flash-Next's kernels.hip.cpp with the shared helpers they use.
#include "src/models/qwen/hip/ops/moe_routed.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip::moe_routed {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

/// block_q6_K: ql[128] (low four bits), qh[64] (high two bits), 16 int8
/// scales (one per 16 elements), then the F16 superblock scale d.
constexpr std::size_t kQ6KBlockBytes = 210;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ6_K ? (k / 256) * kQ6KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k, const void* __restrict__ w_up) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  // Q6_K is staged like Q8_0: its 6-bit codes are assembled into one byte
  // per element at fetch time, with a scale per 16-element half.
  constexpr bool kQ6 = kType == WeightType::kQ6_K;
  constexpr bool kByteCodes = kQ8 || kQ6;
  // Paired gate/up tiles stage both projections through LDS like separate
  // tiles, gate rows in waves 0-3 and up rows in 4-7. Paired K-quants take
  // RoutedKQuantF16Kernel instead.
  constexpr bool kLdsPair = kPair && kByteCodes;
  static_assert(!kPair || kLdsPair);
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, the two byte-code blocks of Q8_0 and Q6_K
  // are four.
  constexpr int kChunks = kByteCodes ? 2 * BK : BK;

  // Codes and affine parameters are staged in LDS, with nearby rows permuted
  // across the bank groups, beside the activation plane
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  // Q5_1 rows stage their blocks as stored: a row's BK blocks are 16-byte
  // aligned, so whole 16-byte chunks load coalesced and each wave reads its
  // rows' scale, high bits and codes from LDS.
  constexpr int kQ5Bytes = BK * 24;  // one row's blocks per stage
  constexpr int kCodeBytes = kQ5 ? BM * kQ5Bytes : BM * kChunks * 16;
  constexpr int kHighBytes = kQ5K ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = kQ5 ? 0 : BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kEpilogueBytes = kLdsPair ? 8 * 16 * 18 * 4 : 8 * 1024;
  constexpr int kLdsBytes =
      kStageBytes > kEpilogueBytes ? kStageBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Thread tid fetches chunk tid % 2 of row tid / 2 (+ 128 u); paired tiles
  // fetch gate rows in the first half of the block and up rows in the second.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kLdsPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];
  // Q5_1 chunk c is part c % kQ5Parts of row c / kQ5Parts.
  constexpr int kQ5Parts = kQ5Bytes / 16;
  constexpr int kQ5Chunks = BM * kQ5Parts;
  constexpr int kQ5Fetch = kQ5 ? (kQ5Chunks + 255) / 256 : 1;
  static_assert(!kQ5 || kQ5Bytes % 16 == 0);
  const std::uint8_t* q5_src[kQ5Fetch];
  uint4 f_q5[kQ5Fetch];
  if constexpr (kQ5) {
#pragma unroll
    for (int i = 0; i < kQ5Fetch; ++i) {
      const int chunk = tid + (i * 256);
      const int r = r_block + (chunk / kQ5Parts);
      q5_src[i] = chunk < kQ5Chunks
                      ? w_expert +
                            (static_cast<std::size_t>(r < m_i ? r : m_i - 1) *
                             row_bytes) +
                            ((chunk % kQ5Parts) * 16)
                      : nullptr;
    }
  }

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        f_q5[i] = q5_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(q5_src[i] + (kb0 * 24))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      if constexpr (kQ6) {
        // K block kbl of a superblock is half n = kbl / 4, quarter
        // qd = kbl % 4: element l takes the low (qd < 2) or high nibble of
        // ql[64 n + 32 (qd & 1) + l] and bits 2 qd of qh[32 n + l], and
        // elements l < 16 / l >= 16 take scales[8 n + 2 qd] / [.. + 1].
        const int kb = kb0 + f_c;
        const int kbl = kb % 8;
        const int n = kbl / 4;
        const int qd = kbl % 4;
        const auto* blk = f_ptr[u] + ((kb / 8) * kQ6KBlockBytes);
        std::uint32_t ql[8];
        std::uint32_t qh[8];
        __builtin_memcpy(ql, blk + (64 * n) + (32 * (qd & 1)), 32);
        __builtin_memcpy(qh, blk + 128 + (32 * n), 32);
        const unsigned nib_shift = qd >= 2 ? 4U : 0U;
        const unsigned high_shift = 2U * static_cast<unsigned>(qd);
        std::uint32_t codes[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          codes[i] = ((ql[i] >> nib_shift) & 0x0F0F0F0FU) |
                     (((qh[i] >> high_shift) & 0x03030303U) << 4U);
        }
        __builtin_memcpy(&f_codes[u], &codes[0], 16);
        __builtin_memcpy(&f_codes_hi[u], &codes[4], 16);
        std::uint16_t d_bits = 0;
        __builtin_memcpy(&d_bits, blk + 208, 2);
        const std::uint32_t sc_lo = blk[192 + (8 * n) + (2 * qd)];
        const std::uint32_t sc_hi = blk[193 + (8 * n) + (2 * qd)];
        f_dm[u] = d_bits | (sc_lo << 16U) | (sc_hi << 24U);
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        if (q5_src[i] != nullptr)
          s_codes[tid + (i * 256)] = f_q5[i];
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kByteCodes) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        if constexpr (kQ8) {
          scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
        }
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ6) {
        // half2 (d * scale of elements 0-15, d * scale of 16-31); the -32
        // code offset is taken out by the magic constant.
        const float d = f_live[u]
                            ? __half2float(__builtin_bit_cast(
                                  __half, static_cast<std::uint16_t>(f_dm[u])))
                            : 0.0F;
        const auto sc_lo = static_cast<std::int8_t>((f_dm[u] >> 16U) & 0xFFU);
        const auto sc_hi = static_cast<std::int8_t>(f_dm[u] >> 24U);
        scale_bias = __builtin_bit_cast(
            std::uint32_t, __floats2half2_rn(d * static_cast<float>(sc_lo),
                                             d * static_cast<float>(sc_hi)));
      } else {
        const int sb32 = f_sb32[u];
        // The first four scale/min bytes live in y/z. Their remaining
        // low nibbles share w; the high two bits reuse those same y/z bytes.
        // Decode the fixed words directly, avoiding header-member selection.
        const uint4 h = f_header[u];
        const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
        const std::uint32_t scale_byte = h.y >> shift;
        const std::uint32_t min_byte = h.z >> shift;
        std::uint32_t sc;
        std::uint32_t mn;
        if (sb32 < 4) {
          sc = scale_byte & 0x3FU;
          mn = min_byte & 0x3FU;
        } else {
          const std::uint32_t packed = h.w >> shift;
          sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
          mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // 1024 + q as F16 minus the magic: Q8_0 carries q + 128 (1152), Q6_K an
  // unsigned code whose zero is 32 (1056), the rest are unsigned (1024).
  constexpr float kMagic = kQ8 ? -1152.0F : kQ6 ? -1056.0F : -1024.0F;
  const __half2 magic = __floats2half2_rn(kMagic, kMagic);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kByteCodes && !kQ5) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        // Q5_1 block kb of the row: d | m, the high-bit word, 16 codes.
        uint2 q5_head{};
        uint2 q5_lo{};
        uint2 q5_hi{};
        std::uint32_t scale_bias;
        if constexpr (kQ5) {
          const std::uint8_t* blk = lds + (row * kQ5Bytes) + (kb * 24);
          q5_head = *reinterpret_cast<const uint2*>(blk);
          q5_lo = *reinterpret_cast<const uint2*>(blk + 8);
          q5_hi = *reinterpret_cast<const uint2*>(blk + 16);
          scale_bias = r_block + row < m_i ? q5_head.x : 0U;
        } else {
          scale_bias = s_scale[(kb * BM) + row];
        }
        const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kByteCodes) {
          // Q8_0 / Q6_K: the block's 32 code bytes are chunks 2 kb and
          // 2 kb + 1. Q8_0 flips the sign bit to carry q + 128, which the
          // 1152 magic takes back out; Q6_K codes are already unsigned.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = kQ8 ? (words[i] ^ 0x80808080U) : words[i];
          }
        } else if constexpr (kQ5) {
          // Q5_1: elements 0-15 take the low nibbles of the 16 code
          // bytes, 16-31 the high, plus bit j of the high-bit word.
          const std::uint32_t high = q5_head.y;
          const std::uint32_t words[4] = {q5_lo.x, q5_lo.y, q5_hi.x, q5_hi.y};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {
              raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w,
              raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if constexpr (kQ6) {
            // nib[0..3] are elements 0-15, nib[4..7] elements 16-31.
            const __half2 half_scale = i < 4 ? scale2 : __high2half2(sb);
            CodesToHalves(nib[i], magic, half_scale, __float2half2_rn(0.0F),
                          h[2 * i], h[2 * i + 1]);
          } else {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i],
                          h[2 * i + 1]);
          }
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kByteCodes) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kByteCodes) && BN >= 48)) &&
                      kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  const auto run_stage = [&](int kb0) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  };
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    run_stage(kb0);
  }

  if constexpr (kLdsPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            __half values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float product = base[idx + v + 4 * plane] * base[idx + v];
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(base[idx + v]);
              asm volatile("" : "+v"(value));
              values[v] = __float2half(value);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<__half2*>(out_half + offset) =
                  __halves2half2(values[0], values[1]);
            } else {
              out_half[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_half[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Lane i receives lane i ^ 16's value (a v_permlanex16) without tying the
/// result to v: every lane is active, so the instruction's old operand is
/// never read and needs no copy of v.
__device__ __forceinline__ std::uint32_t SwapHalves(std::uint32_t v) {
  std::uint32_t old;
  asm("" : "=v"(old));
  return static_cast<std::uint32_t>(
      __builtin_amdgcn_permlanex16(static_cast<int>(old), static_cast<int>(v),
                                   0x76543210, 0xfedcba98, false, false));
}

/// Stores a 128-row routed tile's F16 results (no SwiGLU). Lane (token n,
/// rows 2i + half) of wave w holds acc[j][i] for rows 16 w + 2i + half: half
/// the tile's tokens are staged in `lds` (kRoutedStoreLdsBytes) at a time,
/// then each token's 128 rows are stored as 16-byte chunks.
constexpr int kRoutedStoreTokens = 64;
constexpr int kRoutedStoreStride = 128 + 8;
constexpr int kRoutedStoreLdsBytes =
    kRoutedStoreTokens * kRoutedStoreStride * 2;

__device__ __forceinline__ void StoreRoutedRowsF16(
    const v8f (&acc)[8], std::uint8_t* lds, int live_tok_tiles, int t_local,
    int bucket_begin, int bucket_rows, const std::int32_t* __restrict__ rows_out,
    __half* __restrict__ out, std::size_t m, int r_block) {
  constexpr int kTokens = kRoutedStoreTokens;
  constexpr int kStride = kRoutedStoreStride;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int sub_lane = tid & 15;
  const int half_id = (tid & 31) >> 4;
  const int m_i = static_cast<int>(m);
  __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int part = 0; part < 128 / kTokens; ++part) {
    if (part * (kTokens / 16) >= live_tok_tiles)
      break;
    if (part > 0)
      __syncthreads();
#pragma unroll
    for (int jj = 0; jj < kTokens / 16; ++jj) {
      const int j = part * (kTokens / 16) + jj;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        scratch[((jj * 16) + sub_lane) * kStride + (wave_id * 16) + (2 * i) +
                half_id] = __float2half(acc[j][i]);
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < kTokens * 16 / 256; ++unit) {
      const int flat = (unit * 256) + tid;
      const int tr = (part * kTokens) + (flat / 16);
      const int chunk = flat % 16;
      const int t = t_local + tr;
      const int r = r_block + (chunk * 8);
      if (tr >= live_tok_tiles * 16 || t >= bucket_rows || r >= m_i)
        continue;
      const std::int32_t dst = rows_out[bucket_begin + t];
      if (dst < 0)
        continue;
      const __half* src = scratch + ((flat / 16) * kStride) + (chunk * 8);
      __half* o = out + (static_cast<std::size_t>(dst) * m) + r;
      if (m % 8 == 0) {
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(src);
      } else {
        for (int e = 0; e < 8 && r + e < m_i; ++e)
          o[e] = src[e];
      }
    }
  }
}

// Register-resident Q4_K / Q5_K routed GEMM over token tiles of up to 128
// rows (F16 rows in and out), from Flash-Next's RoutedPairF16Kernel. Each
// wave owns one 16-row WMMA tile with its codes in registers; activations are
// the only LDS stage. A stage is one 32-byte nibble group: K32 blocks 2g (low
// nibbles) and 2g + 1 (high), whose 16-byte halves each hold one K16
// fragment of both blocks. kGated tiles are eight gate rows and their
// matching up rows (64 output rows per block, SwiGLU in the epilogue);
// otherwise sixteen rows of one projection (128 per block), as the expert
// down projection needs.
//
// A gfx1151 WMMA reads the A operand of row r from one lane only: lane r for
// even rows, lane r + 16 for odd ones. So each lane decodes the K16 fragment
// the reading lane needs first (lo in the reading lane, hi in its partner)
// and one v_permlanex16 per dword supplies the other. A superblock's codes
// and header are loaded raw one stage ahead and decoded when their stage is
// committed, so no stage waits on the loads it has just issued.
//
// Every output keeps RoutedF16GEMMKernel's values exactly: the same F16
// weights (code * (d * sc) - dmin * mn, rounded once to F16 pairs), the same
// F16 activations, the same chain of two K16 WMMAs per K32 block from zero in
// K order, the same SwiGLU rounding and one round-to-nearest F16 store, so a
// row does not depend on the tile height or on the other rows in the pass.
//
// grid (ceil(m / rows per block), tiles): `tiles[y]` is expert | (tile << 16)
// in units of `tile_rows` (a multiple of 16, at most 128).
template<WeightType kType, bool kGated>
__launch_bounds__(256) __global__
    void RoutedKQuantF16Kernel(const void* __restrict__ w,
                               const void* __restrict__ w_up,
                               const __half* __restrict__ x,
                               const std::int32_t* __restrict__ tiles,
                               std::uint32_t tile_rows,
                               const std::int32_t* __restrict__ pad_bounds,
                               const std::int32_t* __restrict__ rows_in,
                               const std::int32_t* __restrict__ rows_out,
                               __half* __restrict__ out, std::size_t m,
                               std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage: one nibble group
  constexpr int kTokTiles = BN / 16;
  constexpr int kRows = kGated ? 64 : 128;  // output rows per block
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  static_assert(kType == WeightType::kQ4_K || kQ5K);
  constexpr std::size_t kBlockBytes = kQ5K ? kQ5KBlockBytes : sizeof(Q4KBlock);
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // Gated: every token's 64 F16 results at once, two padding halves per row.
  constexpr int kPairStride = kRows + 2;
  constexpr int kEpilogueBytes =
      kGated ? BN * kPairStride * 2 : kRoutedStoreLdsBytes;
  constexpr int kLdsBytes =
      kActBytes > kEpilogueBytes ? kActBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * static_cast<int>(tile_rows);
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(static_cast<int>(tile_rows) / 16,
               (bucket_rows - t_local + 15) / 16);
  const int num_sb = static_cast<int>(k / 256);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 256) * kBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;
  // Gated tile rows 0-7 are gate rows, 8-15 the matching up rows.
  const int row = r_block + (kGated ? (wave_id * 8) + (sub_lane % 8)
                                    : (wave_id * 16) + sub_lane);
  const bool row_live = row < m_i;
  // The K16 half this lane decodes: lo where the WMMA reads A, hi in the
  // partner lane.
  const int f_c = half_id ^ (sub_lane & 1);
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(kGated && sub_lane >= 8 ? w_up : w) +
      (static_cast<std::size_t>(expert) * m +
       static_cast<std::size_t>(row_live ? row : m_i - 1)) *
          row_bytes;

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; four per thread.
  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        t < live_tok_tiles * 16 && c_row < bucket_rows
            ? rows_in[bucket_begin + c_row]
            : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // A superblock's raw header, high bits and this lane's code chunks,
  // loaded with its first stage's fetch and decoded at that stage's commit.
  uint4 raw_header;
  uint4 raw_codes[4];  ///< chunk f_c of groups 0-3
  uint4 raw_qh[2];     ///< Q5_K: the 32 high-bit bytes
  // The committed superblock: header, nibble-packed codes per group.
  uint4 header = make_uint4(0u, 0u, 0u, 0u);
  uint4 codes[4];
  uint4 qh[2];
  // Padding tokens keep whatever their slots held: their columns are never
  // stored, and WMMA columns do not mix.
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);

  const auto fetch_stage = [&]<int G>(int sb) {
    if constexpr (G == 0) {
      const std::uint8_t* blk = w_row + (sb * kBlockBytes);
      const auto* chunks = reinterpret_cast<const uint4*>(blk);
      constexpr int kCodeChunk = kQ5K ? 3 : 1;
      raw_header = chunks[0];
      if constexpr (kQ5K) {
        raw_qh[0] = chunks[1];
        raw_qh[1] = chunks[2];
      }
#pragma unroll
      for (int g = 0; g < 4; ++g)
        raw_codes[g] = chunks[kCodeChunk + (g * 2) + f_c];
    }
    const int kb0 = (sb * 8) + (G * 2);
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  // The committed stage's (scale, bias) for K32 block 2G + f_c, and Q5_K's
  // high-bit word for it.
  std::uint32_t own_scale = 0;
  std::uint32_t own_high = 0;
  const auto commit_stage = [&]<int G>() {
    if constexpr (G == 0) {
      header = raw_header;
#pragma unroll
      for (int g = 0; g < 4; ++g)
        codes[g] = raw_codes[g];
      if constexpr (kQ5K) {
        qh[0] = raw_qh[0];
        qh[1] = raw_qh[1];
      }
    }
    const int sb32 = (2 * G) + f_c;
    // The first four scale/min bytes live in y/z; the others' low nibbles
    // share w, their high two bits reuse those y/z bytes.
    const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
    const std::uint32_t scale_byte = header.y >> shift;
    const std::uint32_t min_byte = header.z >> shift;
    std::uint32_t sc;
    std::uint32_t mn;
    if (sb32 < 4) {
      sc = scale_byte & 0x3FU;
      mn = min_byte & 0x3FU;
    } else {
      const std::uint32_t packed = header.w >> shift;
      sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
      mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
    }
    const __half2 dm = __builtin_bit_cast(__half2, header.x);
    const float scale =
        row_live ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
    const float offset =
        row_live ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
    own_scale =
        __builtin_bit_cast(std::uint32_t, __floats2half2_rn(scale, -offset));
    if constexpr (kQ5K) {
      // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
      const std::uint32_t q[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                  qh[1].x, qh[1].y, qh[1].z, qh[1].w};
      std::uint32_t high = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        high |= GatherBit(q[i], sb32) << (4 * i);
      own_high = high;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  const __half2 magic = __floats2half2_rn(-1024.0F, -1024.0F);
  const auto compute_stage = [&]<int G>() {
    const uint4 c = codes[G];
    const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
    const std::uint32_t peer_scale = SwapHalves(own_scale);
    const std::uint32_t peer_high = kQ5K ? SwapHalves(own_high) : 0U;
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 sb =
          __builtin_bit_cast(__half2, f_c == kb ? own_scale : peer_scale);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      const std::uint32_t high =
          kQ5K ? ((f_c == kb ? own_high : peer_high) >> (16U * f_c)) : 0U;
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        std::uint32_t nib = (words[i] >> (4U * kb)) & 0x0F0F0F0FU;
        if constexpr (kQ5K)
          nib |= SpreadHighBits((high >> (4 * i)) & 0xFU);
        CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int G>(int sb) {
    commit_stage.template operator()<G>();
    __syncthreads();
    if constexpr (G < 3)
      fetch_stage.template operator()<G + 1>(sb);
    else if (sb + 1 < num_sb)
      fetch_stage.template operator()<0>(sb + 1);
    compute_stage.template operator()<G>();
    __syncthreads();
  };
  fetch_stage.template operator()<0>(0);
  for (int sb = 0; sb < num_sb; ++sb) {
    run_stage.template operator()<0>(sb);
    run_stage.template operator()<1>(sb);
    run_stage.template operator()<2>(sb);
    run_stage.template operator()<3>(sb);
  }

  if constexpr (kGated) {
    __half* scratch = reinterpret_cast<__half*>(lds);
    // Lane (token n, rows 2l + half) holds gate rows in acc[j][l] and the
    // matching up rows in acc[j][l + 4]. SwiGLU in registers, then all F16
    // results are staged once so stores cover the full 64-column output row.
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles)
        continue;
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        float product = acc[j][l + 4] * acc[j][l];
        asm volatile("" : "+v"(product));
        float value = product * SigmoidF(acc[j][l]);
        asm volatile("" : "+v"(value));
        scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l +
                half_id] = __float2half(value);
      }
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < BN * kRows / (2 * 256); ++unit) {
      const int flat = (unit * 256 + tid) * 2;
      const int tr = flat / kRows;
      const int t = t_local + tr;
      const int r = r_block + flat % kRows;
      if (tr < live_tok_tiles * 16 && t < bucket_rows && r < m_i) {
        const std::int32_t dst = rows_out[bucket_begin + t];
        if (dst >= 0) {
          const auto offset = static_cast<std::size_t>(dst) * m + r;
          const int src = tr * kPairStride + flat % kRows;
          if (m % 2 == 0 && r + 1 < m_i) {
            *reinterpret_cast<__half2*>(out + offset) =
                *reinterpret_cast<const __half2*>(scratch + src);
          } else {
            out[offset] = scratch[src];
            if (r + 1 < m_i)
              out[offset + 1] = scratch[src + 1];
          }
        }
      }
    }
  } else {
    StoreRoutedRowsF16(acc, lds, live_tok_tiles, t_local, bucket_begin,
                       bucket_rows, rows_out, out, m, r_block);
  }
}

// Register-resident Q6_K routed projection over token tiles of up to 128
// rows (F16 rows in and out, no SwiGLU): RoutedF16GEMMKernel's Q6_K
// arithmetic laid out like RoutedKQuantF16Kernel's ungated tiles. Each wave
// owns sixteen rows and decodes their codes in registers; activations are
// the only LDS stage. A stage is K32 blocks 2G and 2G + 1 of a superblock,
// which share half n = G / 2 of its ql and qh bytes: the low (G even) or
// high (G odd) nibbles of ql[64 n + 32 kb + l] and bits 2 (2 (G % 2) + kb)
// of qh[32 n + l] for element l of block kb. Each lane decodes the K16 half
// the WMMA reading it needs first, with that half's scale, and one
// v_permlanex16 per dword supplies the other. The raw bytes of the next
// stage load into the second register set while the current one computes.
//
// Every output keeps RoutedF16GEMMKernel's Q6_K values exactly: the same F16
// weights (6-bit code + 1024 - 1056 times the F16-rounded d * scale), the same
// F16 activations, the same chain of two K16 WMMAs per K32 block from zero in
// K order, and one round-to-nearest F16 store.
//
// grid (ceil(m / 128), tiles): `tiles[y]` is expert | (tile << 16) in units
// of `tile_rows` (a multiple of 16, at most 128).
__launch_bounds__(256) __global__
    void RoutedQ6KF16Kernel(const void* __restrict__ w,
                            const __half* __restrict__ x,
                            const std::int32_t* __restrict__ tiles,
                            std::uint32_t tile_rows,
                            const std::int32_t* __restrict__ pad_bounds,
                            const std::int32_t* __restrict__ rows_in,
                            const std::int32_t* __restrict__ rows_out,
                            __half* __restrict__ out, std::size_t m,
                            std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage
  constexpr int kTokTiles = BN / 16;
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kLdsBytes = kActBytes > kRoutedStoreLdsBytes
                                ? kActBytes
                                : kRoutedStoreLdsBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * static_cast<int>(tile_rows);
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(static_cast<int>(tile_rows) / 16,
               (bucket_rows - t_local + 15) / 16);
  const int num_sb = static_cast<int>(k / 256);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 256) * kQ6KBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * 128;
  const int row = r_block + (wave_id * 16) + sub_lane;
  const bool row_live = row < m_i;
  // The K16 half this lane decodes: lo where the WMMA reads A, hi in the
  // partner lane.
  const int f_c = half_id ^ (sub_lane & 1);
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(w) +
      (static_cast<std::size_t>(expert) * m +
       static_cast<std::size_t>(row_live ? row : m_i - 1)) *
          row_bytes;

  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        t < live_tok_tiles * 16 && c_row < bucket_rows
            ? rows_in[bucket_begin + c_row]
            : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // Raw bytes of a stage in two sets: this lane's 16 ql bytes per K32 block,
  // its 16 qh bytes, and the two blocks' scale bytes with d above them.
  uint4 w_ql[2][BK];
  uint4 w_qh[2];
  std::uint32_t w_sd[2];  ///< scale(kb 0) | scale(kb 1) << 8 | d << 16
  // Padding tokens keep whatever their slots held: their columns are never
  // stored, and WMMA columns do not mix.
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);

  const auto fetch_stage = [&]<int S, int G>(int sb) {
    // block_q6_K is 210 bytes, so these loads are only 2-byte aligned.
    const std::uint8_t* blk = w_row + (sb * kQ6KBlockBytes);
    constexpr int n = G / 2;
    constexpr int qd0 = 2 * (G % 2);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb)
      __builtin_memcpy(&w_ql[S][kb], blk + (64 * n) + (32 * kb) + (16 * f_c),
                       16);
    __builtin_memcpy(&w_qh[S], blk + 128 + (32 * n) + (16 * f_c), 16);
    std::uint16_t d_bits = 0;
    __builtin_memcpy(&d_bits, blk + 208, 2);
    const std::uint32_t sc0 = blk[192 + (8 * n) + (2 * qd0) + f_c];
    const std::uint32_t sc1 = blk[192 + (8 * n) + (2 * (qd0 + 1)) + f_c];
    w_sd[S] = sc0 | (sc1 << 8U) | (static_cast<std::uint32_t>(d_bits) << 16U);
    const int kb0 = (sb * 8) + (G * 2);
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  // 1024 + q as F16 minus 1056: the -32 code offset of Q6_K.
  const __half2 magic = __floats2half2_rn(-1056.0F, -1056.0F);
  const __half2 zero2 = __float2half2_rn(0.0F);
  const auto compute_stage = [&]<int S, int G>() {
    constexpr unsigned kNibShift = (G % 2) != 0 ? 4U : 0U;
    const float d =
        row_live ? __half2float(__builtin_bit_cast(
                       __half, static_cast<std::uint16_t>(w_sd[S] >> 16U)))
                 : 0.0F;
    const uint4 hq = w_qh[S];
    const std::uint32_t high_words[4] = {hq.x, hq.y, hq.z, hq.w};
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      constexpr unsigned kHighShift0 = 2U * (2U * (G % 2));
      const unsigned high_shift = kHighShift0 + (2U * kb);
      const auto sc = static_cast<std::int8_t>((w_sd[S] >> (8U * kb)) & 0xFFU);
      const float scale = d * static_cast<float>(sc);
      const __half2 scale2 = __floats2half2_rn(scale, scale);
      const uint4 lq = w_ql[S][kb];
      const std::uint32_t low_words[4] = {lq.x, lq.y, lq.z, lq.w};
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::uint32_t codes =
            ((low_words[i] >> kNibShift) & 0x0F0F0F0FU) |
            (((high_words[i] >> high_shift) & 0x03030303U) << 4U);
        CodesToHalves(codes, magic, scale2, zero2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int S, int G>(int sb) {
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
    __syncthreads();
    if constexpr (G < 3)
      fetch_stage.template operator()<1 - S, G + 1>(sb);
    else if (sb + 1 < num_sb)
      fetch_stage.template operator()<1 - S, 0>(sb + 1);
    compute_stage.template operator()<S, G>();
    __syncthreads();
  };
  fetch_stage.template operator()<0, 0>(0);
  for (int sb = 0; sb < num_sb; ++sb) {
    run_stage.template operator()<0, 0>(sb);
    run_stage.template operator()<1, 1>(sb);
    run_stage.template operator()<0, 2>(sb);
    run_stage.template operator()<1, 3>(sb);
  }

  StoreRoutedRowsF16(acc, lds, live_tok_tiles, t_local, bucket_begin,
                     bucket_rows, rows_out, out, m, r_block);
}
/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_token,
                                      std::int32_t* __restrict__ rows_slot,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      const auto at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}


}  // namespace

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, rows_token, rows_slot,
                     n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ6_K:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ6_K, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K ||
       type == WeightType::kQ6_K)
          ? 256
          : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  // Wide K-quant tiles with plain F16 output (the down projection's) take
  // the register-resident kernels.
  const bool k_quant = type == WeightType::kQ4_K ||
                       type == WeightType::kQ5_K || type == WeightType::kQ6_K;
  if (k_quant && (tile_rows == 64 || tile_rows == 128) &&
      out_half != nullptr && swiglu_gate == nullptr) {
    const dim3 grid(static_cast<unsigned int>((m + 127) / 128), n_tiles);
    if (type == WeightType::kQ6_K) {
      hipLaunchKernelGGL(RoutedQ6KF16Kernel, grid, dim3(kThreads), 0, stream,
                         w, x, tiles, tile_rows, pad_bounds, rows_in, rows_out,
                         out_half, m, k);
      return true;
    }
    const auto kernel = type == WeightType::kQ4_K
                            ? RoutedKQuantF16Kernel<WeightType::kQ4_K, false>
                            : RoutedKQuantF16Kernel<WeightType::kQ5_K, false>;
    hipLaunchKernelGGL(kernel, grid, dim3(kThreads), 0, stream, w, nullptr, x,
                       tiles, tile_rows, pad_bounds, rows_in, rows_out,
                       out_half, m, k);
    return true;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 128:
      return LaunchRoutedF16<128>(w, type, x, tiles, n_tiles, pad_bounds,
                                  rows_in, rows_out, swiglu_gate, out, out_half,
                                  m, k, stream);
    default:
      return false;
  }
}

template<int BN>
bool LaunchRoutedGatedF16(const void* gate, const void* up, WeightType type,
                          const __half* x, const std::int32_t* tiles,
                          std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, __half* out,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || k % 256 != 0 || n_tiles == 0 || out == nullptr) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL((RoutedKQuantF16Kernel<WeightType::kQ4_K, true>),
                         grid, dim3(kThreads), 0, stream, gate, up, x, tiles,
                         static_cast<std::uint32_t>(BN), pad_bounds, rows_in,
                         rows_out, out, m, k);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL((RoutedKQuantF16Kernel<WeightType::kQ5_K, true>),
                         grid, dim3(kThreads), 0, stream, gate, up, x, tiles,
                         static_cast<std::uint32_t>(BN), pad_bounds, rows_in,
                         rows_out, out, m, k);
      return true;
    case WeightType::kQ6_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ6_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ8_0, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    default:
      return false;
  }
}

bool RoutedGatedF16Supported(WeightType type) {
  return type == WeightType::kQ4_K || type == WeightType::kQ5_K ||
         type == WeightType::kQ6_K || type == WeightType::kQ8_0;
}

bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  // 128-token tiles decode each expert's weights once per wide bucket; in
  // prefill they also beat narrower tiles on short buckets. Callers that size
  // tiles per bucket also use 64-token tiles.
  if (tile_rows == 128) {
    return LaunchRoutedGatedF16<128>(gate, up, type, x, tiles, n_tiles,
                                     pad_bounds, rows_in, rows_out, out, m, k,
                                     stream);
  }
  return tile_rows == 64 &&
         LaunchRoutedGatedF16<64>(gate, up, type, x, tiles, n_tiles, pad_bounds,
                                  rows_in, rows_out, out, m, k, stream);
}

// Keep the separate four-tap convolution's F32 rounding order when its

}  // namespace gufo::hip::moe_routed
