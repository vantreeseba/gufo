#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_MOE_EXPERIMENTS_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_MOE_EXPERIMENTS_HPP_

namespace gufo::hip::detail {

/// Temporary A/B switches for qwen35moe decode experiments, read once from
/// the environment at first use. Each one either becomes the default or is
/// removed; see https://github.com/vantreeseba/gufo/issues/1.
///
///   GUFO_EXP_MOE_Q6K_MMVQ=1   Q6_K experts on the MMQ int8 vector kernels
///   GUFO_EXP_MOE_WAVE64=1     wave64 FP32-activation expert GEMVs
///   GUFO_EXP_MOE_FUSE=<list>  comma list of: router, combine (or "all")
///
/// Captured decode graphs record whichever route was active, so the switches
/// are fixed for the life of the process.
struct MoeDecodeExperiments {
  bool q6k_mmvq{false};
  bool wave64{false};
  bool fuse_router{false};
  bool fuse_combine{false};
};

[[nodiscard]] const MoeDecodeExperiments& GetMoeDecodeExperiments();

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_MOE_EXPERIMENTS_HPP_
