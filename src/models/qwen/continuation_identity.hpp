#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::speculative {
struct SpeculativeOptions;
}

namespace gufo::models::qwen {

inline constexpr std::uint32_t kQwenComponentPayloadVersion = 2;

/// Draft speculator bound to a Qwen text runner. The value selects both the
/// state ABI and the disk-cache identity, so an MTP server can never reuse a
/// DFlash2 prefix and vice versa.
enum class QwenDraftKind : std::uint8_t { kNone, kDFlash, kMtp };

inline constexpr std::string_view QwenStateAbi(
    QwenDraftKind draft, bool fp16_attention_kv,
    bool bf16_recurrent_state) noexcept {
  if (draft == QwenDraftKind::kMtp) {
    if (bf16_recurrent_state) {
      return fp16_attention_kv
                 ? "qwen-gfx1151-mtp-state-v1-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-mtp-state-v1-fp32-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-mtp-state-v1-fp16-kv"
                             : "qwen-gfx1151-mtp-state-v1-fp32-kv";
  }
  if (draft == QwenDraftKind::kDFlash) {
    if (bf16_recurrent_state) {
      return fp16_attention_kv
                 ? "qwen-gfx1151-dflash-state-v4-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-dflash-state-v4-fp32-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-dflash-state-v3-fp16-kv"
                             : "qwen-gfx1151-dflash-state-v3-fp32-kv";
  }
  if (bf16_recurrent_state) {
    return fp16_attention_kv ? "qwen-gfx1151-state-v3-fp16-kv-bf16-recurrent"
                             : "qwen-gfx1151-state-v3-fp32-kv-bf16-recurrent";
  }
  return fp16_attention_kv ? "qwen-gfx1151-state-v2-fp16-kv"
                           : "qwen-gfx1151-state-v2-fp32-kv";
}

bool IsSha256Hex(std::string_view value) noexcept;

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, QwenDraftKind draft_kind,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy);

/// `mtp` names the MTP draft model instance, if any; DFlash2 uses `draft`.
std::vector<std::uint8_t> QwenProcessIdentity(const void* model,
                                              const void* draft,
                                              std::uint32_t max_context,
                                              const void* mtp = nullptr);

}  // namespace gufo::models::qwen

namespace gufo::server {
using models::qwen::kQwenComponentPayloadVersion;
using models::qwen::QwenCompatibilityIdentity;
using models::qwen::QwenDraftKind;
using models::qwen::QwenProcessIdentity;
using models::qwen::QwenStateAbi;
}  // namespace gufo::server
