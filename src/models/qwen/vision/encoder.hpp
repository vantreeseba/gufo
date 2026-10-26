#ifndef GUFO_MODELS_QWEN_VISION_ENCODER_HPP_
#define GUFO_MODELS_QWEN_VISION_ENCODER_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/image.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::qwen::vision {

/// Shared Qwen3.8 ViT operator graph. Each model supplies and validates its
/// own projector/output width; language-model state never lives here.
class Encoder {
public:
  class Embedding {
  public:
    ~Embedding();
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

  private:
    friend class Encoder;
    Embedding() = default;
    float* data_{nullptr};
    std::uint32_t rows_{0};
    std::uint32_t width_{0};
  };

  using Observer =
      std::function<void(std::string_view, std::span<const float>)>;
  using CancellationCheck = std::function<bool()>;
  Encoder(const std::filesystem::path& path, std::uint32_t output_width);
  /// Projector output width for a Qwen text target with a supported vision
  /// sidecar (Qwen3.8-27B or Qwen3.6-35B-A3B), or 0 when it has none.
  [[nodiscard]] static std::uint32_t TargetWidth(
      const core::GgufReader& target);
  /// An explicit sidecar is required to exist. Otherwise discover the canonical
  /// BF16 or F16 sidecar beside the target or its quantization subdirectory.
  [[nodiscard]] static std::shared_ptr<Encoder> Open(
      const std::filesystem::path& target, const std::filesystem::path& sidecar,
      std::uint32_t output_width);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Upload the weights now instead of in the first Encode. Repeated calls
  /// keep the existing upload.
  void Upload();
  /// Serialized by the owning model scheduler. Pixel grid is already resized.
  /// Uploads the weights if Upload has not; scratch is bounded by image
  /// dimensions.
  [[nodiscard]] std::shared_ptr<const Embedding> Encode(
      const core::Image& image, const Observer& observer = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen::vision
#endif
