#ifndef GUFO_CLI_SERVE_CACHE_CHECKPOINT_MODES_HPP_
#define GUFO_CLI_SERVE_CACHE_CHECKPOINT_MODES_HPP_

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gufo::server {

/// How the RAM cache treats one kind of optional checkpoint. Both kinds
/// rarely restore in append-only agent loops, but clients that regenerate,
/// retry or edit earlier messages rely on them.
enum class CacheCheckpointMode : std::uint8_t {
  /// Capture it and evict by the usual byte-pressure order (the default).
  kEvict,
  /// Capture it only when space comes from redundant copies and covered
  /// history (rank 0-1); never displace a conversation's checkpoint.
  kYield,
  /// Do not capture it.
  kOff,
};

/// Highest retention rank a capture in `mode` may evict, if bounded.
[[nodiscard]] inline std::optional<std::uint32_t> MaxVictimRank(
    CacheCheckpointMode mode) {
  if (mode == CacheCheckpointMode::kYield)
    return 1;
  return std::nullopt;
}

/// Parses "evict", "yield" or "off"; unset or empty selects evict.
[[nodiscard]] inline CacheCheckpointMode ParseCacheCheckpointMode(
    std::string_view name, const char* value) {
  const std::string_view mode = value != nullptr ? value : "";
  if (mode.empty() || mode == "evict")
    return CacheCheckpointMode::kEvict;
  if (mode == "yield")
    return CacheCheckpointMode::kYield;
  if (mode == "off")
    return CacheCheckpointMode::kOff;
  throw std::invalid_argument(std::string(name) +
                              " must be \"evict\", \"yield\" or \"off\"");
}

struct CacheCheckpointModes {
  /// GUFO_CACHE_GRID_CHECKPOINTS: intermediate prefill checkpoints every
  /// ~2048 tokens, which serve edits to earlier messages.
  CacheCheckpointMode grid{CacheCheckpointMode::kEvict};
  /// GUFO_CACHE_RETRY_CHECKPOINTS: the full-prompt copy taken after a turn's
  /// stable boundary, which serves an identical resent prompt.
  CacheCheckpointMode retry{CacheCheckpointMode::kEvict};
};

/// Read once per process. Throws std::invalid_argument on an unknown value.
[[nodiscard]] inline const CacheCheckpointModes& CacheCheckpointModesFromEnv() {
  static const CacheCheckpointModes modes{
      ParseCacheCheckpointMode("GUFO_CACHE_GRID_CHECKPOINTS",
                               std::getenv("GUFO_CACHE_GRID_CHECKPOINTS")),
      ParseCacheCheckpointMode("GUFO_CACHE_RETRY_CHECKPOINTS",
                               std::getenv("GUFO_CACHE_RETRY_CHECKPOINTS")),
  };
  return modes;
}

}  // namespace gufo::server

#endif
