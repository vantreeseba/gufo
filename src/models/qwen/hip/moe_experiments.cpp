#include "src/models/qwen/hip/detail/moe_experiments.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace gufo::hip::detail {
namespace {

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string_view text(value);
  return !text.empty() && text != "0" && text != "false" && text != "off";
}

MoeDecodeExperiments ReadMoeDecodeExperiments() {
  MoeDecodeExperiments experiments;
  experiments.q6k_mmvq = EnvEnabled("GUFO_EXP_MOE_Q6K_MMVQ");
  experiments.wave64 = EnvEnabled("GUFO_EXP_MOE_WAVE64");
  if (const char* fuse = std::getenv("GUFO_EXP_MOE_FUSE"); fuse != nullptr) {
    std::stringstream list(fuse);
    std::string item;
    while (std::getline(list, item, ',')) {
      if (item == "all") {
        experiments.fuse_router = true;
        experiments.fuse_combine = true;
      } else if (item == "router") {
        experiments.fuse_router = true;
      } else if (item == "combine") {
        experiments.fuse_combine = true;
      } else if (!item.empty()) {
        std::cerr << "gufo: ignoring unknown GUFO_EXP_MOE_FUSE item '" << item
                  << "'\n";
      }
    }
  }
  if (experiments.q6k_mmvq || experiments.wave64 || experiments.fuse_router ||
      experiments.fuse_combine) {
    std::cerr << "gufo: experimental MoE decode: q6k_mmvq="
              << experiments.q6k_mmvq << " wave64=" << experiments.wave64
              << " fuse_router=" << experiments.fuse_router
              << " fuse_combine=" << experiments.fuse_combine << "\n";
  }
  return experiments;
}

}  // namespace

const MoeDecodeExperiments& GetMoeDecodeExperiments() {
  static const MoeDecodeExperiments experiments = ReadMoeDecodeExperiments();
  return experiments;
}

}  // namespace gufo::hip::detail
