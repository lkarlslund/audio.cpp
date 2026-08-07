#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/plapre/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::plapre {

struct PlapreGenerationOptions {
  int64_t max_tokens = 700;
  int top_k = 50;
  float top_p = 0.95F;
  float temperature = 0.8F;
  uint32_t seed = 42;
  bool do_sample = true;
};

struct PlapreGeneratedTokens {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> codec_tokens;
};

std::vector<float>
project_speaker_embedding(const PlapreAssets &assets,
                          const std::vector<float> &speaker_embedding);

class PlapreCausalLMRuntime {
public:
  struct Impl;

  PlapreCausalLMRuntime(std::shared_ptr<const PlapreAssets> assets,
                        core::ExecutionContext &execution,
                        size_t prefill_graph_arena_bytes,
                        size_t decode_graph_arena_bytes,
                        size_t weight_context_bytes,
                        assets::TensorStorageType weight_storage_type);
  ~PlapreCausalLMRuntime();

  PlapreGeneratedTokens generate(const std::vector<int32_t> &prompt_ids,
                                 const std::string &speaker,
                                 const PlapreGenerationOptions &options);

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::plapre
