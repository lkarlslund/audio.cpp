#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::runtime {

struct SpeechLlamaConfig {
  std::string name;
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t max_position_embeddings = 0;
  int64_t eos_token_id = 0;
  int64_t pad_token_id = 0;
  int64_t speech_token_start_id = 0;
  int64_t speech_token_count = 0;
  float rms_norm_eps = 1.0e-6F;
  float rope_theta = 10000.0F;
  bool qk_norm = false;
};

struct SpeechLlamaAssets {
  SpeechLlamaConfig config;
  std::shared_ptr<const assets::TensorSource> model_weights;
};

struct SpeechLlamaGenerationOptions {
  int64_t max_tokens = 700;
  int top_k = 50;
  float top_p = 1.0F;
  float temperature = 0.8F;
  float repetition_penalty = 1.0F;
  float presence_penalty = 0.0F;
  float frequency_penalty = 0.0F;
  uint32_t seed = 0;
  bool do_sample = true;
};

struct SpeechLlamaPrompt {
  std::vector<int32_t> input_ids;
  std::vector<float> prefix_embeddings;
};

struct SpeechLlamaGeneratedTokens {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> codec_tokens;
};

class SpeechLlamaRuntime {
public:
  struct Impl;

  SpeechLlamaRuntime(std::shared_ptr<const SpeechLlamaAssets> assets,
                     core::ExecutionContext &execution,
                     size_t prefill_graph_arena_bytes,
                     size_t decode_graph_arena_bytes,
                     size_t weight_context_bytes,
                     assets::TensorStorageType weight_storage_type);
  ~SpeechLlamaRuntime();

  SpeechLlamaGeneratedTokens
  generate(const SpeechLlamaPrompt &prompt,
           const SpeechLlamaGenerationOptions &options);

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::runtime
