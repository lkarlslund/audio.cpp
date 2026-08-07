#include "engine/models/miotts/causal_lm.h"

#include "engine/framework/runtime/speech_llama.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miotts {
namespace {

std::shared_ptr<const runtime::SpeechLlamaAssets>
adapt_assets(const std::shared_ptr<const MioTTSAssets> &source) {
  if (source == nullptr) {
    throw std::runtime_error("MioTTS lm requires assets");
  }
  auto out = std::make_shared<runtime::SpeechLlamaAssets>();
  const auto &config = source->config;
  out->config.name = "MioTTS";
  out->config.vocab_size = config.vocab_size;
  out->config.hidden_size = config.hidden_size;
  out->config.intermediate_size = config.intermediate_size;
  out->config.num_hidden_layers = config.num_hidden_layers;
  out->config.num_attention_heads = config.num_attention_heads;
  out->config.num_key_value_heads = config.num_key_value_heads;
  out->config.head_dim = config.head_dim;
  out->config.max_position_embeddings = config.max_position_embeddings;
  out->config.eos_token_id = config.eos_token_id;
  out->config.pad_token_id = config.pad_token_id;
  out->config.speech_token_start_id = config.speech_token_start_id;
  out->config.speech_token_count = config.speech_token_count;
  out->config.rms_norm_eps = config.rms_norm_eps;
  out->config.rope_theta = config.rope_theta;
  out->config.qk_norm = true;
  out->model_weights = source->model_weights;
  return out;
}

runtime::SpeechLlamaGenerationOptions
adapt_options(const MioTTSGenerationOptions &source) {
  runtime::SpeechLlamaGenerationOptions out;
  out.max_tokens = source.max_tokens;
  out.top_k = source.top_k;
  out.top_p = source.top_p;
  out.temperature = source.temperature;
  out.repetition_penalty = source.repetition_penalty;
  out.presence_penalty = source.presence_penalty;
  out.frequency_penalty = source.frequency_penalty;
  out.seed = source.seed;
  out.do_sample = source.do_sample;
  return out;
}

} // namespace

struct MioTTSCausalLMRuntime::Impl {
  Impl(std::shared_ptr<const MioTTSAssets> assets,
       core::ExecutionContext &execution, size_t prefill_graph_arena_bytes,
       size_t decode_graph_arena_bytes, size_t weight_context_bytes,
       assets::TensorStorageType storage_type)
      : runtime(adapt_assets(assets), execution, prefill_graph_arena_bytes,
                decode_graph_arena_bytes, weight_context_bytes, storage_type) {}

  runtime::SpeechLlamaRuntime runtime;
};

MioTTSCausalLMRuntime::MioTTSCausalLMRuntime(
    std::shared_ptr<const MioTTSAssets> assets,
    core::ExecutionContext &execution, size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes, size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution,
                                   prefill_graph_arena_bytes,
                                   decode_graph_arena_bytes,
                                   weight_context_bytes, weight_storage_type)) {
}

MioTTSCausalLMRuntime::~MioTTSCausalLMRuntime() = default;

MioTTSGeneratedTokens
MioTTSCausalLMRuntime::generate(const MioTTSPrompt &prompt,
                                const MioTTSGenerationOptions &options) {
  runtime::SpeechLlamaPrompt adapted_prompt;
  adapted_prompt.input_ids = prompt.input_ids;
  const auto generated =
      impl_->runtime.generate(adapted_prompt, adapt_options(options));
  MioTTSGeneratedTokens out;
  out.token_ids = generated.token_ids;
  out.codec_tokens = generated.codec_tokens;
  return out;
}

} // namespace engine::models::miotts
