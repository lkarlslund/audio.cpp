#include "engine/models/plapre/causal_lm.h"

#include "engine/framework/runtime/speech_llama.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::models::plapre {
namespace {

std::shared_ptr<const runtime::SpeechLlamaAssets>
adapt_assets(const std::shared_ptr<const PlapreAssets> &source) {
  if (source == nullptr) {
    throw std::runtime_error("Plapre lm requires assets");
  }
  auto out = std::make_shared<runtime::SpeechLlamaAssets>();
  const auto &config = source->config;
  out->config.name = "Plapre";
  out->config.vocab_size = config.vocab_size;
  out->config.hidden_size = config.hidden_size;
  out->config.intermediate_size = config.intermediate_size;
  out->config.num_hidden_layers = config.num_hidden_layers;
  out->config.num_attention_heads = config.num_attention_heads;
  out->config.num_key_value_heads = config.num_key_value_heads;
  out->config.head_dim = config.head_dim;
  out->config.max_position_embeddings = config.max_position_embeddings;
  out->config.eos_token_id = source->task_tokens.eos;
  out->config.pad_token_id = source->task_tokens.eos;
  out->config.speech_token_start_id = source->task_tokens.audio_base;
  out->config.speech_token_count = kAudioCodebookSize;
  out->config.stop_token_ids = source->task_tokens.stop_ids();
  out->config.rms_norm_eps = config.rms_norm_eps;
  out->config.rope_theta = config.rope_theta;
  out->config.qk_norm = false;
  out->model_weights = source->model_weights;
  return out;
}

runtime::SpeechLlamaGenerationOptions
adapt_options(const PlapreGenerationOptions &source) {
  runtime::SpeechLlamaGenerationOptions out;
  out.max_tokens = source.max_tokens;
  out.top_k = source.top_k;
  out.top_p = source.top_p;
  out.temperature = source.temperature;
  out.seed = source.seed;
  out.do_sample = source.do_sample;
  return out;
}

} // namespace

std::vector<float>
project_speaker_embedding(const PlapreAssets &assets,
                          const std::vector<float> &speaker_embedding) {
  const auto input_size = assets.config.speaker_embedding_size;
  const auto output_size = assets.config.hidden_size;
  if (static_cast<int64_t>(speaker_embedding.size()) != input_size) {
    throw std::runtime_error("Plapre speaker embedding size mismatch");
  }
  const auto weights = assets.speaker_projection_weights->require_f32(
      "weight", {output_size, input_size});
  const auto bias =
      assets.speaker_projection_weights->require_f32("bias", {output_size});
  std::vector<float> projected(static_cast<size_t>(output_size));
  for (int64_t output = 0; output < output_size; ++output) {
    float value = bias[static_cast<size_t>(output)];
    const size_t row = static_cast<size_t>(output * input_size);
    for (int64_t input = 0; input < input_size; ++input) {
      value += weights[row + static_cast<size_t>(input)] *
               speaker_embedding[static_cast<size_t>(input)];
    }
    projected[static_cast<size_t>(output)] = value;
  }
  return projected;
}

struct PlapreCausalLMRuntime::Impl {
  Impl(std::shared_ptr<const PlapreAssets> source_assets,
       core::ExecutionContext &execution, size_t prefill_graph_arena_bytes,
       size_t decode_graph_arena_bytes, size_t weight_context_bytes,
       assets::TensorStorageType storage_type)
      : assets(std::move(source_assets)),
        runtime(adapt_assets(assets), execution, prefill_graph_arena_bytes,
                decode_graph_arena_bytes, weight_context_bytes, storage_type) {
    for (const auto &[name, embedding] : assets->speakers) {
      projected_speakers.emplace(
          name, project_speaker_embedding(*assets, embedding));
    }
  }

  std::shared_ptr<const PlapreAssets> assets;
  runtime::SpeechLlamaRuntime runtime;
  std::unordered_map<std::string, std::vector<float>> projected_speakers;
};

PlapreCausalLMRuntime::PlapreCausalLMRuntime(
    std::shared_ptr<const PlapreAssets> assets,
    core::ExecutionContext &execution, size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes, size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution,
                                   prefill_graph_arena_bytes,
                                   decode_graph_arena_bytes,
                                   weight_context_bytes, weight_storage_type)) {
}

PlapreCausalLMRuntime::~PlapreCausalLMRuntime() = default;

PlapreGeneratedTokens
PlapreCausalLMRuntime::generate(const std::vector<int32_t> &prompt_ids,
                                const std::string &speaker,
                                const PlapreGenerationOptions &options) {
  const auto voice = impl_->projected_speakers.find(speaker);
  if (voice == impl_->projected_speakers.end()) {
    throw std::runtime_error("unknown Plapre speaker: " + speaker);
  }
  runtime::SpeechLlamaPrompt prompt;
  prompt.input_ids = prompt_ids;
  prompt.prefix_embeddings = voice->second;
  const auto generated =
      impl_->runtime.generate(prompt, adapt_options(options));
  PlapreGeneratedTokens out;
  out.token_ids = generated.token_ids;
  out.codec_tokens = generated.codec_tokens;
  return out;
}

} // namespace engine::models::plapre
