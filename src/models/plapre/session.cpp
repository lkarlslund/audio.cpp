#include "engine/models/plapre/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::plapre {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultLmWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultLmPrefillGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultLmDecodeGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultKanadeWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultKanadeGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultKanadeConstantContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kDefaultTextChunkCodepoints = 240;
constexpr float kDefaultSilenceDurationSeconds = 0.1F;

std::shared_ptr<const PlapreAssets> require_assets(std::shared_ptr<const PlapreAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Plapre session requires assets");
    }
    return assets;
}

assets::TensorStorageType weight_type(
    const runtime::SessionOptions & options,
    const char * key) {
    return runtime::parse_tensor_storage_option(
        options.options,
        key,
        assets::TensorStorageType::Native,
        {
            assets::TensorStorageType::Native,
            assets::TensorStorageType::F32,
            assets::TensorStorageType::F16,
            assets::TensorStorageType::BF16,
            assets::TensorStorageType::Q8_0,
        });
}

modules::HiftVocoderConfig hift_config(assets::TensorStorageType storage_type) {
    modules::HiftVocoderConfig config;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    config.weight_storage_type = storage_type;
    config.tensor_prefix = "";
    config.weight_layout = modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    return config;
}

void validate_request_options(const PlapreGenerationOptions & options) {
    if (options.max_tokens <= 0) {
        throw std::runtime_error("Plapre max_tokens must be positive");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("Plapre top_k must be non-negative");
    }
    if (options.top_p < 0.0F || options.top_p > 1.0F) {
        throw std::runtime_error("Plapre top_p must be in [0, 1]");
    }
    if (options.temperature < 0.0F || options.temperature > 2.0F) {
        throw std::runtime_error("Plapre temperature must be in [0, 2]");
    }
}

}  // namespace

PlapreSession::PlapreSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const PlapreAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Plapre only supports the Tts task");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Plapre only supports offline and streaming sessions");
    }
    text_chunk_size_ = engine::text::parse_text_chunk_size_override(options.options)
        .value_or(kDefaultTextChunkCodepoints);
    const auto lm_weight_type = weight_type(options, "plapre.weight_type");
    const auto acoustic_weight_type = weight_type(options, "plapre.acoustic_weight_type");
    language_model_ = std::make_unique<PlapreCausalLMRuntime>(
        assets_,
        execution_context(),
        runtime::parse_size_mb_option(
            options.options, {"plapre.prefill_graph_arena_mb"}, kDefaultLmPrefillGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options, {"plapre.decode_graph_arena_mb"}, kDefaultLmDecodeGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options, {"plapre.weight_context_mb"}, kDefaultLmWeightContextBytes),
        lm_weight_type);
    assets_->model_weights->release_storage();
    assets_->speaker_projection_weights->release_storage();

    kanade_ = std::make_unique<KanadeDecoder>(
        assets_->kanade_weights,
        execution_context(),
        runtime::parse_size_mb_option(
            options.options, {"plapre.kanade_weight_context_mb"}, kDefaultKanadeWeightContextBytes),
        runtime::parse_size_mb_option(
            options.options, {"plapre.kanade_graph_arena_mb"}, kDefaultKanadeGraphArenaBytes),
        runtime::parse_size_mb_option(
            options.options, {"plapre.kanade_constant_context_mb"}, kDefaultKanadeConstantContextBytes),
        acoustic_weight_type);
    assets_->kanade_weights->release_storage();

    hift_ = modules::HiftVocoderComponent::load_from_tensor_source(
        assets_->hift_weights,
        options.backend,
        hift_config(acoustic_weight_type));
    assets_->hift_weights->release_storage();
}

PlapreSession::~PlapreSession() = default;

std::string PlapreSession::family() const {
    return "plapre";
}

runtime::VoiceTaskKind PlapreSession::task_kind() const {
    return task_.task;
}

runtime::RunMode PlapreSession::run_mode() const {
    return task_.mode;
}

void PlapreSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

void PlapreSession::validate_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Plapre run requires text_input");
    }
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Plapre TTS does not accept audio_input");
    }
}

std::vector<runtime::TaskRequest> PlapreSession::chunk_requests(const runtime::TaskRequest & request) const {
    validate_request(request);
    const auto chunk_size = engine::text::parse_text_chunk_size_override(request.options)
        .value_or(text_chunk_size_);
    return runtime::chunk_text_request(request, chunk_size);
}

PlapreGenerationOptions PlapreSession::generation_options(const runtime::TaskRequest & request) const {
    PlapreGenerationOptions options;
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        options.max_tokens = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    options.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    validate_request_options(options);
    return options;
}

std::string PlapreSession::voice_name(const runtime::TaskRequest & request) const {
    std::string voice = runtime::find_option(request.options, {"voice_id", "speaker"})
        .value_or("tor");
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->cached_voice_id.has_value()) {
        voice = *request.voice->speaker->cached_voice_id;
    }
    if (assets_->speakers.find(voice) == assets_->speakers.end()) {
        throw std::runtime_error("unknown Plapre voice: " + voice);
    }
    return voice;
}

runtime::AudioBuffer PlapreSession::synthesize_chunk(
    const runtime::TaskRequest & request,
    size_t chunk_index) {
    const auto voice = voice_name(request);
    auto generation = generation_options(request);
    generation.seed += static_cast<uint32_t>(chunk_index);
    auto stage_start = Clock::now();
    const auto prompt = tokenizer_.build_base_prompt(request.text_input->text);
    engine::debug::timing_log_scalar("plapre.prompt_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = Clock::now();
    const auto generated = language_model_->generate(
        prompt, voice, generation);
    engine::debug::timing_log_scalar("plapre.lm_ms", engine::debug::elapsed_ms(stage_start));
    if (generated.codec_tokens.empty()) {
        throw std::runtime_error("Plapre generated no speech tokens");
    }
    engine::debug::trace_log_scalar(
        "plapre.generated_tokens", static_cast<int64_t>(generated.codec_tokens.size()));
    stage_start = Clock::now();
    const auto mel = kanade_->decode(generated.codec_tokens, assets_->speakers.at(voice));
    engine::debug::timing_log_scalar("plapre.kanade_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = Clock::now();
    const auto waveform = hift_.synthesize(mel.values, mel.frames, generation.seed);
    engine::debug::timing_log_scalar("plapre.hift_ms", engine::debug::elapsed_ms(stage_start));
    return runtime::AudioBuffer{
        static_cast<int>(waveform.sample_rate),
        1,
        waveform.waveform,
    };
}

void PlapreSession::append_inter_chunk_silence(
    runtime::AudioBuffer & audio,
    const runtime::TaskRequest & request) const {
    const float seconds = runtime::parse_finite_float_option(request.options, {"silence_duration"})
        .value_or(kDefaultSilenceDurationSeconds);
    if (seconds < 0.0F || seconds > 10.0F) {
        throw std::runtime_error("Plapre silence_duration must be in [0, 10]");
    }
    const auto samples = static_cast<size_t>(std::llround(seconds * static_cast<float>(audio.sample_rate)));
    audio.samples.insert(audio.samples.end(), samples, 0.0F);
}

runtime::TaskResult PlapreSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("Plapre run");
    const auto voice = voice_name(request);
    const auto generation = generation_options(request);
    const auto chunks = chunk_requests(request);
    engine::debug::trace_log_scalar("plapre.voice", voice);
    engine::debug::trace_log_scalar("plapre.text_chunk_count", static_cast<int64_t>(chunks.size()));
    engine::debug::trace_log_scalar("plapre.sampling.max_tokens", generation.max_tokens);
    engine::debug::trace_log_scalar("plapre.sampling.seed", generation.seed);

    runtime::AudioBuffer merged;
    for (size_t index = 0; index < chunks.size(); ++index) {
        auto audio = synthesize_chunk(chunks[index], index);
        if (index + 1 < chunks.size()) {
            append_inter_chunk_silence(audio, request);
        }
        runtime::append_audio_buffer(merged, audio);
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamingPolicy PlapreSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void PlapreSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Plapre streaming");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Plapre start_stream requires a streaming session");
    }
    reset();
    stream_chunk_requests_ = chunk_requests(request);
    stream_started_ = true;
}

std::optional<runtime::StreamEvent> PlapreSession::next_stream_event() {
    if (!stream_started_) {
        throw std::runtime_error("Plapre streaming has not been started");
    }
    if (stream_chunk_index_ >= stream_chunk_requests_.size()) {
        return std::nullopt;
    }
    const size_t index = stream_chunk_index_++;
    auto audio = synthesize_chunk(stream_chunk_requests_[index], index);
    if (index + 1 < stream_chunk_requests_.size()) {
        append_inter_chunk_silence(audio, stream_chunk_requests_[index]);
    }
    runtime::append_audio_buffer(stream_merged_audio_, audio);
    runtime::StreamEvent event;
    event.named_audio_outputs.push_back({
        "chunk_" + std::to_string(index),
        std::move(audio),
        {},
    });
    return event;
}

void PlapreSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    (void)sink;
}

runtime::TaskResult PlapreSession::finish_stream() {
    if (!stream_started_) {
        throw std::runtime_error("Plapre streaming has not been started");
    }
    runtime::TaskResult result;
    result.audio_output = std::move(stream_merged_audio_);
    reset();
    return result;
}

void PlapreSession::reset() {
    stream_chunk_requests_.clear();
    stream_merged_audio_ = runtime::AudioBuffer{};
    stream_chunk_index_ = 0;
    stream_started_ = false;
}

runtime::StreamEvent PlapreSession::process_audio_chunk(const runtime::AudioChunk &) {
    throw std::runtime_error("Plapre streaming does not consume audio chunks");
}

runtime::TaskResult PlapreSession::finalize() {
    return finish_stream();
}

}  // namespace engine::models::plapre
