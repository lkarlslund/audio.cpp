#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/plapre/assets.h"
#include "engine/models/plapre/causal_lm.h"
#include "engine/models/plapre/kanade.h"
#include "engine/models/plapre/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::plapre {

class PlapreSession final
    : public runtime::RuntimeSessionBase,
      public runtime::IOfflineVoiceTaskSession,
      public runtime::IStreamingVoiceTaskSession {
public:
    PlapreSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const PlapreAssets> assets);
    ~PlapreSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    std::optional<runtime::StreamEvent> next_stream_event() override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    runtime::TaskResult finish_stream() override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    void validate_request(const runtime::TaskRequest & request) const;
    std::vector<runtime::TaskRequest> chunk_requests(const runtime::TaskRequest & request) const;
    PlapreGenerationOptions generation_options(const runtime::TaskRequest & request) const;
    std::string voice_name(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize_chunk(const runtime::TaskRequest & request, size_t chunk_index);
    void append_inter_chunk_silence(runtime::AudioBuffer & audio, const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const PlapreAssets> assets_;
    int64_t text_chunk_size_ = 240;
    PlapreTokenizer tokenizer_;
    std::unique_ptr<PlapreCausalLMRuntime> language_model_;
    std::unique_ptr<KanadeDecoder> kanade_;
    modules::HiftVocoderComponent hift_;
    std::vector<runtime::TaskRequest> stream_chunk_requests_;
    runtime::AudioBuffer stream_merged_audio_;
    size_t stream_chunk_index_ = 0;
    bool stream_started_ = false;
};

}  // namespace engine::models::plapre
