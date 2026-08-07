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
      public runtime::IOfflineVoiceTaskSession {
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

private:
    PlapreGenerationOptions generation_options(const runtime::TaskRequest & request) const;
    std::string voice_name(const runtime::TaskRequest & request) const;
    runtime::AudioBuffer synthesize_chunk(
        const std::string & text,
        const std::string & voice,
        const PlapreGenerationOptions & generation);

    runtime::TaskSpec task_;
    std::shared_ptr<const PlapreAssets> assets_;
    int64_t text_chunk_size_ = 240;
    PlapreTokenizer tokenizer_;
    std::unique_ptr<PlapreCausalLMRuntime> language_model_;
    std::unique_ptr<KanadeDecoder> kanade_;
    modules::HiftVocoderComponent hift_;
};

}  // namespace engine::models::plapre
