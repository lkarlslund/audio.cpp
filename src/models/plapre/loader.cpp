#include "engine/models/plapre/loader.h"

#include "engine/framework/model_spec/package.h"
#include "engine/models/plapre/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::plapre {
namespace {

runtime::ModelMetadata metadata(const PlapreAssets & assets) {
    runtime::ModelMetadata out;
    out.family = "plapre";
    out.variant = assets.resources.model_root().filename().string();
    out.description = "Danish Plapre TTS with native Kanade and HiFT decoding.";
    return out;
}

runtime::CapabilitySet capabilities(const PlapreAssets &) {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.languages = {"da"};
    return out;
}

runtime::ModelCliInterface cli(const PlapreAssets &) {
    runtime::ModelCliInterface out;
    out.request_options = {
        {"voice_id", "tor|ida|liv|ask|kaj", "Built-in Danish voice; default tor."},
        {"max_tokens", "n", "Maximum generated 25 Hz speech tokens; default 700."},
        {"top_k", "n", "Sampling top-k; default 50."},
        {"top_p", "float", "Sampling top-p; default 0.95."},
        {"temperature", "float", "Sampling temperature; default 0.8."},
        {"seed", "n", "Sampling and vocoder noise seed."},
        {"do_sample", "true|false", "Enable sampling; default true."},
        {"text_chunk_size", "n", "Maximum Unicode codepoints per sentence before fallback splitting."},
        {"silence_duration", "seconds", "Silence inserted between text chunks; default 0.1."},
    };
    out.session_options = {
        {"plapre.weight_type", "native|f32|f16|bf16|q8_0", "Plapre language-model weight storage type."},
        {"plapre.acoustic_weight_type", "native|f32|f16|bf16|q8_0", "Kanade and HiFT weight storage type."},
        {"plapre.gpu_graphs", "true|false", "GPU graph capture; defaults off on HIP and on elsewhere."},
    };
    return out;
}

class PlapreLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "plapre";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.languages = {"da"};
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle(
                request.model_path,
                engine::model_spec::default_spec_path(family()));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_plapre_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->resources.model_root();
        inspection.metadata = metadata(*assets);
        inspection.capabilities = capabilities(*assets);
        inspection.cli = cli(*assets);
        const auto spec_path = engine::model_spec::default_spec_path(family());
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(
        const runtime::ModelLoadRequest & request) const override {
        return load_plapre_model(request.model_path);
    }
};

}  // namespace

PlapreLoadedModel::PlapreLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const PlapreAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & PlapreLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & PlapreLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> PlapreLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts ||
        (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming)) {
        throw std::runtime_error("Plapre only supports offline and streaming TTS sessions");
    }
    return std::make_unique<PlapreSession>(task, options, assets_);
}

std::unique_ptr<PlapreLoadedModel> load_plapre_model(const std::filesystem::path & model_path) {
    auto assets = load_plapre_assets(model_path);
    return std::make_unique<PlapreLoadedModel>(
        metadata(*assets), capabilities(*assets), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_plapre_loader() {
    return std::make_shared<PlapreLoader>();
}

}  // namespace engine::models::plapre
