#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/plapre/assets.h"

#include <memory>

namespace engine::models::plapre {

class PlapreLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    PlapreLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const PlapreAssets> assets);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const PlapreAssets> assets_;
};

std::unique_ptr<PlapreLoadedModel> load_plapre_model(const std::filesystem::path & model_path);
std::shared_ptr<runtime::IVoiceModelLoader> make_plapre_loader();

}  // namespace engine::models::plapre
