#pragma once

#include "engine/models/plapre/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::tokenizers {
class LlamaBpeTokenizer;
}

namespace engine::models::plapre {

class PlapreTokenizer {
public:
    explicit PlapreTokenizer(std::shared_ptr<const PlapreAssets> assets);

    std::vector<int32_t> encode_text(const std::string & text) const;
    std::vector<int32_t> build_base_prompt(const std::string & text) const;
    const TaskTokens & task_tokens() const noexcept;

private:
    std::shared_ptr<const PlapreAssets> assets_;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer_;
};

}  // namespace engine::models::plapre
