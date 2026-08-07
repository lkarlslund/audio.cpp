#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/plapre/prompt.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::plapre {

struct PlapreConfig {
    std::string model_type;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t speaker_embedding_size = 128;
    int64_t eos_token_id = -1;
    int64_t pad_token_id = -1;
    float rms_norm_eps = 1.0e-5F;
    float rope_theta = 10000.0F;
    bool tie_word_embeddings = true;
};

struct PlapreAssets {
    assets::ResourceBundle resources;
    PlapreConfig config;
    TaskTokens task_tokens;
    std::unordered_map<std::string, std::vector<float>> speakers;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> speaker_projection_weights;
    std::shared_ptr<const assets::TensorSource> kanade_weights;
    std::shared_ptr<const assets::TensorSource> hift_weights;
};

std::shared_ptr<const PlapreAssets> load_plapre_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::plapre
