#include "engine/models/plapre/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>

namespace engine::models::plapre {
namespace {

namespace json = engine::io::json;

int32_t require_added_token_id(const json::Value & tokenizer_config, const std::string & content) {
    const auto & tokens = tokenizer_config.require("added_tokens_decoder").as_object();
    for (const auto & [key, value] : tokens) {
        if (json::require_string(value, "content") == content) {
            return static_cast<int32_t>(std::stol(key));
        }
    }
    throw std::runtime_error("Plapre tokenizer_config is missing token: " + content);
}

TaskTokens parse_task_tokens(const json::Value & tokenizer_config) {
    TaskTokens tokens;
    tokens.eos = require_added_token_id(tokenizer_config, "<eos>");
    tokens.text = require_added_token_id(tokenizer_config, "<text>");
    tokens.audio = require_added_token_id(tokenizer_config, "<audio>");
    tokens.audio_base = require_added_token_id(tokenizer_config, "<audio_0>");
    tokens.duration = require_added_token_id(tokenizer_config, "<dur>");
    tokens.duration_end = require_added_token_id(tokenizer_config, "</dur>");
    tokens.duration_base = require_added_token_id(tokenizer_config, "<dur_0>");
    tokens.reference_text = require_added_token_id(tokenizer_config, "<ref_text>");
    tokens.reference_text_end = require_added_token_id(tokenizer_config, "</ref_text>");
    tokens.reference_audio = require_added_token_id(tokenizer_config, "<ref_audio>");
    tokens.reference_audio_end = require_added_token_id(tokenizer_config, "</ref_audio>");
    tokens.context = require_added_token_id(tokenizer_config, "<context>");
    tokens.context_end = require_added_token_id(tokenizer_config, "</context>");
    tokens.speaker_same = require_added_token_id(tokenizer_config, "<spk_same>");
    tokens.speaker_different = require_added_token_id(tokenizer_config, "<spk_diff>");
    tokens.context_audio = require_added_token_id(tokenizer_config, "<ctx_audio>");
    tokens.context_audio_end = require_added_token_id(tokenizer_config, "</ctx_audio>");
    tokens.edit_mask = require_added_token_id(tokenizer_config, "<edit_mask>");
    tokens.edit_fill = require_added_token_id(tokenizer_config, "<edit_fill>");
    tokens.edit_end = require_added_token_id(tokenizer_config, "<edit_end>");
    tokens.pronunciation = require_added_token_id(tokenizer_config, "<lex>");
    tokens.pronunciation_end = require_added_token_id(tokenizer_config, "</lex>");
    tokens.pronunciation_word = require_added_token_id(tokenizer_config, "<lex_word>");
    tokens.pronunciation_word_end = require_added_token_id(tokenizer_config, "</lex_word>");
    tokens.pronunciation_audio = require_added_token_id(tokenizer_config, "<lex_audio>");
    tokens.pronunciation_audio_end = require_added_token_id(tokenizer_config, "</lex_audio>");
    tokens.audio_end = require_added_token_id(tokenizer_config, "</audio>");
    tokens.validate_controls();
    if (tokens.audio_base + kAudioCodebookSize > tokens.duration) {
        throw std::runtime_error("Plapre audio token range overlaps control tokens");
    }
    return tokens;
}

PlapreConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto generation = resources.parse_json("generation_config");
    PlapreConfig config;
    config.model_type = json::require_string(root, "model_type");
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.hidden_size = json::require_i64(root, "hidden_size");
    config.intermediate_size = json::require_i64(root, "intermediate_size");
    config.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
    config.num_attention_heads = json::require_i64(root, "num_attention_heads");
    config.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
    config.head_dim = json::optional_i64(root, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = json::require_i64(root, "max_position_embeddings");
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);
    config.eos_token_id = json::optional_i64(
        generation,
        "eos_token_id",
        json::optional_i64(root, "eos_token_id", config.eos_token_id));
    config.pad_token_id = json::optional_i64(
        generation,
        "pad_token_id",
        json::optional_i64(root, "pad_token_id", config.pad_token_id));
    if (config.model_type != "llama") {
        throw std::runtime_error("Plapre expects a llama causal LM config");
    }
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("Plapre currently requires tied input and output embeddings");
    }
    if (config.hidden_size <= 0 || config.head_dim <= 0 ||
        config.num_attention_heads * config.head_dim != config.hidden_size ||
        config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Plapre attention dimensions are invalid");
    }
    return config;
}

}  // namespace

std::shared_ptr<const PlapreAssets> load_plapre_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<PlapreAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "plapre");
    assets->config = parse_config(assets->resources);
    assets->task_tokens = parse_task_tokens(assets->resources.parse_json("tokenizer_config"));
    if (assets->task_tokens.audio_base + kAudioCodebookSize > assets->config.vocab_size) {
        throw std::runtime_error("Plapre audio token range is outside the model vocabulary");
    }
    assets->model_weights = assets->resources.open_tensor_source("weights");
    assets->speaker_projection_weights = assets->resources.open_tensor_source("speaker_projection_weights");
    assets->kanade_weights = assets->resources.open_tensor_source("kanade_weights");
    assets->hift_weights = assets->resources.open_tensor_source("hift_weights");
    return assets;
}

}  // namespace engine::models::plapre
