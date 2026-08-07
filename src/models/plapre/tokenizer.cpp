#include "engine/models/plapre/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::plapre {
namespace {

std::string normalize_whitespace(const std::string & text) {
    std::string normalized;
    normalized.reserve(text.size());
    bool pending_space = false;
    for (const unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            pending_space = !normalized.empty();
            continue;
        }
        if (pending_space) {
            normalized.push_back(' ');
            pending_space = false;
        }
        normalized.push_back(static_cast<char>(ch));
    }
    return normalized;
}

}  // namespace

std::string normalize_target_text(const std::string & text) {
    std::string normalized = normalize_whitespace(text);
    if (normalized.empty()) {
        return normalized;
    }
    if (normalized.back() == ',') {
        normalized.pop_back();
    }
    if (normalized.empty()) {
        return normalized;
    }
    const char last = normalized.back();
    const bool has_terminal_punctuation =
        last == '.' || last == '?' || last == '!' || last == ':' ||
        (normalized.size() >= 3 && normalized.compare(normalized.size() - 3, 3, "\xE2\x80\xA6") == 0);
    if (!has_terminal_punctuation) {
        normalized.push_back('.');
    }
    return normalized;
}

PlapreTokenizer::PlapreTokenizer(std::shared_ptr<const PlapreAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Plapre tokenizer requires assets");
    }
    engine::tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_json_path = assets_->resources.require_file("tokenizer_json");
    spec.tokenizer_config_path = assets_->resources.require_file("tokenizer_config");
    spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Gpt2;
    tokenizer_ = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    for (const auto & [name, expected] : std::vector<std::pair<std::string, int32_t>>{
             {"<eos>", assets_->task_tokens.eos},
             {"<text>", assets_->task_tokens.text},
             {"<audio>", assets_->task_tokens.audio},
             {"<audio_0>", assets_->task_tokens.audio_base},
             {"</audio>", *assets_->task_tokens.audio_end}}) {
        const auto actual = tokenizer_->find_token_id(name);
        if (!actual.has_value() || *actual != expected) {
            throw std::runtime_error("Plapre tokenizer token id mismatch for " + name);
        }
    }
}

std::vector<int32_t> PlapreTokenizer::encode_text(const std::string & text) const {
    const std::string normalized = normalize_whitespace(text);
    if (normalized.empty()) {
        throw std::runtime_error("Plapre requires non-empty text input");
    }
    const auto encoded = tokenizer_->encode(normalized, false);
    if (encoded.empty()) {
        throw std::runtime_error("Plapre tokenizer produced no text tokens");
    }
    return encoded;
}

std::vector<int32_t> PlapreTokenizer::build_base_prompt(const std::string & text) const {
    return base_prompt(assets_->task_tokens, encode_text(normalize_target_text(text)));
}

const TaskTokens & PlapreTokenizer::task_tokens() const noexcept {
    return assets_->task_tokens;
}

}  // namespace engine::models::plapre
