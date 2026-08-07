#include "engine/models/plapre/prompt.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace engine::models::plapre {
namespace {

void require_token(int32_t token, const char * name) {
    if (token < 0) {
        throw std::runtime_error(std::string("Plapre tokenizer is missing ") + name);
    }
}

void require_codec_token(int32_t token) {
    if (token < 0 || token >= kAudioCodebookSize) {
        throw std::runtime_error("Plapre codec token is outside the 12800-entry Kanade codebook");
    }
}

void append(std::vector<int32_t> & target, const std::vector<int32_t> & values) {
    target.insert(target.end(), values.begin(), values.end());
}

void append_duration(
    std::vector<int32_t> & prompt,
    const TaskTokens & tokens,
    const std::vector<int32_t> & duration_frames) {
    if (duration_frames.empty()) {
        return;
    }
    prompt.push_back(tokens.duration);
    append(prompt, map_duration_frames(tokens, duration_frames));
    prompt.push_back(tokens.duration_end);
}

}  // namespace

void TaskTokens::validate_base() const {
    require_token(eos, "<eos>");
    require_token(text, "<text>");
    require_token(audio, "<audio>");
    require_token(audio_base, "<audio_0>");
}

void TaskTokens::validate_controls() const {
    validate_base();
    require_token(duration, "<dur>");
    require_token(duration_end, "</dur>");
    require_token(duration_base, "<dur_0>");
    require_token(reference_text, "<ref_text>");
    require_token(reference_text_end, "</ref_text>");
    require_token(reference_audio, "<ref_audio>");
    require_token(reference_audio_end, "</ref_audio>");
    require_token(context, "<context>");
    require_token(context_end, "</context>");
    require_token(speaker_same, "<spk_same>");
    require_token(speaker_different, "<spk_diff>");
    require_token(context_audio, "<ctx_audio>");
    require_token(context_audio_end, "</ctx_audio>");
    require_token(edit_mask, "<edit_mask>");
    require_token(edit_fill, "<edit_fill>");
    require_token(edit_end, "<edit_end>");
    require_token(pronunciation, "<lex>");
    require_token(pronunciation_end, "</lex>");
    require_token(pronunciation_word, "<lex_word>");
    require_token(pronunciation_word_end, "</lex_word>");
    require_token(pronunciation_audio, "<lex_audio>");
    require_token(pronunciation_audio_end, "</lex_audio>");
}

std::vector<int32_t> TaskTokens::stop_ids() const {
    validate_base();
    std::vector<int32_t> stops{eos};
    if (audio_end.has_value()) {
        require_token(*audio_end, "</audio>");
        if (*audio_end != eos) {
            stops.push_back(*audio_end);
        }
    }
    return stops;
}

std::vector<int32_t> map_audio_tokens(
    const TaskTokens & tokens,
    const std::vector<int32_t> & codec_tokens) {
    tokens.validate_base();
    std::vector<int32_t> mapped;
    mapped.reserve(codec_tokens.size());
    for (const int32_t token : codec_tokens) {
        require_codec_token(token);
        mapped.push_back(tokens.audio_base + token);
    }
    return mapped;
}

std::vector<int32_t> extract_audio_tokens(
    const TaskTokens & tokens,
    const std::vector<int32_t> & generated_tokens) {
    tokens.validate_base();
    std::vector<int32_t> codec_tokens;
    codec_tokens.reserve(generated_tokens.size());
    const int32_t end = tokens.audio_base + kAudioCodebookSize;
    for (const int32_t token : generated_tokens) {
        if (token >= tokens.audio_base && token < end) {
            codec_tokens.push_back(token - tokens.audio_base);
        }
    }
    return codec_tokens;
}

std::vector<int32_t> map_duration_frames(
    const TaskTokens & tokens,
    const std::vector<int32_t> & frames) {
    tokens.validate_controls();
    std::vector<int32_t> mapped;
    mapped.reserve(frames.size());
    for (const int32_t frame : frames) {
        mapped.push_back(tokens.duration_base + std::clamp(frame, int32_t{0}, kDurationTokenCount - 1));
    }
    return mapped;
}

std::vector<int32_t> pronunciation_prefix(
    const TaskTokens & tokens,
    const std::vector<PronunciationReference> & references) {
    tokens.validate_controls();
    std::vector<int32_t> prompt{tokens.pronunciation};
    size_t total_audio_tokens = 0;
    const size_t reference_count = std::min(references.size(), kPronunciationReferenceLimit);
    for (size_t index = 0; index < reference_count; ++index) {
        const auto & reference = references[index];
        const size_t remaining = kPronunciationTotalAudioTokenLimit - total_audio_tokens;
        const size_t count = std::min({reference.audio_tokens.size(), kPronunciationAudioTokenLimit, remaining});
        if (count == 0) {
            break;
        }
        prompt.push_back(tokens.pronunciation_word);
        append(prompt, reference.word_tokens);
        prompt.push_back(tokens.pronunciation_word_end);
        prompt.push_back(tokens.pronunciation_audio);
        const std::vector<int32_t> selected(reference.audio_tokens.begin(), reference.audio_tokens.begin() + count);
        append(prompt, map_audio_tokens(tokens, selected));
        prompt.push_back(tokens.pronunciation_audio_end);
        total_audio_tokens += count;
    }
    prompt.push_back(tokens.pronunciation_end);
    return prompt;
}

std::vector<int32_t> base_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & duration_frames,
    const std::vector<int32_t> & pronunciation) {
    tokens.validate_base();
    std::vector<int32_t> prompt = pronunciation;
    prompt.push_back(tokens.text);
    append(prompt, text_tokens);
    append_duration(prompt, tokens, duration_frames);
    prompt.push_back(tokens.audio);
    return prompt;
}

std::vector<int32_t> clone_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & reference_text_tokens,
    const std::vector<int32_t> & reference_audio_tokens,
    const std::vector<int32_t> & duration_frames,
    const std::vector<int32_t> & pronunciation) {
    tokens.validate_controls();
    std::vector<int32_t> prompt = pronunciation;
    prompt.push_back(tokens.reference_text);
    append(prompt, reference_text_tokens);
    prompt.push_back(tokens.reference_text_end);
    prompt.push_back(tokens.reference_audio);
    const size_t count = std::min(reference_audio_tokens.size(), kReferenceAudioTokenLimit);
    const std::vector<int32_t> selected(reference_audio_tokens.begin(), reference_audio_tokens.begin() + count);
    append(prompt, map_audio_tokens(tokens, selected));
    prompt.push_back(tokens.reference_audio_end);
    prompt.push_back(tokens.text);
    append(prompt, text_tokens);
    append_duration(prompt, tokens, duration_frames);
    prompt.push_back(tokens.audio);
    return prompt;
}

std::vector<int32_t> context_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & previous_text_tokens,
    bool same_speaker,
    const std::optional<std::vector<int32_t>> & previous_audio_tokens,
    const std::vector<int32_t> & duration_frames,
    const std::vector<int32_t> & pronunciation) {
    tokens.validate_controls();
    std::vector<int32_t> prompt = pronunciation;
    prompt.push_back(tokens.context);
    prompt.push_back(same_speaker ? tokens.speaker_same : tokens.speaker_different);
    append(prompt, previous_text_tokens);
    if (previous_audio_tokens.has_value()) {
        prompt.push_back(tokens.context_audio);
        const size_t count = std::min(previous_audio_tokens->size(), kContextAudioTokenLimit);
        const std::vector<int32_t> selected(previous_audio_tokens->begin(), previous_audio_tokens->begin() + count);
        append(prompt, map_audio_tokens(tokens, selected));
        prompt.push_back(tokens.context_audio_end);
    }
    prompt.push_back(tokens.context_end);
    prompt.push_back(tokens.text);
    append(prompt, text_tokens);
    append_duration(prompt, tokens, duration_frames);
    prompt.push_back(tokens.audio);
    return prompt;
}

std::vector<int32_t> edit_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & new_text_tokens,
    const std::vector<int32_t> & original_audio_tokens,
    size_t mask_start,
    size_t mask_end,
    const std::vector<int32_t> & pronunciation) {
    tokens.validate_controls();
    if (mask_start > mask_end || mask_end > original_audio_tokens.size()) {
        throw std::runtime_error("Plapre edit mask is outside the original audio token sequence");
    }
    const auto mapped = map_audio_tokens(tokens, original_audio_tokens);
    std::vector<int32_t> prompt = pronunciation;
    prompt.push_back(tokens.text);
    append(prompt, new_text_tokens);
    prompt.push_back(tokens.audio);
    prompt.insert(prompt.end(), mapped.begin(), mapped.begin() + static_cast<std::ptrdiff_t>(mask_start));
    prompt.push_back(tokens.edit_mask);
    prompt.insert(prompt.end(), mapped.begin() + static_cast<std::ptrdiff_t>(mask_end), mapped.end());
    prompt.push_back(tokens.edit_fill);
    prompt.push_back(tokens.edit_mask);
    return prompt;
}

}  // namespace engine::models::plapre
