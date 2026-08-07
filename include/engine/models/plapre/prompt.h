#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace engine::models::plapre {

constexpr int32_t kAudioCodebookSize = 12800;
constexpr int32_t kDurationTokenCount = 400;
constexpr size_t kReferenceAudioTokenLimit = 250;
constexpr size_t kContextAudioTokenLimit = 150;
constexpr size_t kPronunciationReferenceLimit = 10;
constexpr size_t kPronunciationAudioTokenLimit = 60;
constexpr size_t kPronunciationTotalAudioTokenLimit = 200;

struct TaskTokens {
    int32_t eos = -1;
    int32_t text = -1;
    int32_t audio = -1;
    int32_t audio_base = -1;
    int32_t duration = -1;
    int32_t duration_end = -1;
    int32_t duration_base = -1;
    int32_t reference_text = -1;
    int32_t reference_text_end = -1;
    int32_t reference_audio = -1;
    int32_t reference_audio_end = -1;
    int32_t context = -1;
    int32_t context_end = -1;
    int32_t speaker_same = -1;
    int32_t speaker_different = -1;
    int32_t context_audio = -1;
    int32_t context_audio_end = -1;
    int32_t edit_mask = -1;
    int32_t edit_fill = -1;
    int32_t edit_end = -1;
    int32_t pronunciation = -1;
    int32_t pronunciation_end = -1;
    int32_t pronunciation_word = -1;
    int32_t pronunciation_word_end = -1;
    int32_t pronunciation_audio = -1;
    int32_t pronunciation_audio_end = -1;
    std::optional<int32_t> audio_end;

    void validate_base() const;
    void validate_controls() const;
    std::vector<int32_t> stop_ids() const;
};

struct PronunciationReference {
    std::vector<int32_t> word_tokens;
    std::vector<int32_t> audio_tokens;
};

std::vector<int32_t> map_audio_tokens(
    const TaskTokens & tokens,
    const std::vector<int32_t> & codec_tokens);

std::vector<int32_t> extract_audio_tokens(
    const TaskTokens & tokens,
    const std::vector<int32_t> & generated_tokens);

std::vector<int32_t> map_duration_frames(
    const TaskTokens & tokens,
    const std::vector<int32_t> & frames);

std::vector<int32_t> pronunciation_prefix(
    const TaskTokens & tokens,
    const std::vector<PronunciationReference> & references);

std::vector<int32_t> base_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & duration_frames = {},
    const std::vector<int32_t> & pronunciation = {});

std::vector<int32_t> clone_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & reference_text_tokens,
    const std::vector<int32_t> & reference_audio_tokens,
    const std::vector<int32_t> & duration_frames = {},
    const std::vector<int32_t> & pronunciation = {});

std::vector<int32_t> context_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & previous_text_tokens,
    bool same_speaker,
    const std::optional<std::vector<int32_t>> & previous_audio_tokens = std::nullopt,
    const std::vector<int32_t> & duration_frames = {},
    const std::vector<int32_t> & pronunciation = {});

std::vector<int32_t> edit_prompt(
    const TaskTokens & tokens,
    const std::vector<int32_t> & new_text_tokens,
    const std::vector<int32_t> & original_audio_tokens,
    size_t mask_start,
    size_t mask_end,
    const std::vector<int32_t> & pronunciation = {});

}  // namespace engine::models::plapre
