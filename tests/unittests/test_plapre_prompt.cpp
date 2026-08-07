#include "engine/framework/core/execution_context.h"
#include "engine/models/plapre/causal_lm.h"
#include "engine/models/plapre/kanade.h"
#include "engine/models/plapre/prompt.h"
#include "engine/models/plapre/session.h"
#include "engine/models/plapre/tokenizer.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::plapre::PronunciationReference;
using engine::models::plapre::TaskTokens;
using engine::test::require;
using engine::test::require_eq;

void require_tokens(
    const std::vector<int32_t> & actual,
    const std::vector<int32_t> & expected,
    const char * label) {
    require(actual == expected, std::string(label) + " token sequence mismatch");
}

TaskTokens tokens() {
    TaskTokens out;
    int32_t id = 100;
    out.eos = id++;
    out.text = id++;
    out.audio = id++;
    out.audio_base = 1000;
    out.duration = id++;
    out.duration_end = id++;
    out.duration_base = 14000;
    out.reference_text = id++;
    out.reference_text_end = id++;
    out.reference_audio = id++;
    out.reference_audio_end = id++;
    out.context = id++;
    out.context_end = id++;
    out.speaker_same = id++;
    out.speaker_different = id++;
    out.context_audio = id++;
    out.context_audio_end = id++;
    out.edit_mask = id++;
    out.edit_fill = id++;
    out.edit_end = id++;
    out.pronunciation = id++;
    out.pronunciation_end = id++;
    out.pronunciation_word = id++;
    out.pronunciation_word_end = id++;
    out.pronunciation_audio = id++;
    out.pronunciation_audio_end = id++;
    out.audio_end = id++;
    return out;
}

void test_base_prompt_and_stops() {
    const auto task = tokens();
    require_tokens(
        engine::models::plapre::base_prompt(task, {7, 8}),
        std::vector<int32_t>({task.text, 7, 8, task.audio}),
        "base prompt");
    require_tokens(
        engine::models::plapre::base_prompt(task, {7}, {-3, 12, 900}),
        std::vector<int32_t>({
            task.text,
            7,
            task.duration,
            task.duration_base,
            task.duration_base + 12,
            task.duration_base + 399,
            task.duration_end,
            task.audio}),
        "duration prompt");
    require_tokens(task.stop_ids(), {task.eos, *task.audio_end}, "stop ids");
}

void test_audio_token_mapping() {
    const auto task = tokens();
    require_tokens(
        engine::models::plapre::map_audio_tokens(task, {0, 1, 12799}),
        std::vector<int32_t>({task.audio_base, task.audio_base + 1, task.audio_base + 12799}),
        "audio token mapping");
    require_tokens(
        engine::models::plapre::extract_audio_tokens(
            task,
            {4, task.audio_base, task.audio_base + 12799, task.audio_base + 12800}),
        std::vector<int32_t>({0, 12799}),
        "audio token extraction");
    bool rejected = false;
    try {
        (void) engine::models::plapre::map_audio_tokens(task, {12800});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "out-of-range codec token must be rejected");
}

void test_control_prompts_and_caps() {
    const auto task = tokens();
    std::vector<int32_t> long_audio(300);
    for (size_t index = 0; index < long_audio.size(); ++index) {
        long_audio[index] = static_cast<int32_t>(index);
    }
    const auto clone = engine::models::plapre::clone_prompt(task, {3}, {1, 2}, long_audio);
    require_eq(clone.size(), size_t{259}, "clone prompt length includes 250-token cap");
    require_eq(clone[4], task.reference_audio, "clone reference audio marker");
    require_eq(clone[255], task.reference_audio_end, "clone reference audio end marker");

    const auto context = engine::models::plapre::context_prompt(task, {8}, {4, 5}, false, long_audio);
    require_eq(context.size(), size_t{160}, "context prompt length includes 150-token cap");
    require_eq(context[1], task.speaker_different, "different-speaker context marker");

    const auto edit = engine::models::plapre::edit_prompt(task, {9}, {1, 2, 3, 4}, 1, 3);
    require_tokens(
        edit,
        std::vector<int32_t>({
            task.text,
            9,
            task.audio,
            task.audio_base + 1,
            task.edit_mask,
            task.audio_base + 4,
            task.edit_fill,
            task.edit_mask}),
        "edit prompt");
}

void test_pronunciation_limits() {
    const auto task = tokens();
    std::vector<PronunciationReference> references(12);
    for (size_t index = 0; index < references.size(); ++index) {
        references[index].word_tokens = {static_cast<int32_t>(200 + index)};
        references[index].audio_tokens.assign(60, static_cast<int32_t>(index));
    }
    const auto prefix = engine::models::plapre::pronunciation_prefix(task, references);
    const auto audio_count = static_cast<size_t>(std::count_if(prefix.begin(), prefix.end(), [&](int32_t token) {
        return token >= task.audio_base && token < task.audio_base + engine::models::plapre::kAudioCodebookSize;
    }));
    require_eq(audio_count, size_t{200}, "pronunciation total audio cap");
    require_eq(prefix.front(), task.pronunciation, "pronunciation start marker");
    require_eq(prefix.back(), task.pronunciation_end, "pronunciation end marker");
}

void test_real_tokenizer_when_configured() {
    const char * model_path = std::getenv("PLAPRE_TEST_MODEL");
    if (model_path == nullptr || *model_path == '\0') {
        return;
    }
    const auto assets = engine::models::plapre::load_plapre_assets(std::filesystem::path(model_path));
    const engine::models::plapre::PlapreTokenizer tokenizer(assets);
    require_tokens(
        tokenizer.encode_text("Hej, hvordan har du det?"),
        {39, 563, 12, 700, 253, 971, 204, 31},
        "real Danish tokenizer");
    require_tokens(
        tokenizer.build_base_prompt("Dette er en dansk stemme."),
        {8000, 2252, 207, 215, 546, 1944, 14, 8001},
        "real Plapre base prompt");
    const auto voice = assets->speakers.find("tor");
    require(voice != assets->speakers.end(), "real Plapre package has the tor voice");
    const auto projected = engine::models::plapre::project_speaker_embedding(*assets, voice->second);
    require_eq(projected.size(), size_t{960}, "projected speaker size");
    require(std::abs(projected[0] - 19.893198F) < 1.0e-4F, "speaker projection first value parity");
    require(std::abs(projected[1] - 0.395991F) < 1.0e-4F, "speaker projection second value parity");
    float sum = 0.0F;
    for (const float value : projected) {
        sum += value;
    }
    require(std::abs(sum - (-1.366891F)) < 1.0e-3F, "speaker projection checksum parity");
}

void test_real_generator_when_configured() {
    const char * model_path = std::getenv("PLAPRE_TEST_GENERATION");
    if (model_path == nullptr || *model_path == '\0') {
        return;
    }
    const auto assets = engine::models::plapre::load_plapre_assets(std::filesystem::path(model_path));
    const engine::models::plapre::PlapreTokenizer tokenizer(assets);
    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 8});
    engine::models::plapre::PlapreCausalLMRuntime generator(
        assets,
        execution,
        1024ull * 1024ull * 1024ull,
        1024ull * 1024ull * 1024ull,
        512ull * 1024ull * 1024ull,
        engine::assets::TensorStorageType::Native);
    engine::models::plapre::PlapreGenerationOptions options;
    options.max_tokens = 1;
    options.do_sample = false;
    const auto generated = generator.generate(
        tokenizer.build_base_prompt("Dette er en dansk stemme."), "tor", options);
    require_tokens(generated.token_ids, {12643}, "real Plapre greedy first-token parity");
    require_tokens(generated.codec_tokens, {4641}, "real Plapre codec token parity");
}

void test_real_kanade_when_configured() {
    const char * model_path = std::getenv("PLAPRE_TEST_KANADE");
    if (model_path == nullptr || *model_path == '\0') {
        return;
    }
    const auto assets = engine::models::plapre::load_plapre_assets(std::filesystem::path(model_path));
    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 8});
    engine::models::plapre::KanadeDecoder decoder(
        assets->kanade_weights,
        execution,
        1024ull * 1024ull * 1024ull,
        1024ull * 1024ull * 1024ull,
        256ull * 1024ull * 1024ull,
        engine::assets::TensorStorageType::F32);
    const auto mel = decoder.decode({4641, 100, 9999, 42}, assets->speakers.at("tor"));
    require_eq(mel.frames, int64_t{8}, "Kanade mel frame count");
    require_eq(mel.values.size(), size_t{640}, "Kanade mel value count");
    require(
        std::abs(mel.values[0] - (-3.625030F)) < 0.5F,
        "Kanade first mel value parity: " + std::to_string(mel.values[0]));
    float sum = 0.0F;
    for (const float value : mel.values) {
        sum += value;
    }
    require(
        std::abs(sum - (-3200.063F)) < 150.0F,
        "Kanade mel checksum parity: " + std::to_string(sum));
}

void test_real_session_when_configured() {
    const char * model_path = std::getenv("PLAPRE_TEST_SESSION");
    if (model_path == nullptr || *model_path == '\0') {
        return;
    }
    const auto assets = engine::models::plapre::load_plapre_assets(std::filesystem::path(model_path));
    engine::runtime::SessionOptions session_options;
    session_options.backend = {engine::core::BackendType::Cpu, 0, 8};
    engine::models::plapre::PlapreSession session(
        {engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
        session_options,
        assets);
    session.prepare({});
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{"Hej.", "da"};
    request.options = {
        {"voice_id", "tor"},
        {"max_tokens", "4"},
        {"do_sample", "false"},
        {"seed", "42"},
    };
    const auto result = session.run(request);
    require(result.audio_output.has_value(), "Plapre session produces audio output");
    const auto & audio = *result.audio_output;
    require_eq(audio.sample_rate, 24000, "Plapre session sample rate");
    require_eq(audio.channels, 1, "Plapre session channel count");
    require(!audio.samples.empty(), "Plapre session output is non-empty");
    float peak = 0.0F;
    for (const float sample : audio.samples) {
        require(std::isfinite(sample), "Plapre session output samples are finite");
        peak = std::max(peak, std::abs(sample));
    }
    require(peak > 1.0e-4F, "Plapre session output is non-silent");
    require(peak <= 1.0F, "Plapre session output is normalized");
}

}  // namespace

int main() {
    try {
        test_base_prompt_and_stops();
        test_audio_token_mapping();
        test_control_prompts_and_caps();
        test_pronunciation_limits();
        test_real_tokenizer_when_configured();
        test_real_generator_when_configured();
        test_real_kanade_when_configured();
        test_real_session_when_configured();
    } catch (const std::exception & error) {
        std::cerr << "plapre prompt test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "plapre prompt test passed\n";
    return 0;
}
