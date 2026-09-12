// Fixed-input NAR benchmark: compare raw float32 latents, before WAV
// quantization.
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/nar_runtime.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string arg(int argc, char **argv, const std::string &key, const std::string &fallback = "") {
    for (int i = 1; i + 1 < argc; ++i)
        if (argv[i] == key)
            return argv[i + 1];
    return fallback;
}
template <class T> std::vector<T> read(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    const auto size = f.tellg();
    if (size <= 0 || size % sizeof(T))
        throw std::runtime_error("invalid size: " + path);
    std::vector<T> out(static_cast<size_t>(size) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(out.data()), size);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        using namespace engine;
        const auto path = arg(argc, argv, "--output");
        if (path.empty() || std::filesystem::exists(path)) {
            throw std::runtime_error("provide a new output path");
        }
        const auto fused = arg(argc, argv, "--fused-swiglu", "false");
        if (fused != "true" && fused != "false") {
            throw std::runtime_error("--fused-swiglu requires true or false");
        }
        const auto steps = std::stoll(arg(argc, argv, "--steps", "32"));
        if (steps <= 0) {
            throw std::runtime_error("--steps must be positive");
        }
        const auto prefix = read<int32_t>(arg(argc, argv, "--prefix"));
        const auto codec = read<int32_t>(arg(argc, argv, "--codec"));
        for (auto token : codec)
            if (token < 0 || token >= models::yue2::kCodecSize)
                throw std::runtime_error("codec IDs must exclude the vocabulary offset");
        auto assets =
            std::make_shared<models::yue2::Yue2Assets>(*models::yue2::load_yue2_assets(arg(argc, argv, "--model")));
        assets->model_weights = assets::open_tensor_source(assets->model_root / "yue2-3b-q8_0.gguf", "model_weights");
        core::BackendConfig config;
        const auto backend = arg(argc, argv, "--backend", "hip");
        if (backend == "hip")
            config.type = core::BackendType::Hip;
        else if (backend == "cuda")
            config.type = core::BackendType::Cuda;
        else if (backend == "cpu")
            config.type = core::BackendType::Cpu;
        else
            throw std::runtime_error("unknown backend");
        config.threads = 8;
        core::ExecutionContext execution(config);
        constexpr size_t MiB = 1024 * 1024;
        models::yue2::Yue2ArRuntime ar(execution, assets, assets::TensorStorageType::Native, 6144 * MiB, 4096 * MiB,
                                       1536 * MiB);
        models::yue2::Yue2NarRuntime nar(execution, assets, assets::TensorStorageType::Native, 6144 * MiB, 6144 * MiB,
                                         fused == "true");
        const auto started = std::chrono::steady_clock::now();
        const auto output = nar.synthesize(
            prefix, codec, [&](const std::vector<int32_t> &tokens) { return ar.prefill_device_state(tokens); }, {},
            std::stoull(arg(argc, argv, "--seed", "832002")), steps, models::yue2::kContextTokens);
        for (float value : output)
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite latent");
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
        if (!file)
            throw std::runtime_error("cannot write " + path);
        std::cout << "seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                  << " values=" << output.size() << '\n';
        const auto reference = arg(argc, argv, "--reference");
        if (!reference.empty()) {
            const auto expected = read<float>(reference);
            const bool exact = output.size() == expected.size() &&
                               std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0;
            std::cout << "bitwise_equal=" << exact << '\n';
            if (!exact)
                return 2;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
