#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::plapre {

struct KanadeMel {
    std::vector<float> values;
    int64_t bins = 80;
    int64_t frames = 0;
};

class KanadeDecoder {
public:
    struct Impl;

    KanadeDecoder(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        assets::TensorStorageType storage_type);
    ~KanadeDecoder();

    KanadeMel decode(
        const std::vector<int32_t> & tokens,
        const std::vector<float> & speaker_embedding);

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::plapre
