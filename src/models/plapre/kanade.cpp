#include "engine/models/plapre/kanade.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/models/miocodec/graph_ops.h"
#include "engine/models/miocodec/weights.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::plapre {
namespace {

namespace graphs = engine::models::miocodec::graphs;
namespace modules = engine::modules;

constexpr int64_t kCodebookSize = 12800;
constexpr int64_t kHidden = 768;
constexpr int64_t kCondition = 128;
constexpr int64_t kMelBins = 80;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct PostNetLayer {
    modules::Conv1dConfig config;
    modules::Conv1dWeights convolution;
    modules::NormWeights norm;
};

struct KanadeWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    miocodec::MioCodecWeights bindings;
    miocodec::MioCodecTransformerWeights prenet;
    miocodec::MioCodecTransformerWeights decoder;
    miocodec::MioCodecConvTranspose1dWeights upsample;
    core::TensorValue token_embeddings;
    std::vector<PostNetLayer> postnet;
};

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_qkv_weight(const std::string & name) {
    return name.find(".layers.") != std::string::npos &&
        (ends_with(name, ".attention.wq.weight") ||
         ends_with(name, ".attention.wk.weight") ||
         ends_with(name, ".attention.wv.weight"));
}

core::TensorValue require_tensor(
    const KanadeWeights & weights,
    const std::string & name,
    std::initializer_list<int64_t> shape) {
    const auto found = weights.bindings.tensors.find(name);
    if (found == weights.bindings.tensors.end()) {
        throw std::runtime_error("Kanade checkpoint is missing tensor: " + name);
    }
    core::validate_shape(found->second, core::TensorShape::from_dims(shape), name.c_str());
    return found->second;
}

modules::NormWeights norm(
    const KanadeWeights & weights,
    const std::string & prefix,
    int64_t channels) {
    return {
        require_tensor(weights, prefix + ".weight", {channels}),
        require_tensor(weights, prefix + ".bias", {channels}),
    };
}

miocodec::MioCodecConvTranspose1dWeights conv_transpose(
    const KanadeWeights & weights,
    const std::string & prefix) {
    miocodec::MioCodecConvTranspose1dWeights out;
    out.config = {kHidden, kHidden, 2, 2, 0, 1, true};
    out.weights.weight = require_tensor(weights, prefix + ".weight", {kHidden, kHidden, 2});
    out.weights.bias = require_tensor(weights, prefix + ".bias", {kHidden});
    return out;
}

std::vector<float> content_embedding_table(const assets::TensorSource & source) {
    constexpr std::array<int64_t, 5> levels{8, 8, 8, 5, 5};
    constexpr std::array<int64_t, 5> basis{1, 8, 64, 512, 2560};
    const auto weight = source.require_f32("local_quantizer.proj_out.weight", {kHidden, 5});
    const auto bias = source.require_f32("local_quantizer.proj_out.bias", {kHidden});
    std::vector<float> table(static_cast<size_t>(kCodebookSize * kHidden));
    for (int64_t token = 0; token < kCodebookSize; ++token) {
        std::array<float, 5> code{};
        for (size_t dim = 0; dim < code.size(); ++dim) {
            const int64_t half = levels[dim] / 2;
            const int64_t component = (token / basis[dim]) % levels[dim];
            code[dim] = static_cast<float>(component - half) / static_cast<float>(half);
        }
        for (int64_t output = 0; output < kHidden; ++output) {
            float value = bias[static_cast<size_t>(output)];
            for (size_t dim = 0; dim < code.size(); ++dim) {
                value += weight[static_cast<size_t>(output * 5) + dim] * code[dim];
            }
            table[static_cast<size_t>(token * kHidden + output)] = value;
        }
    }
    return table;
}

std::shared_ptr<const KanadeWeights> load_weights(
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<KanadeWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "plapre.kanade.weights", context_bytes);
    weights->bindings.store = weights->store;
    weights->bindings.source_path = source.source_path();
    for (const auto & tensor : source.tensors()) {
        if (is_qkv_weight(tensor.name)) {
            continue;
        }
        const bool force_f32 = tensor.shape.size() == 1 || ends_with(tensor.name, ".bias") ||
            tensor.name.find(".norm.") != std::string::npos;
        weights->bindings.tensors.emplace(
            tensor.name,
            weights->store->load_tensor(
                source,
                tensor.name,
                force_f32 ? assets::TensorStorageType::F32 : storage_type,
                tensor.shape));
    }
    weights->prenet = miocodec::bind_miocodec_transformer(
        weights->bindings, source, "mel_prenet", kHidden, 6, 12, 65, false, 0, std::nullopt, storage_type);
    weights->decoder = miocodec::bind_miocodec_transformer(
        weights->bindings, source, "mel_decoder", kHidden, 6, 12, 65, true, kCondition, kMelBins, storage_type);
    weights->upsample = conv_transpose(*weights, "mel_conv_upsample");
    weights->token_embeddings = weights->store->make_from_f32(
        core::TensorShape::from_dims({kCodebookSize, kHidden}),
        storage_type,
        content_embedding_table(source));
    weights->postnet.reserve(4);
    for (int64_t layer = 0; layer < 4; ++layer) {
        const int64_t input = layer == 0 ? kMelBins : 256;
        const int64_t output = layer == 3 ? kMelBins : 256;
        const std::string prefix = "mel_postnet.convolutions." + std::to_string(layer);
        PostNetLayer item;
        item.config = {input, output, 7, 1, 3, 1, true};
        item.convolution.weight = require_tensor(*weights, prefix + ".0.weight", {output, input, 7});
        item.convolution.bias = require_tensor(*weights, prefix + ".0.bias", {output});
        item.norm = norm(*weights, prefix + ".1.norm", output);
        weights->postnet.push_back(std::move(item));
    }
    weights->store->upload();
    return weights;
}

core::TensorValue postnet(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const KanadeWeights & weights) {
    auto x = input;
    for (size_t index = 0; index < weights.postnet.size(); ++index) {
        const auto & layer = weights.postnet[index];
        x = modules::Conv1dModule(layer.config).build(ctx, x, layer.convolution);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = modules::LayerNormModule({layer.config.out_channels, 1.0e-5F, true, true})
                .build(ctx, x, layer.norm);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        if (index + 1 < weights.postnet.size()) {
            x = core::ensure_backend_addressable_layout(ctx, x);
            x = core::wrap_tensor(ggml_tanh(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        }
    }
    return modules::ResidualAddModule{}.build(ctx, input, x);
}

class DecoderGraph {
public:
    DecoderGraph(
        std::shared_ptr<const KanadeWeights> weights,
        int64_t token_frames,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t constant_context_bytes)
        : weights_(std::move(weights)),
          token_frames_(token_frames),
          mel_frames_(token_frames * 2),
          backend_(execution.backend()),
          constants_(std::make_unique<core::ConstantTensorCache>(
              backend_, execution.config().threads, "plapre.kanade.constants", constant_context_bytes)) {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Kanade decoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "plapre.kanade", execution.backend_type()};
        token_ids_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({token_frames_}));
        condition_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, kCondition}));
        ggml_set_input(token_ids_.tensor);
        ggml_set_input(condition_.tensor);
        constants_->begin_graph();
        auto x = modules::EmbeddingModule({kCodebookSize, kHidden})
                     .build(ctx, token_ids_, weights_->token_embeddings);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, token_frames_, kHidden}));
        x = graphs::transformer(ctx, *constants_, x, weights_->prenet, std::nullopt, std::nullopt, true);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = modules::ConvTranspose1dModule(weights_->upsample.config).build(ctx, x, weights_->upsample.weights);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = graphs::transformer(ctx, *constants_, x, weights_->decoder, condition_, std::nullopt, true);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        x = postnet(ctx, x, *weights_);
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        constants_->finish_graph();
        constants_->ensure_uploaded();
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (allocator_ == nullptr || !ggml_gallocr_reserve(allocator_, graph_) ||
            !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("failed to allocate Kanade decoder graph");
        }
    }

    ~DecoderGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (allocator_ != nullptr) {
            ggml_gallocr_free(allocator_);
        }
    }

    bool matches(size_t token_count) const noexcept {
        return token_count == static_cast<size_t>(token_frames_);
    }

    KanadeMel run(const std::vector<int32_t> & tokens, const std::vector<float> & condition) {
        if (!matches(tokens.size())) {
            throw std::runtime_error("Kanade token count does not match decoder graph");
        }
        if (condition.size() != kCondition) {
            throw std::runtime_error("Kanade speaker embedding must contain 128 values");
        }
        for (const int32_t token : tokens) {
            if (token < 0 || token >= kCodebookSize) {
                throw std::runtime_error("Kanade token is outside the codebook");
            }
        }
        core::write_tensor_i32(token_ids_, tokens);
        core::write_tensor_f32(condition_, condition);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Kanade decoder graph execution failed");
        }
        KanadeMel out;
        out.frames = mel_frames_;
        out.values = core::read_tensor_f32(output_);
        return out;
    }

private:
    std::shared_ptr<const KanadeWeights> weights_;
    int64_t token_frames_ = 0;
    int64_t mel_frames_ = 0;
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue token_ids_;
    core::TensorValue condition_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
};

}  // namespace

struct KanadeDecoder::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_bytes,
        size_t constant_bytes,
        assets::TensorStorageType storage_type)
        : execution(execution_context),
          weights(load_weights(*source, execution_context, weight_context_bytes, storage_type)),
          graph_arena_bytes(graph_bytes),
          constant_context_bytes(constant_bytes) {}

    core::ExecutionContext & execution;
    std::shared_ptr<const KanadeWeights> weights;
    size_t graph_arena_bytes = 0;
    size_t constant_context_bytes = 0;
    std::unique_ptr<DecoderGraph> graph;
};

KanadeDecoder::KanadeDecoder(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(source),
          execution,
          weight_context_bytes,
          graph_arena_bytes,
          constant_context_bytes,
          storage_type)) {}

KanadeDecoder::~KanadeDecoder() = default;

KanadeMel KanadeDecoder::decode(
    const std::vector<int32_t> & tokens,
    const std::vector<float> & speaker_embedding) {
    if (tokens.empty()) {
        throw std::runtime_error("Kanade decoder requires at least one token");
    }
    if (impl_->graph == nullptr || !impl_->graph->matches(tokens.size())) {
        impl_->graph = std::make_unique<DecoderGraph>(
            impl_->weights,
            static_cast<int64_t>(tokens.size()),
            impl_->execution,
            impl_->graph_arena_bytes,
            impl_->constant_context_bytes);
    }
    return impl_->graph->run(tokens, speaker_embedding);
}

}  // namespace engine::models::plapre
