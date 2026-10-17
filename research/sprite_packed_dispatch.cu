// Research-only packed RGB input. All families receive the same pixels + XY.
#undef OBS_HEIGHT
#undef OBS_WIDTH
#define OBS_HEIGHT SPRITE_PACKED_HEIGHT
#define OBS_WIDTH SPRITE_PACKED_WIDTH
#define OBS_SIZE SPRITE_VIEW_PIXELS
#include "../ocean/connect4cnn/nature.cu"
#include "../ocean/connect4cnn/impala.cu"
namespace c4_cnn {
using ::imp_pool;
using ::imp_pool_grad;
using ::imp_add;
}
#include "../ocean/connect4cnn/flex.cu"
#include "sprite_attention_encoder.cu"

struct SpriteImageWeights {
    Encoder image;
    void* weights;
    Prec weight, bias;
    int hidden;
};

struct SpriteImageActivations {
    void* image;
    Prec pixels, features, output;
    Prec grad_output, grad_features, grad_image, weight_grad, bias_grad;
};

__global__ void sprite_image_pixels(const precision_t* input, precision_t* pixels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pixels[i] = input[i / SPRITE_VIEW_PIXELS * SPRITE_VIEW_OBS + i % SPRITE_VIEW_PIXELS];
    }
}

__global__ void sprite_image_features(const precision_t* image, const precision_t* input,
        precision_t* features, int B, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * (hidden + 4)) {
        int b = i / (hidden + 4), c = i % (hidden + 4);
        features[i] = c < hidden ? image[b * hidden + c]
            : input[b * SPRITE_VIEW_OBS + SPRITE_VIEW_PIXELS + c - hidden];
    }
}

__global__ void sprite_image_gradient(const precision_t* features, precision_t* image,
        int n, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        image[i] = features[i / hidden * (hidden + 4) + i % hidden];
    }
}

static Prec sprite_image_forward(void* weights, void* activations, Prec input,
        cudaStream_t stream) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    SpriteImageActivations* a = (SpriteImageActivations*)activations;
    int B = a->output.shape[0];
    int n = B * SPRITE_VIEW_PIXELS;
    sprite_image_pixels<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data, a->pixels.data, n);
    Prec image = w->image.forward(w->weights, a->image, a->pixels, stream);
    n = B * (w->hidden + 4);
    sprite_image_features<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        image.data, input.data, a->features.data, B, w->hidden);
    puf_mm(&a->features, &w->weight, &a->output, stream);
    n = B * w->hidden;
    nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        a->output.data, w->bias.data, n, w->hidden);
    return a->output;
}

static void sprite_image_backward(void* weights, void* activations, Prec grad,
        cudaStream_t stream) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    SpriteImageActivations* a = (SpriteImageActivations*)activations;
    int B = a->output.shape[0];
    puf_copy(&a->grad_output, &grad, stream);
    nature_relu_backward<<<grid_size(B * w->hidden), BLOCK_SIZE, 0, stream>>>(
        a->grad_output.data, a->output.data, B * w->hidden);
    puf_mm_tn(&a->grad_output, &a->features, &a->weight_grad, stream);
    nature_bias_backward<<<w->hidden, 256, 0, stream>>>(
        a->grad_output.data, a->bias_grad.data, B, w->hidden);
    puf_mm_nn(&a->grad_output, &w->weight, &a->grad_features, stream);
    sprite_image_gradient<<<grid_size(B * w->hidden), BLOCK_SIZE, 0, stream>>>(
        a->grad_features.data, a->grad_image.data, B * w->hidden, w->hidden);
    w->image.backward(w->weights, a->image, a->grad_image, stream);
}

static void* sprite_image_create(void* self) {
    Encoder* enc = (Encoder*)self;
    SpriteImageWeights* w = (SpriteImageWeights*)calloc(1, sizeof(*w));
    w->hidden = enc->out_dim;
    w->image = {.in_dim = SPRITE_VIEW_PIXELS, .out_dim = w->hidden};
    int family = dict_get(enc->config, "encoder");
    if (family == 2) {
        create_nature_encoder(&w->image);
    } else if (family == 4) {
        create_flex_encoder(&w->image, enc->config);
    } else {
        assert(family == 0 || family == 7);
        create_connect4_encoder(&w->image);
    }
    w->weights = w->image.create_weights(&w->image);
    if (family == 7) {
        ImpalaWeights* imp = (ImpalaWeights*)w->weights;
        imp->gap = true;
        imp->readout = 32;
    }
    return w;
}

static void sprite_image_params(void* weights, Allocator* alloc) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    w->image.reg_params(w->weights, alloc);
    w->weight = {.shape = {w->hidden, w->hidden + 4}};
    w->bias = {.shape = {w->hidden}};
    alloc_register(alloc, &w->weight);
    alloc_register(alloc, &w->bias);
}

static void sprite_image_init(void* weights, ulong* seed, cudaStream_t stream) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    w->image.init_weights(w->weights, seed, stream);
    puf_kaiming_init(&w->weight, sqrtf(2.0f), (*seed)++, stream);
    cudaMemsetAsync(w->bias.data, 0, w->hidden * sizeof(precision_t), stream);
}

static void sprite_image_buffers(SpriteImageWeights* w, SpriteImageActivations* a,
        Allocator* alloc, int B) {
    a->pixels = {.shape = {B, SPRITE_VIEW_PIXELS}};
    a->features = {.shape = {B, w->hidden + 4}};
    a->output = {.shape = {B, w->hidden}};
    alloc_register(alloc, &a->pixels);
    alloc_register(alloc, &a->features);
    alloc_register(alloc, &a->output);
}

static void sprite_image_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    SpriteImageActivations* a = (SpriteImageActivations*)activations;
    *a = {};
    a->image = calloc(1, w->image.activation_size);
    w->image.reg_rollout(w->weights, a->image, alloc, B);
    sprite_image_buffers(w, a, alloc, B);
}

static void sprite_image_train(void* weights, void* activations,
        Allocator* acts, Allocator* grads, int B) {
    SpriteImageWeights* w = (SpriteImageWeights*)weights;
    SpriteImageActivations* a = (SpriteImageActivations*)activations;
    *a = {};
    a->image = calloc(1, w->image.activation_size);
    w->image.reg_train(w->weights, a->image, acts, grads, B);
    sprite_image_buffers(w, a, acts, B);
    a->grad_output = {.shape = {B, w->hidden}};
    a->grad_features = {.shape = {B, w->hidden + 4}};
    a->grad_image = {.shape = {B, w->hidden}};
    a->weight_grad = {.shape = {w->hidden, w->hidden + 4}};
    a->bias_grad = {.shape = {w->hidden}};
    alloc_register(acts, &a->grad_output);
    alloc_register(acts, &a->grad_features);
    alloc_register(acts, &a->grad_image);
    alloc_register(grads, &a->weight_grad);
    alloc_register(grads, &a->bias_grad);
}

static void create_sprite_research_encoder(Encoder* enc, Dict* policy) {
    assert(enc->in_dim == SPRITE_VIEW_OBS);
    DictItem* item = dict_find(policy, "encoder");
    assert(item && item->value == (int)item->value);
    if (item->value == 6) {
        create_forage_attention_encoder(enc);
        return;
    }
    assert(item->value == 0 || item->value == 2 || item->value == 4 || item->value == 7);
    *enc = Encoder{.forward = sprite_image_forward, .backward = sprite_image_backward,
        .init_weights = sprite_image_init, .reg_params = sprite_image_params,
        .reg_train = sprite_image_train, .reg_rollout = sprite_image_rollout,
        .create_weights = sprite_image_create, .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(SpriteImageActivations), .config = policy};
}

#undef OBS_SIZE
#undef OBS_HEIGHT
#undef OBS_WIDTH
#define OBS_HEIGHT (GRID_HEIGHT * SPRITE_CELL)
#define OBS_WIDTH (GRID_WIDTH * SPRITE_CELL)
