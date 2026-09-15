// Tiny pixel encoder. Included by src/ocean.cu.
// Conv4x4/s4 (1 -> 8), ReLU, flatten NHWC, Linear -> hidden. No biases.

static constexpr int C4_KERNEL = 4;
static constexpr int C4_CHANNELS = 8;
static constexpr int C4_PATCH = C4_KERNEL * C4_KERNEL;
static constexpr int C4_HEIGHT = OBS_HEIGHT / C4_KERNEL;
static constexpr int C4_WIDTH = OBS_WIDTH / C4_KERNEL;
static constexpr int C4_SPATIAL = C4_HEIGHT * C4_WIDTH;
static constexpr int C4_FLAT = C4_SPATIAL * C4_CHANNELS;
static_assert(OBS_CHANNELS == 1 && OBS_HEIGHT % C4_KERNEL == 0
    && OBS_WIDTH % C4_KERNEL == 0, "tiny CNN requires whole grayscale patches");

struct Connect4EncoderWeights {
    Prec conv_w, proj_w;
    int hidden;
};

struct Connect4EncoderActivations {
    Prec patches, conv, out, grad_conv;
    Prec conv_wgrad, proj_wgrad;
};

__global__ void c4_im2col(const precision_t* __restrict__ input,
        precision_t* __restrict__ patches, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * C4_SPATIAL * C4_PATCH) return;
    int k = idx % C4_PATCH;
    int spatial = (idx / C4_PATCH) % C4_SPATIAL;
    int b = idx / (C4_SPATIAL * C4_PATCH);
    int y = (spatial / C4_WIDTH) * C4_KERNEL + k / C4_KERNEL;
    int x = (spatial % C4_WIDTH) * C4_KERNEL + k % C4_KERNEL;
    patches[idx] = input[(int64_t)b * OBS_SIZE + y * OBS_WIDTH + x];
}

__global__ void c4_relu(precision_t* values, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) values[idx] = from_float(fmaxf(0.0f, to_float(values[idx])));
}

__global__ void c4_relu_backward(precision_t* grad,
        const precision_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && to_float(output[idx]) <= 0.0f) grad[idx] = from_float(0.0f);
}

static Prec c4_encoder_forward(void* w, void* activations,
        Prec input, cudaStream_t stream) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    Connect4EncoderActivations* a = (Connect4EncoderActivations*)activations;
    int B = (int)(numel(input.shape) / OBS_SIZE);
    assert(numel(input.shape) == (int64_t)B * OBS_SIZE && B == a->out.shape[0]);
    c4_im2col<<<grid_size(B * C4_SPATIAL * C4_PATCH), BLOCK_SIZE, 0, stream>>>(
        input.data, a->patches.data, B);
    puf_mm(&a->patches, &ew->conv_w, &a->conv, stream);
    c4_relu<<<grid_size(B * C4_FLAT), BLOCK_SIZE, 0, stream>>>(a->conv.data, B * C4_FLAT);
    Prec flat = {.data = a->conv.data, .shape = {B, C4_FLAT}};
    puf_mm(&flat, &ew->proj_w, &a->out, stream);
    return a->out;
}

static void c4_encoder_backward(void* w, void* activations,
        Prec grad, cudaStream_t stream) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    Connect4EncoderActivations* a = (Connect4EncoderActivations*)activations;
    int B = (int)a->out.shape[0];
    assert(numel(grad.shape) == (int64_t)B * ew->hidden);
    Prec g = {.data = grad.data, .shape = {B, ew->hidden}};
    Prec flat = {.data = a->conv.data, .shape = {B, C4_FLAT}};
    Prec grad_flat = {.data = a->grad_conv.data, .shape = {B, C4_FLAT}};
    // All encoder GEMMs use the caller's stream. No side-stream buffer reuse.
    puf_mm_tn(&g, &flat, &a->proj_wgrad, stream);
    puf_mm_nn(&g, &ew->proj_w, &grad_flat, stream);
    c4_relu_backward<<<grid_size(B * C4_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->grad_conv.data, a->conv.data, B * C4_FLAT);
    puf_mm_tn(&a->grad_conv, &a->patches, &a->conv_wgrad, stream);
    // Observations are external inputs; the encoder API does not return dPixels.
}

static void c4_encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    puf_kaiming_init(&ew->conv_w, sqrtf(2.0f), (*seed)++, stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
}

static void c4_encoder_reg_params(void* w, Allocator* alloc) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    ew->conv_w = {.shape = {C4_CHANNELS, C4_PATCH}};
    ew->proj_w = {.shape = {ew->hidden, C4_FLAT}};
    alloc_register(alloc, &ew->conv_w);
    alloc_register(alloc, &ew->proj_w);
}

static void c4_encoder_reg_rollout(void* w, void* activations,
        Allocator* alloc, int B) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    Connect4EncoderActivations* a = (Connect4EncoderActivations*)activations;
    *a = {};
    a->patches = {.shape = {B * C4_SPATIAL, C4_PATCH}};
    a->conv = {.shape = {B * C4_SPATIAL, C4_CHANNELS}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->patches);
    alloc_register(alloc, &a->conv);
    alloc_register(alloc, &a->out);
}

static void c4_encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B) {
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)w;
    Connect4EncoderActivations* a = (Connect4EncoderActivations*)activations;
    c4_encoder_reg_rollout(w, activations, acts, B);
    a->grad_conv = {.shape = {B * C4_SPATIAL, C4_CHANNELS}};
    alloc_register(acts, &a->grad_conv);
    // Registration order and sizes match the parameters, including alignment.
    a->conv_wgrad = {.shape = {C4_CHANNELS, C4_PATCH}};
    a->proj_wgrad = {.shape = {ew->hidden, C4_FLAT}};
    alloc_register(grads, &a->conv_wgrad);
    alloc_register(grads, &a->proj_wgrad);
}

static void* c4_encoder_create_weights(void* self) {
    Encoder* enc = (Encoder*)self;
    assert(enc->in_dim == OBS_SIZE && enc->out_dim > 0);
    Connect4EncoderWeights* ew = (Connect4EncoderWeights*)calloc(1, sizeof(*ew));
    ew->hidden = enc->out_dim;
    return ew;
}

static void create_connect4_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = c4_encoder_forward,
        .backward = c4_encoder_backward,
        .init_weights = c4_encoder_init_weights,
        .reg_params = c4_encoder_reg_params,
        .reg_train = c4_encoder_reg_train,
        .reg_rollout = c4_encoder_reg_rollout,
        .create_weights = c4_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(Connect4EncoderActivations),
    };
}
