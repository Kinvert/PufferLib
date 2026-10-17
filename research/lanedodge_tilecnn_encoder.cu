// Research encoder: shared tile CNN, then spatially ordered features + geometry.
// Same architecture as the Sprite Forage tile CNN (4x4/s2 3->8, 3x3/s2 8->16,
// 16 features per 32x32 RGB tile), applied to the lane-dodge 18-tile agent view.
#define LD_TILE_FEATURES 16
#define LD_TILE_PIXELS (LD_CELL * LD_CELL * 3)
#define LD_TILECNN_FEATURES (LD_VIEW_TILES * LD_TILE_FEATURES + LD_VIEW_GEOMETRY)

struct LanedodgeTileCnnWeights {
    NatureWeights tile;
    Prec weight, bias;
    int hidden;
};

struct LanedodgeTileCnnActivations {
    NatureActivations tile;
    Prec pixels, features, output;
    Prec grad_output, grad_features, grad_tile, weight_grad, bias_grad;
};

__global__ void ld_pack_tiles(const precision_t* input, precision_t* pixels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = i / LD_VIEW_PIXELS;
        pixels[i] = input[b * LD_VIEW_OBS + i % LD_VIEW_PIXELS];
    }
}

__global__ void ld_pack_features(const precision_t* tiles, const precision_t* input,
        precision_t* features, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = i / LD_TILECNN_FEATURES;
        int c = i % LD_TILECNN_FEATURES;
        int count = LD_VIEW_TILES * LD_TILE_FEATURES;
        features[i] = c < count ? tiles[b * count + c]
            : input[b * LD_VIEW_OBS + LD_VIEW_PIXELS + c - count];
    }
}

__global__ void ld_unpack_grad(const precision_t* features, precision_t* tiles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int count = LD_VIEW_TILES * LD_TILE_FEATURES;
        tiles[i] = features[(i / count) * LD_TILECNN_FEATURES + i % count];
    }
}

static Prec lanedodge_tilecnn_forward(void* weights, void* activations, Prec input,
        cudaStream_t stream) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    LanedodgeTileCnnActivations* a = (LanedodgeTileCnnActivations*)activations;
    int B = a->output.shape[0];
    int n = B * LD_VIEW_PIXELS;
    ld_pack_tiles<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data, a->pixels.data, n);
    Prec x = a->pixels;
    for (int i = 0; i < w->tile.count; i++) {
        NatureLayer d = w->tile.layer[i];
        int rows = B * LD_VIEW_TILES * d.oh * d.ow;
        int patches = rows * d.k * d.k * d.ci;
        nature_im2col<<<grid_size(patches), BLOCK_SIZE, 0, stream>>>(
            x.data, a->tile.patches[i].data, d, B * LD_VIEW_TILES);
        puf_mm(&a->tile.patches[i], &w->tile.weight[i], &a->tile.output[i], stream);
        nature_bias_relu<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
            a->tile.output[i].data, w->tile.bias[i].data, rows * d.co, d.co);
        x = a->tile.output[i];
    }
    n = B * LD_TILECNN_FEATURES;
    ld_pack_features<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        x.data, input.data, a->features.data, n);
    puf_mm(&a->features, &w->weight, &a->output, stream);
    n = B * w->hidden;
    nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        a->output.data, w->bias.data, n, w->hidden);
    return a->output;
}

static void lanedodge_tilecnn_backward(void* weights, void* activations, Prec grad,
        cudaStream_t stream) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    LanedodgeTileCnnActivations* a = (LanedodgeTileCnnActivations*)activations;
    int B = a->output.shape[0];
    puf_copy(&a->grad_output, &grad, stream);
    nature_relu_backward<<<grid_size(B * w->hidden), BLOCK_SIZE, 0, stream>>>(
        a->grad_output.data, a->output.data, B * w->hidden);
    puf_mm_tn(&a->grad_output, &a->features, &a->weight_grad, stream);
    nature_bias_backward<<<w->hidden, 256, 0, stream>>>(
        a->grad_output.data, a->bias_grad.data, B, w->hidden);
    puf_mm_nn(&a->grad_output, &w->weight, &a->grad_features, stream);
    int n = B * LD_VIEW_TILES * LD_TILE_FEATURES;
    ld_unpack_grad<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        a->grad_features.data, a->grad_tile.data, n);
    nature_backward(&w->tile, &a->tile, a->grad_tile, stream);
}

static void* lanedodge_tilecnn_create(void* self) {
    Encoder* enc = (Encoder*)self;
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)calloc(1, sizeof(*w));
    w->hidden = enc->out_dim;
    w->tile.count = 3;
    w->tile.layer[0] = {32, 32, 3, 8, 4, 2, 15, 15};
    w->tile.layer[1] = {15, 15, 8, 16, 3, 2, 7, 7};
    w->tile.layer[2] = {1, 1, 7 * 7 * 16, LD_TILE_FEATURES, 1, 1, 1, 1};
    return w;
}

static void lanedodge_tilecnn_params(void* weights, Allocator* alloc) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    nature_reg_params(&w->tile, alloc);
    w->weight = {.shape = {w->hidden, LD_TILECNN_FEATURES}};
    w->bias = {.shape = {w->hidden}};
    alloc_register(alloc, &w->weight);
    alloc_register(alloc, &w->bias);
}

static void lanedodge_tilecnn_init(void* weights, ulong* seed, cudaStream_t stream) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    nature_init_weights(&w->tile, seed, stream);
    puf_kaiming_init(&w->weight, sqrtf(2.0f), (*seed)++, stream);
    cudaMemsetAsync(w->bias.data, 0, w->hidden * sizeof(precision_t), stream);
}

static void lanedodge_tilecnn_buffers(LanedodgeTileCnnWeights* w,
        LanedodgeTileCnnActivations* a, Allocator* alloc, int B) {
    a->pixels = {.shape = {B * LD_VIEW_TILES, LD_TILE_PIXELS}};
    a->features = {.shape = {B, LD_TILECNN_FEATURES}};
    a->output = {.shape = {B, w->hidden}};
    alloc_register(alloc, &a->pixels);
    alloc_register(alloc, &a->features);
    alloc_register(alloc, &a->output);
}

static void lanedodge_tilecnn_rollout(void* weights, void* activations,
        Allocator* alloc, int B) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    LanedodgeTileCnnActivations* a = (LanedodgeTileCnnActivations*)activations;
    *a = {};
    nature_reg_rollout(&w->tile, &a->tile, alloc, B * LD_VIEW_TILES);
    lanedodge_tilecnn_buffers(w, a, alloc, B);
}

static void lanedodge_tilecnn_train(void* weights, void* activations,
        Allocator* acts, Allocator* grads, int B) {
    LanedodgeTileCnnWeights* w = (LanedodgeTileCnnWeights*)weights;
    LanedodgeTileCnnActivations* a = (LanedodgeTileCnnActivations*)activations;
    *a = {};
    nature_reg_train(&w->tile, &a->tile, acts, grads, B * LD_VIEW_TILES);
    lanedodge_tilecnn_buffers(w, a, acts, B);
    a->grad_output = {.shape = {B, w->hidden}};
    a->grad_features = {.shape = {B, LD_TILECNN_FEATURES}};
    a->grad_tile = {.shape = {B * LD_VIEW_TILES, LD_TILE_FEATURES}};
    alloc_register(acts, &a->grad_output);
    alloc_register(acts, &a->grad_features);
    alloc_register(acts, &a->grad_tile);
    a->weight_grad = {.shape = {w->hidden, LD_TILECNN_FEATURES}};
    a->bias_grad = {.shape = {w->hidden}};
    alloc_register(grads, &a->weight_grad);
    alloc_register(grads, &a->bias_grad);
}

static void create_lanedodge_tilecnn_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = lanedodge_tilecnn_forward, .backward = lanedodge_tilecnn_backward,
        .init_weights = lanedodge_tilecnn_init, .reg_params = lanedodge_tilecnn_params,
        .reg_train = lanedodge_tilecnn_train, .reg_rollout = lanedodge_tilecnn_rollout,
        .create_weights = lanedodge_tilecnn_create,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(LanedodgeTileCnnActivations),
    };
}
