// Nature convolution stack adapted to grayscale 36x44 and the common core width.
// Valid Conv8/s4,32 -> Conv4/s2,64 -> Conv3/s1,64 -> Linear(hidden).
// Bias and ReLU after every layer. NHWC intermediates; no input gradient returned.

struct NatureLayer {
    int ih, iw, ci, co, k, stride, oh, ow;
};

struct NatureWeights {
    NatureLayer layer[4];
    Prec weight[4], bias[4];
};

struct NatureActivations {
    Prec patches[4], output[4], grad_output[4], grad_patches[4];
    Prec weight_grad[4], bias_grad[4];
};

__global__ void nature_im2col(const precision_t* input, precision_t* patches,
        NatureLayer d, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int patch = d.k * d.k * d.ci;
    if (idx >= B * d.oh * d.ow * patch) return;
    int q = idx % patch;
    int p = idx / patch;
    int x = p % d.ow;
    int y = (p / d.ow) % d.oh;
    int b = p / (d.oh * d.ow);
    int iy = y * d.stride + q / (d.k * d.ci);
    int ix = x * d.stride + (q / d.ci) % d.k;
    patches[idx] = input[((int64_t)b * d.ih * d.iw + iy * d.iw + ix) * d.ci + q % d.ci];
}

// Each input element gathers its contributors in a fixed order; no atomics.
__global__ void nature_col2im(const precision_t* patches, precision_t* grad,
        NatureLayer d, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d.ih * d.iw * d.ci) return;
    int c = idx % d.ci;
    int x = (idx / d.ci) % d.iw;
    int y = (idx / (d.ci * d.iw)) % d.ih;
    int b = idx / (d.ci * d.iw * d.ih);
    float sum = 0;
    for (int ky = 0; ky < d.k; ky++) {
        int oy = y - ky;
        if (oy < 0 || oy % d.stride || oy / d.stride >= d.oh) continue;
        for (int kx = 0; kx < d.k; kx++) {
            int ox = x - kx;
            if (ox < 0 || ox % d.stride || ox / d.stride >= d.ow) continue;
            int64_t row = ((int64_t)b * d.oh + oy / d.stride) * d.ow + ox / d.stride;
            sum += to_float(patches[row * d.k * d.k * d.ci + (ky * d.k + kx) * d.ci + c]);
        }
    }
    grad[idx] = from_float(sum);
}

__global__ void nature_bias_relu(precision_t* output, const precision_t* bias, int n, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = from_float(fmaxf(0.0f, to_float(output[idx]) + to_float(bias[idx % channels])));
}

__global__ void nature_relu_backward(precision_t* grad, const precision_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && to_float(output[idx]) <= 0.0f) grad[idx] = from_float(0.0f);
}

__global__ void nature_bias_backward(const precision_t* grad, precision_t* bias_grad, int rows, int channels) {
    __shared__ float sums[256];
    int c = blockIdx.x;
    float sum = 0;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) sum += to_float(grad[(int64_t)row * channels + c]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias_grad[c] = from_float(sums[0]);
}

static Prec nature_forward(void* weights, void* activations, Prec input, cudaStream_t stream) {
    NatureWeights* w = (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int B = (int)(numel(input.shape) / OBS_SIZE);
    assert(numel(input.shape) == (int64_t)B * OBS_SIZE && B == a->output[3].shape[0]);
    for (int i = 0; i < 4; i++) {
        NatureLayer d = w->layer[i];
        int n = B * d.oh * d.ow * d.k * d.k * d.ci;
        nature_im2col<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data, a->patches[i].data, d, B);
        puf_mm(&a->patches[i], &w->weight[i], &a->output[i], stream);
        n = B * d.oh * d.ow * d.co;
        nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->output[i].data, w->bias[i].data, n, d.co);
        input = a->output[i];
    }
    return a->output[3];
}

static void nature_backward(void* weights, void* activations, Prec grad, cudaStream_t stream) {
    NatureWeights* w = (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int B = (int)a->output[3].shape[0];
    assert(numel(grad.shape) == numel(a->grad_output[3].shape));
    puf_copy(&a->grad_output[3], &grad, stream);
    for (int i = 3; i >= 0; i--) {
        NatureLayer d = w->layer[i];
        int rows = B * d.oh * d.ow;
        nature_relu_backward<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(a->grad_output[i].data, a->output[i].data, rows * d.co);
        puf_mm_tn(&a->grad_output[i], &a->patches[i], &a->weight_grad[i], stream);
        nature_bias_backward<<<d.co, 256, 0, stream>>>(a->grad_output[i].data, a->bias_grad[i].data, rows, d.co);
        if (i > 0) {
            puf_mm_nn(&a->grad_output[i], &w->weight[i], &a->grad_patches[i], stream);
            int n = B * d.ih * d.iw * d.ci;
            nature_col2im<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->grad_patches[i].data, a->grad_output[i - 1].data, d, B);
        }
    }
}

static void nature_init_weights(void* weights, ulong* seed, cudaStream_t stream) {
    NatureWeights* w = (NatureWeights*)weights;
    for (int i = 0; i < 4; i++) {
        puf_kaiming_init(&w->weight[i], sqrtf(2.0f), (*seed)++, stream);
        cudaMemsetAsync(w->bias[i].data, 0, numel(w->bias[i].shape) * sizeof(precision_t), stream);
    }
}

static void nature_reg_params(void* weights, Allocator* alloc) {
    NatureWeights* w = (NatureWeights*)weights;
    for (int i = 0; i < 4; i++) {
        NatureLayer d = w->layer[i];
        w->weight[i] = {.shape = {d.co, d.k * d.k * d.ci}};
        w->bias[i] = {.shape = {d.co}};
        alloc_register(alloc, &w->weight[i]);
        alloc_register(alloc, &w->bias[i]);
    }
}

static void nature_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    NatureWeights* w = (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    *a = {};
    for (int i = 0; i < 4; i++) {
        NatureLayer d = w->layer[i];
        a->patches[i] = {.shape = {B * d.oh * d.ow, d.k * d.k * d.ci}};
        a->output[i] = {.shape = {B * d.oh * d.ow, d.co}};
        alloc_register(alloc, &a->patches[i]);
        alloc_register(alloc, &a->output[i]);
    }
}

static void nature_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    NatureWeights* w = (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    nature_reg_rollout(w, activations, acts, B);
    for (int i = 0; i < 4; i++) {
        a->grad_output[i] = {.shape = {a->output[i].shape[0], a->output[i].shape[1]}};
        alloc_register(acts, &a->grad_output[i]);
        if (i > 0) {
            a->grad_patches[i] = {.shape = {a->patches[i].shape[0], a->patches[i].shape[1]}};
            alloc_register(acts, &a->grad_patches[i]);
        }
        NatureLayer d = w->layer[i];
        a->weight_grad[i] = {.shape = {d.co, d.k * d.k * d.ci}};
        a->bias_grad[i] = {.shape = {d.co}};
        alloc_register(grads, &a->weight_grad[i]);
        alloc_register(grads, &a->bias_grad[i]);
    }
}

static void* nature_create_weights(void* self) {
    Encoder* enc = (Encoder*)self;
    static_assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44, "Nature Connect4 input contract");
    assert(enc->in_dim == OBS_SIZE && enc->out_dim > 0);
    NatureWeights* w = (NatureWeights*)calloc(1, sizeof(*w));
    w->layer[0] = {36, 44, 1, 32, 8, 4, 8, 10};
    w->layer[1] = {8, 10, 32, 64, 4, 2, 3, 4};
    w->layer[2] = {3, 4, 64, 64, 3, 1, 1, 2};
    // Flatten the last spatial map before the fully connected projection.
    w->layer[3] = {1, 1, 128, enc->out_dim, 1, 1, 1, 1};
    return w;
}

static void create_connect4_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = nature_forward, .backward = nature_backward,
        .init_weights = nature_init_weights, .reg_params = nature_reg_params,
        .reg_train = nature_reg_train, .reg_rollout = nature_reg_rollout,
        .create_weights = nature_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NatureActivations),
    };
}
