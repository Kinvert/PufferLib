// IMPALA: [16,32,32], SAME Conv3 + SAME MaxPool3/s2 + two preactivation
// residual blocks per stage. Impoola changes only flatten to GAP after ReLU.
// NHWC buffers, fixed reduction order, shared convolution scratch, no atomics.

struct ImpalaWeights {
    Prec weight[16], bias[16];
    int hidden, readout;
    bool gap;
};

struct ImpalaActivations {
    Prec conv[15], pool[3], input, readout, output;
    Int winner[3];
    Prec patches, grad_patches, grad[2], skip_grad;
    Prec weight_grad[16], bias_grad[16];
    int batch;
};

static constexpr int IMP_H[3] = {36, 18, 9};
static constexpr int IMP_W[3] = {44, 22, 11};
static constexpr int IMP_C[3] = {16, 32, 32};

__global__ void imp_patches(const precision_t* input, precision_t* patches,
        int B, int H, int W, int C, bool relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W * 9 * C) return;
    int c = idx % C, k = (idx / C) % 9, p = idx / (9 * C);
    int x = p % W + k % 3 - 1, y = (p / W) % H + k / 3 - 1;
    float v = 0;
    if (x >= 0 && x < W && y >= 0 && y < H)
        v = to_float(input[((int64_t)(p / (H * W)) * H * W + y * W + x) * C + c]);
    patches[idx] = from_float(relu ? fmaxf(v, 0.0f) : v);
}

__global__ void imp_unpatch(const precision_t* patches, const precision_t* input,
        precision_t* grad, int B, int H, int W, int C, bool relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W * C) return;
    int c = idx % C, x = (idx / C) % W, y = (idx / (C * W)) % H;
    int b = idx / (C * W * H);
    float sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        int oy = y + 1 - ky;
        if (oy < 0 || oy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ox = x + 1 - kx;
            if (ox < 0 || ox >= W) continue;
            int64_t row = ((int64_t)b * H + oy) * W + ox;
            sum += to_float(patches[row * 9 * C + (ky * 3 + kx) * C + c]);
        }
    }
    grad[idx] = from_float(relu && to_float(input[idx]) <= 0 ? 0.0f : sum);
}

__global__ void imp_bias(precision_t* output, const precision_t* bias, int n, int C, bool relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = to_float(output[idx]) + to_float(bias[idx % C]);
    output[idx] = from_float(relu ? fmaxf(v, 0.0f) : v);
}

__global__ void imp_bias_grad(const precision_t* grad, precision_t* bias, int rows, int C) {
    __shared__ float sums[256];
    float v = 0;
    for (int r = threadIdx.x; r < rows; r += 256) v += to_float(grad[(int64_t)r * C + blockIdx.x]);
    sums[threadIdx.x] = v;
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (threadIdx.x < k) sums[threadIdx.x] += sums[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias[blockIdx.x] = from_float(sums[0]);
}

__global__ void imp_add(precision_t* dst, const precision_t* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
}

__global__ void imp_pool(const precision_t* input, precision_t* output, int* winner,
        int B, int H, int W, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int PH = (H + 1) / 2, PW = (W + 1) / 2;
    if (idx >= B * PH * PW * C) return;
    int c = idx % C, x = (idx / C) % PW, y = (idx / (C * PW)) % PH;
    int b = idx / (C * PW * PH);
    float best = -INFINITY;
    int at = -1;
    // TensorFlow SAME: odd inputs pad both edges; even inputs pad trailing edge.
    for (int ky = 0; ky < 3; ky++) {
        int iy = y * 2 + ky - H % 2;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ix = x * 2 + kx - W % 2;
            if (ix < 0 || ix >= W) continue;
            int pos = ((b * H + iy) * W + ix) * C + c;
            float v = to_float(input[pos]);
            if (at == -1 || v > best) { best = v; at = pos; }
        }
    }
    output[idx] = from_float(best);
    if (winner) winner[idx] = at;
}

__global__ void imp_pool_grad(const precision_t* grad, const int* winner, precision_t* output,
        int B, int H, int W, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W * C) return;
    int c = idx % C, x = (idx / C) % W, y = (idx / (C * W)) % H;
    int b = idx / (C * W * H), PH = (H + 1) / 2, PW = (W + 1) / 2;
    float sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        int oy = y + H % 2 - ky;
        if (oy < 0 || oy % 2 || oy / 2 >= PH) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ox = x + W % 2 - kx;
            if (ox < 0 || ox % 2 || ox / 2 >= PW) continue;
            int pos = ((b * PH + oy / 2) * PW + ox / 2) * C + c;
            if (winner[pos] == idx) sum += to_float(grad[pos]);
        }
    }
    output[idx] = from_float(sum);
}

__global__ void imp_readout(const precision_t* input, precision_t* output, int B, bool gap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D = gap ? 32 : 960;
    if (idx >= B * D) return;
    float sum = 0;
    if (gap) {
        int b = idx / 32, c = idx % 32;
        for (int p = 0; p < 30; p++) sum += fmaxf(0.0f, to_float(input[(b * 30 + p) * 32 + c]));
        sum /= 30;
    } else sum = fmaxf(0.0f, to_float(input[idx]));
    output[idx] = from_float(sum);
}

__global__ void imp_readout_grad(const precision_t* input, const precision_t* grad,
        precision_t* output, int B, bool gap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * 960) return;
    float v = gap ? to_float(grad[(idx / 960) * 32 + idx % 32]) / 30 : to_float(grad[idx]);
    output[idx] = from_float(to_float(input[idx]) > 0 ? v : 0.0f);
}

__global__ void imp_relu_grad(precision_t* grad, const precision_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && to_float(output[idx]) <= 0) grad[idx] = from_float(0.0f);
}

static void imp_conv_forward(ImpalaWeights* w, ImpalaActivations* a, int i,
        Prec input, int H, int W, bool relu, cudaStream_t stream) {
    int ci = w->weight[i].shape[1] / 9, co = w->weight[i].shape[0];
    int rows = a->batch * H * W;
    Prec patches = {.data = a->patches.data, .shape = {rows, 9 * ci}};
    imp_patches<<<grid_size(rows * 9 * ci), BLOCK_SIZE, 0, stream>>>(input.data, patches.data, a->batch, H, W, ci, relu);
    puf_mm(&patches, &w->weight[i], &a->conv[i], stream);
    imp_bias<<<grid_size(rows * co), BLOCK_SIZE, 0, stream>>>(a->conv[i].data, w->bias[i].data, rows * co, co, false);
}

static void imp_conv_backward(ImpalaWeights* w, ImpalaActivations* a, int i,
        Prec input, int H, int W, bool relu, int gin, int gout, cudaStream_t stream) {
    int ci = w->weight[i].shape[1] / 9, co = w->weight[i].shape[0];
    int rows = a->batch * H * W;
    Prec patches = {.data = a->patches.data, .shape = {rows, 9 * ci}};
    Prec grad = {.data = a->grad[gin].data, .shape = {rows, co}};
    imp_patches<<<grid_size(rows * 9 * ci), BLOCK_SIZE, 0, stream>>>(input.data, patches.data, a->batch, H, W, ci, relu);
    puf_mm_tn(&grad, &patches, &a->weight_grad[i], stream);
    imp_bias_grad<<<co, 256, 0, stream>>>(grad.data, a->bias_grad[i].data, rows, co);
    if (gout >= 0) {
        Prec gp = {.data = a->grad_patches.data, .shape = {rows, 9 * ci}};
        puf_mm_nn(&grad, &w->weight[i], &gp, stream);
        imp_unpatch<<<grid_size(rows * ci), BLOCK_SIZE, 0, stream>>>(gp.data, input.data,
            a->grad[gout].data, a->batch, H, W, ci, relu);
    }
}

static Prec imp_forward(void* weights, void* activations, Prec input, cudaStream_t stream) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    ImpalaActivations* a = (ImpalaActivations*)activations;
    assert(numel(input.shape) == (int64_t)a->batch * OBS_SIZE);
    if (a->input.data) puf_copy(&a->input, &input, stream);
    for (int s = 0; s < 3; s++) {
        int H = IMP_H[s], W = IMP_W[s], C = IMP_C[s], i = s * 5;
        imp_conv_forward(w, a, i, input, H, W, false, stream);
        int PH = (H + 1) / 2, PW = (W + 1) / 2, n = a->batch * PH * PW * C;
        imp_pool<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->conv[i].data, a->pool[s].data,
            a->winner[s].data, a->batch, H, W, C);
        input = a->pool[s];
        for (int r = 0; r < 2; r++) {
            int j = i + 1 + 2 * r;
            imp_conv_forward(w, a, j, input, PH, PW, true, stream);
            imp_conv_forward(w, a, j + 1, a->conv[j], PH, PW, true, stream);
            imp_add<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->conv[j + 1].data, input.data, n);
            input = a->conv[j + 1];
        }
    }
    imp_readout<<<grid_size(a->batch * w->readout), BLOCK_SIZE, 0, stream>>>(input.data, a->readout.data, a->batch, w->gap);
    puf_mm(&a->readout, &w->weight[15], &a->output, stream);
    int n = a->batch * w->hidden;
    imp_bias<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->output.data, w->bias[15].data, n, w->hidden, true);
    return a->output;
}

static void imp_backward(void* weights, void* activations, Prec upstream, cudaStream_t stream) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    ImpalaActivations* a = (ImpalaActivations*)activations;
    Prec g = {.data = a->grad[0].data, .shape = {a->batch, w->hidden}};
    assert(numel(upstream.shape) == numel(g.shape));
    puf_copy(&g, &upstream, stream);
    imp_relu_grad<<<grid_size(a->batch * w->hidden), BLOCK_SIZE, 0, stream>>>(g.data, a->output.data, a->batch * w->hidden);
    puf_mm_tn(&g, &a->readout, &a->weight_grad[15], stream);
    imp_bias_grad<<<w->hidden, 256, 0, stream>>>(g.data, a->bias_grad[15].data, a->batch, w->hidden);
    Prec gr = {.data = a->grad[1].data, .shape = {a->batch, w->readout}};
    puf_mm_nn(&g, &w->weight[15], &gr, stream);
    imp_readout_grad<<<grid_size(a->batch * 960), BLOCK_SIZE, 0, stream>>>(a->conv[14].data, gr.data, g.data, a->batch, w->gap);
    for (int s = 2; s >= 0; s--) {
        int H = IMP_H[s], W = IMP_W[s], C = IMP_C[s], i = s * 5;
        int PH = (H + 1) / 2, PW = (W + 1) / 2, n = a->batch * PH * PW * C;
        Prec skip = {.data = a->skip_grad.data, .shape = {n}};
        Prec gin = {.data = a->grad[0].data, .shape = {n}};
        for (int r = 1; r >= 0; r--) {
            int j = i + 1 + 2 * r;
            puf_copy(&skip, &gin, stream);
            imp_conv_backward(w, a, j + 1, a->conv[j], PH, PW, true, 0, 1, stream);
            Prec x = r ? a->conv[i + 2] : a->pool[s];
            imp_conv_backward(w, a, j, x, PH, PW, true, 1, 0, stream);
            imp_add<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(gin.data, skip.data, n);
        }
        imp_pool_grad<<<grid_size(a->batch * H * W * C), BLOCK_SIZE, 0, stream>>>(gin.data,
            a->winner[s].data, a->grad[1].data, a->batch, H, W, C);
        Prec x = s ? a->conv[i - 1] : a->input;
        imp_conv_backward(w, a, i, x, H, W, false, 1, s ? 0 : -1, stream);
    }
}

static void imp_reg_params(void* weights, Allocator* alloc) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    for (int s = 0; s < 3; s++) {
        for (int j = 0; j < 5; j++) {
            int i = s * 5 + j, ci = j ? IMP_C[s] : (s ? IMP_C[s - 1] : 1);
            w->weight[i] = {.shape = {IMP_C[s], 9 * ci}};
            w->bias[i] = {.shape = {IMP_C[s]}};
        }
    }
    w->weight[15] = {.shape = {w->hidden, w->readout}};
    w->bias[15] = {.shape = {w->hidden}};
    for (int i = 0; i < 16; i++) {
        alloc_register(alloc, &w->weight[i]); alloc_register(alloc, &w->bias[i]);
    }
}

static void imp_init_weights(void* weights, ulong* seed, cudaStream_t stream) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    for (int i = 0; i < 16; i++) {
        puf_kaiming_init(&w->weight[i], sqrtf(2.0f), (*seed)++, stream);
        cudaMemsetAsync(w->bias[i].data, 0, numel(w->bias[i].shape) * sizeof(precision_t), stream);
    }
}

static void imp_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    ImpalaActivations* a = (ImpalaActivations*)activations;
    *a = {}; a->batch = B;
    int max_patch = 0;
    for (int s = 0; s < 3; s++) {
        int H = IMP_H[s], W = IMP_W[s], C = IMP_C[s];
        for (int j = 0; j < 5; j++) {
            int h = j ? (H + 1) / 2 : H, width = j ? (W + 1) / 2 : W;
            int ci = j ? C : (s ? IMP_C[s - 1] : 1);
            a->conv[s * 5 + j] = {.shape = {B * h * width, C}};
            alloc_register(alloc, &a->conv[s * 5 + j]);
            max_patch = max(max_patch, h * width * 9 * ci);
        }
        a->pool[s] = {.shape = {B * ((H + 1) / 2) * ((W + 1) / 2), C}};
        alloc_register(alloc, &a->pool[s]);
    }
    a->patches = {.shape = {(int64_t)B * max_patch}};
    a->readout = {.shape = {B, w->readout}};
    a->output = {.shape = {B, w->hidden}};
    alloc_register(alloc, &a->patches); alloc_register(alloc, &a->readout); alloc_register(alloc, &a->output);
}

static void imp_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    ImpalaWeights* w = (ImpalaWeights*)weights;
    ImpalaActivations* a = (ImpalaActivations*)activations;
    imp_reg_rollout(w, activations, acts, B);
    a->input = {.shape = {B, OBS_SIZE}};
    alloc_register(acts, &a->input);
    a->grad_patches = {.shape = {numel(a->patches.shape)}};
    alloc_register(acts, &a->grad_patches);
    for (int i = 0; i < 2; i++) {
        a->grad[i] = {.shape = {(int64_t)B * max(36 * 44 * 16, w->hidden)}};
        alloc_register(acts, &a->grad[i]);
    }
    a->skip_grad = {.shape = {(int64_t)B * 18 * 22 * 16}};
    alloc_register(acts, &a->skip_grad);
    for (int s = 0; s < 3; s++) {
        a->winner[s] = {.shape = {numel(a->pool[s].shape)}};
        alloc_register(acts, &a->winner[s]);
    }
    for (int i = 0; i < 16; i++) {
        a->weight_grad[i] = {.shape = {w->weight[i].shape[0], w->weight[i].shape[1]}};
        a->bias_grad[i] = {.shape = {w->bias[i].shape[0]}};
        alloc_register(grads, &a->weight_grad[i]); alloc_register(grads, &a->bias_grad[i]);
    }
}

static void* imp_create_weights(void* self) {
    Encoder* enc = (Encoder*)self;
    static_assert(OBS_CHANNELS == 1 && OBS_HEIGHT == 36 && OBS_WIDTH == 44, "IMPALA Connect4 input contract");
    assert(enc->in_dim == OBS_SIZE && enc->out_dim > 0);
    ImpalaWeights* w = (ImpalaWeights*)calloc(1, sizeof(*w));
    w->hidden = enc->out_dim;
#ifdef C4_IMPOOLA_CNN
    w->gap = true;
#endif
    w->readout = w->gap ? 32 : 960;
    return w;
}

static void create_connect4_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = imp_forward, .backward = imp_backward,
        .init_weights = imp_init_weights, .reg_params = imp_reg_params,
        .reg_train = imp_reg_train, .reg_rollout = imp_reg_rollout,
        .create_weights = imp_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(ImpalaActivations),
    };
}
