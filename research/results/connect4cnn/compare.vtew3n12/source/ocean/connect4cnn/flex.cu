// Small configurable stages: strided SAME convolution, optional residual, pool.
// Uses Nature's matrix/bias/activation path and fixed-order pooling gradients.
struct FlexWeights {
    NatureWeights net;
    int type[12], pad_h[12], pad_w[12]; // 0 conv, 1 max pool, 2 average pool, 3 GAP
    bool skip[12];
};

__global__ void flex_patches(const precision_t* input, precision_t* patches,
        NatureLayer d, int B, int ph, int pw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K = d.k * d.k * d.ci;
    if (idx >= B * d.oh * d.ow * K) return;
    int q = idx % K, p = idx / K;
    int y = (p / d.ow) % d.oh * d.stride + q / (d.k * d.ci) - ph;
    int x = p % d.ow * d.stride + (q / d.ci) % d.k - pw;
    patches[idx] = y >= 0 && y < d.ih && x >= 0 && x < d.iw
        ? input[((int64_t)(p / (d.oh * d.ow)) * d.ih * d.iw + y * d.iw + x) * d.ci + q % d.ci]
        : from_float(0.0f);
}

__global__ void flex_unpatch(const precision_t* patches, precision_t* grad,
        NatureLayer d, int B, int ph, int pw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d.ih * d.iw * d.ci) return;
    int c = idx % d.ci, x = (idx / d.ci) % d.iw;
    int y = (idx / (d.ci * d.iw)) % d.ih, b = idx / (d.ci * d.iw * d.ih);
    float sum = 0;
    for (int ky = 0; ky < d.k; ky++) {
        int oy = y + ph - ky;
        if (oy < 0 || oy % d.stride || oy / d.stride >= d.oh) continue;
        for (int kx = 0; kx < d.k; kx++) {
            int ox = x + pw - kx;
            if (ox < 0 || ox % d.stride || ox / d.stride >= d.ow) continue;
            int64_t row = ((int64_t)b * d.oh + oy / d.stride) * d.ow + ox / d.stride;
            sum += to_float(patches[row * d.k * d.k * d.ci + (ky * d.k + kx) * d.ci + c]);
        }
    }
    grad[idx] = from_float(sum);
}

__global__ void flex_skip_relu(precision_t* out, const precision_t* bias,
        const precision_t* skip, int n, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_float(fmaxf(0.0f, to_float(out[i]) + to_float(bias[i % C]) + to_float(skip[i])));
}

__global__ void flex_average(const precision_t* input, precision_t* out, NatureLayer d, int B, bool global) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d.oh * d.ow * d.co) return;
    int c = i % d.co, x = (i / d.co) % d.ow, y = (i / (d.co * d.ow)) % d.oh;
    int b = i / (d.co * d.ow * d.oh);
    int top = global ? 0 : y * 2 - d.ih % 2, left = global ? 0 : x * 2 - d.iw % 2;
    int bottom = global ? d.ih : min(top + 3, d.ih), right = global ? d.iw : min(left + 3, d.iw);
    float sum = 0; int count = 0;
    for (int iy = max(top, 0); iy < bottom; iy++)
        for (int ix = max(left, 0); ix < right; ix++) {
            sum += to_float(input[((b * d.ih + iy) * d.iw + ix) * d.ci + c]);
            count++;
        }
    out[i] = from_float(sum / count);
}

__global__ void flex_average_grad(const precision_t* grad, precision_t* out, NatureLayer d, int B, bool global) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * d.ih * d.iw * d.ci) return;
    int c = i % d.ci, x = (i / d.ci) % d.iw, y = (i / (d.ci * d.iw)) % d.ih;
    int b = i / (d.ci * d.iw * d.ih);
    if (global) {
        out[i] = from_float(to_float(grad[b * d.co + c]) / (d.ih * d.iw));
        return;
    }
    float sum = 0;
    for (int ky = 0; ky < 3; ky++) {
        int oy = y + d.ih % 2 - ky;
        if (oy < 0 || oy % 2 || oy / 2 >= d.oh) continue;
        int top = oy - d.ih % 2;
        int nh = min(top + 3, d.ih) - max(top, 0);
        for (int kx = 0; kx < 3; kx++) {
            int ox = x + d.iw % 2 - kx;
            if (ox < 0 || ox % 2 || ox / 2 >= d.ow) continue;
            int left = ox - d.iw % 2;
            int nw = min(left + 3, d.iw) - max(left, 0);
            sum += to_float(grad[((b * d.oh + oy / 2) * d.ow + ox / 2) * d.co + c]) / (nh * nw);
        }
    }
    out[i] = from_float(sum);
}

static Prec flex_forward(void* weights, void* activations, Prec input, cudaStream_t stream) {
    FlexWeights* w = (FlexWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int B = a->output[0].shape[0] / (w->net.layer[0].oh * w->net.layer[0].ow);
    assert(numel(input.shape) == (int64_t)B * OBS_SIZE);
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        int n = B * d.oh * d.ow * d.co;
        if (w->type[i] == 1) {
            c4_cnn::imp_pool<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data, a->output[i].data,
                a->winner[i].data, B, d.ih, d.iw, d.ci);
        } else if (w->type[i]) {
            flex_average<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data, a->output[i].data, d, B, w->type[i] == 3);
        } else {
            flex_patches<<<grid_size(B * d.oh * d.ow * d.k * d.k * d.ci), BLOCK_SIZE, 0, stream>>>(
                input.data, a->patches[i].data, d, B, w->pad_h[i], w->pad_w[i]);
            puf_mm(&a->patches[i], &w->net.weight[i], &a->output[i], stream);
            if (w->skip[i]) flex_skip_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->output[i].data,
                w->net.bias[i].data, input.data, n, d.co);
            else nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->output[i].data, w->net.bias[i].data, n, d.co);
        }
        input = a->output[i];
    }
    return input;
}

static void flex_backward(void* weights, void* activations, Prec grad, cudaStream_t stream) {
    FlexWeights* w = (FlexWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int last = w->net.count - 1, B = a->output[last].shape[0];
    puf_copy(&a->grad_output[last], &grad, stream);
    for (int i = last; i >= 0; i--) {
        NatureLayer d = w->net.layer[i];
        int rows = B * d.oh * d.ow;
        if (w->type[i] == 1) {
            c4_cnn::imp_pool_grad<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->winner[i].data, a->grad_output[i-1].data, B, d.ih, d.iw, d.ci);
        } else if (w->type[i]) {
            flex_average_grad<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->grad_output[i-1].data, d, B, w->type[i] == 3);
        } else {
            nature_relu_backward<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->output[i].data, rows * d.co);
            puf_mm_tn(&a->grad_output[i], &a->patches[i], &a->weight_grad[i], stream);
            nature_bias_backward<<<d.co, 256, 0, stream>>>(a->grad_output[i].data, a->bias_grad[i].data, rows, d.co);
            if (i) {
                puf_mm_nn(&a->grad_output[i], &w->net.weight[i], &a->grad_patches[i], stream);
                flex_unpatch<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(a->grad_patches[i].data,
                    a->grad_output[i-1].data, d, B, w->pad_h[i], w->pad_w[i]);
                if (w->skip[i]) c4_cnn::imp_add<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
                    a->grad_output[i-1].data, a->grad_output[i].data, rows * d.co);
            }
        }
    }
}

static void flex_reg_params(void* weights, Allocator* alloc) {
    FlexWeights* w = (FlexWeights*)weights;
    for (int i = 0; i < w->net.count; i++) if (!w->type[i]) {
        NatureLayer d = w->net.layer[i];
        w->net.weight[i] = {.shape = {d.co, d.k * d.k * d.ci}};
        w->net.bias[i] = {.shape = {d.co}};
        alloc_register(alloc, &w->net.weight[i]); alloc_register(alloc, &w->net.bias[i]);
    }
}

static void flex_init(void* weights, ulong* seed, cudaStream_t stream) {
    FlexWeights* w = (FlexWeights*)weights;
    for (int i = 0; i < w->net.count; i++) if (!w->type[i]) {
        puf_kaiming_init(&w->net.weight[i], sqrtf(2.0f), (*seed)++, stream);
        cudaMemsetAsync(w->net.bias[i].data, 0, numel(w->net.bias[i].shape) * sizeof(precision_t), stream);
    }
}

static void flex_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    FlexWeights* w = (FlexWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    *a = {};
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        a->output[i] = {.shape = {B * d.oh * d.ow, d.co}};
        alloc_register(alloc, &a->output[i]);
        if (!w->type[i]) {
            a->patches[i] = {.shape = {B * d.oh * d.ow, d.k * d.k * d.ci}};
            alloc_register(alloc, &a->patches[i]);
        }
    }
}

static void flex_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    FlexWeights* w = (FlexWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    flex_reg_rollout(weights, activations, acts, B);
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        a->grad_output[i] = {.shape = {B * d.oh * d.ow, d.co}};
        alloc_register(acts, &a->grad_output[i]);
        if (w->type[i] == 1) {
            a->winner[i] = {.shape = {B * d.oh * d.ow * d.co}};
            alloc_register(acts, &a->winner[i]);
        }
        if (w->type[i]) continue;
        if (i) {
            a->grad_patches[i] = {.shape = {B * d.oh * d.ow, d.k * d.k * d.ci}};
            alloc_register(acts, &a->grad_patches[i]);
        }
        a->weight_grad[i] = {.shape = {d.co, d.k * d.k * d.ci}};
        a->bias_grad[i] = {.shape = {d.co}};
        alloc_register(grads, &a->weight_grad[i]); alloc_register(grads, &a->bias_grad[i]);
    }
}

static void flex_conv(FlexWeights* w, int H, int W, int ci, int co, int k, int s, bool skip = false) {
    int i = w->net.count++;
    assert(i < 12);
    int oh = (H + s - 1) / s, ow = (W + s - 1) / s;
    w->net.layer[i] = {H, W, ci, co, k, s, oh, ow};
    w->pad_h[i] = max((oh - 1) * s + k - H, 0) / 2;
    w->pad_w[i] = max((ow - 1) * s + k - W, 0) / 2;
    w->skip[i] = skip;
}

static int flex_option(Dict* policy, const char* key, int lo, int hi) {
    double v = dict_get(policy, key);
    assert(v >= lo && v <= hi && v == (int)v);
    return (int)v;
}

static void* flex_create_weights(void* self) {
    Encoder* enc = (Encoder*)self;
    assert(enc->in_dim == OBS_SIZE && enc->out_dim >= 8 && enc->out_dim % 8 == 0);
    FlexWeights* w = (FlexWeights*)calloc(1, sizeof(*w));
    int depth = flex_option(enc->config, "cnn_depth", 1, 3);
    int projection = flex_option(enc->config, "cnn_projection", 16, 128);
    assert((projection & (projection - 1)) == 0);
    int H = 36, W = 44, ci = 1;
    for (int s = 1; s <= depth; s++) {
        char key[64];
        snprintf(key, sizeof(key), "cnn_channels_%d", s);
        int co = flex_option(enc->config, key, 8, 32);
        assert((co & (co - 1)) == 0);
        snprintf(key, sizeof(key), "cnn_kernel_%d", s);
        int k = flex_option(enc->config, key, 1, s == 1 ? 8 : 5);
        snprintf(key, sizeof(key), "cnn_stride_%d", s);
        int stride = flex_option(enc->config, key, s == 1 ? 4 : 1, s == 1 ? 8 : 4);
        assert((stride & (stride - 1)) == 0);
        flex_conv(w, H, W, ci, co, k, stride);
        H = (H + stride - 1) / stride; W = (W + stride - 1) / stride; ci = co;
        snprintf(key, sizeof(key), "cnn_residual_%d", s);
        if (flex_option(enc->config, key, 0, 1)) flex_conv(w, H, W, co, co, 3, 1, true);
        snprintf(key, sizeof(key), "cnn_pool_%d", s);
        int pool = flex_option(enc->config, key, 0, 2);
        if (pool) {
            int i = w->net.count++;
            w->type[i] = pool;
            w->net.layer[i] = {H, W, co, co, 3, 2, (H + 1) / 2, (W + 1) / 2};
            H = (H + 1) / 2; W = (W + 1) / 2;
        }
    }
    if (flex_option(enc->config, "cnn_global_pool", 0, 1)) {
        int i = w->net.count++;
        w->type[i] = 3; w->net.layer[i] = {H, W, ci, ci, 1, 1, 1, 1}; H = W = 1;
    }
    flex_conv(w, 1, 1, H * W * ci, projection, 1, 1);
    if (projection != enc->out_dim) flex_conv(w, 1, 1, projection, enc->out_dim, 1, 1);
    return w;
}

static void create_flex_encoder(Encoder* enc, Dict* policy) {
    *enc = Encoder{.forward = flex_forward, .backward = flex_backward,
        .init_weights = flex_init, .reg_params = flex_reg_params,
        .reg_train = flex_reg_train, .reg_rollout = flex_reg_rollout,
        .create_weights = flex_create_weights, .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NatureActivations), .config = policy};
}
