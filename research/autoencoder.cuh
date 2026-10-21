// Training-only decoder. Encoder and ordinary PufferLib code are unchanged.
struct AEDecoder {
    NatureWeights net;
    int source_h[4], source_w[4];
    bool spatial;
    int latent_values;
};

struct AEActivations {
    NatureActivations net;
    Prec resized[4], grad_resized[4], latent, grad_latent;
};

__global__ void ae_resize(const float* x, float* y, int B, int H, int W, int OH, int OW, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * OH * OW * C) return;
    int c = i % C, xx = i / C % OW, yy = i / (C * OW) % OH, b = i / (C * OW * OH);
    y[i] = x[((b * H + yy * H / OH) * W + xx * W / OW) * C + c];
}

__global__ void ae_resize_backward(const float* dy, float* dx, int B, int H, int W, int OH, int OW, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * W * C) return;
    int c = i % C, x = i / C % W, y = i / (C * W) % H, b = i / (C * W * H);
    float sum = 0;
    for (int yy = (y * OH + H - 1) / H; yy < ((y + 1) * OH + H - 1) / H; yy++)
        for (int xx = (x * OW + W - 1) / W; xx < ((x + 1) * OW + W - 1) / W; xx++)
            sum += dy[((b * OH + yy) * OW + xx) * C + c];
    dx[i] = sum;
}

__global__ void ae_bias(float* x, const float* bias, int n, int C, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + bias[i % C];
        x[i] = relu ? fmaxf(v, 0) : v;
    }
}

static AEDecoder ae_decoder(int H, int W) {
    AEDecoder d = {};
    d.latent_values = 128;
    d.net.count = 4;
    d.net.layer[0] = {1, 1, 128, 256, 1, 1, 1, 1};
    int heights[] = {1, (H + 15) / 16, (H + 3) / 4, H};
    int widths[] = {1, (W + 15) / 16, (W + 3) / 4, W};
    int channels[] = {16, 8, 8, 3};
    for (int i = 1; i < 4; i++) {
        d.source_h[i] = i == 1 ? 4 : heights[i - 1];
        d.source_w[i] = i == 1 ? 4 : widths[i - 1];
        d.net.layer[i] = {heights[i], widths[i], channels[i - 1], channels[i], 3, 1, heights[i], widths[i]};
    }
    return d;
}

#ifdef CNN_SPATIAL_RESEARCH
static AEDecoder ae_spatial_decoder(int H, int W, int stride, int channels) {
    AEDecoder d = {}; d.spatial = true; d.net.count = 3;
    int hs[] = {(H + 3) / 4, (H + 1) / 2, H};
    int ws[] = {(W + 3) / 4, (W + 1) / 2, W};
    int cs[] = {channels, 16, 8, 3};
    int sh = (H + stride - 1) / stride, sw = (W + stride - 1) / stride;
    d.latent_values = sh * sw * channels;
    for (int i = 0; i < 3; i++) {
        d.source_h[i] = i ? hs[i - 1] : sh; d.source_w[i] = i ? ws[i - 1] : sw;
        d.net.layer[i] = {hs[i], ws[i], cs[i], cs[i + 1], 3, 1, hs[i], ws[i]};
    }
    return d;
}
#endif

static void ae_register(AEDecoder* d, AEActivations* a, Allocator* acts, Allocator* grads, int B) {
    a->grad_latent = {.shape = {B, d->latent_values}};
    alloc_register(acts, &a->grad_latent);
    for (int i = 0; i < d->net.count; i++) {
        NatureLayer l = d->net.layer[i];
        a->net.output[i] = {.shape = {B * l.oh * l.ow, l.co}};
        a->net.grad_output[i] = {.shape = {B * l.oh * l.ow, l.co}};
        alloc_register(acts, &a->net.output[i]);
        alloc_register(acts, &a->net.grad_output[i]);
        if (i || d->spatial) {
            a->resized[i] = {.shape = {B, l.ih * l.iw * l.ci}};
            a->grad_resized[i] = {.shape = {B, l.ih * l.iw * l.ci}};
            a->net.patches[i] = {.shape = {B * l.oh * l.ow, l.k * l.k * l.ci}};
            a->net.grad_patches[i] = {.shape = {B * l.oh * l.ow, l.k * l.k * l.ci}};
            alloc_register(acts, &a->resized[i]);
            alloc_register(acts, &a->grad_resized[i]);
            alloc_register(acts, &a->net.patches[i]);
            alloc_register(acts, &a->net.grad_patches[i]);
        }
        a->net.weight_grad[i] = {.shape = {l.co, l.k * l.k * l.ci}};
        a->net.bias_grad[i] = {.shape = {l.co}};
        alloc_register(grads, &a->net.weight_grad[i]);
        alloc_register(grads, &a->net.bias_grad[i]);
    }
}

static Prec ae_forward(AEDecoder* d, AEActivations* a, Prec latent, cudaStream_t stream) {
    int B = latent.shape[0];
    a->latent = latent;
    Prec x = latent;
    for (int i = 0; i < d->net.count; i++) {
        NatureLayer l = d->net.layer[i];
        if (i || d->spatial) {
            ae_resize<<<grid_size(B * l.ih * l.iw * l.ci), BLOCK_SIZE, 0, stream>>>(
                x.data, a->resized[i].data, B, d->source_h[i], d->source_w[i], l.ih, l.iw, l.ci);
            flex_patches<<<grid_size(numel(a->net.patches[i].shape)), BLOCK_SIZE, 0, stream>>>(
                a->resized[i].data, a->net.patches[i].data, l, B, 1, 1);
            x = a->net.patches[i];
        }
        puf_mm(&x, &d->net.weight[i], &a->net.output[i], stream);
        ae_bias<<<grid_size(numel(a->net.output[i].shape)), BLOCK_SIZE, 0, stream>>>(
            a->net.output[i].data, d->net.bias[i].data, numel(a->net.output[i].shape), l.co, i != d->net.count - 1);
        x = a->net.output[i];
    }
    return x;
}

static Prec ae_backward(AEDecoder* d, AEActivations* a, cudaStream_t stream) {
    int B = a->latent.shape[0];
    for (int i = d->net.count - 1; i >= 0; i--) {
        NatureLayer l = d->net.layer[i];
        Prec* dy = &a->net.grad_output[i];
        if (i != d->net.count - 1) nature_relu_backward<<<grid_size(numel(dy->shape)), BLOCK_SIZE, 0, stream>>>(
            dy->data, a->net.output[i].data, numel(dy->shape));
        Prec x = i || d->spatial ? a->net.patches[i] : a->latent;
        puf_mm_tn(dy, &x, &a->net.weight_grad[i], stream);
        nature_bias_backward<<<l.co, 256, 0, stream>>>(dy->data, a->net.bias_grad[i].data, B * l.oh * l.ow, l.co);
        if (!i && !d->spatial) {
            puf_mm_nn(dy, &d->net.weight[i], &a->grad_latent, stream);
        } else {
            puf_mm_nn(dy, &d->net.weight[i], &a->net.grad_patches[i], stream);
            flex_unpatch<<<grid_size(numel(a->resized[i].shape)), BLOCK_SIZE, 0, stream>>>(
                a->net.grad_patches[i].data, a->grad_resized[i].data, l, B, 1, 1);
            int n = B * d->source_h[i] * d->source_w[i] * l.ci;
            Prec target = i ? a->net.grad_output[i - 1] : a->grad_latent;
            ae_resize_backward<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->grad_resized[i].data,
                target.data, B, d->source_h[i], d->source_w[i], l.ih, l.iw, l.ci);
        }
    }
    return a->grad_latent;
}

__global__ void ae_loss(const float* y, const float* target, float* dy, float* parts,
        int n, int active, int H, int W, float edge_weight, const float* pixel_weights = NULL) {
    __shared__ float sums[512];
    float raw = 0, weighted = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float grad = 0;
        if (i < active) {
            int x = i / 3 % W, yy = i / (3 * W) % H;
            bool edge = (x && fabsf(target[i] - target[i - 3]) > .05f)
                || (yy && fabsf(target[i] - target[i - 3 * W]) > .05f);
            float error = y[i] - target[i];
            float weight = pixel_weights ? pixel_weights[i / 3] : 1 + edge_weight * edge;
            raw += error * error;
            weighted += weight * error * error;
            grad = 2 * weight * error / active;
        }
        dy[i] = grad;
    }
    sums[threadIdx.x] = raw;
    sums[256 + threadIdx.x] = weighted;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) {
            sums[threadIdx.x] += sums[threadIdx.x + step];
            sums[256 + threadIdx.x] += sums[256 + threadIdx.x + step];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        parts[blockIdx.x] = sums[0] / active;
        parts[256 + blockIdx.x] = sums[256] / active;
    }
}

__global__ void ae_adam(float* w, const float* g, float* m, float* v, const float* sum_sq,
        int n, float lr, float clip, float correction1, float correction2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float scale = fminf(1, clip / (sqrtf(*sum_sq) + 1e-6f));
    float grad = g[i] * scale;
    m[i] = .9f * m[i] + .1f * grad;
    v[i] = .999f * v[i] + .001f * grad * grad;
    w[i] -= lr * (m[i] / correction1) / (sqrtf(v[i] / correction2) + 1e-8f);
}

static void ae_bind(Allocator* view, Allocator* owner) {
    ffc_require(view->total_bytes == owner->total_bytes && view->num_regs == owner->num_regs, "shared parameter layout");
    long offset = 0;
    for (int i = 0; i < view->num_regs; i++) {
        ffc_require(numel(view->regs[i].shape) == numel(owner->regs[i].shape), "shared parameter shape");
        offset = (offset + 15) & ~15;
        *view->regs[i].data_ptr = (char*)owner->mem + offset;
        offset += numel(view->regs[i].shape) * view->regs[i].elem_size;
    }
}
