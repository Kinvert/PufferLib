// Research-only full-frame RGB encoder. Raw observations never change shape.
struct FFCWeights {
    NatureWeights net;
    int model, ph[4], pw[4], height, width;
    bool input_grad; // Test harness requests d(image); the learner does not consume it.
    char* initial;
};

struct FFCActivations {
    NatureActivations net;
    Prec reduced, spatial, grad_spatial, grad_reduced, grad_input;
};

static void ffc_import(FFCWeights* w, const char* directory, cudaStream_t stream);

// Fixed row-major box means. Edge blocks divide by their actual area.
__global__ void ffc_area(const precision_t* x, precision_t* y, int B, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int OH = (H + 3) / 4, OW = (W + 3) / 4;
    if (i >= B * OH * OW * 3) return;
    int c = i % 3, ox = i / 3 % OW, oy = i / (3 * OW) % OH;
    int b = i / (3 * OW * OH), endy = min(oy * 4 + 4, H), endx = min(ox * 4 + 4, W);
    float sum = 0;
    for (int yy = oy * 4; yy < endy; yy++)
        for (int xx = ox * 4; xx < endx; xx++)
            sum += to_float(x[((b * H + yy) * W + xx) * 3 + c]);
    y[i] = from_float(sum / ((endy - oy * 4) * (endx - ox * 4)));
}

__global__ void ffc_area_backward(const precision_t* dy, precision_t* dx, int B, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * W * 3) return;
    int c = i % 3, x = i / 3 % W, y = i / (3 * W) % H, b = i / (3 * W * H);
    int OH = (H + 3) / 4, OW = (W + 3) / 4;
    int count = min(4, H - y / 4 * 4) * min(4, W - x / 4 * 4);
    dx[i] = from_float(to_float(dy[((b * OH + y / 4) * OW + x / 4) * 3 + c]) / count);
}

__global__ void ffc_adaptive(const precision_t* x, precision_t* y, int B, int H, int W, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * 16 * C) return;
    int c = i % C, ox = i / C % 4, oy = i / (4 * C) % 4, b = i / (16 * C);
    int top = oy * H / 4, endy = ((oy + 1) * H + 3) / 4;
    int left = ox * W / 4, endx = ((ox + 1) * W + 3) / 4;
    float sum = 0;
    for (int yy = top; yy < endy; yy++)
        for (int xx = left; xx < endx; xx++)
            sum += to_float(x[((b * H + yy) * W + xx) * C + c]);
    y[i] = from_float(sum / ((endy - top) * (endx - left)));
}

__global__ void ffc_adaptive_backward(const precision_t* dy, precision_t* dx,
        int B, int H, int W, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * W * C) return;
    int c = i % C, x = i / C % W, y = i / (C * W) % H, b = i / (C * W * H);
    float sum = 0;
    for (int oy = 0; oy < 4; oy++) {
        int top = oy * H / 4, endy = ((oy + 1) * H + 3) / 4;
        if (y < top || y >= endy) continue;
        for (int ox = 0; ox < 4; ox++) {
            int left = ox * W / 4, endx = ((ox + 1) * W + 3) / 4;
            if (x >= left && x < endx)
                sum += to_float(dy[((b * 4 + oy) * 4 + ox) * C + c]) / ((endy - top) * (endx - left));
        }
    }
    dx[i] = from_float(sum);
}

static Prec ffc_forward(void* weights, void* activations, Prec input, cudaStream_t stream) {
    FFCWeights* w = (FFCWeights*)weights;
    FFCActivations* a = (FFCActivations*)activations;
    NatureLayer last = w->net.layer[w->net.count - 1];
    int B = a->net.output[w->net.count - 1].shape[0] / (last.oh * last.ow);
    assert(numel(input.shape) == (long)B * w->height * w->width * w->net.layer[0].ci);
    if (w->model) {
        ffc_area<<<grid_size(numel(a->reduced.shape)), BLOCK_SIZE, 0, stream>>>(
            input.data, a->reduced.data, B, w->height, w->width);
        input = a->reduced;
    }
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        if (i == 3 && w->model != 2) {
            NatureLayer prev = w->net.layer[2];
            ffc_adaptive<<<grid_size(B * 256), BLOCK_SIZE, 0, stream>>>(
                input.data, a->spatial.data, B, prev.oh, prev.ow, 16);
            input = a->spatial;
        }
        if (i == 3) {
            puf_mm(&input, &w->net.weight[i], &a->net.output[i], stream);
        } else {
            int n = B * d.oh * d.ow * d.k * d.k * d.ci;
            flex_patches<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                input.data, a->net.patches[i].data, d, B, w->ph[i], w->pw[i]);
            puf_mm(&a->net.patches[i], &w->net.weight[i], &a->net.output[i], stream);
        }
        int n = B * d.oh * d.ow * d.co;
        nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->net.output[i].data, w->net.bias[i].data, n, d.co);
        input = a->net.output[i];
        if (i == 2 && w->model == 2) input.shape[0] = B, input.shape[1] = d.oh * d.ow * d.co;
    }
    input.shape[0] = B;
    input.shape[1] = last.oh * last.ow * last.co;
    return input;
}

static void ffc_backward(void* weights, void* activations, Prec grad, cudaStream_t stream) {
    FFCWeights* w = (FFCWeights*)weights;
    FFCActivations* a = (FFCActivations*)activations;
    NatureActivations* n = &a->net;
    int end = w->net.count - 1;
    NatureLayer last = w->net.layer[end];
    int B = n->output[end].shape[0] / (last.oh * last.ow);
    // The spatial API exposes B x (HWC), while conv GEMMs use (BHW) x C.
    Prec incoming = grad;
    incoming.shape[0] = B * last.oh * last.ow; incoming.shape[1] = last.co;
    puf_copy(&n->grad_output[end], &incoming, stream);
    for (int i = end; i >= 0; i--) {
        NatureLayer d = w->net.layer[i];
        int rows = B * d.oh * d.ow;
        nature_relu_backward<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
            n->grad_output[i].data, n->output[i].data, rows * d.co);
        Prec x = i == 3 ? a->spatial : n->patches[i];
        if (i == 3 && w->model == 2) {
            x = n->output[2]; x.shape[0] = B; x.shape[1] = d.ci;
        }
        puf_mm_tn(&n->grad_output[i], &x, &n->weight_grad[i], stream);
        nature_bias_backward<<<d.co, 256, 0, stream>>>(n->grad_output[i].data, n->bias_grad[i].data, rows, d.co);
        if (i == 3) {
            Prec dx = w->model == 2 ? n->grad_output[2] : a->grad_spatial;
            dx.shape[0] = B; dx.shape[1] = d.ci;
            puf_mm_nn(&n->grad_output[i], &w->net.weight[i], &dx, stream);
            if (w->model != 2) {
                NatureLayer prev = w->net.layer[2];
                int count = B * prev.oh * prev.ow * 16;
                ffc_adaptive_backward<<<grid_size(count), BLOCK_SIZE, 0, stream>>>(
                    dx.data, n->grad_output[2].data, B, prev.oh, prev.ow, 16);
            }
        } else if (i || w->input_grad) {
            puf_mm_nn(&n->grad_output[i], &w->net.weight[i], &n->grad_patches[i], stream);
            Prec dx = i ? n->grad_output[i - 1] : (w->model ? a->grad_reduced : a->grad_input);
            flex_unpatch<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                n->grad_patches[i].data, dx.data, d, B, w->ph[i], w->pw[i]);
        }
    }
    if (w->model && w->input_grad) {
        int count = B * w->height * w->width * 3;
        ffc_area_backward<<<grid_size(count), BLOCK_SIZE, 0, stream>>>(
            a->grad_reduced.data, a->grad_input.data, B, w->height, w->width);
    }
}

static FFCWeights* ffc_graph(int model, int height, int width) {
    FFCWeights* w = (FFCWeights*)calloc(1, sizeof(*w));
    assert(model == (int)model && model >= 0 && model <= 2);
    w->model = model; w->height = height; w->width = width;
    int H = w->height, W = w->width, ci = 3;
    if (w->model) H = (H + 3) / 4, W = (W + 3) / 4;
    for (int i = 0; i < 3; i++) {
        int k = w->model == 2 ? (i == 0 ? 8 : i == 1 ? 4 : 3) : (i == 0 && !w->model ? 4 : 3);
        int s = w->model == 2 ? (i == 0 ? 4 : i == 1 ? 2 : 1) : (i == 0 ? (w->model ? 1 : 4) : 2);
        int co = w->model == 2 ? (i ? 64 : 32) : (i ? 16 : 8);
        int oh = w->model == 2 ? (H - k) / s + 1 : (H + s - 1) / s;
        int ow = w->model == 2 ? (W - k) / s + 1 : (W + s - 1) / s;
        assert(H >= k && W >= k && oh > 0 && ow > 0);
        w->net.layer[i] = {H, W, ci, co, k, s, oh, ow};
        if (w->model != 2) {
            w->ph[i] = max((oh - 1) * s + k - H, 0) / 2;
            w->pw[i] = max((ow - 1) * s + k - W, 0) / 2;
        }
        H = oh; W = ow; ci = co;
    }
    w->net.count = 4;
    w->net.layer[3] = {1, 1, w->model == 2 ? H * W * ci : 256, 128, 1, 1, 1, 1};
    return w;
}

static void* ffc_create(void* self) {
    Encoder* enc = (Encoder*)self;
    assert(!USE_BF16 && OBS_CHANNELS == 3 && enc->in_dim == OBS_SIZE && enc->out_dim == 128);
    double model = dict_get(enc->config, "compression_model");
    assert(model == (int)model && model >= 0 && model <= 2);
    FFCWeights* w = ffc_graph(model, OBS_HEIGHT, OBS_WIDTH);
    const char* path = dict_get_str(enc->config, "compression_init");
    if (path && strcmp(path, "None")) w->initial = strdup(path);
    return w;
}

static void ffc_init(void* weights, ulong* seed, cudaStream_t stream) {
    FFCWeights* w = (FFCWeights*)weights;
    nature_init_weights(&w->net, seed, stream);
    if (w->initial) ffc_import(w, w->initial, stream);
}

static void ffc_params(void* weights, Allocator* alloc) {
    nature_reg_params(&((FFCWeights*)weights)->net, alloc);
}

static void ffc_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    FFCWeights* w = (FFCWeights*)weights;
    FFCActivations* a = (FFCActivations*)activations;
    if (w->model) {
        a->reduced = {.shape = {B, ((w->height + 3) / 4) * ((w->width + 3) / 4) * 3}};
        alloc_register(alloc, &a->reduced);
    }
    if (w->model != 2 && w->net.count == 4) {
        a->spatial = {.shape = {B, 256}};
        alloc_register(alloc, &a->spatial);
    }
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        if (i < 3) {
            a->net.patches[i] = {.shape = {B * d.oh * d.ow, d.k * d.k * d.ci}};
            alloc_register(alloc, &a->net.patches[i]);
        }
        a->net.output[i] = {.shape = {B * d.oh * d.ow, d.co}};
        alloc_register(alloc, &a->net.output[i]);
    }
}

static void ffc_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    FFCWeights* w = (FFCWeights*)weights;
    FFCActivations* a = (FFCActivations*)activations;
    ffc_rollout(weights, activations, acts, B);
    if (w->input_grad) {
        a->grad_input = {.shape = {B, w->height * w->width * w->net.layer[0].ci}};
        alloc_register(acts, &a->grad_input);
    }
    if (w->model && w->input_grad) {
        a->grad_reduced = {.shape = {B, ((w->height + 3) / 4) * ((w->width + 3) / 4) * 3}};
        alloc_register(acts, &a->grad_reduced);
    }
    if (w->model != 2 && w->net.count == 4) {
        a->grad_spatial = {.shape = {B, 256}};
        alloc_register(acts, &a->grad_spatial);
    }
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        a->net.grad_output[i] = {.shape = {B * d.oh * d.ow, d.co}};
        alloc_register(acts, &a->net.grad_output[i]);
        if (i < 3 && (i || w->input_grad)) {
            a->net.grad_patches[i] = {.shape = {B * d.oh * d.ow, d.k * d.k * d.ci}};
            alloc_register(acts, &a->net.grad_patches[i]);
        }
        a->net.weight_grad[i] = {.shape = {d.co, d.k * d.k * d.ci}};
        a->net.bias_grad[i] = {.shape = {d.co}};
        alloc_register(grads, &a->net.weight_grad[i]);
        alloc_register(grads, &a->net.bias_grad[i]);
    }
}

static void create_ffc_encoder(Encoder* enc, Dict* policy) {
    *enc = Encoder{.forward = ffc_forward, .backward = ffc_backward,
        .init_weights = ffc_init, .reg_params = ffc_params, .reg_train = ffc_train,
        .reg_rollout = ffc_rollout, .create_weights = ffc_create,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(FFCActivations), .config = policy};
}

#include "fullframe_compression_io.h"
