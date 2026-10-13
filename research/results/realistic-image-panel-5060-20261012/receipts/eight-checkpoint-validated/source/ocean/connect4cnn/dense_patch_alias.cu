// Profiler/test-only candidate. Production encoders and owned buffers are unchanged.
// Alias only retained internal dense inputs; never the caller's observation buffer.
static bool dense_patch_alias_eligible(NatureLayer d, int layer, int ph, int pw) {
    return layer > 0 && d.ih == 1 && d.iw == 1 && d.oh == 1 && d.ow == 1
        && d.k == 1 && d.stride == 1 && ph == 0 && pw == 0;
}

static Prec dense_patch_alias_view(NatureActivations* a, int layer) {
    Prec view = a->patches[layer];
    assert(layer > 0 && numel(view.shape) == numel(a->output[layer - 1].shape));
    view.data = a->output[layer - 1].data;
    return view;
}

static Prec dense_patch_alias_forward(void* weights, void* activations,
        Prec input, cudaStream_t stream, bool flexible) {
    FlexWeights* f = flexible ? (FlexWeights*)weights : NULL;
    NatureWeights* w = flexible ? &f->net : (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int B = a->output[w->count - 1].shape[0];
    assert(numel(input.shape) == (int64_t)B * OBS_SIZE);
    for (int i = 0; i < w->count; i++) {
        NatureLayer d = w->layer[i];
        int n = B * d.oh * d.ow * d.co;
        if (f && f->type[i] == 1) {
            c4_cnn::imp_pool<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data,
                a->output[i].data, a->winner[i].data, B, d.ih, d.iw, d.ci);
        } else if (f && f->type[i]) {
            flex_average<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(input.data,
                a->output[i].data, d, B, f->type[i] == 3);
        } else {
            int ph = f ? f->pad_h[i] : 0, pw = f ? f->pad_w[i] : 0;
            Prec patches = a->patches[i];
            if (dense_patch_alias_eligible(d, i, ph, pw)) {
                patches = dense_patch_alias_view(a, i);
                assert(patches.data == input.data);
            } else if (f) {
                flex_patches<<<grid_size(B * d.oh * d.ow * d.k * d.k * d.ci), BLOCK_SIZE, 0, stream>>>(
                    input.data, patches.data, d, B, ph, pw);
            } else {
                nature_im2col<<<grid_size(B * d.oh * d.ow * d.k * d.k * d.ci), BLOCK_SIZE, 0, stream>>>(
                    input.data, patches.data, d, B);
            }
            puf_mm(&patches, &w->weight[i], &a->output[i], stream);
            if (f && f->skip[i]) {
                flex_skip_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(a->output[i].data,
                    w->bias[i].data, input.data, n, d.co);
            } else {
                nature_bias_relu<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
                    a->output[i].data, w->bias[i].data, n, d.co);
            }
        }
        input = a->output[i];
    }
    return input;
}

static void dense_patch_alias_backward(void* weights, void* activations,
        Prec grad, cudaStream_t stream, bool flexible) {
    FlexWeights* f = flexible ? (FlexWeights*)weights : NULL;
    NatureWeights* w = flexible ? &f->net : (NatureWeights*)weights;
    NatureActivations* a = (NatureActivations*)activations;
    int last = w->count - 1, B = a->output[last].shape[0];
    assert(numel(grad.shape) == numel(a->grad_output[last].shape));
    puf_copy(&a->grad_output[last], &grad, stream);
    for (int i = last; i >= 0; i--) {
        NatureLayer d = w->layer[i];
        int rows = B * d.oh * d.ow;
        if (f && f->type[i] == 1) {
            c4_cnn::imp_pool_grad<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->winner[i].data, a->grad_output[i-1].data,
                B, d.ih, d.iw, d.ci);
        } else if (f && f->type[i]) {
            flex_average_grad<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->grad_output[i-1].data, d, B, f->type[i] == 3);
        } else {
            int ph = f ? f->pad_h[i] : 0, pw = f ? f->pad_w[i] : 0;
            Prec patches = dense_patch_alias_eligible(d, i, ph, pw)
                ? dense_patch_alias_view(a, i) : a->patches[i];
            nature_relu_backward<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
                a->grad_output[i].data, a->output[i].data, rows * d.co);
            puf_mm_tn(&a->grad_output[i], &patches, &a->weight_grad[i], stream);
            nature_bias_backward<<<d.co, 256, 0, stream>>>(
                a->grad_output[i].data, a->bias_grad[i].data, rows, d.co);
            if (i) {
                puf_mm_nn(&a->grad_output[i], &w->weight[i], &a->grad_patches[i], stream);
                // Keep the original gather/reduction, including signed-zero behavior.
                if (f) {
                    flex_unpatch<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                        a->grad_patches[i].data, a->grad_output[i-1].data, d, B, ph, pw);
                    if (f->skip[i]) {
                        c4_cnn::imp_add<<<grid_size(rows * d.co), BLOCK_SIZE, 0, stream>>>(
                            a->grad_output[i-1].data, a->grad_output[i].data, rows * d.co);
                    }
                } else {
                    nature_col2im<<<grid_size(B * d.ih * d.iw * d.ci), BLOCK_SIZE, 0, stream>>>(
                        a->grad_patches[i].data, a->grad_output[i-1].data, d, B);
                }
            }
        }
    }
}

static Prec dense_patch_alias_nature_forward(void* w, void* a, Prec x, cudaStream_t stream) {
    return dense_patch_alias_forward(w, a, x, stream, false);
}
static Prec dense_patch_alias_flex_forward(void* w, void* a, Prec x, cudaStream_t stream) {
    return dense_patch_alias_forward(w, a, x, stream, true);
}
static void dense_patch_alias_nature_backward(void* w, void* a, Prec g, cudaStream_t stream) {
    dense_patch_alias_backward(w, a, g, stream, false);
}
static void dense_patch_alias_flex_backward(void* w, void* a, Prec g, cudaStream_t stream) {
    dense_patch_alias_backward(w, a, g, stream, true);
}

static bool dense_patch_alias_select(Encoder* enc) {
    if (enc->forward == nature_forward && enc->backward == nature_backward) {
        enc->forward = dense_patch_alias_nature_forward;
        enc->backward = dense_patch_alias_nature_backward;
        return true;
    }
    if (enc->forward == flex_forward && enc->backward == flex_backward) {
        enc->forward = dense_patch_alias_flex_forward;
        enc->backward = dense_patch_alias_flex_backward;
        return true;
    }
    return false;
}

static long dense_patch_alias_payload(Encoder* enc, void* weights, int B, int* layers) {
    bool flexible = enc->forward == dense_patch_alias_flex_forward;
    *layers = 0;
    if (!flexible && enc->forward != dense_patch_alias_nature_forward) return 0;
    FlexWeights* f = flexible ? (FlexWeights*)weights : NULL;
    NatureWeights* w = flexible ? &f->net : (NatureWeights*)weights;
    long bytes = 0;
    for (int i = 0; i < w->count; i++) {
        NatureLayer d = w->layer[i];
        if ((!f || f->type[i] == 0) && dense_patch_alias_eligible(d, i,
                f ? f->pad_h[i] : 0, f ? f->pad_w[i] : 0)) {
            (*layers)++;
            bytes += (long)B * d.ci * sizeof(precision_t);
        }
    }
    return bytes;
}

static void dense_patch_alias_receipt(Encoder* enc, void* weights, int B, bool active) {
    int layers = 0;
    long bytes = dense_patch_alias_payload(enc, weights, B, &layers);
    printf("\"dense_patch_alias_candidate\":true,\"dense_patch_alias_active\":%s,"
        "\"dense_patch_alias_layers\":%d,\"skipped_forward_patch_payload_bytes\":%ld,",
        active ? "true" : "false", layers, bytes);
}
