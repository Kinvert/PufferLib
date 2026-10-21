// Research-only spatial compressor and CNN consumer. No decoder in rollout.
static FFCWeights* sc_graph(int H, int W, int stride, int channels) {
    ffc_require((stride == 8 || stride == 16 || stride == 32)
        && (channels == 4 || channels == 8 || channels == 16 || channels == 32), "spatial size choices");
    FFCWeights* w = ffc_graph(0, H, W);
    w->net.count = 3;
    NatureLayer* l = &w->net.layer[2];
    // Kernel covers stride 4 too: the strongest compression has no blind gaps.
    l->k = 4; l->stride = stride / 8; l->co = channels;
    l->oh = (l->ih + l->stride - 1) / l->stride;
    l->ow = (l->iw + l->stride - 1) / l->stride;
    w->ph[2] = max(0, (l->oh - 1) * l->stride + l->k - l->ih) / 2;
    w->pw[2] = max(0, (l->ow - 1) * l->stride + l->k - l->iw) / 2;
    return w;
}

static cJSON* sc_manifest(FFCWeights* w) {
    // Parameter registration initializes tensor descriptors. Never register a
    // live model just to inspect it: that would erase its device pointers.
    FFCWeights shape = *w; Allocator p = {}; ffc_params(&shape, &p);
    cJSON* root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "schema", "spatial-compressor-v1");
    cJSON_AddStringToObject(root, "dtype", "float32");
    cJSON_AddStringToObject(root, "layout", "NHWC");
    cJSON_AddStringToObject(root, "input_range", "0,1");
    cJSON_AddStringToObject(root, "activation", "relu-zero-derivative-0");
    cJSON_AddStringToObject(root, "padding", "same-trailing-extra");
    cJSON_AddNumberToObject(root, "height", w->height);
    cJSON_AddNumberToObject(root, "width", w->width);
    cJSON_AddNumberToObject(root, "channels", 3);
    NatureLayer last = w->net.layer[2];
    int latent[] = {last.oh, last.ow, last.co};
    cJSON_AddItemToObject(root, "latent_hwc", cJSON_CreateIntArray(latent, 3));
    cJSON_AddNumberToObject(root, "latent_stride", last.stride * 8);
    cJSON_AddNumberToObject(root, "elements", p.total_elems);
    cJSON* layers = cJSON_AddArrayToObject(root, "layers");
    for (int i = 0; i < 3; i++) {
        NatureLayer l = w->net.layer[i];
        int values[] = {l.ih,l.iw,l.ci,l.co,l.k,l.stride,l.oh,l.ow,w->ph[i],w->pw[i]};
        cJSON_AddItemToArray(layers, cJSON_CreateIntArray(values, 10));
    }
    cJSON* tensors = cJSON_AddArrayToObject(root, "tensors"); long offset = 0;
    for (int i = 0; i < p.num_regs; i++) {
        cJSON* t = cJSON_CreateObject(); char name[32];
        snprintf(name, sizeof(name), "%s_%d", i % 2 ? "bias" : "weight", i / 2);
        cJSON_AddStringToObject(t, "name", name); cJSON_AddNumberToObject(t, "offset", offset);
        long count = numel(p.regs[i].shape); cJSON_AddNumberToObject(t, "elements", count);
        cJSON* shape = cJSON_AddArrayToObject(t, "shape");
        for (int j = 0; j < ndim(p.regs[i].shape); j++)
            cJSON_AddItemToArray(shape, cJSON_CreateNumber(p.regs[i].shape[j]));
        cJSON_AddItemToArray(tensors, t); offset += count;
    }
    free(p.regs); return root;
}

static void sc_import(FFCWeights* w, const char* directory, cudaStream_t stream) {
    std::string path = std::string(directory) + "/encoder.json";
    size_t size; char* text = ffc_read_file(path.c_str(), &size);
    cJSON* root = cJSON_ParseWithLength(text, size + 1);
    cJSON* expected = sc_manifest(w);
    // Compare every structural field, including tensor order and spatial shape.
    for (cJSON* key = expected->child; key; key = key->next)
        ffc_require(cJSON_Compare(key, cJSON_GetObjectItemCaseSensitive(root, key->string), true),
            "incompatible spatial encoder manifest");
    cJSON_Delete(expected);
    path = std::string(directory) + "/encoder.bin";
    char* data = ffc_read_file(path.c_str(), &size); char digest[65];
    ffc_sha256((unsigned char*)data, size, digest);
    long elements = 0;
    for (int i = 0; i < 3; i++) elements += numel(w->net.weight[i].shape) + numel(w->net.bias[i].shape);
    ffc_require(size == elements * 4 && ffc_string(root, "sha256", digest), "spatial encoder bytes/hash");
    for (long i = 0; i < elements; i++) ffc_require(isfinite(((float*)data)[i]), "finite spatial weights");
    long offset = 0;
    for (int i = 0; i < 3; i++) for (Prec* t : {&w->net.weight[i], &w->net.bias[i]}) {
        ffc_require(cudaMemcpyAsync(t->data, data + offset * 4, numel(t->shape) * 4,
            cudaMemcpyHostToDevice, stream) == cudaSuccess, "spatial encoder upload");
        offset += numel(t->shape);
    }
    ffc_require(cudaStreamSynchronize(stream) == cudaSuccess, "spatial import synchronization");
    free(data); free(text); cJSON_Delete(root);
}

struct SCPolicy {
    FFCWeights *prefix, *cnn;
    Allocator frozen_parameters;
    bool frozen;
    char* initial;
};
struct SCPolicyActs { FFCActivations prefix, cnn; };

static void* sc_policy_create(void* self) {
    Encoder* enc = (Encoder*)self;
    ffc_require(!USE_BF16 && OBS_CHANNELS == 3 && enc->out_dim == 128, "spatial policy requires float32/H128");
    SCPolicy* w = (SCPolicy*)calloc(1, sizeof(*w));
    int stride, channels;
    std::string initial;
    if (dict_find(enc->config, "compression_bank")) {
        const char* bank = dict_get_str(enc->config, "compression_bank");
        std::string path = std::string(bank) + "/bank.json"; size_t size;
        char* text = ffc_read_file(path.c_str(), &size); cJSON* root = cJSON_ParseWithLength(text, size + 1);
        double id = dict_get(enc->config, "compression_variant");
        cJSON* variants = cJSON_GetObjectItemCaseSensitive(root, "variants");
        ffc_require(ffc_string(root, "schema", "spatial-compressor-bank-v1") && id == (int)id
            && id >= 0 && id < cJSON_GetArraySize(variants), "compression bank variant");
        cJSON* item = cJSON_GetArrayItem(variants, (int)id);
        cJSON *s = cJSON_GetObjectItemCaseSensitive(item, "stride"), *c = cJSON_GetObjectItemCaseSensitive(item, "channels");
        ffc_require(cJSON_IsNumber(s) && cJSON_IsNumber(c), "compression bank dimensions");
        stride = s->valueint; channels = c->valueint;
        initial = std::string(bank) + "/s" + std::to_string(stride) + "-c" + std::to_string(channels);
        std::string manifest = initial + "/encoder.json"; size_t bytes;
        char* content = ffc_read_file(manifest.c_str(), &bytes); cJSON* encoder = cJSON_ParseWithLength(content, bytes + 1);
        cJSON* expected = cJSON_GetObjectItemCaseSensitive(item, "sha256");
        ffc_require(cJSON_IsString(expected) && ffc_string(encoder, "sha256", expected->valuestring), "compression bank hash binding");
        free(content); cJSON_Delete(encoder);
        free(text); cJSON_Delete(root);
    } else {
        stride = dict_get(enc->config, "latent_stride"); channels = dict_get(enc->config, "latent_channels");
        initial = dict_get_str(enc->config, "compression_init");
    }
    w->prefix = sc_graph(OBS_HEIGHT, OBS_WIDTH, stride, channels);
    NatureLayer last = w->prefix->net.layer[2];
    w->cnn = (FFCWeights*)calloc(1, sizeof(FFCWeights));
    w->cnn->height = last.oh; w->cnn->width = last.ow; w->cnn->net.count = 4;
    w->cnn->net.layer[3] = {1, 1, 256, 128, 1, 1, 1, 1};
    // CNN operates on learned channels, not RGB. Keep its footprint small.
    NatureLayer* first = &w->cnn->net.layer[0];
    *first = {last.oh, last.ow, last.co, 8, 3, 1, last.oh, last.ow};
    w->cnn->ph[0] = w->cnn->pw[0] = 1;
    int H = first->oh, W = first->ow;
    for (int i = 1; i < 3; i++) {
        NatureLayer* l = &w->cnn->net.layer[i];
        *l = {H, W, i == 1 ? 8 : 16, 16, 3, 2, (H + 1) / 2, (W + 1) / 2};
        w->cnn->ph[i] = max(0, (l->oh - 1) * 2 + 3 - H) / 2;
        w->cnn->pw[i] = max(0, (l->ow - 1) * 2 + 3 - W) / 2;
        H = l->oh; W = l->ow;
    }
    double frozen = dict_get(enc->config, "compression_frozen");
    ffc_require(frozen == 0 || frozen == 1, "compression_frozen 0/1");
    w->frozen = frozen; w->cnn->input_grad = !w->frozen;
    ffc_require(initial != "None" && !initial.empty(), "spatial policy needs an encoder artifact");
    w->initial = strdup(initial.c_str()); return w;
}

static void sc_policy_params(void* weights, Allocator* p) {
    SCPolicy* w = (SCPolicy*)weights;
    ffc_params(w->prefix, w->frozen ? &w->frozen_parameters : p);
    ffc_params(w->cnn, p);
}

static void sc_policy_init(void* weights, ulong* seed, cudaStream_t stream) {
    SCPolicy* w = (SCPolicy*)weights;
    if (w->frozen) alloc_create(&w->frozen_parameters);
    ffc_init(w->prefix, seed, stream); sc_import(w->prefix, w->initial, stream);
    ffc_init(w->cnn, seed, stream);
}

static void sc_policy_rollout(void* weights, void* acts, Allocator* p, int B) {
    SCPolicy* w = (SCPolicy*)weights; SCPolicyActs* a = (SCPolicyActs*)acts;
    ffc_rollout(w->prefix, &a->prefix, p, B); ffc_rollout(w->cnn, &a->cnn, p, B);
}

static void sc_policy_train(void* weights, void* acts, Allocator* p, Allocator* g, int B) {
    SCPolicy* w = (SCPolicy*)weights; SCPolicyActs* a = (SCPolicyActs*)acts;
    if (w->frozen) ffc_rollout(w->prefix, &a->prefix, p, B);
    else ffc_train(w->prefix, &a->prefix, p, g, B);
    ffc_train(w->cnn, &a->cnn, p, g, B);
}

static Prec sc_policy_forward(void* weights, void* acts, Prec x, cudaStream_t stream) {
    SCPolicy* w = (SCPolicy*)weights; SCPolicyActs* a = (SCPolicyActs*)acts;
    Prec z = ffc_forward(w->prefix, &a->prefix, x, stream);
    return ffc_forward(w->cnn, &a->cnn, z, stream);
}

static void sc_policy_backward(void* weights, void* acts, Prec grad, cudaStream_t stream) {
    SCPolicy* w = (SCPolicy*)weights; SCPolicyActs* a = (SCPolicyActs*)acts;
    ffc_backward(w->cnn, &a->cnn, grad, stream);
    if (!w->frozen) ffc_backward(w->prefix, &a->prefix, a->cnn.grad_input, stream);
}

static void create_sc_policy_encoder(Encoder* enc, Dict* cfg) {
    *enc = Encoder{.forward=sc_policy_forward, .backward=sc_policy_backward,
        .init_weights=sc_policy_init, .reg_params=sc_policy_params, .reg_train=sc_policy_train,
        .reg_rollout=sc_policy_rollout, .create_weights=sc_policy_create,
        .in_dim=enc->in_dim, .out_dim=enc->out_dim, .activation_size=sizeof(SCPolicyActs), .config=cfg};
}
