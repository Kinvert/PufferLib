// Encoder 5: expanded native stage grammar. Encoder 4 keeps its original layout.
#include <limits.h>

struct Flex2Weights {
    NatureWeights net;
    int type[NATURE_MAX_LAYERS], ph[NATURE_MAX_LAYERS], pw[NATURE_MAX_LAYERS];
    int dilation[NATURE_MAX_LAYERS], activation[NATURE_MAX_LAYERS];
    bool skip[NATURE_MAX_LAYERS];
    Prec function[NATURE_MAX_LAYERS];
};
struct Flex2Activations {
    NatureActivations net;
    Prec preactivation[NATURE_MAX_LAYERS], function_grad[NATURE_MAX_LAYERS];
};

static int flex2_option(Dict* policy, const char* key, int fallback, int lo, int hi) {
    DictItem* item = policy ? dict_find(policy, key) : NULL;
    double value = item ? item->value : fallback;
    if ((item && (item->str || item->values)) || !isfinite(value)
            || value < lo || value > hi || value != (int)value) {
        fprintf(stderr, "invalid encoder 5 option: %s\n", key); exit(1);
    }
    return (int)value;
}
static int flex2_pow2(Dict* policy, const char* key, int fallback, int lo, int hi) {
    int value = flex2_option(policy, key, fallback, lo, hi);
    if (value & (value-1)) { fprintf(stderr, "%s must be a power of two\n", key); exit(1); }
    return value;
}

// 0 ReLU; 1 SiLU; 2 exact GELU; 3 learned PReLU; 4 learned rational (5,4).
// Rational: P(x)/(1+|Q(x)|), with six numerator and four denominator coefficients.
// Two unused coefficient slots keep FP32 allocator/optimizer offsets aligned.
__device__ float flex2_function(float x, int kind, const precision_t* p, float* dx, float* dp) {
    if (dp) for (int j = 0; j < 12; j++) dp[j] = 0;
    if (kind == 0) { *dx = x > 0; return fmaxf(x, 0); }
    if (kind == 1) {
        float q = x >= 0 ? 1/(1+expf(-x)) : expf(x)/(1+expf(x));
        *dx = q+x*q*(1-q); return x*q;
    }
    if (kind == 2) {
        float q = 0.5f*erfcf(-x*0.7071067811865475f);
        *dx = q+x*expf(-0.5f*x*x)*0.3989422804014327f; return x*q;
    }
    if (kind == 3) {
        *dx = x >= 0 ? 1 : to_float(p[0]);
        if (dp) dp[0] = x < 0 ? x : 0;
        return x * *dx;
    }
    float powers[6] = {1,x};
    for (int j = 2; j <= 5; j++) powers[j] = powers[j-1]*x;
    float a = 0, da = 0, b = 0, db = 0;
    for (int j = 0; j <= 5; j++) {
        a += to_float(p[j])*powers[j];
        if (j) da += j*to_float(p[j])*powers[j-1];
    }
    for (int j = 1; j <= 4; j++) {
        b += to_float(p[j+5])*powers[j]; db += j*to_float(p[j+5])*powers[j-1];
    }
    float den = 1+fabsf(b), sign = (b > 0)-(b < 0);
    *dx = da/den-a*sign*db/(den*den);
    if (dp) {
        for (int j = 0; j <= 5; j++) dp[j] = powers[j]/den;
        for (int j = 1; j <= 4; j++) dp[j+5] = -a*sign*powers[j]/(den*den);
    }
    return a/den;
}
__global__ void flex2_activate(precision_t* values, precision_t* saved,
        const precision_t* bias, const precision_t* skip, const precision_t* function,
        int kind, int n, int channels) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
        float x = to_float(values[i])+to_float(bias[i%channels]);
        if (skip) x += to_float(skip[i]);
        float dx; saved[i] = from_float(x);
        values[i] = from_float(flex2_function(x,kind,function,&dx,NULL));
    }
}
__global__ void flex2_activation_grad(precision_t* grad, const precision_t* saved,
        const precision_t* function, int kind, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) {
        float dx; flex2_function(to_float(saved[i]),kind,function,&dx,NULL);
        grad[i] = from_float(to_float(grad[i])*dx);
    }
}
__global__ void flex2_function_grad(const precision_t* grad, const precision_t* saved,
        const precision_t* function, precision_t* out, int kind, int n) {
    __shared__ float partial[256];
    int j = blockIdx.x; float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float dx, dp[12]; flex2_function(to_float(saved[i]),kind,function,&dx,dp);
        sum += to_float(grad[i])*dp[j];
    }
    partial[threadIdx.x] = sum; __syncthreads();
    for (int s = 128; s; s /= 2) {
        if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x+s];
        __syncthreads();
    }
    if (!threadIdx.x) out[j] = from_float(partial[0]);
}

// Adaptive average pooling. Overlapping bins on odd input sizes are deliberate.
__global__ void flex2_adaptive(const precision_t* input, precision_t* output,
        NatureLayer d, int B) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= B*d.oh*d.ow*d.co) return;
    int c = i%d.co, x = i/d.co%d.ow, y = i/(d.co*d.ow)%d.oh, b = i/(d.co*d.ow*d.oh);
    int y0 = y*d.ih/d.oh, y1 = ((y+1)*d.ih+d.oh-1)/d.oh;
    int x0 = x*d.iw/d.ow, x1 = ((x+1)*d.iw+d.ow-1)/d.ow;
    float sum = 0;
    for (int iy = y0; iy < y1; iy++) for (int ix = x0; ix < x1; ix++)
        sum += to_float(input[((b*d.ih+iy)*d.iw+ix)*d.ci+c]);
    output[i] = from_float(sum/((y1-y0)*(x1-x0)));
}
__global__ void flex2_adaptive_grad(const precision_t* grad, precision_t* output,
        NatureLayer d, int B) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= B*d.ih*d.iw*d.ci) return;
    int c = i%d.ci, x = i/d.ci%d.iw, y = i/(d.ci*d.iw)%d.ih, b = i/(d.ci*d.iw*d.ih);
    float sum = 0;
    for (int oy = 0; oy < d.oh; oy++) for (int ox = 0; ox < d.ow; ox++) {
        int y0 = oy*d.ih/d.oh, y1 = ((oy+1)*d.ih+d.oh-1)/d.oh;
        int x0 = ox*d.iw/d.ow, x1 = ((ox+1)*d.iw+d.ow-1)/d.ow;
        if (y >= y0 && y < y1 && x >= x0 && x < x1)
            sum += to_float(grad[((b*d.oh+oy)*d.ow+ox)*d.co+c])/((y1-y0)*(x1-x0));
    }
    output[i] = from_float(sum);
}

static Prec flex2_forward(void* weights, void* activations, Prec input, cudaStream_t stream) {
    Flex2Weights* w = (Flex2Weights*)weights; Flex2Activations* aa = (Flex2Activations*)activations;
    NatureActivations* a = &aa->net;
    int B = a->output[0].shape[0]/(w->net.layer[0].oh*w->net.layer[0].ow);
    assert(numel(input.shape) == (int64_t)B*OBS_SIZE);
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i]; int n = B*d.oh*d.ow*d.co;
        if (w->type[i] == 1)
            c4_cnn::imp_pool<<<grid_size(n),BLOCK_SIZE,0,stream>>>(input.data,a->output[i].data,a->winner[i].data,B,d.ih,d.iw,d.ci);
        else if (w->type[i] == 2)
            flex_average<<<grid_size(n),BLOCK_SIZE,0,stream>>>(input.data,a->output[i].data,d,B,false);
        else if (w->type[i] == 3)
            flex2_adaptive<<<grid_size(n),BLOCK_SIZE,0,stream>>>(input.data,a->output[i].data,d,B);
        else {
            flex_patches<<<grid_size(B*d.oh*d.ow*d.k*d.k*d.ci),BLOCK_SIZE,0,stream>>>(input.data,a->patches[i].data,d,B,w->ph[i],w->pw[i],w->dilation[i]);
            puf_mm(&a->patches[i],&w->net.weight[i],&a->output[i],stream);
            flex2_activate<<<grid_size(n),BLOCK_SIZE,0,stream>>>(a->output[i].data,aa->preactivation[i].data,
                w->net.bias[i].data,w->skip[i]?input.data:NULL,w->function[i].data,w->activation[i],n,d.co);
        }
        input = a->output[i];
    }
    return input;
}
static void flex2_backward(void* weights, void* activations, Prec grad, cudaStream_t stream) {
    Flex2Weights* w = (Flex2Weights*)weights; Flex2Activations* aa = (Flex2Activations*)activations;
    NatureActivations* a = &aa->net; int last = w->net.count-1, B = a->output[last].shape[0];
    puf_copy(&a->grad_output[last],&grad,stream);
    for (int i = last; i >= 0; i--) {
        NatureLayer d = w->net.layer[i]; int rows = B*d.oh*d.ow;
        if (w->type[i] == 1)
            c4_cnn::imp_pool_grad<<<grid_size(B*d.ih*d.iw*d.ci),BLOCK_SIZE,0,stream>>>(a->grad_output[i].data,a->winner[i].data,a->grad_output[i-1].data,B,d.ih,d.iw,d.ci);
        else if (w->type[i] == 2)
            flex_average_grad<<<grid_size(B*d.ih*d.iw*d.ci),BLOCK_SIZE,0,stream>>>(a->grad_output[i].data,a->grad_output[i-1].data,d,B,false);
        else if (w->type[i] == 3)
            flex2_adaptive_grad<<<grid_size(B*d.ih*d.iw*d.ci),BLOCK_SIZE,0,stream>>>(a->grad_output[i].data,a->grad_output[i-1].data,d,B);
        else {
            if (w->activation[i] >= 3)
                flex2_function_grad<<<12,256,0,stream>>>(a->grad_output[i].data,aa->preactivation[i].data,w->function[i].data,aa->function_grad[i].data,w->activation[i],rows*d.co);
            flex2_activation_grad<<<grid_size(rows*d.co),BLOCK_SIZE,0,stream>>>(a->grad_output[i].data,aa->preactivation[i].data,w->function[i].data,w->activation[i],rows*d.co);
            puf_mm_tn(&a->grad_output[i],&a->patches[i],&a->weight_grad[i],stream);
            nature_bias_backward<<<d.co,256,0,stream>>>(a->grad_output[i].data,a->bias_grad[i].data,rows,d.co);
            if (i) {
                puf_mm_nn(&a->grad_output[i],&w->net.weight[i],&a->grad_patches[i],stream);
                flex_unpatch<<<grid_size(B*d.ih*d.iw*d.ci),BLOCK_SIZE,0,stream>>>(a->grad_patches[i].data,a->grad_output[i-1].data,d,B,w->ph[i],w->pw[i],w->dilation[i]);
                if (w->skip[i]) c4_cnn::imp_add<<<grid_size(rows*d.co),BLOCK_SIZE,0,stream>>>(a->grad_output[i-1].data,a->grad_output[i].data,rows*d.co);
            }
        }
    }
}
static void flex2_reg_params(void* weights, Allocator* alloc) {
    Flex2Weights* w = (Flex2Weights*)weights;
    for (int i = 0; i < w->net.count; i++) if (!w->type[i]) {
        NatureLayer d = w->net.layer[i];
        w->net.weight[i] = {.shape={d.co,d.k*d.k*d.ci}}; w->net.bias[i] = {.shape={d.co}};
        alloc_register(alloc,&w->net.weight[i]); alloc_register(alloc,&w->net.bias[i]);
        if (w->activation[i] >= 3) { w->function[i] = {.shape={12}}; alloc_register(alloc,&w->function[i]); }
    }
}
static void flex2_init(void* weights, ulong* seed, cudaStream_t stream) {
    Flex2Weights* w = (Flex2Weights*)weights;
    for (int i = 0; i < w->net.count; i++) if (!w->type[i]) {
        puf_kaiming_init(&w->net.weight[i],sqrtf(2.0f),(*seed)++,stream);
        cudaMemsetAsync(w->net.bias[i].data,0,numel(w->net.bias[i].shape)*sizeof(precision_t),stream);
        if (w->activation[i] >= 3) {
            precision_t function[12] = {};
            if (w->activation[i] == 3) function[0] = from_float(0.25f);
            else { function[1] = from_float(1); function[7] = from_float(0.001f); }
            cudaMemcpyAsync(w->function[i].data,function,sizeof(function),cudaMemcpyHostToDevice,stream);
            cudaStreamSynchronize(stream); // initialization only: stack source lifetime
        }
    }
}
static void flex2_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    Flex2Weights* w = (Flex2Weights*)weights; Flex2Activations* aa = (Flex2Activations*)activations;
    *aa = {}; NatureActivations* a = &aa->net;
    if (B <= 0 || (int64_t)B*OBS_SIZE > INT_MAX) {
        fprintf(stderr,"encoder 5 input batch exceeds kernel index range\n"); exit(1);
    }
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        int64_t rows = (int64_t)B*d.oh*d.ow, K = (int64_t)d.k*d.k*d.ci;
        if (B <= 0 || rows*d.co > INT_MAX || (!w->type[i] && rows*K > INT_MAX)) {
            fprintf(stderr,"encoder 5 batch exceeds kernel index range\n"); exit(1);
        }
        a->output[i] = {.shape={rows,d.co}}; alloc_register(alloc,&a->output[i]);
        if (w->type[i] == 1) {
            // Both training and actor paths need winners; allocate in their own buffers.
            a->winner[i] = {.shape={rows*d.co}}; alloc_register(alloc,&a->winner[i]);
        }
        if (!w->type[i]) {
            a->patches[i] = {.shape={rows,K}}; alloc_register(alloc,&a->patches[i]);
            aa->preactivation[i] = {.shape={rows,d.co}}; alloc_register(alloc,&aa->preactivation[i]);
        }
    }
}
static void flex2_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    Flex2Weights* w = (Flex2Weights*)weights; Flex2Activations* aa = (Flex2Activations*)activations;
    flex2_reg_rollout(weights,activations,acts,B); NatureActivations* a = &aa->net;
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        a->grad_output[i] = {.shape={a->output[i].shape[0],d.co}}; alloc_register(acts,&a->grad_output[i]);
        if (w->type[i]) continue;
        if (i) { a->grad_patches[i] = {.shape={a->patches[i].shape[0],a->patches[i].shape[1]}}; alloc_register(acts,&a->grad_patches[i]); }
        a->weight_grad[i] = {.shape={d.co,d.k*d.k*d.ci}}; a->bias_grad[i] = {.shape={d.co}};
        alloc_register(grads,&a->weight_grad[i]); alloc_register(grads,&a->bias_grad[i]);
        if (w->activation[i] >= 3) { aa->function_grad[i] = {.shape={12}}; alloc_register(grads,&aa->function_grad[i]); }
    }
}
static void flex2_conv(Flex2Weights* w, int H, int W, int ci, int co,
        int k, int stride, int dilation, int activation, bool skip = false) {
    int i = w->net.count++; assert(i < NATURE_MAX_LAYERS);
    int oh = (H+stride-1)/stride, ow = (W+stride-1)/stride, effective = (k-1)*dilation+1;
    w->net.layer[i] = {H,W,ci,co,k,stride,oh,ow};
    w->ph[i] = max((oh-1)*stride+effective-H,0)/2;
    w->pw[i] = max((ow-1)*stride+effective-W,0)/2;
    w->dilation[i] = k == 1 ? 1 : dilation;
    w->activation[i] = activation; w->skip[i] = skip;
}
static void* flex2_create_weights(void* self) {
    Encoder* enc = (Encoder*)self;
    if (USE_BF16) { fprintf(stderr,"encoder 5 requires --float until BF16 validation\n"); exit(1); }
    assert(enc->in_dim == OBS_SIZE && enc->out_dim >= 8 && enc->out_dim % 8 == 0);
    Flex2Weights* w = (Flex2Weights*)calloc(1,sizeof(*w)); assert(w);
    int depth = flex2_option(enc->config,"cnn_depth",2,1,4);
    int projection = flex2_pow2(enc->config,"cnn_projection",32,16,128);
    int readout = flex2_option(enc->config,"cnn_readout",0,0,3);
    int H = OBS_HEIGHT, W = OBS_WIDTH, ci = OBS_CHANNELS;
    for (int stage = 1; stage <= depth; stage++) {
        char key[64];
#define F2OPT(name, def, lo, hi) snprintf(key,sizeof(key),"cnn_" #name "_%d",stage); int name = flex2_option(enc->config,key,def,lo,hi)
        F2OPT(channels,8,8,64);
        F2OPT(kernel,stage == 1 ? 7 : 3,1,stage == 1 ? 8 : 5);
        F2OPT(stride,stage == 1 ? 4 : 1,1,stage == 1 ? 8 : 4);
        F2OPT(dilation,1,1,4);
        F2OPT(activation,0,0,4);
        F2OPT(residual,0,0,2);
        F2OPT(pool,0,0,2);
#undef F2OPT
        assert(!(channels & (channels-1)) && !(stride & (stride-1)));
        flex2_conv(w,H,W,ci,channels,kernel,stride,dilation,activation);
        H = (H+stride-1)/stride; W = (W+stride-1)/stride; ci = channels;
        for (int repeat = 0; repeat < residual; repeat++)
            flex2_conv(w,H,W,ci,ci,3,1,dilation,activation,true);
        if (pool) {
            int i = w->net.count++; assert(i < NATURE_MAX_LAYERS);
            w->type[i] = pool;
            w->net.layer[i] = {H,W,ci,ci,3,2,(H+1)/2,(W+1)/2};
            H = (H+1)/2; W = (W+1)/2;
        }
    }
    if (readout) {
        int i = w->net.count++, G = readout == 3 ? 4 : readout;
        assert(i < NATURE_MAX_LAYERS);
        w->type[i] = 3; w->net.layer[i] = {H,W,ci,ci,1,1,G,G}; H = W = G;
    }
    int head_act = flex2_option(enc->config,"cnn_projection_activation",0,0,4);
    flex2_conv(w,1,1,H*W*ci,projection,1,1,1,head_act);
    if (projection != enc->out_dim) flex2_conv(w,1,1,projection,enc->out_dim,1,1,1,head_act);
    return w;
}
static void create_flex2_encoder(Encoder* enc, Dict* policy) {
    *enc = Encoder{.forward=flex2_forward,.backward=flex2_backward,.init_weights=flex2_init,
        .reg_params=flex2_reg_params,.reg_train=flex2_reg_train,.reg_rollout=flex2_reg_rollout,
        .create_weights=flex2_create_weights,.in_dim=enc->in_dim,.out_dim=enc->out_dim,
        .activation_size=sizeof(Flex2Activations),.config=policy};
}

// Campaign-local graph/recipe fingerprint, before starting a duplicate worker.
// Not a cryptographic provenance hash; the reporting sidecar records SHA256 too.
static uint64_t flex2_hash_bytes(uint64_t h, const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i++) h = (h^p[i])*UINT64_C(1099511628211);
    return h;
}
static uint64_t flex2_trial_hash(Ini* ini) {
    Dict* policy = puf_ini_section(ini,"policy",0);
    Encoder enc = {.in_dim=OBS_SIZE,.out_dim=(int)dict_get(policy,"hidden_size"),.config=policy};
    Flex2Weights* w = (Flex2Weights*)flex2_create_weights(&enc);
    uint64_t h = UINT64_C(14695981039346656037);
    int geometry[] = {OBS_HEIGHT,OBS_WIDTH,OBS_CHANNELS,w->net.count};
    h = flex2_hash_bytes(h,geometry,sizeof(geometry));
    for (int i = 0; i < w->net.count; i++) {
        NatureLayer d = w->net.layer[i];
        int layer[] = {d.ih,d.iw,d.ci,d.co,d.k,d.stride,d.oh,d.ow,w->type[i],
            w->ph[i],w->pw[i],w->dilation[i],w->activation[i],w->skip[i]};
        h = flex2_hash_bytes(h,layer,sizeof(layer));
    }
    free(w);
    for (int s = 0; s < ini->num_sections; s++) {
        Dict* section = &ini->sections[s];
        if (strcmp(section->name,"train") && strcmp(section->name,"env")
                && strcmp(section->name,"vec") && strcmp(section->name,"policy")
                && strcmp(section->name,"base")) continue;
        for (int j = 0; j < section->size; j++) {
            DictItem* item = &section->items[j];
            if (!strcmp(section->name,"policy") && !strncmp(item->key,"cnn_",4)) continue;
            if (!strcmp(section->name,"base") && strcmp(item->key,"seed")) continue;
            Dict* env = puf_ini_section(ini,"env",0);
            DictItem* mode = dict_find(env,"representation_mode");
            if (!strcmp(section->name,"env") && mode && mode->value == 1
                    && !strcmp(item->key,"representation")) continue;
            h = flex2_hash_bytes(h,section->name,strlen(section->name)+1);
            h = flex2_hash_bytes(h,item->key,strlen(item->key)+1);
            int kind = item->values ? 2 : item->str ? 1 : 0;
            h = flex2_hash_bytes(h,&kind,sizeof(kind));
            if (item->values) {
                h = flex2_hash_bytes(h,&item->len,sizeof(item->len));
                h = flex2_hash_bytes(h,item->values,item->len*sizeof(double));
            } else if (item->str) h = flex2_hash_bytes(h,item->str,strlen(item->str)+1);
            else h = flex2_hash_bytes(h,&item->value,sizeof(item->value));
        }
    }
    return h;
}
