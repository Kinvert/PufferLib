// Test-only independent float64 CUDA DAG. No CPU neural computation.
// Native callbacks are the subject; the oracle uses direct CUDA sums.
#define main packed_training_main
#include "../src/pufferl.cu"
#undef main
#include "native_encoder_reference.cu"
#include <sys/stat.h>

enum { REF_INPUT, REF_CONV, REF_RELU, REF_ADD, REF_POOL, REF_AVG, REF_GEOMETRY };
struct RefNode {
    int op, a, b, batch, h, w, c;
    Layer layer;
    double* y;
    double* g;
    int* winner;
};

__global__ void packed_ref_input(const double* obs, double* out, int B, bool tiles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * 55296) return;
    int at = i % 55296;
    if (tiles) {
        int tile = at / 3072, p = at % 3072;
        at = ((tile / 3 * 32 + p / 96) * 96 + tile % 3 * 32 + p / 3 % 32) * 3 + p % 3;
    }
    out[i] = obs[i / 55296 * 55300 + at];
}

__global__ void packed_ref_conv(const double* x, const double* weights, double* y,
        Layer l, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * l.oh * l.ow * l.co) return;
    int c = i % l.co, ox = i / l.co % l.ow, oy = i / (l.co * l.ow) % l.oh;
    int b = i / (l.co * l.ow * l.oh), K = l.k * l.k * l.ci;
    double sum = weights[l.offset + c + l.co * K];
    for (int ky = 0; ky < l.k; ky++) {
        for (int kx = 0; kx < l.k; kx++) {
            int y0 = oy * l.stride + ky - l.ph, x0 = ox * l.stride + kx - l.pw;
            if (y0 < 0 || y0 >= l.h || x0 < 0 || x0 >= l.w) continue;
            for (int ci = 0; ci < l.ci; ci++) {
                sum += x[((b * l.h + y0) * l.w + x0) * l.ci + ci]
                    * weights[l.offset + c * K + (ky * l.k + kx) * l.ci + ci];
            }
        }
    }
    y[i] = sum;
}

__global__ void packed_ref_element(const double* a, const double* b, double* out,
        int n, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = op == REF_RELU ? fmax(a[i], 0.0) : a[i] + b[i];
}

__global__ void packed_ref_accumulate(const double* g, const double* y, double* out,
        int n, bool relu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += !relu || y[i] > 0 ? g[i] : 0;
}

__global__ void packed_ref_pool(const double* x, double* y, int* winner,
        int B, int H, int W, int C, bool average, bool global) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int OH = global ? 1 : (H + 1) / 2, OW = global ? 1 : (W + 1) / 2;
    if (i >= B * OH * OW * C) return;
    int c = i % C, ox = i / C % OW, oy = i / (C * OW) % OH;
    int b = i / (C * OW * OH);
    int top = global ? 0 : oy * 2 - H % 2;
    int left = global ? 0 : ox * 2 - W % 2;
    int endy = global ? H : min(top + 3, H), endx = global ? W : min(left + 3, W);
    double sum = 0, best = -INFINITY;
    int count = 0, at = -1;
    for (int y0 = max(top, 0); y0 < endy; y0++) {
        for (int x0 = max(left, 0); x0 < endx; x0++) {
            int p = ((b * H + y0) * W + x0) * C + c;
            sum += x[p];
            count++;
            if (at < 0 || x[p] > best) { best = x[p]; at = p; }
        }
    }
    y[i] = average ? sum / count : best;
    if (winner) winner[i] = at;
}

__global__ void packed_ref_pool_back(const double* dy, const int* winner, double* dx,
        int B, int H, int W, int C, bool average, bool global) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H * W * C) return;
    int c = i % C, x = i / C % W, y = i / (C * W) % H, b = i / (C * W * H);
    int OH = global ? 1 : (H + 1) / 2, OW = global ? 1 : (W + 1) / 2;
    double sum = 0;
    int firsty = global ? 0 : max(0, (y + H % 2 - 2 + 1) / 2);
    int firstx = global ? 0 : max(0, (x + W % 2 - 2 + 1) / 2);
    int lasty = global ? 0 : min(OH - 1, (y + H % 2) / 2);
    int lastx = global ? 0 : min(OW - 1, (x + W % 2) / 2);
    for (int oy = firsty; oy <= lasty; oy++) {
        for (int ox = firstx; ox <= lastx; ox++) {
            int top = global ? 0 : oy * 2 - H % 2;
            int left = global ? 0 : ox * 2 - W % 2;
            if (!global && (y < top || y >= top + 3 || x < left || x >= left + 3)) continue;
            int p = ((b * OH + oy) * OW + ox) * C + c;
            int count = global ? H * W
                : (min(top + 3, H) - max(top, 0)) * (min(left + 3, W) - max(left, 0));
            if (average) sum += dy[p] / count;
            else if (winner[p] == i) sum += dy[p];
        }
    }
    dx[i] += sum;
}

__global__ void packed_ref_geometry(const double* x, const double* obs, double* out,
        int B, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * (features + 4)) return;
    int b = i / (features + 4), c = i % (features + 4);
    out[i] = c < features ? x[b * features + c] : obs[b * 55300 + 55296 + c - features];
}

__global__ void packed_ref_geometry_back(const double* g, double* dx, int B, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * features) dx[i] += g[i / features * (features + 4) + i % features];
}

struct PackedReference {
    std::vector<RefNode> nodes;
    Storage storage;
    int parameters = 0, batch;
    bool tile;
    PackedReference(int B, int family, Dict* policy) : batch(B), tile(family == 6) {
        nodes.push_back({REF_INPUT, -1, -1, tile ? B * 18 : B,
            tile ? 32 : 192, tile ? 32 : 96, 3});
        int x = 0;
        if (tile) {
            x = relu(conv(x, 8, 4, 2, false));
            x = relu(conv(x, 16, 3, 2, false));
            x = relu(linear(x, 16));
        } else if (family == 2) {
            x = relu(conv(x, 32, 8, 4, false));
            x = relu(conv(x, 64, 4, 2, false));
            x = relu(conv(x, 64, 3, 1, false));
            x = relu(linear(x, 128));
        } else if (family == 0 || family == 7) {
            for (int channels : {16, 32, 32}) {
                x = conv(x, channels, 3, 1, true);
                x = pool(x, false, false);
                for (int block = 0; block < 2; block++) {
                    int skip = x;
                    x = conv(relu(x), channels, 3, 1, true);
                    x = conv(relu(x), channels, 3, 1, true);
                    RefNode n = nodes[x];
                    n.op = REF_ADD; n.a = x; n.b = skip;
                    nodes.push_back(n); x = nodes.size() - 1;
                }
            }
            x = relu(x);
            if (family == 7) x = pool(x, true, true);
            x = relu(linear(x, 128));
        } else {
            assert(family == 4);
            int depth = dict_get(policy, "cnn_depth");
            for (int stage = 1; stage <= depth; stage++) {
                char key[64];
                snprintf(key, sizeof(key), "cnn_channels_%d", stage);
                int channels = dict_get(policy, key);
                snprintf(key, sizeof(key), "cnn_kernel_%d", stage);
                int kernel = dict_get(policy, key);
                snprintf(key, sizeof(key), "cnn_stride_%d", stage);
                int stride = dict_get(policy, key);
                x = relu(conv(x, channels, kernel, stride, true));
                snprintf(key, sizeof(key), "cnn_residual_%d", stage);
                if (dict_get(policy, key)) {
                    int skip = x;
                    x = conv(x, channels, 3, 1, true);
                    RefNode n = nodes[x];
                    n.op = REF_ADD; n.a = x; n.b = skip;
                    nodes.push_back(n); x = relu(nodes.size() - 1);
                }
                snprintf(key, sizeof(key), "cnn_pool_%d", stage);
                int p = dict_get(policy, key);
                if (p) x = pool(x, p == 2, false);
            }
            if (dict_get(policy, "cnn_global_pool")) x = pool(x, true, true);
            int projection = dict_get(policy, "cnn_projection");
            x = relu(linear(x, projection));
            if (projection != 128) x = relu(linear(x, 128));
        }
        int features = nodes[x].batch * nodes[x].h * nodes[x].w * nodes[x].c / B;
        nodes.push_back({REF_GEOMETRY, x, -1, B, 1, 1, features + 4});
        relu(linear(nodes.size() - 1, 128));
        for (auto& n : nodes) {
            int count = n.batch * n.h * n.w * n.c;
            n.y = storage.add(count); n.g = storage.add(count);
            n.winner = NULL;
            if (n.op == REF_POOL) checked(cudaMalloc(&n.winner, count * sizeof(int)));
        }
    }
    ~PackedReference() {
        for (auto& n : nodes) if (n.winner) cudaFree(n.winner);
    }
    int conv(int x, int co, int k, int stride, bool same, bool flat = false) {
        RefNode p = nodes[x];
        int H = flat ? 1 : p.h, W = flat ? 1 : p.w, C = flat ? p.h * p.w * p.c : p.c;
        int oh = same ? (H + stride - 1) / stride : (H - k) / stride + 1;
        int ow = same ? (W + stride - 1) / stride : (W - k) / stride + 1;
        int ph = same ? max((oh - 1) * stride + k - H, 0) / 2 : 0;
        int pw = same ? max((ow - 1) * stride + k - W, 0) / 2 : 0;
        Layer l = {H, W, C, oh, ow, co, k, stride, ph, pw, parameters};
        parameters += co * (k * k * C + 1);
        nodes.push_back({REF_CONV, x, -1, p.batch, oh, ow, co, l});
        return nodes.size() - 1;
    }
    int linear(int x, int co) { return conv(x, co, 1, 1, false, true); }
    int relu(int x) {
        RefNode n = nodes[x]; n.op = REF_RELU; n.a = x; n.b = -1;
        nodes.push_back(n); return nodes.size() - 1;
    }
    int pool(int x, bool average, bool global) {
        RefNode n = nodes[x]; n.op = average ? REF_AVG : REF_POOL; n.a = x; n.b = global;
        n.h = global ? 1 : (n.h + 1) / 2; n.w = global ? 1 : (n.w + 1) / 2;
        nodes.push_back(n); return nodes.size() - 1;
    }
    void run(const double* obs, const double* weights, const double* up, double* dw) {
        int count = batch * 55296;
        packed_ref_input<<<(count + 127) / 128,128>>>(obs, nodes[0].y, batch, tile);
        for (auto& n : nodes) checked(cudaMemset(n.g, 0, n.batch * n.h * n.w * n.c * sizeof(double)));
        for (int i = 1; i < (int)nodes.size(); i++) {
            auto& n = nodes[i]; auto& p = nodes[n.a];
            int size = n.batch * n.h * n.w * n.c;
            if (n.op == REF_CONV) {
                packed_ref_conv<<<(size+127)/128,128>>>(p.y, weights, n.y, n.layer, n.batch);
            } else if (n.op == REF_RELU || n.op == REF_ADD) {
                packed_ref_element<<<(size+127)/128,128>>>(p.y,
                    n.op == REF_ADD ? nodes[n.b].y : NULL, n.y, size, n.op);
            } else if (n.op == REF_POOL || n.op == REF_AVG) {
                packed_ref_pool<<<(size+127)/128,128>>>(p.y,n.y,n.winner,p.batch,p.h,p.w,p.c,
                    n.op == REF_AVG,n.b);
            } else {
                packed_ref_geometry<<<(size+127)/128,128>>>(p.y,obs,n.y,batch,n.c-4);
            }
        }
        checked(cudaMemcpy(nodes.back().g, up, batch * 128 * sizeof(double), cudaMemcpyDeviceToDevice));
        for (int i = nodes.size() - 1; i > 0; i--) {
            auto& n = nodes[i]; auto& p = nodes[n.a];
            int size = p.batch * p.h * p.w * p.c;
            if (n.op == REF_CONV) {
                Layer l = n.layer;
                int nw = l.co * l.k * l.k * l.ci;
                weight_backward<<<(nw+127)/128,128>>>(p.y,n.g,dw,l,n.batch);
                bias_backward<<<(l.co+127)/128,128>>>(n.g,dw,l,n.batch);
                if (n.a) {
                    double* dx = storage.add(size);
                    input_backward<<<(size+127)/128,128>>>(weights,n.g,dx,l,n.batch);
                    packed_ref_accumulate<<<(size+127)/128,128>>>(dx,NULL,p.g,size,false);
                }
            } else if (n.op == REF_RELU || n.op == REF_ADD) {
                packed_ref_accumulate<<<(size+127)/128,128>>>(n.g,p.y,p.g,size,n.op == REF_RELU);
                if (n.op == REF_ADD) {
                    packed_ref_accumulate<<<(size+127)/128,128>>>(n.g,NULL,nodes[n.b].g,size,false);
                }
            } else if (n.op == REF_POOL || n.op == REF_AVG) {
                packed_ref_pool_back<<<(size+127)/128,128>>>(n.g,n.winner,p.g,p.batch,p.h,p.w,p.c,
                    n.op == REF_AVG,n.b);
            } else {
                packed_ref_geometry_back<<<(size+127)/128,128>>>(n.g,p.g,batch,n.c-4);
            }
        }
        checked(cudaDeviceSynchronize());
    }
};

static void packed_compare(const float* actual, const double* expected, int n, const char* label) {
    double ratio = 0;
    for (int i = 0; i < n; i++) {
        assert(isfinite(actual[i]) && isfinite(expected[i]));
        double r = fabs(actual[i] - expected[i]) / (6e-5 + 8e-4 * fabs(expected[i]));
        ratio = fmax(ratio, r);
        if (r > 1) {
            fprintf(stderr,"FAIL %s index=%d native=%.12g oracle=%.12g ratio=%.6g\n",
                label,i,actual[i],expected[i],r);
            abort();
        }
    }
    printf("COMPARE %s count=%d maximum_tolerance_ratio=%.9g\n",label,n,ratio);
}

int main(int argc, char** argv) {
    assert(argc == 5 && "checks FULL.ini BATCH FIXTURE OUTPUT_DIR");
    Ini ini = {}; puf_ini_load_file(&ini, argv[1]);
    Dict* policy = puf_ini_section(&ini,"policy",0);
    int B = atoi(argv[2]), fixture = atoi(argv[3]);
    assert((B == 1 || B == 3 || B == 32) && fixture >= 0 && fixture <= 1);
    assert(mkdir(argv[4],0700) == 0);
    int family = dict_get(policy,"encoder");
    Encoder enc = {.in_dim=55300,.out_dim=128}; create_sprite_research_encoder(&enc,policy);
    void* weights = enc.create_weights(&enc);
    void* activations = calloc(1,enc.activation_size);
    Allocator params={},acts={},grads={};
    enc.reg_params(weights,&params); enc.reg_train(weights,activations,&acts,&grads,B);
    PackedReference ref(B,family,policy);
    int P = ref.parameters;
    assert(P == params.total_elems && P == grads.total_elems && params.total_bytes == P * 4L);
    cublas_init_handle(); cudaStream_t stream; checked(cudaStreamCreate(&stream));
    alloc_create(&params); alloc_create(&acts); alloc_create(&grads);
    std::vector<float> hw(P),hx(B*55300),hu(B*128),y(B*128),dw(P);
    std::vector<double> rw(P),rx(B*55300),ru(B*128),ry(B*128),rdw(P);
    for (int i=0;i<P;i++) hw[i]=fixture ? 0 : .001f + (i%23)*.00001f;
    for (int i=0;i<B*55300;i++) {
        unsigned int hash=(i+1)*2654435761u; hash^=hash>>16;
        hx[i]=(hash%1023+1)/1024.f;
    }
    for (int i=0;i<B*128;i++) hu[i]=(1+i%7)*.01f;
    for (int i=0;i<P;i++) rw[i]=hw[i];
    for (int i=0;i<B*55300;i++) rx[i]=hx[i];
    for (int i=0;i<B*128;i++) ru[i]=hu[i];
    Prec input={.shape={B,55300}},up={.shape={B,128}};
    checked(cudaMalloc(&input.data,hx.size()*4)); checked(cudaMalloc(&up.data,hu.size()*4));
    checked(cudaMemcpy(params.mem,hw.data(),P*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(input.data,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(up.data,hu.data(),hu.size()*4,cudaMemcpyHostToDevice));
    double* obs=ref.storage.add(rx.size()),*w=ref.storage.add(P),*g=ref.storage.add(ru.size());
    double* gradient=ref.storage.add(P);
    checked(cudaMemcpy(obs,rx.data(),rx.size()*8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(w,rw.data(),P*8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(g,ru.data(),ru.size()*8,cudaMemcpyHostToDevice));
    ref.run(obs,w,g,gradient);
    checked(cudaMemcpy(ry.data(),ref.nodes.back().y,ry.size()*8,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(rdw.data(),gradient,P*8,cudaMemcpyDeviceToHost));
    Prec output;
    auto run=[&]() { output=enc.forward(weights,activations,input,stream);
        enc.backward(weights,activations,up,stream); };
    run(); checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(y.data(),output.data,y.size()*4,cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(dw.data(),grads.mem,P*4,cudaMemcpyDeviceToHost));
    packed_compare(y.data(),ry.data(),y.size(),"output");
    packed_compare(dw.data(),rdw.data(),P,"all-parameter-gradients");
    cudaGraph_t graph; cudaGraphExec_t execution;
    checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); run();
    checked(cudaStreamEndCapture(stream,&graph)); checked(cudaGraphInstantiate(&execution,graph,0));
    for (int repeat=0;repeat<2;repeat++) {
        checked(cudaGraphLaunch(execution,stream)); checked(cudaDeviceSynchronize());
        std::vector<float> again(y.size()),again_dw(P),again_w(P);
        checked(cudaMemcpy(again.data(),output.data,y.size()*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(again_dw.data(),grads.mem,P*4,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(again_w.data(),params.mem,P*4,cudaMemcpyDeviceToHost));
        assert(!memcmp(y.data(),again.data(),y.size()*4));
        assert(!memcmp(dw.data(),again_dw.data(),P*4));
        assert(!memcmp(hw.data(),again_w.data(),P*4));
    }
    const char* names[]={"weights.f32","input.f32","upstream.f32","output.f32","gradient.f32",
        "output-reference.f64","gradient-reference.f64"};
    const void* data[]={hw.data(),hx.data(),hu.data(),y.data(),dw.data(),ry.data(),rdw.data()};
    size_t bytes[]={P*4UL,hx.size()*4,hu.size()*4,y.size()*4,P*4UL,ry.size()*8,P*8UL};
    for (int i=0;i<7;i++) {
        char path[4096]; snprintf(path,sizeof(path),"%s/%s",argv[4],names[i]);
        FILE* file=fopen(path,"wx"); assert(file);
        assert(fwrite(data[i],1,bytes[i],file)==bytes[i]); assert(!fclose(file));
    }
    printf("PACKED CUDA PASS family=%d B=%d fixture=%d parameters=%d eager-graph-repeat\n",family,B,fixture,P);
    return 0;
}
