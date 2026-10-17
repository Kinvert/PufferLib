// Test-only CUDA oracle/wiring check. No CPU neural reference.
#define main forage_native_main
#include "../src/pufferl.cu"
#undef main
#include "native_encoder_reference.cu"

// Independently stated view sizes; production descriptors do not define these.
constexpr int REF_TILES = 18;
constexpr int REF_INPUT = REF_TILES * 3072 + 4;
constexpr int REF_FEATURES = REF_TILES * 16 + 4;
constexpr int REF_PARAMS = 14120 + 128 * (REF_FEATURES + 1);

__global__ void ref_tiles(const double* input, double* tiles, int B) {
    int b = blockIdx.x;
    for (int tile = 0; tile < REF_TILES; tile++) {
        for (int pixel = threadIdx.x; pixel < 3072; pixel += blockDim.x) {
#ifdef SPRITE_PACKED_VIEW
            int row = tile / 3 * 32 + pixel / (32 * 3);
            int column = tile % 3 * 32 + pixel / 3 % 32;
            int at = (row * 96 + column) * 3 + pixel % 3;
#else
            int at = tile * 3072 + pixel;
#endif
            tiles[(b * REF_TILES + tile) * 3072 + pixel] = input[b * REF_INPUT + at];
        }
    }
}

__global__ void ref_features(const double* tiles, const double* obs, double* out, int B) {
    int b = blockIdx.x;
    for (int c = threadIdx.x; c < REF_FEATURES; c += blockDim.x) {
        int count = REF_TILES * 16;
        out[b * REF_FEATURES + c] = c < count ? tiles[b * count + c]
            : obs[b * REF_INPUT + REF_TILES * 3072 + c - count];
    }
}

__global__ void ref_tile_grad(const double* packed, double* tiles, int B) {
    int b = blockIdx.x;
    for (int c = threadIdx.x; c < REF_TILES * 16; c += blockDim.x) {
        tiles[b * REF_TILES * 16 + c] = packed[b * REF_FEATURES + c];
    }
}

static void oracle(int B, const double* input, const double* weights, const double* up,
        double* out, double* gradient) {
    // Independently stated topology/offsets, not production descriptors.
    Layer layer[] = {
        {32,32,3,15,15,8,4,2,0,0,0},
        {15,15,8,7,7,16,3,2,0,0,392},
        {1,1,784,1,1,16,1,1,0,0,1560},
        {1,1,REF_FEATURES,1,1,128,1,1,0,0,14120}
    };
    Storage mem;
    double* w = mem.add(REF_PARAMS);
    double* obs = mem.add(B * REF_INPUT);
    double* g = mem.add(B * 128);
    checked(cudaMemcpy(w, weights, REF_PARAMS*sizeof(double), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(obs, input, B*REF_INPUT*sizeof(double), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(g, up, B*128*sizeof(double), cudaMemcpyHostToDevice));
    double* x[5];
    x[0] = mem.add(B*REF_TILES*3072);
    ref_tiles<<<B,128>>>(obs,x[0],B);
    for (int i = 0; i < 3; i++) {
        Layer l = layer[i];
        int n = B*REF_TILES*l.oh*l.ow*l.co;
        x[i+1] = mem.add(n);
        forward<<<(n+127)/128,128>>>(x[i],w,x[i+1],l,B*REF_TILES);
    }
    double* features = mem.add(B*REF_FEATURES);
    ref_features<<<B,128>>>(x[3],obs,features,B);
    double* y = mem.add(B*128);
    forward<<<(B*128+127)/128,128>>>(features,w,y,layer[3],B);
    double* delta = mem.add(B*128);
    relu_backward<<<(B*128+127)/128,128>>>(y,g,delta,B*128);
    double* dw = mem.add(REF_PARAMS);
    weight_backward<<<(128*REF_FEATURES+127)/128,128>>>(features,delta,dw,layer[3],B);
    bias_backward<<<1,128>>>(delta,dw,layer[3],B);
    double* df = mem.add(B*REF_FEATURES);
    input_backward<<<(B*REF_FEATURES+127)/128,128>>>(w,delta,df,layer[3],B);
    g = mem.add(B*REF_TILES*16);
    ref_tile_grad<<<B,128>>>(df,g,B);
    for (int i = 2; i >= 0; i--) {
        Layer l = layer[i];
        int n = B*REF_TILES*l.oh*l.ow*l.co;
        delta = mem.add(n);
        relu_backward<<<(n+127)/128,128>>>(x[i+1],g,delta,n);
        int nw = l.co*l.k*l.k*l.ci;
        weight_backward<<<(nw+127)/128,128>>>(x[i],delta,dw,l,B*REF_TILES);
        bias_backward<<<(l.co+127)/128,128>>>(delta,dw,l,B*REF_TILES);
        if (i) {
            int nx = B*REF_TILES*l.h*l.w*l.ci;
            g = mem.add(nx);
            input_backward<<<(nx+127)/128,128>>>(w,delta,g,l,B*REF_TILES);
        }
    }
    checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(out,y,B*128*sizeof(double),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(gradient,dw,REF_PARAMS*sizeof(double),cudaMemcpyDeviceToHost));
}

static void compare(const float* actual, const double* expected, int n) {
    for (int i = 0; i < n; i++) {
        assert(isfinite(actual[i]) && isfinite(expected[i]));
        double tolerance = 6e-5 + 8e-4*fabs(expected[i]);
        if (fabs(actual[i]-expected[i]) > tolerance) {
            fprintf(stderr,"Attention mismatch i=%d actual=%.9g expected=%.12g tolerance=%.9g\n",
                i,actual[i],expected[i],tolerance);
            abort();
        }
    }
}

int main(void) {
    cublas_init_handle();
    cudaStream_t stream;
    checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    assert(cnnref_selftest() == 0);
    for (int B : {1,3,8,64,256}) {
        Encoder enc = {.in_dim = SPRITE_VIEW_OBS, .out_dim = 128};
        create_forage_attention_encoder(&enc);
        void* w = enc.create_weights(&enc);
        ForageAttentionActivations a = {};
        Allocator params = {}, acts = {}, grads = {};
        enc.reg_params(w,&params);
        enc.reg_train(w,&a,&acts,&grads,B);
        assert(params.total_elems == REF_PARAMS && params.total_bytes == REF_PARAMS*4);
        assert(grads.total_elems == REF_PARAMS && grads.total_bytes == REF_PARAMS*4);
        alloc_create(&params); alloc_create(&acts); alloc_create(&grads);
        Prec input = {.shape = {B,REF_INPUT}}, up = {.shape = {B,128}};
        checked(cudaMalloc(&input.data,B*REF_INPUT*sizeof(float)));
        checked(cudaMalloc(&up.data,B*128*sizeof(float)));
        std::vector<float> hw(REF_PARAMS), hx(B*REF_INPUT), hu(B*128), y(B*128), dw(REF_PARAMS);
        std::vector<double> rw(REF_PARAMS), rx(B*REF_INPUT), ru(B*128), ry(B*128), rdw(REF_PARAMS);
        for (int state = 0; state < 2; state++) {
            for (int i = 0; i < REF_PARAMS; i++) hw[i] = state ? 0 : .002f + (i%23)*.00001f;
            for (int b = 0; b < B; b++) for (int i = 0; i < REF_INPUT; i++) {
                hx[b*REF_INPUT+i] = (1 + (b*31+i/3072*13+i%19)%127)/128.f;
            }
            for (int i = 0; i < B*128; i++) hu[i] = (1+i%7)*.01f;
            for (int i = 0; i < REF_PARAMS; i++) rw[i] = hw[i];
            for (int i = 0; i < B*REF_INPUT; i++) rx[i] = hx[i];
            for (int i = 0; i < B*128; i++) ru[i] = hu[i];
            checked(cudaMemcpy(params.mem,hw.data(),REF_PARAMS*4,cudaMemcpyHostToDevice));
            checked(cudaMemcpy(input.data,hx.data(),B*REF_INPUT*4,cudaMemcpyHostToDevice));
            checked(cudaMemcpy(up.data,hu.data(),B*128*4,cudaMemcpyHostToDevice));
            oracle(B,rx.data(),rw.data(),ru.data(),ry.data(),rdw.data());
            auto run = [&]() {
                enc.forward(w,&a,input,stream);
                enc.backward(w,&a,up,stream);
            };
            run();
            checked(cudaDeviceSynchronize());
            checked(cudaMemcpy(y.data(),a.output.data,B*128*4,cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(dw.data(),grads.mem,REF_PARAMS*4,cudaMemcpyDeviceToHost));
            compare(y.data(),ry.data(),B*128);
            compare(dw.data(),rdw.data(),REF_PARAMS);
            cudaGraph_t graph;
            checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            run();
            checked(cudaStreamEndCapture(stream,&graph));
            cudaGraphExec_t execution;
            checked(cudaGraphInstantiate(&execution,graph,0));
            for (int repeat = 0; repeat < 2; repeat++) {
                checked(cudaGraphLaunch(execution,stream));
                checked(cudaDeviceSynchronize());
                std::vector<float> again_y(B*128), again_dw(REF_PARAMS), again_weights(REF_PARAMS);
                checked(cudaMemcpy(again_y.data(),a.output.data,B*128*4,cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(again_dw.data(),grads.mem,REF_PARAMS*4,cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(again_weights.data(),params.mem,REF_PARAMS*4,cudaMemcpyDeviceToHost));
                assert(!memcmp(y.data(),again_y.data(),B*128*4));
                assert(!memcmp(dw.data(),again_dw.data(),REF_PARAMS*4));
                assert(!memcmp(hw.data(),again_weights.data(),REF_PARAMS*4));
            }
            cudaGraphExecDestroy(execution); cudaGraphDestroy(graph);
            printf("Attention CUDA PASS B=%d tiles=%d fixture=%s output/all%dgradients/weights/eager-graph-repeat\n",
                B,REF_TILES,state ? "zero-weights" : "smooth-positive",REF_PARAMS);
        }
        cudaFree(params.mem); cudaFree(acts.mem); cudaFree(grads.mem);
        cudaFree(input.data); cudaFree(up.data);
        free(params.regs); free(acts.regs); free(grads.regs); free(w);
    }
}
