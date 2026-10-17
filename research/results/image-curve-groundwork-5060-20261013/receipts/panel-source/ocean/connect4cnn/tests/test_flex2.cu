// Actual GPU encoder-5 forward/backward; host code only supplies fixtures.
#include <vector>
#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#include "src/pufferl.cu"

#define CUDA_CHECK(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(err)); abort(); } \
} while (0)

static Encoder enc;
static Flex2Weights* weights;
static Flex2Activations train_acts, rollout_acts;
static Allocator params, acts, grads, rollout;
static Prec input, upstream;
static cudaStream_t test_stream;

// Direct probes isolate boundary behavior from matrix multiplication rounding.
__global__ void flex2test_function_kernel(const float* x, const float* p,
        float* y, float* dx, float* dp, int kind, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) y[i] = flex2_function(x[i],kind,p,&dx[i],dp+12*i);
}
extern "C" void flex2test_activation(float* x, float* p, float* y,
        float* dx, float* dp, int kind, int n) {
    float *input, *coefficients, *output, *derivative, *partials;
    CUDA_CHECK(cudaMalloc(&input,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&coefficients,12*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&derivative,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&partials,12*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input,x,n*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(coefficients,p,12*sizeof(float),cudaMemcpyHostToDevice));
    flex2test_function_kernel<<<grid_size(n),BLOCK_SIZE>>>(input,coefficients,output,derivative,partials,kind,n);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y,output,n*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dx,derivative,n*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dp,partials,12*n*sizeof(float),cudaMemcpyDeviceToHost));
    for (float* ptr : {input,coefficients,output,derivative,partials}) CUDA_CHECK(cudaFree(ptr));
}
extern "C" void flex2test_pool(float* x, float* grad, float* y, float* dx,
        int kind, int B, int H, int W, int C, int oh, int ow) {
    int ni = B*H*W*C, no = B*oh*ow*C;
    float *input, *upstream, *output, *derivative; int* winner;
    CUDA_CHECK(cudaMalloc(&input,ni*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&upstream,no*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output,no*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&derivative,ni*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&winner,no*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(input,x,ni*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(upstream,grad,no*sizeof(float),cudaMemcpyHostToDevice));
    NatureLayer d = {H,W,C,C,3,2,oh,ow};
    if (kind == 1) {
        assert(oh == (H+1)/2 && ow == (W+1)/2);
        c4_cnn::imp_pool<<<grid_size(no),BLOCK_SIZE>>>(input,output,winner,B,H,W,C);
        c4_cnn::imp_pool_grad<<<grid_size(ni),BLOCK_SIZE>>>(upstream,winner,derivative,B,H,W,C);
    } else if (kind == 2) {
        assert(oh == (H+1)/2 && ow == (W+1)/2);
        flex_average<<<grid_size(no),BLOCK_SIZE>>>(input,output,d,B,false);
        flex_average_grad<<<grid_size(ni),BLOCK_SIZE>>>(upstream,derivative,d,B,false);
    } else {
        assert(kind == 3);
        flex2_adaptive<<<grid_size(no),BLOCK_SIZE>>>(input,output,d,B);
        flex2_adaptive_grad<<<grid_size(ni),BLOCK_SIZE>>>(upstream,derivative,d,B);
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y,output,no*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dx,derivative,ni*sizeof(float),cudaMemcpyDeviceToHost));
    for (float* ptr : {input,upstream,output,derivative}) CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaFree(winner));
}

extern "C" long flex2test_init(const char* path, int B, int hidden, ulong seed) {
    Ini ini = {}; puf_ini_load_file(&ini,path);
    cublas_init_handle(); CUDA_CHECK(cudaStreamCreate(&test_stream));
    enc.in_dim = OBS_SIZE; enc.out_dim = hidden;
    create_custom_encoder(&enc,puf_ini_section(&ini,"policy",0));
    assert(enc.forward == flex2_forward);
    weights = (Flex2Weights*)enc.create_weights(&enc);
    enc.reg_params(weights,&params);
    enc.reg_train(weights,&train_acts,&acts,&grads,B);
    enc.reg_rollout(weights,&rollout_acts,&rollout,B);
    assert(params.total_bytes == grads.total_bytes);
    assert(params.total_bytes == params.total_elems*sizeof(float));
    input = {.shape={B,OBS_SIZE}}; upstream = {.shape={B,hidden}};
    alloc_register(&acts,&input); alloc_register(&acts,&upstream);
    alloc_create(&params); alloc_create(&acts); alloc_create(&grads); alloc_create(&rollout);
    enc.init_weights(weights,&seed,test_stream); CUDA_CHECK(cudaStreamSynchronize(test_stream));
    puf_ini_free(&ini); enc.config = NULL;
    return params.total_elems;
}

// Registration offsets enable weight/bias/coefficient checks at every operation.
extern "C" int flex2test_tensors(long* metadata) {
    int j = 0;
    for (int i = 0; i < weights->net.count; i++) if (!weights->type[i]) {
        Prec* tensors[] = {&weights->net.weight[i],&weights->net.bias[i],&weights->function[i]};
        for (int kind = 0; kind < (weights->activation[i] >= 3 ? 3 : 2); kind++) {
            Prec* t = tensors[kind];
            metadata[4*j] = t->data-(float*)params.mem;
            metadata[4*j+1] = numel(t->shape);
            metadata[4*j+2] = kind; metadata[4*j+3] = weights->activation[i]; j++;
        }
    }
    return j;
}
extern "C" void flex2test_params(float* values) {
    CUDA_CHECK(cudaMemcpy(values,params.mem,params.total_bytes,cudaMemcpyDeviceToHost));
}
extern "C" void flex2test_run(float* obs, float* values, float* grad,
        float* output, float* parameter_grad, int graph) {
    CUDA_CHECK(cudaMemcpy(input.data,obs,numel(input.shape)*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params.mem,values,params.total_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(upstream.data,grad,numel(upstream.shape)*sizeof(float),cudaMemcpyHostToDevice));
    enc.forward(weights,&train_acts,input,test_stream);
    enc.backward(weights,&train_acts,upstream,test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    if (graph) {
        cudaGraph_t capture; cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeGlobal));
        enc.forward(weights,&train_acts,input,test_stream);
        enc.backward(weights,&train_acts,upstream,test_stream);
        CUDA_CHECK(cudaStreamEndCapture(test_stream,&capture));
        CUDA_CHECK(cudaGraphInstantiate(&executable,capture,0));
        CUDA_CHECK(cudaGraphLaunch(executable,test_stream));
        CUDA_CHECK(cudaStreamSynchronize(test_stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable)); CUDA_CHECK(cudaGraphDestroy(capture));
    }
    Prec actor = enc.forward(weights,&rollout_acts,input,test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    std::vector<float> actor_values(numel(actor.shape));
    CUDA_CHECK(cudaMemcpy(actor_values.data(),actor.data,actor_values.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output,train_acts.net.output[weights->net.count-1].data,actor_values.size()*sizeof(float),cudaMemcpyDeviceToHost));
    assert(memcmp(output,actor_values.data(),actor_values.size()*sizeof(float)) == 0);
    CUDA_CHECK(cudaMemcpy(parameter_grad,grads.mem,grads.total_bytes,cudaMemcpyDeviceToHost));
}
extern "C" void flex2test_close() {
    for (Allocator* alloc : {&params,&acts,&grads,&rollout}) {
        CUDA_CHECK(cudaFree(alloc->mem)); free(alloc->regs); *alloc = {};
    }
    free(weights); weights = NULL; train_acts = {}; rollout_acts = {};
    CUDA_CHECK(cudaStreamDestroy(test_stream));
    cublasDestroy(g_cublas_handle); cublasDestroy(g_cublas_dw_handle);
    CUDA_CHECK(cudaFree(g_cublas_workspace)); CUDA_CHECK(cudaFree(g_cublas_dw_workspace));
    CUDA_CHECK(cudaEventDestroy(g_main_ready)); CUDA_CHECK(cudaEventDestroy(g_dw_done));
    CUDA_CHECK(cudaStreamDestroy(g_dw_stream));
    g_cublas_handle = NULL; g_cublas_dw_handle = NULL;
}
