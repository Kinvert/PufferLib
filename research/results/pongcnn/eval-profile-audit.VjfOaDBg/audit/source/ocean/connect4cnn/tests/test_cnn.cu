// Float32 numerical harness for all allowed experimental CNN configurations.
#include <vector>
#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#include "src/pufferl.cu"
using namespace c4_cnn;

#define CUDA_CHECK(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); abort(); } \
} while (0)

static Encoder enc;
static ImpalaWeights* weights;
static ImpalaActivations train_acts, rollout_acts;
static Allocator params, acts, grads, rollout;
static Prec input, upstream;
static cudaStream_t test_stream;

extern "C" int impalatest_init(int B, int hidden, int channels, int blocks, int gap) {
    cublas_init_handle();
    CUDA_CHECK(cudaStreamCreate(&test_stream));
    enc.in_dim = OBS_SIZE; enc.out_dim = hidden;
    Dict policy = {};
    dict_set(&policy, "encoder", 1);
    dict_set(&policy, "cnn_channels", channels);
    dict_set(&policy, "cnn_blocks", blocks);
    dict_set(&policy, "cnn_global_pool", gap);
    create_custom_encoder(&enc, &policy);
    assert(enc.forward == imp_forward);
    weights = (ImpalaWeights*)enc.create_weights(&enc);
    dict_clear(&policy); enc.config = NULL;
    enc.reg_params(weights, &params);
    enc.reg_train(weights, &train_acts, &acts, &grads, B);
    enc.reg_rollout(weights, &rollout_acts, &rollout, B);
    assert(params.total_elems == grads.total_elems);
    assert(params.total_bytes == params.total_elems * sizeof(float));
    assert(grads.total_bytes == grads.total_elems * sizeof(float));
    input = {.shape = {B, OBS_SIZE}}; upstream = {.shape = {B, hidden}};
    alloc_register(&acts, &input); alloc_register(&acts, &upstream);
    alloc_create(&params); alloc_create(&acts); alloc_create(&grads); alloc_create(&rollout);
    CUDA_CHECK(cudaDeviceSynchronize());
    return params.total_elems;
}

extern "C" void impalatest_run(float* obs, float* values, float* grad,
        float* output, float* parameter_grad, int graph) {
    CUDA_CHECK(cudaMemcpy(input.data, obs, numel(input.shape) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params.mem, values, params.total_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(upstream.data, grad, numel(upstream.shape) * sizeof(float), cudaMemcpyHostToDevice));
    enc.forward(weights, &train_acts, input, test_stream);
    enc.backward(weights, &train_acts, upstream, test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    if (graph) {
        cudaGraph_t capture;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(test_stream, cudaStreamCaptureModeGlobal));
        enc.forward(weights, &train_acts, input, test_stream);
        enc.backward(weights, &train_acts, upstream, test_stream);
        CUDA_CHECK(cudaStreamEndCapture(test_stream, &capture));
        CUDA_CHECK(cudaGraphInstantiate(&executable, capture, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, test_stream));
        CUDA_CHECK(cudaStreamSynchronize(test_stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable)); CUDA_CHECK(cudaGraphDestroy(capture));
    }
    Prec actor = enc.forward(weights, &rollout_acts, input, test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    std::vector<float> actor_values(numel(actor.shape));
    CUDA_CHECK(cudaMemcpy(actor_values.data(), actor.data, actor_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, train_acts.output.data, actor_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    assert(memcmp(output, actor_values.data(), actor_values.size() * sizeof(float)) == 0);
    CUDA_CHECK(cudaMemcpy(parameter_grad, grads.mem, grads.total_bytes, cudaMemcpyDeviceToHost));
}

extern "C" void impalatest_close() {
    for (Allocator* alloc : {&params, &acts, &grads, &rollout}) {
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

extern "C" void impalatest_pool(int B, int H, int W, int C, float* x, float* g, float* y, float* dx) {
    int n = B * H * W * C, m = B * ((H + 1) / 2) * ((W + 1) / 2) * C;
    float *in, *grad, *out, *back;
    int* winners;
    CUDA_CHECK(cudaMalloc(&in, n * sizeof(float))); CUDA_CHECK(cudaMalloc(&back, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grad, m * sizeof(float))); CUDA_CHECK(cudaMalloc(&out, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&winners, m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(in, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grad, g, m * sizeof(float), cudaMemcpyHostToDevice));
    imp_pool<<<grid_size(m), BLOCK_SIZE>>>(in, out, winners, B, H, W, C);
    imp_pool_grad<<<grid_size(n), BLOCK_SIZE>>>(grad, winners, back, B, H, W, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y, out, m * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dx, back, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(in)); CUDA_CHECK(cudaFree(back)); CUDA_CHECK(cudaFree(grad));
    CUDA_CHECK(cudaFree(out)); CUDA_CHECK(cudaFree(winners));
}
