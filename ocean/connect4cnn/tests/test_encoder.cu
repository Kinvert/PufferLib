// Float32 numerical harness using the actual native encoder and allocator.
#include <vector>
#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#include "src/pufferl.cu"

static Encoder enc;
static Connect4EncoderWeights* weights;
static Connect4EncoderActivations train_acts, rollout_acts;
static Allocator params, acts, grads, rollout;
static Prec input, upstream;
static cudaStream_t test_stream;

#define CUDA_CHECK(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); abort(); } \
} while (0)

extern "C" int c4test_init(int B, int hidden) {
    int devices = 0;
    cudaError_t err = cudaGetDeviceCount(&devices);
    if (err != cudaSuccess || devices == 0) {
        fprintf(stderr, "GPU unavailable: %s (devices=%d)\n", cudaGetErrorString(err), devices);
        return 0;
    }
    cublas_init_handle();
    CUDA_CHECK(cudaStreamCreate(&test_stream));
    enc.in_dim = OBS_SIZE;
    enc.out_dim = hidden;
    create_custom_encoder(&enc);
    assert(enc.forward == c4_encoder_forward);
    weights = (Connect4EncoderWeights*)enc.create_weights(&enc);
    enc.reg_params(weights, &params);
    enc.reg_train(weights, &train_acts, &acts, &grads, B);
    enc.reg_rollout(weights, &rollout_acts, &rollout, B);
    assert(params.total_elems == grads.total_elems);
    assert(params.total_bytes == grads.total_bytes);
    input.shape[0] = B; input.shape[1] = OBS_SIZE;
    upstream.shape[0] = B; upstream.shape[1] = hidden;
    alloc_register(&acts, &input);
    alloc_register(&acts, &upstream);
    alloc_create(&params); alloc_create(&acts);
    alloc_create(&grads); alloc_create(&rollout);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 1;
}

extern "C" void c4test_run(float* obs, float* conv_w, float* proj_w, float* grad,
        float* output, float* conv_grad, float* proj_grad, int graph) {
    CUDA_CHECK(cudaMemcpy(input.data, obs, numel(input.shape) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->conv_w.data, conv_w, numel(weights->conv_w.shape) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights->proj_w.data, proj_w, numel(weights->proj_w.shape) * sizeof(float), cudaMemcpyHostToDevice));
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
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(capture));
    }
    Prec out = enc.forward(weights, &rollout_acts, input, test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    std::vector<float> actor(numel(out.shape));
    CUDA_CHECK(cudaMemcpy(actor.data(), out.data, actor.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, train_acts.out.data, numel(train_acts.out.shape) * sizeof(float), cudaMemcpyDeviceToHost));
    assert(memcmp(output, actor.data(), actor.size() * sizeof(float)) == 0);
    CUDA_CHECK(cudaMemcpy(conv_grad, train_acts.conv_wgrad.data, numel(weights->conv_w.shape) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(proj_grad, train_acts.proj_wgrad.data, numel(weights->proj_w.shape) * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" void c4test_close() {
    for (Allocator* alloc : {&params, &acts, &grads, &rollout}) {
        CUDA_CHECK(cudaFree(alloc->mem));
        free(alloc->regs);
        *alloc = {};
    }
    free(weights);
    weights = NULL;
    train_acts = {}; rollout_acts = {};
    CUDA_CHECK(cudaStreamDestroy(test_stream));
    cublasDestroy(g_cublas_handle); cublasDestroy(g_cublas_dw_handle);
    CUDA_CHECK(cudaFree(g_cublas_workspace));
    CUDA_CHECK(cudaFree(g_cublas_dw_workspace));
    CUDA_CHECK(cudaEventDestroy(g_main_ready));
    CUDA_CHECK(cudaEventDestroy(g_dw_done));
    CUDA_CHECK(cudaStreamDestroy(g_dw_stream));
    g_cublas_handle = NULL; g_cublas_dw_handle = NULL;
}
