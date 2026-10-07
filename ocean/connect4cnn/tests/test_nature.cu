// Float32 numerical harness for the actual native Nature encoder.
#include <vector>
#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define C4_NATURE_CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#include "src/pufferl.cu"

#ifdef C4_DENSE_PATCH_ALIAS
#include "ocean/connect4cnn/dense_patch_alias.cu"
#endif

#define CUDA_CHECK(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); abort(); } \
} while (0)

static Encoder enc;
static NatureWeights* weights;
static NatureActivations train_acts, rollout_acts;
static Allocator params, acts, grads, rollout;
static Prec input, upstream;
static cudaStream_t test_stream;

extern "C" int naturetest_alias_enabled() {
#ifdef C4_DENSE_PATCH_ALIAS
    return 1;
#else
    return 0;
#endif
}

static int test_init(int B, int hidden, Dict* policy = NULL) {
    cublas_init_handle();
    CUDA_CHECK(cudaStreamCreate(&test_stream));
    enc.in_dim = OBS_SIZE; enc.out_dim = hidden;
    create_custom_encoder(&enc, policy);
    assert(enc.forward == nature_forward || enc.forward == flex_forward);
#ifdef C4_DENSE_PATCH_ALIAS
    assert(dense_patch_alias_select(&enc));
#endif
    weights = (NatureWeights*)enc.create_weights(&enc);
    enc.reg_params(weights, &params);
    enc.reg_train(weights, &train_acts, &acts, &grads, B);
    enc.reg_rollout(weights, &rollout_acts, &rollout, B);
    assert(params.total_elems == grads.total_elems);
    input = {.shape = {B, OBS_SIZE}};
    upstream = {.shape = {B, hidden}};
    alloc_register(&acts, &input); alloc_register(&acts, &upstream);
    alloc_create(&params); alloc_create(&acts); alloc_create(&grads); alloc_create(&rollout);
    CUDA_CHECK(cudaDeviceSynchronize());
    return params.total_elems;
}

extern "C" int naturetest_init(int B, int hidden) { return test_init(B, hidden); }

extern "C" int compacttest_init(int B, int hidden, int channels, int depth, int stride, int projection) {
    Dict policy = {};
    dict_set(&policy, "encoder", 3);
    dict_set(&policy, "cnn_channels", channels);
    dict_set(&policy, "cnn_depth", depth);
    dict_set(&policy, "cnn_stride", stride);
    dict_set(&policy, "cnn_projection", projection);
    int n = test_init(B, hidden, &policy);
    dict_clear(&policy); enc.config = NULL;
    return n;
}

extern "C" int flextest_init(int B, int hidden, int* settings) {
    Dict policy = {};
    dict_set(&policy, "encoder", 4);
    dict_set(&policy, "cnn_depth", settings[0]);
    dict_set(&policy, "cnn_projection", settings[1]);
    dict_set(&policy, "cnn_global_pool", settings[2]);
    const char* names[] = {"channels", "kernel", "stride", "pool", "residual"};
    for (int stage = 1; stage <= 3; stage++) for (int j = 0; j < 5; j++) {
        char key[64]; snprintf(key, sizeof(key), "cnn_%s_%d", names[j], stage);
        dict_set(&policy, key, settings[3 + (stage-1)*5 + j]);
    }
    int n = test_init(B, hidden, &policy);
    dict_clear(&policy); enc.config = NULL;
    return n;
}

extern "C" void naturetest_run(float* obs, float* values, float* grad,
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
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(capture));
    }
    Prec actor = enc.forward(weights, &rollout_acts, input, test_stream);
    CUDA_CHECK(cudaStreamSynchronize(test_stream));
    std::vector<float> actor_values(numel(actor.shape));
    CUDA_CHECK(cudaMemcpy(actor_values.data(), actor.data, actor_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output, train_acts.output[weights->count - 1].data, actor_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    assert(memcmp(output, actor_values.data(), actor_values.size() * sizeof(float)) == 0);
    CUDA_CHECK(cudaMemcpy(parameter_grad, grads.mem, grads.total_bytes, cudaMemcpyDeviceToHost));
}

extern "C" void naturetest_close() {
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

extern "C" void naturetest_read_params(float* values) {
    CUDA_CHECK(cudaMemcpy(values, params.mem, params.total_bytes, cudaMemcpyDeviceToHost));
}
