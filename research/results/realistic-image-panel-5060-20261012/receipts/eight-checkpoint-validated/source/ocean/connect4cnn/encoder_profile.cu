// Native fixed-encoder metadata and scheduled CUDA profiling. No trainer edits.
#include <vector>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

static cublasStatus_t real_stream(cublasHandle_t handle, cudaStream_t stream) {
    return cublasSetStream(handle, stream);
}

static bool reassign_workspace;
static long stream_calls, dw_stream_calls, workspace_calls, gemm_calls;
static cublasStatus_t profile_stream(cublasHandle_t, cudaStream_t);
template<class... Args> static cublasStatus_t profile_gemm(Args... args) {
    cublasStatus_t status = cublasGemmEx(args...);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM failed: %d\n", (int)status); exit(1);
    }
    gemm_calls++;
    return status;
}

#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
// Instrument this executable only. Headers above retain the real API declarations.
#undef cublasSetStream
#define cublasSetStream profile_stream
#define cublasGemmEx profile_gemm
#include "src/pufferl.cu"
#undef cublasSetStream
#undef cublasGemmEx

#ifdef C4_DENSE_PATCH_ALIAS
#include "dense_patch_alias.cu"
#endif

#define GPU_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); exit(1); } } while (0)

static void require(bool condition, const char* message) {
    if (!condition) { fprintf(stderr, "%s\n", message); exit(1); }
}

static cudaStream_t bound_main, bound_dw;
static cublasStatus_t profile_stream(cublasHandle_t handle, cudaStream_t stream) {
    // Never reuse a handle/workspace across concurrent profiler streams.
    bool main = handle == g_cublas_handle;
    require(main || handle == g_cublas_dw_handle, "Unknown cuBLAS handle");
    cudaStream_t* bound = main ? &bound_main : &bound_dw;
    require(stream != NULL && (!*bound || *bound == stream), "Handle changed stream");
    *bound = stream;
    cublasStatus_t status = real_stream(handle, stream);
    require(status == CUBLAS_STATUS_SUCCESS, "cuBLAS stream assignment failed");
    stream_calls++;
    if (!main) dw_stream_calls++;
    if (reassign_workspace) {
        void* workspace = main ? g_cublas_workspace : g_cublas_dw_workspace;
        require(workspace != NULL, "Missing explicit workspace");
        status = cublasSetWorkspace(handle, workspace, 32 * 1024 * 1024);
        require(status == CUBLAS_STATUS_SUCCESS, "cuBLAS workspace assignment failed");
        workspace_calls++;
    }
    return status;
}

static int integer(const char* text, int low, int high) {
    char* end; errno = 0;
    long value = strtol(text, &end, 10);
    require(!errno && text[0] && !*end && value >= low && value <= high, "Invalid integer");
    return value;
}

static long payload(const Allocator& a) {
    long bytes = 0;
    for (int i = 0; i < a.num_regs; i++) bytes += numel(a.regs[i].shape) * a.regs[i].elem_size;
    return bytes;
}

static const char* fixed_model(Encoder* enc, Dict* policy) {
    int type = (int)dict_get(policy, "encoder");
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
    require(type == 0, "Residual profiler requires encoder=0 in matching INI");
#ifdef C4_IMPOOLA_CNN
    return "impoola_cnn";
#else
    return "impala_cnn";
#endif
#else
    if (type == 2) return "nature_cnn";
    require(type == 4 && enc->forward == flex_forward, "Only fixed quality/Nature supported");
    const char* keys[] = {"cnn_depth", "cnn_projection", "cnn_global_pool", "cnn_channels_1",
        "cnn_kernel_1", "cnn_stride_1", "cnn_pool_1", "cnn_residual_1"};
    int values[] = {1, 64, 0, 16, 7, 4, 0, 0};
    for (int i = 0; i < 8; i++) require(dict_get(policy, keys[i]) == values[i], "Not the locked quality graph");
    return "flex_quality";
#endif
}

static uint64_t hash_bytes(const std::vector<unsigned char>& bytes) {
    uint64_t value = UINT64_C(14695981039346656037);
    for (unsigned char byte : bytes) { value ^= byte; value *= UINT64_C(1099511628211); }
    return value;
}

static void write_receipt(const char* directory, const char* name, const void* data, size_t bytes) {
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%s", directory, name);
    require(n > 0 && n < sizeof(path), "Receipt path too long");
    FILE* file = fopen(path, "wx");
    require(file != NULL, "Receipt file exists or directory is unavailable");
    require(fwrite(data, 1, bytes, file) == bytes, "Receipt write failed");
    require(fclose(file) == 0, "Receipt close failed");
}

static std::vector<unsigned char> snapshot(Prec output, const Allocator* grads) {
    std::vector<unsigned char> result(numel(output.shape) * sizeof(float));
    GPU_CHECK(cudaMemcpy(result.data(), output.data, result.size(), cudaMemcpyDeviceToHost));
    if (grads) for (int i = 0; i < grads->num_regs; i++) {
        const AllocEntry& entry = grads->regs[i];
        require(entry.elem_size == sizeof(float), "Unexpected gradient element type");
        size_t start = result.size(), bytes = numel(entry.shape) * entry.elem_size;
        result.resize(start + bytes);
        GPU_CHECK(cudaMemcpy(result.data() + start, *entry.data_ptr, bytes, cudaMemcpyDeviceToHost));
    }
    for (size_t offset = 0; offset < result.size(); offset += sizeof(float)) {
        float value; memcpy(&value, result.data() + offset, sizeof(float));
        require(isfinite(value), "Nonfinite output/parameter gradient");
    }
    return result;
}

int main(int argc, char** argv) {
    // Artifact hashing only: return before INI parsing, CUDA or encoder construction.
    if (argc == 3 && !strcmp(argv[1], "--hash")) {
        FILE* file = fopen(argv[2], "rb");
        require(file != NULL, "Cannot read receipt");
        unsigned char bytes[4096]; size_t count;
        uint64_t value = UINT64_C(14695981039346656037);
        while ((count = fread(bytes, 1, sizeof(bytes), file))) {
            for (size_t i = 0; i < count; i++) { value ^= bytes[i]; value *= UINT64_C(1099511628211); }
        }
        require(!ferror(file), "Receipt read failed");
        require(fclose(file) == 0, "Receipt close failed");
        printf("%016llx\n", (unsigned long long)value);
        return 0;
    }
    bool describe = argc == 4 && !strcmp(argv[1], "--describe");
    bool run = argc == 10 && !strcmp(argv[1], "--run");
    if (!describe && !run) {
        fprintf(stderr, "Usage: %s --describe FULL.ini BATCH\n"
            "       %s --run FULL.ini BATCH rollout|learner eager|graph default|reassign WARMUP SAMPLES RECEIPT_DIR\n", argv[0], argv[0]);
        return 1;
    }
    // Large host descriptors support learner memory audits without enabling
    // larger unvalidated CUDA executions in the scheduled profiler.
    int B = integer(argv[3], 1, describe ? 65536 : 2048);
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    Dict* policy = puf_ini_section(&ini, "policy", 0);
    double hidden_value = dict_get(policy, "hidden_size");
    require(isfinite(hidden_value) && hidden_value == floor(hidden_value)
        && hidden_value >= 8 && hidden_value <= 4096 && (int)hidden_value % 8 == 0, "Invalid hidden size");
    double type = dict_get(policy, "encoder");
    require(isfinite(type) && (type == 0 || type == 2 || type == 4), "Unqualified encoder ID");
    Encoder enc = {}; enc.in_dim = OBS_SIZE; enc.out_dim = hidden_value;
    create_custom_encoder(&enc, policy);
    const char* model = fixed_model(&enc, policy);
#ifdef C4_DENSE_PATCH_ALIAS
    bool alias_active = dense_patch_alias_select(&enc);
#endif
    void* weights = enc.create_weights(&enc);
    void* train = calloc(1, enc.activation_size);
    void* actor = calloc(1, enc.activation_size);
    require(weights && train && actor, "Host metadata allocation failed");
    Allocator params = {}, acts = {}, grads = {}, rollout = {};
    enc.reg_params(weights, &params);
    enc.reg_train(weights, train, &acts, &grads, B);
    enc.reg_rollout(weights, actor, &rollout, B);
    require(params.total_elems == grads.total_elems && payload(params) == payload(grads), "Parameter/gradient count mismatch");
    // Until this branch returns, no CUDA allocation, initialization or policy executes.
    if (describe) {
        printf("{");
#ifdef C4_DENSE_PATCH_ALIAS
        dense_patch_alias_receipt(&enc, weights, B, alias_active);
#endif
        printf("\"schema\":1,\"model\":\"%s\",\"batch\":%d,\"hidden\":%d,"
            "\"encoder_parameters\":%ld,\"parameter_payload_bytes\":%ld,"
            "\"rollout_payload_bytes\":%ld,\"rollout_allocator_bytes\":%ld,"
            "\"train_payload_bytes\":%ld,\"train_allocator_bytes\":%ld,"
            "\"receipt_exports\":true,\"host_hash\":true,\"cuda_executed\":false,\"policy_executed\":false}\n",
            model, B, enc.out_dim, params.total_elems, payload(params), payload(rollout),
            rollout.total_bytes, payload(acts), acts.total_bytes);
    } else {
        require(!strcmp(argv[4], "rollout") || !strcmp(argv[4], "learner"), "Invalid phase");
        require(!strcmp(argv[5], "eager") || !strcmp(argv[5], "graph"), "Invalid execution mode");
        require(!strcmp(argv[6], "default") || !strcmp(argv[6], "reassign"), "Invalid workspace mode");
        bool learner = !strcmp(argv[4], "learner"), graph = !strcmp(argv[5], "graph");
        reassign_workspace = !strcmp(argv[6], "reassign");
        int warmup = integer(argv[7], 5, 1000), samples = integer(argv[8], 5, 10000);
        cudaStream_t stream; GPU_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cublas_init_handle();
        require(g_cublas_handle && g_cublas_dw_handle && g_cublas_workspace && g_cublas_dw_workspace, "cuBLAS initialization failed");
        Prec input = {.shape = {B, OBS_SIZE}}, upstream = {.shape = {B, enc.out_dim}};
        Allocator io = {};
        alloc_register(&io, &input); alloc_register(&io, &upstream);
        alloc_create(&params); alloc_create(&io);
        if (learner) { alloc_create(&acts); alloc_create(&grads); }
        else alloc_create(&rollout);
        ulong seed = 173;
        enc.init_weights(weights, &seed, stream);
        std::vector<float> obs(numel(input.shape)), grad(numel(upstream.shape));
        // Exactly representable nonzero synthetic inputs; no game or CPU model.
        for (size_t i = 0; i < obs.size(); i++) obs[i] = ((i * 37 + 11) % 256) / 256.0f;
        for (size_t i = 0; i < grad.size(); i++) grad[i] = ((int)((i * 17 + 3) % 127) - 63) / 128.0f;
        GPU_CHECK(cudaMemcpy(input.data, obs.data(), obs.size() * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(upstream.data, grad.data(), grad.size() * sizeof(float), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> initial_params(params.total_bytes);
        GPU_CHECK(cudaMemcpy(initial_params.data(), params.mem, params.total_bytes, cudaMemcpyDeviceToHost));
        Prec output = {};
        auto step = [&]() {
            output = enc.forward(weights, learner ? train : actor, input, stream);
            if (learner) enc.backward(weights, train, upstream, stream);
            GPU_CHECK(cudaGetLastError());
        };
        step(); GPU_CHECK(cudaStreamSynchronize(stream));
        auto reference = snapshot(output, learner ? &grads : NULL);
        cudaGraph_t capture = NULL; cudaGraphExec_t executable = NULL;
        if (graph) {
            GPU_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            step(); GPU_CHECK(cudaStreamEndCapture(stream, &capture));
            GPU_CHECK(cudaGraphInstantiate(&executable, capture, 0));
        }
        auto execute = [&]() { if (graph) GPU_CHECK(cudaGraphLaunch(executable, stream)); else step(); };
        for (int i = 0; i < warmup; i++) execute();
        GPU_CHECK(cudaStreamSynchronize(stream));
        require(reference == snapshot(output, learner ? &grads : NULL), "Eager/mode/warmup byte mismatch");
        cudaEvent_t start, finish;
        GPU_CHECK(cudaEventCreate(&start)); GPU_CHECK(cudaEventCreate(&finish));
        std::vector<float> milliseconds(samples);
        long calls_before = gemm_calls;
        for (int i = 0; i < samples; i++) {
            GPU_CHECK(cudaEventRecord(start, stream)); execute();
            GPU_CHECK(cudaEventRecord(finish, stream)); GPU_CHECK(cudaEventSynchronize(finish));
            GPU_CHECK(cudaEventElapsedTime(&milliseconds[i], start, finish));
            require(isfinite(milliseconds[i]) && milliseconds[i] > 0, "Invalid CUDA event duration");
        }
        require(reference == snapshot(output, learner ? &grads : NULL), "Repeatability byte mismatch");
        double sum = 0; for (float ms : milliseconds) sum += ms;
        std::vector<unsigned char> param_bytes(params.total_bytes);
        GPU_CHECK(cudaMemcpy(param_bytes.data(), params.mem, params.total_bytes, cudaMemcpyDeviceToHost));
        require(initial_params == param_bytes, "Profiling mutated encoder parameters");
        // Outside event timing; retain bytes for SHA256 and exact cross-process audits.
        write_receipt(argv[9], "input.f32", obs.data(), obs.size() * sizeof(float));
        write_receipt(argv[9], "upstream.f32", grad.data(), grad.size() * sizeof(float));
        write_receipt(argv[9], "parameters.f32", param_bytes.data(), param_bytes.size());
        write_receipt(argv[9], "output-grad.f32", reference.data(), reference.size());
        printf("{");
#ifdef C4_DENSE_PATCH_ALIAS
        dense_patch_alias_receipt(&enc, weights, B, alias_active);
#endif
        printf("\"schema\":1,\"model\":\"%s\",\"batch\":%d,\"hidden\":%d,"
            "\"phase\":\"%s\",\"mode\":\"%s\",\"workspace\":\"%s\","
            "\"seed\":173,\"warmup\":%d,\"samples\":%d,\"encoder_parameters\":%ld,"
            "\"input_fnv64\":\"%016llx\",\"output_grad_fnv64\":\"%016llx\","
            "\"parameter_fnv64\":\"%016llx\",\"upstream_fnv64\":\"%016llx\",\"stream_api_calls\":%ld,"
            "\"dw_stream_api_calls\":%ld,\"parameters_unchanged\":true,"
            "\"workspace_api_calls\":%ld,\"timed_host_gemm_calls\":%ld,"
            "\"eager_mode_repeat_bytes_equal\":true,\"receipts_written\":true,\"runtime_qualified\":false,"
            "\"mean_ms\":%.9g,\"samples_ms\":[",
            model, B, enc.out_dim, argv[4], argv[5], argv[6], warmup, samples, params.total_elems,
            (unsigned long long)hash_bytes(std::vector<unsigned char>((unsigned char*)obs.data(), (unsigned char*)(obs.data() + obs.size()))),
            (unsigned long long)hash_bytes(reference), (unsigned long long)hash_bytes(param_bytes),
            (unsigned long long)hash_bytes(std::vector<unsigned char>((unsigned char*)grad.data(), (unsigned char*)(grad.data() + grad.size()))),
            stream_calls, dw_stream_calls, workspace_calls, gemm_calls - calls_before, sum / samples);
        for (int i = 0; i < samples; i++) printf("%s%.9g", i ? "," : "", milliseconds[i]);
        puts("]}");
        GPU_CHECK(cudaEventDestroy(start)); GPU_CHECK(cudaEventDestroy(finish));
        if (graph) { GPU_CHECK(cudaGraphExecDestroy(executable)); GPU_CHECK(cudaGraphDestroy(capture)); }
        for (Allocator* a : {&params, &acts, &grads, &rollout, &io}) {
            if (a->mem) GPU_CHECK(cudaFree(a->mem));
        }
        free(io.regs);
        cublasDestroy(g_cublas_handle); cublasDestroy(g_cublas_dw_handle);
        GPU_CHECK(cudaFree(g_cublas_workspace)); GPU_CHECK(cudaFree(g_cublas_dw_workspace));
        GPU_CHECK(cudaEventDestroy(g_main_ready)); GPU_CHECK(cudaEventDestroy(g_dw_done));
        GPU_CHECK(cudaStreamDestroy(g_dw_stream)); GPU_CHECK(cudaStreamDestroy(stream));
    }
    for (Allocator* a : {&params, &acts, &grads, &rollout}) free(a->regs);
    free(weights); free(train); free(actor); puf_ini_free(&ini);
    return 0;
}
