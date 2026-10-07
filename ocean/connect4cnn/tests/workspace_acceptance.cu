// Isolated shared-GEMM acceptance. No production helper or encoder edits.
#include <vector>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasStatus_t original_stream(cublasHandle_t handle, cudaStream_t stream) {
    return cublasSetStream(handle, stream);
}
static cublasStatus_t checked_stream(cublasHandle_t, cudaStream_t);
static long gemm_calls, main_calls, dw_calls, workspace_calls;
static bool reassign;
template<class... Args> static cublasStatus_t checked_gemm(Args... args) {
    cublasStatus_t status = cublasGemmEx(args...);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM status %d\n", (int)status); exit(1);
    }
    gemm_calls++;
    return status;
}

static cudaError_t checked_record(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t status = cudaEventRecord(event, stream);
    if (status != cudaSuccess) { fprintf(stderr, "Event record: %s\n", cudaGetErrorString(status)); exit(1); }
    return status;
}
static cudaError_t checked_wait(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    cudaError_t status = cudaStreamWaitEvent(stream, event, flags);
    if (status != cudaSuccess) { fprintf(stderr, "Event wait: %s\n", cudaGetErrorString(status)); exit(1); }
    return status;
}

#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#undef cublasSetStream
#define cublasSetStream checked_stream
#define cublasGemmEx checked_gemm
#define cudaEventRecord checked_record
#define cudaStreamWaitEvent checked_wait
#include "src/pufferl.cu"
#undef cublasSetStream
#undef cublasGemmEx
#undef cudaEventRecord
#undef cudaStreamWaitEvent

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(err)); exit(1); } } while (0)

static void require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "%s\n", message); exit(1); }
}

struct Fixture { const char* name; int m, k, n; };
static const Fixture fixtures[] = {
    {"odd", 3, 7, 5},
    {"projection", 64, 128, 128},
    {"quality-conv", 6336, 49, 16},
    {"nature-conv2", 768, 512, 64},
    {"impala-residual", 25344, 144, 16},
    {"learner-core", 2048, 128, 128}
};
static const int lanes = 2, iterations = 3;
static cudaStream_t bound_main, bound_dw;

static cublasStatus_t checked_stream(cublasHandle_t handle, cudaStream_t stream) {
    bool main = handle == g_cublas_handle;
    require(main || handle == g_cublas_dw_handle, "Unknown handle");
    cudaStream_t* bound = main ? &bound_main : &bound_dw;
    require(stream && (!*bound || *bound == stream), "Handle moved across streams");
    *bound = stream;
    cublasStatus_t status = original_stream(handle, stream);
    require(status == CUBLAS_STATUS_SUCCESS, "Stream assignment failed");
    if (main) main_calls++; else dw_calls++;
    if (reassign) {
        void* workspace = main ? g_cublas_workspace : g_cublas_dw_workspace;
        require(workspace && g_cublas_workspace != g_cublas_dw_workspace
                && (uintptr_t)workspace % 256 == 0, "Unsafe workspace allocation");
        status = cublasSetWorkspace(handle, workspace, 32 * 1024 * 1024);
        require(status == CUBLAS_STATUS_SUCCESS, "Workspace assignment failed");
        workspace_calls++;
    }
    return status;
}

static int integer(const char* text, int limit) {
    char* end; errno = 0; long value = strtol(text, &end, 10);
    require(!errno && text[0] && !*end && value >= 0 && value < limit, "Invalid fixture ID");
    return (int)value;
}

// Inputs are exact multiples of 1/8 with magnitude <= 1/2. Products and every
// intermediate sum fit float32 exactly for these bounded fixture dimensions.
__global__ void fill_input(float* out, int count, unsigned salt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned value = (unsigned)i + salt * 0x9e3779b9u;
        value = (value ^ (value >> 16)) * 0x7feb352du;
        value = (value ^ (value >> 15)) * 0x846ca68bu;
        value ^= value >> 16;
        out[i] = ((int)(value % 9) - 4) / 8.0f;
    }
}

// Independent row-major scalar reference; no cuBLAS or production GEMM code.
// op 0: Y = X W^T; op 1: dW = G^T X; op 2: dX = G W.
__global__ void reference(const float* x, const float* w, const float* g,
        float* out, int m, int k, int n, int op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = op == 0 ? m*n : op == 1 ? n*k : m*k;
    if (idx >= count) return;
    double sum = 0.0;
    if (op == 0) {
        int row = idx / n, col = idx % n;
        for (int j = 0; j < k; j++) sum += (double)x[row*k+j] * w[col*k+j];
    } else if (op == 1) {
        int row = idx / k, col = idx % k;
        for (int j = 0; j < m; j++) sum += (double)g[j*n+row] * x[j*k+col];
    } else {
        int row = idx / k, col = idx % k;
        for (int j = 0; j < n; j++) sum += (double)g[row*n+j] * w[j*k+col];
    }
    out[idx] = (float)sum;
}

struct Buffers { Prec x, w, g, y, dw, dx; float *ry, *rw, *rx; };
static Prec allocate(int rows, int cols) {
    Prec tensor = {}; tensor.shape[0] = rows; tensor.shape[1] = cols;
    CUDA_CHECK(cudaMalloc(&tensor.data, (size_t)rows * cols * sizeof(float)));
    return tensor;
}

static void submit(Buffers* data, cudaStream_t stream, bool overlap) {
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane];
        // Replays must overwrite every element, including outputs on dW.
        CUDA_CHECK(cudaMemsetAsync(b->y.data, 0xff, numel(b->y.shape)*4, stream));
        CUDA_CHECK(cudaMemsetAsync(b->dw.data, 0xff, numel(b->dw.shape)*4, stream));
        CUDA_CHECK(cudaMemsetAsync(b->dx.data, 0xff, numel(b->dx.shape)*4, stream));
        puf_mm(&b->x, &b->w, &b->y, stream);
        puf_mm_tn_async_after(&b->g, &b->x, &b->dw, stream);
        if (!overlap) puf_dw_join(stream);
        puf_mm_nn(&b->g, &b->w, &b->dx, stream);
    }
    // Multiple disjoint dWs may queue, just as in native linear backwards.
    puf_dw_join(stream);
    CUDA_CHECK(cudaGetLastError());
}

static void append_device(std::vector<float>& values, const float* device, size_t count) {
    size_t offset = values.size(); values.resize(offset + count);
    CUDA_CHECK(cudaMemcpy(values.data()+offset, device, count*4, cudaMemcpyDeviceToHost));
}

static std::vector<float> outputs(Buffers* data, bool oracle) {
    std::vector<float> values;
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane];
        append_device(values, oracle ? b->ry : b->y.data, numel(b->y.shape));
        append_device(values, oracle ? b->rw : b->dw.data, numel(b->dw.shape));
        append_device(values, oracle ? b->rx : b->dx.data, numel(b->dx.shape));
    }
    return values;
}

static void verify(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "Wrong output quota");
    for (size_t i = 0; i < actual.size(); i++) {
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
            fprintf(stderr, "Oracle mismatch at %zu: %a != %a\n", i, actual[i], expected[i]); exit(1);
        }
    }
}

static void write_values(const char* directory, const char* name, const std::vector<float>& values) {
    std::string path = std::string(directory) + "/" + name;
    FILE* file = fopen(path.c_str(), "wx"); require(file, "Output exists or cannot be created");
    require(fwrite(values.data(), 4, values.size(), file) == values.size(), "Short export");
    require(fclose(file) == 0, "Export close failed");
}

static void describe() {
    printf("{\"protocol\":\"shared-gemm-workspace-v1\",\"lanes\":%d,\"iterations\":%d,"
           "\"precision\":\"float32\",\"oracle\":\"gpu-scalar-double-exact-dyadic-v1\",\"fixtures\":[", lanes, iterations);
    for (int i = 0; i < 6; i++) {
        Fixture f = fixtures[i];
        long input = (long)lanes * (f.m*f.k + f.n*f.k + f.m*f.n);
        printf("%s{\"id\":%d,\"name\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,"
               "\"input_bytes\":%ld,\"output_bytes\":%ld}", i ? "," : "", i, f.name, f.m, f.k, f.n, input*4, input*4);
    }
    puts("]}");
}

int main(int argc, char** argv) {
    // This branch is pure metadata: no CUDA/device/library/context operation.
    if (argc == 2 && !strcmp(argv[1], "--describe")) { describe(); return 0; }
    require(argc == 7 && !strcmp(argv[1], "--run"),
            "Host: --describe; scheduled only: --run ID default|reassign serial|overlap eager|graph FRESH_RECEIPT_DIR");
    int id = integer(argv[2], 6); Fixture f = fixtures[id];
    require(!strcmp(argv[3], "default") || !strcmp(argv[3], "reassign"), "Invalid workspace mode");
    require(!strcmp(argv[4], "serial") || !strcmp(argv[4], "overlap"), "Invalid schedule");
    require(!strcmp(argv[5], "eager") || !strcmp(argv[5], "graph"), "Invalid graph mode");
    struct stat status; require(stat(argv[6], &status) != 0 && errno == ENOENT, "Need fresh receipt directory");
    require(mkdir(argv[6], 0755) == 0, "Cannot create receipt directory");
    reassign = !strcmp(argv[3], "reassign"); bool overlap = !strcmp(argv[4], "overlap");
    cublas_init_handle(); cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    require(g_cublas_handle && g_cublas_dw_handle && g_cublas_workspace && g_cublas_dw_workspace
            && g_cublas_workspace != g_cublas_dw_workspace && g_dw_stream && g_main_ready && g_dw_done,
            "Shared handle/stream initialization failed");
    for (cublasHandle_t handle : {g_cublas_handle, g_cublas_dw_handle}) {
        cublasMath_t math; cublasPointerMode_t pointer;
        require(cublasGetMathMode(handle, &math) == CUBLAS_STATUS_SUCCESS && math == CUBLAS_DEFAULT_MATH
                && cublasGetPointerMode(handle, &pointer) == CUBLAS_STATUS_SUCCESS
                && pointer == CUBLAS_POINTER_MODE_HOST, "Unexpected cuBLAS math/pointer mode");
    }
    Buffers data[lanes] = {};
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane];
        b->x = allocate(f.m, f.k); b->w = allocate(f.n, f.k); b->g = allocate(f.m, f.n);
        b->y = allocate(f.m, f.n); b->dw = allocate(f.n, f.k); b->dx = allocate(f.m, f.k);
        CUDA_CHECK(cudaMalloc(&b->ry, numel(b->y.shape)*4));
        CUDA_CHECK(cudaMalloc(&b->rw, numel(b->dw.shape)*4));
        CUDA_CHECK(cudaMalloc(&b->rx, numel(b->dx.shape)*4));
        Prec inputs[] = {b->x, b->w, b->g};
        for (int j = 0; j < 3; j++) {
            int size = (int)numel(inputs[j].shape);
            fill_input<<<grid_size(size), BLOCK_SIZE, 0, stream>>>(inputs[j].data, size, 1+lane*3+j);
        }
        float* refs[] = {b->ry, b->rw, b->rx};
        int sizes[] = {f.m*f.n, f.n*f.k, f.m*f.k};
        for (int op = 0; op < 3; op++) {
            reference<<<grid_size(sizes[op]), BLOCK_SIZE, 0, stream>>>(b->x.data, b->w.data, b->g.data,
                refs[op], f.m, f.k, f.n, op);
        }
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> inputs;
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane];
        append_device(inputs, b->x.data, numel(b->x.shape)); append_device(inputs, b->w.data, numel(b->w.shape));
        append_device(inputs, b->g.data, numel(b->g.shape));
    }
    std::vector<float> expected = outputs(data, true);
    submit(data, stream, overlap); CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> first = outputs(data, false); verify(first, expected);
    cudaGraph_t graph = NULL; cudaGraphExec_t executable = NULL;
    if (!strcmp(argv[5], "graph")) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        submit(data, stream, overlap);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    }
    for (int i = 0; i < iterations; i++) {
        if (executable) CUDA_CHECK(cudaGraphLaunch(executable, stream)); else submit(data, stream, overlap);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> actual = outputs(data, false); verify(actual, expected);
        require(memcmp(first.data(), actual.data(), first.size()*4) == 0, "Replay bytes differ");
    }
    std::vector<float> final_inputs;
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane];
        append_device(final_inputs, b->x.data, numel(b->x.shape)); append_device(final_inputs, b->w.data, numel(b->w.shape));
        append_device(final_inputs, b->g.data, numel(b->g.shape));
    }
    require(inputs.size() == final_inputs.size()
            && memcmp(inputs.data(), final_inputs.data(), inputs.size()*4) == 0, "GEMM changed an input");
    require(main_calls > 0 && dw_calls > 0 && gemm_calls == main_calls+dw_calls
            && workspace_calls == (reassign ? gemm_calls : 0), "Missing API coverage");
    write_values(argv[6], "input.f32", inputs); write_values(argv[6], "reference.f32", expected);
    write_values(argv[6], "actual.f32", first);
    std::string receipt = std::string(argv[6]) + "/native.json";
    FILE* file = fopen(receipt.c_str(), "wx"); require(file, "Cannot create receipt");
    int runtime, driver; CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
    int blas; require(cublasGetVersion(g_cublas_handle, &blas) == CUBLAS_STATUS_SUCCESS, "cuBLAS version failed");
    fprintf(file, "{\"protocol\":\"shared-gemm-workspace-v1\",\"status\":\"scoped-gpu-checks-passed\","
        "\"fixture\":%d,\"workspace\":\"%s\",\"schedule\":\"%s\",\"mode\":\"%s\","
        "\"lanes\":%d,\"iterations\":%d,\"main_api_calls\":%ld,\"dw_api_calls\":%ld,"
        "\"gemm_api_calls\":%ld,\"workspace_api_calls\":%ld,\"input_bytes\":%zu,\"output_bytes\":%zu,"
        "\"runtime\":%d,\"driver\":%d,\"cublas\":%d,\"measured_overlap\":false,"
        "\"encoder_math_certified\":false,\"frontier_superiority_certified\":false}\n",
        id, argv[3], argv[4], argv[5], lanes, iterations, main_calls, dw_calls, gemm_calls, workspace_calls,
        inputs.size()*4, first.size()*4, runtime, driver, blas);
    require(fclose(file) == 0, "Receipt close failed");
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    for (int lane = 0; lane < lanes; lane++) {
        Buffers* b = &data[lane]; Prec tensors[] = {b->x, b->w, b->g, b->y, b->dw, b->dx};
        for (Prec tensor : tensors) CUDA_CHECK(cudaFree(tensor.data));
        CUDA_CHECK(cudaFree(b->ry)); CUDA_CHECK(cudaFree(b->rw)); CUDA_CHECK(cudaFree(b->rx));
    }
    require(cublasDestroy(g_cublas_handle) == CUBLAS_STATUS_SUCCESS
            && cublasDestroy(g_cublas_dw_handle) == CUBLAS_STATUS_SUCCESS, "Handle cleanup failed");
    CUDA_CHECK(cudaFree(g_cublas_workspace)); CUDA_CHECK(cudaFree(g_cublas_dw_workspace));
    CUDA_CHECK(cudaEventDestroy(g_main_ready)); CUDA_CHECK(cudaEventDestroy(g_dw_done));
    CUDA_CHECK(cudaStreamDestroy(g_dw_stream)); CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
