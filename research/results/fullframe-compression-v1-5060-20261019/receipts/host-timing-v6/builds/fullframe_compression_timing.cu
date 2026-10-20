// Optional encoder-only timing. Native frames and callbacks; no training/dataset.
#define main ffc_unused_training_main
#include "src/pufferl.cu"
#undef main
#include <sys/stat.h>
#include <vector>

static void timing_checked(cudaError_t status) {
    ffc_require(status == cudaSuccess, cudaGetErrorString(status));
}

int main(int argc, char** argv) {
    ffc_require(argc == 5, "timing CONFIG MODEL BATCH FRESH_OUT");
    int model = atoi(argv[2]), batch = atoi(argv[3]);
    ffc_require(model >= 0 && model < 3 && (batch == 1 || batch == 16), "timing bounds");
    ffc_require(!mkdir(argv[4], 0700), "fresh timing output");
    Ini ini = {};
    puf_ini_load_file(&ini, argv[1]);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    long count = (long)batch * OBS_SIZE;
    float* frames;
    timing_checked(cudaMallocHost(&frames, count * sizeof(float)));
    for (int b = 0; b < batch; b++) {
        Env* env = (Env*)calloc(1, sizeof(Env));
        float action = 0, reward = 0, terminal = 0;
        env->rng = 7 + b;
        env->agents[0].observations = frames + (long)b * OBS_SIZE;
        env->agents[0].actions = &action;
        env->agents[0].rewards = &reward;
        env->agents[0].terminals = &terminal;
        puf_init(env, kwargs);
        puf_reset(env);
        puf_close(env);
        free(env);
    }
    char frame_hash[65];
    ffc_sha256((unsigned char*)frames, count * sizeof(float), frame_hash);
    for (long i = 0; i < count; i++)
        ffc_require(isfinite(frames[i]) && frames[i] >= 0 && frames[i] <= 1, "invalid rendered RGB");

    cublas_init_handle();
    cudaStream_t stream;
    timing_checked(cudaStreamCreate(&stream));
    FFCWeights* weights = ffc_graph(model, OBS_HEIGHT, OBS_WIDTH);
    FFCActivations acts = {};
    Allocator params = {}, activations = {}, gradients = {};
    ffc_params(weights, &params);
    ffc_train(weights, &acts, &activations, &gradients, batch);
    alloc_create(&params);
    alloc_create(&activations);
    alloc_create(&gradients);
    ulong seed = 7;
    ffc_init(weights, &seed, stream);
    Prec input = {.shape = {batch, OBS_SIZE}}, upstream = {.shape = {batch, 128}};
    timing_checked(cudaMalloc(&input.data, count * sizeof(float)));
    timing_checked(cudaMalloc(&upstream.data, batch * 128 * sizeof(float)));
    std::vector<float> up(batch * 128, .01f);
    timing_checked(cudaMemcpyAsync(upstream.data, up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    Prec output;
    auto step = [&](bool backward) {
        timing_checked(cudaMemcpyAsync(input.data, frames, count * sizeof(float), cudaMemcpyHostToDevice, stream));
        output = ffc_forward(weights, &acts, input, stream);
        if (backward) ffc_backward(weights, &acts, upstream, stream);
    };
    float microseconds[2];
    for (int mode = 0; mode < 2; mode++) {
        for (int i = 0; i < 20; i++) step(mode);
        timing_checked(cudaStreamSynchronize(stream));
        cudaEvent_t start, stop;
        timing_checked(cudaEventCreate(&start));
        timing_checked(cudaEventCreate(&stop));
        timing_checked(cudaEventRecord(start, stream));
        for (int i = 0; i < 100; i++) step(mode);
        timing_checked(cudaEventRecord(stop, stream));
        timing_checked(cudaEventSynchronize(stop));
        float milliseconds;
        timing_checked(cudaEventElapsedTime(&milliseconds, start, stop));
        microseconds[mode] = milliseconds * 1000 / 100;
        timing_checked(cudaEventDestroy(start));
        timing_checked(cudaEventDestroy(stop));
    }
    std::vector<float> y(batch * 128);
    timing_checked(cudaMemcpy(y.data(), output.data, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : y) ffc_require(isfinite(value), "nonfinite timing output");
    char output_hash[65];
    ffc_sha256((unsigned char*)y.data(), y.size() * sizeof(float), output_hash);
    char result[2048];
    snprintf(result, sizeof(result),
        "{\"schema\":\"fullframe-compression-timing-v1\",\"status\":\"passed\","
        "\"model\":%d,\"batch\":%d,\"shape_hwc\":[%d,%d,3],\"warmup\":20,\"iterations\":100,"
        "\"forward_us\":%.9g,\"forward_backward_us\":%.9g,\"input_h2d_included\":true,"
        "\"graph\":false,\"optimizer_included\":false,\"render_included\":false,"
        "\"input_source\":\"native-reset-frames-seeds-7-through-7-plus-batch-minus-1\","
        "\"frame_sha256\":\"%s\",\"output_sha256\":\"%s\",\"allocated_native_bytes\":%ld}\n",
        model, batch, OBS_HEIGHT, OBS_WIDTH, microseconds[0], microseconds[1], frame_hash, output_hash,
        params.total_bytes + activations.total_bytes + gradients.total_bytes + (count + batch * 128) * 4);
    char file[4096];
    snprintf(file, sizeof(file), "%s/result.json", argv[4]);
    FILE* f = fopen(file, "wx");
    ffc_require(f && fputs(result, f) >= 0 && !fclose(f), "write timing result");
    fputs(result, stdout);
    timing_checked(cudaFreeHost(frames));
    puf_ini_free(&ini);
    return 0;
}
