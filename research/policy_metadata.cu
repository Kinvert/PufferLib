// Native parameter registration only. Never allocate/init/execute a CUDA policy.
#include <cerrno>
#include <climits>
#include <sys/stat.h>
#define PRECISION_FLOAT
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#include "src/pufferl.cu"

static void metadata_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "%s\n", message); exit(1); }
}

static int metadata_integer(Dict* section, const char* name, int low, int high) {
    double value = dict_get(section, name);
    metadata_require(isfinite(value) && value == floor(value) && value >= low && value <= high,
        "Invalid integer policy/core setting");
    return (int)value;
}

static int metadata_actions(const char* text) {
    char* end; errno = 0;
    long count = strtol(text, &end, 10);
    metadata_require(!errno && text[0] && !*end && count >= 1 && count <= 64,
        "Need an explicit discrete action count from 1 through 64");
    return count;
}

static const char* metadata_model(Dict* policy, bool allow_flex) {
    int type = metadata_integer(policy, "encoder", 0, 4);
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
    metadata_require(type == 0, "Residual metadata target requires encoder=0");
#ifdef C4_IMPOOLA_CNN
    return "impoola_cnn";
#else
    return "impala_cnn";
#endif
#else
    if (type == 2) return "nature_cnn";
    metadata_require(type == 4, "Metadata target supports only the frozen quality and Nature graphs");
    if (allow_flex) {
        // The production constructor validates the legacy grammar before any
        // registration. This opt-in describes shapes, not tensor/model math.
        return "flex";
    }
    const char* keys[] = {"cnn_depth", "cnn_projection", "cnn_global_pool", "cnn_channels_1",
        "cnn_kernel_1", "cnn_stride_1", "cnn_pool_1", "cnn_residual_1"};
    int values[] = {1, 64, 0, 16, 7, 4, 0, 0};
    for (int i = 0; i < 8; i++) metadata_require(dict_get(policy, keys[i]) == values[i], "Not the locked quality graph");
    return "flex_quality";
#endif
}

int main(int argc, char** argv) {
    bool allow_flex = argc > 1 && (!strcmp(argv[1], "--describe-flex") || !strcmp(argv[1], "--check-flex-bytes"));
    bool describe = argc == 4 && (!strcmp(argv[1], "--describe") || !strcmp(argv[1], "--describe-flex"));
    bool check = argc == 5 && (!strcmp(argv[1], "--check-bytes") || !strcmp(argv[1], "--check-flex-bytes"));
    if (!describe && !check) {
        fprintf(stderr, "Usage: %s --describe FULL.ini DISCRETE_ACTIONS\n"
            "       %s --check-bytes FULL.ini DISCRETE_ACTIONS CHECKPOINT\n"
            "       --describe-flex / --check-flex-bytes also permit legacy encoder 4 shapes\n", argv[0], argv[0]);
        return 1;
    }
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    Dict* policy = puf_ini_section(&ini, "policy", 0);
    int hidden = metadata_integer(policy, "hidden_size", 8, 4096);
    metadata_require(hidden % 8 == 0, "Hidden size must be divisible by eight");
    int layers = metadata_integer(policy, "num_layers", 1, 16);
    int horizon = metadata_integer(puf_ini_section(&ini, "train", 0), "horizon", 1, 65536);
    int actions = metadata_actions(argv[3]);
    // Qualify graph settings before constructing descriptors, especially ID 5.
    const char* model = metadata_model(policy, allow_flex);
    Arch arch = build_arch(OBS_SIZE, hidden, layers, actions, false, horizon, policy);
    if (!strcmp(model, "flex_quality") || !strcmp(model, "flex"))
        metadata_require(arch.encoder.forward == flex_forward, "Wrong encoder callback");
    Allocator params = {}, encoder = {}, decoder = {}, core = {};
    Weights weights = weights_create(&arch, &params);
    // Re-register the same descriptors solely to obtain component boundaries.
    arch.encoder.reg_params(weights.encoder, &encoder);
    arch.decoder.reg_params(weights.decoder, &decoder);
    arch.network.reg_params(weights.network, &core);
    metadata_require(params.mem == NULL && params.total_elems > 0 && params.total_elems <= LONG_MAX / 4,
        "Invalid/unexpected parameter allocation");
    metadata_require(params.num_regs == encoder.num_regs + decoder.num_regs + core.num_regs
        && params.total_elems == encoder.total_elems + decoder.total_elems + core.total_elems,
        "Component registrations disagree with actual weights_create");
    long payload = params.total_elems * sizeof(float);
    long checkpoint_bytes = -1;
    if (check) {
        struct stat info;
        metadata_require(!stat(argv[4], &info) && S_ISREG(info.st_mode), "Need a regular checkpoint artifact");
        metadata_require(info.st_size == payload, "Checkpoint byte count differs from native policy registration");
        // Raw native checkpoints serialize total_elems floats. Reject any layout
        // whose alignment padding would invalidate that contiguous assumption.
        metadata_require(params.total_bytes == payload, "Padded parameter layout is not raw-checkpoint qualified");
        checkpoint_bytes = info.st_size;
    }
    printf("{\"schema\":\"native-policy-metadata-v1\",\"model\":\"%s\","
        "\"observation_shape\":[%d,%d,%d],\"discrete_actions\":%d,\"hidden\":%d,\"layers\":%d,\"horizon\":%d,"
        "\"encoder_parameters\":%ld,\"decoder_parameters\":%ld,\"core_parameters\":%ld,"
        "\"total_parameters\":%ld,\"parameter_payload_bytes\":%ld,\"parameter_allocator_bytes\":%ld,"
        "\"raw_checkpoint_contiguous\":%s,\"checkpoint_bytes\":",
        model, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH, actions, hidden, layers, horizon,
        encoder.total_elems, decoder.total_elems, core.total_elems, params.total_elems, payload,
        params.total_bytes, params.total_bytes == payload ? "true" : "false");
    if (check) printf("%ld", checkpoint_bytes); else printf("null");
    printf(",\"parameter_tensors\":[");
    long offset = 0;
    for (int i = 0; i < params.num_regs; i++) {
        const AllocEntry& entry = params.regs[i];
        metadata_require(entry.elem_size == sizeof(float) && *entry.data_ptr == NULL,
            "Non-float/unexpected allocated parameter descriptor");
        offset = (offset + 15) & ~15;
        const char* group = i < encoder.num_regs ? "encoder" :
            i < encoder.num_regs + decoder.num_regs ? "decoder" : "core";
        printf("%s{\"index\":%d,\"group\":\"%s\",\"offset_bytes\":%ld,\"elements\":%ld,\"shape\":[",
            i ? "," : "", i, group, offset, (long)numel(entry.shape));
        for (int d = 0; d < ndim(entry.shape); d++) {
            metadata_require(entry.shape[d] > 0, "Invalid parameter shape");
            printf("%s%ld", d ? "," : "", entry.shape[d]);
        }
        printf("]}"); offset += numel(entry.shape) * sizeof(float);
    }
    metadata_require(offset == params.total_bytes, "Parameter offsets disagree with allocator");
    printf("],\"cuda_executed\":false,\"gpu_queried\":false,\"weights_initialized\":false,"
        "\"policy_executed\":false,\"checkpoint_contents_validated\":false,\"numerical_acceptance\":false}\n");
    // Host descriptor objects die with this isolated process; no alloc_create,
    // CUDA initialization, tensor computation or environment stepping occurs.
    free(params.regs); free(encoder.regs); free(decoder.regs); free(core.regs);
    puf_ini_free(&ini);
    return 0;
}
