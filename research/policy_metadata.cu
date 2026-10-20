// Native parameter registration only. Never allocate/init/execute a CUDA policy.
#include <cerrno>
#include <climits>
#include <sys/stat.h>
#define PRECISION_FLOAT
#ifndef ENV_HEADER
#define PUFFER_CONNECT4CNN
#define ENV_HEADER "ocean/connect4cnn/connect4cnn.h"
#define PUFFER_ENV_NAME "connect4cnn"
#endif
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
    int type = metadata_integer(policy, "encoder", 0,
#ifdef CNN_COMPRESSION_RESEARCH
        8
#else
        4
#endif
    );
#ifdef CNN_COMPRESSION_RESEARCH
    if (type == 8) return "fullframe_compression";
#endif
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

static void metadata_kernel_limits(Weights weights, Dict* policy, long batch) {
    metadata_require(batch * OBS_SIZE <= INT_MAX, "Image batch exceeds native 32-bit kernel indexing");
#if defined(C4_IMPALA_CNN) || defined(C4_IMPOOLA_CNN)
    for (int s = 0; s < 3; s++) {
        int ci = s ? IMP_C[s - 1] : OBS_CHANNELS;
        metadata_require(batch * IMP_H[s] * IMP_W[s] * 9 * ci <= INT_MAX
            && batch * IMP_H[s] * IMP_W[s] * IMP_C[s] <= INT_MAX,
            "IMPALA stage exceeds native 32-bit kernel indexing");
        metadata_require(batch * ((IMP_H[s] + 1) / 2) * ((IMP_W[s] + 1) / 2)
            * 9 * IMP_C[s] <= INT_MAX, "IMPALA residual exceeds native 32-bit kernel indexing");
    }
#else
    NatureWeights* w;
#ifdef CNN_COMPRESSION_RESEARCH
    if (dict_get(policy,"encoder") == 8) w = &((FFCWeights*)weights.encoder)->net;
    else
#endif
    w = dict_get(policy, "encoder") == 4
        ? &((FlexWeights*)weights.encoder)->net : (NatureWeights*)weights.encoder;
    for (int i = 0; i < w->count; i++) {
        NatureLayer d = w->layer[i];
        metadata_require(batch * d.oh * d.ow * d.co <= INT_MAX
            && batch * d.oh * d.ow * d.k * d.k * d.ci <= INT_MAX
            && batch * d.ih * d.iw * d.ci <= INT_MAX,
            "CNN stage exceeds native 32-bit kernel indexing");
    }
#endif
}

static int metadata_storage(const char* filename) {
    Ini ini = {}; puf_ini_load_file(&ini, filename);
    Dict* policy = puf_ini_section(&ini, "policy", 0);
    Dict* train = puf_ini_section(&ini, "train", 0);
    Dict* vec = puf_ini_section(&ini, "vec", 0);
    const char* model = metadata_model(policy, true);
    int hidden = metadata_integer(policy, "hidden_size", 8, 4096);
    int layers = metadata_integer(policy, "num_layers", 1, 16);
    int horizon = metadata_integer(train, "horizon", 1, 65536);
    int batch = metadata_integer(train, "minibatch_size", 1, 65536);
    int agents = metadata_integer(vec, "total_agents", 1, 65536);
    int buffers = metadata_integer(vec, "num_buffers", 1, agents);
    metadata_require(batch % horizon == 0 && agents % buffers == 0 && hidden % 8 == 0,
        "Unaligned learner or vector geometry");
    metadata_require(puf_ini_get(&ini, "base", "async") == 0 && dict_get(vec, "num_policies") == 1,
        "Storage preflight covers synchronous single-policy jobs only");
    int actions[] = ACT_SIZES;
    Arch arch = build_arch(OBS_SIZE, hidden, layers, actions[0], false, horizon, policy);
    Allocator params = {}, train_acts = {}, grads = {}, rollout_acts = {};
    Weights weights = weights_create(&arch, &params);
    metadata_require(params.total_elems <= INT_MAX, "Parameters exceed native optimizer indexing");
    metadata_kernel_limits(weights, policy, max(batch, agents / buffers));
    Activations learner = arch_reg_train(&arch, weights, &train_acts, &grads, batch);
    Activations actor = arch_reg_rollout(&arch, weights, &rollout_acts, agents / buffers);
    metadata_require(params.mem == NULL && train_acts.mem == NULL && grads.mem == NULL
        && rollout_acts.mem == NULL, "Storage preflight allocated device memory");
    long observation_bytes = (long)OBS_SIZE * sizeof(float);
    // This is a conservative rejection floor, not a peak-VRAM certificate.
    // Includes two float32 rollout copies, both encoder/head/core registrations,
    // all actor buffers, params + gradients + three optimizer parameter copies.
    long rollout_bytes = 2L * agents * horizon * observation_bytes;
    long environment_bytes = (long)agents * observation_bytes;
    long floor = params.total_bytes * 4 + grads.total_bytes + train_acts.total_bytes
        + buffers * rollout_acts.total_bytes + rollout_bytes + environment_bytes;
    printf("{\"schema\":\"native-image-storage-v1\",\"model\":\"%s\","
        "\"shape_hwc\":[%d,%d,%d],\"layout\":\"NHWC\",\"dtype\":\"float32\","
        "\"agents\":%d,\"buffers\":%d,\"horizon\":%d,\"learner_batch\":%d,"
        "\"parameters\":%ld,\"parameter_bytes\":%ld,\"gradient_bytes\":%ld,"
        "\"learner_activation_bytes\":%ld,\"actor_activation_bytes_per_buffer\":%ld,"
        "\"rollout_observation_bytes\":%ld,\"storage_floor_bytes\":%ld,"
        "\"peak_memory_certified\":false,\"gpu_queried\":false,\"cuda_executed\":false}\n",
        model, OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS, agents, buffers, horizon, batch,
        params.total_elems, params.total_bytes, grads.total_bytes, train_acts.total_bytes,
        rollout_acts.total_bytes, rollout_bytes, floor);
    free(learner.encoder); free(learner.decoder); free(learner.network);
    free(actor.encoder); free(actor.decoder); free(actor.network);
    free(weights.encoder); free(weights.decoder); free(weights.network);
    free(params.regs); free(train_acts.regs); free(grads.regs); free(rollout_acts.regs);
    puf_ini_free(&ini);
    return 0;
}

int main(int argc, char** argv) {
#ifdef CNN_IMAGE_V1
    if (argc == 2 && !strcmp(argv[1], "--image-info")) {
        printf("{\"schema\":\"native-image-build-v1\",\"environment\":\"%s\",\"compiled_family\":\"%s\""
            CNN_IMAGE_INFO_JSON ",\"gpu_queried\":false,\"cuda_executed\":false}\n", PUFFER_ENV_NAME,
#ifdef C4_IMPOOLA_CNN
            "impoola"
#elif defined(C4_IMPALA_CNN)
            "impala"
#else
            "default"
#endif
        );
        return 0;
    }
#endif
    if (argc == 3 && !strcmp(argv[1], "--storage")) return metadata_storage(argv[2]);
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
    printf("{\"schema\":\"%s\",\"model\":\"%s\","
        "\"observation_shape\":[%d,%d,%d],\"discrete_actions\":%d,\"hidden\":%d,\"layers\":%d,\"horizon\":%d,"
        "\"encoder_parameters\":%ld,\"decoder_parameters\":%ld,\"core_parameters\":%ld,"
        "\"total_parameters\":%ld,\"parameter_payload_bytes\":%ld,\"parameter_allocator_bytes\":%ld,"
        "\"raw_checkpoint_contiguous\":%s,\"checkpoint_bytes\":",
#ifdef CNN_IMAGE_V1
        "native-policy-metadata-v2",
#else
        "native-policy-metadata-v1",
#endif
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
    printf("]");
#ifdef CNN_COMPRESSION_RESEARCH
    if (dict_get(policy,"encoder") == 8) {
        FFCWeights* f = (FFCWeights*)weights.encoder;
        printf(",\"compression\":{\"model\":%d,\"layers\":[",f->model);
        for (int i=0;i<4;i++) {
            NatureLayer d=f->net.layer[i];
            printf("%s[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]",i?",":"",
                d.ih,d.iw,d.ci,d.co,d.k,d.stride,d.oh,d.ow,f->ph[i],f->pw[i]);
        }
        printf("]}");
    }
#endif
#ifdef CNN_IMAGE_V1
    printf(",\"environment\":\"%s\"" CNN_IMAGE_INFO_JSON, PUFFER_ENV_NAME);
#endif
    printf(",\"cuda_executed\":false,\"gpu_queried\":false,\"weights_initialized\":false,"
        "\"policy_executed\":false,\"checkpoint_contents_validated\":false,\"numerical_acceptance\":false}\n");
    // Host descriptor objects die with this isolated process; no alloc_create,
    // CUDA initialization, tensor computation or environment stepping occurs.
    free(params.regs); free(encoder.regs); free(decoder.regs); free(core.regs);
    puf_ini_free(&ini);
    return 0;
}
