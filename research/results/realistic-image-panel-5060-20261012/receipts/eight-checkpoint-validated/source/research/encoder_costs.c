// Fixed-graph shape arithmetic only. No policy, weights, inference or CUDA.
// Mirrors the four audited 1x36x44 graphs; not a general sweep constructor.
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint64_t params, macs, patches, outputs, first_macs, first_patches;
    int convolutions, dense;
} Cost;

static void layer(Cost* c, int h, int w, int ci, int co, int k, int dense) {
    uint64_t rows = (uint64_t)h*w, reduction = (uint64_t)ci*k*k;
    uint64_t patches = rows*reduction, macs = patches*co;
    if (!c->convolutions && !c->dense) { c->first_macs = macs; c->first_patches = patches; }
    c->params += reduction*co+co;
    c->macs += macs; c->patches += patches; c->outputs += rows*co;
    if (dense) c->dense++; else c->convolutions++;
}

static int integer(const char* text, int low, int high) {
    char* end; errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !text[0] || *end || value < low || value > high) {
        fprintf(stderr, "Invalid integer: %s\n", text); exit(1);
    }
    return (int)value;
}

static void report(const char* name, Cost c, int hidden, int layers, int actions,
        uint64_t rollout, uint64_t training, uint64_t rebuilt) {
    uint64_t core = (uint64_t)3*hidden*hidden*layers, head = (uint64_t)hidden*(actions+1);
    // Existing backward skips the raw-pixel input gradient on the first conv.
    uint64_t train_macs = 3*c.macs-c.first_macs;
    printf("%s,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
        ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
        ",%" PRIu64 ",%" PRIu64 ",%d\n", name, c.convolutions, c.dense,
        c.params, core, head, c.params+core+head, c.macs, 2*c.macs, train_macs,
        c.patches, rebuilt, 4*rollout, 4*training, c.convolutions+c.dense);
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 4) {
        fprintf(stderr, "Usage: %s [hidden layers discrete_actions]\n", argv[0]); return 1;
    }
    int hidden = argc == 4 ? integer(argv[1], 8, 4096) : 128;
    int layers = argc == 4 ? integer(argv[2], 1, 16) : 1;
    int actions = argc == 4 ? integer(argv[3], 1, 64) : 7;
    if (hidden % 8) { fprintf(stderr, "Hidden width must be a multiple of eight\n"); return 1; }
    puts("model,conv_layers,dense_layers,encoder_parameters,core_parameters,head_parameters,total_parameters,forward_gemm_macs_per_sample,forward_gemm_flops_per_sample,forward_backward_gemm_macs_per_sample,forward_patch_elements_per_sample,backward_rebuilt_patch_elements_per_sample,rollout_encoder_tensor_bytes_per_sample,train_encoder_tensor_bytes_per_sample,forward_gemm_calls");
    Cost quality = {0};
    layer(&quality, 9, 11, 1, 16, 7, 0);
    layer(&quality, 1, 1, 9*11*16, 64, 1, 1);
    if (hidden != 64) layer(&quality, 1, 1, 64, hidden, 1, 1);
    report("flex_quality", quality, hidden, layers, actions, quality.patches+quality.outputs,
           2*quality.patches-quality.first_patches+2*quality.outputs, 0);
    Cost nature = {0};
    layer(&nature, 8, 10, 1, 32, 8, 0);
    layer(&nature, 3, 4, 32, 64, 4, 0);
    layer(&nature, 1, 2, 64, 64, 3, 0);
    layer(&nature, 1, 1, 128, hidden, 1, 1);
    report("nature_cnn", nature, hidden, layers, actions, nature.patches+nature.outputs,
           2*nature.patches-nature.first_patches+2*nature.outputs, 0);
    for (int gap = 0; gap <= 1; gap++) {
        Cost impala = {0};
        const int height[3] = {36, 18, 9}, width[3] = {44, 22, 11}, channels[3] = {16, 32, 32};
        uint64_t max_patch = 0, pools = 0;
        for (int s = 0; s < 3; s++) {
            int h = height[s], w = width[s], co = channels[s], ci = s ? channels[s-1] : 1;
            uint64_t p = (uint64_t)h*w*9*ci;
            if (p > max_patch) max_patch = p;
            layer(&impala, h, w, ci, co, 3, 0);
            h = (h+1)/2; w = (w+1)/2;
            pools += (uint64_t)h*w*co;
            p = (uint64_t)h*w*9*co;
            if (p > max_patch) max_patch = p;
            for (int j = 0; j < 4; j++) layer(&impala, h, w, co, co, 3, 0);
        }
        uint64_t rebuilt = impala.patches, conv_outputs = impala.outputs;
        int readout = gap ? 32 : 960;
        layer(&impala, 1, 1, readout, hidden, 1, 1);
        // Dense uses direct puf_mm, with no materialized dense patch tensor.
        impala.patches -= readout;
        uint64_t rollout = conv_outputs+pools+max_patch+readout+hidden;
        uint64_t grad_size = hidden > 36*44*16 ? hidden : 36*44*16;
        // Winners are int32: counted as four bytes each, not as float tensors.
        uint64_t training = rollout+36*44+max_patch+2*grad_size+18*22*16+pools;
        report(gap ? "impoola_cnn" : "impala_cnn", impala, hidden, layers, actions,
               rollout, training, rebuilt);
    }
    return 0;
}
