// Host-only registration of the real native policy. No CUDA allocation/query.
#define PRECISION_FLOAT
#define PUFFER_SPRITE_FORAGE
#define PUFFER_ENV_NAME "sprite_forage"
#define ENV_HEADER "ocean/sprite_forage/sprite_forage.h"
#include "src/pufferl.cu"

int main(void) {
    Ini ini = {};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, "config/sprite_forage.ini");
    Dict* env = puf_ini_section(&ini, "env", 0);
    int widths[] = {64, 128, 256};
    for (int mode = 0; mode < 2; mode++) {
        dict_set(env, "observation_mode", mode);
        int input = puf_observation_size(env);
        for (int index = 0; index < 3; index++) {
            int hidden = widths[index];
            Arch arch = build_arch(input, hidden, 1, 5, false, 16);
            Allocator params = {};
            Weights weights = weights_create(&arch, &params);
            long expected = (long)hidden * input;
            if (mode) {
                NatureWeights* w = (NatureWeights*)weights.encoder;
                assert(w->count == 4);
                // Independent arithmetic for valid Conv8/s4, Conv4/s2, Conv3/s1.
                expected = 32 * (8 * 8 * 3 + 1) + 64 * (4 * 4 * 32 + 1)
                    + 64 * (3 * 3 * 64 + 1) + (long)hidden * (56 * 76 * 64 + 1);
                assert(w->layer[0].oh == 119 && w->layer[0].ow == 159);
                assert(w->layer[1].oh == 58 && w->layer[1].ow == 78);
                assert(w->layer[2].oh == 56 && w->layer[2].ow == 76);
                assert(arch.encoder.forward == nature_forward);
            } else {
                assert(arch.encoder.forward == encoder_forward);
            }
            Allocator encoder_params = {};
            arch.encoder.reg_params(weights.encoder, &encoder_params);
            long count = 0;
            for (int i = 0; i < encoder_params.num_regs; i++) {
                count += numel(encoder_params.regs[i].shape);
            }
            assert(count == expected);
            Allocator acts = {}, grads = {};
            void* activations = calloc(1, arch.encoder.activation_size);
            arch.encoder.reg_train(weights.encoder, activations, &acts, &grads, 16);
            assert(grads.total_elems == expected);
            if (mode) {
                pid_t pid = fork();
                assert(pid >= 0);
                if (!pid) {
                    freopen("/dev/null", "w", stderr);
                    arch.encoder.reg_rollout(weights.encoder, activations, &acts, 2048);
                    _exit(0);
                }
                int status;
                assert(waitpid(pid, &status, 0) == pid);
                assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
            }
            printf("mode=%d observations=%d hidden=%d encoder_parameters=%ld PASS\n",
                mode, input, hidden, count);
            free(weights.encoder);
            free(weights.decoder);
            free(weights.network);
            free(params.regs);
            free(encoder_params.regs);
            free(activations);
            free(acts.regs);
            free(grads.regs);
        }
    }
    // Both too-short and too-long checkpoints must stop before any CUDA copy.
    char path[] = "build/forage-checkpoint-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    char bytes[8] = {0};
    int lengths[] = {3, 5, 8};
    for (int i = 0; i < 3; i++) {
        assert(ftruncate(fd, 0) == 0 && lseek(fd, 0, SEEK_SET) == 0);
        assert(write(fd, bytes, lengths[i]) == lengths[i]);
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) {
            freopen("/dev/null", "w", stderr);
            Float dst = {.shape = {1}};
            Prec params = {};
            puf_load_weights_into(dst, params, 0, path);
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    }
    close(fd);
    assert(unlink(path) == 0);
    puts("Native checkpoints: 3 short/extra/oversized files rejected before CUDA PASS");
    puf_ini_free(&ini);
}
