// Included after eval_make. Exact evaluation does not alter training or kernels.
#include "exact_episode.h"
#include <float.h>

static void breakout_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "Breakout exact evaluation: %s\n", message); exit(1); }
}

static uint32_t breakout_exact_option(Ini* ini, const char* key, uint32_t lo, uint32_t hi) {
    double value = puf_ini_get(ini, "eval_exact", key);
    breakout_exact_require(isfinite(value) && value >= lo && value <= hi && value == floor(value), key);
    return (uint32_t)value;
}

static void breakout_exact_world(Ini* ini) {
    const char* integers[] = {"frameskip", "width", "height", "ball_width", "ball_height",
        "brick_width", "brick_height", "brick_rows", "brick_cols"};
    for (int i = 0; i < 9; i++) {
        double value = puf_ini_get(ini, "env", integers[i]);
        breakout_exact_require(isfinite(value) && value >= 1 && value <= 1000000
            && value == floor(value), integers[i]);
    }
    const char* positive[] = {"paddle_width", "paddle_height", "initial_ball_speed",
        "max_ball_speed", "paddle_speed"};
    for (int i = 0; i < 5; i++) {
        double value = puf_ini_get(ini, "env", positive[i]);
        breakout_exact_require(isfinite(value) && value > 0 && value <= 1000000
            && (float)value > 0 && fabs(value) <= FLT_MAX, positive[i]);
    }
    breakout_exact_require(puf_ini_get(ini, "env", "brick_rows") == 6
        && puf_ini_get(ini, "env", "brick_cols") == 18
        && puf_ini_get(ini, "env", "continuous") == 0, "stock brick count and discrete actions required");
    breakout_exact_require(puf_ini_get(ini, "env", "initial_ball_speed")
        <= puf_ini_get(ini, "env", "max_ball_speed"), "initial speed exceeds maximum");
    breakout_exact_require(puf_ini_get(ini, "env", "paddle_width") <= puf_ini_get(ini, "env", "width")
        && puf_ini_get(ini, "env", "paddle_height") < puf_ini_get(ini, "env", "height")
        && puf_ini_get(ini, "env", "ball_width") <= puf_ini_get(ini, "env", "width")
        && puf_ini_get(ini, "env", "ball_height") < puf_ini_get(ini, "env", "height"), "objects exceed world");
#ifdef PUFFER_BREAKOUTCNN
    breakout_exact_require(puf_ini_get(ini, "env", "representation_mode") == 0, "fixed appearance required");
#endif
}

static void breakout_exact_info() {
#ifdef PUFFER_BREAKOUT
    const char* encoder = "state";
#elif defined(C4_IMPOOLA_CNN)
    const char* encoder = "impoola";
#elif defined(C4_IMPALA_CNN)
    const char* encoder = "impala";
#elif defined(C4_NATURE_CNN)
    const char* encoder = "nature";
#else
    const char* encoder = "tiny";
#endif
    printf("{\"environment\":\"%s\",\"default_encoder\":\"%s\",\"float32\":%s,"
        "\"rules\":\"breakout-native-v1\",\"receipt_version\":1,"
        "\"evaluation\":\"first-terminal-or-frame-cap-v1\"}\n",
        PUFFER_ENV_NAME, encoder, USE_BF16 ? "false" : "true");
}

// This binary's actual host reset/raster, before CUDA initialization or inference.
static void breakout_exact_manifest(int argc, char** argv) {
    breakout_exact_require(argc == 3, "eval_exact_manifest requires one full INI path");
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    breakout_exact_require(!USE_BF16, "float32 manifest required");
    breakout_exact_world(&ini);
    uint32_t n = breakout_exact_option(&ini, "episodes", 1, 1000000);
    uint32_t first = breakout_exact_option(&ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = breakout_exact_option(&ini, "seed", 0, UINT32_MAX);
    uint32_t slots = breakout_exact_option(&ini, "slots", 1, 1024);
    breakout_exact_option(&ini, "max_frames", 1, 16777216);
    breakout_exact_require((uint64_t)first+n+slots <= UINT32_MAX, "episode IDs overflow");
    Env env = {}; obs_t observations[OBS_SIZE];
    float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations;
    env.agents[0].actions = &action;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    puf_init(&env, puf_ini_section(&ini, "env", 0));
    int representation = 0;
#ifdef PUFFER_BREAKOUTCNN
    representation = env.representation;
#endif
    printf("episode_id,env_seed,policy_seed,representation,start_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = first+i; BreakoutExactEpisode episode;
        breakout_exact_start(&env, &episode, seed, id);
        printf("%u,%u,%llu,%d,%u,%016llx,%016llx\n", id, c4_exact_env_seed(seed, id),
            (unsigned long long)c4_exact_policy_seed(seed, id), representation, env.rng,
            (unsigned long long)breakout_exact_world_hash(&env),
            (unsigned long long)breakout_exact_observation_hash(observations));
    }
    puf_close(&env); puf_ini_free(&ini);
}

__global__ void breakout_exact_rng(curandStatePhilox4_32_10_t* states,
        uint32_t seed, uint32_t first, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) curand_init(c4_exact_policy_seed(seed, first+i), 0, 0, &states[i]);
}

static void run_breakout_exact_eval(Ini* ini, TrainContext* ctx) {
    breakout_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU, "float32 CPU environment required");
    breakout_exact_require(ctx->world_size == 1 && puf_ini_get(ini, "selfplay", "enabled") == 0,
        "single GPU, no selfplay required");
    breakout_exact_world(ini);
    uint32_t episodes = breakout_exact_option(ini, "episodes", 1, 1000000);
    uint32_t first = breakout_exact_option(ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = breakout_exact_option(ini, "seed", 0, UINT32_MAX);
    uint32_t slots = breakout_exact_option(ini, "slots", 1, 1024);
    int cap = breakout_exact_option(ini, "max_frames", 1, 16777216);
    breakout_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX, "episode IDs overflow");
    const char* output = puf_ini_get_str(ini, "eval_exact", "output");
    breakout_exact_require(output && output[0] && strcmp(output, "None"), "explicit output required");
    FILE* fp = fopen(output, "wx");
    breakout_exact_require(fp != NULL, "output exists or cannot be created");
    fprintf(fp, "version,block_seed,episode_id,slot,representation,env_seed,policy_seed,decisions,frames,score,perf,episode_return,completed,capped,action_hash,start_observation_hash,start_world_hash,start_rng\n");
    breakout_exact_require(fflush(fp) == 0, "header write failed");
    char count[32]; snprintf(count, sizeof(count), "%u", slots);
    puf_ini_put(ini, "base.async", "0");
    puf_ini_put(ini, "base.eval_agents", count);
    puf_ini_put(ini, "vec.total_agents", count);
    puf_ini_put(ini, "vec.num_buffers", "1");
    puf_ini_put(ini, "vec.num_threads", "1");
    puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "vec.hist_policy_percent", "0");
    puf_ini_put(ini, "train.horizon", "1");
    puf_ini_put(ini, "train.minibatch_size", count);
    puf_ini_put(ini, "train.replay_ratio", "1");
    const char* checkpoint = puf_ini_get_str(ini, "base", "load_model_path");
    breakout_exact_require(checkpoint && strcmp(checkpoint, "None"), "explicit checkpoint required");
    PuffeRL* p = eval_make(ini, ctx, EVAL_SCORE, 0);
    VecEnv* v = p->vec;
    breakout_exact_require(v->size == (int)slots && v->total_agents == (int)slots, "one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape);
    struct stat st;
    breakout_exact_require(stat(checkpoint, &st) == 0 && st.st_size == parameters*(long)sizeof(float), "checkpoint size mismatch");
    FILE* weights = fopen(checkpoint, "rb");
    breakout_exact_require(weights != NULL, "cannot audit checkpoint");
    float value;
    for (long i = 0; i < parameters; i++)
        breakout_exact_require(fread(&value, sizeof(value), 1, weights) == 1 && isfinite(value), "nonfinite/truncated weights");
    breakout_exact_require(fclose(weights) == 0, "weights read failed");
    breakout_exact_require(cudaStreamSynchronize(p->default_stream) == cudaSuccess, "checkpoint upload failed");
    cudaStream_t stream = p->streams[0];
    BreakoutExactEpisode* history = (BreakoutExactEpisode*)calloc(slots, sizeof(BreakoutExactEpisode));
    unsigned char* active = (unsigned char*)calloc(slots, 1);
    uint64_t* action_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* observation_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* world_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint32_t* start_rng = (uint32_t*)calloc(slots, sizeof(uint32_t));
    breakout_exact_require(history && active && action_hash && observation_hash && world_hash && start_rng, "allocation failed");
    uint32_t evaluated = 0, completed = 0, capped = 0;
    uint64_t score = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots, episodes-base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            breakout_exact_start(env, &history[i], seed, first+base+i);
            breakout_exact_require(env->tick == 0 && env->score == 0 && env->log.n == 0
                && env->num_balls == 5 && env->balls_fired == 0, "starting state mismatch");
            observation_hash[i] = breakout_exact_observation_hash(env->agents[0].observations);
            world_hash[i] = breakout_exact_world_hash(env); start_rng[i] = env->rng;
            active[i] = i < n; action_hash[i] = UINT64_C(14695981039346656037);
        }
        breakout_exact_rng<<<grid_size(slots), BLOCK_SIZE, 0, stream>>>(p->rng_states[0], seed, first+base, slots);
        uint32_t remaining = n;
        int max_decisions = 1 + (cap - 1)/v->envs[0].frameskip;
        for (int decision = 1; decision <= max_decisions && remaining; decision++) {
            cpu_upload(p, 0, slots, stream);
            pufferl_forward(p, 0, 0, stream);
            breakout_exact_require(cudaMemcpyAsync(v->actions, p->env.actions.data, slots*sizeof(float),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess, "action download failed");
            breakout_exact_require(cudaStreamSynchronize(stream) == cudaSuccess, "GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i]; BreakoutExactEpisode* e = &history[i];
                float action = v->actions[i];
                breakout_exact_require(action == NOOP || action == LEFT || action == RIGHT, "invalid policy action");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action)*UINT64_C(1099511628211);
                breakout_exact_require(breakout_exact_step(env, e, action, cap), "episode accounting mismatch");
                breakout_exact_require(e->decisions == decision, "decision count mismatch");
                if (!e->completed && !e->capped) continue;
                uint32_t id = first+base+i; int representation = 0;
#ifdef PUFFER_BREAKOUTCNN
                representation = env->representation;
#endif
                fprintf(fp, "1,%u,%u,%u,%d,%u,%llu,%d,%d,%d,%a,%a,%d,%d,%016llx,%016llx,%016llx,%u\n",
                    seed, id, i, representation, c4_exact_env_seed(seed, id),
                    (unsigned long long)c4_exact_policy_seed(seed, id), e->decisions, e->frames, e->score,
                    e->score/(float)env->max_score, e->episode_return, e->completed, e->capped,
                    (unsigned long long)action_hash[i], (unsigned long long)observation_hash[i],
                    (unsigned long long)world_hash[i], start_rng[i]);
                breakout_exact_require(fflush(fp) == 0, "episode receipt write failed");
                active[i] = 0; remaining--; evaluated++; completed += e->completed; capped += e->capped; score += e->score;
            }
        }
        breakout_exact_require(remaining == 0, "assigned episode exceeded frame cap");
        printf("BREAKOUT_EXACT_PROGRESS evaluated=%u requested=%u\n", evaluated, episodes);
    }
    breakout_exact_require(evaluated == episodes && completed+capped == episodes && fclose(fp) == 0, "incomplete evaluation");
    printf("BREAKOUT_EXACT_EVAL version=1 requested=%u evaluated=%u completed=%u capped=%u score=%llu params=%ld\n",
        episodes, evaluated, completed, capped, (unsigned long long)score, parameters);
    free(history); free(active); free(action_hash); free(observation_hash); free(world_hash); free(start_rng);
    close_pufferl(p);
}
