// Included after eval_make. Native inference with evaluation-only match receipts.
#include "exact_episode.h"

static void pong_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "Pong exact evaluation: %s\n", message); exit(1); }
}
static uint32_t pong_exact_option(Ini* ini, const char* key, uint32_t low, uint32_t high) {
    double value = puf_ini_get(ini, "eval_exact", key);
    pong_exact_require(isfinite(value) && value >= low && value <= high && value == floor(value), key);
    return (uint32_t)value;
}
static void pong_exact_world(Ini* ini) {
    const char* positive[] = {"width", "height", "paddle_width", "paddle_height", "ball_width",
        "ball_height", "paddle_speed", "ball_initial_speed_x", "ball_max_speed_y"};
    for (int i = 0; i < 9; i++) {
        double value = puf_ini_get(ini, "env", positive[i]);
        pong_exact_require(isfinite(value) && value > 0 && value <= 1000000 && (float)value > 0, positive[i]);
    }
    const char* nonnegative[] = {"ball_initial_speed_y", "ball_speed_y_increment"};
    for (int i = 0; i < 2; i++) {
        double value = puf_ini_get(ini, "env", nonnegative[i]);
        pong_exact_require(isfinite(value) && value >= 0 && value <= 1000000, nonnegative[i]);
    }
    const char* integers[] = {"max_score", "frameskip"};
    for (int i = 0; i < 2; i++) {
        double value = puf_ini_get(ini, "env", integers[i]);
        pong_exact_require(isfinite(value) && value >= 1 && value <= 1000000 && value == floor(value), integers[i]);
    }
    pong_exact_require(puf_ini_get(ini, "env", "continuous") == 0, "discrete actions required");
    pong_exact_require(puf_ini_get(ini, "env", "paddle_height") < puf_ini_get(ini, "env", "height")
        && puf_ini_get(ini, "env", "ball_height") < puf_ini_get(ini, "env", "height")
        && puf_ini_get(ini, "env", "ball_width") < puf_ini_get(ini, "env", "width"), "objects exceed world");
    pong_exact_require(puf_ini_get(ini, "env", "ball_initial_speed_y") <= puf_ini_get(ini, "env", "ball_max_speed_y"),
        "initial vertical speed exceeds maximum");
#ifdef PUFFER_PONGCNN
    pong_exact_require(puf_ini_get(ini, "env", "representation_mode") == 0, "fixed appearance required");
#endif
}
static void pong_exact_info() {
#ifdef PUFFER_PONG
    const int representations = 1;
#else
    const int representations = PONGCNN_NUM_REPRESENTATIONS;
#endif
#ifdef PUFFER_PONG
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
        "\"rules\":\"pong-native-v1\",\"receipt_version\":1,\"representation_count\":%d,"
        "\"evaluation\":\"first-match-terminal-or-decision-cap-v1\"}\n",
        PUFFER_ENV_NAME, encoder, USE_BF16 ? "false" : "true", representations);
}
// Actual binary's host reset/raster before any CUDA device query or policy.
static void pong_exact_manifest(int argc, char** argv) {
    pong_exact_require(argc == 3, "eval_exact_manifest requires one full INI path");
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    pong_exact_require(!USE_BF16, "float32 manifest required");
    pong_exact_world(&ini);
    uint32_t episodes = pong_exact_option(&ini, "episodes", 1, 1000000);
    uint32_t first = pong_exact_option(&ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = pong_exact_option(&ini, "seed", 0, UINT32_MAX);
    uint32_t slots = pong_exact_option(&ini, "slots", 1, 1024);
    pong_exact_option(&ini, "max_decisions", 1, 16777216);
    pong_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX, "episode IDs overflow");
    Env env = {}; obs_t observations[OBS_SIZE];
    float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations; env.agents[0].actions = &action;
    env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
    puf_init(&env, puf_ini_section(&ini, "env", 0));
    int representation = 0;
#ifdef PUFFER_PONGCNN
    representation = env.representation;
#endif
    printf("episode_id,env_seed,policy_seed,representation,start_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < episodes; i++) {
        uint32_t id = first+i; PongExactEpisode episode;
        pong_exact_start(&env, &episode, seed, id);
        printf("%u,%u,%llu,%d,%u,%016llx,%016llx\n", id, c4_exact_env_seed(seed, id),
            (unsigned long long)c4_exact_policy_seed(seed, id), representation, env.rng,
            (unsigned long long)pong_exact_world_hash(&env),
            (unsigned long long)pong_exact_observation_hash(observations));
    }
    puf_close(&env); puf_ini_free(&ini);
}
__global__ void pong_exact_rng(curandStatePhilox4_32_10_t* states, uint32_t seed, uint32_t first, int slots) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < slots) curand_init(c4_exact_policy_seed(seed, first+i), 0, 0, &states[i]);
}
static void run_pong_exact_eval(Ini* ini, TrainContext* ctx) {
    pong_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU, "float32 native CPU environment required");
    pong_exact_require(ctx->world_size == 1 && puf_ini_get(ini, "selfplay", "enabled") == 0, "single GPU, no selfplay required");
    pong_exact_world(ini);
    uint32_t episodes = pong_exact_option(ini, "episodes", 1, 1000000);
    uint32_t first = pong_exact_option(ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = pong_exact_option(ini, "seed", 0, UINT32_MAX);
    uint32_t slots = pong_exact_option(ini, "slots", 1, 1024);
    int cap = pong_exact_option(ini, "max_decisions", 1, 16777216);
    pong_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX, "episode IDs overflow");
    const char* output = puf_ini_get_str(ini, "eval_exact", "output");
    pong_exact_require(output && output[0] && strcmp(output, "None"), "explicit output required");
    FILE* fp = fopen(output, "wx"); pong_exact_require(fp != NULL, "output exists or cannot be created");
    fprintf(fp, "version,block_seed,episode_id,slot,representation,env_seed,policy_seed,decisions,last_rally_decisions,right_points,left_points,episode_return,completed,capped,point_fraction_lower,point_fraction_upper,native_perf,action_hash,start_observation_hash,start_world_hash,start_rng\n");
    pong_exact_require(fflush(fp) == 0, "header write failed");
    char count[32]; snprintf(count, sizeof(count), "%u", slots);
    puf_ini_put(ini, "base.async", "0"); puf_ini_put(ini, "base.eval_agents", count);
    puf_ini_put(ini, "vec.total_agents", count); puf_ini_put(ini, "vec.num_buffers", "1");
    puf_ini_put(ini, "vec.num_threads", "1"); puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "vec.hist_policy_percent", "0"); puf_ini_put(ini, "train.horizon", "1");
    puf_ini_put(ini, "train.minibatch_size", count); puf_ini_put(ini, "train.replay_ratio", "1");
    const char* checkpoint = puf_ini_get_str(ini, "base", "load_model_path");
    pong_exact_require(checkpoint && strcmp(checkpoint, "None"), "explicit checkpoint required");
    PuffeRL* p = eval_make(ini, ctx, EVAL_SCORE, 0); VecEnv* v = p->vec;
    pong_exact_require(v->size == (int)slots && v->total_agents == (int)slots, "one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape); struct stat st;
    pong_exact_require(stat(checkpoint, &st) == 0 && st.st_size == parameters*(long)sizeof(float), "checkpoint size mismatch");
    FILE* weights = fopen(checkpoint, "rb"); pong_exact_require(weights != NULL, "cannot audit checkpoint");
    float value;
    for (long i = 0; i < parameters; i++)
        pong_exact_require(fread(&value, sizeof(value), 1, weights) == 1 && isfinite(value), "nonfinite/truncated weights");
    pong_exact_require(fclose(weights) == 0, "weights read failed");
    pong_exact_require(cudaStreamSynchronize(p->default_stream) == cudaSuccess, "checkpoint upload failed");
    cudaStream_t stream = p->streams[0];
    PongExactEpisode* history = (PongExactEpisode*)calloc(slots, sizeof(PongExactEpisode));
    unsigned char* active = (unsigned char*)calloc(slots, 1);
    uint64_t* action_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* observation_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* world_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint32_t* start_rng = (uint32_t*)calloc(slots, sizeof(uint32_t));
    pong_exact_require(history && active && action_hash && observation_hash && world_hash && start_rng, "allocation failed");
    uint32_t evaluated = 0, completed = 0, capped = 0, wins = 0;
    uint64_t right_points = 0, left_points = 0; double lower_sum = 0, upper_sum = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots, episodes-base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i]; pong_exact_start(env, &history[i], seed, first+base+i);
            pong_exact_require(env->tick == 0 && env->score_l == 0 && env->score_r == 0
                && env->log.n == 0 && env->paddle_dir == 0 && env->win == 0, "starting state mismatch");
            observation_hash[i] = pong_exact_observation_hash(env->agents[0].observations);
            world_hash[i] = pong_exact_world_hash(env); start_rng[i] = env->rng;
            active[i] = i < n; action_hash[i] = UINT64_C(14695981039346656037);
        }
        pong_exact_rng<<<grid_size(slots), BLOCK_SIZE, 0, stream>>>(p->rng_states[0], seed, first+base, slots);
        uint32_t remaining = n;
        for (int decision = 1; decision <= cap && remaining; decision++) {
            cpu_upload(p, 0, slots, stream); pufferl_forward(p, 0, 0, stream);
            pong_exact_require(cudaMemcpyAsync(v->actions, p->env.actions.data, slots*sizeof(float),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess, "action download failed");
            pong_exact_require(cudaStreamSynchronize(stream) == cudaSuccess, "GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i]; PongExactEpisode* e = &history[i]; float action = v->actions[i];
                pong_exact_require(action == 0 || action == 1 || action == 2, "invalid policy action");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action)*UINT64_C(1099511628211);
                pong_exact_require(pong_exact_step(env, e, action, cap) && e->decisions == decision, "match accounting mismatch");
                if (!e->completed && !e->capped) continue;
                double lower, upper;
                pong_exact_require(pong_exact_bounds(e->right, e->left, env->max_score, e->completed, &lower, &upper), "impossible final-score bounds");
                char perf[64] = "";
                if (e->completed) snprintf(perf, sizeof(perf), "%a", env->log.perf);
                int representation = 0;
#ifdef PUFFER_PONGCNN
                representation = env->representation;
#endif
                uint32_t id = first+base+i;
                fprintf(fp, "1,%u,%u,%u,%d,%u,%llu,%d,%d,%u,%u,%a,%d,%d,%a,%a,%s,%016llx,%016llx,%016llx,%u\n",
                    seed, id, i, representation, c4_exact_env_seed(seed, id), (unsigned long long)c4_exact_policy_seed(seed, id),
                    e->decisions, e->rally_decisions, e->right, e->left, e->episode_return, e->completed, e->capped,
                    lower, upper, perf, (unsigned long long)action_hash[i], (unsigned long long)observation_hash[i],
                    (unsigned long long)world_hash[i], start_rng[i]);
                pong_exact_require(fflush(fp) == 0, "episode receipt write failed");
                active[i] = 0; remaining--; evaluated++; completed += e->completed; capped += e->capped;
                wins += e->completed && e->right == env->max_score;
                right_points += e->right; left_points += e->left; lower_sum += lower; upper_sum += upper;
            }
        }
        pong_exact_require(remaining == 0, "assigned match exceeded decision cap");
        printf("PONG_EXACT_PROGRESS evaluated=%u requested=%u\n", evaluated, episodes);
    }
    pong_exact_require(evaluated == episodes && completed+capped == episodes && fclose(fp) == 0, "incomplete evaluation");
    printf("PONG_EXACT_EVAL version=1 requested=%u evaluated=%u completed=%u capped=%u wins=%u right=%llu left=%llu params=%ld\n",
        episodes, evaluated, completed, capped, wins, (unsigned long long)right_points, (unsigned long long)left_points, parameters);
    printf("PONG_EXACT_BOUNDS mean_lower=%.17g mean_upper=%.17g kind=censoring-not-confidence\n", lower_sum/episodes, upper_sum/episodes);
    free(history); free(active); free(action_hash); free(observation_hash); free(world_hash); free(start_rng);
    close_pufferl(p);
}
