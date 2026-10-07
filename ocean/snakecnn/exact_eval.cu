// Included after eval_make. Exact native Snake episode allocation, evaluation only.
#include "exact_episode.h"

static void snake_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "Snake exact evaluation: %s\n", message); exit(1); }
}
static uint32_t snake_exact_option(Ini* ini, const char* key, uint32_t low, uint32_t high) {
    double value = puf_ini_get(ini, "eval_exact", key);
    snake_exact_require(isfinite(value) && value >= low && value <= high && value == floor(value), key);
    return (uint32_t)value;
}
static void snake_exact_world(Ini* ini) {
    // The shared initializer validates every game setting before allocation.
    DictItem* mode = dict_find(puf_ini_section(ini, "env", 0), "representation_mode");
    snake_exact_require(!mode || mode->value == 0, "fixed appearance required");
}
static void snake_exact_info() {
#ifdef PUFFER_SNAKEBENCH
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
        "\"rules\":\"snake-local-episodic-v1\",\"receipt_version\":1,"
        "\"evaluation\":\"first-death-or-game-horizon-v1\"}\n",
        PUFFER_ENV_NAME, encoder, USE_BF16 ? "false" : "true");
}
static void snake_exact_manifest(int argc, char** argv) {
    snake_exact_require(argc == 3, "eval_exact_manifest requires one full INI path");
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    snake_exact_require(!USE_BF16, "float32 manifest required");
    snake_exact_world(&ini);
    uint32_t episodes = snake_exact_option(&ini, "episodes", 1, 1000000);
    uint32_t first = snake_exact_option(&ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = snake_exact_option(&ini, "seed", 0, UINT32_MAX);
    uint32_t slots = snake_exact_option(&ini, "slots", 1, 1024);
    snake_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX, "episode IDs overflow");
    Env env = {}; obs_t observations[OBS_SIZE];
    float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations; env.agents[0].actions = &action;
    env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
    puf_init(&env, puf_ini_section(&ini, "env", 0));
    int representation = SNAKEBENCH_PIXELS ? env.representation : 0;
    printf("episode_id,env_seed,policy_seed,representation,start_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < episodes; i++) {
        uint32_t id = first+i; SnakeExactEpisode episode;
        snake_exact_start(&env, &episode, seed, id);
        snake_exact_require(snake_exact_valid(&env, &episode), "invalid starting state");
        printf("%u,%u,%llu,%d,%u,%016llx,%016llx\n", id, c4_exact_env_seed(seed, id),
            (unsigned long long)c4_exact_policy_seed(seed, id), representation, env.rng,
            (unsigned long long)snake_exact_world_hash(&env),
            (unsigned long long)snake_exact_observation_hash(observations));
    }
    puf_close(&env); puf_ini_free(&ini);
}
__global__ void snake_exact_rng(curandStatePhilox4_32_10_t* states, uint32_t seed, uint32_t first, int slots) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < slots) curand_init(c4_exact_policy_seed(seed, first+i), 0, 0, &states[i]);
}
static void run_snake_exact_eval(Ini* ini, TrainContext* ctx) {
    snake_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU, "float32 native CPU environment required");
    snake_exact_require(ctx->world_size == 1 && puf_ini_get(ini, "train", "gpus") == 1
        && puf_ini_get(ini, "selfplay", "enabled") == 0, "single GPU, no selfplay required");
    snake_exact_world(ini);
    uint32_t episodes = snake_exact_option(ini, "episodes", 1, 1000000);
    uint32_t first = snake_exact_option(ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = snake_exact_option(ini, "seed", 0, UINT32_MAX);
    uint32_t slots = snake_exact_option(ini, "slots", 1, 1024);
    snake_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX, "episode IDs overflow");
    const char* output = puf_ini_get_str(ini, "eval_exact", "output");
    snake_exact_require(output && output[0] && strcmp(output, "None"), "explicit output required");
    FILE* fp = fopen(output, "wx"); snake_exact_require(fp != NULL, "output exists or cannot be created");
    fprintf(fp, "version,block_seed,episode_id,slot,representation,env_seed,policy_seed,decisions,score,foods,perf,episode_return,end_reason,action_hash,start_observation_hash,start_world_hash,start_rng\n");
    snake_exact_require(fflush(fp) == 0, "header write failed");
    char count[32]; snprintf(count, sizeof(count), "%u", slots);
    puf_ini_put(ini, "base.async", "0"); puf_ini_put(ini, "base.eval_agents", count);
    puf_ini_put(ini, "vec.total_agents", count); puf_ini_put(ini, "vec.num_buffers", "1");
    puf_ini_put(ini, "vec.num_threads", "1"); puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "vec.hist_policy_percent", "0"); puf_ini_put(ini, "train.horizon", "1");
    puf_ini_put(ini, "train.minibatch_size", count); puf_ini_put(ini, "train.replay_ratio", "1");
    const char* checkpoint = puf_ini_get_str(ini, "base", "load_model_path");
    snake_exact_require(checkpoint && strcmp(checkpoint, "None"), "explicit checkpoint required");
    PuffeRL* p = eval_make(ini, ctx, EVAL_SCORE, 0);
    VecEnv* v = p->vec;
    snake_exact_require(v->size == (int)slots && v->total_agents == (int)slots, "one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape);
    struct stat st;
    snake_exact_require(stat(checkpoint, &st) == 0 && st.st_size == parameters*(long)sizeof(float), "checkpoint size mismatch");
    FILE* weights = fopen(checkpoint, "rb"); snake_exact_require(weights != NULL, "cannot audit weights");
    float value;
    for (long i = 0; i < parameters; i++)
        snake_exact_require(fread(&value, sizeof(value), 1, weights) == 1 && isfinite(value), "nonfinite/truncated weights");
    snake_exact_require(fclose(weights) == 0 && cudaStreamSynchronize(p->default_stream) == cudaSuccess, "checkpoint upload failed");
    cudaStream_t stream = p->streams[0];
    unsigned char* active = (unsigned char*)calloc(slots, 1);
    SnakeExactEpisode* counters = (SnakeExactEpisode*)calloc(slots, sizeof(SnakeExactEpisode));
    uint64_t* action_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* observation_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* world_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint32_t* start_rng = (uint32_t*)calloc(slots, sizeof(uint32_t));
    snake_exact_require(active && counters && action_hash && observation_hash && world_hash && start_rng, "allocation failed");
    uint32_t completed = 0, deaths = 0, horizons = 0;
    uint64_t scores = 0, foods = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots, episodes-base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            snake_exact_start(env, &counters[i], seed, first+base+i);
            snake_exact_require(snake_exact_valid(env, &counters[i]), "invalid starting state");
            observation_hash[i] = snake_exact_observation_hash(env->agents[0].observations);
            world_hash[i] = snake_exact_world_hash(env); start_rng[i] = env->rng;
            active[i] = i < n; action_hash[i] = UINT64_C(14695981039346656037);
        }
        snake_exact_rng<<<grid_size(slots), BLOCK_SIZE, 0, stream>>>(p->rng_states[0], seed, first+base, slots);
        uint32_t remaining = n;
        int horizon = v->envs[0].max_steps;
        for (int decision = 1; decision <= horizon && remaining; decision++) {
            cpu_upload(p, 0, slots, stream); pufferl_forward(p, 0, 0, stream);
            snake_exact_require(cudaMemcpyAsync(v->actions, p->env.actions.data, slots*sizeof(float),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess, "action download failed");
            snake_exact_require(cudaStreamSynchronize(stream) == cudaSuccess, "GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i]; SnakeExactEpisode* e = &counters[i];
                float action = v->actions[i];
                snake_exact_require(snake_exact_step(env, e, action), "action/episode accounting mismatch");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action)*UINT64_C(1099511628211);
                if (!e->end_reason) continue;
                uint32_t id = first+base+i;
                int representation = SNAKEBENCH_PIXELS ? env->representation : 0;
                fprintf(fp, "1,%u,%u,%u,%d,%u,%llu,%d,%d,%d,%a,%a,%d,%016llx,%016llx,%016llx,%u\n",
                    seed, id, i, representation, c4_exact_env_seed(seed, id),
                    (unsigned long long)c4_exact_policy_seed(seed, id), e->decisions, e->score, e->foods,
                    env->log.perf, e->episode_return, e->end_reason, (unsigned long long)action_hash[i],
                    (unsigned long long)observation_hash[i], (unsigned long long)world_hash[i], start_rng[i]);
                snake_exact_require(fflush(fp) == 0, "episode receipt write failed");
                active[i] = 0; remaining--; completed++; scores += e->score; foods += e->foods;
                deaths += e->end_reason == 1; horizons += e->end_reason == 2;
            }
        }
        snake_exact_require(remaining == 0, "assigned episode exceeded declared game horizon");
        printf("SNAKE_EXACT_PROGRESS completed=%u requested=%u\n", completed, episodes);
    }
    snake_exact_require(completed == episodes && fclose(fp) == 0, "incomplete evaluation");
    printf("SNAKE_EXACT_EVAL version=1 requested=%u completed=%u deaths=%u horizons=%u score=%llu foods=%llu params=%ld\n",
        episodes, completed, deaths, horizons, (unsigned long long)scores, (unsigned long long)foods, parameters);
    free(active); free(counters); free(action_hash); free(observation_hash); free(world_hash); free(start_rng);
    close_pufferl(p);
}
