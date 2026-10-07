// Evaluation only, included after eval_make. Policy execution stays native C/CUDA.
#include "exact_episode.h"

static void maze_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr, "Maze exact evaluation: %s\n", message); exit(1); }
}
static uint32_t maze_exact_option(Ini* ini, const char* key, uint32_t low, uint32_t high) {
    double value = puf_ini_get(ini, "eval_exact", key);
    maze_exact_require(isfinite(value) && value >= low && value <= high && value == floor(value), key);
    return (uint32_t)value;
}
static void maze_exact_world(Ini* ini) {
    double maps = puf_ini_get(ini, "env", "num_maps"), size = puf_ini_get(ini, "env", "map_size");
    maze_exact_require(isfinite(maps) && maps == floor(maps) && maps >= 1 && maps <= 8192, "num_maps");
    maze_exact_require(isfinite(size) && size == floor(size) && (size == -1 || (size >= 5 && size <= MAX_SIZE)), "map_size");
    uint32_t first = maze_exact_option(ini, "level_offset", 0, 8191);
    uint32_t count = maze_exact_option(ini, "level_count", 1, 8192);
    maze_exact_require((uint64_t)first + count <= (uint32_t)maps, "level panel exceeds table");
    DictItem* mode = dict_find(puf_ini_section(ini, "env", 0), "representation_mode");
    maze_exact_require(!mode || mode->value == 0, "fixed appearance required");
}
static void maze_exact_info() {
#ifdef PUFFER_MAZE
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
        "\"rules\":\"maze-native-v1\",\"receipt_version\":1,"
        "\"evaluation\":\"first-goal-or-native-area-timeout-v1\","
        "\"levels\":\"seed-rotated-cyclic-v1\"}\n", PUFFER_ENV_NAME, encoder, USE_BF16 ? "false" : "true");
}
static void maze_exact_manifest(int argc, char** argv) {
    maze_exact_require(argc == 3, "eval_exact_manifest requires one full INI path");
    Ini ini = {}; puf_ini_load_file(&ini, argv[2]);
    maze_exact_require(!USE_BF16, "float32 build required"); maze_exact_world(&ini);
    uint32_t episodes = maze_exact_option(&ini, "episodes", 1, 1000000);
    uint32_t first = maze_exact_option(&ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = maze_exact_option(&ini, "seed", 0, UINT32_MAX);
    uint32_t slots = maze_exact_option(&ini, "slots", 1, 1024);
    uint32_t level_first = maze_exact_option(&ini, "level_offset", 0, 8191);
    uint32_t level_count = maze_exact_option(&ini, "level_count", 1, 8192);
    maze_exact_require((uint64_t)first + episodes + slots <= UINT32_MAX, "episode IDs overflow");
    Env env = {}; obs_t observations[OBS_SIZE]; float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations; env.agents[0].actions = &action;
    env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
    puf_init(&env, puf_ini_section(&ini, "env", 0));
    int representation = 0;
#ifdef PUFFER_MAZECNN
    representation = env.representation;
#endif
    printf("episode_id,env_seed,policy_seed,representation,level_id,width,height,horizon,start_rng,terminal_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < episodes; i++) {
        uint32_t id = first + i; MazeExactEpisode e;
        maze_exact_require(maze_exact_start(&env, &e, seed, id, level_first, level_count), "invalid starting state");
        unsigned int terminal_rng = env.rng; rand_r(&terminal_rng); rand_r(&terminal_rng);
        printf("%u,%u,%llu,%d,%u,%d,%d,%d,%u,%u,%016llx,%016llx\n", id, c4_exact_env_seed(seed, id),
            (unsigned long long)c4_exact_policy_seed(seed, id), representation, e.level, e.width, e.height,
            e.horizon, env.rng, terminal_rng, (unsigned long long)maze_exact_world_hash(&env),
            (unsigned long long)maze_exact_observation_hash(observations));
    }
    puf_close(&env); puf_ini_free(&ini);
}
__global__ void maze_exact_rng(curandStatePhilox4_32_10_t* states, uint32_t seed, uint32_t first, int slots) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < slots) curand_init(c4_exact_policy_seed(seed, first + i), 0, 0, &states[i]);
}
static void run_maze_exact_eval(Ini* ini, TrainContext* ctx) {
    maze_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU, "float32 native CPU environment required");
    maze_exact_require(ctx->world_size == 1 && puf_ini_get(ini, "train", "gpus") == 1
        && puf_ini_get(ini, "selfplay", "enabled") == 0, "single GPU, no selfplay required");
    maze_exact_world(ini);
    uint32_t episodes = maze_exact_option(ini, "episodes", 1, 1000000);
    uint32_t first = maze_exact_option(ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = maze_exact_option(ini, "seed", 0, UINT32_MAX);
    uint32_t slots = maze_exact_option(ini, "slots", 1, 1024);
    uint32_t level_first = maze_exact_option(ini, "level_offset", 0, 8191);
    uint32_t level_count = maze_exact_option(ini, "level_count", 1, 8192);
    maze_exact_require((uint64_t)first + episodes + slots <= UINT32_MAX, "episode IDs overflow");
    const char* output = puf_ini_get_str(ini, "eval_exact", "output");
    maze_exact_require(output && output[0] && strcmp(output, "None"), "explicit output required");
    FILE* fp = fopen(output, "wx"); maze_exact_require(fp != NULL, "output exists or cannot be created");
    fprintf(fp, "version,block_seed,episode_id,slot,representation,env_seed,policy_seed,level_id,width,height,horizon,decisions,success,episode_return,end_reason,native_log_entries,end_rng,action_hash,start_observation_hash,start_world_hash,start_rng\n");
    maze_exact_require(fflush(fp) == 0, "header write failed");
    char count[32]; snprintf(count, sizeof(count), "%u", slots);
    puf_ini_put(ini, "base.async", "0"); puf_ini_put(ini, "base.eval_agents", count);
    puf_ini_put(ini, "vec.total_agents", count); puf_ini_put(ini, "vec.num_buffers", "1");
    puf_ini_put(ini, "vec.num_threads", "1"); puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "vec.hist_policy_percent", "0"); puf_ini_put(ini, "train.horizon", "1");
    puf_ini_put(ini, "train.minibatch_size", count); puf_ini_put(ini, "train.replay_ratio", "1");
    const char* checkpoint = puf_ini_get_str(ini, "base", "load_model_path");
    maze_exact_require(checkpoint && strcmp(checkpoint, "None"), "explicit checkpoint required");
    PuffeRL* p = eval_make(ini, ctx, EVAL_SCORE, 0); VecEnv* v = p->vec;
    maze_exact_require(v->size == (int)slots && v->total_agents == (int)slots, "one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape); struct stat st;
    maze_exact_require(stat(checkpoint, &st) == 0 && st.st_size == parameters * (long)sizeof(float), "checkpoint size mismatch");
    FILE* weights = fopen(checkpoint, "rb"); maze_exact_require(weights != NULL, "cannot audit weights");
    float value;
    for (long i = 0; i < parameters; i++)
        maze_exact_require(fread(&value, sizeof(value), 1, weights) == 1 && isfinite(value), "nonfinite/truncated weights");
    maze_exact_require(fclose(weights) == 0 && cudaStreamSynchronize(p->default_stream) == cudaSuccess, "checkpoint upload failed");
    cudaStream_t stream = p->streams[0];
    unsigned char* active = (unsigned char*)calloc(slots, 1);
    MazeExactEpisode* counters = (MazeExactEpisode*)calloc(slots, sizeof(MazeExactEpisode));
    uint64_t* action_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* observation_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* world_hash = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint32_t* start_rng = (uint32_t*)calloc(slots, sizeof(uint32_t));
    maze_exact_require(active && counters && action_hash && observation_hash && world_hash && start_rng, "allocation failed");
    uint32_t completed = 0, successes = 0, timeouts = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots, episodes - base); int horizon = 0;
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            maze_exact_require(maze_exact_start(env, &counters[i], seed, first + base + i, level_first, level_count), "invalid start");
            observation_hash[i] = maze_exact_observation_hash(env->agents[0].observations);
            world_hash[i] = maze_exact_world_hash(env); start_rng[i] = env->rng;
            active[i] = i < n; action_hash[i] = UINT64_C(14695981039346656037);
            if (i < n) horizon = max(horizon, counters[i].horizon);
        }
        maze_exact_rng<<<grid_size(slots), BLOCK_SIZE, 0, stream>>>(p->rng_states[0], seed, first + base, slots);
        uint32_t remaining = n;
        for (int decision = 1; decision <= horizon && remaining; decision++) {
            cpu_upload(p, 0, slots, stream); pufferl_forward(p, 0, 0, stream);
            maze_exact_require(cudaMemcpyAsync(v->actions, p->env.actions.data, slots * sizeof(float),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess, "action download failed");
            maze_exact_require(cudaStreamSynchronize(stream) == cudaSuccess, "GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i]; MazeExactEpisode* e = &counters[i]; float action = v->actions[i];
                maze_exact_require(maze_exact_step(env, e, action), "action/episode accounting mismatch");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action) * UINT64_C(1099511628211);
                if (!e->end_reason) continue;
                uint32_t id = first + base + i; int representation = 0;
#ifdef PUFFER_MAZECNN
                representation = env->representation;
#endif
                fprintf(fp, "1,%u,%u,%u,%d,%u,%llu,%u,%d,%d,%d,%d,%d,%a,%d,%d,%u,%016llx,%016llx,%016llx,%u\n",
                    seed, id, i, representation, c4_exact_env_seed(seed, id),
                    (unsigned long long)c4_exact_policy_seed(seed, id), e->level, e->width, e->height, e->horizon,
                    e->decisions, e->success, e->episode_return, e->end_reason, e->native_logs, env->rng,
                    (unsigned long long)action_hash[i], (unsigned long long)observation_hash[i],
                    (unsigned long long)world_hash[i], start_rng[i]);
                maze_exact_require(fflush(fp) == 0, "episode receipt write failed");
                active[i] = 0; remaining--; completed++; successes += e->success; timeouts += e->end_reason == 2;
            }
        }
        maze_exact_require(remaining == 0, "assigned episode exceeded native horizon");
        printf("MAZE_EXACT_PROGRESS completed=%u requested=%u\n", completed, episodes);
    }
    maze_exact_require(completed == episodes && fclose(fp) == 0, "incomplete evaluation");
    printf("MAZE_EXACT_EVAL version=1 requested=%u completed=%u successes=%u timeouts=%u params=%ld\n",
        episodes, completed, successes, timeouts, parameters);
    free(active); free(counters); free(action_hash); free(observation_hash); free(world_hash); free(start_rng); close_pufferl(p);
}
