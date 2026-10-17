// Sprite-suite research evaluator: fixed episodes, every movement/gaze head.
#include "../ocean/connect4cnn/exact_protocol.h"

static uint32_t sprite_eval_option(Ini* ini, const char* key, uint32_t low, uint32_t high) {
    double value = puf_ini_get(ini, "sprite_eval", key);
    assert(isfinite(value) && value >= low && value <= high && value == floor(value));
    return value;
}

static uint64_t sprite_eval_hash(const void* data, size_t size) {
    const unsigned char* bytes = (const unsigned char*)data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; i++) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t sprite_eval_start(Env* env, uint32_t seed, uint32_t episode) {
    env->rng = c4_exact_env_seed(seed, episode);
    env->log = {};
    puf_reset(env);
    uint32_t fields[] = {(uint32_t)env->player, env->rng};
    return sprite_eval_hash(env->grid, sizeof(env->grid))
        ^ sprite_eval_hash(fields, sizeof(fields));
}

// Scalar game/raster manifest only; returns before any CUDA query/allocation.
static void sprite_eval_manifest(int argc, char** argv) {
    assert(argc == 3 && "sprite_eval_manifest requires one full INI path");
    Ini ini = {};
    puf_ini_load_file(&ini, argv[2]);
    uint32_t count = sprite_eval_option(&ini, "episodes", 1, 1000000);
    uint32_t first = sprite_eval_option(&ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = sprite_eval_option(&ini, "seed", 0, UINT32_MAX);
    assert((uint64_t)first + count <= UINT32_MAX);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    int size = puf_observation_size(kwargs);
    float action[NUM_ATNS] = {0}, reward = 0, terminal = 0;
    Env env = {};
    env.agents[0] = {.observations = (obs_t*)calloc(size, sizeof(obs_t)),
        .actions = action, .rewards = &reward, .terminals = &terminal};
    assert(env.agents[0].observations);
    puf_init(&env, kwargs);
    printf("episode_id,env_seed,policy_seed,start_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < count; i++) {
        uint32_t id = first + i;
        uint64_t world = sprite_eval_start(&env, seed, id);
        printf("%u,%u,%llu,%u,%016llx,%016llx\n", id, c4_exact_env_seed(seed, id),
            (unsigned long long)c4_exact_policy_seed(seed, id), env.rng,
            (unsigned long long)world,
            (unsigned long long)sprite_eval_hash(env.agents[0].observations, size * sizeof(obs_t)));
    }
    puf_close(&env);
    free(env.agents[0].observations);
    puf_ini_free(&ini);
}

__global__ void sprite_eval_rng(curandStatePhilox4_32_10_t* states,
        uint32_t seed, uint32_t first, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        curand_init(c4_exact_policy_seed(seed, first + i), 0, 0, &states[i]);
    }
}

static void run_sprite_exact_eval(Ini* ini, TrainContext* ctx) {
    assert(!USE_BF16 && PUF_BACKEND == PUF_CPU && ctx->world_size == 1);
    assert(puf_ini_get(ini, "selfplay", "enabled") == 0);
    uint32_t episodes = sprite_eval_option(ini, "episodes", 1, 1000000);
    uint32_t first = sprite_eval_option(ini, "episode_offset", 0, UINT32_MAX);
    uint32_t seed = sprite_eval_option(ini, "seed", 0, UINT32_MAX);
    uint32_t slots = sprite_eval_option(ini, "slots", 1, 64);
    uint32_t cap = sprite_eval_option(ini, "max_steps", 1, 1000000);
    assert((uint64_t)first + episodes + slots <= UINT32_MAX);
    const char* output = puf_ini_get_str(ini, "sprite_eval", "output");
    assert(output && output[0] && strcmp(output, "None"));
    FILE* fp = fopen(output, "wx");
    assert(fp && "Refusing to overwrite evaluation evidence");
    fprintf(fp, "episode_id,slot,env_seed,policy_seed,start_rng,start_world_hash,"
        "start_observation_hash,decisions,perf,successes,deaths,episode_return,action_hash\n");
    assert(fflush(fp) == 0);
    char count[32];
    snprintf(count, sizeof(count), "%u", slots);
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
    assert(checkpoint && strcmp(checkpoint, "None"));
    PuffeRL* p = eval_make(ini, ctx, EVAL_SCORE, 0);
    VecEnv* v = p->vec;
    assert(v->size == (int)slots && v->total_agents == (int)slots);
    long parameters = numel(p->policies[0].master_weights.shape);
    FILE* weights = fopen(checkpoint, "rb");
    assert(weights);
    float value;
    for (long i = 0; i < parameters; i++) {
        assert(fread(&value, sizeof(value), 1, weights) == 1 && isfinite(value));
    }
    assert(fgetc(weights) == EOF && fclose(weights) == 0);
    assert(cudaStreamSynchronize(p->default_stream) == cudaSuccess);
    cudaStream_t stream = p->streams[0];
    unsigned char* active = (unsigned char*)calloc(slots, 1);
    float* returns = (float*)calloc(slots, sizeof(float));
    uint64_t* actions = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* observations = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint64_t* worlds = (uint64_t*)calloc(slots, sizeof(uint64_t));
    uint32_t* starts = (uint32_t*)calloc(slots, sizeof(uint32_t));
    assert(active && returns && actions && observations && worlds && starts);
    int obs_size = puf_observation_size(puf_ini_section(ini, "env", 0));
    const int action_sizes[] = ACT_SIZES;
    uint32_t completed = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots, episodes - base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            worlds[i] = sprite_eval_start(env, seed, first + base + i);
            observations[i] = sprite_eval_hash(env->agents[0].observations,
                obs_size * sizeof(obs_t));
            starts[i] = env->rng;
            active[i] = i < n;
            returns[i] = 0;
            actions[i] = UINT64_C(14695981039346656037);
            v->terminals[i] = 1; // Reset every policy's normal recurrent-state path.
        }
        sprite_eval_rng<<<grid_size(slots), BLOCK_SIZE, 0, stream>>>(
            p->rng_states[0], seed, first + base, slots);
        uint32_t remaining = n;
        for (uint32_t decision = 1; decision <= cap && remaining; decision++) {
            cpu_upload(p, 0, slots, stream);
            pufferl_forward(p, 0, 0, stream);
            assert(cudaMemcpyAsync(v->actions, p->env.actions.data, slots * NUM_ATNS * sizeof(float),
                cudaMemcpyDeviceToHost, stream) == cudaSuccess);
            assert(cudaStreamSynchronize(stream) == cudaSuccess);
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) {
                    continue;
                }
                Env* env = &v->envs[i];
                for (int head = 0; head < NUM_ATNS; head++) {
                    float action = env->agents[0].actions[head];
                    assert(isfinite(action) && action >= 0 && action < action_sizes[head]);
                    assert(action == floorf(action));
                    actions[i] = (actions[i] ^ (uint32_t)action) * UINT64_C(1099511628211);
                }
                puf_step(env);
                assert(isfinite(v->rewards[i]));
                returns[i] += v->rewards[i];
                if (!v->terminals[i]) {
                    assert(env->log.n == 0 && env->tick == (int)decision);
                    continue;
                }
                Log log = env->log;
                assert(v->terminals[i] == 1 && log.n == 1 && log.episode_length == decision);
                assert(log.episode_return == returns[i] && log.perf >= 0 && log.perf <= 1);
                assert((log.successes == 0 || log.successes == 1)
                    && (log.deaths == 0 || log.deaths == 1));
                assert(log.successes + log.deaths <= 1);
                uint32_t id = first + base + i;
                fprintf(fp, "%u,%u,%u,%llu,%u,%016llx,%016llx,%u,%a,%.0f,%.0f,%a,%016llx\n",
                    id, i, c4_exact_env_seed(seed, id),
                    (unsigned long long)c4_exact_policy_seed(seed, id), starts[i],
                    (unsigned long long)worlds[i], (unsigned long long)observations[i], decision,
                    log.perf, log.successes, log.deaths, log.episode_return,
                    (unsigned long long)actions[i]);
                assert(fflush(fp) == 0);
                active[i] = 0;
                remaining--;
                completed++;
            }
        }
        assert(remaining == 0 && "Assigned episode exceeded declared cap; preserve partial CSV");
    }
    assert(completed == episodes && fclose(fp) == 0);
    printf("SPRITE_EXACT_EVAL version=2 requested=%u completed=%u params=%ld heads=%d\n",
        episodes, completed, parameters, NUM_ATNS);
    free(active);
    free(returns);
    free(actions);
    free(observations);
    free(worlds);
    free(starts);
    close_pufferl(p);
}
