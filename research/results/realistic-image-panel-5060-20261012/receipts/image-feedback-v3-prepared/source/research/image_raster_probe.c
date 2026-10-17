// Host game/raster checks only. No neural computation or CUDA.
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include ENV_HEADER

static void raster(Env* env) {
#if defined(PUFFER_CONNECT4CNN)
    compute_observation(env);
#elif defined(PUFFER_FLAPPYCNN)
    flappy_compute_observations(env);
#else
    compute_observations(env);
#endif
}

static uint64_t hash(const void* data, size_t count) {
    const unsigned char* bytes = (const unsigned char*)data;
    uint64_t value = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; i++) value = (value ^ bytes[i]) * UINT64_C(1099511628211);
    return value;
}

int main(int argc, char** argv) {
    assert(argc == 3 || argc == 4);
    Ini ini = {0};
    puf_ini_load_file(&ini, "config/default.ini");
    puf_ini_load_file(&ini, argv[1]);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    dict_set(kwargs, "representation", 0);
    dict_set(kwargs, "representation_mode", 0);
#ifdef PUFFER_MAZECNN
    dict_set(kwargs, "num_maps", 32); // Bounded host fixture, not a training recipe.
#endif
    Env* env = (Env*)calloc(1, sizeof(Env));
    obs_t* guarded = (obs_t*)malloc((OBS_SIZE + 2) * sizeof(obs_t));
    obs_t* saved = (obs_t*)malloc(OBS_SIZE * sizeof(obs_t));
    assert(env && guarded && saved);
    guarded[0] = 73; guarded[OBS_SIZE + 1] = 91;
    float action = 0, reward = 0, terminal = 0;
    env->rng = 7;
    env->agents[0].observations = guarded + 1;
    env->agents[0].actions = &action;
    env->agents[0].rewards = &reward;
    env->agents[0].terminals = &terminal;
    puf_init(env, kwargs); puf_reset(env);
    int actions[] = ACT_SIZES;
    FILE* receipt = fopen(argv[2], "w"); assert(receipt);
    fprintf(receipt, "step,rng,tick,reward,terminal\n");
    for (int step = 0; step < 64; step++) {
        action = step % actions[0];
        puf_step(env);
        fprintf(receipt, "%d,%u,%d,%.9g,%.9g\n", step, env->rng, env->tick, reward, terminal);
        for (int rep = 0; rep < IMAGE_REPRESENTATIONS; rep++) {
            env->representation = rep;
            unsigned int rng = env->rng;
            Env before; memcpy(&before, env, sizeof(Env));
            for (int i = 0; i < OBS_SIZE; i++) guarded[i + 1] = -10;
            raster(env);
            assert(!memcmp(&before, env, sizeof(Env)));
            assert(guarded[0] == 73 && guarded[OBS_SIZE + 1] == 91 && rng == env->rng);
            for (int i = 0; i < OBS_SIZE; i++) assert(isfinite(guarded[i + 1]) && guarded[i + 1] >= 0 && guarded[i + 1] <= 1);
            memcpy(saved, guarded + 1, OBS_SIZE * sizeof(obs_t));
            memset(guarded + 1, 0, OBS_SIZE * sizeof(obs_t));
            raster(env);
            assert(!memcmp(saved, guarded + 1, OBS_SIZE * sizeof(obs_t)) && rng == env->rng);
        }
        env->representation = 0;
    }
    fclose(receipt);
    raster(env);
#ifdef CNN_IMAGE_V1
    // The image must not gain the hidden velocity/global-map state.
    memcpy(saved, guarded + 1, OBS_SIZE * sizeof(obs_t));
#if defined(PUFFER_FLAPPYCNN)
    float velocity = env->bird_vy; env->bird_vy += 123;
    raster(env); assert(!memcmp(saved, guarded + 1, OBS_SIZE * sizeof(obs_t))); env->bird_vy = velocity;
#elif defined(PUFFER_PONGCNN) || defined(PUFFER_BREAKOUTCNN)
    float velocity = env->ball_vx; env->ball_vx += 123;
    raster(env); assert(!memcmp(saved, guarded + 1, OBS_SIZE * sizeof(obs_t))); env->ball_vx = velocity;
#elif defined(PUFFER_SNAKECNN)
    int head = env->snake[env->head_ptr], outside = -1;
    for (int i = 0; i < env->width * env->height; i++) {
        if (abs(i / env->width - head / env->width) > 5 || abs(i % env->width - head % env->width) > 5) { outside = i; break; }
    }
    assert(outside >= 0); unsigned char tile = env->grid[outside]; env->grid[outside] = tile == SNAKE_FOOD ? SNAKE_WALL : SNAKE_FOOD;
    raster(env); assert(!memcmp(saved, guarded + 1, OBS_SIZE * sizeof(obs_t))); env->grid[outside] = tile;
#elif defined(PUFFER_MAZECNN)
    int outside = -1;
    for (int y = 0; y < MAX_SIZE && outside < 0; y++) for (int x = 0; x < MAX_SIZE; x++) {
        if (abs(y - env->state.y) > VISION || abs(x - env->state.x) > VISION) { outside = maze_offset(y,x); break; }
    }
    assert(outside >= 0); unsigned char tile = env->state.maze[outside]; env->state.maze[outside] = tile == WALL ? GOAL : WALL;
    raster(env); assert(!memcmp(saved, guarded + 1, OBS_SIZE * sizeof(obs_t))); env->state.maze[outside] = tile;
#elif defined(PUFFER_CONNECT4CNN)
    uint64_t player = env->player_pieces, opponent = env->env_pieces;
    env->player_pieces = 1; env->env_pieces = UINT64_C(1) << 47;
    uint64_t fingerprints[IMAGE_REPRESENTATIONS];
    for (int rep = 0; rep < IMAGE_REPRESENTATIONS; rep++) {
        env->representation = rep; raster(env);
        fingerprints[rep] = hash(guarded + 1, OBS_SIZE * sizeof(obs_t));
        for (int prior = 0; prior < rep; prior++) assert(fingerprints[prior] != fingerprints[rep]);
    }
    env->player_pieces = player; env->env_pieces = opponent; env->representation = 0;
#endif
    raster(env);
#endif
    if (argc == 4) {
        FILE* frame = fopen(argv[3], "wb"); assert(frame);
        fprintf(frame, "P6\n%d %d\n255\n", OBS_WIDTH, OBS_HEIGHT);
        for (int p = 0; p < OBS_HEIGHT * OBS_WIDTH; p++) for (int c = 0; c < 3; c++) {
            int at = p * OBS_CHANNELS + (OBS_CHANNELS == 1 ? 0 : c);
            unsigned char value = (unsigned char)lroundf(255 * guarded[at + 1]);
            assert(fwrite(&value, 1, 1, frame) == 1);
        }
        fclose(frame);
    }
    printf("{\"shape_hwc\":[%d,%d,%d],\"layout\":\"NHWC\",\"dtype\":\"float32\","
        "\"frame_hash_fnv64\":\"%016" PRIx64 "\",\"steps\":64,\"drawings\":%d,"
        "\"neural_execution\":false,\"cuda_execution\":false}\n",
        OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS, hash(guarded + 1, OBS_SIZE * sizeof(obs_t)), IMAGE_REPRESENTATIONS);
    puf_close(env); free(saved); free(guarded); free(env); puf_ini_free(&ini);
    return 0;
}
