// Pixel adaptation of ocean/flappy/flappy.h, blob d5b194c7fc9ccde430689b1562c975203cc58f0d.
// Same game/physics/RNG; policy observations are generated in memory.
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include "../connect4cnn/appearance.h"

#define ACT_SIZES {2}
#define OBS_CHANNELS 1
#define OBS_HEIGHT 36
#define OBS_WIDTH 44
#define OBS_SIZE (OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH)
#define NUM_ATNS 1
#define FLAPPYCNN_NUM_REPRESENTATIONS 4

#define FLAPPY_NUM_PIPES 3
#define FLAPPY_NOOP 0
#define FLAPPY_FLAP 1

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

typedef struct Client Client;
struct Client {
    Texture2D puffer;
};

typedef struct Pipe Pipe;
struct Pipe {
    float x;
    float gap_y;
    int passed;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    Client* client;
    unsigned int rng;

    int width;
    int height;
    int max_steps;
    float gravity;
    float flap_velocity;
    float pipe_speed;
    float pipe_gap;
    float pipe_width;
    float pipe_spacing;
    float first_pipe_x;
    float bird_x;
    float bird_radius;
    float alive_reward;
    float pass_reward;
    float crash_reward;
    float center_reward;

    float bird_y;
    float bird_vy;
    int tick;
    int score;
    float episode_return;
    Pipe pipes[FLAPPY_NUM_PIPES];
    int representation;
};
typedef Env Flappy;

static inline float flappy_randf(Flappy* env) {
    return (float)rand_r(&env->rng) / (float)RAND_MAX;
}

static inline float flappy_clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

static inline float flappy_pipe_gap_y(Flappy* env) {
    float margin = env->pipe_gap * 0.6f;
    return margin + flappy_randf(env) * (env->height - 2.0f * margin);
}

static inline void flappy_init_pipe(Flappy* env, int i, float x) {
    env->pipes[i].x = x;
    env->pipes[i].gap_y = flappy_pipe_gap_y(env);
    env->pipes[i].passed = 0;
}

static inline void flappy_next_pipes(Flappy* env, Pipe** first, Pipe** second) {
    *first = NULL;
    *second = NULL;

    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        float dx = pipe->x + env->pipe_width - env->bird_x;
        if (dx < 0.0f) {
            continue;
        }

        if (*first == NULL || pipe->x < (*first)->x) {
            *second = *first;
            *first = pipe;
        } else if (*second == NULL || pipe->x < (*second)->x) {
            *second = pipe;
        }
    }
}

static inline void flappycnn_rect(obs_t* obs, float x0, float y0,
        float x1, float y1, float value) {
    int left = (int)fmaxf(0, fminf(OBS_WIDTH, floorf(x0)));
    int right = (int)fmaxf(0, fminf(OBS_WIDTH, ceilf(x1)));
    int top = (int)fmaxf(0, fminf(OBS_HEIGHT, floorf(y0)));
    int bottom = (int)fmaxf(0, fminf(OBS_HEIGHT, ceilf(y1)));
    for (int y = top; y < bottom; y++) {
        for (int x = left; x < right; x++) obs[y * OBS_WIDTH + x] = value;
    }
}

static inline void flappy_compute_observations(Flappy* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    float sx = (float)OBS_WIDTH / env->width;
    float sy = (float)OBS_HEIGHT / env->height;
    // World and image y both point down. Draw visible pipe geometry, then bird.
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        float x0 = pipe->x * sx, x1 = (pipe->x + env->pipe_width) * sx;
        if (x1 <= 0 || x0 >= OBS_WIDTH) continue;
        flappycnn_rect(obs, x0, 0, x1, (pipe->gap_y - env->pipe_gap * 0.5f) * sy, 0.5f);
        flappycnn_rect(obs, x0, (pipe->gap_y + env->pipe_gap * 0.5f) * sy, x1, OBS_HEIGHT, 0.5f);
    }
    flappycnn_rect(obs, (env->bird_x - env->bird_radius) * sx,
        (env->bird_y - env->bird_radius) * sy, (env->bird_x + env->bird_radius) * sx,
        (env->bird_y + env->bird_radius) * sy, 1.0f);
    if (env->representation == 1) {
        for (int y = 0; y < OBS_HEIGHT; y++) for (int x = 0; x < OBS_WIDTH / 2; x++) {
            float tmp = obs[y * OBS_WIDTH + x];
            obs[y * OBS_WIDTH + x] = obs[y * OBS_WIDTH + OBS_WIDTH - 1 - x];
            obs[y * OBS_WIDTH + OBS_WIDTH - 1 - x] = tmp;
        }
    } else if (env->representation == 2) {
        for (int y = 0; y < OBS_HEIGHT / 2; y++) for (int x = 0; x < OBS_WIDTH; x++) {
            int other = (OBS_HEIGHT - 1 - y) * OBS_WIDTH + x;
            float tmp = obs[y * OBS_WIDTH + x];
            obs[y * OBS_WIDTH + x] = obs[other]; obs[other] = tmp;
        }
    } else if (env->representation == 3) {
        for (int i = 0; i < OBS_SIZE; i++) obs[i] = 1.0f - obs[i];
    }
}

static inline void flappy_add_log(Flappy* env) {
    env->log.perf += flappy_clampf((float)env->score / 20.0f, 0.0f, 1.0f);
    env->log.score += env->score;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.n += 1.0f;
}

void init(Flappy* env) {
    env->num_agents = 1;
    env->client = NULL;
    memset(&env->log, 0, sizeof(Log));
}

void puf_reset(Flappy* env) {
    env->bird_y = env->height * 0.5f;
    env->bird_vy = 0.0f;
    env->tick = 0;
    env->score = 0;
    env->episode_return = 0.0f;
    float start_x = env->first_pipe_x;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        flappy_init_pipe(env, i, start_x + i * env->pipe_spacing);
    }

    flappy_compute_observations(env);
}

// Hold Left Shift + space/up to flap.
static void flappy_human_controls(Flappy *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_UP)) {
        env->agents[0].actions[0] = FLAPPY_FLAP;
    } else {
        env->agents[0].actions[0] = FLAPPY_NOOP;
    }
}

void puf_step(Flappy* env) {
    flappy_human_controls(env);
    env->tick += 1;
    env->agents[0].rewards[0] = env->alive_reward;
    env->agents[0].terminals[0] = 0.0f;

    int action = (int)env->agents[0].actions[0];
    if (action == FLAPPY_FLAP) {
        env->bird_vy = env->flap_velocity;
    }

    env->bird_vy += env->gravity;
    env->bird_y += env->bird_vy;

    float max_x = 0.0f;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        if (env->pipes[i].x > max_x) {
            max_x = env->pipes[i].x;
        }
    }

    bool done = false;
    if (env->bird_y - env->bird_radius < 0.0f || env->bird_y + env->bird_radius > env->height) {
        done = true;
    }

    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        pipe->x -= env->pipe_speed;

        if (!pipe->passed && pipe->x + env->pipe_width < env->bird_x) {
            pipe->passed = 1;
            env->score += 1;
            env->agents[0].rewards[0] += env->pass_reward;
        }

        bool overlap_x = env->bird_x + env->bird_radius > pipe->x &&
            env->bird_x - env->bird_radius < pipe->x + env->pipe_width;
        bool outside_gap = env->bird_y - env->bird_radius < pipe->gap_y - env->pipe_gap * 0.5f ||
            env->bird_y + env->bird_radius > pipe->gap_y + env->pipe_gap * 0.5f;
        if (overlap_x && outside_gap) {
            done = true;
        }

        if (pipe->x + env->pipe_width < 0.0f) {
            flappy_init_pipe(env, i, max_x + env->pipe_spacing);
            max_x = env->pipes[i].x;
        }
    }

    Pipe* next;
    Pipe* ignored;
    flappy_next_pipes(env, &next, &ignored);
    if (next == NULL) {
        next = &env->pipes[0];
    }
    float center_error = fabsf(env->bird_y - next->gap_y) / (env->height * 0.5f);
    env->agents[0].rewards[0] += env->center_reward * (1.0f - flappy_clampf(center_error, 0.0f, 1.0f));

    if (env->tick >= env->max_steps) {
        done = true;
    }

    if (done) {
        env->agents[0].rewards[0] += env->crash_reward;
        env->agents[0].terminals[0] = 1.0f;
        env->episode_return += env->agents[0].rewards[0];
        flappy_add_log(env);
        puf_reset(env);
        return;
    }

    env->episode_return += env->agents[0].rewards[0];
    flappy_compute_observations(env);
}

void puf_render(Flappy* env) {
    if (env->client == NULL) {
        env->client = (Client*)calloc(1, sizeof(Client));
        InitWindow(env->width, env->height, "PufferLib Flappy");
        SetTargetFPS(60);
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    flappy_human_controls(env);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    DrawRectangle(0, env->height - 24, env->width, 24, (Color){40, 120, 84, 255});
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        int gap_top = (int)(pipe->gap_y - env->pipe_gap * 0.5f);
        int gap_bottom = (int)(pipe->gap_y + env->pipe_gap * 0.5f);
        DrawRectangle((int)pipe->x, 0, (int)env->pipe_width, gap_top, (Color){0, 187, 187, 255});
        DrawRectangle((int)pipe->x, gap_bottom, (int)env->pipe_width,
            env->height - gap_bottom, (Color){0, 187, 187, 255});
    }

    float sprite_size = env->bird_radius * 3.0f;
    float rotation = flappy_clampf(env->bird_vy * 3.0f, -35.0f, 35.0f);
    DrawTexturePro(
        env->client->puffer,
        (Rectangle){0, 0, 128, 128},
        (Rectangle){env->bird_x, env->bird_y, sprite_size, sprite_size},
        (Vector2){sprite_size * 0.5f, sprite_size * 0.5f},
        rotation,
        WHITE
    );
    DrawText(TextFormat("Score: %i", env->score), 12, 12, 24, (Color){241, 241, 241, 255});
    DrawText("[Shift] space/up flap", 12, 40, 16, (Color){241, 241, 241, 255});
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Flappy* env) {
    if (env->client != NULL) {
        if (IsWindowReady()) {
            UnloadTexture(env->client->puffer);
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->max_steps = dict_get(kwargs, "max_steps");
    env->gravity = dict_get(kwargs, "gravity");
    env->flap_velocity = dict_get(kwargs, "flap_velocity");
    env->pipe_speed = dict_get(kwargs, "pipe_speed");
    env->pipe_gap = dict_get(kwargs, "pipe_gap");
    env->pipe_width = dict_get(kwargs, "pipe_width");
    env->pipe_spacing = dict_get(kwargs, "pipe_spacing");
    env->first_pipe_x = dict_get(kwargs, "first_pipe_x");
    env->bird_x = dict_get(kwargs, "bird_x");
    env->bird_radius = dict_get(kwargs, "bird_radius");
    env->alive_reward = dict_get(kwargs, "alive_reward");
    env->pass_reward = dict_get(kwargs, "pass_reward");
    env->crash_reward = dict_get(kwargs, "crash_reward");
    env->center_reward = dict_get(kwargs, "center_reward");
    if (env->width <= 0 || env->height <= 0 || env->max_steps <= 0
            || !(env->pipe_width > 0 && env->pipe_gap > 0 && env->bird_radius > 0)) {
        fprintf(stderr, "FlappyCNN requires positive dimensions, max_steps and object sizes\n");
        exit(1);
    }
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
    env->representation = cnn_appearance_init(kwargs, env->rng, FLAPPYCNN_NUM_REPRESENTATIONS);
}
