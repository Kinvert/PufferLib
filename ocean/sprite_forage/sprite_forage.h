#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"
#include "sprites.h"
#include "imported_sprites.h"

#ifdef FORAGE_ATTENTION
#ifndef FORAGE_GAZE_MODE
#define FORAGE_GAZE_MODE 0
#endif
#ifndef FORAGE_RELATIVE_GAZE
#define FORAGE_RELATIVE_GAZE 0
#endif
#ifndef FORAGE_STATE_ONEHOT
#define FORAGE_STATE_ONEHOT 0
#endif
#ifndef FORAGE_PLAYER_VIEW
#define FORAGE_PLAYER_VIEW 1
#endif
#if FORAGE_GAZE_MODE == 1
#define ACT_SIZES {5, 5}
#define NUM_ATNS 2
#elif FORAGE_GAZE_MODE == 2
#define ACT_SIZES {5, FORAGE_WIDTH, FORAGE_HEIGHT}
#define NUM_ATNS 3
#else
#define ACT_SIZES {5, FORAGE_AREA}
#define NUM_ATNS 2
#endif
#define FORAGE_VIEW_TILES (9 + 8 * FORAGE_PLAYER_VIEW)
#define FORAGE_STATE_VALUES (FORAGE_VIEW_TILES * (FORAGE_STATE_ONEHOT ? FORAGE_TILE_TYPES : 1))
#define FORAGE_TILE_PIXELS (FORAGE_CELL * FORAGE_CELL * OBS_CHANNELS)
#define FORAGE_VIEW_PIXELS (FORAGE_VIEW_TILES * FORAGE_TILE_PIXELS)
#define FORAGE_VIEW_OBS (FORAGE_VIEW_PIXELS + 4)
#else
#define ACT_SIZES {5}
#define NUM_ATNS 1
#endif
#define PUF_STEPS_PER_SEC 8
#define FORAGE_WIDTH 20
#define FORAGE_HEIGHT 15
#define FORAGE_AREA (FORAGE_WIDTH * FORAGE_HEIGHT)
#define FORAGE_CELL 32 // Each cell is 32x32 pixels.
#define OBS_WIDTH (FORAGE_WIDTH * FORAGE_CELL)
#define OBS_HEIGHT (FORAGE_HEIGHT * FORAGE_CELL)
#define OBS_CHANNELS 3
#define FORAGE_PIXEL_OBS_SIZE (OBS_WIDTH * OBS_HEIGHT * OBS_CHANNELS)
#define FORAGE_REPRESENTATIONS 5
// Temporary research version: read by research/sprite_forage_report.py and
// checked by research/run_sprite_forage_*.sh. Remove before the clean env PR.
#define FORAGE_RULES "sprite-forage-state-v7"
#define PUF_NATURE_ENCODER

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float deaths;
    float successes;
    float n;
};

typedef struct {
    Texture2D texture;
    Color* pixels;
} Client;

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    unsigned char grid[FORAGE_AREA];
    unsigned char pixel_grid[FORAGE_AREA]; // Last drawn tile IDs, including the player.
    int player_idx;
    int tick;
    int observation_mode;
#ifdef FORAGE_ATTENTION
    int partial_observation;
    int gaze_idx;
#endif
    int state_layout;
    int representation;
    int copies;
    int trees;
    int max_steps;
    int apples;
    int negatives;
    float episode_return;
    float reward_good;
    float reward_other;
    float reward_grape;
    float reward_bad;
    float reward_death;
    float step_cost;
    Client* client;
};

float forage_tiles[FORAGE_REPRESENTATIONS][FORAGE_TILE_TYPES][FORAGE_CELL * FORAGE_CELL * 3];

float clampf(float value, float low, float high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

int puf_observation_size(Dict* kwargs) {
    int mode = dict_get(kwargs, "observation_mode");
#ifdef FORAGE_ATTENTION
    if (dict_get(kwargs, "partial_observation")) {
        return (mode ? FORAGE_VIEW_PIXELS : FORAGE_STATE_VALUES) + 4;
    }
#endif
    if (mode) {
        return FORAGE_PIXEL_OBS_SIZE;
    }
    int copies = dict_get(kwargs, "copies_per_type");
    int trees = dict_get(kwargs, "trees");
    int layout = dict_get(kwargs, "state_layout");
    return layout == 1 ? FORAGE_AREA : 2 + 3 * (10 * copies + trees) + 25 * (layout == 2);
}

int forage_random_empty_idx(Env* env) {
    int free_cells[FORAGE_AREA];
    int count = 0;
    for (int i = 0; i < FORAGE_AREA; i++) {
        if (!env->grid[i] && i != env->player_idx) {
            free_cells[count++] = i;
        }
    }
    int threshold = (0u - count) % count;
    while (true) {
        env->rng = env->rng * 1664525u + 1013904223u;
        if (env->rng >= threshold) {
            break;
        }
    }
    return free_cells[env->rng % count];
}

Color forage_color(unsigned char glyph, int theme) {
    Color color;
    switch (glyph) {
        case 'k': color = (Color){16, 20, 31, 255}; break;    // Near-black
        case 'r': color = (Color){203, 53, 65, 255}; break;   // Red
        case 'o': color = (Color){246, 148, 77, 255}; break;  // Orange
        case 'y': color = (Color){242, 207, 78, 255}; break;  // Yellow
        case 'g': color = (Color){91, 165, 79, 255}; break;   // Green
        case 'G': color = (Color){43, 103, 65, 255}; break;   // Dark green
        case 'p': color = (Color){155, 94, 178, 255}; break;  // Purple
        case 'b': color = (Color){77, 137, 192, 255}; break;  // Blue
        case 'w': color = (Color){240, 226, 191, 255}; break; // Cream
        case 's': color = (Color){125, 83, 58, 255}; break;   // Brown
        case 'n': color = (Color){226, 171, 128, 255}; break; // Peach
        default: color = (Color){30, 49, 38, 255}; break;     // Ground
    }
    if (theme) {
        unsigned char red = color.r;
        color.r = color.b;
        color.b = red;
    }
    return color;
}

void forage_make_tiles(int rep) {
    // puf_init runs serially, before environment workers start.
    static int ready[FORAGE_REPRESENTATIONS] = {0}; // Build each skin once per process.
    if (ready[rep]) {
        return;
    }
    for (int kind = 0; kind < FORAGE_TILE_TYPES; kind++) {
        for (int pixel = 0; pixel < FORAGE_CELL * FORAGE_CELL; pixel++) {
            int x = pixel % FORAGE_CELL;
            int y = pixel / FORAGE_CELL;
            // Enlarge 16x16 art by 2; representation 2 mirrors it horizontally.
            int sprite_x = rep == 2 ? 15 - x / 2 : x / 2;
            unsigned char glyph = kind ? FORAGE_ART[kind][y / 2][sprite_x] : '.';
            Color color = forage_color(glyph, rep == 1);
            if (rep == 4 && kind != FORAGE_EMPTY) {
                const unsigned char* source = FORAGE_IMPORTED[kind] + pixel * 4;
                glyph = source[3] ? '!' : '.';
                color = (Color){source[0], source[1], source[2], source[3]};
            }
            if (glyph == '.') {
                color = forage_color('.', rep == 1);
                int ground_texture = rep == 3
                    ? ((x / 2 + 3 * (y / 2)) % 7 == 0 || (x + y) % 11 == 0)
                    : ((x == 4 || x == 24) && y % 16 >= 9 && y % 16 <= 12);
                if (ground_texture) {
                    color = forage_color('G', rep == 1);
                }
            }
            forage_tiles[rep][kind][pixel * 3] = color.r / 255.f;
            forage_tiles[rep][kind][pixel * 3 + 1] = color.g / 255.f;
            forage_tiles[rep][kind][pixel * 3 + 2] = color.b / 255.f;
        }
    }
    ready[rep] = 1;
}

// Copy one cached tile into the observation.
void write_pixel_cell(Env* env, int cell) {
    int kind = cell == env->player_idx ? FORAGE_PLAYER : env->grid[cell];
    obs_t* obs = env->agents[0].observations;
    int left = (cell % FORAGE_WIDTH) * FORAGE_CELL;
    int top = (cell / FORAGE_WIDTH) * FORAGE_CELL;
    float* tile = forage_tiles[env->representation][kind];
    obs += (top * OBS_WIDTH + left) * 3;
    for (int y = 0; y < FORAGE_CELL; y++) {
        memcpy(obs + y * OBS_WIDTH * 3, tile + y * FORAGE_CELL * 3,
            FORAGE_CELL * 3 * sizeof(obs_t));
    }
    env->pixel_grid[cell] = kind;
}

#ifdef FORAGE_ATTENTION
#include "attention.h"
#endif

void compute_observations(Env* env) {
#ifdef FORAGE_ATTENTION
    if (env->partial_observation) {
        forage_attention_observations(env);
        return;
    }
#endif
    if (env->observation_mode) {
        // Explicit full rebuild; normal steps and resets update changed cells.
        for (int cell = 0; cell < FORAGE_AREA; cell++) {
            write_pixel_cell(env, cell);
        }
        return;
    }
    obs_t* obs = env->agents[0].observations;
    // Distinct semantic codes, not reward magnitudes. Walls differ from empty ground.
    const float tile_codes[] = {0, .75f, .25f, .5f, .25f, .25f, .25f, .25f,
        -.25f, -.25f, -.5f, -1, 1};
    if (env->state_layout == 1) {
        for (int cell = 0; cell < FORAGE_AREA; cell++) {
            obs[cell] = tile_codes[env->grid[cell]];
        }
        obs[env->player_idx] = 1;
        return;
    }
    memset(obs, 0, (2 + 3 * (10 * env->copies + env->trees)) * sizeof(obs_t));
    int px = env->player_idx % FORAGE_WIDTH;
    int py = env->player_idx / FORAGE_WIDTH;
    obs[0] = px / (float)FORAGE_WIDTH;
    obs[1] = py / (float)FORAGE_HEIGHT;
    int count[FORAGE_TILE_TYPES] = {0};
    // Group by type, in board row order.
    for (int y = 1; y < FORAGE_HEIGHT - 1; y++) {
        for (int x = 1; x < FORAGE_WIDTH - 1; x++) {
            int kind = env->grid[y * FORAGE_WIDTH + x];
            if (!kind) {
                continue;
            }
            int first = kind == FORAGE_TREE ? 10 * env->copies : (kind - 1) * env->copies;
            int slot = first + count[kind]++;
            float dx = (x - px) / (float)FORAGE_WIDTH;
            float dy = (y - py) / (float)FORAGE_HEIGHT;
            obs[2 + 3 * slot] = 1;
            obs[3 + 3 * slot] = dx;
            obs[4 + 3 * slot] = dy;
        }
    }
    if (env->state_layout == 2) {
        int offset = 2 + 3 * (10 * env->copies + env->trees);
        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int sx = px + x, sy = py + y;
                float value = sx < 0 || sx >= FORAGE_WIDTH || sy < 0 || sy >= FORAGE_HEIGHT
                    ? -1 : tile_codes[env->grid[sy * FORAGE_WIDTH + sx]];
                obs[offset++] = x == 0 && y == 0 ? 1 : value;
            }
        }
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->rng += (unsigned int)dict_get(kwargs, "seed");
    env->observation_mode = dict_get(kwargs, "observation_mode");
#ifdef FORAGE_ATTENTION // todo seems like too many macros, maybe find a cleaner simple C way to do it
    env->partial_observation = dict_get(kwargs, "partial_observation");
#endif
    env->state_layout = dict_get(kwargs, "state_layout");
    env->representation = dict_get(kwargs, "representation");
    env->copies = dict_get(kwargs, "copies_per_type");
    env->trees = dict_get(kwargs, "trees");
    env->max_steps = dict_get(kwargs, "max_steps");
    env->reward_good = dict_get(kwargs, "reward_good");
    env->reward_other = dict_get(kwargs, "reward_other");
    env->reward_grape = dict_get(kwargs, "reward_grape");
    env->reward_bad = dict_get(kwargs, "reward_bad");
    env->reward_death = dict_get(kwargs, "reward_death");
    env->step_cost = dict_get(kwargs, "step_cost");
    env->num_agents = 1;
    env->agents[0].policy = 0;
    if (env->observation_mode) {
        forage_make_tiles(env->representation);
        // Force every tile to be drawn on the first reset.
        memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
    }
}

void puf_reset(Env* env) {
    memset(env->grid, 0, sizeof(env->grid));
    env->player_idx = -1;
    for (int y = 0; y < FORAGE_HEIGHT; y++) {
        for (int x = 0; x < FORAGE_WIDTH; x++) {
            if (!x || !y || x == FORAGE_WIDTH - 1 || y == FORAGE_HEIGHT - 1) {
                env->grid[y * FORAGE_WIDTH + x] = FORAGE_TREE;
            }
        }
    }
    env->player_idx = forage_random_empty_idx(env);
    for (int n = 0; n < env->trees + 10 * env->copies; n++) {
        int kind = n < env->trees ? FORAGE_TREE : 1 + (n - env->trees) / env->copies;
        // Exactly one grape; every other collectible uses copies_per_type.
        if (kind == FORAGE_GRAPE && (n - env->trees) % env->copies != 0) {
            continue;
        }
        int cell = forage_random_empty_idx(env);
        env->grid[cell] = kind;
    }
    env->tick = 0;
    env->negatives = 0;
    env->apples = 0;
    env->episode_return = 0;
    env->agents[0].rewards[0] = 0;
    env->agents[0].terminals[0] = 0;
#ifdef FORAGE_ATTENTION // todo not a huge fan of this macro, is there a better cleaner way?
    env->gaze_idx = env->player_idx;
    if (env->partial_observation) {
        compute_observations(env);
        return;
    }
#endif
    if (env->observation_mode) {
        for (int cell = 0; cell < FORAGE_AREA; cell++) {
            int kind = cell == env->player_idx ? FORAGE_PLAYER : env->grid[cell];
            if (env->pixel_grid[cell] != kind) {
                write_pixel_cell(env, cell);
            }
        }
    } else {
        compute_observations(env);
    }
}

void puf_step(Env* env) {
    if (env->client && IsKeyDown(KEY_LEFT_SHIFT)) {
        env->agents[0].actions[0] = 0;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) {
            env->agents[0].actions[0] = 1;
        } else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) {
            env->agents[0].actions[0] = 2;
        } else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            env->agents[0].actions[0] = 3;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            env->agents[0].actions[0] = 4;
        }
    }
    int action = env->agents[0].actions[0];
#ifdef FORAGE_ATTENTION // todo maybe too many macros here like this idk, I don't like the initial look of this.
    forage_attention_action(env);
#endif
    env->agents[0].rewards[0] = env->step_cost;
    env->agents[0].terminals[0] = 0;
    env->tick++;
    int delta[] = {0, -FORAGE_WIDTH, FORAGE_WIDTH, -1, 1};
    int prev_idx = env->player_idx;
    int next_idx = prev_idx + delta[action];
    int kind = FORAGE_EMPTY;
    if (env->grid[next_idx] != FORAGE_TREE) {
        env->player_idx = next_idx;
        kind = env->grid[next_idx];
        if (kind) {
            float reward = env->reward_other;
            if (kind == FORAGE_APPLE) {
                reward = env->reward_good;
            } else if (kind == FORAGE_GRAPE) {
                reward = env->reward_grape;
            } else if (kind == FORAGE_TOADSTOOL) {
                reward = env->reward_death;
            } else if (kind >= FORAGE_SPOTTED_APPLE) {
                reward = env->reward_bad;
            }
            env->agents[0].rewards[0] += reward;
            env->grid[next_idx] = FORAGE_EMPTY;
            if (kind == FORAGE_APPLE) {
                env->apples++;
            } else if (kind >= FORAGE_SPOTTED_APPLE) {
                env->negatives++;
            }
        }
    }
    env->episode_return += env->agents[0].rewards[0];
    int died = kind == FORAGE_TOADSTOOL;
    int got_grape = kind == FORAGE_GRAPE;
    if (died || got_grape || env->tick >= env->max_steps) {
        float reward = env->agents[0].rewards[0];
        int collected = env->apples + got_grape - env->negatives;
        env->log.perf += .25f * clampf(collected, 0.0f, 4.0f);
        env->log.score += env->episode_return;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += env->tick;
        env->log.deaths += died;
        env->log.successes += got_grape;
        env->log.n++;
        puf_reset(env);
        env->agents[0].rewards[0] = reward;
        env->agents[0].terminals[0] = 1;
    }
#ifdef FORAGE_ATTENTION
    else if (env->partial_observation) {
        compute_observations(env);
    }
#endif
    else if (env->player_idx != prev_idx) {
        if (env->observation_mode == 0) {
            compute_observations(env);
        } else {
            // Only the old and new player cells change.
            write_pixel_cell(env, prev_idx);
            write_pixel_cell(env, env->player_idx);
        }
    }
}

void puf_render(Env* env) {
    if (!env->client) {
        InitWindow(OBS_WIDTH, OBS_HEIGHT, "PufferLib Sprite Forage");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->pixels = (Color*)calloc(OBS_WIDTH * OBS_HEIGHT, sizeof(Color));
        Image image = {env->client->pixels, OBS_WIDTH, OBS_HEIGHT, 1,
            PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        env->client->texture = LoadTextureFromImage(image);
        forage_make_tiles(env->representation);
    }
    for (int cell = 0; cell < FORAGE_AREA; cell++) {
        int kind = cell == env->player_idx ? FORAGE_PLAYER : env->grid[cell];
        const float* tile = forage_tiles[env->representation][kind];
        for (int y = 0; y < FORAGE_CELL; y++) {
            for (int x = 0; x < FORAGE_CELL; x++) {
                int source = (y * FORAGE_CELL + x) * 3;
                int target = (cell / FORAGE_WIDTH * FORAGE_CELL + y) * OBS_WIDTH
                    + cell % FORAGE_WIDTH * FORAGE_CELL + x;
                env->client->pixels[target] = (Color){
                    (unsigned char)(tile[source] * 255 + .5f),
                    (unsigned char)(tile[source + 1] * 255 + .5f),
                    (unsigned char)(tile[source + 2] * 255 + .5f), 255};
            }
        }
    }
    UpdateTexture(env->client->texture, env->client->pixels);
    BeginDrawing();
    DrawTexture(env->client->texture, 0, 0, WHITE);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Env* env) {
    if (env->client) {
        UnloadTexture(env->client->texture);
        free(env->client->pixels);
        free(env->client);
        env->client = NULL;
        CloseWindow();
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "deaths", log->deaths);
    dict_set(out, "successes", log->successes);
    dict_set(out, "n", log->n);
}
