#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"
#include "sprites.h"

#define ACT_SIZES {3, LD_AREA}
#define NUM_ATNS 2
#define PUF_STEPS_PER_SEC 16
#define LD_WIDTH 20
#define LD_HEIGHT 15
#define LD_AREA (LD_WIDTH * LD_HEIGHT)
#define LD_CELL 32 // Each cell is 32x32 pixels.
#define OBS_WIDTH (LD_WIDTH * LD_CELL)
#define OBS_HEIGHT (LD_HEIGHT * LD_CELL)
#define OBS_CHANNELS 3
#define LD_PIXEL_OBS_SIZE (OBS_WIDTH * OBS_HEIGHT * OBS_CHANNELS)
#define STATE_OBS_SIZE (1 + 2 * LD_LANES)
#define LD_ROI_OBS_SIZE (9 * LD_CELL * LD_CELL * 3 + 3)
#define LD_REPRESENTATIONS 3
#define LD_LANES 10
#define LD_LANE_X(l) (5 + (l)) // Lane columns 5-14, adjacent.
#define LD_PLAYER_ROW (LD_HEIGHT - 2) // Row 13; row 14 is the road shoulder.
#define LD_RULES "sprite-lanedodge-v2"
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
    Texture2D roi_texture; // 96x96 gaze window, observation mode 2.
    Color* roi_pixels;
} Client;

struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    unsigned int object_rng;
    unsigned char grid[LD_AREA];
    unsigned char pixel_grid[LD_AREA]; // Last drawn tile IDs, including the player.
    int lane;
    int player_idx; // Row-major grid cell of the player vehicle.
    int gaze_idx; // ROI window center; row-major cell, like Forage's gaze.
    int tick;
    int observation_mode;
    int representation;
    int coins;
    int coins_spawned;
    int max_steps;
    int hazard_rate;
    int coin_rate;
    float episode_return;
    float step_cost;
    float reward_coin;
    float reward_death;
    Client* client;
};

float ld_tiles[LD_REPRESENTATIONS][LD_TILE_TYPES][LD_CELL * LD_CELL * 3];

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
    if (mode == 1) {
        return LD_PIXEL_OBS_SIZE;
    }
    if (mode == 2) {
        return LD_ROI_OBS_SIZE;
    }
    return STATE_OBS_SIZE;
}

int ld_cell_bg(int x) {
    if (x < 5 || x > 14) {
        return LD_GRASS;
    }
    if (x == 5) {
        return LD_ROAD;
    }
    // Lane columns carry the dashed edge stripe on the tile's left edge.
    return LD_DIVIDER;
}

Color ld_color(unsigned char glyph, int theme) {
    Color color;
    switch (glyph) {
        case 'a': color = (Color){58, 60, 66, 255}; break;    // Asphalt
        case 'w': color = (Color){235, 235, 240, 255}; break; // Road marking
        case 'R': color = (Color){203, 53, 65, 255}; break;   // Hazard body
        case 'D': color = (Color){122, 32, 44, 255}; break;  // Hazard dark
        case 'Y': color = (Color){242, 207, 78, 255}; break; // Coin body
        case 'O': color = (Color){168, 128, 40, 255}; break; // Coin dark
        case 'b': color = (Color){77, 137, 192, 255}; break; // Player body
        case 'B': color = (Color){45, 80, 140, 255}; break;  // Player dark
        case 'k': color = (Color){24, 28, 31, 255}; break;   // Wheels
        case 's': color = (Color){240, 240, 245, 255}; break; // Shine
        default: color = (Color){96, 138, 74, 255}; break;   // Grass
    }
    if (theme) {
        unsigned char red = color.r;
        color.r = color.b;
        color.b = red;
    }
    return color;
}

void ld_make_tiles(int rep) {
    // puf_init runs serially, before environment workers start.
    static int ready[LD_REPRESENTATIONS] = {0}; // Build each skin once per process.
    if (ready[rep]) {
        return;
    }
    for (int kind = 0; kind < LD_TILE_TYPES; kind++) {
        for (int pixel = 0; pixel < LD_CELL * LD_CELL; pixel++) {
            int x = pixel % LD_CELL;
            int y = pixel / LD_CELL;
            // Enlarge 16x16 art by 2; representation 2 mirrors it horizontally.
            int sprite_x = rep == 2 ? 15 - x / 2 : x / 2;
            unsigned char glyph = LD_ART[kind][y / 2][sprite_x];
            Color color = ld_color(glyph, rep == 1);
            // Deterministic per-art-pixel shading keeps ground fills from being flat.
            if (glyph == 'a' || glyph == '.') {
                int shade = 4 * ((sprite_x * 7 + (y / 2) * 3) % 3 - 1);
                color.r = (unsigned char)clampf(color.r + shade, 0, 255);
                color.g = (unsigned char)clampf(color.g + shade, 0, 255);
                color.b = (unsigned char)clampf(color.b + shade, 0, 255);
            }
            ld_tiles[rep][kind][pixel * 3] = color.r / 255.f;
            ld_tiles[rep][kind][pixel * 3 + 1] = color.g / 255.f;
            ld_tiles[rep][kind][pixel * 3 + 2] = color.b / 255.f;
        }
    }
    // Entities only travel on road cells, so their transparent art composites
    // over the shaded road instead of the grass default.
    for (int kind = LD_HAZARD; kind < LD_TILE_TYPES; kind++) {
        for (int pixel = 0; pixel < LD_CELL * LD_CELL; pixel++) {
            int x = pixel % LD_CELL;
            int y = pixel / LD_CELL;
            int sprite_x = rep == 2 ? 15 - x / 2 : x / 2;
            if (LD_ART[kind][y / 2][sprite_x] != '.') {
                continue;
            }
            for (int channel = 0; channel < 3; channel++) {
                ld_tiles[rep][kind][pixel * 3 + channel] =
                    ld_tiles[rep][LD_ROAD][pixel * 3 + channel];
            }
        }
    }
    ready[rep] = 1;
}

// Copy one cached tile into the observation.
void write_pixel_cell(Env* env, int cell) {
    int kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
    obs_t* obs = env->agents[0].observations;
    int left = (cell % LD_WIDTH) * LD_CELL;
    int top = (cell / LD_WIDTH) * LD_CELL;
    float* tile = ld_tiles[env->representation][kind];
    obs += (top * OBS_WIDTH + left) * 3;
    for (int y = 0; y < LD_CELL; y++) {
        memcpy(obs + y * OBS_WIDTH * 3, tile + y * LD_CELL * 3,
            LD_CELL * 3 * sizeof(obs_t));
    }
    env->pixel_grid[cell] = kind;
}
// Write the 3x3 cell window around the gaze cell; out-of-range cells are grass.
void write_roi(Env* env) {
    obs_t* obs = env->agents[0].observations;
    int gx = env->gaze_idx % LD_WIDTH;
    int gy = env->gaze_idx / LD_WIDTH;
    int slot = 0;
    for (int sy = -1; sy <= 1; sy++) {
        for (int sx = -1; sx <= 1; sx++) {
            int x = gx + sx;
            int y = gy + sy;
            int kind = LD_GRASS;
            if (x >= 0 && x < LD_WIDTH && y >= 0 && y < LD_HEIGHT) {
                int cell = y * LD_WIDTH + x;
                kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
            }
            memcpy(obs + slot * LD_CELL * LD_CELL * 3,
                ld_tiles[env->representation][kind],
                LD_CELL * LD_CELL * 3 * sizeof(obs_t));
            slot++;
        }
    }
    obs[9 * LD_CELL * LD_CELL * 3] = env->lane / (float)(LD_LANES - 1);
    obs[9 * LD_CELL * LD_CELL * 3 + 1] = (float)gx / (float)(LD_WIDTH - 1);
    obs[9 * LD_CELL * LD_CELL * 3 + 2] = (float)gy / (float)(LD_HEIGHT - 1);
}

void compute_observations(Env* env) {
    obs_t* obs = env->agents[0].observations;
    obs[0] = env->lane / (float)(LD_LANES - 1);
    for (int l = 0; l < LD_LANES; l++) {
        int hy = -1, cy = -1;
        for (int y = LD_PLAYER_ROW - 1; y >= 0 && (hy < 0 || cy < 0); y--) {
            int kind = env->grid[y * LD_WIDTH + LD_LANE_X(l)];
            if (kind == LD_HAZARD && hy < 0) {
                hy = y;
            } else if (kind == LD_COIN && cy < 0) {
                cy = y;
            }
        }
        obs[1 + l] = hy >= 0 ? (LD_HEIGHT - 1 - hy) / (float)(LD_HEIGHT - 1) : 0.f;
        obs[1 + LD_LANES + l] = cy >= 0 ? (LD_HEIGHT - 1 - cy) / (float)(LD_HEIGHT - 1) : 0.f;
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->rng += (unsigned int)dict_get(kwargs, "seed");
    env->observation_mode = dict_get(kwargs, "observation_mode");
    env->representation = dict_get(kwargs, "representation");
    env->max_steps = dict_get(kwargs, "max_steps");
    env->hazard_rate = dict_get(kwargs, "hazard_rate");
    env->coin_rate = dict_get(kwargs, "coin_rate");
    env->step_cost = dict_get(kwargs, "step_cost");
    env->reward_coin = dict_get(kwargs, "reward_coin");
    env->reward_death = dict_get(kwargs, "reward_death");
    env->num_agents = 1;
    env->agents[0].policy = 0;
    if (env->observation_mode) {
        ld_make_tiles(env->representation);
        if (env->observation_mode == 1) {
            // Force every tile to be drawn on the first reset.
            memset(env->pixel_grid, 255, sizeof(env->pixel_grid));
        }
    }
}

void puf_reset(Env* env) {
    // One world draw per episode; spawn draws never advance the next world RNG.
    env->rng = env->rng * 1664525u + 1013904223u;
    env->object_rng = env->rng ^ 0x9e3779b9u;
    env->object_rng ^= env->object_rng >> 16;
    env->object_rng *= 0x85ebca6bu;
    for (int y = 0; y < LD_HEIGHT; y++) {
        for (int x = 0; x < LD_WIDTH; x++) {
            env->grid[y * LD_WIDTH + x] = (unsigned char)ld_cell_bg(x);
        }
    }
    env->lane = 1;
    env->player_idx = LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(env->lane);
    env->gaze_idx = env->player_idx;
    env->tick = env->coins = env->coins_spawned = 0;
    env->episode_return = 0;
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    if (env->observation_mode == 1) {
        for (int cell = 0; cell < LD_AREA; cell++) {
            int kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
            if (env->pixel_grid[cell] != kind) {
                write_pixel_cell(env, cell);
            }
        }
    } else if (env->observation_mode == 2) {
        write_roi(env);
    } else {
        compute_observations(env);
    }
}

void puf_step(Env* env) {
    if (env->client && IsKeyDown(KEY_LEFT_SHIFT)) {
        env->agents[0].actions[0] = 0;
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) {
            env->agents[0].actions[0] = 1;
        } else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) {
            env->agents[0].actions[0] = 2;
        }
    }
    int gaze = (int)env->agents[0].actions[1];
    if (gaze >= 0 && gaze < LD_AREA) {
        env->gaze_idx = gaze;
    }
    int action = env->agents[0].actions[0];
    env->agents[0].rewards[0] = env->step_cost;
    env->agents[0].terminals[0] = 0;
    env->tick++;
    int prev_player = env->player_idx;
    env->lane += (action == 2) - (action == 1);
    if (env->lane < 0) {
        env->lane = 0;
    } else if (env->lane > LD_LANES - 1) {
        env->lane = LD_LANES - 1;
    }
    env->player_idx = LD_PLAYER_ROW * LD_WIDTH + LD_LANE_X(env->lane);
    int player_x = env->player_idx % LD_WIDTH;
    int cell = LD_PLAYER_ROW * LD_WIDTH + player_x;
    int kind = env->grid[cell];
    int died = 0;
    if (kind == LD_HAZARD) {
        env->grid[cell] = (unsigned char)ld_cell_bg(player_x);
        died = 1;
    } else if (kind == LD_COIN) {
        env->grid[cell] = (unsigned char)ld_cell_bg(player_x);
        env->coins++;
        env->agents[0].rewards[0] += env->reward_coin;
    }
    int dirty[LD_AREA * 2]; // Every cell marked twice at most, plus spawns.
    int nd = 0;
    // Entities advance one row toward the player; row 13 exits, row 12 collides.
    for (int y = LD_PLAYER_ROW; y >= 0; y--) {
        for (int l = 0; l < LD_LANES; l++) {
            int x = LD_LANE_X(l);
            int idx = y * LD_WIDTH + x;
            int e = env->grid[idx];
            if (e != LD_HAZARD && e != LD_COIN) {
                continue;
            }
            int bg = ld_cell_bg(x);
            if (y == LD_PLAYER_ROW) {
                env->grid[idx] = (unsigned char)bg;
                dirty[nd++] = idx;
                continue;
            }
            int target = idx + LD_WIDTH;
            if (y + 1 == LD_PLAYER_ROW && x == player_x) {
                if (e == LD_COIN) {
                    env->coins++;
                    env->agents[0].rewards[0] += env->reward_coin;
                } else {
                    died = 1;
                }
                // Consume the arriving entity on both rows so it cannot repeat.
                env->grid[idx] = (unsigned char)bg;
                env->grid[target] = (unsigned char)bg;
                dirty[nd++] = idx;
                dirty[nd++] = target;
                continue;
            }
            env->grid[target] = e;
            env->grid[idx] = (unsigned char)bg;
            dirty[nd++] = idx;
            dirty[nd++] = target;
        }
    }
    env->object_rng = env->object_rng * 1664525u + 1013904223u;
    if ((env->object_rng >> 16) % 100 < env->hazard_rate) {
        env->object_rng = env->object_rng * 1664525u + 1013904223u;
        int x = LD_LANE_X((env->object_rng >> 16) % LD_LANES);
        if (env->grid[x] == (unsigned char)ld_cell_bg(x)) {
            env->grid[x] = LD_HAZARD;
            dirty[nd++] = x;
        }
    }
    env->object_rng = env->object_rng * 1664525u + 1013904223u;
    if ((env->object_rng >> 16) % 100 < env->coin_rate) {
        env->object_rng = env->object_rng * 1664525u + 1013904223u;
        int x = LD_LANE_X((env->object_rng >> 16) % LD_LANES);
        if (env->grid[x] == (unsigned char)ld_cell_bg(x)) {
            env->grid[x] = LD_COIN;
            env->coins_spawned++;
            dirty[nd++] = x;
        }
    }
    if (died) {
        env->agents[0].rewards[0] += env->reward_death;
    }
    env->episode_return += env->agents[0].rewards[0];
    if (died || env->tick >= env->max_steps) {
        float reward = env->agents[0].rewards[0];
        float survival = died ? env->tick / (float)env->max_steps : 1.0f;
        float coin_frac = env->coins_spawned
            ? env->coins / (float)env->coins_spawned : 0.0f;
        env->log.perf += clampf(0.5f * survival + 0.5f * coin_frac, 0.0f, 1.0f);
        env->log.score += env->episode_return;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += env->tick;
        env->log.deaths += died;
        env->log.successes += !died;
        env->log.n++;
        puf_reset(env);
        env->agents[0].rewards[0] = reward;
        env->agents[0].terminals[0] = 1;
    } else if (env->observation_mode == 0) {
        compute_observations(env);
    } else if (env->observation_mode == 1) {
        if (env->player_idx != prev_player) {
            write_pixel_cell(env, prev_player);
            write_pixel_cell(env, env->player_idx);
        }
        for (int i = 0; i < nd; i++) {
            write_pixel_cell(env, dirty[i]);
        }
    } else {
        write_roi(env);
    }
}

void puf_render(Env* env) {
    if (!env->client) {
        InitWindow(OBS_WIDTH, OBS_HEIGHT, "PufferLib Sprite Lane Dodge");
        SetTargetFPS(PUF_STEPS_PER_SEC);
        env->client = (Client*)calloc(1, sizeof(Client));
        env->client->pixels = (Color*)calloc(OBS_WIDTH * OBS_HEIGHT, sizeof(Color));
        Image image = {env->client->pixels, OBS_WIDTH, OBS_HEIGHT, 1,
            PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        env->client->texture = LoadTextureFromImage(image);
        env->client->roi_pixels = (Color*)calloc(96 * 96, sizeof(Color));
        Image roi_image = {env->client->roi_pixels, 96, 96, 1,
            PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        env->client->roi_texture = LoadTextureFromImage(roi_image);
        ld_make_tiles(env->representation);
    }
    if (env->observation_mode == 2) {
        // Show the agent's view: the 3x3 gaze window, scaled to full height.
        int gx = env->gaze_idx % LD_WIDTH;
        int gy = env->gaze_idx / LD_WIDTH;
        for (int sy = 0; sy < 3; sy++) {
            for (int sx = 0; sx < 3; sx++) {
                int x = gx + sx - 1;
                int y = gy + sy - 1;
                int kind = LD_GRASS;
                if (x >= 0 && x < LD_WIDTH && y >= 0 && y < LD_HEIGHT) {
                    int cell = y * LD_WIDTH + x;
                    kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
                }
                const float* tile = ld_tiles[env->representation][kind];
                for (int y2 = 0; y2 < LD_CELL; y2++) {
                    for (int x2 = 0; x2 < LD_CELL; x2++) {
                        int source = (y2 * LD_CELL + x2) * 3;
                        int target = (sy * LD_CELL + y2) * 96 + sx * LD_CELL + x2;
                        env->client->roi_pixels[target] = (Color){
                            (unsigned char)(tile[source] * 255 + .5f),
                            (unsigned char)(tile[source + 1] * 255 + .5f),
                            (unsigned char)(tile[source + 2] * 255 + .5f), 255};
                    }
                }
            }
        }
        UpdateTexture(env->client->roi_texture, env->client->roi_pixels);
        BeginDrawing();
        ClearBackground((Color){0, 0, 0, 255});
        DrawTextureEx(env->client->roi_texture, (Vector2){80, 0}, 0, 5.f, WHITE);
        EndDrawing();
        puf_web_vsync();
        return;
    }
    for (int cell = 0; cell < LD_AREA; cell++) {
        int kind = cell == env->player_idx ? LD_PLAYER : env->grid[cell];
        const float* tile = ld_tiles[env->representation][kind];
        for (int y = 0; y < LD_CELL; y++) {
            for (int x = 0; x < LD_CELL; x++) {
                int source = (y * LD_CELL + x) * 3;
                int target = (cell / LD_WIDTH * LD_CELL + y) * OBS_WIDTH
                    + cell % LD_WIDTH * LD_CELL + x;
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
    UnloadTexture(env->client->roi_texture);
    free(env->client->roi_pixels);
    UnloadTexture(env->client->texture);
    free(env->client->pixels);
    free(env->client);
    env->client = NULL;
    CloseWindow();
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
