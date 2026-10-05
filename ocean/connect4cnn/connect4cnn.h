// Copied from connect4/connect4.h at upstream 89414204.
// Rules/opponent match native Connect4; observations/viewer differ.
// Both copies include the October 5, 2026 full-board draw correction.
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
#include "appearance.h"

#define ACT_SIZES {7}
#define CONNECT4CNN_CELL_PIXELS 6
#define CONNECT4CNN_SIDE_PADDING 1
#define OBS_CHANNELS 1
#define OBS_HEIGHT (6 * CONNECT4CNN_CELL_PIXELS)
#define OBS_WIDTH (7 * CONNECT4CNN_CELL_PIXELS + 2 * CONNECT4CNN_SIDE_PADDING)
#define OBS_SIZE (OBS_CHANNELS * OBS_HEIGHT * OBS_WIDTH)
#define NUM_ATNS 1
#define HOLD_FRAMES 30

// Cell pitch, occupied shape size, glyph: square=0, disk=1, X/O=2.
// Every preset fits the same 36x44 observation. ID 0 is the original layout.
static const int CONNECT4CNN_REPRESENTATIONS[][5] = {
    {6, 6, 6, 6, 0}, {6, 6, 4, 4, 0}, {6, 6, 2, 2, 0},
    {6, 6, 6, 6, 1}, {6, 6, 4, 4, 1}, {6, 6, 6, 6, 2},
    {4, 4, 4, 4, 0}, {2, 2, 2, 2, 0}, {1, 1, 1, 1, 0},
    {2, 4, 2, 4, 0},
};
#define CONNECT4CNN_NUM_REPRESENTATIONS 10

#define WIN_CONDITION 4
const int PLAYER_WIN = 1.0;
const int ENV_WIN = -1.0;
const int ROWS = 6;
const int COLUMNS = 7;
const int WIDTH = 672;
const int HEIGHT = 576;
const int PIECE_WIDTH = 96;
const int PIECE_HEIGHT = 96;

const float MAX_VALUE = 31;
const float WIN_VALUE = 30;
const float DRAW_VALUE = 0;

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float invalids;
};

typedef struct Client Client;
struct Env {
    int num_agents;
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    Client* client;

    // Bit string representation from:
    //  https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0
    //  & http://blog.gamesolver.org/solving-connect-four/01-introduction/
    uint64_t player_pieces;
    uint64_t env_pieces;
    uint64_t last_env_bit;
    int pending_reset;

    int tick;
    unsigned int rng;
    int representation;
    uint64_t glyph[2];
};
typedef Env Connect4;

void add_log(Connect4* env) {
    env->log.perf += (float)(env->agents[0].rewards[0] == PLAYER_WIN);
    env->log.score += env->agents[0].rewards[0];
    env->log.episode_return += env->agents[0].rewards[0];
    env->log.episode_length += env->tick;
    env->log.n += 1;
}

void init(Connect4* env) {
    env->tick = 0;
}

// Get the bit at the top of 'column'. Column can be played if bit is 0
uint64_t top_mask(uint64_t column) {
    return (UINT64_C(1) << (ROWS - 1)) << column * (ROWS + 1);
}

// Get a bit mask for where a piece played at 'column' would end up.
uint64_t bottom_mask(uint64_t column) {
    return UINT64_C(1) << column * (ROWS + 1);
}

// A bit mask used to create unique representation of the game state.
uint64_t c_bottom() {
    return UINT64_C(1) << (COLUMNS - 1) * (ROWS + 1);
}

bool invalid_move(int column, uint64_t mask) {
    return (mask & top_mask(column)) != 0;
}

uint64_t play(int column, uint64_t mask,  uint64_t other_pieces) {
    mask |= mask + bottom_mask(column); // Somehow faster than |= bottom_mask(column)
    return other_pieces ^ mask;
}

// All 42 playable cells; the seventh bit of each column is unused.
bool draw(uint64_t mask) {
    return mask == UINT64_C(0xfdfbf7efdfbf);
}

// Determine if 'pieces' contains at least one line of connected pieces.
bool won(uint64_t pieces) {
    // Horizontal 
    uint64_t m = pieces & (pieces >> (ROWS + 1));
    if(m & (m >> (2 * (ROWS + 1)))) {
        return true;
    }

    // Diagonal 1
    m = pieces & (pieces >> ROWS);
    if(m & (m >> (2 * ROWS))) {
        return true;
    }

    // Diagonal 2 
    m = pieces & (pieces >> (ROWS + 2));
    if(m & (m >> (2 * (ROWS + 2)))) {
        return true;
    }

    // Vertical;
    m = pieces & (pieces >> 1);
    if(m & (m >> 2)) {
        return true;
    }

    return false;
}

// https://en.wikipedia.org/wiki/Negamax#Negamax_variant_with_no_color_parameter
float negamax(uint64_t pieces, uint64_t other_pieces, int depth) {
    uint64_t piece_mask = pieces | other_pieces;
    if (won(other_pieces)) {
        return pow(10, depth);
    }
    if (won(pieces)) {
        return 0;
    }

    if (depth == 0 || draw(piece_mask)) {
        return 0;
    }

    float value = 0;
    for (uint64_t column = 0; column < 7; column ++) {
        if (invalid_move(column, piece_mask)) {
            continue;
        }
        uint64_t child_pieces = play(column, piece_mask, other_pieces);
        value -= negamax(other_pieces, child_pieces, depth - 1);
    }
    return value;
}

int compute_env_move(Connect4* env) {
    uint64_t piece_mask = env->player_pieces | env->env_pieces;
    uint64_t hash = env->player_pieces + piece_mask + c_bottom();

    // Hard coded opening book to handle some early game traps
    // TODO: Add more opening book moves
    switch (hash) {
        case 4398050705408:
            // Respond to _ _ _ o _ _ _
            // with       _ _ x o _ _ _
            return 2;
        case 4398583382016:
            // Respond to _ _ _ _ o _ _
            // with       _ _ _ x o _ _
            return 3;
    }

    float best_value = 9999;
    float values[7];
    for (int i = 0; i < 7; i++) {
        values[i] = 9999;
    }
    for (uint64_t column = 0; column < 7; column ++) {
        if (invalid_move(column, piece_mask)) {
            continue;
        }
        uint64_t child_env_pieces = play(column, piece_mask, env->player_pieces);
        if (won(child_env_pieces)) {
            return column;
        }
        float val = -negamax(env->player_pieces, child_env_pieces, 3);
        values[column] = val;
        if (val < best_value) {
            best_value = val;
        }
    }
    int num_ties = 0;
    for (uint64_t column = 0; column < 7; column ++) {
        if (values[column] == best_value) {
            num_ties++;
        }
    }
    if (num_ties <= 0) {
        return 0;
    }
    int best_tie = rand_r(&env->rng) % num_ties;
    for (uint64_t column = 0; column < 7; column ++) {
        if (values[column] == best_value) {
            if (best_tie == 0) {
                return column;

            }
            best_tie--;
        }
    }
    return 0;
}

void compute_observation(Connect4* env) {
    obs_t* obs = env->agents[0].observations;
    if (env->representation != 0) {
        const int* spec = CONNECT4CNN_REPRESENTATIONS[env->representation];
        int cw = spec[0], ch = spec[1];
        int left = (OBS_WIDTH - COLUMNS * cw) / 2;
        int top = (OBS_HEIGHT - ROWS * ch) / 2;
        memset(obs, 0, OBS_SIZE * sizeof(obs_t));
        for (int row = 0; row < ROWS; row++) {
            for (int column = 0; column < COLUMNS; column++) {
                uint64_t bit = UINT64_C(1) << (column * (ROWS + 1) + row);
                int owner = env->player_pieces & bit ? 0 : env->env_pieces & bit ? 1 : -1;
                if (owner < 0) continue;
                float pixel = owner == 0 ? 1.0f : 0.5f;
                int base = (top + (ROWS - 1 - row) * ch) * OBS_WIDTH + left + column * cw;
                uint64_t mask = env->glyph[owner];
                for (int y = 0; y < ch; y++) {
                    for (int x = 0; x < cw; x++) {
                        if ((mask >> (y * cw + x)) & 1) obs[base + y * OBS_WIDTH + x] = pixel;
                    }
                }
            }
        }
        return;
    }
    // Normalized grayscale, row-major [1,36,44], top row first.
    // 6x6 cells, with one black column on each side. Without the padding,
    // Nature's valid strided convolutions would ignore the seventh column.
    for (int y = 0; y < OBS_HEIGHT; y++) {
        int row = ROWS - 1 - y / CONNECT4CNN_CELL_PIXELS;
        obs[y * OBS_WIDTH] = 0;
        obs[y * OBS_WIDTH + OBS_WIDTH - 1] = 0;
        for (int column = 0; column < COLUMNS; column++) {
            int bit = column * (ROWS + 1) + row;
            float pixel = ((env->player_pieces >> bit) & 1) ? 1.0f
                        : ((env->env_pieces >> bit) & 1) ? 0.5f : 0.0f;
            int x = CONNECT4CNN_SIDE_PADDING + column * CONNECT4CNN_CELL_PIXELS;
            for (int dx = 0; dx < CONNECT4CNN_CELL_PIXELS; dx++) {
                obs[y * OBS_WIDTH + x + dx] = pixel;
            }
        }
    }
}

static void reset_board(Connect4* env) {
    float* obs = env->agents[0].observations;
    env->tick = 0;
    env->player_pieces = 0;
    env->env_pieces = 0;
    env->last_env_bit = 0;
    for (int i = 0; i < OBS_SIZE; i++) {
        obs[i] = 0;
    }
}

void puf_reset(Connect4* env) {
    reset_board(env);
    env->pending_reset = 0;
    env->agents[0].terminals[0] = 0;
    if (env->agents[0].rewards) {
        env->agents[0].rewards[0] = 0;
    }
}

// Same-step auto-reset as go. A live client delays reset until after render.
void finish_game(Connect4* env, float reward, int invalid) {
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = 1;
    env->log.invalids += (float)invalid;
    add_log(env);
    if (env->client == NULL) {
        reset_board(env);
    } else {
        compute_observation(env);
        env->pending_reset = 1;
    }
}

void puf_step(Connect4* env) {
    if (IsWindowReady() && IsKeyDown(KEY_LEFT_SHIFT)) {
        int col = -1;
        for (int k = 0; k < COLUMNS; k++) {
            if (IsKeyPressed(KEY_ONE + k)) {
                col = k;
            }
        }
        if (col < 0) {
            return;
        }
        env->agents[0].actions[0] = col;
    }
    env->tick += 1;
    env->last_env_bit = 0;
    env->agents[0].rewards[0] = 0.0;
    env->agents[0].terminals[0] = 0;

    // Player action (PLAYER_WIN)
    uint64_t column = (uint64_t)env->agents[0].actions[0];
    uint64_t piece_mask = env->player_pieces | env->env_pieces;
    if (invalid_move(column, piece_mask)) {
        finish_game(env, ENV_WIN, 1);
        return;
    }

    env->player_pieces = play(column, piece_mask, env->env_pieces);
    if (won(env->player_pieces)) {
        finish_game(env, PLAYER_WIN, 0);
        return;
    }
    if (draw(env->player_pieces | env->env_pieces)) {
        finish_game(env, 0, 0);
        return;
    }

    // Environment action (ENV_WIN)
    column = compute_env_move(env);
    piece_mask = env->player_pieces | env->env_pieces;
    if (invalid_move(column, piece_mask)) {
        finish_game(env, PLAYER_WIN, 0);
        return;
    }

    uint64_t new_env = play(column, piece_mask, env->player_pieces);
    env->last_env_bit = new_env ^ env->env_pieces;
    env->env_pieces = new_env;
    if (won(env->env_pieces)) {
        finish_game(env, ENV_WIN, 0);
        return;
    }
    if (draw(env->env_pieces | env->player_pieces)) {
        finish_game(env, 0, 0);
        return;
    }

    compute_observation(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct Client Client;
struct Client {
    float width;
    float height;
    Texture2D puffers;
};

Client* make_client() {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = WIDTH;
    client->height = HEIGHT;

    InitWindow(WIDTH, HEIGHT, "PufferLib Connect4CNN");
    SetTargetFPS(60);

    client->puffers = LoadTexture("resources/shared/puffers_128.png");
    return client;
}

void puf_render(Connect4* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    if (env->client == NULL) {
        env->client = make_client();
    }

    Client* client = env->client;
    int frames = (env->last_env_bit || env->pending_reset) ? HOLD_FRAMES : 0;
    uint64_t hide = env->last_env_bit;
    env->last_env_bit = 0;
    int f = 0;
redraw:
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    int y_offset = client->height - PIECE_HEIGHT;
    for (int i = 0; i < 49; i++) {
        // Read the board directly: the policy observation is now an image.
        if ((i + 1) % 7 == 0) {
            continue;
        }

        int row = i % (ROWS + 1);
        int column = i / (ROWS + 1);
        int y = y_offset - row * PIECE_HEIGHT;
        int x = column * PIECE_WIDTH;

        Color piece_color=PURPLE;
        int color_idx = 0;
        float cell = ((env->player_pieces >> i) & 1) ? (float)PLAYER_WIN
                   : ((env->env_pieces >> i) & 1) ? (float)ENV_WIN : 0.0f;
        if (f < frames && ((hide >> i) & 1)) {
            cell = 0;
        }
        if (cell == 0.0f) {
            piece_color = BLACK;
        } else if (cell == (float)PLAYER_WIN) {
            piece_color = PUFF_CYAN;
            color_idx = 1;
        } else if (cell == (float)ENV_WIN) {
            piece_color = PUFF_RED;
            color_idx = 2;
        }

        Color board_color = (Color){0, 80, 80, 255};
        DrawRectangle(x , y , PIECE_WIDTH, PIECE_WIDTH, board_color);
        DrawCircle(x + PIECE_WIDTH/2, y + PIECE_WIDTH/2, PIECE_WIDTH/2, piece_color);
        if (color_idx == 0) {
            continue;
        }

        DrawTexturePro(
            client->puffers,
            (Rectangle){
                (color_idx == 1) ? 0 : 128,
                0, 128, 128,
            },
            (Rectangle){x+16, y+16, PIECE_WIDTH-32, PIECE_WIDTH-32},
            (Vector2){0, 0},
            0,
            WHITE
        );
    }
    EndDrawing();
    puf_web_vsync();
    if (f++ < frames) {
        goto redraw;
    }
    if (env->pending_reset) {
        reset_board(env);
        env->pending_reset = 0;
    }
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void puf_close(Connect4* env) {
    if (env->client != NULL) {
        close_client(env->client);
        env->client = NULL;
    }
}

// --- Native trainer (pufferl) API ---
void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "n", log->n);
    dict_set(out, "invalids", log->invalids);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->player_pieces = dict_get(kwargs, "player_pieces");
    env->env_pieces = dict_get(kwargs, "env_pieces");
    env->representation = cnn_appearance_init(kwargs, env->rng, CONNECT4CNN_NUM_REPRESENTATIONS);
    const int* spec = CONNECT4CNN_REPRESENTATIONS[env->representation];
    int cw = spec[0], ch = spec[1], width = spec[2], height = spec[3], style = spec[4];
    int left = (cw - width) / 2, top = (ch - height) / 2;
    env->glyph[0] = env->glyph[1] = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = 2 * x - width + 1, dy = 2 * y - height + 1;
            int disk = dx * dx + dy * dy <= width * width;
            uint64_t bit = UINT64_C(1) << ((y + top) * cw + x + left);
            if (style == 2) {
                if (x == y || x + y == width - 1) env->glyph[0] |= bit;
                if (disk && dx * dx + dy * dy >= (width - 2) * (width - 2)) env->glyph[1] |= bit;
            } else if (style == 0 || disk) {
                env->glyph[0] |= bit;
                env->glyph[1] |= bit;
            }
        }
    }
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
    init(env);
}
