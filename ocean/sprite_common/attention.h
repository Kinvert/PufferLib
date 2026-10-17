// Temporary research view: player-centered 3x3, then agent-selected 3x3.
// Both state and pixels receive the same cells and relative gaze geometry.
void sprite_attention_action(Env* env) {
    int x = env->agents[0].actions[1];
    int y = env->agents[0].actions[2];
    env->gaze_idx = y * GRID_WIDTH + x;
}

int sprite_attention_cell(Env* env, int slot) {
    int center = slot < 9 ? env->player : env->gaze_idx;
    int at = slot % 9;
    int x = center % GRID_WIDTH + at % 3 - 1;
    int y = center / GRID_WIDTH + at / 3 - 1;
    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT) {
        return -1;
    }
    return y * GRID_WIDTH + x;
}

void sprite_attention_observations(Env* env) {
    float* obs = env->agents[0].observations;
    if (!env->observation_mode) {
        memset(obs, 0, SPRITE_VIEW_STATE * sizeof(float));
    }
    for (int slot = 0; slot < SPRITE_VIEW_TILES; slot++) {
        int cell = sprite_attention_cell(env, slot);
        int kind = cell < 0 ? SPRITE_OUTSIDE
            : cell == env->player ? SPRITE_PLAYER : env->grid[cell];
        if (env->observation_mode) {
            unsigned int value = cell < 0 ? 0 : env->floor_seed ^ (cell * 0x9e3779b9u);
            value ^= value >> 16;
            value *= 0x85ebca6bu;
            const float* tile = sprite_tiles[env->representation][kind][value >> 30];
#ifdef SPRITE_PACKED_VIEW
            int left = slot % 3 * SPRITE_CELL;
            int top = slot / 3 * SPRITE_CELL;
            for (int y = 0; y < SPRITE_CELL; y++) {
                memcpy(obs + ((top + y) * SPRITE_PACKED_WIDTH + left) * 3,
                    tile + y * SPRITE_CELL * 3, SPRITE_CELL * 3 * sizeof(float));
            }
#else
            memcpy(obs + slot * SPRITE_TILE_PIXELS, tile, SPRITE_TILE_PIXELS * sizeof(float));
#endif
        } else {
            obs[slot * SPRITE_TILE_TYPES + kind] = 1;
        }
    }
    int offset = env->observation_mode ? SPRITE_VIEW_PIXELS : SPRITE_VIEW_STATE;
    int px = env->player % GRID_WIDTH;
    int py = env->player / GRID_WIDTH;
    obs[offset] = px / (float)(GRID_WIDTH - 1);
    obs[offset + 1] = py / (float)(GRID_HEIGHT - 1);
    obs[offset + 2] = (env->gaze_idx % GRID_WIDTH - px) / (float)(GRID_WIDTH - 1);
    obs[offset + 3] = (env->gaze_idx / GRID_WIDTH - py) / (float)(GRID_HEIGHT - 1);
}
