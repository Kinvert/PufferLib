// Research view: optional player neighbors, then the gaze's row-major 3x3 cells.
void forage_attention_action(Env* env) {
#if FORAGE_GAZE_MODE == 1
    // Same ordering as player movement: noop, up, down, left, right.
    int action = env->agents[0].actions[1];
    int x = env->gaze_idx % FORAGE_WIDTH + (action == 4) - (action == 3);
    int y = env->gaze_idx / FORAGE_WIDTH + (action == 2) - (action == 1);
    if (x < 0) x = 0;
    if (x >= FORAGE_WIDTH) x = FORAGE_WIDTH - 1;
    if (y < 0) y = 0;
    if (y >= FORAGE_HEIGHT) y = FORAGE_HEIGHT - 1;
    env->gaze_idx = y * FORAGE_WIDTH + x;
#elif FORAGE_GAZE_MODE == 2
    int x = env->agents[0].actions[1];
    int y = env->agents[0].actions[2];
    env->gaze_idx = y * FORAGE_WIDTH + x;
#else
    env->gaze_idx = env->agents[0].actions[1];
#endif
}

int forage_attention_cell(Env* env, int slot) {
#if FORAGE_PLAYER_VIEW
    int center = slot < 8 ? env->player_idx : env->gaze_idx;
    int at = slot < 8 ? slot + (slot >= 4) : slot - 8;
#else
    int center = env->gaze_idx;
    int at = slot;
#endif
    int x = center % FORAGE_WIDTH + at % 3 - 1;
    int y = center / FORAGE_WIDTH + at / 3 - 1;
    if (x < 0 || x >= FORAGE_WIDTH || y < 0 || y >= FORAGE_HEIGHT) {
        return -1;
    }
    return y * FORAGE_WIDTH + x;
}

void forage_attention_observations(Env* env) {
    obs_t* obs = env->agents[0].observations;
#if FORAGE_STATE_ONEHOT
    if (!env->observation_mode) {
        memset(obs, 0, FORAGE_STATE_VALUES * sizeof(obs_t));
    }
#endif
    for (int slot = 0; slot < FORAGE_VIEW_TILES; slot++) {
        int cell = forage_attention_cell(env, slot);
        int kind = cell < 0 ? FORAGE_TREE
            : cell == env->player_idx ? FORAGE_PLAYER : env->grid[cell];
        if (env->observation_mode) {
            memcpy(obs + slot * FORAGE_TILE_PIXELS,
                forage_tiles[env->representation][kind], FORAGE_TILE_PIXELS * sizeof(obs_t));
        } else {
#if FORAGE_STATE_ONEHOT
            obs[slot * FORAGE_TILE_TYPES + kind] = 1;
#else
            obs[slot] = kind / (float)(FORAGE_TILE_TYPES - 1);
#endif
        }
    }
    int offset = env->observation_mode ? FORAGE_VIEW_PIXELS : FORAGE_STATE_VALUES;
    // Identical geometry metadata in state/pixels; no hidden object information.
    obs[offset] = (env->player_idx % FORAGE_WIDTH) / (float)(FORAGE_WIDTH - 1);
    obs[offset + 1] = (env->player_idx / FORAGE_WIDTH) / (float)(FORAGE_HEIGHT - 1);
    int x = env->gaze_idx % FORAGE_WIDTH;
    int y = env->gaze_idx / FORAGE_WIDTH;
#if FORAGE_RELATIVE_GAZE
    x -= env->player_idx % FORAGE_WIDTH;
    y -= env->player_idx / FORAGE_WIDTH;
#endif
    obs[offset + 2] = x / (float)(FORAGE_WIDTH - 1);
    obs[offset + 3] = y / (float)(FORAGE_HEIGHT - 1);
}
