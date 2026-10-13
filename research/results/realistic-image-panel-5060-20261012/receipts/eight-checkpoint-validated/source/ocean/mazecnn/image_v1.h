void compute_observations(Grid* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    State* s = &env->state;
    for (int row = 0; row < WINDOW; row++) for (int col = 0; col < WINDOW; col++) {
        int y = s->y - VISION + row, x = s->x - VISION + col;
        int tile = y >= 0 && y < MAX_SIZE && x >= 0 && x < MAX_SIZE
            ? s->maze[maze_offset(y, x)] : EMPTY;
        CnnColor color = tile == WALL ? CNN_GRAY : tile == GOAL ? CNN_GREEN
            : tile == AGENT ? CNN_WHITE : CNN_BLACK;
        if (env->representation == 4 && tile == WALL) color = CNN_GREEN;
        if (env->representation == 4 && tile == GOAL) color = CNN_GRAY;
        cnn_image_cell(obs, col, row, WINDOW, color, env->representation);
    }
    if (env->representation == 3) cnn_image_invert(obs, 0);
    if (env->representation == 5) cnn_image_flip(obs, false, 0);
}
