void compute_observations(Env* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    int head = env->snake[env->head_ptr];
    int row = head / env->width - 5, col = head % env->width - 5;
    for (int y = 0; y < 11; y++) for (int x = 0; x < 11; x++) {
        int tile = env->grid[(row + y) * env->width + col + x];
        CnnColor color = tile == SNAKE_FOOD ? CNN_RED : tile == SNAKE_WALL ? CNN_GRAY
            : tile == SNAKE_BODY ? CNN_GREEN : CNN_BLACK;
        if (env->representation == 4 && tile == SNAKE_FOOD) color = CNN_GREEN;
        if (env->representation == 4 && tile == SNAKE_BODY) color = CNN_RED;
        color.gray = tile == SNAKE_FOOD ? 0.25f : tile == SNAKE_BODY ? 0.75f : color.gray;
        if (env->representation == 4 && tile == SNAKE_FOOD) color.gray = 0.75f;
        if (env->representation == 4 && tile == SNAKE_BODY) color.gray = 0.25f;
        if (x == 5 && y == 5) color = CNN_WHITE;
        cnn_image_cell(obs, x, y, 11, color, env->representation);
    }
    if (env->representation == 3) cnn_image_invert(obs, 0);
    if (env->representation == 5) cnn_image_flip(obs, false, 0);
}
