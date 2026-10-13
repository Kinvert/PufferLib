void compute_observations(Breakout* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    float sx = (float)OBS_WIDTH / env->width, sy = (float)OBS_HEIGHT / env->height;
    CnnColor brick = env->representation == 4 ? CNN_YELLOW : CNN_RED;
    CnnColor paddle = env->representation == 4 ? CNN_RED : CNN_YELLOW;
    for (int i = 0; i < env->num_bricks; i++) {
        if (env->brick_states[i] == 1) continue;
        cnn_image_rect(obs, env->brick_x[i] * sx, env->brick_y[i] * sy,
            (env->brick_x[i] + env->brick_width) * sx,
            (env->brick_y[i] + env->brick_height) * sy, brick, 0);
    }
    cnn_image_rect(obs, env->paddle_x * sx, env->paddle_y * sy,
        (env->paddle_x + env->paddle_width) * sx,
        (env->paddle_y + env->paddle_height) * sy, paddle, 0);
    cnn_image_rect(obs, env->ball_x * sx, env->ball_y * sy,
        (env->ball_x + env->ball_width) * sx,
        (env->ball_y + env->ball_height) * sy, CNN_WHITE, 0);
    if (env->representation == 1) cnn_image_flip(obs, false, 0);
    if (env->representation == 2) cnn_image_flip(obs, true, 0);
    if (env->representation == 3) cnn_image_invert(obs, 0);
}
