static inline void flappy_compute_observations(Flappy* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    float sx = (float)OBS_WIDTH / env->width, sy = (float)OBS_HEIGHT / env->height;
    int outline = env->representation == 5 || env->representation == 6
        ? (int)fmaxf(1, 3 * fminf(sx, sy)) : 0;
    for (int i = 0; i < FLAPPY_NUM_PIPES; i++) {
        Pipe* pipe = &env->pipes[i];
        float x0 = pipe->x * sx, x1 = (pipe->x + env->pipe_width) * sx;
        cnn_image_rect(obs, x0, 0, x1, (pipe->gap_y - env->pipe_gap / 2) * sy, CNN_GREEN, outline);
        cnn_image_rect(obs, x0, (pipe->gap_y + env->pipe_gap / 2) * sy, x1, OBS_HEIGHT, CNN_GREEN, outline);
    }
    float cx = env->bird_x * sx, cy = env->bird_y * sy;
    float rx = env->bird_radius * sx, ry = env->bird_radius * sy;
    if (env->representation == 4 || env->representation == 6)
        cnn_image_ellipse(obs, cx, cy, rx, ry, CNN_YELLOW, false);
    else cnn_image_rect(obs, cx - rx, cy - ry, cx + rx, cy + ry, CNN_YELLOW, 0);
    if (env->representation == 1) cnn_image_flip(obs, false, 0);
    if (env->representation == 2) cnn_image_flip(obs, true, 0);
    if (env->representation == 3) cnn_image_invert(obs, 0);
}
