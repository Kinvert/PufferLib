void compute_observations(Pong* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    // Match the viewer's three-paddle-width side margins. Score bars occupy a
    // declared HUD strip; the world extent/physics and upward y stay unchanged.
    int hud = (OBS_HEIGHT + 39) / 40;
    float padding = 3 * env->paddle_width;
    float sx = (float)OBS_WIDTH / (env->width + 2 * padding);
    float sy = (float)(OBS_HEIGHT - hud) / env->height;
    float margin = padding * sx;
    cnn_image_rect(obs, 0, 0, OBS_WIDTH * (float)env->score_l / env->max_score, hud / 2, CNN_RED, 0);
    cnn_image_rect(obs, 0, hud / 2, OBS_WIDTH * (float)env->score_r / env->max_score, hud, CNN_YELLOW, 0);
    CnnColor left = env->representation == 1 ? CNN_YELLOW : CNN_RED;
    CnnColor right = env->representation == 1 ? CNN_RED : CNN_YELLOW;
    cnn_image_rect(obs, margin - env->paddle_width * sx,
        OBS_HEIGHT - (env->paddle_yl + env->paddle_height) * sy, margin,
        OBS_HEIGHT - env->paddle_yl * sy, left, 0);
    cnn_image_rect(obs, OBS_WIDTH - margin,
        OBS_HEIGHT - (env->paddle_yr + env->paddle_height) * sy,
        OBS_WIDTH - margin + env->paddle_width * sx, OBS_HEIGHT - env->paddle_yr * sy, right, 0);
    cnn_image_rect(obs, margin + env->ball_x * sx,
        OBS_HEIGHT - (env->ball_y + env->ball_height) * sy,
        margin + (env->ball_x + env->ball_width) * sx,
        OBS_HEIGHT - env->ball_y * sy, CNN_WHITE, 0);
    if (env->representation == 2) cnn_image_flip(obs, false, hud);
    if (env->representation == 3) cnn_image_flip(obs, true, hud);
    if (env->representation == 4) cnn_image_invert(obs, hud);
    // Broad visible checker/scanline textures retain separate object color bands.
    if (env->representation >= 5) {
        for (int y = hud; y < OBS_HEIGHT; y++) for (int x = 0; x < OBS_WIDTH; x++) {
            int bit = env->representation == 5 ? ((x / 8 + y / 8) & 1) : ((y / 4) & 1);
            if (!bit) continue;
            for (int c = 0; c < OBS_CHANNELS; c++) {
                int i = (y * OBS_WIDTH + x) * OBS_CHANNELS + c;
                obs[i] = 0.0625f + 0.875f * obs[i];
            }
        }
    }
}
