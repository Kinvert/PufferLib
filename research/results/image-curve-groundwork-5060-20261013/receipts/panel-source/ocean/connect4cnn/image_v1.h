// human-geometry-v1: full 7x6 board; glyphs rasterized at output pixel centers.
void compute_observation(Connect4* env) {
    obs_t* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    const int* spec = CONNECT4CNN_REPRESENTATIONS[env->representation];
    // These are geometric sizes relative to full cells, not tiny-raster pixels.
    // Ten IDs remain distinct even though every version now spans the full board.
    const float size_x[] = {0.90f,0.60f,0.30f,0.90f,0.60f,0.90f,0.80f,0.50f,0.25f,0.45f};
    const float size_y[] = {0.90f,0.60f,0.30f,0.90f,0.60f,0.90f,0.80f,0.50f,0.25f,0.85f};
    float cw = (float)OBS_WIDTH / COLUMNS, ch = (float)OBS_HEIGHT / ROWS;
    for (int row = 0; row < ROWS; row++) for (int col = 0; col < COLUMNS; col++) {
        uint64_t bit = UINT64_C(1) << (col * (ROWS + 1) + row);
        int owner = env->player_pieces & bit ? 0 : env->env_pieces & bit ? 1 : -1;
        float cx = (col + 0.5f) * cw, cy = (ROWS - row - 0.5f) * ch;
        // Thin grid remains visible on an empty board, without state side channels.
        cnn_image_rect(obs, col * cw, (ROWS - 1 - row) * ch,
            (col + 1) * cw, (ROWS - row) * ch, CNN_GRAY, 1);
        if (owner < 0) continue;
        CnnColor color = owner == 0 ? CNN_RED : CNN_YELLOW;
        float rx = 0.5f * cw * size_x[env->representation], ry = 0.5f * ch * size_y[env->representation];
        if (spec[4] == 1) cnn_image_ellipse(obs, cx, cy, rx, ry, color, false);
        else if (spec[4] == 2 && owner == 1) cnn_image_ellipse(obs, cx, cy, rx, ry, color, true);
        else if (spec[4] == 2) {
            int left = (int)floorf(cx - rx), right = (int)ceilf(cx + rx);
            int top = (int)floorf(cy - ry), bottom = (int)ceilf(cy + ry);
            for (int y = top; y < bottom; y++) for (int x = left; x < right; x++) {
                float dx = (x + 0.5f - cx) / rx, dy = (y + 0.5f - cy) / ry;
                if (fabsf(dx - dy) < 0.2f || fabsf(dx + dy) < 0.2f) cnn_image_pixel(obs, x, y, color);
            }
        } else cnn_image_rect(obs, cx - rx, cy - ry, cx + rx, cy + ry, color, 0);
    }
}
