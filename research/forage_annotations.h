// Pixel-space annotation helpers, used only by research collectors/evaluators.
// Call after including Sprite Forage. Labels are targets, never policy inputs.
static int forage_foreground(int skin, int kind, int pixel) {
    if (kind == FORAGE_EMPTY) return 0;
    if (skin == 4) return FORAGE_IMPORTED[kind][4 * pixel + 3] != 0;
    int x = pixel % FORAGE_CELL / 2;
    if (skin == 2) x = 15 - x;
    return FORAGE_ART[kind][pixel / FORAGE_CELL / 2][x] != '.';
}

static void forage_visible_labels(const Env* env, unsigned char* labels) {
    for (int cell = 0; cell < FORAGE_AREA; cell++)
        labels[cell] = cell == env->player_idx ? FORAGE_PLAYER : env->grid[cell];
}

static int forage_annotation_checks() {
    for (int skin = 0; skin < FORAGE_REPRESENTATIONS; skin++) {
        forage_make_tiles(skin);
        for (int kind = 0; kind < FORAGE_TILE_TYPES; kind++) {
            int foreground = 0;
            for (int pixel = 0; pixel < FORAGE_CELL * FORAGE_CELL; pixel++) {
                int visible = forage_foreground(skin, kind, pixel);
                foreground += visible;
                if (!visible && memcmp(forage_tiles[skin][kind] + 3 * pixel,
                        forage_tiles[skin][FORAGE_EMPTY] + 3 * pixel, 3 * sizeof(float))) return 1;
            }
            if ((kind == FORAGE_EMPTY && foreground) || (kind && !foreground)) return 2;
        }
        if (!memcmp(forage_tiles[skin][FORAGE_APPLE], forage_tiles[skin][FORAGE_SPOTTED_APPLE],
                FORAGE_CELL * FORAGE_CELL * 3 * sizeof(float))
            || !memcmp(forage_tiles[skin][FORAGE_MUSHROOM], forage_tiles[skin][FORAGE_TOADSTOOL],
                FORAGE_CELL * FORAGE_CELL * 3 * sizeof(float))) return 3;
    }
    return 0;
}
