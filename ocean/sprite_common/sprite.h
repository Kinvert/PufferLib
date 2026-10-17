// Original 8x8 art is enlarged once; steps copy cached float32 RGB tile rows.
// Each game defines its grid dimensions, artwork, palette and player tile.
#define SPRITE_CELL 32
#define OBS_WIDTH (GRID_WIDTH * SPRITE_CELL)
#define OBS_HEIGHT (GRID_HEIGHT * SPRITE_CELL)
#define OBS_CHANNELS 3
#define PIXEL_OBS_SIZE (OBS_WIDTH * OBS_HEIGHT * OBS_CHANNELS)
#define PUF_NATURE_ENCODER
#define SPRITE_TEXTURES 4
#define SPRITE_PIXELS "sprite-background-v2"
#ifdef SPRITE_ATTENTION
#define SPRITE_VIEW_TILES 18
#define SPRITE_TILE_PIXELS (SPRITE_CELL * SPRITE_CELL * OBS_CHANNELS)
#define SPRITE_VIEW_PIXELS (SPRITE_VIEW_TILES * SPRITE_TILE_PIXELS)
#define SPRITE_VIEW_OBS (SPRITE_VIEW_PIXELS + 4)
#define SPRITE_VIEW_STATE (SPRITE_VIEW_TILES * SPRITE_TILE_TYPES)
#endif

typedef struct {
    Texture2D texture;
    Color* pixels;
    float* frame;
} SpriteClient;

float sprite_tiles[3][SPRITE_TILE_TYPES][SPRITE_TEXTURES][SPRITE_CELL * SPRITE_CELL * 3];

void sprite_make_tiles(int rep) {
    // Environment setup is serial; prepare only the skin being used.
    static int ready[3];
    if (ready[rep]) {
        return;
    }
    for (int kind = 0; kind < SPRITE_TILE_TYPES; kind++) {
        for (int pixel = 0; pixel < SPRITE_TEXTURES * SPRITE_CELL * SPRITE_CELL; pixel++) {
            int texture = pixel / (SPRITE_CELL * SPRITE_CELL);
            int at = pixel % (SPRITE_CELL * SPRITE_CELL);
            int x = at % SPRITE_CELL, y = at / SPRITE_CELL;
            if (rep == 2) {
                x = SPRITE_CELL - 1 - x;
            }
            char glyph = SPRITE_ART[kind][y / 4][x / 4];
            int index = glyph <= '9' ? glyph - '0' : glyph - 'a' + 10;
            unsigned int grain = (x / 2 + 17 * (y / 2) + 1) * 0x45d9f3bu;
            grain ^= texture * 0x27d4eb2du;
            grain ^= grain >> 16;
            int shade = index == 0 ? (int)(grain >> 28) - 7 : 0;
            for (int channel = 0; channel < 3; channel++) {
                int source = rep == 1 ? (channel + 1) % 3 : channel;
                int value = SPRITE_PALETTE[index][source] + shade;
                if (value < 0) {
                    value = 0;
                }
                if (value > 255) {
                    value = 255;
                }
                sprite_tiles[rep][kind][texture][3 * at + channel] = value / 255.f;
            }
        }
    }
    ready[rep] = 1;
}

void sprite_write_cell(float* frame, const unsigned char* grid, int player,
        int cell, int representation, unsigned int floor_seed, unsigned char* pixel_grid) {
    int kind = cell == player ? SPRITE_PLAYER : grid[cell];
    unsigned int value = floor_seed ^ (cell * 0x9e3779b9u);
    value ^= value >> 16;
    value *= 0x85ebca6bu;
    int texture = value >> 30;
    int stamp = kind * SPRITE_TEXTURES + texture;
    if (pixel_grid && pixel_grid[cell] == stamp) {
        return;
    }
    const float* tile = sprite_tiles[representation][kind][texture];
    int left = cell % GRID_WIDTH * SPRITE_CELL;
    int top = cell / GRID_WIDTH * SPRITE_CELL;
    for (int y = 0; y < SPRITE_CELL; y++) {
        memcpy(frame + ((top + y) * OBS_WIDTH + left) * 3,
            tile + y * SPRITE_CELL * 3, SPRITE_CELL * 3 * sizeof(float));
    }
    if (pixel_grid) {
        pixel_grid[cell] = stamp;
    }
}

void sprite_observe(float* frame, const unsigned char* grid, int player, int rep,
        unsigned int floor_seed, unsigned char* pixel_grid) {
    for (int cell = 0; cell < GRID_WIDTH * GRID_HEIGHT; cell++) {
        sprite_write_cell(frame, grid, player, cell, rep, floor_seed, pixel_grid);
    }
}

void sprite_render(SpriteClient** client, const unsigned char* grid, int player,
        int representation, unsigned int floor_seed, const char* title, int fps) {
    if (!*client) {
        InitWindow(OBS_WIDTH, OBS_HEIGHT, title);
        SetTargetFPS(fps);
        *client = (SpriteClient*)calloc(1, sizeof(SpriteClient));
        (*client)->pixels = (Color*)calloc(OBS_WIDTH * OBS_HEIGHT, sizeof(Color));
        (*client)->frame = (float*)calloc(PIXEL_OBS_SIZE, sizeof(float));
        Image image = {(*client)->pixels, OBS_WIDTH, OBS_HEIGHT, 1,
            PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
        (*client)->texture = LoadTextureFromImage(image);
        sprite_make_tiles(representation);
    }
    SpriteClient* c = *client;
    sprite_observe(c->frame, grid, player, representation, floor_seed, NULL);
    for (int i = 0; i < OBS_WIDTH * OBS_HEIGHT; i++) {
        c->pixels[i] = (Color){c->frame[3 * i] * 255 + .5f,
            c->frame[3 * i + 1] * 255 + .5f, c->frame[3 * i + 2] * 255 + .5f, 255};
    }
    UpdateTexture(c->texture, c->pixels);
    BeginDrawing();
    DrawTexture(c->texture, 0, 0, WHITE);
    EndDrawing();
    puf_web_vsync();
}

void sprite_close(SpriteClient* client) {
    if (client) {
        UnloadTexture(client->texture);
        free(client->pixels);
        free(client->frame);
        free(client);
        CloseWindow();
    }
}
