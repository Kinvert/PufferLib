#pragma once
#include <stdbool.h>

// Explicit build-time image contract. Shapes stay fixed within a CUDA graph.
// No flags keeps the historical float32, grayscale 36x44 input unchanged.
#ifdef CNN_IMAGE_V1
#if !defined(CNN_IMAGE_HEIGHT) || !defined(CNN_IMAGE_WIDTH) || !defined(CNN_IMAGE_CHANNELS)
#error "CNN_IMAGE_V1 requires explicit height, width and channels"
#endif
#if CNN_IMAGE_HEIGHT < 36 || CNN_IMAGE_WIDTH < 36 || CNN_IMAGE_HEIGHT > 1024 || CNN_IMAGE_WIDTH > 1024
#error "Image dimensions must be between 36 and 1024"
#endif
#if CNN_IMAGE_CHANNELS != 1 && CNN_IMAGE_CHANNELS != 3
#error "Image channels must be 1 (grayscale) or 3 (RGB)"
#endif
#define OBS_HEIGHT CNN_IMAGE_HEIGHT
#define OBS_WIDTH CNN_IMAGE_WIDTH
#define OBS_CHANNELS CNN_IMAGE_CHANNELS
#define CNN_IMAGE_STRING_IMPL(x) #x
#define CNN_IMAGE_STRING(x) CNN_IMAGE_STRING_IMPL(x)
#define CNN_IMAGE_INFO_JSON ",\"image_contract\":{\"version\":\"human-geometry-v1\",\"shape_hwc\":[" \
    CNN_IMAGE_STRING(OBS_HEIGHT) "," CNN_IMAGE_STRING(OBS_WIDTH) "," CNN_IMAGE_STRING(OBS_CHANNELS) \
    "],\"layout\":\"NHWC\",\"dtype\":\"float32\",\"preprocessing\":\"none\"}"
#else
#define OBS_HEIGHT 36
#define OBS_WIDTH 44
#define OBS_CHANNELS 1
#endif
#define OBS_SIZE (OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS)

#ifdef CNN_IMAGE_V1
// NHWC float32 in [0,1]. Rasterize world geometry directly at this resolution.
typedef struct { float gray, r, g, b; } CnnColor;
static const CnnColor CNN_BLACK = {0, 0, 0, 0};
static const CnnColor CNN_RED = {0.5f, 0.875f, 0.125f, 0.125f};
static const CnnColor CNN_GREEN = {0.5f, 0.125f, 0.75f, 0.25f};
static const CnnColor CNN_YELLOW = {0.75f, 0.875f, 0.75f, 0.125f};
static const CnnColor CNN_GRAY = {0.25f, 0.375f, 0.375f, 0.375f};
static const CnnColor CNN_WHITE = {1, 1, 1, 1};

static inline void cnn_image_pixel(obs_t* obs, int x, int y, CnnColor color) {
    int i = (y * OBS_WIDTH + x) * OBS_CHANNELS;
    if (OBS_CHANNELS == 1) obs[i] = color.gray;
    else { obs[i] = color.r; obs[i + 1] = color.g; obs[i + 2] = color.b; }
}

static inline void cnn_image_rect(obs_t* obs, float x0, float y0,
        float x1, float y1, CnnColor color, int outline) {
    int left = (int)fmaxf(0, fminf(OBS_WIDTH, floorf(x0)));
    int right = (int)fmaxf(0, fminf(OBS_WIDTH, ceilf(x1)));
    int top = (int)fmaxf(0, fminf(OBS_HEIGHT, floorf(y0)));
    int bottom = (int)fmaxf(0, fminf(OBS_HEIGHT, ceilf(y1)));
    for (int y = top; y < bottom; y++) for (int x = left; x < right; x++) {
        if (!outline || x - left < outline || right - 1 - x < outline
                || y - top < outline || bottom - 1 - y < outline)
            cnn_image_pixel(obs, x, y, color);
    }
}

static inline void cnn_image_ellipse(obs_t* obs, float cx, float cy,
        float rx, float ry, CnnColor color, bool ring) {
    if (rx <= 0 || ry <= 0) return;
    int left = (int)fmaxf(0, floorf(cx - rx)), right = (int)fminf(OBS_WIDTH, ceilf(cx + rx));
    int top = (int)fmaxf(0, floorf(cy - ry)), bottom = (int)fminf(OBS_HEIGHT, ceilf(cy + ry));
    for (int y = top; y < bottom; y++) for (int x = left; x < right; x++) {
        float dx = (x + 0.5f - cx) / rx, dy = (y + 0.5f - cy) / ry;
        float d = dx * dx + dy * dy;
        if (d <= 1 && (!ring || d >= 0.55f)) cnn_image_pixel(obs, x, y, color);
    }
}

static inline void cnn_image_flip(obs_t* obs, bool vertical, int first_row) {
    int height = OBS_HEIGHT - first_row;
    for (int y = first_row; y < OBS_HEIGHT; y++) for (int x = 0; x < OBS_WIDTH; x++) {
        int other_y = vertical ? OBS_HEIGHT - 1 - (y - first_row) : y;
        int other_x = vertical ? x : OBS_WIDTH - 1 - x;
        if ((vertical && y - first_row >= height / 2) || (!vertical && x >= OBS_WIDTH / 2)) continue;
        for (int c = 0; c < OBS_CHANNELS; c++) {
            int i = (y * OBS_WIDTH + x) * OBS_CHANNELS + c;
            int j = (other_y * OBS_WIDTH + other_x) * OBS_CHANNELS + c;
            float tmp = obs[i]; obs[i] = obs[j]; obs[j] = tmp;
        }
    }
}

static inline void cnn_image_invert(obs_t* obs, int first_row) {
    for (int i = first_row * OBS_WIDTH * OBS_CHANNELS; i < OBS_SIZE; i++) obs[i] = 1 - obs[i];
}

// Visible cell styles: solid, cross, inset. No change to visibility or state.
static inline void cnn_image_cell(obs_t* obs, int col, int row, int cells,
        CnnColor color, int style) {
    float cw = (float)OBS_WIDTH / cells, ch = (float)OBS_HEIGHT / cells;
    float x = col * cw, y = row * ch;
    if (style == 1) {
        cnn_image_rect(obs, x + cw / 3, y, x + 2 * cw / 3, y + ch, color, 0);
        cnn_image_rect(obs, x, y + ch / 3, x + cw, y + 2 * ch / 3, color, 0);
    } else {
        float pad = style == 2 ? 0.16f : 0;
        cnn_image_rect(obs, x + pad * cw, y + pad * ch,
            x + (1 - pad) * cw, y + (1 - pad) * ch, color, 0);
    }
}
#endif
