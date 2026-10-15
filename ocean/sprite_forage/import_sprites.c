// Offline asset compiler only. Runtime uses the generated arrays and memcpy.
// Build: clang -Iraylib-5.5_linux_amd64/include ... libraylib.a -lGL -lm -lpthread -ldl -lrt
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

int main(int argc, char** argv) {
    SetTraceLogLevel(LOG_WARNING);
    if (argc == 5 && strcmp(argv[1], "--scale") == 0) {
        Image image = LoadImage(argv[2]);
        if (!image.data) return 1;
        int scale = atoi(argv[4]); if (scale < 1 || scale > 16) return 1;
        ImageResizeNN(&image, image.width * scale, image.height * scale);
        bool ok = ExportImage(image, argv[3]); UnloadImage(image); return ok ? 0 : 1;
    }
    int cell = 32, output = 1, input = 2;
    if (argc == 16 && strcmp(argv[1], "--cell") == 0) { cell = atoi(argv[2]); output = 3; input = 4; }
    else if (argc != 14) { fprintf(stderr, "Usage: import_sprites [--cell PIXELS] OUTPUT.h TILE1.png ... TILE12.png\n"); return 1; }
    if (cell < 4 || cell > 96) { fprintf(stderr, "Cell size must be4..96\n"); return 1; }
    FILE* out = fopen(argv[output], "w"); if (!out) return 1;
    fprintf(out, "// Generated offline. Bundled asset provenance: assets/SOURCES.md.\n#pragma once\n"
        "#define FORAGE_IMPORTED_CELL_PIXELS %d\n"
        "static const unsigned char FORAGE_IMPORTED[13][%d] = {\n{0},\n",cell,cell*cell*4);
    for (int tile = 0; tile < 12; tile++) {
        Image image = LoadImage(argv[tile + input]);
        if (!image.data || image.width < 1 || image.height < 1 || image.width > cell || image.height > cell) {
            fprintf(stderr, "Sprite must fit the cell without downsampling: %s\n", argv[tile + input]); fclose(out); return 1;
        }
        Color* colors = LoadImageColors(image);
        int sx = cell / image.width, sy = cell / image.height, scale = sx < sy ? sx : sy;
        int ox = (cell - image.width * scale) / 2, oy = cell - image.height * scale;
        fprintf(out, "{ // slot%d\n", tile + 1);
        for (int y = 0; y < cell; y++) {
            for (int x = 0; x < cell; x++) {
                Color color = {0};
                if (x >= ox && y >= oy && x < ox + image.width * scale)
                    color = colors[((y - oy) / scale) * image.width + (x - ox) / scale];
                if (color.a != 0 && color.a != 255) { fprintf(stderr, "Semitransparent pixel unsupported\n"); return 1; }
                fprintf(out, "%u,%u,%u,%u,", color.r,color.g,color.b,color.a);
            }
            fprintf(out, "\n");
        }
        fprintf(out, "},\n"); UnloadImageColors(colors); UnloadImage(image);
    }
    fprintf(out, "};\n"); fclose(out); return 0;
}
