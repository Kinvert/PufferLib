// Verify compiled RGBA values against the original image files, no GPU.
#include <assert.h>
#include <stdio.h>
#include "raylib.h"
#include IMPORTED_HEADER
int main(int argc, char** argv) {
    assert(argc == 13); SetTraceLogLevel(LOG_WARNING);
    const int cell = FORAGE_IMPORTED_CELL_PIXELS;
    for (int kind = 1; kind <= 12; kind++) {
        Image image = LoadImage(argv[kind]); assert(image.data);
        Color* original = LoadImageColors(image);
        int scale = cell / image.height;
        if (cell / image.width < scale) scale = cell / image.width;
        assert(scale > 0);
        int left = (cell - scale * image.width) / 2, top = cell - scale * image.height;
        for (int y = 0; y < cell; y++) for (int x = 0; x < cell; x++) {
            Color expected = {0};
            int source_x = (x - left) / scale, source_y = (y - top) / scale;
            if (x >= left && y >= top && x < left + scale * image.width)
                expected = original[source_y * image.width + source_x];
            const unsigned char* actual = FORAGE_IMPORTED[kind] + (y * cell + x) * 4;
            assert(actual[0] == expected.r && actual[1] == expected.g && actual[2] == expected.b && actual[3] == expected.a);
        }
        UnloadImageColors(original); UnloadImage(image);
    }
    printf("Imported atlas:12 source sprites preserve RGBA/aspect ratio/integer scaling/padding at cell%d PASS\n", cell);
    return 0;
}
