// Native scalar appearance receipt. No game, images, model or CUDA executes.
#include <math.h>
#include <stdint.h>
#include "../src/ini.h"
#include "../ocean/connect4cnn/appearance.h"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    char* end;
    long count = strtol(argv[2], &end, 10);
    if (*end || !*argv[2] || count < 1 || count > 64) return 2;
    Ini ini = {0};
    puf_ini_load_file(&ini, argv[1]);
    Dict* kwargs = puf_ini_section(&ini, "env", 0);
    if (cnn_appearance_option(kwargs, "representation_mode", 1) != 1) return 2;
    double slots = puf_ini_get(&ini, "vec", "total_agents");
    if (!(slots >= 1 && slots <= 1048576 && floor(slots) == slots)) return 2;
    puts("slot,representation");
    for (unsigned int slot = 0; slot < (unsigned int)slots; slot++) {
        printf("%u,%d\n", slot, cnn_appearance_init(kwargs, slot, (int)count));
    }
    puf_ini_free(&ini);
    return 0;
}
