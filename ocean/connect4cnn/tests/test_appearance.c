// Native assignment receipt and invariants; no GPU, model or game RNG needed.
#include <assert.h>
typedef float obs_t;
#include "pufferenv.h"
#include "ocean/connect4cnn/appearance.h"

int main(int argc, char** argv) {
    Dict kwargs = {0};
    dict_set(&kwargs, "representation_mode", 1);
    dict_set(&kwargs, "representation_seed", 12345);
    const int expected[] = {0,0,7,8,6,4,4,4,6,4,7,3,8,8,1,5};
    for (int slot = 15; slot >= 0; slot--) {
        assert(cnn_appearance_init(&kwargs, slot, 10) == expected[slot]);
        assert(cnn_appearance_init(&kwargs, slot, 5) == expected[slot] % 5);
    }
    int counts[10] = {0}, changed = 0;
    for (unsigned int slot = 0; slot < 4096; slot++) {
        dict_set(&kwargs, "representation_seed", 12345);
        int first = cnn_appearance_init(&kwargs, slot, 10);
        counts[first]++;
        // Interleaved calls, iteration order and seed changes cannot alter it.
        dict_set(&kwargs, "representation_seed", 12346);
        changed += first != cnn_appearance_init(&kwargs, slot, 10);
        dict_set(&kwargs, "representation_seed", 12345);
        assert(first == cnn_appearance_init(&kwargs, slot, 10));
    }
    assert(changed > 0);
    for (int i = 0; i < 10; i++) assert(counts[i] > 0);
    dict_set(&kwargs, "representation_mode", 0);
    for (int id = 0; id < 10; id++) {
        dict_set(&kwargs, "representation", id);
        assert(cnn_appearance_init(&kwargs, 999, 10) == id);
    }
    if (argc == 4) {
        // Usage: test_appearance connect4cnn|pongcnn seed slots
        int count = strcmp(argv[1], "connect4cnn") == 0 ? 10
            : strcmp(argv[1], "pongcnn") == 0 ? 5 : 0;
        double slots = strtod(argv[3], NULL);
        if (!count || !(slots >= 1 && slots <= 1000000 && slots == floor(slots))) return 2;
        dict_set(&kwargs, "representation", 0);
        dict_set(&kwargs, "representation_mode", 1);
        dict_set(&kwargs, "representation_seed", strtod(argv[2], NULL));
        puts("slot,representation");
        for (unsigned int slot = 0; slot < (unsigned int)slots; slot++) {
            printf("%u,%d\n", slot, cnn_appearance_init(&kwargs, slot, count));
        }
    } else if (argc != 1) return 2;
    free(kwargs.items);
    return 0;
}
