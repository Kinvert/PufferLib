// Preserve the spawn sequence across the while-loop rewrite, including rejection.
#include <assert.h>
#include "../sprite_forage.h"

int main(void) {
    for (int count = 1; count <= FORAGE_AREA; count++) {
        int threshold = (0u - count) % count;
        unsigned int candidates[] = {0, 1, threshold ? threshold - 1 : 0,
            threshold, 0xffffffffu};
        for (int index = 0; index < 5; index++) {
            Env env = {0};
            env.player_idx = -1;
            memset(env.grid, FORAGE_TREE, sizeof(env.grid));
            memset(env.grid, FORAGE_EMPTY, count);
            // Inverse of1664525 modulo2^32: force the next candidate exactly.
            env.rng = (candidates[index] - 1013904223u) * 4276115653u;
            assert(env.rng * 1664525u + 1013904223u == candidates[index]);
            unsigned int expected = candidates[index];
            while (expected < threshold) {
                expected = expected * 1664525u + 1013904223u;
            }
            int cell = forage_random_empty_idx(&env);
            assert(env.rng == expected && cell == expected % count);
        }
    }
    puts("Spawn RNG: all300 free-cell counts, 1500 forced rejection/wrap boundaries PASS");
}
