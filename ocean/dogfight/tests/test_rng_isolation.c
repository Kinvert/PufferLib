#include <stdio.h>
#include <stdlib.h>

#include "../flightlib.h"

int main(void) {
    unsigned int actual_a = 17;
    unsigned int actual_b = 29;
    unsigned int expected_a = actual_a;
    unsigned int expected_b = actual_b;

    unsigned int expected_a1 = (unsigned int)rand_r(&expected_a);
    unsigned int expected_a2 = (unsigned int)rand_r(&expected_a);
    unsigned int expected_b1 = (unsigned int)rand_r(&expected_b);
    unsigned int expected_b2 = (unsigned int)rand_r(&expected_b);

    dogfight_bind_rng(&actual_a);
    unsigned int actual_a1 = dogfight_rand();
    dogfight_bind_rng(&actual_b);
    unsigned int actual_b1 = dogfight_rand();
    dogfight_bind_rng(&actual_a);
    unsigned int actual_a2 = dogfight_rand();
    dogfight_bind_rng(&actual_b);
    unsigned int actual_b2 = dogfight_rand();
    dogfight_bind_rng(NULL);

    if (actual_a1 != expected_a1 || actual_a2 != expected_a2 ||
            actual_b1 != expected_b1 || actual_b2 != expected_b2) {
        fprintf(stderr, "interleaved per-environment RNG streams diverged\n");
        return 1;
    }

    puts("per-environment RNG isolation: ok");
    return 0;
}
