#include <stdint.h>
#include <stdio.h>

#include "../curriculum_mix.h"

static int test_disabled_mix_uses_primary_stage(void) {
    for (uint32_t identity = 0; identity < 128; identity++) {
        if (dogfight_select_fixed_stage(identity, 20, -1, 8) != 20) {
            fprintf(stderr, "disabled rehearsal stage changed assignment\n");
            return 1;
        }
        if (dogfight_select_fixed_stage(identity, 20, 11, 0) != 20) {
            fprintf(stderr, "disabled rehearsal stride changed assignment\n");
            return 1;
        }
    }
    return 0;
}

static int test_stride_eight_distribution(void) {
    int primary_count = 0;
    int rehearsal_count = 0;
    for (uint32_t identity = 0; identity < 4096; identity++) {
        int stage = dogfight_select_fixed_stage(identity, 20, 11, 8);
        if (stage == 20) {
            primary_count++;
        } else if (stage == 11) {
            rehearsal_count++;
        } else {
            fprintf(stderr, "stage mix selected unexpected stage %d\n", stage);
            return 1;
        }
        if (stage != dogfight_select_fixed_stage(identity, 20, 11, 8)) {
            fprintf(stderr, "stage mix is not deterministic\n");
            return 1;
        }
    }

    if (primary_count != 3584 || rehearsal_count != 512) {
        fprintf(
            stderr,
            "stride-eight distribution drifted: primary=%d rehearsal=%d\n",
            primary_count,
            rehearsal_count
        );
        return 1;
    }
    return 0;
}

static int test_stride_one_uses_only_rehearsal(void) {
    for (uint32_t identity = 0; identity < 128; identity++) {
        if (dogfight_select_fixed_stage(identity, 20, 11, 1) != 11) {
            fprintf(stderr, "stride one must select rehearsal stage\n");
            return 1;
        }
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_disabled_mix_uses_primary_stage();
    failures += test_stride_eight_distribution();
    failures += test_stride_one_uses_only_rehearsal();
    if (failures == 0) {
        printf("curriculum stage mix: PASS\n");
    }
    return failures;
}
