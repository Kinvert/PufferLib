#include <math.h>
#include <stdio.h>

#include "../dogfight.h"

static int closef(float left, float right) {
    return fabsf(left - right) < 1.0e-6f;
}

static int expect_pair(
        const char* label,
        const float actual[2],
        float expected_0,
        float expected_1) {
    if (closef(actual[0], expected_0)
            && closef(actual[1], expected_1)) {
        return 0;
    }
    fprintf(stderr,
        "%s: got [%.6f, %.6f], expected [%.6f, %.6f]\n",
        label, actual[0], actual[1], expected_0, expected_1);
    return 1;
}

int main(void) {
    int failures = 0;
    float published[2];

    const float legacy[2] = {0.6f, -0.2f};
    dogfight_two_agent_combine_dense_rewards(0, legacy, published);
    failures += expect_pair(
        "internal default parity", published, 0.6f, -0.2f);

    dogfight_two_agent_combine_dense_rewards(1, legacy, published);
    failures += expect_pair("legacy parity", published, 0.6f, -0.2f);

    dogfight_two_agent_combine_dense_rewards(2, legacy, published);
    failures += expect_pair(
        "competitive advantage", published, 0.4f, -0.4f);

    const float common_mode[2] = {0.35f, 0.35f};
    dogfight_two_agent_combine_dense_rewards(
        2, common_mode, published);
    failures += expect_pair(
        "common-mode cancellation", published, 0.0f, 0.0f);

    const float swapped[2] = {-0.2f, 0.6f};
    dogfight_two_agent_combine_dense_rewards(2, swapped, published);
    failures += expect_pair("seat swap", published, -0.4f, 0.4f);

    const float clipped[2] = {1.0f, -1.0f};
    dogfight_two_agent_combine_dense_rewards(2, clipped, published);
    failures += expect_pair("bounded advantage", published, 1.0f, -1.0f);

    if (failures != 0) {
        fprintf(stderr, "%d competitive reward tests failed\n", failures);
        return 1;
    }
    printf("two-agent competitive reward contract passed\n");
    return 0;
}
