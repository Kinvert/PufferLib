#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../dogfight.h"

static Plane level_plane(float x) {
    Plane plane;
    memset(&plane, 0, sizeof(plane));
    plane.pos = vec3(x, 0.0f, 2000.0f);
    plane.vel = vec3(80.0f, 0.0f, 0.0f);
    plane.ori.w = 1.0f;
    plane.g_force = 1.0f;
    return plane;
}

int main(void) {
    Dogfight neutral_env;
    Dogfight deflected_env;
    memset(&neutral_env, 0, sizeof(neutral_env));
    memset(&deflected_env, 0, sizeof(deflected_env));
    neutral_env.rcfg.aileron_magnitude_penalty = 0.01f;
    deflected_env.rcfg.aileron_magnitude_penalty = 0.01f;

    Plane self = level_plane(0.0f);
    Plane other = level_plane(100.0f);
    float neutral_actions[NUM_ATNS] = {0};
    float deflected_actions[NUM_ATNS] = {0};
    deflected_actions[2] = 0.75f;

    float neutral = dogfight_two_agent_dense_reward(
        &neutral_env, &self, &other, neutral_actions, 0, 0.0f, 0.0f);
    float deflected = dogfight_two_agent_dense_reward(
        &deflected_env, &self, &other, deflected_actions, 0, 0.0f, 0.0f);
    float expected_delta = 0.75f * 0.01f;
    float actual_delta = neutral - deflected;

    if (fabsf(actual_delta - expected_delta) > 1.0e-6f) {
        fprintf(stderr,
            "aileron penalty delta %.8f, expected %.8f\n",
            actual_delta, expected_delta);
        return 1;
    }
    printf("two-agent aileron magnitude penalty passed\n");
    return 0;
}
