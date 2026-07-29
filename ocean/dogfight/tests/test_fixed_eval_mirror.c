#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../dogfight.h"

#define TOL 1.0e-5f

typedef struct {
    Dogfight env;
    float observations[OBS_SIZE];
} FixedEnv;

static int closef(float a, float b) {
    return fabsf(a - b) <= TOL;
}

static void setup_fixed(
        FixedEnv* test, unsigned int seed, int stage, int mirror) {
    memset(test, 0, sizeof(*test));
    RewardConfig reward_config = {0};
    reward_config.speed_min = 50.0f;
    reward_config.low_altitude_threshold = 1200.0f;

    test->env.rng = seed;
    dogfight_bind_rng(&test->env.rng);
    init(&test->env, OBS_OPPONENT_AWARE, &reward_config, 1, 0, seed);
    test->env.observations = test->observations;
    test->env.configured_max_steps = 300;
    test->env.local_curriculum.enabled = 1;
    test->env.local_curriculum.fixed_stage = stage;
    test->env.eval_lateral_mirror = mirror;
    set_curriculum_target(&test->env, (float)stage);
    dogfight_bind_rng(&test->env.rng);
    c_reset(&test->env);
}

static int mirrored_vec(Vec3 original, Vec3 reflected) {
    return closef(original.x, reflected.x)
        && closef(-original.y, reflected.y)
        && closef(original.z, reflected.z);
}

static int mirrored_quat(Quat original, Quat reflected) {
    return closef(original.w, reflected.w)
        && closef(-original.x, reflected.x)
        && closef(original.y, reflected.y)
        && closef(-original.z, reflected.z);
}

static int mirrored_plane(const Plane* original, const Plane* reflected) {
    return mirrored_vec(original->pos, reflected->pos)
        && mirrored_vec(original->vel, reflected->vel)
        && mirrored_vec(original->prev_vel, reflected->prev_vel)
        && mirrored_quat(original->ori, reflected->ori)
        && closef(-original->omega.x, reflected->omega.x)
        && closef(original->omega.y, reflected->omega.y)
        && closef(-original->omega.z, reflected->omega.z);
}

static AutopilotMode mirrored_mode(AutopilotMode mode) {
    if (mode == AP_TURN_LEFT) return AP_TURN_RIGHT;
    if (mode == AP_TURN_RIGHT) return AP_TURN_LEFT;
    if (mode == AP_HARD_TURN_LEFT) return AP_HARD_TURN_RIGHT;
    if (mode == AP_HARD_TURN_RIGHT) return AP_HARD_TURN_LEFT;
    return mode;
}

static int mirrored_observations(
        const float original[OBS_SIZE], const float reflected[OBS_SIZE]) {
    static const float signs[OBS_SIZE] = {
        1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
        -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1,
    };
    for (int i = 0; i < OBS_SIZE; i++) {
        if (!closef(signs[i] * original[i], reflected[i])) {
            fprintf(stderr,
                "stage observation %d mismatch: %.6f -> %.6f\n",
                i, original[i], reflected[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    for (int stage = 0; stage <= 10; stage++) {
        FixedEnv original;
        FixedEnv reflected;
        setup_fixed(&original, 0xC0FFEEU + (unsigned int)stage, stage, 0);
        setup_fixed(&reflected, 0xC0FFEEU + (unsigned int)stage, stage, 1);

        if (original.env.rng != reflected.env.rng
                || original.env.stage != reflected.env.stage
                || original.env.max_steps != reflected.env.max_steps
                || !closef(original.env.player.throttle,
                    reflected.env.player.throttle)
                || !closef(original.env.opponent_ap.throttle,
                    reflected.env.opponent_ap.throttle)
                || !mirrored_plane(&original.env.player, &reflected.env.player)
                || !mirrored_plane(
                    &original.env.opponent, &reflected.env.opponent)
                || mirrored_mode(original.env.opponent_ap.mode)
                    != reflected.env.opponent_ap.mode
                || !mirrored_observations(
                    original.observations, reflected.observations)) {
            fprintf(stderr, "fixed eval mirror failed at stage %d\n", stage);
            return 1;
        }
    }
    return 0;
}
