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

static Plane posed_plane(
        float x,
        float y,
        float yaw_degrees,
        float bank_degrees,
        float roll_rate) {
    Plane plane = level_plane(x);
    plane.pos.y = y;
    Quat yaw = quat_from_axis_angle(
        vec3(0.0f, 0.0f, 1.0f), yaw_degrees * DEG_TO_RAD);
    Quat bank = quat_from_axis_angle(
        vec3(1.0f, 0.0f, 0.0f), bank_degrees * DEG_TO_RAD);
    plane.ori = quat_mul(yaw, bank);
    plane.omega.x = roll_rate;
    return plane;
}

static int test_magnitude_penalty(void) {
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
    return 0;
}

static int test_mirrored_steering_alignment(void) {
    const float scale = 0.02f;
    float left_correct = dogfight_two_agent_steering_alignment_reward(
        0.25f, -1.0f, scale);
    float left_wrong = dogfight_two_agent_steering_alignment_reward(
        0.25f, 1.0f, scale);
    float right_correct = dogfight_two_agent_steering_alignment_reward(
        -0.25f, 1.0f, scale);
    float right_wrong = dogfight_two_agent_steering_alignment_reward(
        -0.25f, -1.0f, scale);
    float neutral = dogfight_two_agent_steering_alignment_reward(
        0.25f, 0.0f, scale);

    if (fabsf(left_correct - right_correct) > 1.0e-6f
            || fabsf(left_wrong - right_wrong) > 1.0e-6f
            || left_correct <= neutral
            || neutral <= left_wrong
            || fabsf(neutral) > 1.0e-6f) {
        fprintf(stderr,
            "mirrored steering alignment failed: "
            "left=(%.6f, %.6f) right=(%.6f, %.6f) neutral=%.6f\n",
            left_correct,
            left_wrong,
            right_correct,
            right_wrong,
            neutral);
        return 1;
    }
    return 0;
}

static int test_steering_alignment_decay(void) {
    const float configured = 0.20f;
    float initial = dogfight_two_agent_steering_scale_for_stage(configured, 0);
    float middle = dogfight_two_agent_steering_scale_for_stage(configured, 5);
    float final = dogfight_two_agent_steering_scale_for_stage(configured, 10);
    if (fabsf(initial - 0.20f) > 1.0e-6f
            || fabsf(middle - 0.10f) > 1.0e-6f
            || fabsf(final) > 1.0e-6f) {
        fprintf(stderr,
            "steering alignment decay failed: %.6f %.6f %.6f\n",
            initial,
            middle,
            final);
        return 1;
    }
    return 0;
}

static int test_independent_steering_publication(void) {
    float left_reward = dogfight_two_agent_publish_steering_reward(
        0.0f, 0.25f, -1.0f, 0.20f);
    float right_reward = dogfight_two_agent_publish_steering_reward(
        0.0f, -0.25f, 1.0f, 0.20f);
    if (left_reward <= 0.0f || right_reward <= 0.0f) {
        fprintf(stderr,
            "independent steering publication failed: %.6f %.6f\n",
            left_reward,
            right_reward);
        return 1;
    }
    return 0;
}

static int test_roll_travel_accumulates_absolute_rotation(void) {
    float positive = 0.0f;
    float negative = 0.0f;
    for (int step = 0; step < 100; step++) {
        positive = dogfight_two_agent_update_roll_travel(
            positive, 2.0f, 0.02f);
        negative = dogfight_two_agent_update_roll_travel(
            negative, -2.0f, 0.02f);
    }
    if (fabsf(positive - 4.0f) > 1.0e-5f
            || fabsf(negative - 4.0f) > 1.0e-5f) {
        fprintf(stderr,
            "absolute roll travel integration failed: %.6f %.6f\n",
            positive,
            negative);
        return 1;
    }
    float reversed = dogfight_two_agent_update_roll_travel(
        positive, -2.0f, 0.02f);
    float arrested = dogfight_two_agent_update_roll_travel(
        positive, 0.0f, 0.02f);
    if (fabsf(reversed - 4.04f) > 1.0e-5f
            || fabsf(arrested - 4.0f) > 1.0e-5f) {
        fprintf(stderr,
            "roll travel was cleared by chatter: reverse=%.6f arrest=%.6f\n",
            reversed,
            arrested);
        return 1;
    }
    return 0;
}

static float roll_penalty(
        float rotations, int stage, long step, float scale) {
    return dogfight_two_agent_roll_discipline_penalty(
        rotations * 2.0f * (float)M_PI,
        scale,
        1.0f,
        2.0f,
        stage,
        step,
        80000000,
        300000000,
        0.50f);
}

static int test_roll_discipline_budget_and_symmetry(void) {
    float free_positive = roll_penalty(1.0f, 0, 0, 0.03f);
    float halfway = roll_penalty(1.5f, 0, 0, 0.03f);
    float full_positive = roll_penalty(2.0f, 0, 0, 0.03f);
    if (fabsf(free_positive) > 1.0e-6f
            || fabsf(halfway + 0.0075f) > 1.0e-6f
            || fabsf(full_positive + 0.03f) > 1.0e-6f) {
        fprintf(stderr,
            "roll discipline budget failed: %.6f %.6f %.6f\n",
            free_positive,
            halfway,
            full_positive);
        return 1;
    }
    return 0;
}

static int test_roll_discipline_stage_and_time_decay(void) {
    float stage_0 = roll_penalty(2.0f, 0, 0, 0.03f);
    float stage_3 = roll_penalty(2.0f, 3, 0, 0.03f);
    float stage_5 = roll_penalty(2.0f, 5, 0, 0.03f);
    float stage_6 = roll_penalty(2.0f, 6, 0, 0.03f);
    float late = roll_penalty(2.0f, 0, 300000000, 0.03f);
    if (fabsf(stage_3 - 0.75f * stage_0) > 1.0e-6f
            || fabsf(stage_5 - 0.25f * stage_0) > 1.0e-6f
            || fabsf(stage_6) > 1.0e-6f
            || fabsf(late - 0.50f * stage_0) > 1.0e-6f) {
        fprintf(stderr,
            "roll discipline schedule failed: %.6f %.6f %.6f %.6f %.6f\n",
            stage_0,
            stage_3,
            stage_5,
            stage_6,
            late);
        return 1;
    }
    return 0;
}

static int test_roll_discipline_does_not_cancel_common_mode(void) {
    Dogfight env;
    memset(&env, 0, sizeof(env));
    env.native_roll_discipline_scale = 0.03f;
    env.native_roll_free_rotations = 1.0f;
    env.native_roll_full_rotations = 2.0f;
    env.native_roll_decay_start = 80000000;
    env.native_roll_decay_end = 300000000;
    env.native_roll_final_fraction = 0.50f;
    env.stage = 0;
    float rotations = 2.0f * 2.0f * (float)M_PI;
    float right = dogfight_two_agent_publish_roll_discipline(
        0.0f, rotations, &env);
    float left = dogfight_two_agent_publish_roll_discipline(
        0.0f, rotations, &env);
    if (right >= 0.0f || left >= 0.0f
            || fabsf(right - left) > 1.0e-6f) {
        fprintf(stderr,
            "common-mode roll penalty canceled: %.6f %.6f\n",
            right,
            left);
        return 1;
    }
    return 0;
}

static int test_bank_guidance_sign_and_mirror_contract(void) {
    Plane right = posed_plane(0.0f, 0.0f, 0.0f, 30.0f, 0.0f);
    Plane left = posed_plane(0.0f, 0.0f, 0.0f, -30.0f, 0.0f);
    float right_bank = dogfight_two_agent_signed_bank(&right);
    float left_bank = dogfight_two_agent_signed_bank(&left);
    if (right_bank <= 0.0f
            || left_bank >= 0.0f
            || fabsf(right_bank + left_bank) > 1.0e-6f) {
        fprintf(stderr,
            "signed bank convention failed: right=%.6f left=%.6f\n",
            right_bank,
            left_bank);
        return 1;
    }

    Plane origin = level_plane(0.0f);
    Plane target_left = posed_plane(100.0f, 100.0f, 0.0f, 0.0f, 0.0f);
    Plane target_right = posed_plane(100.0f, -100.0f, 0.0f, 0.0f, 0.0f);
    Plane bank_left = posed_plane(0.0f, 0.0f, 0.0f, -45.0f, 0.0f);
    Plane bank_right = posed_plane(0.0f, 0.0f, 0.0f, 45.0f, 0.0f);
    float left_reward = dogfight_two_agent_bank_guidance_penalty(
        &bank_left, &target_left, 0.001f, 0.0005f);
    float right_reward = dogfight_two_agent_bank_guidance_penalty(
        &bank_right, &target_right, 0.001f, 0.0005f);
    float level_left = dogfight_two_agent_bank_guidance_penalty(
        &origin, &target_left, 0.001f, 0.0005f);
    if (fabsf(left_reward - right_reward) > 1.0e-6f
            || fabsf(left_reward) > 1.0e-6f
            || level_left >= left_reward) {
        fprintf(stderr,
            "bank guidance mirror/alignment failed: %.6f %.6f %.6f\n",
            left_reward,
            right_reward,
            level_left);
        return 1;
    }
    return 0;
}

static int test_bank_guidance_damps_and_prefers_corrective_roll(void) {
    Plane target = posed_plane(100.0f, 100.0f, 0.0f, 0.0f, 0.0f);
    Plane aligned = posed_plane(0.0f, 0.0f, 0.0f, -45.0f, 0.0f);
    Plane continuing = posed_plane(0.0f, 0.0f, 0.0f, -45.0f, -1.0f);
    Plane level_toward = posed_plane(0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
    Plane level_away = posed_plane(0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    Plane wrong = posed_plane(0.0f, 0.0f, 0.0f, 45.0f, 0.0f);
    float aligned_reward = dogfight_two_agent_bank_guidance_penalty(
        &aligned, &target, 0.001f, 0.0005f);
    float continuing_reward = dogfight_two_agent_bank_guidance_penalty(
        &continuing, &target, 0.001f, 0.0005f);
    float toward_reward = dogfight_two_agent_bank_guidance_penalty(
        &level_toward, &target, 0.001f, 0.0005f);
    float away_reward = dogfight_two_agent_bank_guidance_penalty(
        &level_away, &target, 0.001f, 0.0005f);
    float wrong_reward = dogfight_two_agent_bank_guidance_penalty(
        &wrong, &target, 0.001f, 0.0005f);
    if (aligned_reward <= continuing_reward
            || toward_reward <= away_reward
            || aligned_reward <= toward_reward
            || toward_reward <= wrong_reward) {
        fprintf(stderr,
            "bank guidance preference failed: "
            "aligned=%.6f continuing=%.6f toward=%.6f "
            "away=%.6f wrong=%.6f\n",
            aligned_reward,
            continuing_reward,
            toward_reward,
            away_reward,
            wrong_reward);
        return 1;
    }
    return 0;
}

static int test_bank_guidance_is_independent_and_stage_gated(void) {
    Dogfight env;
    memset(&env, 0, sizeof(env));
    env.native_bank_guidance_scale = 0.001f;
    env.native_roll_guidance_scale = 0.0005f;
    env.native_roll_decay_start = 80000000;
    env.native_roll_decay_end = 300000000;
    env.native_roll_final_fraction = 0.50f;
    env.stage = 0;

    Plane first = posed_plane(0.0f, 0.0f, 0.0f, 45.0f, 1.0f);
    Plane second = posed_plane(
        100.0f, 100.0f, 180.0f, 45.0f, 1.0f);
    float first_reward = dogfight_two_agent_publish_bank_guidance(
        0.0f, &first, &second, &env);
    float second_reward = dogfight_two_agent_publish_bank_guidance(
        0.0f, &second, &first, &env);
    if (first_reward >= 0.0f || second_reward >= 0.0f) {
        fprintf(stderr,
            "common-mode bank guidance canceled: %.6f %.6f\n",
            first_reward,
            second_reward);
        return 1;
    }

    env.stage = 6;
    float unguided = dogfight_two_agent_publish_bank_guidance(
        0.25f, &first, &second, &env);
    if (fabsf(unguided - 0.25f) > 1.0e-6f) {
        fprintf(stderr,
            "late-stage bank guidance did not turn off: %.6f\n",
            unguided);
        return 1;
    }
    return 0;
}

static int test_native_acquisition_contract(void) {
    Dogfight env;
    memset(&env, 0, sizeof(env));
    env.native_spawn_curriculum = 1;
    env.native_acquisition_steps = 64;
    env.native_acquisition_rehearsal_cycle_steps = 32;
    env.native_acquisition_rehearsal_steps = 4;
    env.global_step = 63;
    if (!dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native acquisition ended before configured step\n");
        return 1;
    }
    env.global_step = 64;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native acquisition continued into competitive phase\n");
        return 1;
    }
    env.global_step = 91;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal began before configured window\n");
        return 1;
    }
    env.global_step = 92;
    if (!dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal did not begin at configured window\n");
        return 1;
    }
    env.global_step = 96;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr, "native rehearsal crossed cycle boundary\n");
        return 1;
    }
    env.global_step = 0;
    env.agents[0].policy = 0;
    env.agents[1].policy = 1;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr,
            "explicit policy match was incorrectly scored as acquisition\n");
        return 1;
    }
    env.agents[1].policy = 0;
    if (!dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr,
            "current-policy training unexpectedly skipped acquisition\n");
        return 1;
    }
    env.native_acquisition_steps = 0;
    env.native_acquisition_rehearsal_steps = 0;
    if (dogfight_two_agent_in_native_acquisition(&env)) {
        fprintf(stderr,
            "disabled acquisition intercepted current-policy self-play\n");
        return 1;
    }

    float left_actions[NUM_ATNS] = {0};
    float right_actions[NUM_ATNS] = {0};
    float left_wrong_actions[NUM_ATNS] = {0};
    float right_wrong_actions[NUM_ATNS] = {0};
    left_actions[2] = -1.0f;
    right_actions[2] = 1.0f;
    left_wrong_actions[2] = 1.0f;
    right_wrong_actions[2] = -1.0f;
    float left_correct = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.1f);
    float left_wrong = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_wrong_actions, 1.0f, 0.1f);
    float right_correct = dogfight_two_agent_native_acquisition_reward(
        -0.25f, right_actions, 1.0f, 0.1f);
    float right_wrong = dogfight_two_agent_native_acquisition_reward(
        -0.25f, right_wrong_actions, 1.0f, 0.1f);
    if (fabsf(left_correct - right_correct) > 1.0e-6f
            || fabsf(left_wrong - right_wrong) > 1.0e-6f
            || left_correct < 0.999f
            || left_wrong > -0.999f) {
        fprintf(stderr,
            "native acquisition reward is not mirrored: %.6f %.6f %.6f %.6f\n",
            left_correct,
            left_wrong,
            right_correct,
            right_wrong);
        return 1;
    }

    left_actions[0] = 1.0f;
    left_actions[1] = -1.0f;
    left_actions[3] = 1.0f;
    float unstable = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.1f);
    if (fabsf(unstable - 0.7f) > 1.0e-6f) {
        fprintf(stderr,
            "native acquisition did not penalize unstable controls: %.6f\n",
            unstable);
        return 1;
    }
    float rehearsal = dogfight_two_agent_native_acquisition_reward(
        0.25f, left_actions, 1.0f, 0.0f);
    if (rehearsal < 0.999f) {
        fprintf(stderr,
            "native rehearsal constrained non-aileron controls: %.6f\n",
            rehearsal);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_magnitude_penalty();
    failures += test_mirrored_steering_alignment();
    failures += test_steering_alignment_decay();
    failures += test_independent_steering_publication();
    failures += test_roll_travel_accumulates_absolute_rotation();
    failures += test_roll_discipline_budget_and_symmetry();
    failures += test_roll_discipline_stage_and_time_decay();
    failures += test_roll_discipline_does_not_cancel_common_mode();
    failures += test_bank_guidance_sign_and_mirror_contract();
    failures += test_bank_guidance_damps_and_prefers_corrective_roll();
    failures += test_bank_guidance_is_independent_and_stage_gated();
    failures += test_native_acquisition_contract();
    if (failures == 0) {
        printf("two-agent aileron reward contract passed\n");
    }
    return failures;
}
