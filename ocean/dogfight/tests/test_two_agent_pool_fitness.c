#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../dogfight.h"

static void set_good_trace(Dogfight* env, int physical) {
    env->two_agent_pool_steps[physical] = 300;
    env->two_agent_pool_controlled_steps[physical] = 300;
    env->two_agent_pool_target_negative_steps[physical] = 100;
    env->two_agent_pool_target_positive_steps[physical] = 100;
    env->two_agent_pool_target_negative_aileron_sum[physical] = 30.0f;
    env->two_agent_pool_target_positive_aileron_sum[physical] = -30.0f;
    env->two_agent_pool_control_rate_sum[physical] = 90.0f;
    env->two_agent_roll_travel_radians[physical] =
        0.5f * 2.0f * (float)M_PI;
}

static void test_balanced_target_response_passes(void) {
    Dogfight* env = calloc(1, sizeof(*env));
    assert(env != NULL);
    set_good_trace(env, 0);
    assert(fabsf(dogfight_two_agent_pool_flight_quality(env, 0) - 1.0f)
        < 1e-6f);
    free(env);
}

static void test_constant_right_or_left_collapse_fails_symmetrically(void) {
    Dogfight* env = calloc(1, sizeof(*env));
    assert(env != NULL);
    for (int physical = 0; physical < 2; physical++) {
        env->two_agent_pool_steps[physical] = 300;
        env->two_agent_pool_controlled_steps[physical] = 300;
        env->two_agent_pool_target_negative_steps[physical] = 100;
        env->two_agent_pool_target_positive_steps[physical] = 100;
    }

    env->two_agent_pool_aileron_sum[0] = 240.0f;
    env->two_agent_pool_target_negative_aileron_sum[0] = 80.0f;
    env->two_agent_pool_target_positive_aileron_sum[0] = 80.0f;
    env->two_agent_pool_aileron_sum[1] = -240.0f;
    env->two_agent_pool_target_negative_aileron_sum[1] = -80.0f;
    env->two_agent_pool_target_positive_aileron_sum[1] = -80.0f;

    float right_quality =
        dogfight_two_agent_pool_flight_quality(env, 0);
    float left_quality =
        dogfight_two_agent_pool_flight_quality(env, 1);
    assert(right_quality == 0.0f);
    assert(left_quality == 0.0f);
    assert(right_quality == left_quality);
    free(env);
}

static void test_roll_and_control_state_are_independent_gates(void) {
    Dogfight* env = calloc(1, sizeof(*env));
    assert(env != NULL);

    set_good_trace(env, 0);
    env->two_agent_pool_excessive_roll_steps[0] = 180;
    assert(dogfight_two_agent_pool_flight_quality(env, 0) == 0.0f);

    set_good_trace(env, 1);
    env->two_agent_pool_controlled_steps[1] = 100;
    assert(dogfight_two_agent_pool_flight_quality(env, 1) == 0.0f);

    set_good_trace(env, 0);
    env->two_agent_pool_excessive_roll_steps[0] = 0;
    env->two_agent_roll_travel_radians[0] =
        2.0f * 2.0f * (float)M_PI;
    assert(dogfight_two_agent_pool_flight_quality(env, 0) == 0.0f);
    free(env);
}

static void test_control_chatter_gate_relaxes_with_stage(void) {
    Dogfight* env = calloc(1, sizeof(*env));
    assert(env != NULL);

    set_good_trace(env, 0);
    env->two_agent_pool_control_rate_sum[0] = 270.0f;
    env->stage = 0;
    assert(dogfight_two_agent_pool_flight_quality(env, 0) == 0.0f);

    env->stage = 10;
    assert(dogfight_two_agent_pool_flight_quality(env, 0) > 0.0f);
    free(env);
}

static void test_short_decisive_episode_is_not_rejected_by_lateral_gate(void) {
    Dogfight* env = calloc(1, sizeof(*env));
    assert(env != NULL);
    env->two_agent_pool_steps[0] = 10;
    env->two_agent_pool_controlled_steps[0] = 10;
    env->two_agent_pool_aileron_sum[0] = 10.0f;
    assert(dogfight_two_agent_pool_flight_quality(env, 0) == 1.0f);
    free(env);
}

static void test_adjusted_outcome_contract(void) {
    float qualities[2] = {1.0f, 1.0f};
    float scores[2];

    dogfight_two_agent_adjusted_pool_scores(
        DEATH_KILL, 1, qualities, scores);
    assert(scores[0] == 1.0f && scores[1] == 0.0f);

    dogfight_two_agent_adjusted_pool_scores(
        DEATH_OOB, 1, qualities, scores);
    assert(scores[0] == DOGFIGHT_POOL_GROUND_WIN_CREDIT);
    assert(scores[1] == 0.0f);

    dogfight_two_agent_adjusted_pool_scores(
        DEATH_TIMEOUT, 0, qualities, scores);
    assert(scores[0] == 0.0f && scores[1] == 0.0f);

    float collapsed[2] = {0.0f, 0.0f};
    dogfight_two_agent_adjusted_pool_scores(
        DEATH_TIMEOUT, 0, collapsed, scores);
    assert(scores[0] == 0.0f && scores[1] == 0.0f);
    assert(scores[0] + scores[1] < 1.0f);

    dogfight_two_agent_adjusted_pool_scores(
        DEATH_KILL, 0, qualities, scores);
    assert(scores[0] == 0.5f && scores[1] == 0.5f);

    dogfight_two_agent_adjusted_pool_scores(
        DEATH_OOB, 0, qualities, scores);
    assert(scores[0] == 0.0f && scores[1] == 0.0f);
}

static void test_raw_outcomes_remain_robocode_constant_sum(void) {
    float scores[2];

    dogfight_two_agent_raw_pool_scores(1, scores);
    assert(scores[0] == 1.0f && scores[1] == 0.0f);
    assert(scores[0] + scores[1] == 1.0f);

    dogfight_two_agent_raw_pool_scores(-1, scores);
    assert(scores[0] == 0.0f && scores[1] == 1.0f);
    assert(scores[0] + scores[1] == 1.0f);

    dogfight_two_agent_raw_pool_scores(0, scores);
    assert(scores[0] == 0.5f && scores[1] == 0.5f);
    assert(scores[0] + scores[1] == 1.0f);
}

int main(void) {
    test_balanced_target_response_passes();
    test_constant_right_or_left_collapse_fails_symmetrically();
    test_roll_and_control_state_are_independent_gates();
    test_control_chatter_gate_relaxes_with_stage();
    test_short_decisive_episode_is_not_rejected_by_lateral_gate();
    test_adjusted_outcome_contract();
    test_raw_outcomes_remain_robocode_constant_sum();
    puts("two-agent pool fitness tests passed");
    return 0;
}
