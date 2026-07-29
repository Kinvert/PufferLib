#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../dogfight.h"

#define TOL 1.0e-5f

typedef struct {
    Env env;
    obs_t observations[2][OBS_SIZE];
    float actions[2][NUM_ATNS];
    float rewards[2];
    float terminals[2];
} DualEnv;

static int closef(float a, float b) {
    return fabsf(a - b) <= TOL;
}

static int plane_close(const Plane* a, const Plane* b) {
    return closef(a->pos.x, b->pos.x)
        && closef(a->pos.y, b->pos.y)
        && closef(a->pos.z, b->pos.z)
        && closef(a->vel.x, b->vel.x)
        && closef(a->vel.y, b->vel.y)
        && closef(a->vel.z, b->vel.z)
        && closef(a->omega.x, b->omega.x)
        && closef(a->omega.y, b->omega.y)
        && closef(a->omega.z, b->omega.z)
        && closef(a->ori.w, b->ori.w)
        && closef(a->ori.x, b->ori.x)
        && closef(a->ori.y, b->ori.y)
        && closef(a->ori.z, b->ori.z);
}

static void bind_agents(DualEnv* t) {
    for (int i = 0; i < 2; i++) {
        t->env.agents[i].observations = t->observations[i];
        t->env.agents[i].actions = t->actions[i];
        t->env.agents[i].rewards = &t->rewards[i];
        t->env.agents[i].terminals = &t->terminals[i];
        t->env.agents[i].action_mask = NULL;
        t->env.agents[i].policy = i;
    }
}

static void setup_dual(DualEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 2;
    t->env.max_steps = 300;
    t->env.configured_max_steps = 300;
    t->env.rng = 42U;
    t->env.global_step_stride = 1;
    bind_agents(t);
    dogfight_bind_rng(&t->env.rng);

    RewardConfig rcfg = {0};
    rcfg.speed_min = 50.0f;
    rcfg.low_altitude_threshold = 1200.0f;
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);
    t->env.two_agent_role_randomization = 1;
    t->env.two_agent_reward_version = 1;
    t->env.two_agent_bootstrap_steps = 100;
    t->env.two_agent_bootstrap_imitation_scale = 0.01f;
    set_curriculum_target(&t->env, 0.0f);
    puf_reset(&t->env);
}

static int test_puf_init_reads_bootstrap_steps(void) {
    Env env = {0};
    DictItem items[3] = {0};
    snprintf(items[0].key, sizeof(items[0].key), "%s", "num_agents");
    items[0].value = 2.0;
    snprintf(items[1].key, sizeof(items[1].key), "%s",
        "selfplay_bootstrap_steps");
    items[1].value = 1234.0;
    snprintf(items[2].key, sizeof(items[2].key), "%s",
        "selfplay_bootstrap_imitation_scale");
    items[2].value = 0.02;
    Dict kwargs = {
        .name = "env",
        .items = items,
        .size = 3,
        .cap = 3,
    };
    env.rng = 42U;

    puf_init(&env, &kwargs);
    if (env.two_agent_bootstrap_steps != 1234) {
        fprintf(stderr, "bootstrap steps were not read from env config\n");
        return 1;
    }
    if (!closef(env.two_agent_bootstrap_imitation_scale, 0.02f)) {
        fprintf(stderr, "bootstrap imitation scale was not read from config\n");
        return 1;
    }
    return 0;
}

static int test_bootstrap_scripts_opponent_and_masks_its_reward(void) {
    DualEnv t;
    setup_dual(&t);
    if (!t.env.two_agent_scripted_episode
            || t.env.two_agent_player_slot != 0) {
        fprintf(stderr, "bootstrap did not pin the trainable physical slot\n");
        return 1;
    }

    Plane teacher_self = t.env.player;
    Plane target_left = t.env.opponent;
    Plane target_right = t.env.opponent;
    target_left.pos = add3(teacher_self.pos, vec3(300.0f, 100.0f, 0.0f));
    target_right.pos = add3(teacher_self.pos, vec3(300.0f, -100.0f, 0.0f));
    float teacher_left[NUM_ATNS];
    float teacher_right[NUM_ATNS];
    dogfight_two_agent_pursuit_teacher_actions(
        &t.env, &teacher_self, &target_left, teacher_left);
    dogfight_two_agent_pursuit_teacher_actions(
        &t.env, &teacher_self, &target_right, teacher_right);
    if (teacher_left[2] * teacher_right[2] >= 0.0f
            || teacher_left[3] * teacher_right[3] >= 0.0f
            || !closef(teacher_left[2], -teacher_right[2])
            || !closef(teacher_left[3], -teacher_right[3])) {
        fprintf(stderr,
            "pursuit teacher is not mirror symmetric: "
            "ail=%.6f/%.6f rud=%.6f/%.6f\n",
            teacher_left[2], teacher_right[2],
            teacher_left[3], teacher_right[3]);
        return 1;
    }

    const float player_actions[NUM_ATNS] =
        {0.0f, -0.1f, 0.2f, 0.0f, -1.0f};
    const float ignored_actions[NUM_ATNS] =
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    memcpy(t.actions[0], player_actions, sizeof(player_actions));
    memcpy(t.actions[1], ignored_actions, sizeof(ignored_actions));

    float player_teacher[NUM_ATNS];
    float opponent_teacher[NUM_ATNS];
    dogfight_two_agent_pursuit_teacher_actions(
        &t.env, &t.env.player, &t.env.opponent, player_teacher);
    dogfight_two_agent_pursuit_teacher_actions(
        &t.env, &t.env.opponent, &t.env.player, opponent_teacher);
    float expected_player_reward =
        dogfight_two_agent_flight_school_reward(
            player_actions,
            player_teacher,
            t.env.two_agent_bootstrap_imitation_scale);
    float expected_opponent_reward =
        dogfight_two_agent_flight_school_reward(
            ignored_actions,
            opponent_teacher,
            t.env.two_agent_bootstrap_imitation_scale);

    puf_step(&t.env);

    if (t.terminals[0] != 1.0f || t.terminals[1] != 1.0f
            || !closef(t.rewards[0], expected_player_reward)
            || !closef(t.rewards[1], expected_opponent_reward)) {
        fprintf(stderr,
            "flight school terminal/reward mismatch: "
            "term=%.1f/%.1f reward=%.6f/%.6f expected=%.6f/%.6f\n",
            t.terminals[0], t.terminals[1],
            t.rewards[0], t.rewards[1],
            expected_player_reward, expected_opponent_reward);
        return 1;
    }
    return 0;
}

static int test_flight_school_isolates_aileron_direction(void) {
    const float teacher[NUM_ATNS] =
        {0.4f, -0.3f, 0.8f, 0.2f, -1.0f};
    const float exact[NUM_ATNS] =
        {0.4f, -0.3f, 0.8f, 0.2f, -1.0f};
    const float correct_aileron_only[NUM_ATNS] =
        {-1.0f, 1.0f, 0.8f, -1.0f, 1.0f};
    const float neutral_aileron[NUM_ATNS] =
        {0.4f, -0.3f, 0.0f, 0.2f, -1.0f};
    const float wrong_aileron[NUM_ATNS] =
        {0.4f, -0.3f, -0.8f, 0.2f, -1.0f};
    float exact_reward = dogfight_two_agent_flight_school_reward(
        exact, teacher, 1.0f);
    float correct_reward = dogfight_two_agent_flight_school_reward(
        correct_aileron_only, teacher, 1.0f);
    float neutral_reward = dogfight_two_agent_flight_school_reward(
        neutral_aileron, teacher, 1.0f);
    float wrong_reward = dogfight_two_agent_flight_school_reward(
        wrong_aileron, teacher, 1.0f);

    if (!closef(exact_reward, correct_reward)
            || !(correct_reward > neutral_reward)
            || !(neutral_reward > wrong_reward)) {
        fprintf(stderr,
            "flight school did not isolate aileron direction: "
            "exact=%.6f correct=%.6f neutral=%.6f wrong=%.6f\n",
            exact_reward, correct_reward, neutral_reward, wrong_reward);
        return 1;
    }
    return 0;
}

static int test_bootstrap_transitions_to_native_actions(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.global_step = t.env.two_agent_bootstrap_steps;
    t.env.two_agent_role_randomization = 0;
    puf_reset(&t.env);
    if (t.env.two_agent_scripted_episode) {
        fprintf(stderr, "bootstrap remained active at its step boundary\n");
        return 1;
    }

    const float player_actions[NUM_ATNS] =
        {0.0f, 0.0f, 0.0f, 0.0f, -1.0f};
    const float opponent_actions[NUM_ATNS] =
        {0.25f, -0.30f, 0.65f, 0.15f, -1.0f};
    memcpy(t.actions[0], player_actions, sizeof(player_actions));
    memcpy(t.actions[1], opponent_actions, sizeof(opponent_actions));
    Plane expected = t.env.opponent;
    step_plane_with_params(
        &expected, opponent_actions, DT, &t.env.flight_params);

    puf_step(&t.env);

    if (!plane_close(&t.env.opponent, &expected)
            || !closef(t.env.last_opp_actions[2], 0.65f)) {
        fprintf(stderr,
            "post-bootstrap episode did not restore native opponent actions\n");
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_puf_init_reads_bootstrap_steps();
    failures += test_bootstrap_scripts_opponent_and_masks_its_reward();
    failures += test_flight_school_isolates_aileron_direction();
    failures += test_bootstrap_transitions_to_native_actions();
    if (failures == 0) {
        puts("two-agent bootstrap: ok");
    }
    return failures;
}
