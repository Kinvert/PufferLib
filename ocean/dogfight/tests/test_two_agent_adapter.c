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

static void bind_dual_agents(DualEnv* t) {
    for (int i = 0; i < 2; i++) {
        t->env.agents[i].observations = t->observations[i];
        t->env.agents[i].actions = t->actions[i];
        t->env.agents[i].rewards = &t->rewards[i];
        t->env.agents[i].terminals = &t->terminals[i];
        t->env.agents[i].action_mask = NULL;
        t->env.agents[i].policy = 0;
    }
}

static void setup_dual(DualEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 2;
    t->env.max_steps = 3000;
    t->env.configured_max_steps = 3000;
    t->env.rng = 42;
    bind_dual_agents(t);
    dogfight_bind_rng(&t->env.rng);

    RewardConfig rcfg = {0};
    rcfg.speed_min = 50.0f;
    rcfg.low_altitude_threshold = 1200.0f;
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 0, 0, 0);
    t->env.use_opponent_override = 1;
    puf_reset(&t->env);
}

static int test_init_declares_robocode_policy_rows(void) {
    Env env = {0};
    DictItem num_agents = {0};
    snprintf(num_agents.key, sizeof(num_agents.key), "%s", "num_agents");
    num_agents.value = 2.0;
    Dict kwargs = {
        .name = "env",
        .items = &num_agents,
        .size = 1,
        .cap = 1,
    };
    env.rng = 42;

    puf_init(&env, &kwargs);

    if (env.num_agents != 2
            || env.agents[0].policy != 0
            || env.agents[1].policy != 1) {
        fprintf(stderr,
            "native policy rows mismatch: agents=%d policies=%d/%d\n",
            env.num_agents, env.agents[0].policy, env.agents[1].policy);
        return 1;
    }
    if (env.agents[0].observations != NULL
            || env.agents[1].observations != NULL) {
        fprintf(stderr, "puf_init touched agent buffers before vector binding\n");
        return 1;
    }
    return 0;
}

static int test_reset_observations_are_both_fresh(void) {
    DualEnv t;
    setup_dual(&t);

    float expected_0[OBS_SIZE] = {0};
    float expected_1[OBS_SIZE] = {0};
    compute_obs_opponent_aware_for_plane(
        &t.env, &t.env.player, &t.env.opponent, expected_0);
    compute_obs_opponent_aware_for_plane(
        &t.env, &t.env.opponent, &t.env.player, expected_1);

    for (int i = 0; i < OBS_SIZE; i++) {
        if (!isfinite(t.observations[0][i])
                || !isfinite(t.observations[1][i])
                || !closef(t.observations[0][i], expected_0[i])
                || !closef(t.observations[1][i], expected_1[i])) {
            fprintf(stderr, "reset observation mismatch at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int test_swapped_planes_swap_observations(void) {
    DualEnv a;
    DualEnv b;
    setup_dual(&a);
    setup_dual(&b);

    b.env.player = a.env.opponent;
    b.env.opponent = a.env.player;
    b.env.tick = a.env.tick;
    compute_observations(&a.env);
    compute_opponent_observations(&a.env, a.observations[1]);
    compute_observations(&b.env);
    compute_opponent_observations(&b.env, b.observations[1]);

    for (int i = 0; i < OBS_SIZE; i++) {
        if (!closef(a.observations[0][i], b.observations[1][i])
                || !closef(a.observations[1][i], b.observations[0][i])) {
            fprintf(stderr, "role-swap observation mismatch at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int test_slot_actions_control_their_own_planes(void) {
    DualEnv t;
    setup_dual(&t);

    float slot_0[NUM_ATNS] = {0.25f, -0.35f, 0.45f, -0.20f, -1.0f};
    float slot_1[NUM_ATNS] = {-0.15f, 0.30f, -0.40f, 0.10f, -1.0f};
    memcpy(t.actions[0], slot_0, sizeof(slot_0));
    memcpy(t.actions[1], slot_1, sizeof(slot_1));

    Plane expected_player = t.env.player;
    Plane expected_opponent = t.env.opponent;
    step_plane_with_params(
        &expected_player, slot_0, DT, &t.env.flight_params);
    step_plane_with_params(
        &expected_opponent, slot_1, DT, &t.env.flight_params);

    puf_step(&t.env);

    if (!plane_close(&t.env.player, &expected_player)) {
        fprintf(stderr, "slot 0 did not control the player plane\n");
        return 1;
    }
    if (!plane_close(&t.env.opponent, &expected_opponent)) {
        fprintf(stderr, "slot 1 did not control the opponent plane\n");
        return 1;
    }
    for (int i = 0; i < NUM_ATNS; i++) {
        if (!closef(t.env.last_opp_actions[i], slot_1[i])) {
            fprintf(stderr, "slot 1 action mismatch at %d\n", i);
            return 1;
        }
    }
    return 0;
}

static int test_terminal_pulse_and_reset_observations_cover_both_slots(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.player.pos.z = -10.0f;
    t.actions[0][4] = -1.0f;
    t.actions[1][4] = -1.0f;

    puf_step(&t.env);

    if (t.terminals[0] != 1.0f || t.terminals[1] != 1.0f) {
        fprintf(stderr, "terminal pulse did not cover both slots\n");
        return 1;
    }
    if (!closef(t.rewards[0], -1.0f) || !closef(t.rewards[1], 1.0f)) {
        fprintf(stderr, "two-slot crash reward mismatch: %.3f %.3f\n",
            t.rewards[0], t.rewards[1]);
        return 1;
    }
    for (int slot = 0; slot < 2; slot++) {
        for (int i = 0; i < OBS_SIZE; i++) {
            if (!isfinite(t.observations[slot][i])) {
                fprintf(stderr, "non-finite reset obs slot=%d index=%d\n",
                    slot, i);
                return 1;
            }
        }
    }
    return 0;
}

static void configure_head_on(DualEnv* t, float separation) {
    t->env.player.pos = vec3(-0.5f * separation, 0.0f, 2000.0f);
    t->env.opponent.pos = vec3(0.5f * separation, 0.0f, 2000.0f);
    t->env.player.vel = vec3(100.0f, 0.0f, 0.0f);
    t->env.opponent.vel = vec3(-100.0f, 0.0f, 0.0f);
    t->env.player.ori.w = 1.0f;
    t->env.player.ori.x = 0.0f;
    t->env.player.ori.y = 0.0f;
    t->env.player.ori.z = 0.0f;
    t->env.opponent.ori.w = 0.0f;
    t->env.opponent.ori.x = 0.0f;
    t->env.opponent.ori.y = 0.0f;
    t->env.opponent.ori.z = 1.0f;
    t->env.player.fire_cooldown = 0;
    t->env.opponent.fire_cooldown = 0;
    t->env.head_on_lockout = 0;
}

static int test_simultaneous_hits_are_an_order_invariant_draw(void) {
    DualEnv t;
    setup_dual(&t);
    configure_head_on(&t, 200.0f);
    t.actions[0][4] = 1.0f;
    t.actions[1][4] = 1.0f;

    puf_step(&t.env);

    if (t.terminals[0] != 1.0f || t.terminals[1] != 1.0f) {
        fprintf(stderr, "simultaneous hit did not terminate both slots\n");
        return 1;
    }
    if (!closef(t.rewards[0], 0.0f) || !closef(t.rewards[1], 0.0f)) {
        fprintf(stderr, "simultaneous hit was order-biased: %.3f %.3f\n",
            t.rewards[0], t.rewards[1]);
        return 1;
    }
    if (!closef(t.env.log.slot_0_score, 0.5f)
            || !closef(t.env.log.slot_1_score, 0.5f)
            || !closef(t.env.log.draw_rate, 1.0f)) {
        fprintf(stderr, "mutual-hit draw metrics were not symmetric\n");
        return 1;
    }
    if (!closef(t.env.log.slot_0_gun_kills, 0.0f)
            || !closef(t.env.log.slot_1_gun_kills, 0.0f)) {
        fprintf(stderr, "mutual hit was incorrectly counted as decisive\n");
        return 1;
    }
    return 0;
}

static int test_simultaneous_crashes_are_an_order_invariant_draw(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.player.pos.z = -10.0f;
    t.env.opponent.pos.z = -10.0f;
    t.actions[0][4] = -1.0f;
    t.actions[1][4] = -1.0f;

    puf_step(&t.env);

    if (t.terminals[0] != 1.0f || t.terminals[1] != 1.0f) {
        fprintf(stderr, "simultaneous crash did not terminate both slots\n");
        return 1;
    }
    if (!closef(t.rewards[0], 0.0f)
            || !closef(t.rewards[1], 0.0f)) {
        fprintf(stderr, "simultaneous crash was order-biased: %.3f %.3f\n",
            t.rewards[0], t.rewards[1]);
        return 1;
    }
    return 0;
}

static int test_timeout_is_a_zero_reward_draw(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.tick = t.env.max_steps - 1;
    t.actions[0][4] = -1.0f;
    t.actions[1][4] = -1.0f;

    puf_step(&t.env);

    if (t.terminals[0] != 1.0f || t.terminals[1] != 1.0f
            || !closef(t.rewards[0], 0.0f)
            || !closef(t.rewards[1], 0.0f)) {
        fprintf(stderr,
            "timeout was not a joint zero-reward terminal: "
            "term=%.1f/%.1f reward=%.3f/%.3f\n",
            t.terminals[0], t.terminals[1],
            t.rewards[0], t.rewards[1]);
        return 1;
    }
    if (!closef(t.env.log.slot_0_score, 0.5f)
            || !closef(t.env.log.slot_1_score, 0.5f)
            || !closef(t.env.log.draw_rate, 1.0f)
            || !closef(t.env.log.timeouts, 1.0f)
            || !closef(
                t.env.log.slot_0_score + t.env.log.slot_1_score, 1.0f)) {
        fprintf(stderr,
            "timeout score did not conserve one match credit\n");
        return 1;
    }
    return 0;
}

static int test_fire_cooldowns_are_identical(void) {
    DualEnv t;
    setup_dual(&t);
    configure_head_on(&t, 2000.0f);
    t.actions[0][4] = 1.0f;
    t.actions[1][4] = 1.0f;

    puf_step(&t.env);

    if (t.env.player.fire_cooldown != FIRE_COOLDOWN
            || t.env.opponent.fire_cooldown != FIRE_COOLDOWN) {
        fprintf(stderr, "cooldown mismatch: %d %d\n",
            t.env.player.fire_cooldown, t.env.opponent.fire_cooldown);
        return 1;
    }
    return 0;
}

static int test_swapping_roles_swaps_dense_rewards(void) {
    DualEnv a;
    DualEnv b;
    setup_dual(&a);
    setup_dual(&b);

    a.env.player.pos = vec3(-400.0f, 50.0f, 2500.0f);
    a.env.opponent.pos = vec3(550.0f, -100.0f, 2800.0f);
    a.env.player.vel = vec3(120.0f, 5.0f, 8.0f);
    a.env.opponent.vel = vec3(85.0f, -12.0f, -4.0f);
    b.env.player = a.env.opponent;
    b.env.opponent = a.env.player;

    float slot_0[NUM_ATNS] = {0.25f, -0.15f, 0.30f, 0.20f, -1.0f};
    float slot_1[NUM_ATNS] = {-0.10f, 0.25f, -0.20f, -0.15f, -1.0f};
    memcpy(a.actions[0], slot_0, sizeof(slot_0));
    memcpy(a.actions[1], slot_1, sizeof(slot_1));
    memcpy(b.actions[0], slot_1, sizeof(slot_1));
    memcpy(b.actions[1], slot_0, sizeof(slot_0));

    puf_step(&a.env);
    puf_step(&b.env);

    if (a.terminals[0] != 0.0f || b.terminals[0] != 0.0f) {
        fprintf(stderr, "role-swap shaping fixture terminated unexpectedly\n");
        return 1;
    }
    if (!closef(a.rewards[0], b.rewards[1])
            || !closef(a.rewards[1], b.rewards[0])) {
        fprintf(stderr,
            "role-swap reward mismatch: A=(%.6f,%.6f) B=(%.6f,%.6f)\n",
            a.rewards[0], a.rewards[1], b.rewards[0], b.rewards[1]);
        return 1;
    }
    return 0;
}

static int test_logical_role_assignment_routes_observations_and_actions(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.two_agent_player_slot = 1;
    dogfight_two_agent_bind_slots(&t.env);

    compute_observations(&t.env);
    compute_opponent_observations(
        &t.env, t.env.opponent_observations);
    float expected_0[OBS_SIZE] = {0};
    float expected_1[OBS_SIZE] = {0};
    compute_obs_opponent_aware_for_plane(
        &t.env, &t.env.opponent, &t.env.player, expected_0);
    compute_obs_opponent_aware_for_plane(
        &t.env, &t.env.player, &t.env.opponent, expected_1);
    for (int i = 0; i < OBS_SIZE; i++) {
        if (!closef(t.observations[0][i], expected_0[i])
                || !closef(t.observations[1][i], expected_1[i])) {
            fprintf(stderr, "logical role observation routing failed at %d\n",
                i);
            return 1;
        }
    }

    float slot_0[NUM_ATNS] = {-0.15f, 0.30f, -0.40f, 0.10f, -1.0f};
    float slot_1[NUM_ATNS] = {0.25f, -0.35f, 0.45f, -0.20f, -1.0f};
    memcpy(t.actions[0], slot_0, sizeof(slot_0));
    memcpy(t.actions[1], slot_1, sizeof(slot_1));
    Plane expected_player = t.env.player;
    Plane expected_opponent = t.env.opponent;
    step_plane_with_params(
        &expected_player, slot_1, DT, &t.env.flight_params);
    step_plane_with_params(
        &expected_opponent, slot_0, DT, &t.env.flight_params);

    puf_step(&t.env);

    if (!plane_close(&t.env.player, &expected_player)
            || !plane_close(&t.env.opponent, &expected_opponent)) {
        fprintf(stderr, "logical role action routing failed\n");
        return 1;
    }
    return 0;
}

static int test_logical_slot_scores_follow_role_assignment(void) {
    DualEnv t;
    setup_dual(&t);
    t.env.two_agent_player_slot = 1;
    dogfight_two_agent_bind_slots(&t.env);
    configure_head_on(&t, 200.0f);
    t.actions[0][4] = -1.0f;
    t.actions[1][4] = 1.0f;

    puf_step(&t.env);

    if (!closef(t.rewards[0], -1.0f)
            || !closef(t.rewards[1], 1.0f)) {
        fprintf(stderr, "logical role terminal reward routing failed\n");
        return 1;
    }
    if (!closef(t.env.log.slot_0_score, 0.0f)
            || !closef(t.env.log.slot_1_score, 1.0f)
            || !closef(
                t.env.log.slot_0_score + t.env.log.slot_1_score, 1.0f)
            || !closef(t.env.log.draw_rate, 0.0f)) {
        fprintf(stderr, "logical slot score routing failed\n");
        return 1;
    }
    if (!closef(t.env.log.shots_fired, 1.0f)
            || !closef(t.env.log.accuracy, 100.0f)) {
        fprintf(stderr, "physical-player shot accounting failed\n");
        return 1;
    }
    if (!closef(t.env.log.slot_0_gun_kills, 0.0f)
            || !closef(t.env.log.slot_1_gun_kills, 1.0f)) {
        fprintf(stderr, "logical slot gun-kill routing failed\n");
        return 1;
    }
    return 0;
}

static int test_role_randomization_is_seed_deterministic(void) {
    DualEnv a;
    DualEnv b;
    setup_dual(&a);
    setup_dual(&b);
    a.env.two_agent_role_randomization = 1;
    b.env.two_agent_role_randomization = 1;
    int saw_slot_0 = 0;
    int saw_slot_1 = 0;

    for (int episode = 0; episode < 16; episode++) {
        puf_reset(&a.env);
        puf_reset(&b.env);
        if (a.env.two_agent_player_slot != b.env.two_agent_player_slot) {
            fprintf(stderr, "seeded role sequence diverged at %d\n", episode);
            return 1;
        }
        saw_slot_0 |= a.env.two_agent_player_slot == 0;
        saw_slot_1 |= a.env.two_agent_player_slot == 1;
    }
    if (!saw_slot_0 || !saw_slot_1) {
        fprintf(stderr, "role randomization did not exercise both slots\n");
        return 1;
    }
    return 0;
}

static int test_historical_boundary_signal_matches_robocode(void) {
    DualEnv historical;
    setup_dual(&historical);
    historical.env.tag = 1;
    historical.env.boundary_reached = 0;
    historical.env.player.pos.z = -10.0f;
    historical.actions[0][4] = -1.0f;
    historical.actions[1][4] = -1.0f;

    puf_step(&historical.env);

    if (historical.env.boundary_reached != 1) {
        fprintf(stderr,
            "tagged historical match did not signal episode boundary\n");
        return 1;
    }
    puf_reset(&historical.env);
    if (historical.env.boundary_reached != 1) {
        fprintf(stderr,
            "environment reset cleared trainer-owned boundary signal\n");
        return 1;
    }

    DualEnv pure;
    setup_dual(&pure);
    pure.env.tag = 0;
    pure.env.boundary_reached = 0;
    pure.env.player.pos.z = -10.0f;
    pure.actions[0][4] = -1.0f;
    pure.actions[1][4] = -1.0f;

    puf_step(&pure.env);

    if (pure.env.boundary_reached != 0) {
        fprintf(stderr, "pure current-current match signaled pool boundary\n");
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_init_declares_robocode_policy_rows();
    failures += test_reset_observations_are_both_fresh();
    failures += test_swapped_planes_swap_observations();
    failures += test_slot_actions_control_their_own_planes();
    failures += test_terminal_pulse_and_reset_observations_cover_both_slots();
    failures += test_simultaneous_hits_are_an_order_invariant_draw();
    failures += test_simultaneous_crashes_are_an_order_invariant_draw();
    failures += test_timeout_is_a_zero_reward_draw();
    failures += test_fire_cooldowns_are_identical();
    failures += test_swapping_roles_swaps_dense_rewards();
    failures += test_logical_role_assignment_routes_observations_and_actions();
    failures += test_logical_slot_scores_follow_role_assignment();
    failures += test_role_randomization_is_seed_deterministic();
    failures += test_historical_boundary_signal_matches_robocode();
    if (failures == 0) {
        puts("two-agent adapter: ok");
    }
    return failures;
}
