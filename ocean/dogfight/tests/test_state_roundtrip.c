#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float opponent_observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float opponent_rewards[1];
    float terminals[1];
} TestEnv;

static int nearly(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static int nearly_vec3(Vec3 a, Vec3 b) {
    return nearly(a.x, b.x) && nearly(a.y, b.y) && nearly(a.z, b.z);
}

static int nearly_quat(Quat a, Quat b) {
    return nearly(a.w, b.w) && nearly(a.x, b.x) && nearly(a.y, b.y) && nearly(a.z, b.z);
}

static int nearly_actions(float a[TEST_NUM_ATNS], float b[TEST_NUM_ATNS]) {
    for (int i = 0; i < TEST_NUM_ATNS; i++) {
        if (!nearly(a[i], b[i])) {
            return 0;
        }
    }
    return 1;
}

static RewardConfig test_default_rcfg(void) {
    RewardConfig r = {0};
    r.aim_scale = 0.001695f;
    r.closing_scale = 0.0001f;
    r.neg_g = 0.035f;
    r.control_rate_penalty = 0.002f;
    r.low_altitude_threshold = 1200.0f;
    r.low_altitude_penalty = 0.005f;
    r.speed_min = 50.0f;
    r.aim_decay_stage = 30.0f;
    r.shaping_decay_start = 100000000L;
    r.shaping_decay_end = 150000000L;
    r.energy_gain_scale = 0.001f;
    r.energy_loss_scale = 0.0005f;
    r.energy_advantage_scale = 0.004f;
    return r;
}

static void setup_env(TestEnv* t, unsigned int rng) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 300;
    t->env.rng = rng;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, 1, &rcfg, 1, 0, 0);
    t->env.opponent_observations = t->opponent_observations;
    t->env.opponent_rewards = t->opponent_rewards;
    set_curriculum_target(&t->env, 5.0f);
    c_reset(&t->env);
}

static void step_with_action(TestEnv* t, const float action[TEST_NUM_ATNS]) {
    memcpy(t->actions, action, TEST_NUM_ATNS * sizeof(float));
    c_step(&t->env);
}

static int compare_observations(TestEnv* a, TestEnv* b) {
    for (int i = 0; i < TEST_OBS_SIZE; i++) {
        if (!nearly(a->observations[i], b->observations[i])) {
            printf("state_roundtrip: obs[%d] %.7f != %.7f [FAIL]\n",
                i, a->observations[i], b->observations[i]);
            return 1;
        }
    }
    return 0;
}

static int compare_future_step(TestEnv* a, TestEnv* b) {
    int fail = 0;
    fail |= compare_observations(a, b);
    if (!nearly(a->rewards[0], b->rewards[0])) {
        printf("state_roundtrip: reward %.7f != %.7f [FAIL]\n",
            a->rewards[0], b->rewards[0]);
        fail = 1;
    }
    if (!nearly(a->opponent_rewards[0], b->opponent_rewards[0])) {
        printf("state_roundtrip: opponent reward %.7f != %.7f [FAIL]\n",
            a->opponent_rewards[0], b->opponent_rewards[0]);
        fail = 1;
    }
    if (!nearly(a->terminals[0], b->terminals[0])) {
        printf("state_roundtrip: terminal %.1f != %.1f [FAIL]\n",
            a->terminals[0], b->terminals[0]);
        fail = 1;
    }
    if (a->env.tick != b->env.tick || a->env.death_reason != b->env.death_reason) {
        printf("state_roundtrip: tick/death %d/%d != %d/%d [FAIL]\n",
            a->env.tick, a->env.death_reason, b->env.tick, b->env.death_reason);
        fail = 1;
    }
    if (!nearly_vec3(a->env.player.pos, b->env.player.pos) ||
            !nearly_vec3(a->env.player.vel, b->env.player.vel) ||
            !nearly_quat(a->env.player.ori, b->env.player.ori) ||
            !nearly_vec3(a->env.opponent.pos, b->env.opponent.pos) ||
            !nearly_vec3(a->env.opponent.vel, b->env.opponent.vel) ||
            !nearly_quat(a->env.opponent.ori, b->env.opponent.ori)) {
        printf("state_roundtrip: plane state diverged [FAIL]\n");
        fail = 1;
    }
    return fail;
}

static int compare_stage_climb_runtime_state(Dogfight* a, Dogfight* b, const char* label) {
    int fail = 0;
    if (a->stage != b->stage ||
            a->selfplay_active != b->selfplay_active ||
            a->use_opponent_override != b->use_opponent_override ||
            a->opponent_recovery_active != b->opponent_recovery_active ||
            a->opponent_recovery_tick_start != b->opponent_recovery_tick_start ||
            a->opponent_above_recovery_threshold != b->opponent_above_recovery_threshold ||
            a->guided_climb_active != b->guided_climb_active ||
            a->guided_climb_ticks_remaining != b->guided_climb_ticks_remaining ||
            a->vertical_level != b->vertical_level ||
            a->vertical_spawn_used != b->vertical_spawn_used ||
            a->side_spawn_variant != b->side_spawn_variant ||
            a->opponent_ap.mode != b->opponent_ap.mode ||
            a->opponent_ap.recovery_phase != b->opponent_ap.recovery_phase ||
            a->recovery_rng_state != b->recovery_rng_state ||
            a->global_step != b->global_step) {
        printf("%s: runtime ints diverged stage=%d/%d selfplay=%d/%d override=%d/%d recovery=%d/%d ap=%d/%d [FAIL]\n",
            label, a->stage, b->stage, a->selfplay_active, b->selfplay_active,
            a->use_opponent_override, b->use_opponent_override,
            a->opponent_recovery_active, b->opponent_recovery_active,
            a->opponent_ap.mode, b->opponent_ap.mode);
        fail = 1;
    }

    if (!nearly(a->curriculum_target, b->curriculum_target) ||
            !nearly(a->selfplay_prob, b->selfplay_prob) ||
            !nearly(a->recovery_altitude_threshold, b->recovery_altitude_threshold) ||
            !nearly(a->recovery_trigger_prob, b->recovery_trigger_prob) ||
            !nearly(a->recovery_speed_threshold, b->recovery_speed_threshold) ||
            !nearly(a->recovery_bank_deg, b->recovery_bank_deg) ||
            !nearly(a->guided_climb_elevator, b->guided_climb_elevator) ||
            !nearly(a->vertical_spawn_prob, b->vertical_spawn_prob) ||
            !nearly(a->stage9_bank_deg, b->stage9_bank_deg) ||
            !nearly(a->side_energy_spawn_prob, b->side_energy_spawn_prob) ||
            !nearly(a->prev_player_target_az, b->prev_player_target_az) ||
            !nearly(a->prev_player_target_el, b->prev_player_target_el) ||
            !nearly(a->prev_player_aspect, b->prev_player_aspect) ||
            !nearly(a->prev_player_eadv, b->prev_player_eadv) ||
            !nearly(a->prev_opp_target_az, b->prev_opp_target_az) ||
            !nearly(a->prev_opp_target_el, b->prev_opp_target_el) ||
            !nearly(a->prev_opp_aspect, b->prev_opp_aspect) ||
            !nearly(a->prev_opp_eadv, b->prev_opp_eadv) ||
            !nearly(a->domain_randomization, b->domain_randomization) ||
            !nearly(a->flight_params.mass, b->flight_params.mass) ||
            !nearly(a->flight_params.ixx, b->flight_params.ixx) ||
            !nearly(a->flight_params.c_l_alpha, b->flight_params.c_l_alpha) ||
            !nearly(a->flight_params.engine_power, b->flight_params.engine_power) ||
            !nearly(a->flight_params.cm_q, b->flight_params.cm_q) ||
            !nearly(a->opponent_ap.target_bank, b->opponent_ap.target_bank) ||
            !nearly(a->opponent_ap.recovery_speed_threshold, b->opponent_ap.recovery_speed_threshold) ||
            !nearly(a->opponent_ap.prev_vz, b->opponent_ap.prev_vz) ||
            !nearly(a->opponent_ap.prev_bank_error, b->opponent_ap.prev_bank_error) ||
            !nearly_actions(a->last_opp_actions, b->last_opp_actions)) {
        printf("%s: runtime floats/actions diverged target=%.3f/%.3f recovery_thr=%.3f/%.3f dr=%.3f/%.3f [FAIL]\n",
            label, a->curriculum_target, b->curriculum_target,
            a->recovery_altitude_threshold, b->recovery_altitude_threshold,
            a->domain_randomization, b->domain_randomization);
        fail = 1;
    }
    return fail;
}

static int compare_terminal_reset_result(TestEnv* a, TestEnv* b) {
    int fail = compare_future_step(a, b);
    if (a->env.last_death_reason != b->env.last_death_reason ||
            a->env.last_winner != b->env.last_winner ||
            a->env.total_episodes != b->env.total_episodes) {
        printf("state_roundtrip_terminal: last result %d/%d/%d != %d/%d/%d [FAIL]\n",
            a->env.last_death_reason, a->env.last_winner, a->env.total_episodes,
            b->env.last_death_reason, b->env.last_winner, b->env.total_episodes);
        fail = 1;
    }
    return fail;
}

static void configure_stage_climb_recovery_state(TestEnv* t) {
    force_state(
        &t->env,
        0.0f, -200.0f, 1600.0f,
        115.0f, 10.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0.55f,
        420.0f, 20.0f, 900.0f,
        95.0f, -15.0f, -35.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        40, 0, 3);

    t->env.stage = CURRICULUM_HARD_MANEUVERING;
    t->env.curriculum_target = 17.0f;
    t->env.global_step = 123456789L;
    t->env.selfplay_active = 1;
    t->env.use_opponent_override = 0;
    t->env.selfplay_prob = 0.25f;
    t->env.head_on_lockout = 1;
    t->env.prev_rel_dot = -500.0f;

    t->env.opponent_recovery_active = 1;
    t->env.opponent_recovery_tick_start = 28;
    t->env.recovery_altitude_threshold = 1000.0f;
    t->env.recovery_trigger_prob = 0.75f;
    t->env.recovery_speed_threshold = 82.0f;
    t->env.recovery_bank_deg = 55.0f;
    t->env.recovery_rng_state = 0x12345678u;
    t->env.opponent_above_recovery_threshold = 0;
    autopilot_start_recovery(&t->env.opponent_ap, 82.0f, 55.0f);
    t->env.opponent_ap.prev_vz = -35.0f;
    t->env.opponent_ap.prev_bank_error = 0.125f;

    t->env.guided_climb_active = 1;
    t->env.guided_climb_ticks_remaining = 37;
    t->env.guided_climb_elevator = 0.42f;
    t->env.vertical_spawn_prob = 0.63f;
    t->env.vertical_level = 3;
    t->env.vertical_spawn_used = 1;
    t->env.stage9_bank_deg = 30.0f;
    t->env.side_energy_spawn_prob = 0.88f;
    t->env.side_spawn_variant = SIDE_SPAWN_ENERGY;

    t->env.prev_player_target_az = 0.11f;
    t->env.prev_player_target_el = -0.12f;
    t->env.prev_player_aspect = 0.13f;
    t->env.prev_player_eadv = -0.14f;
    t->env.prev_opp_target_az = -0.21f;
    t->env.prev_opp_target_el = 0.22f;
    t->env.prev_opp_aspect = -0.23f;
    t->env.prev_opp_eadv = 0.24f;

    t->env.domain_randomization = 0.05f;
    t->env.flight_params.mass *= 1.03f;
    t->env.flight_params.inv_mass = 1.0f / t->env.flight_params.mass;
    t->env.flight_params.ixx *= 0.97f;
    t->env.flight_params.c_l_alpha *= 1.02f;
    t->env.flight_params.engine_power *= 1.04f;
    t->env.flight_params.cm_q *= 0.96f;

    const float last_opp[TEST_NUM_ATNS] = {0.1f, -0.2f, 0.3f, -0.4f, -1.0f};
    memcpy(t->env.last_opp_actions, last_opp, sizeof(last_opp));
    dogfight_state_capture(&t->env);
}

static int test_state_roundtrip_restores_stage_climb_runtime_state(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 777);
    setup_env(&restored, 888);

    configure_stage_climb_recovery_state(&source);
    State snapshot = source.env.state;

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);

    int fail = compare_stage_climb_runtime_state(
        &source.env, &restored.env, "state_roundtrip_stage_climb_restore");

    const float probe_action[TEST_NUM_ATNS] = {0.10f, 0.05f, -0.15f, 0.20f, -1.0f};
    step_with_action(&source, probe_action);
    step_with_action(&restored, probe_action);

    fail |= compare_future_step(&source, &restored);
    fail |= compare_stage_climb_runtime_state(
        &source.env, &restored.env, "state_roundtrip_stage_climb_future");

    if (!fail) {
        printf("state_roundtrip_stage_climb: restored stage-climb runtime state matches future [OK]\n");
    }
    return fail;
}

static int test_state_refresh_preserves_applied_runtime_config(void) {
    TestEnv t;
    setup_env(&t, 135);

    RuntimeConfig cfg = {
        .eval_spawn_mode = 2,
        .recovery_enabled = 1,
        .recovery_altitude_threshold = 1400.0f,
        .recovery_trigger_prob = 0.73f,
        .recovery_speed_threshold = 91.0f,
        .recovery_bank_deg = 47.0f,
        .domain_randomization = 0.09f,
        .vertical_spawn_prob = 0.41f,
        .stage9_bank_deg = -1.0f,
        .side_energy_spawn_prob = 0.67f,
    };
    apply_runtime_config(&t.env, &cfg);

    dogfight_state_refresh(&t.env);

    int fail = 0;
    if (t.env.eval_spawn_mode != 2 ||
            !nearly(t.env.recovery_altitude_threshold, 1400.0f) ||
            !nearly(t.env.recovery_trigger_prob, 0.73f) ||
            !nearly(t.env.recovery_speed_threshold, 91.0f) ||
            !nearly(t.env.recovery_bank_deg, 47.0f) ||
            !nearly(t.env.domain_randomization, 0.09f) ||
            !nearly(t.env.vertical_spawn_prob, 0.41f) ||
            !nearly(t.env.stage9_bank_deg, -1.0f) ||
            !nearly(t.env.side_energy_spawn_prob, 0.67f)) {
        printf("state_refresh_runtime_config: runtime config was reverted by state refresh [FAIL]\n");
        fail = 1;
    }
    if (!fail) {
        printf("state_refresh_runtime_config: applied runtime config survives refresh [OK]\n");
    }
    return fail;
}

static int test_state_refresh_preserves_curriculum_setters(void) {
    TestEnv target_env;
    TestEnv stage_env;
    setup_env(&target_env, 246);
    setup_env(&stage_env, 357);

    set_curriculum_target(&target_env.env, 11.9f);
    dogfight_state_refresh(&target_env.env);

    set_curriculum_stage(&stage_env.env, CURRICULUM_HARD_MANEUVERING);
    dogfight_state_refresh(&stage_env.env);

    int fail = 0;
    if (!nearly(target_env.env.curriculum_target, 11.9f)) {
        printf("state_refresh_curriculum_target: target reverted to %.3f [FAIL]\n",
            target_env.env.curriculum_target);
        fail = 1;
    }
    if (stage_env.env.stage != CURRICULUM_HARD_MANEUVERING ||
            !nearly(stage_env.env.curriculum_target, (float)CURRICULUM_HARD_MANEUVERING)) {
        printf("state_refresh_curriculum_stage: stage/target reverted to %d/%.3f [FAIL]\n",
            stage_env.env.stage, stage_env.env.curriculum_target);
        fail = 1;
    }
    if (!fail) {
        printf("state_refresh_curriculum_setters: curriculum setters survive refresh [OK]\n");
    }
    return fail;
}

static int test_state_roundtrip_restores_future_deterministically(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 42);
    setup_env(&restored, 999);

    const float warmup_action[TEST_NUM_ATNS] = {0.20f, 0.15f, -0.35f, 0.05f, -1.0f};
    for (int i = 0; i < 8; i++) {
        step_with_action(&source, warmup_action);
    }

    State snapshot = source.env.state;

    const float probe_action[TEST_NUM_ATNS] = {0.35f, -0.20f, 0.40f, -0.10f, 1.0f};
    step_with_action(&source, probe_action);

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);
    step_with_action(&restored, probe_action);

    int fail = compare_future_step(&source, &restored);
    if (!fail) {
        printf("state_roundtrip: restored Dogfight state matches scripted future [OK]\n");
    }
    return fail;
}

static int test_state_roundtrip_restores_terminal_future_deterministically(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 123);
    setup_env(&restored, 456);

    force_state(
        &source.env,
        0.0f, 0.0f, 1000.0f,
        150.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,
        250.0f, 0.0f, 1000.0f,
        150.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        3, 0, 0);

    State snapshot = source.env.state;

    const float kill_action[TEST_NUM_ATNS] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    step_with_action(&source, kill_action);

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);
    step_with_action(&restored, kill_action);

    int fail = compare_terminal_reset_result(&source, &restored);
    if (!fail) {
        printf("state_roundtrip_terminal: restored terminal future matches scripted kill [OK]\n");
    }
    return fail;
}

static void configure_selfplay_opponent_kill(TestEnv* t) {
    force_state(
        &t->env,
        260.0f, 0.0f, 1800.0f,
        105.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0.5f,
        0.0f, 0.0f, 1800.0f,
        120.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        0, 0, 0);

    t->env.stage = CURRICULUM_TAIL_CHASE;
    t->env.curriculum_target = 0.0f;
    t->env.use_opponent_override = 1;
    t->env.selfplay_active = 1;
    t->env.selfplay_prob = 1.0f;
    t->env.head_on_lockout = 0;
    t->env.player.fire_cooldown = 0;
    t->env.opponent.fire_cooldown = 0;

    const float player_action[TEST_NUM_ATNS] = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f};
    const float opponent_action[TEST_NUM_ATNS] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    memcpy(t->actions, player_action, sizeof(player_action));
    memcpy(t->env.opponent_actions_override, opponent_action, sizeof(opponent_action));
    dogfight_state_capture(&t->env);
}

static int test_state_roundtrip_restores_selfplay_opponent_terminal_future(void) {
    TestEnv source;
    TestEnv restored;
    setup_env(&source, 321);
    setup_env(&restored, 654);

    configure_selfplay_opponent_kill(&source);
    State snapshot = source.env.state;

    step_with_action(&source, source.actions);

    restored.env.state = snapshot;
    dogfight_state_restore(&restored.env);
    step_with_action(&restored, source.actions);

    int fail = compare_terminal_reset_result(&source, &restored);
    if (source.rewards[0] != -1.0f || source.opponent_rewards[0] != 1.0f ||
            source.env.last_death_reason != DEATH_KILL ||
            source.env.last_winner != -1 ||
            source.env.log.sp_opp_kills != 1.0f) {
        printf("state_roundtrip_selfplay_terminal: source terminal reward=%.1f opp=%.1f last=%d winner=%d sp_opp=%.1f [FAIL]\n",
            source.rewards[0], source.opponent_rewards[0],
            source.env.last_death_reason, source.env.last_winner,
            source.env.log.sp_opp_kills);
        fail = 1;
    }
    if (!fail) {
        printf("state_roundtrip_selfplay_terminal: restored opponent-kill future matches [OK]\n");
    }
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= test_state_roundtrip_restores_future_deterministically();
    fail |= test_state_roundtrip_restores_terminal_future_deterministically();
    fail |= test_state_roundtrip_restores_selfplay_opponent_terminal_future();
    fail |= test_state_roundtrip_restores_stage_climb_runtime_state();
    fail |= test_state_refresh_preserves_applied_runtime_config();
    fail |= test_state_refresh_preserves_curriculum_setters();
    return fail;
}
