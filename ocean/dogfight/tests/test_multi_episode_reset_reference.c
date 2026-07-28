#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dogfight.h"

#define TEST_OBS_SIZE 26
#define TEST_NUM_ATNS 5

typedef struct TestEnv {
    Dogfight env;
    float observations[TEST_OBS_SIZE];
    float actions[TEST_NUM_ATNS];
    float rewards[1];
    float opponent_rewards[1];
    float terminals[1];
} TestEnv;

static int nearly(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static RewardConfig test_rcfg(void) {
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

static void setup_env(TestEnv* t) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 1200;
    t->env.rng = 424242;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_rcfg();
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);
    t->env.opponent_rewards = t->opponent_rewards;
}

static void set_action(float* action, float throttle, float elevator,
        float aileron, float rudder, float trigger) {
    action[0] = throttle;
    action[1] = elevator;
    action[2] = aileron;
    action[3] = rudder;
    action[4] = trigger;
}

static void prepare_forced_episode(TestEnv* t, int selfplay_active, int max_steps) {
    t->env.max_steps = max_steps;
    t->env.configured_max_steps = max_steps;
    t->env.stage = CURRICULUM_TAIL_CHASE;
    t->env.curriculum_target = 0.0f;
    t->env.global_step = 25L;
    t->env.use_opponent_override = 1;
    t->env.selfplay_active = selfplay_active;
    t->env.selfplay_prob = 1.0f;
    t->env.head_on_lockout = 0;
    t->env.death_reason = DEATH_NONE;
    t->env.player.fire_cooldown = 0;
    t->env.opponent.fire_cooldown = 0;
    t->env.flight_params = default_flight_params();
    set_action(t->actions, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
    set_action(t->env.opponent_actions_override, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
}

static void set_plane(Plane* p, Vec3 pos, Vec3 vel, Quat ori, float throttle) {
    p->pos = pos;
    p->vel = vel;
    p->prev_vel = vel;
    p->omega = vec3(0.0f, 0.0f, 0.0f);
    p->ori = ori;
    quat_normalize(&p->ori);
    p->throttle = throttle;
    p->g_force = 1.0f;
    p->yaw_from_rudder = 0.0f;
}

static void refresh_episode_energy(TestEnv* t) {
    t->env.player.prev_energy =
        calc_specific_energy_with_params(&t->env.player, &t->env.flight_params);
    t->env.opponent.prev_energy =
        calc_specific_energy_with_params(&t->env.opponent, &t->env.flight_params);
}

static void run_player_kill(TestEnv* t) {
    prepare_forced_episode(t, 0, 1200);
    set_plane(&t->env.player, vec3(0.0f, 0.0f, 1800.0f),
        vec3(120.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.6f);
    set_plane(&t->env.opponent, vec3(260.0f, 0.0f, 1800.0f),
        vec3(105.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    refresh_episode_energy(t);
    t->actions[4] = 1.0f;
    c_step(&t->env);
}

static void run_timeout(TestEnv* t) {
    prepare_forced_episode(t, 0, 1);
    set_plane(&t->env.player, vec3(0.0f, 0.0f, 1800.0f),
        vec3(90.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    set_plane(&t->env.opponent, vec3(260.0f, 0.0f, 1800.0f),
        vec3(80.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    refresh_episode_energy(t);
    c_step(&t->env);
}

static void run_player_oob(TestEnv* t) {
    prepare_forced_episode(t, 0, 1200);
    set_plane(&t->env.player, vec3(0.0f, 0.0f, -5.0f),
        vec3(90.0f, 0.0f, -10.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    set_plane(&t->env.opponent, vec3(260.0f, 0.0f, 1800.0f),
        vec3(80.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    refresh_episode_energy(t);
    c_step(&t->env);
}

static void run_selfplay_opponent_kill(TestEnv* t) {
    prepare_forced_episode(t, 1, 1200);
    set_plane(&t->env.player, vec3(260.0f, 0.0f, 1800.0f),
        vec3(105.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.5f);
    set_plane(&t->env.opponent, vec3(0.0f, 0.0f, 1800.0f),
        vec3(120.0f, 0.0f, 0.0f), quat(1.0f, 0.0f, 0.0f, 0.0f), 0.6f);
    refresh_episode_energy(t);
    t->env.opponent_actions_override[4] = 1.0f;
    c_step(&t->env);
}

static int test_multi_episode_reset_and_logs_match_df36_fixtures(void) {
    TestEnv t;
    setup_env(&t);

    run_player_kill(&t);
    run_timeout(&t);
    run_player_oob(&t);
    run_selfplay_opponent_kill(&t);

    if (t.env.tick != 0 || t.env.death_reason != DEATH_NONE ||
            t.env.last_death_reason != DEATH_KILL || t.env.last_winner != -1 ||
            t.env.kill != 0 || t.env.opp_kill != 0 ||
            t.env.total_episodes != 4 ||
            !nearly(t.env.episode_return, 0.0f, 1e-6f) ||
            !nearly(t.env.episode_shots_fired, 0.0f, 1e-6f)) {
        printf("multi_episode_reset: reset flags tick=%d death=%d last=%d winner=%d kill=%d opp=%d eps=%d [FAIL]\n",
            t.env.tick, t.env.death_reason, t.env.last_death_reason,
            t.env.last_winner, t.env.kill, t.env.opp_kill, t.env.total_episodes);
        return 1;
    }

    if (!nearly(t.env.log.episode_return, 0.992029309f, 1e-6f) ||
            !nearly(t.env.log.episode_length, 4.0f, 1e-6f) ||
            !nearly(t.env.log.perf, 1.0f, 1e-6f) ||
            !nearly(t.env.log.score, -1.5f, 1e-6f) ||
            !nearly(t.env.log.sp_player_kills, 0.0f, 1e-6f) ||
            !nearly(t.env.log.sp_opp_kills, 1.0f, 1e-6f) ||
            !nearly(t.env.log.shots_fired, 1.0f, 1e-6f) ||
            !nearly(t.env.log.accuracy, 100.0f, 1e-5f) ||
            !nearly(t.env.log.stage, 0.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_kills, 1.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_eps, 4.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_ground, 1.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_timeouts, 1.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_episode_length, 4.0f, 1e-6f) ||
            !nearly(t.env.log.player_ground_hits, 1.0f, 1e-6f) ||
            !nearly(t.env.log.opponent_ground_hits, 0.0f, 1e-6f) ||
            !nearly(t.env.log.clean_fights, 1.0f, 1e-6f) ||
            !nearly(t.env.log.n, 4.0f, 1e-6f) ||
            !nearly(t.env.log.kill_rate, 0.25f, 1e-6f) ||
            !nearly(t.env.log.avg_stage, 0.0f, 1e-6f) ||
            !nearly(t.env.log.ultimate, 0.0025f, 1e-6f) ||
            !nearly(t.env.log.ultimate2, 0.0625f, 1e-6f) ||
            !nearly(t.env.log.total_stage_weight, 0.04f, 1e-6f) ||
            !nearly(t.env.log.total_abs_bias, 0.0f, 1e-6f) ||
            !nearly(t.env.log.total_control_rate, 0.0f, 1e-6f)) {
        printf("multi_episode_reset: logs ret=%.9f len=%.9f perf=%.9f score=%.9f sp=%.9f/%.9f base=%.9f/%.9f ground=%.9f timeout=%.9f n=%.9f [FAIL]\n",
            t.env.log.episode_return, t.env.log.episode_length,
            t.env.log.perf, t.env.log.score,
            t.env.log.sp_player_kills, t.env.log.sp_opp_kills,
            t.env.log.base_stage_kills, t.env.log.base_stage_eps,
            t.env.log.base_stage_ground, t.env.log.base_stage_timeouts,
            t.env.log.n);
        return 1;
    }

    printf("multi_episode_reset: aggregate reset-boundary logs match df36 terminal fixtures [OK]\n");
    return 0;
}

int main(void) {
    return test_multi_episode_reset_and_logs_match_df36_fixtures();
}
