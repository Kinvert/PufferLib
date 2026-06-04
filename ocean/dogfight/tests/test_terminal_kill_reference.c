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
    t->env.rng = 24680;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_rcfg();
    init(&t->env, OBS_OPPONENT_AWARE, &rcfg, 1, 0, 0);
    t->env.opponent_rewards = t->opponent_rewards;

    t->env.tick = 0;
    t->env.stage = CURRICULUM_TAIL_CHASE;
    t->env.curriculum_target = 0.0f;
    t->env.global_step = 25L;
    t->env.use_opponent_override = 1;
    t->env.head_on_lockout = 0;
    t->env.selfplay_active = 0;
    t->env.player.fire_cooldown = 0;
    t->env.opponent.fire_cooldown = 0;

    t->env.player.pos = vec3(0.0f, 0.0f, 1800.0f);
    t->env.player.vel = vec3(120.0f, 0.0f, 0.0f);
    t->env.player.prev_vel = t->env.player.vel;
    t->env.player.omega = vec3(0.0f, 0.0f, 0.0f);
    t->env.player.ori = quat(1.0f, 0.0f, 0.0f, 0.0f);
    t->env.player.throttle = 0.6f;
    t->env.player.g_force = 1.0f;

    t->env.opponent.pos = vec3(260.0f, 0.0f, 1800.0f);
    t->env.opponent.vel = vec3(105.0f, 0.0f, 0.0f);
    t->env.opponent.prev_vel = t->env.opponent.vel;
    t->env.opponent.omega = vec3(0.0f, 0.0f, 0.0f);
    t->env.opponent.ori = quat(1.0f, 0.0f, 0.0f, 0.0f);
    t->env.opponent.throttle = 0.5f;
    t->env.opponent.g_force = 1.0f;

    t->env.player.prev_energy =
        calc_specific_energy_with_params(&t->env.player, &t->env.flight_params);
    t->env.opponent.prev_energy =
        calc_specific_energy_with_params(&t->env.opponent, &t->env.flight_params);

    t->actions[0] = 0.0f;
    t->actions[1] = 0.0f;
    t->actions[2] = 0.0f;
    t->actions[3] = 0.0f;
    t->actions[4] = 1.0f;

    t->env.opponent_actions_override[0] = 0.0f;
    t->env.opponent_actions_override[1] = 0.0f;
    t->env.opponent_actions_override[2] = 0.0f;
    t->env.opponent_actions_override[3] = 0.0f;
    t->env.opponent_actions_override[4] = -1.0f;
}

static int test_terminal_player_kill_matches_dogfight3_fixture(void) {
    TestEnv t;
    setup_env(&t);
    c_step(&t.env);

    if (!nearly(t.rewards[0], 1.0f, 1e-6f) ||
            !nearly(t.opponent_rewards[0], -1.0f, 1e-6f) ||
            !nearly(t.terminals[0], 1.0f, 1e-6f)) {
        printf("terminal_kill: rewards player=%.9f opponent=%.9f terminal=%.1f [FAIL]\n",
            t.rewards[0], t.opponent_rewards[0], t.terminals[0]);
        return 1;
    }

    if (t.env.tick != 0 || t.env.death_reason != DEATH_NONE ||
            t.env.last_death_reason != DEATH_KILL || t.env.last_winner != 1 ||
            t.env.kill != 0 || t.env.opp_kill != 0 ||
            t.env.total_episodes != 1 ||
            !nearly(t.env.episode_return, 0.0f, 1e-6f) ||
            !nearly(t.env.episode_shots_fired, 0.0f, 1e-6f)) {
        printf("terminal_kill: reset flags tick=%d death=%d last=%d winner=%d kill=%d opp=%d eps=%d [FAIL]\n",
            t.env.tick, t.env.death_reason, t.env.last_death_reason,
            t.env.last_winner, t.env.kill, t.env.opp_kill, t.env.total_episodes);
        return 1;
    }

    if (!nearly(t.env.log.episode_return, 1.0f, 1e-6f) ||
            !nearly(t.env.log.episode_length, 1.0f, 1e-6f) ||
            !nearly(t.env.log.perf, 1.0f, 1e-6f) ||
            !nearly(t.env.log.score, 1.0f, 1e-6f) ||
            !nearly(t.env.log.shots_fired, 1.0f, 1e-6f) ||
            !nearly(t.env.log.accuracy, 100.0f, 1e-5f) ||
            !nearly(t.env.log.stage, 0.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_kills, 1.0f, 1e-6f) ||
            !nearly(t.env.log.base_stage_eps, 1.0f, 1e-6f) ||
            !nearly(t.env.log.n, 1.0f, 1e-6f) ||
            !nearly(t.env.log.kill_rate, 1.0f, 1e-6f) ||
            !nearly(t.env.log.avg_stage, 0.0f, 1e-6f) ||
            !nearly(t.env.log.avg_abs_bias, 0.0f, 1e-6f) ||
            !nearly(t.env.log.ultimate, 0.01f, 1e-6f) ||
            !nearly(t.env.log.ultimate2, 0.0f, 1e-6f) ||
            !nearly(t.env.log.total_stage_weight, 0.01f, 1e-6f) ||
            !nearly(t.env.log.total_abs_bias, 0.0f, 1e-6f) ||
            !nearly(t.env.log.total_control_rate, 0.0f, 1e-6f)) {
        printf("terminal_kill: log return=%.9f len=%.9f perf=%.9f score=%.9f shots=%.9f base=%.9f/%.9f ultimate=%.9f [FAIL]\n",
            t.env.log.episode_return, t.env.log.episode_length, t.env.log.perf,
            t.env.log.score, t.env.log.shots_fired,
            t.env.log.base_stage_kills, t.env.log.base_stage_eps,
            t.env.log.ultimate);
        return 1;
    }

    printf("terminal_kill: player kill reward, reset flags, and log counters match Dogfight3 fixture [OK]\n");
    return 0;
}

int main(void) {
    return test_terminal_player_kill_matches_dogfight3_fixture();
}
