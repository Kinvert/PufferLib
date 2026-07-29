#include <math.h>
#include <stdio.h>

#include "dogfight.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        return 1; \
    } \
} while (0)

static int close_enough(float actual, float expected) {
    return fabsf(actual - expected) < 1.0e-6f;
}

int main(void) {
    Dict kwargs = {0};
    dict_set(&kwargs, "curriculum_enabled", 1.0);
    dict_set(&kwargs, "curriculum_target", 0.9);
    dict_set(&kwargs, "warmup_steps", 100.0);
    dict_set(&kwargs, "eval_interval", 50.0);
    dict_set(&kwargs, "min_eval_episodes", 5.0);
    dict_set(&kwargs, "max_stage", 2.0);

    PufCurriculumState state;
    puf_curriculum_init(&state, &kwargs, 1000);
    CHECK(close_enough(state.target, 0.9f), "initial target must match Dogfight3");
    CHECK(state.mastered_stage == -1, "no stage starts mastered");

    Env envs[2] = {0};
    puf_set_global_step(&envs[0], 1234);
    CHECK(envs[0].global_step == 1234, "global step reaches the environment");

    CHECK(!puf_curriculum_observe(&state, 99, 5.0, 5.0),
        "pre-warmup results must not trigger evaluation");
    CHECK(state.base_stage_eps == 0.0,
        "pre-warmup results must not enter the mastery window");

    CHECK(puf_curriculum_observe(&state, 150, 5.0, 5.0),
        "first interval must evaluate");
    CHECK(state.mastered_stage == 1, "90 percent gate must master stage 1");
    CHECK(close_enough(state.target, 1.9f),
        "stage 1 mastery must expose stage 2 at 90 percent");

    CHECK(puf_curriculum_observe(&state, 200, 4.0, 5.0),
        "second interval must evaluate");
    CHECK(state.mastered_stage == 1,
        "sub-90 percent performance must not advance");
    CHECK(close_enough(state.target, 1.9f),
        "failed mastery must retain the current target");

    CHECK(puf_curriculum_observe(&state, 250, 10.0, 10.0),
        "third interval must evaluate");
    CHECK(state.mastered_stage == 2, "recovered window must master stage 2");
    CHECK(close_enough(state.target, 2.0f),
        "target must respect the configured stage cap");

    PufCurriculumState sparse;
    puf_curriculum_init(&sparse, &kwargs, 1000);
    CHECK(!puf_curriculum_observe(&sparse, 150, 4.0, 4.0),
        "an interval with too few episodes must wait for evidence");
    CHECK(close_enough(sparse.target, 0.9f),
        "insufficient evidence must not retreat the curriculum");
    CHECK(puf_curriculum_observe(&sparse, 151, 1.0, 1.0),
        "evaluation must run as soon as the evidence floor is reached");
    CHECK(sparse.mastered_stage == 1,
        "the completed sparse window must still master its stage");

    PufCurriculumState fallback;
    puf_curriculum_init(&fallback, &kwargs, 1000);
    CHECK(puf_curriculum_observe(&fallback, 150, 0.0, 5.0),
        "failed initial window must evaluate");
    CHECK(close_enough(fallback.target, 0.9f),
        "failed local mastery must retain the current target");
    puf_curriculum_apply(&fallback, envs, 2);
    CHECK(close_enough(envs[0].curriculum_target, 0.9f),
        "failed local mastery must not lower future spawn difficulty");
    CHECK(close_enough(envs[1].curriculum_target, 0.9f),
        "retained curriculum target must apply consistently");

    double observed = 0.0;
    CHECK(puf_curriculum_delta(5.0, &observed) == 5.0,
        "first non-clearing poll must count all new episodes");
    CHECK(puf_curriculum_delta(5.0, &observed) == 0.0,
        "repeated non-clearing poll must not double-count");
    CHECK(puf_curriculum_delta(2.0, &observed) == 0.0,
        "counter decreases must not infer a wall-clock clear");
    state.observed_base_stage_kills = observed;
    state.observed_base_stage_eps = observed;
    puf_curriculum_counters_cleared(&state);
    CHECK(state.observed_base_stage_kills == 0.0 &&
            state.observed_base_stage_eps == 0.0,
        "dashboard clear must explicitly reset observed counters");
    observed = state.observed_base_stage_eps;
    CHECK(puf_curriculum_delta(2.0, &observed) == 2.0,
        "new window after explicit clear must count all episodes");

    dict_set(&kwargs, "fixed_stage", 2.0);
    PufCurriculumState fixed;
    puf_curriculum_init(&fixed, &kwargs, 1000);
    CHECK(close_enough(fixed.target, 2.0f), "fixed stage must override target");
    CHECK(!puf_curriculum_observe(&fixed, 1000, 100.0, 100.0),
        "fixed-stage eval must not run mastery progression");

    Dict local_kwargs = {0};
    dict_set(&local_kwargs, "curriculum_enabled", 1.0);
    dict_set(&local_kwargs, "curriculum_target", 0.9);
    dict_set(&local_kwargs, "warmup_steps", 100.0);
    dict_set(&local_kwargs, "eval_interval", 50.0);
    dict_set(&local_kwargs, "min_eval_episodes", 5.0);
    dict_set(&local_kwargs, "mastery_threshold", 0.60);
    dict_set(&local_kwargs, "max_stage", 2.0);

    Env local = {0};
    local.curriculum_enabled = 1;
    local.curriculum_target = 0.9f;
    local.global_step_stride = 1024;
    puf_curriculum_init(&local.local_curriculum, &local_kwargs, 1000);

    dogfight_advance_local_global_step(&local);
    CHECK(local.global_step == 1024,
        "Dogfight-local clock must replace the removed runner callback");

    local.global_step = 100;
    for (int episode = 1; episode <= 5; episode++) {
        local.global_step = 100 + episode * 10;
        dogfight_local_curriculum_episode(&local, 1, episode <= 3);
    }
    CHECK(local.local_curriculum.mastered_stage == 1,
        "local 60 percent mastery window must advance stage 1");
    CHECK(close_enough(local.curriculum_target, 1.9f),
        "local mastery must immediately update future spawn targets");

    dict_clear(&local_kwargs);
    dict_clear(&kwargs);
    return 0;
}
