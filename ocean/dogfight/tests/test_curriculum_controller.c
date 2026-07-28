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

    PufCurriculumState fallback;
    puf_curriculum_init(&fallback, &kwargs, 1000);
    CHECK(puf_curriculum_observe(&fallback, 150, 0.0, 5.0),
        "failed initial window must evaluate");
    CHECK(close_enough(fallback.target, -0.1f),
        "unmastered donor controller must retreat toward stage 0");
    puf_curriculum_apply(&fallback, envs, 2);
    CHECK(close_enough(envs[0].curriculum_target, 0.0f),
        "environment must clamp a retreat to stage 0");
    CHECK(close_enough(envs[1].curriculum_target, 0.0f),
        "curriculum target must be global across environments");

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

    dict_clear(&kwargs);
    return 0;
}
