#include <assert.h>

#include "../dogfight.h"

int main(void) {
    Dogfight env = {0};
    RewardConfig reward_config = {0};
    float observations[256] = {0};

    init(&env, OBS_OPPONENT_AWARE, &reward_config, 1, 0, 0);
    env.observations = observations;
    dogfight_bind_rng(&env.rng);
    set_curriculum_target(&env, 3.0f);
    c_reset(&env);

    assert(env.stage == (CurriculumStage)3);
    return 0;
}
