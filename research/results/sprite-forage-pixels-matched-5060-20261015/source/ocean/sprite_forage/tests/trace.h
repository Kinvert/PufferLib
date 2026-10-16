// Test-only native CUDA evaluator: record every applied action and game state.
// Run with exactly one state agent, default geometry, and FORAGE_TRACE_FILE set.
#include <assert.h>
#define puf_step forage_original_step
#include "../sprite_forage.h"
#undef puf_step
#include "trace_types.h"

void puf_step(Env* env) {
    static FILE* file;
    static int index;
    if (!file) {
        const char* path = getenv("FORAGE_TRACE_FILE");
        assert(path && env->observation_mode == 0 && env->copies == 3 && env->trees == 8);
        file = fopen(path, "wbx");
        assert(file);
    }
    ForageTrace row = {};
    row.index = index++;
    row.action = env->agents[0].actions[0];
    forage_snapshot(env, &row.before);
    forage_original_step(env);
    forage_snapshot(env, &row.after);
    memcpy(row.observation, env->agents[0].observations, sizeof(row.observation));
    assert(fwrite(&row, sizeof(row), 1, file) == 1);
}
