// Native configuration arithmetic only: no policy, images, weights or CUDA.
// Scope: float32, one trainable policy, no selfplay/historical agents.
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include "../src/ini.h"

static void require(int ok, const char* message) {
    if (!ok) { fprintf(stderr, "%s\n", message); exit(1); }
}

static double number(Ini* ini, const char* section, const char* key) {
    double value;
    const char* raw = puf_ini_get_str(ini, section, key);
    require(puf_ini_parse_val(raw, &value) && isfinite(value), "Need a finite scalar setting");
    return value;
}

static uint64_t integer(Ini* ini, const char* section, const char* key, uint64_t limit) {
    double value = number(ini, section, key);
    require(value >= 0 && value <= (double)limit && value == floor(value), "Invalid integer setting");
    return (uint64_t)value;
}

static uint64_t product(uint64_t left, uint64_t right) {
    require(!right || left <= UINT64_MAX/right, "Geometry count overflow");
    return left*right;
}

static float schedule(float base, float min_value, uint64_t epoch, uint64_t epochs) {
    double u = (double)epoch/(double)epochs;
    return min_value+0.5f*(base-min_value)*(1.0f+(float)cos(3.14159265358979323846*u));
}

int main(int argc, char** argv) {
    require(argc > 1, "Usage: learner_geometry FULL.ini [OVERLAY.ini ...]");
    Ini ini = {0};
    for (int i = 1; i < argc; i++) puf_ini_load_file(&ini, argv[i]);
    const char* environment = puf_ini_get_str(&ini, "base", "env_name");
    require(environment[0] != 0, "Need an environment name");
    for (const char* p = environment; *p; p++)
        require(isalnum((unsigned char)*p) || *p == '_' || *p == '-', "Invalid environment name");
    require(integer(&ini, "vec", "num_policies", INT_MAX) == 1
        && number(&ini, "vec", "hist_policy_percent") == 0
        && integer(&ini, "selfplay", "enabled", 1) == 0, "Only single-policy/no-selfplay geometry is supported");
    uint64_t agents = integer(&ini, "vec", "total_agents", INT_MAX);
    uint64_t buffers = integer(&ini, "vec", "num_buffers", INT_MAX);
    uint64_t horizon = integer(&ini, "train", "horizon", INT_MAX);
    uint64_t minibatch = integer(&ini, "train", "minibatch_size", INT_MAX);
    uint64_t gpus = integer(&ini, "train", "gpus", INT_MAX);
    uint64_t requested = integer(&ini, "train", "total_timesteps", UINT64_C(9007199254740991));
    uint64_t cadence = integer(&ini, "base", "checkpoint_interval", INT_MAX);
    require(agents && buffers && agents%buffers == 0 && horizon && horizon%4 == 0
        && minibatch && minibatch%horizon == 0 && gpus && requested, "Invalid float32 rollout/minibatch geometry");
    uint64_t batch = product(agents, horizon);
    require(batch <= INT_MAX && minibatch <= batch && agents%(minibatch/horizon) == 0, "Native learner shape constraints fail");
    float replay = (float)number(&ini, "train", "replay_ratio");
    require(isfinite(replay) && replay >= 0, "Invalid replay ratio");
    // Match the native float product/division and truncation to int, not
    // double arithmetic or independently rounded replay values.
    float update_product = replay*(float)batch;
    float raw_updates = update_product/(float)minibatch;
    require(isfinite(raw_updates) && (double)raw_updates <= INT_MAX, "Native optimizer minibatch count overflows int");
    uint64_t updates = (int)raw_updates;
    require(!updates || product(updates-1, minibatch/horizon) <= INT_MAX,
        "Native minibatch row-offset multiplication overflows int");
    uint64_t slices = batch/minibatch;
    uint64_t passes = updates/slices, prefix = updates%slices;
    uint64_t epochs = requested/gpus/batch;
    require(epochs < INT_MAX, "Native epoch/log-history count overflows int");
    uint64_t local_decisions = product(epochs, batch);
    uint64_t actual = product(local_decisions, gpus);
    uint64_t total_updates = product(epochs, updates);
    uint64_t processed = product(product(total_updates, minibatch), gpus);
    uint64_t checkpoints = epochs ? (cadence ? epochs/cadence+(epochs%cadence != 0) : 1) : 0;
    float lr = (float)number(&ini, "train", "learning_rate");
    float min_ratio = (float)number(&ini, "train", "min_lr_ratio");
    int anneal = integer(&ini, "train", "anneal_lr", 1);
    require(isfinite(lr) && lr >= 0 && isfinite(min_ratio) && min_ratio >= 0 && min_ratio <= 1, "Invalid LR schedule");
    float first_lr = epochs && anneal ? schedule(lr, min_ratio*lr, 0, epochs) : lr;
    float last_lr = epochs && anneal ? schedule(lr, min_ratio*lr, epochs-1, epochs) : lr;
    printf("{\"protocol\":\"native-learner-geometry-v1\",\"environment\":\"%s\","
        "\"float32_geometry\":true,\"policy_executed\":false,\"gpu_runtime_validated\":false,"
        "\"agents_per_rank\":%" PRIu64 ",\"buffers\":%" PRIu64 ",\"agents_per_buffer\":%" PRIu64 ","
        "\"horizon\":%" PRIu64 ",\"minibatch_decisions\":%" PRIu64 ",\"minibatch_rows\":%" PRIu64 ","
        "\"world_size\":%" PRIu64 ",\"rollout_decisions_per_rank\":%" PRIu64 ","
        "\"requested_global_decisions\":%" PRIu64 ",\"actual_global_decisions\":%" PRIu64 ","
        "\"dropped_requested_decisions\":%" PRIu64 ",\"epochs\":%" PRIu64 ","
        "\"replay_ratio_f32\":%.9g,\"untruncated_minibatches_f32\":%.9g,\"optimizer_minibatches_per_epoch\":%" PRIu64 ","
        "\"optimizer_updates_per_rank\":%" PRIu64 ",\"processed_training_decisions_all_ranks\":%" PRIu64 ","
        "\"effective_replay_ratio\":%.17g,\"optimizer_updates_per_1000_local_decisions\":%.17g,"
        "\"distinct_slices_per_rollout\":%" PRIu64 ",\"complete_passes_per_rollout\":%" PRIu64 ","
        "\"additional_prefix_slices\":%" PRIu64 ",\"rows_covered_per_rollout\":%" PRIu64 ","
        "\"coverage_is_prefix_not_random\":true,\"nonzero_training_updates\":%s,"
        "\"checkpoint_count\":%" PRIu64 ",\"checkpoint_interval_epochs\":%" PRIu64 ","
        "\"anneal_lr\":%s,\"lr_base_f32\":%.9g,\"first_epoch_lr_f32\":%.9g,\"last_epoch_lr_f32\":%.9g,"
        "\"lr_schedule_evaluated\":%s}\n",
        environment, agents, buffers, agents/buffers, horizon, minibatch, minibatch/horizon,
        gpus, batch, requested, actual, requested-actual, epochs, replay, raw_updates, updates,
        total_updates, processed, (double)updates*minibatch/batch, 1000.0*updates/batch,
        slices, passes, prefix, updates >= slices ? agents : updates*(minibatch/horizon),
        epochs && updates ? "true" : "false", checkpoints, cadence, anneal ? "true" : "false",
        lr, first_lr, last_lr, epochs ? "true" : "false");
    puf_ini_free(&ini);
    return 0;
}
