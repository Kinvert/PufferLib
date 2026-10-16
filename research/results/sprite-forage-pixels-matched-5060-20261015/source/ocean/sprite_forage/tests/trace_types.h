// Local binary receipts use their captured source layout. Zero padding before comparisons.
typedef struct {
    unsigned int rng;
    int player, tick, apples, bad;
    float episode_return, reward, terminal;
    Log log;
    unsigned char grid[FORAGE_AREA];
} ForageSnapshot;

typedef struct {
    int index;
    float action;
    ForageSnapshot before, after;
    float observation[116];
} ForageTrace;

void forage_snapshot(Env* env, ForageSnapshot* snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->rng = env->rng;
    snapshot->player = env->player_idx;
    snapshot->tick = env->tick;
    snapshot->apples = env->apples;
    snapshot->bad = env->negatives;
    snapshot->episode_return = env->episode_return;
    snapshot->reward = env->agents[0].rewards[0];
    snapshot->terminal = env->agents[0].terminals[0];
    snapshot->log = env->log;
    memcpy(snapshot->grid, env->grid, FORAGE_AREA);
}
