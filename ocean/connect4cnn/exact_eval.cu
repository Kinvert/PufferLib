// Included after eval_make. Opt-in Connect4 measurement, outside the learner.
#include "exact_protocol.h"

static void c4_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr,"Connect4 exact evaluation: %s\n",message); exit(1); }
}
static uint32_t c4_exact_option(Ini* ini, const char* key, uint32_t lo, uint32_t hi) {
    double value = puf_ini_get(ini,"eval_exact",key);
    c4_exact_require(isfinite(value) && value >= lo && value <= hi && value == floor(value),key);
    return (uint32_t)value;
}
__global__ void c4_exact_rng(curandStatePhilox4_32_10_t* states,
        uint32_t seed, uint32_t first, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) curand_init(c4_exact_policy_seed(seed,first+i),0,0,&states[i]);
}

static void run_connect4_exact_eval(Ini* ini, TrainContext* ctx) {
    c4_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU,"requires float32 and native CPU environment");
    c4_exact_require(ctx->world_size == 1,"single GPU required");
    c4_exact_require(puf_ini_get(ini,"selfplay","enabled") == 0,"selfplay must be disabled");
    c4_exact_require(puf_ini_get(ini,"env","player_pieces") == 0
        && puf_ini_get(ini,"env","env_pieces") == 0,"requires empty starting boards");
    uint32_t episodes = c4_exact_option(ini,"episodes",1,1000000);
    uint32_t first = c4_exact_option(ini,"episode_offset",0,UINT32_MAX);
    uint32_t seed = c4_exact_option(ini,"seed",0,UINT32_MAX);
    uint32_t slots = c4_exact_option(ini,"slots",1,1024);
    c4_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX,"episode IDs overflow");
    const char* output = puf_ini_get_str(ini,"eval_exact","output");
    c4_exact_require(output && output[0] && strcmp(output,"None"),"explicit output CSV required");
    // Refuse clobbering a partial/failed evaluation. Its records remain evidence.
    FILE* fp = fopen(output,"wx");
    c4_exact_require(fp != NULL,"output exists or cannot be created");
    fprintf(fp,"version,block_seed,episode_id,slot,representation,env_seed,policy_seed,decisions,win,score,invalid,action_hash\n");
    c4_exact_require(fflush(fp) == 0,"cannot write CSV header");

    // Evaluation-only allocation. One inference step, no optimizer updates.
    char count[32]; snprintf(count,sizeof(count),"%u",slots);
    puf_ini_put(ini,"base.async","0");
    puf_ini_put(ini,"base.eval_agents",count);
    puf_ini_put(ini,"vec.total_agents",count);
    puf_ini_put(ini,"vec.num_buffers","1");
    puf_ini_put(ini,"vec.num_policies","1");
    puf_ini_put(ini,"vec.hist_policy_percent","0");
    puf_ini_put(ini,"train.horizon","1");
    puf_ini_put(ini,"train.minibatch_size",count);
    puf_ini_put(ini,"train.replay_ratio","1");
    const char* checkpoint = puf_ini_get_str(ini,"base","load_model_path");
    c4_exact_require(checkpoint && strcmp(checkpoint,"None"),"explicit checkpoint required");
    PuffeRL* p = eval_make(ini,ctx,EVAL_SCORE,0);
    VecEnv* v = p->vec;
    c4_exact_require(v->size == (int)slots && v->total_agents == (int)slots,"one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape);
    struct stat st;
    c4_exact_require(stat(checkpoint,&st) == 0 && st.st_size == parameters*(long)sizeof(float),"checkpoint size mismatch");
    FILE* weights = fopen(checkpoint,"rb");
    c4_exact_require(weights != NULL,"cannot audit checkpoint");
    float value;
    for (long j = 0; j < parameters; j++)
        c4_exact_require(fread(&value,sizeof(value),1,weights) == 1 && isfinite(value),"nonfinite or truncated checkpoint");
    c4_exact_require(fclose(weights) == 0,"checkpoint read failed");
    cudaStream_t stream = p->default_stream;
    unsigned char* active = (unsigned char*)calloc(slots,1);
    uint64_t* action_hash = (uint64_t*)calloc(slots,sizeof(uint64_t));
    c4_exact_require(active && action_hash,"allocation failed");
    uint32_t completed = 0, wins = 0;
    // Waves give each assigned episode its own RNG and fresh recurrent state.
    // Finished slots never step again; dummy inference cannot add extra games.
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots,episodes-base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            env->rng = c4_exact_env_seed(seed,first+base+i);
            env->log = {};
            puf_reset(env);
            active[i] = i < n;
            action_hash[i] = UINT64_C(14695981039346656037);
            v->terminals[i] = 1; // Exercise the production recurrent-reset path.
        }
        c4_exact_rng<<<grid_size(slots),BLOCK_SIZE,0,stream>>>(p->rng_states[0],seed,first+base,slots);
        uint32_t remaining = n;
        for (int decision = 1; decision <= 21 && remaining; decision++) {
            cpu_upload(p,0,slots,stream);
            pufferl_forward(p,0,0,stream);
            c4_exact_require(cudaMemcpyAsync(v->actions,p->env.actions.data,
                slots*sizeof(float),cudaMemcpyDeviceToHost,stream) == cudaSuccess,"action download failed");
            c4_exact_require(cudaStreamSynchronize(stream) == cudaSuccess,"GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i];
                float action = v->actions[i];
                c4_exact_require(isfinite(action) && action >= 0 && action < 7 && action == floorf(action),"invalid policy action");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action)*UINT64_C(1099511628211);
                puf_step(env);
                if (!v->terminals[i]) continue;
                float score = v->rewards[i];
                c4_exact_require((score == -1 || score == 0 || score == 1)
                    && env->log.n == 1 && env->log.episode_length == decision
                    && env->log.score == score && (env->log.invalids == 0 || env->log.invalids == 1),"terminal accounting mismatch");
                uint32_t id = first+base+i;
                fprintf(fp,"1,%u,%u,%u,%d,%u,%llu,%d,%d,%.0f,%.0f,%016llx\n",
                    seed,id,i,env->representation,c4_exact_env_seed(seed,id),
                    (unsigned long long)c4_exact_policy_seed(seed,id),decision,score == 1,
                    score,env->log.invalids,(unsigned long long)action_hash[i]);
                c4_exact_require(fflush(fp) == 0,"episode receipt write failed");
                active[i] = 0; remaining--; completed++; wins += score == 1;
            }
        }
        c4_exact_require(remaining == 0,"assigned game exceeded 21 decisions");
    }
    c4_exact_require(completed == episodes && fclose(fp) == 0,"incomplete evaluation");
    printf("CONNECT4_EXACT_EVAL version=1 requested=%u completed=%u wins=%u params=%ld\n",episodes,completed,wins,parameters);
    free(active); free(action_hash); close_pufferl(p);
}
