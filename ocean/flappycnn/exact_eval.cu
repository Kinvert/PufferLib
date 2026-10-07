// Included after eval_make. Evaluation only; learner/environment rules unchanged.
#include "exact_episode.h"
#include <float.h>

static void flappy_exact_require(bool ok, const char* message) {
    if (!ok) { fprintf(stderr,"Flappy exact evaluation: %s\n",message); exit(1); }
}
static uint32_t flappy_exact_option(Ini* ini, const char* key, uint32_t lo, uint32_t hi) {
    double value = puf_ini_get(ini,"eval_exact",key);
    flappy_exact_require(isfinite(value) && value >= lo && value <= hi && value == floor(value),key);
    return (uint32_t)value;
}
static void flappy_exact_world(Ini* ini) {
    const char* keys[] = {"width","height","max_steps","gravity","flap_velocity","pipe_speed",
        "pipe_gap","pipe_width","pipe_spacing","first_pipe_x","bird_x","bird_radius",
        "alive_reward","pass_reward","crash_reward","center_reward"};
    for (int i = 0; i < 16; i++) {
        double value = puf_ini_get(ini,"env",keys[i]);
        flappy_exact_require(isfinite(value) && fabs(value) <= FLT_MAX,"nonfinite/overflowing world parameter");
        if (i < 3) flappy_exact_require(value >= 1 && value <= 1000000 && value == floor(value),keys[i]);
    }
    const char* positive[] = {"gravity","pipe_speed","pipe_gap","pipe_width","pipe_spacing","bird_radius"};
    for (int i = 0; i < 6; i++) flappy_exact_require(puf_ini_get(ini,"env",positive[i]) > 0,positive[i]);
    flappy_exact_require(puf_ini_get(ini,"env","pipe_gap") < puf_ini_get(ini,"env","height"),"pipe gap exceeds world");
}
static void flappy_exact_info() {
#ifdef PUFFER_FLAPPY
    const char* encoder = "state";
#elif defined(C4_IMPOOLA_CNN)
    const char* encoder = "impoola";
#elif defined(C4_IMPALA_CNN)
    const char* encoder = "impala";
#elif defined(C4_NATURE_CNN)
    const char* encoder = "nature";
#else
    const char* encoder = "tiny";
#endif
    printf("{\"environment\":\"%s\",\"default_encoder\":\"%s\",\"float32\":%s,"
        "\"rules\":\"flappy-native-v1\",\"receipt_version\":1,\"representation_count\":%d}\n",
        PUFFER_ENV_NAME,encoder,USE_BF16 ? "false" : "true",
#ifdef PUFFER_FLAPPYCNN
        FLAPPYCNN_NUM_REPRESENTATIONS
#else
        1
#endif
    );
}
// Manifest uses this binary's actual host math and rasterization, no policy/CUDA.
static void flappy_exact_manifest(int argc, char** argv) {
    flappy_exact_require(argc == 3,"eval_exact_manifest requires one full INI path");
    Ini ini = {};
    puf_ini_load_file(&ini,argv[2]);
    flappy_exact_require(!USE_BF16,"float32 manifest required");
    flappy_exact_world(&ini);
    uint32_t n = flappy_exact_option(&ini,"episodes",1,1000000);
    uint32_t first = flappy_exact_option(&ini,"episode_offset",0,UINT32_MAX);
    uint32_t seed = flappy_exact_option(&ini,"seed",0,UINT32_MAX);
    uint32_t slots = flappy_exact_option(&ini,"slots",1,1024);
    flappy_exact_require((uint64_t)first+n+slots <= UINT32_MAX,"episode IDs overflow");
    flappy_exact_require(puf_ini_get(&ini,"env","max_steps") >= 1
        && puf_ini_get(&ini,"env","max_steps") <= 1000000,"invalid native cap");
#ifdef PUFFER_FLAPPYCNN
    flappy_exact_require(puf_ini_get(&ini,"env","representation_mode") == 0,"fixed appearance required");
#endif
    Env env = {};
    obs_t observations[OBS_SIZE];
    float action = 0, reward = 0, terminal = 0;
    env.agents[0].observations = observations;
    env.agents[0].actions = &action;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    puf_init(&env,puf_ini_section(&ini,"env",0));
    int representation = 0;
#ifdef PUFFER_FLAPPYCNN
    representation = env.representation;
#endif
    printf("episode_id,env_seed,policy_seed,representation,start_rng,start_world_hash,start_observation_hash\n");
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = first+i;
        flappy_exact_start(&env,seed,id);
        printf("%u,%u,%llu,%d,%u,%016llx,%016llx\n",id,c4_exact_env_seed(seed,id),
            (unsigned long long)c4_exact_policy_seed(seed,id),representation,env.rng,
            (unsigned long long)flappy_exact_world_hash(&env),
            (unsigned long long)flappy_exact_observation_hash(observations));
    }
    puf_close(&env); puf_ini_free(&ini);
}
__global__ void flappy_exact_rng(curandStatePhilox4_32_10_t* states,
        uint32_t seed, uint32_t first, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) curand_init(c4_exact_policy_seed(seed,first+i),0,0,&states[i]);
}
static void run_flappy_exact_eval(Ini* ini, TrainContext* ctx) {
    flappy_exact_require(!USE_BF16 && PUF_BACKEND == PUF_CPU,"requires float32 native CPU environment");
    flappy_exact_require(ctx->world_size == 1 && puf_ini_get(ini,"selfplay","enabled") == 0,
        "single GPU, no selfplay required");
    flappy_exact_world(ini);
#ifdef PUFFER_FLAPPYCNN
    flappy_exact_require(puf_ini_get(ini,"env","representation_mode") == 0,"fixed appearance suite required");
#endif
    uint32_t episodes = flappy_exact_option(ini,"episodes",1,1000000);
    uint32_t first = flappy_exact_option(ini,"episode_offset",0,UINT32_MAX);
    uint32_t seed = flappy_exact_option(ini,"seed",0,UINT32_MAX);
    uint32_t slots = flappy_exact_option(ini,"slots",1,1024);
    uint32_t cap = flappy_exact_option(ini,"max_steps",1,1000000);
    flappy_exact_require(puf_ini_get(ini,"env","max_steps") == cap,"suite/native cap mismatch");
    flappy_exact_require((uint64_t)first+episodes+slots <= UINT32_MAX,"episode IDs overflow");
    const char* output = puf_ini_get_str(ini,"eval_exact","output");
    flappy_exact_require(output && output[0] && strcmp(output,"None"),"explicit output required");
    FILE* fp = fopen(output,"wx");
    flappy_exact_require(fp != NULL,"output exists or cannot be created");
    fprintf(fp,"version,block_seed,episode_id,slot,representation,env_seed,policy_seed,decisions,pipes,perf,episode_return,cap_reached,action_hash,start_observation_hash,start_world_hash,start_rng\n");
    flappy_exact_require(fflush(fp) == 0,"header write failed");
    char count[32]; snprintf(count,sizeof(count),"%u",slots);
    puf_ini_put(ini,"base.async","0");
    puf_ini_put(ini,"base.eval_agents",count);
    puf_ini_put(ini,"vec.total_agents",count);
    puf_ini_put(ini,"vec.num_buffers","1");
    puf_ini_put(ini,"vec.num_threads","1");
    puf_ini_put(ini,"vec.num_policies","1");
    puf_ini_put(ini,"vec.hist_policy_percent","0");
    puf_ini_put(ini,"train.horizon","1");
    puf_ini_put(ini,"train.minibatch_size",count);
    puf_ini_put(ini,"train.replay_ratio","1");
    const char* checkpoint = puf_ini_get_str(ini,"base","load_model_path");
    flappy_exact_require(checkpoint && strcmp(checkpoint,"None"),"explicit checkpoint required");
    PuffeRL* p = eval_make(ini,ctx,EVAL_SCORE,0);
    VecEnv* v = p->vec;
    flappy_exact_require(v->size == (int)slots && v->total_agents == (int)slots,"one agent per slot required");
    long parameters = numel(p->policies[0].master_weights.shape);
    struct stat st;
    flappy_exact_require(stat(checkpoint,&st) == 0 && st.st_size == parameters*(long)sizeof(float),"checkpoint size mismatch");
    FILE* weights = fopen(checkpoint,"rb");
    flappy_exact_require(weights != NULL,"cannot audit checkpoint");
    float value;
    for (long i = 0; i < parameters; i++)
        flappy_exact_require(fread(&value,sizeof(value),1,weights) == 1 && isfinite(value),"nonfinite/truncated weights");
    flappy_exact_require(fclose(weights) == 0,"weights read failed");
    flappy_exact_require(cudaStreamSynchronize(p->default_stream) == cudaSuccess,"checkpoint upload failed");
    cudaStream_t stream = p->streams[0];
    unsigned char* active = (unsigned char*)calloc(slots,1);
    float* returns = (float*)calloc(slots,sizeof(float));
    uint64_t* action_hash = (uint64_t*)calloc(slots,sizeof(uint64_t));
    uint64_t* observation_hash = (uint64_t*)calloc(slots,sizeof(uint64_t));
    uint64_t* world_hash = (uint64_t*)calloc(slots,sizeof(uint64_t));
    uint32_t* start_rng = (uint32_t*)calloc(slots,sizeof(uint32_t));
    flappy_exact_require(active && returns && action_hash && observation_hash && world_hash && start_rng,"allocation failed");
    uint32_t completed = 0, capped = 0;
    uint64_t pipes = 0;
    for (uint32_t base = 0; base < episodes; base += slots) {
        uint32_t n = min(slots,episodes-base);
        for (uint32_t i = 0; i < slots; i++) {
            Env* env = &v->envs[i];
            flappy_exact_start(env,seed,first+base+i);
            flappy_exact_require(env->tick == 0 && env->score == 0 && env->log.n == 0
                && env->bird_y == env->height*0.5f && env->bird_vy == 0,"starting state mismatch");
            observation_hash[i] = flappy_exact_observation_hash(env->agents[0].observations);
            world_hash[i] = flappy_exact_world_hash(env);
            start_rng[i] = env->rng;
            active[i] = i < n; returns[i] = 0;
            action_hash[i] = UINT64_C(14695981039346656037);
        }
        flappy_exact_rng<<<grid_size(slots),BLOCK_SIZE,0,stream>>>(p->rng_states[0],seed,first+base,slots);
        uint32_t remaining = n;
        for (uint32_t decision = 1; decision <= cap && remaining; decision++) {
            cpu_upload(p,0,slots,stream);
            pufferl_forward(p,0,0,stream);
            flappy_exact_require(cudaMemcpyAsync(v->actions,p->env.actions.data,slots*sizeof(float),
                cudaMemcpyDeviceToHost,stream) == cudaSuccess,"action download failed");
            flappy_exact_require(cudaStreamSynchronize(stream) == cudaSuccess,"GPU inference failed");
            for (uint32_t i = 0; i < slots; i++) {
                if (!active[i]) continue;
                Env* env = &v->envs[i];
                float action = v->actions[i];
                flappy_exact_require(action == 0 || action == 1,"invalid policy action");
                action_hash[i] = (action_hash[i] ^ (uint32_t)action)*UINT64_C(1099511628211);
                puf_step(env);
                flappy_exact_require(isfinite(v->rewards[i]),"nonfinite reward");
                returns[i] += v->rewards[i];
                if (!v->terminals[i]) {
                    flappy_exact_require(env->log.n == 0 && env->tick == (int)decision,"nonterminal accounting mismatch");
                    continue;
                }
                flappy_exact_require(v->terminals[i] == 1 && flappy_exact_terminal(env,decision,returns[i]),
                    "terminal accounting mismatch");
                uint32_t id = first+base+i;
                int representation = 0;
#ifdef PUFFER_FLAPPYCNN
                representation = env->representation;
#endif
                fprintf(fp,"1,%u,%u,%u,%d,%u,%llu,%u,%.0f,%a,%a,%d,%016llx,%016llx,%016llx,%u\n",
                    seed,id,i,representation,c4_exact_env_seed(seed,id),
                    (unsigned long long)c4_exact_policy_seed(seed,id),decision,env->log.score,
                    env->log.perf,env->log.episode_return,decision == cap,
                    (unsigned long long)action_hash[i],(unsigned long long)observation_hash[i],
                    (unsigned long long)world_hash[i],start_rng[i]);
                flappy_exact_require(fflush(fp) == 0,"episode receipt write failed");
                active[i] = 0; remaining--; completed++; pipes += (uint32_t)env->log.score;
                capped += decision == cap;
            }
        }
        flappy_exact_require(remaining == 0,"assigned episode exceeded native max_steps");
        printf("FLAPPY_EXACT_PROGRESS completed=%u requested=%u\n",completed,episodes);
    }
    flappy_exact_require(completed == episodes && fclose(fp) == 0,"incomplete evaluation");
    printf("FLAPPY_EXACT_EVAL version=1 requested=%u completed=%u pipes=%llu capped=%u params=%ld\n",
        episodes,completed,(unsigned long long)pipes,capped,parameters);
    free(active); free(returns); free(action_hash); free(observation_hash); free(world_hash); free(start_rng);
    close_pufferl(p);
}
