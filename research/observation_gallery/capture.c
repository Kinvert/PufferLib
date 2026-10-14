// Temporary observation export. Native game steps and native observation writer;
// no policy, GPU, Raylib rendering/window, or fabricated game state.
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include ENV_HEADER

static void observe(Env* e) {
#ifdef PUFFER_CONNECT4CNN
    compute_observation(e);
#elif defined(PUFFER_FLAPPYCNN)
    flappy_compute_observations(e);
#else
    compute_observations(e);
#endif
}

static uint64_t fingerprint(const void* data, size_t count) {
    const unsigned char* bytes = data;
    uint64_t value = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; i++) value = (value ^ bytes[i]) * UINT64_C(1099511628211);
    return value;
}

#ifdef PUFFER_MAZECNN
static int goal_distance(Env* e) {
    int queue[MAX_SIZE*MAX_SIZE], distance[MAX_SIZE*MAX_SIZE];
    for (int i=0;i<MAX_SIZE*MAX_SIZE;i++) distance[i]=-1;
    int head=maze_offset(e->state.y,e->state.x),start=0,end=1;
    queue[0]=head;distance[head]=0;
    int dx[]={1,0,-1,0},dy[]={0,-1,0,1};
    while(start<end) {
        int at=queue[start++];
        if(e->state.maze[at]==GOAL) return distance[at];
        for(int a=0;a<4;a++) {
            int x=at%MAX_SIZE+dx[a],y=at/MAX_SIZE+dy[a];
            if(x<0||y<0||x>=e->state.width||y>=e->state.height)continue;
            int to=maze_offset(y,x);
            if(distance[to]>=0||e->state.maze[to]==WALL)continue;
            distance[to]=distance[at]+1;queue[end++]=to;
        }
    }
    return -1;
}
#endif

static int action_for(Env* e, int step) {
#ifdef PUFFER_CONNECT4CNN
    // Reuse the native opponent's rule controller on a shallow player-swapped
    // copy. The real environment RNG/state is untouched when selecting actions.
    Env view = *e;
    view.player_pieces = e->env_pieces;
    view.env_pieces = e->player_pieces;
    return compute_env_move(&view);
#elif defined(PUFFER_PONGCNN)
    float delta = e->ball_y + e->ball_height / 2 - (e->paddle_yr + e->paddle_height / 2);
    return fabsf(delta) < e->paddle_speed * e->frameskip / 2 ? 0 : delta > 0 ? 1 : 2;
#elif defined(PUFFER_FLAPPYCNN)
    Pipe *next, *ignored;
    flappy_next_pipes(e, &next, &ignored);
    float target = next ? next->gap_y : e->height / 2;
    return e->bird_y > target + 18 && e->bird_vy > -1;
#elif defined(PUFFER_BREAKOUTCNN)
    float delta = e->ball_x + e->ball_width / 2 - (e->paddle_x + e->paddle_width / 2);
    return fabsf(delta) < 4 ? NOOP : delta > 0 ? RIGHT : LEFT;
#elif defined(PUFFER_SNAKECNN)
    int count = e->width * e->height, queue[count], parent[count], first[count];
    for (int i = 0; i < count; i++) parent[i] = -1;
    int head = e->snake[e->head_ptr], start = 0, end = 1;
    int delta[] = {-e->width, e->width, -1, 1};
    queue[0] = head; parent[head] = head;
    while (start < end) {
        int at = queue[start++];
        if (e->grid[at] == SNAKE_FOOD) return first[at];
        for (int a = 0; a < 4; a++) {
            int to = at + delta[a];
            if (to < 0 || to >= count || parent[to] != -1 || e->grid[to] >= SNAKE_WALL) continue;
            parent[to] = at; first[to] = at == head ? a : first[at]; queue[end++] = to;
        }
    }
    for (int a = 0; a < 4; a++) if (e->grid[head + delta[a]] < SNAKE_WALL) return a;
    return 0;
#elif defined(PUFFER_FORAGECNN)
    int queue[FORAGE_AREA],first[FORAGE_AREA],head=0,tail=1;
    unsigned char visited[FORAGE_AREA]={0};
    int delta[]={0,-FORAGE_WIDTH,FORAGE_WIDTH,-1,1};
    queue[0]=e->player;visited[e->player]=1;
    while(head<tail) {
        int cell=queue[head++],kind=e->grid[cell];
        if(kind>=FORAGE_APPLE&&kind<=FORAGE_MUSHROOM)return first[cell];
        for(int a=1;a<=4;a++) {
            int to=cell+delta[a];
            if(to<0||to>=FORAGE_AREA||visited[to]||e->grid[to]>FORAGE_MUSHROOM)continue;
            visited[to]=1;first[to]=cell==e->player?a:first[cell];queue[tail++]=to;
        }
    }
    return 0;
#else
    int count = MAX_SIZE * MAX_SIZE, queue[count], parent[count], first[count];
    for (int i = 0; i < count; i++) parent[i] = -1;
    int head = maze_offset(e->state.y, e->state.x), start = 0, end = 1;
    int dx[] = {0,1,0,-1,0}, dy[] = {0,0,-1,0,1};
    queue[0] = head; parent[head] = head;
    while (start < end) {
        int at = queue[start++];
        if (e->state.maze[at] == GOAL) return first[at];
        for (int a = 1; a <= 4; a++) {
            int x = at % MAX_SIZE + dx[a], y = at / MAX_SIZE + dy[a];
            if (x < 0 || y < 0 || x >= e->state.width || y >= e->state.height) continue;
            int to = maze_offset(y, x);
            if (parent[to] != -1 || e->state.maze[to] == WALL) continue;
            parent[to] = at; first[to] = at == head ? a : first[at]; queue[end++] = to;
        }
    }
    return ATN_PASS;
#endif
}

static void describe_state(Env* e) {
#ifdef PUFFER_CONNECT4CNN
    printf("\"pieces\":%d,\"player_bits\":\"%" PRIu64 "\",\"opponent_bits\":\"%" PRIu64 "\"",
        __builtin_popcountll(e->player_pieces | e->env_pieces),e->player_pieces,e->env_pieces);
#elif defined(PUFFER_PONGCNN)
    printf("\"ball\":[%.9g,%.9g],\"paddles\":[%.9g,%.9g],\"score\":[%u,%u]",
        e->ball_x,e->ball_y,e->paddle_yl,e->paddle_yr,e->score_l,e->score_r);
#elif defined(PUFFER_FLAPPYCNN)
    printf("\"bird_y\":%.9g,\"score\":%d,\"pipes\":[",e->bird_y,e->score);
    for (int i=0;i<FLAPPY_NUM_PIPES;i++) printf("%s[%.9g,%.9g]",i?",":"",e->pipes[i].x,e->pipes[i].gap_y);
    printf("]");
#elif defined(PUFFER_BREAKOUTCNN)
    int remaining=0; for (int i=0;i<e->num_bricks;i++) remaining += e->brick_states[i] != 1;
    printf("\"ball\":[%.9g,%.9g],\"paddle_x\":%.9g,\"score\":%d,\"remaining_bricks\":%d,\"lives\":%d",
        e->ball_x,e->ball_y,e->paddle_x,e->score,remaining,e->num_balls);
#elif defined(PUFFER_SNAKECNN)
    printf("\"head\":[%d,%d],\"length\":%d,\"food_collected\":%d,\"grid_hash\":\"%016" PRIx64 "\"",
        e->snake[e->head_ptr]%e->width,e->snake[e->head_ptr]/e->width,e->length,e->food_collected,
        fingerprint(e->grid,e->width*e->height));
#elif defined(PUFFER_FORAGECNN)
    printf("\"player\":[%d,%d],\"positive_collected\":%d,\"remaining\":%d,\"episode_return\":%.9g,\"grid_hash\":\"%016" PRIx64 "\"",
        e->player%FORAGE_WIDTH,e->player/FORAGE_WIDTH,e->positives,e->remaining,e->episode_return,
        fingerprint(e->grid,sizeof(e->grid)));
#else
    printf("\"agent\":[%d,%d],\"map_size\":[%d,%d],\"map_hash\":\"%016" PRIx64 "\"",
        e->state.x,e->state.y,e->state.width,e->state.height,fingerprint(e->state.maze,sizeof(e->state.maze)));
#endif
}

int main(int argc, char** argv) {
    assert(argc == 3);
    int representation = atoi(argv[2]);
    assert(representation >= 0 && representation < GALLERY_REPS);
    Ini ini = {0};
    puf_ini_load_file(&ini,"config/default.ini");
    puf_ini_load_file(&ini,argv[1]);
    Dict* kwargs = puf_ini_section(&ini,"env",0);
    dict_set(kwargs,"representation",representation);
    dict_set(kwargs,"representation_mode",0);
    Env* e = calloc(1,sizeof(Env));
    float* observation=calloc(OBS_SIZE,sizeof(float));
    float* repeat=calloc(OBS_SIZE,sizeof(float));
    assert(e && observation && repeat);
    float action=0,reward=0,terminal=0;
    e->rng=7;
    e->agents[0].observations=observation;e->agents[0].actions=&action;
    e->agents[0].rewards=&reward;e->agents[0].terminals=&terminal;
    puf_init(e,kwargs);puf_reset(e);
    int targets[3] = {GALLERY_T1,GALLERY_T2,GALLERY_T3};
#ifdef PUFFER_MAZECNN
    int distance=goal_distance(e);assert(distance>=4);
    targets[0]=distance/4;targets[1]=distance/2;targets[2]=3*distance/4;
#endif
    int consecutive=0, episode=0, exported=0;
    for (int step=1;step<=2000 && exported<3;step++) {
        action=action_for(e,step);puf_step(e);consecutive++;
        if (terminal) { episode++;consecutive=0;continue; }
        if (consecutive!=targets[exported]) continue;
        Env before=*e;
        observe(e);assert(!memcmp(&before,e,sizeof(Env)));
        for(int i=0;i<OBS_SIZE;i++) assert(isfinite(observation[i]) && observation[i]>=0 && observation[i]<=1);
        memcpy(repeat,observation,OBS_SIZE*sizeof(float));
        observe(e);assert(!memcmp(repeat,observation,OBS_SIZE*sizeof(float)));
        assert(!memcmp(&before,e,sizeof(Env)));
        printf("{\"representation\":%d,\"moment\":%d,\"shape_hwc\":[%d,%d,%d],\"seed\":7,\"decision\":%d,\"episode\":%d,\"episode_decision\":%d,\"tick\":%d,\"rng\":%u,\"repeat_exact\":true,\"observation_fnv64\":\"%016" PRIx64 "\",\"state\":{",
            representation,exported,OBS_HEIGHT,OBS_WIDTH,OBS_CHANNELS,step,episode,consecutive,e->tick,e->rng,
            fingerprint(observation,OBS_SIZE*sizeof(float)));
        describe_state(e);printf("}}\n");
        assert(fwrite(observation,sizeof(float),OBS_SIZE,stdout)==OBS_SIZE);fflush(stdout);
        exported++;
    }
    if (exported != 3) fprintf(stderr,"Only %d/3 progressed snapshots captured\n",exported);
    puf_close(e);free(repeat);free(observation);free(e);puf_ini_free(&ini);
    return exported==3?0:1;
}
