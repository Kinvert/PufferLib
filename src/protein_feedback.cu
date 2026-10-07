// Temporary research bridge, enabled only by PUFFER_RESEARCH_PROTEIN_FEEDBACK.
// Replays previous native suggestions/observations in a fresh CUDA process.
// The process exits before training/evaluation, leaving no resident GP context.

#define FEEDBACK_MAX_DIMS 64
#define FEEDBACK_MAX_ROWS 512

static void feedback_require(int ok, const char* message) {
    if (!ok) {
        fprintf(stderr, "research protein: %s\n", message);
        exit(1);
    }
}

static double feedback_option(Ini* ini, const char* key, double fallback) {
    Dict* section = puf_ini_section(ini, "protein", 0);
    DictItem* item = section ? dict_find(section, key) : NULL;
    return item ? item->value : fallback;
}

static int research_protein_feedback(int argc, char** argv) {
    int describe = strcmp(argv[1], "research_protein_describe") == 0;
    feedback_require(argc == (describe ? 3 : 5),
        "usage: research_protein_describe RECIPE | research_protein_propose RECIPE HISTORY OUTPUT");
    Ini ini = {0};
    puf_ini_load_file(&ini, argv[2]);
    feedback_require(puf_ini_get(&ini, "policy", "encoder") == 4
        && puf_ini_get(&ini, "policy", "hidden_size") == 128
        && puf_ini_get(&ini, "policy", "num_layers") == 1,
        "requires research encoder4/H128/L1");
    SweepSpace space = {0};
    Space spaces[FEEDBACK_MAX_DIMS];
    char keys[FEEDBACK_MAX_DIMS][PUF_DICT_MAX_KEY];
    float defaults[FEEDBACK_MAX_DIMS];
    space.spaces = spaces;
    space.cost_idx = -1;
    space.optimize_direction = 1;
    for (int i = 0; i < ini.num_sections; i++) {
        Dict* section = &ini.sections[i];
        if (strncmp(section->name, "sweep.", 6) != 0) continue;
        feedback_require(strncmp(section->name, "sweep.policy.cnn_", 17) == 0,
            "only CNN coordinates may be searched");
        feedback_require(space.num < FEEDBACK_MAX_DIMS, "too many dimensions");
        const char* distribution = dict_get_str(section, "distribution");
        feedback_require(strcmp(distribution, "int_uniform") == 0
            || strcmp(distribution, "uniform_pow2") == 0, "unsupported distribution");
        Space s = {.type = strcmp(distribution, "uniform_pow2") == 0
            ? SPACE_POW2 : SPACE_LINEAR, .min = (float)dict_get(section, "min"),
            .max = (float)dict_get(section, "max"), .scale = 0.5f, .is_integer = 1};
        feedback_require(isfinite(s.min) && isfinite(s.max) && s.min < s.max
            && (s.type != SPACE_POW2 || s.min > 0), "invalid coordinate range");
        const char* scale = dict_get_str(section, "scale");
        if (strcmp(scale, "auto") != 0) s.scale = (float)dict_get(section, "scale");
        feedback_require(isfinite(s.scale) && s.scale > 0, "invalid coordinate scale");
        int d = space.num++;
        snprintf(keys[d], sizeof(keys[d]), "%s", section->name + 13);
        spaces[d] = s;
        float value = (float)puf_ini_get(&ini, "policy", keys[d]);
        defaults[d] = space_normalize(&s, value);
        feedback_require(isfinite(defaults[d]) && defaults[d] >= -1 && defaults[d] <= 1
            && space_unnormalize(&s, defaults[d]) == value, "invalid default coordinate");
    }
    feedback_require(space.num > 0, "empty CNN search space");
    if (describe) {
        printf("{\"protocol\":\"native-cross-game-protein-v1\",\"dimensions\":%d,"
            "\"gpu_queried\":false,\"policy_executed\":false,\"coordinates\":[", space.num);
        for (int d = 0; d < space.num; d++) {
            printf("%s\"%s\"", d ? "," : "", keys[d]);
        }
        printf("]}\n");
        puf_ini_free(&ini);
        return 0;
    }

    // Validate the complete fixed-format ledger before any CUDA initialization.
    FILE* history = fopen(argv[3], "r");
    feedback_require(history != NULL, "cannot read history");
    int dimensions = 0, rows = 0;
    char extra = 0;
    char header[128], line[8192];
    feedback_require(fgets(header, sizeof(header), history) != NULL
        && strchr(header, '\n') != NULL
        && sscanf(header, "PUFFER_CROSS_GAME_V1 %d %c", &dimensions, &extra) == 1
        && dimensions == space.num, "wrong history header");
    float* samples = (float*)calloc(FEEDBACK_MAX_ROWS * space.num, sizeof(float));
    float scores[FEEDBACK_MAX_ROWS], costs[FEEDBACK_MAX_ROWS];
    int failures[FEEDBACK_MAX_ROWS];
    while (fgets(line, sizeof(line), history)) {
        feedback_require(strchr(line, '\n') != NULL && rows < FEEDBACK_MAX_ROWS,
            "truncated/oversized history");
        char* cursor = line;
        char* end = NULL;
        long failure = strtol(cursor, &end, 10);
        feedback_require(end != cursor && (failure == 0 || failure == 1), "bad history status");
        failures[rows] = (int)failure;
        cursor = end;
        scores[rows] = strtof(cursor, &end);
        feedback_require(end != cursor && isfinite(scores[rows])
            && scores[rows] >= 0 && scores[rows] <= 1, "bad history score");
        cursor = end;
        costs[rows] = strtof(cursor, &end);
        feedback_require(end != cursor && isfinite(costs[rows]) && costs[rows] > 0,
            "bad history cost");
        cursor = end;
        for (int d = 0; d < space.num; d++) {
            float value = strtof(cursor, &end);
            feedback_require(end != cursor && isfinite(value) && value >= -1 && value <= 1,
                "bad normalized coordinate");
            samples[rows * space.num + d] = value;
            cursor = end;
        }
        while (isspace((unsigned char)*cursor)) cursor++;
        feedback_require(*cursor == 0, "extra history data");
        rows++;
    }
    feedback_require(!ferror(history), "history read failure");
    fclose(history);
    double random_control = feedback_option(&ini, "num_random_samples", 10);
    double iteration_control = feedback_option(&ini, "gp_training_iter", 50);
    double rng_seed = feedback_option(&ini, "seed", 73);
    feedback_require(isfinite(random_control) && random_control >= 0 && random_control <= 100
        && floor(random_control) == random_control
        && isfinite(iteration_control) && iteration_control >= 1 && iteration_control <= 100
        && floor(iteration_control) == iteration_control
        && isfinite(rng_seed) && rng_seed >= 0 && rng_seed <= 4294967295.0
        && floor(rng_seed) == rng_seed, "bad optimizer controls");
    int random_samples = (int)random_control;
    int gp_iterations = (int)iteration_control;
    int devices = 0;
    feedback_require(cudaGetDeviceCount(&devices) == cudaSuccess && devices == 1,
        "requires one scheduled GPU");
    feedback_require(cudaSetDevice(0) == cudaSuccess, "cannot select GPU");
    ProteinSweep* protein = protein_sweep_create((ProteinSweep){
        .space = &space, .num_random_samples = random_samples,
        .suggestions_per_pareto = 256, .gp_training_iter = gp_iterations,
        .gp_learning_rate = 0.001f, .optimizer_reset_frequency = 50,
        .gp_max_obs = 750, .infer_batch_size = 4096, .use_success_prob = 1,
        .prune_pareto = 1, .use_logit = 0, .global_search_scale = 1,
        .max_suggestion_cost = 3600, .expansion_rate = 0.1f,
        .cost_random_suggestion = -0.8f, .early_stop_quantile = 0.5f,
        .success_cap = 8192, .failure_cap = 1024, .top_k = 5,
        .rng_seed = (unsigned long long)rng_seed,
    });
    float next[FEEDBACK_MAX_DIMS];
    ProteinSweepInfo info = {0};
    for (int row = 0; row <= rows; row++) {
        if (row == 0) memcpy(next, defaults, space.num * sizeof(float));
        else info = protein_sweep_suggest(protein, next, NAN);
        if (row == rows) break;
        feedback_require(memcmp(next, samples + row * space.num,
            space.num * sizeof(float)) == 0, "native proposal replay changed; refuse feedback");
        protein_sweep_observe(protein, next, scores[row], costs[row], failures[row]);
    }
    feedback_require(cudaDeviceSynchronize() == cudaSuccess, "optimizer CUDA failure");
    FILE* output = fopen(argv[4], "wx");
    feedback_require(output != NULL, "proposal output exists or cannot be written");
    fprintf(output, "{\"protocol\":\"native-cross-game-protein-v1\",\"trial\":%d,"
        "\"observations_replayed\":%d,\"gp_observations\":%d,\"random\":%d,"
        "\"pareto\":%d,\"normalized\":[", rows, rows, info.n_gp_obs, info.is_random, info.n_pareto);
    for (int d = 0; d < space.num; d++) {
        feedback_require(isfinite(next[d]) && next[d] >= -1 && next[d] <= 1,
            "invalid native proposal");
        fprintf(output, "%s%.9g", d ? "," : "", next[d]);
    }
    fprintf(output, "],\"policy\":{");
    for (int d = 0; d < space.num; d++) {
        fprintf(output, "%s\"%s\":%.9g", d ? "," : "", keys[d],
            space_unnormalize(&spaces[d], next[d]));
    }
    fprintf(output, "},\"success_observations\":%d,\"failure_observations\":%d}\n",
        protein->succ_n, protein->fail_n);
    feedback_require(fclose(output) == 0, "proposal write failure");
    free(samples);
    puf_ini_free(&ini);
    // Process exit owns cleanup of PROTEIN's CUDA allocations/handles.
    return 0;
}
