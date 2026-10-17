// Host descriptors only: no CUDA query, allocation, initialization or tensor math.
static void sprite_benchmark_describe(int argc, char** argv) {
    assert(argc == 3);
    Ini ini = {};
    puf_ini_load_file(&ini, argv[2]);
    Dict* env = puf_ini_section(&ini, "env", 0);
    Dict* policy = puf_ini_section(&ini, "policy", 0);
    assert(dict_get(env, "partial_observation") == 1 && dict_get(env, "observation_mode") == 1);
    int hidden = dict_get(policy, "hidden_size"), layers = dict_get(policy, "num_layers");
    int horizon = puf_ini_get(&ini, "train", "horizon");
    int sizes[] = ACT_SIZES, actions = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        actions += sizes[i];
    }
    Arch arch = build_arch(SPRITE_VIEW_OBS, hidden, layers, actions, false, horizon, policy);
    Allocator params = {}, encoder = {}, rollout = {}, acts = {}, grads = {};
    Weights weights = weights_create(&arch, &params);
    arch.encoder.reg_params(weights.encoder, &encoder);
    void* activations = calloc(1, arch.encoder.activation_size);
    arch.encoder.reg_rollout(weights.encoder, activations, &rollout, 1);
    arch.encoder.reg_train(weights.encoder, activations, &acts, &grads, 1);
    assert(!params.mem && !acts.mem && params.total_bytes == params.total_elems * 4);
    printf("{\"schema\":\"sprite-packed-metadata-v1\",\"encoder\":%d,"
        "\"width\":96,\"height\":192,\"channels\":3,\"geometry\":4,"
        "\"input_size\":%d,\"action_sizes\":[%d,%d,%d],\"hidden\":%d,\"layers\":%d,"
        "\"parameters\":%ld,\"encoder_parameters\":%ld,\"checkpoint_bytes\":%ld,"
        "\"encoder_rollout_bytes_per_item\":%ld,\"encoder_train_bytes_per_item\":%ld,"
        "\"encoder_gradient_bytes\":%ld,\"gpu_execution\":false}\n",
        (int)dict_get(policy, "encoder"), SPRITE_VIEW_OBS, sizes[0], sizes[1], sizes[2],
        hidden, layers, params.total_elems, encoder.total_elems, params.total_bytes,
        rollout.total_bytes, acts.total_bytes, grads.total_bytes);
    puf_ini_free(&ini);
}
