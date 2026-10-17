#pragma once

// Appearance has its own explicit seed and the native environment slot.
// Never advance the environment/opponent RNG to select a rendering.
static unsigned int cnn_appearance_mix(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static unsigned int cnn_appearance_option(Dict* kwargs, const char* key,
        unsigned int max) {
    DictItem* item = dict_find(kwargs, key);
    double value = item ? item->value : 0;
    if (!(value >= 0 && value <= max && value == floor(value))) {
        fprintf(stderr, "CNN appearance: %s must be an integer from 0 to %u\n", key, max);
        exit(1);
    }
    return (unsigned int)value;
}

static int cnn_appearance_init(Dict* kwargs, unsigned int slot, int count) {
    unsigned int fixed = cnn_appearance_option(kwargs, "representation", count - 1);
    unsigned int mode = cnn_appearance_option(kwargs, "representation_mode", 1);
    unsigned int seed = cnn_appearance_option(kwargs, "representation_seed", UINT32_MAX);
    if (mode == 0) return fixed;
    unsigned int draw = cnn_appearance_mix(slot ^ cnn_appearance_mix(seed ^ 0x9e3779b9u));
    unsigned int threshold = (0u - (unsigned int)count) % (unsigned int)count;
    while (draw < threshold) draw = cnn_appearance_mix(draw + 0x9e3779b9u);
    return draw % (unsigned int)count;
}

// Expanded catalogs must not silently change historical mixed assignments.
// Fixed mode accepts all IDs; mixed mode defaults to the legacy alphabet.
static int cnn_appearance_init_catalog(Dict* kwargs, unsigned int slot,
        int legacy_count, int count) {
    unsigned int catalog = cnn_appearance_option(kwargs, "representation_mix_catalog", 1);
    unsigned int mode = cnn_appearance_option(kwargs, "representation_mode", 1);
    return cnn_appearance_init(kwargs, slot, mode && !catalog ? legacy_count : count);
}
