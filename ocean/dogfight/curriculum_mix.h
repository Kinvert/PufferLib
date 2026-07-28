#pragma once

#include <stdint.h>

static inline int dogfight_select_fixed_stage(
        uint32_t env_identity,
        int primary_stage,
        int rehearsal_stage,
        int rehearsal_stride) {
    if (rehearsal_stage < 0 || rehearsal_stride <= 0) {
        return primary_stage;
    }
    return env_identity % (uint32_t)rehearsal_stride == 0
        ? rehearsal_stage
        : primary_stage;
}
