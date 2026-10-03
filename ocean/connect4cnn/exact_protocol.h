#pragma once
#include <stdint.h>

// Version 1: episode identity, not slot/scheduling order, determines randomness.
#ifdef __CUDACC__
#define C4_EXACT_HD __host__ __device__
#else
#define C4_EXACT_HD
#endif
static C4_EXACT_HD uint64_t c4_exact_mix(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static C4_EXACT_HD uint64_t c4_exact_policy_seed(uint32_t seed, uint32_t episode) {
    return c4_exact_mix(((uint64_t)seed << 32 | episode) ^ UINT64_C(0x706f6c6963797631));
}
static C4_EXACT_HD uint32_t c4_exact_env_seed(uint32_t seed, uint32_t episode) {
    return (uint32_t)c4_exact_mix(((uint64_t)seed << 32 | episode) ^ UINT64_C(0x656e7669726f7631));
}
#undef C4_EXACT_HD
