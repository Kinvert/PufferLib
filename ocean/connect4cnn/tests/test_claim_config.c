// Native configuration/RNG receipt check only. Does not include neural code.
#include "src/ini.h"
#include "ocean/connect4cnn/exact_protocol.h"
#include <inttypes.h>

int main(int argc, char** argv) {
    if (argc == 2) {
        Ini ini = {0};
        puf_ini_load_file(&ini,argv[1]);
        puf_ini_write(stdout,&ini);
        puf_ini_free(&ini);
        return 0;
    }
    uint32_t seeds[] = {0,147001,UINT32_MAX};
    uint32_t episodes[] = {0,91,UINT32_MAX};
    for (int s = 0; s < 3; s++) for (int e = 0; e < 3; e++)
        printf("%u,%u,%u,%" PRIu64 "\n",seeds[s],episodes[e],
            c4_exact_env_seed(seeds[s],episodes[e]),c4_exact_policy_seed(seeds[s],episodes[e]));
    return 0;
}
