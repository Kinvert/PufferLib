#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

prepare_source() {
    local destination="$1"
    local core_source="src/pufferl.cu"
    local horizon_line='    puf_ini_put(ini, "train.horizon", "1");'
    local eval_agents_guard='            if (eval_agents > num_games && num_games >= 1024) {'
    local eval_branch='    } else if (strcmp(mode, "eval") == 0 || strcmp(mode, "eval_bot") == 0) {'

    if [[ "$(grep -Fxc "$horizon_line" "$core_source")" -ne 1 ]]; then
        echo "Expected exactly one upstream eval horizon assignment" >&2
        exit 1
    fi
    if [[ "$(grep -Fxc "$eval_agents_guard" "$core_source")" -ne 1 ]]; then
        echo "Expected exactly one upstream small-eval agent guard" >&2
        exit 1
    fi
    if [[ "$(grep -Fxc "$eval_branch" "$core_source")" -ne 1 ]]; then
        echo "Expected exactly one upstream eval CLI branch" >&2
        exit 1
    fi

    mkdir -p "$(dirname "$destination")"
    sed \
        -e 's/puf_ini_put(ini, "train.horizon", "1");/puf_ini_put(ini, "train.horizon", "8");/' \
        -e 's/if (eval_agents > num_games && num_games >= 1024)/if (eval_agents > num_games)/' \
        -e 's/puf_ini_put(&ini, "env.dr", "0");/puf_ini_put(\&ini, "env.domain_randomization", "0");/' \
        "$core_source" > "$destination"

    awk '
        $0 == "    // Selfplay end rating: match final checkpoint vs a small fixed opponent pool." {
            print "    // run_eval mutates the shared INI for match topology. Preserve the"
            print "    // trained configuration so native logs and observers report the"
            print "    // parameters Protein actually evaluated."
            print "    const char* dogfight_saved_eval_keys[][2] = {"
            print "        {\"base\", \"num_games\"},"
            print "        {\"base\", \"load_model_path\"},"
            print "        {\"base\", \"load_enemy_model_path\"},"
            print "        {\"base\", \"reset_every_horizon\"},"
            print "        {\"vec\", \"num_buffers\"},"
            print "        {\"vec\", \"total_agents\"},"
            print "        {\"vec\", \"num_frozen_banks\"},"
            print "        {\"vec\", \"frozen_bank_pct\"},"
            print "        {\"vec\", \"frozen_bank_hidden_size\"},"
            print "        {\"vec\", \"frozen_bank_num_layers\"},"
            print "        {\"selfplay\", \"enabled\"},"
            print "        {\"env\", \"dr\"},"
            print "        {\"env\", \"domain_randomization\"},"
            print "        {\"env\", \"num_agents\"},"
            print "        {\"env\", \"num_bots\"},"
            print "        {\"train\", \"horizon\"},"
            print "    };"
            print "    constexpr int dogfight_saved_eval_count ="
            print "        sizeof(dogfight_saved_eval_keys) / sizeof(dogfight_saved_eval_keys[0]);"
            print "    char dogfight_saved_eval_values[dogfight_saved_eval_count][4096] = {{0}};"
            print "    if (use_selfplay) {"
            print "        for (int i = 0; i < dogfight_saved_eval_count; i++) {"
            print "            snprintf(dogfight_saved_eval_values[i],"
            print "                sizeof(dogfight_saved_eval_values[i]), \"%s\","
            print "                puf_ini_get_str(ini, dogfight_saved_eval_keys[i][0],"
            print "                    dogfight_saved_eval_keys[i][1]));"
            print "        }"
            print "    }"
            print ""
            print
            native_pool_eval = 1
            next
        }
        native_pool_eval && $0 == "    if (ctx->artifact_owner) {" {
            print "    if (use_selfplay) {"
            print "        for (int i = 0; i < dogfight_saved_eval_count; i++) {"
            print "            puf_ini_set("
            print "                puf_ini_section(ini, dogfight_saved_eval_keys[i][0], 0),"
            print "                dogfight_saved_eval_keys[i][1],"
            print "                dogfight_saved_eval_values[i]);"
            print "        }"
            print "    }"
            print ""
            native_pool_eval = 0
            print
            next
        }
        $0 == "            puf_dashboard_print(ini, pufferl, &log, 0);" {
            print
            finite_render = 1
            next
        }
        finite_render && $0 == "            continue;" {
            print "            if (num_games <= 0) {"
            print "                continue;"
            print "            }"
            finite_render = 0
            next
        }
        $0 == "    } else if (strcmp(mode, \"eval\") == 0 || strcmp(mode, \"eval_bot\") == 0) {" {
            print "    } else if (strcmp(mode, \"render\") == 0) {"
            print "        puf_ini_put(&ini, \"vec.num_buffers\", \"2\");"
            print "        puf_ini_put(&ini, \"vec.total_agents\", \"2\");"
            print "        puf_ini_put(&ini, \"vec.num_frozen_banks\", \"0\");"
            print "        puf_ini_put(&ini, \"vec.frozen_bank_pct\", \"0\");"
            print "        puf_ini_put(&ini, \"selfplay.enabled\", \"0\");"
            print "        puf_ini_put(&ini, \"env.domain_randomization\", \"0\");"
            print "        puf_ini_put(&ini, \"env.vertical_spawn_prob\", \"0\");"
            print "        puf_ini_put(&ini, \"env.num_agents\", \"1\");"
            print "        puf_ini_put(&ini, \"env.num_bots\", \"1\");"
            print "        puf_ini_put(&ini, \"env.role_randomization\", \"0\");"
            print "        run_eval(&ini, &ctx, EVAL_RENDER, 1);"
            print $0
            next
        }
        $0 == "        if (strcmp(mode, \"eval_bot\") == 0) {" {
            print
            print "            puf_ini_put(&ini, \"base.eval_agents\", \"2\");"
            print "            puf_ini_put(&ini, \"vec.num_buffers\", \"2\");"
            print "            puf_ini_put(&ini, \"env.vertical_spawn_prob\", \"0\");"
            print "            puf_ini_put(&ini, \"env.role_randomization\", \"0\");"
            next
        }
        $0 == "            result.games = (int)scored_n;" {
            print
            print "            double az_neg_sum = dict_get(&log, \"env/target_az_neg_aileron_sum\");"
            print "            double az_pos_sum = dict_get(&log, \"env/target_az_pos_aileron_sum\");"
            print "            double az_neg_steps = dict_get(&log, \"env/target_az_neg_steps\");"
            print "            double az_pos_steps = dict_get(&log, \"env/target_az_pos_steps\");"
            print "            printf(\"dogfight_eval_controls avg_abs_bias=%.6f avg_signed_bias=%.6f avg_roll_rotations=%.6f avg_control_rate=%.6f az_neg_mean_aileron=%.6f az_pos_mean_aileron=%.6f az_neg_steps=%.1f az_pos_steps=%.1f\\n\","
            print "                dict_get(&log, \"env/avg_abs_bias\"),"
            print "                dict_get(&log, \"env/avg_signed_bias\"),"
            print "                dict_get(&log, \"env/avg_roll_rotations\"),"
            print "                dict_get(&log, \"env/avg_control_rate\"),"
            print "                az_neg_steps > 0.0 ? az_neg_sum / az_neg_steps : 0.0,"
            print "                az_pos_steps > 0.0 ? az_pos_sum / az_pos_steps : 0.0,"
            print "                az_neg_steps, az_pos_steps);"
            print "            printf(\"dogfight_eval_outcomes perf=%.6f score=%.6f timeouts=%.6f player_ground_hits=%.6f opponent_ground_hits=%.6f\\n\","
            print "                dict_get(&log, \"env/perf\"),"
            print "                dict_get(&log, \"env/score\"),"
            print "                dict_get(&log, \"env/timeouts\"),"
            print "                dict_get(&log, \"env/player_ground\"),"
            print "                dict_get(&log, \"env/opponent_ground\"));"
            next
        }
        { print }
    ' "$destination" > "$destination.tmp"
    mv "$destination.tmp" "$destination"
}

if [[ "${1:-}" == "--prepare-only" ]]; then
    if [[ "$#" -ne 2 ]]; then
        echo "usage: $0 --prepare-only OUTPUT_SOURCE" >&2
        exit 1
    fi
    prepare_source "$2"
    exit 0
fi

output="${1:-build/puffer-dogfight-eval}"
generated_source="build/pufferl_dogfight_eval.cu"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
nccl_include="$(python -c \
    "import nvidia.nccl, os; print(os.path.join(nvidia.nccl.__path__[0], 'include'))")"
nccl_lib="$(python -c \
    "import nvidia.nccl, os; print(os.path.join(nvidia.nccl.__path__[0], 'lib'))")"

prepare_source "$generated_source"
mkdir -p "$(dirname "$output")"

nvcc=("$cuda_home/bin/nvcc")
if command -v ccache >/dev/null 2>&1; then
    nvcc=(ccache "${nvcc[@]}")
fi

echo "Compiling Dogfight-local native binary..."
"${nvcc[@]}" \
    -O2 --threads 0 -arch=native -std=c++17 \
    -I. -Isrc -Iocean/dogfight -Ivendor \
    -Iraylib-5.5_linux_amd64/include \
    -I"$cuda_home/include" -I"$cuda_home/include/cccl" -I"$nccl_include" \
    '-DENV_HEADER="ocean/dogfight/dogfight.h"' \
    -DENV_NAME=dogfight '-DPUFFER_ENV_NAME="dogfight"' \
    -DPUFFERLIB_BUILD_MAIN \
    -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp \
    "$generated_source" raylib-5.5_linux_amd64/lib/libraylib.a \
    -L"$cuda_home/lib64" -L"$nccl_lib" \
    -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
    -lm -lpthread -lomp5 -lGL \
    -o "$output"
echo "Built: $output"
