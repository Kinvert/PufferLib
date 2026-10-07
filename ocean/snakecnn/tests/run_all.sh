#!/usr/bin/env bash
# CPU environment/raster only, never neural inference or training.
set -euo pipefail
cd "$(dirname "$0")/../../.."
out=${1:-}
if [[ -z "$out" ]]; then
    mkdir -p build/snakecnn
    out=$(mktemp -d build/snakecnn/environment.XXXXXXXX)
else
    mkdir -p "$(dirname "$out")"
    mkdir "$out"
fi
raylib_dir=raylib-5.5_linux_amd64
test -f "$raylib_dir/lib/libraylib.a"
flags=(-std=c11 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -Wextra
       -Wno-unused-function -Wno-unused-parameter -O1 -g
       -fsanitize=address,undefined -fno-omit-frame-pointer
       -I. -Isrc -I"$raylib_dir/include")
for variant in snakebench snakecnn; do
    "${CC:-clang}" "${flags[@]}" "-DENV_HEADER=\"ocean/$variant/$variant.h\"" \
        ocean/snakecnn/tests/test_environment.c "$raylib_dir/lib/libraylib.a" \
        -lGL -lpthread -ldl -lrt -lm -o "$out/test_$variant"
    "$out/test_$variant" > "$out/$variant.trace"
done
cmp "$out/snakebench.trace" "$out/snakecnn.trace"
"$out/test_snakecnn" > "$out/repeat.trace"
cmp "$out/snakecnn.trace" "$out/repeat.trace"
for representation in {1..5}; do
    "$out/test_snakecnn" "$representation" > "$out/r$representation.trace"
    cmp "$out/snakebench.trace" "$out/r$representation.trace"
done
for seed in 0 12345 4294967295; do
    "$out/test_snakecnn" 0 1 "$seed" > "$out/mixed-$seed.trace"
    cmp "$out/snakebench.trace" "$out/mixed-$seed.trace"
done
for option in 'rule_version 0' 'num_agents 2' 'vision 4' 'leave_corpse_on_death 1' \
              'width 12' 'width 129' 'height 26.5' 'num_food 0' 'num_food 256' \
              'max_snake_length 1' 'max_snake_length 254' 'max_steps 0' 'max_steps 16777217' \
              'reward_food nan' 'reward_death 1' 'cell_size 0' 'representation -1' \
              'representation 6' 'representation 1.5' 'representation_mode 2' \
              'representation_seed -1' 'representation_seed 4294967296'; do
    if "$out/test_snakecnn" --invalid $option > /dev/null 2> "$out/invalid-${option// /-}.log"; then
        echo "Invalid configuration accepted: $option" >&2; exit 1
    fi
    rg -q 'Snake benchmark:|CNN appearance:' "$out/invalid-${option// /-}.log"
done
for action in -1 4 1.5 nan inf; do
    if "$out/test_snakecnn" --action "$action" > /dev/null 2> "$out/action-$action.log"; then
        echo "Invalid action accepted: $action" >&2; exit 1
    fi
    rg -q 'invalid action' "$out/action-$action.log"
done
sha256sum "$out/"*.trace > "$out/traces.sha256"
echo "PASS: 49,152 decisions per panel match independent deque reference; state/pixels and all six/three mixed drawings agree."
echo "PASS: spawn/RNG, growth/neck/tail/death/horizon/reset, local pixels/guards/overwrite, worker order, repeat, ASan/UBSan."
echo "Receipts: $out"
