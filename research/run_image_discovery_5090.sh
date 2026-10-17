#!/usr/bin/env bash
# Native discovery orchestration. Requires completed fixed-image calibration.
set -euo pipefail
cd "$(dirname "$0")/.."
image_command=${1:-}
case "$image_command" in
    prepare-pilot) [[ $# == 3 ]] ;;
    prepare-sweep) [[ $# == 4 ]] ;;
    run|inspect|handoff) [[ $# == 2 ]] ;;
    *) echo 'Usage: run_image_discovery_5090.sh prepare-pilot CALIBRATION FRESH_OUT' >&2
       echo '       run_image_discovery_5090.sh prepare-sweep CALIBRATION PILOT FRESH_OUT' >&2
       echo '       run_image_discovery_5090.sh run|inspect|handoff DELIVERY' >&2; exit 2 ;;
esac
if [[ -f build/connect4cnn/runtime-5090.sh ]]; then source build/connect4cnn/runtime-5090.sh; fi
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
[[ -z "${NVCC_EXTRA:-}${NVCC_FLAGS:-}${NVCC_APPEND_FLAGS:-}${NVCC_PREPEND_FLAGS:-}" ]] || {
    echo 'Unset compiler overrides; use the documented existing runtime.' >&2; exit 1;
}
image_python=${IMAGE_PYTHON:-"$PWD/.venv/bin/python"}
[[ -x "$image_python" ]] || { echo 'Reuse the calibration uv Python3.12 venv; run its setup first.' >&2; exit 1; }
"$image_python" -c 'import numpy,sys; assert sys.version_info[:2] == (3,12)'
if [[ "$image_command" == prepare-pilot ]]; then
    "$image_python" research/prepare_image_discovery.py prepare --phase pilot --calibration "$2" --out "$3"
elif [[ "$image_command" == prepare-sweep ]]; then
    "$image_python" research/prepare_image_discovery.py prepare --phase sweep --calibration "$2" --pilot "$3" --out "$4"
elif [[ "$image_command" == run ]]; then
    image_status=0
    "$image_python" research/prepare_image_discovery.py run --delivery "$2" --allow-gpu || image_status=$?
    "$image_python" research/prepare_image_discovery.py handoff --delivery "$2" || {
        echo 'Summary packaging failed; original evidence remains intact.' >&2
        [[ "$image_status" != 0 ]] || image_status=1
    }
    exit "$image_status"
else
    "$image_python" research/prepare_image_discovery.py "$image_command" --delivery "$2"
fi
