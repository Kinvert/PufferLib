#!/usr/bin/env bash
# No source editing on 5090: prepare, explicitly run, or package a retained run.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 2 && ( "$1" == prepare || "$1" == run || "$1" == handoff || "$1" == inspect ) ]] || {
    echo 'Usage: bash research/run_image_calibration_5090.sh prepare FRESH_DIRECTORY' >&2
    echo '       bash research/run_image_calibration_5090.sh run PREPARED_DIRECTORY' >&2
    echo '       bash research/run_image_calibration_5090.sh inspect PREPARED_DIRECTORY' >&2
    echo '       bash research/run_image_calibration_5090.sh handoff PREPARED_DIRECTORY' >&2
    exit 2
}
image_stage=$1
image_directory=$2
if [[ -f build/connect4cnn/runtime-5090.sh ]]; then
    source build/connect4cnn/runtime-5090.sh
fi
source ocean/connect4cnn/runtime_env.sh
export NVCC_ARCH=sm_120
[[ -z "${NVCC_EXTRA:-}${NVCC_FLAGS:-}${NVCC_APPEND_FLAGS:-}${NVCC_PREPEND_FLAGS:-}" ]] || {
    echo 'Compiler overrides are not accepted; start with the documented runtime helper only.' >&2
    exit 1
}
if [[ -z "${IMAGE_PYTHON:-}" && ! -x .venv/bin/python ]]; then
    [[ "$image_stage" == prepare ]] || { echo 'Run prepare first.' >&2; exit 1; }
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python 'numpy==2.5.3'
fi
image_python=${IMAGE_PYTHON:-"$PWD/.venv/bin/python"}
[[ -x "$image_python" ]] || { echo 'IMAGE_PYTHON must point to the existing uv Python3.12 interpreter.' >&2; exit 1; }
"$image_python" -c 'import numpy,sys; assert sys.version_info[:2] == (3,12), "Python3.12 required"'
if [[ "$image_stage" == prepare ]]; then
    [[ ! -e "$image_directory" ]] || { echo 'Use a fresh output directory; no overwrite/restart.' >&2; exit 1; }
    "$image_python" research/image_calibration_campaign.py prepare --out "$image_directory"
elif [[ "$image_stage" == inspect ]]; then
    "$image_python" research/image_calibration_campaign.py inspect --prepared "$image_directory"
elif [[ "$image_stage" == handoff ]]; then
    "$image_python" research/image_calibration_campaign.py handoff --prepared "$image_directory"
else
    [[ -f "$image_directory/CAMPAIGN.json" && ! -e "$image_directory/execution" ]] || {
        echo 'Run prepare first; preserve every started allocation, including failures.' >&2; exit 1;
    }
    image_status=0
    "$image_python" research/image_calibration_campaign.py run --prepared "$image_directory" --allow-gpu || image_status=$?
    # Package only summaries/configs/receipts. Never SCP huge numerical arrays by default.
    "$image_python" research/image_calibration_campaign.py handoff --prepared "$image_directory" || {
        echo 'Summary packaging failed; the complete original tree is retained.' >&2
        [[ "$image_status" != 0 ]] || image_status=1
    }
    exit "$image_status"
fi
