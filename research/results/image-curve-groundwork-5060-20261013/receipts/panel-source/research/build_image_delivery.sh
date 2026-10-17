#!/usr/bin/env bash
# Fresh six-game native RGB builds and host registration; no GPU execution.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 1 && ! -e "$1" ]] || { echo 'Usage: build_image_delivery.sh FRESH_DIR' >&2; exit 1; }
image_python=${IMAGE_PYTHON:-.venv/bin/python}
[[ -x "$image_python" ]] || { echo 'Set IMAGE_PYTHON to the existing uv Python 3.12 venv.' >&2; exit 1; }
# Keep the venv path: resolving its interpreter symlink bypasses pyvenv.cfg.
[[ "$image_python" == /* ]] || image_python="$PWD/$image_python"
[[ "$("$image_python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == 3.12 ]] || { echo 'Python 3.12 required.' >&2; exit 1; }
[[ "${NVCC_ARCH:-}" =~ ^sm_[0-9]+[a-z]?$ ]] || { echo 'Set NVCC_ARCH explicitly for the target GPU.' >&2; exit 1; }
mkdir -p "$1"
image_out=$(realpath "$1")
while read -r image_game image_height image_width; do
    bash research/build_image_policy.sh "$image_game" "$image_height" "$image_width" 3 default "$image_out/native/$image_game"
done <<'SHAPES'
connect4cnn 576 672
pongcnn 640 620
flappycnn 640 420
breakoutcnn 330 576
snakecnn 616 616
mazecnn 616 616
SHAPES
cat > "$image_out/builds.json" <<'BUILDS'
{
  "connect4cnn": {"default": "native/connect4cnn"},
  "pongcnn": {"default": "native/pongcnn"},
  "flappycnn": {"default": "native/flappycnn"},
  "breakoutcnn": {"default": "native/breakoutcnn"},
  "snakecnn": {"default": "native/snakecnn"},
  "mazecnn": {"default": "native/mazecnn"}
}
BUILDS
"$image_python" research/import_image_builds.py --builds "$image_out/builds.json" --out "$image_out/registry"
"$image_python" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"research"); import candidate_panel as p; root=Path(sys.argv[1]); builds=json.loads((root/"builds.json").read_text()); p.policy_layout.freeze_images({game:{family:root/path for family,path in groups.items()} for game,groups in builds.items()},root/"metadata")' "$image_out"
echo "Prepared native RGB trainers and host metadata: $image_out"
echo 'No GPU query, numerical check, training or evaluation executed.'
