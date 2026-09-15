#!/usr/bin/env bash
# Package small comparison receipts; retain the original run and checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
if (( $# != 1 )); then
    echo "Usage: bash ocean/connect4cnn/package_results.sh RUN_DIRECTORY" >&2
    exit 2
fi
run=$(realpath "$1")
test -f "$run/protocol.json"
test -f "$run/jobs.json"
test -f "$run/results.csv"
mkdir -p "$root/build/hardware-artifacts"
archive=$(mktemp "$root/build/hardware-artifacts/$(basename "$run").XXXXXXXX.tar.gz")
(
    cd "$run"
    find . -type d \( -name checkpoints -o -name comparison-sidecar -o -name __pycache__ \) -prune -o \
        -type f \( -name '*.md' -o -name '*.html' -o -name '*.tsv' -o -name '*.patch' \
            -o -name '*.csv' -o -name '*.json' -o -name '*.jsonl' \
            -o -name '*.ini' -o -name '*.txt' -o -name '*.log' -o -name '*.sha256' \
            -o -name '*.c' -o -name '*.cu' -o -name '*.h' -o -name '*.py' -o -name '*.sh' -o -name '*.js' \) -print0 \
        | tar --create --gzip --file="$archive" --null --verbatim-files-from --files-from=-
)
printf 'Evidence archive: %s\n' "$archive"
sha256sum "$archive"
