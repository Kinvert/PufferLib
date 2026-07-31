#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PUFFER_VENV:-}" ]]; then
    puffer_venv="$PUFFER_VENV"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    puffer_venv="$repo_root/.venv"
elif command -v python >/dev/null 2>&1; then
    python="$(command -v python)"
    puffer_venv="$("$python" -c 'import sys; print(sys.prefix)')"
else
    echo "missing Python; activate the PufferLib virtualenv first" >&2
    exit 2
fi
python="$puffer_venv/bin/python"
launcher="$repo_root/ocean/dogfight/native_sweep.py"

[[ -x "$python" ]] || {
    echo "missing PufferLib virtualenv Python: $python" >&2
    exit 2
}

export PUFFER_VENV="$puffer_venv"
if [[ "${1:-}" == "--screen-existing" ]]; then
    [[ "$#" -eq 2 ]] || {
        echo "usage: $0 --screen-existing EXPERIMENT_DIR" >&2
        exit 2
    }
    exec "$python" "$launcher" screen --experiment-dir "$2"
fi

exec "$python" "$launcher" launch "$@"
