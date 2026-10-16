#!/usr/bin/env bash
# Source from the repository root. Process-local paths only; installs nothing.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -z "${NCCL_ROOT:-}" ]; then
    for c4_nccl in "$PWD"/.venv/lib/python*/site-packages/nvidia/nccl \
            /home/claude/PufferLib/.venv/lib/python3.12/site-packages/nvidia/nccl; do
        if [ -f "$c4_nccl/include/nccl.h" ]; then
            export NCCL_ROOT="$c4_nccl"
            break
        fi
    done
fi
if [ ! -f "${NCCL_ROOT:-}/include/nccl.h" ]; then
    echo "Set NCCL_ROOT to an existing NCCL package (include/ and lib/)." >&2
    return 1
fi
export CPATH="$NCCL_ROOT/include${CPATH:+:$CPATH}"
# Standard build.sh links -lnvidia-ml. The toolkit provides its link-time stub;
# at runtime WSL supplies the actual driver library, never the stub directory.
export LIBRARY_PATH="$NCCL_ROOT/lib:$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCACHE_DIR="$PWD/build/ccache"
unset c4_nccl

# Native Linux first; WSL exposes the driver tool outside PATH on G240.
puffer_find_nvidia_smi() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        command -v nvidia-smi
    elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
        echo /usr/lib/wsl/lib/nvidia-smi
    else
        echo 'Cannot find nvidia-smi in PATH or the WSL driver directory.' >&2
        return 1
    fi
}

puffer_require_idle_gpu() {
    local active
    active=$("$1" --query-compute-apps=pid --format=csv,noheader) || return
    if [[ -n "${active//[[:space:]]/}" ]]; then
        echo 'GPU busy; stopping without interrupting other work.' >&2
        return 1
    fi
}
