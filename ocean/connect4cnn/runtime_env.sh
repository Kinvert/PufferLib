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
