#!/usr/bin/env bash
# Source from the repository root. PATH-first native Linux, then WSL.
pong_find_smi() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        command -v nvidia-smi
    elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        echo /usr/lib/wsl/lib/nvidia-smi
    else
        echo 'nvidia-smi not found in PATH or WSL driver directory' >&2
        return 1
    fi
}
pong_gpu_idle() {
    local active
    active=$("$PONG_SMI" --query-compute-apps=pid --format=csv,noheader) || return
    [[ -z "${active//[[:space:]]/}" ]] || {
        echo 'GPU busy; stopping without interrupting other work.' >&2
        return 1
    }
}
