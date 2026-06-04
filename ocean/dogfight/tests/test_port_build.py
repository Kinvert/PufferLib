import os
import subprocess
from pathlib import Path

import pytest


def build_dogfight_backend(repo_root):
    python_bin = repo_root / ".venv" / "bin" / "python"
    env = os.environ.copy()
    env["PYTHON"] = str(python_bin)
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["CCACHE_DIR"] = str(repo_root / "build" / "ccache")

    return subprocess.run(
        [str(repo_root / "build.sh"), "dogfight"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )


def test_dogfight_training_backend_builds():
    repo_root = Path(__file__).resolve().parents[3]
    result = build_dogfight_backend(repo_root)

    assert result.returncode == 0, result.stdout


def test_dogfight_gpu_vec_creates_and_resets():
    repo_root = Path(__file__).resolve().parents[3]
    result = build_dogfight_backend(repo_root)
    assert result.returncode == 0, result.stdout

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable; Dogfight GPU vec reset smoke is deferred and not counted as coverage")

    from pufferlib import _C

    args = {
        "vec": {
            "total_agents": 1,
            "num_buffers": 1,
        },
        "env": {
            "max_steps": 300,
            "obs_scheme": 1,
            "curriculum_enabled": 1,
            "curriculum_randomize": 0,
            "reward_aim_scale": 0.001695,
            "reward_closing_scale": 0.0001,
            "penalty_neg_g": 0.035,
            "control_rate_penalty": 0.002,
            "low_altitude_threshold": 1200.0,
            "low_altitude_penalty": 0.005,
            "speed_min": 50.0,
            "aim_decay_stage": 30.0,
            "shaping_decay_start": 100_000_000,
            "shaping_decay_end": 150_000_000,
            "energy_gain_scale": 0.001,
            "energy_loss_scale": 0.0005,
            "energy_advantage_scale": 0.004,
            "eval_spawn_mode": 0,
            "recovery_enabled": 1,
            "recovery_altitude_threshold": 500.0,
            "recovery_trigger_prob": 0.1,
            "recovery_speed_threshold": 70.0,
            "recovery_bank_deg": 60.0,
            "domain_randomization": 0.0,
            "vertical_spawn_prob": 0.0,
        },
    }

    vec = _C.create_vec(args, 1)
    try:
        assert _C.env_name == "dogfight"
        assert _C.gpu == 1
        assert vec.gpu == 1
        assert vec.total_agents == 1
        assert vec.obs_size == 26
        assert vec.num_atns == 5
        assert vec.act_sizes == [1, 1, 1, 1, 1]
        vec.reset()
    finally:
        vec.close()
