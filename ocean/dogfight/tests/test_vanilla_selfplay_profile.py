import os
import subprocess
from pathlib import Path


def run_profile(*args, env=None):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "ocean" / "dogfight" / "train_vanilla_selfplay.sh"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )


def test_vanilla_profile_emits_official_native_selfplay_contract():
    result = run_profile(
        "--dry-run",
        "vanilla_canary",
        "8388608",
        "4096",
    )

    assert result.returncode == 0, result.stdout
    expected = [
        "train dogfight",
        "base.run_id=vanilla_canary",
        "base.seed=42",
        "base.async=0",
        "base.checkpoint_interval=8",
        "vec.total_agents=4096",
        "vec.num_buffers=4",
        "vec.num_threads=8",
        "vec.num_frozen_banks=1",
        "vec.frozen_bank_pct=0.1",
        "vec.frozen_bank_hidden_size=64",
        "vec.frozen_bank_num_layers=3",
        "selfplay.mode=vanilla",
        "selfplay.enabled=1",
        "selfplay.max_size=8",
        "selfplay.opp_timeout_steps=2097152",
        "policy.hidden_size=64",
        "policy.num_layers=3",
        "train.total_timesteps=8388608",
        "train.gpus=1",
        "train.horizon=64",
        "env.num_agents=2",
        "env.curriculum_enabled=1",
        "env.fixed_stage=0",
        "env.reward_version=1",
        "env.selfplay_bootstrap_steps=0",
        "env.selfplay_bootstrap_imitation_scale=0",
        "env.role_randomization=1",
        "env.domain_randomization=0",
        "env.vertical_spawn_prob=0",
        "env.recovery_enabled=0",
    ]
    for argument in expected:
        assert argument in result.stdout
    assert "env.curriculum_enabled=0" not in result.stdout
    assert "env.reward_version=2" not in result.stdout
    assert "base.load_model_path=" not in result.stdout


def test_vanilla_profile_rejects_invalid_agent_partition():
    result = run_profile(
        "--dry-run",
        "invalid_agents",
        "8388608",
        "4097",
    )

    assert result.returncode == 2
    assert "divisible by 8" in result.stdout


def test_vanilla_profile_propagates_explicit_seed():
    result = run_profile(
        "--dry-run",
        "seed_canary",
        "8388608",
        "4096",
        "7",
    )

    assert result.returncode == 0, result.stdout
    assert "base.seed=7" in result.stdout
    assert "selfplay.seed=7" in result.stdout
    assert "base.seed=42" not in result.stdout


def test_vanilla_profile_propagates_bootstrap_steps():
    result = run_profile(
        "--dry-run",
        "bootstrap_canary",
        "134217728",
        "16384",
        "7",
        "33554432",
    )

    assert result.returncode == 0, result.stdout
    assert "base.seed=7" in result.stdout
    assert "env.selfplay_bootstrap_steps=33554432" in result.stdout


def test_vanilla_profile_propagates_bootstrap_imitation_scale():
    env = os.environ.copy()
    env["DOGFIGHT_BOOTSTRAP_IMITATION_SCALE"] = "0.01"
    result = run_profile(
        "--dry-run",
        "bootstrap_imitation_canary",
        "67108864",
        "16384",
        "42",
        "33554432",
        env=env,
    )

    assert result.returncode == 0, result.stdout
    assert (
        "env.selfplay_bootstrap_imitation_scale=0.01"
        in result.stdout.split()
    )


def test_vanilla_profile_can_disable_frozen_training_rows():
    env = os.environ.copy()
    env["DOGFIGHT_FROZEN_BANK_PCT"] = "0"
    result = run_profile(
        "--dry-run",
        "current_current_canary",
        "67108864",
        "16384",
        "42",
        "0",
        env=env,
    )

    assert result.returncode == 0, result.stdout
    tokens = result.stdout.split()
    assert "vec.frozen_bank_pct=0" in tokens
    assert "vec.num_frozen_banks=0" in tokens
    assert "selfplay.enabled=0" in tokens


def test_vanilla_profile_can_disable_control_rate_penalty():
    env = os.environ.copy()
    env["DOGFIGHT_CONTROL_RATE_PENALTY"] = "0"
    result = run_profile(
        "--dry-run",
        "no_rate_penalty_canary",
        "67108864",
        "16384",
        "42",
        "0",
        env=env,
    )

    assert result.returncode == 0, result.stdout
    assert "env.control_rate_penalty=0" in result.stdout.split()


def test_vanilla_profile_propagates_aileron_magnitude_penalty():
    env = os.environ.copy()
    env["DOGFIGHT_AILERON_MAGNITUDE_PENALTY"] = "0.01"
    result = run_profile(
        "--dry-run",
        "aileron_penalty_canary",
        "67108864",
        "16384",
        "42",
        "0",
        env=env,
    )

    assert result.returncode == 0, result.stdout
    assert "env.aileron_magnitude_penalty=0.01" in result.stdout.split()
