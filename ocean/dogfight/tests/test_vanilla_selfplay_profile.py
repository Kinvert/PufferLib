import configparser
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "config" / "dogfight.ini"


def run_profile(*args, env=None):
    script = REPO_ROOT / "ocean" / "dogfight" / "train_vanilla_selfplay.sh"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )


def test_vanilla_profile_emits_official_native_selfplay_contract():
    config = configparser.ConfigParser()
    config.read(CONFIG)
    assert config["base"]["wandb_project"] == "df42"
    assert config["vec"]["num_frozen_banks"] == "1"
    assert config["vec"]["frozen_bank_pct"] == "0.1"
    assert config["selfplay"]["mode"] == "vanilla"
    assert config["selfplay"]["enabled"] == "1"
    assert config["selfplay"]["max_size"] == "100"
    assert config["env"]["num_agents"] == "2"
    assert config["env"]["curriculum_enabled"] == "1"
    assert config["env"]["curriculum_target"] == "0"
    assert config["env"]["fixed_stage"] == "0"
    assert config["env"]["max_stage"] == "10"
    assert config["env"]["native_spawn_curriculum"] == "1"
    assert config["env"]["native_acquisition_steps"] == "134217728"
    assert config["env"]["native_acquisition_reward_scale"] == "1.0"
    assert config["env"]["native_acquisition_neutral_scale"] == "0.1"
    assert config["env"]["native_acquisition_rehearsal_cycle_steps"] == "134217728"
    assert config["env"]["native_acquisition_rehearsal_steps"] == "0"
    assert config["env"]["native_spawn_total_steps"] == "12_079_595_520"
    assert config["env"]["native_frontier_fraction"] == "0.50"
    assert config["env"]["reward_version"] == "2"
    assert config["env"]["steering_alignment_scale"] == "0.01"
    assert config["env"]["role_randomization"] == "1"
    assert config["env"]["selfplay_bootstrap_steps"] == "0"
    assert config["env"]["selfplay_bootstrap_imitation_scale"] == "0"
    assert config["env"]["recovery_enabled"] == "0"
    assert config["train"]["ent_coef"] == "0.0001"

    result = run_profile(
        "--dry-run",
        "vanilla_canary",
        "8388608",
        "4096",
    )

    assert result.returncode == 0, result.stdout
    expected = [
        "train dogfight",
        "--wandb --wandb-project=df42",
        "base.run_id=vanilla_canary",
        "base.seed=42",
        "train.seed=42",
        "selfplay.seed=42",
        "vec.total_agents=4096",
        "train.total_timesteps=8388608",
        "env.global_step_stride=4096",
        "env.curriculum_total_steps=8388608",
    ]
    for argument in expected:
        assert argument in result.stdout
    assert len(result.stdout.split()) <= 13
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
    assert "train.seed=7" in result.stdout
    assert "base.seed=42" not in result.stdout


def test_vanilla_profile_rejects_bootstrap_argument():
    result = run_profile(
        "--dry-run",
        "bootstrap_canary",
        "134217728",
        "16384",
        "7",
        "33554432",
    )

    assert result.returncode == 2
