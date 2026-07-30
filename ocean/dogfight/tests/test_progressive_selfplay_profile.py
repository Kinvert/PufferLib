import configparser
from pathlib import Path
import subprocess


DOGFIGHT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = DOGFIGHT_DIR / "train_vanilla_selfplay_progressive.sh"


def run_dry(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=DOGFIGHT_DIR.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_progressive_profile_composes_native_selfplay_defaults():
    result = run_dry("progressive_profile_test", "134217728", "16384")

    assert result.returncode == 0, result.stderr
    command = result.stdout
    assert "--wandb" not in command
    assert "vec.total_agents=16384" in command
    assert "env.global_step_stride=16384" in command
    assert "env.curriculum_total_steps=134217728" in command
    assert len(command.split()) <= 11

    config = configparser.ConfigParser()
    config.read(DOGFIGHT_DIR.parents[1] / "config" / "dogfight.ini")
    assert config["base"]["wandb_project"] == "df43"
    assert config["selfplay"]["mode"] == "vanilla"
    assert config["selfplay"]["enabled"] == "1"
    assert config["vec"]["frozen_bank_pct"] == "0.1"
    assert config["env"]["num_agents"] == "2"
    assert config["env"]["curriculum_enabled"] == "1"
    assert config["env"]["curriculum_randomize"] == "0"
    assert config["env"]["curriculum_target"] == "0"
    assert config["env"]["fixed_stage"] == "0"
    assert config["env"]["max_stage"] == "10"
    assert config["env"]["native_spawn_curriculum"] == "1"
    assert config["env"]["native_acquisition_steps"] == "0"
    assert config["env"]["native_acquisition_reward_scale"] == "1.0"
    assert config["env"]["native_acquisition_neutral_scale"] == "0.1"
    assert config["env"]["native_acquisition_rehearsal_cycle_steps"] == "134217728"
    assert config["env"]["native_acquisition_rehearsal_steps"] == "0"
    assert config.getint("env", "native_spawn_total_steps") > 0
    assert config["env"]["native_frontier_fraction"] == "0.50"
    assert config.getint("train", "total_timesteps") == config.getint(
        "env", "native_spawn_total_steps"
    )
    assert not config.has_option("sweep", "sweep_only")
    assert config["env"]["steering_alignment_scale"] == "0"
    assert config["train"]["ent_coef"] == "0.0001"


def test_progressive_profile_preserves_base_validation():
    result = run_dry("progressive_profile_test", "1048576", "4097")

    assert result.returncode != 0
    assert "divisible by 8" in result.stderr
