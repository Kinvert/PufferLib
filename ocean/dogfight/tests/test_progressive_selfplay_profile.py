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
    tokens = command.split()
    assert "selfplay.mode=vanilla" in command
    assert "selfplay.enabled=1" in command
    assert "vec.total_agents=16384" in command
    assert "vec.frozen_bank_pct=0.1" in command
    assert "env.num_agents=2" in command
    assert "env.curriculum_enabled=1" in command
    assert "env.curriculum_randomize=0" in command
    assert "env.curriculum_target=0.9" in command
    assert "env.fixed_stage=-1" in command
    assert "env.max_stage=20" in command
    assert "env.global_step_stride=16384" in command
    assert "env.curriculum_total_steps=134217728" in command

    assert "env.curriculum_enabled=0" not in tokens
    assert "env.curriculum_target=0" not in tokens
    assert "env.fixed_stage=0" not in tokens
    assert "env.max_stage=0" not in tokens


def test_progressive_profile_preserves_base_validation():
    result = run_dry("progressive_profile_test", "1048576", "4097")

    assert result.returncode != 0
    assert "divisible by 8" in result.stderr
