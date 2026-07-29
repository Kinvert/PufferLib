from pathlib import Path
import subprocess


DOGFIGHT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = DOGFIGHT_DIR / "train_vanilla_selfplay_history.sh"


def run_dry(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", *args],
        cwd=DOGFIGHT_DIR.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_history_profile_uses_official_pool_with_stronger_exposure():
    result = run_dry("history_profile_test", "134217728", "16384")

    assert result.returncode == 0, result.stderr
    command = result.stdout
    tokens = command.split()
    assert "selfplay.mode=vanilla" in tokens
    assert "selfplay.enabled=1" in tokens
    assert "selfplay.max_size=100" in tokens
    assert "vec.frozen_bank_pct=0.5" in tokens
    assert "vec.num_frozen_banks=1" in tokens
    assert "env.num_agents=2" in tokens
    assert "env.curriculum_enabled=1" in tokens
    assert "env.fixed_stage=0" in tokens
    assert "vec.frozen_bank_pct=0.1" not in tokens
    assert "selfplay.max_size=8" not in tokens


def test_history_profile_rejects_unsupported_fraction():
    result = run_dry(
        "history_profile_test", "1048576", "16384", "0.33", "100"
    )

    assert result.returncode != 0
    assert "frozen fraction" in result.stderr
