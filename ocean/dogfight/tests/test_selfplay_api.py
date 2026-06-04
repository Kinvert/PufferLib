from pathlib import Path

from test_port_build import build_dogfight_backend


def test_native_vecenv_exposes_selfplay_opponent_controls():
    repo_root = Path(__file__).resolve().parents[3]
    result = build_dogfight_backend(repo_root)
    assert result.returncode == 0, result.stdout

    from pufferlib import _C

    expected = {
        "get_opponent_observations",
        "set_opponent_actions",
        "enable_opponent_override",
        "set_opponent_obs_scheme",
        "set_opponent_buffers",
    }
    missing = sorted(name for name in expected if not hasattr(_C.VecEnv, name))

    assert missing == []
