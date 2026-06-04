from pathlib import Path

from test_port_build import build_dogfight_backend


def test_native_backend_exposes_curriculum_target_setter():
    repo_root = Path(__file__).resolve().parents[3]
    result = build_dogfight_backend(repo_root)
    assert result.returncode == 0, result.stdout

    from pufferlib import _C

    assert _C.env_name == "dogfight"
    assert hasattr(_C, "set_curriculum_target")
