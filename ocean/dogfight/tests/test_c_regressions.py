import os
import subprocess
from pathlib import Path

import pytest


TESTS = [
    "test_observation_scheme1_reference",
    "test_scripted_step_reference",
    "test_scripted_trace_reference",
    "test_terminal_kill_reference",
    "test_terminal_opponent_kill_reference",
    "test_terminal_oob_reference",
    "test_terminal_timeout_reference",
    "test_multi_episode_reset_reference",
    "test_action_bounds",
    "test_state_roundtrip",
    "test_curriculum_max_steps",
    "test_curriculum_stage_geometry",
]


def compile_and_run_c_test(repo_root: Path, test_name: str, tmp_path: Path):
    dogfight_dir = repo_root / "ocean" / "dogfight"
    dogfight_header = dogfight_dir / "dogfight.h"
    raylib_dir = repo_root / "raylib-5.5_linux_amd64"
    raylib_include = raylib_dir / "include"
    raylib_library = raylib_dir / "lib" / "libraylib.a"
    source = dogfight_dir / "tests" / f"{test_name}.c"
    binary = tmp_path / test_name

    assert dogfight_header.is_file(), (
        "Phase 1 must provide ocean/dogfight/dogfight.h before C regressions run"
    )
    assert raylib_library.is_file(), (
        "Run the verified Dogfight build command once to populate raylib"
    )

    result = subprocess.run(
        [
            os.environ.get("CC", "gcc"),
            "-O2",
            "-Wall",
            "-I",
            str(dogfight_dir),
            "-I",
            str(raylib_include),
            str(source),
            str(raylib_library),
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            str(binary),
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    result = subprocess.run(
        [str(binary)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("test_name", TESTS)
def test_dogfight_c_regression(test_name, tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    compile_and_run_c_test(repo_root, test_name, tmp_path)

