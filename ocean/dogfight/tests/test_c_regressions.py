import os
import subprocess
from pathlib import Path

import pytest


def compile_and_run_c_test(repo_root: Path, test_name: str, tmp_path: Path):
    raylib_inc = repo_root / "raylib-5.5_linux_amd64" / "include"
    raylib_lib = repo_root / "raylib-5.5_linux_amd64" / "lib" / "libraylib.a"
    dogfight_inc = repo_root / "ocean" / "dogfight"
    src = dogfight_inc / "tests" / f"{test_name}.c"
    binary = tmp_path / test_name
    cc = os.environ.get("CC", "gcc")

    assert raylib_lib.is_file(), "run the default Dogfight build once to populate raylib"

    compile_result = subprocess.run(
        [
            cc,
            "-O2",
            "-Wall",
            "-I",
            str(dogfight_inc),
            "-I",
            str(raylib_inc),
            str(src),
            str(raylib_lib),
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
    assert compile_result.returncode == 0, compile_result.stdout

    run_result = subprocess.run(
        [str(binary)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout


@pytest.mark.parametrize(
    "test_name",
    [
        "test_training_telemetry",
        "test_curriculum_stage_geometry",
        "test_curriculum_max_steps",
        "test_action_bounds",
        "test_observation_padding",
        "test_observation_scheme1_reference",
        "test_scripted_step_reference",
        "test_terminal_kill_reference",
        "test_state_roundtrip",
    ],
)
def test_dogfight_c_regression(test_name, tmp_path):
    repo_root = Path(__file__).resolve().parents[3]

    compile_and_run_c_test(repo_root, test_name, tmp_path)
