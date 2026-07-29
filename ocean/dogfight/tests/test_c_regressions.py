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
    pytest.param(
    "test_multi_episode_reset_reference",
        marks=pytest.mark.skip(reason="Phase 2 logging/reset contract"),
    ),
    "test_action_bounds",
    "test_rng_isolation",
    "test_curriculum_controller",
    "test_curriculum_stage_mix",
    "test_flight_autoace",
    "test_flight_energy",
    "test_flight_obs_dynamic",
    "test_flight_obs_static",
    "test_flight_physics",
    "test_flight_recovery",
    pytest.param(
        "test_state_roundtrip",
        marks=pytest.mark.skip(
            reason="Rewrite in Phase 2 without the rejected duplicated State mirror"
        ),
    ),
    pytest.param(
        "test_curriculum_max_steps",
        marks=pytest.mark.skip(reason="Phase 2 curriculum contract"),
    ),
    "test_curriculum_stage_geometry",
    "test_curriculum_reset_selection",
    "test_native_selfplay_spawn",
    "test_two_agent_adapter",
    "test_two_agent_aileron_penalty",
    "test_two_agent_competitive_reward",
    "test_two_agent_bootstrap",
    "test_two_agent_lateral_symmetry",
    "test_fixed_eval_mirror",
]


def compile_and_run_c_test(repo_root: Path, test_name: str, tmp_path: Path):
    dogfight_dir = repo_root / "ocean" / "dogfight"
    dogfight_header = dogfight_dir / "dogfight.h"
    source_include = repo_root / "src"
    vendor_include = repo_root / "vendor"
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
            str(source_include),
            "-I",
            str(vendor_include),
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
