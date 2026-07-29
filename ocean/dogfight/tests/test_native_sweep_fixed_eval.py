import configparser
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DOGFIGHT_DIR = REPO_ROOT / "ocean" / "dogfight"
SCRIPT = DOGFIGHT_DIR / "native_sweep.py"
PROFILE = DOGFIGHT_DIR / "native_sweep_profile.ini"
WRAPPER = DOGFIGHT_DIR / "sweep_vanilla_selfplay.sh"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location(
        "dogfight_native_sweep_fixed_eval", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_config(timeout_seconds=1800):
    return {
        "enabled": True,
        "stages": "0..10",
        "seeds": "42,314159,271828",
        "mirrors": "0,1",
        "episodes": 128,
        "threshold": 0.90,
        "timeout_seconds": timeout_seconds,
        "max_attempts": 3,
    }


def write_native_log(
    experiment: Path,
    run_id: str = "sweep_1_0000",
    final_steps: int = 100,
) -> Path:
    log_path = experiment / "logs" / "dogfight" / f"{run_id}.ini"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            "[base]\n"
            f"run_id = {run_id}\n"
            "\n"
            "[metrics]\n"
            f"agent_steps = {final_steps}\n"
            "env/perf = 0.5\n"
        ),
        encoding="ascii",
    )
    return log_path


def write_checkpoint(
    experiment: Path,
    run_id: str,
    filename: str,
    content: bytes = b"checkpoint",
) -> Path:
    path = (
        experiment
        / "checkpoints"
        / "dogfight"
        / run_id
        / filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def rejected_result(checkpoint: Path) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "threshold": 0.9,
        "highest_contiguous_stage": -1,
        "all_mastered": False,
        "min_perf": 0.35,
        "mean_perf": 0.35,
        "stopped_after_stage": 0,
        "stages": {
            "0": {
                "mastered": False,
                "cells": 6,
                "episodes": 768,
                "min_perf": 0.35,
                "mean_perf": 0.35,
                "max_abs_signed_bias": 62.0,
            }
        },
        "samples": [
            {
                "stage": 0,
                "seed": 42,
                "mirror": 0,
                "episodes": 128,
                "perf": 0.35,
                "score": 0.35,
                "timeouts": 0.0,
                "player_ground_hits": 0.0,
                "opponent_ground_hits": 0.0,
                "avg_abs_bias": 62.0,
                "avg_signed_bias": 62.0,
            }
        ],
    }


def test_profile_is_manifested_but_not_forwarded_to_puffer(tmp_path):
    module = load_sweep_module()
    experiment = tmp_path / "experiment"

    manifest = module.prepare_experiment(experiment, 1, 8_388_608)

    staged = configparser.ConfigParser(interpolation=None)
    staged.read(experiment / "config" / "dogfight.ini")
    assert not staged.has_section("fixed_eval")
    assert manifest["fixed_eval"] == fixed_config()
    assert manifest["inputs"][str(module.EVAL_STAGE_MATRIX)] == module.sha256(
        module.EVAL_STAGE_MATRIX
    )
    assert manifest["inputs"][str(module.EVAL_CHECKPOINT)] == module.sha256(
        module.EVAL_CHECKPOINT
    )

    profile = configparser.ConfigParser(interpolation=None)
    profile.read(PROFILE)
    assert profile["fixed_eval"]["stages"] == "0..10"
    assert 'export PUFFER_VENV="$puffer_venv"' in WRAPPER.read_text()
    assert "--screen-existing" in WRAPPER.read_text()


def test_final_checkpoint_matches_numeric_final_agent_step(tmp_path):
    module = load_sweep_module()
    log_path = write_native_log(tmp_path, final_steps=100)
    write_checkpoint(tmp_path, "sweep_1_0000", "9.bin")
    expected = write_checkpoint(
        tmp_path, "sweep_1_0000", "100.bin", b"final"
    )

    checkpoint, final_steps = module.resolve_final_checkpoint(
        tmp_path, log_path
    )

    assert checkpoint == expected
    assert final_steps == 100


def test_final_checkpoint_rejects_latest_periodic_checkpoint(tmp_path):
    module = load_sweep_module()
    log_path = write_native_log(tmp_path, final_steps=100)
    write_checkpoint(tmp_path, "sweep_1_0000", "99.bin")

    with pytest.raises(module.SweepError, match="final agent step 100"):
        module.resolve_final_checkpoint(tmp_path, log_path)


def test_rc2_with_valid_json_is_an_atomic_completed_rejection(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    log_path = write_native_log(tmp_path)
    checkpoint = write_checkpoint(
        tmp_path, "sweep_1_0000", "100.bin", b"final"
    )

    def fake_run(command, **kwargs):
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 60
        assert kwargs["env"]["PUFFER_VENV"] == str(module.PUFFER_VENV)
        output_path = Path(
            command[command.index("--json-output") + 1]
        )
        output_path.write_text(
            json.dumps(rejected_result(checkpoint)), encoding="ascii"
        )
        return subprocess.CompletedProcess(
            command, 2, stdout="candidate rejected\n"
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    entry = module.run_fixed_evaluation(
        tmp_path, log_path, fixed_config(timeout_seconds=60)
    )

    assert entry["status"] == "rejected"
    result_path = Path(entry["result_path"])
    result = json.loads(result_path.read_text())
    assert result["run_id"] == "sweep_1_0000"
    assert result["checkpoint_sha256"] == hashlib.sha256(b"final").hexdigest()
    assert result["evaluation_returncode"] == 2
    assert not list(result_path.parent.glob("*.tmp"))
    assert Path(entry["output_log"]).read_text() == "candidate rejected\n"


def test_rc2_without_valid_json_is_infrastructure_failure(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    log_path = write_native_log(tmp_path)
    write_checkpoint(tmp_path, "sweep_1_0000", "100.bin")

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, stdout="argument failure\n"
        ),
    )

    entry = module.run_fixed_evaluation(
        tmp_path, log_path, fixed_config(timeout_seconds=60)
    )

    assert entry["status"] == "failed"
    assert "valid JSON" in entry["error"]
    assert not (tmp_path / "fixed_eval" / "sweep_1_0000.json").exists()


def test_fixed_metrics_resume_same_wandb_run_without_history(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    result = rejected_result(Path("/tmp/final.bin"))
    result.update(
        {
            "run_id": "sweep_1_0000",
            "checkpoint_sha256": "abc123",
        }
    )
    captured = {}

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.finished = False

        def log(self, *args, **kwargs):
            raise AssertionError("fixed screening must not add W&B history")

        def finish(self):
            self.finished = True

    fake_run = FakeRun()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake_run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    module.upload_fixed_wandb_run(result, "df42")

    assert captured["id"] == "sweep_1_0000"
    assert captured["resume"] == "allow"
    assert fake_run.finished
    assert fake_run.summary == {
        "fixed/highest_contiguous_stage": -1,
        "fixed/all_mastered": False,
        "fixed/min_perf": 0.35,
        "fixed/mean_perf": 0.35,
        "fixed/stopped_after_stage": 0,
        "fixed/max_abs_signed_bias": 62.0,
        "fixed/evaluated_cells": 1,
        "fixed/checkpoint_sha256": "abc123",
    }


def test_drain_resume_skips_completed_hash_and_retries_only_wandb(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    log_path = write_native_log(tmp_path)
    checkpoint = write_checkpoint(
        tmp_path, "sweep_1_0000", "100.bin", b"final"
    )
    eval_calls = []
    upload_calls = []

    def fake_run(command, **kwargs):
        eval_calls.append(command)
        output_path = Path(
            command[command.index("--json-output") + 1]
        )
        output_path.write_text(
            json.dumps(rejected_result(checkpoint)), encoding="ascii"
        )
        return subprocess.CompletedProcess(command, 2, stdout="rejected\n")

    def failing_upload(result, project):
        upload_calls.append((result["run_id"], project))
        raise RuntimeError("offline")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module, "upload_fixed_wandb_run", failing_upload
    )

    first = module.drain_fixed_evaluations(
        tmp_path, "df42", fixed_config()
    )

    assert first["completed"] == 1
    assert first["rejected"] == 1
    assert first["wandb_pending"] == 1
    assert len(eval_calls) == 1

    monkeypatch.setattr(
        module,
        "upload_fixed_wandb_run",
        lambda result, project: upload_calls.append(
            (result["run_id"], project)
        ),
    )

    second = module.drain_fixed_evaluations(
        tmp_path, "df42", fixed_config()
    )

    assert second["completed"] == 1
    assert second["wandb_pending"] == 0
    assert len(eval_calls) == 1
    assert len(upload_calls) == 2
    state = json.loads(
        (tmp_path / "fixed_eval" / "state.json").read_text()
    )
    checkpoint_hash = hashlib.sha256(b"final").hexdigest()
    assert f"sweep_1_0000:{checkpoint_hash}" in state["runs"]


def test_nonzero_native_exit_drains_only_after_process_exit(
    tmp_path, monkeypatch
):
    module = load_sweep_module()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    command = tmp_path / "puffer"
    command.write_text("", encoding="ascii")
    (experiment / "manifest.json").write_text(
        json.dumps(
            {
                "native_command": [str(command)],
                "wandb_project": "df42",
                "fixed_eval": fixed_config(),
            }
        ),
        encoding="ascii",
    )
    events = []

    class FakeProcess:
        pid = 123

        def __init__(self):
            self.alive = True
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                return None
            return 7

        def wait(self, timeout=None):
            self.alive = False
            return 7

    process = FakeProcess()
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *args, **kwargs: process
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "load_upload_state", lambda path: set())
    monkeypatch.setattr(
        module,
        "upload_completed_runs",
        lambda *args, **kwargs: events.append(
            ("upload", process.alive)
        ),
    )

    def fake_drain(*args, **kwargs):
        events.append(("drain", process.alive))
        return {
            "completed": 0,
            "mastered": 0,
            "rejected": 0,
            "failures": 0,
            "wandb_pending": 0,
        }

    monkeypatch.setattr(module, "drain_fixed_evaluations", fake_drain)

    exit_code = module.run_experiment(experiment)

    assert exit_code == 7
    assert ("upload", True) in events
    assert events[-1] == ("drain", False)


@pytest.mark.parametrize(
    (
        "native_exit_code",
        "initial_status",
        "fixed_failures",
        "expected_status",
        "expected_returncode",
    ),
    [
        (0, "completed_with_screen_errors", 0, "completed", 0),
        (0, "completed", 1, "completed_with_screen_errors", 3),
        (7, "failed", 0, "failed", 0),
    ],
)
def test_screen_existing_recomputes_status_from_native_and_fixed_results(
    tmp_path,
    monkeypatch,
    native_exit_code,
    initial_status,
    fixed_failures,
    expected_status,
    expected_returncode,
):
    module = load_sweep_module()
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    manifest_path = experiment / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": initial_status,
                "exit_code": native_exit_code,
                "finished_utc": "old-finished-time",
                "wandb_project": "df42",
                "fixed_eval": fixed_config(),
            }
        ),
        encoding="ascii",
    )
    summary = {
        "completed": 1,
        "mastered": 0,
        "rejected": 1,
        "failures": fixed_failures,
        "wandb_pending": 0,
    }
    monkeypatch.setattr(
        module,
        "drain_fixed_evaluations",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        module, "utc_now", lambda: "new-screen-finished-time"
    )

    returncode = module.screen_existing(experiment)

    updated = json.loads(manifest_path.read_text())
    assert returncode == expected_returncode
    assert updated["status"] == expected_status
    assert updated["finished_utc"] == "new-screen-finished-time"
    assert (
        updated["fixed_eval_last_screen_utc"]
        == "new-screen-finished-time"
    )
    assert updated["fixed_eval_summary"] == summary
