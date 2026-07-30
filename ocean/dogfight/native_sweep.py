#!/usr/bin/env python3
"""Dogfight-local orchestration for PufferLib 5c's native Protein sweep."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOGFIGHT_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOGFIGHT_DIR.parents[1]
SOURCE_DEFAULT = REPO_ROOT / "config" / "default.ini"
SOURCE_DOGFIGHT = REPO_ROOT / "config" / "dogfight.ini"
SWEEP_PROFILE = DOGFIGHT_DIR / "native_sweep_profile.ini"
EVAL_STAGE_MATRIX = DOGFIGHT_DIR / "eval_stage_matrix.py"
EVAL_CHECKPOINT = DOGFIGHT_DIR / "eval_checkpoint.sh"
PUFFER_BINARY = REPO_ROOT / "puffer"
PUFFER_VENV = Path(sys.prefix)
VENV_PYTHON = Path(sys.executable)
NCCL_LIB = (
    PUFFER_VENV
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
    / "nccl"
    / "lib"
)
DEFAULT_EXPERIMENT_ROOT = DOGFIGHT_DIR / "sweeps"
MAX_EXPLICIT_METRICS = 31
FIXED_EVAL_EXIT_INFRASTRUCTURE_FAILURE = 3

# Put decision-critical and flight-quality measurements ahead of optional
# diagnostics if a future Dogfight log grows beyond the 31-value contract.
METRIC_PRIORITY = (
    "SPS",
    "uptime",
    "epoch",
    "env/perf",
    "env/score",
    "env/slot_0_score",
    "env/slot_1_score",
    "env/draw_rate",
    "env/slot_0_gun_kills",
    "env/slot_1_gun_kills",
    "env/episode_return",
    "env/episode_length",
    "env/avg_stage",
    "env/avg_stage_weight",
    "env/avg_abs_bias",
    "env/avg_signed_bias",
    "env/target_az_neg_aileron_sum",
    "env/target_az_pos_aileron_sum",
    "env/target_az_neg_steps",
    "env/target_az_pos_steps",
    "env/avg_control_rate",
    "env/base_stage_kills",
    "env/base_stage_eps",
    "env/clean_fights",
    "env/player_ground_hits",
    "env/opponent_ground_hits",
    "env/recovery_triggers",
    "env/mastered_stage",
    "env/curriculum_target",
)


class SweepError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ini(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    if not config.read(path):
        raise SweepError(f"could not read INI: {path}")
    return config


def write_ini(config: configparser.ConfigParser, path: Path) -> None:
    with path.open("w", encoding="ascii") as output:
        config.write(output)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def overlay_ini(
    destination: configparser.ConfigParser,
    overlay: configparser.ConfigParser,
) -> None:
    for section in overlay.sections():
        if not destination.has_section(section):
            destination.add_section(section)
        for key, value in overlay.items(section):
            destination.set(section, key, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = git("status", "--short")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def fixed_eval_profile(
    profile: configparser.ConfigParser,
) -> dict[str, Any]:
    section = "fixed_eval"
    if not profile.has_section(section):
        raise SweepError("[fixed_eval] is required in native sweep profile")
    result = {
        "enabled": profile.getboolean(section, "enabled"),
        "stages": profile.get(section, "stages"),
        "seeds": profile.get(section, "seeds"),
        "mirrors": profile.get(section, "mirrors"),
        "episodes": profile.getint(section, "episodes"),
        "threshold": profile.getfloat(section, "threshold"),
        "timeout_seconds": profile.getint(section, "timeout_seconds"),
        "max_attempts": profile.getint(section, "max_attempts"),
    }
    if not result["stages"].strip() or not result["seeds"].strip():
        raise SweepError("fixed evaluation stages and seeds cannot be empty")
    if result["mirrors"].replace(" ", "") != "0,1":
        raise SweepError("fixed evaluation mirrors must be exactly 0,1")
    if result["episodes"] < 128:
        raise SweepError("fixed evaluation episodes must be at least 128")
    if not 0.0 <= result["threshold"] <= 1.0:
        raise SweepError("fixed evaluation threshold must be in [0, 1]")
    if result["timeout_seconds"] < 1 or result["max_attempts"] < 1:
        raise SweepError("fixed evaluation timeout/attempts must be positive")
    return result


def selected_sweep_dimensions(
    dogfight: configparser.ConfigParser,
) -> list[str]:
    if not dogfight.has_option("sweep", "sweep_only"):
        raise SweepError("[sweep] sweep_only is required")
    selected = [
        name.strip()
        for name in dogfight.get("sweep", "sweep_only").split(",")
        if name.strip()
    ]
    if not selected:
        raise SweepError("[sweep] sweep_only must select at least one dimension")
    if len(selected) != len(set(selected)):
        raise SweepError("[sweep] sweep_only contains duplicate dimensions")
    if len(selected) > MAX_EXPLICIT_METRICS:
        raise SweepError("sweep dimension count exceeds 31")

    present = {
        section.removeprefix("sweep.")
        for section in dogfight.sections()
        if section.startswith("sweep.")
    }
    missing = set(selected) - present
    if missing:
        raise SweepError(
            "Dogfight sweep_only references missing sections: "
            f"{sorted(missing)}"
        )
    for dimension in present - set(selected):
        dogfight.remove_section(f"sweep.{dimension}")
    return sorted(selected)


def native_command(
    max_runs: int,
    timesteps: int | None,
    project: str,
) -> list[str]:
    command = [
        str(PUFFER_BINARY),
        "sweep",
        "dogfight",
        "--wandb",
        f"--wandb-project={project}",
        f"sweep.max_runs={max_runs}",
        "sweep.gpus=1",
        "train.gpus=1",
    ]
    if timesteps is not None:
        command.append(f"train.total_timesteps={timesteps}")
    return command


def prepare_experiment(
    experiment_dir: Path,
    max_runs: int,
    timesteps: int | None,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise SweepError(f"experiment directory is not empty: {experiment_dir}")
    config_dir = experiment_dir / "config"
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    default = read_ini(SOURCE_DEFAULT)
    for section in list(default.sections()):
        if section.startswith("sweep."):
            default.remove_section(section)

    dogfight = read_ini(SOURCE_DOGFIGHT)
    profile = read_ini(SWEEP_PROFILE)
    fixed_eval = fixed_eval_profile(profile)
    profile.remove_section("fixed_eval")
    overlay_ini(dogfight, profile)
    dogfight.set("base", "checkpoint_dir", str(checkpoint_dir))
    dogfight.set("base", "log_dir", str(log_dir))
    dogfight.set("sweep", "max_runs", str(max_runs))
    if timesteps is not None:
        dogfight.set("train", "total_timesteps", str(timesteps))

    dimensions = selected_sweep_dimensions(dogfight)
    explicit_log_values = count_explicit_log_values()
    if explicit_log_values > MAX_EXPLICIT_METRICS:
        raise SweepError(
            f"Dogfight exports {explicit_log_values} values; maximum is 31"
        )

    staged_default = config_dir / "default.ini"
    staged_dogfight = config_dir / "dogfight.ini"
    write_ini(default, staged_default)
    write_ini(dogfight, staged_dogfight)

    project = dogfight.get("base", "wandb_project")
    command = native_command(max_runs, timesteps, project)
    manifest = {
        "format": "dogfight-native-protein-sweep-v1",
        "created_utc": utc_now(),
        "experiment_dir": str(experiment_dir),
        "max_runs": max_runs,
        "timesteps_override": timesteps,
        "wandb_project": project,
        "fixed_eval": fixed_eval,
        "native_sweep_dimensions": dimensions,
        "explicit_dogfight_log_values": explicit_log_values,
        "checkpoint_interval": dogfight.getint(
            "base", "checkpoint_interval"
        ),
        "native_command": command,
        "source": source_revision(),
        "inputs": {
            str(SOURCE_DEFAULT): sha256(SOURCE_DEFAULT),
            str(SOURCE_DOGFIGHT): sha256(SOURCE_DOGFIGHT),
            str(SWEEP_PROFILE): sha256(SWEEP_PROFILE),
            str(EVAL_STAGE_MATRIX): sha256(EVAL_STAGE_MATRIX),
            str(EVAL_CHECKPOINT): sha256(EVAL_CHECKPOINT),
            str(PUFFER_BINARY): sha256(PUFFER_BINARY)
            if PUFFER_BINARY.exists()
            else "missing",
        },
        "staged": {
            str(staged_default): sha256(staged_default),
            str(staged_dogfight): sha256(staged_dogfight),
        },
        "status": "prepared",
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return manifest


def count_explicit_log_values() -> int:
    source = (DOGFIGHT_DIR / "dogfight_puffer.h").read_text(encoding="ascii")
    match = re.search(
        r"void puf_log\(Log\* log, Dict\* out\) \{(?P<body>.*?)\n\}",
        source,
        re.DOTALL,
    )
    if match is None:
        raise SweepError("could not find puf_log in dogfight_puffer.h")
    keys = re.findall(
        r'dict_set\(\s*out\s*,\s*"([^"]+)"',
        match.group("body"),
        re.DOTALL,
    )
    if len(keys) != len(set(keys)):
        raise SweepError("puf_log contains duplicate output keys")
    return len(keys)


def parse_metric_values(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def build_wandb_payload(log_path: Path) -> dict[str, Any]:
    log = read_ini(log_path)
    if not log.has_section("metrics"):
        raise SweepError(f"native log has no [metrics] section: {log_path}")
    if not log.has_option("metrics", "agent_steps"):
        raise SweepError(f"native log has no agent_steps: {log_path}")

    metric_values = {
        key: parse_metric_values(value)
        for key, value in log.items("metrics")
        if key != "agent_steps"
    }
    steps = [
        int(round(value))
        for value in parse_metric_values(log.get("metrics", "agent_steps"))
    ]
    if not steps:
        raise SweepError(f"native log has empty agent_steps: {log_path}")

    selected = [
        key for key in METRIC_PRIORITY if key in metric_values
    ]
    selected.extend(
        key
        for key in sorted(metric_values)
        if key not in selected
    )
    selected = selected[:MAX_EXPLICIT_METRICS]

    history = []
    for index, step in enumerate(steps):
        row = {}
        for key in selected:
            values = metric_values[key]
            if index < len(values):
                row[key] = values[index]
        history.append({"step": step, "metrics": row})

    run_id = log.get("base", "run_id")
    flattened_config = {
        f"{section}.{key}": value
        for section in log.sections()
        if section != "metrics"
        for key, value in log.items(section)
    }
    summary: dict[str, Any] = {"final_agent_steps": steps[-1]}
    for key in selected:
        values = metric_values[key]
        if values:
            summary[f"final_{key}"] = values[-1]

    return {
        "run_id": run_id,
        "config": flattened_config,
        "history": history,
        "summary": summary,
        "selected_metrics": selected,
    }


def final_checkpoint(experiment_dir: Path, run_id: str) -> Path | None:
    run_dir = experiment_dir / "checkpoints" / "dogfight" / run_id
    checkpoints = sorted(run_dir.glob("*.bin"))
    return checkpoints[-1] if checkpoints else None


def checkpoint_agent_step(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def resolve_final_checkpoint(
    experiment_dir: Path,
    log_path: Path,
) -> tuple[Path, int]:
    payload = build_wandb_payload(log_path)
    run_id = payload["run_id"]
    final_steps = int(payload["summary"]["final_agent_steps"])
    run_dir = experiment_dir / "checkpoints" / "dogfight" / run_id
    checkpoints = [
        path
        for path in run_dir.glob("*.bin")
        if checkpoint_agent_step(path) == final_steps
    ]
    if len(checkpoints) != 1:
        available = sorted(
            step
            for step in (
                checkpoint_agent_step(path)
                for path in run_dir.glob("*.bin")
            )
            if step is not None
        )
        raise SweepError(
            f"{run_id} has no unique checkpoint at final agent step "
            f"{final_steps}; available={available}"
        )
    checkpoint = checkpoints[0]
    if checkpoint.stat().st_size < 1:
        raise SweepError(f"final checkpoint is empty: {checkpoint}")
    return checkpoint, final_steps


def upload_wandb_run(
    experiment_dir: Path,
    log_path: Path,
    project: str,
) -> None:
    payload = build_wandb_payload(log_path)
    import wandb

    run = wandb.init(
        project=project,
        id=payload["run_id"],
        job_type="native-protein-trial",
        config=payload["config"],
        resume="allow",
        reinit=True,
        settings=wandb.Settings(console="off"),
    )
    try:
        for row in payload["history"]:
            run.log(
                {"agent_steps": row["step"], **row["metrics"]},
                step=row["step"],
            )
        run.summary.update(payload["summary"])
        checkpoint = final_checkpoint(experiment_dir, payload["run_id"])
        if checkpoint is not None:
            run.summary["checkpoint_path"] = str(checkpoint)
            run.summary["checkpoint_sha256"] = sha256(checkpoint)
        run.summary["native_log_path"] = str(log_path)
        run.summary["sweep_manifest_path"] = str(
            experiment_dir / "manifest.json"
        )
    finally:
        run.finish()


def fixed_wandb_summary(result: dict[str, Any]) -> dict[str, Any]:
    samples = result.get("samples", [])
    signed_biases = [
        abs(float(sample["avg_signed_bias"]))
        for sample in samples
        if sample.get("avg_signed_bias") is not None
    ]
    return {
        "fixed/highest_contiguous_stage": result[
            "highest_contiguous_stage"
        ],
        "fixed/all_mastered": result["all_mastered"],
        "fixed/min_perf": result["min_perf"],
        "fixed/mean_perf": result["mean_perf"],
        "fixed/stopped_after_stage": result["stopped_after_stage"],
        "fixed/max_abs_signed_bias": (
            max(signed_biases) if signed_biases else None
        ),
        "fixed/evaluated_cells": len(samples),
        "fixed/checkpoint_sha256": result["checkpoint_sha256"],
    }


def upload_fixed_wandb_run(
    result: dict[str, Any],
    project: str,
) -> None:
    import wandb

    run = wandb.init(
        project=project,
        id=result["run_id"],
        job_type="native-protein-trial",
        resume="allow",
        reinit=True,
        settings=wandb.Settings(console="off"),
    )
    try:
        run.summary.update(fixed_wandb_summary(result))
    finally:
        run.finish()


def load_upload_state(experiment_dir: Path) -> set[str]:
    path = experiment_dir / "wandb_uploaded.json"
    if not path.exists():
        return set()
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (json.JSONDecodeError, OSError):
        return set()
    return set(value.get("uploaded", []))


def save_upload_state(experiment_dir: Path, uploaded: set[str]) -> None:
    path = experiment_dir / "wandb_uploaded.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"updated_utc": utc_now(), "uploaded": sorted(uploaded)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def fixed_eval_root(experiment_dir: Path) -> Path:
    root = experiment_dir / "fixed_eval"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_fixed_eval_state(experiment_dir: Path) -> dict[str, Any]:
    path = fixed_eval_root(experiment_dir) / "state.json"
    if not path.exists():
        return {"format": "dogfight-fixed-eval-state-v1", "runs": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise SweepError(f"could not read fixed evaluation state: {error}")
    if (
        state.get("format") != "dogfight-fixed-eval-state-v1"
        or not isinstance(state.get("runs"), dict)
    ):
        raise SweepError(f"invalid fixed evaluation state: {path}")
    return state


def save_fixed_eval_state(
    experiment_dir: Path,
    state: dict[str, Any],
) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(fixed_eval_root(experiment_dir) / "state.json", state)


def load_fixed_wandb_state(experiment_dir: Path) -> set[str]:
    path = fixed_eval_root(experiment_dir) / "wandb_uploaded.json"
    if not path.exists():
        return set()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise SweepError(f"could not read fixed W&B state: {error}")
    uploaded = value.get("uploaded")
    if not isinstance(uploaded, list):
        raise SweepError(f"invalid fixed W&B state: {path}")
    return set(uploaded)


def save_fixed_wandb_state(
    experiment_dir: Path,
    uploaded: set[str],
) -> None:
    atomic_write_json(
        fixed_eval_root(experiment_dir) / "wandb_uploaded.json",
        {"updated_utc": utc_now(), "uploaded": sorted(uploaded)},
    )


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SweepError(f"unsafe native run id: {run_id!r}")
    return run_id


def _validate_fixed_eval_result(
    result: Any,
    checkpoint: Path,
    returncode: int,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise SweepError("fixed evaluator did not produce valid JSON object")
    required = {
        "checkpoint",
        "highest_contiguous_stage",
        "all_mastered",
        "min_perf",
        "mean_perf",
        "stopped_after_stage",
        "samples",
    }
    missing = sorted(required - set(result))
    if missing or not isinstance(result["samples"], list):
        raise SweepError(
            "fixed evaluator did not produce valid JSON "
            f"(missing={missing})"
        )
    if Path(result["checkpoint"]).resolve() != checkpoint.resolve():
        raise SweepError("fixed evaluator JSON references wrong checkpoint")
    if returncode == 0 and result["all_mastered"] is not True:
        raise SweepError("fixed evaluator rc 0 contradicts all_mastered")
    if returncode == 2 and result["all_mastered"] is not False:
        raise SweepError("fixed evaluator rc 2 contradicts all_mastered")
    return result


def run_fixed_evaluation(
    experiment_dir: Path,
    log_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    root = fixed_eval_root(experiment_dir)
    run_id = _safe_run_id(
        read_ini(log_path).get("base", "run_id")
    )
    started_utc = utc_now()
    output_log = root / f"{run_id}-unresolved.log"
    temporary_json: Path | None = None
    captured_output = ""
    try:
        checkpoint, final_steps = resolve_final_checkpoint(
            experiment_dir, log_path
        )
        checkpoint_hash = sha256(checkpoint)
        stem = f"{run_id}-{checkpoint_hash[:12]}"
        output_log = root / f"{stem}.log"
        result_path = root / f"{stem}.json"
        temporary_json = root / f".{stem}.result.tmp"
        temporary_json.unlink(missing_ok=True)
        command = [
            str(VENV_PYTHON),
            str(EVAL_STAGE_MATRIX),
            str(checkpoint),
            "--stages",
            str(config["stages"]),
            "--seeds",
            str(config["seeds"]),
            "--mirrors",
            str(config["mirrors"]),
            "--episodes",
            str(config["episodes"]),
            "--threshold",
            str(config["threshold"]),
            "--eval-script",
            str(EVAL_CHECKPOINT),
            "--json-output",
            str(temporary_json),
        ]
        environment = os.environ.copy()
        environment["PUFFER_VENV"] = str(PUFFER_VENV)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(config["timeout_seconds"]),
        )
        captured_output = completed.stdout or ""
        atomic_write_text(output_log, captured_output)
        if completed.returncode not in (0, 2):
            raise SweepError(
                f"fixed evaluator exited {completed.returncode}"
            )
        if not temporary_json.is_file():
            raise SweepError("fixed evaluator did not produce valid JSON")
        try:
            raw_result = json.loads(
                temporary_json.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as error:
            raise SweepError(
                f"fixed evaluator did not produce valid JSON: {error}"
            )
        result = _validate_fixed_eval_result(
            raw_result, checkpoint, completed.returncode
        )
        result.update(
            {
                "format": "dogfight-fixed-stage-screen-v1",
                "run_id": run_id,
                "native_log_path": str(log_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "final_agent_steps": final_steps,
                "evaluation_returncode": completed.returncode,
                "evaluation_command": command,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
            }
        )
        atomic_write_json(result_path, result)
        return {
            "status": (
                "mastered" if result["all_mastered"] else "rejected"
            ),
            "run_id": run_id,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "result_path": str(result_path),
            "output_log": str(output_log),
            "finished_utc": utc_now(),
        }
    except KeyboardInterrupt:
        raise
    except subprocess.TimeoutExpired as error:
        timeout_output = error.stdout or ""
        if isinstance(timeout_output, bytes):
            timeout_output = timeout_output.decode(
                "utf-8", errors="replace"
            )
        captured_output = str(timeout_output)
        atomic_write_text(output_log, captured_output)
        return {
            "status": "failed",
            "run_id": run_id,
            "error": (
                f"fixed evaluator timed out after "
                f"{config['timeout_seconds']} seconds"
            ),
            "output_log": str(output_log),
            "finished_utc": utc_now(),
        }
    except Exception as error:
        if not output_log.exists():
            atomic_write_text(output_log, captured_output)
        return {
            "status": "failed",
            "run_id": run_id,
            "error": str(error),
            "output_log": str(output_log),
            "finished_utc": utc_now(),
        }
    finally:
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)


def _read_completed_fixed_result(
    entry: dict[str, Any],
    checkpoint_hash: str,
) -> dict[str, Any] | None:
    if entry.get("status") not in ("mastered", "rejected"):
        return None
    if entry.get("checkpoint_sha256") != checkpoint_hash:
        return None
    try:
        result = json.loads(
            Path(entry["result_path"]).read_text(encoding="utf-8")
        )
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    if (
        result.get("checkpoint_sha256") != checkpoint_hash
        or result.get("run_id") != entry.get("run_id")
    ):
        return None
    return result


def _append_fixed_failure(
    experiment_dir: Path,
    filename: str,
    message: str,
) -> None:
    path = fixed_eval_root(experiment_dir) / filename
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{utc_now()} {message}\n")


def drain_fixed_evaluations(
    experiment_dir: Path,
    project: str,
    config: dict[str, Any],
) -> dict[str, int]:
    summary = {
        "completed": 0,
        "mastered": 0,
        "rejected": 0,
        "failures": 0,
        "wandb_pending": 0,
    }
    if not config.get("enabled", False):
        return summary

    state = load_fixed_eval_state(experiment_dir)
    uploaded = load_fixed_wandb_state(experiment_dir)
    log_root = experiment_dir / "logs" / "dogfight"
    for log_path in sorted(log_root.glob("sweep_*.ini")):
        try:
            payload = build_wandb_payload(log_path)
            run_id = _safe_run_id(payload["run_id"])
            checkpoint, _ = resolve_final_checkpoint(
                experiment_dir, log_path
            )
            checkpoint_hash = sha256(checkpoint)
            key = f"{run_id}:{checkpoint_hash}"
            previous = state["runs"].get(key, {})
            result = _read_completed_fixed_result(
                previous, checkpoint_hash
            )
            if result is None:
                entry = run_fixed_evaluation(
                    experiment_dir, log_path, config
                )
                entry["attempts"] = int(previous.get("attempts", 0)) + 1
                state["runs"][key] = entry
                save_fixed_eval_state(experiment_dir, state)
                if entry["status"] == "failed":
                    summary["failures"] += 1
                    _append_fixed_failure(
                        experiment_dir,
                        "failures.log",
                        f"{key}: {entry['error']}",
                    )
                    continue
                result = _read_completed_fixed_result(
                    entry, checkpoint_hash
                )
                if result is None:
                    raise SweepError(
                        f"fixed result was not durable for {key}"
                    )
            entry = state["runs"][key]
            summary["completed"] += 1
            summary[entry["status"]] += 1

            if key in uploaded:
                continue
            try:
                upload_fixed_wandb_run(result, project)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                summary["wandb_pending"] += 1
                _append_fixed_failure(
                    experiment_dir,
                    "wandb_failures.log",
                    f"{key}: {error!r}",
                )
                continue
            uploaded.add(key)
            save_fixed_wandb_state(experiment_dir, uploaded)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            summary["failures"] += 1
            failure_key = f"log:{log_path.name}"
            previous = state["runs"].get(failure_key, {})
            state["runs"][failure_key] = {
                "status": "failed",
                "run_id": log_path.stem,
                "native_log_path": str(log_path),
                "attempts": int(previous.get("attempts", 0)) + 1,
                "error": str(error),
                "finished_utc": utc_now(),
            }
            save_fixed_eval_state(experiment_dir, state)
            _append_fixed_failure(
                experiment_dir,
                "failures.log",
                f"{failure_key}: {error}",
            )
    return summary


def upload_completed_runs(
    experiment_dir: Path,
    project: str,
    uploaded: set[str],
    retry_after: dict[str, float],
) -> None:
    log_root = experiment_dir / "logs" / "dogfight"
    failures = experiment_dir / "wandb_upload_failures.log"
    now = time.monotonic()
    for log_path in sorted(log_root.glob("sweep_*.ini")):
        key = log_path.name
        if key in uploaded or now < retry_after.get(key, 0.0):
            continue
        try:
            upload_wandb_run(experiment_dir, log_path, project)
        except Exception as error:
            retry_after[key] = now + 60.0
            with failures.open("a", encoding="utf-8") as output:
                output.write(f"{utc_now()} {key}: {error!r}\n")
            continue
        uploaded.add(key)
        retry_after.pop(key, None)
        save_upload_state(experiment_dir, uploaded)


def update_manifest(experiment_dir: Path, **updates: Any) -> None:
    path = experiment_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="ascii"))
    manifest.update(updates)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def run_experiment(experiment_dir: Path) -> int:
    experiment_dir = experiment_dir.resolve()
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        raise SweepError(f"missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    command = [str(value) for value in manifest["native_command"]]
    if not Path(command[0]).is_file():
        raise SweepError(f"missing native Puffer binary: {command[0]}")

    environment = os.environ.copy()
    if NCCL_LIB.is_dir():
        old_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{NCCL_LIB}:{old_path}" if old_path else str(NCCL_LIB)
        )
    environment["WANDB_PROJECT"] = manifest["wandb_project"]

    sweep_log = experiment_dir / "sweep.log"
    uploaded = load_upload_state(experiment_dir)
    retry_after: dict[str, float] = {}
    interrupted: KeyboardInterrupt | None = None
    with sweep_log.open("a", encoding="utf-8") as output:
        output.write(f"{utc_now()} command: {shlex.join(command)}\n")
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=experiment_dir,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        update_manifest(
            experiment_dir,
            status="running",
            started_utc=utc_now(),
            pid=process.pid,
        )
        try:
            while process.poll() is None:
                upload_completed_runs(
                    experiment_dir,
                    manifest["wandb_project"],
                    uploaded,
                    retry_after,
                )
                time.sleep(5.0)
            exit_code = process.wait()
        except KeyboardInterrupt as error:
            interrupted = error
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
            exit_code = process.wait(timeout=30.0)

    # Native logs are durable by worker exit. Make one final upload pass without
    # delaying or terminating training if W&B is temporarily unavailable.
    retry_after.clear()
    upload_completed_runs(
        experiment_dir,
        manifest["wandb_project"],
        uploaded,
        retry_after,
    )
    fixed_eval_summary = drain_fixed_evaluations(
        experiment_dir,
        manifest["wandb_project"],
        manifest["fixed_eval"],
    )
    screen_failed = fixed_eval_summary["failures"] > 0
    update_manifest(
        experiment_dir,
        status=(
            "failed"
            if exit_code != 0
            else (
                "completed_with_screen_errors"
                if screen_failed
                else "completed"
            )
        ),
        finished_utc=utc_now(),
        exit_code=exit_code,
        uploaded_runs=len(uploaded),
        fixed_eval_summary=fixed_eval_summary,
    )
    if interrupted is not None:
        raise interrupted
    if exit_code != 0:
        return exit_code
    if screen_failed:
        return FIXED_EVAL_EXIT_INFRASTRUCTURE_FAILURE
    return 0


def screen_existing(experiment_dir: Path) -> int:
    experiment_dir = experiment_dir.resolve()
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        raise SweepError(f"missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "fixed_eval" not in manifest:
        raise SweepError("experiment manifest has no fixed_eval protocol")
    summary = drain_fixed_evaluations(
        experiment_dir,
        manifest["wandb_project"],
        manifest["fixed_eval"],
    )
    native_exit_code = manifest.get("exit_code")
    if native_exit_code == 0:
        status = (
            "completed_with_screen_errors"
            if summary["failures"]
            else "completed"
        )
    else:
        status = manifest.get("status", "failed")
    screen_finished_utc = utc_now()
    update_manifest(
        experiment_dir,
        status=status,
        finished_utc=screen_finished_utc,
        fixed_eval_summary=summary,
        fixed_eval_last_screen_utc=screen_finished_utc,
    )
    return (
        FIXED_EVAL_EXIT_INFRASTRUCTURE_FAILURE
        if summary["failures"]
        else 0
    )


def validate_positive(name: str, value: int) -> int:
    if value < 1:
        raise argparse.ArgumentTypeError(f"{name} must be at least 1")
    return value


def session_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "session may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def default_experiment_id() -> str:
    return datetime.now().strftime("df42-native-sweep-%Y%m%d-%H%M%S")


def launch_command(
    experiment_dir: Path,
    session: str,
) -> list[str]:
    inner = [
        str(VENV_PYTHON),
        str(Path(__file__).resolve()),
        "run",
        "--experiment-dir",
        str(experiment_dir),
    ]
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(REPO_ROOT),
        shlex.join(inner),
    ]


def launch(args: argparse.Namespace) -> int:
    experiment_id = args.experiment_id or default_experiment_id()
    experiment_dir = (args.experiment_root / experiment_id).resolve()
    session = args.session or experiment_id
    native = native_command(args.max_runs, args.timesteps, "df42")
    tmux = launch_command(experiment_dir, session)

    if args.dry_run:
        print(f"native: {shlex.join(native)}")
        print(f"tmux: {shlex.join(tmux)}")
        return 0

    prepare_experiment(experiment_dir, args.max_runs, args.timesteps)
    if args.foreground:
        return run_experiment(experiment_dir)

    subprocess.run(tmux, cwd=REPO_ROOT, check=True)
    print(f"session={session}")
    print(f"experiment={experiment_dir}")
    print(f"log={experiment_dir / 'sweep.log'}")
    print(f"attach: tmux attach -t {shlex.quote(session)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dogfight-local PufferLib 5c native sweep launcher"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument(
        "--experiment-dir", type=Path, required=True
    )
    prepare.add_argument(
        "--max-runs",
        type=lambda value: validate_positive("max-runs", int(value)),
        required=True,
    )
    prepare.add_argument(
        "--timesteps",
        type=lambda value: validate_positive("timesteps", int(value)),
    )

    run = subparsers.add_parser("run")
    run.add_argument("--experiment-dir", type=Path, required=True)

    screen = subparsers.add_parser("screen")
    screen.add_argument("--experiment-dir", type=Path, required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--foreground", action="store_true")
    launch_parser.add_argument(
        "--max-runs",
        type=lambda value: validate_positive("max-runs", int(value)),
        default=1000,
    )
    launch_parser.add_argument(
        "--timesteps",
        type=lambda value: validate_positive("timesteps", int(value)),
    )
    launch_parser.add_argument("--session", type=session_name)
    launch_parser.add_argument("--experiment-id", type=session_name)
    launch_parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_experiment(
            args.experiment_dir,
            args.max_runs,
            args.timesteps,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return run_experiment(args.experiment_dir)
    if args.command == "screen":
        return screen_existing(args.experiment_dir)
    if args.command == "launch":
        return launch(args)
    raise SweepError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (SweepError, subprocess.CalledProcessError) as error:
        print(f"native sweep error: {error}", file=sys.stderr)
        raise SystemExit(2)
