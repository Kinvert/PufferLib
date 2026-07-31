#!/usr/bin/env python3
"""Upload completed native Protein trials to W&B without controlling the sweep."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FORMAT = "dogfight-wandb-sidecar-v1"
POOL_SCORE_KEY = "selfplay/pool_score"
POOL_QUALITY_KEY = "selfplay/pool_flight_quality"
POOL_FITNESS_KEY = "selfplay/pool_fitness"
FINAL_POOL_KEYS = frozenset(
    (POOL_SCORE_KEY, POOL_QUALITY_KEY, POOL_FITNESS_KEY)
)


class SidecarError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    compact = value.replace("_", "")
    try:
        return int(compact)
    except ValueError:
        pass
    try:
        return float(compact)
    except ValueError:
        return value


def parse_metric_values(raw: str, key: str) -> list[float]:
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return []
    try:
        return [
            float(item.strip().replace("_", ""))
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as error:
        raise SidecarError(f"invalid native metric {key}: {raw!r}") from error


def final_metric(metrics: dict[str, list[float]], key: str) -> float | None:
    values = metrics.get(key, [])
    if not values:
        return None
    value = values[-1]
    return value if math.isfinite(value) else None


def load_native_log(path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        with path.open(encoding="utf-8") as source:
            parser.read_file(source)
    except (OSError, configparser.Error) as error:
        raise SidecarError(f"could not read native log {path}: {error}") from error

    if not parser.has_section("metrics"):
        raise SidecarError(f"native log has no [metrics] section: {path}")
    if not parser.has_option("metrics", "agent_steps"):
        raise SidecarError(f"native log has no agent_steps metric: {path}")

    run_id = parser.get("base", "run_id", fallback=path.stem).strip("'\"")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise SidecarError(f"unsafe native run id: {run_id!r}")

    config: dict[str, Any] = {}
    for section in parser.sections():
        if section == "metrics":
            continue
        for key, value in parser.items(section):
            config[f"{section}.{key}"] = parse_scalar(value)

    metrics = {
        key: parse_metric_values(value, key)
        for key, value in parser.items("metrics")
    }
    steps = [int(round(value)) for value in metrics.pop("agent_steps")]
    if not steps:
        raise SidecarError(f"native log has empty agent_steps: {path}")
    if any(step < 0 for step in steps):
        raise SidecarError(f"native log has negative agent_steps: {path}")

    history = []
    for index, step in enumerate(steps):
        row: dict[str, float] = {}
        for key, values in metrics.items():
            # Native serialization backfills final-only pool metrics into
            # every downsample bucket. They are measured only after training,
            # so expose them at the final step instead of as fake flat curves.
            if key in FINAL_POOL_KEYS:
                if index == len(steps) - 1 and values:
                    row[key] = values[-1]
                continue
            # Native metrics can appear only at the end of a run. Align shorter
            # columns to the final steps instead of incorrectly logging at step 0.
            offset = max(len(steps) - len(values), 0)
            value_index = index - offset
            if 0 <= value_index < len(values):
                row[key] = values[value_index]
        history.append({"step": step, "metrics": row})

    pool_score = final_metric(metrics, POOL_SCORE_KEY)
    pool_quality = final_metric(metrics, POOL_QUALITY_KEY)
    pool_fitness = final_metric(metrics, POOL_FITNESS_KEY)
    valid_pool_eval = all(
        value is not None
        for value in (pool_score, pool_quality, pool_fitness)
    )
    final_steps = steps[-1]
    final_uptime = final_metric(metrics, "uptime")
    summary: dict[str, Any] = {
        "native/run_id": run_id,
        "native/log_path": str(path.resolve()),
        "native/final_agent_steps": final_steps,
        "selfplay/valid_pool_eval": valid_pool_eval,
        "selfplay/pool_winrate": pool_score,
        "selfplay/pool_flight_quality": pool_quality,
        "selfplay/pool_fitness": pool_fitness,
        "protein/fitness": pool_fitness,
    }
    if final_uptime is not None:
        summary["native/uptime"] = final_uptime
        if final_uptime > 0:
            summary["native/mean_sps"] = final_steps / final_uptime
    final_sps = final_metric(metrics, "SPS")
    if final_sps is not None:
        summary["native/final_sps"] = final_sps

    return {
        "run_id": run_id,
        "config": config,
        "metrics": metrics,
        "history": history,
        "summary": summary,
        "valid_pool_eval": valid_pool_eval,
    }


def load_or_create_state(
    path: Path,
    existing_logs: list[Path],
    include_existing: bool = False,
) -> dict[str, Any]:
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SidecarError(f"could not read sidecar state {path}: {error}")
        if state.get("format") != STATE_FORMAT:
            raise SidecarError(f"unsupported sidecar state: {path}")
        if not isinstance(state.get("runs"), dict):
            raise SidecarError(f"invalid sidecar runs state: {path}")
        state.setdefault("baseline", [])
        return state

    state = {
        "format": STATE_FORMAT,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "baseline": (
            [] if include_existing else sorted(item.name for item in existing_logs)
        ),
        "runs": {},
    }
    atomic_write_json(path, state)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_write_json(path, state)


def discover_logs(
    log_dir: Path,
    state: dict[str, Any],
    settle_seconds: float,
    now: float | None = None,
) -> list[Path]:
    now = time.time() if now is None else now
    baseline = set(state.get("baseline", []))
    runs = state.get("runs", {})
    candidates = []
    for path in log_dir.glob("*.ini"):
        if path.name in baseline:
            continue
        if runs.get(path.name, {}).get("status") == "uploaded":
            continue
        try:
            if now - path.stat().st_mtime < settle_seconds:
                continue
        except OSError:
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda item: (item.stat().st_mtime_ns, item.name))


def final_checkpoint(
    checkpoint_root: Path,
    env_name: str,
    run_id: str,
) -> Path | None:
    run_dir = checkpoint_root / env_name / run_id
    checkpoints = [
        path
        for path in run_dir.glob("*.bin")
        if path.is_file() and path.stem.isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem))


def upload_native_log(
    log_path: Path,
    project: str,
    state: dict[str, Any],
    state_path: Path,
    checkpoint_root: Path,
    env_name: str,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    payload = load_native_log(log_path)
    key = log_path.name
    entry = state["runs"].setdefault(key, {})
    wandb = wandb_module
    if wandb is None:
        import wandb as wandb_module_import

        wandb = wandb_module_import

    init_args: dict[str, Any] = {
        "project": project,
        "config": payload["config"],
        "reinit": True,
        "settings": wandb.Settings(console="off"),
    }
    if entry.get("wandb_id"):
        init_args["id"] = entry["wandb_id"]
        init_args["resume"] = "allow"

    run = None
    try:
        # On first creation, intentionally omit both id and name. W&B therefore
        # generates its normal random ID and human-readable display name.
        run = wandb.init(**init_args)
        if run is None:
            raise SidecarError("wandb.init returned no run")
        entry.update(
            {
                "native_run_id": payload["run_id"],
                "wandb_id": getattr(run, "id", entry.get("wandb_id")),
                "wandb_name": getattr(run, "name", entry.get("wandb_name")),
                "status": "uploading",
                "started_utc": utc_now(),
            }
        )
        save_state(state_path, state)

        for row in payload["history"]:
            run.log(
                {"agent_steps": row["step"], **row["metrics"]},
                step=row["step"],
            )
        run.summary.update(payload["summary"])
        checkpoint = final_checkpoint(
            checkpoint_root, env_name, payload["run_id"]
        )
        if checkpoint is not None:
            run.summary["native/checkpoint_path"] = str(checkpoint.resolve())
        run.finish()
        run = None
    except Exception as error:
        entry.update(
            {
                "status": "failed",
                "error": repr(error),
                "failed_utc": utc_now(),
            }
        )
        save_state(state_path, state)
        raise
    finally:
        if run is not None:
            run.finish()

    entry.update(
        {
            "status": "uploaded",
            "valid_pool_eval": payload["valid_pool_eval"],
            "uploaded_utc": utc_now(),
            "error": None,
        }
    )
    save_state(state_path, state)
    return payload


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload completed native Protein logs to W&B without controlling "
            "training or sweep suggestions."
        )
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Accepted for parity with the traditional Puffer command.",
    )
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--env", default="dogfight")
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("checkpoints")
    )
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Exit after uploading this many new native logs; 0 follows forever.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Upload logs that already exist when a new state file is created.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--allow-missing-pool-score",
        action="store_true",
        help="Return success even when a trial lacks historical-pool evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_runs < 0:
        raise SidecarError("--max-runs must be nonnegative")
    if args.poll_seconds <= 0 or args.settle_seconds < 0:
        raise SidecarError("poll and settle intervals must be valid")

    repo_root = Path(__file__).resolve().parents[2]
    log_root = resolve_path(repo_root, args.log_root)
    log_dir = log_root / args.env
    checkpoint_root = resolve_path(repo_root, args.checkpoint_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    project_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.wandb_project)
    state_path = args.state_path or (
        log_dir / f".wandb-sidecar-{project_key}.json"
    )
    state_path = resolve_path(repo_root, state_path)
    existing_logs = sorted(log_dir.glob("*.ini"))
    state = load_or_create_state(
        state_path, existing_logs, include_existing=args.include_existing
    )

    print(
        f"wandb sidecar project={args.wandb_project} env={args.env} "
        f"baseline={len(state.get('baseline', []))} state={state_path}",
        flush=True,
    )
    uploaded = 0
    invalid = 0
    while True:
        for log_path in discover_logs(
            log_dir, state, args.settle_seconds
        ):
            try:
                payload = upload_native_log(
                    log_path=log_path,
                    project=args.wandb_project,
                    state=state,
                    state_path=state_path,
                    checkpoint_root=checkpoint_root,
                    env_name=args.env,
                )
            except KeyboardInterrupt:
                raise
            except Exception as error:
                print(
                    f"wandb sidecar retry pending {log_path.name}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            uploaded += 1
            if not payload["valid_pool_eval"]:
                invalid += 1
            entry = state["runs"][log_path.name]
            print(
                f"wandb uploaded native={payload['run_id']} "
                f"wandb={entry.get('wandb_name') or entry.get('wandb_id')} "
                f"pool_score={payload['summary']['selfplay/pool_winrate']} "
                f"flight_quality={payload['summary']['selfplay/pool_flight_quality']} "
                f"fitness={payload['summary']['protein/fitness']}",
                flush=True,
            )
            if args.max_runs and uploaded >= args.max_runs:
                return (
                    0
                    if invalid == 0 or args.allow_missing_pool_score
                    else 2
                )

        if args.once:
            return (
                0
                if invalid == 0 or args.allow_missing_pool_score
                else 2
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SidecarError as error:
        print(f"wandb sidecar error: {error}", file=sys.stderr)
        raise SystemExit(2)
