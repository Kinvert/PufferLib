#!/usr/bin/env python3
"""Publish and validate Dogfight native checkpoint manifests.

The native checkpoint payload is published by PufferLib first. This tool then
validates the complete payload and atomically publishes `<checkpoint>.manifest.json`
as the final artifact in the checkpoint transaction.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


PINNED_5C_COMMIT = "ebf5ed03cc3524076b6c1a4033bd69cec0b3db22"
FORMAT = "pufferlib-native-fp32-weights-v1"
OBS_WIDTH = 26
OBS_SCHEMA = "opponent-aware-v1"
ACTION_SCHEMA = ["throttle", "elevator", "aileron", "rudder", "trigger"]
ENCODER = "native-linear-no-bias"
RECURRENT = "mingru"
DECODER = "native-fused-policy-value"
CURRICULUM_STAGE_MAX = 20
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    """Checkpoint or manifest validation failed."""


def expected_weight_count(
    hidden_size: int,
    num_layers: int,
    obs_width: int = OBS_WIDTH,
    action_width: int = len(ACTION_SCHEMA),
) -> int:
    if hidden_size <= 0 or num_layers <= 0:
        raise ManifestError("hidden_size and num_layers must be positive")
    if obs_width <= 0 or action_width <= 0:
        raise ManifestError("observation and action widths must be positive")

    encoder_weights = obs_width * hidden_size
    recurrent_weights = num_layers * 3 * hidden_size * hidden_size
    decoder_weights = (action_width + 1) * hidden_size
    continuous_logstd = action_width
    return (
        encoder_weights
        + recurrent_weights
        + decoder_weights
        + continuous_logstd
    )


def expected_byte_count(
    hidden_size: int,
    num_layers: int,
    obs_width: int = OBS_WIDTH,
    action_width: int = len(ACTION_SCHEMA),
) -> int:
    return 4 * expected_weight_count(
        hidden_size,
        num_layers,
        obs_width,
        action_width,
    )


def manifest_path(checkpoint: Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")


def _regular_file(path: Path, label: str):
    try:
        file_stat = path.stat()
    except FileNotFoundError as exc:
        raise ManifestError(f"{label} does not exist: {path}") from exc
    if not path.is_file():
        raise ManifestError(f"{label} is not a regular file: {path}")
    return file_stat


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload:
        for block in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_payload(before, after) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )


def _validate_commit(value: str, label: str) -> None:
    if not COMMIT_PATTERN.fullmatch(value):
        raise ManifestError(f"{label} must be a full lowercase 40-character commit")


def _validate_manifest_payload(checkpoint: Path, manifest: dict) -> None:
    required = {
        "format",
        "pufferlib_commit",
        "dogfight_behavior_version",
        "obs_width",
        "obs_schema",
        "action_schema",
        "encoder",
        "recurrent",
        "decoder",
        "hidden_size",
        "num_layers",
        "weight_count",
        "byte_count",
        "global_agent_steps",
        "curriculum_stage",
        "parent",
        "warm_start",
        "optimizer_state_resumed",
        "scheduler_state_resumed",
        "lr_schedule_decision",
        "sha256",
        "published_at_utc",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ManifestError(f"manifest is missing required fields: {', '.join(missing)}")

    if manifest["format"] != FORMAT:
        raise ManifestError(f"unsupported checkpoint format: {manifest['format']}")
    _validate_commit(manifest["pufferlib_commit"], "pufferlib_commit")
    _validate_commit(
        manifest["dogfight_behavior_version"],
        "dogfight_behavior_version",
    )
    if manifest["obs_width"] != OBS_WIDTH or manifest["obs_schema"] != OBS_SCHEMA:
        raise ManifestError("observation schema mismatch")
    if manifest["action_schema"] != ACTION_SCHEMA:
        raise ManifestError("action schema mismatch")
    if (
        manifest["encoder"] != ENCODER
        or manifest["recurrent"] != RECURRENT
        or manifest["decoder"] != DECODER
    ):
        raise ManifestError("policy topology schema mismatch")

    hidden_size = manifest["hidden_size"]
    num_layers = manifest["num_layers"]
    weight_count = expected_weight_count(hidden_size, num_layers)
    byte_count = 4 * weight_count
    if manifest["weight_count"] != weight_count:
        raise ManifestError(
            "weight count mismatch: "
            f"expected={weight_count} actual={manifest['weight_count']}"
        )
    if manifest["byte_count"] != byte_count:
        raise ManifestError(
            "manifest byte count mismatch: "
            f"expected={byte_count} actual={manifest['byte_count']}"
        )

    payload_stat = _regular_file(checkpoint, "checkpoint")
    if payload_stat.st_size != byte_count:
        raise ManifestError(
            "checkpoint size mismatch: "
            f"expected={byte_count} actual={payload_stat.st_size}"
        )
    payload_sha256 = _sha256(checkpoint)
    if not SHA256_PATTERN.fullmatch(manifest["sha256"]):
        raise ManifestError("manifest sha256 is not a lowercase SHA-256 digest")
    if payload_sha256 != manifest["sha256"]:
        raise ManifestError(
            "checkpoint sha256 mismatch: "
            f"expected={manifest['sha256']} actual={payload_sha256}"
        )

    if manifest["global_agent_steps"] < 0:
        raise ManifestError("global_agent_steps must be nonnegative")
    stage = manifest["curriculum_stage"]
    if stage is not None and not 0 <= stage <= CURRICULUM_STAGE_MAX:
        raise ManifestError(
            f"curriculum_stage must be null or 0..{CURRICULUM_STAGE_MAX}"
        )
    if not str(manifest["lr_schedule_decision"]).strip():
        raise ManifestError("an explicit learning-rate schedule decision is required")
    if manifest["optimizer_state_resumed"] or manifest["scheduler_state_resumed"]:
        raise ManifestError("native checkpoints are weights-only warm starts")

    parent = manifest["parent"]
    if parent is not None:
        if set(parent) != {"path", "sha256"}:
            raise ManifestError("parent must contain exactly path and sha256")
        parent_path = Path(parent["path"])
        parent_stat = _regular_file(parent_path, "parent checkpoint")
        if parent_stat.st_size != byte_count:
            raise ManifestError(
                "parent checkpoint size mismatch: "
                f"expected={byte_count} actual={parent_stat.st_size}"
            )
        parent_sha256 = _sha256(parent_path)
        if parent_sha256 != parent["sha256"]:
            raise ManifestError(
                "parent checkpoint sha256 mismatch: "
                f"expected={parent['sha256']} actual={parent_sha256}"
            )


def publish_manifest(
    checkpoint: Path,
    *,
    dogfight_behavior_version: str,
    global_agent_steps: int,
    curriculum_stage: int | None,
    hidden_size: int,
    num_layers: int,
    lr_schedule_decision: str,
    parent: Path | None = None,
    pufferlib_commit: str = PINNED_5C_COMMIT,
    warm_start: bool = True,
) -> Path:
    checkpoint = Path(checkpoint)
    sidecar = manifest_path(checkpoint)
    if sidecar.exists():
        raise ManifestError(f"manifest already exists: {sidecar}")
    _validate_commit(pufferlib_commit, "pufferlib_commit")
    _validate_commit(dogfight_behavior_version, "dogfight_behavior_version")
    if not str(lr_schedule_decision).strip():
        raise ManifestError("an explicit learning-rate schedule decision is required")

    weight_count = expected_weight_count(hidden_size, num_layers)
    byte_count = 4 * weight_count
    before = _regular_file(checkpoint, "checkpoint")
    if before.st_size != byte_count:
        raise ManifestError(
            "checkpoint size mismatch: "
            f"expected={byte_count} actual={before.st_size}"
        )
    payload_sha256 = _sha256(checkpoint)
    after_hash = _regular_file(checkpoint, "checkpoint")
    if not _same_payload(before, after_hash):
        raise ManifestError("checkpoint changed while its hash was being computed")

    parent_record = None
    if parent is not None:
        parent = Path(parent)
        parent_stat = _regular_file(parent, "parent checkpoint")
        if parent_stat.st_size != byte_count:
            raise ManifestError(
                "parent checkpoint size mismatch: "
                f"expected={byte_count} actual={parent_stat.st_size}"
            )
        parent_record = {"path": str(parent), "sha256": _sha256(parent)}

    manifest = {
        "format": FORMAT,
        "pufferlib_commit": pufferlib_commit,
        "dogfight_behavior_version": dogfight_behavior_version,
        "obs_width": OBS_WIDTH,
        "obs_schema": OBS_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "encoder": ENCODER,
        "recurrent": RECURRENT,
        "decoder": DECODER,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "weight_count": weight_count,
        "byte_count": byte_count,
        "global_agent_steps": global_agent_steps,
        "curriculum_stage": curriculum_stage,
        "parent": parent_record,
        "warm_start": bool(warm_start),
        "optimizer_state_resumed": False,
        "scheduler_state_resumed": False,
        "lr_schedule_decision": str(lr_schedule_decision).strip(),
        "sha256": payload_sha256,
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _validate_manifest_payload(checkpoint, manifest)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.tmp.",
        dir=sidecar.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

        before_publish = _regular_file(checkpoint, "checkpoint")
        if not _same_payload(before, before_publish):
            raise ManifestError("checkpoint changed before manifest publication")
        if sidecar.exists():
            raise ManifestError(f"manifest already exists: {sidecar}")
        os.replace(temporary_path, sidecar)

        directory_descriptor = os.open(sidecar.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return sidecar


def validate_manifest(
    checkpoint: Path,
    sidecar: Path | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    sidecar = Path(sidecar) if sidecar is not None else manifest_path(checkpoint)
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest does not exist: {sidecar}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {sidecar}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be a JSON object")
    _validate_manifest_payload(checkpoint, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    publish = commands.add_parser("publish", help="publish a validated sidecar")
    publish.add_argument("checkpoint", type=Path)
    publish.add_argument("--dogfight-behavior-version", required=True)
    publish.add_argument("--global-agent-steps", type=int, required=True)
    publish.add_argument("--curriculum-stage", type=int)
    publish.add_argument("--hidden-size", type=int, required=True)
    publish.add_argument("--num-layers", type=int, required=True)
    publish.add_argument("--lr-schedule-decision", required=True)
    publish.add_argument("--parent", type=Path)
    publish.add_argument("--pufferlib-commit", default=PINNED_5C_COMMIT)
    publish.add_argument(
        "--from-scratch",
        action="store_true",
        help="record that this lineage did not warm-start from supplied weights",
    )

    validate = commands.add_parser("validate", help="validate payload and sidecar")
    validate.add_argument("checkpoint", type=Path)
    validate.add_argument("--manifest", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "publish":
            sidecar = publish_manifest(
                args.checkpoint,
                dogfight_behavior_version=args.dogfight_behavior_version,
                global_agent_steps=args.global_agent_steps,
                curriculum_stage=args.curriculum_stage,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                lr_schedule_decision=args.lr_schedule_decision,
                parent=args.parent,
                pufferlib_commit=args.pufferlib_commit,
                warm_start=not args.from_scratch,
            )
            print(sidecar)
        else:
            manifest = validate_manifest(args.checkpoint, args.manifest)
            print(
                f"valid checkpoint: bytes={manifest['byte_count']} "
                f"sha256={manifest['sha256']}"
            )
    except ManifestError as exc:
        print(f"checkpoint manifest error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
