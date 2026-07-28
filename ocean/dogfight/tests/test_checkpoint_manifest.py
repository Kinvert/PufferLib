import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


DOGFIGHT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = DOGFIGHT_DIR / "checkpoint_manifest.py"
SPEC = importlib.util.spec_from_file_location("dogfight_checkpoint_manifest", MODULE_PATH)
checkpoint_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_manifest)


def _write_checkpoint(path: Path, size: int, fill: bytes = b"\0") -> None:
    path.write_bytes(fill * size)


def _publish(checkpoint: Path, **overrides) -> Path:
    kwargs = {
        "dogfight_behavior_version": "a" * 40,
        "global_agent_steps": 57_671_680,
        "curriculum_stage": 11,
        "hidden_size": 64,
        "num_layers": 3,
        "lr_schedule_decision": "fresh fixed learning rate for this warm-start segment",
    }
    kwargs.update(overrides)
    return checkpoint_manifest.publish_manifest(checkpoint, **kwargs)


def test_default_native_topology_has_exact_known_size():
    assert checkpoint_manifest.expected_weight_count(64, 3) == 38_917
    assert checkpoint_manifest.expected_byte_count(64, 3) == 155_668


def test_publish_writes_validated_manifest_after_payload(tmp_path):
    checkpoint = tmp_path / "0000000057671680.bin"
    _write_checkpoint(checkpoint, checkpoint_manifest.expected_byte_count(64, 3))

    sidecar = _publish(checkpoint)
    manifest = json.loads(sidecar.read_text())

    assert sidecar == checkpoint.with_suffix(".bin.manifest.json")
    assert manifest["format"] == "pufferlib-native-fp32-weights-v1"
    assert manifest["pufferlib_commit"] == checkpoint_manifest.PINNED_5C_COMMIT
    assert manifest["dogfight_behavior_version"] == "a" * 40
    assert manifest["obs_width"] == 26
    assert manifest["obs_schema"] == "opponent-aware-v1"
    assert manifest["action_schema"] == [
        "throttle",
        "elevator",
        "aileron",
        "rudder",
        "trigger",
    ]
    assert manifest["encoder"] == "native-linear-no-bias"
    assert manifest["recurrent"] == "mingru"
    assert manifest["decoder"] == "native-fused-policy-value"
    assert manifest["hidden_size"] == 64
    assert manifest["num_layers"] == 3
    assert manifest["weight_count"] == 38_917
    assert manifest["byte_count"] == 155_668
    assert manifest["global_agent_steps"] == 57_671_680
    assert manifest["curriculum_stage"] == 11
    assert manifest["warm_start"] is True
    assert manifest["optimizer_state_resumed"] is False
    assert manifest["scheduler_state_resumed"] is False
    assert manifest["lr_schedule_decision"] == (
        "fresh fixed learning rate for this warm-start segment"
    )
    assert manifest["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_manifest.validate_manifest(checkpoint) == manifest


def test_publish_rejects_wrong_size_without_sidecar(tmp_path):
    checkpoint = tmp_path / "trailing.bin"
    _write_checkpoint(
        checkpoint,
        checkpoint_manifest.expected_byte_count(64, 3) + 1,
    )

    with pytest.raises(checkpoint_manifest.ManifestError, match="size mismatch"):
        _publish(checkpoint)

    assert not checkpoint.with_suffix(".bin.manifest.json").exists()


def test_validate_detects_payload_change(tmp_path):
    checkpoint = tmp_path / "changed.bin"
    _write_checkpoint(checkpoint, checkpoint_manifest.expected_byte_count(64, 3))
    _publish(checkpoint)

    with checkpoint.open("r+b") as payload:
        payload.write(b"\1")

    with pytest.raises(checkpoint_manifest.ManifestError, match="sha256 mismatch"):
        checkpoint_manifest.validate_manifest(checkpoint)


def test_publish_records_parent_payload_hash(tmp_path):
    byte_count = checkpoint_manifest.expected_byte_count(64, 3)
    parent = tmp_path / "parent.bin"
    checkpoint = tmp_path / "child.bin"
    _write_checkpoint(parent, byte_count, b"\1")
    _write_checkpoint(checkpoint, byte_count, b"\2")

    sidecar = _publish(checkpoint, parent=parent)
    manifest = json.loads(sidecar.read_text())

    assert manifest["parent"] == {
        "path": str(parent),
        "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    }


def test_publish_requires_explicit_lr_schedule_decision(tmp_path):
    checkpoint = tmp_path / "missing-schedule.bin"
    _write_checkpoint(checkpoint, checkpoint_manifest.expected_byte_count(64, 3))

    with pytest.raises(checkpoint_manifest.ManifestError, match="learning-rate"):
        _publish(checkpoint, lr_schedule_decision="")
