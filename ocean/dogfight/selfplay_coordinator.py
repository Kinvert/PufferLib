"""Dogfight-native self-play pool and promotion coordination.

This module deliberately does not implement PPO or policy inference. It owns the
durable metadata around native checkpoints and match evidence so native
``puffer`` train/match operations can remain the source of policy behavior.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence


MANIFEST_VERSION = 1
DEFAULT_PROMOTION_POLICY = {
    "min_decisive_gun_kills": 10,
    "min_learner_kill_share": 0.55,
    "min_clean_fight_rate": 0.80,
    "min_successful_rotations": 2,
}
_MATCH_EVIDENCE_PATTERN = re.compile(
    r"match/evidence games=(?P<games>\d+) "
    r"slot0_wins=(?P<slot0_wins>\d+) "
    r"slot1_wins=(?P<slot1_wins>\d+) "
    r"draws=(?P<draws>\d+) "
    r"slot0_gun_kills=(?P<slot0_gun_kills>\d+) "
    r"slot1_gun_kills=(?P<slot1_gun_kills>\d+) "
    r"clean_fights=(?P<clean_fights>\d+)"
)


class ManifestError(ValueError):
    """Raised when persisted coordinator state violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_entry(
    checkpoint_id: str,
    path: str | os.PathLike[str],
    *,
    generation: int,
    parent_id: str | None = None,
    architecture: Mapping[str, Any] | None = None,
    behavior_version: int = 1,
    reward_version: int = 1,
    fixed: bool = False,
) -> dict[str, Any]:
    """Create immutable checkpoint metadata from checkpoint bytes on disk."""
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise ManifestError(f"checkpoint is not a file: {checkpoint_path}")
    if not checkpoint_id:
        raise ManifestError("checkpoint_id must not be empty")
    if type(generation) is not int or generation < 0:
        raise ManifestError("generation must be a non-negative integer")

    return {
        "id": checkpoint_id,
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "byte_size": checkpoint_path.stat().st_size,
        "generation": generation,
        "parent_id": parent_id,
        "architecture": dict(architecture or {}),
        "behavior_version": behavior_version,
        "reward_version": reward_version,
        "fixed": bool(fixed),
    }


def new_manifest(
    *,
    seed: int,
    pool_capacity: int,
    promotion_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an empty, deterministic coordinator manifest."""
    if type(seed) is not int:
        raise ManifestError("seed must be an integer")
    if type(pool_capacity) is not int or pool_capacity < 1:
        raise ManifestError("pool_capacity must be a positive integer")

    policy = dict(DEFAULT_PROMOTION_POLICY)
    if promotion_policy:
        policy.update(promotion_policy)
    manifest = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "selection_counter": 0,
        "pool_capacity": pool_capacity,
        "current_opponent_id": None,
        "checkpoints": {},
        "anchors": [],
        "pool": [],
        "candidates": {},
        "matches": [],
        "ratings": {},
        "promotions": [],
        "promotion_policy": policy,
    }
    validate_manifest(manifest, verify_files=False)
    return manifest


def _require_exact_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ManifestError(f"{name} must be an integer >= {minimum}")
    return value


def _require_rate(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ManifestError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ManifestError(f"{name} must be between 0 and 1")
    return result


def validate_manifest(
    manifest: Mapping[str, Any], *, verify_files: bool = True
) -> None:
    """Validate schema, references, immutable bytes, and promotion policy."""
    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest must be an object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ManifestError(
            f"unsupported manifest version: {manifest.get('version')!r}"
        )
    _require_exact_int(manifest.get("seed"), "seed")
    _require_exact_int(manifest.get("selection_counter"), "selection_counter")
    capacity = _require_exact_int(manifest.get("pool_capacity"), "pool_capacity", 1)

    checkpoints = manifest.get("checkpoints")
    anchors = manifest.get("anchors")
    pool = manifest.get("pool")
    candidates = manifest.get("candidates")
    matches = manifest.get("matches")
    ratings = manifest.get("ratings")
    promotions = manifest.get("promotions")
    policy = manifest.get("promotion_policy")
    if not isinstance(checkpoints, Mapping):
        raise ManifestError("checkpoints must be an object")
    if not isinstance(anchors, list) or not isinstance(pool, list):
        raise ManifestError("anchors and pool must be lists")
    if not isinstance(candidates, Mapping):
        raise ManifestError("candidates must be an object")
    if not isinstance(matches, list) or not isinstance(promotions, list):
        raise ManifestError("matches and promotions must be lists")
    if not isinstance(ratings, Mapping):
        raise ManifestError("ratings must be an object")
    if not isinstance(policy, Mapping):
        raise ManifestError("promotion_policy must be an object")
    if len(pool) > capacity:
        raise ManifestError("pool exceeds pool_capacity")
    if len(set(anchors)) != len(anchors) or len(set(pool)) != len(pool):
        raise ManifestError("anchor and pool ids must be unique")
    if set(anchors) & set(pool):
        raise ManifestError("anchors and pool must be disjoint")

    paths: dict[str, str] = {}
    for checkpoint_id, entry in checkpoints.items():
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ManifestError("checkpoint ids must be non-empty strings")
        if not isinstance(entry, Mapping) or entry.get("id") != checkpoint_id:
            raise ManifestError(f"checkpoint key/id mismatch: {checkpoint_id}")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        if not isinstance(path, str) or not path:
            raise ManifestError(f"checkpoint {checkpoint_id} has no path")
        if path in paths:
            raise ManifestError(
                f"checkpoint path reused by {checkpoint_id} and {paths[path]}"
            )
        paths[path] = checkpoint_id
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)
        ):
            raise ManifestError(f"checkpoint {checkpoint_id} has invalid sha256")
        size = _require_exact_int(
            entry.get("byte_size"), f"{checkpoint_id}.byte_size"
        )
        _require_exact_int(entry.get("generation"), f"{checkpoint_id}.generation")
        if not isinstance(entry.get("architecture"), Mapping):
            raise ManifestError(f"{checkpoint_id}.architecture must be an object")
        _require_exact_int(
            entry.get("behavior_version"), f"{checkpoint_id}.behavior_version"
        )
        _require_exact_int(
            entry.get("reward_version"), f"{checkpoint_id}.reward_version"
        )
        if type(entry.get("fixed")) is not bool:
            raise ManifestError(f"{checkpoint_id}.fixed must be boolean")
        checkpoint_path = Path(path)
        if verify_files:
            if not checkpoint_path.is_file():
                raise ManifestError(f"checkpoint missing: {checkpoint_path}")
            if checkpoint_path.stat().st_size != size:
                raise ManifestError(f"checkpoint size changed: {checkpoint_id}")
            if _sha256(checkpoint_path) != sha256:
                raise ManifestError(f"checkpoint hash changed: {checkpoint_id}")

    for anchor_id in anchors:
        if anchor_id not in checkpoints:
            raise ManifestError(f"unknown anchor checkpoint: {anchor_id}")
        if checkpoints[anchor_id].get("fixed") is not True:
            raise ManifestError(f"anchor is not fixed: {anchor_id}")
    for checkpoint_id in pool:
        if checkpoint_id not in checkpoints:
            raise ManifestError(f"unknown pool checkpoint: {checkpoint_id}")
        if checkpoints[checkpoint_id].get("fixed") is True:
            raise ManifestError(f"fixed checkpoint must be an anchor: {checkpoint_id}")

    current = manifest.get("current_opponent_id")
    selectable = set(anchors) | set(pool)
    if current is not None and current not in selectable:
        raise ManifestError(f"current opponent is not selectable: {current}")

    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ManifestError("candidate ids must be non-empty strings")
        if not isinstance(candidate, Mapping):
            raise ManifestError(f"candidate {candidate_id} must be an object")
        checkpoint_id = candidate.get("checkpoint_id")
        parent_id = candidate.get("parent_checkpoint_id")
        if checkpoint_id not in checkpoints:
            raise ManifestError(f"candidate checkpoint missing: {checkpoint_id}")
        if parent_id not in checkpoints:
            raise ManifestError(f"candidate parent missing: {parent_id}")
        if candidate.get("status") not in {"pending", "promoted", "rejected"}:
            raise ManifestError(f"candidate status invalid: {candidate_id}")

    match_ids: set[str] = set()
    for evidence in matches:
        _validate_match_evidence(evidence, manifest)
        evidence_id = evidence["id"]
        if evidence_id in match_ids:
            raise ManifestError(f"duplicate match evidence id: {evidence_id}")
        match_ids.add(evidence_id)

    for checkpoint_id, rating in ratings.items():
        if checkpoint_id not in checkpoints:
            raise ManifestError(f"rating references unknown checkpoint: {checkpoint_id}")
        if not isinstance(rating, (int, float)) or not math.isfinite(float(rating)):
            raise ManifestError(f"invalid rating for checkpoint: {checkpoint_id}")

    _require_exact_int(
        policy.get("min_decisive_gun_kills"), "min_decisive_gun_kills", 1
    )
    _require_rate(policy.get("min_learner_kill_share"), "min_learner_kill_share")
    _require_rate(policy.get("min_clean_fight_rate"), "min_clean_fight_rate")
    _require_exact_int(
        policy.get("min_successful_rotations"), "min_successful_rotations", 1
    )


def write_manifest_atomic(
    path: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> None:
    """Durably replace a manifest without exposing partial JSON."""
    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_manifest(
    path: str | os.PathLike[str], *, verify_files: bool = True
) -> dict[str, Any]:
    """Load and fully validate a persisted manifest."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load manifest {source}: {exc}") from exc
    validate_manifest(manifest, verify_files=verify_files)
    return manifest


def update_manifest(
    path: str | os.PathLike[str],
    mutation: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Lock, load, mutate, validate, and atomically replace a manifest."""
    destination = Path(path)
    lock_path = destination.with_name(f"{destination.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        manifest = load_manifest(destination)
        updated = mutation(manifest)
        write_manifest_atomic(destination, updated)
        return updated


def add_checkpoint(
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    category: str = "pool",
) -> dict[str, Any]:
    """Add an immutable checkpoint to the FIFO pool, anchors, or candidates."""
    if category not in {"pool", "anchor", "candidate"}:
        raise ManifestError(f"unsupported checkpoint category: {category}")
    updated = copy.deepcopy(dict(manifest))
    candidate_entry = copy.deepcopy(dict(entry))
    checkpoint_id = candidate_entry.get("id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ManifestError("checkpoint entry has no id")
    expected_fixed = category == "anchor"
    if candidate_entry.get("fixed") is not expected_fixed:
        raise ManifestError(
            f"{category} checkpoint fixed must be {expected_fixed}: {checkpoint_id}"
        )

    existing = updated["checkpoints"].get(checkpoint_id)
    if existing is not None and existing != candidate_entry:
        raise ManifestError(f"checkpoint metadata is immutable: {checkpoint_id}")
    for other_id, other in updated["checkpoints"].items():
        if other_id != checkpoint_id and other["path"] == candidate_entry.get("path"):
            raise ManifestError(f"checkpoint path already registered as {other_id}")
    updated["checkpoints"][checkpoint_id] = candidate_entry
    updated["ratings"].setdefault(checkpoint_id, 1000.0)

    if category == "anchor":
        if checkpoint_id in updated["pool"]:
            raise ManifestError("pool checkpoint cannot become a fixed anchor")
        if checkpoint_id not in updated["anchors"]:
            updated["anchors"].append(checkpoint_id)
    elif category == "pool":
        if checkpoint_id in updated["anchors"]:
            raise ManifestError("fixed anchor cannot enter FIFO pool")
        if checkpoint_id not in updated["pool"]:
            updated["pool"].append(checkpoint_id)
        while len(updated["pool"]) > updated["pool_capacity"]:
            evicted = updated["pool"].pop(0)
            if updated["current_opponent_id"] == evicted:
                updated["current_opponent_id"] = None

    validate_manifest(updated)
    return updated


def register_candidate(
    manifest: Mapping[str, Any],
    *,
    candidate_id: str,
    checkpoint_entry_data: Mapping[str, Any],
    parent_checkpoint_id: str,
) -> dict[str, Any]:
    """Register a pending league candidate backed by immutable checkpoint bytes."""
    if not candidate_id:
        raise ManifestError("candidate_id must not be empty")
    updated = add_checkpoint(
        manifest, checkpoint_entry_data, category="candidate"
    )
    checkpoint_id = checkpoint_entry_data["id"]
    if parent_checkpoint_id not in updated["checkpoints"]:
        raise ManifestError(f"candidate parent missing: {parent_checkpoint_id}")
    candidate = {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": parent_checkpoint_id,
        "status": "pending",
    }
    existing = updated["candidates"].get(candidate_id)
    if existing is not None and existing != candidate:
        raise ManifestError(f"candidate metadata is immutable: {candidate_id}")
    updated["candidates"][candidate_id] = candidate
    validate_manifest(updated)
    return updated


def _selection_digest(seed: int, counter: int, strategy: str) -> int:
    value = f"{seed}:{counter}:{strategy}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _learner_kill_share(manifest: Mapping[str, Any], opponent_id: str) -> float:
    learner_kills = 0
    opponent_kills = 0
    for evidence in manifest["matches"]:
        if evidence["opponent_checkpoint_id"] != opponent_id:
            continue
        learner_kills += evidence["learner_gun_kills"]
        opponent_kills += evidence["opponent_gun_kills"]
    decisive = learner_kills + opponent_kills
    return learner_kills / decisive if decisive else 0.5


def eligible_opponents(
    manifest: Mapping[str, Any],
    *,
    strategy: str = "uniform",
    top_n: int = 4,
) -> list[tuple[str, float]]:
    """Return current-excluding candidates and their deterministic weights."""
    validate_manifest(manifest)
    candidates = list(manifest["anchors"]) + list(manifest["pool"])
    if not candidates:
        raise ManifestError("opponent pool is empty")
    current = manifest["current_opponent_id"]
    if len(candidates) > 1 and current in candidates:
        candidates.remove(current)

    if strategy == "uniform":
        return [(checkpoint_id, 1.0) for checkpoint_id in candidates]
    if strategy == "top_n":
        if type(top_n) is not int or top_n < 1:
            raise ManifestError("top_n must be a positive integer")
        ranked = sorted(
            candidates,
            key=lambda checkpoint_id: (
                -float(manifest["ratings"].get(checkpoint_id, 1000.0)),
                checkpoint_id,
            ),
        )
        return [(checkpoint_id, 1.0) for checkpoint_id in ranked[:top_n]]
    if strategy == "pfsp":
        weighted = []
        for checkpoint_id in candidates:
            learner_share = _learner_kill_share(manifest, checkpoint_id)
            weighted.append((checkpoint_id, max((1.0 - learner_share) ** 2, 1e-9)))
        return weighted
    raise ManifestError(f"unsupported selection strategy: {strategy}")


def select_opponent(
    manifest: Mapping[str, Any],
    *,
    strategy: str = "uniform",
    top_n: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Select an opponent deterministically and advance persisted RNG state."""
    weighted = eligible_opponents(manifest, strategy=strategy, top_n=top_n)
    counter = manifest["selection_counter"]
    digest = _selection_digest(manifest["seed"], counter, strategy)
    total_weight = sum(weight for _, weight in weighted)
    threshold = (digest / 2**64) * total_weight
    selected = weighted[-1][0]
    cumulative = 0.0
    for checkpoint_id, weight in weighted:
        cumulative += weight
        if threshold < cumulative:
            selected = checkpoint_id
            break

    updated = copy.deepcopy(dict(manifest))
    updated["selection_counter"] = counter + 1
    updated["current_opponent_id"] = selected
    validate_manifest(updated)
    return selected, updated


def select_and_persist(
    path: str | os.PathLike[str],
    *,
    strategy: str = "uniform",
    top_n: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Atomically select an opponent and persist the advanced selection state."""
    selected: list[str] = []

    def mutation(manifest: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id, updated = select_opponent(
            manifest, strategy=strategy, top_n=top_n
        )
        selected.append(checkpoint_id)
        return updated

    updated = update_manifest(path, mutation)
    return selected[0], updated


def _validate_match_evidence(
    evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if not isinstance(evidence, Mapping):
        raise ManifestError("match evidence must be an object")
    for key in ("id", "candidate_id", "opponent_checkpoint_id", "rotation_id"):
        if not isinstance(evidence.get(key), str) or not evidence[key]:
            raise ManifestError(f"match evidence {key} must be a non-empty string")
    candidate_id = evidence["candidate_id"]
    if candidate_id not in manifest["candidates"]:
        raise ManifestError(f"match references unknown candidate: {candidate_id}")
    if evidence["opponent_checkpoint_id"] not in manifest["checkpoints"]:
        raise ManifestError("match references unknown opponent checkpoint")
    if evidence.get("seat") not in (0, 1):
        raise ManifestError("match seat must be 0 or 1")

    games = _require_exact_int(evidence.get("games"), "games", 1)
    learner_wins = _require_exact_int(evidence.get("learner_wins"), "learner_wins")
    opponent_wins = _require_exact_int(
        evidence.get("opponent_wins"), "opponent_wins"
    )
    draws = _require_exact_int(evidence.get("draws"), "draws")
    if learner_wins + opponent_wins + draws != games:
        raise ManifestError("match outcomes must sum to games")
    learner_kills = _require_exact_int(
        evidence.get("learner_gun_kills"), "learner_gun_kills"
    )
    opponent_kills = _require_exact_int(
        evidence.get("opponent_gun_kills"), "opponent_gun_kills"
    )
    if learner_kills > learner_wins or opponent_kills > opponent_wins:
        raise ManifestError("decisive gun kills cannot exceed wins")
    clean_fights = _require_exact_int(
        evidence.get("clean_fights"), "clean_fights"
    )
    if clean_fights > games:
        raise ManifestError("clean_fights cannot exceed games")
    command = evidence.get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) for token in command
    ):
        raise ManifestError("match command must be a list of strings")


def _elo_update(
    learner_rating: float,
    opponent_rating: float,
    learner_score: float,
    *,
    k_factor: float = 32.0,
) -> tuple[float, float]:
    expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - learner_rating) / 400.0))
    delta = k_factor * (learner_score - expected)
    return learner_rating + delta, opponent_rating - delta


def record_match(
    manifest: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist native both-seat evidence and update candidate/opponent Elo."""
    updated = copy.deepcopy(dict(manifest))
    candidate_id = evidence.get("candidate_id")
    if candidate_id in updated["candidates"]:
        status = updated["candidates"][candidate_id]["status"]
        if status != "pending":
            raise ManifestError(f"candidate is not pending: {candidate_id}")
    _validate_match_evidence(evidence, updated)
    for existing in updated["matches"]:
        if existing["id"] == evidence["id"]:
            if existing == evidence:
                return updated
            raise ManifestError(f"match evidence is immutable: {evidence['id']}")

    stored_evidence = copy.deepcopy(dict(evidence))
    updated["matches"].append(stored_evidence)
    candidate_checkpoint = updated["candidates"][candidate_id]["checkpoint_id"]
    opponent_checkpoint = evidence["opponent_checkpoint_id"]
    learner_rating = float(updated["ratings"].get(candidate_checkpoint, 1000.0))
    opponent_rating = float(updated["ratings"].get(opponent_checkpoint, 1000.0))
    learner_score = (
        evidence["learner_wins"] + 0.5 * evidence["draws"]
    ) / evidence["games"]
    learner_rating, opponent_rating = _elo_update(
        learner_rating, opponent_rating, learner_score
    )
    updated["ratings"][candidate_checkpoint] = learner_rating
    updated["ratings"][opponent_checkpoint] = opponent_rating
    validate_manifest(updated)
    return updated


def parse_native_match_output(output: str) -> dict[str, int]:
    """Parse the single fail-closed evidence record emitted by native match."""
    matches = [
        match
        for line in output.replace("\r", "\n").splitlines()
        if (match := _MATCH_EVIDENCE_PATTERN.fullmatch(line.strip()))
    ]
    if len(matches) != 1:
        raise ManifestError(
            "native match did not emit exactly one structured evidence record"
        )
    result = {
        key: int(value)
        for key, value in matches[0].groupdict().items()
    }
    if (
        result["slot0_wins"] + result["slot1_wins"] + result["draws"]
        != result["games"]
    ):
        raise ManifestError("native match evidence outcomes do not sum to games")
    if (
        result["slot0_gun_kills"] > result["slot0_wins"]
        or result["slot1_gun_kills"] > result["slot1_wins"]
    ):
        raise ManifestError("native match evidence gun kills exceed wins")
    if result["clean_fights"] > result["games"]:
        raise ManifestError("native match evidence clean fights exceed games")
    return result


def _native_match_command(
    *,
    binary: str,
    env_name: str,
    primary_path: str,
    enemy_path: str,
    num_games: int,
    seed: int,
    architecture: Mapping[str, Any],
    overrides: Sequence[str],
) -> list[str]:
    hidden_size = _require_exact_int(
        architecture.get("hidden_size"), "architecture.hidden_size", 1
    )
    num_layers = _require_exact_int(
        architecture.get("num_layers"), "architecture.num_layers", 1
    )
    reserved = {
        "base.load_model_path",
        "base.load_enemy_model_path",
        "base.num_games",
        "base.seed",
        "env.num_agents",
        "policy.hidden_size",
        "policy.num_layers",
        "vec.num_frozen_banks",
        "vec.frozen_bank_hidden_size",
        "vec.frozen_bank_num_layers",
    }
    for override in overrides:
        if not isinstance(override, str) or "=" not in override:
            raise ManifestError(f"invalid native override: {override!r}")
        if override.split("=", 1)[0] in reserved:
            raise ManifestError(f"native override is coordinator-owned: {override}")
    return [
        binary,
        "match",
        env_name,
        f"base.seed={seed}",
        f"base.load_model_path={primary_path}",
        f"base.load_enemy_model_path={enemy_path}",
        f"base.num_games={num_games}",
        "env.num_agents=2",
        f"policy.hidden_size={hidden_size}",
        f"policy.num_layers={num_layers}",
        "vec.num_frozen_banks=1",
        f"vec.frozen_bank_hidden_size={hidden_size}",
        f"vec.frozen_bank_num_layers={num_layers}",
        *overrides,
    ]


def collect_rotation(
    manifest: Mapping[str, Any],
    *,
    candidate_id: str,
    opponent_checkpoint_id: str,
    rotation_id: str,
    binary: str = "./puffer",
    env_name: str = "dogfight",
    num_games: int,
    seed: int,
    overrides: Sequence[str] = (),
    timeout: float = 300.0,
    process_env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run a native match in both seats and record the pair only on success."""
    validate_manifest(manifest)
    _require_exact_int(num_games, "num_games", 1)
    _require_exact_int(seed, "seed")
    if not rotation_id:
        raise ManifestError("rotation_id must not be empty")
    if candidate_id not in manifest["candidates"]:
        raise ManifestError(f"unknown candidate: {candidate_id}")
    candidate = manifest["candidates"][candidate_id]
    if candidate["status"] != "pending":
        raise ManifestError(f"candidate is not pending: {candidate_id}")
    if opponent_checkpoint_id not in (
        set(manifest["anchors"]) | set(manifest["pool"])
    ):
        raise ManifestError(
            f"opponent is not active or fixed: {opponent_checkpoint_id}"
        )

    candidate_entry_data = manifest["checkpoints"][candidate["checkpoint_id"]]
    opponent_entry = manifest["checkpoints"][opponent_checkpoint_id]
    if candidate_entry_data["architecture"] != opponent_entry["architecture"]:
        raise ManifestError("candidate and opponent architectures do not match")
    architecture = candidate_entry_data["architecture"]
    match_specs = (
        (0, candidate_entry_data["path"], opponent_entry["path"]),
        (1, opponent_entry["path"], candidate_entry_data["path"]),
    )
    collected: list[dict[str, Any]] = []
    child_env = dict(os.environ)
    if process_env:
        child_env.update(process_env)

    for seat, primary_path, enemy_path in match_specs:
        command = _native_match_command(
            binary=binary,
            env_name=env_name,
            primary_path=primary_path,
            enemy_path=enemy_path,
            num_games=num_games,
            seed=seed,
            architecture=architecture,
            overrides=overrides,
        )
        try:
            completed = runner(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                shell=False,
                env=child_env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ManifestError(f"native match failed for seat {seat}: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ManifestError(
                f"native match failed for seat {seat} "
                f"with exit {completed.returncode}: {detail}"
            )
        result = parse_native_match_output(completed.stdout)
        if seat == 0:
            learner_wins = result["slot0_wins"]
            opponent_wins = result["slot1_wins"]
            learner_kills = result["slot0_gun_kills"]
            opponent_kills = result["slot1_gun_kills"]
        else:
            learner_wins = result["slot1_wins"]
            opponent_wins = result["slot0_wins"]
            learner_kills = result["slot1_gun_kills"]
            opponent_kills = result["slot0_gun_kills"]
        collected.append(
            {
                "id": f"{candidate_id}:{rotation_id}:seat{seat}",
                "candidate_id": candidate_id,
                "opponent_checkpoint_id": opponent_checkpoint_id,
                "rotation_id": rotation_id,
                "seat": seat,
                "games": result["games"],
                "learner_wins": learner_wins,
                "opponent_wins": opponent_wins,
                "draws": result["draws"],
                "learner_gun_kills": learner_kills,
                "opponent_gun_kills": opponent_kills,
                "clean_fights": result["clean_fights"],
                "command": command,
            }
        )

    updated = copy.deepcopy(dict(manifest))
    for evidence in collected:
        updated = record_match(updated, evidence)
    return updated


def collect_rotation_and_persist(
    path: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run both native seats, then atomically append their evidence pair."""
    starting = load_manifest(path)
    collected = collect_rotation(starting, **kwargs)
    starting_ids = {match["id"] for match in starting["matches"]}
    new_evidence = [
        match for match in collected["matches"] if match["id"] not in starting_ids
    ]
    candidate_id = kwargs["candidate_id"]
    opponent_id = kwargs["opponent_checkpoint_id"]
    checkpoint_ids = (
        starting["candidates"][candidate_id]["checkpoint_id"],
        opponent_id,
    )

    def mutation(current: dict[str, Any]) -> dict[str, Any]:
        if current["candidates"].get(candidate_id) != starting["candidates"][candidate_id]:
            raise ManifestError("candidate changed while native matches were running")
        for checkpoint_id in checkpoint_ids:
            if (
                current["checkpoints"].get(checkpoint_id)
                != starting["checkpoints"][checkpoint_id]
            ):
                raise ManifestError(
                    "checkpoint changed while native matches were running"
                )
        updated = current
        for evidence in new_evidence:
            updated = record_match(updated, evidence)
        return updated

    return update_manifest(path, mutation)


def _aggregate_evidence(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "games": 0,
        "learner_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "learner_gun_kills": 0,
        "opponent_gun_kills": 0,
        "clean_fights": 0,
    }
    for match in evidence:
        for key in totals:
            totals[key] += match[key]
    decisive = totals["learner_gun_kills"] + totals["opponent_gun_kills"]
    totals["decisive_gun_kills"] = decisive
    totals["learner_kill_share"] = (
        totals["learner_gun_kills"] / decisive if decisive else 0.0
    )
    totals["clean_fight_rate"] = (
        totals["clean_fights"] / totals["games"] if totals["games"] else 0.0
    )
    return totals


def promotion_report(
    manifest: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Evaluate persisted, both-seat rotation evidence against release gates."""
    validate_manifest(manifest)
    if candidate_id not in manifest["candidates"]:
        raise ManifestError(f"unknown candidate: {candidate_id}")
    policy = manifest["promotion_policy"]
    candidate_matches = [
        match for match in manifest["matches"] if match["candidate_id"] == candidate_id
    ]
    rotations: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for match in candidate_matches:
        key = (match["rotation_id"], match["opponent_checkpoint_id"])
        rotations.setdefault(key, []).append(match)

    rotation_reports = []
    successful = 0
    covered_anchors: set[str] = set()
    for (rotation_id, opponent_id), evidence in sorted(rotations.items()):
        totals = _aggregate_evidence(evidence)
        both_seats = {match["seat"] for match in evidence} == {0, 1}
        passed = (
            both_seats
            and totals["decisive_gun_kills"]
            >= policy["min_decisive_gun_kills"]
            and totals["learner_kill_share"] >= policy["min_learner_kill_share"]
            and totals["clean_fight_rate"] >= policy["min_clean_fight_rate"]
        )
        if passed:
            successful += 1
            if opponent_id in manifest["anchors"]:
                covered_anchors.add(opponent_id)
        rotation_reports.append(
            {
                "rotation_id": rotation_id,
                "opponent_checkpoint_id": opponent_id,
                "both_seats": both_seats,
                "passed": passed,
                **totals,
            }
        )

    aggregate = _aggregate_evidence(candidate_matches)
    anchors_covered = set(manifest["anchors"]).issubset(covered_anchors)
    aggregate_passed = (
        aggregate["decisive_gun_kills"] >= policy["min_decisive_gun_kills"]
        and aggregate["learner_kill_share"] >= policy["min_learner_kill_share"]
        and aggregate["clean_fight_rate"] >= policy["min_clean_fight_rate"]
    )
    passed = (
        successful >= policy["min_successful_rotations"]
        and anchors_covered
        and aggregate_passed
    )
    return {
        "candidate_id": candidate_id,
        "passed": passed,
        "successful_rotations": successful,
        "required_successful_rotations": policy["min_successful_rotations"],
        "anchors_covered": anchors_covered,
        "aggregate": aggregate,
        "rotations": rotation_reports,
    }


def promote_candidate(
    manifest: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    """Promote a passing candidate into FIFO history without mutating anchors."""
    report = promotion_report(manifest, candidate_id)
    if not report["passed"]:
        raise ManifestError(f"candidate failed promotion gates: {candidate_id}")
    updated = copy.deepcopy(dict(manifest))
    candidate = updated["candidates"][candidate_id]
    if candidate["status"] != "pending":
        raise ManifestError(f"candidate is not pending: {candidate_id}")
    checkpoint_id = candidate["checkpoint_id"]
    if checkpoint_id not in updated["pool"]:
        updated["pool"].append(checkpoint_id)
    while len(updated["pool"]) > updated["pool_capacity"]:
        evicted = updated["pool"].pop(0)
        if updated["current_opponent_id"] == evicted:
            updated["current_opponent_id"] = None
    candidate["status"] = "promoted"
    updated["promotions"].append(
        {
            "candidate_id": candidate_id,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": candidate["parent_checkpoint_id"],
            "selection_counter": updated["selection_counter"],
            "evidence_ids": [
                match["id"]
                for match in updated["matches"]
                if match["candidate_id"] == candidate_id
            ],
        }
    )
    validate_manifest(updated)
    return updated


def promote_and_persist(
    path: str | os.PathLike[str], candidate_id: str
) -> dict[str, Any]:
    """Atomically gate and promote a candidate in the persisted manifest."""
    return update_manifest(
        path, lambda manifest: promote_candidate(manifest, candidate_id)
    )


def _parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("manifest")
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.add_argument("--pool-capacity", type=int, required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("manifest")
    add_parser.add_argument("checkpoint_id")
    add_parser.add_argument("checkpoint_path")
    add_parser.add_argument("--generation", type=int, required=True)
    add_parser.add_argument("--parent-id")
    add_parser.add_argument(
        "--category", choices=("pool", "anchor", "candidate"), required=True
    )
    add_parser.add_argument("--architecture", type=_parse_json_object, default={})
    add_parser.add_argument("--behavior-version", type=int, default=1)
    add_parser.add_argument("--reward-version", type=int, default=1)

    candidate_parser = subparsers.add_parser("register-candidate")
    candidate_parser.add_argument("manifest")
    candidate_parser.add_argument("candidate_id")
    candidate_parser.add_argument("checkpoint_id")
    candidate_parser.add_argument("checkpoint_path")
    candidate_parser.add_argument("--parent-id", required=True)
    candidate_parser.add_argument("--generation", type=int, required=True)
    candidate_parser.add_argument(
        "--architecture", type=_parse_json_object, required=True
    )
    candidate_parser.add_argument("--behavior-version", type=int, default=1)
    candidate_parser.add_argument("--reward-version", type=int, default=1)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("manifest")
    select_parser.add_argument(
        "--strategy", choices=("uniform", "top_n", "pfsp"), default="uniform"
    )
    select_parser.add_argument("--top-n", type=int, default=4)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("manifest")
    collect_parser.add_argument("candidate_id")
    collect_parser.add_argument("opponent_checkpoint_id")
    collect_parser.add_argument("rotation_id")
    collect_parser.add_argument("--binary", default="./puffer")
    collect_parser.add_argument("--env-name", default="dogfight")
    collect_parser.add_argument("--num-games", type=int, required=True)
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--timeout", type=float, default=300.0)
    collect_parser.add_argument("--override", action="append", default=[])

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("manifest")
    report_parser.add_argument("candidate_id")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("manifest")
    promote_parser.add_argument("candidate_id")

    args = parser.parse_args()
    if args.action == "init":
        manifest = new_manifest(seed=args.seed, pool_capacity=args.pool_capacity)
        write_manifest_atomic(args.manifest, manifest)
        return 0
    if args.action == "add":
        entry = checkpoint_entry(
            args.checkpoint_id,
            args.checkpoint_path,
            generation=args.generation,
            parent_id=args.parent_id,
            architecture=args.architecture,
            behavior_version=args.behavior_version,
            reward_version=args.reward_version,
            fixed=args.category == "anchor",
        )
        update_manifest(
            args.manifest,
            lambda manifest: add_checkpoint(
                manifest, entry, category=args.category
            ),
        )
        return 0
    if args.action == "register-candidate":
        entry = checkpoint_entry(
            args.checkpoint_id,
            args.checkpoint_path,
            generation=args.generation,
            parent_id=args.parent_id,
            architecture=args.architecture,
            behavior_version=args.behavior_version,
            reward_version=args.reward_version,
        )
        update_manifest(
            args.manifest,
            lambda manifest: register_candidate(
                manifest,
                candidate_id=args.candidate_id,
                checkpoint_entry_data=entry,
                parent_checkpoint_id=args.parent_id,
            ),
        )
        return 0
    if args.action == "select":
        selected, _ = select_and_persist(
            args.manifest, strategy=args.strategy, top_n=args.top_n
        )
        print(selected)
        return 0
    if args.action == "collect":
        updated = collect_rotation_and_persist(
            args.manifest,
            candidate_id=args.candidate_id,
            opponent_checkpoint_id=args.opponent_checkpoint_id,
            rotation_id=args.rotation_id,
            binary=args.binary,
            env_name=args.env_name,
            num_games=args.num_games,
            seed=args.seed,
            timeout=args.timeout,
            overrides=tuple(args.override),
        )
        print(
            json.dumps(
                promotion_report(updated, args.candidate_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.action == "report":
        report = promotion_report(load_manifest(args.manifest), args.candidate_id)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    if args.action == "promote":
        promote_and_persist(args.manifest, args.candidate_id)
        return 0
    raise AssertionError(f"unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(_main())
