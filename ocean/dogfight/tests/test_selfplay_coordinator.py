from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest

from ocean.dogfight.selfplay_coordinator import (
    ManifestError,
    add_checkpoint,
    checkpoint_entry,
    collect_rotation,
    eligible_opponents,
    load_manifest,
    new_manifest,
    parse_native_match_output,
    promote_and_persist,
    promotion_report,
    record_match,
    register_candidate,
    select_and_persist,
    select_opponent,
    write_manifest_atomic,
)


ARCHITECTURE = {"hidden_size": 64, "num_layers": 3}


def _entry(
    tmp_path: Path,
    checkpoint_id: str,
    *,
    generation: int,
    fixed: bool = False,
):
    checkpoint = tmp_path / f"{checkpoint_id}.bin"
    checkpoint.write_bytes((checkpoint_id * 17).encode("ascii"))
    return checkpoint_entry(
        checkpoint_id,
        checkpoint,
        generation=generation,
        architecture=ARCHITECTURE,
        fixed=fixed,
    )


def _manifest_with_anchor_and_pool(tmp_path: Path):
    manifest = new_manifest(seed=42, pool_capacity=2)
    anchor = _entry(tmp_path, "anchor", generation=0, fixed=True)
    parent = _entry(tmp_path, "parent", generation=1)
    history = _entry(tmp_path, "history", generation=2)
    manifest = add_checkpoint(manifest, anchor, category="anchor")
    manifest = add_checkpoint(manifest, parent)
    manifest = add_checkpoint(manifest, history)
    return manifest


def _candidate_manifest(tmp_path: Path):
    manifest = _manifest_with_anchor_and_pool(tmp_path)
    candidate = _entry(tmp_path, "candidate-checkpoint", generation=3)
    return register_candidate(
        manifest,
        candidate_id="candidate-a",
        checkpoint_entry_data=candidate,
        parent_checkpoint_id="parent",
    )


def _evidence(
    evidence_id: str,
    *,
    rotation: str,
    seat: int,
    learner_wins: int = 6,
    opponent_wins: int = 4,
    draws: int = 0,
    learner_kills: int = 6,
    opponent_kills: int = 4,
    clean_fights: int = 8,
):
    games = learner_wins + opponent_wins + draws
    return {
        "id": evidence_id,
        "candidate_id": "candidate-a",
        "opponent_checkpoint_id": "anchor",
        "rotation_id": rotation,
        "seat": seat,
        "games": games,
        "learner_wins": learner_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "learner_gun_kills": learner_kills,
        "opponent_gun_kills": opponent_kills,
        "clean_fights": clean_fights,
        "command": ["./puffer", "match", "dogfight", f"match.seat={seat}"],
    }


def test_checkpoint_bytes_are_verified_on_restart(tmp_path):
    manifest = new_manifest(seed=1, pool_capacity=2)
    entry = _entry(tmp_path, "parent", generation=0)
    manifest = add_checkpoint(manifest, entry)
    manifest_path = tmp_path / "manifest.json"
    write_manifest_atomic(manifest_path, manifest)

    assert load_manifest(manifest_path) == manifest
    Path(entry["path"]).write_bytes(b"tampered")

    with pytest.raises(ManifestError, match="size changed|hash changed"):
        load_manifest(manifest_path)


def test_checkpoint_ids_and_fixed_anchors_are_immutable(tmp_path):
    manifest = new_manifest(seed=1, pool_capacity=1)
    anchor = _entry(tmp_path, "anchor", generation=0, fixed=True)
    manifest = add_checkpoint(manifest, anchor, category="anchor")
    changed = copy.deepcopy(anchor)
    changed["generation"] = 99

    with pytest.raises(ManifestError, match="immutable"):
        add_checkpoint(manifest, changed, category="anchor")

    for generation in (1, 2, 3):
        manifest = add_checkpoint(
            manifest,
            _entry(tmp_path, f"history-{generation}", generation=generation),
        )

    assert manifest["anchors"] == ["anchor"]
    assert manifest["pool"] == ["history-3"]
    assert manifest["checkpoints"]["anchor"] == anchor


def test_restart_preserves_exact_selection_sequence(tmp_path):
    manifest = _manifest_with_anchor_and_pool(tmp_path)
    path = tmp_path / "manifest.json"
    write_manifest_atomic(path, manifest)

    first, persisted = select_and_persist(path, strategy="uniform")
    expected_second, expected_state = select_opponent(persisted, strategy="uniform")
    restarted = load_manifest(path)
    actual_second, actual_state = select_and_persist(path, strategy="uniform")

    assert first != actual_second
    assert actual_second == expected_second
    assert actual_state == expected_state
    assert load_manifest(path) == expected_state


def test_selection_excludes_loaded_opponent_when_alternatives_exist(tmp_path):
    manifest = _manifest_with_anchor_and_pool(tmp_path)
    selected, manifest = select_opponent(manifest)

    eligible = dict(eligible_opponents(manifest))
    assert selected not in eligible
    assert len(eligible) == 2

    sole = new_manifest(seed=9, pool_capacity=1)
    sole = add_checkpoint(
        sole, _entry(tmp_path, "sole", generation=0), category="pool"
    )
    selected, sole = select_opponent(sole)
    assert dict(eligible_opponents(sole)) == {selected: 1.0}


def test_top_n_and_pfsp_use_persisted_evidence(tmp_path):
    manifest = _manifest_with_anchor_and_pool(tmp_path)
    manifest["ratings"].update({"anchor": 900.0, "parent": 1200.0, "history": 1100.0})
    top = eligible_opponents(manifest, strategy="top_n", top_n=2)
    assert [checkpoint_id for checkpoint_id, _ in top] == ["parent", "history"]

    manifest = _candidate_manifest(tmp_path)
    manifest = record_match(
        manifest,
        _evidence(
            "easy",
            rotation="baseline",
            seat=0,
            learner_wins=9,
            opponent_wins=1,
            learner_kills=9,
            opponent_kills=1,
            clean_fights=10,
        ),
    )
    manifest["matches"][-1]["opponent_checkpoint_id"] = "parent"
    hard_weight = dict(eligible_opponents(manifest, strategy="pfsp"))["anchor"]
    easy_weight = dict(eligible_opponents(manifest, strategy="pfsp"))["parent"]
    assert hard_weight > easy_weight


def test_promotion_requires_two_passing_both_seat_anchor_rotations(tmp_path):
    manifest = _candidate_manifest(tmp_path)
    for rotation in ("rotation-1", "rotation-2"):
        for seat in (0, 1):
            manifest = record_match(
                manifest,
                _evidence(f"{rotation}-seat-{seat}", rotation=rotation, seat=seat),
            )

    report = promotion_report(manifest, "candidate-a")
    assert report["passed"] is True
    assert report["successful_rotations"] == 2
    assert report["anchors_covered"] is True
    assert report["aggregate"]["learner_kill_share"] == pytest.approx(0.6)
    assert report["aggregate"]["clean_fight_rate"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "failure",
    ("one_rotation", "one_seat", "too_few_kills", "low_share", "dirty_flights"),
)
def test_promotion_fails_closed_when_any_release_gate_is_missing(tmp_path, failure):
    manifest = _candidate_manifest(tmp_path)
    rotations = ("rotation-1",) if failure == "one_rotation" else (
        "rotation-1",
        "rotation-2",
    )
    seats = (0,) if failure == "one_seat" else (0, 1)
    for rotation in rotations:
        for seat in seats:
            overrides = {}
            if failure == "too_few_kills":
                overrides = {
                    "learner_wins": 3,
                    "opponent_wins": 1,
                    "draws": 6,
                    "learner_kills": 3,
                    "opponent_kills": 1,
                }
            elif failure == "low_share":
                overrides = {
                    "learner_wins": 5,
                    "opponent_wins": 5,
                    "learner_kills": 5,
                    "opponent_kills": 5,
                }
            elif failure == "dirty_flights":
                overrides = {"clean_fights": 7}
            manifest = record_match(
                manifest,
                _evidence(
                    f"{rotation}-seat-{seat}",
                    rotation=rotation,
                    seat=seat,
                    **overrides,
                ),
            )

    assert promotion_report(manifest, "candidate-a")["passed"] is False


def test_promotion_is_atomic_and_preserves_fixed_anchor(tmp_path):
    manifest = _candidate_manifest(tmp_path)
    anchor_before = copy.deepcopy(manifest["checkpoints"]["anchor"])
    for rotation in ("rotation-1", "rotation-2"):
        for seat in (0, 1):
            manifest = record_match(
                manifest,
                _evidence(f"{rotation}-seat-{seat}", rotation=rotation, seat=seat),
            )
    path = tmp_path / "manifest.json"
    write_manifest_atomic(path, manifest)

    promoted = promote_and_persist(path, "candidate-a")

    assert promoted["candidates"]["candidate-a"]["status"] == "promoted"
    assert promoted["pool"][-1] == "candidate-checkpoint"
    assert promoted["checkpoints"]["anchor"] == anchor_before
    assert load_manifest(path) == promoted
    assert not list(tmp_path.glob(".manifest.json.tmp.*"))


def test_native_match_output_parser_requires_complete_final_evidence():
    output = (
        "\rgames=31/32  A=0.600  B=0.400  draw=0.125\n"
        "match/evidence games=32 slot0_wins=18 slot1_wins=10 draws=4 "
        "slot0_gun_kills=12 slot1_gun_kills=7 clean_fights=29\n"
    )

    assert parse_native_match_output(output) == {
        "games": 32,
        "slot0_wins": 18,
        "slot1_wins": 10,
        "draws": 4,
        "slot0_gun_kills": 12,
        "slot1_gun_kills": 7,
        "clean_fights": 29,
    }
    with pytest.raises(ManifestError, match="structured evidence"):
        parse_native_match_output("games=32/32 A=.5 B=.5 draw=.1")


def test_collect_rotation_runs_both_seats_and_fails_closed(tmp_path):
    manifest = _candidate_manifest(tmp_path)
    calls = []
    outputs = iter(
        (
            "match/evidence games=20 slot0_wins=12 slot1_wins=6 draws=2 "
            "slot0_gun_kills=8 slot1_gun_kills=4 clean_fights=18\n",
            "match/evidence games=20 slot0_wins=7 slot1_wins=11 draws=2 "
            "slot0_gun_kills=4 slot1_gun_kills=7 clean_fights=17\n",
        )
    )

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, next(outputs), "")

    updated = collect_rotation(
        manifest,
        candidate_id="candidate-a",
        opponent_checkpoint_id="anchor",
        rotation_id="rotation-native",
        binary="./puffer",
        env_name="dogfight",
        num_games=20,
        seed=42,
        overrides=("env.fixed_stage=3",),
        runner=runner,
    )

    assert len(calls) == 2
    candidate_path = manifest["checkpoints"]["candidate-checkpoint"]["path"]
    anchor_path = manifest["checkpoints"]["anchor"]["path"]
    assert f"base.load_model_path={candidate_path}" in calls[0][0]
    assert f"base.load_enemy_model_path={anchor_path}" in calls[0][0]
    assert f"base.load_model_path={anchor_path}" in calls[1][0]
    assert f"base.load_enemy_model_path={candidate_path}" in calls[1][0]
    assert calls[0][1]["shell"] is False
    assert [match["seat"] for match in updated["matches"]] == [0, 1]
    assert updated["matches"][0]["learner_wins"] == 12
    assert updated["matches"][1]["learner_wins"] == 11
    assert updated["matches"][0]["learner_gun_kills"] == 8
    assert updated["matches"][1]["learner_gun_kills"] == 7
    assert updated["matches"][0]["command"] == calls[0][0]
    assert updated["matches"][1]["command"] == calls[1][0]

    failed_calls = 0

    def failing_runner(command, **kwargs):
        nonlocal failed_calls
        failed_calls += 1
        if failed_calls == 1:
            return subprocess.CompletedProcess(
                command,
                0,
                "match/evidence games=20 slot0_wins=12 slot1_wins=6 draws=2 "
                "slot0_gun_kills=8 slot1_gun_kills=4 clean_fights=18\n",
                "",
            )
        return subprocess.CompletedProcess(command, 2, "", "native failure")

    before = copy.deepcopy(manifest)
    with pytest.raises(ManifestError, match="native match failed"):
        collect_rotation(
            manifest,
            candidate_id="candidate-a",
            opponent_checkpoint_id="anchor",
            rotation_id="rotation-failed",
            binary="./puffer",
            env_name="dogfight",
            num_games=20,
            seed=42,
            runner=failing_runner,
        )
    assert manifest == before
