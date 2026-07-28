import json
import re
from pathlib import Path


SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FIXTURES = {
    "opponent_aware_26_fixed_states",
    "scripted_one_step",
    "scripted_twelve_step",
    "terminal_slot0_kill",
    "terminal_slot1_kill",
    "terminal_vertical_bounds",
    "terminal_timeout",
    "multi_episode_reset",
    "unbounded_action_clamp",
    "state_roundtrip_determinism",
    "curriculum_max_steps",
    "curriculum_stage_geometry",
}


def load_manifest():
    path = Path(__file__).with_name("FIXTURE_PROVENANCE.json")
    return path, json.loads(path.read_text())


def test_fixture_manifest_is_complete_and_portable():
    manifest_path, manifest = load_manifest()
    repo_root = manifest_path.parents[3]

    for key in (
        "target_5c_base",
        "dogfight3_behavior_head",
        "dogfight3_comparison_oracle",
        "dogfight5_fixture_donor",
    ):
        assert SHA_PATTERN.fullmatch(manifest[key]), key

    policy = manifest["golden_update_policy"]
    assert policy["approval_required"] is True
    assert policy["generator_revision_required"] is True
    assert policy["regeneration_command_required"] is True
    assert policy["silent_updates_forbidden"] is True

    fixtures = manifest["fixtures"]
    ids = [fixture["id"] for fixture in fixtures]
    assert len(ids) == len(set(ids))
    assert set(ids) == REQUIRED_FIXTURES

    for fixture in fixtures:
        assert fixture["category"]
        assert fixture["source_test"]
        assert SHA_PATTERN.fullmatch(fixture["source_commit"])

        for field in ("source_path", "target_path"):
            path = Path(fixture[field])
            assert not path.is_absolute()
            assert ".." not in path.parts

        assert (repo_root / fixture["target_path"]).is_file()

        status = fixture["provenance_status"]
        oracle = fixture["behavior_oracle_commit"]
        if status == "quarantined_unpinned_oracle":
            assert oracle is None
            assert fixture["quarantine_reason"]
        elif status == "accepted":
            assert SHA_PATTERN.fullmatch(oracle)
        else:
            assert status == "invariant_source_pinned"

