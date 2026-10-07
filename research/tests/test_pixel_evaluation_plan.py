"""GPU-free configuration/identity binding checks, not policy/model tests."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_pixel_panel as audit
import pixel_evaluation_plan as bridge
import prepare_pixel_robustness as panel


class PanelAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "panel"
        self.value = panel.prepare(self.root, ["connect4cnn"], "default", [53111], 65536, 32768)
        self.path = self.root / "protocol.json"

    def save(self):
        self.path.write_text(json.dumps(self.value))

    def test_full_inventory_and_relocation(self):
        root = Path(self.tmp.name) / "all"
        value = panel.prepare(root, list(panel.ENVIRONMENTS), "all", [53111, 53112], 65536, 32768)
        self.assertEqual(len(audit.load(root / "protocol.json")["jobs"]), 8*sum(panel.ENVIRONMENTS.values()))
        self.assertEqual(len(value["builds"]), 18)
        relocated = Path(self.tmp.name) / "relocated"; shutil.copytree(root, relocated)
        self.assertEqual(audit.load(relocated / "protocol.json"), value)

    def test_missing_duplicate_and_reordered_jobs(self):
        original = list(self.value["jobs"])
        for jobs in (original[:-1], [original[0]]*4, list(reversed(original))):
            with self.subTest(jobs=jobs):
                self.value["jobs"] = jobs; self.save()
                with self.assertRaises(ValueError): audit.load(self.path)

    def test_rehashed_one_and_all_model_drift_is_rejected(self):
        original = {j["config"]: (self.root / j["config"]).read_bytes() for j in self.value["jobs"]}
        for section, key, number in (("train", "learning_rate", "0.004"), ("env", "player_pieces", "1"),
                                      ("policy", "hidden_size", "256"), ("policy", "cnn_kernel", "3"),
                                      ("policy", "encoder", "1"),
                                      ("base", "load_model_path", "pretend.bin")):
            for all_models in (False, True):
                with self.subTest(section=section, key=key, all_models=all_models):
                    for i, job in enumerate(self.value["jobs"]):
                        path = self.root / job["config"]; path.write_bytes(original[job["config"]])
                        if all_models or i == 0:
                            c = audit.config(path); c[section][key] = number
                            with path.open("w") as stream: c.write(stream)
                        job["config_sha256"] = audit.sha(path)
                    self.save()
                    with self.assertRaisesRegex(ValueError, "captured recipe"): audit.load(self.path)

    def test_source_and_checkpoint_controls_fail(self):
        source = self.root / "source/config/default.ini"; source.write_text(source.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "source snapshot"): audit.load(self.path)
        self.value["source_sha256"]["config/default.ini"] = audit.sha(source); self.save()
        # Even a refreshed JSON receipt must agree with the separately captured build manifest.
        with self.assertRaisesRegex(ValueError, "manifest differs"): audit.load(self.path)

    def test_cadence_selector_build_and_stale_gate_fail(self):
        original = json.loads(json.dumps(self.value))
        changes = (lambda v: v["jobs"][0].update(checkpoint_steps=[65536]),
                   lambda v: v["jobs"][0].update(evaluation_gate="pooled-v1"),
                   lambda v: v["builds"][0]["command"].append("--bf16"),
                   lambda v: v.update(publication_confirmation_launchable=True))
        for change in changes:
            self.value = json.loads(json.dumps(original)); change(self.value); self.save()
            with self.assertRaises(ValueError): audit.load(self.path)

    def test_secondary_native_game_ini_cannot_override_audited_default(self):
        for all_models in (False, True):
            with self.subTest(all_models=all_models):
                for i, job in enumerate(self.value["jobs"]):
                    path = (self.root / job["config"]).parent / "connect4cnn.ini"
                    path.write_text("[train]\nlearning_rate = 0.004\n" if all_models or i == 0
                                    else "# Full resolved settings in default.ini.\n")
                with self.assertRaisesRegex(ValueError, "game INI overrides"): audit.load(self.path)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.panel = self.root / "panel"
        panel.prepare(self.panel, ["connect4cnn"], "all", [53111, 53112], 65536, 32768)
        self.registry = self.root / "registry.json"
        builds = {}
        for family in bridge.BUILD_FAMILIES:
            binary = self.root / family; binary.write_bytes(b"host-metadata-fixture-"+family.encode())
            builds[family] = str(binary)
        self.registry.write_text(json.dumps({"connect4cnn": builds}))
        self.args = SimpleNamespace(panel=self.panel / "protocol.json", registry=self.registry,
            out=self.root / "bound", seed=56123, episodes=3, slots=2, pong_max_decisions=16384, breakout_max_frames=8192)

    def metadata(self, command, **kwargs):
        self.assertEqual(command[1:], ["eval_exact_info"])
        return json.dumps(dict(environment="connect4cnn", rules=panel.ENVIRONMENT_RULES["connect4cnn"],
            float32=True, receipt_version=2, default_encoder="tiny" if Path(command[0]).name == "default" else Path(command[0]).name))

    def prepare(self):
        with patch.object(bridge.subprocess, "check_output", side_effect=self.metadata), \
             patch.object(bridge.subprocess, "run", side_effect=AssertionError("No native execution expected for Connect4 suite")):
            return bridge.prepare(self.args)

    def test_paired_20_conditions_80_jobs_without_gpu_or_model(self):
        value = self.prepare()
        self.assertEqual(value["conditions"], 20); self.assertEqual(len(value["bindings"]), 80)
        self.assertFalse(value["policy_executed"]); self.assertFalse(value["caps_calibrated"])
        self.assertEqual({b["evaluation_seed"] for b in value["bindings"]}, {56123, 56124})
        for binding in value["bindings"]:
            self.assertEqual(binding["checkpoint_steps"], [32768, 65536])
            self.assertIsNone(binding["host_check"])
        relocated = self.root / "relocated"; shutil.copytree(self.args.out, relocated)
        self.assertEqual(bridge.inspect(relocated / "plan.json"), value)
        # Offline artifact inspection executes no native command, including metadata.
        with patch.object(bridge.subprocess, "check_output", side_effect=AssertionError("Must be offline")), \
             patch.object(bridge.subprocess, "run", side_effect=AssertionError("Must be offline")):
            bridge.inspect(self.args.out / "plan.json", check_native=True)
        with self.assertRaises(FileExistsError): self.prepare()

    def test_v3_task_budget_binds_every_declared_checkpoint_without_gpu(self):
        directory = self.root / "budget-panel"
        panel.prepare(directory, ["connect4cnn"], "all", [53111, 53112], 65536, 32768,
            task_budgets={"connect4cnn": dict(steps=98304, checkpoint_steps=65536)})
        self.args.panel = directory / "protocol.json"
        value = self.prepare()
        self.assertEqual(len(value["bindings"]), 80)
        self.assertTrue(all(b["checkpoint_steps"] == [65536, 98304] for b in value["bindings"]))
        self.assertFalse(value["policy_executed"])
        self.assertEqual(bridge.inspect(self.args.out / "plan.json"), value)

    def test_changed_packet_and_job_rebinding_fail(self):
        self.prepare(); path = self.args.out / "plan.json"; original = path.read_text()
        value = json.loads(original); value["bindings"][0]["suite"] = value["bindings"][4]["suite"]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "suite allocation"): bridge.inspect(path)
        path.write_text(original)
        config = self.args.out / "panel" / json.loads(original)["bindings"][0]["config"].removeprefix("panel/")
        config.write_text(config.read_text()+"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "packet files"): bridge.inspect(path)

    def test_changed_binary_and_improper_claim_flags_fail(self):
        self.prepare(); path = self.args.out / "plan.json"; original = path.read_text()
        for key in ("policy_executed", "caps_calibrated", "publication_confirmation_launchable", "gpu_execution_authorized"):
            value = json.loads(original); value[key] = True; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "upgraded"): bridge.inspect(path)
        path.write_text(original); (self.root / "default").write_bytes(b"changed")
        bridge.inspect(path)  # Offline packet still describes its historical binary.
        with self.assertRaisesRegex(ValueError, "binary changed"): bridge.inspect(path, check_native=True)

    def test_invalid_quota_registry_family_and_failure_preservation(self):
        self.args.episodes = 0
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.args.out.exists()); self.args.episodes = 3
        self.registry.write_text(json.dumps({"connect4cnn": {"default": str(self.root / "default")}}))
        with self.assertRaisesRegex(ValueError, "three native"): self.prepare()
        self.assertTrue((self.args.out / "failure.json").is_file())
        self.assertFalse((self.args.out / "plan.json").exists())

    def test_cross_drawing_world_comparison_keeps_rng_and_level(self):
        first = [dict(episode_id="1", env_seed="22", policy_seed="33", level_id="4",
                      representation="0", start_world_hash="abc", start_observation_hash="def")]
        reflected = [dict(first[0], representation="1", start_observation_hash="ghi")]
        self.assertEqual(bridge.worlds(first), bridge.worlds(reflected))
        self.assertNotEqual(bridge.worlds(first), bridge.worlds([dict(reflected[0], env_seed="23")]))
        self.assertNotEqual(bridge.worlds(first), bridge.worlds([dict(reflected[0], level_id="5")]))

    def test_wrong_compiled_family_is_preserved_as_failure(self):
        def wrong(command, **kwargs):
            identity = json.loads(self.metadata(command, **kwargs)); identity["default_encoder"] = "nature"
            return json.dumps(identity)
        with patch.object(bridge.subprocess, "check_output", side_effect=wrong):
            with self.assertRaisesRegex(ValueError, "compiled encoder family"): bridge.prepare(self.args)
        self.assertTrue((self.args.out / "failure.json").is_file())
        self.assertFalse((self.args.out / "plan.json").exists())

    def test_output_overlap_and_evaluation_seed_overflow_fail_early(self):
        self.args.out = self.panel / "nested"
        with self.assertRaisesRegex(ValueError, "overlaps"): self.prepare()
        self.assertFalse(self.args.out.exists())
        self.args.out = self.root / "bound"; self.args.seed = 2**32-1
        with self.assertRaisesRegex(ValueError, "overflows"): self.prepare()
        self.assertFalse(self.args.out.exists())


if __name__ == "__main__": unittest.main()
