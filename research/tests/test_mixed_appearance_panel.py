"""Native scalar mixed-slot/configuration checks; no neural model or GPU."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from research.tests import test_pixel_evaluation_plan as fixtures
import audit_pixel_panel as audit
import pixel_evaluation_plan as bridge
import prepare_pixel_robustness as panel


class MixedPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, name="panel", **kwargs):
        args = dict(environments=list(panel.ENVIRONMENTS), appearances="mixed", seeds=[53151, 53152],
                    steps=65536, checkpoint_steps=32768, appearance_seed=12345)
        args.update(kwargs)
        return panel.prepare(self.root / name, **args)

    def test_all_six_games_pair_slots_settings_and_full_catalogs(self):
        value = self.prepare(); root = self.root / "panel"
        self.assertEqual(value["version"], "pixel-robustness-preparation-v4")
        self.assertEqual(value["learner_contract"], "shared-native-geometry-v3")
        self.assertEqual(len(value["jobs"]), 48)
        self.assertEqual(len(value["builds"]), 18)
        self.assertEqual(audit.load(root / "protocol.json"), value)
        groups = {}
        for job in value["jobs"]:
            c = audit.config(root / job["config"])
            self.assertEqual(c.getint("env", "representation_mode"), 1)
            self.assertEqual(c.getint("env", "representation_seed"), 12345)
            self.assertEqual(c.getint("env", "representation"), 0)
            self.assertEqual(c.get("base", "load_model_path"), "None")
            self.assertEqual(job["checkpoint_steps"], [32768, 65536])
            groups.setdefault((job["environment"], job["seed"]), []).append(audit.common(c))
            if job["environment"] == "pongcnn":
                self.assertEqual(c.getint("env", "representation_mix_catalog"), 1)
        self.assertTrue(all(len(g) == 4 and all(c == g[0] for c in g) for g in groups.values()))
        for task, record in value["appearance_assignments"].items():
            self.assertEqual(record["catalog_count"], panel.ENVIRONMENTS[task])
            self.assertEqual(sum(record["counts"]), 64)
            self.assertTrue(record["all_drawings_assigned"])
            self.assertFalse(record["native_vector_initialization_validated"])
            with (root / record["csv"]).open() as stream: actual = list(csv.DictReader(stream))
            self.assertEqual([int(r["representation"]) for r in actual],
                             [panel.appearance_assignment(12345, i, panel.ENVIRONMENTS[task]) for i in range(64)])
        relocated = self.root / "relocated"; shutil.copytree(root, relocated)
        (relocated / "validation/learner_geometry").unlink()
        (relocated / "validation/appearance_assignments").unlink()
        with patch.object(panel.subprocess, "run", side_effect=AssertionError("Must be offline")), \
             patch.object(panel.subprocess, "check_output", side_effect=AssertionError("Must be offline")):
            self.assertEqual(audit.load(relocated / "protocol.json"), value)

    def test_mixed_with_shared_stock_overlay_and_per_game_budgets(self):
        value = self.prepare(environments=["connect4cnn", "flappycnn"],
            learner_recipes={"flappycnn": panel.ROOT / "ocean/flappycnn/learner_stock_small_batch.ini"},
            task_budgets={"flappycnn": dict(steps=262144, checkpoint_steps=131072)}, appearance_seed=2**32-1)
        audit.load(self.root / "panel/protocol.json")
        self.assertEqual(value["appearance_assignments"]["flappycnn"]["slots"], 2048)
        for job in value["jobs"]:
            self.assertEqual(job["checkpoint_steps"], [131072, 262144] if job["environment"] == "flappycnn" else [32768, 65536])

    def test_seed_extremes_and_rejection_branch_match_native_under_ubsan(self):
        binary = self.root / "assignment-ubsan"
        subprocess.run(["cc", "-std=c11", "-O1", "-g", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                        str(panel.ROOT / "research/appearance_assignments.c"), "-lm", "-o", str(binary)], check=True)
        config = self.root / "assignment.ini"
        for seed in (0, 12345, 0x9e3779b9, 2**32-1):
            for count in (4, 5, 6, 7, 10):
                # With seed==salt, slot 0 hashes to 0 and triggers rejection for 5/6/7/10.
                config.write_text(f"[env]\nrepresentation_mode = 1\nrepresentation_seed = {seed}\n[vec]\ntotal_agents = 128\n")
                raw = subprocess.check_output([str(binary), str(config), str(count)], text=True)
                rows = list(csv.DictReader(raw.splitlines()))
                self.assertEqual([int(r["representation"]) for r in rows],
                                 [panel.appearance_assignment(seed, i, count) for i in range(128)])
        self.assertEqual([panel.appearance_assignment(12345, i, 10) for i in range(16)],
                         [0,0,7,8,6,4,4,4,6,4,7,3,8,8,1,5])

    def test_bad_seed_or_fixed_mode_seed_rejects_before_output_creation(self):
        for index, seed in enumerate((-1, 2**32, True, 12345.0)):
            with self.subTest(seed=seed), self.assertRaises(ValueError): self.prepare(name=f"bad-{index}", appearance_seed=seed)
            self.assertFalse((self.root / f"bad-{index}").exists())
        with self.assertRaisesRegex(ValueError, "requires mixed"):
            self.prepare(name="fixed", appearances="default", appearance_seed=0)
        self.assertFalse((self.root / "fixed").exists())

    def test_native_receipt_rejects_bad_mode_seed_slot_and_catalog_inputs(self):
        binary = self.root / "assignment"
        subprocess.run(["cc", "-std=c11", "-O2", str(panel.ROOT / "research/appearance_assignments.c"),
                        "-lm", "-o", str(binary)], check=True)
        config = self.root / "assignment.ini"
        cases = [("0", "12345", "64", "10"), ("2", "12345", "64", "10"),
                 ("1", "-1", "64", "10"), ("1", "4294967296", "64", "10"),
                 ("1", "nan", "64", "10"), ("1", "1.5", "64", "10"),
                 ("1", "12345", "0", "10"), ("1", "12345", "1.5", "10"),
                 ("1", "12345", "1048577", "10"), ("1", "12345", "64", "0"),
                 ("1", "12345", "64", "65"), ("1", "12345", "64", "3.5")]
        for mode, seed, slots, count in cases:
            config.write_text(f"[env]\nrepresentation_mode = {mode}\nrepresentation_seed = {seed}\n[vec]\ntotal_agents = {slots}\n")
            result = subprocess.run([str(binary), str(config), count], text=True, capture_output=True)
            with self.subTest(mode=mode, seed=seed, slots=slots, count=count):
                self.assertNotEqual(result.returncode, 0)
                self.assertLessEqual(len(result.stdout.splitlines()), 1)  # No accepted slot record.

    def test_assignment_budget_metadata_and_rehashed_one_model_drift_reject(self):
        self.prepare(environments=["connect4cnn"])
        root = self.root / "panel"; path = root / "protocol.json"; original = path.read_text()
        changes = (
            lambda v: v.update(appearance_seed=12346),
            lambda v: v["appearance_assignments"]["connect4cnn"].update(slots=63),
            lambda v: v["appearance_assignments"]["connect4cnn"].update(catalog_count=9),
            lambda v: v["appearance_assignments"]["connect4cnn"].update(mode=True),
            lambda v: v["appearance_assignments"]["connect4cnn"].update(native_vector_initialization_validated=True),
            lambda v: v["appearance_assignments"]["connect4cnn"]["counts"].__setitem__(0, 99),
            lambda v: v["task_budgets"]["connect4cnn"].update(steps=98304),
            lambda v: v.update(appearances="default"),
        )
        for change in changes:
            value = json.loads(original); change(value); path.write_text(json.dumps(value))
            with self.subTest(change=change), self.assertRaises(ValueError): audit.load(path)
        path.write_text(original); value = json.loads(original)
        job = value["jobs"][0]; config = root / job["config"]; c = audit.config(config)
        c["env"]["representation_mode"] = "0"
        with config.open("w") as stream: c.write(stream)
        job["config_sha256"] = audit.sha(config); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "captured recipe"): audit.load(path)

    def test_rehashed_csv_assignment_and_changed_helper_binary_reject(self):
        value = self.prepare(environments=["connect4cnn"]); root = self.root / "panel"
        path = root / "protocol.json"; original = path.read_text()
        record = value["appearance_assignments"]["connect4cnn"]; csv_path = root / record["csv"]
        csv_path.write_text(csv_path.read_text().replace("0,0\n", "0,1\n", 1))
        record["csv_sha256"] = audit.sha(csv_path); path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Mixed slots"): audit.load(path)
        path.write_text(original)
        binary = root / "validation/appearance_assignments"; binary.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "scalar binary"): audit.load(path)


class MixedBindingTests(unittest.TestCase):
    metadata = fixtures.BindingTests.metadata
    prepare = fixtures.BindingTests.prepare

    def setUp(self):
        fixtures.BindingTests.setUp(self)
        mixed = self.root / "mixed-panel"
        panel.prepare(mixed, ["connect4cnn"], "mixed", [53111, 53112], 65536, 32768, appearance_seed=12345)
        self.args.panel = mixed / "protocol.json"
        self.args.evaluation_appearances = "all"

    def test_mixed_jobs_bind_every_fixed_drawing_and_relocate_offline(self):
        value = self.prepare()
        self.assertEqual(value["protocol"], bridge.MIXED_PROTOCOL)
        self.assertEqual(value["training_appearances"], "mixed-fixed-per-slot")
        self.assertEqual(value["training_jobs"], 8)
        self.assertEqual(len(value["bindings"]), 80)
        for binding in value["bindings"]:
            self.assertIsNone(binding["training_representation"])
            self.assertEqual(binding["training_representation_seed"], 12345)
            self.assertEqual(binding["training_catalog_count"], 10)
            self.assertEqual(binding["training_representation_mode"], 1)
            config = audit.config(self.args.out / binding["config"])
            self.assertEqual(config.getint("env", "representation_mode"), 1)
            suite = bridge.adapter("connect4cnn").load_suite(self.args.out / binding["suite"])
            self.assertEqual(suite["representation"], binding["evaluation_representation"])
        relocated = self.root / "relocated"; shutil.copytree(self.args.out, relocated)
        self.assertEqual(bridge.inspect(relocated / "plan.json"), value)

    def test_matched_mixed_evaluation_is_rejected_before_output_creation(self):
        self.args.evaluation_appearances = "matched"
        with self.assertRaisesRegex(ValueError, "explicit all-drawing"): self.prepare()
        self.assertFalse(self.args.out.exists())

    def test_wrong_protocol_or_ambiguous_seed_drawing_metadata_rejects(self):
        self.prepare(); path = self.args.out / "plan.json"; original = path.read_text()
        changes = (
            lambda v: v.update(protocol=bridge.TRANSFER_PROTOCOL),
            lambda v: v.update(training_appearances="fixed"),
            lambda v: v["bindings"][0].update(training_representation=0),
            lambda v: v["bindings"][0].update(training_representation_seed=12346),
            lambda v: v["bindings"][0].update(training_catalog_count=9),
            lambda v: v["bindings"][0].update(training_representation_mode=True),
        )
        for change in changes:
            value = json.loads(original); change(value); path.write_text(json.dumps(value))
            with self.subTest(change=change), self.assertRaises(ValueError): bridge.inspect(path)


if __name__ == "__main__": unittest.main()
