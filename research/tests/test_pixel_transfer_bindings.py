"""Cross-drawing checkpoint allocation checks; no GPU/policy or CPU model."""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from research.tests import test_pixel_evaluation_plan as fixtures
import audit_pixel_panel as audit
import pixel_evaluation_plan as bridge
import prepare_pixel_robustness as panel


class TransferTests(unittest.TestCase):
    metadata = fixtures.BindingTests.metadata
    prepare = fixtures.BindingTests.prepare

    def setUp(self):
        fixtures.BindingTests.setUp(self)
        self.args.evaluation_appearances = "all"
        self.default_panel = self.root / "default-panel"
        panel.prepare(self.default_panel, ["connect4cnn"], "default", [53111, 53112], 65536, 32768)
        self.args.panel = self.default_panel / "protocol.json"

    def test_same_eight_training_jobs_bind_all_twenty_conditions(self):
        value = self.prepare()
        self.assertEqual(value["protocol"], bridge.TRANSFER_PROTOCOL)
        self.assertEqual(value["training_jobs"], 8)
        self.assertEqual(value["conditions"], 20)
        self.assertEqual(len(value["bindings"]), 80)
        self.assertFalse(value["unseen_appearance_selection_exclusion_certified"])
        jobs = audit.load(self.args.panel)["jobs"]
        for index, job in enumerate(jobs):
            bindings = [b for b in value["bindings"] if b["job_index"] == index]
            self.assertEqual([b["evaluation_representation"] for b in bindings], list(range(10)))
            self.assertEqual({b["training_representation"] for b in bindings}, {0})
            self.assertEqual({b["config"] for b in bindings}, {"panel/"+job["config"]})
            self.assertTrue(all(b["checkpoint_steps"] == [32768, 65536] for b in bindings))
            self.assertEqual({b["evaluation_seed"] for b in bindings},
                             {56123 + [53111, 53112].index(job["seed"])})
        for seed in (53111, 53112):
            for drawing in range(10):
                bindings = [b for b in value["bindings"] if b["training_seed"] == seed
                            and b["evaluation_representation"] == drawing]
                self.assertEqual({b["model"] for b in bindings}, set(panel.MODELS))
                self.assertEqual(len({b["suite"] for b in bindings}), 1)
        relocated = self.root / "relocated"; shutil.copytree(self.args.out, relocated)
        with patch.object(bridge.subprocess, "check_output", side_effect=AssertionError("Must remain offline")), \
             patch.object(bridge.subprocess, "run", side_effect=AssertionError("Must remain offline")):
            self.assertEqual(bridge.inspect(relocated / "plan.json", check_native=True), value)

    def test_all_training_drawings_are_a_complete_transfer_matrix(self):
        self.args.panel = self.panel / "protocol.json"
        value = self.prepare()
        self.assertEqual(value["training_jobs"], 80)
        self.assertEqual(value["conditions"], 20)
        self.assertEqual(len(value["bindings"]), 800)
        for model in panel.MODELS:
            pairs = [(b["training_representation"], b["evaluation_representation"], b["training_seed"])
                     for b in value["bindings"] if b["model"] == model]
            self.assertEqual(len(pairs), 200)
            self.assertEqual(set(pairs), {(train, test, seed) for train in range(10)
                                         for test in range(10) for seed in (53111, 53112)})

    def test_v3_budget_keeps_all_checkpoints_for_every_target(self):
        directory = self.root / "budget-panel"
        panel.prepare(directory, ["connect4cnn"], "default", [53111, 53112], 65536, 32768,
                      task_budgets={"connect4cnn": dict(steps=98304, checkpoint_steps=65536)})
        self.args.panel = directory / "protocol.json"
        value = self.prepare()
        self.assertEqual(value["training_jobs"], 8)
        self.assertTrue(all(b["checkpoint_steps"] == [65536, 98304] for b in value["bindings"]))

    def test_missing_duplicate_reordered_rebound_or_wrong_drawings_reject(self):
        self.prepare(); path = self.args.out / "plan.json"; original = path.read_text()
        changes = (
            lambda p: p["bindings"].pop(),
            lambda p: p["bindings"].__setitem__(1, dict(p["bindings"][0])),
            lambda p: p["bindings"].reverse(),
            lambda p: p["bindings"][0].update(job_index=1),
            lambda p: p["bindings"][0].update(config=p["bindings"][10]["config"]),
            lambda p: p["bindings"][0].update(training_representation=1),
            lambda p: p["bindings"][0].update(evaluation_representation=1),
            lambda p: p["bindings"][0].update(evaluation_representation=0.0),
            lambda p: p["bindings"][0].update(suite=p["bindings"][1]["suite"]),
            lambda p: p["bindings"][0].update(checkpoint_steps=[65536]),
            lambda p: p.update(training_jobs=80),
            lambda p: p.update(evaluation_appearances="matched"),
            lambda p: p.update(unseen_appearance_selection_exclusion_certified=True),
        )
        for change in changes:
            value = json.loads(original); change(value); path.write_text(json.dumps(value))
            with self.subTest(change=change), self.assertRaises(ValueError): bridge.inspect(path)
        path.write_text(original); bridge.inspect(path)

    def test_legacy_mode_stays_matched_and_cannot_claim_transfer(self):
        self.args.evaluation_appearances = "matched"
        value = self.prepare()
        self.assertEqual(value["protocol"], bridge.PROTOCOL)
        self.assertEqual(len(value["bindings"]), 8)
        self.assertEqual(value["conditions"], 2)
        self.assertNotIn("training_jobs", value)
        self.assertTrue(all("evaluation_representation" not in b for b in value["bindings"]))
        path = self.args.out / "plan.json"; value["evaluation_appearances"] = "all"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Legacy binding"): bridge.inspect(path)

    def test_all_six_games_have_complete_target_allocation(self):
        # Pure scalar enumeration, not a native world or model fixture.
        jobs = [dict(environment=task, representation=0, model=model, seed=53111)
                for task in panel.ENVIRONMENTS for model in panel.MODELS]
        specs = list(bridge.binding_specs(dict(jobs=jobs), "all"))
        self.assertEqual(len(specs), 4*sum(panel.ENVIRONMENTS.values()))
        self.assertEqual(len({(j["environment"], r) for _, j, r in specs}), sum(panel.ENVIRONMENTS.values()))
        for index, job in enumerate(jobs):
            self.assertEqual([r for i, _, r in specs if i == index], list(range(panel.ENVIRONMENTS[job["environment"]])))

    def test_bad_mode_rejects_before_output_creation(self):
        self.args.evaluation_appearances = "best"
        with self.assertRaisesRegex(ValueError, "appearance mode"): self.prepare()
        self.assertFalse(self.args.out.exists())


if __name__ == "__main__": unittest.main()
