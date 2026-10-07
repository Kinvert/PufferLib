"""Native scalar configuration arithmetic, never a CPU policy/model test."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_learning_recipes as recipes
import prepare_pixel_robustness as panel


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        cls.binary = cls.root / "geometry"; cls.sanitized = cls.root / "geometry-ubsan"
        flags = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]
        for binary, extra in ((cls.binary, []), (cls.sanitized, ["-fsanitize=undefined", "-fno-sanitize-recover=all"])):
            subprocess.run([*flags, *extra, str(panel.ROOT / "research/learner_geometry.c"), "-lm", "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=60)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def config(self, **overrides):
        c = panel.read(panel.ROOT / "config/default.ini", panel.ROOT / "config/flappycnn.ini",
                       panel.ROOT / "ocean/flappycnn/compare.ini")
        c["base"]["checkpoint_interval"] = "16"
        for name, value in overrides.items():
            section, key = name.split("__"); c[section][key] = str(value)
        directory = Path(tempfile.mkdtemp(dir=self.root)); path = directory / "full.ini"
        with path.open("x") as stream: c.write(stream)
        return path

    def value(self, path, sanitized=False):
        return recipes.geometry(self.sanitized if sanitized else self.binary, [path])

    def test_common_rollout_budget_schedule_and_checkpoint_arithmetic(self):
        value = self.value(self.config())
        expected = dict(rollout_decisions_per_rank=2048, epochs=32, optimizer_minibatches_per_epoch=1,
            optimizer_updates_per_rank=32, actual_global_decisions=65536, dropped_requested_decisions=0,
            distinct_slices_per_rollout=1, complete_passes_per_rollout=1, additional_prefix_slices=0,
            rows_covered_per_rollout=64, checkpoint_count=2, nonzero_training_updates=True)
        for key, number in expected.items(): self.assertEqual(value[key], number)
        self.assertAlmostEqual(value["first_epoch_lr_f32"], .001, places=9)
        self.assertAlmostEqual(value["last_epoch_lr_f32"], 0.000002407637, places=11)
        self.assertFalse(value["policy_executed"])

    def test_hand_calculated_current_state_controls(self):
        expected = {
            "connect4": (131072, 101, 50, 5050, 13238272, 3.125),
            "pong": (65536, 122, 16, 1952, 7995392, 4),
            "flappy": (131072, 152, 8, 1216, 19922944, 1),
            "breakout": (131072, 419, 3, 1257, 54919168, 1.5),
            "snakebench": (2048, 6500, 1, 6500, 13312000, 1),
            "maze": (131072, 2578, 22, 56716, 337903616, 2.75),
        }
        keys = ("rollout_decisions_per_rank", "epochs", "optimizer_minibatches_per_epoch",
                "optimizer_updates_per_rank", "actual_global_decisions", "effective_replay_ratio")
        for task, expected_row in expected.items():
            with self.subTest(task=task):
                value = recipes.geometry(self.binary, [panel.ROOT / "config/default.ini", panel.ROOT / "config" / (task+".ini")])
                self.assertEqual(tuple(value[k] for k in keys), expected_row)

    def test_executed_flappy_geometry_matches_retained_counts(self):
        for name in ("stock-pilot.DN7C2JF6", "stock-pilot.tgKBYrNs"):
            files = list((panel.ROOT / "research/results/flappycnn" / name / "metrics").rglob("*.ini"))
            self.assertEqual(len(files), 1)
            value = self.value(files[0])
            self.assertEqual(value["actual_global_decisions"], 19922944)
            self.assertEqual(value["optimizer_updates_per_rank"], 1216)
            self.assertEqual(value["checkpoint_count"], 19)

    def test_replay_quantization_and_zero_updates_are_explicit(self):
        for replay, updates in ((.25, 0), (.99999997, 0), (.99999998, 1), (1.91617525, 1), (3.57716227, 3)):
            with self.subTest(replay=replay):
                value = self.value(self.config(train__replay_ratio=replay), sanitized=True)
                self.assertEqual(value["optimizer_minibatches_per_epoch"], updates)
                self.assertEqual(value["nonzero_training_updates"], updates > 0)
                self.assertEqual(value["rows_covered_per_rollout"], 64 if updates else 0)
                self.assertEqual(value["processed_training_decisions_all_ranks"], 65536*updates)
        # Near one, native float32 conversion can cross a floor boundary;
        # independently flooring the decimal/double would give zero instead.
        self.assertEqual(self.value(self.config(train__replay_ratio=.99999998))["replay_ratio_f32"], 1)

    def test_partial_replay_is_prefix_coverage_not_random_sampling(self):
        value = self.value(self.config(train__minibatch_size=512, train__replay_ratio=.625))
        self.assertEqual(value["optimizer_minibatches_per_epoch"], 2)
        self.assertEqual(value["distinct_slices_per_rollout"], 4)
        self.assertEqual(value["complete_passes_per_rollout"], 0)
        self.assertEqual(value["additional_prefix_slices"], 2)
        self.assertEqual(value["rows_covered_per_rollout"], 32)
        self.assertEqual(value["effective_replay_ratio"], .5)

    def test_zero_epochs_and_multi_rank_budget_rounding(self):
        value = self.value(self.config(train__total_timesteps=1024), sanitized=True)
        self.assertEqual(value["epochs"], 0); self.assertEqual(value["checkpoint_count"], 0)
        self.assertFalse(value["nonzero_training_updates"]); self.assertFalse(value["lr_schedule_evaluated"])
        value = self.value(self.config(train__gpus=2, train__total_timesteps=65600))
        self.assertEqual(value["epochs"], 16); self.assertEqual(value["actual_global_decisions"], 65536)
        self.assertEqual(value["dropped_requested_decisions"], 64)
        self.assertEqual(value["processed_training_decisions_all_ranks"], 65536)

    def test_no_annealing_underscore_parser_and_final_cadence(self):
        value = self.value(self.config(train__total_timesteps="65_537", train__anneal_lr=0,
            train__learning_rate=.01, base__checkpoint_interval=7))
        self.assertEqual(value["actual_global_decisions"], 65536)
        self.assertEqual(value["dropped_requested_decisions"], 1)
        self.assertEqual(value["checkpoint_count"], 5)
        self.assertEqual(value["first_epoch_lr_f32"], value["last_epoch_lr_f32"])
        value = self.value(self.config(base__checkpoint_interval=0)); self.assertEqual(value["checkpoint_count"], 1)

    def test_bad_geometry_scalars_and_overflow_fail_without_undefined_behavior(self):
        cases = (dict(train__horizon=0), dict(train__horizon=30), dict(train__minibatch_size=0),
            dict(train__minibatch_size=4096), dict(train__minibatch_size=1000), dict(vec__total_agents=63),
            dict(vec__num_buffers=3), dict(train__gpus=0), dict(train__replay_ratio=-1), dict(train__replay_ratio="nan"),
            dict(train__replay_ratio="1,2"), dict(train__replay_ratio="None"), dict(train__replay_ratio="1e30"),
            dict(vec__total_agents=2147483647), dict(train__total_timesteps=9007199254740992),
            dict(train__replay_ratio=100000000), dict(train__total_timesteps=(2**31-1)*2048),
            dict(train__learning_rate="inf"), dict(train__min_lr_ratio=1.1), dict(vec__num_policies=2),
            dict(selfplay__enabled=1), dict(vec__hist_policy_percent=.5))
        for case in cases:
            with self.subTest(case=case):
                path = self.config(**case)
                result = subprocess.run([str(self.sanitized), str(path)], capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 1); self.assertFalse(result.stdout)
                self.assertNotIn("runtime error", result.stderr)

    def test_source_guard_rejects_changed_native_count_semantics(self):
        text = (panel.ROOT / "src/pufferl.cu").read_text(); path = self.root / "changed.cu"
        path.write_text(text.replace("int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;",
                                    "int total_minibatches = 1;"))
        with self.assertRaisesRegex(ValueError, "arithmetic changed"): recipes.source_guard(path)

    def test_panel_geometry_full_allocation_and_archived_inspection(self):
        directory = self.root / "panel-fixture"
        panel.prepare(directory, list(panel.ENVIRONMENTS), "all", [53111, 53112], 65536, 32768)
        packet = self.root / "recipe-audit"
        result = recipes.prepare(packet, directory / "protocol.json")
        jobs = 8*sum(panel.ENVIRONMENTS.values())
        self.assertEqual(result["cases"], jobs+14); self.assertEqual(result["panel_jobs"], jobs)
        self.assertFalse(result["learning_calibrated"])
        self.assertEqual(recipes.inspect(packet / "audit.json"), result)
        relocated = self.root / "relocated"; shutil.copytree(packet, relocated)
        (relocated / "learner_geometry").unlink()  # No executable transfer needed for artifact inspection.
        self.assertEqual(recipes.inspect(relocated / "audit.json"), result)
        (relocated / "geometry.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "audit packet"): recipes.inspect(relocated / "audit.json")
        with self.assertRaises(FileExistsError): recipes.prepare(packet, directory / "protocol.json")


if __name__ == "__main__": unittest.main()
