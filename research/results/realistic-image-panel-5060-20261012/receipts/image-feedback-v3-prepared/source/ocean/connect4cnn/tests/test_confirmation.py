"""Frozen-panel selection and training/evaluation history separation checks."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from compare_sidecar import history
from wandb_sidecar import read_ini


class ConfirmationTests(unittest.TestCase):
    def test_panel_matches_development_choices(self):
        panel = json.loads((HERE / "confirmation.json").read_text())
        evidence = HERE.parents[1] / "research/results/connect4cnn/sweep._u86vi03/analysis/summary.json"
        selected = json.loads(evidence.read_text())["selected_before_evaluation"]
        for variant, trial in zip(("flex_quality", "flex_fast", "flex_small"), selected):
            active = {k.removeprefix("policy."): v for k, v in trial["architecture"].items() if k.startswith("policy.cnn_")}
            self.assertEqual(panel["policies"][variant], {"encoder": 4, **active})
            self.assertEqual(panel["parameters"][variant], trial["params"])
        for settings in (panel, panel["canary"]):
            self.assertEqual(settings["steps"] % (2048 * settings["checkpoints"]), 0)
            self.assertTrue(set(settings["seeds"]).isdisjoint(range(settings["eval_seed"], settings["eval_seed"] + len(settings["seeds"]))))
            self.assertTrue(set(settings["seeds"]).isdisjoint({73, 74, 75}))

    def test_history_merges_eval_without_overwriting_training(self):
        ini = {"metrics": {"agent_steps": "10,30", "env/perf": ".2,.4", "SPS": "100,110", "uptime": ".1,.3"}}
        evaluations = [{"steps": "20", "win_rate": ".5", "score": "0", "games": "1024", "checkpoint_wall_s": ".2"},
                       {"steps": "30", "win_rate": ".6", "score": ".2", "games": "1040", "checkpoint_wall_s": ".31"}]
        rows = history(ini, evaluations)
        self.assertEqual([r["agent_steps"] for r in rows], [10, 20, 30])
        self.assertNotIn("env/perf", rows[1])
        self.assertEqual(rows[-1]["env/perf"], .4)
        self.assertEqual(rows[-1]["eval/perf"], .6)
        self.assertEqual(rows[-1]["SPS"], 110)


if __name__ == "__main__":
    unittest.main()
