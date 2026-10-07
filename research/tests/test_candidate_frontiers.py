"""Synthetic scalar full-curve accounting, never a CPU/GPU neural model."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import candidate_panel as tool


def fixture():
    candidates = [dict(name="happy-cat-1"), dict(name="nature-cnn"), dict(name="impala-cnn")]
    value = dict(candidates=candidates, jobs=[], seeds=[101, 102], appearance_seed=501,
                 assignments={task: dict(catalog_count=n) for task, n in tool.panel.ENVIRONMENTS.items()})
    rows = []
    for task, count in tool.panel.ENVIRONMENTS.items():
        for seed in value["seeds"]:
            for ci, candidate in enumerate(candidates):
                value["jobs"].append(dict(environment=task, seed=seed, candidate=candidate["name"],
                    checkpoint_steps=[65536, 131072, 196608], targets=[dict(representation=i) for i in range(count)]))
                for step in (65536, 131072, 196608):
                    for drawing in range(count):
                        # Include a decline; retain every point instead of best checkpoint.
                        low = (0.8 if step == 131072 else 0.4) - ci*0.05
                        if seed == 102: low -= 0.2
                        rows.append(dict(environment=task, candidate=candidate["name"], seed=seed, representation=drawing,
                            decisions=step, train_seconds=step/65536*(ci+1), score_lower=low,
                            score_upper=low+0.1 if task == "pongcnn" else low))
    return value, rows


class CandidateFrontierTests(unittest.TestCase):
    def test_full_curve_and_all_drawings_remain_separate(self):
        value, rows = fixture(); result = tool.candidate_frontiers(value, rows, "development")
        self.assertEqual(len(result["points"]), 738)
        self.assertEqual(len(result["means"]), 369)
        self.assertEqual(len(result["conditions"]), 41)
        self.assertEqual(result["status"], "complete_descriptive")
        self.assertFalse(result["dominance_certified"])
        self.assertFalse(result["publication_claim_qualified"])
        self.assertFalse(result["quality_selection_allowed"])
        self.assertFalse(result["cross_game_score_average_enabled"])
        chosen = [r for r in result["means"] if r["environment"] == "connect4cnn" and r["evaluation_representation"] == 0 and r["model"] == "happy-cat-1"]
        self.assertEqual([r["steps"] for r in chosen], [65536, 131072, 196608])
        self.assertGreater(chosen[1]["score_lower"], chosen[2]["score_lower"])
        self.assertAlmostEqual(chosen[0]["score_lower"], .3)

    def test_missing_seed_does_not_compute_partial_seed_mean_or_front(self):
        value, rows = fixture(); omitted = next(r for r in rows if r["environment"] == "connect4cnn" and r["candidate"] == "happy-cat-1"
                                               and r["representation"] == 0 and r["seed"] == 102 and r["decisions"] == 131072)
        rows.remove(omitted); result = tool.candidate_frontiers(value, rows, "development")
        self.assertEqual(len(result["missing"]), 1)
        condition = [r for r in result["means"] if r["environment"] == "connect4cnn" and r["evaluation_representation"] == 0]
        self.assertFalse(any(r["model"] == "happy-cat-1" and r["steps"] == 131072 for r in condition))
        self.assertTrue(all(r["frontier_seconds_lower"] is None and r["frontier_seconds_upper"] is None for r in condition))

    def test_smoke_and_preparation_have_no_quality_front_flags(self):
        value, rows = fixture(); result = tool.candidate_frontiers(value, rows, "smoke")
        self.assertTrue(all(r["frontier_seconds_lower"] is None and r["frontier_seconds_upper"] is None for r in result["means"]))
        empty = tool.candidate_frontiers(value, [], "preparation")
        self.assertEqual(len(empty["missing"]), 738)
        self.assertEqual(empty["points"], [])
        self.assertEqual(empty["means"], [])
        self.assertTrue(all(c["observed"] == 0 for c in empty["conditions"]))

    def test_pong_bounds_are_not_collapsed_or_drawing_averaged(self):
        value, rows = fixture(); result = tool.candidate_frontiers(value, rows, "development")
        pong = [r for r in result["means"] if r["environment"] == "pongcnn"]
        self.assertEqual(len(pong), 63)
        self.assertTrue(all(abs(r["score_upper"]-r["score_lower"]-.1) < 1e-12 for r in pong))
        self.assertTrue(all(c["score_bounds_kind"] == "deterministic-censoring-not-confidence" for c in result["conditions"] if c["environment"] == "pongcnn"))

    def test_duplicate_unassigned_and_nonfinite_points_rejected(self):
        value, rows = fixture()
        for extra in (copy.deepcopy(rows[0]), dict(rows[0], seed=999), dict(rows[0], train_seconds=float("nan")), dict(rows[0], decisions=1234)):
            with self.subTest(extra=extra), self.assertRaises(ValueError): tool.candidate_frontiers(value, rows+[extra], "development")

    def test_drawing_targets_reuse_same_training_time(self):
        value, rows = fixture(); result = tool.candidate_frontiers(value, rows, "development")
        # No train-time sum over appearances: each mean point keeps the one job coordinate.
        condition_times = [r["seconds"] for r in result["means"] if r["environment"] == "connect4cnn"
                           and r["model"] == "happy-cat-1" and r["steps"] == 196608]
        self.assertEqual(condition_times, [3]*10)
        self.assertEqual(result["cost_scope"], "native-training-through-checkpoint-once-per-job")


if __name__ == "__main__": unittest.main()
