"""Offline discovery diagnosis tests, without neural or native execution."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/analysis"))
spec = importlib.util.spec_from_file_location("generalization_diagnostics", ROOT / "research/analysis/generalization_diagnostics.py")
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
PACKET = ROOT / "research/results/learning-continuation-5090-20261009-prefix45"


class GeneralizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = tool.diagnose(PACKET)

    def test_decline_is_not_replaced_by_best_earlier_score(self):
        points = [dict(decisions=100, seconds=1, score_lower=.9),
                  dict(decisions=200, seconds=2, score_lower=.1),
                  dict(decisions=300, seconds=4, score_lower=1)]
        self.assertEqual(tool.latest_affordable(points, 3), points[1])

    def test_unmeasured_early_policy_is_not_interpolated(self):
        self.assertIsNone(tool.latest_affordable([dict(decisions=100, seconds=4)], 3))

    def test_actual_packet_shows_concentration_and_matched_time_regressions(self):
        value = self.result
        coverage = {r["candidate"]: r for r in value["matched_time_coverage"]}
        self.assertEqual(len(value["matched_nature_final_time"]), 282)
        self.assertEqual(coverage["quiet-owl-45"]["games_higher_lower_endpoint"], 3)
        self.assertEqual(coverage["quiet-owl-45"]["games_lower_lower_endpoint"], 3)
        self.assertEqual(coverage["quality-reference"]["games_higher_lower_endpoint"], 4)
        self.assertEqual(max(r["games_higher_lower_endpoint"] for r in coverage.values()), 4)
        contribution = value["normalized_game_contributions"]
        self.assertAlmostEqual(sum(r["contribution_to_mean_gain_over_quality"] for r in contribution),
                               .422731105917145 - .35391309916541164)
        self.assertAlmostEqual(next(r for r in contribution if r["environment"] == "pongcnn")["fraction_of_net_gain"],
                               .7457033312263126)

    def test_duplicate_ties_are_not_ranked_as_different_effective_winners(self):
        for value in self.result["leave_one_game_out"]:
            self.assertEqual(value["ranks"]["quiet-owl-45"], 1)
        self.assertEqual(self.result["blind_spot_trials"], 4)


if __name__ == "__main__":
    unittest.main()
