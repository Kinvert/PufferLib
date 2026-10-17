"""Transported artifact/scalar tests. No policy math or native execution."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/analysis"))
spec = importlib.util.spec_from_file_location("continuation_exports", ROOT / "research/analysis/continuation_exports.py")
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
PACKET = ROOT / "research/results/learning-continuation-5090-20261009-prefix45"


class ContinuationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = tool.audit(PACKET)
        cls.parent = tool.read(PACKET / "parent-search-result.json")
        cls.allocation = tool.read(PACKET / "allocation-result.json")
        cls.replay = tool.read(PACKET / "last-native-proposal.json")
        cls.ledger = (PACKET / "completed-history.tsv").read_text()

    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in tool.FILES:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PACKET / name, root / name)
            yield root

    def rehash(self, root, name):
        receipt = tool.read(root / "analysis.json")
        if name in receipt["files_sha256"]:
            receipt["files_sha256"][name] = tool.sha(root / name)
        (root / "analysis.json").write_text(json.dumps(receipt))

    def test_actual_prefix_reconstructs_feedback_and_keeps_failed_allocation(self):
        result = self.result
        self.assertEqual((result["point_rows"], result["comparison_jobs"], result["training_jobs"]), (15416, 564, 594))
        self.assertEqual(result["completed_observations"], 45)
        self.assertEqual(result["effective_graphs"], 35)
        self.assertEqual(result["allocation_status"], "failed")
        self.assertFalse(result["raw_packet_audited"])
        self.assertEqual(len(result["aggregate_checkpoints"]), 188)
        self.assertTrue(all(d["all_point_scores_identical"] and d["all_checkpoint_hashes_identical"]
                            for d in result["duplicates"]))

    def test_history_cannot_discard_or_change_an_observation(self):
        allocation = copy.deepcopy(self.allocation)
        allocation["trials"][0]["feedback"]["score"] += .001
        with self.assertRaisesRegex(ValueError, "Changed history"):
            tool.check_history(self.parent, allocation, self.replay, self.ledger)

    def test_effective_graph_cannot_be_relabelled(self):
        allocation = copy.deepcopy(self.allocation)
        allocation["trials"][0]["architecture"]["cnn_kernel_1"] += 1
        with self.assertRaisesRegex(ValueError, "architecture disagrees"):
            tool.check_history(self.parent, allocation, self.replay, self.ledger)

    def test_failure_cannot_be_upgraded_to_success(self):
        allocation = copy.deepcopy(self.allocation); allocation["status"] = "ok"
        with self.assertRaisesRegex(ValueError, "relabeled"):
            tool.check_history(self.parent, allocation, self.replay, self.ledger)

    def test_rehashed_missing_point_is_still_incomplete(self):
        with self.fixture() as root:
            name = "review/combined/curves.json"
            curve = tool.read(root / name); curve["points"].pop()
            (root / name).write_text(json.dumps(curve)); self.rehash(root, name)
            with self.assertRaisesRegex(ValueError, "Point coverage incomplete"):
                tool.audit(root)

    def test_rehashed_doubled_cost_is_not_accepted(self):
        with self.fixture() as root:
            name = "exports/normalized-final-feedback.csv"
            values = tool.rows(root / name)
            values[0]["total_checkpoint_training_seconds"] = 2 * float(values[0]["total_checkpoint_training_seconds"])
            tool.write_csv(root / name, values); self.rehash(root, name)
            with self.assertRaisesRegex(ValueError, "Aggregate clipping/cost mismatch"):
                tool.audit(root)

    def test_rehashed_clipping_after_game_average_is_not_accepted(self):
        with self.fixture() as root:
            name = "exports/normalized-final-feedback.csv"
            values = tool.rows(root / name)
            row = next(r for r in values if r["candidate"] == "quiet-owl-45")
            # This tempting reconstruction uses already averaged game scores,
            # rather than clipping each seed/drawing point before averaging.
            finals = [r for r in tool.rows(root / "exports/per-environment-final.csv")
                      if r["candidate"] == "quiet-owl-45"]
            anchors = dict(connect4cnn=(0, 1), pongcnn=(0, 1), flappycnn=(0, 100),
                           breakoutcnn=(0, 20), snakecnn=(1, 20), mazecnn=(0, 1))
            wrong = sum(max(0, min(1, (float(r["score_lower"]) - anchors[r["environment"]][0])
                        / (anchors[r["environment"]][1] - anchors[r["environment"]][0]))) for r in finals) / 6
            self.assertNotAlmostEqual(wrong, float(row["equal_game_normalized_final_score"]))
            row["equal_game_normalized_final_score"] = wrong
            tool.write_csv(root / name, values); self.rehash(root, name)
            with self.assertRaisesRegex(ValueError, "Aggregate clipping/cost mismatch"):
                tool.audit(root)

    def test_original_fourteen_models_keep_all_previous_export_rows(self):
        previous = ROOT / "research/results/learning-feedback-5090-20261008T173307Z"
        old = tool.rows(previous / "per-environment-checkpoints.csv")
        new = tool.rows(PACKET / "exports/per-environment-checkpoints.csv")
        key = lambda r: (r["environment"], r["candidate"], r["decisions"])
        lookup = {key(r): r for r in new}
        self.assertEqual(len(old), 336)
        for row in old:
            self.assertEqual(row, lookup[key(row)])


if __name__ == "__main__":
    unittest.main()
