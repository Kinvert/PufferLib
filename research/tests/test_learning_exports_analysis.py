"""Scalar/artifact consistency tests; no model, GPU or native execution."""
import csv
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("learning_exports", ROOT/"research/analysis/learning_exports.py")
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
EXPORTS = ROOT/"research/results/learning-feedback-5090-20261008T173307Z"


class ExportTests(unittest.TestCase):
    def copy(self, directory):
        for name in tool.FILES: shutil.copyfile(EXPORTS/name,directory/name)

    def change(self, path, field, value):
        values=tool.rows(path); values[0][field]=value
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)

    def test_actual_exports_have_complete_cost_and_point_coverage(self):
        result=tool.analyze(EXPORTS)
        self.assertEqual(result["points"],336); self.assertEqual(result["training_jobs"],192)
        self.assertEqual(len(result["models"]),14)
        self.assertFalse(result["raw_packet_audited"])

    def test_wrong_sps_fails_without_rewriting_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.copy(root); self.change(root/"training-sps.csv","process_sps","1")
            with self.assertRaisesRegex(ValueError,"Wrong process SPS"): tool.analyze(root)

    def test_missing_checkpoint_cannot_form_complete_frontiers(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.copy(root); path=root/"checkpoint-sps.csv"
            lines=path.read_text().splitlines(); path.write_text("\n".join(lines[:-1])+"\n")
            with self.assertRaisesRegex(ValueError,"Incomplete checkpoint"): tool.analyze(root)

    def test_averaged_drawing_cost_cannot_be_counted_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.copy(root)
            self.change(root/"per-environment-checkpoints.csv","mean_checkpoint_seconds","999")
            with self.assertRaisesRegex(ValueError,"Mean checkpoint cost mismatch"): tool.analyze(root)

    def test_overlapping_censoring_bounds_do_not_establish_bound_dominance(self):
        fast=dict(seconds=1,score_lower=.5,score_upper=.6)
        uncertain=dict(seconds=2,score_lower=.4,score_upper=.7)
        self.assertTrue(tool.dominates(fast,uncertain))
        self.assertFalse(tool.dominates(fast,uncertain,"score_lower","score_upper"))


if __name__=="__main__": unittest.main()
