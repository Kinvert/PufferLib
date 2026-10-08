"""Graph metadata and audit guards only; no model/GPU execution."""
import ctypes
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cnn_graph_acceptance as tool


class GraphTests(unittest.TestCase):
    def test_independent_scalar_nature_and_quality_counts(self):
        config=tool.panels.panel.read(tool.ROOT/"research/recipes/panel_smoke_quality.ini")
        model,settings=tool.settings(config); rows,n=tool.topology(model,settings)
        self.assertEqual(n,110560); self.assertEqual(len(rows),3)
        rows,n=tool.topology(1,settings); self.assertEqual(n,88352); self.assertEqual(len(rows),4)

    def test_three_stage_plan_covers_residual_max_average_and_global(self):
        cfg=tool.panels.panel.read(tool.ROOT/"research/recipes/panel_prepare_three_stage.ini")
        rows,n=tool.topology(*tool.settings(cfg))
        self.assertGreater(n,0); self.assertEqual({r[11] for r in rows},{0,1,2,3})
        self.assertEqual(sum(r[12] for r in rows),2)

    def test_case_matrix_includes_actual_learner_batch_and_zero_branch(self):
        self.assertEqual(len(tool.cases()),7)
        self.assertIn((2048,"signed"),tool.cases()); self.assertIn((1,"zero"),tool.cases())
        self.assertEqual(len(tool.MODES),4)

    def test_unsupported_graphs_fail_metadata_before_library_load(self):
        with patch.object(ctypes,"CDLL",side_effect=AssertionError("No load")):
            for encoder in (0,1,3,5):
                cfg=tool.panels.panel.read(); cfg["policy"]={"encoder":str(encoder),"hidden_size":"128"}
                with self.assertRaises(ValueError): tool.settings(cfg)

    def test_missing_report_cannot_audit_as_pass(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(ctypes,"CDLL",side_effect=AssertionError("No load")):
            with self.assertRaises(FileNotFoundError): tool.audit(Path(temp))

    def test_gate_timeout_is_bounded_before_output_allocation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(tool,"inspect_bundle"):
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,"deadline"): tool.run(root,root/"missing",root/"out",timeout=121)
            self.assertFalse((root/"out").exists())


if __name__=="__main__": unittest.main()
