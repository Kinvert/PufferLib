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

    def test_partition_preserves_two_processes_per_case_and_every_native_call(self):
        contract=dict(schema=tool.IMAGE_SCHEMA,batches=[1,4,16],worker_partition="per-batch-v1")
        groups=tool.worker_groups(contract)
        self.assertEqual(len(groups),6)
        self.assertEqual(sum(map(len,groups)),18)
        self.assertEqual(groups[0],groups[1]); self.assertEqual(groups[2],groups[3]); self.assertEqual(groups[4],groups[5])
        from collections import Counter
        self.assertEqual(Counter(case for group in groups for case in group),Counter({case:2 for case in tool.cases(contract)}))
        self.assertEqual(tool.worker_groups({}),[tool.cases(),tool.cases()])

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

    def test_each_worker_uses_remaining_outer_deadline(self):
        with patch.object(tool.time,"monotonic",side_effect=[100.,117.,120.]):
            self.assertEqual(tool.worker_timeout(120,120),20)
            self.assertEqual(tool.worker_timeout(120,120),3)
            with self.assertRaisesRegex(ValueError,"Campaign deadline"):
                tool.worker_timeout(120,120)
        self.assertEqual(tool.worker_timeout(30),30)
        for timeout in (float("nan"),float("inf"),0,121):
            with self.assertRaises(ValueError): tool.worker_timeout(timeout)

    def test_compact_exports_keep_all_paths_and_refuse_any_different_mode_byte(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); value=np.array([1.,-2.,3.],dtype=np.float32)
            tool.export_mode(root,"eager-0",[("gradient",value)],True)
            tool.export_mode(root,"graph-0",[("gradient",value.copy())],True)
            self.assertEqual((root/"eager-0-gradient.f32").stat().st_ino,(root/"graph-0-gradient.f32").stat().st_ino)
            self.assertEqual(tool.array_hashes(root),tool.array_hashes(root,True))
            changed=value.copy();changed[0]=4.
            with self.assertRaisesRegex(ValueError,"actual current native bytes"):
                tool.export_mode(root,"graph-1",[("gradient",changed)],True)
            self.assertFalse((root/"graph-1-gradient.f32").exists())
            before=tool.array_hashes(root,True)
            (root/"graph-0-gradient.f32").write_bytes(changed.tobytes())
            after=tool.array_hashes(root,True)
            self.assertNotEqual(before,after)
            self.assertEqual(after["eager-0-gradient.f32"],after["graph-0-gradient.f32"])
            self.assertEqual(after,tool.array_hashes(root))


if __name__=="__main__": unittest.main()
