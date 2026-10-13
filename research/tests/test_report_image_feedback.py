"""Scalar/artifact export checks; no GPU or neural reference."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import report_image_feedback as tool


class ImageReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        metric = Path(self.temp.name)/"native.ini"
        metric.write_text("[metrics]\nagent_steps=64,512\nSPS=100,200\nuptime=0.2,3.0\n")
        self.plan = dict(protocol=tool.panels.IMAGE_CURVE_PROTOCOL,
            jobs=[dict(environment="flappycnn",candidate="happy-cat-1",seed=7,checkpoint_steps=[0,64,512])],
            task_budgets={"flappycnn":dict(steps=512)})
        self.analysis = dict(status="ok",audited_jobs=1,missing_evaluations=0,
            training=[dict(environment="flappycnn",candidate="happy-cat-1",seed=7,
                parameters=123,train_seconds=2.,process_sps=256.,native_metrics_ini=str(metric))])
        self.execution = dict(jobs=[dict(job_index=0,process_seconds=2.,process_sps=256.,parameters=123,
            checkpoint_seconds={"0":0.1,"64":0.5,"512":1.5},
            checkpoints={str(s):dict(sha256="a"*64) for s in (0,64,512)})])

    def export(self):
        return tool.panel_exports("calibration",self.plan,self.analysis,self.execution)

    def test_initial_cost_is_retained_without_invented_initial_training_sps(self):
        training,checkpoints,native = self.export()
        self.assertEqual(training[0]["process_sps"],256)
        self.assertIsNone(training[0]["native_avg_sps"])
        self.assertEqual(checkpoints[0]["checkpoint_seconds"],0.1)
        self.assertIsNone(checkpoints[0]["cumulative_checkpoint_sps"])
        self.assertIsNone(checkpoints[0]["interval_checkpoint_sps"])
        self.assertEqual(checkpoints[1]["interval_decisions"],64)
        self.assertEqual(checkpoints[1]["interval_checkpoint_sps"],160)
        self.assertEqual(checkpoints[2]["interval_checkpoint_sps"],448)
        self.assertEqual(native[1]["native_bin_mean_sps"],200)
        self.assertEqual(native[1]["native_bin_mean_uptime"],3.)
        self.assertTrue(native[1]["uptime_exceeds_process_seconds"])

    def test_wrong_process_times_sps_layout_or_duplicate_jobs_are_rejected(self):
        original = copy.deepcopy(self.execution)
        for key,value in (("process_seconds",3.),("process_sps",257.),("parameters",124),
                          ("job_index",True)):
            self.execution = copy.deepcopy(original); self.execution["jobs"][0][key]=value
            with self.assertRaises(ValueError): self.export()
        self.execution=copy.deepcopy(original); self.execution["jobs"]*=2
        with self.assertRaises(ValueError): self.export()

    def test_nonmonotonic_costs_and_malformed_native_bins_are_rejected(self):
        self.execution["jobs"][0]["checkpoint_seconds"]["512"]=0.3
        with self.assertRaises(ValueError): self.export()
        self.execution["jobs"][0]["checkpoint_seconds"]["512"]=1.5
        metric=Path(self.analysis["training"][0]["native_metrics_ini"])
        for content in ("[metrics]\nagent_steps=64,512\nSPS=100\nuptime=0.2,3\n",
                        "[metrics]\nagent_steps=64\nSPS=nan\nuptime=0.2\n"):
            metric.write_text(content)
            with self.assertRaises(ValueError): self.export()


if __name__ == "__main__": unittest.main()
