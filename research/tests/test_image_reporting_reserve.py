"""Scalar measured-clock fixtures; no GPU, training or neural computation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import image_reporting_reserve as tool


class ReportingReserveTests(unittest.TestCase):
    def setUp(self):
        self.value=dict(schema="realistic-image-campaign-budget-v1",empirical_estimate=True,
            hard_runtime_guarantee=False,publication_claim_qualified=False,safety_factor=2.,
            measured_panel_seconds=[100.,120.],next_panel_seconds=240.,
            measured_final_replay_seconds=2.,measured_report_seconds=10.,post_panel_seconds=24.)

    def test_larger_history_scales_report_and_smaller_history_keeps_pilot_floor(self):
        self.assertEqual([tool.seconds(self.value,n) for n in (0,1,2,3,6)], [24.,24.,24.,34.,64.])

    def test_bad_counts_or_modified_measured_reserve_are_refused(self):
        for count in (-1,True,1.5,float("nan")):
            with self.subTest(count=count),self.assertRaises(ValueError):tool.seconds(self.value,count)
        changed=copy.deepcopy(self.value);changed["measured_report_seconds"]=.01
        with self.assertRaises(ValueError):tool.seconds(changed,3)

    def test_observed_slower_optimizer_reserves_final_replay_without_shrinking(self):
        self.assertEqual(tool.seconds(self.value,3,[5]),40.)
        self.assertEqual(tool.seconds(self.value,3,[.1]),34.)
        for values in ([0],[float("nan")],[True]):
            with self.assertRaises(ValueError): tool.seconds(self.value,3,values)


if __name__=="__main__":unittest.main()
