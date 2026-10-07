"""Host-only scheduling boundaries; no policy, library or GPU execution."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_panel as panel


class MiniBounds(unittest.TestCase):
    def setUp(self):
        self.value = dict(candidates=[dict(name="happy-cat-1", architecture=panel.architecture(
            panel.ROOT / "research/recipes/panel_smoke_quality.ini")),
            dict(panel.BASELINES["nature_cnn"], kind="baseline", baseline="nature_cnn")],
            seeds=[59173, 59174], task_budgets={t:dict(steps=524288) for t in panel.TASKS},
            jobs=[dict(checkpoint_steps=[262144,524288]) for _ in range(24)],
            planned_evaluations=328, episodes=17, slots=16)

    def test_paired_short_quality_nature_routes_to_5060(self):
        self.assertEqual(panel.run_hardware(self.value, "mini"), "5060")
        self.assertEqual(panel.run_hardware(self.value, "development"), "5090")

    def test_mini_rejects_long_extra_seed_or_large_evaluation(self):
        for change in (dict(seeds=[1,2,3]), dict(episodes=34), dict(slots=33),
                       dict(planned_evaluations=657), dict(jobs=self.value["jobs"]*2),
                       dict(task_budgets={"connect4cnn":dict(steps=2097152)}),
                       dict(jobs=[dict(checkpoint_steps=list(range(5)))])):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "bounded short"):
                panel.run_hardware({**self.value, **change}, "mini")

    def test_mini_rejects_slow_baseline_or_new_shape(self):
        for baseline in ("impala_cnn", "impoola_cnn"):
            value=copy.deepcopy(self.value); value["candidates"][1]["baseline"]=baseline
            with self.assertRaisesRegex(ValueError, "Nature only"):
                panel.run_hardware(value, "mini")
        value=copy.deepcopy(self.value); value["candidates"][0]["architecture"]["cnn_channels_1"]=8
        with self.assertRaisesRegex(ValueError, "frozen quality"):
            panel.run_hardware(value, "mini")

    def test_legacy_smoke_and_unknown_mode_remain_restricted(self):
        with self.assertRaisesRegex(ValueError, "Smoke exceeds"):
            panel.run_hardware(self.value, "smoke")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            panel.run_hardware(self.value, "whatever")

    def test_single_candidate_research_has_bounded_local_route(self):
        value=copy.deepcopy(self.value);value["candidates"]=value["candidates"][:1]
        self.assertEqual(panel.run_hardware(value,"research"),"5060")
        for change in (dict(episodes=34),dict(slots=33),dict(planned_evaluations=329),
                       dict(seeds=[1,2,3]),dict(candidates=self.value["candidates"])):
            with self.assertRaisesRegex(ValueError,"bounded single"):
                panel.run_hardware({**value,**change},"research")

    def test_remote_canary_is_small_and_routes_only_to_5090(self):
        value=copy.deepcopy(self.value)
        value.update(candidates=value["candidates"][:1], seeds=[60173], episodes=17,
            slots=16, planned_evaluations=41,
            task_budgets={t:dict(steps=65536) for t in panel.TASKS},
            jobs=[dict(checkpoint_steps=[65536]) for _ in range(6)])
        self.assertEqual(panel.run_hardware(value,"canary5090"),"5090")
        for change in (dict(seeds=[1,2]),dict(episodes=18),dict(slots=17),
                dict(planned_evaluations=42),dict(jobs=value["jobs"]*2),
                dict(jobs=[dict(checkpoint_steps=[32768,65536]) for _ in range(6)]),
                dict(task_budgets={t:dict(steps=262144) for t in panel.TASKS}),
                dict(candidates=[self.value["candidates"][1]])):
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,"5090 canary"):
                panel.run_hardware({**value,**change},"canary5090")


if __name__ == "__main__":
    unittest.main()
