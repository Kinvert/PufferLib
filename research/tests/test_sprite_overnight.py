"""Scalar, receipt and supervision tests only. No neural computation/GPU queries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sprite_overnight as night


def jobs(cfg):
    return [dict(game=g, seed=s, training=dict(process=dict(seconds=10)),
        evaluations=[dict(decisions=cfg['decisions'], skin=k, perf=(i + 1) / 10)
            for k in cfg['skins']]) for i, g in enumerate(night.GAMES) for s in cfg['seeds']]


class SpriteOvernightTests(unittest.TestCase):
    def test_fixed_seven_replicas(self):
        cfg = night.settings(night.RECIPE, 'overnight')
        self.assertEqual(cfg['seeds'], [7, 42, 69, 123, 173, 2027, 9001])
        self.assertEqual(cfg['seconds'], 43200)
        self.assertEqual(cfg['checkpoint_count'], 8)
        self.assertEqual(cfg['baselines'], ['tile', 'nature', 'impala', 'impoola'])

    def test_bounded_smoke(self):
        cfg = night.settings(night.RECIPE, 'smoke')
        self.assertEqual((cfg['seconds'], cfg['trials'], cfg['decisions']), (900, 2, 8192))
        self.assertEqual(cfg['seeds'], [7])

    def test_equal_games_charge_training_once(self):
        cfg = night.settings(night.RECIPE, 'overnight')
        feedback = night.aggregate(jobs(cfg), cfg)
        self.assertAlmostEqual(feedback['score'], .35)
        self.assertEqual(feedback['cost'], 420)
        self.assertEqual(feedback['training_jobs_charged'], 42)
        self.assertEqual(feedback['worst_game_utility'], .1)

    def test_missing_or_duplicate_replica_rejected(self):
        cfg = night.settings(night.RECIPE, 'overnight')
        data = jobs(cfg)
        for changed in (data[:-1], data + [copy.deepcopy(data[0])]):
            with self.assertRaises(ValueError):
                night.aggregate(changed, cfg)

    def test_missing_skin_rejected(self):
        cfg = night.settings(night.RECIPE, 'overnight')
        data = jobs(cfg)
        data[0]['evaluations'].pop()
        with self.assertRaises(ValueError):
            night.aggregate(data, cfg)

    def test_history_keeps_duplicate_and_zero_score(self):
        trial = dict(feedback=dict(score=0, cost=2.5), proposal=dict(normalized=[0] * 18))
        history = night.history_text([trial, copy.deepcopy(trial)])
        self.assertEqual(len(history.splitlines()), 3)
        self.assertEqual(history.splitlines()[1], history.splitlines()[2])
        self.assertTrue(history.endswith('\n'))

    def test_seed_sweep_recipe_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            file = Path(name) / 'recipe.ini'
            file.write_text(night.RECIPE.read_text().replace('7,42,69,123,173,2027,9001', '7,42'))
            with self.assertRaises(ValueError):
                night.settings(file, 'overnight')

    def test_process_failure_keeps_receipt_and_refuses_retry(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            executor = object.__new__(night.Executor)
            executor.out = out
            executor.settings = night.settings(night.RECIPE, 'smoke')
            executor.deadline = time.monotonic() + 10
            executor.hardware = 'synthetic-host-test'
            executor.processes = []
            executor.idle = lambda: None
            with self.assertRaises(ValueError):
                executor.process('failed', ['/bin/sh', '-c', 'printf retained; exit 7'], out, 2)
            receipt = night.read(out / 'processes/failed.json')
            self.assertEqual(receipt['returncode'], 7)
            self.assertEqual((out / receipt['log']).read_text(), 'retained')
            with self.assertRaises(ValueError):
                executor.process('failed', ['/bin/true'], out, 2)

    def test_timeout_kills_owned_process_group(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            executor = object.__new__(night.Executor)
            executor.out = out
            executor.settings = night.settings(night.RECIPE, 'smoke')
            executor.deadline = time.monotonic() + 10
            executor.hardware = 'synthetic-host-test'
            executor.processes = []
            executor.idle = lambda: None
            with self.assertRaises(ValueError):
                executor.process('timeout', ['/bin/sh', '-c', 'sleep 5'], out, .05)
            receipt = night.read(out / 'processes/timeout.json')
            self.assertTrue(receipt['timed_out'])
            self.assertNotEqual(receipt['returncode'], 0)

    def test_all_missing_baselines_remain_visible(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            cfg = night.settings(night.RECIPE, 'overnight')
            night.save(out / 'plan.json', dict(settings=cfg))
            night.save(out / 'result.json', dict(status='failed'))
            night.report(out)
            report = night.read(out / 'report.json')
            self.assertEqual(report['complete_curve_points'], 0)
            self.assertEqual(report['missing_curve_points'], 4 * 6 * 3 * 8)
            page = (out / 'curves.html').read_text()
            self.assertIn('impala', page)
            self.assertIn('nature', page)
            self.assertIn('missing', page)


if __name__ == '__main__':
    unittest.main()
