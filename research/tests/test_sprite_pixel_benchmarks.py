"""Actual prepared artifacts and independent scalar counts; no neural execution."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from research import sprite_pixel_benchmarks as benchmark


class SpritePixelPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = Path(os.environ.get('SPRITE_PIXEL_PACKET',
            benchmark.ROOT / 'build/sprite-packed-preparation-20261016-v3'))
        cls.plan = json.loads((cls.packet / 'plan.json').read_text())

    def test_actual_packet(self):
        result = benchmark.inspect(self.packet)
        self.assertEqual(result['jobs'], 30)
        self.assertEqual(result['assigned_world_pixel_manifests'], 36)
        self.assertFalse(result['gpu_execution'])

    def test_independent_counts(self):
        hidden = 128
        conv = 32 * (8 * 8 * 3 + 1) + 64 * (4 * 4 * 32 + 1) + 64 * (3 * 3 * 64 + 1)
        height, width = 192, 96
        for kernel, stride in ((8, 4), (4, 2), (3, 1)):
            height = (height - kernel) // stride + 1
            width = (width - kernel) // stride + 1
        adapter = hidden * (hidden + 4 + 1)
        residual = 0
        previous = 3
        for channels in (16, 32, 32):
            residual += channels * (9 * previous + 1) + 4 * channels * (9 * channels + 1)
            previous = channels
        encoders = dict(tile=14120 + hidden * (18 * 16 + 4 + 1),
            nature=conv + hidden * (height * width * 64 + 1) + adapter,
            impala=residual + hidden * (24 * 12 * 32 + 1) + adapter,
            impoola=residual + hidden * (32 + 1) + adapter,
            flex=8 * (8 * 8 * 3 + 1) + 32 * (48 * 24 * 8 + 1)
                + hidden * (32 + 1) + adapter)
        for job in self.plan['jobs']:
            meta = job['metadata']
            self.assertEqual(meta['encoder_parameters'], encoders[job['family']])
            core = 3 * hidden * hidden * meta['layers']
            decoder = (sum(meta['action_sizes']) + 1) * hidden
            self.assertEqual(meta['parameters'], encoders[job['family']] + core + decoder)

    def damaged_plan(self, change):
        with tempfile.TemporaryDirectory() as name:
            packet = Path(name) / 'packet'
            shutil.copytree(self.packet, packet)
            plan = copy.deepcopy(self.plan)
            change(plan)
            (packet / 'plan.json').write_text(json.dumps(plan))
            with self.assertRaises(AssertionError):
                benchmark.inspect(packet)

    def test_missing_job_rejected(self):
        self.damaged_plan(lambda p: p['jobs'].pop())

    def test_seed_search_rejected(self):
        self.damaged_plan(lambda p: p.update(seeds_swept=True))

    def test_wrong_metadata_rejected(self):
        self.damaged_plan(lambda p: p['jobs'][0]['metadata'].update(parameters=1))

    def test_wrong_schedule_rejected(self):
        self.damaged_plan(lambda p: p.update(checkpoints=[p['decisions']]))

    def test_sweep_axes_and_seeds(self):
        self.assertEqual(benchmark.TRAINING_SEEDS, (7, 42, 69, 123, 173, 2027, 9001))
        self.assertTrue(set(benchmark.TRAINING_SEEDS).isdisjoint(benchmark.EVAL_SEEDS))
        recipe = benchmark.read_config(benchmark.RECIPE)
        axes = [s for s in recipe.sections() if s.startswith('sweep.')]
        self.assertEqual(len(axes), 18)
        self.assertTrue(all(s.startswith('sweep.policy.cnn_') for s in axes))


if __name__ == '__main__':
    unittest.main()
