"""Host artifact rejection tests; no CPU neural execution or GPU discovery."""
import array
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
import fullframe_compression as ffc


class Contracts(unittest.TestCase):
    def test_timing_cannot_bypass_incomplete_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ffc.save(out / 'stages/math.json', dict(status='passed'))
            ffc.save(out / 'stages/smoke.json', dict(status='incomplete'))
            with patch.object(ffc, 'inspect'), patch.object(ffc.GPU, '__enter__') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Timing requires accepted'):
                    ffc.timing_stage(out, '5060')
                launch.assert_not_called()

    def test_timing_refuses_mutated_captured_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ffc.save(out / 'sources.json', {})
            folder = out / 'timing-build'
            folder.mkdir()
            (folder / 'fullframe_compression_timing.cu').write_text('original')
            (folder / 'supervisor.py').write_text('supervisor')
            ffc.save(out / 'timing-prepared.json', dict(schema='fullframe-compression-timing-build-v1',
                native_source_manifest_sha256=ffc.sha(out / 'sources.json'),
                source_sha256=ffc.sha(folder / 'fullframe_compression_timing.cu'),
                supervisor_sha256=ffc.sha(folder / 'supervisor.py'), builds={g: 'unused' for g in ffc.GAMES}))
            (folder / 'fullframe_compression_timing.cu').write_text('changed!')
            with self.assertRaisesRegex(RuntimeError, 'Changed timing build provenance'):
                ffc.inspect_timing(out)

    def test_changed_export_cannot_keep_its_old_import_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export = out / 'exports/flappycnn-0'
            export.mkdir(parents=True)
            (export / 'encoder.bin').write_bytes(b'old!')
            value = dict(sha256=ffc.sha(export / 'encoder.bin'))
            ffc.save(export / 'encoder.json', value)
            (export / 'encoder.bin').write_bytes(b'new!')
            with self.assertRaisesRegex(RuntimeError, 'Encoder export changed'):
                ffc.audit_import(out, dict(model=0, export=value))

    def test_expired_stage_cannot_launch_a_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = ffc.GPU(Path(tmp), '5060', 'smoke', -1)
            with patch.object(gpu, 'pids', return_value=[]), patch.object(ffc, 'process') as launch:
                with self.assertRaisesRegex(RuntimeError, 'Stage wall-clock cap reached'):
                    gpu.run(['not-a-real-policy'], Path(tmp) / 'never.log')
                launch.assert_not_called()
            self.assertFalse((Path(tmp) / 'never.log').exists())

    def test_native_allocation_at_limit_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            recipe = out / 'configs/flappycnn-0.ini'
            recipe.parent.mkdir()
            recipe.write_text('[policy]\nencoder=8\n')
            layout = dict(encoder_parameters=36776, image_contract=dict(shape_hwc=[640, 420, 3]),
                compression=dict(model=0), discrete_actions=2, raw_checkpoint_contiguous=True)
            storage = dict(learner_batch=16, storage_floor_bytes=4 * 2**30)
            ffc.save(out / 'layouts/flappycnn-0.json', dict(config_sha256=ffc.sha(recipe),
                layout=layout, storage=storage))
            for prefix, value in (('layout', layout), ('storage', storage)):
                log = out / f'builds/flappycnn/{prefix}-0.log'
                ffc.save(log, value)
                ffc.save(log.with_suffix('.process.json'), dict(returncode=0, timeout=False,
                    interference=[], log_sha256=ffc.sha(log)))
            with self.assertRaisesRegex(RuntimeError, 'Registration leaves fixed packet contract'):
                ffc.inspect_registration(out, 'flappycnn', 0)

    def test_changed_registration_is_not_blessed_by_its_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            recipe = out / 'configs/flappycnn-0.ini'
            recipe.parent.mkdir()
            recipe.write_text('[policy]\nencoder=8\n')
            ffc.save(out / 'layouts/flappycnn-0.json', dict(config_sha256=ffc.sha(recipe),
                layout=dict(encoder_parameters=36776), storage={}))
            log = out / 'builds/flappycnn/layout-0.log'
            ffc.save(log, dict(encoder_parameters=36608))
            ffc.save(log.with_suffix('.process.json'), dict(returncode=0, timeout=False,
                interference=[], log_sha256=ffc.sha(log)))
            with self.assertRaisesRegex(RuntimeError, 'Registration differs'):
                ffc.inspect_registration(out, 'flappycnn', 0)

    def test_native_bin_means_are_not_final_training_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.ini'
            path.write_text('[metrics]\nuptime=1,3\nagent_steps=1024,6144\nSPS=2000,2100\n')
            value = ffc.native_metrics(path, 4)
            self.assertIsNone(value['native_average_sps'])
            self.assertEqual(value['series']['agent_steps'], [1024, 6144])
            self.assertTrue(value['binned_uptime_within_process'])
            self.assertFalse(ffc.native_metrics(path, 2)['binned_uptime_within_process'])

    def test_busy_admission_creates_no_numerical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ffc.save(out / 'host-check.json', dict(status='passed'))
            with patch.object(ffc.GPU, '__enter__', side_effect=RuntimeError('Shared GPU busy')):
                with self.assertRaisesRegex(RuntimeError, 'Shared GPU busy'):
                    ffc.math_stage(out, '5060')
            self.assertFalse((out / 'math').exists())

    def test_seed_and_learner_are_fixed_across_models(self):
        for game in ffc.GAMES:
            recipes = [ffc.common_config(game, m) for m in range(3)]
            for cfg in recipes:
                self.assertEqual(cfg.getint('base', 'seed'), 7)
                self.assertFalse(any(s.startswith('sweep.') for s in cfg.sections()))
            for section in ('base', 'train', 'vec', 'env'):
                self.assertEqual(dict(recipes[0][section]), dict(recipes[1][section]))
                self.assertEqual(dict(recipes[0][section]), dict(recipes[2][section]))

    def test_evaluator_rejects_missing_and_duplicate_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.csv'
            for text in ('episode_id,perf\n0,0\n', 'episode_id,perf\n0,0\n0,0\n'):
                p.write_text(text)
                with self.assertRaises(RuntimeError):
                    ffc.episode_rows(p)

    def test_slot_order_is_not_a_different_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.csv'
            p.write_text('episode_id,slot,perf\n1,1,0x1p-1\n0,0,0x0p+0\n')
            one = ffc.episode_rows(p)
            p.write_text('episode_id,slot,perf\n0,0,0x0p+0\n1,0,0x1p-1\n')
            self.assertEqual(one, ffc.episode_rows(p))

    def test_nonfinite_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.csv'
            p.write_text('episode_id,perf\n0,nan\n1,0\n')
            with self.assertRaises(RuntimeError):
                ffc.episode_rows(p)

    def test_failed_process_keeps_negative_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'failure.log'
            with self.assertRaises(RuntimeError):
                ffc.process([sys.executable, '-c', 'raise SystemExit(3)'], tmp, log, 5)
            proof = ffc.read(log.with_suffix('.process.json'))
            self.assertEqual(proof['returncode'], 3)
            self.assertEqual(proof['log_sha256'], ffc.sha(log))
            with self.assertRaises(RuntimeError):
                ffc.process([sys.executable, '-c', 'pass'], tmp, log, 5)

    def test_expected_rejection_cannot_accept_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                ffc.process([sys.executable, '-c', 'pass'], tmp, Path(tmp) / 'reject.log', 5, expect_failure=True)

    def test_timeout_keeps_terminal_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'timeout.log'
            with self.assertRaises(RuntimeError):
                ffc.process([sys.executable, '-c', 'import time; time.sleep(60)'], tmp, log, .1)
            self.assertTrue(ffc.read(log.with_suffix('.process.json'))['timeout'])

    def test_equal_size_checkpoint_with_wrong_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            checkpoint = out / 'weights.bin'
            checkpoint.write_bytes(array.array('f', [0] * 4).tobytes())
            ffc.save(out / 'sources.json', {})
            ffc.save(out / 'plan.json', {})
            layout = dict(parameter_payload_bytes=16)
            ffc.save(out / 'layouts/flappycnn-0.json', dict(layout=layout))
            value = dict(schema='fullframe-compression-checkpoint-v1', game='pongcnn', model=0,
                sha256=ffc.sha(checkpoint), bytes=16, layout=layout,
                source_manifest_sha256=ffc.sha(out / 'sources.json'), plan_sha256=ffc.sha(out / 'plan.json'))
            ffc.save(checkpoint.with_suffix('.manifest.json'), value)
            with self.assertRaises(RuntimeError):
                ffc.validate_checkpoint(out, 'flappycnn', 0, checkpoint)

    def test_export_uses_registered_offsets_instead_of_a_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            checkpoint = out / 'weights.bin'
            checkpoint.write_bytes(array.array('f', range(20)).tobytes())
            ffc.save(out / 'sources.json', {})
            ffc.save(out / 'plan.json', {})
            (out / 'recipe.ini').write_text('[policy]\nencoder=8\n')
            tensors = [dict(group='encoder', offset_bytes=i*8, elements=1, shape=[1]) for i in range(8)]
            layout = dict(parameter_payload_bytes=80, encoder_parameters=8,
                parameter_tensors=tensors, compression=dict(layers=[]))
            ffc.save(out / 'layouts/flappycnn-0.json', dict(layout=layout))
            value = dict(schema='fullframe-compression-checkpoint-v1', game='flappycnn', model=0,
                sha256=ffc.sha(checkpoint), bytes=80, layout=layout, initialization='fixture',
                config_path='recipe.ini', config_sha256=ffc.sha(out / 'recipe.ini'),
                source_manifest_sha256=ffc.sha(out / 'sources.json'), plan_sha256=ffc.sha(out / 'plan.json'))
            ffc.save(checkpoint.with_suffix('.manifest.json'), value)
            directory = out / 'export'
            ffc.export_encoder(out, 'flappycnn', 0, checkpoint, directory)
            actual = array.array('f')
            actual.frombytes((directory / 'encoder.bin').read_bytes())
            self.assertEqual(list(actual), list(range(0, 16, 2)))
            self.assertEqual(ffc.read(directory / 'encoder.json')['sha256'], ffc.sha(directory / 'encoder.bin'))
            with self.assertRaises(RuntimeError):
                ffc.export_encoder(out, 'flappycnn', 0, checkpoint, directory)
            (out / 'recipe.ini').write_text('[policy]\nencoder=2\n')
            with self.assertRaises(RuntimeError):
                ffc.export_encoder(out, 'flappycnn', 0, checkpoint, out / 'wrong-recipe')


if __name__ == '__main__':
    unittest.main()
