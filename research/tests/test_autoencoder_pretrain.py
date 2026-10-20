"""Host contracts only; no CPU neural computation, collection or GPU query."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import autoencoder_pretrain as ae
import backbone_assets as assets


def fixture(root):
    image = bytes(144 * 144 * 3 * 4)
    digest = hashlib.sha256(image).hexdigest()
    rows = []
    for split, seed in (('train', 7), ('validation', 420)):
        (root / f'{split}.f32').write_bytes(image)
        rows.append(dict(id=split, game='fixture', split=split, env_seed=seed,
            shape_hwc=[144, 144, 3], frames=1, path=f'{split}.f32', sha256=digest, frame_sha256=[digest]))
    value = dict(schema='native-game-frames-v1', dtype='float32', layout='NHWC', value_range=[0, 1], episodes=rows)
    ae.save(root / 'dataset.json', value)
    return value


class DataContracts(unittest.TestCase):
    def test_episode_separation_does_not_hide_duplicate_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            result = ae.dataset(root)
            self.assertEqual(result['exact_frame_overlap'], 1)
            self.assertFalse(result['generalization_claim'])

    def test_world_seed_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); value = fixture(root)
            value['episodes'][1]['env_seed'] = 7; ae.save(root / 'dataset.json', value)
            with self.assertRaisesRegex(RuntimeError, 'seed leaks'):
                ae.dataset(root)

    def test_modified_frame_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            path = root / 'train.f32'
            with path.open('r+b') as f:
                f.write(struct.pack('<f', 1))
            with self.assertRaisesRegex(RuntimeError, 'Changed frame shard'):
                ae.dataset(root)

    def test_missing_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); value = fixture(root)
            value['episodes'].pop(); ae.save(root / 'dataset.json', value)
            with self.assertRaisesRegex(RuntimeError, 'Missing stream/split'):
                ae.dataset(root)

    def test_paths_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); value = fixture(root)
            value['episodes'][0]['path'] = '../outside.f32'; ae.save(root / 'dataset.json', value)
            with self.assertRaisesRegex(RuntimeError, 'escapes root'):
                ae.dataset(root)


class StateContracts(unittest.TestCase):
    def test_export_rejects_changed_recipe_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); packet = root / 'packet'; run = root / 'run'
            packet.mkdir(); run.mkdir()
            ae.save(packet / 'sources.json', {})
            binary = packet / 'builds/flappycnn/autoencoder'
            binary.parent.mkdir(parents=True); binary.write_bytes(b'binary')
            recipe = run / 'recipe.ini'
            recipe.write_text('[autoencoder]\nsteps=8\nmax_seconds=60\nseed=7\nlearning_rate=.001\n')
            data = dict(manifest_sha256='data')
            contract = ae.training_contract(packet, recipe, data)
            ae.save(run / 'dataset-review.json', data)
            (run / 'index.tsv').write_text('index')
            ae.save(run / 'request.json', dict(contract=contract,
                recipe_sha256=ae.sha(recipe), index_sha256=ae.sha(run / 'index.tsv')))
            ae.save(run / 'REPORT.json', dict(contract=contract, checkpoints=[dict(path='missing.state')]))
            recipe.write_text(recipe.read_text().replace('.001', '.002'))
            with patch.object(ae, 'inspect'), self.assertRaisesRegex(RuntimeError, 'Export run/source/recipe'):
                ae.export(packet, run, 'stream', root / 'export')

    def test_optimizer_state_is_included_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state'
            count = 36776 + ae.DECODER_PARAMETERS
            raw = ae.STATE.pack(ae.MAGIC, 1, count, 8, 7) + bytes(count * 12)
            path.write_bytes(raw)
            self.assertEqual(ae.state(path)[0]['step'], 8)
            for offset in (0, count * 4, count * 8):
                corrupt = bytearray(raw)
                corrupt[ae.STATE.size + offset:ae.STATE.size + offset + 4] = struct.pack('<f', float('nan'))
                path.write_bytes(corrupt)
                with self.assertRaisesRegex(RuntimeError, 'Nonfinite checkpoint'):
                    ae.state(path)

    def test_truncated_or_extended_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state'; count = 36776 + ae.DECODER_PARAMETERS
            raw = ae.STATE.pack(ae.MAGIC, 1, count, 8, 7) + bytes(count * 12)
            for wrong in (raw[:-4], raw + bytes(4)):
                path.write_bytes(wrong)
                with self.assertRaisesRegex(RuntimeError, 'state layout'):
                    ae.state(path)

    def test_resume_extends_steps_but_cannot_change_learning_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ae.save(root / 'sources.json', {})
            binary = root / 'builds/flappycnn/autoencoder'; binary.parent.mkdir(parents=True); binary.write_bytes(b'binary')
            recipe = root / 'recipe.ini'
            recipe.write_text('[autoencoder]\nsteps=4\nmax_seconds=60\nseed=7\nlearning_rate=.001\n')
            data = dict(manifest_sha256='same-data')
            first = ae.training_contract(root, recipe, data)
            recipe.write_text('[autoencoder]\nsteps=8\nmax_seconds=120\nseed=7\nlearning_rate=.001\n')
            self.assertEqual(first, ae.training_contract(root, recipe, data))
            recipe.write_text('[autoencoder]\nsteps=8\nmax_seconds=120\nseed=7\nlearning_rate=.002\n')
            self.assertNotEqual(first, ae.training_contract(root, recipe, data))


class SafeWeights(unittest.TestCase):
    def test_integer_scalar_normalization_buffers_are_inventory_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.safetensors'
            header = json.dumps({'bn.num_batches_tracked': dict(dtype='I64', shape=[], data_offsets=[0, 8])}).encode()
            p.write_bytes(struct.pack('<Q', len(header)) + header + bytes(8))
            result = assets.inventory(p)
            self.assertEqual(result['stored_elements'], 1)
            self.assertEqual(result['encoder_parameters'], 0)
            self.assertFalse(result['execution'])

    def test_safetensors_inventory_is_data_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.safetensors'
            header = json.dumps({'encoder.test': dict(dtype='F32', shape=[2, 2], data_offsets=[0, 16])}).encode()
            p.write_bytes(struct.pack('<Q', len(header)) + header + bytes(16))
            result = assets.inventory(p)
            self.assertEqual(result['encoder_parameters'], 4)
            self.assertFalse(result['execution'])
            self.assertFalse(result['native_conversion'])

    def test_safetensors_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'model.safetensors'
            header = json.dumps({name: dict(dtype='F32', shape=[4], data_offsets=[0, 16])
                for name in ('encoder.a', 'encoder.b')}).encode()
            p.write_bytes(struct.pack('<Q', len(header)) + header + bytes(16))
            with self.assertRaisesRegex(RuntimeError, 'overlap/gap'):
                assets.inventory(p)


if __name__ == '__main__':
    unittest.main()
