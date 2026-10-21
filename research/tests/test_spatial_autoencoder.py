"""Artifact and scalar-report contracts; no CPU neural network or GPU query."""
import csv
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_autoencoder as sc


class SpatialContracts(unittest.TestCase):
    def test_native_bank_sweep_has_required_fields_and_only_architecture_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); packet=root/'packet'; bank=root/'bank'
            cfg=sc.f.config(sc.ROOT/'config/default.ini',sc.ROOT/'config/sprite_forage.ini')
            sc.f.write_config(packet/'configs/sprite_forage.ini',cfg)
            sc.save(bank/'bank.json',dict(variants=[dict(stride=8,channels=8),dict(stride=16,channels=16)]))
            result=sc.bank_config(packet,bank,root/'policy')
            self.assertEqual([s for s in result.sections() if s.startswith('sweep.')],['sweep.policy.compression_variant'])
            for key in ('metric','metric_distribution','goal','max_runs','gpus','max_suggestion_cost','downsample','prune_pareto','early_stop_quantile'):
                self.assertIn(key,result['sweep'])
            self.assertEqual(dict(result['train']),dict(cfg['train'],total_timesteps='8192'))
            self.assertEqual(result['base']['seed'],'7')

    def test_candidate_grid_rejects_invalid_or_duplicate_shapes(self):
        self.assertEqual(sc.variants('8:4,16:8,32:32'), [(8, 4), (16, 8), (32, 32)])
        for value in ('16:8,16:8', '4:8', '16:3'):
            with self.assertRaises(RuntimeError): sc.variants(value)

    def test_vector_state_is_not_silently_treated_as_spatial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'
            count = sum(sc.counts(8))
            path.write_bytes(sc.ae.STATE.pack(sc.ae.MAGIC, 1, count, 4, 7) + bytes(count * 12))
            with self.assertRaisesRegex(RuntimeError, 'header'): sc.state(path, 8)

    def test_state_binds_width_optimizer_and_finiteness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'; count = sum(sc.counts(8))
            raw = sc.ae.STATE.pack(sc.ae.MAGIC, 2, count, 4, 7) + bytes(count * 12)
            path.write_bytes(raw); self.assertEqual(sc.state(path, 8)[0]['step'], 4)
            with self.assertRaisesRegex(RuntimeError, 'header'): sc.state(path, 16)
            for offset in (0, count * 4, count * 8):
                damaged = bytearray(raw); struct.pack_into('<f', damaged, sc.ae.STATE.size + offset, float('nan'))
                path.write_bytes(damaged)
                with self.assertRaisesRegex(RuntimeError, 'Nonfinite'): sc.state(path, 8)
            path.write_bytes(raw + b'extra')
            with self.assertRaisesRegex(RuntimeError, 'header'): sc.state(path, 8)

    def fixture(self, root):
        regions, features = root / 'regions.csv', root / 'features.csv'
        with regions.open('w') as f:
            f.write('split,frame,skin,cell,kind,foreground_sum,foreground_count,background_sum,background_count,mark_sum,mark_count\n')
            for cell in range(300):
                # Background is perfect; a tiny marking is wrong by one.
                f.write(f'2,0,4,{cell},1,1,1,0,1023,1,1\n')
        features.write_text('split,frame,skin,sha256\n2,0,4,' + 'a'*64 + '\n')
        expected = [dict(split=2, skin=4, labels=bytes([1]*300))]
        return regions, features, expected

    def test_correct_background_cannot_hide_marking_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            regions, features, expected = self.fixture(Path(tmp))
            sc.audit_regions(regions, features, expected, 1)
            row = sc.summarize_regions(regions)[0]
            self.assertEqual(row['background_mse'], 0)
            self.assertEqual(row['marking_mse'], 1)
            self.assertEqual(row['object_mse'], 1)

    def test_missing_or_duplicate_cells_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            regions, features, expected = self.fixture(Path(tmp)); original = regions.read_text()
            regions.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'Missing'): sc.audit_regions(regions, features, expected, 1)
            regions.write_text(original + original.splitlines()[-1] + '\n')
            with self.assertRaisesRegex(RuntimeError, 'identity'): sc.audit_regions(regions, features, expected, 1)

    def test_wrong_skin_truth_or_frame_quota_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            regions, features, expected = self.fixture(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'quota'): sc.audit_regions(regions, features, expected, 2)
            expected[0]['labels'] = bytes([8]*300)
            with self.assertRaisesRegex(RuntimeError, 'truth'): sc.audit_regions(regions, features, expected, 1)

    def test_missing_feature_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            regions, features, expected = self.fixture(Path(tmp))
            features.write_text('split,frame,skin,sha256\n')
            with self.assertRaisesRegex(RuntimeError, 'Missing'): sc.audit_regions(regions, features, expected, 1)

    def test_preview_preserves_pixel_order_with_padded_bmp_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ppm, bmp = root / 'image.ppm', root / 'image.bmp'
            ppm.write_bytes(b'P6\n1 2\n255\n' + bytes([255,0,0,0,255,0]))
            sc.ppm_to_bmp(ppm, bmp)
            self.assertEqual(bmp.read_bytes()[54:], bytes([0,255,0,0,0,0,255,0]))


if __name__ == '__main__': unittest.main()
