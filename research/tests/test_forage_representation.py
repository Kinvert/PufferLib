"""Scalar/data receipt checks; no CPU neural model, GPU query or actual corpus."""
import array
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import forage_representation as fr
ae = fr.ae


def synthetic(root):
    packet, data_root, run = (root / n for n in ('packet', 'data', 'probe'))
    for p in (packet, data_root, run):p.mkdir()
    ae.save(packet / 'sources.json', {})
    binary=packet / 'builds/sprite_forage/autoencoder';binary.parent.mkdir(parents=True);binary.write_bytes(b'native-fixture')
    ae.save(packet / 'plan.json', dict(games=dict(sprite_forage=[480,640,5])))
    labels=bytes(c%13 for c in range(300));episodes=[]
    for split in fr.SPLITS:
        path=data_root/(split+'.labels');path.write_bytes(labels)
        episodes.append(dict(id=split,world_id=split,split=split,game='sprite_forage',frames=1,skin=0,
            shape_hwc=[480,640,3],path=split+'.f32',labels_path=path.name))
    data=dict(manifest=dict(episodes=episodes,source_manifest_sha256=ae.sha(packet/'sources.json')),manifest_sha256='frozen-data')
    (run/'recipe.ini').write_text('[probe]\nsteps=4\n');(run/'index.tsv').write_text('frozen-index')
    ae.save(run/'request.json',dict(dataset_sha256='frozen-data',sources_sha256=ae.sha(packet/'sources.json'),
        binary_sha256=ae.sha(binary),recipe_sha256=ae.sha(run/'recipe.ini'),index_sha256=ae.sha(run/'index.tsv'),
        checkpoint='random',checkpoint_sha256=None,train_skins=[0]))
    (run/'native.log').write_text('')
    ae.save(run/'native.process.json',dict(returncode=0,timeout=False,interference=[],seconds=1,
        log_sha256=ae.sha(run/'native.log'),cwd=str(packet/'source'),
        argv=list(map(str,[binary,'forage-probe',packet/'configs/sprite_forage.ini',run/'index.tsv','random',run/'recipe.ini',run]))))
    rows=fr.expected_frames(data,data_root)
    (run/'frames.tsv').write_text(''.join('\t'.join(map(str,(i,r['split'],r['skin'],r['path'],r['frame'])))+'\n' for i,r in enumerate(rows)))
    (run/'features.f32').write_bytes(bytes(3*128*4));(run/'head.f32').write_bytes(bytes(300*13*129*4))
    (run/'normalization.f32').write_bytes(array.array('f',[0]*128+[1]*128).tobytes())
    (run/'labels.u8').write_bytes(labels*3)
    with (run/'predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['frame','split','skin','cell','truth','prediction','majority','apple_pair_prediction','mushroom_pair_prediction'])
        for i,r in enumerate(rows):
            for c,k in enumerate(labels):w.writerow([i,r['split'],0,c,k,k,0,1 if k!=8 else 8,7 if k!=10 else 10])
    region=[]
    for split in range(3):
        region.append(dict(split=split,skin=0,global_mse=.01,background_mse=.001,objects=[dict(
            **{'class':c},instances=labels.count(c) if c else 0,mse=.02,mark_instances=0) for c in range(13)]))
    ae.save(run/'native.json',dict(frames=3,encoder_decoder_unchanged=True,constant_feature_control=False,
        probe_parameters=300*13*129,status='complete',reconstruction=region))
    return packet,data_root,run,data


class Metrics(unittest.TestCase):
    def test_empty_space_cannot_make_item_f1_good(self):
        matrix=[[0]*13 for _ in range(13)];matrix[0][0]=100000;matrix[1][0]=10;matrix[8][0]=10
        value=fr.confusion_metrics(matrix)
        self.assertEqual(value['collectible_macro_f1'],0)
        self.assertEqual(value['per_class'][1]['recall'],0)

    def test_absent_classes_are_not_perfect(self):
        matrix=[[0]*13 for _ in range(13)];matrix[1][1]=4
        value=fr.confusion_metrics(matrix)
        self.assertIsNone(value['per_class'][8]['f1'])
        self.assertIn('spotted apple',value['absent_classes'])

    def test_fine_identity_errors_are_visible(self):
        matrix=[[0]*13 for _ in range(13)];matrix[1][8]=8;matrix[8][1]=8
        value=fr.confusion_metrics(matrix)
        self.assertEqual(value['collectible_macro_f1'],0)
        self.assertEqual(value['confusion'][1][8],8)


class AnnotationContracts(unittest.TestCase):
    def data(self,root):
        episodes=[];image=bytes(480*640*12);digest=fr.hashlib.sha256(image).hexdigest()
        for split,seed in zip(fr.SPLITS,(7,420,1337)):
            path=root/(split+'.f32');path.write_bytes(image);labels=root/(split+'.labels');labels.write_bytes(bytes(300))
            episodes.append(dict(id=split,game='sprite_forage',split=split,env_seed=seed,skin=0,
                shape_hwc=[480,640,3],frames=1,path=path.name,sha256=digest,frame_sha256=[digest],
                labels_path=labels.name,labels_sha256=ae.sha(labels)))
        value=dict(schema='native-game-frames-v1',dtype='float32',layout='NHWC',value_range=[0,1],episodes=episodes)
        ae.save(root/'dataset.json',value);return value

    def test_final_worlds_must_not_be_training_worlds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=self.data(root);v['episodes'][2]['env_seed']=7;ae.save(root/'dataset.json',v)
            with self.assertRaisesRegex(RuntimeError,'seed leaks'):ae.dataset(root)

    def test_annotation_hashes_are_rebound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.data(root);(root/'test.labels').write_bytes(bytes([1])*300)
            with self.assertRaisesRegex(RuntimeError,'annotation shard'):ae.dataset(root)

    def test_invalid_identity_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);v=self.data(root);p=root/'test.labels';p.write_bytes(bytes([13])*300)
            v['episodes'][2]['labels_sha256']=ae.sha(p);ae.save(root/'dataset.json',v)
            with self.assertRaisesRegex(RuntimeError,'pixel-label contract'):ae.dataset(root)

    def test_cross_split_duplicate_images_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.data(root);value=ae.dataset(root)
            self.assertEqual(value['test_train_overlap'],1)
            self.assertEqual(value['test_validation_overlap'],1)


class ReceiptAudit(unittest.TestCase):
    def audit(self,packet,root,run,data):
        with patch.object(ae,'inspect'),patch.object(ae,'dataset',return_value=data):
            return fr.audit_probe(packet,root,run)

    def test_full_quota_and_known_confusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=synthetic(Path(tmp));value=self.audit(*args)
            self.assertEqual(len(value['rows']),3)
            self.assertEqual(value['rows'][-1]['probe']['collectible_macro_f1'],1)
            self.assertEqual(value['rows'][-1]['lookalikes']['apple']['balanced_accuracy'],1)
            self.assertTrue(value['no_rl_or_pareto_claim'])

    def test_missing_row_is_not_silently_averaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'predictions.csv'
            p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(RuntimeError,'Missing prediction'):
                self.audit(packet,root,run,data)

    def test_wrong_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'labels.u8'
            p.write_bytes(bytes([12])+p.read_bytes()[1:])
            with self.assertRaisesRegex(RuntimeError,'Cached targets'):
                self.audit(packet,root,run,data)

    def test_missing_object_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'native.json';v=ae.read(p)
            del v['reconstruction'][0]['objects'][1]['mse'];ae.save(p,v)
            with self.assertRaisesRegex(RuntimeError,'Missing foreground'):
                self.audit(packet,root,run,data)

    def test_changed_native_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'native.process.json';v=ae.read(p)
            v['argv'][4]='different-weights';ae.save(p,v)
            with self.assertRaisesRegex(RuntimeError,'Probe process'):
                self.audit(packet,root,run,data)

    def test_changed_retained_feature_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));value=self.audit(packet,root,run,data)
            ae.save(run/'REPORT.json',value);p=run/'features.f32';raw=bytearray(p.read_bytes());raw[3]=63;p.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError,'Changed retained'):
                self.audit(packet,root,run,data)

    def test_nonfinite_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'features.f32';raw=bytearray(p.read_bytes())
            raw[:4]=array.array('f',[float('nan')]).tobytes();p.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError,'Nonfinite probe'):
                self.audit(packet,root,run,data)

    def test_test_frames_cannot_enter_autoencoder_training_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet,root,run,data=synthetic(Path(tmp));p=run/'training.tsv';ae.write_index(data,root,p)
            self.assertNotIn('\ntest\t',p.read_text())
            self.assertTrue(all(len(r.split('\t'))==8 for r in p.read_text().splitlines()[1:]))
            ae.write_index(data,root,run/'probe.tsv',include_test=True)
            self.assertIn('\ntest\t',(run/'probe.tsv').read_text())


if __name__=='__main__':unittest.main()
