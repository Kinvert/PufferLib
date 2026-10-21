"""Native full-frame Forage recognition pilot; stdlib manifests/metrics only.

Preparation never queries a GPU or collects data. Explicit run requires target
CUDA qualification, then collects one bounded shared dataset and runs controls.
Neural training/inference is native CUDA; scalar reports are ordinary CPU work.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import autoencoder_pretrain as ae

f = ae.f
ROOT, require, read, save, sha = ae.ROOT, ae.require, ae.read, ae.save, ae.sha
SOURCE = Path(__file__).read_bytes()
CLASSES = ['ground', 'apple', 'pear', 'grapes', 'leaf', 'flower', 'beetle',
    'striped mushroom', 'spotted apple', 'wasp', 'spotted toadstool', 'tree', 'explorer']
SPLITS = ('train', 'validation', 'test')


def prepare(out):
    ae.prepare(out, ['sprite_forage'])
    value = f.run_json([ae.trainer_binary(out), 'forage-host-checks'], out / 'source', out / 'forage-host-checks.log')
    require(value == dict(status='passed', skins=5, classes=13, cuda_execution=False), 'Annotation template check failed')
    command = ['clang', '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-D_DEFAULT_SOURCE', '-O1', '-g',
        '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-I.', '-Isrc', '-Iresearch',
        '-Iraylib-5.5_linux_amd64/include', 'research/test_forage_annotations.c',
        'raylib-5.5_linux_amd64/lib/libraylib.a', '-lGL', '-lpthread', '-ldl', '-lrt', '-lm',
        '-o', str(out / 'annotation-check')]
    f.process(command, out / 'source', out / 'annotation-build.log', 120)
    receipt = f.process([out / 'annotation-check'], out / 'source', out / 'annotation-check.log', 60)
    save(out / 'annotation-check.json', dict(status='passed', binary_sha256=sha(out / 'annotation-check'), process=receipt))
    return ae.inspect(out)


def settings(path):
    cfg = f.config(path)
    require(set(cfg.sections()) == {'experiment', 'data', 'autoencoder', 'probe'}, 'Pilot recipe sections')
    data = cfg['data']
    for key in ('train_skins', 'validation_skins', 'test_skins'):
        values = [int(n) for n in data[key].split(',')]
        require(values and len(set(values)) == len(values) and all(0 <= v < 5 for v in values), 'Skin list')
    require(1 <= cfg['experiment'].getint('max_seconds') <= 14400
        and 1 <= data.getint('frames_per_episode') <= 64
        and 1 <= data.getint('max_steps') <= 4096
        and 1 <= data.getint('cadence') <= data.getint('max_steps')
        and 1 <= data.getint('max_gib') <= 16, 'Pilot data/time cap')
    require(all(1 <= data.getint(s + '_worlds') <= 256 for s in SPLITS), 'Fixed world count bounds')
    require(cfg['autoencoder'].getint('seed') == 7 and cfg['autoencoder'].getfloat('edge_weight') == 0,
        'Fixed seed/plain-vs-object comparison')
    return cfg


def collect(packet, out, recipe, max_seconds=300):
    ae.inspect(packet); cfg = settings(recipe); data = cfg['data']
    started = time.monotonic(); deadline = started + min(300, max_seconds)
    require(set(ae.packet_games(packet)) == {'sprite_forage'} and not out.exists(), 'Fresh Forage dataset required')
    frame_cap = data.getint('frames_per_episode'); episodes = []
    total_episodes = sum(data.getint(s + '_worlds') * len(data[s + '_skins'].split(',')) for s in SPLITS)
    worst = total_episodes * frame_cap * (480 * 640 * 12 + 300)
    require(worst <= data.getint('max_gib') * 2**30 and shutil.disk_usage(out.parent).free >= worst + 20 * 2**30,
        'Collection byte budget/free reserve')
    out.mkdir(parents=True)
    save(out / 'collection-plan.json', dict(recipe_sha256=sha(recipe), source_manifest_sha256=sha(packet / 'sources.json'),
        worst_bytes=worst, classes=CLASSES, input='full-640x480-RGB', target='visible cell identity; not a reward tier',
        fixed_split_bases=dict(train=7, validation=420, test=1337), seed_sweep=False))
    for split, base in zip(SPLITS, (7, 420, 1337)):
        for world in range(data.getint(split + '_worlds')):
            identity = f'sprite_forage:{split}:{world}'
            seed = int.from_bytes(hashlib.sha256(f'{base}:{world}:env'.encode()).digest()[:4], 'little')
            action_seed = int.from_bytes(hashlib.sha256(f'{base}:{world}:actions'.encode()).digest()[:8], 'little')
            paired = []
            for skin in [int(n) for n in data[split + '_skins'].split(',')]:
                remaining = deadline - time.monotonic(); require(remaining > 1, 'Collection wall-clock cap reached')
                folder = out / split / f'world-{world:04d}' / f'skin-{skin}'; folder.mkdir(parents=True)
                path = folder / 'frames.f32'; label = folder / 'frames.f32.labels'
                result = f.run_json([ae.trainer_binary(packet), 'collect', packet / 'configs/sprite_forage.ini', path,
                    seed, action_seed, skin, data.getint('max_steps'), data.getint('cadence'), frame_cap],
                    packet / 'source', folder / 'collect.log', min(60, remaining))
                whole, frames = ae.frame_hashes(path, 480 * 640 * 3)
                require(result['frames'] == len(frames) and result['shape_hwc'] == [480, 640, 3]
                    and result['cuda_execution'] is False, 'Native collection contract')
                row = dict(id=identity + f':skin-{skin}', world_id=identity, game='sprite_forage', split=split,
                    env_seed=seed, action_seed=action_seed, skin=skin, path=str(path.relative_to(out)),
                    shape_hwc=[480, 640, 3], frames=len(frames), sha256=whole, frame_sha256=frames,
                    labels_path=str(label.relative_to(out)), labels_sha256=sha(label), decisions=result['decisions'], terminal=result['terminal'])
                episodes.append(row); paired.append((row['frames'], row['labels_sha256'], row['decisions'], row['terminal']))
                save(out / 'progress.json', dict(episodes=episodes))
            require(len(set(paired)) == 1, 'Appearance changed the world/action/visible identity stream')
    save(out / 'dataset.json', dict(schema='native-game-frames-v1', dtype='float32', layout='NHWC', value_range=[0, 1],
        source_manifest_sha256=sha(packet / 'sources.json'), annotation='forage-visible-grid-v1', classes=CLASSES,
        collection_plan_sha256=sha(out / 'collection-plan.json'), episodes=episodes))
    save(out / 'collection-time.json', dict(process_seconds=time.monotonic()-started, episode_processes=len(episodes)))
    return ae.dataset(out)


def expected_frames(data, root):
    rows = []
    for split in SPLITS:
        for ep in sorted(data['manifest']['episodes'], key=lambda e: e['id']):
            if ep['split'] != split:
                continue
            labels = (root / ep['labels_path']).read_bytes()
            for frame in range(ep['frames']):
                rows.append(dict(split=SPLITS.index(split), skin=ep['skin'], path=str((root / ep['path']).resolve()),
                    frame=frame, world_id=ep.get('world_id', ep['id']), labels=labels[frame * 300:(frame + 1) * 300]))
    return rows


def confusion_metrics(matrix):
    per_class = []
    for c, name in enumerate(CLASSES):
        support, predicted, tp = sum(matrix[c]), sum(row[c] for row in matrix), matrix[c][c]
        per_class.append(dict(id=c, name=name, support=support, predictions=predicted,
            recall=tp / support if support else None, precision=tp / predicted if predicted else None,
            f1=2 * tp / (support + predicted) if support else None))
    def macro(ids):
        values = [per_class[c]['f1'] for c in ids if per_class[c]['support']]
        return sum(values) / len(values) if values else None
    return dict(per_class=per_class, macro_f1=macro(range(13)), collectible_macro_f1=macro(range(1, 11)),
        confusion=matrix, absent_classes=[c['name'] for c in per_class if not c['support']])


def audit_probe(packet, data_root, run):
    ae.inspect(packet); request = read(run / 'request.json'); data = ae.dataset(data_root)
    require(data['manifest_sha256'] == request['dataset_sha256'] and request['sources_sha256'] == sha(packet / 'sources.json')
        and request['binary_sha256'] == sha(ae.trainer_binary(packet)), 'Probe source/data binding')
    require(data['manifest']['source_manifest_sha256'] == sha(packet / 'sources.json'), 'Forage renderer/annotation source differs')
    require(sha(run / 'recipe.ini') == request['recipe_sha256'] and sha(run / 'index.tsv') == request['index_sha256'], 'Probe inputs changed')
    checkpoint = request['checkpoint']
    parameter_hash = None
    if checkpoint not in ('random', 'constant'):
        state, raw = ae.state(Path(checkpoint))
        require(state['sha256'] == request['checkpoint_sha256'], 'Probe checkpoint changed')
        parameter_hash = hashlib.sha256(raw[ae.STATE.size:ae.STATE.size+state['parameters']*4]).hexdigest()
    receipt = read(run / 'native.process.json')
    command = list(map(str, [ae.trainer_binary(packet), 'forage-probe', packet / 'configs/sprite_forage.ini',
        run / 'index.tsv', checkpoint, run / 'recipe.ini', run]))
    require(receipt['returncode'] == 0 and not receipt['timeout'] and not receipt['interference']
        and sha(run / 'native.log') == receipt['log_sha256'] and receipt['argv'] == command
        and receipt['cwd'] == str(packet / 'source') and math.isfinite(receipt['seconds'])
        and receipt['seconds'] > 0, 'Probe process failed/changed')
    expected = expected_frames(data, data_root); native = read(run / 'native.json')
    require(native['frames'] == len(expected) and native['encoder_decoder_unchanged'] is True
        and native['probe_parameters'] == 300 * 13 * 129
        and native['constant_feature_control'] == (checkpoint == 'constant'), 'Native feature/readout contract')
    require(parameter_hash is None or parameter_hash == native['parameter_sha256'], 'Native loaded different checkpoint parameters')
    mapping = (run / 'frames.tsv').read_text().splitlines()
    require(len(mapping) == len(expected), 'Frame mapping quota')
    for i, (line, row) in enumerate(zip(mapping, expected)):
        require(line.split('\t') == list(map(str, (i, row['split'], row['skin'], row['path'], row['frame']))), 'Frame mapping differs')
    for name, elements in (('features.f32', len(expected) * 128), ('head.f32', 300 * 13 * 129), ('normalization.f32', 256)):
        path = run / name; require(path.stat().st_size == elements * 4, 'Probe array size')
        values = ae.array.array('f'); values.frombytes(path.read_bytes())
        require(all(map(math.isfinite, values)), 'Nonfinite probe arrays')
        if name == 'features.f32' and checkpoint == 'constant':
            require(all(v == 0 for v in values), 'Constant-feature control was not constant')
        if name == 'normalization.f32':
            require(all(v > 0 for v in values[128:]), 'Invalid normalization scale')
    require((run / 'labels.u8').read_bytes() == b''.join(row['labels'] for row in expected), 'Cached targets differ')
    groups = {}
    with (run / 'predictions.csv').open() as stream:
        reader = csv.DictReader(stream); count = 0
        for row in reader:
            values = {k: int(v) for k, v in row.items()}; i, cell = divmod(count, 300)
            require(i < len(expected) and values['frame'] == i and values['cell'] == cell, 'Prediction order/quota')
            target = expected[i]
            require(values['split'] == target['split'] and values['skin'] == target['skin']
                and values['truth'] == target['labels'][cell] and all(0 <= values[k] < 13 for k in ('prediction', 'majority'))
                and values['apple_pair_prediction'] in (1, 8) and values['mushroom_pair_prediction'] in (7, 10), 'Prediction/target binding')
            key = (values['split'], values['skin'])
            group = groups.setdefault(key, dict(confusion=[[0] * 13 for _ in range(13)], majority=[[0] * 13 for _ in range(13)],
                pairs={name: dict(count=0, correct=0, support={}, hits={}) for name in ('apple', 'mushroom')}, worlds=set()))
            truth = values['truth']; group['worlds'].add(target['world_id'])
            group['confusion'][truth][values['prediction']] += 1; group['majority'][truth][values['majority']] += 1
            for name, classes in (('apple', (1, 8)), ('mushroom', (7, 10))):
                if truth in classes:
                    pair = group['pairs'][name]; correct = values[name + '_pair_prediction'] == truth
                    pair['count'] += 1; pair['correct'] += correct
                    pair['support'][truth] = pair['support'].get(truth, 0) + 1
                    pair['hits'][truth] = pair['hits'].get(truth, 0) + correct
            count += 1
    require(count == len(expected) * 300, 'Missing prediction rows')
    require({(r['split'], r['skin']) for r in native['reconstruction']} == set(groups)
        and len(native['reconstruction']) == len(groups), 'Missing reconstruction groups')
    for row in native['reconstruction']:
        require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0
            for k in ('global_mse', 'background_mse')), 'Invalid reconstruction metric')
        require([v['class'] for v in row['objects']] == list(range(13)), 'Missing object reconstruction classes')
        support = groups[row['split'], row['skin']]['confusion']
        for item in row['objects']:
            c = item['class']
            require(item['instances'] == (sum(support[c]) if c else 0), 'Object instance quota differs')
            for key in ('mse', 'mark_mse'):
                require(key not in item or isinstance(item[key], (int, float)) and math.isfinite(item[key]) and item[key] >= 0,
                    'Invalid object reconstruction score')
            require(not item['instances'] or 'mse' in item, 'Missing foreground error')
    rows = []
    for (split, skin), group in sorted(groups.items()):
        pairs = {}
        for name, pair in group['pairs'].items():
            recalls = [pair['hits'][c] / n for c, n in pair['support'].items()]
            pairs[name] = dict(count=pair['count'], accuracy=pair['correct'] / pair['count'] if pair['count'] else None,
                balanced_accuracy=sum(recalls) / 2 if len(recalls) == 2 else None,
                conditional_on_true_object_location=True)
        rows.append(dict(split=SPLITS[split], skin=skin, independent_worlds=len(group['worlds']),
            trained_skin=skin in request['train_skins'], probe=confusion_metrics(group['confusion']),
            majority=confusion_metrics(group['majority']), lookalikes=pairs))
    files = {p.name: sha(p) for p in run.iterdir() if p.is_file() and p.name not in ('REPORT.json', 'REPORT.md')}
    if (run / 'REPORT.json').exists():
        require(read(run / 'REPORT.json')['artifacts'] == files, 'Changed retained probe artifact')
    return dict(status=native['status'], rows=rows, reconstruction=native['reconstruction'],
        native=native, artifacts=files, process_seconds=receipt['seconds'], dataset_sha256=data['manifest_sha256'],
        no_rl_or_pareto_claim=True, probe_target='full-frame latent -> all 300 cell identities; no oracle crop')


def probe(packet, data_root, recipe, checkpoint, out, gpu):
    require(not out.exists(), 'Fresh probe output required'); ae.inspect(packet); data = ae.dataset(data_root)
    out.mkdir(parents=True); shutil.copyfile(recipe, out / 'recipe.ini')
    require(data['manifest']['source_manifest_sha256'] == sha(packet / 'sources.json'), 'Use the captured Forage renderer dataset')
    ae.write_index(data, data_root, out / 'index.tsv', include_test=True)
    request = dict(dataset_sha256=data['manifest_sha256'], sources_sha256=sha(packet / 'sources.json'),
        binary_sha256=sha(ae.trainer_binary(packet)), recipe_sha256=sha(out / 'recipe.ini'), index_sha256=sha(out / 'index.tsv'),
        checkpoint=str(checkpoint), checkpoint_sha256=ae.state(checkpoint)[0]['sha256'] if checkpoint not in ('random', 'constant') else None,
        train_skins=sorted({ep['skin'] for ep in data['manifest']['episodes'] if ep['split'] == 'train'}))
    save(out / 'request.json', request)
    seconds = f.config(recipe)['probe'].getint('max_seconds')
    gpu.run([ae.trainer_binary(packet), 'forage-probe', packet / 'configs/sprite_forage.ini', out / 'index.tsv',
        checkpoint, out / 'recipe.ini', out], out / 'native.log', seconds + 60)
    result = audit_probe(packet, data_root, out); save(out / 'REPORT.json', result); return result


def run(packet, out, recipe, hardware):
    ae.inspect(packet); proof = ae.audit_qualification(packet)
    require(hardware in proof['hardware'] and not out.exists(), 'Target qualification/fresh campaign required')
    require(not (packet / 'stages/forage-campaign.started.json').exists(), 'Campaign allocation already used')
    require((packet / 'source/research/forage_representation.py').read_bytes() == SOURCE, 'Reprepare changed supervisor')
    settings(recipe); out.mkdir(parents=True); shutil.copyfile(recipe, out / 'recipe.ini'); cfg = settings(out / 'recipe.ini')
    started = time.monotonic(); deadline = started + cfg['experiment'].getint('max_seconds')
    save(out / 'request.json', dict(sources_sha256=sha(packet / 'sources.json'), recipe_sha256=sha(out / 'recipe.ini'),
        hardware=hardware, qualified=True, candidates=['constant', 'random', 'plain', 'object-balanced']))
    progress = dict(training={}, probes={})
    try:
        data_root = out / 'data'; collect(packet, data_root, out / 'recipe.ini', deadline-time.monotonic())
        require(deadline-time.monotonic()>1, 'Campaign cap reached during collection')
        with ae.Reservation(packet, hardware, 'forage-campaign', deadline-time.monotonic()) as gpu:
            candidates = [('constant', 'constant'), ('random', 'random')]
            for name, mode in (('plain', 0), ('object-balanced', 1)):
                cfg['autoencoder']['loss_mode'] = str(mode); path = out / (name + '.ini'); f.write_config(path, cfg)
                train_root = out / 'training' / name
                result = ae.train(packet, data_root, path, train_root, hardware, qualification_gpu=gpu)
                progress['training'][name] = dict(status=result['status'], report=str(train_root / 'REPORT.json'), sha256=sha(train_root / 'REPORT.json'))
                require(result['checkpoints'][-1]['step'] > 0, 'No learned encoder before cap')
                candidates.append((name, train_root / result['checkpoints'][-1]['path']))
                ae.export(packet, train_root, 'sprite_forage-480x640', out / 'encoder-exports' / name)
                save(out / 'progress.json', progress)
            initial = [out / 'training' / name / 'step-000000000000.state' for name in ('plain', 'object-balanced')]
            require(sha(initial[0]) == sha(initial[1]), 'Pretraining initial weights/Adam differ')
            state, raw = ae.state(initial[0]); initial_parameter_hash = hashlib.sha256(raw[ae.STATE.size:ae.STATE.size+state['parameters']*4]).hexdigest()
            progress['initial_parameter_sha256'] = initial_parameter_hash
            for name, checkpoint in candidates:
                probe_root = out / 'probes' / name
                result = probe(packet, data_root, out / 'recipe.ini', checkpoint, probe_root, gpu)
                if checkpoint in ('random', 'constant'):
                    require(result['native']['parameter_sha256'] == initial_parameter_hash, 'Random controls differ from pretraining initialization')
                progress['probes'][name] = dict(status=result['status'], report=str(probe_root / 'REPORT.json'), sha256=sha(probe_root / 'REPORT.json'))
                save(out / 'progress.json', progress)
        return report(packet, out)
    except Exception as error:
        save(out / 'failure.json', dict(error=str(error), progress=progress)); report(packet, out); raise


def report(packet, out):
    require(sha(out / 'recipe.ini') == read(out / 'request.json')['recipe_sha256'], 'Campaign recipe changed')
    progress = read(out / 'progress.json') if (out / 'progress.json').exists() else dict(training={}, probes={})
    training = {}
    for name, entry in progress['training'].items():
        directory = Path(entry['report']).parent
        require(sha(directory / 'REPORT.json') == entry['sha256'], 'Training report changed')
        trained = read(directory / 'REPORT.json')
        require(trained['contract']['dataset_manifest_sha256'] == sha(out / 'data/dataset.json')
            and trained['contract']['sources_sha256'] == sha(packet / 'sources.json'), 'Training source/data differs')
        for checkpoint in trained['checkpoints']:
            current = ae.state(directory / checkpoint['path'])[0]
            require(all(current[k] == checkpoint[k] for k in current), 'Retained training checkpoint changed')
        training[name] = dict(status=trained['status'], final_step=trained['checkpoints'][-1]['step'],
            process_seconds=trained['process_seconds'], process_training_images_per_second=trained['process_training_images_per_second'])
    results = {}
    for name, entry in progress['probes'].items():
        directory = Path(entry['report']).parent
        require(sha(directory / 'REPORT.json') == entry['sha256'], 'Probe report changed')
        result = audit_probe(packet, out / 'data', directory)
        require(result == read(directory / 'REPORT.json'), 'Probe audit differs')
        results[name] = result
        if name in ('random', 'constant'):
            require(result['native']['parameter_sha256'] == progress['initial_parameter_sha256'], 'Retained random control binding differs')
    if 'constant' in results and 'random' in results:
        require(results['constant']['reconstruction'] == results['random']['reconstruction'], 'Constant control changed the random decoder')
    finished = set(results) == {'constant', 'random', 'plain', 'object-balanced'} and not (out / 'failure.json').exists()
    complete = finished and all(r['status'] == 'complete' for r in results.values()) \
        and len(progress['training']) == 2 and all(r['status'] == 'complete' for r in progress['training'].values())
    value = dict(status='complete' if complete else 'budget_exhausted' if finished else 'failed' if (out / 'failure.json').exists() else 'incomplete',
        candidates=results, training=training, progress=progress, no_rl_or_pareto_claim=True)
    save(out / 'REPORT.json', value)
    lines = ['# Forage representation pilot', '', f'Status: **{value["status"]}**.', '',
        'Full 640x480 RGB input; shared 128-value latent; 503,100-parameter linear probe predicts all 300 cells.',
        'Pair accuracy below is conditional on the true object location; it is not detection or game success.', '',
        '|Candidate|Split|Skin|Seen skin|Worlds|Item macro F1|Apple pair balanced accuracy|Mushroom pair balanced accuracy|',
        '|---|---|---:|---|---:|---:|---:|---:|']
    def number(x): return 'N/A' if x is None else f'{100*x:.2f}%'
    for name, result in results.items():
        for row in result['rows']:
            if row['split'] == 'train': continue
            lines.append('|'+ '|'.join(map(str, (name,row['split'],row['skin'],row['trained_skin'],row['independent_worlds'],
                number(row['probe']['collectible_macro_f1']),number(row['lookalikes']['apple']['balanced_accuracy']),
                number(row['lookalikes']['mushroom']['balanced_accuracy']))))+'|')
    lines += ['', 'See REPORT.json for all class recalls/confusions, majority controls, per-object and distinguishing-mark reconstruction errors,',
        'actual update counts, time caps and source/artifact bindings. A low global/background MSE does not qualify useful perception.',
        'This is a small fixed-pretraining-seed diagnostic, not downstream RL, a seven-replica study or an architecture winner.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n');return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'qualify', 'collect', 'run', 'report'))
    parser.add_argument('--packet', type=Path); parser.add_argument('--out', type=Path)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'research/recipes/forage_representation.ini')
    parser.add_argument('--hardware', choices=('5060', '5090'))
    args = parser.parse_args()
    for name in ('packet', 'out', 'recipe'):
        if getattr(args,name):setattr(args,name,getattr(args,name).resolve())
    if args.command == 'prepare':
        require(args.out is not None,'Need --out');result=prepare(args.out)
    else:
        require(args.packet is not None,'Need --packet')
        if args.command == 'qualify':
            require(args.hardware is not None,'Need --hardware');result=ae.qualify(args.packet,args.hardware)
        else:
            require(args.out is not None,'Need --out')
            if args.command == 'collect':result=collect(args.packet,args.out,args.recipe)
            elif args.command == 'run':
                require(args.hardware is not None,'Need --hardware');result=run(args.packet,args.out,args.recipe,args.hardware)
            else:result=report(args.packet,args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':main()
