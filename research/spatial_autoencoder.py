"""Native spatial AE pilot: shared data, ROI reconstruction, encoder-only bank.

Python only supervises processes, verifies artifacts and tabulates scalar errors.
All convolutions, neural gradients and optimizer updates execute in native CUDA.
"""
import argparse
import array
import csv
import html
import json
import math
from pathlib import Path
import re
import shutil

import autoencoder_pretrain as ae
import forage_representation as fr

f = ae.f
ROOT, require, read, save, sha = ae.ROOT, ae.require, ae.read, ae.save, ae.sha
SOURCE = Path(__file__).read_bytes()


def variants(text):
    result = [tuple(map(int, part.split(':'))) for part in text.split(',')]
    require(result and len(result) == len(set(result)) and all(s in (8, 16, 32) and c in (4, 8, 16, 32)
        for s, c in result), 'Choose unique stride:channels pairs (8/16/32 : 4/8/16/32)')
    return result


def counts(channels):
    return 392 + 1168 + channels * 257, 16 * (9 * channels + 1) + 8 * 145 + 3 * 73


def state(path, channels):
    raw = path.read_bytes(); require(len(raw) >= ae.STATE.size, 'Truncated spatial state')
    magic, version, parameters, step, seed = ae.STATE.unpack_from(raw)
    require(magic == ae.MAGIC and version == 2 and parameters == sum(counts(channels)) and seed == 7
        and len(raw) == ae.STATE.size + parameters * 12, 'Spatial state header/size differs')
    values = array.array('f'); values.frombytes(raw[ae.STATE.size:])
    require(all(map(math.isfinite, values)), 'Nonfinite spatial weights or Adam state')
    return dict(parameters=parameters, step=step, seed=seed, sha256=sha(path)), raw


def inspect(packet):
    ae.inspect(packet); plan = read(packet / 'spatial-plan.json')
    require(read(packet / 'plan.json').get('spatial') is True, 'Not a spatial packet')
    require(sha(packet / 'recipe.ini') == plan['recipe_sha256'], 'Changed spatial recipe')
    require(sha(packet / 'source/research/spatial_autoencoder.py') == ae.hashlib.sha256(SOURCE).hexdigest(),
        'Use the captured spatial supervisor revision')
    return plan


def prepare(packet, recipe):
    cfg = fr.settings(recipe); grid = variants(cfg['experiment']['variants'])
    fr.prepare(packet, spatial=True); shutil.copyfile(recipe, packet / 'recipe.ini')
    manifests = {}
    for stride, channels in grid:
        key = f's{stride}-c{channels}'
        m = f.run_json([ae.trainer_binary(packet), 'spatial-manifest', 480, 640, stride, channels],
            packet / 'source', packet / f'logs/prepare/{key}.log')
        require(m['schema'] == 'spatial-compressor-v1' and m['elements'] == counts(channels)[0]
            and m['latent_hwc'] == [(480 + stride - 1) // stride, (640 + stride - 1) // stride, channels]
            and len(m['layers']) == 3 and len(m['tensors']) == 6, 'Independent spatial manifest differs')
        save(packet / f'spatial-manifests/{key}.json', m); manifests[key] = sha(packet / f'spatial-manifests/{key}.json')
    save(packet / 'spatial-plan.json', dict(schema='native-spatial-ae-packet-v1', variants=grid,
        recipe_sha256=sha(packet / 'recipe.ini'), manifests=manifests, gpu_execution=False))
    return inspect(packet)


class Reservation(ae.Reservation):
    def __enter__(self):
        inspect(self.out)
        result = super().__enter__()
        (self.out / f'stages/{self.stage}.spatial-supervisor.py').write_bytes(SOURCE)
        return result


def recipe_for(packet, stride, channels, loss_mode=1, steps=None, batch=None):
    cfg = f.config(packet / 'recipe.ini')
    cfg['autoencoder'].update(latent_stride=str(stride), latent_channels=str(channels), loss_mode=str(loss_mode))
    if steps is not None:
        cfg['autoencoder'].update(steps=str(steps), checkpoint_interval=str(max(1, steps // 2)), max_seconds='60')
    if batch is not None:
        cfg['autoencoder']['batch'] = str(batch)
    return cfg


def fit(packet, data_root, cfg, out, gpu, resume=None):
    inspect(packet); require(not out.exists(), 'Fresh spatial training directory required')
    require(gpu.out == packet and gpu.stage in ('spatial-smoke', 'spatial-qualify', 'spatial-run'), 'Spatial reservation context')
    data = ae.dataset(data_root); a = cfg['autoencoder']; channels = a.getint('latent_channels')
    require(a.getint('seed') == 7 and (a.getint('latent_stride'), channels) in
        [tuple(v) for v in inspect(packet)['variants']], 'Unregistered spatial candidate/seed')
    out.mkdir(parents=True); f.write_config(out / 'recipe.ini', cfg)
    ae.write_index(data, data_root, out / 'index.tsv')  # Never gives final-test frames to the trainer.
    contract = ae.training_contract(packet, out / 'recipe.ini', data)
    if resume:
        previous = read(resume.with_suffix('.json'))
        require(previous['contract'] == contract and previous['state'] == state(resume, channels)[0], 'Spatial resume binding differs')
    save(out / 'request.json', dict(contract=contract, recipe_sha256=sha(out / 'recipe.ini'),
        index_sha256=sha(out / 'index.tsv'), resume=str(resume) if resume else None,
        resume_sha256=sha(resume) if resume else None))
    receipt = gpu.run([ae.trainer_binary(packet), 'train', out / 'recipe.ini', out / 'index.tsv', out,
        resume or 'None', 'v1'], out / 'train.log', a.getint('max_seconds') + 60)
    rows = [json.loads(line) for line in (out / 'train.log').read_text().splitlines() if line.startswith('{')]
    require(rows and rows[-1]['status'] in ('complete', 'budget_exhausted'), 'Missing native spatial training result')
    checkpoints = []
    for path in sorted(out.glob('step-*.state')):
        metadata = state(path, channels)[0]
        save(path.with_suffix('.json'), dict(state=metadata, contract=contract))
        checkpoints.append(dict(path=path.name, **metadata))
    require(checkpoints and checkpoints[-1]['step'] == rows[-1]['final_step'] and checkpoints[-1]['step'] > 0,
        'Missing final spatial checkpoint/no training updates')
    request = read(out / 'request.json')
    require(sha(out / 'recipe.ini') == request['recipe_sha256'] and sha(out / 'index.tsv') == request['index_sha256'],
        'Spatial training inputs changed')
    ae.dataset(data_root)
    initial = state(resume, channels)[0]['step'] if resume else 0
    result = dict(status=rows[-1]['status'], hardware=gpu.identity, contract=contract, process=receipt,
        rows=rows, checkpoints=checkpoints, training_images=(checkpoints[-1]['step'] - initial) * a.getint('batch'),
        exact_frame_overlap=data['exact_frame_overlap'], learning_or_frontier_claim=False)
    result['process_images_per_second'] = result['training_images'] / receipt['seconds']
    save(out / 'REPORT.json', result); return result


def export(packet, run, out):
    inspect(packet); report = read(run / 'REPORT.json'); request = read(run / 'request.json')
    cfg = f.config(run / 'recipe.ini'); a = cfg['autoencoder']; channels = a.getint('latent_channels')
    require(not out.exists() and report['contract'] == request['contract']
        and sha(run / 'recipe.ini') == request['recipe_sha256']
        and sha(run / 'index.tsv') == request['index_sha256'], 'Spatial export binding/fresh directory')
    # The scheduled final state is used irrespective of any reconstruction score.
    path = run / report['checkpoints'][-1]['path']; metadata, raw = state(path, channels)
    require(metadata == read(path.with_suffix('.json'))['state'] and metadata['sha256'] == report['checkpoints'][-1]['sha256']
        and read(path.with_suffix('.json'))['contract'] == report['contract'], 'Changed spatial export state')
    stride = a.getint('latent_stride'); manifest = read(packet / f'spatial-manifests/s{stride}-c{channels}.json')
    require(sha(packet / f'spatial-manifests/s{stride}-c{channels}.json') == inspect(packet)['manifests'][f's{stride}-c{channels}'],
        'Changed spatial manifest')
    payload = b''.join(raw[ae.STATE.size + t['offset'] * 4:ae.STATE.size + (t['offset'] + t['elements']) * 4]
        for t in manifest['tensors'])
    require(len(payload) == counts(channels)[0] * 4, 'Spatial encoder extraction size')
    out.mkdir(parents=True); (out / 'encoder.bin').write_bytes(payload)
    manifest.update(sha256=sha(out / 'encoder.bin'), source_checkpoint_sha256=metadata['sha256'],
        source_manifest_sha256=sha(packet / 'sources.json'), pretraining_contract=report['contract'], step=metadata['step'])
    save(out / 'encoder.json', manifest); return manifest


def reconstruction(packet, data_root, run, artifact, out, gpu):
    cfg = f.config(run / 'recipe.ini'); channels = cfg['autoencoder'].getint('latent_channels')
    proof = read(run / 'REPORT.json'); checkpoint = run / proof['checkpoints'][-1]['path']
    require(state(checkpoint, channels)[0]['sha256'] == proof['checkpoints'][-1]['sha256'], 'Reconstruction checkpoint binding')
    index = out.with_suffix('.index.tsv'); require(not index.exists(), 'Fresh evaluation index')
    data = ae.dataset(data_root); ae.write_index(data, data_root, index, include_test=True)
    receipt = gpu.run([ae.trainer_binary(packet), 'spatial-reconstruct', run / 'recipe.ini', index,
        checkpoint, out, artifact], out.with_suffix('.log'), 600)
    result = json.loads(out.with_suffix('.log').read_text())
    require(result['status'] == 'passed' and result['export_features_exact'] and result['parameters_unchanged'],
        'Spatial reconstruction/import validation failed')
    audit_regions(out / 'regions.csv', out / 'features.csv', fr.expected_frames(data, data_root), result['frames'])
    metrics = summarize_regions(out / 'regions.csv')
    save(out / 'REPORT.json', dict(native=result, metrics=metrics, process=receipt,
        checkpoint_sha256=sha(checkpoint), encoder_sha256=sha(artifact / 'encoder.bin'),
        dataset_sha256=sha(data_root / 'dataset.json'), feature_hashes_sha256=sha(out / 'features.csv'),
        regions_sha256=sha(out / 'regions.csv')))
    return read(out / 'REPORT.json')


def audit_regions(regions, features, expected, actual_frames):
    require(actual_frames == len(expected) and actual_frames > 0, 'Reconstruction frame quota differs')
    frames = {}; counters = [0, 0, 0]
    for row in expected:
        split = row['split']; frame = counters[split]; counters[split] += 1
        frames[split, frame] = row
    seen = set()
    with regions.open() as stream:
        for row in csv.DictReader(stream):
            split, frame, cell = [int(row[k]) for k in ('split', 'frame', 'cell')]
            key = split, frame, cell; target = frames.get((split, frame))
            require(key not in seen and target is not None and 0 <= cell < 300
                and int(row['skin']) == target['skin'] and int(row['kind']) == target['labels'][cell],
                'Reconstruction row identity/truth differs')
            seen.add(key)
    require(len(seen) == actual_frames * 300, 'Missing reconstruction region rows')
    seen = set()
    with features.open() as stream:
        for row in csv.DictReader(stream):
            key = int(row['split']), int(row['frame']); target = frames.get(key)
            digest = row['sha256']
            require(key not in seen and target is not None and int(row['skin']) == target['skin']
                and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Feature quota/identity/hash differs')
            seen.add(key)
    require(set(frames) == seen, 'Missing reconstruction feature hashes')


def summarize_regions(path):
    # Scalar reduction only. Equal instance weighting prevents large objects
    # dominating; keep every class, split and skin separately, including absence.
    table = {}; seen = set()
    with path.open() as stream:
        for r in csv.DictReader(stream):
            identity = tuple(int(r[k]) for k in ('split', 'frame', 'skin', 'cell'))
            require(identity not in seen, 'Duplicate reconstruction cell row'); seen.add(identity)
            split, frame, skin, cell = identity; kind = int(r['kind'])
            require(0 <= split <= 2 and 0 <= skin < 5 and 0 <= cell < 300 and 0 <= kind < 13, 'Region identity')
            value = table.setdefault((split, skin, kind), dict(instances=0, object_sum=0., marking_instances=0, mark_sum=0.,
                background_sum=0., background_pixels=0))
            fg, bg, mark = [float(r[k]) for k in ('foreground_sum', 'background_sum', 'mark_sum')]
            nf, nb, nm = [int(float(r[k])) for k in ('foreground_count', 'background_count', 'mark_count')]
            require(all(math.isfinite(v) and v >= 0 for v in (fg, bg, mark)) and nf + nb == 1024 and 0 <= nm <= 1024,
                'Invalid region sums/counts')
            if nf:
                value['instances'] += 1; value['object_sum'] += fg / nf
            if nm:
                value['marking_instances'] += 1; value['mark_sum'] += mark / nm
            value['background_sum'] += bg; value['background_pixels'] += nb
    require(seen, 'No reconstruction rows')
    for split, frame, skin in {r[:3] for r in seen}:
        require(sum((split, frame, skin, c) in seen for c in range(300)) == 300, 'Missing reconstruction cells')
    rows = []
    for (split, skin, kind), v in sorted(table.items()):
        rows.append(dict(split=fr.SPLITS[split], skin=skin, kind=kind, name=fr.CLASSES[kind],
            instances=v['instances'], object_mse=v['object_sum'] / v['instances'] if v['instances'] else None,
            marking_instances=v['marking_instances'], marking_mse=v['mark_sum'] / v['marking_instances'] if v['marking_instances'] else None,
            background_pixels=v['background_pixels'], background_mse=v['background_sum'] / v['background_pixels'] if v['background_pixels'] else None))
    return rows


def policy_config(packet, stride, channels, artifact, folder, frozen=1):
    cfg = f.config(packet / 'configs/sprite_forage.ini')
    cfg['policy'].update(compression_model='3', compression_init=str(artifact.resolve()), compression_frozen=str(frozen),
        latent_stride=str(stride), latent_channels=str(channels))
    cfg['base'].update(run_id='happy-fox-1', checkpoint_dir=str((folder / 'checkpoints').resolve()), log_dir=str((folder / 'metrics').resolve()))
    for name in list(cfg.sections()):
        if name.startswith('sweep.'):
            cfg.remove_section(name)
    return cfg


def policy_canary(packet, stride, channels, artifact, folder, gpu, frozen=1, sanitizer=None):
    require(not folder.exists(), 'Fresh policy canary directory')
    cfg = policy_config(packet, stride, channels, artifact, folder, frozen)
    f.write_config(folder / 'config/sprite_forage.ini', cfg); (folder / 'config/default.ini').touch()
    command = [ae.trainer_binary(packet), 'policy', 'train', '--headless']
    if sanitizer:
        command = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99'] + command
    receipt = gpu.run(command, folder / 'train.log', 180 if sanitizer else 60, folder)
    if sanitizer:
        require('ERROR SUMMARY: 0 errors' in (folder / 'train.log').read_text(), 'Policy learner memory gate')
    files = sorted((folder / 'checkpoints').rglob('*.bin'))
    require(len(files) == 2, 'Policy canary scheduled checkpoints differ')
    for path in files:
        values = array.array('f'); values.frombytes(path.read_bytes())
        require(values and all(map(math.isfinite, values)), 'Nonfinite policy checkpoint')
    reload = folder / 'reload'; reload_cfg = policy_config(packet, stride, channels, artifact, reload, frozen)
    reload_cfg['base'].update(load_model_path=str(files[-1].resolve()), checkpoint_interval='1')
    reload_cfg['train']['total_timesteps'] = '256'
    f.write_config(reload / 'config/sprite_forage.ini', reload_cfg); (reload / 'config/default.ini').touch()
    reloaded = gpu.run(command, reload / 'train.log', 180 if sanitizer else 60, reload)
    if sanitizer:
        require('ERROR SUMMARY: 0 errors' in (reload / 'train.log').read_text(), 'Policy reload memory gate')
    weights = list((reload / 'checkpoints').rglob('*.bin'))
    require(len(weights) == 1, 'Reload canary checkpoint quota')
    values = array.array('f'); values.frombytes(weights[0].read_bytes())
    require(all(map(math.isfinite, values)) and len(values) * 4 == files[-1].stat().st_size, 'Reload weights invalid')
    return dict(process=receipt, checkpoint_sha256=[sha(p) for p in files], decision_budget=8192,
        process_sps=8192 / receipt['seconds'], frozen=bool(frozen), encoder_sha256=sha(artifact / 'encoder.bin'),
        reload_process=reloaded, reload_checkpoint_sha256=sha(weights[0]))


def bank_config(packet, bank, folder, training_decisions=8192, max_runs=3):
    entries = read(bank / 'bank.json')['variants']; first = entries[0]
    cfg = policy_config(packet, first['stride'], first['channels'],
        bank / f"s{first['stride']}-c{first['channels']}", folder)
    cfg['policy'].update(compression_bank=str(bank.resolve()), compression_variant='0')
    cfg['train']['total_timesteps'] = str(training_decisions)
    cfg['sweep'].update(metric='perf', metric_distribution='linear', goal='maximize', max_runs=str(max_runs),
        gpus='1', max_suggestion_cost='300', downsample='2', prune_pareto='1', early_stop_quantile='0.3')
    if len(entries) > 1:
        cfg['sweep.policy.compression_variant'] = dict(distribution='int_uniform', min='0', max=str(len(entries)-1), scale='auto')
    return cfg


def seal(packet, stage, result):
    # Freeze successful evidence including data, states and exported weights.
    result['receipts'] = {str(p.relative_to(packet)): sha(p) for p in sorted((packet / stage).rglob('*')) if p.is_file()}
    result.update(status='passed', source_manifest_sha256=sha(packet / 'sources.json'), learning_or_frontier_claim=False)
    save(packet / f'stages/{stage}.json', result); return result


def audit(packet, stage):
    inspect(packet); proof = read(packet / f'stages/{stage}.json')
    require(proof['status'] == 'passed' and proof['source_manifest_sha256'] == sha(packet / 'sources.json'), 'Spatial gate binding differs')
    for path, value in proof['receipts'].items():
        require(sha(packet / path) == value, 'Changed spatial receipt: ' + path)
    return proof


def qualify(packet, hardware, smoke=False):
    plan = inspect(packet); stage = 'spatial-smoke' if smoke else 'spatial-qualify'
    grid = [tuple(v) for v in plan['variants']]
    rows = []; comparisons = []; root = packet / stage
    with Reservation(packet, hardware, stage, 600 if smoke else 3600) as gpu:
        root.mkdir(); binary = ae.trainer_binary(packet)
        for stride, channels in grid:
            cases = [(160, 192, 1)] if smoke else [(160, 192, 1), (161, 193, 2), (480, 640, 1)]
            for H, W, B in cases:
                pair = []
                for repeat in range(1 if smoke else 2):
                    name = f's{stride}-c{channels}-{H}x{W}-b{B}-r{repeat}'
                    out = root / 'math' / name; out.parent.mkdir(exist_ok=True)
                    receipt = gpu.run([binary, 'spatial-math', H, W, B, stride, channels, out], out.with_suffix('.log'), 120)
                    value = read(out / 'result.json')
                    require(value['status'] == 'passed' and value['parameters'] == sum(counts(channels)), 'Spatial CUDA math')
                    pair.append(value); rows.append(dict(case=name, result=value, process=receipt))
                require(len(pair) == 1 or pair[0] == pair[1], 'Spatial math fresh-process repeat differs')
        for command in ('auxiliary', 'forage-checks'):
            receipt = gpu.run([binary, command], root / f'{command}.log', 60)
            require(json.loads((root / f'{command}.log').read_text())['status'] == 'passed', 'Loss/metric primitive check')
            rows.append(dict(case=command, process=receipt))
        # New-code vector regression, not a restart of any retained old packet.
        if not smoke:
            gpu.run([binary, 'math', 160, 192, 1, root / 'vector-regression'], root / 'vector-regression.log', 120)
        data = root / 'fixture'; ae.collect(packet, data, 1, 1, 2, 4, 1, 128 * 2**20, test=1)
        for stride, channels in grid:
            base = root / f's{stride}-c{channels}'; base.mkdir()
            cfg = recipe_for(packet, stride, channels, steps=4, batch=1)
            full = fit(packet, data, cfg, base / 'full', gpu)
            repeat = fit(packet, data, cfg, base / 'repeat', gpu)
            require([c['sha256'] for c in full['checkpoints']] == [c['sha256'] for c in repeat['checkpoints']], 'Spatial same-seed weights/moments differ')
            # Cadence is part of the strict resume contract: keep it at two.
            half_cfg = recipe_for(packet, stride, channels, steps=4, batch=1); half_cfg['autoencoder']['steps'] = '2'
            half = fit(packet, data, half_cfg, base / 'half', gpu)
            resumed = fit(packet, data, cfg, base / 'resumed', gpu, base / 'half' / half['checkpoints'][-1]['path'])
            require(resumed['checkpoints'][-1]['sha256'] == full['checkpoints'][-1]['sha256'], 'Spatial resume differs')
            artifact = base / 'encoder'; manifest = export(packet, base / 'full', artifact)
            reconstructed = reconstruction(packet, data, base / 'full', artifact, base / 'reconstruction', gpu)
            comparisons.append(dict(stride=stride, channels=channels, loss='objects', training=full, encoder=manifest,
                reconstruction=reconstructed, folder=str(base.relative_to(packet))))
            if smoke:
                plain_cfg = recipe_for(packet, stride, channels, loss_mode=0, steps=4, batch=1)
                plain = fit(packet, data, plain_cfg, base / 'plain/training', gpu)
                require(plain['checkpoints'][0]['sha256'] == full['checkpoints'][0]['sha256'], 'Plain/object initial states differ')
                plain_manifest = export(packet, base / 'plain/training', base / 'plain/encoder')
                plain_recon = reconstruction(packet, data, base / 'plain/training', base / 'plain/encoder', base / 'plain/reconstruction', gpu)
                comparisons.append(dict(stride=stride, channels=channels, loss='plain', training=plain, encoder=plain_manifest,
                    reconstruction=plain_recon, folder=str((base / 'plain').relative_to(packet))))
            policy = policy_config(packet, stride, channels, artifact, base / 'unused')
            f.write_config(base / 'policy.ini', policy)
            for frozen in (1, 0):
                log = base / f'consumer-{frozen}.log'
                receipt = gpu.run([binary, 'spatial-policy-check', base / 'policy.ini', artifact, frozen], log, 90)
                require(json.loads(log.read_text())['status'] == 'passed', 'CNN consumer boundary check')
                rows.append(dict(case=f'consumer-s{stride}-c{channels}-f{frozen}', process=receipt))
            invalid = base / 'wrong-shape'; shutil.copytree(artifact, invalid)
            m = read(invalid / 'encoder.json'); m['latent_hwc'][0] += 1; save(invalid / 'encoder.json', m)
            gpu.run([binary, 'spatial-policy-check', base / 'policy.ini', invalid, 1], base / 'wrong-shape.log', 60, expect_failure=True)
            require('incompatible spatial encoder manifest' in (base / 'wrong-shape.log').read_text(), 'Wrong-shape rejection reason')
            if smoke:
                rows.append(dict(case=f'smoke-policy-s{stride}-c{channels}', result=policy_canary(
                    packet, stride, channels, artifact, base / 'policy-smoke', gpu)))
                if (stride, channels) == grid[0]:
                    rows.append(dict(case='smoke-policy-tuned', result=policy_canary(
                        packet, stride, channels, artifact, base / 'policy-tuned', gpu, frozen=0)))
            if not smoke:
                first = policy_canary(packet, stride, channels, artifact, base / 'policy-first', gpu)
                second = policy_canary(packet, stride, channels, artifact, base / 'policy-repeat', gpu)
                require(first['checkpoint_sha256'] == second['checkpoint_sha256'], 'Native policy training repeat differs')
                rows.append(dict(case=f'policy-s{stride}-c{channels}', result=first))
                rows.append(dict(case=f'policy-tuned-s{stride}-c{channels}', result=policy_canary(
                    packet, stride, channels, artifact, base / 'policy-tuned', gpu, frozen=0)))
                sanitizer = Path(ae.os.environ['CUDA_HOME']) / 'bin/compute-sanitizer'
                require(sanitizer.is_file(), 'Existing compute-sanitizer required; change nothing')
                command = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', binary]
                for suffix, tail in (
                    ('math', ['spatial-math', 161, 193, 1, stride, channels, base / 'memory-math']),
                    ('consumer-frozen', ['spatial-policy-check', base / 'policy.ini', artifact, 1]),
                    ('consumer-tuned', ['spatial-policy-check', base / 'policy.ini', artifact, 0])):
                    log = base / f'memory-{suffix}.log'; receipt = gpu.run(command + tail, log, 180)
                    require('ERROR SUMMARY: 0 errors' in log.read_text(), 'Memory gate failed/unsupported')
                    rows.append(dict(case=f'memory-{suffix}-s{stride}-c{channels}', process=receipt))
                memory = base / 'memory-train'; memory.mkdir()
                receipt = gpu.run(command + ['train', base / 'half/recipe.ini', base / 'full/index.tsv', memory, 'None', 'v1'],
                    base / 'memory-train.log', 180)
                require('ERROR SUMMARY: 0 errors' in (base / 'memory-train.log').read_text()
                    and state(memory / 'step-000000000002.state', channels)[0]['step'] == 2, 'AE learner memory gate')
                rows.append(dict(case=f'memory-train-s{stride}-c{channels}', process=receipt))
                rows.append(dict(case=f'memory-policy-s{stride}-c{channels}', result=policy_canary(
                    packet, stride, channels, artifact, base / 'memory-policy', gpu, sanitizer=sanitizer)))
        bank = root / 'encoder-bank'; bank.mkdir(); entries = []
        for stride, channels in grid:
            artifact = bank / f's{stride}-c{channels}'
            shutil.copytree(root / f's{stride}-c{channels}/encoder', artifact)
            entries.append(dict(stride=stride, channels=channels, sha256=sha(artifact / 'encoder.bin')))
        save(bank / 'bank.json', dict(schema='spatial-compressor-bank-v1', variants=entries))
        config = bank_config(packet, bank, root / 'native-sweep')
        f.write_config(root / 'native-sweep/config/sprite_forage.ini', config)
        (root / 'native-sweep/config/default.ini').touch()
        if len(entries) > 1:
            receipt = gpu.run([binary, 'policy', 'sweep'], root / 'native-sweep/train.log', 180, root / 'native-sweep')
            log = (root / 'native-sweep/train.log').read_text()
            require(len(re.findall(r'sweep run=\d+ score=', log)) == 3 and 'marking sample bad' not in log,
                'Native bank-to-PROTEIN three-trial canary failed')
            configs = list((root / 'native-sweep/checkpoints').rglob('resolved.ini'))
            require(len(configs) == 3 and all(f.config(c)['base'].getint('seed') == 7 for c in configs), 'Native sweep changed fixed seed/quota')
            rows.append(dict(case='native-bank-protein', process=receipt, trials=3,
                variants=[f.config(c)['policy'].getint('compression_variant') for c in configs]))
        if smoke:
            write_report(packet, dict(rows=comparisons), root=root)
        return seal(packet, stage, dict(hardware=gpu.identity, cases=rows, full_qualification=not smoke))


def run(packet, hardware):
    proof = audit(packet, 'spatial-qualify'); require(hardware in proof['hardware'] and proof['full_qualification'], 'Target full qualification required')
    cfg = f.config(packet / 'recipe.ini'); grid = [tuple(v) for v in inspect(packet)['variants']]
    root = packet / 'spatial-run'; rows = []
    with Reservation(packet, hardware, 'spatial-run', cfg['experiment'].getint('max_seconds')) as gpu:
        root.mkdir(); data = root / 'data'; fr.collect(packet, data, packet / 'recipe.ini')
        bank = root / 'encoder-bank'; bank.mkdir(); entries = []
        for stride, channels in grid:
            initial_hashes = []
            for name, mode in (('plain', 0), ('objects', 1)):
                base = root / f's{stride}-c{channels}' / name
                recipe = recipe_for(packet, stride, channels, mode)
                trained = fit(packet, data, recipe, base / 'training', gpu)
                initial_hashes.append(trained['checkpoints'][0]['sha256'])
                artifact = base / 'encoder'; manifest = export(packet, base / 'training', artifact)
                report = reconstruction(packet, data, base / 'training', artifact, base / 'reconstruction', gpu)
                rows.append(dict(stride=stride, channels=channels, loss=name, training=trained,
                    encoder=manifest, reconstruction=report, folder=str(base.relative_to(packet))))
                save(root / 'progress.json', dict(rows=rows))
                if mode == 1:
                    target = bank / f's{stride}-c{channels}'; shutil.copytree(artifact, target)
                    entries.append(dict(stride=stride, channels=channels, sha256=sha(target / 'encoder.bin')))
            require(initial_hashes[0] == initial_hashes[1], 'Plain/object controls did not share initial weights')
        save(bank / 'bank.json', dict(schema='spatial-compressor-bank-v1', variants=entries))
        # Ready-to-run configs retain the per-env learner and sweep only bank ID.
        policy = bank_config(packet, bank, root / 'policy-discovery', max_runs=24)
        f.write_config(root / 'policy-config/config/sprite_forage.ini', policy)
        (root / 'policy-config/config/default.ini').touch()
        result = dict(hardware=gpu.identity, rows=rows, no_rl_quality_claim=True, classifier_fitted=False)
        write_report(packet, result)
        return seal(packet, 'spatial-run', result)


def write_report(packet, result, root=None):
    root = root or packet / 'spatial-run'
    lines = ['# Spatial autoencoder reconstruction pilot', '',
        'Full RGB -> spatial feature grid -> decoder. No classification head was trained.',
        'Object/marking errors are instance means, split by class and skin. No RL/Pareto claim.', '',
        '|Stride|Channels|Latent HWC|Loss|Updates|Train process seconds|Images/s|', '|---|---|---|---|---|---|---|']
    panels = ['<!doctype html><meta charset="utf-8"><title>Spatial reconstruction</title>',
        '<style>body{font:16px sans-serif;background:#222;color:#eee}img{max-width:46%;image-rendering:pixelated}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #777}</style>',
        '<h1>Spatial reconstruction</h1><p>Original / reconstruction. Full frames; no classifier score. Pixel errors use unclipped output; preview RGB is clipped.</p>']
    for r in result['rows']:
        t = r['training']; z = r['encoder']['latent_hwc']
        lines.append(f"|{r['stride']}|{r['channels']}|{z}|{r['loss']}|{t['checkpoints'][-1]['step']}|{t['process']['seconds']:.3f}|{t['process_images_per_second']:.2f}|")
        folder = packet / r['folder'] / 'reconstruction'
        panels.append(f"<h2>Stride {r['stride']}, channels {r['channels']}, {r['loss']}</h2>")
        panels.append('<table><tr><th>Split</th><th>Skin</th><th>Object</th><th>Instances</th><th>Object MSE</th><th>Mark MSE</th><th>Background MSE</th></tr>')
        lines += ['', f"## Stride {r['stride']}, channels {r['channels']}, {r['loss']}", '',
            '|Split|Skin|Object|Instances|Object MSE|Mark MSE|Background MSE|', '|---|---|---|---|---|---|---|']
        for m in r['reconstruction']['metrics']:
            def value(v): return 'N/A' if v is None else f'{v:.8g}'
            vals = [m['split'], m['skin'], m['name'], m['instances'], value(m['object_mse']), value(m['marking_mse']), value(m['background_mse'])]
            lines.append('|' + '|'.join(map(str, vals)) + '|')
            panels.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in vals) + '</tr>')
        panels.append('</table>')
        for original in sorted(folder.glob('*-original.ppm')):
            # Browser-friendly BMP uses no image library or dependency. PPM source
            # is kept for exact raw preview provenance.
            pair = [original, original.with_name(original.name.replace('-original', '-reconstruction'))]
            panels.append('<p>' + html.escape(original.stem) + '</p>')
            for path in pair:
                bitmap = path.with_suffix('.bmp'); ppm_to_bmp(path, bitmap)
                panels.append('<img src="' + html.escape(str(bitmap.relative_to(root)), quote=True) + '">')
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (root / 'reconstructions.html').write_text('\n'.join(panels))


def ppm_to_bmp(path, target):
    with path.open('rb') as stream:
        require(stream.readline() == b'P6\n', 'Preview format')
        W, H = map(int, stream.readline().split()); require(stream.readline() == b'255\n', 'Preview range')
        rgb = stream.read(); require(len(rgb) == W * H * 3, 'Preview size')
    stride = (W * 3 + 3) // 4 * 4; pixels = bytearray()
    for y in range(H-1, -1, -1):
        row = bytearray(rgb[y*W*3:(y+1)*W*3]); row[0::3], row[2::3] = row[2::3], row[0::3]
        pixels += row + bytes(stride - W * 3)
    header = ae.struct.pack('<2sIHHI', b'BM', 54+len(pixels), 0, 0, 54)
    header += ae.struct.pack('<IiiHHIIiiII', 40, W, H, 1, 24, 0, len(pixels), 0, 0, 0, 0)
    target.write_bytes(header + pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recipe', type=Path, default=ROOT / 'research/recipes/spatial_autoencoder.ini')
    for name in ('smoke', 'qualify', 'run', 'report'):
        p = sub.add_parser(name); p.add_argument('--packet', type=Path, required=True)
        if name != 'report': p.add_argument('--hardware', choices=('5060', '5090'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.out.resolve(), args.recipe.resolve())
    elif args.command in ('smoke', 'qualify'): result = qualify(args.packet.resolve(), args.hardware, smoke=args.command == 'smoke')
    elif args.command == 'run': result = run(args.packet.resolve(), args.hardware)
    else:
        packet = args.packet.resolve(); inspect(packet)
        result = {name: audit(packet, name) for name in ('spatial-smoke', 'spatial-qualify', 'spatial-run')
            if (packet / f'stages/{name}.json').exists()}
        result['failures'] = {p.name: read(p) for p in (packet / 'stages').glob('*.failure.json')}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
