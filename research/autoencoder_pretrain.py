"""Optional native full-frame autoencoder: capture, collect, train and export.

No Python neural model. Preparation/inspection never query the GPU. Collection
and pretraining are explicit commands, not side effects of building this tool.
"""
import argparse
import array
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import fullframe_compression as f

ROOT = f.ROOT
SOURCE = Path(__file__).read_bytes()
require, sha, save, read = f.require, f.sha, f.save, f.read
STATE = struct.Struct('<5Q')
MAGIC = 0x31454146465550
NATIVE = ['autoencoder.cuh', 'pretrain_autoencoder.cu', 'fullframe_compression.cu',
    'fullframe_compression_io.h', 'autoencoder_checks.cuh', 'fullframe_compression_reference.cu',
    'native_encoder_reference.cu', 'autoencoder_pretrain.py', 'autoencoder_pretrain.sh']
DECODER_PARAMETERS = 128 * 256 + 256 + 8 * (9 * 16 + 1) + 8 * (9 * 8 + 1) + 3 * (9 * 8 + 1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def prepare(out):
    require(not out.exists(), 'Fresh packet required')
    require(sys.byteorder == 'little' and os.environ.get('NVCC_ARCH') == 'sm_120', 'Need little-endian host and NVCC_ARCH=sm_120')
    require(not any(os.environ.get(k) for k in ('NVCC_EXTRA', 'NVCC_APPEND_FLAGS', 'NVCC_PREPEND_FLAGS')), 'Unrecorded compiler flags refused')
    source = out / 'source'; source.mkdir(parents=True)
    files = []
    for folder in ('src', 'ocean/connect4cnn', 'ocean/flappycnn', 'ocean/pongcnn'):
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.h', '.c', '.cu', '.sh')]
    files += [ROOT / 'research' / n for n in NATIVE]
    files += [ROOT / 'research/fullframe_compression.py', ROOT / 'research/recipes/autoencoder.ini']
    files += [ROOT / 'config/default.ini'] + [ROOT / f'config/{g}.ini' for g in f.GAMES]
    hashes = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT); target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True); hashes[str(rel)] = sha(p)
        shutil.copyfile(p, target); require(sha(target) == hashes[str(rel)], 'Source capture differs')
    external = {}
    for name in ('vendor', 'raylib-5.5_linux_amd64'):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
        for p in (ROOT / name).rglob('*'):
            if p.is_file() and p.suffix in ('.h', '.hpp', '.c', '.a'):
                external[str(p.relative_to(ROOT))] = sha(p)
    save(out / 'sources.json', hashes); save(out / 'external.json', external)
    cuda, nccl = Path(os.environ['CUDA_HOME']), Path(os.environ['NCCL_ROOT'])
    save(out / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        nvcc_sha256=sha(cuda / 'bin/nvcc'), compiler=subprocess.check_output([cuda / 'bin/nvcc', '--version'], text=True)))
    binaries = {}
    for game, (H, W, _) in f.GAMES.items():
        folder = out / 'builds' / game; folder.mkdir(parents=True)
        cmd = [str(cuda / 'bin/nvcc'), '-O2', '-std=c++17', '-arch=sm_120', '--threads', '1',
            '-Xcompiler=-fopenmp', '-Xcompiler=-Wno-narrowing', '--diag-suppress=2361', '--diag-suppress=111', '--diag-suppress=128',
            '-I.', '-Iresearch', '-Isrc', '-Ivendor', '-Iraylib-5.5_linux_amd64/include',
            '-I' + str(cuda / 'include/cccl'), '-I' + str(nccl / 'include'),
            '-DCNN_IMAGE_V1', '-DCNN_COMPRESSION_RESEARCH', f'-DCNN_IMAGE_HEIGHT={H}', f'-DCNN_IMAGE_WIDTH={W}', '-DCNN_IMAGE_CHANNELS=3',
            '-DPUFFERLIB_BUILD_MAIN', '-DPRECISION_FLOAT', '-DPUFFER_' + game.upper(),
            '-DENV_HEADER="ocean/' + game + '/' + game + '.h"', '-DPUFFER_ENV_NAME="' + game + '"',
            'research/pretrain_autoencoder.cu', 'raylib-5.5_linux_amd64/lib/libraylib.a',
            '-L' + str(cuda / 'lib64'), '-L' + str(cuda / 'lib64/stubs'), '-L' + str(nccl / 'lib'),
            '-lcudart', '-lcublas', '-lcurand', '-lcusolver', '-lnccl', '-lnvidia-ml', '-lGL', '-lpthread', '-ldl', '-lrt',
            '-o', str(folder / 'autoencoder')]
        f.process(cmd, source, folder / 'build.log', 300)
        binaries[game] = sha(folder / 'autoencoder')
        value = f.run_json([folder / 'autoencoder', 'describe'], source, folder / 'describe.log')
        require(value == dict(decoder_parameters=DECODER_PARAMETERS, encoder_parameters=36776, latent_values=128, gpu_queried=False), 'Independent parameter count differs')
        manifest = f.run_json([folder / 'autoencoder', 'manifest', H, W], source, folder / 'encoder-manifest.log')
        require(manifest['elements'] == 36776 and len(manifest['tensors']) == 8, 'Encoder registration differs')
        save(out / f'manifests/{game}.json', manifest)
        f.write_config(out / f'configs/{game}.ini', f.common_config(game, 0))
    for rel, value in hashes.items():
        require(sha(ROOT / rel) == value and sha(source / rel) == value, 'Source changed during preparation')
    save(out / 'plan.json', dict(schema='native-autoencoder-packet-v1', sources_sha256=sha(out / 'sources.json'),
        external_sha256=sha(out / 'external.json'), binaries=binaries,
        manifests={p.name: sha(p) for p in (out / 'manifests').glob('*.json')},
        configs={p.name: sha(p) for p in (out / 'configs').glob('*.ini')}, gpu_execution=False))
    return inspect(out)


def inspect(out):
    plan = read(out / 'plan.json')
    require(plan['schema'] == 'native-autoencoder-packet-v1' and sha(out / 'sources.json') == plan['sources_sha256']
        and sha(out / 'external.json') == plan['external_sha256'], 'Changed source manifest')
    for rel, value in read(out / 'sources.json').items():
        require(sha(out / 'source' / rel) == value, 'Changed captured source: ' + rel)
    for rel, value in read(out / 'external.json').items():
        require(sha(out / 'source' / rel) == value, 'Changed external input: ' + rel)
    for game, value in plan['binaries'].items():
        require(sha(out / f'builds/{game}/autoencoder') == value, 'Changed native trainer')
    for folder in ('manifests', 'configs'):
        for name, value in plan[folder].items():
            require(sha(out / folder / name) == value, 'Changed prepared ' + folder)
    return plan


def frame_hashes(path, elements):
    whole = hashlib.sha256(); frames = []
    with path.open('rb') as stream:
        while block := stream.read(elements * 4):
            require(len(block) == elements * 4, 'Truncated frame')
            whole.update(block); frames.append(hashlib.sha256(block).hexdigest())
    return whole.hexdigest(), frames


def collect(packet, out, training, validation, maximum, steps, cadence, cap):
    inspect(packet)
    require(not out.exists() and 1 <= training <= 10000 and 1 <= validation <= 10000
        and 1 <= maximum <= 1024 and 1 <= steps <= 65536 and 1 <= cadence <= steps, 'Collection bounds/fresh output')
    worst = (training + validation) * maximum * sum(H * W * 3 * 4 for H, W, _ in f.GAMES.values())
    require(worst <= cap and cap <= 256 * 2**30 and shutil.disk_usage(out.parent).free >= worst + 20 * 2**30, 'Collection exceeds disk budget/reserve')
    out.mkdir(parents=True); episodes = []
    save(out / 'collection-plan.json', dict(training_episodes=training, validation_episodes=validation,
        maximum_frames=maximum, max_steps=steps, cadence=cadence, worst_bytes=worst, byte_cap=cap,
        action_policy='fixed-random-splitmix64-v1', source_manifest_sha256=sha(packet / 'sources.json')))
    for game, (H, W, _) in f.GAMES.items():
        for split, count, base in (('train', training, 7), ('validation', validation, 420)):
            for episode in range(count):
                identity = f'{game}:{split}:{episode}'
                seed = int.from_bytes(hashlib.sha256(f'{base}:{game}:{episode}:env'.encode()).digest()[:4], 'little')
                action_seed = int.from_bytes(hashlib.sha256(f'{base}:{game}:{episode}:actions'.encode()).digest()[:8], 'little')
                skin = episode % 7
                folder = out / game / split / f'episode-{episode:06d}'; folder.mkdir(parents=True)
                path = folder / 'frames.f32'
                require(shutil.disk_usage(out).free > maximum * H * W * 12 + 20 * 2**30, 'Collection reserve reached')
                result = f.run_json([packet / f'builds/{game}/autoencoder', 'collect', packet / f'configs/{game}.ini',
                    path, seed, action_seed, skin, steps, cadence, maximum], packet / 'source', folder / 'collector.log', 60)
                file_hash, hashes = frame_hashes(path, H * W * 3)
                require(result['frames'] == len(hashes) and result['shape_hwc'] == [H, W, 3]
                    and result['cuda_execution'] is False, 'Collection receipt differs')
                episodes.append(dict(id=identity, game=game, split=split, env_seed=seed, action_seed=action_seed,
                    skin=skin, path=str(path.relative_to(out)), shape_hwc=[H, W, 3], frames=len(hashes),
                    sha256=file_hash, frame_sha256=hashes, decisions=result['decisions'], terminal=result['terminal']))
                save(out / 'progress.json', dict(episodes=episodes))
    save(out / 'dataset.json', dict(schema='native-game-frames-v1', dtype='float32', layout='NHWC', value_range=[0, 1],
        source_manifest_sha256=sha(packet / 'sources.json'), collection_plan_sha256=sha(out / 'collection-plan.json'), episodes=episodes))
    return dataset(out)


def dataset(root):
    value = read(root / 'dataset.json')
    require(value['schema'] == 'native-game-frames-v1' and value['dtype'] == 'float32'
        and value['layout'] == 'NHWC' and value['value_range'] == [0, 1], 'Dataset pixel contract')
    ids, seeds, splits, images = set(), {}, {}, {'train': set(), 'validation': set()}
    total = 0
    for ep in value['episodes']:
        require(ep['id'] not in ids and ep['split'] in images, 'Duplicate episode/invalid split')
        ids.add(ep['id']); H, W, C = ep['shape_hwc']
        require(C == 3 and 144 <= H <= 1024 and 144 <= W <= 1024 and ep['frames'] > 0, 'Dataset frame geometry')
        key = (ep['game'], ep['env_seed'])
        require(key not in seeds or seeds[key] == ep['split'], 'Environment seed leaks across splits')
        seeds[key] = ep['split']; splits.setdefault((ep['game'], H, W), set()).add(ep['split'])
        path = (root / ep['path']).resolve()
        require(path.is_relative_to(root.resolve()) and '\t' not in str(path) and '\n' not in str(path), 'Dataset path escapes root/index')
        require(path.stat().st_size == ep['frames'] * H * W * 3 * 4, 'Frame shard size differs')
        file_hash, hashes = frame_hashes(path, H * W * 3)
        require(file_hash == ep['sha256'] and hashes == ep['frame_sha256'], 'Changed frame shard')
        images[ep['split']].update(hashes); total += path.stat().st_size
    require(splits and len(splits) <= 8 and all(s == {'train', 'validation'} for s in splits.values()), 'Missing stream/split')
    return dict(manifest=value, manifest_sha256=sha(root / 'dataset.json'), bytes=total,
        exact_frame_overlap=len(images['train'] & images['validation']),
        stream_keys=sorted(splits), generalization_claim=False)


class Reservation(f.GPU):
    def __enter__(self):
        inspect(self.out)
        marker = self.out / f'stages/{self.stage}.started.json'
        require(not marker.exists(), 'Stage already started; do not retry allocations')
        lock = ROOT / 'build/connect4cnn/hardware-benchmark.lock'; lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock.open('a'); fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not self.pids(), 'Shared GPU busy')
        identity = subprocess.check_output([self.smi, '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        require(len(identity.splitlines()) == 1 and self.hardware in identity, 'GPU identity differs')
        self.identity = identity
        marker.parent.mkdir(parents=True, exist_ok=True)
        supervisor = marker.with_name(self.stage + '.supervisor.py'); supervisor.write_bytes(SOURCE)
        save(marker, dict(hardware=identity, started_ns=time.monotonic_ns(), supervisor_sha256=sha(supervisor)))
        return self


def state(path):
    raw = path.read_bytes(); require(len(raw) >= STATE.size, 'Truncated state header')
    magic, version, parameters, step, seed = STATE.unpack_from(raw)
    require(magic == MAGIC and version == 1 and parameters == 36776 + DECODER_PARAMETERS
        and len(raw) == STATE.size + parameters * 12, 'Autoencoder state layout')
    values = array.array('f'); values.frombytes(raw[STATE.size:])
    require(all(map(math.isfinite, values)), 'Nonfinite checkpoint state')
    return dict(parameters=parameters, step=step, seed=seed, sha256=sha(path)), raw


def training_contract(packet, recipe, data):
    cfg = f.config(recipe); values = dict(cfg['autoencoder'])
    for key in ('steps', 'max_seconds'):
        values.pop(key)
    return dict(recipe=values, dataset_manifest_sha256=data['manifest_sha256'],
        sources_sha256=sha(packet / 'sources.json'), binary_sha256=sha(packet / 'builds/flappycnn/autoencoder'))


def train(packet, data_root, recipe, out, hardware, resume=None, qualification_gpu=None):
    inspect(packet); require(not out.exists(), 'Fresh training output required')
    data = dataset(data_root); cfg = f.config(recipe); settings = cfg['autoencoder']
    require(settings.getint('seed') == 7 and 1 <= settings.getint('batch') <= 16
        and 1 <= settings.getint('steps') <= 10000000 and 1 <= settings.getint('max_seconds') <= 43200,
        'Pretraining seed/budget contract')
    if qualification_gpu is None:
        proof = audit_qualification(packet)
        require(hardware in proof['hardware'], 'Qualify on the target hardware before pretraining')
    else:
        require(qualification_gpu.stage == 'qualify' and qualification_gpu.out == packet,
            'Internal qualification context differs')
    contract = training_contract(packet, recipe, data)
    if resume:
        previous = read(resume.with_suffix('.json'))
        require(previous['contract'] == contract and previous['state'] == state(resume)[0], 'Resume data/optimizer/config binding differs')
    out.mkdir(parents=True)
    shutil.copyfile(recipe, out / 'recipe.ini'); save(out / 'dataset-review.json', data)
    lines = ['fullframe-autoencoder-index-v1']
    for ep in sorted(data['manifest']['episodes'], key=lambda e: e['id']):
        H, W, _ = ep['shape_hwc']; name = ep['game'] + f'-{H}x{W}'
        require(name.replace('-', '').replace('_', '').isalnum(), 'Unsafe stream name')
        lines.append('\t'.join(map(str, (ep['split'], name, H, W, ep['frames'], (data_root / ep['path']).resolve()))))
    (out / 'index.tsv').write_text('\n'.join(lines) + '\n')
    save(out / 'request.json', dict(contract=contract, packet=str(packet), recipe_sha256=sha(out / 'recipe.ini'),
        index_sha256=sha(out / 'index.tsv'), resume=str(resume) if resume else None,
        resume_sha256=sha(resume) if resume else None))
    stage = 'pretrain-' + out.name
    require(all(c.isalnum() or c in '-_.' for c in stage), 'Safe run name required')
    context = contextlib.nullcontext(qualification_gpu) if qualification_gpu else Reservation(packet, hardware, stage, settings.getint('max_seconds') + 60)
    with context as gpu:
        receipt = gpu.run([packet / 'builds/flappycnn/autoencoder', 'train', out / 'recipe.ini', out / 'index.tsv',
            out, resume or 'None', 'v1'], out / 'train.log', settings.getint('max_seconds') + 60)
    require(sha(out / 'recipe.ini') == read(out / 'request.json')['recipe_sha256']
        and sha(out / 'index.tsv') == read(out / 'request.json')['index_sha256'], 'Training inputs changed')
    dataset(data_root)  # Rebind actual frame bytes after training too.
    rows = [json.loads(line) for line in (out / 'train.log').read_text().splitlines() if line.startswith('{')]
    checkpoints = []
    for path in sorted(out.glob('step-*.state')):
        metadata = state(path)[0]
        save(path.with_suffix('.json'), dict(state=metadata, contract=contract, process=receipt))
        checkpoints.append(dict(path=path.name, **metadata))
    require(checkpoints and any(r.get('final_step') == checkpoints[-1]['step'] for r in rows), 'Missing final checkpoint/receipt')
    result = dict(status=rows[-1]['status'], hardware=gpu.identity, process=receipt, rows=rows, checkpoints=checkpoints,
        contract=contract, exact_frame_overlap=data['exact_frame_overlap'], learning_or_frontier_claim=False)
    initial_step = state(resume)[0]['step'] if resume else 0
    result['training_images'] = (checkpoints[-1]['step'] - initial_step) * settings.getint('batch')
    result['process_seconds'] = receipt['seconds']
    result['process_training_images_per_second'] = result['training_images'] / receipt['seconds']
    save(out / 'REPORT.json', result); return result


def export(packet, run, stream, out):
    inspect(packet); report = read(run / 'REPORT.json')
    require(not out.exists() and report['checkpoints'], 'Fresh encoder export required')
    request = read(run / 'request.json')
    data = read(run / 'dataset-review.json')
    require(request['contract'] == report['contract'] == training_contract(packet, run / 'recipe.ini', data)
        and sha(run / 'recipe.ini') == request['recipe_sha256']
        and sha(run / 'index.tsv') == request['index_sha256'], 'Export run/source/recipe binding differs')
    checkpoint = run / report['checkpoints'][-1]['path']; proof, raw = state(checkpoint)
    require(proof == {k: report['checkpoints'][-1][k] for k in proof}
        and read(checkpoint.with_suffix('.json'))['contract'] == report['contract']
        and read(checkpoint.with_suffix('.json'))['state'] == proof, 'Export checkpoint binding differs')
    require(proof['step'] > 0, 'An untrained step-zero state is not a pretrained encoder')
    candidates = [(g, H, W) for g, H, W in data['stream_keys'] if g + f'-{H}x{W}' == stream]
    require(len(candidates) == 1, 'Choose one recorded raw-shape stream')
    _, H, W = candidates[0]
    temporary = out.with_name(out.name + '.tmp'); require(not temporary.exists(), 'Unfinished export exists')
    temporary.mkdir(parents=True)
    manifest = f.run_json([packet / 'builds/flappycnn/autoencoder', 'manifest', H, W], packet / 'source', temporary / 'manifest.log')
    blocks = [raw[STATE.size + t['offset'] * 4:STATE.size + (t['offset'] + t['elements']) * 4] for t in manifest['tensors']]
    payload = b''.join(blocks); require(len(payload) == manifest['elements'] * 4, 'Native encoder extraction differs')
    (temporary / 'encoder.bin').write_bytes(payload)
    manifest.update(sha256=sha(temporary / 'encoder.bin'), initialization='autoencoder-reconstruction-v1',
        source_checkpoint_sha256=proof['sha256'], source_manifest_sha256=sha(packet / 'sources.json'),
        pretraining_contract=report['contract'], step=proof['step'])
    save(temporary / 'encoder.json', manifest); temporary.rename(out); return manifest


def qualify(packet, hardware):
    inspect(packet)
    progress = dict(math=[], auxiliary=[], runs=[], imports=[], memory=[])
    with Reservation(packet, hardware, 'qualify', 300 if hardware == '5060' else 900) as gpu:
        for game, (H, W, _) in f.GAMES.items():
            binary = packet / f'builds/{game}/autoencoder'
            cases = [(h, w, b) for h, w in ((160, 192), (161, 193), (H, W)) for b in (1, 2)]
            for h, w, batch in cases:
                pair = []
                for repeat in range(2):
                    name = f'{game}-h{h}-w{w}-b{batch}-r{repeat}'
                    folder = packet / f'qualification/math/{name}'; folder.parent.mkdir(parents=True, exist_ok=True)
                    receipt = gpu.run([binary, 'math', h, w, batch, folder], packet / f'logs/qualify/{name}.log', 60)
                    value = read(folder / 'result.json')
                    require(value['status'] == 'passed' and value['parameters'] == 36776 + DECODER_PARAMETERS
                        and value['eager_graph_repeats'] == 3, 'Incomplete CUDA math case')
                    pair.append(value); progress['math'].append(dict(name=name, result=value, process=receipt,
                        result_sha256=sha(folder / 'result.json')))
                    save(packet / 'qualification-progress.json', progress)
                require(pair[0] == pair[1], 'CUDA math fresh-process results differ')
            receipt = gpu.run([binary, 'auxiliary'], packet / f'logs/qualify/{game}-auxiliary.log', 30)
            value = json.loads((packet / f'logs/qualify/{game}-auxiliary.log').read_text())
            require(value['status'] == 'passed' and value['inactive_tail_checked'], 'MSE/Adam qualification failed')
            progress['auxiliary'].append(dict(game=game, result=value, process=receipt))
        save(packet / 'qualification-progress.json', progress)

        # Tiny explicit test fixture, not a pretraining corpus. Raw native frames,
        # both aspect ratios, disjoint episodes, and a real shard-reader canary.
        fixture = packet / 'qualification/fixture'
        collect(packet, fixture, 1, 1, 2, 4, 1, 128 * 2**20)
        cfg = f.config(packet / 'source/research/recipes/autoencoder.ini')
        cfg['autoencoder'].update(batch='1', steps='8', checkpoint_interval='4', max_seconds='60')
        recipe = packet / 'qualification/canary.ini'; f.write_config(recipe, cfg)
        for name in ('full', 'repeat'):
            folder = packet / 'qualification' / name
            value = train(packet, fixture, recipe, folder, hardware, qualification_gpu=gpu)
            require(value['status'] == 'complete', 'Canary budget exhausted')
            progress['runs'].append(dict(name=name, report=value, report_sha256=sha(folder / 'REPORT.json')))
        require([c['sha256'] for c in progress['runs'][0]['report']['checkpoints']] ==
            [c['sha256'] for c in progress['runs'][1]['report']['checkpoints']], 'Canary fresh-process state differs')
        cfg['autoencoder']['steps'] = '4'
        half_recipe = packet / 'qualification/half.ini'; f.write_config(half_recipe, cfg)
        half = packet / 'qualification/half'
        value = train(packet, fixture, half_recipe, half, hardware, qualification_gpu=gpu)
        require(value['status'] == 'complete', 'Half canary incomplete')
        progress['runs'].append(dict(name='half', report=value, report_sha256=sha(half / 'REPORT.json')))
        resumed = packet / 'qualification/resumed'
        value = train(packet, fixture, recipe, resumed, hardware, half / 'step-000000000004.state', qualification_gpu=gpu)
        require(value['status'] == 'complete' and value['checkpoints'][-1]['sha256'] ==
            progress['runs'][0]['report']['checkpoints'][-1]['sha256'], 'Resume weights/moments/data-order differ')
        progress['runs'].append(dict(name='resumed', report=value, report_sha256=sha(resumed / 'REPORT.json')))
        full = progress['runs'][0]['report']
        for game, (H, W, _) in f.GAMES.items():
            stream = game + f'-{H}x{W}'
            values = [r['mse'] for r in full['rows'] if r.get('validation') == stream]
            require(len(values) >= 2 and values[-1] < values[0], 'Canary reconstruction did not improve')
            directory = packet / f'qualification/exports/{game}'
            manifest = export(packet, packet / 'qualification/full', stream, directory)
            log = packet / f'logs/qualify/{game}-import.log'
            receipt = gpu.run([packet / f'builds/{game}/autoencoder', 'import-check', packet / f'configs/{game}.ini', directory], log, 30)
            require(json.loads(log.read_text()) == dict(status='passed', core_head_unchanged=True, rng_unchanged=True, encoder_exact=True),
                'Native autoencoder import proof differs')
            progress['imports'].append(dict(game=game, manifest=manifest, process=receipt))
            invalid = packet / f'qualification/invalid/{game}'; shutil.copytree(directory, invalid)
            value = read(invalid / 'encoder.json'); value['width'] += 1; save(invalid / 'encoder.json', value)
            log = packet / f'logs/qualify/{game}-reject.log'
            receipt = gpu.run([packet / f'builds/{game}/autoencoder', 'import-check', packet / f'configs/{game}.ini', invalid], log, 30, expect_failure=True)
            require('incompatible encoder manifest' in log.read_text(), 'Wrong-shape import failed for unexpected reason')
            progress['imports'].append(dict(game=game, rejected=True, process=receipt))
        save(packet / 'qualification-progress.json', progress)
        # Memory instrumentation is mandatory for enabling the optional long job.
        # Keep useful short canary receipts even when WSL cannot initialize it.
        sanitizer = Path(os.environ['CUDA_HOME']) / 'bin/compute-sanitizer'
        require(sanitizer.is_file(), 'Existing compute-sanitizer required; install/change nothing')
        for game in f.GAMES:
            folder = packet / f'qualification/memcheck/{game}'; folder.parent.mkdir(parents=True, exist_ok=True)
            log = packet / f'logs/qualify/{game}-memcheck.log'
            receipt = gpu.run([sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', packet / f'builds/{game}/autoencoder',
                'math', 161, 193, 1, folder], log, 120)
            require('ERROR SUMMARY: 0 errors' in log.read_text() and read(folder / 'result.json')['status'] == 'passed',
                'Device memory gate failed/unsupported')
            progress['memory'].append(dict(game=game, process=receipt))
            save(packet / 'qualification-progress.json', progress)
        cfg['autoencoder']['steps'] = '2'
        memory_recipe = packet / 'qualification/memory-train.ini'; f.write_config(memory_recipe, cfg)
        folder = packet / 'qualification/memory-train'; folder.mkdir()
        log = packet / 'logs/qualify/train-memcheck.log'
        receipt = gpu.run([sanitizer, '--tool', 'memcheck', '--error-exitcode', '99',
            packet / 'builds/flappycnn/autoencoder', 'train', memory_recipe,
            packet / 'qualification/full/index.tsv', folder, 'None', 'v1'], log, 120)
        require('ERROR SUMMARY: 0 errors' in log.read_text()
            and state(folder / 'step-000000000002.state')[0]['step'] == 2,
            'Mixed-shape training memory gate failed')
        progress['memory'].append(dict(game='mixed-shape-training', process=receipt))
        save(packet / 'qualification-progress.json', progress)
        value = dict(status='passed', hardware=gpu.identity, evidence=progress,
            source_manifest_sha256=sha(packet / 'sources.json'), no_long_pretraining=True, learning_or_frontier_claim=False)
        save(packet / 'stages/qualify.json', value); return value


def audit_qualification(packet):
    inspect(packet)
    require(not (packet / 'stages/qualify.failure.json').exists(), 'Qualification failed; preserve its evidence')
    path = packet / 'stages/qualify.json'
    require(path.exists(), 'Native autoencoder qualification required')
    proof = read(path)
    require(proof['status'] == 'passed' and proof['source_manifest_sha256'] == sha(packet / 'sources.json'),
        'Qualification source/status binding differs')
    started = read(packet / 'stages/qualify.started.json')
    require(sha(packet / 'stages/qualify.supervisor.py') == started['supervisor_sha256'], 'Qualification supervisor changed')
    evidence = proof['evidence']
    require(len(evidence['math']) == 24 and len(evidence['auxiliary']) == 2 and len(evidence['runs']) == 4
        and len(evidence['imports']) == 4 and len(evidence['memory']) == 3, 'Qualification coverage missing')
    for row in evidence['math']:
        file = packet / f'qualification/math/{row["name"]}/result.json'
        require(read(file) == row['result'] and sha(file) == row['result_sha256'], 'Changed numerical evidence')
    for row in evidence['runs']:
        directory = packet / f'qualification/{row["name"]}'
        require(sha(directory / 'REPORT.json') == row['report_sha256'], 'Changed canary report')
        for checkpoint in row['report']['checkpoints']:
            current = state(directory / checkpoint['path'])[0]
            require(all(current[k] == checkpoint[k] for k in current), 'Changed canary checkpoint/moments')
    require({r['game'] for r in evidence['memory']} == set(f.GAMES) | {'mixed-shape-training'}, 'Missing memory suite')
    for row in evidence['memory']:
        name = 'train' if row['game'] == 'mixed-shape-training' else row['game']
        log = packet / f'logs/qualify/{name}-memcheck.log'
        require('ERROR SUMMARY: 0 errors' in log.read_text(), 'Memory instrumentation did not pass')
    for log in (packet / 'logs/qualify').glob('*.process.json'):
        value = read(log); source = log.with_suffix('').with_suffix('.log')
        require(sha(source) == value['log_sha256'] and not value['timeout'] and not value['interference']
            and (value['returncode'] != 0 if value['expected_failure'] else value['returncode'] == 0), 'Failed/changed native qualification process')
    return proof


def report(packet):
    inspect(packet)
    proof = packet / 'stages/qualify.json'; failure = packet / 'stages/qualify.failure.json'
    progress = read(packet / 'qualification-progress.json') if (packet / 'qualification-progress.json').exists() else {}
    if proof.exists():
        audit_qualification(packet)
    result = dict(status='failed' if failure.exists() else 'passed' if proof.exists() else 'incomplete',
        progress=progress, failure=read(failure) if failure.exists() else None,
        source_manifest_sha256=sha(packet / 'sources.json'), learning_or_frontier_claim=False)
    save(packet / 'REPORT.json', result)
    lines = ['# Native autoencoder preparation/qualification', '', f'Status: **{result["status"]}**.', '',
        f'Encoder:36,776parameters;decoder:{DECODER_PARAMETERS};latent128values. Full frames, native CUDA, no Torch.',
        f'CUDA math workers:{len(progress.get("math", []))};short canary training processes:{len(progress.get("runs", []))};memory suites:{len(progress.get("memory", []))}.',
        '', 'These are correctness/short reconstruction checks, not evidence of improved RL or generalization.',
        'Failed/unsupported memory instrumentation leaves long pretraining disabled.']
    (packet / 'REPORT.md').write_text('\n'.join(lines) + '\n'); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'inspect', 'collect', 'dataset', 'qualify', 'train', 'export', 'report'))
    parser.add_argument('--packet', type=Path); parser.add_argument('--out', type=Path)
    parser.add_argument('--dataset', type=Path); parser.add_argument('--run', type=Path)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'research/recipes/autoencoder.ini')
    parser.add_argument('--hardware', choices=('5060', '5090')); parser.add_argument('--resume', type=Path)
    parser.add_argument('--stream'); parser.add_argument('--training-episodes', type=int, default=16)
    parser.add_argument('--validation-episodes', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=32); parser.add_argument('--max-steps', type=int, default=256)
    parser.add_argument('--cadence', type=int, default=4); parser.add_argument('--max-gib', type=int, default=8)
    args = parser.parse_args()
    for name in ('packet', 'out', 'dataset', 'run', 'recipe', 'resume'):
        if getattr(args, name): setattr(args, name, getattr(args, name).resolve())
    if args.command == 'prepare':
        require(args.out is not None, 'Need --out'); result = prepare(args.out)
    elif args.command == 'dataset':
        require(args.dataset is not None, 'Need --dataset'); result = dataset(args.dataset)
    else:
        require(args.packet is not None, 'Need --packet')
        if args.command == 'inspect': result = inspect(args.packet)
        elif args.command == 'qualify':
            require(args.hardware is not None, 'Need --hardware'); result = qualify(args.packet, args.hardware)
        elif args.command == 'report': result = report(args.packet)
        elif args.command == 'collect':
            require(args.out is not None, 'Need --out')
            result = collect(args.packet, args.out, args.training_episodes, args.validation_episodes,
                args.max_frames, args.max_steps, args.cadence, args.max_gib * 2**30)
        elif args.command == 'train':
            require(args.dataset is not None and args.out is not None and args.hardware is not None, 'Need --dataset --out --hardware')
            result = train(args.packet, args.dataset, args.recipe, args.out, args.hardware, args.resume)
        elif args.command == 'export':
            require(args.run is not None and args.stream is not None and args.out is not None, 'Need --run --stream --out')
            result = export(args.packet, args.run, args.stream, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
