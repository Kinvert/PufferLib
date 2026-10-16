"""Matched native partial-state controls; no Python learning or CPU neural work."""
import argparse
import array
import configparser
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from run_forage_attention_budget import ROOT, BASELINE, read, sha

STEPS = 83886080
WORLDS = (420, 1337)
WORLD_KEYS = ('episode_id', 'env_seed', 'policy_seed', 'start_rng', 'start_world_hash')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    common = read(BASELINE / 'mossy-fern-2/runtime/config/sprite_forage.ini')
    defaults = BASELINE / 'default.ini'
    jobs = []
    names = ['quiet-otter-1', 'mystic-tree-2', 'happy-heron-3',
        'sunny-gecko-4', 'gentle-fern-5', 'silver-robin-6', 'bright-maple-7']
    definitions = [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1), (2, 1, 1), (0, 0, 0)]
    world_reference = {}
    for index, (gaze, relative, partial) in enumerate(definitions):
        build = ROOT / f'build/sprite-forage-controls-build-20261015-{gaze}-{relative}'
        binary = build / 'puffer'
        assert binary.is_file()
        for local, captured in [('ocean/sprite_forage/sprite_forage.h', 'ocean/sprite_forage/sprite_forage.h'),
                ('ocean/sprite_forage/attention.h', 'ocean/sprite_forage/attention.h'),
                ('research/sprite_attention_eval.cu', 'research/sprite_exact_eval.cu')]:
            assert sha(ROOT / local) == sha(build / 'source' / captured)
        folder = out / names[index]
        (folder / 'config').mkdir(parents=True)
        config = configparser.ConfigParser(interpolation=None)
        config.read_dict({s: dict(common[s]) for s in common.sections()})
        config['env'].update(observation_mode='0', partial_observation=str(partial))
        config['train']['total_timesteps'] = str(STEPS)
        config['base'].update(seed='7', load_model_path='None', eval_episodes='0', checkpoint_interval='640')
        config['sprite_eval'] = dict(episodes='128', episode_offset='0', seed='420', slots='16',
            max_steps='512', output='None')
        config_path = folder / 'config/sprite_forage.ini'
        with config_path.open('x') as file:
            config.write(file)
        shutil.copyfile(defaults, folder / 'config/default.ini')
        action_sizes = [5, 300] if gaze == 0 else [5, 5] if gaze == 1 else [5, 20, 15]
        parameters = 91008 + (0 if partial else 120 * 128) + (sum(action_sizes) - 305) * 128
        job = dict(name=names[index], gaze_mode=gaze, relative=relative, partial=partial,
            action_sizes=action_sizes, parameters=parameters, binary=str(binary), binary_sha256=sha(binary),
            config_sha256=sha(config_path), default_sha256=sha(folder / 'config/default.ini'),
            build_flags=(build / 'attention-build.env').read_text(), manifests={})
        for world in WORLDS:
            manifest_config = read(config_path)
            manifest_config['env']['max_steps'] = '512'
            manifest_config['sprite_eval']['seed'] = str(world)
            manifest_ini = folder / f'manifest-{world}.ini'
            with manifest_ini.open('x') as file:
                manifest_config.write(file)
            command = [str(binary), 'sprite_eval_manifest', str(manifest_ini)]
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            assert result.returncode == 0, result.stderr
            path = folder / f'manifest-{world}.csv'
            path.write_text(result.stdout)
            data = rows(path)
            assert len(data) == 128 and [int(r['episode_id']) for r in data] == list(range(128))
            worlds = [[r[k] for k in WORLD_KEYS] for r in data]
            if world in world_reference:
                assert worlds == world_reference[world]
            else:
                world_reference[world] = worlds
            job['manifests'][str(world)] = dict(path=str(path), sha256=sha(path), command=command)
        jobs.append(job)
    save(out / 'plan.json', dict(schema='forage-state-controls-v1', jobs=jobs,
        fixed_train_seed=7, seed_swept=False, training_steps=STEPS, checkpoint_steps=[10485760*i for i in range(1, 9)],
        evaluation_worlds=list(WORLDS), evaluation_episodes_each=128, evaluation_slots=16,
        learner='Same historical tuned state recipe, A256/H64/MB2048/H128-L1, same decision/LR schedule',
        timing='Approximately two minutes, fixed decisions for deterministic training; actual monotonic times reported',
        sources={p: sha(ROOT / p) for p in ('ocean/sprite_forage/sprite_forage.h',
            'ocean/sprite_forage/attention.h', 'research/sprite_attention_eval.cu')},
        tool_sha256=sha(Path(__file__)), primary_n=1, quality_or_pareto_claim=False))
    print('Prepared seven jobs; all fixed episode worlds match. No GPU query or policy execution.')


def execute(out, smoke):
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['tool_sha256'] == sha(Path(__file__))
    for path, digest in plan['sources'].items():
        assert sha(ROOT / path) == digest
    run = out / ('smoke' if smoke else 'full')
    if not smoke:
        assert (out / 'smoke/COMPLETE').is_file()
    run.mkdir(exist_ok=False)
    began = time.monotonic()
    save(run / 'started.json', dict(monotonic_started=began))
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    training, evaluations = [], []

    def process(label, command, cwd, timeout):
        assert sha(Path(command[0])) == next(j['binary_sha256'] for j in plan['jobs'] if j['binary'] == command[0])
        deadline = time.monotonic() + 5
        queries = []
        while True:
            active = subprocess.check_output([smi, '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            queries.append(dict(compute_pids=active, monotonic_seconds=time.monotonic()))
            save(run / (label + '.gpu-query.json'), queries)
            if not active.strip():
                break
            assert time.monotonic() < deadline, 'GPU occupied; no policy retry'
            time.sleep(.2)
        remaining = (180 if smoke else 1500) - (time.monotonic() - began)
        assert remaining > 0
        save(run / (label + '.command.json'), command)
        begin = time.monotonic()
        with (run / (label + '.log')).open('x') as file:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining), env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=timed_out, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end-begin, log_sha256=sha(run / (label + '.log')))
        save(run / (label + '.process.json'), receipt)
        assert code == 0 and not timed_out, (label, receipt)
        return receipt

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(run / 'hardware.json', dict(query=subprocess.check_output([smi,
            '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)))
        for job in plan['jobs']:
            folder = run / job['name']
            (folder / 'runtime/config').mkdir(parents=True)
            original = out / job['name'] / 'config/sprite_forage.ini'
            assert sha(original) == job['config_sha256']
            config = read(original)
            steps = 65536 if smoke else STEPS
            config['train']['total_timesteps'] = str(steps)
            config['base'].update(run_id=job['name'], checkpoint_dir=str(folder / 'checkpoints'),
                log_dir=str(folder / 'logs'), checkpoint_interval='0' if smoke else '640')
            with (folder / 'runtime/config/sprite_forage.ini').open('x') as file:
                config.write(file)
            shutil.copyfile(out / job['name'] / 'config/default.ini', folder / 'runtime/config/default.ini')
            receipt = process(job['name'], [job['binary'], 'train'], folder / 'runtime', 20 if smoke else 210)
            weights = sorted((folder / 'checkpoints/sprite_forage' / job['name']).glob('*.bin'))
            assert [int(w.stem) for w in weights] == ([steps] if smoke else plan['checkpoint_steps'])
            for weight in weights:
                assert weight.stat().st_size == job['parameters'] * 4
                assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            resolved = read(weights[-1].parent / 'resolved.ini')
            for section in ('env', 'policy', 'train', 'vec'):
                assert dict(resolved[section]) == dict(config[section])
            metrics = read(folder / 'logs/sprite_forage' / (job['name'] + '.ini'))['metrics']
            last = lambda key: float(metrics[key].split(',')[-1])
            assert last('agent_steps') == steps
            row = dict(name=job['name'], parameters=job['parameters'], steps=steps, **receipt,
                process_sps=steps/receipt['process_seconds'], native_uptime=last('uptime'),
                native_last_bin_sps=last('sps'), final_training_perf=last('env/perf'),
                checkpoints={str(w.relative_to(out)): sha(w) for w in weights}, final_checkpoint=str(weights[-1]))
            training.append(row)
            save(run / 'training.json', training)
            modes = [('graph', 1, 0, 17), ('repeat', 1, 0, 17), ('eager', -1, 0, 17), ('tail', 1, 16, 1)] if smoke else [
                (f'world-{w}', 1, 0, 128) for w in WORLDS]
            first_data = None
            for label, graphs, first, count in modes:
                world = 420 if smoke else int(label.split('-')[-1])
                prefix = job['name'] + '-' + label
                csv_path = run / (prefix + '.csv')
                command = [job['binary'], 'sprite_eval', '--base.load_model_path=' + str(weights[-1]),
                    '--base.seed=7', '--env.seed=0', '--env.max_steps=512', f'--base.cudagraphs={graphs}',
                    f'--sprite_eval.seed={world}', f'--sprite_eval.episode_offset={first}',
                    f'--sprite_eval.episodes={count}', '--sprite_eval.slots=16', '--sprite_eval.max_steps=512',
                    '--sprite_eval.output=' + str(csv_path)]
                receipt = process(prefix, command, folder / 'runtime', 20 if smoke else 35)
                data = rows(csv_path)
                assert len(data) == count and sorted(int(r['episode_id']) for r in data) == list(range(first, first+count))
                manifest = {r['episode_id']: r for r in rows(Path(job['manifests'][str(world)]['path']))}
                for r in data:
                    assert all(r[k] == manifest[r['episode_id']][k] for k in WORLD_KEYS + ('start_observation_hash',))
                    assert 1 <= int(r['decisions']) <= 512 and 0 <= float.fromhex(r['perf']) <= 1
                if smoke:
                    ordered = sorted(data, key=lambda r: int(r['episode_id']))
                    if label == 'graph':
                        first_data = ordered
                    else:
                        assert ordered == (first_data[-1:] if label == 'tail' else first_data)
                evaluations.append(dict(name=job['name'], label=label, world=world, episodes=count,
                    perf=statistics.fmean(float.fromhex(r['perf']) for r in data),
                    grape_success=statistics.fmean(float(r['successes']) for r in data),
                    csv_path=str(csv_path), csv_sha256=sha(csv_path), **receipt))
                save(run / 'evaluations.json', evaluations)
    summary = dict(campaign_seconds=time.monotonic()-began, smoke=smoke, primary_training_n=1,
        fixed_training_seed=7, quality_or_pareto_claim=False, rows=[])
    for job, tr in zip(plan['jobs'], training):
        ev = [r for r in evaluations if r['name'] == job['name'] and (not smoke or r['label'] == 'graph')]
        summary['rows'].append(dict(name=job['name'], gaze_mode=job['gaze_mode'], relative=job['relative'],
            partial=job['partial'], parameters=job['parameters'], steps=tr['steps'], train_seconds=tr['process_seconds'],
            process_sps=tr['process_sps'], perf=statistics.fmean(r['perf'] for r in ev),
            grape_success=statistics.fmean(r['grape_success'] for r in ev)))
    save(run / 'summary.json', summary)
    (run / 'COMPLETE').write_text('All seven assigned trainings/evaluations complete.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'smoke', 'run'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        prepare(out) if args.mode == 'prepare' else execute(out, args.mode == 'smoke')
    except BaseException as error:
        folder = out if args.mode == 'prepare' else out / ('smoke' if args.mode == 'smoke' else 'full')
        if folder.is_dir():
            save(folder / 'failure.json', dict(error=repr(error)))
        raise
