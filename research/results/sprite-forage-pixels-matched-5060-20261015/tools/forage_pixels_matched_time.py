"""Five-minute native state/pixel pilots with matched partial controls and batching.

No Python learner, CPU neural reference, seed search, or image downsampling.
Calibration selects decision budgets from throughput only. Keep all checkpoints;
the primary comparison uses the checkpoint nearest 300 monotonic seconds, with
ties going to the earlier checkpoint. Never select using evaluation scores.
"""
import argparse
import array
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from forage_state_ablations import WORLD_KEYS, WORLDS, rows, save
from run_forage_attention_budget import ROOT, read, sha

BUILD = ROOT / 'build/sprite-forage-onehot-relative-build-20261015'
PREVIOUS = ROOT / 'build/sprite-forage-state-relative-5060-20261015/full/mossy-robin-4/runtime/config'
DEFINITIONS = [('mystic-otter-1', 0, 83200, 5242880),
               ('happy-heron-2', 1, 103976, 131072)]
BATCH = 64 * 64
TARGET = 300


def prepare(out):
    out.mkdir(exist_ok=False)
    # Bind the actual captured executable sources, including unchanged CNN math.
    sources = {}
    for current, captured in (
            ('ocean/sprite_forage/sprite_forage.h', 'ocean/sprite_forage/sprite_forage.h'),
            ('ocean/sprite_forage/attention.h', 'ocean/sprite_forage/attention.h'),
            ('research/forage_attention_encoder.cu', 'research/forage_attention_encoder.cu'),
            ('research/forage_attention_dispatch.cu', 'research/sprite_encoders.cu'),
            ('research/sprite_attention_eval.cu', 'research/sprite_exact_eval.cu')):
        assert sha(ROOT / current) == sha(BUILD / 'source' / captured), current
        sources[current] = sha(ROOT / current)
    old = ROOT / 'build/sprite-forage-attention-build-20261015'
    assert (old / 'math.log').read_text().count('Attention CUDA PASS') == 8
    assert sha(old / 'source/research/forage_attention_encoder.cu') == sources['research/forage_attention_encoder.cu']
    jobs, worlds = [], {}
    for name, pixels, parameters, calibration in DEFINITIONS:
        folder = out / name
        (folder / 'config').mkdir(parents=True)
        config = read(PREVIOUS / 'sprite_forage.ini')
        config['env'].update(observation_mode=str(pixels), partial_observation='1')
        config['vec'].update(total_agents='64', num_threads='4')
        config['train'].update(minibatch_size='256', horizon='64')
        config['base'].update(seed='7', load_model_path='None', eval_episodes='0')
        ini = folder / 'config/sprite_forage.ini'
        with ini.open('x') as file:
            config.write(file)
        shutil.copyfile(PREVIOUS / 'default.ini', folder / 'config/default.ini')
        manifests = {}
        for world in WORLDS:
            manifest = read(ini)
            manifest['env']['max_steps'] = '512'
            manifest['sprite_eval']['seed'] = str(world)
            path = folder / f'manifest-{world}.ini'
            with path.open('x') as file:
                manifest.write(file)
            command = [str(BUILD / 'puffer'), 'sprite_eval_manifest', str(path)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, result.stderr
            csv = folder / f'manifest-{world}.csv'
            csv.write_text(result.stdout)
            actual = rows(csv)
            assert [int(r['episode_id']) for r in actual] == list(range(128))
            keys = [[r[k] for k in WORLD_KEYS] for r in actual]
            assert keys == worlds.setdefault(world, keys)
            manifests[str(world)] = dict(path=str(csv), sha256=sha(csv), command=command)
        jobs.append(dict(name=name, pixels=pixels, parameters=parameters,
            calibration_steps=calibration, manifests=manifests,
            config_sha256=sha(ini), default_sha256=sha(folder / 'config/default.ini')))
    save(out / 'plan.json', dict(schema='forage-matched-time-v1', jobs=jobs,
        binary=str(BUILD / 'puffer'), binary_sha256=sha(BUILD / 'puffer'),
        sources=sources, tool_sha256=sha(Path(__file__)),
        build_flags=(BUILD / 'attention-build.env').read_text(),
        target_seconds=TARGET, fixed_train_seed=7, seed_swept=False, independent_training_n=1,
        common_recipe='A64/H64/MB256, four threads, sync, H128/L1, historical learner and rewards',
        historical_difference='Both new modes change A256/MB2048 to A64/MB256; horizon64 unchanged',
        images='17 full-resolution 32x32 RGB tiles, four common geometry floats; no downsampling',
        actions=[5, 20, 15], state_inputs=225, pixel_inputs=52228,
        timepoint_rule='Nearest checkpoint to 300 monotonic seconds from process start; tie earlier',
        final_checkpoint_also_reported=True, budget_selection_uses_scores=False,
        calibration_and_training_separate=True, quality_or_pareto_claim=False))
    print('PREPARED: matched worlds, 225 state / 52228 pixel inputs, three identical action heads.', flush=True)


def run(out, calibration):
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['tool_sha256'] == sha(Path(__file__))
    assert sha(Path(plan['binary'])) == plan['binary_sha256']
    for path, digest in plan['sources'].items():
        assert sha(ROOT / path) == digest
    stage = out / ('calibration' if calibration else 'full')
    stage.mkdir(exist_ok=False)
    if not calibration:
        assert (out / 'calibration/COMPLETE').is_file()
        budgets = json.loads((out / 'budgets.json').read_text())
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    started = time.monotonic()
    save(stage / 'started.json', dict(monotonic_started=started))

    def process(label, command, cwd, timeout):
        assert command[0] == plan['binary'] and sha(Path(command[0])) == plan['binary_sha256']
        query = subprocess.check_output([smi, '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        save(stage / (label + '.gpu-query.json'), dict(compute_pids=query, monotonic=time.monotonic()))
        assert not query.strip(), 'GPU occupied: stop, do not retry policies'
        save(stage / (label + '.command.json'), command)
        begin = time.monotonic()
        assert begin - started < (240 if calibration else 1100), 'Campaign deadline'
        log = stage / (label + '.log')
        with log.open('x') as file:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=timeout, env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=timed_out, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end-begin, log_sha256=sha(log))
        save(stage / (label + '.process.json'), receipt)
        assert code == 0 and not timed_out, (label, receipt)
        return receipt

    training, evaluations = [], []
    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(stage / 'hardware.json', dict(query=subprocess.check_output([smi,
            '--query-gpu=name,uuid,driver_version,memory.total,memory.used', '--format=csv,noheader'], text=True)))
        for job in plan['jobs']:
            name = job['name']
            folder = stage / name
            runtime = folder / 'runtime'
            (runtime / 'config').mkdir(parents=True)
            source = out / name / 'config/sprite_forage.ini'
            assert sha(source) == job['config_sha256']
            config = read(source)
            steps = job['calibration_steps'] if calibration else budgets[name]['steps']
            cadence = 0 if calibration else steps // BATCH // 8
            config['train']['total_timesteps'] = str(steps)
            config['base'].update(run_id=name, checkpoint_dir=str(folder / 'checkpoints'),
                log_dir=str(folder / 'logs'), checkpoint_interval=str(cadence))
            with (runtime / 'config/sprite_forage.ini').open('x') as file:
                config.write(file)
            default = out / name / 'config/default.ini'
            assert sha(default) == job['default_sha256']
            shutil.copyfile(default, runtime / 'config/default.ini')
            save(folder / 'assignment.json', dict(steps=steps, cadence=cadence, parameters=job['parameters']))
            print(f'START {stage.name} {name}: {steps} decisions', flush=True)
            receipt = process(name, [plan['binary'], 'train'], runtime, 60 if calibration else 450)
            weights = sorted((folder / 'checkpoints/sprite_forage' / name).glob('*.bin'))
            expected = [steps] if calibration else [steps*i//8 for i in range(1, 9)]
            assert [int(w.stem) for w in weights] == expected
            for weight in weights:
                assert weight.stat().st_size == job['parameters'] * 4
                assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            resolved = read(weights[-1].parent / 'resolved.ini')
            for section in ('env', 'policy', 'train', 'vec'):
                assert dict(resolved[section]) == dict(config[section]), section
            metrics = read(folder / 'logs/sprite_forage' / (name + '.ini'))['metrics']
            last = lambda key: float(metrics[key].split(',')[-1])
            assert last('agent_steps') == steps
            marks = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)',
                (stage / (name + '.log')).read_text())
            assert [int(m[0]) for m in marks] == expected
            checkpoints = []
            for weight, (count, size, stamp) in zip(weights, marks):
                elapsed = int(stamp)/1e9 - receipt['monotonic_started']
                assert 0 < elapsed <= receipt['process_seconds'] and int(size) == weight.stat().st_size
                checkpoints.append(dict(path=str(weight), sha256=sha(weight), steps=int(count),
                    elapsed_seconds=elapsed, elapsed_sps=int(count)/elapsed))
            selected = min(checkpoints, key=lambda c: (abs(c['elapsed_seconds']-TARGET), c['elapsed_seconds']))
            row = dict(name=name, pixels=job['pixels'], parameters=job['parameters'], steps=steps,
                **receipt, process_sps=steps/receipt['process_seconds'],
                native_uptime=last('uptime'), native_last_bin_sps=last('sps'),
                final_training_perf=last('env/perf'), checkpoints=checkpoints,
                selected_checkpoint=selected, final_checkpoint=checkpoints[-1])
            training.append(row)
            save(stage / 'training.json', training)
            print(f'DONE {stage.name} {name}: {receipt["process_seconds"]:.3f}s, {row["process_sps"]:.0f} SPS', flush=True)
            # All evaluations happen outside timed training. Every ID is assigned.
            selections = [('canary', checkpoints[-1])] if calibration else [('nearest300', selected)]
            if not calibration and selected['path'] != checkpoints[-1]['path']:
                selections.append(('final', checkpoints[-1]))
            for selection, checkpoint in selections:
                modes = [('graph', 1, 420, 0, 17), ('repeat', 1, 420, 0, 17),
                    ('eager', -1, 420, 0, 17), ('tail', 1, 420, 16, 1)] if calibration else [
                    (f'world-{world}', 1, world, 0, 128) for world in WORLDS]
                first = None
                for label, graph, world, offset, count in modes:
                    prefix = f'{name}-{selection}-{label}'
                    csv = stage / (prefix + '.csv')
                    command = [plan['binary'], 'sprite_eval', '--base.load_model_path=' + checkpoint['path'],
                        '--base.seed=7', '--env.seed=0', '--env.max_steps=512', f'--base.cudagraphs={graph}',
                        f'--sprite_eval.seed={world}', f'--sprite_eval.episode_offset={offset}',
                        f'--sprite_eval.episodes={count}', '--sprite_eval.slots=16',
                        '--sprite_eval.max_steps=512', '--sprite_eval.output=' + str(csv)]
                    ev_receipt = process(prefix, command, runtime, 45)
                    data = rows(csv)
                    assert sorted(int(r['episode_id']) for r in data) == list(range(offset, offset+count))
                    manifest = {r['episode_id']: r for r in rows(Path(job['manifests'][str(world)]['path']))}
                    for r in data:
                        assert all(r[k] == manifest[r['episode_id']][k] for k in WORLD_KEYS + ('start_observation_hash',))
                        assert 1 <= int(r['decisions']) <= 512 and 0 <= float.fromhex(r['perf']) <= 1
                    log = (stage / (prefix + '.log')).read_text()
                    assert f'completed={count} params={job["parameters"]} heads=3' in log
                    ordered = sorted(data, key=lambda r: int(r['episode_id']))
                    if calibration:
                        if label == 'graph':
                            first = ordered
                        else:
                            assert ordered == (first[-1:] if label == 'tail' else first)
                    evaluations.append(dict(name=name, selection=selection, label=label, world=world,
                        episodes=count, checkpoint=checkpoint['path'], csv=str(csv), csv_sha256=sha(csv),
                        perf=statistics.fmean(float.fromhex(r['perf']) for r in data),
                        grape_exits=statistics.fmean(float(r['successes']) for r in data), **ev_receipt))
                    save(stage / 'evaluations.json', evaluations)
    if calibration:
        budgets = {}
        for tr in training:
            rate = tr['native_last_bin_sps']
            assert math.isfinite(rate) and rate > 0
            overhead = max(0, tr['process_seconds'] - tr['steps']/rate)
            steps = max(8*BATCH, int((TARGET-overhead)*rate // (8*BATCH)) * (8*BATCH))
            assert steps <= (500_000_000 if not tr['pixels'] else 20_000_000)
            budgets[tr['name']] = dict(steps=steps, rate=rate, overhead_seconds=overhead,
                selected_without_scores=True, calibration_process_sha256=sha(stage / (tr['name'] + '.process.json')))
        save(out / 'budgets.json', budgets)
    summary = dict(training=training, evaluations=evaluations, campaign_seconds=time.monotonic()-started,
        fixed_seed=7, independent_training_n=1, score_or_pareto_claim=False)
    save(stage / 'summary.json', summary)
    (stage / 'COMPLETE').write_text('Every assigned training, checkpoint and exact evaluation passed.\n')
    print(f'COMPLETE {stage.name}: {len(training)} trainings / {len(evaluations)} exact evaluations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'calibrate', 'run'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        prepare(out) if args.mode == 'prepare' else run(out, args.mode == 'calibrate')
    except BaseException as error:
        if out.is_dir():
            save(out / ('failure-' + args.mode + '.json'), dict(error=repr(error)))
        raise
