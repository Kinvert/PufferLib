"""Native Forage learning pilots targeting the measured full-state wall time.

Python only prepares configs, supervises processes and records evidence.
Calibration determines decision budgets from speed, never from policy scores.
"""
import argparse
import array
import configparser
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'build/sprite-forage-cleanup-sps-20261015'
BUILD = ROOT / 'build/sprite-forage-attention-build-20261015'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    assert config.read(path), path
    return config


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')

    baseline = json.loads((BASELINE / 'summary.json').read_text())['after']
    target = baseline['train_seconds'] / baseline['runs']
    common = read(BASELINE / 'mossy-fern-2/runtime/config/sprite_forage.ini')
    defaults = BASELINE / 'default.ini'
    assert (BUILD / 'math.log').read_text().count('Attention CUDA PASS') == 8
    pairs = {
        ROOT / 'ocean/sprite_forage/sprite_forage.h': BUILD / 'source/ocean/sprite_forage/sprite_forage.h',
        ROOT / 'ocean/sprite_forage/attention.h': BUILD / 'source/ocean/sprite_forage/attention.h',
        ROOT / 'research/forage_attention_encoder.cu': BUILD / 'source/research/forage_attention_encoder.cu',
        ROOT / 'research/forage_attention_dispatch.cu': BUILD / 'source/research/sprite_encoders.cu',
    }
    assert all(sha(a) == sha(b) for a, b in pairs.items())
    definitions = [
        dict(mode='full_state', name='happy-otter-1', binary=BASELINE / 'after-puffer',
            partial=0, pixels=0, parameters=67968, agents=256, horizon=64, minibatch=2048,
            calibration_steps=0),
        dict(mode='partial_state', name='mystic-tree-2', binary=BUILD / 'puffer',
            partial=1, pixels=0, parameters=91008, agents=256, horizon=64, minibatch=2048,
            calibration_steps=1048576),
        dict(mode='partial_pixels', name='sunny-heron-3', binary=BUILD / 'puffer',
            partial=1, pixels=1, parameters=137896, agents=8, horizon=8, minibatch=64,
            calibration_steps=16384),
    ]
    hashes = {str(d['binary']): sha(d['binary']) for d in definitions}
    save('plan.json', dict(target_process_seconds=target, fixed_train_seed=7,
        eval_worlds=[420, 1337], eval_episodes_per_world=32, eval_max_steps=512,
        definitions=[{**d, 'binary': str(d['binary'])} for d in definitions],
        baseline_summary_sha256=sha(BASELINE / 'summary.json'), binaries=hashes,
        main_sources={str(p.relative_to(ROOT)): sha(p) for p in pairs},
        tool_sha256=sha(Path(__file__)), default_sha256=sha(defaults),
        learner='Historical tuned state learner for all modes; pixels change agents/horizon/minibatch only',
        budget_rule='Subtract estimated startup from target; steady rate from last native SPS bin; round down to rollout batch',
        calibration_score_used=False, independent_training_seeds=1, seed_swept=False,
        evaluation_scope='Fixed single-slot sequential world streams, aggregate metrics; no per-episode/action trace qualification',
        train_process_timeout=45, campaign_timeout=240, score_or_pareto_claim=False))
    began = time.monotonic()
    save('started.json', dict(monotonic_started=began))
    training, evaluations, calibrations = [], [], []
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'

    def process(label, command, cwd, timeout=45):
        assert sha(Path(command[0])) == hashes[command[0]]
        queries = []
        deadline = time.monotonic() + 5
        while True:
            active = subprocess.check_output([smi, '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True)
            queries.append(dict(compute_pids=active, monotonic_seconds=time.monotonic()))
            save(label + '.gpu-query.json', queries)
            if not active.strip():
                break
            assert time.monotonic() < deadline, 'GPU occupied; stop without retry'
            time.sleep(.2)
        save(label + '.command.json', command)
        begin = time.monotonic()
        remaining = 240 - (begin - began)
        assert remaining > 0
        with (out / (label + '.log')).open('x') as log:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining),
                    env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=timed_out, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end - begin,
            log_sha256=sha(out / (label + '.log')))
        save(label + '.process.json', receipt)
        assert code == 0 and not timed_out, (label, receipt)
        return receipt

    def train(definition, steps, calibration=False):
        d = definition
        label = d['name'] + ('-calibration' if calibration else '')
        folder = out / label
        (folder / 'runtime/config').mkdir(parents=True)
        config = configparser.ConfigParser(interpolation=None)
        config.read_dict({s: dict(common[s]) for s in common.sections()})
        config['env'].update(partial_observation=str(d['partial']), observation_mode=str(d['pixels']))
        config['vec'].update(total_agents=str(d['agents']), num_threads='2' if d['pixels'] else '4')
        config['train'].update(total_timesteps=str(steps), horizon=str(d['horizon']),
            minibatch_size=str(d['minibatch']))
        batch = d['agents'] * d['horizon']
        cadence = max(1, steps // batch // 4)
        config['base'].update(seed='7', run_id=label, checkpoint_interval='0' if calibration else str(cadence),
            checkpoint_dir=str(folder / 'checkpoints'), log_dir=str(folder / 'logs'),
            load_model_path='None', eval_episodes='0')
        requested = folder / 'runtime/config/sprite_forage.ini'
        with requested.open('x') as file:
            config.write(file)
        shutil.copyfile(defaults, folder / 'runtime/config/default.ini')
        save(label + '.assignment.json', dict(steps=steps, batch=batch, checkpoint_epoch_cadence=cadence,
            requested_sha256=sha(requested), parameters=d['parameters'], calibration=calibration))
        receipt = process(label, [str(d['binary']), 'train'], folder / 'runtime')
        weights = sorted((folder / 'checkpoints/sprite_forage' / label).glob('*.bin'))
        assert weights and int(weights[-1].stem) == steps
        for weight in weights:
            assert weight.stat().st_size == d['parameters'] * 4
            assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
        resolved = read(weights[-1].parent / 'resolved.ini')
        for section in ('env', 'policy', 'vec', 'train'):
            assert dict(resolved[section]) == dict(config[section]), section
        assert resolved.getint('base', 'seed') == 7
        metrics = read(folder / 'logs/sprite_forage' / (label + '.ini'))['metrics']
        last = lambda key: float(metrics[key].split(',')[-1])
        assert last('agent_steps') == steps
        uptime = last('uptime')
        return dict(mode=d['mode'], name=label, calibration=calibration, steps=steps,
            parameters=d['parameters'], **receipt, process_sps=steps / receipt['process_seconds'],
            native_last_bin_sps=last('sps'), native_uptime=uptime,
            native_average_sps=steps / uptime if 0 < uptime <= receipt['process_seconds'] else None,
            final_training_perf=last('env/perf'), final_training_grape_success=last('env/successes'),
            target_seconds=target, target_percent_error=100 * (receipt['process_seconds'] / target - 1),
            checkpoints={str(w.relative_to(out)): sha(w) for w in weights},
            final_checkpoint=str(weights[-1]), runtime=str(folder / 'runtime'))

    def evaluate(d, trained, world, suffix=''):
        label = d['name'] + '-eval-' + str(world) + suffix
        command = [str(d['binary']), 'eval', '--headless', '--base.result_fd=0',
            '--base.seed=7', '--base.eval_agents=1', '--base.eval_episodes=32',
            '--vec.num_threads=1', '--vec.num_buffers=1', '--train.horizon=1',
            '--train.minibatch_size=1', '--env.max_steps=512', f'--env.seed={world}',
            '--sweep.metric=successes', '--base.load_model_path=' + trained['final_checkpoint']]
        receipt = process(label, command, trained['runtime'], timeout=35)
        match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
            r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', (out / (label + '.log')).read_text())
        assert match and int(match[3]) == 32 and int(match[4]) == d['parameters']
        return dict(mode=d['mode'], world=world, episodes=32, repeat=bool(suffix),
            perf=float(match[2]), grape_success=float(match[1]), **receipt)

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('hardware.json', dict(query=subprocess.check_output([smi,
            '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)))
        for d in definitions:
            steps = 8388608
            if d['calibration_steps']:
                calibration = train(d, d['calibration_steps'], calibration=True)
                rate = calibration['native_last_bin_sps']
                assert math.isfinite(rate) and rate > 0
                overhead = max(0, calibration['process_seconds'] - calibration['steps'] / rate)
                batch = d['agents'] * d['horizon']
                steps = int((target - overhead) * rate // batch) * batch
                assert batch <= steps <= (16777216 if not d['pixels'] else 262144)
                calibrations.append(dict(**calibration, estimated_overhead_seconds=overhead,
                    frozen_full_steps=steps, budget_selected_without_evaluation=True))
                save('calibrations.json', calibrations)
            trained = train(d, steps)
            training.append(trained)
            save('training.json', training)
            # Final scheduled checkpoint only; no best-checkpoint selection.
            for world in (420, 1337):
                evaluations.append(evaluate(d, trained, world))
                save('evaluations.json', evaluations)
            if d['pixels']:
                repeat = evaluate(d, trained, 420, '-repeat')
                first = next(e for e in evaluations if e['mode'] == d['mode'] and e['world'] == 420)
                evaluations.append(repeat)
                save('evaluations.json', evaluations)
                assert (repeat['perf'], repeat['grape_success']) == (first['perf'], first['grape_success'])
    summary = dict(target_seconds=target, historical_state_perf=baseline['evaluation_perf'],
        campaign_seconds=time.monotonic() - began, independent_training_seeds=1,
        score_or_pareto_claim=False, rows=[])
    for trained in training:
        ev = [e for e in evaluations if e['mode'] == trained['mode'] and not e['repeat']]
        summary['rows'].append(dict(mode=trained['mode'], steps=trained['steps'],
            train_seconds=trained['process_seconds'], process_sps=trained['process_sps'],
            native_average_sps=trained['native_average_sps'], native_last_bin_sps=trained['native_last_bin_sps'],
            target_percent_error=trained['target_percent_error'], parameters=trained['parameters'],
            evaluation_perf=sum(e['perf'] for e in ev) / len(ev),
            evaluation_grape_success=sum(e['grape_success'] for e in ev) / len(ev)))
    previous = BASELINE / 'mossy-fern-2/checkpoints/sprite_forage/mossy-fern-2/0000000008388608.bin'
    summary['full_state_final_matches_historical_bytes'] = sha(Path(training[0]['final_checkpoint'])) == sha(previous)
    save('summary.json', summary)
    (out / 'COMPLETE').write_text('Three fixed-budget learning pilots, two speed calibrations and seven evaluations complete.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    try:
        main(args.out)
    except BaseException as error:
        if args.out.is_dir():
            (args.out / 'failure.json').write_text(json.dumps(dict(error=repr(error)), indent=2) + '\n')
        raise
