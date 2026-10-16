"""Fresh native partial-state/pixel trainings, targeting five minutes each."""
import argparse
import array
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from run_forage_attention_budget import BUILD, ROOT, read, sha

SHORT = ROOT / 'build/sprite-forage-attention-budget-5060-20261015'


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    assert (SHORT / 'COMPLETE').is_file()
    short = json.loads((SHORT / 'training.json').read_text())
    binary = BUILD / 'puffer'
    digest = sha(binary)
    prior_plan = json.loads((SHORT / 'plan.json').read_text())
    for path, expected in prior_plan['main_sources'].items():
        assert sha(ROOT / path) == expected
    assert digest == prior_plan['binaries'][str(binary)]
    target = 300
    jobs = []
    for mode, name in [('partial_state', 'mystic-otter-1'), ('partial_pixels', 'happy-heron-2')]:
        reference = next(row for row in short if row['mode'] == mode)
        source = Path(reference['runtime']) / 'config/sprite_forage.ini'
        config = read(source)
        batch = config.getint('vec', 'total_agents') * config.getint('train', 'horizon')
        rate = reference['native_last_bin_sps']
        overhead = max(0, reference['process_seconds'] - reference['steps'] / rate)
        # Eight evenly spaced checkpoints; decision count fixed before launch.
        epochs = int((target - overhead) * rate // batch // 8) * 8
        assert epochs > 0
        steps = epochs * batch
        folder = out / name
        (folder / 'runtime/config').mkdir(parents=True)
        config['train']['total_timesteps'] = str(steps)
        config['base'].update(run_id=name, seed='7', load_model_path='None', eval_episodes='0',
            checkpoint_interval=str(epochs // 8), checkpoint_dir=str(folder / 'checkpoints'),
            log_dir=str(folder / 'logs'))
        with (folder / 'runtime/config/sprite_forage.ini').open('x') as file:
            config.write(file)
        shutil.copyfile(Path(reference['runtime']) / 'config/default.ini', folder / 'runtime/config/default.ini')
        jobs.append(dict(mode=mode, name=name, steps=steps, batch=batch,
            checkpoint_steps=[steps * i // 8 for i in range(1, 9)],
            parameters=reference['parameters'], target_seconds=target, reference_native_sps=rate,
            estimated_overhead_seconds=overhead, reference_config_sha256=sha(source),
            config_sha256=sha(folder / 'runtime/config/sprite_forage.ini'),
            default_sha256=sha(folder / 'runtime/config/default.ini')))
    save('plan.json', dict(schema='forage-attention-five-minute-v1', jobs=jobs,
        target_train_seconds_each=300, fixed_seed=7, seed_swept=False, fresh_random_initialization=True,
        binary=str(binary), binary_sha256=digest, source_sha256=prior_plan['main_sources'],
        short_training_sha256=sha(SHORT / 'training.json'), tool_sha256=sha(Path(__file__)),
        evaluation_worlds=[420, 1337], evaluation_episodes_per_world=32, evaluation_max_steps=512,
        reference='Existing 97.65625% full-state result; no new full-state allocation',
        budget_rule='Prior speed only, no score selection; fixed whole-rollout count; actual time reported',
        train_process_timeout=450, campaign_timeout=1020, score_or_pareto_claim=False))
    started = time.monotonic()
    save('started.json', dict(monotonic_started=started))
    training, evaluations = [], []
    smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'

    def process(label, command, cwd, timeout):
        assert sha(binary) == digest
        query = subprocess.check_output([smi, '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        save(label + '.gpu-query.json', dict(compute_pids=query, monotonic_seconds=time.monotonic()))
        assert not query.strip(), 'GPU occupied; stop without launching or retrying'
        remaining = 1020 - (time.monotonic() - started)
        assert remaining > 0
        save(label + '.command.json', command)
        begin = time.monotonic()
        with (out / (label + '.log')).open('x') as log:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining), env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=timed_out, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end - begin, log_sha256=sha(out / (label + '.log')))
        save(label + '.process.json', receipt)
        assert code == 0 and not timed_out, (label, receipt)
        return receipt

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save('hardware.json', dict(query=subprocess.check_output([smi,
            '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)))
        for job in jobs:
            folder = out / job['name']
            config_path = folder / 'runtime/config/sprite_forage.ini'
            assert sha(config_path) == job['config_sha256']
            receipt = process(job['name'], [str(binary), 'train'], folder / 'runtime', 450)
            checkpoint_dir = folder / 'checkpoints/sprite_forage' / job['name']
            weights = sorted(checkpoint_dir.glob('*.bin'))
            assert [int(w.stem) for w in weights] == job['checkpoint_steps']
            for weight in weights:
                assert weight.stat().st_size == job['parameters'] * 4
                assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            requested, resolved = read(config_path), read(checkpoint_dir / 'resolved.ini')
            for section in ('env', 'policy', 'train', 'vec'):
                assert dict(requested[section]) == dict(resolved[section])
            metrics = read(folder / 'logs/sprite_forage' / (job['name'] + '.ini'))['metrics']
            last = lambda key: float(metrics[key].split(',')[-1])
            assert last('agent_steps') == job['steps']
            uptime = last('uptime')
            row = dict(**job, **receipt, process_sps=job['steps'] / receipt['process_seconds'],
                native_uptime=uptime, native_last_bin_sps=last('sps'),
                native_average_sps=job['steps'] / uptime if 0 < uptime <= receipt['process_seconds'] else None,
                target_percent_error=100 * (receipt['process_seconds'] / target - 1),
                final_training_perf=last('env/perf'), final_training_grape_success=last('env/successes'),
                checkpoints={str(w.relative_to(out)): sha(w) for w in weights}, final_checkpoint=str(weights[-1]))
            training.append(row)
            save('training.json', training)
            for world in (420, 1337):
                label = job['name'] + '-eval-' + str(world)
                command = [str(binary), 'eval', '--headless', '--base.result_fd=0',
                    '--base.seed=7', '--base.eval_agents=1', '--base.eval_episodes=32',
                    '--vec.num_threads=1', '--vec.num_buffers=1', '--train.horizon=1',
                    '--train.minibatch_size=1', '--env.max_steps=512', f'--env.seed={world}',
                    '--sweep.metric=successes', '--base.load_model_path=' + row['final_checkpoint']]
                receipt = process(label, command, folder / 'runtime', 35)
                match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
                    r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', (out / (label + '.log')).read_text())
                assert match and int(match[3]) == 32 and int(match[4]) == job['parameters']
                evaluations.append(dict(mode=job['mode'], world=world, episodes=32,
                    perf=float(match[2]), grape_success=float(match[1]), **receipt))
                save('evaluations.json', evaluations)
    summary = dict(campaign_seconds=time.monotonic() - started, rows=[], independent_training_seeds=1,
        historical_full_state_perf=.9765625, score_or_pareto_claim=False)
    for tr in training:
        ev = [e for e in evaluations if e['mode'] == tr['mode']]
        summary['rows'].append(dict(mode=tr['mode'], steps=tr['steps'], train_seconds=tr['process_seconds'],
            process_sps=tr['process_sps'], native_average_sps=tr['native_average_sps'],
            native_last_bin_sps=tr['native_last_bin_sps'], target_percent_error=tr['target_percent_error'],
            final_training_perf=tr['final_training_perf'], parameters=tr['parameters'],
            evaluation_perf=sum(e['perf'] for e in ev) / 2,
            evaluation_grape_success=sum(e['grape_success'] for e in ev) / 2))
    save('summary.json', summary)
    (out / 'COMPLETE').write_text('Two fresh five-minute-target trainings and four evaluations complete.\n')
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
