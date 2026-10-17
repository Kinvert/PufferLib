"""Fixed-seed native Forage map-generator comparison; no Python learner."""
import array
import configparser
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/claude/cnn')
OUT = ROOT / 'build/sprite-forage-connectivity-20261016'
sys.path.insert(0, str(OUT))
from sprite_gpu_idle import require_idle

STEPS = 8388608
WORLDS = (420, 1337)
FAMILIES = ('connected', 'unchecked')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    assert config.read(path)
    return config


def prepare():
    assert not (OUT / 'plan.json').exists()
    jobs = []
    for index, family in enumerate(('connected', 'unchecked', 'unchecked', 'connected')):
        name = ('happy-otter-' if family == 'connected' else 'mystic-fern-') + str(index + 1)
        folder = OUT / name
        (folder / 'runtime/config').mkdir(parents=True)
        config = read(OUT / 'default.ini')
        config.read(OUT / 'recipe.ini')
        config['base'].update(seed='7', run_id=name, eval_episodes='0',
            load_model_path='None', checkpoint_interval='128',
            checkpoint_dir=str(folder / 'checkpoints'), log_dir=str(folder / 'logs'))
        config['train']['total_timesteps'] = str(STEPS)
        config['sprite_eval'] = dict(episodes='64', episode_offset='0', seed='420',
            slots='16', max_steps='676', output='None')
        with (folder / 'runtime/config/sprite_forage.ini').open('x') as file:
            config.write(file)
        (folder / 'runtime/config/default.ini').write_bytes((OUT / 'default.ini').read_bytes())
        jobs.append(dict(name=name, family=family,
            config_sha256=sha(folder / 'runtime/config/sprite_forage.ini')))
    source_hashes = {}
    for family in FAMILIES:
        source = OUT / (family + '-source')
        for base in ('src', 'ocean', 'research'):
            for path in (source / base).rglob('*'):
                if path.is_file() and not path.is_symlink():
                    source_hashes[str(path.relative_to(OUT))] = sha(path)
        source_hashes[str((source / 'build.sh').relative_to(OUT))] = sha(source / 'build.sh')
    save(OUT / 'plan.json', dict(schema='forage-connectivity-ablation-v1', jobs=jobs,
        steps=STEPS, fixed_training_seed=7, independent_training_seeds=1, seed_swept=False,
        evaluation_worlds=list(WORLDS), evaluation_episodes_per_world=64,
        evaluation_cap=676, evaluation='Both trained policies on both map generators',
        parameters=67968, train_timeout=45, eval_timeout=30, campaign_timeout=480,
        source_sha256=source_hashes, binary_sha256={family: sha(OUT / (family + '-puffer'))
            for family in FAMILIES}, tool_sha256=sha(Path(__file__)),
        default_sha256=sha(OUT / 'default.ini'), idle_helper_sha256=sha(OUT / 'sprite_gpu_idle.py')))
    print('Prepared four native trainings and eight exact assigned-episode evaluations.', flush=True)


def run():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert not (OUT / 'started.json').exists(), 'No automatic restart of this allocation'
    assert sha(Path(__file__)) == plan['tool_sha256']
    assert sha(OUT / 'sprite_gpu_idle.py') == plan['idle_helper_sha256']
    assert sha(OUT / 'default.ini') == plan['default_sha256']
    for path, digest in plan['source_sha256'].items():
        assert sha(OUT / path) == digest, path
    for family, digest in plan['binary_sha256'].items():
        assert sha(OUT / (family + '-puffer')) == digest
    started = time.monotonic()
    training, evaluations = [], []

    def process(label, command, cwd, timeout):
        remaining = 480 - (time.monotonic() - started)
        assert remaining > 0
        require_idle(OUT / (label + '.gpu-query.json'))
        save(OUT / (label + '.command.json'), dict(argv=command, cwd=str(cwd)))
        begin = time.monotonic()
        code, expired = None, False
        with (OUT / (label + '.log')).open('x') as file:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining),
                    env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code = result.returncode
            except subprocess.TimeoutExpired:
                expired = True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=expired, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end - begin,
            log_sha256=sha(OUT / (label + '.log')))
        save(OUT / (label + '.process.json'), receipt)
        assert code == 0 and not expired, (label, receipt)
        return receipt

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(OUT / 'started.json', dict(monotonic_started=started,
            hardware=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True)))
        for job in plan['jobs']:
            name, family = job['name'], job['family']
            folder = OUT / name
            assert sha(folder / 'runtime/config/sprite_forage.ini') == job['config_sha256']
            receipt = process(name, [str(OUT / (family + '-puffer')), 'train'],
                folder / 'runtime', 45)
            weights = sorted((folder / 'checkpoints/sprite_forage' / name).glob('*.bin'))
            assert [int(path.stem) for path in weights] == [2097152, 4194304, 6291456, STEPS]
            for weight in weights:
                assert weight.stat().st_size == 67968 * 4
                assert all(math.isfinite(x) for x in array.array('f', weight.read_bytes()))
            resolved = read(weights[-1].parent / 'resolved.ini')
            requested = read(folder / 'runtime/config/sprite_forage.ini')
            for section in ('env', 'policy', 'train', 'vec'):
                assert dict(resolved[section]) == dict(requested[section]), section
            assert resolved.getint('base', 'seed') == 7
            metrics = read(folder / 'logs/sprite_forage' / (name + '.ini'))['metrics']
            last = lambda key: float(metrics[key].split(',')[-1])
            assert last('agent_steps') == STEPS
            training.append(dict(name=name, family=family, **receipt, steps=STEPS,
                process_sps=STEPS / receipt['process_seconds'], native_uptime=last('uptime'),
                native_last_bin_sps=last('sps'), training_perf=last('env/perf'),
                training_grape_exits=last('env/successes'), parameters=67968,
                checkpoints={p.name: sha(p) for p in weights}, final_checkpoint=str(weights[-1])))
            save(OUT / 'training.json', training)

        for trained in FAMILIES:
            reference = next(row for row in training if row['family'] == trained)
            folder = OUT / reference['name']
            for generator in FAMILIES:
                for world in WORLDS:
                    label = f'eval-{trained}-on-{generator}-{world}'
                    csv_path = OUT / (label + '.csv')
                    command = [str(OUT / (generator + '-puffer')), 'sprite_eval', '--headless',
                        '--base.result_fd=0', '--base.load_model_path=' + reference['final_checkpoint'],
                        '--sprite_eval.output=' + str(csv_path), f'--sprite_eval.seed={world}']
                    receipt = process(label, command, folder / 'runtime', 30)
                    with csv_path.open() as file:
                        data = list(csv.DictReader(file))
                    assert len(data) == 64 and sorted(int(r['episode_id']) for r in data) == list(range(64))
                    assert all(1 <= int(r['decisions']) <= 676 for r in data)
                    previous = next((r for r in evaluations if r['generator'] == generator
                        and r['world'] == world), None)
                    world_rows = sorted([[r[key] for key in ('episode_id', 'env_seed', 'policy_seed',
                        'start_rng', 'start_world_hash', 'start_observation_hash')] for r in data])
                    if previous:
                        assert world_rows == previous['world_rows'], 'Unmatched evaluation worlds'
                    evaluations.append(dict(trained=trained, generator=generator, world=world,
                        episodes=64, perf=statistics.mean(float.fromhex(r['perf']) for r in data),
                        grape_exits=statistics.mean(float(r['successes']) for r in data),
                        deaths=statistics.mean(float(r['deaths']) for r in data),
                        episode_length=statistics.mean(int(r['decisions']) for r in data),
                        csv_sha256=sha(csv_path), world_rows=world_rows, **receipt))
                    save(OUT / 'evaluations.json', evaluations)

    summary = dict(training={}, evaluation=[], training_repeats_exact={},
        independent_training_seeds=1, seed_swept=False, campaign_seconds=time.monotonic() - started)
    for family in FAMILIES:
        group = [r for r in training if r['family'] == family]
        assert len(group) == 2
        same = group[0]['checkpoints'] == group[1]['checkpoints']
        summary['training_repeats_exact'][family] = same
        summary['training'][family] = dict(process_seconds_mean=statistics.mean(r['process_seconds'] for r in group),
            process_sps=2 * STEPS / sum(r['process_seconds'] for r in group),
            native_last_bin_sps=[r['native_last_bin_sps'] for r in group],
            training_perf=[r['training_perf'] for r in group])
        for generator in FAMILIES:
            group_eval = [r for r in evaluations if r['trained'] == family and r['generator'] == generator]
            summary['evaluation'].append(dict(trained=family, generator=generator, episodes=128,
                perf=statistics.mean(r['perf'] for r in group_eval),
                grape_exits=statistics.mean(r['grape_exits'] for r in group_eval),
                deaths=statistics.mean(r['deaths'] for r in group_eval)))
    summary['process_sps_change_percent'] = 100 * (summary['training']['unchecked']['process_sps']
        / summary['training']['connected']['process_sps'] - 1)
    save(OUT / 'summary.json', summary)
    assert all(summary['training_repeats_exact'].values()), 'Training repeats differ'
    (OUT / 'COMPLETE').write_text('Four trainings/eight exact evaluations; state only.\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] in (['prepare'], ['run'])
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        try:
            run()
        except BaseException as error:
            save(OUT / 'failure.json', dict(error=str(error)))
            raise
