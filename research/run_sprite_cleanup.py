"""Short serial native sprite cleanup comparisons; no Python learner."""
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
OUT = ROOT / 'build/sprite-cleanup-20261016'
GAMES = ('catch', 'platform', 'maze', 'runner', 'key', 'push')
STEPS = 8388608


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
    for game in GAMES:
        for index, version in enumerate(('before', 'after', 'after', 'before')):
            name = f'{game}-' + ('happy-otter-' if version == 'before' else 'mystic-fern-') + str(index + 1)
            folder = OUT / name
            (folder / 'runtime/config').mkdir(parents=True)
            config = read(ROOT / 'config/default.ini')
            config.read(ROOT / f'config/sprite_{game}.ini')
            config['base'].update(seed='7', run_id=name, eval_episodes='0', load_model_path='None',
                checkpoint_interval='128', checkpoint_dir=str(folder / 'checkpoints'),
                log_dir=str(folder / 'logs'))
            config['vec'].update(total_agents='256', num_buffers='1', num_threads='4')
            config['train'].update(total_timesteps=str(STEPS), horizon='64', minibatch_size='2048')
            config['sprite_eval'] = dict(episodes='32', episode_offset='0', seed='420',
                slots='16', max_steps='2048', output='None')
            with (folder / f'runtime/config/sprite_{game}.ini').open('x') as file:
                config.write(file)
            (folder / 'runtime/config/default.ini').write_bytes((ROOT / 'config/default.ini').read_bytes())
            jobs.append(dict(name=name, game=game, version=version,
                config_sha256=sha(folder / f'runtime/config/sprite_{game}.ini')))
    save(OUT / 'plan.json', dict(schema='sprite-cleanup-v1', jobs=jobs,
        state_steps=STEPS, training_seed=7, seed_swept=False, independent_training_seeds=1,
        order='before/after/after/before for each game', state_eval_worlds=[420, 1337],
        episodes_per_world=32, evaluation_cap=2048, pixel_canary_decisions=128,
        pixel_eval_world=420, pixel_eval_episodes=4, campaign_timeout=900,
        tool_sha256=sha(Path(__file__)), evaluator_sha256=sha(ROOT / 'research/sprite_suite_eval.cu'),
        default_sha256=sha(ROOT / 'config/default.ini'), binaries={f'{version}-sprite_{game}':
            sha(OUT / f'{version}-sprite_{game}') for version in ('before', 'after') for game in GAMES},
        source_manifests={version: sha(OUT / f'{version}-source.sha256') for version in ('before', 'after')}))
    print('Prepared 24 state trainings, 24 state evaluations, 12 pixel canaries and 12 pixel evaluations.', flush=True)


def run():
    plan = json.loads((OUT / 'plan.json').read_text())
    assert not (OUT / 'started.json').exists(), 'Fresh allocation only; no automatic retry'
    assert sha(Path(__file__)) == plan['tool_sha256']
    for binary, digest in plan['binaries'].items():
        assert sha(OUT / binary) == digest
    for version, digest in plan['source_manifests'].items():
        assert sha(OUT / f'{version}-source.sha256') == digest
        subprocess.run(['sha256sum', '-c', str(OUT / f'{version}-source.sha256')],
            cwd=OUT / f'{version}-source', check=True, stdout=subprocess.DEVNULL)
    began = time.monotonic()
    training, evaluations, pixel = [], [], []

    def process(label, command, cwd, timeout):
        remaining = 900 - (time.monotonic() - began)
        assert remaining > 0
        smi = '/usr/lib/wsl/lib/nvidia-smi'
        queries = []
        deadline = time.monotonic() + 5
        while True:
            active = subprocess.check_output([smi, '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            queries.append(dict(pids=active, monotonic=time.monotonic()))
            save(OUT / f'{label}.gpu-query.json', queries)
            if not active.strip():
                break
            assert time.monotonic() < deadline, 'GPU busy; no job retry'
            time.sleep(.2)
        save(OUT / f'{label}.command.json', dict(argv=command, cwd=str(cwd)))
        start = time.monotonic()
        code, expired = None, False
        with (OUT / f'{label}.log').open('x') as file:
            try:
                completed = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining), env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code = completed.returncode
            except subprocess.TimeoutExpired:
                expired = True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=expired, monotonic_started=start,
            monotonic_finished=end, process_seconds=end - start, log_sha256=sha(OUT / f'{label}.log'))
        save(OUT / f'{label}.process.json', receipt)
        assert code == 0 and not expired, (label, receipt)
        return receipt

    def eval_policy(game, version, checkpoint, folder, world, episodes, label):
        output = OUT / f'{label}.csv'
        binary = OUT / f'{version}-sprite_{game}'
        receipt = process(label, [str(binary), 'sprite_eval', '--headless',
            '--base.result_fd=0', f'--base.load_model_path={checkpoint}',
            f'--sprite_eval.output={output}', f'--sprite_eval.seed={world}',
            f'--sprite_eval.episodes={episodes}', f'--sprite_eval.slots={min(16, episodes)}'],
            folder / 'runtime', 30)
        with output.open() as file:
            data = list(csv.DictReader(file))
        assert len(data) == episodes and sorted(int(r['episode_id']) for r in data) == list(range(episodes))
        assert all(1 <= int(r['decisions']) <= 2048 for r in data)
        return dict(game=game, version=version, world=world, episodes=episodes,
            perf=statistics.mean(float.fromhex(r['perf']) for r in data),
            successes=statistics.mean(float(r['successes']) for r in data),
            deaths=statistics.mean(float(r['deaths']) for r in data),
            csv=str(output), csv_sha256=sha(output), **receipt)

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(OUT / 'started.json', dict(monotonic=began, hardware=subprocess.check_output([
            '/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version',
            '--format=csv,noheader'], text=True)))
        for job in plan['jobs']:
            name, game, version = job['name'], job['game'], job['version']
            folder = OUT / name
            assert sha(folder / f'runtime/config/sprite_{game}.ini') == job['config_sha256']
            receipt = process(name, [str(OUT / f'{version}-sprite_{game}'), 'train'], folder / 'runtime', 60)
            weights = sorted((folder / f'checkpoints/sprite_{game}' / name).glob('*.bin'))
            assert [int(p.stem) for p in weights] == [2097152, 4194304, 6291456, STEPS]
            for weight in weights:
                assert weight.stat().st_size > 0 and weight.stat().st_size % 4 == 0
                assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            requested = read(folder / f'runtime/config/sprite_{game}.ini')
            resolved = read(weights[-1].parent / 'resolved.ini')
            for section in ('env', 'policy', 'train', 'vec'):
                assert dict(requested[section]) == dict(resolved[section]), (game, section)
            metrics = read(folder / f'logs/sprite_{game}/{name}.ini')['metrics']
            last = lambda key: float(metrics[key].split(',')[-1])
            assert last('agent_steps') == STEPS
            training.append(dict(**job, **receipt, steps=STEPS, process_sps=STEPS / receipt['process_seconds'],
                native_uptime=last('uptime'), native_last_bin_sps=last('sps'),
                training_perf=last('env/perf'), parameters=weights[-1].stat().st_size // 4,
                checkpoints={p.name: sha(p) for p in weights}, final_checkpoint=str(weights[-1])))
            save(OUT / 'training.json', training)
        for game in GAMES:
            for version in ('before', 'after'):
                row = next(r for r in training if r['game'] == game and r['version'] == version)
                for world in (420, 1337):
                    label = f'eval-{game}-{version}-{world}'
                    evaluations.append(eval_policy(game, version, row['final_checkpoint'],
                        OUT / row['name'], world, 32, label))
                    save(OUT / 'evaluations.json', evaluations)
            for world in (420, 1337):
                paths = [OUT / f'eval-{game}-{version}-{world}.csv' for version in ('before', 'after')]
                def ordered(path):
                    with path.open() as file:
                        return sorted(csv.DictReader(file), key=lambda r: int(r['episode_id']))
                assert ordered(paths[0]) == ordered(paths[1]), (game, world, 'State evaluations differ')

        for game in GAMES:
            for version in ('before', 'after'):
                reference = next(r for r in training if r['game'] == game and r['version'] == version)
                name = f'{game}-pixel-{version}'
                folder = OUT / name
                (folder / 'runtime/config').mkdir(parents=True)
                config = read(OUT / reference['name'] / f'runtime/config/sprite_{game}.ini')
                config['base'].update(run_id=name, checkpoint_dir=str(folder / 'checkpoints'),
                    log_dir=str(folder / 'logs'), checkpoint_interval='0')
                config['env']['observation_mode'] = '1'
                config['vec'].update(total_agents='8', num_threads='2')
                config['train'].update(total_timesteps='128', horizon='8', minibatch_size='8')
                with (folder / f'runtime/config/sprite_{game}.ini').open('x') as file:
                    config.write(file)
                (folder / 'runtime/config/default.ini').write_bytes((ROOT / 'config/default.ini').read_bytes())
                receipt = process(name, [str(OUT / f'{version}-sprite_{game}'), 'train'], folder / 'runtime', 30)
                checkpoint = folder / f'checkpoints/sprite_{game}/{name}/0000000000000128.bin'
                values = array.array('f', checkpoint.read_bytes())
                assert values and all(math.isfinite(v) for v in values)
                pixel.append(dict(game=game, version=version, steps=128, **receipt,
                    process_sps=128 / receipt['process_seconds'], parameters=len(values),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
                save(OUT / 'pixel-canaries.json', pixel)
                result = eval_policy(game, version, checkpoint, folder, 420, 4, f'eval-{name}')
                save(OUT / f'eval-{name}.json', result)
            group = [r for r in pixel if r['game'] == game]
            assert group[0]['checkpoint_sha256'] == group[1]['checkpoint_sha256'], (game, 'pixel weights differ')

    summary = dict(games={}, campaign_seconds=time.monotonic() - began,
        independent_training_seeds=1, seed_swept=False, pixel_quality_claim=False)
    for game in GAMES:
        group = [r for r in training if r['game'] == game]
        assert all(r['checkpoints'] == group[0]['checkpoints'] for r in group)
        entry = dict(all_state_checkpoints_exact=True, state={}, pixel_canary={})
        for version in ('before', 'after'):
            runs = [r for r in group if r['version'] == version]
            ev = [r for r in evaluations if r['game'] == game and r['version'] == version]
            entry['state'][version] = dict(process_seconds_mean=statistics.mean(r['process_seconds'] for r in runs),
                process_sps=2 * STEPS / sum(r['process_seconds'] for r in runs),
                perf=statistics.mean(r['perf'] for r in ev), successes=statistics.mean(r['successes'] for r in ev),
                native_last_bin_sps=[r['native_last_bin_sps'] for r in runs])
            entry['pixel_canary'][version] = next(r for r in pixel if r['game'] == game and r['version'] == version)
        entry['state_process_sps_change_percent'] = 100 * (entry['state']['after']['process_sps']
            / entry['state']['before']['process_sps'] - 1)
        summary['games'][game] = entry
    save(OUT / 'summary.json', summary)
    (OUT / 'COMPLETE').write_text('24 state trainings/24 exact evals; 12 pixel canaries/12 exact evals.\n')
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
