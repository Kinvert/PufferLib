"""Native matched partial-state/tile-CNN pilots for the six sprite games."""
import array
import configparser
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/claude/cnn')
BUILD = ROOT / 'build/sprite-attention-20261016-v2'
OUT = ROOT / 'build/sprite-attention-pilots-20261016'
GAMES = ('catch', 'platform', 'maze', 'runner', 'key', 'push')
STEPS = 1048576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    assert config.read(path)
    return config


def rows(path):
    with path.open() as file:
        return sorted(csv.DictReader(file), key=lambda r: int(r['episode_id']))


def prepare():
    OUT.mkdir(exist_ok=False)
    jobs = []
    for game in GAMES:
        art = (BUILD / f'source/ocean/sprite_{game}/sprites.h').read_text()
        kinds = int(re.search(r'#define SPRITE_TILE_TYPES (\d+)', art)[1])
        movement = 3 if game == 'catch' else 2 if game == 'runner' else 5
        config = read(BUILD / 'runtime/config/default.ini')
        config.read(BUILD / f'runtime/config/sprite_{game}.ini')
        layers = config.getint('policy', 'num_layers')
        for pixels in (0, 1):
            name = f'{game}-' + ('happy-maple-1' if not pixels else 'mystic-otter-2')
            folder = OUT / name
            (folder / 'config').mkdir(parents=True)
            cfg = configparser.ConfigParser(interpolation=None)
            cfg.read_dict({s: dict(config[s]) for s in config.sections()})
            cfg['env'].update(observation_mode=str(pixels), partial_observation='1')
            cfg['base'].update(seed='7', eval_episodes='0', load_model_path='None', checkpoint_interval='256')
            cfg['vec'].update(total_agents='32', num_buffers='1', num_threads='2')
            cfg['train'].update(total_timesteps=str(STEPS), horizon='32', minibatch_size='256')
            cfg['sprite_eval'] = dict(episodes='32', episode_offset='0', seed='420', slots='16',
                max_steps='2048', output='None')
            with (folder / f'config/sprite_{game}.ini').open('x') as file:
                cfg.write(file)
            (folder / 'config/default.ini').write_bytes((BUILD / 'runtime/config/default.ini').read_bytes())
            input_size = 55300 if pixels else 18 * kinds + 4
            encoder = 51624 if pixels else input_size * 128
            parameters = encoder + layers * 49152 + (movement + 35 + 1) * 128
            jobs.append(dict(name=name, game=game, pixels=pixels, input_size=input_size,
                parameters=parameters, binary=str(BUILD / f'sprite_{game}'),
                binary_sha256=sha(BUILD / f'sprite_{game}'),
                config_sha256=sha(folder / f'config/sprite_{game}.ini')))
    save(OUT / 'plan.json', dict(schema='sprite-attention-pilots-v1', jobs=jobs,
        state_pixel_same='18 cells, factored movement/X/Y heads, four geometry floats, per-game learners and decisions',
        fixed_training_seed=7, independent_training_seeds=1, seed_swept=False,
        decisions=STEPS, canary_decisions=8192, canary_repeats=2,
        canary_eval_episodes=3, canary_eval_slots=2, evaluation_worlds=[420, 1337],
        final_episodes_per_world=32, final_slots=16, cap=2048,
        campaign_timeout=1500, full_train_timeout=120, canary_timeout=30, eval_timeout=45,
        checks_sha256=sha(BUILD / 'checks'), tool_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(BUILD / 'source.sha256')))
    print('Prepared 12 matched pilots, 24 repeated canary trainings, 36 exact canary evals and 24 final evals.', flush=True)


def run(resume=False):
    plan = json.loads((OUT / 'plan.json').read_text())
    if resume:
        assert (OUT / 'started.json').exists() and (OUT / 'failure.json').exists()
        assert not (OUT / 'COMPLETE').exists()
        assert sha(OUT / 'runner-at-initial-failure.py') == plan['tool_sha256']
        save(OUT / 'resume.json', dict(reason='Reporting reader lacked env/perf for an episode-free canary',
            prior_failure=json.loads((OUT / 'failure.json').read_text()),
            original_tool_sha256=plan['tool_sha256'], resumed_tool_sha256=sha(Path(__file__)),
            completed_processes_reused=True, native_failures_retried=False))
    else:
        assert not (OUT / 'started.json').exists(), 'No automatic restart'
        assert sha(Path(__file__)) == plan['tool_sha256']
    assert sha(BUILD / 'source.sha256') == plan['source_manifest_sha256']
    subprocess.run(['sha256sum', '-c', str(BUILD / 'source.sha256')], cwd=BUILD / 'source',
        check=True, stdout=subprocess.DEVNULL)
    assert sha(BUILD / 'checks') == plan['checks_sha256']
    began = json.loads((OUT / 'started.json').read_text())['monotonic'] if resume else time.monotonic()
    training, evaluations, canaries = [], [], []

    def process(label, command, cwd, timeout):
        previous = OUT / f'{label}.process.json'
        if resume and previous.exists():
            receipt = json.loads(previous.read_text())
            assert receipt['returncode'] == 0 and not receipt['timed_out'], 'Never retry failed native work'
            assert sha(OUT / f'{label}.log') == receipt['log_sha256']
            assert json.loads((OUT / f'{label}.command.json').read_text()) == dict(argv=command, cwd=str(cwd))
            return receipt
        remaining = 1500 - (time.monotonic() - began)
        assert remaining > 0
        queries = []
        deadline = time.monotonic() + 5
        while True:
            active = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            queries.append(dict(pids=active, monotonic=time.monotonic()))
            save(OUT / f'{label}.gpu-query.json', queries)
            if not active.strip():
                break
            assert time.monotonic() < deadline, 'GPU occupied; no native retry'
            time.sleep(.2)
        save(OUT / f'{label}.command.json', dict(argv=command, cwd=str(cwd)))
        start = time.monotonic()
        code, expired = None, False
        with (OUT / f'{label}.log').open('x') as file:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=min(timeout, remaining), env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code = result.returncode
            except subprocess.TimeoutExpired:
                expired = True
        finish = time.monotonic()
        receipt = dict(returncode=code, timed_out=expired, monotonic_started=start,
            monotonic_finished=finish, process_seconds=finish-start, log_sha256=sha(OUT / f'{label}.log'))
        save(OUT / f'{label}.process.json', receipt)
        assert code == 0 and not expired, (label, receipt)
        return receipt

    def train(job, label, steps):
        folder = OUT / label
        (folder / 'runtime/config').mkdir(parents=True, exist_ok=resume)
        cfg = read(OUT / job['name'] / f'config/sprite_{job["game"]}.ini')
        cfg['base'].update(run_id=label, checkpoint_dir=str(folder / 'checkpoints'),
            log_dir=str(folder / 'logs'), checkpoint_interval='256' if steps == STEPS else '0')
        cfg['train']['total_timesteps'] = str(steps)
        config_path = folder / f'runtime/config/sprite_{job["game"]}.ini'
        default_path = folder / 'runtime/config/default.ini'
        if config_path.exists():
            assert resume and {s: dict(read(config_path)[s]) for s in cfg.sections()} == {
                s: dict(cfg[s]) for s in cfg.sections()}
            assert default_path.read_bytes() == (BUILD / 'runtime/config/default.ini').read_bytes()
        else:
            with config_path.open('x') as file:
                cfg.write(file)
            default_path.write_bytes((BUILD / 'runtime/config/default.ini').read_bytes())
        receipt = process(label, [job['binary'], 'train'], folder / 'runtime', 120 if steps == STEPS else 30)
        weights = sorted((folder / f'checkpoints/sprite_{job["game"]}/{label}').glob('*.bin'))
        expected = [262144, 524288, 786432, STEPS] if steps == STEPS else [steps]
        assert [int(p.stem) for p in weights] == expected
        for path in weights:
            assert path.stat().st_size == job['parameters'] * 4
            assert all(math.isfinite(v) for v in array.array('f', path.read_bytes()))
        resolved = read(weights[-1].parent / 'resolved.ini')
        for section in ('env', 'policy', 'train', 'vec'):
            assert dict(resolved[section]) == dict(cfg[section]), (label, section)
        metrics = read(folder / f'logs/sprite_{job["game"]}/{label}.ini')['metrics']
        last = lambda key: float(metrics[key].split(',')[-1])
        assert last('agent_steps') == steps
        return dict(name=label, game=job['game'], pixels=job['pixels'], steps=steps,
            parameters=job['parameters'], process_sps=steps/receipt['process_seconds'],
            native_uptime=last('uptime'), native_last_bin_sps=last('sps'),
            training_perf=last('env/perf') if 'env/perf' in metrics else None,
            checkpoint=str(weights[-1]),
            checkpoints={p.name: sha(p) for p in weights}, **receipt)

    def evaluate(job, trained, world, label, episodes, slots, eager=False):
        output = OUT / f'{label}.csv'
        command = [job['binary'], 'sprite_eval', '--headless', '--base.result_fd=0',
            '--base.load_model_path=' + trained['checkpoint'], f'--sprite_eval.output={output}',
            f'--sprite_eval.seed={world}', f'--sprite_eval.episodes={episodes}', f'--sprite_eval.slots={slots}']
        if eager:
            command += ['--base.cudagraphs=-1']
        receipt = process(label, command, OUT / trained['name'] / 'runtime', 45)
        data = rows(output)
        assert len(data) == episodes and [int(r['episode_id']) for r in data] == list(range(episodes))
        assert all(1 <= int(r['decisions']) <= 2048 for r in data)
        return dict(name=label, game=job['game'], pixels=job['pixels'], world=world, episodes=episodes,
            perf=statistics.mean(float.fromhex(r['perf']) for r in data),
            successes=statistics.mean(float(r['successes']) for r in data),
            deaths=statistics.mean(float(r['deaths']) for r in data), csv=str(output), csv_sha256=sha(output), **receipt)

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            save(OUT / 'started.json', dict(monotonic=began, hardware=subprocess.check_output([
                '/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version',
                '--format=csv,noheader'], text=True)))
        process('encoder-math', [str(BUILD / 'checks')], ROOT, 30)
        assert (OUT / 'encoder-math.log').read_text().count('Attention CUDA PASS') == 10
        for job in plan['jobs']:
            assert sha(Path(job['binary'])) == job['binary_sha256']
            assert sha(OUT / job['name'] / f'config/sprite_{job["game"]}.ini') == job['config_sha256']
            first = train(job, job['name'] + '-canary-a', 8192)
            canaries.append(first)
            save(OUT / 'canaries.json', canaries)
            second = train(job, job['name'] + '-canary-b', 8192)
            canaries.append(second)
            save(OUT / 'canaries.json', canaries)
            assert first['checkpoints'] == second['checkpoints'], 'Canary training repeat differs'
            checks = []
            for mode in ('graph', 'repeat', 'eager'):
                label = job['name'] + '-check-' + mode
                checks.append(evaluate(job, first, 420, label, 3, 2, mode == 'eager'))
            assert rows(Path(checks[0]['csv'])) == rows(Path(checks[1]['csv'])) == rows(Path(checks[2]['csv']))
            save(OUT / (job['name'] + '-checks.json'), checks)

        for job in plan['jobs']:
            trained = train(job, job['name'] + '-pilot', STEPS)
            training.append(trained)
            save(OUT / 'training.json', training)
            for world in (420, 1337):
                evaluations.append(evaluate(job, trained, world, job['name'] + f'-world-{world}', 32, 16))
                save(OUT / 'evaluations.json', evaluations)
        for game in GAMES:
            for world in (420, 1337):
                pair = [r for r in evaluations if r['game'] == game and r['world'] == world]
                assert len(pair) == 2
                key = lambda path: [[r[k] for k in ('episode_id', 'env_seed', 'policy_seed', 'start_rng',
                    'start_world_hash')] for r in rows(Path(path))]
                assert key(pair[0]['csv']) == key(pair[1]['csv']), 'State/pixel evaluation worlds differ'

    summary = dict(games={}, campaign_seconds=time.monotonic()-began,
        independent_training_seeds=1, seed_swept=False, pareto_or_generalization_claim=False)
    for game in GAMES:
        summary['games'][game] = {}
        for pixels in (0, 1):
            trained = next(r for r in training if r['game'] == game and r['pixels'] == pixels)
            ev = [r for r in evaluations if r['game'] == game and r['pixels'] == pixels]
            summary['games'][game]['pixels' if pixels else 'state'] = dict(
                process_seconds=trained['process_seconds'], process_sps=trained['process_sps'],
                decisions=STEPS, parameters=trained['parameters'], perf=statistics.mean(r['perf'] for r in ev),
                successes=statistics.mean(r['successes'] for r in ev), deaths=statistics.mean(r['deaths'] for r in ev))
    save(OUT / 'summary.json', summary)
    (OUT / 'COMPLETE').write_text('Math, 24 canary trainings, 36 exact checks, 12 matched pilots, 24 exact evaluations complete.\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    assert sys.argv[1:] in (['prepare'], ['run'], ['resume'])
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        try:
            run(resume=sys.argv[1] == 'resume')
        except BaseException as error:
            save(OUT / 'failure.json', dict(error=str(error)))
            raise
