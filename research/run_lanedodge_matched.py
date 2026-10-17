"""Native Lane Dodge matched state/pixel pilot; fixed seeds, exact episodes."""
import array
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from run_sprite_attention import read, rows, save, sha

ROOT = Path('/home/claude/cnn')
BUILD = ROOT / 'build/sprite-lanedodge-matched-20261016'
OUT = ROOT / 'build/sprite-lanedodge-pilots-20261016'
STEPS = 1048576


def main():
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    save(OUT / 'plan.json', dict(seed=7, seed_swept=False, independent_training_seeds=1,
        modes=dict(state=4, pixels=3), decisions=STEPS, agents=32, horizon=64, minibatch=2048,
        canary_repeats=2, canary_steps=8192, eval_worlds=[420, 1337], episodes_per_world=32,
        cap=400, binary_sha256=sha(BUILD / 'puffer'), checks_sha256=sha(BUILD / 'checks'),
        tool_sha256=sha(Path(__file__)), helper_sha256=sha(ROOT / 'research/run_sprite_attention.py')))
    subprocess.run(['sha256sum', '-c', str(BUILD / 'source.sha256')], cwd=BUILD / 'source',
        check=True, stdout=subprocess.DEVNULL)
    training, evaluations = [], []

    def process(label, command, cwd, timeout):
        assert time.monotonic() - started < 360
        deadline = time.monotonic() + 5
        queries = []
        while True:
            active = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            queries.append(dict(monotonic=time.monotonic(), pids=active))
            save(OUT / f'{label}.gpu-query.json', queries)
            if not active.strip():
                break
            assert time.monotonic() < deadline, 'GPU occupied; no native retry'
            time.sleep(.2)
        save(OUT / f'{label}.command.json', dict(argv=command, cwd=str(cwd)))
        begin = time.monotonic()
        code, expired = None, False
        with (OUT / f'{label}.log').open('x') as file:
            try:
                result = subprocess.run(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                    timeout=min(timeout, 360 - (begin - started)),
                    env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
                code = result.returncode
            except subprocess.TimeoutExpired:
                expired = True
        end = time.monotonic()
        receipt = dict(returncode=code, timed_out=expired, monotonic_started=begin,
            monotonic_finished=end, process_seconds=end-begin, log_sha256=sha(OUT / f'{label}.log'))
        save(OUT / f'{label}.process.json', receipt)
        assert code == 0 and not expired, (label, receipt)
        return receipt

    def train(mode, name, steps):
        folder = OUT / name
        (folder / 'runtime/config').mkdir(parents=True)
        cfg = read(BUILD / 'runtime/config/default.ini')
        cfg.read(BUILD / 'runtime/config/sprite_lanedodge.ini')
        cfg['env']['observation_mode'] = str(mode)
        cfg['base'].update(seed='7', run_id=name, eval_episodes='0', load_model_path='None',
            checkpoint_interval='128' if steps == STEPS else '0',
            checkpoint_dir=str(folder / 'checkpoints'), log_dir=str(folder / 'logs'))
        cfg['vec'].update(total_agents='32', num_buffers='1', num_threads='2')
        cfg['train'].update(total_timesteps=str(steps), horizon='64', minibatch_size='2048')
        cfg['sprite_eval'] = dict(episodes='32', episode_offset='0', seed='420', slots='16',
            max_steps='400', output='None')
        with (folder / 'runtime/config/sprite_lanedodge.ini').open('x') as file:
            cfg.write(file)
        (folder / 'runtime/config/default.ini').write_bytes((BUILD / 'runtime/config/default.ini').read_bytes())
        receipt = process(name, [str(BUILD / 'puffer'), 'train'], folder / 'runtime', 120)
        weights = sorted((folder / f'checkpoints/sprite_lanedodge/{name}').glob('*.bin'))
        expected = [262144, 524288, 786432, STEPS] if steps == STEPS else [steps]
        assert [int(p.stem) for p in weights] == expected
        parameters = 102272 if mode == 4 else 139560
        for weight in weights:
            assert weight.stat().st_size == parameters * 4
            assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
        resolved = read(weights[-1].parent / 'resolved.ini')
        for section in ('env', 'policy', 'train', 'vec'):
            assert dict(resolved[section]) == dict(cfg[section])
        metrics = read(folder / f'logs/sprite_lanedodge/{name}.ini')['metrics']
        last = lambda key: float(metrics[key].split(',')[-1])
        assert last('agent_steps') == steps
        return dict(name=name, mode=mode, steps=steps, parameters=parameters,
            checkpoint=str(weights[-1]), checkpoints={p.name: sha(p) for p in weights},
            process_sps=steps/receipt['process_seconds'], native_uptime=last('uptime'),
            native_last_bin_sps=last('sps'), training_perf=last('env/perf') if 'env/perf' in metrics else None,
            **receipt)

    def evaluate(trained, world, label, count, slots, eager=False):
        output = OUT / f'{label}.csv'
        command = [str(BUILD / 'puffer'), 'sprite_eval', '--headless', '--base.result_fd=0',
            '--base.load_model_path=' + trained['checkpoint'], f'--sprite_eval.output={output}',
            f'--sprite_eval.seed={world}', f'--sprite_eval.episodes={count}', f'--sprite_eval.slots={slots}']
        if eager:
            command += ['--base.cudagraphs=-1']
        receipt = process(label, command, OUT / trained['name'] / 'runtime', 30)
        data = rows(output)
        assert len(data) == count and [int(r['episode_id']) for r in data] == list(range(count))
        return dict(mode=trained['mode'], world=world, episodes=count, csv=str(output),
            csv_sha256=sha(output), perf=statistics.mean(float.fromhex(r['perf']) for r in data),
            successes=statistics.mean(float(r['successes']) for r in data),
            deaths=statistics.mean(float(r['deaths']) for r in data), **receipt)

    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(OUT / 'started.json', dict(monotonic=started, hardware=subprocess.check_output([
            '/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=name,uuid,driver_version',
            '--format=csv,noheader'], text=True)))
        process('encoder-math', [str(BUILD / 'checks')], ROOT, 30)
        assert (OUT / 'encoder-math.log').read_text().count('Attention CUDA PASS') == 10
        for mode, name in ((4, 'happy-otter'), (3, 'mystic-maple')):
            pair = [train(mode, name + '-canary-' + str(i), 8192) for i in (1, 2)]
            save(OUT / f'{name}-canaries.json', pair)
            assert pair[0]['checkpoints'] == pair[1]['checkpoints']
            checks = [evaluate(pair[0], 420, f'{name}-check-{style}', 3, 2, style == 'eager')
                for style in ('graph', 'repeat', 'eager')]
            save(OUT / f'{name}-checks.json', checks)
            assert rows(Path(checks[0]['csv'])) == rows(Path(checks[1]['csv'])) == rows(Path(checks[2]['csv']))
            trained = train(mode, name + '-pilot', STEPS)
            training.append(trained)
            save(OUT / 'training.json', training)
            for world in (420, 1337):
                evaluations.append(evaluate(trained, world, f'{name}-world-{world}', 32, 16))
                save(OUT / 'evaluations.json', evaluations)
        for world in (420, 1337):
            pair = [r for r in evaluations if r['world'] == world]
            starts = lambda path: [[r[k] for k in ('episode_id', 'env_seed', 'policy_seed',
                'start_rng', 'start_world_hash')] for r in rows(Path(path))]
            assert starts(pair[0]['csv']) == starts(pair[1]['csv'])
    summary = dict(models={}, campaign_seconds=time.monotonic()-started, independent_training_seeds=1,
        seed_swept=False, frontier_claim=False)
    for trained in training:
        ev = [r for r in evaluations if r['mode'] == trained['mode']]
        summary['models']['state' if trained['mode'] == 4 else 'pixels'] = dict(
            process_seconds=trained['process_seconds'], process_sps=trained['process_sps'],
            parameters=trained['parameters'], steps=STEPS,
            perf=statistics.mean(r['perf'] for r in ev), successes=statistics.mean(r['successes'] for r in ev),
            deaths=statistics.mean(r['deaths'] for r in ev))
    save(OUT / 'summary.json', summary)
    (OUT / 'COMPLETE').write_text('Math, four canary trainings/six exact checks, two pilots/four exact evaluations complete.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.is_dir():
            save(OUT / 'failure.json', dict(error=str(error)))
        raise
