"""Bounded native all-sprite PROTEIN search; Python only supervises/audits processes."""
import argparse
import array
import configparser
import csv
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

import sprite_pixel_benchmarks as packet

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'research/recipes/sprite_overnight.ini'
SEEDS = packet.TRAINING_SEEDS
GAMES = packet.GAMES
FAMILIES = packet.FAMILIES


def sha(path):
    return packet.sha(Path(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integers(text):
    return [int(s.strip()) for s in text.split(',')]


def settings(recipe, mode):
    cfg = packet.read_config(recipe)
    require(integers(cfg['campaign']['training_seeds']) == list(SEEDS), 'Never sweep/change fixed replicas')
    require(cfg.getint('campaign', 'checkpoint_count') == 8, 'Expected eight curve checkpoints')
    value = dict(seconds=cfg.getint('campaign', 'seconds'), trials=cfg.getint('campaign', 'max_trials'),
        decisions=cfg.getint('campaign', 'decisions'), seeds=list(SEEDS),
        baselines=cfg['campaign']['baselines'].split(','), checkpoint_count=8,
        eval_seed=cfg.getint('evaluation', 'seed'), episodes=cfg.getint('evaluation', 'episodes'),
        slots=cfg.getint('evaluation', 'slots'), skins=integers(cfg['evaluation']['skins']),
        max_steps=cfg.getint('evaluation', 'max_steps'),
        min_free_gib=cfg.getint('campaign', 'min_free_gib'),
        max_artifact_gib=cfg.getint('campaign', 'max_artifact_gib'),
        offset=cfg.getfloat('objective', 'offset'), target=cfg.getfloat('objective', 'target'),
        timeouts={k: int(v) for k, v in cfg['timeouts'].items()})
    require(0 < value['seconds'] <= 43200 and 1 <= value['trials'] <= 128, 'Unbounded campaign')
    require(value['decisions'] >= 8192 and value['decisions'] % 8192 == 0, 'Unaligned decision budget')
    require(value['offset'] == 0 and value['target'] == 1, 'Unexpected objective anchors')
    require(value['skins'] == [0, 1, 2] and value['eval_seed'] not in SEEDS, 'Wrong evaluation assignment')
    require(1 <= value['slots'] <= 64 and value['episodes'] >= value['slots'], 'Wrong evaluation quotas')
    require(all(0 < v <= 1800 for v in value['timeouts'].values()), 'Unbounded process timeout')
    require(value['baselines'] == ['tile', 'nature', 'impala', 'impoola'], 'Missing baseline')
    if mode == 'smoke':
        value.update(seconds=900, trials=2, decisions=8192, seeds=[7], baselines=['tile', 'nature'],
            checkpoint_count=1, episodes=3, slots=2)
        value['timeouts'] = dict(training=45, evaluation=30, numerical=120, proposal=60)
    return value


def prepare(args):
    out, build = args.out.resolve(), args.build.resolve()
    require(not out.exists(), 'Use a fresh campaign directory')
    cfg = settings(args.recipe, args.mode)
    out.mkdir(parents=True)
    (out / 'recipes').mkdir()
    shutil.copyfile(args.recipe, out / 'campaign.ini')
    architecture = packet.read_config(packet.RECIPE)
    architecture['policy'].update(hidden_size='128', num_layers='1')
    architecture['protein'] = dict(packet.read_config(args.recipe)['protein'])
    packet.write_config(out / 'optimizer.ini', architecture)
    frozen = {}
    for game in GAMES:
        recipe = packet.read_config(build / 'runtime/config/default.ini',
            build / f'runtime/config/sprite_{game}.ini')
        packet.remove_sweeps(recipe)
        recipe['env'].update(observation_mode='1', partial_observation='1', representation='0')
        recipe['base'].update({'async': '0', 'eval_episodes': '0', 'result_fd': '0',
            'load_model_path': 'None', 'cudagraphs': '1'})
        recipe['vec'].update(total_agents='32', num_buffers='1', num_threads='2',
            num_policies='1', hist_policy_percent='0')
        recipe['train'].update(gpus='1', horizon='32', minibatch_size='32')
        recipe['selfplay']['enabled'] = '0'
        recipe['sweep'] = dict(metric='perf', metric_distribution='linear', goal='maximize',
            max_runs='1', downsample='8', gpus='1', max_suggestion_cost='600')
        recipe['sprite_eval'] = dict(seed=str(cfg['eval_seed']), episodes=str(cfg['episodes']),
            slots=str(cfg['slots']), episode_offset='0', max_steps=str(cfg['max_steps']), output='None')
        file = out / f'recipes/{game}.ini'
        packet.write_config(file, recipe)
        frozen[game] = sha(file)
    plan = dict(schema='sprite-overnight-v1', mode=args.mode, build=str(build),
        settings=cfg, games=list(GAMES), recipes=frozen, prepared_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        seeds_swept=False, complete_training_replicas_required=True, architecture_dimensions=18,
        gpu_executed=False, held_out_confirmation=False, scientific_claim=False)
    files = [Path(__file__), ROOT / 'research/sprite_pixel_benchmarks.py',
        out / 'campaign.ini', out / 'optimizer.ini', build / 'source.sha256', build / 'packed-checks',
        build / 'packed-checks.sha256']
    files += [build / f'sprite_{g}' for g in GAMES]
    plan['inputs'] = {str(p.resolve()): sha(p) for p in files}
    describe = packet.host_call(build / 'sprite_catch', 'research_protein_describe', out / 'optimizer.ini')
    plan['optimizer_description'] = describe
    require(json.loads(describe['stdout'])['dimensions'] == 18, 'Wrong native search dimensions')
    save(out / 'plan.json', plan)
    inspect(out)
    print(f'Prepared {args.mode}: six games, {len(cfg["seeds"])} fixed replicas, {cfg["seconds"]/3600:g}h cap; no GPU work.')


def inspect(out):
    plan = read(out / 'plan.json')
    require(plan['schema'] == 'sprite-overnight-v1' and not plan['seeds_swept'], 'Wrong campaign')
    require(plan['settings'] == settings(out / 'campaign.ini', plan['mode']), 'Changed frozen campaign settings')
    require(plan['games'] == list(GAMES), 'Missing sprite games')
    for path, digest in plan['inputs'].items():
        require(sha(path) == digest, f'Changed campaign input: {path}')
    for game, digest in plan['recipes'].items():
        require(sha(out / f'recipes/{game}.ini') == digest, 'Changed per-game learner')
    build = Path(plan['build'])
    subprocess.run(['sha256sum', '-c', str(build / 'source.sha256')], cwd=build / 'source',
        check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['sha256sum', '-c', str(build / 'packed-checks.sha256')],
        check=True, stdout=subprocess.DEVNULL)
    return plan


class CampaignEnd(Exception):
    pass


class Executor:
    def __init__(self, out, plan, expected):
        self.out, self.plan, self.settings = out, plan, plan['settings']
        self.start = time.monotonic_ns()
        self.deadline = time.monotonic() + self.settings['seconds']
        self.smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
        hardware = subprocess.check_output([self.smi, '--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        require(len(hardware.splitlines()) == 1 and expected in hardware, 'Wrong/multiple scheduled GPUs')
        self.hardware = hardware
        self.processes = []
        self.idle()

    def pids(self):
        data = subprocess.check_output([self.smi, '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        return [int(s.strip()) for s in data.splitlines() if s.strip()]

    def idle(self):
        require(not self.pids(), 'GPU busy; no competing work will be interrupted')

    def disk(self, forecast=0):
        free = shutil.disk_usage(self.out).free
        used = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if free - forecast < self.settings['min_free_gib'] * 2**30:
            raise CampaignEnd('disk_free_reserve')
        if used + forecast > self.settings['max_artifact_gib'] * 2**30:
            raise CampaignEnd('artifact_storage_cap')

    def label(self, folder, suffix):
        return str(folder.relative_to(self.out)).replace('/', '-') + '-' + suffix

    def process(self, label, argv, cwd, timeout):
        self.idle()
        remaining = self.deadline - time.monotonic()
        if remaining <= 1:
            raise CampaignEnd('wall_clock_cap')
        folder = self.out / 'processes'
        folder.mkdir(exist_ok=True)
        file = folder / f'{label}.json'
        require(not file.exists(), 'Never overwrite/retry a process')
        logfile = folder / f'{label}.log'
        receipt = dict(label=label, argv=list(map(str, argv)), cwd=str(cwd), hardware=self.hardware,
            monotonic_started_ns=time.monotonic_ns(), log=str(logfile.relative_to(self.out)))
        save(folder / f'{label}.started.json', receipt)
        monitor_stop = threading.Event()
        interference = []
        samples = []
        with logfile.open('xb') as log:
            proc = subprocess.Popen(receipt['argv'], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))

            def monitor():
                known = set()
                while not monitor_stop.wait(5):
                    try:
                        active = self.pids()
                        foreign = []
                        for pid in active:
                            try:
                                ours = os.getpgid(pid) == proc.pid
                            except ProcessLookupError:
                                ours = pid in known
                            if ours:
                                known.add(pid)
                            else:
                                foreign.append(pid)
                        samples.append(dict(monotonic_ns=time.monotonic_ns(), pids=active, foreign=foreign))
                        if foreign:
                            interference.extend(foreign)
                            os.killpg(proc.pid, signal.SIGTERM)
                            return
                    except Exception as error:
                        interference.append(str(error))
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        return

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            timed_out = False
            interrupted = None
            try:
                proc.wait(timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            except BaseException as error:
                interrupted = f'{type(error).__name__}: {error}'
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                except ProcessLookupError:
                    proc.wait()
            finally:
                finished = time.monotonic_ns()
                monitor_stop.set()
                thread.join(timeout=20)
        receipt.update(monotonic_finished_ns=finished, returncode=proc.returncode,
            timed_out=timed_out, interrupted=interrupted, interference=interference,
            gpu_samples=samples, log_sha256=sha(logfile))
        receipt['seconds'] = (receipt['monotonic_finished_ns'] - receipt['monotonic_started_ns']) / 1e9
        save(file, receipt)
        self.processes.append(receipt)
        require(not interrupted, f'Interrupted native process: {label}; receipt preserved')
        require(not interference, 'GPU interference/query failure; retain timing as contaminated')
        if timed_out and time.monotonic() >= self.deadline:
            raise CampaignEnd('wall_clock_cap_during_process')
        require(proc.returncode == 0 and not timed_out, f'Native process failed: {label}; see {logfile}')
        self.idle()
        return receipt


def config_for(out, game, family, policy, seed, steps, checkpoints, folder, cfg):
    recipe = packet.read_config(out / f'recipes/{game}.ini')
    recipe['policy'].update(policy)
    recipe['policy']['encoder'] = str(FAMILIES[family])
    recipe['base'].update(seed=str(seed), run_id=folder.name,
        checkpoint_dir=str(folder / 'checkpoints'), log_dir=str(folder / 'logs'),
        checkpoint_interval=str(steps // (1024 * checkpoints)))
    recipe['train']['total_timesteps'] = str(steps)
    (folder / 'config').mkdir(parents=True)
    file = folder / f'config/sprite_{game}.ini'
    packet.write_config(file, recipe)
    (folder / 'config/default.ini').write_text('')
    return file


def describe(build, game, config):
    receipt = packet.host_call(build / f'sprite_{game}', 'sprite_benchmark_describe', config)
    return json.loads(receipt['stdout'])


def weight_check(path, expected):
    require(path.stat().st_size == expected, 'Wrong checkpoint length')
    with path.open('rb') as file:
        while block := file.read(1024 * 1024):
            require(all(math.isfinite(x) for x in array.array('f', block)), 'Nonfinite checkpoint')
    return sha(path)


def train(executor, build, game, folder, config, steps, count):
    meta = describe(build, game, config)
    executor.disk(meta['checkpoint_bytes'] * count)
    receipt = executor.process(executor.label(folder, 'train'), [build / f'sprite_{game}', 'train',
        '--headless', '--base.result_fd=0'], folder, executor.settings['timeouts']['training'])
    log = (executor.out / receipt['log']).read_text()
    times = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)', log)
    expected = [steps // count * i for i in range(1, count + 1)]
    require([int(row[0]) for row in times] == expected, 'Missing/extra checkpoint receipt')
    checkpoints = []
    for decision, size, stamp in times:
        decision, size, stamp = int(decision), int(size), int(stamp)
        file = folder / f'checkpoints/sprite_{game}/{folder.name}/{decision:016d}.bin'
        require(size == meta['checkpoint_bytes'], 'Wrong serialized parameter size')
        require(receipt['monotonic_started_ns'] <= stamp <= receipt['monotonic_finished_ns'],
            'Checkpoint clock is outside process')
        checkpoints.append(dict(decisions=decision, path=str(file.relative_to(executor.out)),
            sha256=weight_check(file, size), bytes=size,
            seconds=(stamp - receipt['monotonic_started_ns']) / 1e9))
    metrics_file = folder / f'logs/sprite_{game}/{folder.name}.ini'
    metrics = packet.read_config(metrics_file)['metrics']
    native = {k: metrics.get(k) for k in ('SPS', 'uptime', 'agent_steps', 'env/perf')}
    result = dict(game=game, name=folder.name, metadata=meta, config_sha256=sha(config),
        process=receipt, process_sps=steps / receipt['seconds'], native_metrics=native,
        checkpoints=checkpoints, metrics_sha256=sha(metrics_file))
    save(folder / 'training.json', result)
    return result


def eval_rows(file, episodes):
    with file.open() as stream:
        data = sorted(csv.DictReader(stream), key=lambda r: int(r['episode_id']))
    require([int(r['episode_id']) for r in data] == list(range(episodes)), 'Wrong assigned episode quota')
    require(all(math.isfinite(float.fromhex(r['perf'])) and 0 <= float.fromhex(r['perf']) <= 1
        for r in data), 'Nonfinite/out-of-range evaluation perf')
    return data


def evaluate(executor, build, game, folder, checkpoint, skin, label, episodes=None, slots=None, eager=False):
    cfg = executor.settings
    episodes = cfg['episodes'] if episodes is None else episodes
    slots = cfg['slots'] if slots is None else slots
    output = folder / f'{label}.csv'
    recipe = packet.read_config(folder / f'config/sprite_{game}.ini')
    recipe['env']['representation'] = str(skin)
    recipe['sprite_eval'].update(episodes=str(episodes), slots=str(slots))
    manifest_config = folder / f'{label}.manifest.ini'
    packet.write_config(manifest_config, recipe)
    manifest = packet.host_call(build / f'sprite_{game}', 'sprite_eval_manifest', manifest_config)
    save(folder / f'{label}.manifest.json', manifest)
    command = [build / f'sprite_{game}', 'sprite_eval', '--headless', '--base.result_fd=0',
        '--base.load_model_path=' + str(executor.out / checkpoint['path']),
        f'--env.representation={skin}', f'--sprite_eval.output={output}',
        f'--sprite_eval.episodes={episodes}', f'--sprite_eval.slots={slots}']
    if eager:
        command.append('--base.cudagraphs=-1')
    receipt = executor.process(executor.label(folder, label), command, folder, cfg['timeouts']['evaluation'])
    data = eval_rows(output, episodes)
    start = list(csv.DictReader(io.StringIO(manifest['stdout'])))
    keys = ('episode_id', 'env_seed', 'policy_seed', 'start_rng', 'start_world_hash', 'start_observation_hash')
    require([[r[k] for k in keys] for r in data] == [[r[k] for k in keys] for r in start],
        'Policy did not receive assigned starting worlds/pixels')
    result = dict(game=game, decisions=checkpoint['decisions'], skin=skin, episodes=episodes,
        perf=statistics.mean(float.fromhex(r['perf']) for r in data),
        csv=str(output.relative_to(executor.out)), csv_sha256=sha(output),
        manifest=str((folder / f'{label}.manifest.json').relative_to(executor.out)),
        manifest_sha256=sha(folder / f'{label}.manifest.json'), process=receipt)
    save(folder / f'{label}.json', result)
    return result


def numerical(executor, build, config, folder, batches, fixtures):
    results = []
    for B in batches:
        for fixture in fixtures:
            case = folder / f'b{B}-f{fixture}'
            case.parent.mkdir(parents=True, exist_ok=True)
            receipt = executor.process(executor.label(folder, f'b{B}-f{fixture}'),
                [build / 'packed-checks', config, str(B), str(fixture), case], executor.out,
                executor.settings['timeouts']['numerical'])
            require('PACKED CUDA PASS ' in (executor.out / receipt['log']).read_text(), 'Missing CUDA numerical pass')
            results.append(dict(receipt=receipt, files={str(p.relative_to(executor.out)): sha(p)
                for p in case.iterdir() if p.is_file()}))
    save(folder / 'checks.json', results)


def aggregate(jobs, cfg):
    wanted = {(g, s) for g in GAMES for s in cfg['seeds']}
    require(len(jobs) == len(wanted) and {(j['game'], j['seed']) for j in jobs} == wanted,
        'Incomplete candidate cannot provide feedback')
    games = []
    for game in GAMES:
        rows = []
        for job in jobs:
            if job['game'] != game:
                continue
            final = [r for r in job['evaluations'] if r['decisions'] == cfg['decisions']]
            require(len(final) == len(cfg['skins']) and {r['skin'] for r in final} == set(cfg['skins']),
                'Incomplete final skin evaluation')
            require(all(math.isfinite(r['perf']) and 0 <= r['perf'] <= 1 for r in final),
                'Nonfinite/out-of-range final utility')
            rows.extend(r['perf'] for r in final)
        score = statistics.mean(rows)
        utility = max(0, min(1, (score - cfg['offset']) / (cfg['target'] - cfg['offset'])))
        games.append(dict(game=game, score=score, utility=utility))
    costs = [j['training']['process']['seconds'] for j in jobs]
    require(all(math.isfinite(c) and c > 0 for c in costs), 'Invalid training cost')
    return dict(score=statistics.mean(r['utility'] for r in games),
        worst_game_utility=min(r['utility'] for r in games), games=games,
        cost=math.fsum(costs),
        training_jobs_charged=len(jobs), scientific_claim=False)


def candidate(executor, build, name, family, policy):
    cfg = executor.settings
    folder = executor.out / 'candidates' / name
    folder.mkdir(parents=True)
    save(folder / 'candidate.json', dict(name=name, family=family, policy=policy, status='running'))
    jobs = []
    for game in GAMES:
        for seed in cfg['seeds']:
            job_dir = folder / f'{game}-s{seed}'
            config = config_for(executor.out, game, family, policy, seed, cfg['decisions'],
                cfg['checkpoint_count'], job_dir, cfg)
            trained = train(executor, build, game, job_dir, config, cfg['decisions'], cfg['checkpoint_count'])
            ev = []
            for checkpoint in trained['checkpoints']:
                for skin in cfg['skins']:
                    label = f'd{checkpoint["decisions"]}-skin{skin}'
                    ev.append(evaluate(executor, build, game, job_dir, checkpoint, skin, label))
            job = dict(game=game, seed=seed, training=trained, evaluations=ev)
            save(job_dir / 'job.json', job)
            jobs.append(job)
            save(folder / 'progress.json', dict(jobs=jobs, expected_jobs=len(GAMES) * len(cfg['seeds'])))
    feedback = aggregate(jobs, cfg)
    result = dict(name=name, family=family, policy=policy, jobs=jobs, feedback=feedback, status='complete')
    save(folder / 'complete.json', result)
    print(f'{name}: complete; aggregate={feedback["score"]:.5f}, training={feedback["cost"]:.1f}s', flush=True)
    return result


def validate(executor, build, policy):
    cfg = executor.settings
    families = ['tile', 'nature', 'flex'] if executor.plan['mode'] == 'smoke' else list(FAMILIES)
    for family in families:
        folder = executor.out / 'validation' / family
        config = config_for(executor.out, 'catch', family, policy, 7, 8192, 1, folder / 'math', cfg)
        numerical(executor, build, config, folder / 'numerical',
            [1] if executor.plan['mode'] == 'smoke' else [1, 3, 32], [0, 1])
        for game in GAMES:
            pair = []
            for repeat in ('a', 'b'):
                job = folder / f'{game}-{repeat}'
                config = config_for(executor.out, game, family, policy, 7, 8192, 1, job, cfg)
                pair.append(train(executor, build, game, job, config, 8192, 1))
            require(pair[0]['checkpoints'][0]['sha256'] == pair[1]['checkpoints'][0]['sha256'],
                'Same-seed native training repeat differs')
            checks = []
            for mode in ('graph', 'repeat', 'eager'):
                checks.append(evaluate(executor, build, game, folder / f'{game}-a',
                    pair[0]['checkpoints'][0], 0, mode, episodes=3, slots=2, eager=mode == 'eager'))
            require(eval_rows(executor.out / checks[0]['csv'], 3) == eval_rows(executor.out / checks[1]['csv'], 3)
                == eval_rows(executor.out / checks[2]['csv'], 3), 'Graph/repeat/eager rows differ')
    save(executor.out / 'VALIDATED.json', dict(families=families, hardware=executor.hardware,
        numerical_fixtures='positive/zero, all registered encoder gradients; not arbitrary branches/full learner',
        training_decisions=8192, exact_tail_episodes=3, slots=2, scientific_claim=False))
    calibration = []
    for family in families:
        for game in GAMES:
            pair = [read(executor.out / f'validation/{family}/{game}-{r}/training.json') for r in ('a', 'b')]
            seconds = statistics.mean(p['process']['seconds'] for p in pair)
            calibration.append(dict(family=family, game=game, canary_decisions=8192,
                process_seconds=seconds, process_sps=8192 / seconds,
                projected_training_seconds_all_replicas=seconds * cfg['decisions'] / 8192 * len(cfg['seeds']),
                projection_is_estimate_not_measurement=True))
    save(executor.out / 'calibration.json', calibration)


def history_text(trials):
    text = 'PUFFER_CROSS_GAME_V1 18\n'
    for trial in trials:
        values = ['0', format(trial['feedback']['score'], '.17g'), format(trial['feedback']['cost'], '.17g')]
        values += [format(float(x), '.9g') for x in trial['proposal']['normalized']]
        text += ' '.join(values) + '\n'
    return text


def propose(executor, build, trials):
    folder = executor.out / 'optimizer' / f'observations-{len(trials):04d}'
    folder.mkdir(parents=True)
    ledger = folder / 'history.tsv'
    ledger.write_text(history_text(trials))
    output = folder / 'proposal.json'
    executor.process(f'proposal-{len(trials):04d}', [build / 'sprite_catch', 'research_protein_propose',
        executor.out / 'optimizer.ini', ledger, output], executor.out, executor.settings['timeouts']['proposal'])
    value = read(output)
    require(value['observations_replayed'] == len(trials) and len(value['normalized']) == 18,
        'Native feedback replay incomplete')
    require(set(value['policy']) == set(json.loads(executor.plan['optimizer_description']['stdout'])['coordinates']),
        'Native proposal coordinate set changed')
    require(all(math.isfinite(v) and v == int(v) for v in value['policy'].values()), 'Invalid native CNN coordinates')
    return value


def run(args):
    out = args.out.resolve()
    plan = inspect(out)
    require(not (out / 'started.json').exists(), 'No automatic campaign restart; prepare a fresh allocation')
    require(args.expected_gpu == ('5060' if plan['mode'] == 'smoke' else '5090'), 'Wrong execution route')
    build = Path(plan['build'])
    policy = dict(packet.read_config(out / 'optimizer.ini')['policy'])
    policy.pop('hidden_size'); policy.pop('num_layers')
    record = dict(status='running', baselines=[], trials=[], scientific_claim=False,
        mode=plan['mode'], plan_sha256=sha(out / 'plan.json'))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Campaign received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    lock_path = ROOT / 'build/connect4cnn/hardware-benchmark.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        executor = Executor(out, plan, args.expected_gpu)
        save(out / 'started.json', dict(monotonic_ns=executor.start, hardware=executor.hardware,
            deadline_seconds=plan['settings']['seconds']))
        try:
            validate(executor, build, policy)
            for family in executor.settings['baselines']:
                inspect(out)
                record['baselines'].append(candidate(executor, build, family, family, policy))
                save(out / 'progress.json', record)
            proposal = propose(executor, build, record['trials'])
            for index in range(executor.settings['trials']):
                inspect(out)
                executor.disk()
                name = ('happy-cat', 'mystic-tree', 'swift-fox', 'quiet-owl')[index % 4] + f'-{index + 1}'
                chosen = dict(policy)
                chosen.update({k: str(int(v)) for k, v in proposal['policy'].items()})
                check_dir = out / 'trial-checks' / name
                config = config_for(out, 'catch', 'flex', chosen, 7, 8192, 1, check_dir / 'recipe', executor.settings)
                numerical(executor, build, config, check_dir,
                    [1] if plan['mode'] == 'smoke' else [1, 32], [0, 1])
                result = candidate(executor, build, name, 'flex', chosen)
                duplicate = next((j for j, old in enumerate(record['trials']) if old['proposal']['policy'] == proposal['policy']), None)
                record['trials'].append(dict(name=name, proposal=proposal, feedback=result['feedback'], duplicate_of=duplicate))
                save(out / 'progress.json', record)
                proposal = propose(executor, build, record['trials'])
                record['observations_replayed'] = proposal['observations_replayed']
                record['final_unused_proposal'] = proposal
                if index >= 2:
                    require(proposal['gp_observations'] > 0, 'Native GP phase was not reached after warmup')
                save(out / 'progress.json', record)
            record['status'] = 'complete'
        except CampaignEnd as error:
            record.update(status='budget_stopped', stop_reason=str(error), incomplete_candidates_retained=True)
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            record.update(ended_monotonic_ns=time.monotonic_ns(), hardware=executor.hardware,
                process_count=len(executor.processes))
            save(out / 'result.json', record)
            report(out)


def report(out):
    """Offline receipt and curve export. Never executes native/GPU/policy code."""
    plan = read(out / 'plan.json')
    require((out / 'result.json').is_file(), 'Report after execution stops; use progress.json while running')
    cfg = plan['settings']
    rows = []
    names = set(cfg['baselines'])
    processes = []
    for file in sorted((out / 'processes').glob('*.json')):
        if file.name.endswith('.started.json'):
            continue
        receipt = read(file)
        require(sha(out / receipt['log']) == receipt['log_sha256'], 'Changed native process log')
        processes.append(receipt)
    processes.sort(key=lambda p: p['monotonic_started_ns'])
    receipts = {r['label']: r for r in processes}
    require(all(a['monotonic_finished_ns'] <= b['monotonic_started_ns']
        for a, b in zip(processes, processes[1:])), 'Timed native processes overlap')
    for folder in sorted((out / 'candidates').glob('*')):
        if not (folder / 'candidate.json').exists():
            continue
        model = read(folder / 'candidate.json')
        names.add(model['name'])
        if (folder / 'complete.json').exists():
            complete = read(folder / 'complete.json')
            require(aggregate(complete['jobs'], cfg) == complete['feedback'], 'Changed aggregate feedback')
        for training_path in sorted(folder.glob('*/training.json')):
            training = read(training_path)
            job_dir = training_path.parent
            game = training['game']
            config = job_dir / f'config/sprite_{game}.ini'
            require(sha(config) == training['config_sha256'], 'Changed requested training recipe')
            seed = packet.read_config(config).getint('base', 'seed')
            require(training['process'] == receipts[training['process']['label']], 'Unbound training receipt')
            require(training['process']['returncode'] == 0 and not training['process']['timed_out']
                and not training['process']['interference'], 'Contaminated/failed training cannot qualify')
            require(sha(out / training['process']['log']) == training['process']['log_sha256'], 'Changed training log')
            for checkpoint in training['checkpoints']:
                require(sha(out / checkpoint['path']) == checkpoint['sha256'], 'Changed checkpoint')
            for eval_file in sorted(job_dir.glob('d*-skin*.json')):
                if eval_file.name.endswith('.manifest.json'):
                    continue
                ev = read(eval_file)
                require(ev['episodes'] == cfg['episodes'] and ev['game'] == game, 'Changed evaluation quota/game')
                require(ev['process'] == receipts[ev['process']['label']], 'Unbound evaluation receipt')
                require(ev['process']['returncode'] == 0 and not ev['process']['timed_out']
                    and not ev['process']['interference'], 'Contaminated/failed evaluation cannot qualify')
                require(sha(out / ev['csv']) == ev['csv_sha256'], 'Changed evaluation CSV')
                raw = eval_rows(out / ev['csv'], ev['episodes'])
                require(statistics.mean(float.fromhex(r['perf']) for r in raw) == ev['perf'], 'Changed score')
                require(sha(out / ev['manifest']) == ev['manifest_sha256'], 'Changed world/pixel manifest')
                starts = list(csv.DictReader(io.StringIO(read(out / ev['manifest'])['stdout'])))
                keys = ('episode_id', 'env_seed', 'policy_seed', 'start_rng', 'start_world_hash', 'start_observation_hash')
                require([[r[k] for k in keys] for r in raw] == [[r[k] for k in keys] for r in starts],
                    'Evaluation assignments differ')
                checkpoint = next(c for c in training['checkpoints'] if c['decisions'] == ev['decisions'])
                rows.append(dict(model=model['name'], family=model['family'], game=game, seed=seed,
                    skin=ev['skin'], decisions=ev['decisions'], score=ev['perf'],
                    train_seconds=checkpoint['seconds'], process_sps=training['process_sps']))
    curves = []
    wanted = [(name, game, skin, cfg['decisions'] // cfg['checkpoint_count'] * i)
        for name in sorted(names) for game in GAMES for skin in cfg['skins']
        for i in range(1, cfg['checkpoint_count'] + 1)]
    for key in wanted:
        cell = [r for r in rows if (r['model'], r['game'], r['skin'], r['decisions']) == key]
        require(len(cell) == len({r['seed'] for r in cell}), 'Duplicate model/game/skin/checkpoint/seed')
        require({r['seed'] for r in cell} <= set(cfg['seeds']), 'Unexpected seed')
        complete = len(cell) == len(cfg['seeds'])
        curves.append(dict(model=key[0], game=key[1], skin=key[2], decisions=key[3], replicas=len(cell),
            score=statistics.mean(r['score'] for r in cell) if complete else None,
            seconds=statistics.mean(r['train_seconds'] for r in cell) if complete else None))
    save(out / 'curves.json', dict(curves=curves, raw_points=rows, scientific_claim=False))
    with (out / 'curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'game', 'skin', 'decisions', 'replicas', 'score', 'seconds'])
        writer.writeheader(); writer.writerows(curves)
    content = ['<!doctype html><meta charset="utf-8"><title>Sprite CNN development curves</title>',
        '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #aaa}svg{background:#fafafa;border:1px solid #bbb}circle:hover{r:6}</style>',
        '<h1>Sprite CNN development curves</h1><p>Complete assigned replicas only. All measured checkpoints and declines retained. '
        'Discovery data; no held-out superiority or statistical frontier claim. See result.json for incomplete/failed work.</p>']
    for game in GAMES:
        content.append(f'<h2>{html.escape(game)}</h2>')
        for skin in cfg['skins']:
            all_cells = [r for r in curves if r['game'] == game and r['skin'] == skin]
            cell = [r for r in all_cells if r['score'] is not None]
            content.append(f'<h3>Skin {skin}</h3>')
            max_time = max((r['seconds'] for r in cell), default=1)
            content.append('<svg viewBox="0 0 720 300" width="720" height="300"><path d="M50 10V260H710" fill="none" stroke="#555"/>')
            for model in sorted({r['model'] for r in cell}):
                points = sorted([r for r in cell if r['model'] == model], key=lambda r: r['decisions'])
                color = f'hsl({int(hashlib.sha256(model.encode()).hexdigest()[:6],16)%360} 70% 40%)'
                coords = [(50 + r['seconds'] / max_time * 650, 260 - r['score'] * 240) for r in points]
                content.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)}" fill="none" stroke="{color}"/>')
                for (x, y), row in zip(coords, points):
                    title = html.escape(f'{model}: {row["score"]:.4f}, {row["seconds"]:.3f}s, {row["decisions"]} decisions')
                    content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>{title}</title></circle>')
            content.append('<text x="270" y="292">training seconds (per-replica mean)</text><text x="5" y="16">1.0</text><text x="12" y="261">0</text></svg>')
            content.append('<table><tr><th>Model</th><th>Final score</th><th>Train seconds</th><th>Replicas</th></tr>')
            for model in sorted(names):
                last = max((r for r in all_cells if r['model'] == model), key=lambda r: r['decisions'])
                score = 'missing' if last['score'] is None else f'{last["score"]:.4f}'
                seconds = 'missing' if last['seconds'] is None else f'{last["seconds"]:.3f}'
                content.append(f'<tr><td>{html.escape(model)}</td><td>{score}</td><td>{seconds}</td><td>{last["replicas"]}/{len(cfg["seeds"])}</td></tr>')
            content.append('</table>')
    (out / 'curves.html').write_text('\n'.join(content))
    save(out / 'report.json', dict(completed_models=len(list((out / 'candidates').glob('*/complete.json'))),
        complete_curve_points=sum(r['score'] is not None for r in curves), missing_curve_points=sum(r['score'] is None for r in curves),
        raw_points=len(rows), native_processes=len(processes), scientific_claim=False, gpu_execution=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--build', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--recipe', type=Path, default=RECIPE)
    prep.add_argument('--mode', choices=('smoke', 'overnight'), default='overnight')
    for command in ('inspect', 'report', 'run'):
        part = sub.add_parser(command)
        part.add_argument('--out', type=Path, required=True)
        if command == 'run':
            part.add_argument('--expected-gpu', choices=('5060', '5090'), required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'inspect': print(json.dumps(inspect(args.out), indent=2))
    elif args.command == 'run': run(args)
    else: report(args.out.resolve())


if __name__ == '__main__':
    main()
