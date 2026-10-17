"""Prepare/inspect matched sprite CNN packets; never start training or query a GPU."""
import argparse
import configparser
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GAMES = ('catch', 'platform', 'maze', 'runner', 'key', 'push')
FAMILIES = {'tile': 6, 'nature': 2, 'impala': 0, 'impoola': 7, 'flex': 4}
TRAINING_SEEDS = (7, 42, 69, 123, 173, 2027, 9001)
EVAL_SEEDS = (2026101601, 2026101602)
RECIPE = ROOT / 'research/recipes/sprite_pixel_architecture.ini'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    with path.open('x') as file:
        json.dump(data, file, indent=2)
        file.write('\n')


def read_config(*paths):
    cfg = configparser.ConfigParser(interpolation=None, strict=False)
    for path in paths:
        with path.open() as file:
            cfg.read_file(file)
    return cfg


def write_config(path, cfg):
    with path.open('x') as file:
        cfg.write(file)


def remove_sweeps(cfg):
    for section in cfg.sections():
        if section == 'sweep' or section.startswith('sweep.'):
            cfg.remove_section(section)


def host_call(binary, mode, config):
    command = [str(binary), mode, str(config)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    return dict(argv=command, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)


def prepare(args):
    build, out = args.build.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('Use a fresh output directory; previous packets stay intact')
    if args.decisions < 8192 or args.decisions % 8192:
        raise ValueError('Decisions must be a positive multiple of 8192 for eight checkpoints')
    games = tuple(args.games)
    if len(games) != len(set(games)) or len(args.families) != len(set(args.families)):
        raise ValueError('Duplicate game/family assignments')
    subprocess.run(['sha256sum', '-c', str(build / 'source.sha256')],
        cwd=build / 'source', check=True, stdout=subprocess.DEVNULL)
    out.mkdir(parents=True)
    write_json(out / 'preparation-started.json', dict(status='preparing', gpu_execution=False))
    seed_list = TRAINING_SEEDS if args.mode == 'confirmation' else (7,)
    # Preparation chooses a conservative COMMON minibatch; calibrate it on GPU.
    # Per-game LR/entropy/gamma/core depth remain the existing game recipe.
    jobs = []
    for game in games:
        for family in args.families:
            for seed in seed_list:
                job_id = f'{game}-{family}-s{seed}'
                folder = out / 'jobs' / job_id
                (folder / 'config').mkdir(parents=True)
                cfg = read_config(build / 'runtime/config/default.ini',
                    build / f'runtime/config/sprite_{game}.ini')
                remove_sweeps(cfg)
                cfg.read(RECIPE)
                remove_sweeps(cfg)
                cfg['env'].update(observation_mode='1', partial_observation='1', representation='0')
                cfg['policy']['encoder'] = str(FAMILIES[family])
                cfg['base']['seed'] = str(seed)
                cfg['base'].update({'async': '0', 'eval_episodes': '0', 'load_model_path': 'None',
                    'run_id': f'happy-maple-{len(jobs) + 1}',
                    'checkpoint_interval': str(args.decisions // 8192),
                    'checkpoint_dir': str(folder / 'checkpoints'), 'log_dir': str(folder / 'logs')})
                cfg['vec'].update(total_agents='32', num_buffers='1', num_threads='2', num_policies='1',
                    hist_policy_percent='0')
                cfg['train'].update(gpus='1', total_timesteps=str(args.decisions),
                    horizon='32', minibatch_size='32')
                cfg['selfplay']['enabled'] = '0'
                cfg['sweep'] = dict(metric='perf', metric_distribution='linear', goal='maximize',
                    max_runs='1', gpus='1', downsample='8', max_suggestion_cost='600')
                cfg['sprite_eval'] = dict(episodes='3' if args.mode == 'canary' else '128',
                    slots='2' if args.mode == 'canary' else '16', seed=str(EVAL_SEEDS[0]),
                    episode_offset='0', max_steps='2048', output='None')
                config = folder / f'config/sprite_{game}.ini'
                write_config(config, cfg)
                # The full recipe above owns all settings, including sweep axes.
                (folder / 'config/default.ini').write_text('')
                binary = build / f'sprite_{game}'
                receipt = host_call(binary, 'sprite_benchmark_describe', config)
                metadata = json.loads(receipt['stdout'])
                write_json(folder / 'metadata.json', dict(metadata=metadata, receipt=receipt))
                manifests = []
                for skin in range(3):
                    for world in EVAL_SEEDS:
                        eval_cfg = read_config(config)
                        eval_cfg['env']['representation'] = str(skin)
                        eval_cfg['sprite_eval']['seed'] = str(world)
                        eval_path = folder / f'eval-skin{skin}-world{world}.ini'
                        write_config(eval_path, eval_cfg)
                        receipt = host_call(binary, 'sprite_eval_manifest', eval_path)
                        write_json(folder / f'{eval_path.stem}.manifest.json', receipt)
                        manifests.append(dict(config=str(eval_path.relative_to(out)),
                            config_sha256=sha(eval_path), skin=skin, world=world,
                            manifest_sha256=sha(folder / f'{eval_path.stem}.manifest.json'),
                            manifest=str((folder / f'{eval_path.stem}.manifest.json').relative_to(out))))
                jobs.append(dict(id=job_id, game=game, family=family, seed=seed,
                    binary=str(binary), binary_sha256=sha(binary),
                    config=str(config.relative_to(out)), config_sha256=sha(config),
                    default_sha256=sha(folder / 'config/default.ini'), metadata=metadata,
                    metadata_receipt_sha256=sha(folder / 'metadata.json'), evaluations=manifests))
                if family == 'flex':
                    sweep = read_config(config)
                    sweep.read(RECIPE)
                    sweep_dir = folder / 'sweep/config'
                    sweep_dir.mkdir(parents=True)
                    write_config(sweep_dir / f'sprite_{game}.ini', sweep)
                    (sweep_dir / 'default.ini').write_text('')
                    jobs[-1]['sweep_config'] = str((sweep_dir / f'sprite_{game}.ini').relative_to(out))
                    jobs[-1]['sweep_config_sha256'] = sha(sweep_dir / f'sprite_{game}.ini')
    plan = dict(schema='sprite-packed-benchmark-v1', status='prepared-not-executed',
        mode=args.mode, games=list(games), families=list(args.families), build=str(build),
        source_manifest_sha256=sha(build / 'source.sha256'),
        tool_sha256=sha(Path(__file__)), recipe_sha256=sha(RECIPE),
        observation=dict(layout='player3x3-above-gaze3x3', width=96, height=192,
            channels=3, geometry=4, input_size=55300, shared_pixel_bytes=True),
        decisions=args.decisions, checkpoints=[args.decisions // 8 * i for i in range(1, 9)],
        training_seeds=list(seed_list), seeds_swept=False, evaluation_seeds=list(EVAL_SEEDS),
        jobs=jobs, gpu_execution=False, completed_observations=0,
        remaining_gates=['GPU numerical baseline/geometry-adapter checks',
            'actual learner-batch memory and runtime calibration',
            'same-seed training and graph/eager/reload canaries',
            'fixed/random gaze controls', 'calibrated per-game budgets',
            'seven-replica confirmation and held-out evaluation',
            'sprite-suite cross-game aggregate-to-PROTEIN integration'])
    write_json(out / 'plan.json', plan)
    report = inspect(out)
    write_json(out / 'preparation-audit.json', report)
    write_json(out / 'PREPARED', dict(training_jobs=len(jobs), gpu_execution=False))
    write_json(out / 'status.json', dict(status='prepared-not-executed', gpu_execution=False))
    print(json.dumps(report, indent=2))


def inspect(out):
    out = out.resolve()
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['schema'] == 'sprite-packed-benchmark-v1' and not plan['seeds_swept']
    assert plan['training_seeds'] == (list(TRAINING_SEEDS) if plan['mode'] == 'confirmation' else [7])
    assert plan['evaluation_seeds'] == list(EVAL_SEEDS)
    expected = {(g, f, s) for g in plan['games'] for f in plan['families']
        for s in plan['training_seeds']}
    assert len(plan['jobs']) == len(expected)
    assert {(j['game'], j['family'], j['seed']) for j in plan['jobs']} == expected
    assert plan['checkpoints'] == [plan['decisions'] // 8 * i for i in range(1, 9)]
    assert sha(Path(__file__)) == plan['tool_sha256'] and sha(RECIPE) == plan['recipe_sha256']
    build = Path(plan['build'])
    assert sha(build / 'source.sha256') == plan['source_manifest_sha256']
    subprocess.run(['sha256sum', '-c', str(build / 'source.sha256')], cwd=build / 'source',
        check=True, stdout=subprocess.DEVNULL)
    recipes, worlds = {}, {}
    for job in plan['jobs']:
        config = out / job['config']
        assert sha(config) == job['config_sha256']
        assert sha(config.with_name('default.ini')) == job['default_sha256']
        assert sha(Path(job['binary'])) == job['binary_sha256']
        assert sha(config.parent.parent / 'metadata.json') == job['metadata_receipt_sha256']
        cfg = read_config(config)
        assert cfg.getint('base', 'seed') == job['seed']
        assert cfg.getint('train', 'total_timesteps') == plan['decisions']
        assert not any(s.startswith('sweep.') for s in cfg.sections())
        assert cfg.getint('policy', 'encoder') == FAMILIES[job['family']]
        assert cfg.getint('env', 'observation_mode') == cfg.getint('env', 'partial_observation') == 1
        meta = job['metadata']
        registered = json.loads((config.parent.parent / 'metadata.json').read_text())
        assert registered['metadata'] == meta
        assert json.loads(registered['receipt']['stdout']) == meta
        assert meta['encoder'] == FAMILIES[job['family']]
        assert (meta['width'], meta['height'], meta['channels'], meta['geometry']) == (96, 192, 3, 4)
        assert meta['input_size'] == 55300 and not meta['gpu_execution']
        assert meta['checkpoint_bytes'] == meta['parameters'] * 4
        recipe = {s: dict(cfg[s]) for s in ('env', 'vec', 'train', 'policy', 'selfplay')}
        del recipe['policy']['encoder']
        key = (job['game'], job['seed'])
        assert recipes.setdefault(key, recipe) == recipe, 'Candidates have different learners/interfaces'
        assert len(job['evaluations']) == 6
        if job['family'] == 'flex':
            sweep_path = out / job['sweep_config']
            assert sha(sweep_path) == job['sweep_config_sha256']
            sweep = read_config(sweep_path)
            axes = [s for s in sweep.sections() if s.startswith('sweep.')]
            assert len(axes) == 18 and all(s.startswith('sweep.policy.cnn_') for s in axes)
            assert sweep.getint('base', 'seed') == job['seed']
        for ev in job['evaluations']:
            assert sha(out / ev['config']) == ev['config_sha256']
            assert sha(out / ev['manifest']) == ev['manifest_sha256']
            receipt = json.loads((out / ev['manifest']).read_text())
            assert receipt['returncode'] == 0
            rows = list(csv.DictReader(io.StringIO(receipt['stdout'])))
            eval_cfg = read_config(out / ev['config'])
            assert eval_cfg.getint('sprite_eval', 'seed') == ev['world']
            assert eval_cfg.getint('env', 'representation') == ev['skin']
            assert [int(r['episode_id']) for r in rows] == list(range(
                eval_cfg.getint('sprite_eval', 'episodes')))
            key = (job['game'], ev['skin'], ev['world'])
            assert worlds.setdefault(key, rows) == rows, 'Assigned worlds/pixels differ across candidates'
    return dict(status='prepared-not-executed', jobs=len(plan['jobs']),
        assigned_world_pixel_manifests=len(worlds), fixed_training_seeds=plan['training_seeds'],
        planned_checkpoints_per_job=8, gpu_execution=False, scientific_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--build', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--mode', choices=('canary', 'calibration', 'confirmation'), default='canary')
    prep.add_argument('--decisions', type=int, default=8192)
    prep.add_argument('--games', nargs='+', choices=GAMES, default=list(GAMES))
    prep.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    audit = sub.add_parser('inspect')
    audit.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        print(json.dumps(inspect(args.out), indent=2))


if __name__ == '__main__':
    main()
