"""Offline Forage control/evaluator/learning receipt audit, no native execution."""
import argparse
import array
import json
import math
from pathlib import Path
import re
import statistics

from forage_state_ablations import WORLD_KEYS, rows
from run_forage_attention_budget import ROOT, read, sha


def audit(packet, mode):
    plan = json.loads((packet / 'plan.json').read_text())
    assert plan['schema'] == 'forage-state-controls-v1'
    assert plan['fixed_train_seed'] == 7 and plan['seed_swept'] is False
    assert sha(ROOT / 'research/forage_state_ablations.py') == plan['tool_sha256']
    scalar_plan = None
    if plan.get('state_encoding_followup'):
        assert sha(ROOT / 'research/forage_state_encoding.py') == plan['encoding_wrapper_sha256']
        scalar_root = ROOT / 'build/sprite-forage-state-ablations-5060-20261015'
        assert sha(scalar_root / 'plan.json') == plan['scalar_reference_plan_sha256']
        assert sha(scalar_root / 'full/summary.json') == plan['scalar_reference_summary_sha256']
        scalar_plan = json.loads((scalar_root / 'plan.json').read_text())
    if plan.get('relative_wrapper_sha256'):
        assert sha(ROOT / 'research/forage_state_relative.py') == plan['relative_wrapper_sha256']
        onehot_reference = ROOT / 'build/sprite-forage-state-onehot-5060-20261015/full/summary.json'
        assert sha(onehot_reference) == plan['onehot_absolute_reference_sha256']
    root = packet / mode
    load = lambda name: json.loads((root / name).read_text())
    assert (root / 'COMPLETE').is_file() and not (root / 'failure.json').exists()
    training, evaluations, summary = load('training.json'), load('evaluations.json'), load('summary.json')
    smoke = mode == 'smoke'
    assert len(training) == len(plan['jobs'])
    assert len(evaluations) == len(plan['jobs']) * (4 if smoke else 2)
    steps = 65536 if smoke else plan['training_steps']
    reference = None
    worlds = {}
    processes = []
    checkpoints = 0
    for job, record in zip(plan['jobs'], training):
        assert record['name'] == job['name']
        assert sha(Path(job['binary'])) == job['binary_sha256']
        build = Path(job['binary']).parent
        assert (build / 'attention-build.env').read_text() == job['build_flags']
        if job.get('state_onehot'):
            assert 'FORAGE_STATE_ONEHOT=1\n' in job['build_flags']
        for source, digest in plan['sources'].items():
            captured = 'research/sprite_exact_eval.cu' if source == 'research/sprite_attention_eval.cu' else source
            assert sha(build / 'source' / captured) == digest
        template = packet / job['name'] / 'config/sprite_forage.ini'
        assert sha(template) == job['config_sha256']
        folder = root / job['name']
        config = read(folder / 'runtime/config/sprite_forage.ini')
        assert sha(folder / 'runtime/config/default.ini') == job['default_sha256']
        assert config.getint('base', 'seed') == 7
        assert config.get('base', 'load_model_path') == 'None'
        assert config.getint('env', 'observation_mode') == 0
        assert config.getint('env', 'partial_observation') == job['partial']
        assert config.getint('train', 'total_timesteps') == steps == record['steps']
        if scalar_plan:
            scalar = next(j for j in scalar_plan['jobs'] if j['name'] == job['scalar_reference'])
            scalar_config = read(scalar_root / scalar['name'] / 'config/sprite_forage.ini')
            for section in ('env', 'policy', 'vec'):
                assert dict(config[section]) == dict(scalar_config[section])
            assert {k: v for k, v in config['train'].items() if k != 'total_timesteps'} == {
                k: v for k, v in scalar_config['train'].items() if k != 'total_timesteps'}
        comparison = {s: dict(config[s]) for s in ('env', 'train', 'policy', 'vec')}
        comparison['env'].pop('partial_observation')
        if reference is None:
            reference = comparison
        assert comparison == reference
        expected_parameters = 91008 + (0 if job['partial'] else 15360) + (sum(job['action_sizes']) - 305) * 128
        expected_parameters += 26112 * job.get('state_onehot', 0)
        assert expected_parameters == job['parameters'] == record['parameters']
        expected_steps = [steps] if smoke else plan['checkpoint_steps']
        assert sorted(int(Path(p).stem) for p in record['checkpoints']) == expected_steps
        for path, digest in record['checkpoints'].items():
            weight = packet / path
            assert weight.stat().st_size == job['parameters'] * 4 and sha(weight) == digest
            assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            checkpoints += 1
        final = Path(record['final_checkpoint'])
        assert int(final.stem) == steps
        resolved = read(final.parent / 'resolved.ini')
        for section in ('env', 'policy', 'train', 'vec'):
            assert dict(resolved[section]) == dict(config[section])
        receipt = load(job['name'] + '.process.json')
        assert all(record[k] == v for k, v in receipt.items())
        assert record['process_sps'] == steps / receipt['process_seconds']
        metrics = read(folder / 'logs/sprite_forage' / (job['name'] + '.ini'))['metrics']
        assert float(metrics['agent_steps'].split(',')[-1]) == steps
        assert float(metrics['sps'].split(',')[-1]) == record['native_last_bin_sps']
        processes.append((job['name'], receipt))
        manifests = {}
        for world, descriptor in job['manifests'].items():
            path = Path(descriptor['path'])
            assert sha(path) == descriptor['sha256']
            data = rows(path)
            assert len(data) == 128 and [int(r['episode_id']) for r in data] == list(range(128))
            projected = [[r[k] for k in WORLD_KEYS] for r in data]
            if scalar_plan:
                scalar_worlds = rows(Path(scalar['manifests'][world]['path']))
                assert projected == [[r[k] for k in WORLD_KEYS] for r in scalar_worlds]
            if world in worlds:
                assert worlds[world] == projected
            else:
                worlds[world] = projected
            manifests[int(world)] = {r['episode_id']: r for r in data}
        group = [e for e in evaluations if e['name'] == job['name']]
        assert len(group) == (4 if smoke else 2)
        graph_rows = None
        for ev in group:
            label = job['name'] + '-' + ev['label']
            receipt = load(label + '.process.json')
            assert all(ev[k] == v for k, v in receipt.items())
            assert sha(Path(ev['csv_path'])) == ev['csv_sha256']
            data = rows(Path(ev['csv_path']))
            assert len(data) == ev['episodes']
            first = 16 if smoke and ev['label'] == 'tail' else 0
            count = 1 if first else (17 if smoke else 128)
            assert sorted(int(r['episode_id']) for r in data) == list(range(first, first + count))
            for row in data:
                start = manifests[ev['world']][row['episode_id']]
                assert all(row[k] == start[k] for k in WORLD_KEYS + ('start_observation_hash',))
                assert 1 <= int(row['decisions']) <= 512
                assert 0 <= float.fromhex(row['perf']) <= 1
                assert float(row['successes']) + float(row['deaths']) <= 1
                assert len(row['action_hash']) == 16
            assert ev['perf'] == statistics.fmean(float.fromhex(r['perf']) for r in data)
            assert ev['grape_success'] == statistics.fmean(float(r['successes']) for r in data)
            command = load(label + '.command.json')
            assert '--base.load_model_path=' + str(final) in command
            assert '--env.max_steps=512' in command and '--sprite_eval.slots=16' in command
            text = (root / (label + '.log')).read_text()
            match = re.search(r'SPRITE_EXACT_EVAL version=2 requested=(\d+) completed=(\d+) params=(\d+) heads=(\d+)', text)
            assert match and int(match[1]) == int(match[2]) == count
            assert int(match[3]) == job['parameters'] and int(match[4]) == len(job['action_sizes'])
            if smoke:
                ordered = sorted(data, key=lambda r: int(r['episode_id']))
                if ev['label'] == 'graph':
                    graph_rows = ordered
                else:
                    assert ordered == (graph_rows[-1:] if ev['label'] == 'tail' else graph_rows)
            processes.append((label, receipt))
        point = next(p for p in summary['rows'] if p['name'] == job['name'])
        chosen = group[:1] if smoke else group
        assert point['perf'] == statistics.fmean(e['perf'] for e in chosen)
        assert point['grape_success'] == statistics.fmean(e['grape_success'] for e in chosen)
        assert point['train_seconds'] == record['process_seconds']
    for label, receipt in processes:
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert receipt['process_seconds'] == receipt['monotonic_finished'] - receipt['monotonic_started']
        assert sha(root / (label + '.log')) == receipt['log_sha256']
        assert not load(label + '.gpu-query.json')[-1]['compute_pids'].strip()
    ordered = sorted((r for _, r in processes), key=lambda r: r['monotonic_started'])
    assert all(a['monotonic_finished'] <= b['monotonic_started'] for a, b in zip(ordered, ordered[1:]))
    legacy = json.loads((packet / 'legacy-control/result.json').read_text()) if (packet / 'legacy-control/COMPLETE').is_file() else None
    if not smoke and not plan.get('state_encoding_followup'):
        assert legacy and legacy['byte_identical'] and legacy['legacy_checkpoint_sha256'] == legacy['new_checkpoint_sha256']
    return dict(status='PASS', mode=mode, native_processes=len(processes), training_jobs=len(plan['jobs']),
        evaluations=len(evaluations), assigned_episode_executions=sum(e['episodes'] for e in evaluations),
        checkpoint_files=checkpoints, manifest_worlds_per_policy=256,
        evaluated_unique_worlds_per_policy=17 if smoke else 256,
        legacy_training_bytes_match=legacy['byte_identical'] if legacy else None,
        primary_training_n=1, seed_swept=False,
        scope='Offline recipes, world/episode quotas, logs, times, weights and outcomes; not independent neural math or statistical superiority',
        quality_or_pareto_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'full'))
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    result = audit(args.packet.resolve(), args.mode)
    (args.packet / args.mode / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
