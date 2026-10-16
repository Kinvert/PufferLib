"""Offline audit of the five-minute native attention training receipts."""
import argparse
import array
import json
import math
from pathlib import Path
import re

from run_forage_attention_budget import ROOT, read, sha


def audit(out):
    load = lambda name: json.loads((out / name).read_text())
    assert (out / 'COMPLETE').is_file() and not (out / 'failure.json').exists()
    plan, summary = load('plan.json'), load('summary.json')
    assert plan['schema'] == 'forage-attention-five-minute-v1'
    assert plan['fixed_seed'] == 7 and not plan['seed_swept']
    assert sha(Path(plan['binary'])) == plan['binary_sha256']
    assert sha(Path(__file__).with_name('run_forage_attention_long.py')) == plan['tool_sha256']
    for path, expected in plan['source_sha256'].items():
        assert sha(ROOT / path) == expected
    short = ROOT / 'build/sprite-forage-attention-budget-5060-20261015'
    assert sha(short / 'training.json') == plan['short_training_sha256']
    previous = json.loads((short / 'training.json').read_text())
    training, evaluations = load('training.json'), load('evaluations.json')
    assert len(training) == 2 and len(evaluations) == 4
    receipts = []
    weights_checked = 0
    for job, record in zip(plan['jobs'], training):
        assert record['name'] == job['name'] and record['mode'] == job['mode']
        assert record['steps'] == job['steps']
        folder = out / job['name']
        requested = folder / 'runtime/config/sprite_forage.ini'
        assert sha(requested) == job['config_sha256']
        assert sha(folder / 'runtime/config/default.ini') == job['default_sha256']
        config = read(requested)
        assert config.getint('base', 'seed') == 7
        assert config.get('base', 'load_model_path') == 'None'
        assert config.getint('train', 'total_timesteps') == job['steps']
        prior = next(r for r in previous if r['mode'] == job['mode'])
        prior_config_path = Path(prior['runtime']) / 'config/sprite_forage.ini'
        assert sha(prior_config_path) == job['reference_config_sha256']
        prior_config = read(prior_config_path)
        for section in ('env', 'policy', 'vec'):
            assert dict(config[section]) == dict(prior_config[section])
        assert {k: v for k, v in config['train'].items() if k != 'total_timesteps'} == {
            k: v for k, v in prior_config['train'].items() if k != 'total_timesteps'}
        checkpoint = Path(record['final_checkpoint'])
        resolved = read(checkpoint.parent / 'resolved.ini')
        for section in ('env', 'policy', 'vec', 'train'):
            assert dict(config[section]) == dict(resolved[section])
        assert sorted(int(Path(p).stem) for p in record['checkpoints']) == job['checkpoint_steps']
        assert len(record['checkpoints']) == 8
        for path, digest in record['checkpoints'].items():
            weight = out / path
            assert sha(weight) == digest and weight.stat().st_size == record['parameters'] * 4
            assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            weights_checked += 1
        receipt = load(job['name'] + '.process.json')
        assert all(record[k] == v for k, v in receipt.items())
        assert record['process_sps'] == record['steps'] / receipt['process_seconds']
        assert record['target_percent_error'] == 100 * (receipt['process_seconds'] / 300 - 1)
        metrics = read(folder / 'logs/sprite_forage' / (job['name'] + '.ini'))['metrics']
        assert float(metrics['agent_steps'].split(',')[-1]) == record['steps']
        assert float(metrics['sps'].split(',')[-1]) == record['native_last_bin_sps']
        assert float(metrics['uptime'].split(',')[-1]) == record['native_uptime']
        receipts.append((job['name'], receipt))
        ev = [e for e in evaluations if e['mode'] == record['mode']]
        assert sorted(e['world'] for e in ev) == [420, 1337]
        for e in ev:
            label = job['name'] + '-eval-' + str(e['world'])
            receipt = load(label + '.process.json')
            assert all(e[k] == v for k, v in receipt.items())
            text = (out / (label + '.log')).read_text()
            match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
                r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', text)
            assert match and float(match[1]) == e['grape_success'] and float(match[2]) == e['perf']
            assert int(match[3]) == e['episodes'] == 32 and int(match[4]) == job['parameters']
            command = load(label + '.command.json')
            for expected in ('--base.load_model_path=' + str(checkpoint), '--base.eval_agents=1',
                    '--base.eval_episodes=32', '--base.seed=7', '--env.seed=' + str(e['world']), '--env.max_steps=512'):
                assert expected in command
            receipts.append((label, receipt))
        row = next(r for r in summary['rows'] if r['mode'] == record['mode'])
        assert row['evaluation_perf'] == sum(e['perf'] for e in ev) / 2
        assert row['evaluation_grape_success'] == sum(e['grape_success'] for e in ev) / 2
        assert row['train_seconds'] == record['process_seconds'] and row['process_sps'] == record['process_sps']
    for label, receipt in receipts:
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert receipt['process_seconds'] == receipt['monotonic_finished'] - receipt['monotonic_started']
        assert sha(out / (label + '.log')) == receipt['log_sha256']
        assert not load(label + '.gpu-query.json')['compute_pids'].strip()
    ordered = sorted((r for _, r in receipts), key=lambda r: r['monotonic_started'])
    assert all(a['monotonic_finished'] <= b['monotonic_started'] for a, b in zip(ordered, ordered[1:]))
    result = dict(status='PASS', native_processes=6, fresh_training_jobs=2, checkpoints=weights_checked,
        evaluations=4, assigned_episode_executions=128, fixed_training_seed=7,
        unchanged_mode_specific_learners=True, all_weights_finite=True,
        actual_time_target_errors_percent=[r['target_percent_error'] for r in training],
        scope='Offline artifact audit, not independent neural math or per-episode evaluator certification',
        quality_or_pareto_claim=False)
    supplemental = out / 'time-matched'
    if (supplemental / 'COMPLETE').is_file():
        extra = lambda name: json.loads((supplemental / name).read_text())
        rule = extra('rule.json')
        assert rule['tool_sha256'] == sha(Path(__file__).with_name('forage_attention_timepoint.py'))
        assert rule['prepared_monotonic'] < extra('started.json')['monotonic_started']
        selections, points, evs = extra('selection.json'), extra('summary.json'), extra('evaluations.json')
        assert len(selections) == len(points) == 2 and len(evs) == 4
        added = []
        for selection in selections:
            tr = next(r for r in training if r['mode'] == selection['mode'])
            captures = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)',
                (out / (tr['name'] + '.log')).read_text())
            assert len(captures) == 8
            candidates = [(abs(int(ns) / 1e9 - tr['monotonic_started'] - 300),
                int(ns), int(s), int(b)) for s, b, ns in captures]
            _, ns, steps, size = min(candidates)
            elapsed = ns / 1e9 - tr['monotonic_started']
            assert selection['steps'] == steps and selection['elapsed_seconds'] == elapsed
            weight = Path(selection['checkpoint'])
            assert sha(weight) == selection['sha256'] == tr['checkpoints'][str(weight.relative_to(out))]
            assert size == weight.stat().st_size == tr['parameters'] * 4
            group = [e for e in evs if e['mode'] == selection['mode']]
            assert sorted(e['world'] for e in group) == [420, 1337]
            for e in group:
                label = tr['name'] + '-eval-' + str(e['world'])
                source = out if e['reused_existing_evaluation'] else supplemental
                command = json.loads((source / (label + '.command.json')).read_text())
                assert '--base.load_model_path=' + str(weight) in command
                text = (source / (label + '.log')).read_text()
                assert sha(source / (label + '.log')) == e['log_sha256']
                match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
                    r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', text)
                assert match and float(match[1]) == e['grape_success'] and float(match[2]) == e['perf']
                assert int(match[3]) == 32 and int(match[4]) == tr['parameters']
                if not e['reused_existing_evaluation']:
                    receipt = extra(label + '.process.json')
                    assert receipt['returncode'] == 0 and all(e[k] == v for k, v in receipt.items())
                    assert not extra(label + '.gpu-query.json')['compute_pids'].strip()
                    assert receipt['monotonic_started'] >= ordered[-1]['monotonic_finished']
                    added.append(receipt)
            point = next(p for p in points if p['mode'] == selection['mode'])
            assert point['steps'] == steps and point['elapsed_seconds'] == elapsed
            assert point['elapsed_sps'] == steps / elapsed
            assert point['perf'] == sum(e['perf'] for e in group) / 2
            assert point['grape_success'] == sum(e['grape_success'] for e in group) / 2
        added.sort(key=lambda r: r['monotonic_started'])
        assert all(a['monotonic_finished'] <= b['monotonic_started'] for a, b in zip(added, added[1:]))
        result['time_selected_supplement'] = dict(status='PASS', added_native_evaluations=len(added),
            added_episode_executions=32 * len(added), target_seconds=300,
            checkpoint_elapsed_seconds=[s['elapsed_seconds'] for s in selections],
            selected_by_score=False, strict_300_second_cap=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = audit(args.out.resolve())
    (args.out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
