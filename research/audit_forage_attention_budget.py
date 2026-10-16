"""Offline receipt audit; no native process, GPU query or model execution."""
import argparse
import array
import json
import math
from pathlib import Path
import re

from run_forage_attention_budget import BASELINE, ROOT, read, sha


def audit(out):
    load = lambda name: json.loads((out / name).read_text())
    assert (out / 'COMPLETE').is_file() and not (out / 'failure.json').exists()
    plan, summary = load('plan.json'), load('summary.json')
    assert sha(Path(__file__).with_name('run_forage_attention_budget.py')) == plan['tool_sha256']
    assert sha(BASELINE / 'summary.json') == plan['baseline_summary_sha256']
    for binary, digest in plan['binaries'].items():
        assert sha(Path(binary)) == digest
    for source, digest in plan['main_sources'].items():
        assert sha(ROOT / source) == digest
    training, calibrations, evaluations = load('training.json'), load('calibrations.json'), load('evaluations.json')
    assert len(training) == 3 and len(calibrations) == 2 and len(evaluations) == 7
    assert [r['mode'] for r in training] == ['full_state', 'partial_state', 'partial_pixels']
    assert plan['fixed_train_seed'] == 7 and plan['seed_swept'] is False
    receipts = []
    weights_checked = 0
    for record in training + calibrations:
        name = record['name']
        receipt = load(name + '.process.json')
        assert all(record[k] == v for k, v in receipt.items())
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert receipt['process_seconds'] == receipt['monotonic_finished'] - receipt['monotonic_started']
        assert sha(out / (name + '.log')) == receipt['log_sha256']
        assert record['process_sps'] == record['steps'] / receipt['process_seconds']
        folder = out / name
        requested = folder / 'runtime/config/sprite_forage.ini'
        assignment = load(name + '.assignment.json')
        assert sha(requested) == assignment['requested_sha256']
        config = read(requested)
        assert sha(folder / 'runtime/config/default.ini') == plan['default_sha256']
        assert config.getint('base', 'seed') == 7
        assert config.getint('train', 'total_timesteps') == assignment['steps'] == record['steps']
        checkpoint = Path(record['final_checkpoint'])
        resolved = read(checkpoint.parent / 'resolved.ini')
        for section in ('env', 'policy', 'train', 'vec'):
            assert dict(resolved[section]) == dict(config[section])
        epochs = record['steps'] // assignment['batch']
        cadence = 0 if record['calibration'] else assignment['checkpoint_epoch_cadence']
        expected = sorted(set([record['steps']] + ([i * assignment['batch']
            for i in range(cadence, epochs + 1, cadence)] if cadence else [])))
        assert sorted(int(Path(p).stem) for p in record['checkpoints']) == expected
        for path, digest in record['checkpoints'].items():
            weight = out / path
            assert sha(weight) == digest
            assert weight.stat().st_size == record['parameters'] * 4
            assert all(math.isfinite(v) for v in array.array('f', weight.read_bytes()))
            weights_checked += 1
        metrics = read(folder / 'logs/sprite_forage' / (name + '.ini'))['metrics']
        assert float(metrics['agent_steps'].split(',')[-1]) == record['steps']
        assert float(metrics['sps'].split(',')[-1]) == record['native_last_bin_sps']
        if record['calibration']:
            full = next(r for r in training if r['mode'] == record['mode'])
            assert full['steps'] == record['frozen_full_steps']
            assert record['monotonic_finished'] < full['monotonic_started']
        receipts.append(receipt)
    for record in evaluations:
        trained = next(r for r in training if r['mode'] == record['mode'])
        label = trained['name'] + '-eval-' + str(record['world']) + ('-repeat' if record['repeat'] else '')
        receipt = load(label + '.process.json')
        assert all(record[k] == v for k, v in receipt.items())
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert receipt['process_seconds'] == receipt['monotonic_finished'] - receipt['monotonic_started']
        assert sha(out / (label + '.log')) == receipt['log_sha256']
        match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
            r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', (out / (label + '.log')).read_text())
        assert match and float(match[1]) == record['grape_success'] and float(match[2]) == record['perf']
        assert int(match[3]) == record['episodes'] == 32 and int(match[4]) == trained['parameters']
        command = load(label + '.command.json')
        assert '--base.load_model_path=' + trained['final_checkpoint'] in command
        assert '--env.max_steps=512' in command and '--base.eval_agents=1' in command
        receipts.append(receipt)
    for row in summary['rows']:
        tr = next(r for r in training if r['mode'] == row['mode'])
        ev = [r for r in evaluations if r['mode'] == row['mode'] and not r['repeat']]
        assert sorted(r['world'] for r in ev) == [420, 1337]
        assert row['evaluation_perf'] == sum(r['perf'] for r in ev) / 2
        assert row['evaluation_grape_success'] == sum(r['grape_success'] for r in ev) / 2
        assert row['train_seconds'] == tr['process_seconds']
    for before, after in zip(sorted(receipts, key=lambda r: r['monotonic_started']),
            sorted(receipts, key=lambda r: r['monotonic_started'])[1:]):
        assert before['monotonic_finished'] <= after['monotonic_started']
    full = training[0]
    for name, digest in full['checkpoints'].items():
        previous = BASELINE / 'mossy-fern-2/checkpoints/sprite_forage/mossy-fern-2' / Path(name).name
        assert sha(previous) == digest
    repeat = next(r for r in evaluations if r['repeat'])
    original = next(r for r in evaluations if r['mode'] == repeat['mode'] and r['world'] == repeat['world'] and not r['repeat'])
    assert (repeat['perf'], repeat['grape_success']) == (original['perf'], original['grape_success'])
    for query in out.glob('*.gpu-query.json'):
        assert not load(query.name)[-1]['compute_pids'].strip()
    return dict(status='PASS', native_processes=len(receipts), training_jobs=3, calibration_jobs=2,
        evaluations=7, assigned_episode_executions=224, checkpoint_files=weights_checked,
        historical_full_state_all_four_checkpoints_byte_identical=True,
        audit_scope='Artifacts, config, times and aggregate metrics only; no fresh neural/evaluator execution',
        pareto_or_generalization_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = audit(args.out.resolve())
    (args.out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
