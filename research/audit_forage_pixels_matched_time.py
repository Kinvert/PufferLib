"""Offline receipt/recipe/world/checkpoint audit for the matched state/pixel pilot."""
import argparse
import array
import json
import math
from pathlib import Path
import re
import statistics

from forage_state_ablations import WORLD_KEYS, rows, save
from run_forage_attention_budget import ROOT, read, sha


def audit(packet):
    plan = json.loads((packet / 'plan.json').read_text())
    assert plan['schema'] == 'forage-matched-time-v1'
    assert plan['fixed_train_seed'] == 7 and plan['seed_swept'] is False
    assert plan['actions'] == [5, 20, 15]
    assert sha(ROOT / 'research/forage_pixels_matched_time.py') == plan['tool_sha256']
    binary = Path(plan['binary'])
    assert sha(binary) == plan['binary_sha256']
    assert (binary.parent / 'attention-build.env').read_text() == plan['build_flags']
    captured = {'research/sprite_attention_eval.cu': 'research/sprite_exact_eval.cu',
        'research/forage_attention_dispatch.cu': 'research/sprite_encoders.cu'}
    for name, digest in plan['sources'].items():
        assert sha(binary.parent / 'source' / captured.get(name, name)) == digest
    budgets = json.loads((packet / 'budgets.json').read_text())
    process_times, checks, output = [], 0, []
    common = None
    for stage_name in ('calibration', 'full'):
        stage = packet / stage_name
        assert (stage / 'COMPLETE').is_file()
        assert not (packet / ('failure-' + ('calibrate' if stage_name == 'calibration' else 'run') + '.json')).exists()
        load = lambda path: json.loads((stage / path).read_text())
        training, evaluations, summary = load('training.json'), load('evaluations.json'), load('summary.json')
        assert summary['training'] == training and summary['evaluations'] == evaluations
        assert len(training) == 2
        processes = list(stage.glob('*.process.json'))
        assert len(processes) == len(training) + len(evaluations)
        for path in processes:
            label = path.name.removesuffix('.process.json')
            receipt = json.loads(path.read_text())
            assert receipt['returncode'] == 0 and not receipt['timed_out']
            assert receipt['process_seconds'] == receipt['monotonic_finished']-receipt['monotonic_started']
            assert receipt['log_sha256'] == sha(stage / (label + '.log'))
            assert not load(label + '.gpu-query.json')['compute_pids'].strip()
            command = load(label + '.command.json')
            assert command[0] == str(binary)
            process_times.append((receipt['monotonic_started'], receipt['monotonic_finished']))
        for job, tr in zip(plan['jobs'], training):
            name = job['name']
            assert tr['name'] == name and tr['pixels'] == job['pixels']
            config = read(stage / name / 'runtime/config/sprite_forage.ini')
            assert sha(packet / name / 'config/sprite_forage.ini') == job['config_sha256']
            assert sha(stage / name / 'runtime/config/default.ini') == job['default_sha256']
            assert config.getint('env', 'observation_mode') == job['pixels']
            assert config.getint('env', 'partial_observation') == 1
            assert config.getint('base', 'seed') == 7 and config.get('base', 'load_model_path') == 'None'
            assert config.getint('vec', 'total_agents') == 64 and config.getint('vec', 'num_threads') == 4
            assert config.getint('train', 'horizon') == 64 and config.getint('train', 'minibatch_size') == 256
            recipe = {s: dict(config[s]) for s in ('env', 'vec', 'policy', 'train')}
            recipe['env'].pop('observation_mode')
            recipe['train'].pop('total_timesteps')
            if common is None:
                common = recipe
            assert recipe == common
            assigned_steps = job['calibration_steps'] if stage_name == 'calibration' else budgets[name]['steps']
            assert config.getint('train', 'total_timesteps') == tr['steps'] == assigned_steps
            assert tr['parameters'] == (103976 if job['pixels'] else 83200)
            assert tr['process_sps'] == assigned_steps/tr['process_seconds']
            receipt = load(name + '.process.json')
            assert all(tr[k] == v for k, v in receipt.items())
            checkpoints = tr['checkpoints']
            expected = [assigned_steps] if stage_name == 'calibration' else [assigned_steps*i//8 for i in range(1, 9)]
            assert [c['steps'] for c in checkpoints] == expected
            marks = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)',
                (stage / (name + '.log')).read_text())
            assert len(marks) == len(checkpoints)
            actual_weights = sorted((stage / name / 'checkpoints/sprite_forage' / name).glob('*.bin'))
            assert [str(w) for w in actual_weights] == [c['path'] for c in checkpoints]
            for c, mark in zip(checkpoints, marks):
                path = Path(c['path'])
                assert sha(path) == c['sha256'] and path.stat().st_size == tr['parameters']*4 == int(mark[1])
                assert int(path.stem) == c['steps'] == int(mark[0])
                assert c['elapsed_seconds'] == int(mark[2])/1e9-tr['monotonic_started']
                assert 0 < c['elapsed_seconds'] <= tr['process_seconds']
                assert all(math.isfinite(v) for v in array.array('f', path.read_bytes()))
                checks += 1
            resolved = read(actual_weights[-1].parent / 'resolved.ini')
            for section in ('env', 'vec', 'train', 'policy'):
                assert dict(resolved[section]) == dict(config[section])
            selected = min(checkpoints, key=lambda c: (abs(c['elapsed_seconds']-300), c['elapsed_seconds']))
            assert selected == tr['selected_checkpoint'] and checkpoints[-1] == tr['final_checkpoint']
            metrics = read(stage / name / 'logs/sprite_forage' / (name + '.ini'))['metrics']
            assert float(metrics['agent_steps'].split(',')[-1]) == assigned_steps
            assert float(metrics['sps'].split(',')[-1]) == tr['native_last_bin_sps']
            if stage_name == 'calibration':
                rate = tr['native_last_bin_sps']
                overhead = max(0, tr['process_seconds']-assigned_steps/rate)
                count = max(32768, int((300-overhead)*rate//32768)*32768)
                assert budgets[name]['steps'] == count and budgets[name]['overhead_seconds'] == overhead
                assert budgets[name]['calibration_process_sha256'] == sha(stage / (name + '.process.json'))
                assert budgets[name]['selected_without_scores'] is True
            records = [e for e in evaluations if e['name'] == name]
            modes = ('canary',) if stage_name == 'calibration' else (('nearest300', 'final')
                if selected != checkpoints[-1] else ('nearest300',))
            assert len(records) == (4 if stage_name == 'calibration' else 2*len(modes))
            original = None
            for record in records:
                path = Path(record['csv'])
                assert sha(path) == record['csv_sha256']
                data = rows(path)
                count = record['episodes']
                first_id = 16 if record['label'] == 'tail' else 0
                assert sorted(int(r['episode_id']) for r in data) == list(range(first_id, first_id+count))
                world = record['world']
                manifest_path = Path(job['manifests'][str(world)]['path'])
                assert sha(manifest_path) == job['manifests'][str(world)]['sha256']
                manifest = {r['episode_id']: r for r in rows(manifest_path)}
                for row in data:
                    assert all(row[k] == manifest[row['episode_id']][k] for k in WORLD_KEYS + ('start_observation_hash',))
                    assert 1 <= int(row['decisions']) <= 512 and 0 <= float.fromhex(row['perf']) <= 1
                assert record['perf'] == statistics.fmean(float.fromhex(r['perf']) for r in data)
                assert record['grape_exits'] == statistics.fmean(float(r['successes']) for r in data)
                label = path.stem
                assert all(record[k] == v for k, v in load(label + '.process.json').items())
                command = load(label + '.command.json')
                assert '--base.load_model_path=' + record['checkpoint'] in command
                if stage_name == 'calibration':
                    ordered = sorted(data, key=lambda r: int(r['episode_id']))
                    if record['label'] == 'graph':
                        original = ordered
                    else:
                        assert ordered == (original[-1:] if record['label'] == 'tail' else original)
            if stage_name == 'full':
                for mode in modes:
                    ev = [e for e in records if e['selection'] == mode]
                    assert sorted(e['world'] for e in ev) == [420, 1337]
                    checkpoint = selected if mode == 'nearest300' else checkpoints[-1]
                    assert all(e['checkpoint'] == checkpoint['path'] for e in ev)
                    output.append(dict(name=name, pixels=job['pixels'], selection=mode,
                        steps=checkpoint['steps'], elapsed_seconds=checkpoint['elapsed_seconds'],
                        elapsed_sps=checkpoint['elapsed_sps'], parameters=tr['parameters'],
                        train_process_seconds=tr['process_seconds'], process_sps=tr['process_sps'],
                        collection_perf=statistics.fmean(e['perf'] for e in ev),
                        grape_exits=statistics.fmean(e['grape_exits'] for e in ev)))
    for world in (420, 1337):
        manifests = [rows(Path(j['manifests'][str(world)]['path'])) for j in plan['jobs']]
        assert [[r[k] for k in WORLD_KEYS] for r in manifests[0]] == [[r[k] for k in WORLD_KEYS] for r in manifests[1]]
    ordered = sorted(process_times)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
    result = dict(audit='PASS', processes=len(ordered), checkpoints=checks,
        exact_episode_executions=sum(e['episodes'] for s in ('calibration', 'full')
            for e in json.loads((packet / s / 'evaluations.json').read_text())),
        fixed_train_seed=7, independent_training_n=1, seed_swept=False,
        rows=output, quality_or_pareto_claim=False)
    save(packet / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.packet.resolve()), indent=2))
