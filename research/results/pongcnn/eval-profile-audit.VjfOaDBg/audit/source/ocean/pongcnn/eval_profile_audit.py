"""Reporting/configuration receipts for the bounded native evaluation audit."""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from confirm_report import MODELS, EVAL, read_ini, save, sha


def prepare(root, original):
    rows = list(csv.DictReader((original / 'results.csv').open()))
    ordered = sorted(rows, key=lambda r: (r['id'], int(r['steps'])))
    failures = [r for r in ordered if r['status'] != 'ok']
    assert len(failures) == 8
    controls = [next(r for r in ordered if r['recipe'] == recipe and
                     r['variant'] == model and r['status'] == 'ok')
                for recipe in 'ab' for model in MODELS]
    fast = [min((r for r in rows if r['variant'] == model and r['status'] == 'ok'),
                key=lambda r: (float(r['eval_wall_s']), r['id'], int(r['steps'])))
            for model in MODELS]
    attempts = []

    def evaluation(row, group, label, version, games, cap):
        name = f'{group}/{label}/{version}'
        directory = root / name
        saved = original / row['id'] / 'evaluations' / f"{int(row['steps']):016d}"
        shutil.copytree(saved / 'config', directory / 'config')
        ini = read_ini(directory / 'config/default.ini', directory / 'config/pongcnn.ini')
        checkpoint = Path(ini['base']['load_model_path'])
        assert sha(checkpoint) == row['checkpoint_sha256']
        attempts.append(dict(name=name, group=group, label=label, version=version,
                             id=row['id'], model=row['variant'], recipe=row['recipe'],
                             steps=int(row['steps']), games=games, cap=cap,
                             original_status=row['status'], original_eval_wall_s=float(row['eval_wall_s']),
                             checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                             original_config={p.name: sha(p) for p in (saved/'config').glob('*.ini')},
                             binary=str((original if version == 'old' else root) / (row['variant']+'.bin'))))

    for r in fast:
        for version in ('old', 'new'):
            evaluation(r, 'validation', r['variant'], version, 32, 30)
    evaluation(fast[0], 'validation', 'intentional-timeout', 'new', 2147483647, 7)
    for i, r in enumerate(failures + controls):
        label = f"{i:02d}-{r['id']}-{r['steps']}"
        for version in (('old', 'new') if i % 2 == 0 else ('new', 'old')):
            evaluation(r, 'panel', label, version, 512, 30)
    for r in controls:
        for version in ('unprofiled', 'instrumented'):
            name = f"performance/{r['recipe']}-{r['variant']}/{version}"
            directory = root/name
            shutil.copytree(original/r['id']/'config', directory/'config')
            attempts.append(dict(name=name, group='performance', label=f"{r['recipe']}-{r['variant']}",
                                 version=version, id=r['id'], model=r['variant'], recipe=r['recipe'],
                                 steps=131072, cap=60 if version == 'unprofiled' else 90,
                                 binary=str(root/(r['variant']+'.bin'))))
    save(root/'plan.json', dict(original=str(original), attempts=attempts, controls=controls,
         failures=failures, validation=fast, selection='Controls: first successful sorted job ID/numeric step per model/recipe. Validation: lowest original successful evaluation wall per model.',
         limits=dict(total_gpu_seconds=2700, diagnostic_evaluations=32, performance_jobs=8, profiled_repeats=8),
         profiler='Nsight Systems CUDA node tracing and NVTX; no CPU sampling or context-switch tracing. One instrumented repeat per cell; no retries.'))
    with (root/'attempts.tsv').open('w') as f:
        for a in attempts:
            f.write('\t'.join(str(a.get(k, '0')) for k in
                              ('group','name','version','binary','cap','games'))+'\n')
    save(root/'original-identities.json', {str(p.relative_to(original)): sha(p)
         for p in original.rglob('*') if p.is_file() and (p.suffix in ('.bin','.ini','.json','.csv','.log','.md') or p.name.endswith('.sha256'))})


def result(root, a):
    directory = root/a['name']
    log = (directory/'output.log').read_text(errors='replace') if (directory/'output.log').exists() else ''
    code = (directory/'exit-code.txt').read_text().strip() if (directory/'exit-code.txt').exists() else 'incomplete'
    final = EVAL.findall(log)
    progress = [dict(steps=int(s), games=int(g), requested=int(n), elapsed=float(t)) for s,g,n,t in
                re.findall(r'CUDA_EVAL_PROGRESS steps=(\d+) games=(\d+) requested=(\d+) elapsed=([\d.]+)', log)]
    r = dict(a, exit_code=code, final=final, progress=progress,
             wall_s=float((directory/'wall.txt').read_text().splitlines()[-1]) if (directory/'wall.txt').exists() else None,
             log_sha256=sha(directory/'output.log') if log else None)
    if a['group'] != 'performance':
        r['checkpoint_unchanged'] = sha(a['checkpoint']) == a['checkpoint_sha256']
        r['success'] = code == '0' and len(final) == 1 and int(final[0][2]) >= a['games'] and int(final[0][3]) == MODELS[a['model']][1]
        r['dashboards'] = log.count('util/gpu_percent')  # Kept alongside raw dashboard text.
        r['missing_key_error'] = 'missing key' in log
    else:
        paths = sorted((directory/'checkpoints/pongcnn/trial').glob('*.bin'))
        r['checkpoints'] = {p.name: dict(sha256=sha(p), params=p.stat().st_size//4,
                                       finite=bool(np.isfinite(np.fromfile(p,dtype=np.float32)).all())) for p in paths}
        metric = directory/'metrics/pongcnn/trial.ini'
        if metric.exists():
            ini = read_ini(metric)
            r['metrics'] = {k: [float(v) for v in value.split(',')] for k,value in ini['metrics'].items()}
            r['effective_updates_per_rollout'] = int(float(ini['train']['replay_ratio']) * 2048 / 2048)
            r['effective_updates_total'] = 64*r['effective_updates_per_rollout']
        r['success'] = (code == '0' and bool(paths) and all(v['finite'] and v['params']==MODELS[a['model']][1] for v in r['checkpoints'].values())
                        and r.get('metrics',{}).get('agent_steps',[0])[-1] == 131072)
    return r


def validate(root):
    plan = json.loads((root/'plan.json').read_text())
    results = [result(root,a) for a in plan['attempts'] if a['group']=='validation']
    pairs = {}
    for model in MODELS:
        pair = [r for r in results if r['label']==model]
        pairs[model] = ('passed' if all(r['success'] and r['checkpoint_unchanged'] and not r['missing_key_error'] for r in pair)
                        and pair[0]['final']==pair[1]['final'] else 'unresolved_or_failed')
    intentional = results[-1]
    timeout_ok = intentional['exit_code']=='124' and bool(intentional['progress']) and not intentional['final']
    passed = all(v=='passed' for v in pairs.values()) and timeout_ok
    save(root/'validation.json', dict(passed=passed, pairs=pairs, intentional_timeout_passed=timeout_ok, attempts=results))
    print(json.dumps(dict(passed=passed,pairs=pairs,intentional_timeout_passed=timeout_ok)))
    return 0 if passed else 1


def report(root):
    plan = json.loads((root/'plan.json').read_text())
    results = [result(root,a) for a in plan['attempts']]
    save(root/'measurements.json', results)
    lines = ['# Pong evaluation/backend audit', '', 'Original: `'+plan['original']+'`.', '',
             'The original 40 training runs and 312 successful / 8 timed-out evaluations remain unchanged.', '',
             '## Evaluation panel', '', '| Checkpoint | Original | Old exit / seconds | New exit / seconds | New last progress steps / matches |',
             '|---|---|---|---|---|']
    for label in dict.fromkeys(a['label'] for a in results if a['group']=='panel'):
        pair = {a['version']:a for a in results if a['group']=='panel' and a['label']==label}
        o,n = pair['old'],pair['new']
        p = n['progress'][-1] if n['progress'] else {}
        lines.append(f"| {label} | {n['original_status']} | {o['exit_code']} / {o['wall_s']} | {n['exit_code']} / {n['wall_s']} | {p.get('steps','—')} / {p.get('games','—')} |")
    lines += ['', '## Short performance diagnostics', '',
              '131,072 decisions, seed 52001; shortened annealing schedule. Instrumented durations include profiler overhead and are not publication throughput.', '',
              '| Cell | Mode | Valid | Process seconds | Native seconds | Updates |', '|---|---|---|---:|---:|---:|']
    for r in results:
        if r['group']=='performance':
            lines.append(f"| {r['label']} | {r['version']} | {r['success']} | {r['wall_s']} | {r.get('metrics',{}).get('uptime',['—'])[-1]} | {r.get('effective_updates_total','—')} |")
    lines += ['', 'Raw configs, commands, low-frequency activity, progress and profiler output accompany each attempt. Full measurements are in `measurements.json`. Interpretation and profiler coverage are recorded separately in `FINDINGS.md`.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('mode', choices=['prepare','validate','report'])
    parser.add_argument('--original', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.root.resolve(), args.original.resolve())
    elif args.mode == 'validate': sys.exit(validate(args.root.resolve()))
    else: report(args.root.resolve())
