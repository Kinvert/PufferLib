"""Matched one-hot state follow-up using the frozen native ablation executor."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import forage_state_ablations as panel
from run_forage_attention_budget import ROOT, read, sha

REFERENCE = ROOT / 'build/sprite-forage-state-ablations-5060-20261015'


def prepare(out):
    assert (REFERENCE / 'full/COMPLETE').is_file()
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((REFERENCE / 'plan.json').read_text())
    jobs = []
    for gaze, name in enumerate(('gentle-maple-1', 'mystic-sparrow-2', 'happy-fern-3')):
        previous = next(j for j in original['jobs'] if j['gaze_mode'] == gaze and not j['relative'] and j['partial'])
        build = ROOT / f'build/sprite-forage-onehot-build-20261015-{gaze}'
        folder = out / name
        (folder / 'config').mkdir(parents=True)
        for file in ('sprite_forage.ini', 'default.ini'):
            shutil.copyfile(REFERENCE / previous['name'] / 'config' / file, folder / 'config' / file)
        binary = build / 'puffer'
        for local, captured in [('ocean/sprite_forage/sprite_forage.h', 'ocean/sprite_forage/sprite_forage.h'),
                ('ocean/sprite_forage/attention.h', 'ocean/sprite_forage/attention.h'),
                ('research/sprite_attention_eval.cu', 'research/sprite_exact_eval.cu')]:
            assert sha(ROOT / local) == sha(build / 'source' / captured)
        job = dict(name=name, gaze_mode=gaze, relative=0, partial=1, state_onehot=1,
            action_sizes=previous['action_sizes'], parameters=previous['parameters']+26112,
            binary=str(binary), binary_sha256=sha(binary), build_flags=(build / 'attention-build.env').read_text(),
            config_sha256=sha(folder / 'config/sprite_forage.ini'),
            default_sha256=sha(folder / 'config/default.ini'), manifests={}, scalar_reference=previous['name'])
        for world in panel.WORLDS:
            config = read(folder / 'config/sprite_forage.ini')
            config['env']['max_steps'] = '512'
            config['sprite_eval']['seed'] = str(world)
            ini = folder / f'manifest-{world}.ini'
            with ini.open('x') as file:
                config.write(file)
            command = [str(binary), 'sprite_eval_manifest', str(ini)]
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            assert result.returncode == 0, result.stderr
            path = folder / f'manifest-{world}.csv'
            path.write_text(result.stdout)
            actual = panel.rows(path)
            reference = panel.rows(Path(previous['manifests'][str(world)]['path']))
            assert len(actual) == len(reference) == 128
            assert [[r[k] for k in panel.WORLD_KEYS] for r in actual] == [[r[k] for k in panel.WORLD_KEYS] for r in reference]
            job['manifests'][str(world)] = dict(path=str(path), sha256=sha(path), command=command)
        jobs.append(job)
    plan = {**original, 'jobs': jobs, 'state_encoding_followup': True,
        'scalar_reference_plan_sha256': sha(REFERENCE / 'plan.json'),
        'scalar_reference_summary_sha256': sha(REFERENCE / 'full/summary.json'),
        'encoding_wrapper_sha256': sha(Path(__file__)),
        'sources': {p: sha(ROOT / p) for p in original['sources']}}
    panel.save(out / 'plan.json', plan)
    print('Prepared three one-hot candidates; all256 episode worlds match their scalar controls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'smoke', 'run'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        if args.mode == 'prepare':
            prepare(out)
        else:
            plan = json.loads((out / 'plan.json').read_text())
            assert plan['encoding_wrapper_sha256'] == sha(Path(__file__))
            panel.execute(out, args.mode == 'smoke')
            folder = out / ('smoke' if args.mode == 'smoke' else 'full')
            (folder / 'COMPLETE').write_text('All three assigned one-hot trainings/evaluations complete.\n')
    except BaseException as error:
        folder = out if args.mode == 'prepare' else out / ('smoke' if args.mode == 'smoke' else 'full')
        if folder.is_dir():
            panel.save(folder / 'failure.json', dict(error=repr(error)))
        raise
