"""One paired geometry interaction: factored XY + one-hot, relative coordinates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import forage_state_ablations as panel
from forage_state_encoding import REFERENCE
from run_forage_attention_budget import ROOT, read, sha

ONEHOT = ROOT / 'build/sprite-forage-state-onehot-5060-20261015'


def prepare(out):
    assert (ONEHOT / 'full/COMPLETE').is_file()
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((ONEHOT / 'plan.json').read_text())
    source = next(j for j in original['jobs'] if j['gaze_mode'] == 2)
    build = ROOT / 'build/sprite-forage-onehot-relative-build-20261015'
    name = 'mossy-robin-4'
    folder = out / name
    (folder / 'config').mkdir(parents=True)
    for file in ('sprite_forage.ini', 'default.ini'):
        shutil.copyfile(ONEHOT / source['name'] / 'config' / file, folder / 'config' / file)
    job = {**source, 'name': name, 'relative': 1, 'binary': str(build / 'puffer'),
        'binary_sha256': sha(build / 'puffer'), 'build_flags': (build / 'attention-build.env').read_text(),
        'scalar_reference': 'silver-robin-6', 'manifests': {}}
    for local, captured in [('ocean/sprite_forage/sprite_forage.h', 'ocean/sprite_forage/sprite_forage.h'),
            ('ocean/sprite_forage/attention.h', 'ocean/sprite_forage/attention.h'),
            ('research/sprite_attention_eval.cu', 'research/sprite_exact_eval.cu')]:
        assert sha(ROOT / local) == sha(build / 'source' / captured) == original['sources'][local]
    for world in panel.WORLDS:
        config = read(folder / 'config/sprite_forage.ini')
        config['env']['max_steps'] = '512'
        config['sprite_eval']['seed'] = str(world)
        ini = folder / f'manifest-{world}.ini'
        with ini.open('x') as file:
            config.write(file)
        command = [job['binary'], 'sprite_eval_manifest', str(ini)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr
        path = folder / f'manifest-{world}.csv'
        path.write_text(result.stdout)
        actual = panel.rows(path)
        reference = panel.rows(Path(source['manifests'][str(world)]['path']))
        assert len(actual) == len(reference) == 128
        assert [[r[k] for k in panel.WORLD_KEYS] for r in actual] == [[r[k] for k in panel.WORLD_KEYS] for r in reference]
        job['manifests'][str(world)] = dict(path=str(path), sha256=sha(path), command=command)
    panel.save(out / 'plan.json', {**original, 'jobs': [job],
        'relative_wrapper_sha256': sha(Path(__file__)),
        'onehot_absolute_reference_sha256': sha(ONEHOT / 'full/summary.json')})


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
            assert plan['relative_wrapper_sha256'] == sha(Path(__file__))
            panel.execute(out, args.mode == 'smoke')
            folder = out / ('smoke' if args.mode == 'smoke' else 'full')
            (folder / 'COMPLETE').write_text('Single assigned one-hot/relative candidate complete.\n')
    except BaseException as error:
        folder = out if args.mode == 'prepare' else out / ('smoke' if args.mode == 'smoke' else 'full')
        if folder.is_dir():
            panel.save(folder / 'failure.json', dict(error=repr(error)))
        raise
