"""Native pixel ablation: chosen gaze only, without automatic player neighbors."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import forage_pixels_matched_time as timed
from forage_state_ablations import WORLD_KEYS, WORLDS, rows, save
from run_forage_attention_budget import ROOT, read, sha

BUILD = ROOT / 'build/sprite-forage-gaze-only-build-20261015'
REFERENCE = ROOT / 'build/sprite-forage-pixels-matched-5060-20261015'


def prepare(out):
    assert (REFERENCE / 'full/COMPLETE').is_file()
    assert json.loads((REFERENCE / 'audit.json').read_text())['audit'] == 'PASS'
    assert (BUILD / 'math.log').read_text().count('Attention CUDA PASS') == 10
    flags = (BUILD / 'attention-build.env').read_text()
    assert 'FORAGE_PLAYER_VIEW=0\n' in flags and 'FORAGE_GAZE_MODE=2\n' in flags
    sources = {}
    for current, captured in (
            ('ocean/sprite_forage/sprite_forage.h', 'ocean/sprite_forage/sprite_forage.h'),
            ('ocean/sprite_forage/attention.h', 'ocean/sprite_forage/attention.h'),
            ('research/forage_attention_encoder.cu', 'research/forage_attention_encoder.cu'),
            ('research/forage_attention_dispatch.cu', 'research/sprite_encoders.cu'),
            ('research/sprite_attention_eval.cu', 'research/sprite_exact_eval.cu')):
        assert sha(ROOT / current) == sha(BUILD / 'source' / captured)
        sources[current] = sha(ROOT / current)
    out.mkdir(exist_ok=False)
    reference_plan = json.loads((REFERENCE / 'plan.json').read_text())
    previous = next(j for j in reference_plan['jobs'] if j['pixels'] == 1)
    name = 'mossy-owl-3'
    folder = out / name
    (folder / 'config').mkdir(parents=True)
    for file in ('sprite_forage.ini', 'default.ini'):
        shutil.copyfile(REFERENCE / previous['name'] / 'config' / file, folder / 'config' / file)
    manifests = {}
    for world in WORLDS:
        config = read(folder / 'config/sprite_forage.ini')
        config['env']['max_steps'] = '512'
        config['sprite_eval']['seed'] = str(world)
        ini = folder / f'manifest-{world}.ini'
        with ini.open('x') as file:
            config.write(file)
        command = [str(BUILD / 'puffer'), 'sprite_eval_manifest', str(ini)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        path = folder / f'manifest-{world}.csv'
        path.write_text(result.stdout)
        data = rows(path)
        reference = rows(Path(previous['manifests'][str(world)]['path']))
        assert len(data) == len(reference) == 128
        assert [[r[k] for k in WORLD_KEYS] for r in data] == [[r[k] for k in WORLD_KEYS] for r in reference]
        manifests[str(world)] = dict(path=str(path), sha256=sha(path), command=command)
    job = dict(name=name, pixels=1, parameters=87592, calibration_steps=131072,
        manifests=manifests, config_sha256=sha(folder / 'config/sprite_forage.ini'),
        default_sha256=sha(folder / 'config/default.ini'))
    save(out / 'plan.json', dict(schema='forage-gaze-only-v1', jobs=[job],
        binary=str(BUILD / 'puffer'), binary_sha256=sha(BUILD / 'puffer'), sources=sources,
        tool_sha256=sha(ROOT / 'research/forage_pixels_matched_time.py'),
        wrapper_sha256=sha(Path(__file__)), build_flags=flags,
        encoder_checks_sha256=sha(BUILD / 'math.log'), target_seconds=300,
        fixed_train_seed=7, seed_swept=False, independent_training_n=1, actions=[5, 20, 15],
        pixel_inputs=27652, view_tiles=9, player_view=False, downsampling=False,
        reference_packet=str(REFERENCE), reference_plan_sha256=sha(REFERENCE / 'plan.json'),
        reference_audit_sha256=sha(REFERENCE / 'audit.json'), reference_name=previous['name'],
        budget_selection_uses_scores=False,
        timepoint_rule='Nearest checkpoint to 300 monotonic seconds, tie earlier; also report final',
        caveat='Removing eight tiles also reduces projection parameters and allows more decisions at equal time',
        quality_or_pareto_claim=False))
    print('Prepared gaze-only pixels: nine32x32 tiles,27652 values,87592 policy parameters.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'calibrate', 'run'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        if args.mode == 'prepare':
            prepare(out)
        else:
            plan = json.loads((out / 'plan.json').read_text())
            assert plan['schema'] == 'forage-gaze-only-v1'
            assert plan['wrapper_sha256'] == sha(Path(__file__))
            assert plan['reference_audit_sha256'] == sha(REFERENCE / 'audit.json')
            timed.run(out, args.mode == 'calibrate')
    except BaseException as error:
        if out.is_dir():
            save(out / ('failure-' + args.mode + '.json'), dict(error=repr(error)))
        raise
