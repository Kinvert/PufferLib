"""Evaluate checkpoints chosen by distance to300 elapsed seconds, never score."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from run_forage_attention_budget import ROOT, sha


def main(packet, mode):
    packet = packet.resolve()
    out = packet / 'time-matched'
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    if mode == 'prepare':
        out.mkdir(exist_ok=False)
        save('rule.json', dict(target_seconds=300,
            selection='Minimize absolute distance of native post-rename checkpoint time from monotonic training process start to300 seconds; earlier time breaks ties',
            score_used=False, prepared_monotonic=time.monotonic(), tool_sha256=sha(Path(__file__))))
        return
    assert (packet / 'COMPLETE').is_file() and not (out / 'started.json').exists()
    rule = json.loads((out / 'rule.json').read_text())
    assert rule['tool_sha256'] == sha(Path(__file__))
    save('started.json', dict(monotonic_started=time.monotonic()))
    plan = json.loads((packet / 'plan.json').read_text())
    training = json.loads((packet / 'training.json').read_text())
    original_evals = json.loads((packet / 'evaluations.json').read_text())
    binary = Path(plan['binary'])
    assert sha(binary) == plan['binary_sha256']
    selected, evaluations = [], []
    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for tr in training:
            captures = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)',
                (packet / (tr['name'] + '.log')).read_text())
            assert len(captures) == 8
            points = [dict(steps=int(s), bytes=int(b), monotonic_ns=int(ns),
                elapsed_seconds=int(ns) / 1e9 - tr['monotonic_started']) for s, b, ns in captures]
            assert [p['steps'] for p in points] == tr['checkpoint_steps']
            point = min(points, key=lambda p: (abs(p['elapsed_seconds'] - 300), p['elapsed_seconds']))
            weight = Path(tr['final_checkpoint']).parent / f"{point['steps']:016d}.bin"
            assert weight.stat().st_size == tr['parameters'] * 4 == point['bytes']
            assert sha(weight) == tr['checkpoints'][str(weight.relative_to(packet))]
            row = dict(mode=tr['mode'], **point, checkpoint=str(weight), sha256=sha(weight),
                elapsed_sps=point['steps'] / point['elapsed_seconds'], candidates=points,
                final_checkpoint_reused=weight == Path(tr['final_checkpoint']))
            selected.append(row)
            save('selection.json', selected)
            for world in (420, 1337):
                if row['final_checkpoint_reused']:
                    ev = next(e for e in original_evals if e['mode'] == tr['mode'] and e['world'] == world)
                    evaluations.append(dict(**ev, reused_existing_evaluation=True))
                    save('evaluations.json', evaluations)
                    continue
                label = tr['name'] + '-eval-' + str(world)
                command = json.loads((packet / (label + '.command.json')).read_text())
                command = ['--base.load_model_path=' + str(weight) if x.startswith('--base.load_model_path=') else x for x in command]
                query = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
                    '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
                save(label + '.gpu-query.json', dict(compute_pids=query))
                assert not query.strip()
                save(label + '.command.json', command)
                begin = time.monotonic()
                with (out / (label + '.log')).open('x') as log:
                    result = subprocess.run(command, cwd=packet / tr['name'] / 'runtime', stdout=log,
                        stderr=subprocess.STDOUT, timeout=35, env=os.environ.copy())
                receipt = dict(returncode=result.returncode, monotonic_started=begin,
                    monotonic_finished=time.monotonic(), log_sha256=sha(out / (label + '.log')))
                receipt['process_seconds'] = receipt['monotonic_finished'] - begin
                save(label + '.process.json', receipt)
                assert result.returncode == 0
                match = re.search(r'CUDA_EVAL env=sprite_forage score=([\d.e+-]+) '
                    r'perf=([\d.e+-]+) games=(\d+) params=(\d+)', (out / (label + '.log')).read_text())
                assert match and int(match[3]) == 32 and int(match[4]) == tr['parameters']
                evaluations.append(dict(mode=tr['mode'], world=world, episodes=32, perf=float(match[2]),
                    grape_success=float(match[1]), reused_existing_evaluation=False, **receipt))
                save('evaluations.json', evaluations)
    summary = []
    for selection in selected:
        ev = [e for e in evaluations if e['mode'] == selection['mode']]
        assert len(ev) == 2
        summary.append(dict(mode=selection['mode'], steps=selection['steps'],
            elapsed_seconds=selection['elapsed_seconds'], elapsed_sps=selection['elapsed_sps'],
            target_percent_error=100 * (selection['elapsed_seconds'] / 300 - 1),
            perf=sum(e['perf'] for e in ev) / 2, grape_success=sum(e['grape_success'] for e in ev) / 2))
    save('summary.json', summary)
    (out / 'COMPLETE').write_text('Time-selected evaluations complete; no checkpoint selected by score.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    main(args.packet, args.mode)
