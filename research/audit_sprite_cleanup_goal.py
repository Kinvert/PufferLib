"""Offline completion audit for the October16 sprite cleanup; never launches GPU work."""
import array
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/claude/cnn')
RESULT = ROOT / 'research/results/sprite-attention-5060-20261016'
GAMES = ('catch', 'platform', 'maze', 'runner', 'key', 'push')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit():
    cleanup = ROOT / 'build/sprite-cleanup-20261016'
    shared = ROOT / 'build/sprite-attention-pilots-20261016'
    lane = ROOT / 'build/sprite-lanedodge-pilots-20261016'
    build = ROOT / 'build/sprite-attention-20261016-v2'
    lane_build = ROOT / 'build/sprite-lanedodge-matched-20261016'
    evidence = {}
    for name in ('common', *GAMES, 'lanedodge'):
        folder = ROOT / f'ocean/sprite_{name}'
        assert folder.is_dir() and not folder.is_symlink(), name
    evidence['authoritative_sources_local'] = True

    source_pairs = []
    for game in GAMES:
        for file in (f'sprite_{game}.h', 'sprites.h'):
            source_pairs.append((ROOT / f'ocean/sprite_{game}/{file}',
                build / f'source/ocean/sprite_{game}/{file}'))
    for file in ('sprite.h', 'attention.h'):
        source_pairs.append((ROOT / f'ocean/sprite_common/{file}',
            build / f'source/ocean/sprite_common/{file}'))
    for file in ('sprite_lanedodge.h', 'sprites.h'):
        source_pairs.append((ROOT / f'ocean/sprite_lanedodge/{file}',
            lane_build / f'source/ocean/sprite_lanedodge/{file}'))
    for main, captured in source_pairs:
        assert sha(main) == sha(captured), main
    evidence['current_production_headers_match_tested_research_builds'] = len(source_pairs)

    for game in GAMES:
        for kind in ('games', 'render', 'attention'):
            text = (cleanup / f'final-tests/sprite_{game}-{kind}.log').read_text()
            assert 'PASS' in text
            assert not any(term in text for term in ('ERROR:', 'runtime error:', 'Assertion', 'LeakSanitizer'))
        assert (cleanup / f'traces/sprite_{game}-before.bin').read_bytes() == (
            cleanup / f'traces/sprite_{game}-after.bin').read_bytes()
    assert (cleanup / 'lane-tests/exit-status.txt').read_text().strip() == '0'
    evidence['main_sanitizer_suites'] = 19
    evidence['before_after_exact_state_traces'] = 6

    for packet in (cleanup, shared, lane):
        assert (packet / 'COMPLETE').is_file()
    baseline = read(cleanup / 'summary.json')
    assert all(x['all_state_checkpoints_exact'] for x in baseline['games'].values())
    evidence['cleanup_state_trainings'] = len(read(cleanup / 'training.json'))
    pixels = read(cleanup / 'pixel-canaries.json')
    for game in GAMES:
        pair = [r for r in pixels if r['game'] == game]
        assert len(pair) == 2 and pair[0]['checkpoint_sha256'] == pair[1]['checkpoint_sha256']
    evidence['cleanup_full_image_pixel_canaries'] = len(pixels)

    all_processes = []
    for packet, count in ((shared, 97), (lane, 17)):
        processes = list(packet.glob('*.process.json'))
        assert len(processes) == count, (packet, len(processes))
        for path in processes:
            data = read(path)
            assert data['returncode'] == 0 and not data['timed_out']
            label = path.name.removesuffix('.process.json')
            assert sha(packet / f'{label}.log') == data['log_sha256']
            assert data['monotonic_finished'] > data['monotonic_started']
            all_processes.append(data)
        math_log = (packet / 'encoder-math.log').read_text()
        assert math_log.count('Attention CUDA PASS') == 10
        assert all(f'B={b} ' in math_log for b in (1, 3, 8, 64, 256))
    all_processes.sort(key=lambda r: r['monotonic_started'])
    assert all(a['monotonic_finished'] <= b['monotonic_started']
        for a, b in zip(all_processes, all_processes[1:]))
    evidence['matched_native_processes_successful_and_nonoverlapping'] = len(all_processes)
    evidence['independent_cuda_encoder_cases'] = 20

    shared_canaries = read(shared / 'canaries.json')
    assert len(shared_canaries) == 24
    for first, second in zip(shared_canaries[::2], shared_canaries[1::2]):
        assert first['checkpoints'] == second['checkpoints']
    for name in ('happy-otter', 'mystic-maple'):
        pair = read(lane / f'{name}-canaries.json')
        assert pair[0]['checkpoints'] == pair[1]['checkpoints']
    evidence['exact_canary_training_pairs'] = 14

    models = read(shared / 'training.json') + read(lane / 'training.json')
    assert len(models) == 14
    checkpoint_count = 0
    for model in models:
        for name, digest in model['checkpoints'].items():
            path = Path(model['checkpoint']).parent / name
            assert path.stat().st_size == model['parameters'] * 4
            assert sha(path) == digest
            assert all(math.isfinite(v) for v in array.array('f', path.read_bytes()))
            checkpoint_count += 1
    evidence['finite_shape_bound_pilot_checkpoints'] = checkpoint_count

    episodes = 0
    for packet in (shared, lane):
        for path in packet.glob('*.csv'):
            with path.open() as file:
                data = list(csv.DictReader(file))
            ids = sorted(int(r['episode_id']) for r in data)
            expected = 32 if '-world-' in path.name else 3
            assert ids == list(range(expected)), path
            assert all(0 <= float.fromhex(r['perf']) <= 1 for r in data)
            episodes += len(data)
    assert episodes == 1022
    evidence['assigned_episode_rows'] = episodes

    old_lane = ROOT / ('build/sprite-lanedodge-tilecnn-1/runtime/checkpoints/'
        'sprite_lanedodge/ld-tilecnn-d1/0000000001048576.bin')
    new_lane = next(m for m in models if m.get('mode') == 3)
    assert sha(old_lane) == sha(Path(new_lane['checkpoint']))
    evidence['lane_original_pixel_training_reproduced'] = True

    for game in (*GAMES, 'lanedodge', 'g2048'):
        binary = cleanup / f'main-builds/sprite_{game}' if game != 'g2048' else cleanup / 'main-builds/g2048'
        assert binary.is_file() and binary.stat().st_size > 0
    assert 'FAILED' not in (cleanup / 'main-builds/sources-audit.log').read_text()
    evidence['main_normal_native_targets_compiled'] = 8
    evidence['original_reader_failure_preserved'] = (shared / 'failure.json').exists()
    assert read(shared / 'resume.json')['native_failures_retried'] is False
    evidence['status'] = 'complete_after_reporting_reader_recovery'
    evidence['limits'] = ['one independent training seed', 'development evaluation worlds',
        'positive/zero encoder math fixtures only', 'learned gaze usefulness unproven',
        'no Nature/Pareto/generalization claim']
    return evidence


if __name__ == '__main__':
    result = audit()
    result['audit_tool_sha256'] = sha(Path(__file__))
    (RESULT / 'completion-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
