"""Captured native full-frame encoder preparation, execution and evidence audit."""
import argparse
import array
import configparser
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODULE_SOURCE = Path(__file__).read_bytes()
SCHEMA = 'fullframe-compression-packet-v1'
GAMES = {'flappycnn': (640, 420, 2), 'pongcnn': (640, 620, 3)}
NEW_SOURCES = ['fullframe_compression.cu', 'fullframe_compression_io.h',
    'fullframe_compression_reference.cu', 'test_fullframe_compression.cu',
    'native_encoder_reference.cu', 'policy_metadata.cu']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def config(*paths):
    cfg = configparser.ConfigParser(interpolation=None)
    for path in paths:
        with Path(path).open() as f:
            cfg.read_file(f)
    return cfg


def write_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        cfg.write(f)


def fixed_logging(cfg):
    # Native train uses these logging keys even when no sweep is launched.
    if not cfg.has_section('sweep'):
        cfg['sweep'] = {}
    cfg['sweep'].update(metric='perf', metric_distribution='linear', goal='maximize',
        max_runs='1', gpus='1', downsample='2', max_suggestion_cost='600')


def process(argv, cwd, log, timeout, env=None, gpu=None, expect_failure=False):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    require(not log.exists(), 'Never overwrite or retry a process log')
    record = dict(argv=list(map(str, argv)), cwd=str(cwd), started_ns=time.monotonic_ns())
    save(log.with_suffix('.started.json'), record)
    with log.open('xb') as f:
        p = subprocess.Popen(record['argv'], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
        record['pid'] = p.pid
        stop, interference, samples = threading.Event(), [], []
        def monitor():
            while not stop.wait(5):
                try:
                    foreign = [pid for pid in gpu.pids() if os.getpgid(pid) != p.pid]
                    samples.append(dict(monotonic_ns=time.monotonic_ns(), foreign=foreign))
                    if foreign:
                        interference.extend(foreign)
                        os.killpg(p.pid, signal.SIGTERM)
                        return
                except ProcessLookupError:
                    continue
                except Exception as error:
                    interference.append(str(error))
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    return
        if gpu:
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
        timeout_hit = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_hit = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        stop.set()
        if gpu:
            thread.join()
    record.update(finished_ns=time.monotonic_ns(), returncode=p.returncode,
        timeout=timeout_hit, log_sha256=sha(log), interference=interference, gpu_samples=samples,
        expected_failure=expect_failure)
    record['seconds'] = (record['finished_ns'] - record['started_ns']) / 1e9
    save(log.with_suffix('.process.json'), record)
    require((p.returncode != 0 if expect_failure else p.returncode == 0)
        and not timeout_hit and not interference, f'Process failed; see {log}')
    return record


def run_json(argv, cwd, log, timeout=30):
    process(argv, cwd, log, timeout)
    return json.loads(Path(log).read_text())


def common_config(game, model):
    cfg = config(ROOT / 'config/default.ini', ROOT / f'config/{game}.ini')
    for section in list(cfg.sections()):
        if section == 'sweep' or section.startswith('sweep.'):
            cfg.remove_section(section)
    values = dict(base=dict(seed='7', **{'async': '0'}, cudagraphs='1', checkpoint_interval='16',
        load_model_path='None', load_enemy_model_path='None', eval_episodes='0'),
        vec=dict(total_agents='16', num_buffers='1', num_threads='2', num_policies='1', hist_policy_percent='0'),
        policy=dict(encoder='8', compression_model=str(model), compression_init='None', hidden_size='128', num_layers='1'),
        train=dict(gpus='1', horizon='16', minibatch_size='16', total_timesteps='8192'),
        selfplay=dict(enabled='0'))
    for section, entries in values.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg[section].update(entries)
    cfg['env'].update(representation='0', representation_mode='0')
    fixed_logging(cfg)
    return cfg


def prepare(args):
    out = args.out.resolve()
    require(not out.exists(), 'Prepare needs a fresh output directory')
    require(os.environ.get('NVCC_ARCH') == 'sm_120', 'Set NVCC_ARCH=sm_120 explicitly')
    require(not any(os.environ.get(k) for k in ('NVCC_EXTRA', 'NVCC_APPEND_FLAGS', 'NVCC_PREPEND_FLAGS')),
        'Unrecorded compiler flags refused')
    out.mkdir(parents=True)
    source = out / 'source'
    source.mkdir()
    files = [ROOT / 'build.sh', ROOT / 'research/fullframe_compression.py',
        ROOT / 'research/fullframe_compression.sh', ROOT / 'research/recipes/fullframe_compression.ini']
    for folder in ('src', 'ocean/connect4cnn', 'ocean/flappycnn', 'ocean/pongcnn'):
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.h', '.cu', '.c', '.sh')]
    files += [ROOT / 'research' / name for name in NEW_SOURCES]
    files += [ROOT / 'config/default.ini'] + [ROOT / f'config/{g}.ini' for g in GAMES]
    hashes = {}
    for path in sorted(set(files)):
        rel = path.relative_to(ROOT)
        dest = source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        hashes[str(rel)] = sha(path)
        shutil.copyfile(path, dest)
        require(sha(dest) == hashes[str(rel)], 'Source capture differs')
    external = {}
    for name in ('vendor', 'raylib-5.5_linux_amd64'):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
        for p in (ROOT / name).rglob('*'):
            if p.is_file() and (p.suffix in ('.h', '.c', '.a') or p.name.endswith('.hpp')):
                external[str(p.relative_to(ROOT))] = sha(p)
    save(out / 'sources.json', hashes)
    save(out / 'external.json', external)
    save(out / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        arch='sm_120', cuda_home=os.environ['CUDA_HOME'], nccl_root=os.environ['NCCL_ROOT']))
    cuda = Path(os.environ['CUDA_HOME'])
    nccl = Path(os.environ['NCCL_ROOT'])
    save(out / 'toolchain.json', dict(nvcc_sha256=sha(cuda / 'bin/nvcc'),
        compiler=subprocess.check_output([cuda / 'bin/nvcc', '--version'], text=True)))
    builds = {}
    for game, (H, W, actions) in GAMES.items():
        folder = out / 'builds' / game
        folder.mkdir(parents=True)
        defs = ['-DCNN_IMAGE_V1', f'-DCNN_IMAGE_HEIGHT={H}', f'-DCNN_IMAGE_WIDTH={W}',
            '-DCNN_IMAGE_CHANNELS=3', '-DCNN_COMPRESSION_RESEARCH']
        environment = dict(os.environ, NVCC_EXTRA=' '.join(defs), NVCC_ARCH='sm_120')
        process(['bash', 'build.sh', game, folder / 'puffer', '--float'], source, folder / 'build.log', 300, environment)
        cmd = [str(cuda / 'bin/nvcc'), '-O2', '-std=c++17', '-arch=sm_120', '--threads', '1',
            '-Xcompiler=-fopenmp', '-Xcompiler=-Wno-narrowing', '--diag-suppress=2361',
            '--diag-suppress=111', '--diag-suppress=128', '-I.', '-Isrc', '-Ivendor',
            '-Iraylib-5.5_linux_amd64/include', '-I' + str(cuda / 'include/cccl'), '-I' + str(nccl / 'include'),
            *defs, '-DPUFFER_' + game.upper(), '-DENV_HEADER="ocean/' + game + '/' + game + '.h"',
            '-DPUFFER_ENV_NAME="' + game + '"', 'research/policy_metadata.cu',
            'raylib-5.5_linux_amd64/lib/libraylib.a', '-L' + str(cuda / 'lib64'),
            '-L' + str(cuda / 'lib64/stubs'), '-L' + str(nccl / 'lib'),
            '-lcudart', '-lcublas', '-lcurand', '-lcusolver', '-lnccl', '-lnvidia-ml',
            '-lGL', '-lpthread', '-ldl', '-lrt', '-o', str(folder / 'metadata')]
        process(cmd, source, folder / 'metadata-build.log', 300)
        builds[game] = {n: sha(folder / n) for n in ('puffer', 'metadata')}
        for model in range(3):
            cfg = common_config(game, model)
            file = out / f'configs/{game}-{model}.ini'
            write_config(file, cfg)
            layout = run_json([folder / 'metadata', '--describe-flex', file, str(actions)], source,
                folder / f'layout-{model}.log')
            storage = run_json([folder / 'metadata', '--storage', file], source, folder / f'storage-{model}.log')
            require(layout['raw_checkpoint_contiguous'] and layout['encoder_parameters'] ==
                expected_parameters(model, H, W), 'Native layout differs from independent count')
            require(storage['storage_floor_bytes'] < 4 * 2**30, 'Registered allocation exceeds initial packet envelope')
            save(out / f'layouts/{game}-{model}.json', dict(layout=layout, storage=storage, config_sha256=sha(file)))
        if game == 'flappycnn':
            checkcmd = cmd.copy()
            checkcmd[checkcmd.index('research/policy_metadata.cu')] = 'research/test_fullframe_compression.cu'
            checkcmd += ['-DPUFFERLIB_BUILD_MAIN', '-DPRECISION_FLOAT']
            checkcmd[checkcmd.index(str(folder / 'metadata'))] = str(out / 'checks')
            process(checkcmd, source, out / 'checks-build.log', 300)
    for rel, digest in hashes.items():
        require(sha(ROOT / rel) == digest and sha(source / rel) == digest, 'Inputs changed during build')
    save(out / 'plan.json', dict(schema=SCHEMA, numerical_protocol='ffc-native-branches-v2', builds=builds, checks_sha256=sha(out / 'checks'),
        configs={p.name: sha(p) for p in (out / 'configs').glob('*.ini')},
        math_seconds=900, smoke_seconds=900, training_seed=7, evaluation_seed=420,
        source_manifest_sha256=sha(out / 'sources.json'), external_manifest_sha256=sha(out / 'external.json')))
    print(json.dumps(inspect(out), indent=2))


def expected_parameters(model, H, W):
    if model < 2:
        return (36776, 36608)[model]
    H, W = (H + 3) // 4, (W + 3) // 4
    total, ci = 0, 3
    for k, s, co in ((8, 4, 32), (4, 2, 64), (3, 1, 64)):
        H, W = (H - k) // s + 1, (W - k) // s + 1
        total += co * (k * k * ci + 1)
        ci = co
    return total + 128 * (H * W * ci + 1)


def inspect_registration(out, game, model):
    value = read(out / f'layouts/{game}-{model}.json')
    require(value['config_sha256'] == sha(out / f'configs/{game}-{model}.ini'), 'Registration recipe changed')
    for key, prefix in (('layout', 'layout'), ('storage', 'storage')):
        log = out / f'builds/{game}/{prefix}-{model}.log'
        proof = read(log.with_suffix('.process.json'))
        require(proof['returncode'] == 0 and not proof['timeout'] and not proof['interference']
            and sha(log) == proof['log_sha256'] and value[key] == read(log), 'Registration differs from native receipt')
    H, W, actions = GAMES[game]
    layout, storage = value['layout'], value['storage']
    require(layout['encoder_parameters'] == expected_parameters(model, H, W)
        and layout['image_contract']['shape_hwc'] == [H, W, 3]
        and layout['compression']['model'] == model and layout['discrete_actions'] == actions
        and layout['raw_checkpoint_contiguous'] and storage['learner_batch'] == 16
        and storage['storage_floor_bytes'] < 4 * 2**30, 'Registration leaves fixed packet contract')


def inspect(out):
    out = Path(out).resolve()
    plan = read(out / 'plan.json')
    require(plan['schema'] == SCHEMA and plan['training_seed'] == 7 and plan['evaluation_seed'] == 420,
        'Unexpected protocol or seed assignment')
    require(sha(out / 'sources.json') == plan['source_manifest_sha256'] and
        sha(out / 'external.json') == plan['external_manifest_sha256'], 'Changed source manifests')
    for rel, digest in read(out / 'sources.json').items():
        require(sha(out / 'source' / rel) == digest, f'Changed captured source: {rel}')
    for rel, digest in read(out / 'external.json').items():
        require(sha(out / 'source' / rel) == digest, f'Changed external build input: {rel}')
    for game, binaries in plan['builds'].items():
        for name, digest in binaries.items():
            require(sha(out / 'builds' / game / name) == digest, 'Changed binary')
    require(sha(out / 'checks') == plan['checks_sha256'], 'Changed CUDA harness')
    for name, digest in plan['configs'].items():
        require(sha(out / 'configs' / name) == digest, 'Changed frozen recipe')
    for game in GAMES:
        for model in range(3):
            inspect_registration(out, game, model)
    return plan


def host_check(out):
    inspect(out)
    target = out / 'host-check.json'
    require(not target.exists(), 'Host checks already completed')
    value = run_json([out / 'checks', '--host'], out / 'source', out / 'host-check.log')
    require(value['status'] == 'passed' and value['gpu_queried'] is False, 'Host check failed')
    save(target, value)


class GPU:
    def __init__(self, out, hardware, stage, seconds):
        require(hardware in ('5060', '5090'), 'Need explicit --hardware')
        self.out, self.hardware, self.stage = out, hardware, stage
        self.start = time.monotonic_ns()
        self.deadline = time.monotonic() + seconds
        self.smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'

    def pids(self):
        data = subprocess.check_output([self.smi, '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=10)
        return [int(line.strip()) for line in data.splitlines() if line.strip()]

    def __enter__(self):
        inspect(self.out)
        marker = self.out / f'stages/{self.stage}.started.json'
        require(not marker.exists(), 'Stage already started; completed/failed allocations cannot restart')
        lock = ROOT / 'build/connect4cnn/hardware-benchmark.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock.open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not self.pids(), 'Shared GPU busy')
        identity = subprocess.check_output([self.smi, '--query-gpu=name,uuid,driver_version,memory.total',
            '--format=csv,noheader'], text=True, timeout=10).strip()
        require(len(identity.splitlines()) == 1 and self.hardware in identity, 'GPU identity mismatch')
        self.identity = identity
        supervisor = self.out / f'stages/{self.stage}.supervisor.py'
        supervisor.parent.mkdir(parents=True, exist_ok=True)
        supervisor.write_bytes(MODULE_SOURCE)
        save(marker, dict(started_ns=self.start, hardware=identity, seconds_limit=self.deadline-time.monotonic(),
            supervisor_sha256=sha(supervisor)))
        return self

    def run(self, argv, log, timeout=240, cwd=None, expect_failure=False):
        require(not self.pids(), 'Competing GPU process detected')
        require(shutil.disk_usage(self.out).free >= 20 * 2**30, 'Free space reserve reached')
        require(sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file() and not p.is_symlink()) < 10 * 2**30,
            'Artifact cap reached')
        remaining = self.deadline - time.monotonic()
        require(remaining > 1, 'Stage wall-clock cap reached')
        environment = dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        environment.pop('PUFFER_WEIGHTS_MIRROR', None)
        environment.pop('PUFFER_SAVE_INITIAL_CHECKPOINT', None)
        result = process(argv, cwd or self.out / 'source', log, min(timeout, remaining), environment,
            gpu=self, expect_failure=expect_failure)
        # A just-exited direct CUDA worker can remain briefly in NVML. Only its
        # exact PID may drain; an unrelated PID is never treated as our cleanup.
        cleanup_deadline = time.monotonic() + 2
        active = self.pids()
        while active and set(active) <= {result['pid']} and time.monotonic() < cleanup_deadline:
            time.sleep(.05)
            active = self.pids()
        require(not active, 'GPU work remains or competing job appeared after process')
        return result

    def __exit__(self, kind, error, traceback):
        if error:
            save(self.out / f'stages/{self.stage}.failure.json', dict(error=str(error), finished_ns=time.monotonic_ns()))
        self.lock.close()


def math_stage(out, hardware):
    require(read(out / 'host-check.json')['status'] == 'passed', 'Run host-check first')
    cases = [(h, w, b, m, f) for h, w in ((160, 192), (161, 193))
        for b in (1, 3) for m in range(3) for f in range(3)]
    cases += [(h, w, b, m, f) for h, w in ((640, 420), (640, 620))
        for b in (1, 16) for m in range(3) for f in (0, 2)]
    rows = []
    with GPU(out, hardware, 'math', 900) as gpu:
        folder = out / 'math'
        folder.mkdir(exist_ok=True)
        require(not any(folder.iterdir()), 'Numerical outputs already exist; never retry a used allocation')
        for h, w, b, m, f in cases:
            repeats = []
            for repeat in range(2):
                name = f'h{h}-w{w}-b{b}-m{m}-f{f}-r{repeat}'
                folder = out / 'math' / name
                receipt = gpu.run([out / 'checks', h, w, b, m, f, folder, 'math'], out / f'logs/math/{name}.log')
                value = read(folder / 'result.json')
                require(value['status'] == 'passed' and value['parameters'] == expected_parameters(m, h, w)
                    and value['numerical_protocol'] == 'ffc-native-branches-v2', 'Incomplete math case')
                row = dict(name=name, result=value, process=receipt,
                    hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
                rows.append(row)
                repeats.append(row)
                save(out / 'math-progress.json', dict(cases=rows))
            require(repeats[0]['hashes'] == repeats[1]['hashes'], 'Fresh-process math results differ')
        sanitizer = Path(os.environ['CUDA_HOME']) / 'bin/compute-sanitizer'
        require(sanitizer.is_file(), 'Need existing compute-sanitizer for device memory gate; install nothing')
        memory = []
        for model in range(3):
            folder = out / f'math/memcheck-m{model}'
            receipt = gpu.run([sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', out / 'checks',
                161, 193, 1, model, 0, folder, 'math'], out / f'logs/math/memcheck-m{model}.log', 120)
            require('ERROR SUMMARY: 0 errors' in (out / f'logs/math/memcheck-m{model}.log').read_text(),
                'Missing clean CUDA memory summary')
            memory.append(dict(model=model, process=receipt, result=read(folder / 'result.json')))
        save(out / 'stages/math.json', dict(status='passed', cases=rows, native_processes=len(rows),
            memory_checks=memory, hardware=gpu.identity, finished_ns=time.monotonic_ns(), mathematical_scope='encoder-only'))


def checkpoint_manifest(out, game, model, checkpoint, cfg_path, layout, receipt, steps, timestamp):
    values = array.array('f')
    with checkpoint.open('rb') as f:
        values.frombytes(f.read())
    require(len(values) == layout['total_parameters'] and all(map(math.isfinite, values)), 'Invalid checkpoint values')
    value = dict(schema='fullframe-compression-checkpoint-v1', game=game, model=model,
        sha256=sha(checkpoint), bytes=checkpoint.stat().st_size, layout=layout,
        config_sha256=sha(cfg_path), config_path=str(cfg_path.relative_to(out)), source_manifest_sha256=sha(out / 'sources.json'),
        plan_sha256=sha(out / 'plan.json'), initialization='scratch', steps=steps,
        train_seconds=(timestamp-receipt['started_ns'])/1e9, process_seconds=receipt['seconds'])
    require(0 < value['train_seconds'] <= receipt['seconds'], 'Invalid checkpoint timestamp')
    save(checkpoint.with_suffix('.manifest.json'), value)
    return value


def validate_checkpoint(out, game, model, checkpoint):
    value = read(checkpoint.with_suffix('.manifest.json'))
    layout = read(out / f'layouts/{game}-{model}.json')['layout']
    require(value['schema'] == 'fullframe-compression-checkpoint-v1' and value['game'] == game
        and value['model'] == model and value['sha256'] == sha(checkpoint)
        and value['bytes'] == checkpoint.stat().st_size == layout['parameter_payload_bytes']
        and value['layout'] == layout and value['source_manifest_sha256'] == sha(out / 'sources.json')
        and value['plan_sha256'] == sha(out / 'plan.json'), 'Checkpoint manifest/layout binding differs')
    require(sha(out / value['config_path']) == value['config_sha256'], 'Checkpoint training configuration differs')
    return value


def export_encoder(out, game, model, checkpoint, directory):
    require(not directory.exists(), 'Export needs a fresh directory')
    proof = validate_checkpoint(out, game, model, checkpoint)
    layout = proof['layout']
    raw = checkpoint.read_bytes()
    tensors, blocks, offset = [], [], 0
    selected = [t for t in layout['parameter_tensors'] if t['group'] == 'encoder']
    require(len(selected) == 8, 'Unexpected encoder registration')
    for i, t in enumerate(selected):
        start, elements = t['offset_bytes'], t['elements']
        block = raw[start:start+elements*4]
        require(len(block) == elements*4, 'Truncated encoder tensor')
        tensors.append(dict(name=('bias' if i % 2 else 'weight') + '_' + str(i//2),
            offset=offset, elements=elements, shape=t['shape']))
        blocks.append(block)
        offset += elements
    payload = b''.join(blocks)
    values = array.array('f')
    values.frombytes(payload)
    require(all(map(math.isfinite, values)), 'Nonfinite export')
    H, W, _ = GAMES[game]
    value = dict(schema='fullframe-compression-encoder-v1', model=model, height=H, width=W,
        channels=3, dtype='float32', layout='NHWC', range='0,1',
        activation='relu-zero-derivative-0', area='box4-rowmajor-partial-edge',
        adaptive='floor-start-ceil-end-4x4', padding='same-trailing-extra-or-nature-valid',
        elements=offset, tensors=tensors, layers=layout['compression']['layers'],
        sha256=hashlib.sha256(payload).hexdigest(), source_checkpoint_sha256=proof['sha256'],
        source_manifest_sha256=proof['source_manifest_sha256'], initialization=proof['initialization'])
    require(offset == layout['encoder_parameters'], 'Export count differs')
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(directory.name + '.tmp')
    require(not temporary.exists(), 'Unfinished export exists')
    temporary.mkdir()
    (temporary / 'encoder.bin').write_bytes(payload)
    save(temporary / 'encoder.json', value)
    temporary.rename(directory)
    return value


def episode_rows(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 2 and {int(r['episode_id']) for r in rows} == {0, 1}, 'Missing/duplicate assigned episode')
    for row in rows:
        for key, value in row.items():
            if key in ('episode_return', 'perf', 'native_perf', 'point_fraction_lower', 'point_fraction_upper'):
                parsed = float.fromhex(value) if value.startswith(('0x', '-0x')) else float(value)
                require(math.isfinite(parsed), 'Nonfinite evaluation row')
    return sorted([{k: v for k, v in row.items() if k != 'slot'} for row in rows], key=lambda r: int(r['episode_id']))


def native_metrics(path, process_seconds):
    cfg = config(path)
    values = {k: [float(v) for v in text.split(',')] for k, text in cfg['metrics'].items()}
    require(all(math.isfinite(v) for series in values.values() for v in series), 'Nonfinite native metric')
    require(all(k in values and values[k] for k in ('uptime', 'agent_steps', 'sps')), 'Missing native timing')
    # PufferLib serializes bin means here, not final steps divided by final time.
    # Keep the native samples without inventing a native-average throughput.
    return dict(series=values, series_are_bin_means=True,
        binned_uptime_within_process=all(0 <= t <= process_seconds for t in values['uptime']),
        native_average_sps=None,
        clock_note='Native bin means are preserved; process timing uses a monotonic clock.')


def smoke_stage(out, hardware):
    require(read(out / 'stages/math.json')['status'] == 'passed', 'Accepted math required')
    jobs, evaluations, imports = [], [], []
    with GPU(out, hardware, 'smoke', 900) as gpu:
        for game in GAMES:
            binary = out / 'builds' / game / 'puffer'
            for model in range(3):
                replicas = []
                for repeat in range(2):
                    folder = out / f'smoke/{game}-m{model}-r{repeat}'
                    cfg = config(out / f'configs/{game}-{model}.ini')
                    fixed_logging(cfg)
                    cfg['base'].update(run_id='happy-cat-1', checkpoint_dir=str(folder / 'checkpoints'),
                        log_dir=str(folder / 'metrics'))
                    write_config(folder / f'config/{game}.ini', cfg)
                    (folder / 'config/default.ini').touch()
                    receipt = gpu.run([binary, 'train', '--headless'], folder / 'train.log', 45, folder)
                    logged = re.findall(r'PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)',
                        (folder / 'train.log').read_text())
                    require([int(r[0]) for r in logged] == [4096, 8192], 'Missing/extra checkpoint schedule')
                    layout = read(out / f'layouts/{game}-{model}.json')['layout']
                    checkpoints = []
                    for step, size, stamp in logged:
                        checkpoint = folder / f'checkpoints/{game}/happy-cat-1/{int(step):016d}.bin'
                        require(int(size) == layout['parameter_payload_bytes'], 'Checkpoint receipt size')
                        checkpoints.append(dict(path=str(checkpoint.relative_to(out)),
                            manifest=checkpoint_manifest(out, game, model, checkpoint,
                                folder / f'config/{game}.ini', layout, receipt, int(step), int(stamp))))
                    resolved = config(folder / f'checkpoints/{game}/happy-cat-1/resolved.ini')
                    require({s: dict(resolved[s]) for s in resolved.sections()} == {s: dict(cfg[s]) for s in cfg.sections()},
                        'Native resolved recipe differs')
                    metrics = folder / f'metrics/{game}/happy-cat-1.ini'
                    job = dict(game=game, model=model, repeat=repeat, process=receipt,
                        process_sps=8192 / receipt['seconds'], checkpoints=checkpoints,
                        metrics=str(metrics.relative_to(out)), metrics_sha256=sha(metrics),
                        native=native_metrics(metrics, receipt['seconds']))
                    jobs.append(job)
                    replicas.append(job)
                    save(out / 'smoke-progress.json', dict(jobs=jobs, evaluations=evaluations, imports=imports))
                require([c['manifest']['sha256'] for c in replicas[0]['checkpoints']] ==
                    [c['manifest']['sha256'] for c in replicas[1]['checkpoints']], 'Same-seed training checkpoints differ')
                checkpoint = out / replicas[0]['checkpoints'][-1]['path']
                reference = None
                for mode, slots, graphs in (('graph', 1, 1), ('repeat', 1, 1), ('eager', 1, 0), ('slots', 2, 1)):
                    folder = out / f'evaluation/{game}-m{model}-{mode}'
                    cfg = config(out / f'configs/{game}-{model}.ini')
                    cfg['base'].update(load_model_path=str(checkpoint), cudagraphs=str(graphs))
                    cfg['policy']['compression_init'] = 'None'
                    cfg['eval_exact'] = dict(seed='420', episodes='2', episode_offset='0', slots=str(slots),
                        output=str(folder / 'episodes.csv'))
                    if game == 'flappycnn':
                        cfg['env']['max_steps'] = '256'
                        cfg['eval_exact']['max_steps'] = '256'
                    else:
                        cfg['eval_exact']['max_decisions'] = '256'
                    write_config(folder / f'config/{game}.ini', cfg)
                    (folder / 'config/default.ini').touch()
                    validate_checkpoint(out, game, model, checkpoint)
                    manifest = folder / 'manifest.csv'
                    process([binary, 'eval_exact_manifest', folder / f'config/{game}.ini'], out / 'source', manifest, 30)
                    receipt = gpu.run([binary, 'eval_exact', '--headless'], folder / 'eval.log', 30, folder)
                    rows = episode_rows(folder / 'episodes.csv')
                    with manifest.open() as stream:
                        starts = list(csv.DictReader(stream))
                    keys = ('episode_id', 'env_seed', 'policy_seed', 'start_rng', 'start_world_hash', 'start_observation_hash')
                    require([[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in starts], 'World assignments differ')
                    require(reference is None or rows == reference, 'Graph/eager/slot episode rows differ')
                    reference = rows
                    evaluations.append(dict(game=game, model=model, mode=mode, slots=slots, process=receipt,
                        rows=rows, csv=str((folder / 'episodes.csv').relative_to(out)), csv_sha256=sha(folder / 'episodes.csv')))
                if game == 'flappycnn':
                    directory = out / f'exports/{game}-{model}'
                    exported = export_encoder(out, game, model, checkpoint, directory)
                    target = out / f'import-checks/model-{model}'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    receipt = gpu.run([out / 'checks', '--import-check', out / f'configs/{game}-{model}.ini',
                        directory, target], out / f'logs/import/model-{model}.log', 30)
                    imports.append(dict(model=model, export=exported, process=receipt, valid=True))
                    for kind in ('model', 'hash', 'same-size-layout', 'nonfinite', 'truncated'):
                        invalid = out / f'invalid-imports/m{model}-{kind}'
                        shutil.copytree(directory, invalid)
                        value = read(invalid / 'encoder.json')
                        if kind == 'model':
                            value['model'] = (model + 1) % 3
                        elif kind == 'hash':
                            value['sha256'] = '0' * 64
                        elif kind == 'same-size-layout':
                            value['layers'][0][8] += 1
                        elif kind == 'nonfinite':
                            data = bytearray((invalid / 'encoder.bin').read_bytes())
                            data[:4] = array.array('f', [float('nan')]).tobytes()
                            (invalid / 'encoder.bin').write_bytes(data)
                            value['sha256'] = sha(invalid / 'encoder.bin')
                        else:
                            (invalid / 'encoder.bin').write_bytes((invalid / 'encoder.bin').read_bytes()[:-4])
                        save(invalid / 'encoder.json', value)
                        receipt = gpu.run([out / 'checks', '--import-check', out / f'configs/{game}-{model}.ini',
                            invalid, out / f'invalid-imports/rejected-{model}-{kind}'],
                            out / f'logs/import/m{model}-{kind}.log', 30, expect_failure=True)
                        reason = {'model': 'incompatible encoder manifest', 'hash': 'encoder hash mismatch',
                            'same-size-layout': 'encoder graph mismatch', 'nonfinite': 'nonfinite encoder weight',
                            'truncated': 'encoder byte count mismatch'}[kind]
                        rejection_log = out / f'logs/import/m{model}-{kind}.log'
                        require(reason in rejection_log.read_text(), 'Import failed for an unexpected reason')
                        imports.append(dict(model=model, kind=kind, process=receipt, rejected=True,
                            reason=reason, log=str(rejection_log.relative_to(out))))
        save(out / 'stages/smoke.json', dict(status='passed', jobs=jobs, evaluations=evaluations,
            imports=imports, hardware=gpu.identity, learning_or_frontier_claim=False))


def report(out):
    inspect(out)
    result = dict(schema='fullframe-compression-report-v1', learning_or_frontier_claim=False,
        stages={}, source_manifest_sha256=sha(out / 'sources.json'))
    result['registered_models'] = []
    for game, (H, W, _) in GAMES.items():
        for model in range(3):
            registered = read(out / f'layouts/{game}-{model}.json')
            result['registered_models'].append(dict(game=game, model=model, raw_shape_hwc=[H, W, 3],
                encoder_parameters=registered['layout']['encoder_parameters'],
                total_parameters=registered['layout']['total_parameters'],
                storage_floor_bytes=registered['storage']['storage_floor_bytes'],
                peak_memory_certified=False, learner_batch=16, actors=16, horizon=16,
                recipe_sha256=registered['config_sha256']))
    for stage in ('math', 'smoke'):
        file = out / f'stages/{stage}.json'
        require(not (out / f'stages/{stage}.failure.json').exists(), 'Failed stage cannot be reported as accepted')
        result['stages'][stage] = 'passed' if file.exists() else 'missing'
        if file.exists():
            require(read(file)['status'] == 'passed', 'Unaccepted stage result')
        started = out / f'stages/{stage}.started.json'
        if started.exists() and 'supervisor_sha256' in read(started):
            require(sha(out / f'stages/{stage}.supervisor.py') == read(started)['supervisor_sha256'],
                'Changed executed supervisor source')
    processes = []
    for file in out.rglob('*.process.json'):
        receipt = read(file)
        log = file.with_suffix('').with_suffix('.log')
        # Manifest CSVs use the same receipt convention with a .csv data file.
        if not log.exists():
            log = file.with_suffix('').with_suffix('.csv')
        require(log.exists() and sha(log) == receipt['log_sha256'], 'Process log hash differs')
        require(not receipt['timeout'] and not receipt['interference'] and
            (receipt['returncode'] != 0 if receipt['expected_failure'] else receipt['returncode'] == 0),
            'Failed or contaminated process in packet')
        processes.append(receipt)
    if result['stages']['math'] == 'passed':
        math_proof = read(out / 'stages/math.json')
        require(len(math_proof['cases']) == 120, 'Incomplete numerical panel')
        expected = {(h, w, b, m, f, r) for h, w in ((160, 192), (161, 193))
            for b in (1, 3) for m in range(3) for f in range(3) for r in range(2)}
        expected |= {(h, w, b, m, f, r) for h, w in ((640, 420), (640, 620))
            for b in (1, 16) for m in range(3) for f in (0, 2) for r in range(2)}
        actual = set()
        paired = {}
        for row in math_proof['cases']:
            value = row['result']
            require(value == read(out / 'math' / row['name'] / 'result.json') and value['status'] == 'passed'
                and value['actor_train_equal'] and value['eager_graph_repeats'] == 3, 'Incomplete numerical result')
            repeat = int(row['name'].rsplit('-r', 1)[1])
            key = tuple(value[k] for k in ('height', 'width', 'batch', 'model', 'fixture'))
            actual.add((*key, repeat))
            paired.setdefault(key, []).append(row['hashes'])
            require(row['process'] == read(out / f'logs/math/{row["name"]}.process.json'),
                'Numerical process receipt differs')
            for name, digest in row['hashes'].items():
                require(sha(out / 'math' / row['name'] / name) == digest, 'Changed numerical array/receipt')
        require(actual == expected and all(len(pair) == 2 and pair[0] == pair[1] for pair in paired.values()),
            'Missing, duplicate or nonrepeatable numerical assignment')
        memory = math_proof['memory_checks']
        require(len(memory) == 3 and {m['model'] for m in memory} == {0, 1, 2}, 'Missing CUDA memory gate')
        for row in memory:
            log = out / f'logs/math/memcheck-m{row["model"]}.log'
            require(row['process'] == read(log.with_suffix('.process.json'))
                and 'ERROR SUMMARY: 0 errors' in log.read_text()
                and row['result'] == read(out / f'math/memcheck-m{row["model"]}/result.json')
                and row['result']['status'] == 'passed', 'Changed or incomplete CUDA memory evidence')
        result['math_processes'] = 120
        values = [row['result'] for row in math_proof['cases']]
        result['numerical'] = dict(protocol='ffc-native-branches-v2', distinct_cases=60,
            fresh_processes=120, device_memory_checks=3,
            finite_difference_probes=sum(v['finite_difference_probes'] for v in values),
            cases_with_branch_disagreement=sum(v['branch_disagreements'] > 0 for v in values),
            maximum_branch_distance=max(v['branch_distance'] for v in values),
            maximum_layer_forward_error=max(v['max_layer_forward_error'] for v in values),
            maximum_actual_branch_gradient_error=max(v['actual_branch_gradient_error'] for v in values),
            maximum_actual_branch_input_error=max(v['actual_branch_input_error'] for v in values),
            full_double_branch_errors_retained=True, mathematical_scope='encoder-only')
    if result['stages']['smoke'] == 'passed':
        smoke = read(out / 'stages/smoke.json')
        require(len(smoke['jobs']) == 12 and len(smoke['evaluations']) == 24
            and len(smoke['imports']) == 18, 'Incomplete smoke/import coverage')
        require({(j['game'], j['model'], j['repeat']) for j in smoke['jobs']} ==
            {(g, m, r) for g in GAMES for m in range(3) for r in range(2)}, 'Missing/duplicate training assignment')
        require({(e['game'], e['model'], e['mode']) for e in smoke['evaluations']} ==
            {(g, m, mode) for g in GAMES for m in range(3) for mode in ('graph', 'repeat', 'eager', 'slots')},
            'Missing/duplicate evaluation assignment')
        for job in smoke['jobs']:
            folder = out / f'smoke/{job["game"]}-m{job["model"]}-r{job["repeat"]}'
            require(job['process'] == read(folder / 'train.process.json'), 'Training process receipt differs')
            require([c['manifest']['steps'] for c in job['checkpoints']] == [4096, 8192], 'Checkpoint schedule differs')
            require(sha(out / job['metrics']) == job['metrics_sha256'] and
                native_metrics(out / job['metrics'], job['process']['seconds']) == job['native'], 'Changed native metrics')
            for entry in job['checkpoints']:
                current = validate_checkpoint(out, job['game'], job['model'], out / entry['path'])
                require(current == entry['manifest'], 'Changed checkpoint provenance')
        for ev in smoke['evaluations']:
            folder = out / f'evaluation/{ev["game"]}-m{ev["model"]}-{ev["mode"]}'
            require(ev['process'] == read(folder / 'eval.process.json'), 'Evaluation process receipt differs')
            require(sha(out / ev['csv']) == ev['csv_sha256'] and episode_rows(out / ev['csv']) == ev['rows'],
                'Changed evaluation evidence')
        require({(c['model'], c.get('kind', 'valid')) for c in smoke['imports']} ==
            {(m, k) for m in range(3) for k in ('valid', 'model', 'hash', 'same-size-layout', 'nonfinite', 'truncated')},
            'Missing/duplicate import assignment')
        for check in smoke['imports']:
            if check.get('rejected'):
                require(check['reason'] in (out / check['log']).read_text(), 'Import rejection reason differs')
        for game in GAMES:
            for model in range(3):
                pair = [j for j in smoke['jobs'] if j['game'] == game and j['model'] == model]
                require([c['manifest']['sha256'] for c in pair[0]['checkpoints']] ==
                    [c['manifest']['sha256'] for c in pair[1]['checkpoints']], 'Training repeat checkpoints differ')
                rows = [e['rows'] for e in smoke['evaluations'] if e['game'] == game and e['model'] == model]
                require(all(r == rows[0] for r in rows), 'Graph/eager/slot-count comparison differs')
            starts = []
            for ev in smoke['evaluations']:
                if ev['game'] == game:
                    starts.append([{k: row[k] for k in ('episode_id', 'env_seed', 'policy_seed',
                        'start_observation_hash', 'start_world_hash', 'start_rng')} for row in ev['rows']])
            require(all(rows == starts[0] for rows in starts), 'Models did not receive identical initial pixel worlds')
        result.update(hardware=smoke['hardware'], training_jobs=12, evaluations=24,
            assigned_episode_executions=48, import_checks=18)
        result['training'] = [dict(game=j['game'], model=j['model'], repeat=j['repeat'],
            process_seconds=j['process']['seconds'], process_sps=j['process_sps'],
            native=j['native'],
            parameters=j['checkpoints'][-1]['manifest']['layout']['total_parameters'],
            checkpoints=[dict(steps=c['manifest']['steps'], train_seconds=c['manifest']['train_seconds'],
                sha256=c['manifest']['sha256']) for c in j['checkpoints']]) for j in smoke['jobs']]
    result['status'] = 'passed' if all(v == 'passed' for v in result['stages'].values()) else 'incomplete'
    result['native_processes'] = len(processes)
    result['gpu_query_or_execution'] = False
    save(out / 'REPORT.json', result)
    lines = ['# Full-frame compression first-increment report', '',
        f'Status: **{result["status"]}**. Encoder correctness and short deterministic integration only.',
        'Source frames: Flappy640x420 RGB; Pong640x620 RGB. No sprite/ROI shortcuts.', '',
        'ProcessSPS includes startup and checkpoint writes; these8192-decision jobs do not establish steady-state speed or learning.', '',
        '|Game|Model|Repeat|Process seconds|Process SPS|Parameters|', '|---|---:|---:|---:|---:|---:|']
    for row in result.get('training', []):
        lines.append(f'|{row["game"]}|{row["model"]}|{row["repeat"]}|{row["process_seconds"]:.6f}|{row["process_sps"]:.1f}|{row["parameters"]}|')
    if 'numerical' in result:
        n = result['numerical']
        lines += ['', f'Numerical acceptance:60declared cases in120fresh processes;{n["finite_difference_probes"]}GPU finite-difference probes;3device memchecks.',
            f'Actual-branch maximum gradient error:{n["maximum_actual_branch_gradient_error"]:.6g};input-gradient error:{n["maximum_actual_branch_input_error"]:.6g}.',
            f'{n["cases_with_branch_disagreement"]}process cases contain recorded near-zero branch differences;full-double branch errors remain in the raw receipts.']
    lines += ['', 'Model0: learned strided compression. Model1: area4 plus matched small CNN. Model2: area4 plus resized RGB Nature/H128.',
        '', 'Read REPORT.json and native receipts for exact coverage. Numerical checks cover the declared fixtures, not arbitrary architectures or independent whole-learner mathematics.',
        'No pretrained learning, held-out generalization, Pareto or SOTA result is claimed.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'inspect', 'host-check', 'math', 'smoke', 'timing', 'report', 'export-encoder'])
    p.add_argument('--out', type=Path)
    p.add_argument('--packet', type=Path)
    p.add_argument('--hardware', choices=['5060', '5090'])
    p.add_argument('--game', choices=list(GAMES))
    p.add_argument('--model', type=int, choices=range(3))
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    if args.command == 'prepare':
        require(args.out is not None, 'Need --out')
        prepare(args)
        return
    require(args.packet is not None, 'Need --packet')
    out = args.packet.resolve()
    if args.command == 'inspect':
        print(json.dumps(inspect(out), indent=2))
    elif args.command == 'host-check':
        host_check(out)
    elif args.command == 'math':
        math_stage(out, args.hardware)
    elif args.command == 'smoke':
        smoke_stage(out, args.hardware)
    elif args.command == 'report':
        report(out)
    elif args.command == 'export-encoder':
        inspect(out)
        require(args.game is not None and args.model is not None and args.checkpoint is not None
            and args.out is not None, 'Need --game --model --checkpoint --out')
        export_encoder(out, args.game, args.model, args.checkpoint.resolve(), args.out.resolve())
    else:
        raise RuntimeError('Optional timing stage is not qualified; use required math/smoke stages first')


if __name__ == '__main__':
    main()
