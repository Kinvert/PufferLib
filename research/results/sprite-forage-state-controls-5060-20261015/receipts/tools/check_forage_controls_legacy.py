"""Fresh bounded native training versus the frozen original attention binary."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_forage_attention_budget import ROOT, BUILD, read, sha


def main():
    packet = ROOT / 'build/sprite-forage-state-ablations-5060-20261015'
    assert (packet / 'smoke/COMPLETE').is_file()
    out = packet / 'legacy-control'
    (out / 'runtime/config').mkdir(parents=True, exist_ok=False)
    source = packet / 'smoke/quiet-otter-1/runtime/config'
    config = read(source / 'sprite_forage.ini')
    config['base'].update(run_id='calm-robin-8', checkpoint_dir=str(out / 'checkpoints'), log_dir=str(out / 'logs'))
    with (out / 'runtime/config/sprite_forage.ini').open('x') as file:
        config.write(file)
    shutil.copyfile(source / 'default.ini', out / 'runtime/config/default.ini')
    command = [str(BUILD / 'puffer'), 'train']
    with (ROOT / 'build/connect4cnn/hardware-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        assert not active.strip()
        begin = time.monotonic()
        with (out / 'train.log').open('x') as file:
            result = subprocess.run(command, cwd=out / 'runtime', stdout=file, stderr=subprocess.STDOUT,
                timeout=20, env=dict(os.environ, PUFFER_CHECKPOINT_RECEIPTS='1'))
        end = time.monotonic()
    assert result.returncode == 0
    weight = out / 'checkpoints/sprite_forage/calm-robin-8/0000000000065536.bin'
    control = packet / 'smoke/quiet-otter-1/checkpoints/sprite_forage/quiet-otter-1/0000000000065536.bin'
    receipt = dict(command=command, returncode=result.returncode, binary_sha256=sha(BUILD / 'puffer'),
        monotonic_started=begin, monotonic_finished=end, process_seconds=end-begin, steps=65536,
        legacy_checkpoint_sha256=sha(weight), new_checkpoint_sha256=sha(control),
        byte_identical=weight.read_bytes() == control.read_bytes(), train_log_sha256=sha(out / 'train.log'))
    (out / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert receipt['byte_identical']
    (out / 'COMPLETE').write_text('Original action/observation contract training matches frozen binary bytes.\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
