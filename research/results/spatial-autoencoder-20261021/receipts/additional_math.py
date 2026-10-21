"""One fresh bounded local CUDA allocation; retained execution driver, no retries."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import autoencoder_pretrain as ae

packet = Path(sys.argv[1]).resolve()
root = packet / 'additional-math'
rows = []
with ae.Reservation(packet, '5060', 'spatial-additional-math', 300) as gpu:
    root.mkdir()
    (root / 'driver.py').write_bytes(Path(__file__).read_bytes())
    binary = ae.trainer_binary(packet)
    for stride, channels in ae.read(packet / 'spatial-plan.json')['variants']:
        for H, W, B in ((161, 193, 2), (480, 640, 1)):
            name = f's{stride}-c{channels}-{H}x{W}-b{B}'
            folder = root / name
            receipt = gpu.run([binary, 'spatial-math', H, W, B, stride, channels, folder], root / f'{name}.log', 90)
            value = ae.read(folder / 'result.json')
            ae.require(value['status'] == 'passed', 'Additional CUDA shape failed')
            rows.append(dict(case=name, result=value, process=receipt, result_sha256=ae.sha(folder / 'result.json')))
    receipt = gpu.run([binary, 'math', 160, 192, 1, root / 'vector'], root / 'vector.log', 60)
    value = ae.read(root / 'vector/result.json')
    ae.require(value['status'] == 'passed', 'Retained vector CUDA regression failed')
    rows.append(dict(case='retained-vector', result=value, process=receipt, result_sha256=ae.sha(root / 'vector/result.json')))
    result = dict(status='passed', hardware=gpu.identity, rows=rows,
        source_manifest_sha256=ae.sha(packet / 'sources.json'), driver_sha256=ae.sha(root / 'driver.py'),
        memory_qualification=False, learning_or_frontier_claim=False)
    ae.save(packet / 'stages/spatial-additional-math.json', result)
    print(json.dumps(result, indent=2))
