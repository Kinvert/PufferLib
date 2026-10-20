"""Fetch pinned optional weights as data; never import Torch or execute model code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

TAESD = dict(repo='madebyollin/taesd', revision='614f76814bbe30edbe2e627ace1c2234c81a2c0e',
    file='diffusion_pytorch_model.safetensors', bytes=9793292,
    sha256='db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549',
    license='mit', native_encoder8_compatible=False,
    source='https://huggingface.co/madebyollin/taesd', checked='2026-10-20')
MOBILENET = dict(repo='timm/mobilenetv3_small_050.lamb_in1k', revision='5188129022623c1fd7e36401459eda145a81d7e9',
    file='model.safetensors', bytes=6420754,
    sha256='2e3f6937afd4b3704450518a2710168775d6c70ebfb7a0e9aaf06200c6fbe0c4',
    license='apache-2.0', native_encoder8_compatible=False,
    source='https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k', checked='2026-10-20')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def inventory(path):
    with path.open('rb') as f:
        raw = f.read(8); require(len(raw) == 8, 'Safetensors header length missing')
        size = struct.unpack('<Q', raw)[0]
        require(2 <= size <= 2**20, 'Safetensors header cap')
        header = json.loads(f.read(size))
    payload = path.stat().st_size - 8 - size
    tensors, ranges = [], []
    for name, value in header.items():
        if name == '__metadata__':
            continue
        shape = value['shape']; lo, hi = value['data_offsets']
        sizes = dict(F32=4, F16=2, BF16=2, I64=8)
        require(value['dtype'] in sizes and all(type(n) is int and n > 0 for n in shape), 'Unsupported tensor metadata')
        count = 1
        for n in shape:
            count *= n
        require(type(lo) is int and type(hi) is int and 0 <= lo <= hi <= payload
            and hi - lo == count * sizes[value['dtype']], 'Invalid tensor span')
        tensors.append(dict(name=name, shape=shape, dtype=value['dtype'], elements=count, offsets=[lo, hi]))
        ranges.append((lo, hi))
    ranges.sort()
    require(ranges and ranges[0][0] == 0 and ranges[-1][1] == payload
        and all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])), 'Tensor payload overlap/gap')
    return dict(tensors=tensors, encoder_parameters=sum(t['elements'] for t in tensors if t['name'].startswith('encoder.')),
        decoder_parameters=sum(t['elements'] for t in tensors if t['name'].startswith('decoder.')),
        stored_elements=sum(t['elements'] for t in tensors),
        execution=False, native_conversion=False)


def fetch(out, asset=TAESD):
    require(not out.exists(), 'Fresh weight directory required')
    temp = out.with_name(out.name + '.tmp'); require(not temp.exists(), 'Unfinished download exists')
    temp.mkdir(parents=True)
    receipts = []
    for remote, local, cap in ((asset['file'], 'model.safetensors', asset['bytes']), ('config.json', 'config.json', 2**20), ('README.md', 'README.md', 2**20)):
        url = f'https://huggingface.co/{asset["repo"]}/resolve/{asset["revision"]}/{remote}'
        path = temp / local; count = 0; digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=30) as source, path.open('xb') as target:
            while block := source.read(2**16):
                count += len(block); require(count <= cap, 'Download byte cap exceeded')
                digest.update(block); target.write(block)
        if local == 'model.safetensors':
            require(count == asset['bytes'] and digest.hexdigest() == asset['sha256'], 'Pinned checkpoint checksum differs')
        receipts.append(dict(url=url, file=local, bytes=count, sha256=digest.hexdigest()))
    value = dict(asset=asset, files=receipts, inventory=inventory(temp / 'model.safetensors'))
    (temp / 'RECEIPT.json').write_text(json.dumps(value, indent=2) + '\n')
    temp.rename(out); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'fetch-taesd', 'fetch-mobilenet', 'inspect'))
    parser.add_argument('--out', type=Path); parser.add_argument('--file', type=Path)
    args = parser.parse_args()
    if args.command == 'list':
        result = dict(taesd=TAESD, mobilenetv3_small_050=MOBILENET)
    elif args.command.startswith('fetch-'):
        require(args.out is not None, 'Need --out')
        result = fetch(args.out.resolve(), TAESD if args.command == 'fetch-taesd' else MOBILENET)
    else:
        require(args.file is not None, 'Need --file'); result = inventory(args.file.resolve())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
