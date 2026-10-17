"""Export compact stopped-run evidence; keep raw checkpoints/arrays on the training host."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(campaign, output):
    campaign, output = campaign.resolve(), output.resolve()
    if not (campaign / 'result.json').is_file():
        raise ValueError('Export after the GPU run stops; no concurrent archive work')
    if output.exists():
        raise ValueError('Use a new archive path')
    plan = json.loads((campaign / 'plan.json').read_text())
    build = Path(plan['build'])
    suffixes = {'.json', '.ini', '.csv', '.tsv', '.txt', '.log', '.md', '.html', '.sha256', '.patch'}
    files = {f'campaign/{p.relative_to(campaign)}': p for p in campaign.rglob('*')
        if p.is_file() and p.suffix in suffixes}
    for p in (build / 'source').rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in {'.c', '.cu', '.h', '.py', '.sh'}:
            files['build-source/' + str(p.relative_to(build / 'source'))] = p
    for path, digest in plan['inputs'].items():
        p = Path(path)
        if p.suffix in {'.py', '.ini', '.sha256'}:
            captured = p
            if sha(p) != digest:
                captured = campaign / 'reader-at-completion.py'
                if p.name != 'sprite_overnight.py' or not captured.is_file() or sha(captured) != digest:
                    raise ValueError(f'Frozen input changed and no matching source receipt exists: {p}')
            files['inputs/' + p.name] = captured
    for name in ('revision.txt', 'worktree.txt', 'compiler.txt', 'toolchain.txt'):
        p = campaign.parent / name
        if p.is_file():
            files['host/' + name] = p
    manifest = dict(schema='sprite-overnight-compact-evidence-v1',
        raw_checkpoints_included=False, raw_numerical_arrays_included=False,
        raw_artifacts_retained_on_training_host=str(campaign),
        files={name: dict(sha256=sha(p), bytes=p.stat().st_size) for name, p in sorted(files.items())})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    temporary.rename(output)
    output.with_name(output.name + '.sha256').write_text(f'{sha(output)}  {output.name}\n')
    print(f'Compact evidence: {output} ({output.stat().st_size:,} bytes). Raw weights remain on host.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.campaign, args.out)
