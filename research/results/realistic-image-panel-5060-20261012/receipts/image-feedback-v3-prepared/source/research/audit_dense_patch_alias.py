#!/usr/bin/env python3
"""Compare native host descriptors only; never query a GPU or execute a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"flex_quality": "default", "nature_cnn": "default",
            "impala_cnn": "impala", "impoola_cnn": "impoola"}
ALIAS_KEYS = {"dense_patch_alias_candidate", "dense_patch_alias_active",
              "dense_patch_alias_layers", "skipped_forward_patch_payload_bytes"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(base, candidate):
    """Shape arithmetic, not forward/backward correctness or measured traffic."""
    if ALIAS_KEYS.intersection(base) or not ALIAS_KEYS.issubset(candidate):
        raise ValueError("Expected ordinary baseline and explicitly marked candidate")
    if {k: v for k, v in candidate.items() if k not in ALIAS_KEYS} != base:
        raise ValueError("Candidate changed registration/configuration metadata")
    if base.get("cuda_executed") is not False or base.get("policy_executed") is not False:
        raise ValueError("Descriptor is not host-only")
    model, hidden, batch = base["model"], base["hidden"], base["batch"]
    if model not in FAMILIES or type(hidden) is not int or type(batch) is not int:
        raise ValueError("Malformed shape/family")
    if hidden not in (64, 128, 256) or not 1 <= batch <= 65536:
        raise ValueError("Unqualified descriptor shape")
    layers, values = 0, 0
    if model == "flex_quality":
        layers, values = (1, 1584) if hidden == 64 else (2, 1648)
    elif model == "nature_cnn":
        layers, values = 1, 128
    expected = dict(dense_patch_alias_candidate=True, dense_patch_alias_active=bool(layers),
                    dense_patch_alias_layers=layers,
                    skipped_forward_patch_payload_bytes=4 * batch * values)
    for key, value in expected.items():
        if type(candidate[key]) is not type(value) or candidate[key] != value:
            raise ValueError(f"Candidate descriptor differs: {key}")
    return expected


def manifest(path, root):
    entries = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError("Unsafe/duplicate source receipt")
        source = root / relative
        if sha(source) != digest:
            raise ValueError(f"Compiled source changed: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError("Empty source receipt")
    return entries


def main():
    import configparser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base_sources = manifest(args.baseline / "source.sha256", ROOT)
    candidate_sources = manifest(args.candidate / "source.sha256", ROOT)
    if base_sources != candidate_sources:
        raise ValueError("Paired builds must use identical source snapshots")
    receipts, rows = {}, []
    for directory in (args.baseline, args.candidate):
        receipts[str((directory / "source.sha256").resolve())] = sha(directory / "source.sha256")
        binaries = {}
        for line in (directory / "binaries.sha256").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            binaries[Path(name).name] = digest
        for family in set(FAMILIES.values()):
            path = directory / family
            if sha(path) != binaries[family]:
                raise ValueError(f"Binary changed since compilation: {path}")
            receipts[str(path.resolve())] = sha(path)
    for model, family in FAMILIES.items():
        matches = sorted(args.configs.glob(f"{model}-s*/config/default.ini"))
        if len(matches) != 1:
            raise ValueError(f"Expected one fixed {model} configuration")
        source = matches[0]
        receipts[str(source.resolve())] = sha(source)
        for hidden in (64, 128, 256):
            config = configparser.ConfigParser(interpolation=None)
            config.read(source)
            config["policy"]["hidden_size"] = str(hidden)
            target = args.out / f"{model}-h{hidden}.ini"
            with target.open("x") as stream:
                config.write(stream)
            for batch in (1, 64, 2048, 16384):
                descriptions, commands = [], []
                for directory in (args.baseline, args.candidate):
                    command = [str((directory / family).resolve()), "--describe", str(target.resolve()), str(batch)]
                    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
                    commands.append(command)
                    descriptions.append(json.loads(result.stdout))
                compare(*descriptions)
                rows.append(dict(baseline=descriptions[0], candidate=descriptions[1],
                                 commands=commands, config_sha256=sha(target)))
    for path, digest in receipts.items():
        if sha(path) != digest:
            raise ValueError(f"Input changed during audit: {path}")
    manifest(args.baseline / "source.sha256", ROOT)
    manifest(args.candidate / "source.sha256", ROOT)
    report = dict(schema=1, status="passed", scope="paired host registration/identity-copy arithmetic",
                  rows=rows, input_sha256=receipts, audit_source_sha256=sha(__file__),
                  cuda_executed=False, policy_executed=False, numerical_qualified=False,
                  performance_qualified=False, production_encoder_changed=False,
                  allocated_buffer_savings_bytes=0)
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(rows)} paired native descriptor comparisons; no CUDA/model execution.")


if __name__ == "__main__":
    main()
