#!/usr/bin/env python3
"""Host-only registration audit. Never invokes profiler --run or a neural policy."""
import argparse
import configparser
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--cost-binary", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True, help="Prepared connect4cnn/r0 directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 2048],
                        help="Host registration batches only, distinct integers 1..65536; default unchanged")
    parser.add_argument("--host-hash", action="store_true", help="Also verify the new native CPU artifact-hash mode")
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches) or any(not 1 <= b <= 65536 for b in args.batches):
        parser.error("Use distinct host registration batches from 1 through 65536")
    args.out.mkdir(parents=True, exist_ok=False)
    import csv
    import io
    families = {"flex_quality": "default", "nature_cnn": "default", "impala_cnn": "impala", "impoola_cnn": "impoola"}
    rows, rejected, receipts, hash_checks = [], [], {}, []
    for name in ("default", "impala", "impoola"):
        path = args.binaries / name
        receipts[str(path.resolve())] = sha(path)
    receipts[str(args.cost_binary.resolve())] = sha(args.cost_binary)
    for hidden in (64, 128, 256):
        costs = {row["model"]: row for row in csv.DictReader(io.StringIO(invoke([args.cost_binary, hidden, 1, 7])))}
        for model, family in families.items():
            matches = sorted(args.configs.glob(f"{model}-s*/config/default.ini"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {model} config, found {matches}")
            original = matches[0]
            receipts[str(original.resolve())] = sha(original)
            config = configparser.ConfigParser(interpolation=None)
            config.read(original)
            config["policy"]["hidden_size"] = str(hidden)
            target = args.out / f"{model}-h{hidden}.ini"
            with target.open("w") as handle:
                config.write(handle)
            for batch in args.batches:
                command = [args.binaries / family, "--describe", target, batch]
                actual = json.loads(invoke(command))
                expected = costs[model]
                checks = {
                    "model": model, "batch": batch, "hidden": hidden,
                    "encoder_parameters": int(expected["encoder_parameters"]),
                    "parameter_payload_bytes": 4 * int(expected["encoder_parameters"]),
                    "rollout_payload_bytes": batch * int(expected["rollout_encoder_tensor_bytes_per_sample"]),
                    "train_payload_bytes": batch * int(expected["train_encoder_tensor_bytes_per_sample"]),
                    "cuda_executed": False, "policy_executed": False,
                }
                for key, value in checks.items():
                    if actual.get(key) != value:
                        raise RuntimeError(f"{model}/H{hidden}/B{batch}: {key}: native={actual.get(key)}, arithmetic={value}")
                for prefix in ("rollout", "train"):
                    if actual[f"{prefix}_allocator_bytes"] < actual[f"{prefix}_payload_bytes"]:
                        raise RuntimeError("Allocator smaller than registered payload")
                rows.append({**actual, "config_sha256": sha(target), "command": [str(x) for x in command]})
    # Fixed-family/config pairing must fail before any GPU operation.
    base = args.out / "flex_quality-h128.ini"
    for label, family, key, value in (
        ("encoder5", "default", "encoder", "5"),
        ("fractional_encoder", "default", "encoder", "2.5"),
        ("fractional_hidden", "default", "hidden_size", "128.5"),
        ("zero_hidden", "default", "hidden_size", "0"),
        ("changed_quality", "default", "cnn_kernel_1", "3"),
        ("impala_wrong_pair", "impala", "encoder", "2"),
        ("impoola_wrong_pair", "impoola", "encoder", "4"),
    ):
        config = configparser.ConfigParser(interpolation=None)
        config.read(base)
        config["policy"][key] = value
        target = args.out / f"reject-{label}.ini"
        with target.open("w") as handle:
            config.write(handle)
        result = subprocess.run([str(args.binaries / family), "--describe", str(target), "64"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            raise RuntimeError(f"Invalid configuration accepted: {label}")
        rejected.append({"case": label, "returncode": result.returncode, "stderr": result.stderr})
    for batch in ("0", "65537", "16384.5"):
        result = subprocess.run([str(args.binaries / "default"), "--describe", str(base), batch],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or result.stdout or "Invalid integer" not in result.stderr:
            raise RuntimeError(f"Invalid host descriptor batch accepted: {batch}")
        rejected.append({"case": "invalid-host-batch-" + batch, "returncode": result.returncode, "stderr": result.stderr})
    if args.host_hash:
        fixtures = {"empty": b"", "hello": b"hello", "block-tail": bytes(range(256)) * 17}
        for label, data in fixtures.items():
            target = args.out / f"hash-{label}.bin"
            target.write_bytes(data)
            value = 14695981039346656037
            for byte in data:
                value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
            expected = f"{value:016x}"
            if label == "hello" and expected != "a430d84680aabd0b":
                raise RuntimeError("Independent FNV known-answer check failed")
            for family in ("default", "impala", "impoola"):
                actual = invoke([args.binaries / family, "--hash", target]).strip()
                if actual != expected:
                    raise RuntimeError(f"Native host hash differs: {family}/{label}")
                hash_checks.append(dict(family=family, fixture=label, bytes=len(data), fnv64=actual, sha256=sha(target)))
    for path, digest in receipts.items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Input changed during audit: {path}")
    report = {"schema": 1, "status": "passed", "scope": "host construction/registration arithmetic",
              "host_batches": args.batches, "large_batch_gpu_execution_qualified": False,
              "policy_executed": False, "cuda_executed": False, "gpu_runtime_validated": False,
              "rows": rows, "rejected": rejected, "host_hash_checks": hash_checks, "input_sha256": receipts,
              "audit_source_sha256": sha(Path(__file__))}
    (args.out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {len(rows)} native registration/arithmetic comparisons, {len(rejected)} rejection cases and {len(hash_checks)} host hashes; no CUDA/policy execution.")


if __name__ == "__main__":
    main()
