#!/usr/bin/env python3
"""Cross-check native shape arithmetic against retained parameter receipts.

No policy is constructed or executed. Counts aren't throughput benchmarks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=ROOT / "research/results/connect4cnn/rtx-5060/hardware-5060.Mz7sDeEe/full/results.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    output = subprocess.check_output([str(args.binary.resolve())], text=True, timeout=30)
    costs = list(csv.DictReader(output.splitlines()))
    expected = {row["model"]: int(row["total_parameters"]) for row in costs}
    if len(expected) != 4: raise ValueError("Need four unique fixed graphs")
    seen = set(); checked = 0
    with args.receipts.open() as stream:
        for row in csv.DictReader(stream):
            if row["variant"] not in expected: continue
            if int(row["params"]) != expected[row["variant"]]:
                raise ValueError("Independent retained parameter count differs")
            seen.add(row["variant"]); checked += 1
    if seen != set(expected): raise ValueError("Missing baseline receipts")
    anchored_sources = {}
    for name in ("ocean/connect4cnn/flex.cu", "ocean/connect4cnn/nature.cu",
                 "ocean/connect4cnn/impala.cu", "src/algo.cu"):
        measured = args.receipts.parent / "source" / name
        if not measured.is_file() or digest(measured) != digest(ROOT / name):
            raise ValueError(f"Retained graph/core source differs: {name}; renew the source/count audit")
        anchored_sources[name] = digest(measured)
    # A separate task's retained executed checkpoint count anchors action-head arithmetic.
    flappy = subprocess.check_output([str(args.binary.resolve()), "128", "1", "2"], text=True, timeout=30)
    flappy_rows = list(csv.DictReader(flappy.splitlines()))
    report = ROOT / "research/results/flappycnn/stock-pilot.DN7C2JF6/REPORT.md"
    if "exactly 160,096 float32 parameters" not in report.read_text():
        raise ValueError("Missing retained Flappy parameter anchor")
    if int(flappy_rows[0]["total_parameters"]) != 160096:
        raise ValueError("Flappy quality head count differs")
    # Exercise the real constructor's projection-equals-core branch, without a network.
    equal = subprocess.check_output([str(args.binary.resolve()), "64", "2", "7"], text=True, timeout=30)
    equal_rows = list(csv.DictReader(equal.splitlines()))
    if int(equal_rows[0]["dense_layers"]) != 1 or int(equal_rows[0]["encoder_parameters"]) != 102240:
        raise ValueError("Projection/core equality count differs")
    for invalid in (("0", "1", "7"), ("129", "1", "7"), ("128", "0", "7"),
                    ("128", "1", "0"), ("128x", "1", "7"), ("128",), ("nan", "1", "7")):
        result = subprocess.run([str(args.binary.resolve()), *invalid], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or result.stdout: raise ValueError("Invalid count input accepted")
    (args.out / "connect4.csv").write_text(output)
    (args.out / "flappy.csv").write_text(flappy)
    (args.out / "projection-equals-core.csv").write_text(equal)
    data = dict(status="passed-shape-arithmetic-only", retained_parameter_rows=checked,
                matched_models=sorted(seen), flappy_quality_parameters=160096,
                policy_executed=False, gpu_profiled=False, backend_efficiency_certified=False,
                binary_sha256=digest(args.binary), receipts=str(args.receipts), receipts_sha256=digest(args.receipts),
                flappy_anchor=str(report), flappy_anchor_sha256=digest(report),
                count_source_sha256=digest(ROOT / "research/encoder_costs.c"),
                parameter_anchor_native_source_sha256=anchored_sources,
                native_source_sha256={name: digest(ROOT / name) for name in
                    ("ocean/connect4cnn/flex.cu", "ocean/connect4cnn/nature.cu",
                     "ocean/connect4cnn/impala.cu", "src/algo.cu", "src/pufferl.cu")},
                limitation="Independent fixed-graph arithmetic, not production introspection or a general constructor. "
                           "Parameter anchors do not independently certify MAC/tensor counts or backend speed.")
    (args.out / "REPORT.json").write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__": main()
