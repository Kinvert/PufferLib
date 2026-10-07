#!/usr/bin/env python3
"""GPU-free shared-GEMM acceptance preparation and exported-array auditing.

There is deliberately no launch command until the exclusive runtime supervisor
is implemented and explicitly scheduled. No CPU GEMM or policy is provided.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from claim import require, save, sha

PROTOCOL = "shared-gemm-workspace-v1"
FIXTURES = (("odd", 3, 7, 5), ("projection", 64, 128, 128), ("quality-conv", 6336, 49, 16),
            ("nature-conv2", 768, 512, 64), ("impala-residual", 25344, 144, 16),
            ("learner-core", 2048, 128, 128))
LANES, ITERATIONS = 2, 3


def fixtures():
    return [dict(id=i, name=name, m=m, k=k, n=n,
                 input_bytes=LANES*4*(m*k+n*k+m*n), output_bytes=LANES*4*(m*n+n*k+m*k))
            for i, (name, m, k, n) in enumerate(FIXTURES)]


def metadata(binary):
    value = json.loads(subprocess.check_output([str(binary), "--describe"], text=True, timeout=30))
    require(value == dict(protocol=PROTOCOL, lanes=LANES, iterations=ITERATIONS,
                         precision="float32", oracle="gpu-scalar-double-exact-dyadic-v1", fixtures=fixtures()),
            "Wrong native metadata; source/fixture review required")
    # Input integer numerators are bounded by four. Every product numerator is
    # <=16; no partial sum can reach the float32 exact-integer limit 2^24.
    require(all(16*max(m, k, n) < 2**24 for _, m, k, n in FIXTURES), "Dyadic exactness bound exceeded")
    return value


def panel():
    cases = []
    for repetition in range(2):
        for fixture in fixtures():
            for workspace in ("default", "reassign"):
                for schedule in ("serial", "overlap"):
                    for mode in ("eager", "graph"):
                        cases.append(dict(name=f"f{fixture['id']}-{workspace}-{schedule}-{mode}-r{repetition}",
                            fixture=fixture["id"], workspace=workspace, schedule=schedule, mode=mode,
                            repetition=repetition))
    return cases


def files(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def prepare(args):
    binary, out = args.binary.resolve(), args.out.resolve()
    require(binary.is_file(), "Need compiled native binary")
    original_hash = sha(binary); description = metadata(binary)
    require(sha(binary) == original_hash, "Native binary changed")
    build = binary.parent
    required = ("source.sha256", "compiler.txt", "command.txt", "binary.sha256", "revision.txt", "worktree.txt",
                "dependencies.sha256")
    require(all((build / name).is_file() for name in required), "Need native build provenance")
    binary_receipt = (build / "binary.sha256").read_text().splitlines()
    require(len(binary_receipt) == 1 and binary_receipt[0].split(None, 1)[0] == original_hash,
            "Binary/build receipt differs")
    dependencies = {}
    for line in (build / "dependencies.sha256").read_text().splitlines():
        digest, name = line.split(None, 1); name = name.strip()
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and name not in dependencies, "Bad dependency receipt")
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(sha(path) == digest, "Build dependency changed; rebuild first")
        dependencies[name] = digest
    out.mkdir(parents=True, exist_ok=False)
    save(out / "request.json", dict(status="preparing", binary=str(binary), binary_sha256=original_hash), exclusive=True)
    try:
        shutil.copyfile(binary, out / "native"); (out / "native").chmod(0o755)
        for name in required:
            shutil.copyfile(build / name, out / name)
        source_hashes = {}
        for line in (out / "source.sha256").read_text().splitlines():
            digest, name = line.split(None, 1); name = name.strip()
            require(re.fullmatch(r"[0-9a-f]{64}", digest) and not Path(name).is_absolute()
                    and ".." not in Path(name).parts and name not in source_hashes, "Unsafe/duplicate source path")
            source = ROOT / name; require(sha(source) == digest, "Compiled source differs; rebuild first")
            target = out / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target); require(sha(target) == digest, "Source changed while copying")
            source_hashes[name] = digest
        require("src/algo.cu" in source_hashes and "ocean/connect4cnn/tests/workspace_acceptance.cu" in source_hashes,
                "Incomplete compiled source manifest")
        for path in (Path(__file__), ROOT / "ocean/connect4cnn/claim.py"):
            target = out / "tooling" / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target); require(sha(path) == sha(target), "Preparation tool changed")
        require(sha(binary) == sha(out / "native") == original_hash, "Binary changed during snapshot")
        for name, digest in source_hashes.items(): require(sha(ROOT / name) == digest, "Source changed during preparation")
        for name, digest in dependencies.items():
            require(sha(Path(name) if Path(name).is_absolute() else ROOT / name) == digest,
                    "Dependency changed during preparation")
        record = dict(protocol=PROTOCOL, status="prepared-not-executed", description=description,
                      cases=panel(), binary_path=str(binary), binary_sha256=original_hash,
                      source_sha256=source_hashes, files_sha256=files(out),
                      gpu_execution_authorized=False, runtime_supervisor_implemented=False,
                      backend_math_certified=False, encoder_math_certified=False,
                      frontier_superiority_certified=False)
        save(out / "packet.json", record, exclusive=True)
    except BaseException as error:
        save(out / "FAILURE.json", dict(status="failed", error=f"{type(error).__name__}: {error}"), exclusive=True)
        raise
    print(out / "packet.json")


def load_packet(path):
    value = json.loads(path.read_text())
    require(value["protocol"] == PROTOCOL and value["status"] == "prepared-not-executed"
            and value["cases"] == panel(), "Wrong protocol/panel")
    require(all(value[key] is False for key in ("gpu_execution_authorized", "runtime_supervisor_implemented",
                "backend_math_certified", "encoder_math_certified", "frontier_superiority_certified")), "Wrong gates")
    for name, digest in value["files_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and sha(path.parent / name) == digest, "Changed/unsafe frozen file")
    require(sha(path.parent / "native") == value["binary_sha256"], "Frozen binary differs")
    require((path.parent / "binary.sha256").read_text().split(None, 1)[0] == value["binary_sha256"],
            "Binary/build receipt differs")
    require(value["description"] == dict(protocol=PROTOCOL, lanes=LANES, iterations=ITERATIONS,
        precision="float32", oracle="gpu-scalar-double-exact-dyadic-v1", fixtures=fixtures()), "Changed description")
    for name, digest in value["source_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and sha(path.parent / "source" / name) == digest, "Changed source snapshot")
    return value


def audit_case(packet, name, directory):
    selected = [case for case in packet["cases"] if case["name"] == name]
    require(len(selected) == 1, "Unknown case")
    case = selected[0]; fixture = fixtures()[case["fixture"]]
    receipt = json.loads((directory / "native.json").read_text())
    require(receipt["protocol"] == PROTOCOL and receipt["status"] == "scoped-gpu-checks-passed", "Failed native check")
    for key in ("fixture", "workspace", "schedule", "mode"):
        require(receipt[key] == case[key], "Wrong native case")
    require(receipt["lanes"] == LANES and receipt["iterations"] == ITERATIONS, "Wrong native quota")
    submissions = 2 if case["mode"] == "graph" else ITERATIONS+1
    require(receipt["main_api_calls"] == LANES*2*submissions
            and receipt["dw_api_calls"] == LANES*submissions
            and receipt["gemm_api_calls"] == LANES*3*submissions
            and receipt["workspace_api_calls"] == (receipt["gemm_api_calls"] if case["workspace"] == "reassign" else 0),
            "Wrong handle/GEMM/workspace coverage")
    require(receipt["measured_overlap"] is False and receipt["encoder_math_certified"] is False
            and receipt["frontier_superiority_certified"] is False, "Unsupported native claims")
    for key in ("runtime", "driver", "cublas"):
        require(type(receipt[key]) is int and receipt[key] > 0, "Missing runtime/library identity")
    arrays = {}
    for filename, key in (("input.f32", "input_bytes"), ("reference.f32", "output_bytes"), ("actual.f32", "output_bytes")):
        path = directory / filename
        require(receipt[key] == fixture[key] and path.stat().st_size == fixture[key], "Wrong exported byte quota")
        arrays[filename] = np.fromfile(path, dtype=np.float32)
        require(np.isfinite(arrays[filename]).all(), "Nonfinite export")
    inputs = arrays["input.f32"]
    require((np.abs(inputs) <= .5).all() and np.equal(inputs*8, np.floor(inputs*8)).all(), "Input exactness contract changed")
    require(np.array_equal(arrays["actual.f32"], arrays["reference.f32"]), "GPU export/oracle differs")
    return dict(case=name, status="export-consistency-passed", files_sha256=files(directory),
                backend_math_certified=False, encoder_math_certified=False, frontier_superiority_certified=False,
                note="Export audit alone cannot prove GPU execution or authenticate the native oracle.")


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare"); create.add_argument("--binary", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("inspect"); check.add_argument("--packet", type=Path, required=True)
    audit = sub.add_parser("audit-case"); audit.add_argument("--packet", type=Path, required=True)
    audit.add_argument("--case", required=True); audit.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "inspect":
        value = load_packet(args.packet.resolve())
        print(json.dumps(dict(status="host-preparation-validated", cases=len(value["cases"]),
            gpu_execution_authorized=False, backend_math_certified=False), indent=2))
    else: print(json.dumps(audit_case(load_packet(args.packet.resolve()), args.case, args.receipts.resolve()), indent=2))


if __name__ == "__main__": main()
