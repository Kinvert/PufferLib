#!/usr/bin/env python3
"""Exclusive scheduled native GEMM acceptance; external audit glue, never trains.

prepare/inspect/audit are GPU-free. run requires a separately scheduled GPU
window; both current execution holds remain. There is no CPU arithmetic fallback.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import prepare_workspace_acceptance as native
import claim

ROOT = native.ROOT
PROTOCOL = "shared-gemm-supervised-v1"
LOCK = ROOT / "build/connect4cnn/hardware-benchmark.lock"
TOOLS = ("research/workspace_supervisor.py", "research/prepare_workspace_acceptance.py", "ocean/connect4cnn/claim.py")
ENVIRONMENT = ("CUDA_HOME", "NCCL_ROOT", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
               "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")


def verify_files(directory, hashes):
    for name, digest in hashes.items():
        claim.require(not Path(name).is_absolute() and ".." not in Path(name).parts
                      and claim.sha(directory / name) == digest, "Changed/unsafe retained file: " + name)


def dependencies(directory):
    entries = {}
    for line in (directory / "dependencies.sha256").read_text().splitlines():
        digest, name = line.split(None, 1); name = name.strip()
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        claim.require(name not in entries and claim.sha(path) == digest, "Native dependency changed; rebuild/reprepare")
        entries[name] = digest
    claim.require(bool(entries), "Empty dependency receipt")
    return entries


def prepare(args):
    source, out = args.packet.resolve(), args.out.resolve()
    packet = native.load_packet(source); original_hash = claim.sha(source)
    tool_hashes = {name: claim.sha(ROOT / name) for name in TOOLS}
    dependencies(source.parent)
    out.mkdir(parents=True, exist_ok=False)
    claim.save(out / "request.json", dict(status="preparing", source_packet=str(source),
        source_packet_sha256=original_hash, gpu_execution_authorized=False), exclusive=True)
    try:
        # Refuse unexpected additions rather than copying unrecorded execution data.
        expected = dict(packet["files_sha256"], **{"packet.json": original_hash})
        claim.require(native.files(source.parent) == expected, "Unexpected files in native preparation")
        shutil.copytree(source.parent, out / "inputs")
        for name, digest in tool_hashes.items():
            target = out / "tooling" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            claim.require(claim.sha(target) == digest, "Tool changed during snapshot")
        native.load_packet(out / "inputs/packet.json")
        claim.require(claim.sha(source) == original_hash, "Source packet changed during preparation")
        verify_files(source.parent, expected)
        dependencies(out / "inputs")
        claim.require(all(claim.sha(ROOT / name) == digest for name, digest in tool_hashes.items()), "Tool changed")
        record = dict(protocol=PROTOCOL, status="prepared-not-executed", cases=packet["cases"],
            native_packet_sha256=original_hash, tool_sha256=tool_hashes, files_sha256=native.files(out),
            runtime_supervisor_implemented=True, gpu_execution_authorized=False,
            backend_math_certified=False, encoder_math_certified=False, frontier_superiority_certified=False)
        claim.save(out / "plan.json", record, exclusive=True)
    except BaseException as error:
        claim.save(out / "FAILURE.json", dict(status="failed", error=f"{type(error).__name__}: {error}"), exclusive=True)
        raise
    print(out / "plan.json")


def load(path, current_tools=False):
    plan = json.loads(path.read_text())
    claim.require(plan["protocol"] == PROTOCOL and plan["status"] == "prepared-not-executed"
        and plan["cases"] == native.panel() and plan["runtime_supervisor_implemented"] is True, "Wrong supervised plan")
    claim.require(all(plan[key] is False for key in ("gpu_execution_authorized", "backend_math_certified",
        "encoder_math_certified", "frontier_superiority_certified")), "Unsupported plan gates")
    verify_files(path.parent, plan["files_sha256"])
    packet = native.load_packet(path.parent / "inputs/packet.json")
    claim.require(claim.sha(path.parent / "inputs/packet.json") == plan["native_packet_sha256"]
                  and plan["cases"] == packet["cases"] and set(plan["tool_sha256"]) == set(TOOLS), "Wrong native packet/tools")
    for name, digest in plan["tool_sha256"].items():
        claim.require(claim.sha(path.parent / "tooling" / name) == digest, "Frozen supervisor tool changed")
        if current_tools: claim.require(claim.sha(ROOT / name) == digest, "Current supervisor differs; reprepare")
    return plan, packet


def gpu(smi):
    hardware = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                                      text=True, timeout=20)
    claim.require(len(hardware.strip().splitlines()) == 1 and ("5060" in hardware or "5090" in hardware),
                  "Unexpected/empty GPU topology")
    busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    claim.require(not busy.strip(), "GPU busy; no processes interrupted")
    return hardware


def audit_process(path, case, directory, allow_relocation=True):
    result = json.loads((directory / "native.log.json").read_text())
    command = [str(path.parent / "inputs/native"), "--run", str(case["fixture"]), case["workspace"],
               case["schedule"], case["mode"], str(directory / "native")]
    # Historical absolute paths are retained on archive relocation.
    original = Path(result["cwd"])
    claim.require(original.is_absolute() and original.name == case["name"], "Wrong recorded case cwd")
    claim.require(allow_relocation or original == directory, "Native process cwd changed during execution")
    if original != directory:
        command[0] = result["command"][0]
        command[-1] = str(original / "native")
        claim.require(Path(command[0]).is_absolute() and Path(command[0]).name == "native"
                      and Path(command[0]).parent.name == "inputs", "Wrong archived binary path")
    claim.require(result["command"] == command and result["status"] == "ok" and result["returncode"] == 0,
                  "Failed/wrong native process; keep partial receipts")
    start, end = result["launch_monotonic_ns"], result["end_monotonic_ns"]
    claim.require(type(start) is int and type(end) is int and end > start
                  and type(result["pid"]) is int and result["pid"] > 0
                  and type(result["timeout"]) is int and 1 <= result["timeout"] <= 600, "Bad native process receipt")
    seconds = (end-start)/1e9; claim.require(math.isfinite(seconds), "Nonfinite process time")
    return seconds


def audit_execution(path, output):
    plan, packet = load(path); records = {}
    versions = set(); groups = {}
    for case in plan["cases"]:
        directory = output / case["name"]
        record = native.audit_case(packet, case["name"], directory / "native")
        record["process_seconds"] = audit_process(path, case, directory)
        claim.require(json.loads((directory / "case-audit.json").read_text()) == record,
                      "Completed case changed; keep the original audit")
        record["files_sha256"] = native.files(directory)
        receipt = json.loads((directory / "native/native.json").read_text())
        versions.add(tuple(receipt[key] for key in ("runtime", "driver", "cublas")))
        hashes = tuple(claim.sha(directory / "native" / name) for name in ("input.f32", "reference.f32", "actual.f32"))
        groups.setdefault(case["fixture"], []).append(hashes)
        records[case["name"]] = record
    claim.require(len(versions) == 1, "Runtime/library identity changed across cases")
    claim.require(set(groups) == set(range(6)) and all(len(group) == 16 and len(set(group)) == 1 for group in groups.values()),
                  "Process/schedule/mode/workspace bytes differ; retain every case")
    return dict(status="full-panel-receipt-consistency-passed", cases=96, records=records,
        paired_fixture_sha256={str(key): list(values[0]) for key, values in groups.items()}, runtime_versions=list(next(iter(versions))),
        backend_math_certified=False, encoder_math_certified=False, frontier_superiority_certified=False,
        note="Scoped native/execution/export consistency. Synthetic fixtures can pass; this is not independent GPU authentication, "
             "general FP32 accuracy, measured overlap, full-model acceptance, speedup or learning evidence.")


def run(args):
    path, output = args.plan.resolve(), args.out.resolve()
    claim.require(type(args.timeout) is int and 1 <= args.timeout <= 600, "Invalid per-case deadline")
    plan, packet = load(path, current_tools=True); plan_hash = claim.sha(path)
    output.mkdir(parents=True, exist_ok=False)
    failure = dict(status="starting", plan_sha256=plan_hash, completed_cases=[], backend_math_certified=False,
                   encoder_math_certified=False, frontier_superiority_certified=False)
    claim.save(output / "request.json", dict(**failure, timeout=args.timeout), exclusive=True)
    retained = {}
    try:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dependencies(path.parent / "inputs")
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            hardware = gpu(smi); (output / "gpu.txt").write_text(hardware)
            claim.save(output / "runtime-environment.json", {key: os.environ.get(key) for key in ENVIRONMENT}, exclusive=True)
            for case in plan["cases"]:
                failure["case"] = case["name"]
                load(path, current_tools=True)
                claim.require(claim.sha(path) == plan_hash, "Plan changed during execution")
                claim.require(gpu(smi) == hardware, "Hardware changed during panel")
                directory = output / case["name"]; directory.mkdir()
                command = [str(path.parent / "inputs/native"), "--run", str(case["fixture"]), case["workspace"],
                           case["schedule"], case["mode"], str(directory / "native")]
                # Native test itself creates its fresh exports directory.
                claim.process(command, directory, directory / "native.log", args.timeout)
                load(path, current_tools=True)
                claim.require(claim.sha(path) == plan_hash, "Plan changed during execution")
                record = native.audit_case(packet, case["name"], directory / "native")
                record["process_seconds"] = audit_process(path, case, directory, allow_relocation=False)
                claim.save(directory / "case-audit.json", record, exclusive=True)
                retained[case["name"]] = native.files(directory)
                failure["completed_cases"].append(case["name"])
            claim.require(gpu(smi) == hardware, "Hardware changed or GPU busy at completion")
            dependencies(path.parent / "inputs")
            load(path, current_tools=True)
            claim.require(claim.sha(path) == plan_hash, "Plan changed during execution")
            for name, hashes in retained.items(): verify_files(output / name, hashes)
            summary = audit_execution(path, output)
            summary.update(plan_sha256=plan_hash, files_sha256=native.files(output))
            claim.save(output / "REPORT.json", summary, exclusive=True)
    except BaseException as error:
        failure.update(status="failed", error=f"{type(error).__name__}: {error}")
        claim.save(output / "FAILURE.json", failure, exclusive=True)
        raise
    print(output / "REPORT.json")


def audit(args):
    path, output = args.plan.resolve(), args.execution.resolve()
    report = json.loads((output / "REPORT.json").read_text())
    claim.require(report["plan_sha256"] == claim.sha(path), "Wrong report/plan")
    verify_files(output, report["files_sha256"])
    expected = audit_execution(path, output)
    claim.require(all(report[key] == value for key, value in expected.items()), "Changed aggregate/receipts")
    print(json.dumps(dict(status="offline-panel-consistency-passed", cases=expected["cases"],
                         backend_math_certified=False, encoder_math_certified=False, frontier_superiority_certified=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare"); create.add_argument("--packet", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("inspect"); check.add_argument("--plan", type=Path, required=True)
    execute = sub.add_parser("run", help="Explicitly scheduled GPU window only; never trains")
    execute.add_argument("--plan", type=Path, required=True); execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--timeout", type=int, default=120)
    review = sub.add_parser("audit"); review.add_argument("--plan", type=Path, required=True)
    review.add_argument("--execution", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run(args)
    elif args.command == "audit": audit(args)
    else:
        plan, _ = load(args.plan.resolve(), current_tools=True)
        print(json.dumps(dict(status="supervised-host-preparation-validated", cases=len(plan["cases"]),
                             gpu_execution_authorized=False, backend_math_certified=False), indent=2))


if __name__ == "__main__": main()
