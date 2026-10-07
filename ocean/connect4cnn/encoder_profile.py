"""Encoder preparation/receipt supervision; all encoder execution is native CUDA.

prepare is GPU-free. run is only for an explicitly scheduled exclusive GPU window.
This is external experiment glue, not a native architecture-search deliverable.
"""
import argparse
import configparser
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import claim

ROOT = claim.ROOT
MODELS = dict(flex_quality="default", nature_cnn="default", impala_cnn="impala", impoola_cnn="impoola")
PARAMS = dict(flex_quality=110560, nature_cnn=88352, impala_cnn=220320, impoola_cnn=101536)
FORWARD_CALLS = dict(flex_quality=3, nature_cnn=4, impala_cnn=16, impoola_cnn=16)
PHASES = dict(rollout=64, learner=2048)
FILES = ("input.f32", "upstream.f32", "parameters.f32", "output-grad.f32")
LOCK_PATH = ROOT / "build/connect4cnn/hardware-benchmark.lock"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_files(root, entries):
    for name, digest in entries.items():
        require(sha(root / name) == digest, f"Immutable input changed: {name}")


def describe(binary, config, batch):
    result = subprocess.run([str(binary), "--describe", str(config), str(batch)],
                            capture_output=True, text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def case_panel(repetitions):
    names, cases = list(MODELS), []
    for repetition in range(repetitions):
        order = names[repetition % 4:] + names[:repetition % 4]
        for model in order:
            for phase, batch in PHASES.items():
                for mode in ("eager", "graph"):
                    for workspace in ("default", "reassign"):
                        cases.append(dict(id=f"r{repetition}-{model}-{phase}-{mode}-{workspace}",
                                          repetition=repetition, model=model, family=MODELS[model],
                                          phase=phase, batch=batch, mode=mode, workspace=workspace))
    return cases


def prepare(args):
    require(5 <= args.warmup <= 1000 and 5 <= args.samples <= 10000, "Invalid warmup/sample count")
    require(2 <= args.repetitions <= 4, "Use two to four independent process repetitions")
    out, binaries, configs = args.out.resolve(), args.binaries.resolve(), args.configs.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Verify the actual build closure, then freeze it and binaries/configs locally.
    originals = {}
    lines = (binaries / "source.sha256").read_text().splitlines()
    for line in lines:
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid build source path")
        source = ROOT / name
        require(sha(source) == digest, f"Compiled source differs: {name}")
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        originals[str(source)] = digest
    binary_receipts = {}
    for line in (binaries / "binaries.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        binary_receipts[Path(name).name] = digest
    (out / "binaries").mkdir()
    for family in set(MODELS.values()):
        source = binaries / family
        require(sha(source) == binary_receipts[family], f"Binary changed since build: {family}")
        shutil.copy2(source, out / "binaries" / family)
        originals[str(source)] = sha(source)
    for name in ("compiler.txt", "revision.txt", "worktree.txt", "source.sha256", "binaries.sha256"):
        shutil.copyfile(binaries / name, out / name)
    (out / "configs").mkdir()
    metadata = {}
    for model, family in MODELS.items():
        matches = sorted(configs.glob(f"{model}-s*/config/default.ini"))
        require(len(matches) == 1, f"Expected exactly one {model} config")
        source = matches[0]
        target = out / "configs" / f"{model}.ini"
        shutil.copyfile(source, target)
        originals[str(source)] = sha(source)
        config = configparser.ConfigParser(interpolation=None)
        config.read(target)
        require(config.getint("policy", "hidden_size") == 128, "Panel requires H128")
        require(config.getint("vec", "total_agents") == 64 and config.getint("vec", "num_buffers") == 1,
                "Rollout recipe does not match B64")
        require(config.getint("train", "minibatch_size") == 2048, "Learner recipe does not match B2048")
        for phase, batch in PHASES.items():
            info = describe(out / "binaries" / family, target, batch)
            require(info["model"] == model and info["batch"] == batch and info["hidden"] == 128
                    and info["encoder_parameters"] == PARAMS[model]
                    and info.get("receipt_exports") is True and info.get("host_hash") is True
                    and info["cuda_executed"] is False and info["policy_executed"] is False,
                    "Native metadata/family mismatch")
            require(info["parameter_payload_bytes"] == 4 * PARAMS[model], "Unexpected parameter payload")
            metadata[f"{model}/{phase}"] = info
    tooling = [Path(__file__), ROOT / "ocean/connect4cnn/claim.py"]
    for source in tooling:
        target = out / "tooling" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        originals[str(source)] = sha(source)
    verify_files(Path("/"), originals)
    cases = case_panel(args.repetitions)
    immutable = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    protocol = dict(schema=1, status="prepared_not_executed", seed=173, hidden=128,
                    warmup=args.warmup, samples=args.samples, repetitions=args.repetitions,
                    cases=cases, metadata=metadata, immutable_sha256=immutable,
                    original_sha256=originals, runtime_qualified=False,
                    supervisor_sha256=sha(Path(__file__)), process_helper_sha256=sha(Path(claim.__file__)),
                    note="Registration-only preparation. GPU execution requires explicit scheduling; neither hold is lifted.")
    save(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    return protocol


def load(out):
    require(sha(out / "protocol.json") == (out / "protocol.sha256").read_text().strip(), "Protocol changed")
    protocol = json.loads((out / "protocol.json").read_text())
    require(protocol["schema"] == 1 and protocol["runtime_qualified"] is False, "Invalid protocol")
    require(2 <= protocol["repetitions"] <= 4 and protocol["cases"] == case_panel(protocol["repetitions"]),
            "Missing/changed predeclared panel")
    verify_files(out, protocol["immutable_sha256"])
    # Supervision code must match its preparation receipt; binaries run frozen snapshots.
    require(sha(Path(__file__)) == protocol["supervisor_sha256"]
            and sha(Path(claim.__file__)) == protocol["process_helper_sha256"], "Supervisor source changed; prepare again")
    return protocol


def file_fnv(binary, path):
    result = subprocess.run([str(binary), "--hash", str(path)], capture_output=True, text=True, timeout=30, check=True)
    value = result.stdout.strip()
    require(re.fullmatch("[0-9a-f]{16}", value), "Malformed native host hash")
    return value


def audit_result(result, case, protocol, native, hash_binary):
    require(type(result) is dict, "Expected one native JSON object")
    expected = dict(schema=1, model=case["model"], batch=case["batch"], hidden=128,
                    phase=case["phase"], mode=case["mode"], workspace=case["workspace"], seed=173,
                    warmup=protocol["warmup"], samples=protocol["samples"], encoder_parameters=PARAMS[case["model"]],
                    parameters_unchanged=True, eager_mode_repeat_bytes_equal=True,
                    receipts_written=True, runtime_qualified=False)
    metadata = protocol["metadata"][f"{case['model']}/{case['phase']}"]
    alias_keys = ("dense_patch_alias_candidate", "dense_patch_alias_active",
                  "dense_patch_alias_layers", "skipped_forward_patch_payload_bytes")
    for key in alias_keys:
        require((key in result) == (key in metadata), f"Candidate receipt presence differs: {key}")
        if key in metadata:
            expected[key] = metadata[key]
    for key, value in expected.items():
        require(type(result.get(key)) is type(value) and result[key] == value, f"Native receipt mismatch: {key}")
    values = result.get("samples_ms")
    require(type(values) is list and len(values) == protocol["samples"], "Missing duration samples")
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "Invalid CUDA duration")
    mean = result.get("mean_ms")
    require(type(mean) in (int, float) and math.isfinite(mean) and mean > 0
            and math.isclose(mean, sum(values) / len(values), rel_tol=2e-8, abs_tol=1e-12), "Duration mean differs")
    for key in ("input_fnv64", "upstream_fnv64", "parameter_fnv64", "output_grad_fnv64"):
        require(type(result.get(key)) is str and re.fullmatch("[0-9a-f]{16}", result[key]), "Malformed fingerprint")
    calls = FORWARD_CALLS[case["model"]]
    if case["phase"] == "learner":
        calls = 3 * calls - 1  # First conv skips raw-pixel input gradients.
    timed = calls * protocol["samples"] if case["mode"] == "eager" else 0
    for key in ("timed_host_gemm_calls", "stream_api_calls", "workspace_api_calls", "dw_stream_api_calls"):
        require(type(result.get(key)) is int and result[key] >= 0, "Invalid host API counter")
    total = calls * (1 + protocol["warmup"] + protocol["samples"]) if case["mode"] == "eager" else 2 * calls
    require(result["timed_host_gemm_calls"] == timed and result["stream_api_calls"] == total,
            "Host API counts differ from fixed graph")
    require(result["dw_stream_api_calls"] == 0, "Unexpected encoder side-stream; qualification remains pending")
    workspace = total if case["workspace"] == "reassign" else 0
    require(result["workspace_api_calls"] == workspace, "Workspace API assignment count differs")
    batch, count = case["batch"], PARAMS[case["model"]]
    sizes = {"input.f32": batch * 1584 * 4, "upstream.f32": batch * 128 * 4,
             "parameters.f32": count * 4,
             "output-grad.f32": (batch * 128 + (count if case["phase"] == "learner" else 0)) * 4}
    import numpy as np
    receipts = {}
    for name, size in sizes.items():
        path = native / name
        require(path.stat().st_size == size, f"Receipt byte count differs: {name}")
        # Inspect retained byte arrays only, never execute a CPU network.
        require(np.isfinite(np.fromfile(path, dtype="<f4")).all(), f"Nonfinite retained bytes: {name}")
        receipts[name] = dict(bytes=size, sha256=sha(path))
    for name, key in zip(FILES, ("input_fnv64", "upstream_fnv64", "parameter_fnv64", "output_grad_fnv64")):
        require(file_fnv(hash_binary, native / name) == result[key], f"Native fingerprint/file mismatch: {name}")
    return dict(native=result, files=receipts)


def audit_panel(protocol, records):
    require(set(records) == {case["id"] for case in protocol["cases"]}, "Panel has missing/extra cases")
    groups, parameters, inputs = {}, {}, {}
    for case in protocol["cases"]:
        receipt = records[case["id"]]["files"]
        key = (case["model"], case["phase"])
        hashes = {name: value["sha256"] for name, value in receipt.items()}
        require(key not in groups or groups[key] == hashes, "Cross-process/mode/workspace byte mismatch")
        groups[key] = hashes
        require(case["model"] not in parameters or parameters[case["model"]] == hashes["parameters.f32"],
                "Parameters changed across phases")
        parameters[case["model"]] = hashes["parameters.f32"]
        shared = (hashes["input.f32"], hashes["upstream.f32"])
        require(case["phase"] not in inputs or inputs[case["phase"]] == shared, "Families received different inputs")
        inputs[case["phase"]] = shared
    return dict(status="complete_profile_panel", cases=len(records), paired_groups=len(groups),
                cross_process_mode_workspace_sha256_equal=True, runtime_qualified=False,
                note="Synthetic encoder diagnostics only. No math-oracle/concurrent-trainer/learning/general-dominance certification.")


def idle_gpu(smi):
    hardware = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                                       text=True, timeout=20)
    require(len(hardware.strip().splitlines()) == 1 and ("5060" in hardware or "5090" in hardware), "Unexpected GPU topology")
    busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    require(not busy.strip(), "GPU busy; no process was interrupted")
    return hardware


def run(args):
    out = args.out.resolve()
    protocol = load(out)
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 600, "Invalid per-case timeout")
    execution = out / "execution"
    execution.mkdir(exist_ok=False)  # Never restart/retry/overwrite a partial panel automatically.
    records = {}
    failure = dict(status="starting", runtime_qualified=False)
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOCK_PATH.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            hardware = idle_gpu(smi)
            (execution / "gpu.txt").write_text(hardware)
            save(execution / "runtime-environment.json", {key: os.environ.get(key) for key in
                 ("CUDA_HOME", "NCCL_ROOT", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                  "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")})
            for case in protocol["cases"]:
                failure["case"] = case["id"]
                verify_files(out, protocol["immutable_sha256"])
                require(idle_gpu(smi) == hardware, "Hardware changed during panel")
                directory = execution / case["id"]
                directory.mkdir()
                native = directory / "native"
                native.mkdir()
                command = [str(out / "binaries" / case["family"]), "--run",
                           str(out / "configs" / (case["model"] + ".ini")), str(case["batch"]),
                           case["phase"], case["mode"], case["workspace"], str(protocol["warmup"]),
                           str(protocol["samples"]), str(native)]
                process = claim.process(command, directory, directory / "native.log", args.timeout)
                verify_files(out, protocol["immutable_sha256"])
                result = json.loads((directory / "native.log").read_text())
                record = audit_result(result, case, protocol, native, out / "binaries" / case["family"])
                record["process_seconds"] = (process["end_monotonic_ns"] - process["launch_monotonic_ns"]) / 1e9
                require(math.isfinite(record["process_seconds"]) and record["process_seconds"] > 0, "Invalid process clock")
                record["native_log_sha256"] = sha(directory / "native.log")
                record["process_receipt_sha256"] = sha(directory / "native.log.json")
                save(directory / "audit.json", record)
                records[case["id"]] = record
            require(idle_gpu(smi) == hardware, "GPU/hardware contention at completion")
            verify_files(out, protocol["immutable_sha256"])
            for case in protocol["cases"]:
                directory = execution / case["id"]
                record = records[case["id"]]
                require(sha(directory / "native.log") == record["native_log_sha256"]
                        and sha(directory / "native.log.json") == record["process_receipt_sha256"], "Case receipts changed")
                for name, receipt in record["files"].items():
                    require(sha(directory / "native" / name) == receipt["sha256"], "Retained bytes changed")
            summary = audit_panel(protocol, records)
            save(execution / "REPORT.json", dict(**summary, records=records))
    except BaseException as error:
        failure.update(status="failed", error=str(error), completed_cases=list(records))
        save(execution / "FAILURE.json", failure)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="No GPU queries/policy execution")
    prep.add_argument("--binaries", type=Path, required=True)
    prep.add_argument("--configs", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--warmup", type=int, default=5)
    prep.add_argument("--samples", type=int, default=20)
    prep.add_argument("--repetitions", type=int, default=2)
    execution = subs.add_parser("run", help="Only after explicit GPU scheduling; hold remains in force")
    execution.add_argument("--out", type=Path, required=True)
    execution.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else run(args)
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]) if isinstance(result.get("cases"), list) else result["cases"],
                          runtime_qualified=False)))


if __name__ == "__main__":
    main()
