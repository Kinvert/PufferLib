#!/usr/bin/env python3
"""Bounded native CUDA encoder checks against an isolated float64 CUDA oracle.

NumPy only creates/inspects artifact arrays; it never computes a CNN reference.
prepare/inspect/audit never load a model library or query a GPU. Scheduled run
uses the shared reservation, idle checks and bounded worker process groups.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

import numpy as np
import dense_alias_acceptance as fixtures

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "native-encoder-cuda-oracle-smoke-v1"
LEARNER_SCHEMA = "native-encoder-cuda-oracle-smoke-v2"
TOOLS = ("research/gpu_encoder_smoke.py", "research/dense_alias_acceptance.py",
         "ocean/connect4cnn/claim.py", "ocean/connect4cnn/encoder_profile.py")
TOLERANCES = {"flex_quality": (8e-4, 6e-5), "nature_cnn": (3e-4, 3e-5)}
require, save, sha = fixtures.require, fixtures.save, fixtures.sha


def cases(learner=False):
    widths, batches = ((64, 128, 256), (1, 3, 64, 2048)) if learner else ((128,), (1, 3, 64))
    return [dict(id=f"{model}-h{hidden}-b{batch}-{state}", model=model, hidden=hidden, batch=batch, state=state)
            for model in fixtures.MODELS for hidden in widths for batch in batches for state in fixtures.STATES]


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def prepare(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    learner = getattr(args, "learner_panel", False)
    schema = LEARNER_SCHEMA if learner else SCHEMA
    save(out / "request.json", dict(schema=schema, status="preparing", gpu_queried=False))
    try:
        originals = {}
        for kind in ("native", "reference"):
            library = getattr(args, kind).resolve(); receipt = Path(str(library) + ".build")
            entries = fixtures.sources(receipt / "source.sha256", ROOT)
            command = (receipt / "command.txt").read_text()
            environment = (receipt / "environment.txt").read_text()
            require("NVCC_ARCH=native" not in environment and "NVCC_ARCH=" in environment, "Explicit architecture required")
            if kind == "native":
                require("tests/test_nature.cu" in command and "-DC4_DENSE_PATCH_ALIAS" not in command
                        and "C4_DENSE_PATCH_ALIAS=0" in environment, "Need the unchanged production callbacks")
            else:
                require("research/native_encoder_reference.cu" in command and "-fmad=false" in command,
                        "Need the independent double CUDA reference")
            require(sha(library) == (receipt / "binary.sha256").read_text().split()[0], "Changed compiled library")
            target = out / "libraries" / (kind + ".so"); target.parent.mkdir(exist_ok=True)
            shutil.copyfile(library, target)
            shutil.copytree(receipt, out / "build-receipts" / kind)
            for name, digest in entries.items():
                source = ROOT / name; destination = out / "source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists(): require(sha(destination) == digest, "Build source snapshots disagree")
                else: shutil.copyfile(source, destination)
                originals[str(source)] = digest
            originals[str(library)] = sha(library)
        for name in TOOLS:
            source = ROOT / name; target = out / "tools" / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            originals[str(source)] = sha(source)
        for name, digest in originals.items(): require(sha(name) == digest, "Preparation source/input changed")
        packet = dict(schema=schema, status="prepared-not-executed", cases=cases(learner), workers=["repeat-0", "repeat-1"],
            modes=list(fixtures.MODES), tolerances=TOLERANCES, files_sha256=hashes(out), original_sha256=originals,
            python_version=sys.version, numpy_version=np.__version__, oracle_device="cuda", oracle_precision="float64",
            cpu_neural_reference=False, full_policy_qualified=False, learner_batch_2048_qualified=False,
            performance_qualified=False)
        save(out / "packet.json", packet)
        return packet
    except BaseException as error:
        save(out / "preparation_failure.json", dict(error=str(error), gpu_queried=False)); raise


def load(out, current=False):
    value = json.loads((out / "packet.json").read_text())
    require(value["schema"] in (SCHEMA, LEARNER_SCHEMA) and value["status"] == "prepared-not-executed"
            and value["cases"] == cases(value["schema"] == LEARNER_SCHEMA) and value["workers"] == ["repeat-0", "repeat-1"]
            and value["modes"] == list(fixtures.MODES)
            and value["tolerances"] == {k: list(v) for k, v in TOLERANCES.items()}
            and value["oracle_device"] == "cuda" and value["cpu_neural_reference"] is False,
            "Changed protocol/allocation/tolerances")
    require(value["oracle_precision"] == "float64" and value["full_policy_qualified"] is False
            and value["learner_batch_2048_qualified"] is False and value["performance_qualified"] is False,
            "Preparation cannot certify full-policy/learner/performance gates")
    fixtures.verify_files(out, value["files_sha256"])
    if current:
        for name, digest in value["original_sha256"].items(): require(sha(name) == digest, "Current source/input changed")
        require(sys.version == value["python_version"] and np.__version__ == value["numpy_version"], "Runtime changed")
    return value


def reference_call(lib, model, data, output=None, gradient=None, loss=None):
    null = ctypes.POINTER(ctypes.c_double)()
    pointer = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if a is not None else null
    code = lib.cnnref_run(model, data[0].shape[0], data[2].shape[1],
        *(pointer(a) for a in (*data, output, gradient, loss)))
    require(code == 0, f"CUDA reference failed: {code}")


def worker(out, name):
    packet = load(out, current=True)
    require(name in packet["workers"] and (out / "execution/request.json").is_file(), "Scheduled supervisor required")
    request = json.loads((out / "execution/request.json").read_text())
    require(request["supervisor_pid"] == os.getppid() and request["nonce"] == os.environ.get("PUFFER_GPU_ORACLE_TOKEN")
            and request["packet_sha256"] == sha(out / "packet.json"), "Worker is not a child of its scheduled supervisor")
    root = out / "execution" / name / "arrays"; root.mkdir()
    native = ctypes.CDLL(str(out / "libraries/native.so"))
    reference = ctypes.CDLL(str(out / "libraries/reference.so"))
    native.naturetest_alias_enabled.restype = ctypes.c_int
    require(native.naturetest_alias_enabled() == 0, "Production callbacks required")
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    native.naturetest_init.argtypes = [ctypes.c_int, ctypes.c_int]; native.naturetest_init.restype = ctypes.c_int
    native.flextest_init.argtypes = [ctypes.c_int, ctypes.c_int, integers]; native.flextest_init.restype = ctypes.c_int
    native.naturetest_run.argtypes = [floats]*5 + [ctypes.c_int]; native.naturetest_run.restype = None
    native.naturetest_read_params.argtypes = [floats]; native.naturetest_read_params.restype = None
    native.naturetest_close.restype = None
    reference.cnnref_run.argtypes = [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_double)]*6
    reference.cnnref_run.restype = ctypes.c_int
    reference.cnnref_parameters.argtypes = [ctypes.c_int, ctypes.c_int]
    reference.cnnref_parameters.restype = ctypes.c_int
    reference.cnnref_selftest.restype = ctypes.c_int
    require(reference.cnnref_selftest() == 0, "Independent CUDA dyadic/ReLU/dX/loss self-test failed")
    for case in packet["cases"]:
        directory = root / case["id"]; directory.mkdir()
        data = fixtures.fixture(case); precise = tuple(a.astype(np.float64) for a in data)
        for label, a in zip(("input", "parameters", "upstream"), data): a.tofile(directory / (label + ".f32"))
        model = fixtures.MODELS.index(case["model"])
        require(reference.cnnref_parameters(model, case["hidden"]) == data[1].size == fixtures.count(case),
                "Independent CUDA reference parameter layout differs")
        expected = (np.empty_like(precise[2]), np.empty_like(precise[1]))
        reference_call(reference, model, precise, *expected)
        for label, a in zip(("output", "gradient"), expected): a.tofile(directory / ("expected-" + label + ".f64"))
        differences = []
        if case["batch"] == 1 and case["state"] == "nonblank":
            offset = 0
            for tensor, shape in enumerate(fixtures.shapes(case)):
                count = int(np.prod(shape)); idx = offset + int(np.argmax(np.abs(expected[1][offset:offset+count])))
                losses = []
                for sign, label in ((1, "plus"), (-1, "minus")):
                    weights = precise[1].copy(); weights[idx] += sign * 1e-6
                    weights.tofile(directory / f"fd-{tensor}-{label}-parameters.f64")
                    loss = np.empty(1, dtype=np.float64)
                    reference_call(reference, model, (precise[0], weights, precise[2]), loss=loss)
                    loss.tofile(directory / f"fd-{tensor}-{label}-loss.f64"); losses.append(float(loss[0]))
                delta = (losses[0]-losses[1])/2e-6
                np.testing.assert_allclose(delta, expected[1][idx], rtol=2e-5, atol=1e-8)
                differences.append(dict(tensor=tensor, index=idx, epsilon=1e-6, derivative=delta, analytic=float(expected[1][idx])))
                offset += count
        init = (native.naturetest_init(case["batch"], case["hidden"]) if model == 1 else
                native.flextest_init(case["batch"], case["hidden"], np.array(fixtures.QUALITY, dtype=np.int32)))
        try:
            require(init == fixtures.count(case), "Native parameter count differs")
            first = None
            for mode in packet["modes"]:
                actual = (np.empty_like(data[2]), np.empty_like(data[1])); parameters = np.empty_like(data[1])
                native.naturetest_run(*data, *actual, int(mode.startswith("graph")))
                native.naturetest_read_params(parameters)
                for label, a in zip(("output", "gradient", "parameters"), (*actual, parameters)):
                    a.tofile(directory / (mode + "-" + label + ".f32"))
                require(parameters.tobytes() == data[1].tobytes(), "Native parameter mutation")
                for got, want in zip(actual, expected): np.testing.assert_allclose(got, want, rtol=TOLERANCES[case["model"]][0], atol=TOLERANCES[case["model"]][1])
                fingerprints = [a.tobytes() for a in actual]
                require(first is None or first == fingerprints, "Eager/graph/repetition bytes differ")
                first = fingerprints
        finally: native.naturetest_close()
        save(directory / "case.json", dict(case=case, status="passed", finite_differences=differences))
        print("PASS " + case["id"], flush=True)
    save(root.parent / "worker.json", dict(status="passed", cases=len(packet["cases"]), dyadic_selftest=True, arrays_sha256=hashes(root)))
    print("COMPLETE " + name, flush=True)


def audit(out, completing=False):
    packet = load(out)
    execution = out / "execution"; seal = json.loads((execution / "seal.json").read_text())
    fixtures.verify_files(execution, seal)
    require(seal == {k: v for k, v in hashes(execution).items() if k not in ("seal.json", "REPORT.json", "REPORT.sha256")},
            "Execution inventory differs from its seal")
    request = json.loads((execution / "request.json").read_text())
    require(request["packet_sha256"] == sha(out / "packet.json") and type(request["timeout"]) is int
            and 0 < request["timeout"] <= 120, "Execution belongs to a different packet/deadline")
    previous_end = None; signatures = None
    for name in packet["workers"]:
        root = execution / name; record = json.loads((root / "worker.json").read_text())
        require(record["status"] == "passed" and record["cases"] == len(packet["cases"]) and record["dyadic_selftest"] is True, "Incomplete worker")
        fixtures.verify_files(root / "arrays", record["arrays_sha256"])
        require(hashes(root / "arrays") == record["arrays_sha256"], "Extra/missing worker arrays")
        timing = json.loads((root / "worker.log.json").read_text())
        require(timing["status"] == "ok" and timing["returncode"] == 0 and timing["command"] == command(out, name)
                and timing["cwd"] == str(root) and timing["end_monotonic_ns"] > timing["launch_monotonic_ns"] > 0
                and timing["timeout"] == request["timeout"]
                and (previous_end is None or timing["launch_monotonic_ns"] >= previous_end), "Invalid worker process receipt")
        previous_end = timing["end_monotonic_ns"]
        lines = (root / "worker.log").read_text().splitlines()
        require([l for l in lines if l.startswith("PASS ")] == ["PASS " + c["id"] for c in packet["cases"]]
                and lines[-1] == "COMPLETE " + name, "Incomplete worker output")
        require(sorted(p.name for p in (root / "arrays").iterdir()) == sorted(c["id"] for c in packet["cases"]), "Wrong case coverage")
        for case in packet["cases"]:
            directory = root / "arrays" / case["id"]; data = fixtures.fixture(case)
            for label, a in zip(("input", "parameters", "upstream"), data):
                require((directory / (label + ".f32")).read_bytes() == a.tobytes(), "Changed fixture")
            expected = [np.fromfile(directory / ("expected-" + label + ".f64"), dtype="<f8") for label in ("output", "gradient")]
            require([a.size for a in expected] == [data[2].size, data[1].size]
                    and all(np.isfinite(a).all() for a in expected), "Malformed GPU oracle artifacts")
            first = None
            for mode in packet["modes"]:
                actual = [np.fromfile(directory / (mode + "-" + label + ".f32"), dtype="<f4") for label in ("output", "gradient")]
                require((directory / (mode + "-parameters.f32")).read_bytes() == data[1].tobytes(), "Changed device parameters")
                for got, want in zip(actual, expected):
                    require(got.size == want.size and np.isfinite(got).all(), "Malformed native arrays")
                    np.testing.assert_allclose(got, want, rtol=TOLERANCES[case["model"]][0], atol=TOLERANCES[case["model"]][1])
                current = [a.tobytes() for a in actual]
                require(first is None or first == current, "Changed eager/graph/repeat bytes"); first = current
            checks = json.loads((directory / "case.json").read_text())
            require(checks["status"] == "passed" and checks["case"] == case, "Wrong case receipt")
            required = len(fixtures.shapes(case)) if case["batch"] == 1 and case["state"] == "nonblank" else 0
            require(len(checks["finite_differences"]) == required, "Missing GPU finite differences")
            offset = 0
            for tensor, check in enumerate(checks["finite_differences"]):
                count = int(np.prod(fixtures.shapes(case)[tensor])); idx = offset + int(np.argmax(np.abs(expected[1][offset:offset+count])))
                losses = []
                require(check["index"] == idx and check["tensor"] == tensor and check["epsilon"] == 1e-6, "Wrong finite difference probe")
                for sign, label in ((1, "plus"), (-1, "minus")):
                    perturbed = data[1].astype("<f8"); perturbed[idx] += sign * 1e-6
                    require((directory / f"fd-{tensor}-{label}-parameters.f64").read_bytes() == perturbed.tobytes(), "Changed GPU perturbation input")
                    loss = np.fromfile(directory / f"fd-{tensor}-{label}-loss.f64", dtype="<f8")
                    require(loss.size == 1 and np.isfinite(loss).all(), "Malformed GPU loss"); losses.append(float(loss[0]))
                delta = (losses[0]-losses[1])/2e-6
                require(check["derivative"] == delta and check["analytic"] == float(expected[1][idx]), "Changed finite difference receipt")
                np.testing.assert_allclose(delta, expected[1][idx], rtol=2e-5, atol=1e-8); offset += count
        current_signatures = record["arrays_sha256"]
        require(signatures is None or signatures == current_signatures, "Independent-process arrays differ")
        signatures = current_signatures
    result = dict(status="scoped-cuda-encoder-comparisons-passed", cases=len(packet["cases"]), workers=2,
        native_calls=len(packet["cases"])*len(packet["workers"])*len(packet["modes"]),
        oracle_device="cuda", oracle_precision="float64", cpu_neural_reference=False,
        full_policy_qualified=False, learner_batch_2048_qualified=False, performance_qualified=False)
    if packet["schema"] == LEARNER_SCHEMA:
        result.update(encoder_batch_2048_numerically_checked=True, encoder_hidden_widths=[64, 128, 256],
                      encoder_batches=[1, 3, 64, 2048])
    if not completing:
        require(sha(execution / "REPORT.json") == (execution / "REPORT.sha256").read_text().strip()
                and json.loads((execution / "REPORT.json").read_text()) == result, "Missing/changed successful report")
    return result


def command(out, name):
    return [sys.executable, str(Path(__file__).resolve()), "_worker", "--out", str(out), "--name", name]


def run(args):
    out = args.out.resolve(); packet = load(out, current=True)
    require(0 < args.timeout <= 120, "Bounded smoke requires a 1–120 second worker timeout")
    execution = out / "execution"; execution.mkdir()
    nonce = secrets.token_hex(16)
    save(execution / "request.json", dict(timeout=args.timeout, packet_sha256=sha(out / "packet.json"),
        status="scheduled-local-smoke", supervisor_pid=os.getpid(), nonce=nonce))
    try:
        with fixtures.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            hardware = fixtures.idle_gpu(smi); (execution / "gpu.txt").write_text(hardware)
            for name in packet["workers"]:
                require(fixtures.idle_gpu(smi) == hardware, "Hardware/idle state changed")
                load(out, current=True); directory = execution / name; directory.mkdir()
                timed = fixtures.claim.process(command(out, name), directory, directory / "worker.log", args.timeout,
                    dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                         PUFFER_GPU_ORACLE_TOKEN=nonce))
                require(timed["status"] == "ok" and timed["returncode"] == 0, "Native/oracle worker failed; retain all evidence")
                load(out, current=True)
            require(fixtures.idle_gpu(smi) == hardware, "Hardware/idle state changed at completion")
            save(execution / "seal.json", hashes(execution))
            result = audit(out, completing=True); save(execution / "REPORT.json", result)
            (execution / "REPORT.sha256").write_text(sha(execution / "REPORT.json") + "\n")
            return result
    except BaseException as error:
        save(execution / "failure.json", dict(status="failed", error=str(error))); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--native", type=Path, required=True); prep.add_argument("--reference", type=Path, required=True)
    prep.add_argument("--learner-panel", action="store_true",
                      help="Opt-in v2: fixed quality/Nature H64/128/256 and B1/3/64/2048, 72 cases")
    for action in ("inspect", "audit", "run", "_worker"): sub.add_parser(action)
    for action in sub.choices.values(): action.add_argument("--out", type=Path, required=True)
    sub.choices["run"].add_argument("--timeout", type=int, default=120)
    sub.choices["_worker"].add_argument("--name", required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if args.action == "_worker": worker(out, args.name); return
    result = {"prepare": lambda: prepare(args), "inspect": lambda: load(out, current=True),
              "audit": lambda: audit(out), "run": lambda: run(args)}[args.action]()
    print(json.dumps({k: result[k] for k in ("status", "cases", "cpu_neural_reference")}, indent=2))


if __name__ == "__main__": main()
