#!/usr/bin/env python3
"""Paired native encoder acceptance. Preparation/audit never execute a network.

run is for a separately scheduled exclusive GPU window; neither hold is lifted.
The private worker calls native CUDA before computing an independent oracle.
This is test/audit glue, not a Python learner or architecture search framework.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
import claim
from encoder_profile import idle_gpu

LOCK = ROOT / "build/connect4cnn/hardware-benchmark.lock"
SCHEMA = "dense-patch-alias-acceptance-v2"
SEED = 53417
MODELS = ("flex_quality", "nature_cnn")
VARIANTS = ("baseline", "candidate")
STATES = ("nonblank", "zero-observation", "zero-weights-and-observation")
MODES = ("eager-0", "eager-1", "graph-0", "graph-1")
QUALITY = [1, 64, 0, 16, 7, 4, 0, 0, 8, 3, 1, 0, 0, 8, 3, 1, 0, 0]
TOOLS = ("research/dense_alias_acceptance.py", "ocean/connect4cnn/claim.py",
         "ocean/connect4cnn/encoder_profile.py", "ocean/connect4cnn/tests/test_nature.py",
         "ocean/connect4cnn/tests/test_flex.py", "ocean/connect4cnn/tests/test_impala.py")
ENV_KEYS = ("CUDA_HOME", "NCCL_ROOT", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cases():
    return [dict(id=f"{model}-h{hidden}-b{batch}-{state}", model=model,
                 hidden=hidden, batch=batch, state=state)
            for model in MODELS for hidden in (64, 128, 256)
            for batch in (1, 3, 64, 2048) for state in STATES]


def jobs():
    return [dict(id=f"r{rep}-{variant}", repetition=rep, variant=variant)
            for rep in range(2) for variant in (VARIANTS if rep == 0 else VARIANTS[::-1])]


def shapes(case):
    h = case["hidden"]
    if case["model"] == "nature_cnn":
        return [(32, 64), (32,), (64, 512), (64,), (64, 576), (64,), (h, 128), (h,)]
    require(case["model"] == "flex_quality", "Unknown fixed graph")
    return [(16, 49), (16,), (64, 1584), (64,)] + ([(h, 64), (h,)] if h != 64 else [])


def count(case):
    return sum(math.prod(shape) for shape in shapes(case))


def sources(path, root):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and name not in result,
                "Unsafe/duplicate native source receipt")
        require(sha(root / name) == digest, f"Native build source changed: {name}")
        result[name] = digest
    require(result, "Empty native source receipt")
    return result


def verify_files(root, entries):
    for name, digest in entries.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe immutable path")
        require(sha(root / name) == digest, f"Immutable input changed: {name}")


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "request.json", dict(schema=SCHEMA, status="preparing", gpu_execution_authorized=False))
    originals, native_sources, paired_environment, compiler = {}, None, None, None
    for variant in VARIANTS:
        library = getattr(args, variant).resolve()
        receipt = Path(str(library) + ".build")
        entries = sources(receipt / "source.sha256", ROOT)
        if native_sources is None:
            native_sources = entries
            for name in entries:
                target = out / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
        require(entries == native_sources, "Paired native builds used different source snapshots")
        environment = dict(line.split("=", 1) for line in (receipt / "environment.txt").read_text().splitlines())
        require(environment["C4_DENSE_PATCH_ALIAS"] == str(VARIANTS.index(variant)), "Wrong candidate build marker")
        require(environment["NVCC_ARCH"] != "native", "Explicit compile architecture required")
        require(environment["NVCC_APPEND_FLAGS"] == "" and environment["NVCC_PREPEND_FLAGS"] in ("", "--threads 1"),
                "Unreviewed compiler overrides")
        shared_environment = {k: v for k, v in environment.items() if k != "C4_DENSE_PATCH_ALIAS"}
        compiler_text = (receipt / "compiler.txt").read_text()
        require(paired_environment is None or paired_environment == shared_environment, "Paired compile environments differ")
        require(compiler is None or compiler == compiler_text, "Paired compilers differ")
        paired_environment, compiler = shared_environment, compiler_text
        command = (receipt / "command.txt").read_text()
        require("tests/test_nature.cu" in command and "-shared" in command, "Wrong native test harness")
        require(("-DC4_DENSE_PATCH_ALIAS" in command) == (variant == "candidate"), "Build command marker differs")
        lines = (receipt / "binary.sha256").read_text().splitlines()
        require(len(lines) == 1 and sha(library) == lines[0].split(maxsplit=1)[0], "Library changed since compilation")
        target = out / "libraries" / f"{variant}.so"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(library, target)
        destination = out / "native-build" / variant
        destination.mkdir(parents=True)
        for path in sorted(receipt.iterdir()):
            require(path.is_file(), "Unexpected build receipt directory")
            shutil.copyfile(path, destination / path.name)
            originals[str(path)] = sha(path)
        originals[str(library)] = sha(library)
    require(sha(out / "libraries/baseline.so") != sha(out / "libraries/candidate.so"), "Both variants contain the same library")
    for name in TOOLS:
        source = ROOT / name
        target = out / "tooling" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        originals[str(source)] = sha(source)
    for name, digest in native_sources.items():
        require(sha(ROOT / name) == digest, "Native source changed during preparation")
    for name, digest in originals.items(): require(sha(name) == digest, "Input changed during preparation")
    immutable = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    protocol = dict(schema=SCHEMA, status="prepared_not_executed", cases=cases(), jobs=jobs(), modes=list(MODES),
                    fixture_seed=SEED, immutable_sha256=immutable, original_sha256=originals,
                    tool_sha256={name: sha(ROOT / name) for name in TOOLS},
                    cuda_executed=False, encoder_executed=False, full_policy_qualified=False,
                    performance_qualified=False, tolerances={"nature_cnn": [3e-4, 3e-5], "flex_quality": [8e-4, 6e-5]},
                    numpy_version=__import__("numpy").__version__, python_version=sys.version,
                    note="Synthetic acceptance fixtures only. Preparation is not GPU authorization.")
    save(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    return protocol


def load(out, current_tools=True):
    require(sha(out / "protocol.json") == (out / "protocol.sha256").read_text().strip(), "Protocol changed")
    p = json.loads((out / "protocol.json").read_text())
    require(p["schema"] == SCHEMA and p["status"] == "prepared_not_executed" and p["cases"] == cases()
            and p["jobs"] == jobs() and p["modes"] == list(MODES) and p["fixture_seed"] == SEED, "Changed fixed acceptance panel")
    for kind, expected in (("cases", cases()), ("jobs", jobs())):
        for actual, template in zip(p[kind], expected):
            require(all(type(actual[key]) is type(value) for key, value in template.items()), "Wrong panel field type")
    require(p["tolerances"] == {"nature_cnn": [3e-4, 3e-5], "flex_quality": [8e-4, 6e-5]}, "Changed oracle tolerances")
    require(p["cuda_executed"] is False and p["encoder_executed"] is False
            and p["full_policy_qualified"] is False and p["performance_qualified"] is False, "Invalid preparation scope")
    verify_files(out, p["immutable_sha256"])
    if current_tools:
        for name, digest in p["tool_sha256"].items(): require(sha(ROOT / name) == digest, "Acceptance tooling changed; prepare afresh")
    return p


def fixture(case):
    """Synthetic test arrays only; no network/reference is evaluated here."""
    import numpy as np
    rng = np.random.default_rng(SEED + case["hidden"])
    scale = .5 if case["model"] == "nature_cnn" else .4
    arrays = [rng.normal(0, scale / np.sqrt(sh[1]), sh) if len(sh) == 2
              else rng.choice([-.2, .2], sh) for sh in shapes(case)]
    obs = np.random.default_rng(SEED + 2).choice([0, .5, 1], (case["batch"], 1584)).astype("<f4")
    values = np.concatenate([a.ravel() for a in arrays]).astype("<f4")
    upstream = np.random.default_rng(SEED + 4).normal(0, .1, (case["batch"], case["hidden"])).astype("<f4")
    if case["state"] != "nonblank": obs.fill(0)
    if case["state"] == "zero-weights-and-observation": values.fill(0)
    return obs, values, upstream


def worker(out, job_id):
    # Never invoked by prepare/inspect/audit. The supervisor schedules this
    # private subprocess only after an explicit exclusive GPU window.
    import ctypes
    import numpy as np
    p = load(out)
    matches = [job for job in p["jobs"] if job["id"] == job_id]
    require(len(matches) == 1 and (out / "execution/runtime-environment.json").is_file(), "Worker requires scheduled supervisor")
    job = matches[0]
    destination = out / "execution" / job_id / "arrays"
    destination.mkdir(exist_ok=False)
    library = ctypes.CDLL(str(out / "libraries" / f"{job['variant']}.so"))
    library.naturetest_alias_enabled.restype = ctypes.c_int
    require(library.naturetest_alias_enabled() == VARIANTS.index(job["variant"]), "Loaded wrong native callback variant")
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    integers = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    library.naturetest_init.argtypes = [ctypes.c_int, ctypes.c_int]
    library.naturetest_init.restype = ctypes.c_int
    library.flextest_init.argtypes = [ctypes.c_int, ctypes.c_int, integers]
    library.flextest_init.restype = ctypes.c_int
    library.naturetest_run.argtypes = [floats] * 5 + [ctypes.c_int]
    library.naturetest_run.restype = None
    library.naturetest_close.restype = None
    library.naturetest_read_params.argtypes = [floats]
    library.naturetest_read_params.restype = None
    # Imports define independent references; no reference runs before CUDA init.
    sys.path.insert(0, str(ROOT / "ocean/connect4cnn/tests"))
    import test_nature
    import test_flex
    for case in p["cases"]:
        directory = destination / case["id"]
        directory.mkdir()
        data = fixture(case)
        for name, values in zip(("input.f32", "parameters.f32", "upstream.f32"), data): values.tofile(directory / name)
        if case["model"] == "nature_cnn":
            n = library.naturetest_init(case["batch"], case["hidden"])
        else:
            n = library.flextest_init(case["batch"], case["hidden"], np.array(QUALITY, np.int32))
        try:
            require(n == count(case), "Native parameter registration differs")
            # GPU buffers are allocated and initialization synchronized above.
            # This is an oracle paired with native CUDA, never a CPU substitute.
            precise = tuple(a.astype(np.float64) for a in data)
            expected = (test_nature.reference(*precise) if case["model"] == "nature_cnn" else
                        test_flex.reference(precise, test_flex.plan(QUALITY, case["hidden"]), shapes(case)))
            for name, values in zip(("expected-output.f64", "expected-gradient.f64"), expected): values.tofile(directory / name)
            first = None
            rtol, atol = p["tolerances"][case["model"]]
            for mode in MODES:
                actual = [np.empty_like(data[2]), np.empty_like(data[1])]
                library.naturetest_run(*data, *actual, int(mode.startswith("graph")))
                parameters = np.empty_like(data[1])
                library.naturetest_read_params(parameters)
                for suffix, values in zip(("output.f32", "gradient.f32", "parameters.f32"), [*actual, parameters]):
                    values.tofile(directory / f"{mode}-{suffix}")
                require(parameters.tobytes() == data[1].tobytes(), "Native execution mutated parameters")
                for got, want in zip(actual, expected): np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
                fingerprints = [a.tobytes() for a in actual]
                require(first is None or first == fingerprints, "Native eager/graph/repetition byte mismatch")
                first = fingerprints
            require(all((directory / name).read_bytes() == array.tobytes() for name, array in
                        zip(("input.f32", "parameters.f32", "upstream.f32"), data)), "Fixture mutated")
            save(directory / "result.json", dict(case=case, status="passed", candidate=job["variant"] == "candidate"))
            print("PASS " + case["id"], flush=True)
        finally:
            library.naturetest_close()
    print("COMPLETE " + job_id, flush=True)


def audit_worker(p, job, directory):
    """Only inspect retained arrays; no reference/model computation."""
    import numpy as np
    root = directory / "arrays"
    require(sorted(path.name for path in root.iterdir()) == sorted(case["id"] for case in p["cases"]), "Missing/extra case receipts")
    records = {}
    for case in p["cases"]:
        folder = root / case["id"]
        result = json.loads((folder / "result.json").read_text())
        require(result == dict(case=case, status="passed", candidate=job["variant"] == "candidate"), "Wrong case/variant receipt")
        require(type(result["candidate"]) is bool and all(type(result["case"][key]) is type(value)
                for key, value in case.items()), "Wrong case receipt field type")
        output_count, param_count = case["batch"] * case["hidden"], count(case)
        sizes = {"input.f32": case["batch"] * 1584, "parameters.f32": param_count, "upstream.f32": output_count,
                 "expected-output.f64": output_count, "expected-gradient.f64": param_count}
        for mode in MODES:
            sizes.update({f"{mode}-output.f32": output_count, f"{mode}-gradient.f32": param_count,
                          f"{mode}-parameters.f32": param_count})
        require({path.name for path in folder.iterdir()} == set(sizes) | {"result.json"}, "Missing/extra retained arrays")
        arrays, hashes = {}, {"result.json": sha(folder / "result.json")}
        for name, elements in sizes.items():
            path = folder / name
            dtype = "<f8" if name.endswith(".f64") else "<f4"
            require(path.stat().st_size == elements * np.dtype(dtype).itemsize, "Retained array size differs")
            arrays[name] = np.fromfile(path, dtype=dtype)
            require(np.isfinite(arrays[name]).all(), "Nonfinite retained array")
            hashes[name] = sha(path)
        for name, values in zip(("input.f32", "parameters.f32", "upstream.f32"), fixture(case)):
            require((folder / name).read_bytes() == values.tobytes(), "Wrong seeded fixture")
        for mode in MODES:
            require(hashes[f"{mode}-parameters.f32"] == hashes["parameters.f32"], "Parameters mutated")
            for field in ("output", "gradient"):
                require(hashes[f"{mode}-{field}.f32"] == hashes[f"eager-0-{field}.f32"], "Mode/repetition byte mismatch")
                rtol, atol = p["tolerances"][case["model"]]
                np.testing.assert_allclose(arrays[f"{mode}-{field}.f32"], arrays[f"expected-{field}.f64"], rtol=rtol, atol=atol)
        records[case["id"]] = hashes
    return records


def compare_workers(p, records):
    require(set(records) == {job["id"] for job in p["jobs"]}, "Missing/extra worker records")
    for case in p["cases"]:
        reference = None
        for job in p["jobs"]:
            hashes = records[job["id"]][case["id"]]
            # result.json differs only by variant; all actual bytes must match.
            actual = {key: value for key, value in hashes.items() if key != "result.json"}
            require(reference is None or reference == actual, "Baseline/candidate/independent-process byte mismatch")
            reference = actual
    return dict(status="paired_encoder_math_passed", cases=len(p["cases"]), workers=len(p["jobs"]),
                native_calls=len(p["cases"]) * len(p["jobs"]) * len(MODES),
                cuda_executed=True, encoder_executed=True, full_policy_qualified=False, performance_qualified=False,
                note="Fixed encoder synthetic numerical acceptance only; no learner concurrency/reload, SPS, score or multi-game dominance.")


def command_for(out, job):
    return [sys.executable, str(Path(__file__).resolve()), "_worker", "--out", str(out), "--job", job["id"]]


def audit_processes(p, execution):
    bindings = json.loads((execution / "process-bindings.json").read_text())
    require(set(bindings) == {job["id"] for job in p["jobs"]}, "Missing/extra process binding")
    previous_end = 0
    for job in p["jobs"]:
        directory = execution / job["id"]
        receipt = json.loads((directory / "worker.log.json").read_text())
        binding = bindings[job["id"]]
        require(receipt["status"] == "ok" and type(receipt["returncode"]) is int and receipt["returncode"] == 0,
                "Worker failed or lacks terminal success")
        require(receipt["command"] == binding["command"] and receipt["cwd"] == binding["cwd"], "Worker command/cwd differ")
        start, end = receipt["launch_monotonic_ns"], receipt["end_monotonic_ns"]
        require(type(start) is int and type(end) is int and end > start > 0 and start >= previous_end,
                "Invalid/overlapping worker process clocks")
        previous_end = end
        lines = (directory / "worker.log").read_text().splitlines()
        require([line for line in lines if line.startswith("PASS ")] == ["PASS " + case["id"] for case in p["cases"]]
                and lines[-1] == "COMPLETE " + job["id"], "Incomplete/changed worker completion log")


def audit(out):
    p = load(out, current_tools=False)
    require(__import__("numpy").__version__ == p["numpy_version"], "Use pinned NumPy for seeded fixture audit")
    execution = out / "execution"
    require(not (execution / "FAILURE.json").exists(), "Failed panel cannot qualify")
    require(sha(execution / "REPORT.json") == (execution / "REPORT.sha256").read_text().strip(), "Report changed")
    verify_files(execution, json.loads((execution / "execution-sha256.json").read_text()))
    audit_processes(p, execution)
    records = {job["id"]: audit_worker(p, job, execution / job["id"]) for job in p["jobs"]}
    summary = compare_workers(p, records)
    require(json.loads((execution / "REPORT.json").read_text()) == dict(**summary, records=records), "Final paired report differs")
    return summary


def run(args):
    out = args.out.resolve()
    p = load(out)
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 1800, "Invalid worker timeout")
    require(__import__("numpy").__version__ == p["numpy_version"] and sys.version == p["python_version"], "Interpreter/NumPy changed")
    execution = out / "execution"
    execution.mkdir(exist_ok=False)
    records, failure = {}, dict(status="starting", full_policy_qualified=False, performance_qualified=False)
    try:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            hardware = idle_gpu(smi)
            (execution / "gpu.txt").write_text(hardware)
            environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            save(execution / "runtime-environment.json", {key: environment.get(key) for key in ENV_KEYS})
            save(execution / "process-bindings.json", {job["id"]: dict(command=command_for(out, job),
                 cwd=str(execution / job["id"])) for job in p["jobs"]})
            for job in p["jobs"]:
                failure["job"] = job["id"]
                load(out)
                require(idle_gpu(smi) == hardware, "GPU topology changed or contention observed")
                directory = execution / job["id"]
                directory.mkdir()
                command = command_for(out, job)
                process = claim.process(command, directory, directory / "worker.log", args.timeout, environment)
                load(out)
                require(process["status"] == "ok" and process["returncode"] == 0
                        and process["command"] == command and process["cwd"] == str(directory), "Wrong worker process receipt")
                require(process["end_monotonic_ns"] > process["launch_monotonic_ns"] > 0, "Invalid process clocks")
                records[job["id"]] = audit_worker(p, job, directory)
            require(idle_gpu(smi) == hardware, "GPU topology changed or contention at completion")
            load(out)
            for job in p["jobs"]:
                require(audit_worker(p, job, execution / job["id"]) == records[job["id"]], "Prior receipts changed")
            audit_processes(p, execution)
            summary = compare_workers(p, records)
            hashes = {str(path.relative_to(execution)): sha(path) for path in sorted(execution.rglob("*")) if path.is_file()}
            save(execution / "execution-sha256.json", hashes)
            save(execution / "REPORT.json", dict(**summary, records=records))
            (execution / "REPORT.sha256").write_text(sha(execution / "REPORT.json") + "\n")
    except BaseException as error:
        failure.update(status="failed", error=str(error), completed_workers=list(records))
        save(execution / "FAILURE.json", failure)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="Copy libraries/build/tool receipts; never load a library or query a GPU")
    prep.add_argument("--baseline", type=Path, required=True)
    prep.add_argument("--candidate", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    inspect = subs.add_parser("inspect", help="Offline immutable-input check")
    inspect.add_argument("--out", type=Path, required=True)
    review = subs.add_parser("audit", help="Offline retained-array/process audit; no model executes")
    review.add_argument("--out", type=Path, required=True)
    launch = subs.add_parser("run", help="Scheduled exclusive GPU acceptance only; currently held")
    launch.add_argument("--out", type=Path, required=True)
    launch.add_argument("--timeout", type=float, default=600)
    hidden = subs.add_parser("_worker", help=argparse.SUPPRESS)
    hidden.add_argument("--out", type=Path, required=True)
    hidden.add_argument("--job", required=True)
    args = parser.parse_args()
    if args.command == "_worker":
        worker(args.out.resolve(), args.job)
        return
    result = (prepare(args) if args.command == "prepare" else run(args) if args.command == "run" else
              audit(args.out.resolve()) if args.command == "audit" else load(args.out.resolve()))
    print(json.dumps({key: result[key] for key in ("status", "full_policy_qualified", "performance_qualified")}))


if __name__ == "__main__": main()
