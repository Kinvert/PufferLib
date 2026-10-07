"""Frozen development replication and timing canaries; native training/evaluation.

Not the gated publication confirmation. No search, no adaptive seed extension.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import deterministic_eval as ev
from claim import ENVIRONMENT_RULES, common_settings, ini, normalized, parse_checkpoints, process, RECEIPT, require

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"research"))
from claim_frontier import report

MODELS = ("flex_quality", "nature_cnn")
PARAMETERS = dict(flex_quality=160736, nature_cnn=138528)
FAMILIES = dict(flex_quality="flex", nature_cnn="nature")


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def gpu():
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    require(not busy.strip(), "GPU occupied; no job was interrupted")
    return subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20)


def train(binary, directory, receipts=True, interrupt=False):
    env = {**os.environ, "PUFFER_CHECKPOINT_RECEIPTS": "1" if receipts else "0"}
    command = [str(binary), "train", "--headless"]
    with (ROOT/"build/connect4cnn/hardware-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory/"gpu.txt").write_text(gpu())
        if not interrupt:
            stop = threading.Event()
            contention = []
            def observe_competitors():
                smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
                while not stop.wait(5):
                    try:
                        active = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"],
                                                         text=True, timeout=10).strip().splitlines()
                        # Single-GPU trainer: at most one compute process.
                        if len([p for p in active if p.strip()]) > 1:
                            contention.append(dict(monotonic_ns=time.monotonic_ns(), pids=active))
                    except (OSError, subprocess.SubprocessError) as error:
                        contention.append(dict(monotonic_ns=time.monotonic_ns(), telemetry_error=str(error)))
            monitor = threading.Thread(target=observe_competitors, daemon=True)
            monitor.start()
            try:
                return process(command, directory, directory/"train.log", 900, env)
            except (ValueError, subprocess.SubprocessError):
                return json.loads((directory/"train.log.json").read_text())
            finally:
                stop.set(); monitor.join(timeout=12)
                save(directory/"contention.json", dict(status="uncontended" if not contention else "contaminated_or_unknown", events=contention,
                    sampling_seconds=5, note="Sampled competitor check; absence of brief unsampled interference is not proven."))
        # Deliberate bounded failure immediately after the first completed file.
        # This is a measurement test, not an early stopping rule for replication.
        record = dict(command=command, cwd=str(directory), status="running",
                      launch_monotonic_ns=time.monotonic_ns(), intentional_interrupt=True,
                      environment=dict(PUFFER_CHECKPOINT_RECEIPTS="1"))
        child = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, start_new_session=True)
        record["pid"] = child.pid
        killed_after = None
        try:
            with (directory/"train.log").open("xb") as stream:
                data = b""
                while child.poll() is None:
                    require(time.monotonic_ns()-record["launch_monotonic_ns"] < 60*10**9, "Failure-canary deadline")
                    readable, _, _ = select.select([child.stdout], [], [], 1)
                    if not readable:
                        continue
                    chunk = os.read(child.stdout.fileno(), 65536)
                    stream.write(chunk); stream.flush(); data += chunk
                    match = RECEIPT.search(data.decode(errors="replace"))
                    if match:
                        killed_after = int(match[1])
                        os.killpg(child.pid, signal.SIGKILL)
                        break
                child.wait(timeout=10)
                stream.write(child.stdout.read())
            require(killed_after is not None and child.returncode != 0, "Failure injection did not occur")
            record.update(status="failed", returncode=child.returncode, killed_after_step=killed_after)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            record["end_monotonic_ns"] = time.monotonic_ns()
            save(directory/"train.log.json", record)
        return record


def verify_checkpoints(directory, timed, steps, parameters):
    times = parse_checkpoints((directory/"train.log").read_text(), timed, steps, parameters, allow_partial=True)
    checkpoints = directory/"checkpoints/connect4cnn/trial"
    complete = {}
    for step, seconds in times.items():
        path = checkpoints/f"{step:016d}.bin"
        weights = np.fromfile(path, dtype=np.float32)
        require(weights.size == parameters and np.isfinite(weights).all(), "Completed checkpoint invalid")
        complete[str(step)] = dict(seconds=seconds, sha256=ev.sha(path))
    require(normalized(ini(checkpoints/"resolved.ini")) == normalized(ini(directory/"config/default.ini")),
            "Native startup config differs from frozen launch")
    if timed["status"] == "ok":
        require(normalized(ini(directory/"metrics/connect4cnn/trial.ini")) == normalized(ini(directory/"config/default.ini")),
                "Native final config differs from frozen launch")
    return complete


def native_sources():
    paths = [ROOT/"build.sh", ROOT/"config/default.ini", ROOT/"config/connect4cnn.ini"]
    for directory in (ROOT/"src", ROOT/"vendor", ROOT/"ocean/connect4cnn"):
        paths += [p for p in directory.rglob("*") if p.is_file() and p.suffix in (".cu", ".c", ".h", ".ini", ".json", ".sh")]
    return {str(p.relative_to(ROOT)): ev.sha(p) for p in paths}


def prepare(out, binaries, canary):
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((ROOT/"ocean/connect4cnn/confirmation.json").read_text())
    base = ini(ROOT/"config/default.ini", ROOT/"config/connect4cnn.ini", ROOT/"ocean/connect4cnn/compare.ini")
    for section in list(base.sections()):
        if section.startswith("sweep."):
            base.remove_section(section)
    for key in list(base["policy"]):
        if key == "encoder" or key.startswith("cnn_"):
            del base["policy"][key]
    steps, cadence = (262144, 32768) if canary else (13312000, 262144)
    seeds = [53201, 53202, 53203, 53204] if canary else [53101, 53102, 53103, 53104, 53105]
    planned = list(range(cadence, steps+1, cadence))
    if planned[-1] != steps:
        planned.append(steps)
    jobs, common = [], None
    (out/"binaries").mkdir()
    for model, name in (("flex_quality", "timing-default"), ("nature_cnn", "timing-nature")):
        shutil.copy2(binaries/name, out/"binaries"/model)
    for index, seed in enumerate(seeds):
        order = MODELS if index % 2 == 0 else MODELS[::-1]
        for model in order:
            flags = (False, True) if canary and index % 2 == 0 else ((True, False) if canary else (True,))
            for receipts in flags:
                name = f"{model}-s{seed}"+(f"-receipts-{int(receipts)}" if canary else "")
                directory = out/name
                (directory/"config").mkdir(parents=True)
                config = copy.deepcopy(base)
                config["policy"].update({k: str(v) for k, v in frozen["policies"].get(model, dict(encoder=0)).items()})
                config["base"].update(seed=str(seed), run_id="trial", checkpoint_dir="checkpoints", log_dir="metrics",
                                       checkpoint_interval=str(cadence//2048), eval_episodes="0", load_model_path="None")
                config["train"]["total_timesteps"] = str(steps)
                with (directory/"config/default.ini").open("x") as stream:
                    config.write(stream)
                (directory/"config/connect4cnn.ini").touch()
                settings = common_settings(config)
                if common is None:
                    common = settings
                require(settings == common, "Learner/core/env settings differ")
                jobs.append(dict(id=name, model=model, seed=seed, receipts=receipts,
                                 config_sha256=ev.sha(directory/"config/default.ini")))
    ev.create_suite(argparse.Namespace(out=out/"suite", seed=253101 if not canary else 253201,
                                      offset=0, episodes=1000 if not canary else 65, slots=64,
                                      representation=0, purpose="development"))
    protocol = dict(version="connect4-development-replication-v1", purpose="timing_canary" if canary else "paired_development_replication",
                    models=list(MODELS), training_seeds=seeds, steps=steps, checkpoint_steps=planned,
                    environment_rules=ENVIRONMENT_RULES, precision="float32", jobs=jobs, common_settings=common,
                    binary_sha256={m: ev.sha(out/"binaries"/m) for m in MODELS}, source_sha256=native_sources(),
                    suite_sha256=ev.sha(out/"suite/suite.json"),
                    timing="CLOCK_MONOTONIC launch to native post-rename complete checkpoint receipt",
                    evaluation_protocol="connect4-fixed-suite-v1", stopping="fixed 5 paired seeds; no adaptation, no selective retries",
                    inference="development only; uncalibrated candidate joint bands cannot certify superiority",
                    candidate_bands=False,
                    selection="existing locked quality Flex versus adapted Nature; fixed common learner and H128/L1 core")
    if not canary:
        protocol["timing_gate_required"] = True
    protocol["tool_sha256"] = {str(p.relative_to(ROOT)): ev.sha(p) for p in (
        Path(__file__), ROOT/"ocean/connect4cnn/deterministic_eval.py", ROOT/"ocean/connect4cnn/claim.py",
        ROOT/"research/claim_frontier.py", ROOT/"research/claim_frontier.html")}
    for name, digest in {**protocol["source_sha256"], **protocol["tool_sha256"]}.items():
        target = out/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        require(ev.sha(target) == digest, "Snapshot changed while preparing")
    ev.save(out/"protocol.json", protocol)
    (out/"protocol.sha256").write_text(ev.sha(out/"protocol.json")+"\n")
    return protocol


def load(out):
    require(ev.sha(out/"protocol.json") == (out/"protocol.sha256").read_text().strip(), "Frozen protocol changed")
    protocol = json.loads((out/"protocol.json").read_text())
    for model, digest in protocol["binary_sha256"].items():
        require(ev.sha(out/"binaries"/model) == digest, "Execution binary changed")
    for name, digest in protocol["source_sha256"].items():
        require(ev.sha(ROOT/name) == digest, f"Source changed after preparation: {name}")
    for name, digest in protocol.get("tool_sha256", {}).items():
        require(ev.sha(ROOT/name) == digest and ev.sha(out/"source"/name) == digest, f"Measurement tool changed: {name}")
    for job in protocol["jobs"]:
        require(ev.sha(out/job["id"]/"config/default.ini") == job["config_sha256"], "Frozen config changed")
    require(ev.sha(out/"suite/suite.json") == protocol["suite_sha256"], "Frozen suite changed")
    return protocol


def evaluate(out, job, step, config):
    directory = out/job["id"]
    target = directory/f"eval-{step:016d}"
    command = [sys.executable, str(Path(ev.__file__)), "run", "--suite", str(out/"suite/suite.json"),
               "--binary", str(out/"binaries"/job["model"]), "--family", FAMILIES[job["model"]],
               "--config", str(config), "--checkpoint", str(directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"),
               "--out", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=330)
    (directory/f"eval-{step:016d}.launcher.log").write_text(result.stdout+result.stderr)
    require(result.returncode == 0, "Checkpoint evaluation failed; partial rows preserved")
    record = json.loads((target/"result.json").read_text())
    require(record["status"] == "ok" and record["suite_sha256"] == ev.sha(out/"suite/suite.json"), "Wrong suite/status")
    return record


def run(out, gate=None):
    protocol = load(out)
    require(not (out/"execution.json").exists(), "Execution already exists; no automatic retries")
    if protocol["purpose"] != "timing_canary":
        require(gate is not None, "Timing gate required")
        measured = json.loads((gate/"timing-gate.json").read_text())
        require(measured["status"] == "ok" and measured["binary_sha256"] == protocol["binary_sha256"], "Timing gate/build mismatch")
    execution = dict(status="running", jobs=[], failures=[])
    save(out/"execution.json", execution)
    points, timings, hashes = [], {}, {}
    for job in protocol["jobs"]:
        directory = out/job["id"]
        record = dict(id=job["id"], status="running", evaluations={}, checkpoints={})
        execution["jobs"].append(record); save(out/"execution.json", execution)
        try:
            timed = train(out/"binaries"/job["model"], directory, job["receipts"])
            record["training"] = timed
            if job["receipts"]:
                complete = verify_checkpoints(directory, timed, protocol["checkpoint_steps"], PARAMETERS[job["model"]])
            else:
                require(timed["status"] == "ok", "Receipt-off control failed")
                complete = {str(k): dict(sha256=ev.sha(directory/f"checkpoints/connect4cnn/trial/{k:016d}.bin"))
                            for k in protocol["checkpoint_steps"]}
                require(normalized(ini(directory/"metrics/connect4cnn/trial.ini")) == normalized(ini(directory/"config/default.ini")), "Control resolved config differs")
            record["checkpoints"] = complete
            elapsed = (timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9
            timings[job["id"]] = elapsed
            hashes[job["id"]] = {k: v["sha256"] for k, v in complete.items()}
            record["process_seconds"] = elapsed
            record["process_sps"] = protocol["steps"]/elapsed if timed["status"] == "ok" else None
            record["native_avg_sps"] = None
            if timed["status"] == "ok":
                metrics = ini(directory/"metrics/connect4cnn/trial.ini")["metrics"]
                uptime = float(metrics["uptime"].split(",")[-1])
                require(uptime > 0 and int(float(metrics["agent_steps"].split(",")[-1])) == protocol["steps"], "Invalid final native counters")
                record.update(native_uptime_seconds=uptime, native_avg_sps=protocol["steps"]/uptime,
                              native_last_sps=float(metrics["sps"].split(",")[-1]),
                              vram_last_gb=float(metrics["util/vram_used_gb"].split(",")[-1]))
            save(out/"execution.json", execution)
            contention_path = directory/"contention.json"
            if contention_path.exists():
                contention = json.loads(contention_path.read_text())
                if contention["status"] != "uncontended":
                    # Preserve scores and curves but never label affected timing clean.
                    execution["failures"].append(dict(job=job["id"], error="GPU contention or missing monitor telemetry", detail=contention))
            # Evaluate all complete cells, even if the training job later failed.
            selected = list(complete) if protocol["purpose"] != "timing_canary" else ([str(protocol["checkpoint_steps"][-1])] if job["receipts"] else [])
            for raw_step in selected:
                step = int(raw_step)
                try:
                    evaluated = evaluate(out, job, step, directory/"checkpoints/connect4cnn/trial/resolved.ini")
                    record["evaluations"][raw_step] = dict(result_sha256=ev.sha(directory/f"eval-{step:016d}/result.json"),
                        episodes_sha256=evaluated["episodes_sha256"])
                    if protocol["purpose"] != "timing_canary":
                        counts = evaluated["counts"]
                        points.append(dict(model=job["model"], seed=job["seed"], steps=step,
                                           seconds=complete[raw_step]["seconds"], win_rate=counts["wins"]/counts["episodes"],
                                           games=counts["episodes"], parameters=PARAMETERS[job["model"]],
                                           process_sps=record["process_sps"], native_sps=record["native_avg_sps"], train_seconds=elapsed))
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    execution["failures"].append(dict(job=job["id"], steps=step, error=str(error)))
                save(out/"execution.json", execution)
            require(timed["status"] == "ok", "Training failed; recovered cells retained")
            record["status"] = "ok"
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
            record["status"] = "failed"
            execution["failures"].append(dict(job=job["id"], error=str(error)))
        save(out/"execution.json", execution)
    if protocol["purpose"] == "timing_canary":
        ratios = {}
        for model in MODELS:
            pairs = []
            for seed in protocol["training_seeds"]:
                off, on = (f"{model}-s{seed}-receipts-{flag}" for flag in (0, 1))
                require(hashes.get(off) == hashes.get(on) and hashes.get(off), "Receipt setting changed checkpoints")
                pairs.append(timings[on]/timings[off])
            ratios[model] = dict(paired_ratios=pairs, median_ratio=statistics.median(pairs))
        # The extra interrupted fixture uses the same LR schedule and full budget.
        original = protocol["jobs"][1]
        fixture = out/"interrupted"
        shutil.copytree(out/original["id"]/"config", fixture/"config")
        stopped = train(out/"binaries"/original["model"], fixture, interrupt=True)
        recovered = verify_checkpoints(fixture, stopped, protocol["checkpoint_steps"], PARAMETERS[original["model"]])
        require(recovered and len(recovered) < len(protocol["checkpoint_steps"]), "No meaningful partial prefix recovered")
        job = dict(id="interrupted", model=original["model"])
        evaluated = evaluate(out, job, int(next(iter(recovered))), fixture/"checkpoints/connect4cnn/trial/resolved.ini")
        require(evaluated["counts"]["episodes"] == 65, "Recovered checkpoint could not complete exact eval")
        ev.save(out/"interrupted/recovery.json", dict(training_status="failed", complete_checkpoints=recovered,
                                                    evaluated_episodes=65, native_final_metrics_available=False))
        require(not execution["failures"], "Timing canary failures retained")
        # Practical screen only; short noisy ratios are not an overhead CI.
        require(all(r["median_ratio"] <= 1.10 for r in ratios.values()), "Receipt overhead exceeds 10% canary screen; investigate before full run")
        ev.save(out/"timing-gate.json", dict(status="ok", binary_sha256=protocol["binary_sha256"],
            paired_timings=ratios, byte_identical_on_off=True, partial_failure_recovered=True,
            note="Short canary screen, not precise overhead inference; includes startup config receipt."))
    else:
        (out/"analysis").mkdir()
        report(out/"analysis", protocol, points, execution["failures"])
    execution["status"] = "ok" if not execution["failures"] else "completed_with_failures"
    save(out/"execution.json", execution)
    print(out/("timing-gate.json" if protocol["purpose"] == "timing_canary" else "analysis/REPORT.md"))
    require(not execution["failures"], "Campaign incomplete; failed jobs and usable cells preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-canary", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binaries", type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out.resolve(), args.gate.resolve() if args.gate else None)
    else:
        require(args.binaries is not None, "--binaries is required for preparation")
        prepare(args.out.resolve(), args.binaries.resolve(), args.command == "prepare-canary")


if __name__ == "__main__":
    main()
