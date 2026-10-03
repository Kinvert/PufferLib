"""Prepare and audit the matched full-frontier experiment. GPU runs are opt-in.

Configuration/process/report glue only. Learning and exact evaluation are native
C/CUDA. Full confirmation remains blocked on measurement/inference calibration.
"""
import argparse
import configparser
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ("flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn")
RECEIPT = re.compile(r"^PUFFER_CHECKPOINT steps=(\d+) bytes=(\d+) monotonic_ns=(\d+)$", re.M)
EXACT = re.compile(r"^CONNECT4_EXACT_EVAL version=1 requested=(\d+) completed=(\d+) wins=(\d+) params=(\d+)$", re.M)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def ini(*paths):
    c = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    for path in paths:
        with Path(path).open() as handle:
            c.read_file(handle)
    return c


def normalized(config):
    def value(raw):
        raw = raw.strip().strip("\"'")
        try:
            number = float(raw.replace("_", ""))
            if math.isfinite(number):
                # Native INI writes %.17g: compare the parsed double, not its
                # different decimal spellings (e.g. 0.8 vs 0.80000000000000004).
                return number.hex()
        except ValueError:
            pass
        return raw
    return {s: {k: value(v) for k, v in config[s].items()} for s in config.sections() if s != "metrics"}


def common_settings(config):
    result = normalized(config)
    for key in ("seed", "run_id", "checkpoint_dir", "log_dir"):
        result["base"].pop(key, None)
    result["policy"] = {k: v for k, v in result["policy"].items() if k != "encoder" and not k.startswith("cnn_")}
    return result


def snapshots(out):
    paths = [ROOT/"build.sh", ROOT/"config/default.ini", ROOT/"config/connect4cnn.ini"]
    for directory in (ROOT/"src", ROOT/"vendor", HERE):
        paths += [p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [ROOT/"research/connect4_claim_design.json", ROOT/"research/CONNECT4_CLAIM_PROTOCOL.md",
              ROOT/"research/claim_frontier.py", ROOT/"research/claim_frontier.html"]
    result = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT)
        target = out/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        result[str(relative)] = sha(target)
    return result


def prepare(out, canary, seeds):
    require(not out.exists(), "Use a fresh campaign directory; evidence must not be overwritten")
    out.mkdir(parents=True)
    frozen = json.loads((HERE/"confirmation.json").read_text())
    seeds = seeds or ([47001, 47002] if canary else None)
    require(seeds is not None and len(seeds) >= 2 and len(set(seeds)) == len(seeds), "Provide distinct training seeds; confirmation seed count is not yet calibrated")
    require(all(0 <= seed <= 0x7fffffff for seed in seeds), "Training seeds must fit native signed int")
    steps, cadence = (65536, 32768) if canary else (13312000, 262144)
    checkpoint_steps = list(range(cadence, steps+1, cadence))
    if checkpoint_steps[-1] != steps:
        checkpoint_steps.append(steps)
    base = ini(ROOT/"config/default.ini", ROOT/"config/connect4cnn.ini", HERE/"compare.ini")
    for section in list(base.sections()):
        if section.startswith("sweep."):
            base.remove_section(section)
    for key in list(base["policy"]):
        if key == "encoder" or key.startswith("cnn_"):
            del base["policy"][key]
    base["base"].update(eval_episodes="0", checkpoint_interval=str(cadence//2048), load_model_path="None")
    base["train"]["total_timesteps"] = str(steps)
    base["sweep"]["downsample"] = "25"
    jobs = []
    common = None
    for index, seed in enumerate(seeds):
        # Cyclic order balances each position in each complete block of four seeds.
        order = MODELS[index % 4:]+MODELS[:index % 4]
        blocks = [dict(seed=147001+2*index+j, offset=0, episodes=n)
                  for j, n in enumerate((65, 8) if canary else (512, 512))]
        for model in order:
            name = f"{model}-s{seed}"
            job_dir = out/name
            (job_dir/"config").mkdir(parents=True)
            c = copy.deepcopy(base)
            policy = frozen["policies"].get(model, {"encoder": 0})
            c["policy"].update({k: str(v) for k, v in policy.items()})
            c["base"].update(seed=str(seed), run_id="trial", checkpoint_dir="checkpoints", log_dir="metrics")
            with (job_dir/"config/default.ini").open("x") as handle:
                c.write(handle)
            (job_dir/"config/connect4cnn.ini").write_text("# All settings frozen in default.ini.\n")
            if common is None:
                common = common_settings(c)
            require(common_settings(c) == common, "Prepared learner settings differ")
            jobs.append(dict(id=name, model=model, seed=seed, parameters=frozen["parameters"][model],
                             config_sha256=sha(job_dir/"config/default.ini"),
                             env_config_sha256=sha(job_dir/"config/connect4cnn.ini"), evaluation=blocks))
    protocol = dict(version="connect4-exact-frontier-v1", purpose="measurement_canary" if canary else "confirmation_draft",
                    status="prepared_not_executed", confirmation_launch_allowed=False,
                    gates_pending=["GPU exact evaluator/recurrent reset and repeatability", "checkpoint receipt/cadence overhead",
                                   "baseline backend audit", "simultaneous inference calibration and replication freeze"],
                    models=list(MODELS), training_seeds=seeds, steps=steps, checkpoint_steps=checkpoint_steps,
                    evaluation_slots=64, precision="float32", source_revision=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256=snapshots(out), common_settings=common, jobs=jobs,
                    timing="CLOCK_MONOTONIC launch to native post-rename checkpoint receipt",
                    episode_allocation="fixed waves; per-episode Philox and environment RNG; completed slots never step again",
                    analysis="full checkpoint means and seed curves; candidate simultaneous bands remain exploratory")
    save(out/"protocol.json", protocol, exclusive=True)
    (out/"protocol.sha256").write_text(sha(out/"protocol.json")+"\n")
    (out/"NEXT.md").write_text(
        "# Prepared, not executed\n\nNo GPU work occurred during preparation.\n\n"
        "Read research/CLAIM_PIPELINE.md from the source checkout. Full confirmation is blocked.\n"
        "The opt-in run-canary command records exact commands, failures and all checkpoint curves.\n")
    return protocol


def load(out, current_source=False):
    require(sha(out/"protocol.json") == (out/"protocol.sha256").read_text().strip(), "Manifest changed after preparation")
    p = json.loads((out/"protocol.json").read_text())
    require(p["version"] == "connect4-exact-frontier-v1", "Unknown protocol")
    for name, digest in p["source_sha256"].items():
        require(sha(out/"source"/name) == digest, f"Source snapshot mismatch: {name}")
        if current_source:
            require(sha(ROOT/name) == digest, f"Current source changed: {name}; prepare a fresh campaign")
    for job in p["jobs"]:
        require(sha(out/job["id"]/"config/default.ini") == job["config_sha256"], "Frozen config changed")
        require(sha(out/job["id"]/"config/connect4cnn.ini") == job["env_config_sha256"], "Environment config changed")
    return p


def process(command, cwd, log, timeout, environment=None):
    require(not log.exists(), f"Refusing to overwrite {log}")
    result = dict(command=list(map(str, command)), cwd=str(cwd), status="starting", timeout=timeout,
                  environment={key: (environment or os.environ).get(key) for key in
                               ("PUFFER_CHECKPOINT_RECEIPTS", "NVCC_ARCH", "NVCC_EXTRA", "NVCC_APPEND_FLAGS", "CUDA_HOME", "NCCL_ROOT")})
    metadata = log.with_suffix(log.suffix+".json")
    save(metadata, result, exclusive=True)
    child = None
    with log.open("x") as handle:
        try:
            started = time.monotonic_ns()
            result["launch_monotonic_ns"] = started
            child = subprocess.Popen(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment)
            result.update(pid=child.pid, status="running")
            save(metadata, result)
            code = child.wait(timeout=timeout)
            result.update(returncode=code, status="ok" if code == 0 else "failed")
            require(code == 0, f"Process failed: {log}")
        except BaseException as error:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            result.update(status="failed", error=str(error), returncode=child.returncode if child else None)
            raise
        finally:
            result["end_monotonic_ns"] = time.monotonic_ns()
            save(metadata, result)
    return result


def build(out):
    p = load(out, current_source=True)
    require(not (out/"builds.json").exists(), "Builds already recorded; preserve them")
    require(not any(os.environ.get(k) for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_EXTRA")), "Unset compiler overrides")
    (out/"binaries").mkdir(exist_ok=False)
    process([str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))/"bin/nvcc"), "--version"],
            ROOT, out/"compiler.log", 30)
    records = {}
    for model in p["models"]:
        env = os.environ.copy()
        if model != "flex_quality":
            env["NVCC_EXTRA"] = "-DC4_"+model.upper()
        binary = out/"binaries"/model
        process(["bash", "build.sh", "connect4cnn", str(binary), "--float"], ROOT,
                out/f"build-{model}.log", 600, env)
        records[model] = sha(binary)
    load(out, current_source=True)
    save(out/"builds.json", records, exclusive=True)


def idle_gpu(out, name):
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    hardware = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20)
    require("5090" in hardware and len(hardware.strip().splitlines()) == 1, "This canary is reserved for the separate single-GPU 5090 host")
    active = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    require(not active.strip(), "GPU busy; no process was interrupted")
    (out/name).write_text(hardware)


def parse_checkpoints(log, process_record, expected_steps, parameters):
    require(process_record["status"] == "ok", "Training process did not complete")
    matches = RECEIPT.findall(log)
    require([int(row[0]) for row in matches] == expected_steps, "Missing, duplicate or unexpected checkpoint receipt")
    result = {}
    previous = process_record["launch_monotonic_ns"]
    for step, size, timestamp in matches:
        step, size, timestamp = int(step), int(size), int(timestamp)
        require(size == parameters*4, "Checkpoint parameter count differs")
        require(previous < timestamp <= process_record["end_monotonic_ns"], "Invalid checkpoint completion time")
        result[step] = (timestamp-process_record["launch_monotonic_ns"])/1e9
        previous = timestamp
    return result


def evaluation_command(out, job, step, index, block, slots):
    directory = out/job["id"]
    checkpoint = directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"
    stem = directory/f"eval-{step:016d}-b{index}"
    return [str(out/"binaries"/job["model"]), "eval_exact", "--headless", f"--base.load_model_path={checkpoint}",
            f"--eval_exact.seed={block['seed']}", f"--eval_exact.episodes={block['episodes']}",
            f"--eval_exact.episode_offset={block['offset']}", f"--eval_exact.slots={slots}",
            f"--eval_exact.output={stem}.csv"]


def repeat_command(out, job, step, label, slots):
    command = evaluation_command(out, job, step, 0, job["evaluation"][0], slots)
    command[-1] = f"--eval_exact.output={out/job['id']}/eval-{step:016d}-{label}.csv"
    return command+{"repeat": [], "eager": ["--base.cudagraphs=-1"], "worker1": ["--vec.num_threads=1"]}[label]


def run_canary(out, allowed):
    require(allowed, "GPU execution requires --allow-gpu; preparation never launches jobs")
    p = load(out, current_source=True)
    require(p["purpose"] == "measurement_canary", "Full confirmation is blocked on the documented gates")
    require(not (out/"execution.json").exists(), "An execution exists; do not silently retry or erase failed jobs")
    binaries = json.loads((out/"builds.json").read_text())
    for model, digest in binaries.items():
        require(sha(out/"binaries"/model) == digest, "Binary changed since build")
    idle_gpu(out, "gpu-before.txt")
    execution = dict(status="running", jobs=[], failures=[])
    save(out/"execution.json", execution, exclusive=True)
    record = None
    try:
        for job in p["jobs"]:
            directory = out/job["id"]
            idle_gpu(directory, "gpu-before.txt")
            binary = out/"binaries"/job["model"]
            record = dict(id=job["id"], status="running", checkpoints={}, evaluations={}, repeats={})
            execution["jobs"].append(record)
            save(out/"execution.json", execution)
            env = {**os.environ, "PUFFER_CHECKPOINT_RECEIPTS": "1"}
            timed = process([str(binary), "train", "--headless"], directory, directory/"train.log", 2400, env)
            parse_checkpoints((directory/"train.log").read_text(), timed, p["checkpoint_steps"], job["parameters"])
            expected = ini(directory/"config/default.ini")
            resolved = ini(directory/"metrics/connect4cnn/trial.ini")
            require(normalized(expected) == normalized(resolved), "Executed config differs from frozen job")
            for step in p["checkpoint_steps"]:
                checkpoint = directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"
                values = np.fromfile(checkpoint, dtype=np.float32)
                require(values.size == job["parameters"] and np.isfinite(values).all(), "Invalid checkpoint")
                record["checkpoints"][str(step)] = sha(checkpoint)
                save(out/"execution.json", execution)
                for index, block in enumerate(job["evaluation"]):
                    stem = directory/f"eval-{step:016d}-b{index}"
                    command = evaluation_command(out, job, step, index, block, p["evaluation_slots"])
                    process(command, directory, Path(str(stem)+".log"), 300)
                    record["evaluations"][stem.name] = {suffix: sha(Path(str(stem)+suffix)) for suffix in (".csv", ".log", ".log.json")}
                    save(out/"execution.json", execution)
            # Fixed checkpoint/episodes, separate repeat and eager-worker checks.
            step, block = p["checkpoint_steps"][-1], job["evaluation"][0]
            reference_csv = directory/f"eval-{step:016d}-b0.csv"
            for label in ("repeat", "eager", "worker1"):
                stem = directory/f"eval-{step:016d}-{label}"
                command = repeat_command(out, job, step, label, p["evaluation_slots"])
                process(command, directory, Path(str(stem)+".log"), 300)
                record["repeats"][label] = {suffix: sha(Path(str(stem)+suffix)) for suffix in (".csv", ".log", ".log.json")}
                save(out/"execution.json", execution)
                require(sha(reference_csv) == record["repeats"][label][".csv"], f"Evaluation {label} differs at fixed episode IDs")
            record["status"] = "ok"
            save(out/"execution.json", execution)
            record = None
        execution["status"] = "ok"
    except BaseException as error:
        execution["status"] = "failed"
        execution["failures"].append(str(error))
        if record is not None:
            record["status"] = "failed"
        raise
    finally:
        save(out/"execution.json", execution)


def episode_seeds(seed, episode):
    def mix(x):
        mask = (1 << 64)-1
        x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
        x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
        return x ^ (x >> 31)
    identity = seed << 32 | episode
    return mix(identity ^ 0x656e7669726f7631) & 0xffffffff, mix(identity ^ 0x706f6c6963797631)


def episodes(path, block, slots, representation=0):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == block["episodes"], "Wrong episode quota")
    seen, wins, score = set(), 0, 0
    for row in rows:
        episode = int(row["episode_id"])
        require(episode not in seen, "Duplicate episode ID")
        seen.add(episode)
        require(int(row["version"]) == 1 and int(row["block_seed"]) == block["seed"], "Wrong evaluation block")
        require(int(row["slot"]) == (episode-block["offset"]) % slots, "Wrong slot assignment")
        require(int(row["representation"]) == representation, "Wrong representation")
        require((int(row["env_seed"]), int(row["policy_seed"])) == episode_seeds(block["seed"], episode), "Episode RNG mismatch")
        require(1 <= int(row["decisions"]) <= 21, "Invalid episode length")
        reward, win, invalid = int(row["score"]), int(row["win"]), int(row["invalid"])
        require(reward in (-1, 0, 1) and win == int(reward == 1) and invalid in (0, 1), "Invalid outcome")
        require(not invalid or reward == -1, "Invalid action must lose")
        require(re.fullmatch(r"[0-9a-f]{16}", row["action_hash"]) is not None, "Missing action trace hash")
        wins += win; score += reward
    require(seen == set(range(block["offset"], block["offset"]+block["episodes"])), "Missing or unexpected episode IDs")
    return dict(wins=wins, score=score, episodes=len(rows))


def audit(out):
    p = load(out)
    execution = json.loads((out/"execution.json").read_text())
    binaries = json.loads((out/"builds.json").read_text())
    by_id = {j["id"]: j for j in execution["jobs"]}
    require(len(by_id) == len(execution["jobs"]), "Duplicate job record")
    points, failures = [], list(execution.get("failures", []))
    if execution["status"] != "ok" and not failures:
        failures.append("Campaign has no successful completion receipt")
    for job in p["jobs"]:
        directory = out/job["id"]
        try:
            require(job["id"] in by_id, "Job never started")
            if by_id[job["id"]]["status"] != "ok":
                failures.append(dict(job=job["id"], error="Job incomplete; retaining independently complete checkpoint cells"))
            require(sha(out/"binaries"/job["model"]) == binaries[job["model"]], "Binary receipt mismatch")
            resolved = ini(directory/"metrics/connect4cnn/trial.ini")
            require(normalized(resolved) == normalized(ini(directory/"config/default.ini")), "Executed job config changed")
            require(common_settings(resolved) == p["common_settings"], "Unmatched learner settings")
            timed = json.loads((directory/"train.log.json").read_text())
            require(timed["command"] == [str(out/"binaries"/job["model"]), "train", "--headless"]
                    and timed["cwd"] == str(directory), "Wrong training command/config directory")
            require(timed["environment"]["PUFFER_CHECKPOINT_RECEIPTS"] == "1", "Checkpoint receipts were not requested")
            times = parse_checkpoints((directory/"train.log").read_text(), timed, p["checkpoint_steps"], job["parameters"])
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(job=job["id"], error=str(error)))
            continue
        duration = (timed["end_monotonic_ns"]-timed["launch_monotonic_ns"])/1e9
        for label in ("repeat", "eager", "worker1"):
            try:
                step = p["checkpoint_steps"][-1]
                stem = directory/f"eval-{step:016d}-{label}"
                hashes = by_id[job["id"]]["repeats"][label]
                for suffix in (".csv", ".log", ".log.json"):
                    require(sha(Path(str(stem)+suffix)) == hashes[suffix], "Repeat evidence changed")
                repeated = json.loads(Path(str(stem)+".log.json").read_text())
                require(repeated["status"] == "ok", "Repeat process failed")
                require(repeated["command"] == repeat_command(out, job, step, label, p["evaluation_slots"])
                        and repeated["cwd"] == str(directory), "Repeat check ran the wrong command")
                require(hashes[".csv"] == sha(directory/f"eval-{step:016d}-b0.csv"), "Repeat/eager/worker outcomes differ")
            except (ValueError, OSError, KeyError) as error:
                failures.append(dict(job=job["id"], check=label, error=str(error)))
        try:
            metrics = resolved["metrics"]
            native_uptime = float(metrics["uptime"].split(",")[-1])
            require(math.isfinite(native_uptime) and native_uptime > 0 and int(float(metrics["agent_steps"].split(",")[-1])) == p["steps"], "Invalid native metrics")
        except (ValueError, KeyError) as error:
            failures.append(dict(job=job["id"], error=str(error)))
            continue
        for step in p["checkpoint_steps"]:
            try:
                checkpoint = directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"
                require(sha(checkpoint) == by_id[job["id"]]["checkpoints"][str(step)], "Checkpoint hash mismatch")
                weights = np.fromfile(checkpoint, dtype=np.float32)
                require(weights.size == job["parameters"] and np.isfinite(weights).all(), "Checkpoint contents invalid")
                totals = dict(wins=0, score=0, episodes=0)
                for index, block in enumerate(job["evaluation"]):
                    stem = directory/f"eval-{step:016d}-b{index}"
                    hashes = by_id[job["id"]]["evaluations"][stem.name]
                    for suffix in (".csv", ".log", ".log.json"):
                        require(sha(Path(str(stem)+suffix)) == hashes[suffix], "Evaluation evidence changed")
                    record = json.loads(Path(str(stem)+".log.json").read_text())
                    require(record["status"] == "ok", "Evaluation process failed")
                    require(record["command"] == evaluation_command(out, job, step, index, block, p["evaluation_slots"])
                            and record["cwd"] == str(directory), "Wrong evaluation checkpoint/command/config directory")
                    result = episodes(Path(str(stem)+".csv"), block, p["evaluation_slots"])
                    summary = EXACT.findall(Path(str(stem)+".log").read_text())
                    require(summary == [(str(block["episodes"]), str(block["episodes"]), str(result["wins"]), str(job["parameters"]))], "Evaluation completion receipt mismatch")
                    for key in totals:
                        totals[key] += result[key]
                points.append(dict(model=job["model"], seed=job["seed"], steps=step, seconds=times[step],
                                   win_rate=totals["wins"]/totals["episodes"], games=totals["episodes"],
                                   parameters=job["parameters"], process_sps=p["steps"]/duration,
                                   native_sps=p["steps"]/native_uptime, train_seconds=duration))
            except (ValueError, OSError, KeyError) as error:
                failures.append(dict(job=job["id"], steps=step, error=str(error)))
    output = Path(tempfile.mkdtemp(prefix="analysis.", dir=out))
    costs = {}
    for kind, paths in (("training", out.glob("*/train.log.json")),
                        ("evaluation_including_repeat_checks", out.glob("*/eval-*.log.json")),
                        ("build", out.glob("build-*.log.json"))):
        total, finished, incomplete = 0., 0, 0
        for path in paths:
            record = json.loads(path.read_text())
            if "launch_monotonic_ns" in record and "end_monotonic_ns" in record:
                total += (record["end_monotonic_ns"]-record["launch_monotonic_ns"])/1e9
                finished += 1
            else:
                incomplete += 1
        costs[kind] = dict(seconds=total, timed_processes=finished, missing_timing=incomplete)
    sys.path.insert(0, str(ROOT/"research"))
    from claim_frontier import report
    report(output, p, points, failures, costs)
    save(output/"analysis-source.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                        "sources": {name: sha(ROOT/name) for name in
                                                    ("ocean/connect4cnn/claim.py", "research/claim_frontier.py", "research/claim_frontier.html")}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "run-canary", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        p = prepare(out, args.canary, args.seeds)
        print(json.dumps(dict(directory=str(out), status=p["status"], jobs=len(p["jobs"]), checkpoints=len(p["checkpoint_steps"]))))
    elif args.action == "build":
        build(out)
    elif args.action == "run-canary":
        run_canary(out, args.allow_gpu)
    else:
        print(audit(out))


if __name__ == "__main__":
    main()
