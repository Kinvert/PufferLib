#!/usr/bin/env python3
"""Maze suites, receipt audits and supervised native checkpoint evaluation.

suite, audit and run --prepare-only execute no policy. run requires an explicitly
scheduled GPU window; inference stays native C/CUDA, never Python or CPU.
"""
import argparse
import configparser
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from claim import episode_seeds, process, require
from deterministic_eval import FAMILIES, load_config, save

TOOL_FILES = {"ocean/mazecnn/exact_eval.py": Path(__file__).resolve(),
              "ocean/connect4cnn/claim.py": ROOT / "ocean/connect4cnn/claim.py",
              "ocean/connect4cnn/deterministic_eval.py": ROOT / "ocean/connect4cnn/deterministic_eval.py"}

RULES = "maze-native-v1"
SEMANTICS = "first-goal-or-native-area-timeout-v1"
LEVELS = "seed-rotated-cyclic-v1"
ESTIMAND = "success-fraction-all-assigned-episodes-v1"
MANIFEST_FIELDS = ("episode_id", "env_seed", "policy_seed", "representation", "level_id", "width",
                   "height", "horizon", "start_rng", "terminal_rng", "start_world_hash", "start_observation_hash")
RESULT_FIELDS = ("version", "block_seed", "episode_id", "slot", "representation", "env_seed", "policy_seed",
                 "level_id", "width", "height", "horizon", "decisions", "success", "episode_return", "end_reason",
                 "native_log_entries", "end_rng", "action_hash", "start_observation_hash", "start_world_hash", "start_rng")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(path), "Need full resolved INI")
    return config


def mix(value):
    mask = 2**64 - 1
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


def level_id(suite, episode):
    count = suite["level_count"]
    rotation = mix(suite["seed"] ^ 0x6d617a656c657631) % count
    return suite["level_offset"] + (episode + rotation) % count


def world(config):
    maps, size = (config.getfloat("env", key) for key in ("num_maps", "map_size"))
    require(math.isfinite(maps) and maps.is_integer() and 1 <= maps <= 8192, "Invalid table size")
    require(math.isfinite(size) and size.is_integer() and (size == -1 or 5 <= size <= 47), "Invalid map size")
    return dict(num_maps=int(maps), map_size=int(size))


def validate_suite(suite):
    require(suite["protocol"] == "maze-fixed-suite-v1" and suite["rules"] == RULES
            and suite["evaluation"] == SEMANTICS and suite["level_assignment"] == LEVELS
            and suite["estimand"] == ESTIMAND, "Wrong protocol/rules/estimand")
    require(suite["environment"] in ("maze", "mazecnn") and suite["sampling"] == "native-philox-categorical-v1"
            and suite["purpose"] in ("development", "heldout"), "Wrong task/sampling/purpose")
    for key, low, high in (("seed", 0, 2**32-1), ("offset", 0, 2**32-1), ("episodes", 1, 1000000),
                           ("slots", 1, 1024), ("representation", 0, 5), ("level_offset", 0, 8191),
                           ("level_count", 1, 8192), ("training_level_count", 0, 8192)):
        require(type(suite[key]) is int and low <= suite[key] <= high, f"Invalid {key}")
    require(suite["environment"] != "maze" or suite["representation"] == 0, "State has no appearance")
    require(suite["offset"] + suite["episodes"] + suite["slots"] <= 2**32-1, "Episode IDs overflow")
    config = configparser.ConfigParser(); config["env"] = {k: str(v) for k, v in suite["world"].items()}
    require(world(config) == suite["world"], "Incomplete/changed world")
    require(suite["level_offset"] + suite["level_count"] <= suite["world"]["num_maps"], "Panel exceeds table")
    require(suite["training_level_count"] <= suite["world"]["num_maps"], "Invalid training prefix")
    if suite["purpose"] == "heldout":
        require(0 < suite["training_level_count"] <= suite["level_offset"], "Heldout panel overlaps declared training prefix")


def info(binary):
    identity = json.loads(subprocess.check_output([str(binary), "eval_exact_info"], text=True, timeout=30))
    require(identity["environment"] in ("maze", "mazecnn") and identity["rules"] == RULES
            and identity["evaluation"] == SEMANTICS and identity["levels"] == LEVELS
            and identity["float32"] and identity["receipt_version"] == 1, "Need current float32 Maze exact build")
    return identity


def validate_manifest(path, suite):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, "Wrong manifest columns")
        rows = list(reader)
    require(len(rows) == suite["episodes"], "Wrong manifest quota")
    for i, row in enumerate(rows):
        episode = suite["offset"] + i; env_seed, policy_seed = episode_seeds(suite["seed"], episode)
        require(int(row["episode_id"]) == episode and int(row["env_seed"]) == env_seed
                and int(row["policy_seed"]) == policy_seed, "Wrong episode identity/seeds")
        require(int(row["representation"]) == suite["representation"] and int(row["start_rng"]) == env_seed,
                "Wrong appearance/initial RNG")
        require(int(row["level_id"]) == level_id(suite, episode), "Wrong cyclic level identity")
        width, height, horizon = (int(row[k]) for k in ("width", "height", "horizon"))
        size = suite["world"]["map_size"]
        require(width == height and 5 <= width <= 47 and width % 2 == 1 and horizon == 2 * width * height,
                "Invalid dimensions/horizon")
        require((size == -1 and width <= 45) or width == size - (size % 2 == 0), "Wrong fixed/generated size")
        require(0 <= int(row["terminal_rng"]) <= 2**32-1, "Invalid terminal RNG")
        for key in ("start_world_hash", "start_observation_hash"):
            require(re.fullmatch(r"[0-9a-f]{16}", row[key]) is not None, "Malformed starting hash")
    return rows


def suite_config(config, suite):
    copy = configparser.ConfigParser(interpolation=None)
    copy.read_dict({s: dict(config[s]) for s in config.sections()})
    copy["env"].update(representation=str(suite["representation"]), representation_mode="0", representation_seed="0")
    copy["eval_exact"] = dict(seed=str(suite["seed"]), episodes=str(suite["episodes"]),
                              episode_offset=str(suite["offset"]), slots=str(suite["slots"]),
                              level_offset=str(suite["level_offset"]), level_count=str(suite["level_count"]))
    return copy


def create_suite(args):
    binary, source, out = args.binary.resolve(), args.config.resolve(), args.out.resolve()
    identity, config = info(binary), read(source)
    require(config.get("base", "env_name") == identity["environment"], "Binary/config mismatch")
    settings = world(config)
    suite = dict(protocol="maze-fixed-suite-v1", rules=RULES, evaluation=SEMANTICS, level_assignment=LEVELS,
                 estimand=ESTIMAND, environment=identity["environment"], seed=args.seed, offset=args.offset,
                 episodes=args.episodes, slots=args.slots, representation=args.representation,
                 level_offset=args.level_offset, level_count=args.level_count, training_level_count=args.training_level_count,
                 purpose=args.purpose, sampling="native-philox-categorical-v1", world=settings)
    validate_suite(suite)
    out.mkdir(parents=True, exist_ok=False); binary_hash, config_hash = sha(binary), sha(source)
    shutil.copyfile(source, out / "source.ini")
    with (out / "manifest.ini").open("x") as stream: suite_config(config, suite).write(stream)
    with (out / "episodes.csv").open("x") as stream:
        subprocess.run([str(binary), "eval_exact_manifest", str(out / "manifest.ini")], stdout=stream,
                       stderr=subprocess.PIPE, text=True, check=True, timeout=120)
    validate_manifest(out / "episodes.csv", suite)
    require(sha(binary) == binary_hash and sha(source) == config_hash, "Inputs changed during preparation")
    suite.update(episodes_sha256=sha(out / "episodes.csv"), source_ini_sha256=config_hash,
                 manifest_ini_sha256=sha(out / "manifest.ini"), binary_sha256=binary_hash, binary_info=identity,
                 evidence_status="host-level-starts-only-gpu-runtime-pending",
                 level_exclusion_status="declared-prefix-not-verified-against-training-artifacts")
    (out / "suite.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(out / "suite.json")


def load_suite(path):
    suite = json.loads(path.read_text()); validate_suite(suite)
    for name, key in (("episodes.csv", "episodes_sha256"), ("source.ini", "source_ini_sha256"),
                      ("manifest.ini", "manifest_ini_sha256")):
        require(sha(path.parent / name) == suite[key], f"Changed {name}")
    source = read(path.parent / "source.ini")
    require(source.get("base", "env_name") == suite["environment"] and world(source) == suite["world"], "Source/suite world differs")
    expected, actual = suite_config(source, suite), read(path.parent / "manifest.ini")
    require({s: dict(expected[s]) for s in expected.sections()} == {s: dict(actual[s]) for s in actual.sections()},
            "Suite/manifest configuration differs")
    return suite, validate_manifest(path.parent / "episodes.csv", suite)


def audit_csv(path, suite, manifest):
    expected = {int(row["episode_id"]): row for row in manifest}
    require(len(expected) == suite["episodes"], "Wrong manifest identities")
    seen, successes, timeouts, decisions_total, duplicate_logs = set(), 0, 0, 0, 0
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == RESULT_FIELDS, "Wrong result columns")
        for row in reader:
            episode = int(row["episode_id"])
            require(episode in expected and episode not in seen, "Extra/duplicate episode")
            seen.add(episode); start = expected[episode]
            require(int(row["version"]) == 1 and int(row["block_seed"]) == suite["seed"], "Wrong version/seed")
            require(int(row["slot"]) == (episode - suite["offset"]) % suite["slots"], "Wrong slot")
            for key in MANIFEST_FIELDS:
                if key != "terminal_rng": require(row[key] == start[key], f"Wrong start/identity: {key}")
            decisions, success, reason = (int(row[k]) for k in ("decisions", "success", "end_reason"))
            horizon = int(start["horizon"])
            require(1 <= decisions <= horizon and success in (0, 1) and reason == (1 if success else 2), "Invalid ending")
            require(success or decisions == horizon, "Timeout before native horizon")
            require(not success or decisions >= 2 * (int(start["width"]) - 3), "Success before minimum goal distance")
            reward = float.fromhex(row["episode_return"])
            require(math.isfinite(reward) and reward == success, "Wrong success/return")
            logs = 1 + (success and decisions == horizon)
            require(int(row["native_log_entries"]) == logs, "Wrong duplicate-log accounting")
            require(row["end_rng"] == start["terminal_rng"], "Wrong terminal reset RNG")
            require(re.fullmatch(r"[0-9a-f]{16}", row["action_hash"]) is not None, "Malformed action hash")
            successes += success; timeouts += not success; decisions_total += decisions; duplicate_logs += logs - 1
    require(seen == set(expected) and len(seen) == suite["episodes"], "Incomplete assigned episode set")
    n = len(seen)
    return dict(episodes=n, successes=successes, timeouts=timeouts, success_fraction=successes/n,
                mean_return=successes/n, mean_decisions=decisions_total/n, duplicate_native_log_entries=duplicate_logs,
                estimand=ESTIMAND, policy_runtime_certified=False, heldout_exclusion_certified=False)


def checkpoint_config(path, identity, family):
    if identity["environment"] != "maze":
        return load_config(path, identity, family)
    config = read(path)
    require(family == "state" and config.get("base", "env_name") == "maze", "State binary/config/family mismatch")
    require(all(s in config for s in ("base", "vec", "selfplay", "env", "policy", "train")), "Need full resolved INI")
    require(config.getint("policy", "hidden_size") > 0 and config.getint("policy", "num_layers") > 0, "Invalid state core")
    return config


def evaluation_world(config, suite):
    """Only a declared held-out prefix may expand to the frozen evaluation table.

    The original table generator is prefix-stable. This configuration check does
    not prove checkpoint training provenance or absence of tuning exposure.
    """
    training = world(config)
    require(training["map_size"] == suite["world"]["map_size"], "Training/suite map generation differs")
    if suite["purpose"] == "heldout":
        require(training["num_maps"] == suite["training_level_count"], "Training table differs from declared heldout prefix")
    else:
        require(training == suite["world"], "Training/suite game settings differ")
    copy = suite_config(config, suite)
    copy["env"].update({key: str(value) for key, value in suite["world"].items()})
    return copy, training


def run(args):
    binary, training, checkpoint = args.binary.resolve(), args.config.resolve(), args.checkpoint.resolve()
    suite_path, out = args.suite.resolve(), args.out.resolve()
    suite, manifest = load_suite(suite_path); identity = info(binary)
    require(identity["environment"] == suite["environment"], "Suite/binary observation type mismatch")
    original = checkpoint_config(training, identity, args.family)
    config, training_world = evaluation_world(original, suite)
    require(args.timeout > 0 and checkpoint.is_file(), "Need weights file/positive timeout")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="pending", suite_sha256=sha(suite_path), manifest_sha256=suite["episodes_sha256"],
                  checkpoint_sha256=sha(checkpoint), binary_sha256=sha(binary), training_ini_sha256=sha(training),
                  family=args.family, binary_info=identity, policy=dict(original["policy"]),
                  training_appearance={k: original.get("env", k, fallback="0") for k in
                                       ("representation", "representation_mode", "representation_seed")},
                  evaluation_representation=suite["representation"], rules=RULES, evaluation=SEMANTICS,
                  estimand=ESTIMAND, training_world=training_world, evaluation_world=suite["world"],
                  level_panel=dict(offset=suite["level_offset"], count=suite["level_count"]),
                  binary=str(binary), checkpoint=str(checkpoint), config=str(training), suite=str(suite_path),
                  cwd=str(out), prepare_only=args.prepare_only, gpu_runtime_qualified=False,
                  heldout_exclusion_certified=False,
                  note="GPU acceptance remains pending. A matching prefix INI is a declaration, not verified "
                       "checkpoint training/selection provenance. Compiled native provenance needs build receipts.")
    save(out / "request.json", record)
    try:
        require(checkpoint.stat().st_size % 4 == 0, "Malformed float32 checkpoint length")
        weights = np.fromfile(checkpoint, dtype=np.float32)
        require(weights.size > 0 and np.isfinite(weights).all(), "Nonfinite/empty checkpoint")
        del weights
        shutil.copyfile(training, out / "training.ini")
        shutil.copytree(suite_path.parent, out / "suite")
        tooling = {}
        for name, source in TOOL_FILES.items():
            target = out / "tooling" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target); tooling[name] = sha(target)
            require(sha(source) == tooling[name], "Tooling changed while snapshotting")
        record["tooling_sha256"] = tooling
        with (out / "spawn-check.ini").open("x") as stream: config.write(stream)
        with (out / "spawn-check.csv").open("x") as stream:
            subprocess.run([str(binary), "eval_exact_manifest", str(out / "spawn-check.ini")], stdout=stream,
                           stderr=subprocess.PIPE, text=True, check=True, timeout=120)
        require((out / "spawn-check.csv").read_bytes() == (out / "suite/episodes.csv").read_bytes(),
                "Binary generates different starts from frozen manifest")
        record.update(spawn_ini_sha256=sha(out / "spawn-check.ini"), spawn_csv_sha256=sha(out / "spawn-check.csv"))
        config["base"].update(load_model_path=str(checkpoint), eval_agents=str(suite["slots"]),
                              run_id="evaluation", checkpoint_dir=str(out / "unused-checkpoints"),
                              log_dir=str(out / "metrics"), seed="0", cudagraphs="-1" if args.eager else "1",
                              **{"async": "0"})
        config["vec"].update(total_agents=str(suite["slots"]), num_buffers="1", num_threads="1",
                             num_policies="1", hist_policy_percent="0")
        config["selfplay"]["enabled"] = "0"
        config["train"].update(gpus="1", horizon="1", minibatch_size=str(suite["slots"]), replay_ratio="1", verb_eps="0")
        config["eval_exact"]["output"] = str(out / "episodes.csv")
        (out / "config").mkdir()
        with (out / "config/default.ini").open("x") as stream: config.write(stream)
        (out / "config" / f"{identity['environment']}.ini").touch()
        record.update(effective_ini_sha256=sha(out / "config/default.ini"),
                      effective_env_ini_sha256=sha(out / "config" / f"{identity['environment']}.ini"),
                      command=[str(binary), "eval_exact", "--headless"])
        if not args.prepare_only:
            smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
            (ROOT / "build/connect4cnn").mkdir(parents=True, exist_ok=True)
            with (ROOT / "build/connect4cnn/hardware-benchmark.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
                require(not busy.strip(), "GPU busy; no processes interrupted")
                hardware = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20)
                require(bool(hardware.strip()), "GPU hardware query returned no devices")
                (out / "gpu.txt").write_text(hardware)
                try:
                    timed = process(record["command"], out, out / "native.log", args.timeout)
                finally:
                    metadata = out / "native.log.json"
                    if metadata.exists():
                        timing = json.loads(metadata.read_text())
                        if "launch_monotonic_ns" in timing:
                            start, end = timing["launch_monotonic_ns"], timing["end_monotonic_ns"]
                            require(type(start) is int and type(end) is int and end >= start, "Invalid process clock receipt")
                            record["evaluation_process_seconds"] = (end-start)/1e9
                require(timed["returncode"] == 0, "Native evaluation failed; preserve partial receipts/native.log")
                require("evaluation_process_seconds" in record, "Missing native process clock receipt")
        for key, path in (("checkpoint_sha256", checkpoint), ("binary_sha256", binary),
                          ("training_ini_sha256", training), ("suite_sha256", suite_path),
                          ("effective_ini_sha256", out / "config/default.ini"),
                          ("effective_env_ini_sha256", out / "config" / f"{identity['environment']}.ini"),
                          ("training_ini_sha256", out / "training.ini"), ("suite_sha256", out / "suite/suite.json"),
                          ("spawn_ini_sha256", out / "spawn-check.ini"), ("spawn_csv_sha256", out / "spawn-check.csv")):
            require(sha(path) == record[key], f"Input changed: {path}")
        require(sha(suite_path.parent / "episodes.csv") == record["manifest_sha256"], "Manifest changed during evaluation")
        require(sha(out / "suite/episodes.csv") == record["manifest_sha256"], "Copied manifest changed during evaluation")
        load_suite(out / "suite/suite.json")
        for name, source in TOOL_FILES.items():
            require(sha(source) == tooling[name] and sha(out / "tooling" / name) == tooling[name], "Tooling changed during evaluation")
        if args.prepare_only:
            record["status"] = "prepared-not-executed"
        else:
            counts = audit_csv(out / "episodes.csv", suite, manifest)
            summary = re.findall(r"MAZE_EXACT_EVAL version=1 requested=(\d+) completed=(\d+) successes=(\d+) timeouts=(\d+) params=(\d+)",
                                 (out / "native.log").read_text())
            require(len(summary) == 1, "Missing/duplicate completion receipt")
            requested, completed, successes, timeouts, parameters = map(int, summary[0])
            require(requested == completed == counts["episodes"] and successes == counts["successes"]
                    and timeouts == counts["timeouts"] and parameters*4 == checkpoint.stat().st_size,
                    "Summary/CSV/weights mismatch")
            record.update(status="ok", counts=counts, parameters=parameters, episodes_sha256=sha(out / "episodes.csv"))
            (out / "REPORT.md").write_text(
                f"# Deterministic Maze evaluation\n\n{args.family}: success {counts['success_fraction']:.6f} "
                f"across {completed} exactly assigned episodes; {successes} goals, {timeouts} native area timeouts. "
                f"Mean decisions {counts['mean_decisions']:.6f}; {counts['duplicate_native_log_entries']} "
                "extra native log entries retained without double-counting successes.\n\n"
                "Every assigned episode contributes once. Training graph/core shapes are preserved. "
                "Evaluation process time isn't training time. This report doesn't certify GPU behavior, "
                "held-out level exclusion, frontier dominance or SOTA.\n")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(out / "result.json", record)
    print(out / ("result.json" if args.prepare_only else "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("suite", help="GPU-free native level/start manifests")
    for name in ("binary", "config", "out"): create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--episodes", type=int, default=1000); create.add_argument("--offset", type=int, default=0)
    create.add_argument("--slots", type=int, default=64); create.add_argument("--representation", type=int, default=0)
    create.add_argument("--level-offset", type=int, default=0); create.add_argument("--level-count", type=int, required=True)
    create.add_argument("--training-level-count", type=int, default=0)
    create.add_argument("--purpose", choices=("development", "heldout"), default="development")
    audit = sub.add_parser("audit", help="GPU-free assigned receipt audit, not runtime qualification")
    audit.add_argument("--suite", type=Path, required=True); audit.add_argument("--episodes", type=Path, required=True)
    evaluate = sub.add_parser("run", help="Scheduled native GPU inference; --prepare-only never queries GPU")
    for name in ("binary", "config", "checkpoint", "suite", "out"):
        evaluate.add_argument("--"+name, type=Path, required=True)
    evaluate.add_argument("--family", choices=("state", "tiny", *FAMILIES.values(), "impala", "impoola"), required=True)
    evaluate.add_argument("--timeout", type=int, default=600)
    evaluate.add_argument("--eager", action="store_true")
    evaluate.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.command == "suite": create_suite(args)
    elif args.command == "run": run(args)
    else:
        suite, manifest = load_suite(args.suite)
        print(json.dumps(audit_csv(args.episodes, suite, manifest), indent=2))


if __name__ == "__main__": main()
