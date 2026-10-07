#!/usr/bin/env python3
"""Snake suite/audit preparation and supervised native checkpoint evaluation.

suite, audit and run --prepare-only execute no policy. run requires a scheduled
GPU window; inference remains native C/CUDA, never Python or CPU.
"""
import argparse
import configparser
import csv
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from claim import episode_seeds, process, require
from deterministic_eval import FAMILIES, load_config, save, sha

TOOL_FILES = {"ocean/snakecnn/exact_eval.py": Path(__file__).resolve(),
              "ocean/connect4cnn/claim.py": ROOT / "ocean/connect4cnn/claim.py",
              "ocean/connect4cnn/deterministic_eval.py": ROOT / "ocean/connect4cnn/deterministic_eval.py"}

RULES = "snake-local-episodic-v1"
SEMANTICS = "first-death-or-game-horizon-v1"
ESTIMAND = "mean-ending-snake-length-all-assigned-episodes-v1"
INTEGER_KEYS = ("rule_version", "num_agents", "vision", "leave_corpse_on_death", "width", "height",
                "num_food", "max_snake_length", "max_steps", "cell_size")
WORLD_KEYS = INTEGER_KEYS + ("reward_food", "reward_death")
MANIFEST_FIELDS = ("episode_id", "env_seed", "policy_seed", "representation", "start_rng",
                   "start_world_hash", "start_observation_hash")
RESULT_FIELDS = ("version", "block_seed", "episode_id", "slot", "representation", "env_seed", "policy_seed",
                 "decisions", "score", "foods", "perf", "episode_return", "end_reason", "action_hash",
                 "start_observation_hash", "start_world_hash", "start_rng")


def read(path):
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(path), "Need full resolved INI")
    return config


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def world(config):
    values = {key: config.getfloat("env", key) for key in WORLD_KEYS}
    require(all(math.isfinite(v) for v in values.values()), "Nonfinite game setting")
    require(all(values[k].is_integer() for k in INTEGER_KEYS), "Fractional integer game setting")
    require(values["rule_version"] == 1 and values["num_agents"] == 1 and values["vision"] == 5
            and values["leave_corpse_on_death"] == 0, "Need one-agent local episodic v1")
    require(13 <= values["width"] <= 128 and 13 <= values["height"] <= 128, "Invalid board")
    capacity = (values["width"]-10)*(values["height"]-10)
    require(1 <= values["num_food"] <= capacity-1
            and 2 <= values["max_snake_length"] <= capacity-values["num_food"]+1, "Invalid spawn/ring capacity")
    require(1 <= values["max_steps"] <= 16777216 and 1 <= values["cell_size"] <= 64, "Invalid horizon/viewer setting")
    require(0 <= values["reward_food"] <= 16 and -16 <= values["reward_death"] <= 0, "Invalid rewards")
    return values


def validate_suite(suite):
    require(suite["protocol"] == "snake-fixed-suite-v1" and suite["rules"] == RULES
            and suite["evaluation"] == SEMANTICS and suite["estimand"] == ESTIMAND, "Wrong protocol/rules/estimand")
    require(suite["environment"] in ("snakecnn", "snakebench"), "Wrong environment")
    require(suite["sampling"] == "native-philox-categorical-v1"
            and suite["purpose"] in ("development", "heldout"), "Wrong sampling/purpose")
    for key, lo, hi in (("seed", 0, 2**32-1), ("offset", 0, 2**32-1), ("episodes", 1, 1000000),
                        ("slots", 1, 1024), ("representation", 0, 5)):
        require(type(suite[key]) is int and lo <= suite[key] <= hi, f"Invalid {key}")
    require(suite["environment"] != "snakebench" or suite["representation"] == 0, "State has no appearance")
    require(suite["offset"] + suite["episodes"] + suite["slots"] <= 2**32-1, "Episode IDs overflow")
    config = configparser.ConfigParser(); config["env"] = {k: str(v) for k, v in suite["world"].items()}
    require(world(config) == suite["world"], "Incomplete/changed world")


def info(binary):
    identity = json.loads(subprocess.check_output([str(binary), "eval_exact_info"], text=True, timeout=30))
    require(identity["environment"] in ("snakecnn", "snakebench") and identity["rules"] == RULES
            and identity["evaluation"] == SEMANTICS and identity["float32"] and identity["receipt_version"] == 1,
            "Need current float32 Snake exact build")
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
        require(int(row["representation"]) == suite["representation"]
                and 0 <= int(row["start_rng"]) <= 2**32-1, "Wrong appearance/post-reset RNG")
        for key in ("start_world_hash", "start_observation_hash"):
            require(re.fullmatch(r"[0-9a-f]{16}", row[key]) is not None, "Malformed start hash")
    return rows


def suite_config(config, suite):
    copy = configparser.ConfigParser(interpolation=None)
    copy.read_dict({s: dict(config[s]) for s in config.sections()})
    copy["env"].update(representation=str(suite["representation"]), representation_mode="0", representation_seed="0")
    copy["eval_exact"] = dict(seed=str(suite["seed"]), episodes=str(suite["episodes"]),
                              episode_offset=str(suite["offset"]), slots=str(suite["slots"]))
    return copy


def create_suite(args):
    binary, source, out = args.binary.resolve(), args.config.resolve(), args.out.resolve()
    identity = info(binary); config = read(source)
    require(config.get("base", "env_name") == identity["environment"], "Binary/config mismatch")
    suite = dict(protocol="snake-fixed-suite-v1", rules=RULES, evaluation=SEMANTICS, estimand=ESTIMAND,
                 environment=identity["environment"], seed=args.seed, offset=args.offset, episodes=args.episodes,
                 slots=args.slots, representation=args.representation, purpose=args.purpose,
                 sampling="native-philox-categorical-v1", world=world(config))
    validate_suite(suite)
    out.mkdir(parents=True, exist_ok=False); binary_hash, config_hash = sha(binary), sha(source)
    shutil.copyfile(source, out / "source.ini")
    with (out / "manifest.ini").open("x") as stream: suite_config(config, suite).write(stream)
    with (out / "episodes.csv").open("x") as stream:
        subprocess.run([str(binary), "eval_exact_manifest", str(out / "manifest.ini")], text=True,
                       stdout=stream, stderr=subprocess.PIPE, check=True, timeout=120)
    validate_manifest(out / "episodes.csv", suite)
    require(sha(binary) == binary_hash and sha(source) == config_hash, "Input changed during preparation")
    suite.update(episodes_sha256=sha(out / "episodes.csv"), source_ini_sha256=config_hash,
                 manifest_ini_sha256=sha(out / "manifest.ini"), binary_sha256=binary_hash,
                 binary_info=identity, evidence_status="host-spawns-only-gpu-runtime-pending")
    save(out / "suite.json", suite); print(out / "suite.json")


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


def food_returns(counts, reward):
    """Exact repeated float32 addition, O(max food count) once per CSV, no model.

    Multiplying count by reward does not reproduce native running-return rounding.
    Retain only the requested counts, avoiding a horizon-sized float cache.
    """
    desired = sorted(set(counts))
    if reward == 0: return {count: 0.0 for count in desired}
    values, position, total = {}, 0, 0.0
    for count in desired:
        while position < count:
            total = f32(total+reward); position += 1
        values[count] = total
    return values


def audit_csv(path, suite, manifest):
    expected = {int(row["episode_id"]): row for row in manifest}
    seen, deaths, horizons, score_total, food_total, decision_total = set(), 0, 0, 0, 0, 0
    return_receipts, perfs = [], []
    horizon, ring = int(suite["world"]["max_steps"]), int(suite["world"]["max_snake_length"])
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == RESULT_FIELDS, "Wrong result columns")
        for row in reader:
            episode = int(row["episode_id"])
            require(episode in expected and episode not in seen, "Extra/duplicate episode")
            seen.add(episode)
            require(int(row["version"]) == 1 and int(row["block_seed"]) == suite["seed"], "Wrong version/seed")
            require(int(row["slot"]) == (episode-suite["offset"]) % suite["slots"], "Wrong slot")
            for key in MANIFEST_FIELDS: require(row[key] == expected[episode][key], f"Wrong start/RNG/appearance: {key}")
            decisions, score, foods, reason = (int(row[k]) for k in ("decisions", "score", "foods", "end_reason"))
            require(1 <= decisions <= horizon and 0 <= foods <= decisions, "Invalid food/decision counts")
            require(reason in (1, 2) and (reason != 2 or decisions == horizon)
                    and (reason != 1 or foods < decisions), "Invalid death/horizon accounting")
            require(score == min(foods+1, ring-1), "Wrong ending length/growth limit")
            perf, reward = float.fromhex(row["perf"]), float.fromhex(row["episode_return"])
            require(perf == min(f32(score/120), 1) and math.isfinite(reward), "Wrong perf/nonfinite return")
            require(re.fullmatch(r"[0-9a-f]{16}", row["action_hash"]) is not None, "Malformed action hash")
            score_total += score; food_total += foods; decision_total += decisions
            deaths += reason == 1; horizons += reason == 2
            return_receipts.append((foods, reason, reward)); perfs.append(perf)
    require(seen == set(expected) and len(seen) == suite["episodes"], "Incomplete assigned episode set")
    prefix = food_returns((count for count, _, _ in return_receipts), f32(suite["world"]["reward_food"]))
    for count, reason, reward in return_receipts:
        reconstructed = f32(prefix[count]+f32(suite["world"]["reward_death"])) if reason == 1 else prefix[count]
        require(reward == reconstructed, "Wrong native float32 running return")
    n = len(seen)
    return dict(episodes=n, deaths=deaths, horizon_ends=horizons, score=score_total, foods=food_total,
                mean_score=score_total/n, mean_foods=food_total/n, mean_decisions=decision_total/n,
                mean_perf=math.fsum(perfs)/n, mean_return=math.fsum(r for _, _, r in return_receipts)/n,
                estimand=ESTIMAND, policy_runtime_certified=False)


def checkpoint_config(path, identity, family):
    if identity["environment"] != "snakebench":
        return load_config(path, identity, family)
    config = read(path)
    require(family == "state" and config.get("base", "env_name") == "snakebench", "State binary/config/family mismatch")
    require(all(s in config for s in ("base", "vec", "selfplay", "env", "policy", "train")), "Need full resolved INI")
    require(config.getint("policy", "hidden_size") > 0 and config.getint("policy", "num_layers") > 0, "Invalid state core")
    return config


def native_manifest(binary, config, directory):
    ini_path, csv_path = directory / "spawn-check.ini", directory / "spawn-check.csv"
    with ini_path.open("x") as stream: config.write(stream)
    with csv_path.open("x") as stream:
        subprocess.run([str(binary), "eval_exact_manifest", str(ini_path)], stdout=stream,
                       stderr=subprocess.PIPE, text=True, check=True, timeout=120)
    return csv_path


def run(args):
    binary, training, checkpoint = args.binary.resolve(), args.config.resolve(), args.checkpoint.resolve()
    suite_path, out = args.suite.resolve(), args.out.resolve()
    suite, manifest = load_suite(suite_path); identity = info(binary)
    require(identity["environment"] == suite["environment"], "Suite/binary observation type mismatch")
    config = checkpoint_config(training, identity, args.family)
    require(world(config) == suite["world"], "Training/suite game settings differ")
    require(args.timeout > 0 and checkpoint.is_file(), "Need weights file/positive timeout")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="pending", suite_sha256=sha(suite_path), manifest_sha256=suite["episodes_sha256"],
                  checkpoint_sha256=sha(checkpoint), binary_sha256=sha(binary), training_ini_sha256=sha(training),
                  family=args.family, binary_info=identity, policy=dict(config["policy"]),
                  training_appearance={k: config.get("env", k, fallback="0") for k in
                                       ("representation", "representation_mode", "representation_seed")},
                  evaluation_representation=suite["representation"], rules=RULES, evaluation=SEMANTICS,
                  estimand=ESTIMAND, world=suite["world"], binary=str(binary), checkpoint=str(checkpoint),
                  config=str(training), suite=str(suite_path), cwd=str(out), prepare_only=args.prepare_only,
                  gpu_runtime_qualified=False,
                  note="Evaluator GPU acceptance is pending. Tooling snapshots describe the launcher; "
                       "compiled native provenance requires separate build receipts.")
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
        config = suite_config(config, suite)
        observed = native_manifest(binary, config, out)
        require(observed.read_bytes() == (out / "suite/episodes.csv").read_bytes(),
                "Binary generates different starts from frozen manifest")
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
                          ("training_ini_sha256", out / "training.ini"), ("suite_sha256", out / "suite/suite.json")):
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
            summary = re.findall(r"SNAKE_EXACT_EVAL version=1 requested=(\d+) completed=(\d+) deaths=(\d+) horizons=(\d+) score=(\d+) foods=(\d+) params=(\d+)",
                                 (out / "native.log").read_text())
            require(len(summary) == 1, "Missing/duplicate completion receipt")
            requested, completed, deaths, horizons, score, foods, parameters = map(int, summary[0])
            require(requested == completed == counts["episodes"] and deaths == counts["deaths"]
                    and horizons == counts["horizon_ends"] and score == counts["score"] and foods == counts["foods"]
                    and parameters*4 == checkpoint.stat().st_size, "Summary/CSV/weights mismatch")
            record.update(status="ok", counts=counts, parameters=parameters, episodes_sha256=sha(out / "episodes.csv"))
            (out / "REPORT.md").write_text(
                f"# Deterministic Snake evaluation\n\n{args.family}: mean ending length {counts['mean_score']:.6f} "
                f"across {completed} exactly assigned episodes; {deaths} deaths, {horizons} game-horizon endings. "
                f"Mean food {counts['mean_foods']:.6f}, clipped length/120 perf {counts['mean_perf']:.6f}, "
                f"return {counts['mean_return']:.6f}, decisions {counts['mean_decisions']:.6f}.\n\n"
                f"Every assigned episode contributes once, including deaths. The {int(suite['world']['max_steps'])}-decision "
                "game horizon is fixed; food after growth saturation still counts. Length/food/perf aren't win rates. "
                "This is the separately named one-agent local Snake protocol, not original stock Snake.\n\n"
                "Training policy/core shapes are preserved. Evaluation process time is not training time. "
                "This checkpoint report does not qualify GPU behavior, replicated frontier superiority or SOTA.\n")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(out / "result.json", record)
    print(out / ("result.json" if args.prepare_only else "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("suite", help="GPU-free native episode starts; no policy executes")
    for key in ("binary", "config", "out"): create.add_argument("--"+key, type=Path, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--episodes", type=int, default=1000)
    create.add_argument("--offset", type=int, default=0)
    create.add_argument("--slots", type=int, default=64)
    create.add_argument("--representation", type=int, default=0)
    create.add_argument("--purpose", choices=("development", "heldout"), default="development")
    audit = sub.add_parser("audit", help="GPU-free assigned receipt audit; cannot certify runtime provenance")
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
