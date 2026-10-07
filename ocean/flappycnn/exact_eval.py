#!/usr/bin/env python3
"""Frozen Flappy suites, receipt audits and supervised native checkpoint evaluation.

suite, audit and run --prepare-only execute no policy. Native GPU inference
requires an explicitly scheduled window; no CPU model execution is provided.
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

TOOL_FILES = {"ocean/flappycnn/exact_eval.py": Path(__file__).resolve(),
              "ocean/connect4cnn/claim.py": ROOT / "ocean/connect4cnn/claim.py",
              "ocean/connect4cnn/deterministic_eval.py": ROOT / "ocean/connect4cnn/deterministic_eval.py"}

RULES = "flappy-native-v1"
WORLD_KEYS = ("width", "height", "max_steps", "gravity", "flap_velocity", "pipe_speed",
              "pipe_gap", "pipe_width", "pipe_spacing", "first_pipe_x", "bird_x", "bird_radius",
              "alive_reward", "pass_reward", "crash_reward", "center_reward")
MANIFEST_FIELDS = ("episode_id", "env_seed", "policy_seed", "representation", "start_rng",
                   "start_world_hash", "start_observation_hash")
RESULT_FIELDS = ("version", "block_seed", "episode_id", "slot", "representation", "env_seed",
                 "policy_seed", "decisions", "pipes", "perf", "episode_return", "cap_reached",
                 "action_hash", "start_observation_hash", "start_world_hash", "start_rng")


def read(path):
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(path), "Need a full resolved INI")
    return config


def world(config):
    values = {key: config.getfloat("env", key) for key in WORLD_KEYS}
    require(all(math.isfinite(v) and abs(v) <= 3.4028234663852886e38 for v in values.values()), "Nonfinite/overflowing world parameter")
    for key in ("width", "height", "max_steps"):
        require(1 <= values[key] <= 1000000 and values[key].is_integer(), f"Invalid {key}")
    for key in ("gravity", "pipe_speed", "pipe_gap", "pipe_width", "pipe_spacing", "bird_radius"):
        require(values[key] > 0, f"Invalid positive {key}")
    require(values["pipe_gap"] < values["height"], "Pipe gap must fit inside world")
    return values


def info(binary):
    identity = json.loads(subprocess.check_output([str(binary), "eval_exact_info"], text=True, timeout=30))
    require(identity["environment"] in ("flappy", "flappycnn") and identity["rules"] == RULES
            and identity["float32"] and identity["receipt_version"] == 1, "Need a current float32 Flappy exact build")
    return identity


def validate_suite(suite):
    require(suite["protocol"] in ("flappy-fixed-suite-v1", "flappy-fixed-suite-v2") and suite["rules"] == RULES, "Wrong suite/rules")
    count = 4
    if suite["protocol"] == "flappy-fixed-suite-v2":
        count = suite["representation_count"]
        require(type(count) is int and count == 7 and suite["environment"] == "flappycnn", "Wrong drawing catalog")
    else:
        require("representation_count" not in suite, "Legacy suite cannot declare an expanded catalog")
    require(suite["environment"] in ("flappy", "flappycnn"), "Wrong environment")
    require(suite["sampling"] == "native-philox-categorical-v1", "Wrong sampling")
    require(suite["purpose"] in ("development", "heldout"), "Wrong purpose")
    for key, low, high in (("seed", 0, 2**32-1), ("offset", 0, 2**32-1), ("episodes", 1, 1000000),
                           ("slots", 1, 1024), ("representation", 0, count-1)):
        require(type(suite[key]) is int and low <= suite[key] <= high, f"Invalid {key}")
    require(suite["environment"] != "flappy" or suite["representation"] == 0, "State suite has no appearance")
    require(suite["offset"] + suite["episodes"] + suite["slots"] <= 2**32-1, "Episode IDs overflow")
    c = configparser.ConfigParser(); c["env"] = {k: str(v) for k, v in suite["world"].items()}
    require(world(c) == suite["world"], "Incomplete/changed world configuration")
    require(suite["cap_semantics"] == "native-terminal-with-crash-reward", "Wrong cap semantics")


def validate_manifest(path, suite):
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == MANIFEST_FIELDS, "Wrong manifest columns")
        rows = list(reader)
    require(len(rows) == suite["episodes"], "Wrong manifest quota")
    for i, row in enumerate(rows):
        episode = suite["offset"] + i
        env_seed, policy_seed = episode_seeds(suite["seed"], episode)
        require(int(row["episode_id"]) == episode and int(row["env_seed"]) == env_seed
                and int(row["policy_seed"]) == policy_seed, "Manifest identity/RNG mismatch")
        require(int(row["representation"]) == suite["representation"], "Manifest appearance mismatch")
        require(0 <= int(row["start_rng"]) <= 2**32-1, "Invalid post-reset RNG")
        for key in ("start_world_hash", "start_observation_hash"):
            require(re.fullmatch(r"[0-9a-f]{16}", row[key]) is not None, "Malformed start hash")
    return rows


def native_manifest(binary, config, path, directory):
    ini_path = directory / "manifest.ini"
    with ini_path.open("x") as stream:
        config.write(stream)
    with path.open("x") as stream:
        subprocess.run([str(binary), "eval_exact_manifest", str(ini_path)], stdout=stream,
                       stderr=subprocess.PIPE, text=True, check=True, timeout=120)


def suite_config(config, suite):
    copy = configparser.ConfigParser(interpolation=None)
    copy.read_dict({s: dict(config[s]) for s in config.sections()})
    if suite["environment"] == "flappycnn":
        copy["env"].update(representation=str(suite["representation"]), representation_mode="0", representation_seed="0")
    copy["eval_exact"] = dict(seed=str(suite["seed"]), episodes=str(suite["episodes"]),
                              episode_offset=str(suite["offset"]), slots=str(suite["slots"]),
                              max_steps=str(int(suite["world"]["max_steps"])))
    return copy


def create_suite(args):
    binary, source, out = args.binary.resolve(), args.config.resolve(), args.out.resolve()
    identity = info(binary)
    config = read(source)
    require(config.get("base", "env_name") == identity["environment"], "Manifest binary/config mismatch")
    suite = dict(protocol="flappy-fixed-suite-v1", rules=RULES, environment=identity["environment"],
                 seed=args.seed, offset=args.offset, episodes=args.episodes, slots=args.slots,
                 representation=args.representation, purpose=args.purpose,
                 sampling="native-philox-categorical-v1", world=world(config),
                 cap_semantics="native-terminal-with-crash-reward")
    if identity["environment"] == "flappycnn" and identity.get("representation_count", 4) > 4:
        suite.update(protocol="flappy-fixed-suite-v2", representation_count=identity["representation_count"])
    validate_suite(suite)
    out.mkdir(parents=True, exist_ok=False)
    binary_hash, config_hash = sha(binary), sha(source)
    shutil.copyfile(source, out / "source.ini")
    native_manifest(binary, suite_config(config, suite), out / "episodes.csv", out)
    validate_manifest(out / "episodes.csv", suite)
    require(sha(binary) == binary_hash and sha(source) == config_hash, "Manifest input changed")
    suite.update(episodes_sha256=sha(out / "episodes.csv"), binary_sha256=binary_hash,
                 source_ini_sha256=config_hash, manifest_ini_sha256=sha(out / "manifest.ini"),
                 configuration_receipt_version=2, binary_info=identity,
                 evidence_status="host-spawns-only-gpu-runtime-pending")
    save(out / "suite.json", suite)
    print(out / "suite.json")


def load_suite(path):
    suite = json.loads(path.read_text()); validate_suite(suite)
    for name, key in (("episodes.csv", "episodes_sha256"), ("source.ini", "source_ini_sha256")):
        require(sha(path.parent / name) == suite[key], f"Changed {name}")
    version = suite.get("configuration_receipt_version", 1)
    require(type(version) is int and version in (1, 2), "Unknown configuration receipt version")
    if version == 2:
        require(sha(path.parent / "manifest.ini") == suite["manifest_ini_sha256"], "Changed manifest.ini")
    else:
        require("manifest_ini_sha256" not in suite, "Unversioned manifest hash")
    source = read(path.parent / "source.ini")
    require(source.get("base", "env_name") == suite["environment"] and world(source) == suite["world"],
            "Suite/source environment/world differs")
    expected, actual = suite_config(source, suite), read(path.parent / "manifest.ini")
    require({s: dict(expected[s]) for s in expected.sections()} == {s: dict(actual[s]) for s in actual.sections()},
            "Suite/manifest configuration differs")
    suite["configuration_integrity"] = ("sha256-and-derived-config" if version == 2 else
                                        "legacy-derived-config-no-original-manifest-hash")
    return suite, validate_manifest(path.parent / "episodes.csv", suite)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def audit_csv(path, suite, manifest):
    expected = {int(row["episode_id"]): row for row in manifest}
    seen, pipes, capped, perf, returns, decisions = set(), 0, 0, [], [], []
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == RESULT_FIELDS, "Wrong result columns")
        for row in reader:
            episode = int(row["episode_id"])
            require(episode in expected and episode not in seen, "Extra/duplicate episode")
            seen.add(episode); start = expected[episode]
            require(int(row["version"]) == 1 and int(row["block_seed"]) == suite["seed"], "Wrong version/seed")
            require(int(row["slot"]) == (episode - suite["offset"]) % suite["slots"], "Wrong episode slot")
            for key in MANIFEST_FIELDS:
                require(row[key] == start[key], f"Start/RNG/appearance mismatch: {key}")
            length, score, cap = int(row["decisions"]), int(row["pipes"]), int(row["cap_reached"])
            require(1 <= length <= suite["world"]["max_steps"] and 0 <= score <= 3*length, "Invalid duration/pipes")
            require(cap == int(length == suite["world"]["max_steps"]), "Native cap accounting mismatch")
            performance, reward = float.fromhex(row["perf"]), float.fromhex(row["episode_return"])
            require(math.isfinite(reward) and performance == f32(min(score / 20, 1)), "Invalid return/perf")
            require(re.fullmatch(r"[0-9a-f]{16}", row["action_hash"]) is not None, "Invalid action hash")
            pipes += score; capped += cap; perf.append(performance); returns.append(reward); decisions.append(length)
    require(seen == set(expected) and len(seen) == suite["episodes"], "Incomplete assigned episode set")
    n = len(seen)
    return dict(episodes=n, pipes=pipes, capped=capped, mean_pipes=pipes/n, mean_perf=math.fsum(perf)/n,
                mean_return=math.fsum(returns)/n, mean_decisions=sum(decisions)/n,
                policy_runtime_certified=False)


def run(args):
    binary, training, checkpoint = args.binary.resolve(), args.config.resolve(), args.checkpoint.resolve()
    suite_path, out = args.suite.resolve(), args.out.resolve()
    suite, manifest = load_suite(suite_path)
    identity = info(binary)
    require(identity["environment"] == suite["environment"], "Suite/binary observation type mismatch")
    require(suite["representation"] < identity.get("representation_count", 4), "Binary does not support this drawing")
    if identity["environment"] == "flappy":
        config = read(training)
        require(args.family == "state" and config.get("base", "env_name") == "flappy", "State binary/config/family mismatch")
        require(all(section in config for section in ("base", "vec", "selfplay", "env", "policy", "train")), "Need full resolved INI")
        require(config.getint("policy", "hidden_size") > 0 and config.getint("policy", "num_layers") > 0, "Invalid state core")
    else:
        config = load_config(training, identity, args.family)
    require(world(config) == suite["world"], "Training/suite physics or native cap differs")
    require(args.timeout > 0 and checkpoint.is_file(), "Need weights/positive timeout")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status="pending", suite_sha256=sha(suite_path), manifest_sha256=suite["episodes_sha256"],
                  checkpoint_sha256=sha(checkpoint), binary_sha256=sha(binary), training_ini_sha256=sha(training),
                  family=args.family, binary_info=identity, policy=dict(config["policy"]),
                  training_appearance={k: config.get("env", k, fallback="0") for k in
                                       ("representation", "representation_mode", "representation_seed")},
                  evaluation_representation=suite["representation"], rules=RULES,
                  binary=str(binary), checkpoint=str(checkpoint), config=str(training), suite=str(suite_path), cwd=str(out),
                  prepare_only=args.prepare_only, gpu_runtime_qualified=False,
                  suite_configuration_integrity=suite["configuration_integrity"],
                  note="GPU runtime/repeatability remains unqualified until scheduled acceptance. "
                       "Tooling snapshots aren't compiled build provenance. Evaluation is not training time.")
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
        # Fresh host manifests catch different binaries/host math before GPU execution.
        native_manifest(binary, config, out / "spawn-check.csv", out)
        require((out / "spawn-check.csv").read_bytes() == (suite_path.parent / "episodes.csv").read_bytes(),
                "Binary generates different starts from frozen manifest")
        record.update(spawn_ini_sha256=sha(out / "manifest.ini"), spawn_csv_sha256=sha(out / "spawn-check.csv"))
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
        with (out / "config/default.ini").open("x") as stream:
            config.write(stream)
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
                          ("spawn_ini_sha256", out / "manifest.ini"), ("spawn_csv_sha256", out / "spawn-check.csv")):
            require(sha(path) == record[key], f"Input changed: {path}")
        require(sha(suite_path.parent / "episodes.csv") == record["manifest_sha256"], "Manifest changed during evaluation")
        require(sha(out / "suite/episodes.csv") == record["manifest_sha256"], "Copied manifest changed during evaluation")
        load_suite(out / "suite/suite.json"); load_suite(suite_path)
        for name, source in TOOL_FILES.items():
            require(sha(source) == tooling[name] and sha(out / "tooling" / name) == tooling[name], "Tooling changed during evaluation")
        if args.prepare_only:
            record["status"] = "prepared-not-executed"
        else:
            counts = audit_csv(out / "episodes.csv", suite, manifest)
            summary = re.findall(r"FLAPPY_EXACT_EVAL version=1 requested=(\d+) completed=(\d+) pipes=(\d+) capped=(\d+) params=(\d+)",
                                 (out / "native.log").read_text())
            require(len(summary) == 1, "Missing/duplicate completion receipt")
            requested, completed, pipes, capped, parameters = map(int, summary[0])
            require(requested == completed == counts["episodes"] and pipes == counts["pipes"]
                    and capped == counts["capped"] and parameters * 4 == checkpoint.stat().st_size, "Summary/CSV/weights mismatch")
            record.update(status="ok", counts=counts, parameters=parameters, episodes_sha256=sha(out / "episodes.csv"))
            (out / "REPORT.md").write_text(
                f"# Deterministic Flappy evaluation\n\n{args.family}: {counts['mean_pipes']:.3f} mean pipes, "
                f"{counts['mean_perf']:.4f} mean clipped perf, {completed} exactly assigned episodes; "
                f"{capped} reached the native {int(suite['world']['max_steps'])}-decision limit.\n\n"
                "Native cap applies the original terminal/crash reward; capped episodes remain in all means. "
                "Perf is clipped pipes/20, not wins. Recurrent core/encoder shape are preserved from the supplied INI. "
                "Per-episode start/world/image/RNG/action receipts and inputs are retained. "
                "This report doesn't qualify GPU behavior, replicated frontier superiority or SOTA.\n")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(out / "result.json", record)
    print(out / ("result.json" if args.prepare_only else "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("suite", help="GPU-free native spawn/image receipt manifest")
    for name in ("binary", "config", "out"):
        create.add_argument(f"--{name}", type=Path, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--episodes", type=int, default=1000)
    create.add_argument("--offset", type=int, default=0)
    create.add_argument("--slots", type=int, default=64)
    create.add_argument("--representation", type=int, default=0)
    create.add_argument("--purpose", choices=("development", "heldout"), default="development")
    audit = sub.add_parser("audit", help="GPU-free assigned receipt audit; not runtime qualification")
    audit.add_argument("--suite", type=Path, required=True); audit.add_argument("--episodes", type=Path, required=True)
    evaluate = sub.add_parser("run", help="Scheduled native GPU inference; --prepare-only never queries GPU")
    for name in ("binary", "config", "checkpoint", "suite", "out"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
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


if __name__ == "__main__":
    main()
