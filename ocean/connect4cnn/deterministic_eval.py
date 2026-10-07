#!/usr/bin/env python3
"""Frozen suite preparation and receipts. Policy inference is native C/CUDA."""
import argparse
import configparser
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from claim import ENVIRONMENT_RULES, episode_seeds, episodes, require

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {1: "experimental", 2: "nature", 3: "compact", 4: "flex", 5: "flex2"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def manifest_rows(suite):
    for episode in range(suite["offset"], suite["offset"] + suite["episodes"]):
        env, policy = episode_seeds(suite["seed"], episode)
        yield dict(episode_id=episode, env_seed=env, policy_seed=policy,
                   representation=suite["representation"], player_pieces=0, env_pieces=0)


def validate_suite(suite):
    require(suite["protocol"] == "connect4-fixed-suite-v1" and
            suite["rules"] == ENVIRONMENT_RULES, "Unsupported suite/rules")
    for key, low, high in (("seed", 0, 2**32-1), ("offset", 0, 2**32-1),
                           ("episodes", 1, 1000000), ("slots", 1, 1024),
                           ("representation", 0, 9)):
        require(type(suite[key]) is int and low <= suite[key] <= high, f"Invalid {key}")
    require(suite["offset"] + suite["episodes"] + suite["slots"] <= 2**32-1,
            "Episode IDs overflow native wave allocation")
    require(suite["sampling"] == "native-philox-categorical-v1", "Unsupported sampling")
    require(suite["purpose"] in ("development", "heldout"), "Invalid suite purpose")


def create_suite(args):
    suite = dict(protocol="connect4-fixed-suite-v1", rules=ENVIRONMENT_RULES,
                 seed=args.seed, offset=args.offset, episodes=args.episodes,
                 slots=args.slots, representation=args.representation,
                 sampling="native-philox-categorical-v1", purpose=args.purpose)
    validate_suite(suite)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out/"episodes.csv").open("x", newline="") as stream:
        rows = manifest_rows(suite)
        first = next(rows)
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)
    suite["episodes_sha256"] = sha(out/"episodes.csv")
    save(out/"suite.json", suite)
    print(out/"suite.json")


def load_suite(path):
    suite = json.loads(path.read_text())
    validate_suite(suite)
    manifest = path.parent/"episodes.csv"
    require(sha(manifest) == suite["episodes_sha256"], "Changed episode manifest")
    with manifest.open() as stream:
        actual = list(csv.DictReader(stream))
    expected = [{k: str(v) for k, v in row.items()} for row in manifest_rows(suite)]
    require(actual == expected, "Manifest does not match native episode identity mapping")
    return suite


def load_config(path, info, family):
    config = configparser.ConfigParser(interpolation=None)
    require(config.read(path), "Cannot read resolved checkpoint INI")
    for section in ("base", "vec", "selfplay", "env", "policy", "train"):
        require(section in config, f"Need full resolved training INI: missing [{section}]")
    require(config["base"]["env_name"] == info["environment"], "Binary/config environment mismatch")
    for key in ("hidden_size", "num_layers"):
        require(config.getint("policy", key) > 0, f"Missing/invalid policy.{key}")
    if info["environment"] == "connect4":
        actual = "state"
    else:
        # Older native resolved INIs predate the optional selector. Absence
        # selects the compiled default, exactly as create_custom_encoder does.
        encoder = config.getint("policy", "encoder", fallback=0)
        require(encoder in (0, *FAMILIES), "Unknown encoder")
        actual = FAMILIES[encoder] if encoder else info["default_encoder"]
    require(actual == family, f"Requested {family}, but binary/config construct {actual}")
    return config


def empty_observation_hash(size):
    value = 14695981039346656037
    for _ in range(size * 4):
        value = (value * 1099511628211) & (2**64-1)
    return f"{value:016x}"


def audit_csv(path, suite, environment):
    representation = suite["representation"] if environment == "connect4cnn" else 0
    result = episodes(path, suite, suite["slots"], representation)
    expected_hash = empty_observation_hash(36*44 if environment == "connect4cnn" else 42)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            require(row["start_observation_hash"] == expected_hash, "Initial observation differs")
            require(row["start_player_pieces"] == row["start_env_pieces"] == "0", "Initial board differs")
            require(row["start_rng"] == row["env_seed"], "Initial environment RNG differs")
    return result


def run(args):
    suite_path, binary = args.suite.resolve(), args.binary.resolve()
    checkpoint, config_path, out = args.checkpoint.resolve(), args.config.resolve(), args.out.resolve()
    suite = load_suite(suite_path)
    info = json.loads(subprocess.check_output([binary, "eval_exact_info"], text=True, timeout=30))
    require(info["float32"] and info["rules"] == ENVIRONMENT_RULES and info["receipt_version"] == 2,
            "Need a current float32 exact-evaluation build")
    require(info["environment"] in ("connect4", "connect4cnn"), "Unsupported environment")
    config = load_config(config_path, info, args.family)
    require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Explicit weights file required")
    require(args.timeout > 0, "Timeout must be positive")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out/"training.ini")
    (out/"suite").mkdir()
    shutil.copyfile(suite_path, out/"suite/suite.json")
    shutil.copyfile(suite_path.parent/"episodes.csv", out/"suite/episodes.csv")
    config["base"].update(load_model_path=str(checkpoint), eval_agents=str(suite["slots"]),
                          checkpoint_dir=str(out/"unused-checkpoints"), log_dir=str(out/"metrics"),
                          run_id="evaluation", seed="0", cudagraphs="-1" if args.eager else "1",
                          **{"async": "0"})
    config["vec"].update(total_agents=str(suite["slots"]), num_buffers="1", num_threads="1",
                         num_policies="1", hist_policy_percent="0")
    config["selfplay"]["enabled"] = "0"
    config["env"].update(player_pieces="0", env_pieces="0")
    if info["environment"] == "connect4cnn":
        config["env"].update(representation=str(suite["representation"]),
                             representation_mode="0", representation_seed="0")
    config["train"].update(gpus="1", horizon="1", minibatch_size=str(suite["slots"]),
                           replay_ratio="1", verb_eps="0")
    config["eval_exact"] = dict(episodes=str(suite["episodes"]), episode_offset=str(suite["offset"]),
                                seed=str(suite["seed"]), slots=str(suite["slots"]), output=str(out/"episodes.csv"))
    (out/"config").mkdir()
    with (out/"config/default.ini").open("x") as stream:
        config.write(stream)
    (out/f"config/{info['environment']}.ini").touch()
    command = [str(binary), "eval_exact", "--headless"]
    record = dict(status="pending", suite_sha256=sha(suite_path), checkpoint_sha256=sha(checkpoint),
                  binary_sha256=sha(binary), training_ini_sha256=sha(config_path),
                  effective_ini_sha256=sha(out/"config/default.ini"), family=args.family,
                  architecture=dict(config["policy"]), binary_info=info, command=command,
                  checkpoint=str(checkpoint), config=str(config_path), cwd=str(out),
                  source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  note="Source HEAD is runner context; binary hash identifies execution. Raw weights require the supplied training INI/build family.")
    save(out/"request.json", record)
    try:
        smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        (ROOT/"build/connect4cnn").mkdir(parents=True, exist_ok=True)
        with (ROOT/"build/connect4cnn/hardware-benchmark.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = subprocess.check_output([smi, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
            require(not busy.strip(), "GPU busy; no processes were interrupted")
            hardware = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20)
            (out/"gpu.txt").write_text(hardware)
            started = time.monotonic()
            try:
                with (out/"native.log").open("x") as stream:
                    result = subprocess.run(command, cwd=out, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=args.timeout, check=False)
            finally:
                record["evaluation_process_seconds"] = time.monotonic() - started
            require(result.returncode == 0, f"Native evaluator exited {result.returncode}; see native.log")
        for key, path in (("checkpoint_sha256", checkpoint), ("binary_sha256", binary),
                          ("training_ini_sha256", config_path), ("suite_sha256", suite_path)):
            require(sha(path) == record[key], f"Input changed during evaluation: {path}")
        counts = audit_csv(out/"episodes.csv", suite, info["environment"])
        summary = re.findall(r"CONNECT4_EXACT_EVAL version=1 requested=(\d+) completed=(\d+) wins=(\d+) params=(\d+)",
                             (out/"native.log").read_text())
        require(len(summary) == 1, "Missing/duplicate native completion receipt")
        requested, completed, wins, parameters = map(int, summary[0])
        require(requested == completed == counts["episodes"] and wins == counts["wins"], "Summary/CSV mismatch")
        require(parameters * 4 == checkpoint.stat().st_size, "Checkpoint size mismatch")
        record.update(status="ok", counts=counts, parameters=parameters, episodes_sha256=sha(out/"episodes.csv"))
        (out/"REPORT.md").write_text(
            f"# Deterministic Connect4 evaluation\n\n{args.family}: {wins}/{completed} wins ({wins/completed:.2%}). "
            f"Exactly {completed} assigned episodes completed once.\n\n"
            f"Suite SHA256: `{record['suite_sha256']}`. Rules: `{ENVIRONMENT_RULES}`. "
            f"Fixed pixel representation: {suite['representation']}; {suite['slots']} inference slots.\n\n"
            "Per-episode IDs, starting observation/board/RNG receipts and action hashes are in `episodes.csv`. "
            "The full supplied policy configuration is preserved; no architecture or core size is substituted. "
            "Evaluation time is not training time. This checkpoint evaluation alone does not establish "
            "a replicated learning advantage or a Pareto-front claim.\n")
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(out/"result.json", record)
    print(out/"REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("suite", help="GPU-free immutable episode manifest")
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--episodes", type=int, default=1000)
    create.add_argument("--offset", type=int, default=0)
    create.add_argument("--slots", type=int, default=64)
    create.add_argument("--representation", type=int, default=0)
    create.add_argument("--purpose", choices=("development", "heldout"), default="development")
    evaluate = sub.add_parser("run", help="Native GPU inference of any supported saved policy shape")
    for key in ("suite", "binary", "config", "checkpoint", "out"):
        evaluate.add_argument(f"--{key}", type=Path, required=True)
    evaluate.add_argument("--family", choices=("state", "tiny", *FAMILIES.values(), "impala", "impoola"), required=True)
    evaluate.add_argument("--timeout", type=int, default=300)
    evaluate.add_argument("--eager", action="store_true", help="Repeatability diagnostic; default CUDA graphs")
    args = parser.parse_args()
    create_suite(args) if args.command == "suite" else run(args)


if __name__ == "__main__":
    main()
