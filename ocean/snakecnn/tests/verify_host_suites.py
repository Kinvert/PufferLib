#!/usr/bin/env python3
"""Retain native Snake starts for all drawings/families, without policy execution."""
import argparse
import configparser
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exact_eval as ev


def write_ini(path, config):
    with path.open("x") as stream: config.write(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args(); binaries = args.binary_dir.resolve(); out = args.out.resolve()
    ev.require(1 <= args.episodes <= 1000000, "Invalid quota")
    out.mkdir(parents=True, exist_ok=False)
    source = configparser.ConfigParser(interpolation=None)
    source.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/snakecnn.ini",
                 ev.ROOT / "ocean/snakecnn/compare.ini"])
    write_ini(out / "source.ini", source)
    common = ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash")
    reference = None; files = []
    for representation in range(6):
        suite_dir = out / f"r{representation}"
        ev.create_suite(argparse.Namespace(binary=binaries / "default", config=out / "source.ini",
                        out=suite_dir, seed=51005, offset=0, episodes=args.episodes, slots=64,
                        representation=representation, purpose="development"))
        suite, rows = ev.load_suite(suite_dir / "suite.json")
        worlds = [{key: row[key] for key in common} for row in rows]
        if reference is None: reference = worlds
        ev.require(worlds == reference, "Appearance changed assigned world")
        config = ev.read(suite_dir / "manifest.ini")
        config["policy"].update(hidden_size="512", num_layers="4", encoder="4")
        write_ini(suite_dir / "large-core.ini", config)
        for family in ("default", "impala", "impoola"):
            path = suite_dir / (family+".csv")
            with path.open("x") as stream:
                subprocess.run([str(binaries / family), "eval_exact_manifest", str(suite_dir / "large-core.ini")],
                               stdout=stream, stderr=subprocess.PIPE, check=True, timeout=120)
            ev.require(path.read_bytes() == (suite_dir / "episodes.csv").read_bytes(), "Family/core changed starts")
            files.append(path)
        config["eval_exact"].update(episode_offset=str(args.episodes-1), episodes="1", slots="1")
        write_ini(suite_dir / "tail.ini", config)
        with (suite_dir / "tail.csv").open("x") as stream:
            subprocess.run([str(binaries / "default"), "eval_exact_manifest", str(suite_dir / "tail.ini")],
                           stdout=stream, stderr=subprocess.PIPE, check=True, timeout=30)
        with (suite_dir / "tail.csv").open() as stream: tail = list(csv.DictReader(stream))
        ev.require(tail == rows[-1:], "Independent final episode differs")
        files.append(suite_dir / "tail.csv")
    source["base"]["env_name"] = "snakebench"
    for key in ("representation", "representation_mode", "representation_seed"):
        del source["env"][key]
    write_ini(out / "state-source.ini", source)
    ev.create_suite(argparse.Namespace(binary=binaries / "state", config=out / "state-source.ini",
                    out=out / "state", seed=51005, offset=0, episodes=args.episodes, slots=64,
                    representation=0, purpose="development"))
    _, rows = ev.load_suite(out / "state/suite.json")
    ev.require([{key: row[key] for key in common} for row in rows] == reference, "State world differs")
    files += sorted(out.glob("**/suite.json")) + sorted(out.glob("**/episodes.csv"))
    report = dict(status="passed-host-starts-only", episodes_per_drawing=args.episodes,
                  drawings=6, native_families=3, state_world_matching=True,
                  independent_final_episode_matching=True, rules=ev.RULES,
                  gpu_policy_executed=False, policy_runtime_certified=False,
                  binary_sha256={family: ev.sha(binaries / family) for family in ("default", "impala", "impoola", "state")},
                  artifact_sha256={str(path.relative_to(out)): ev.sha(path) for path in files})
    ev.save(out / "REPORT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
