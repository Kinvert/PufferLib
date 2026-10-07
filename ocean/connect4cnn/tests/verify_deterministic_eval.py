#!/usr/bin/env python3
"""Opt-in serial GPU acceptance using existing checkpoints; never trains.

Cases JSON is a list of {name, family, binary, config, checkpoint}. Each path
is interpreted relative to the invoking directory. Source runtime_env.sh first.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deterministic_eval as ev


def rows(path):
    with path.open() as stream:
        return {int(row["episode_id"]): row for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = ev.load_suite(args.suite)
    cases = json.loads(args.cases.read_text())
    names = [case["name"] for case in cases]
    ev.require(len(set(names)) == len(names) and all(Path(n).name == n for n in names), "Invalid case names")
    args.out.mkdir(parents=True, exist_ok=False)
    ev.save(args.out/"cases.json", cases)
    # Replay the final wave alone: recurrent state must not retain prior waves.
    tail = dict(suite)
    tail["offset"] += ((suite["episodes"]-1)//suite["slots"])*suite["slots"]
    tail["episodes"] = suite["offset"]+suite["episodes"]-tail["offset"]
    ev.create_suite(argparse.Namespace(out=args.out/"tail-suite", **tail))
    results = []
    for case in cases:
        for mode in ("graph", "repeat", "eager", "tail"):
            path = args.out/f"{case['name']}-{mode}"
            command = [sys.executable, str(Path(ev.__file__)), "run", "--suite",
                       str(args.out/"tail-suite/suite.json" if mode == "tail" else args.suite),
                       "--out", str(path)]
            for key in ("family", "binary", "config", "checkpoint"):
                command += [f"--{key}", case[key]]
            if mode == "eager":
                command += ["--eager"]
            subprocess.run(command, check=True, timeout=330)
        original = rows(args.out/f"{case['name']}-graph/episodes.csv")
        for mode in ("repeat", "eager"):
            other = rows(args.out/f"{case['name']}-{mode}/episodes.csv")
            ev.require(original == other, f"{case['name']}: {mode} differs")
        final = rows(args.out/f"{case['name']}-tail/episodes.csv")
        ev.require(final == {key: original[key] for key in final}, f"{case['name']}: prior waves affected final wave")
        record = json.loads((args.out/f"{case['name']}-graph/result.json").read_text())
        results.append(dict(name=case["name"], family=case["family"], parameters=record["parameters"],
                            architecture=record["architecture"], counts=record["counts"],
                            repeat_equal=True, eager_equal=True, independent_tail_equal=True))
        print(f"PASS {case['name']}: exact quota, repeat, eager/graph, fresh recurrent state", flush=True)
    ev.save(args.out/"acceptance.json", dict(status="ok", suite_sha256=ev.sha(args.suite), cases=results))


if __name__ == "__main__":
    main()
