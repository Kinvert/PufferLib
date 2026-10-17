#!/usr/bin/env python3
"""Retain Pong world/raster start receipts; no policy or CUDA execution."""
import argparse
import configparser
import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exact_eval as ev


def write(path, config):
    with path.open("x") as stream: config.write(stream)


def manifest(binary, config, path):
    with path.open("x") as stream:
        subprocess.run([str(binary), "eval_exact_manifest", str(config)], stdout=stream,
                       stderr=subprocess.PIPE, text=True, check=True, timeout=120)
    with path.open() as stream: return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--legacy-binary-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    ev.require(1 <= args.episodes <= 1000000, "Invalid start quota")
    binaries, legacy, out = args.binary_dir.resolve(), args.legacy_binary_dir.resolve(), args.out.resolve()
    ev.require(ev.info(binaries / "default")["representation_count"] == 7, "Need new seven-ID native build")
    ev.require(ev.info(legacy / "default")["representation_count"] == 5, "Need original five-ID native build")
    out.mkdir(parents=True, exist_ok=False)
    source = configparser.ConfigParser(interpolation=None)
    source.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/pongcnn.ini", ev.ROOT / "ocean/pongcnn/compare.ini"])
    write(out / "source.ini", source)
    common = ("episode_id", "env_seed", "policy_seed", "start_rng", "start_world_hash")
    reference, files, legacy_checked = None, [], []
    binary_sha = {str(path): ev.sha(path) for path in [binaries / name for name in ("default", "impala", "impoola", "state")] + [legacy / "default"]}
    for representation in range(7):
        directory = out / f"r{representation}"
        ev.create_suite(argparse.Namespace(binary=binaries / "default", config=out / "source.ini", out=directory,
                         seed=53118, offset=0, episodes=args.episodes, slots=64, representation=representation,
                         max_decisions=64, purpose="development"))
        suite, rows = ev.load_suite(directory / "suite.json")
        worlds = [{key: row[key] for key in common} for row in rows]
        if reference is None: reference = worlds
        ev.require(worlds == reference, "Appearance changed assigned worlds")
        config = ev.read(directory / "manifest.ini")
        config["policy"].update(hidden_size="512", num_layers="4", encoder="4")
        write(directory / "large-core.ini", config)
        for family in ("default", "impala", "impoola"):
            path = directory / (family + ".csv")
            ev.require(manifest(binaries / family, directory / "large-core.ini", path) == rows, "Family/core changed starts")
            ev.require(path.read_bytes() == (directory / "episodes.csv").read_bytes(), "Manifest serialization differs")
            files.append(path)
        if representation < 5:
            path = directory / "legacy.csv"
            ev.require(manifest(legacy / "default", directory / "manifest.ini", path) == rows, "Legacy world/raster starts changed")
            ev.require(path.read_bytes() == (directory / "episodes.csv").read_bytes(), "Legacy manifest bytes differ")
            files.append(path); legacy_checked.append(representation)
        config["eval_exact"].update(episode_offset=str(args.episodes - 1), episodes="1", slots="1")
        write(directory / "tail.ini", config)
        path = directory / "tail.csv"
        ev.require(manifest(binaries / "default", directory / "tail.ini", path) == rows[-1:], "Independent tail start differs")
        files.append(path)
    state = ev.suite_config(source, dict(suite, environment="pong", representation=0))
    state["base"]["env_name"] = "pong"
    write(out / "state.ini", state)
    state_rows = manifest(binaries / "state", out / "state.ini", out / "state.csv")
    ev.require([{key: row[key] for key in common} for row in state_rows] == reference, "State/pixel worlds differ")
    files.append(out / "state.csv")
    for path, digest in binary_sha.items(): ev.require(ev.sha(Path(path)) == digest, "Binary changed during check")
    hashes = {str(path.relative_to(out)): ev.sha(path) for path in out.rglob("*") if path.is_file()}
    report = dict(status="host_start_checks_passed", episodes_per_appearance=args.episodes, appearances=7,
                  legacy_ids=legacy_checked, legacy_manifest_bytes_equal=True, families=3, large_core="H512/L4",
                  independent_tails_equal=True, state_pixel_worlds_equal=True, files_sha256=hashes, binary_sha256=binary_sha,
                  policy_executed=False, cuda_executed=False, gpu_runtime_validated=False,
                  source_sha256=ev.sha(Path(__file__)), note="World/raster initialization only; hashes don't certify inference, complete episodes or learning.")
    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {args.episodes} assigned starts for each of seven appearances; five legacy CSVs unchanged; no policy/CUDA execution.")


if __name__ == "__main__":
    main()
