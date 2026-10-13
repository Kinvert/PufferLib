#!/usr/bin/env python3
"""Retain full-table Maze level/world/raster starts; no policy or CUDA executes."""
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    ev.require(1 <= args.episodes <= 1000000, "Invalid start quota")
    binaries, out = args.binary_dir.resolve(), args.out.resolve()
    for name in ("default", "impala", "impoola", "state"): ev.info(binaries / name)
    out.mkdir(parents=True, exist_ok=False)
    source = configparser.ConfigParser(interpolation=None)
    source.read([ev.ROOT / "config/default.ini", ev.ROOT / "config/mazecnn.ini", ev.ROOT / "ocean/mazecnn/compare.ini"])
    ev.require(ev.world(source) == dict(num_maps=8192, map_size=-1), "Need original full level table")
    write(out / "source.ini", source)
    common = [k for k in ev.MANIFEST_FIELDS if k not in ("representation", "start_observation_hash")]
    reference, final_suite = None, None
    hashes = {str(binaries / name): ev.sha(binaries / name) for name in ("default", "impala", "impoola", "state")}
    for representation in range(6):
        directory = out / f"r{representation}"
        ev.create_suite(argparse.Namespace(binary=binaries / "default", config=out / "source.ini", out=directory,
            seed=56118, offset=0, episodes=args.episodes, slots=64, representation=representation,
            purpose="heldout", level_offset=7168, level_count=1024, training_level_count=7168))
        suite, rows = ev.load_suite(directory / "suite.json"); final_suite = suite
        worlds = [{k: row[k] for k in common} for row in rows]
        if reference is None: reference = worlds
        ev.require(worlds == reference, "Appearance changed levels/worlds")
        config = ev.read(directory / "manifest.ini")
        config["policy"].update(hidden_size="512", num_layers="4", encoder="4")
        write(directory / "large-core.ini", config)
        for family in ("default", "impala", "impoola"):
            path = directory / f"{family}.csv"
            ev.require(manifest(binaries / family, directory / "large-core.ini", path) == rows, "Build/core changed starts")
            ev.require(path.read_bytes() == (directory / "episodes.csv").read_bytes(), "Manifest serialization differs")
        config["eval_exact"].update(episodes="1", episode_offset=str(args.episodes - 1), slots="1")
        write(directory / "tail.ini", config)
        ev.require(manifest(binaries / "default", directory / "tail.ini", directory / "tail.csv") == rows[-1:], "Tail start differs")
    state = ev.suite_config(source, dict(final_suite, environment="maze", representation=0))
    state["base"]["env_name"] = "maze"; write(out / "state.ini", state)
    rows = manifest(binaries / "state", out / "state.ini", out / "state.csv")
    ev.require([{k: row[k] for k in common} for row in rows] == reference, "State/pixels have different worlds")
    for path, digest in hashes.items(): ev.require(ev.sha(Path(path)) == digest, "Binary changed during checks")
    files = {str(path.relative_to(out)): ev.sha(path) for path in out.rglob("*") if path.is_file()}
    report = dict(status="host_level_start_checks_passed", episodes_per_appearance=args.episodes, appearances=6,
        table=dict(num_maps=8192, map_size=-1), level_panel=dict(offset=7168, count=1024),
        declared_training_prefix=7168, heldout_exclusion_certified=False, families=3, large_core="H512/L4",
        state_pixel_worlds_equal=True, independent_tails_equal=True, files_sha256=files, binary_sha256=hashes,
        source_sha256=ev.sha(Path(__file__)), policy_executed=False, cuda_executed=False, gpu_runtime_validated=False,
        note="Initialization manifests only. No episodes/policies executed; training prefix is a declaration, not audited training evidence.")
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {args.episodes} native assigned starts per drawing; full 8192-level table, 1024-level panel, families/state/tails match; no policy.")


if __name__ == "__main__": main()
