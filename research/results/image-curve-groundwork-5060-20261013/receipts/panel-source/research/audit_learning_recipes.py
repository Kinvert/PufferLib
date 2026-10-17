#!/usr/bin/env python3
"""Compile/run native learner configuration arithmetic; no model or GPU.

Keep configuration geometry distinct from successful learning calibration.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import audit_pixel_panel as audit
import prepare_pixel_robustness as panel
from claim import require, save, sha

ROOT = panel.ROOT
CONTROLS = dict(connect4cnn="connect4", pongcnn="pong", flappycnn="flappy",
                breakoutcnn="breakout", snakecnn="snakebench", mazecnn="maze")
SOURCE_MARKERS = (
    ("int total_minibatches = hypers->replay_ratio * batch_size / hypers->minibatch_size;", 2),
    ("int total_epochs = hypers->total_timesteps / hypers->world_size / batch_size;", 1),
    ("long local_timesteps = total_timesteps / ctx->world_size;", 1),
    ("long train_epochs = local_timesteps / batch_size;", 1),
    ("int dest_off = (mb * mb_segs) % n_rows;", 1),
    ("muon_step(&pufferl->muon, primary->master_weights,", 1),
)


def source_guard(path):
    text = path.read_text()
    require(all(text.count(marker) == count for marker, count in SOURCE_MARKERS),
            "Native learner arithmetic changed; review the geometry helper before reuse")


def file_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def read(paths):
    return panel.read(*paths)


def geometry(binary, paths):
    result = json.loads(subprocess.check_output([str(binary), *map(str, paths)], text=True, timeout=20))
    require(result["protocol"] == "native-learner-geometry-v1" and result["float32_geometry"] is True
            and result["policy_executed"] is False and result["gpu_runtime_validated"] is False, "Wrong geometry executable output")
    return result


def prepare(out, path_panel=None):
    source_guard(ROOT / "src/pufferl.cu")
    protocol = audit.load(path_panel) if path_panel else None
    out = out.resolve()
    if path_panel: require(not out.is_relative_to(path_panel.parent) and not path_panel.parent.is_relative_to(out), "Output overlaps panel")
    out.mkdir(parents=True, exist_ok=False)
    try:
        cases = []
        for task, control in CONTROLS.items():
            cases.append(dict(name="control-"+control, task=task,
                kind="versioned-local-state-control" if control == "snakebench" else "current-stock-state-config",
                files=[ROOT / "config/default.ini", ROOT / "config" / f"{control}.ini"]))
            cases.append(dict(name="common-"+task, task=task, kind="untuned-common-recipe",
                files=[ROOT / "config/default.ini", ROOT / "config" / f"{task}.ini", ROOT / "ocean" / task / "compare.ini"]))
        for name in ("stock-pilot.DN7C2JF6", "stock-pilot.tgKBYrNs"):
            directory = ROOT / "research/results/flappycnn" / name / "metrics"
            files = sorted(directory.rglob("*.ini"))
            require(len(files) == 1, "Need the retained executed Flappy training INI")
            cases.append(dict(name="executed-"+name, task="flappycnn", kind="historical-executed-config-not-new-training", files=files))
        if protocol:
            for i, job in enumerate(protocol["jobs"]):
                cases.append(dict(name=f"panel-job-{i:04d}", task=job["environment"], kind="prepared-panel-job",
                    files=[path_panel.parent / job["config"], path_panel.parent / job["config"].replace("default.ini", job["environment"]+".ini")],
                    job_index=i, representation=job["representation"], seed=job["seed"], model=job["model"]))
        tools = [Path(__file__), ROOT / "research/learner_geometry.c", ROOT / "research/audit_pixel_panel.py",
                 ROOT / "research/prepare_pixel_robustness.py", ROOT / "ocean/connect4cnn/claim.py", ROOT / "src/ini.h", ROOT / "src/pufferl.cu"]
        inputs = {p for case in cases for p in case["files"]} | set(tools)
        if path_panel:
            inputs.add(path_panel)
            require(sha(ROOT / "src/pufferl.cu") == protocol["source_sha256"]["src/pufferl.cu"]
                    and sha(ROOT / "src/ini.h") == protocol["source_sha256"]["src/ini.h"], "Panel has a different learner/parser; audit that source separately")
        receipts = {str(p): sha(p) for p in inputs}
        for path in tools:
            target = out / "source" / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        binary = out / "learner_geometry"
        command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                   str(out / "source/research/learner_geometry.c"), "-lm", "-o", str(binary)]
        with (out / "build.txt").open("x") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=60)
        compiler = subprocess.check_output(["cc", "--version"], text=True, timeout=20)
        binary_hash = sha(binary); records = []
        for case in cases:
            paths = []
            for i, path in enumerate(case["files"]):
                target = out / "configs" / case["name"] / f"{i}.ini"; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target); require(sha(target) == receipts[str(path)], "Input changed during copy")
                paths.append(target)
            merged = read(paths)
            result = geometry(binary, paths)
            require(result["environment"] == merged.get("base", "env_name"), "Geometry environment differs")
            record = {k: v for k, v in case.items() if k != "files"}
            record.update(configs=[str(p.relative_to(out)) for p in paths], geometry=result,
                policy_core=dict(hidden_size=merged.getint("policy", "hidden_size"), num_layers=merged.getint("policy", "num_layers")),
                async_actor=merged.getint("base", "async"), num_threads=merged.getint("vec", "num_threads"),
                learning_settings=dict(merged["train"]))
            records.append(record)
        if protocol:
            groups = {}
            for record in records:
                if record["kind"] != "prepared-panel-job": continue
                key = record["task"], record["representation"], record["seed"]
                require(record["geometry"]["nonzero_training_updates"] and record["geometry"]["dropped_requested_decisions"] == 0,
                        "Prepared panel has zero updates or an unreported rounded budget")
                groups.setdefault(key, []).append(record["geometry"])
            require(all(len(group) == 4 and all(value == group[0] for value in group) for group in groups.values()), "Paired optimizer geometry differs")
            audit.load(path_panel)
        require(sha(binary) == binary_hash and all(sha(Path(name)) == digest for name, digest in receipts.items()), "Source/config/executable changed")
        save(out / "records.json", records)
        columns = ("name", "task", "kind", "agents_per_rank", "horizon", "minibatch_decisions", "rollout_decisions_per_rank",
            "replay_ratio_f32", "optimizer_minibatches_per_epoch", "effective_replay_ratio", "optimizer_updates_per_1000_local_decisions",
            "rows_covered_per_rollout", "epochs", "actual_global_decisions", "dropped_requested_decisions", "nonzero_training_updates")
        with (out / "geometry.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for record in records: writer.writerow({key: record.get(key, record["geometry"].get(key)) for key in columns})
        source_receipts = file_hashes(out / "source")
        report = dict(protocol="learning-recipe-geometry-audit-v1", status="configuration-arithmetic-audited",
            policy_executed=False, gpu_runtime_validated=False, learning_calibrated=False, publication_confirmation_launchable=False,
            cases=len(records), panel_jobs=len(protocol["jobs"]) if protocol else 0, binary_sha256=binary_hash,
            compile_command=command, compiler=compiler, source_sha256=source_receipts, original_inputs_sha256=receipts,
            files_sha256={name: digest for name, digest in file_hashes(out).items() if name != "learner_geometry"},
            warning="Arithmetic/configuration receipt only; source markers aren't a complete control-flow proof. No optimizer, model, learning or speed was executed.")
        save(out / "audit.json", report)
        return report
    except Exception as error:
        save(out / "failure.json", dict(status="failed-audit", error=str(error), policy_executed=False))
        raise


def inspect(path):
    root = path.parent; value = json.loads(path.read_text())
    require(not (root / "failure.json").exists() and value["protocol"] == "learning-recipe-geometry-audit-v1"
            and value["status"] == "configuration-arithmetic-audited", "Wrong/failed recipe audit")
    for key in ("policy_executed", "gpu_runtime_validated", "learning_calibrated", "publication_confirmation_launchable"):
        require(value[key] is False, "Arithmetic upgraded into unsupported runtime/learning evidence")
    entries = file_hashes(root); entries.pop("audit.json", None); entries.pop("learner_geometry", None)
    require(entries == value["files_sha256"] and file_hashes(root / "source") == value["source_sha256"], "Changed recipe audit packet")
    source_guard(root / "source/src/pufferl.cu")
    records = json.loads((root / "records.json").read_text())
    require(len(records) == value["cases"] and sum(r["kind"] == "prepared-panel-job" for r in records) == value["panel_jobs"], "Wrong case allocation")
    if (root / "learner_geometry").exists(): require(sha(root / "learner_geometry") == value["binary_sha256"], "Changed arithmetic executable")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare"); create.add_argument("--out", type=Path, required=True); create.add_argument("--panel", type=Path)
    check = sub.add_parser("inspect"); check.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out, args.panel.resolve() if args.panel else None) if args.command == "prepare" else inspect(args.audit.resolve())
    print(json.dumps({k: result[k] for k in ("status", "cases", "panel_jobs", "policy_executed", "learning_calibrated")}, indent=2))


if __name__ == "__main__": main()
