"""Audit the fixed hardware run and write its final-checkpoint handoff table."""
import csv
import configparser
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

run = Path(sys.argv[1]).resolve()
protocol = json.loads((run / "protocol.json").read_text())
finished = json.loads((run / "finished.json").read_text())
jobs = json.loads((run / "jobs.json").read_text())
manifest = json.loads((run / "source/ocean/connect4cnn/confirmation.json").read_text())
with (run / "results.csv").open() as stream:
    rows = list(csv.DictReader(stream))
names = {"flex_quality": "Ours quality", "nature_cnn": "Nature",
         "impala_cnn": "IMPALA", "impoola_cnn": "Impoola"}
assert finished == {"status": "ok", "jobs": 4, "evaluations": 52, "logging_failures": 0}, finished
assert len(jobs) == 4 and len(rows) == 52
assert set(protocol["variants"]) == set(names)
assert protocol["steps"] == protocol["completed_steps"] == 13312000
assert protocol["seeds"] == [173] and protocol["eval_seed"] == 20173
assert protocol["precision"] == "float32" and protocol["build_arch"] == "sm_120"
assert protocol["eval_games"] == 1024 and protocol["checkpoints"] == 13

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

for name, expected in protocol["source_sha256"].items():
    assert digest(run / "source" / name) == expected, name
assert digest(run / "recipe.ini") == protocol["recipe_sha256"]
for name, expected in protocol["binary_sha256"].items():
    assert digest(run / name) == expected, name

table = ["| Model | Decisions | Held-out wins | Train wall seconds | Process SPS | Native average SPS | Parameters |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for job in jobs:
    variant = job["variant"]
    assert job["status"] == "ok" and job["seed"] == 173
    resolved = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    resolved.read(run / f"{variant}-s173/metrics/connect4cnn/trial.ini")
    assert resolved.getint("policy", "hidden_size") == 128
    assert resolved.getint("policy", "num_layers") == 1
    assert resolved.getint("env", "representation") == 0
    assert resolved.getint("train", "gpus") == 1
    curve = sorted((row for row in rows if row["variant"] == variant), key=lambda row: int(row["steps"]))
    assert [int(row["steps"]) for row in curve] == protocol["checkpoint_steps"]
    for row in curve:
        assert int(row["seed"]) == 173 and int(row["eval_seed"]) == 20173
        assert int(row["games"]) >= 1024 and 0 <= float(row["win_rate"]) <= 1
        assert int(row["params"]) == manifest["parameters"][variant]
        checkpoint = run / row["checkpoint"]
        assert digest(checkpoint) == job["checkpoint_sha256"][checkpoint.name]
        weights = np.fromfile(checkpoint, dtype=np.float32)
        assert weights.size == int(row["params"]) and np.isfinite(weights).all()
    row = curve[-1]
    table.append(f"| {names[variant]} | {int(row['steps']):,} | {float(row['win_rate']):.2%} | "
                 f"{float(row['train_process_wall_s']):.3f} | {float(row['process_sps']):,.0f} | "
                 f"{float(row['native_avg_sps']):,.0f} | {int(row['params']):,} |")

lines = ["# RTX 5090 fixed comparison", "",
         "Completed four jobs and 52 held-out evaluations with zero failures.", "",
         *table, "", "## Protocol and hardware", "",
         f"- Revision: `{protocol['revision']}`; measured source hashes verified against the captured files.",
         "- RTX 5090, AMD Ryzen 9 9950X3D, CUDA compiler 13.1.115, NVIDIA driver 580.105.08.",
         "- Native C/CUDA float32; seed 173; held-out seed 20173; common H128/L1 core and representation 0.",
         "- Command: `source build/connect4cnn/runtime-5090.sh && bash ocean/connect4cnn/hardware_compare.sh --full`.",
         "- Existing F-Zero NCCL reused read-only; linker alias lives in this checkout. Exact paths/hashes are in `validation-5090/setup.txt`.",
         "- Source, recipe, binary, and all 52 checkpoint hashes verified. All checkpoint arrays have the expected parameter count and finite float32 values.", "",
         "## Validation", "",
         "- 17 configuration/runner tests passed.",
         "- Ten representation pixel/reset/parity/repeatability checks and ASan/UBSan passed.",
         "- Nature, Flex, IMPALA and Impoola numerical/gradient and eager/graph repeatability checks passed.",
         "- Canary `compare.bfi2gjs_`: four successful jobs and 16 evaluations. Canary scores/timing are excluded from this table.", "",
         "## Interpretation limits", "",
         "The table uses each model's final 13.312M-decision checkpoint. Process SPS includes trainer startup and checkpoint writes; native average SPS uses the trainer's adjusted timer. Builds and post-training evaluations are excluded from training wall time. Parameter counts differ.", "",
         "This is one training seed on one game. It does not establish score superiority or a general-purpose CNN ranking. No same-revision RTX 5060 comparison has been run here; historical 5060 results use different captured source. Desktop graphics processes were present, with no competing compute job at launch.", "",
         "Full checkpoint curves and actual batched evaluation counts are in `REPORT.md` and `results.csv`. Raw evidence is retained in this run directory.", ""]
(run / "SUMMARY-5090.md").write_text("\n".join(lines))
(run / "audit-5090.json").write_text(json.dumps({"status": "ok", "jobs": 4, "evaluations": 52,
    "finite_checkpoints": 52, "source_hashes_verified": len(protocol["source_sha256"]),
    "binary_hashes_verified": 4, "checkpoint_hashes_verified": 52}, indent=2) + "\n")
print("\n".join(lines))
