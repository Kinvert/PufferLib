"""Archive one fixed four-model hardware run and plot its descriptive frontier."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from compare import EVAL, sha256
from claim_frontier import report


def check(ok, message):
    if not ok:
        raise ValueError(message)


def archive(run, output, gpu):
    check(not output.exists(), "Use a fresh output directory")
    protocol = json.loads((run / "protocol.json").read_text())
    jobs = json.loads((run / "jobs.json").read_text())
    finished = json.loads((run / "finished.json").read_text())
    models = ["flex_quality", "nature_cnn", "impala_cnn", "impoola_cnn"]
    check(finished == dict(status="ok", jobs=4, evaluations=52, logging_failures=0), "Incomplete comparison")
    check(protocol["seeds"] == [173] and protocol["eval_seed"] == 20173, "Unexpected seed panel")
    check(set(protocol["variants"]) == set(models), "Unexpected models")
    check(protocol["steps"] == protocol["completed_steps"] == 13312000, "Unexpected budget")
    check(protocol["checkpoint_steps"] == list(range(1024000, 13312001, 1024000)), "Unexpected checkpoints")
    check(protocol["environment_rules"] == "connect4-full-board-draw-v2", "Wrong game revision")
    check(protocol["precision"] == "float32" and protocol["evaluation_protocol"] == "pooled-v1", "Wrong protocol")
    check(f"RTX {gpu}" in (run / "gpu.txt").read_text(), "Wrong hardware")
    for name, digest in protocol["source_sha256"].items():
        check(sha256(run / "source" / name) == digest, f"Changed source: {name}")
    for model, digest in protocol["binary_sha256"].items():
        check(sha256(run / model) == digest, f"Changed binary: {model}")
    selected = json.loads((run / "source/ocean/connect4cnn/confirmation.json").read_text())
    by_model = {job["variant"]: job for job in jobs}
    check(len(jobs) == len(by_model) == 4 and all(j["status"] == "ok" for j in jobs), "Failed or duplicate job")
    with (run / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == 52, "Missing observations")
    points, seen = [], set()
    for row in rows:
        model, step = row["variant"], int(row["steps"])
        check((model, step) not in seen, "Duplicate observation")
        seen.add((model, step))
        check(int(row["seed"]) == 173 and int(row["eval_seed"]) == 20173, "Wrong evaluation seed")
        checkpoint = run / row["checkpoint"]
        expected = f"{model}-s173/checkpoints/connect4cnn/trial/{step:016d}.bin"
        check(row["checkpoint"] == expected, "Wrong checkpoint path")
        check(sha256(checkpoint) == by_model[model]["checkpoint_sha256"][checkpoint.name], "Checkpoint changed")
        weights = np.fromfile(checkpoint, dtype=np.float32)
        check(weights.size == selected["parameters"][model] == int(row["params"]) and np.isfinite(weights).all(), "Invalid weights")
        matches = EVAL.findall((run / f"{model}-s173/eval-{step:016d}.log").read_text())
        check(len(matches) == 1, "Missing/duplicate evaluation summary")
        env, score, perf, games, params = matches[0]
        check(env == "connect4cnn" and float(score) == float(row["score"]) and float(perf) == float(row["win_rate"])
              and int(games) == int(row["games"]) >= 1024 and int(params) == weights.size, "CSV/log mismatch")
        point = dict(model=model, seed=173, steps=step, seconds=float(row["checkpoint_wall_s"]),
                     win_rate=float(perf), games=int(games), parameters=weights.size,
                     process_sps=float(row["process_sps"]), native_sps=float(row["native_avg_sps"]),
                     train_seconds=float(row["train_process_wall_s"]))
        check(all(np.isfinite(v) for k, v in point.items() if k != "model"), "Nonfinite measurement")
        check(0 < point["seconds"] <= point["train_seconds"] and 0 <= point["win_rate"] <= 1, "Invalid observation")
        points.append(point)
    check(seen == {(m, k) for m in models for k in protocol["checkpoint_steps"]}, "Incomplete panel")
    for model in models:
        curve = sorted((p for p in points if p["model"] == model), key=lambda p: p["steps"])
        check(all(a["seconds"] < b["seconds"] for a, b in zip(curve, curve[1:])), "Nonmonotonic time")
    # Preserve small receipts and original reports; binaries/weights stay in build/.
    output.mkdir(parents=True)
    suffixes = {".md", ".csv", ".json", ".jsonl", ".ini", ".txt", ".log", ".sha256", ".c", ".cu", ".h", ".py", ".sh"}
    for path in run.rglob("*"):
        relative = path.relative_to(run)
        if path.is_file() and path.suffix in suffixes and not {"checkpoints", "comparison-sidecar", "__pycache__"}.intersection(relative.parts):
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    analysis = output / "analysis"
    analysis.mkdir()
    chart_protocol = dict(models=models, training_seeds=[173], checkpoint_steps=protocol["checkpoint_steps"],
                          purpose=f"RTX {gpu}: one-seed exploratory pooled-v1 comparison; approximate checkpoint times",
                          environment_rules=protocol["environment_rules"], timing="launch wall clock to checkpoint file mtime (approximate)")
    report(analysis, chart_protocol, points, [])
    (analysis / "hardware.json").write_text(json.dumps(dict(gpu=gpu, source_revision=protocol["revision"],
        evaluation_protocol="pooled-v1", local_run=str(run), checkpoints_checked=52,
        analyst_source_sha256={name: sha256(ROOT / name) for name in
            ("research/report_hardware_run.py", "research/claim_frontier.py", "research/claim_frontier.html")}), indent=2)+"\n")
    print(f"Archived RTX {gpu} results: {output}")
    print(f"Whole observed frontier: {analysis / 'curves.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("5060", "5090"), required=True)
    args = parser.parse_args()
    archive(args.run.resolve(), args.output.resolve(), args.gpu)
