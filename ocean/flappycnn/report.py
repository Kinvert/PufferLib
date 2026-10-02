"""External canary configuration/evidence reporting. No model, training or search."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "connect4cnn"))
from wandb_sidecar import read_ini, scalar

VARIANTS = {"state": ("flappy", 0), "flex_quality": ("flappycnn", 4),
            "nature_cnn": ("flappycnn", 2), "impala_cnn": ("flappycnn", 0), "impoola_cnn": ("flappycnn", 0)}


def validate(root):
    reference = read_ini(Path(__file__).with_name("compare.ini"))
    for variant, (environment, encoder) in VARIANTS.items():
        ini = read_ini(root / variant / "config/default.ini")
        ini.read(root / variant / "config" / f"{environment}.ini")
        for section in ("env", "vec", "train", "selfplay"):
            for key, value in reference[section].items():
                assert scalar(ini.get(section, key)) == scalar(value), (variant, section, key)
        for key, value in reference["base"].items():
            assert scalar(ini.get("base", key)) == scalar(value), (variant, "base", key)
        for key, value in reference["policy"].items():
            expected = encoder if key == "encoder" else scalar(value)
            assert scalar(ini.get("policy", key)) == expected, (variant, key)
        assert ini.get("base", "env_name") == environment
        assert ini.getint("train", "gpus") == 1 and ini.getint("base", "eval_episodes") == 0
        assert not any(section.startswith("sweep.") for section in ini)
    return reference


def report(root):
    reference = validate(root)
    steps = reference.getint("train", "total_timesteps")
    rows = []
    for variant, (environment, encoder) in VARIANTS.items():
        job = root / variant
        checkpoint = job / "checkpoints" / environment / "trial" / f"{steps:016d}.bin"
        data = checkpoint.read_bytes()
        assert data and len(data) % 4 == 0
        weights = np.frombuffer(data, dtype=np.float32)
        assert np.isfinite(weights).all(), (variant, "nonfinite checkpoint")
        checkpoints = sorted(checkpoint.parent.glob("*.bin"))
        batch = reference.getint("vec", "total_agents") * reference.getint("train", "horizon")
        cadence = reference.getint("base", "checkpoint_interval") * batch
        expected = list(range(cadence, steps + 1, cadence))
        if not expected or expected[-1] != steps:
            expected.append(steps)
        assert [int(path.stem) for path in checkpoints] == expected, (variant, "checkpoint cadence")
        for path in checkpoints:
            snapshot = np.fromfile(path, dtype=np.float32)
            assert snapshot.size == weights.size and np.isfinite(snapshot).all(), (variant, path.name)
        metrics = read_ini(job / "metrics" / environment / "trial.ini")
        assert metrics.getint("policy", "encoder") == encoder
        # Audit executed settings, not merely prepared configs.
        for section in ("env", "vec", "train", "selfplay"):
            for key, value in reference[section].items():
                assert scalar(metrics.get(section, key)) == scalar(value), (variant, section, key, "executed")
        for key in ("hidden_size", "num_layers"):
            assert metrics.getint("policy", key) == reference.getint("policy", key)
        for key in ("async", "cudagraphs", "reset_every_horizon", "seed", "checkpoint_interval", "eval_episodes"):
            assert scalar(metrics.get("base", key)) == scalar(reference.get("base", key)), (variant, key, "executed")
        recorded_steps = [float(v) for v in metrics.get("metrics", "agent_steps").split(",")]
        assert int(recorded_steps[-1]) == steps
        series = {k: [float(v) for v in values.split(",")] for k, values in metrics["metrics"].items()}
        uptime = series["uptime"][-1]
        wall = float((job / "train-wall.txt").read_text())
        assert wall > 0 and uptime > 0
        match = re.findall(r"^CUDA_EVAL env=" + environment + r" score=([-+\d.eE]+) perf=([-+\d.eE]+) games=(\d+) params=(\d+)$", (job / "eval.txt").read_text(), re.M)
        assert len(match) == 1, (variant, "missing evaluation completion")
        score, perf, games, parameters = match[0]
        assert int(games) >= 64 and int(parameters) == weights.size
        assert np.isfinite([float(score), float(perf)]).all() and 0 <= float(perf) <= 1
        assert float(score) >= 0
        rows.append(dict(model=variant, steps=steps, process_train_seconds=wall,
                         process_sps=steps/wall, native_uptime_seconds=uptime,
                         native_avg_sps=steps/uptime, native_last_sps=series["SPS"][-1],
                         evaluation_score=float(score), evaluation_perf=float(perf),
                         completed_episodes=int(games), params=int(parameters),
                         checkpoint_sha256=hashlib.sha256(data).hexdigest()))
    with (root / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# FlappyCNN native plumbing canary", "", "Five matched short runs. No search or quality confirmation.",
             "Score is mean pipes passed; perf is mean min(pipes/20, 1), not a win rate.",
             "Evaluation uses pooled native v1 accounting (at least 64 completed episodes), not exact quotas.",
             "Process SPS includes startup/checkpoint writes; native average SPS uses the saved native uptime.", "",
             "| Model | Decisions | Process seconds | Process SPS | Native avg SPS | Eval score | Eval perf | Episodes | Parameters |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['steps']:,} | {row['process_train_seconds']:.3f} | {row['process_sps']:,.0f} | {row['native_avg_sps']:,.0f} | {row['evaluation_score']:.3f} | {row['evaluation_perf']:.3f} | {row['completed_episodes']} | {row['params']:,} |")
    lines += ["", "Raw native metrics, commands, configs, checkpoint hashes, binary/source receipts and GPU metadata are retained in this directory.",
              "These short, startup-dominated timings are not a Pareto claim. State includes velocity and offscreen future-pipe features absent from pixels.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(args.root)
        print("PASS: five models share Flappy environment/learner/core/budget settings")
    else:
        report(args.root)


if __name__ == "__main__":
    main()
