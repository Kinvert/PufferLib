"""Audit a completed flex sweep and evaluate training-selected saved checkpoints.

No training or W&B writes. Run with the existing runtime_env.sh sourced.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ocean/connect4cnn"))
from compare import EVAL
from wandb_sidecar import trials, display_name, read_ini, scalar


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all-final", action="store_true", help="Evaluate every final checkpoint; resume saved receipts")
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rows = trials(root)
    finished = json.loads((root / "finished.json").read_text())
    assert finished["status"] == "ok" and finished["completed"] == len(rows)
    protocol = json.loads((root / "protocol.json").read_text())
    assert sha(root / "cnn") == protocol["binary_sha256"]
    assert len({r["config"]["base.seed"] for r in rows}) == 1
    refs = []
    for campaign in ("compare.2s__8t8l", "compare.l6d5sbk2"):
        archive = ROOT / "research/results/connect4cnn" / campaign
        for r in csv.DictReader((archive / "results.csv").open()):
            refs.append({k: scalar(v) for k, v in r.items()})

    # Compare effective learner, core, environment and vectorization settings.
    def fixed(config):
        return {k: v for k, v in config.items()
                if (k.startswith(("train.", "vec.", "env.", "selfplay."))
                    and k != "train.total_timesteps")
                or k in ("policy.hidden_size", "policy.num_layers", "base.async",
                         "base.cudagraphs", "base.reset_every_horizon")}

    reference_files = sorted((ROOT / "research/results/connect4cnn/compare.2s__8t8l").glob("*/metrics/connect4cnn/trial.ini"))
    reference_files += sorted((ROOT / "research/results/connect4cnn/compare.l6d5sbk2").glob("*/metrics/connect4cnn/trial.ini"))
    anchor = fixed(rows[0]["config"])
    differences = {}
    for r in rows:
        assert fixed(r["config"]) == anchor, r["run_id"]
        assert np.isfinite(np.fromfile(root / r["checkpoint"], dtype=np.float32)).all(), r["run_id"]
    for path in reference_files:
        ini = read_ini(path)
        config = {f"{s}.{k}": scalar(v) for s in ini.sections() if s != "metrics" for k, v in ini[s].items()}
        other = fixed(config)
        differences[str(path.relative_to(ROOT))] = {
            k: [anchor.get(k), other.get(k)] for k in anchor.keys() | other.keys()
            if anchor.get(k) != other.get(k)}
    assert not any(differences.values()), differences

    winner = max(rows, key=lambda r: r["score"])
    fast = min((r for r in rows if r["score"] >= .8), key=lambda r: r["cost"])
    small = min((r for r in rows if r["score"] >= .85), key=lambda r: (r["params"], -r["score"]))
    selected = list({r["index"]: r for r in (winner, fast, small)}.values())
    compact = lambda r: {"name": display_name(root, r["index"]), **{k: r[k] for k in (
        "index", "run_id", "steps", "score", "cost", "native_avg_sps", "params", "architecture", "checkpoint_sha256")}}
    repeats = defaultdict(list)
    for r in rows:
        # Effective active shape, exact requested budget and all fixed settings.
        key = (r["architecture_sha256"], r["config"]["train.total_timesteps"], r["config"]["base.seed"])
        repeats[key].append(r)
    repeated = [{"count": len(group), "distinct_checkpoints": len({r["checkpoint_sha256"] for r in group}),
                 "indices": [r["index"] for r in group], "requested_steps": key[1]}
                for key, group in repeats.items() if len(group) > 1]
    summary = {"campaign": root.name, "completion": finished, "binary_sha256": protocol["binary_sha256"],
               "analysis_source_sha256": sha(Path(__file__).resolve()),
               "finite_final_checkpoints": len(rows), "fixed_config_differences": differences,
               "unique_architectures": len({r["architecture_sha256"] for r in rows}),
               "total_decisions": sum(r["steps"] for r in rows),
               "depth_counts": dict(Counter(r["architecture"]["policy.cnn_depth"] for r in rows)),
               "budget_counts": {"under_6.6M": sum(r["steps"] < 6600000 for r in rows),
                                 "at_least_6.6M": sum(r["steps"] >= 6600000 for r in rows)},
               "score_counts": {str(t): sum(r["score"] >= t for r in rows) for t in (.5, .8, .85, .9)},
               "selected_before_evaluation": [compact(r) for r in selected],
               "winner_checkpoint_wall": [{"steps": int(p.stem), "mtime": p.stat().st_mtime,
                   "seconds_from_launch_id": p.stat().st_mtime - int(winner["run_id"].split("_")[1]) / 1000}
                   for p in sorted((root / winner["checkpoint"]).parent.glob("*.bin"))],
               "selection_rules": ["maximum training score", "minimum cost with training score >= .8",
                                   "minimum parameters with training score >= .85; score breaks ties"],
               "exact_active_configuration_repeats": repeated,
               "baseline_final": [r for r in refs if r["steps"] == 13279232]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not (args.evaluate or args.all_final):
        return
    existing_path = out / "evaluations.csv"
    existing = list(csv.DictReader(existing_path.open())) if existing_path.exists() else []
    completed = {(int(r["index"]), int(r["steps"]), r["checkpoint_sha256"]) for r in existing}
    gpu = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi"], capture_output=True, text=True, check=True)
    (out / "gpu.txt").write_text(gpu.stdout)
    fields = ["name", "index", "eval_seed", "steps", "win_rate", "score", "games", "params",
              "checkpoint_sha256", "eval_wall_s", "training_native_cost_s", "checkpoint"]
    with existing_path.open("a", newline="") as csvfile, (out / "commands.jsonl").open("a") as commands:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if not existing:
            writer.writeheader()
        for r in (rows if args.all_final else selected):
            checkpoints = sorted((root / r["checkpoint"]).parent.glob("*.bin")) if r is winner and not args.all_final else [root / r["checkpoint"]]
            for checkpoint in checkpoints:
                if (r["index"], int(checkpoint.stem), sha(checkpoint)) in completed:
                    continue
                config = {k: v for k, v in r["config"].items() if not k.startswith("sweep.")}
                config.update({"base.result_fd": 0, "base.seed": 10073, "base.eval_episodes": 1024,
                               "base.load_model_path": str(checkpoint), "base.log_dir": str(out / "metrics"),
                               "sweep.metric": "score"})
                command = [str(root / "cnn"), "eval", "--headless"] + [f"--{k}={v}" for k, v in config.items()]
                log = out / f"eval-{r['index']:03d}-{checkpoint.stem}.txt"
                commands.write(json.dumps({"argv": command, "cwd": str(root), "log": log.name}) + "\n")
                commands.flush()
                start = time.perf_counter()
                with log.open("w") as output:
                    subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=60)
                elapsed = time.perf_counter() - start
                match = EVAL.search(log.read_text())
                assert match and match[1] == "connect4cnn" and int(match[5]) == r["params"], log
                result = dict(name=display_name(root, r["index"]), index=r["index"], eval_seed=10073,
                              steps=int(checkpoint.stem), win_rate=float(match[3]), score=float(match[2]),
                              games=int(match[4]), params=int(match[5]), checkpoint_sha256=sha(checkpoint),
                              eval_wall_s=elapsed, training_native_cost_s=r["cost"] if int(checkpoint.stem) == r["steps"] else "",
                              checkpoint=str(checkpoint.relative_to(root)))
                writer.writerow(result)
                csvfile.flush()
                print(f"{result['name']} steps={result['steps']} eval={result['win_rate']:.2%} games={result['games']}", flush=True)


if __name__ == "__main__":
    main()
