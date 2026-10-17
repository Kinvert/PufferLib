"""Validate locked Pong models and report native search/evaluation receipts. No training/search logic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "connect4cnn"))
from wandb_sidecar import architecture, display_name, read_ini, scalar, trials

PARAMS = {"flex_quality": 160224, "nature_cnn": 138016, "impala_cnn": 269984, "impoola_cnn": 151200}
ENCODERS = {"flex_quality": 4, "nature_cnn": 2, "impala_cnn": 0, "impoola_cnn": 0}
DIMENSIONS = {"total_timesteps", "learning_rate", "ent_coef", "gamma", "gae_lambda", "replay_ratio", "clip_coef", "vf_coef", "max_grad_norm"}
EVAL = re.compile(r"^CUDA_EVAL env=pongcnn score=([-+\d.eE]+) perf=([-+\d.eE]+) games=(\d+) params=(\d+)$", re.M)


def validate(root):
    variant = (root / "variant.txt").read_text().strip()
    ini = read_ini(root / "config/default.ini")
    ini.read(root / "config/pongcnn.ini")
    fixed = read_ini(Path(__file__).with_name("compare.ini"))
    expected = dict(fixed["policy"])
    expected["encoder"] = str(ENCODERS[variant])
    assert dict(ini["policy"]) == expected, "Encoder/core architecture must stay locked"
    assert {s for s in ini if s.startswith("sweep.")} == {"sweep.train." + k for k in DIMENSIONS}, "Unexpected search dimensions"
    for section in ("env", "vec", "selfplay"):
        for key, value in fixed[section].items():
            assert ini.get(section, key) == value, (section, key, "must stay fixed")
    assert ini.getint("train", "gpus") == ini.getint("sweep", "gpus") == 1
    assert ini.getint("base", "eval_episodes") == 0
    assert ini.get("sweep", "metric") == "perf"
    for key in DIMENSIONS:
        value = ini.getfloat("train", key)
        assert ini.getfloat("sweep.train." + key, "min") <= value <= ini.getfloat("sweep.train." + key, "max"), key
    return ini, variant


def report(root, require_evals=False, allow_eval_failures=False):
    ini, variant = validate(root)
    rows = trials(root)
    if not rows:
        raise ValueError("No completed trials; preserve failure receipts and investigate")
    fields = ["name", "run_id", "variant", "steps", "train_perf", "native_cost_s", "native_avg_sps", "params",
              "eval_perf", "eval_score", "eval_games", "eval_status", "training_pareto", "architecture_sha256", "checkpoint_sha256"]
    flat = []
    expected = {f"policy.{k}": scalar(v) for k, v in ini["policy"].items()}
    failures = []
    for row in rows:
        assert all(row["config"].get(k) == v for k, v in expected.items()), "Trial changed fixed architecture/core"
        assert row["params"] == PARAMS[variant], "Unexpected parameter count"
        weights = np.fromfile(root / row["checkpoint"], dtype=np.float32)
        assert np.isfinite(weights).all(), "Nonfinite checkpoint"
        # Check every fixed setting, not only the shape. Native bookkeeping keys change per job.
        mutable = {"base.run_id", "base.result_fd", "base.gpu_offset", "base.env_name"} | {"train." + k for k in DIMENSIONS}
        for section in ini.sections():
            if section.startswith("sweep"):
                continue
            for key, value in ini[section].items():
                full = section + "." + key
                if full not in mutable:
                    assert row["config"].get(full) == scalar(value.strip("\"'")), (row["run_id"], full, "changed")
        evaluation = root / "evaluations" / row["run_id"]
        status = "pending"
        parsed = {}
        if (evaluation / "exit-code.txt").exists():
            matches = EVAL.findall((evaluation / "eval.txt").read_text())
            if (evaluation / "exit-code.txt").read_text().strip() == "0" and len(matches) == 1:
                score, perf, games, params = matches[0]
                assert np.isfinite([float(score), float(perf)]).all()
                eval_ini = read_ini(evaluation / "config/default.ini")
                requested_games = int((evaluation / "requested-games.txt").read_text())
                assert int(params) == row["params"] and int(games) >= requested_games and 0 <= float(perf) <= 1
                assert all(scalar(eval_ini.get("policy", k[7:])) == v for k, v in expected.items())
                parsed = dict(score=float(score), perf=float(perf), games=int(games), seed=29173,
                              checkpoint_sha256=row["checkpoint_sha256"])
                (root / "evaluations" / (row["run_id"] + ".json")).write_text(json.dumps(parsed, indent=2) + "\n")
                status = "ok"
            else:
                status = "failed"
        if require_evals and status == "pending":
            raise ValueError(f"Missing evaluation attempt: {row['run_id']}")
        if status == "failed":
            failures.append(row["run_id"])
        dominated = any(r["cost"] <= row["cost"] and r["score"] >= row["score"] and
                        (r["cost"] < row["cost"] or r["score"] > row["score"]) for r in rows)
        flat.append(dict(name=display_name(root, row["index"]), run_id=row["run_id"], variant=variant,
                         steps=row["steps"], train_perf=row["score"], native_cost_s=row["cost"], native_avg_sps=row["native_avg_sps"],
                         params=row["params"], eval_perf=parsed.get("perf", ""), eval_score=parsed.get("score", ""),
                         eval_games=parsed.get("games", ""), eval_status=status, training_pareto=not dominated,
                         architecture_sha256=row["architecture_sha256"], checkpoint_sha256=row["checkpoint_sha256"]))
    assert len({r["architecture_sha256"] for r in rows}) == 1
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    (root / "trials.tsv").write_text("".join(f"{r['run_id']}\t{r['checkpoint']}\n" for r in rows))
    native_failures = [line for line in (root / "sweep.log").read_text().splitlines() if "failed; marking sample bad" in line]
    receipt = dict(variant=variant, completed=len(rows), architectures=1, native_failures=native_failures,
                   evaluation_failures=failures, status=(root / "status.txt").read_text().strip(),
                   config_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "config").glob("*.ini"))})
    if receipt["status"] == "completed":
        assert len(rows) == ini.getint("sweep", "max_runs"), "Missing completed native trials"
    (root / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (root / "evaluation-failures.txt").write_text("".join(run_id + "\n" for run_id in failures))
    lines = [f"# Pong fixed-architecture development search: {variant}", "",
             f"{len(rows)} completed trials; one verified architecture; {len(native_failures)} native worker failures; {len(failures)} evaluation failures. Training status: {receipt['status']}.",
             "PufferLib Pong, not ALE Pong. Perf is episode-averaged fraction of points won, not match win rate.",
             "Training seed 9173; separate development evaluation seed 29173. This is hyperparameter discovery, not fresh-seed confirmation.",
             "PROTEIN uses downsampled training curves. Native cost excludes evaluation/upload and uses adjusted uptime rounded in stdout.",
             "Sweep wall includes search/worker overhead; see sweep-wall.txt. W&B uploads happen after the family, outside timing.",
             "A resource cap may interrupt a trial; partial checkpoints/logs remain but are not labeled completed observations.", "",
             "| Run | Decisions | Train point fraction | Native seconds | Native SPS | Eval point fraction | Eval status |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in flat:
        perf = f"{row['eval_perf']:.2%}" if row["eval_perf"] != "" else ""
        lines.append(f"| {row['name']} | {row['steps']:,} | {row['train_perf']:.2%} | {row['native_cost_s']:.2f} | {row['native_avg_sps']:,.0f} | {perf} | {row['eval_status']} |")
    lines += ["", "All per-trial native curves and effective settings: metrics/pongcnn/*.ini. Full final-point data: results.csv.",
              "Source/build provenance: parent campaign source.sha256, revision.txt, working.patch; per-family inputs.sha256/build-command.txt.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    if failures and require_evals and not allow_eval_failures:
        raise ValueError(f"{len(failures)} evaluation failures; see saved receipts")
    print(f"{variant}: validated {len(rows)} completed trials, one fixed architecture", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--require-evals", action="store_true")
    parser.add_argument("--allow-eval-failures", action="store_true", help="Retain attempted evaluation failures without aborting other families; all other audits remain strict")
    args = parser.parse_args()
    if args.validate:
        validate(args.root)
    else:
        report(args.root, args.require_evals, args.allow_eval_failures)
