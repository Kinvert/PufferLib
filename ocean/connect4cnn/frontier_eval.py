"""Reevaluate all saved checkpoints, retaining historical training-time receipts.

Experiment/report glue only. Uses deterministic_eval.py and native GPU inference.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import deterministic_eval as ev
from claim import ENVIRONMENT_RULES, require

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"research"))
from claim_frontier import report

FAMILIES = dict(flex_quality="flex", nature_cnn="nature", impala_cnn="impala", impoola_cnn="impoola")
BINARY_NAMES = dict(flex_quality="default", nature_cnn="nature", impala_cnn="impala", impoola_cnn="impoola")


def audit(out):
    """Rebuild analysis from intact episode evidence; never rerun GPU inference."""
    protocol = json.loads((out/"protocol.json").read_text())
    progress = json.loads((out/"progress.json").read_text())
    original = Path(protocol["original_run"])
    require(ev.sha(original/"protocol.json") == protocol["original_protocol_sha256"], "Historical protocol changed")
    suite = ev.load_suite(out/"suite/suite.json")
    require(ev.sha(out/"suite/suite.json") == protocol["suite_sha256"], "Suite changed")
    with (out/"historical-pooled-v1.csv").open() as stream:
        historical = {(r["variant"], int(r["seed"]), int(r["steps"])): r for r in csv.DictReader(stream)}
    jobs = {(j["variant"], j["seed"]): j for j in json.loads((original/"jobs.json").read_text())}
    expected = {(m, s, k) for m in protocol["models"] for s in protocol["training_seeds"] for k in protocol["checkpoint_steps"]}
    require(set(historical) == expected, "Historical timing panel differs")
    points, seen = [], set()
    for receipt in progress["receipts"]:
        key = receipt["model"], receipt["seed"], receipt["steps"]
        require(key in expected and key not in seen, "Duplicate/unexpected receipt")
        seen.add(key)
        directory = out/receipt["evaluation"]
        require(ev.sha(directory/"result.json") == receipt["result_sha256"] and
                ev.sha(directory/"episodes.csv") == receipt["episodes_sha256"], "Evaluation evidence changed")
        measured = json.loads((directory/"result.json").read_text())
        require(measured["status"] == "ok" and measured["suite_sha256"] == protocol["suite_sha256"], "Wrong suite/status")
        checkpoint = Path(measured["checkpoint"])
        require(ev.sha(checkpoint) == measured["checkpoint_sha256"] == jobs[key[:2]]["checkpoint_sha256"][checkpoint.name], "Checkpoint changed")
        require(ev.sha(Path(measured["command"][0])) == measured["binary_sha256"], "Binary changed")
        counts = ev.audit_csv(directory/"episodes.csv", suite, "connect4cnn")
        row = historical[key]
        require(measured["parameters"] == int(row["params"]), "Wrong parameter count")
        points.append(dict(model=key[0], seed=key[1], steps=key[2], seconds=float(row["checkpoint_wall_s"]),
                           win_rate=counts["wins"]/counts["episodes"], games=counts["episodes"], parameters=measured["parameters"]))
    target = out/"analysis.audited"
    target.mkdir(exist_ok=False)
    report(target, protocol, points, progress["failures"])
    ev.save(out/"finished.audited.json", dict(status="ok" if seen == expected and not progress["failures"] else "incomplete",
        evaluations=len(points), failures=progress["failures"], receipts=progress["receipts"], analysis=str(target)))
    require(seen == expected and not progress["failures"], "Incomplete panel; partial analysis preserved")
    print(target/"REPORT.md")


def reevaluate(run, binaries, suite_path, out):
    suite = ev.load_suite(suite_path)
    original = json.loads((run/"protocol.json").read_text())
    require(original["environment_rules"] == ENVIRONMENT_RULES, "Use corrected-rules checkpoints")
    require(original["precision"] == "float32", "Need float32 checkpoints")
    with (run/"results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    jobs = {(j["variant"], j["seed"]): j for j in json.loads((run/"jobs.json").read_text())}
    expected = {(m, s, k) for m in original["variants"] for s in original["seeds"] for k in original["checkpoint_steps"]}
    require(len(rows) == len(expected) and {(r["variant"], int(r["seed"]), int(r["steps"])) for r in rows} == expected,
            "Historical checkpoint panel is missing/duplicated")
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(models=original["variants"], training_seeds=original["seeds"],
                    checkpoint_steps=original["checkpoint_steps"], environment_rules=ENVIRONMENT_RULES,
                    purpose="development: exact-suite reevaluation of existing single-seed curves",
                    evaluation_protocol="connect4-fixed-suite-v1", suite_sha256=ev.sha(suite_path),
                    timing="historical launch wall clock to file mtime (approximate); NOT new monotonic timing",
                    original_run=str(run), original_protocol_sha256=ev.sha(run/"protocol.json"))
    ev.save(out/"protocol.json", protocol)
    (out/"suite").mkdir()
    shutil.copyfile(suite_path, out/"suite/suite.json")
    shutil.copyfile(suite_path.parent/"episodes.csv", out/"suite/episodes.csv")
    shutil.copyfile(run/"results.csv", out/"historical-pooled-v1.csv")
    points, failures, receipts = [], [], []
    for row in rows:
        model, seed, step = row["variant"], int(row["seed"]), int(row["steps"])
        target = out/f"{model}-s{seed}-{step:016d}"
        checkpoint = run/row["checkpoint"]
        try:
            require(ev.sha(checkpoint) == jobs[model, seed]["checkpoint_sha256"][checkpoint.name], "Historical weights changed")
            config = run/f"{model}-s{seed}/metrics/connect4cnn/trial.ini"
            binary = binaries/BINARY_NAMES[model]
            command = [sys.executable, str(Path(ev.__file__)), "run", "--suite", str(suite_path),
                       "--binary", str(binary), "--family", FAMILIES[model], "--config", str(config),
                       "--checkpoint", str(checkpoint), "--out", str(target)]
            result = subprocess.run(command, timeout=330, check=False, capture_output=True, text=True)
            (out/f"{target.name}.launcher.log").write_text(result.stdout+result.stderr)
            require(result.returncode == 0, f"Exact evaluation failed: {target.name}")
            measured = json.loads((target/"result.json").read_text())
            require(measured["status"] == "ok" and measured["suite_sha256"] == protocol["suite_sha256"], "Wrong evaluation suite/status")
            require(measured["parameters"] == int(row["params"]), "Wrong policy parameter count")
            counts = ev.audit_csv(target/"episodes.csv", suite, "connect4cnn")
            points.append(dict(model=model, seed=seed, steps=step, seconds=float(row["checkpoint_wall_s"]),
                               win_rate=counts["wins"]/counts["episodes"], games=counts["episodes"],
                               parameters=measured["parameters"]))
            receipts.append(dict(model=model, seed=seed, steps=step, evaluation=str(target.relative_to(out)),
                                 result_sha256=ev.sha(target/"result.json"), episodes_sha256=ev.sha(target/"episodes.csv")))
        except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
            failures.append(dict(model=model, seed=seed, steps=step, error=str(error)))
        # Retain usable cells even if a different checkpoint fails.
        state = dict(status="running", points=points, failures=failures, receipts=receipts)
        (out/"progress.json").write_text(json.dumps(state, indent=2)+"\n")
    (out/"analysis").mkdir()
    report(out/"analysis", protocol, points, failures)
    ev.save(out/"finished.json", dict(status="ok" if not failures else "completed_with_failures",
                                     evaluations=len(points), failures=failures, receipts=receipts))
    require(not failures, "Incomplete panel; partial results preserved")
    print(out/"analysis/REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "binaries", "suite", "out"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true", help="Audit retained episodes and regenerate report without GPU")
    args = parser.parse_args()
    if args.audit_only:
        audit(args.out.resolve())
    else:
        reevaluate(*(getattr(args, key).resolve() for key in ("run", "binaries", "suite", "out")))
