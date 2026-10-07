"""GPU-free independent audit/archive of the native development replication."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"ocean/connect4cnn"))
import deterministic_eval as ev
from claim import ini, normalized, common_settings, parse_checkpoints, require
from claim_frontier import report


def archive(run, output):
    require(not output.exists(), "Fresh archive directory required")
    protocol = json.loads((run/"protocol.json").read_text())
    require(ev.sha(run/"protocol.json") == (run/"protocol.sha256").read_text().strip(), "Protocol changed")
    require(protocol["purpose"] == "paired_development_replication" and not protocol["candidate_bands"], "Wrong inference protocol")
    execution = json.loads((run/"execution.json").read_text())
    require(execution["status"] != "running", "Campaign still running; no premature final report")
    for name, digest in {**protocol["source_sha256"], **protocol["tool_sha256"]}.items():
        require(ev.sha(run/"source"/name) == digest, f"Captured source changed: {name}")
    suite = ev.load_suite(run/"suite/suite.json")
    require(ev.sha(run/"suite/suite.json") == protocol["suite_sha256"], "Suite changed")
    records = {r["id"]: r for r in execution["jobs"]}
    require(len(records) == len(execution["jobs"]), "Duplicate training job")
    points, failures = [], list(execution["failures"])
    for job in protocol["jobs"]:
        directory = run/job["id"]
        try:
            record = records[job["id"]]
            timed = json.loads((directory/"train.log.json").read_text())
            require(record["training"] == timed, "Training process receipt changed")
            require(timed["command"] == [str(run/"binaries"/job["model"]), "train", "--headless"] and
                    timed["cwd"] == str(directory) and timed["environment"]["PUFFER_CHECKPOINT_RECEIPTS"] == "1", "Wrong training command/flags")
            require(ev.sha(run/"binaries"/job["model"]) == protocol["binary_sha256"][job["model"]], "Changed binary")
            require(ev.sha(directory/"config/default.ini") == job["config_sha256"], "Changed launch config")
            expected = ini(directory/"config/default.ini")
            require(normalized(ini(directory/"checkpoints/connect4cnn/trial/resolved.ini")) == normalized(expected), "Native startup config mismatch")
            require(common_settings(expected) == protocol["common_settings"], "Unmatched learner settings")
            params = 160736 if job["model"] == "flex_quality" else 138528
            times = parse_checkpoints((directory/"train.log").read_text(), timed, protocol["checkpoint_steps"], params, allow_partial=True)
            require(json.loads((directory/"contention.json").read_text())["status"] == "uncontended", "Contaminated/unknown timing")
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(job=job["id"], error=str(error)))
            continue
        for step in protocol["checkpoint_steps"]:
            try:
                require(step in times, "Missing completed checkpoint receipt")
                checkpoint = directory/f"checkpoints/connect4cnn/trial/{step:016d}.bin"
                require(ev.sha(checkpoint) == record["checkpoints"][str(step)]["sha256"], "Changed checkpoint")
                weights = np.fromfile(checkpoint, dtype=np.float32)
                require(weights.size == params and np.isfinite(weights).all(), "Nonfinite or wrong-sized checkpoint")
                target = directory/f"eval-{step:016d}"
                receipt = record["evaluations"][str(step)]
                require(ev.sha(target/"result.json") == receipt["result_sha256"] and
                        ev.sha(target/"episodes.csv") == receipt["episodes_sha256"], "Changed evaluation receipt")
                measured = json.loads((target/"result.json").read_text())
                require(measured["status"] == "ok" and measured["suite_sha256"] == protocol["suite_sha256"], "Wrong suite/status")
                require(measured["checkpoint_sha256"] == ev.sha(checkpoint) and measured["binary_sha256"] == protocol["binary_sha256"][job["model"]], "Wrong evaluated checkpoint/build")
                counts = ev.audit_csv(target/"episodes.csv", suite, "connect4cnn")
                points.append(dict(model=job["model"], seed=job["seed"], steps=step, seconds=times[step],
                                   win_rate=counts["wins"]/counts["episodes"], games=counts["episodes"], parameters=params,
                                   process_sps=record.get("process_sps"), native_sps=record.get("native_avg_sps"),
                                   train_seconds=record["process_seconds"]))
            except (ValueError, OSError, KeyError) as error:
                failures.append(dict(job=job["id"], steps=step, error=str(error)))
    output.mkdir(parents=True)
    # Native weights/binaries stay local. Compress complete raw episode CSVs;
    # their original uncompressed SHA256 remains in the evaluation receipts.
    index = {}
    for path in run.rglob("*"):
        relative = path.relative_to(run)
        if not path.is_file() or "binaries" in relative.parts or path.suffix == ".bin":
            continue
        if path.suffix not in (".ini", ".json", ".csv", ".sha256", ".log", ".md", ".html", ".txt", ".h", ".cu", ".c", ".py", ".sh"):
            continue
        if path.name == "episodes.csv":
            target = output/Path(str(relative)+".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source, target.open("wb") as dest, gzip.GzipFile(fileobj=dest, mode="wb", mtime=0) as zipped:
                shutil.copyfileobj(source, zipped)
            index[str(relative)] = dict(archive=str(target.relative_to(output)), uncompressed_sha256=ev.sha(path), sha256=ev.sha(target))
        else:
            target = output/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (output/"independent-analysis").mkdir()
    analysis = report(output/"independent-analysis", protocol, points, failures)
    ev.save(output/"compressed-episodes.json", index)
    ev.save(output/"audit.json", dict(status="ok" if not failures and not analysis["missing"] else "incomplete",
        checkpoints_verified=len(points), failures=failures, parameters_checked=True, finite_weights_checked=True,
        weights_and_binaries_transferred=False, protocol_sha256=ev.sha(run/"protocol.json"),
        analyst_source_sha256={str(p.relative_to(ROOT)): ev.sha(p) for p in (Path(__file__), ROOT/"research/claim_frontier.py")}))
    status = "completed_and_audited" if not failures and not analysis["missing"] else "completed_with_missing_or_failed_cells"
    ev.save(output.parent/"STATUS.json", dict(status=status, observed_cells=len(points), missing=len(analysis["missing"]), failures=len(failures)))
    (output.parent/"README.md").write_text(
        f"# Paired Connect4 development replication — {status}\n\n"
        f"Observed/audited cells: {len(points)}. Missing cells: {len(analysis['missing'])}. Failure records: {len(failures)}.\n\n"
        "[Independent full-frontier report](completed/independent-analysis/REPORT.md) · "
        "[All curves](completed/independent-analysis/curves.html) · [Audit](completed/audit.json).\n\n"
        "Five fresh paired training seeds, fixed quality/Nature models, common learner/core, 13.312M decisions "
        "and all 51 checkpoints per run. Development evidence; no certified superiority or SOTA conclusion. "
        "All original curves/failures remain; raw episode CSVs are compressed with original hashes retained. "
        "Native weights and binaries were checked locally and were not transferred.\n\n"
        "[Methods](../../../CONNECT4_DEVELOPMENT_REPLICATION.md).\n")
    print(output/"independent-analysis/REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive(args.run.resolve(), args.out.resolve())
