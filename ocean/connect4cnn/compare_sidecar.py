"""Upload completed fixed-comparison jobs between training runs, never during them."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from wandb_sidecar import display_name, read_ini, scalar


def history(ini, evaluations):
    columns = {k: [float(v) for v in values.split(",")] for k, values in ini["metrics"].items()}
    steps = columns.pop("agent_steps")
    rows = {}
    for i, step in enumerate(steps):
        row = rows.setdefault(int(round(step)), {"agent_steps": int(round(step))})
        for key, values in columns.items():
            j = i - (len(steps) - len(values))
            if 0 <= j < len(values) and math.isfinite(values[j]):
                row[key] = values[j]
    for result in evaluations:
        step = int(result["steps"])
        rows.setdefault(step, {"agent_steps": step}).update({
            "eval/perf": float(result["win_rate"]), "eval/score": float(result["score"]),
            "eval/games": int(result["games"]), "eval/train_wall_s": float(result["checkpoint_wall_s"]),
            "eval/perf_vs_time": float(result["win_rate"])})
    return [rows[k] for k in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--project", default="cnn2")
    parser.add_argument("--entity", default="kinvert-k")
    args = parser.parse_args()
    root = args.campaign.resolve()
    output = root / "comparison-sidecar"
    output.mkdir(exist_ok=True)
    for key, folder in (("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "data"), ("WANDB_CONFIG_DIR", "config")):
        os.environ.setdefault(key, str(output / folder))
    import wandb
    token = hashlib.sha256(f"{args.mode}/{args.entity}/{args.project}".encode()).hexdigest()[:12]
    state_path = output / f"state-{token}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    jobs = json.loads((root / "jobs.json").read_text())
    results = list(csv.DictReader((root / "results.csv").open()))
    protocol = json.loads((root / "protocol.json").read_text())
    for index, job in enumerate(jobs):
        key = f"{job['variant']}-s{job['seed']}"
        if job["status"] != "ok" or key in state:
            continue
        env = protocol["variants"][job["variant"]]
        ini = read_ini(root / key / "metrics" / env / "trial.ini")
        evals = [r for r in results if r["variant"] == job["variant"] and int(r["seed"]) == job["seed"]]
        config = {f"{s}.{k}": scalar(v) for s in ini.sections() if s != "metrics" for k, v in ini[s].items()}
        config.update(variant=job["variant"], protocol="confirmation-v1" if protocol["confirmation"] else "comparison",
                      canary=protocol["canary"], binary_sha256=protocol["binary_sha256"][job["variant"]])
        identifier = hashlib.sha256(f"{root.name}/{key}".encode()).hexdigest()[:24]
        rows = history(ini, evals)
        (output / f"{key}.json").write_text(json.dumps({"config": config, "history": rows, "job": job}, indent=2) + "\n")
        run = wandb.init(project=args.project, entity=args.entity, id=identifier, name=display_name(root, index),
                         group=root.name, tags=[job["variant"], "canary" if protocol["canary"] else "confirmation"],
                         mode=args.mode, dir=str(output), config=config, resume="allow" if args.mode == "online" else None,
                         settings=wandb.Settings(console="off", disable_git=True, x_disable_stats=True))
        try:
            run.define_metric("agent_steps")
            run.define_metric("*", step_metric="agent_steps")
            run.define_metric("eval/perf_vs_time", step_metric="eval/train_wall_s")
            for row in rows:
                run.log(row, step=row["agent_steps"])
            final = max(evals, key=lambda r: int(r["steps"]))
            run.summary.update({"process_sps": float(final["process_sps"]),
                                "train_process_wall_s": float(final["train_process_wall_s"]),
                                "native_avg_sps": float(final["native_avg_sps"]),
                                "params": int(final["params"]), "eval_seed": int(final["eval_seed"]),
                                "checkpoint_sha256": job.get("checkpoint_sha256", {})})
        finally:
            run.finish()
        state[key] = identifier
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(f"Uploaded {key} as {display_name(root, index)}", flush=True)


if __name__ == "__main__":
    main()
