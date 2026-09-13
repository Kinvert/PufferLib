"""Prepare an isolated native PROTEIN campaign for the experimental CNN only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from wandb_sidecar import read_ini, trials

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIMENSIONS = {"policy.cnn_channels", "policy.cnn_blocks", "policy.cnn_global_pool", "train.total_timesteps"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, recipe, max_runs):
    ini = read_ini(ROOT / "config/default.ini")
    ini.read([ROOT / "config/connect4cnn.ini", HERE / "compare.ini"])
    for section in list(ini.sections()):
        if section.startswith("sweep."):
            ini.remove_section(section)
    ini.read(recipe)
    dimensions = {s[6:] for s in ini.sections() if s.startswith("sweep.")}
    if dimensions != DIMENSIONS or ini.getint("policy", "encoder") != 1:
        raise ValueError("Sweep only encoder=1 and the four declared architecture/budget dimensions")
    if ini.getint("train", "gpus") != 1 or ini.getint("sweep", "gpus") != 1 or ini.getint("selfplay", "enabled"):
        raise ValueError("This runner requires one GPU and no selfplay")
    if ini.getint("vec", "num_policies") != 1 or ini.getint("base", "eval_episodes") != 0:
        raise ValueError("Use one policy and training metrics for the discovery sweep")
    if max_runs is not None:
        ini.set("sweep", "max_runs", str(max_runs))
    if ini.getint("sweep", "max_runs") < 1:
        raise ValueError("max-runs must be positive")
    if ini.get("sweep", "metric") != "perf":
        raise ValueError("Use sweep.metric=perf (training win rate)")
    limits = {"policy.cnn_channels": ("uniform_pow2", {8, 16, 32}),
              "policy.cnn_blocks": ("int_uniform", {0, 1, 2}),
              "policy.cnn_global_pool": ("int_uniform", {0, 1})}
    for key, (distribution, allowed) in limits.items():
        section = "sweep." + key
        lo, hi = ini.getfloat(section, "min"), ini.getfloat(section, "max")
        base_section, name = key.split(".")
        value = ini.getfloat(base_section, name)
        if ini.get(section, "distribution") != distribution or lo not in allowed or hi not in allowed or not lo < hi or value not in allowed or not lo <= value <= hi:
            raise ValueError(f"Unsupported architecture range/default: {key}")
    batch = ini.getint("vec", "total_agents") * ini.getint("train", "horizon")
    budget = "sweep.train.total_timesteps"
    lo, hi = ini.getfloat(budget, "min"), ini.getfloat(budget, "max")
    if not batch <= lo < hi or not lo <= ini.getfloat("train", "total_timesteps") <= hi:
        raise ValueError("Budget range must cover the default and at least one rollout batch")
    if os.environ.get("NVCC_PREPEND_FLAGS"):
        raise ValueError("Unset NVCC_PREPEND_FLAGS for this experimental-only sweep")
    ini.set("base", "checkpoint_dir", str(out / "checkpoints"))
    ini.set("base", "log_dir", str(out / "metrics"))
    (out / "config").mkdir()
    with (out / "config/default.ini").open("w") as f:
        ini.write(f)
    (out / "config/connect4cnn.ini").write_text("# All effective settings are in this campaign's default.ini.\n")
    return ini


def execute(command, cwd, log, timeout):
    started = time.monotonic()
    with log.open("w") as f:
        child = subprocess.Popen(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
    return time.monotonic() - started


def write_report(out, rows, wall, status):
    fields = ["index", "run_id", "channels", "blocks", "global_pool", "steps", "score", "cost", "native_avg_sps", "params", "random", "gp_obs", "architecture_sha256", "pareto"]
    flat = []
    for row in rows:
        dominated = any(r["cost"] <= row["cost"] and r["score"] >= row["score"] and
                        (r["cost"] < row["cost"] or r["score"] > row["score"]) for r in rows)
        flat.append({**{k: row[k] for k in fields if k in row}, "channels": row["config"]["policy.cnn_channels"],
                     "blocks": row["config"]["policy.cnn_blocks"], "global_pool": row["config"]["policy.cnn_global_pool"], "pareto": not dominated})
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)
    lines = ["# Experimental CNN native PROTEIN sweep", "", f"Status: {status}. Sweep process wall: {wall:.3f} seconds.",
             "Same Connect4CNN pixels and locked learner/core recipe; architecture and training budget vary.",
             "Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.",
             "Canary budgets validate infrastructure, not learning quality. Pareto flags use rounded final observations.", "",
             "| Trial | Channels | Blocks | GAP | Steps | Training wins | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in flat:
        lines.append(f"| {r['index']} | {r['channels']} | {r['blocks']} | {r['global_pool']} | {r['steps']:,} | {r['score']:.2%} | {r['cost']:.2f} | {r['native_avg_sps']:,.0f} | {r['params']:,} | {bool(r['gp_obs'])} | {r['pareto']} |")
    lines += ["", "Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.",
              "Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.",
              "Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and post-run W&B synchronization.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=HERE / "sweep.ini")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--timeout", type=int, default=600, help="Hard deadline for the entire sweep and its worker process group")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="offline")
    parser.add_argument("--project", default="puffer-cnn")
    parser.add_argument("--entity")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    (ROOT / "build/connect4cnn").mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="sweep.", dir=ROOT / "build/connect4cnn"))
    ini = prepare(out, args.recipe.resolve(), args.max_runs)
    print(f"Sweep: {out}", flush=True)
    sources = [ROOT / "build.sh", ROOT / "config/default.ini", ROOT / "config/connect4cnn.ini",
               *sorted((ROOT / "src").glob("*")), *sorted(HERE.glob("*"))]
    sources = [p for p in sources if p.is_file()]
    sources += sorted((HERE / "tests").glob("*.cu")) + sorted((HERE / "tests").glob("*.py"))
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    protocol = dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    started_utc=datetime.now(timezone.utc).isoformat(), precision="float32", build_arch=os.environ.get("NVCC_ARCH", "native"),
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    config_sha256=sha(out / "config/default.ini"), dimensions=sorted(DIMENSIONS),
                    max_runs=ini.getint("sweep", "max_runs"), timeout=args.timeout)
    binary = out / "cnn"
    build = ["bash", "build.sh", "connect4cnn", str(binary), "--float"]
    command = [str(binary), "sweep", "--headless"]
    protocol.update(build_command=build, sweep_command=command, sweep_cwd=str(out))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    execute(["/usr/lib/wsl/lib/nvidia-smi"], ROOT, out / "gpu.txt", 20)
    execute(build, ROOT, out / "build.log", 300)
    protocol["binary_sha256"] = sha(binary)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    status, error, wall = "ok", None, 0
    started = time.monotonic()
    try:
        wall = execute(command, out, out / "sweep.log", args.timeout)
    except Exception as exc:
        status, error, wall = "failed", str(exc), time.monotonic() - started
    rows = trials(out)
    failures = [line for line in (out / "sweep.log").read_text().splitlines() if "failed; marking sample bad" in line]
    if failures:
        status, error = "failed", error or f"{len(failures)} native workers failed; see sweep.log"
    if len(rows) != protocol["max_runs"]:
        status, error = "failed", error or "Missing completed trials"
    write_report(out, rows, wall, status)
    (out / "finished.json").write_text(json.dumps(dict(status=status, error=error, completed=len(rows), wall_seconds=wall, worker_failures=failures), indent=2) + "\n")
    sidecar = [sys.executable, str(HERE / "wandb_sidecar.py"), str(out), "--mode", args.wandb, "--project", args.project]
    if args.entity:
        sidecar += ["--entity", args.entity]
    execute(sidecar, ROOT, out / "sidecar.log", 300)
    print(f"{status}: {len(rows)} trials, {len({r['architecture_sha256'] for r in rows})} architectures; report: {out / 'REPORT.md'}", flush=True)
    if status != "ok":
        raise SystemExit(error)


if __name__ == "__main__":
    main()
