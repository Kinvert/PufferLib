"""Compare state/pixel policies with a common recipe, or measure stock Connect4 separately."""
import argparse
import configparser
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VARIANTS = {"state": "connect4", "tiny_cnn": "connect4cnn", "nature_cnn": "connect4cnn",
            "impala_cnn": "connect4cnn", "impoola_cnn": "connect4cnn",
            "flex_quality": "connect4cnn", "flex_fast": "connect4cnn", "flex_small": "connect4cnn"}
EVAL = re.compile(r"CUDA_EVAL env=(\S+) score=([-+\d.eE]+) perf=([-+\d.eE]+) games=(\d+) params=(\d+)")


def find_nvidia_smi():
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable
    fallback = "/usr/lib/wsl/lib/nvidia-smi"
    if Path(fallback).is_file():
        return fallback
    raise RuntimeError("nvidia-smi not found in PATH or the WSL driver directory; use the host's existing GPU environment")


def require_idle_gpu(executable):
    active = subprocess.check_output([executable, "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=20)
    if active.strip():
        raise RuntimeError("GPU has a compute process; stop this comparison without interrupting the existing job")


def read_ini(path):
    ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    with Path(path).open() as f:
        ini.read_file(f)
    return ini


def overrides(ini):
    return [f"--{section}.{key}={value}" for section in ini for key, value in ini[section].items()]


def run(command, log, timeout, commands, cwd=ROOT):
    with commands.open("a") as f:
        f.write(json.dumps({"argv": command, "log": str(log), "cwd": str(cwd)}) + "\n")
    started = time.perf_counter()
    with log.open("w") as output:
        child = subprocess.Popen(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
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
    return time.perf_counter() - started


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def append_history(out, rows, jobs, protocol):
    history = ROOT / "research/EXPERIMENT_LOG.md"
    relative = os.path.relpath(out, history.parent)
    lines = [f"\n## {datetime.now(timezone.utc).isoformat(timespec='seconds')} — {out.name}", "",
             f"Change/purpose: {protocol['note']}", "",
             f"Revision `{protocol['revision'][:12]}`; recipe SHA256 `{protocol['recipe_sha256']}`. "
             f"[Report]({relative}/REPORT.md) · [CSV]({relative}/results.csv) · [Source/build hashes]({relative}/protocol.json) · [GPU]({relative}/gpu.txt)", "",
             "| Policy | Seed | Steps | Win rate | Score | Parameters | Wall s | Process SPS | Native avg SPS | Native last SPS | VRAM last GB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for job in jobs:
        if job["status"] != "ok":
            lines.append(f"| {job['variant']} | {job['seed']} | — | FAILED | — | — | — | — | — | — | — |")
            continue
        r = max((r for r in rows if r["variant"] == job["variant"] and r["seed"] == job["seed"]), key=lambda r: r["steps"])
        lines.append(f"| {r['variant']} | {r['seed']} | {r['steps']:,} | {r['win_rate']:.2%} | {r['score']:.4f} | {r['params']:,} | {r['train_process_wall_s']:.3f} | {r['process_sps']:,.0f} | {r['native_avg_sps']:,.0f} | {r['native_last_sps']:,.0f} | {r['vram_last_gb']:.3f} |")
    lines += ["", ("Stock training configuration in float32; common 64-agent evaluation. Training hypers differ from the earlier state/tiny-CNN comparison."
                       if protocol["stock"] else "Matched learner/core recipe; state and CNN parameter counts differ."),
              "See the report for checkpoint curves and actual evaluation counts.", ""]
    with history.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("\n".join(lines))


def report(out, rows, jobs, protocol):
    fields = ["variant", "seed", "eval_seed", "steps", "win_rate", "score", "games",
              "params", "checkpoint_wall_s", "train_process_wall_s", "eval_wall_s",
              "process_sps", "native_uptime_s", "native_avg_sps", "native_last_sps", "vram_last_gb", "checkpoint"]
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    description = ("Stock Connect4 training configuration, float32, one GPU, serial seeds. "
                   "Evaluation uses the common 64-agent setup. This is a separately configured baseline, not a matched-hyperparameter comparison."
                   if protocol["stock"] else
                   "Same 7-column × 6-row game, opponent, learner recipe and core. Float32, one GPU, serial trials. "
                   "Encoders differ; parameters/FLOPs are not matched. State, if included, uses a modified configuration.")
    lines = ["# Connect4 training results", "", description,
             f"Environment rules: `{protocol.get('environment_rules', 'legacy-unversioned-see-source')}`; evaluation: pooled-v1 (exploratory).",
             f"Requested decisions: {protocol['steps']:,}; expected completed decisions: {protocol['completed_steps']:,}.", "",
             "## Final results by training seed", "",
             "| Policy | Seed | Steps | Win rate | Score | Games | Parameters | Train wall seconds | Process SPS | Native avg SPS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for job in jobs:
        if job["status"] != "ok":
            lines.append(f"| {job['variant']} | {job['seed']} | — | **FAILED** | — | — | — | — | — | — |")
            continue
        row = max((r for r in rows if r["variant"] == job["variant"] and r["seed"] == job["seed"]), key=lambda r: r["steps"])
        lines.append(f"| {row['variant']} | {row['seed']} | {row['steps']:,} | {row['win_rate']:.2%} | {row['score']:.4f} | {row['games']} | {row['params']:,} | {row['train_process_wall_s']:.3f} | {row['process_sps']:,.0f} | {row['native_avg_sps']:,.0f} |")
    lines += ["", "## Checkpoint evaluation curves", "",
              "| Policy | Seed | Steps | Win rate | Training wall time at checkpoint (s) |",
              "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['steps']:,} | {r['win_rate']:.2%} | {r['checkpoint_wall_s']:.3f} |")
    lines += ["", "## Reading these results", "",
              "- Win rate is native `perf`. Evaluation seeds are separate from training seeds and shared across policies.",
              "- Evaluation is batched and may exceed the requested game count; actual counts are shown. Games are not independent training seeds.",
              "- Train wall time includes process startup and checkpoint writing; builds and later evaluations are excluded. Per-checkpoint time uses file modification time relative to process launch, so it is approximate.",
              "- Process SPS = full training steps / process wall time. Native average SPS = full training steps / the trainer's final logged uptime. Native last SPS is the last logged SPS sample/bin, not a whole-run average. These training-run statistics repeat on checkpoint rows; they are not checkpoint-specific throughput.",
              "- Evaluation happens after training at fixed saved checkpoints. It does not affect training timing or select a best checkpoint.",
              "- Confirmation rotates model order by seed according to protocol.json; other comparisons alternate forward/reverse order. Short smoke timings are not steady-state speed benchmarks.",
              "- CSV retains each checkpoint result. `protocol.json`, `source/`, `commands.jsonl`, resolved INIs, logs, hashes, and checkpoints preserve the run context.",
              "- A short run or one training seed cannot establish a reliable performance ranking. Use the same longer budget and several seeds before drawing conclusions."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stock", action="store_true", help="Run only original Connect4 with stock training settings; float32 and common evaluation")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=None)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--checkpoints", type=int)
    parser.add_argument("--timeout", type=int, help="Seconds per training process; evaluation capped at 60 seconds")
    parser.add_argument("--confirmation", action="store_true", help="Use the frozen six-model/five-seed confirmation manifest")
    parser.add_argument("--canary", action="store_true", help="Use the manifest's short one-seed confirmation validation")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--project", default="cnn2")
    parser.add_argument("--entity", default="kinvert-k")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--note", help="Change or purpose recorded in the experiment history")
    parser.add_argument("--require-idle-gpu", action="store_true", help="Refuse competing GPU compute processes before builds and each training job")
    args = parser.parse_args()
    manifest = json.loads((HERE / "confirmation.json").read_text())
    controlled = ("steps", "seeds", "eval_seed", "eval_games", "checkpoints", "timeout")
    if args.canary and not args.confirmation:
        parser.error("--canary requires --confirmation")
    if args.confirmation:
        if args.stock or args.recipe or args.variants or any(getattr(args, k) is not None for k in controlled):
            parser.error("Confirmation settings are frozen in confirmation.json; omit individual overrides")
        chosen = manifest["canary"] if args.canary else manifest
        for key in controlled:
            setattr(args, key, chosen[key])
        args.variants = manifest["variants"]
        args.note = args.note or ("Confirmation CANARY; plumbing only" if args.canary else "Frozen five-seed Connect4CNN architecture confirmation v1")
    else:
        for key, value in dict(seeds=[73, 74, 75], eval_seed=10073, eval_games=1024, checkpoints=4, timeout=120).items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if args.stock and (args.recipe is not None or args.steps is not None or args.variants is not None):
        parser.error("--stock preserves the stock training recipe and budget; omit --recipe, --steps and --variants")
    args.recipe = args.recipe or HERE / "compare.ini"
    recipe = read_ini(ROOT / "config/default.ini") if args.stock else read_ini(args.recipe)
    if args.stock:
        recipe.read(ROOT / "config/connect4.ini")
    args.steps = recipe.getint("train", "total_timesteps") if args.stock else (65536 if args.steps is None else args.steps)
    selected = args.variants or ["state", "tiny_cnn"]
    if len(selected) != len(set(selected)):
        parser.error("Use unique variants")
    variants = {"stock_state": "connect4"} if args.stock else {v: VARIANTS[v] for v in selected}
    args.note = args.note or ("Stock Connect4 training configuration in float32" if args.stock
                              else "Common-recipe comparison: " + ", ".join(selected))
    # CUDA_EVAL.score follows sweep.metric; fix it to episode score so the
    # report can show both return and the separately reported win rate.
    if recipe.get("sweep", "metric", fallback="score") != "score":
        parser.error("Keep sweep.metric=score so the report's score column is episode return")
    batch = recipe.getint("vec", "total_agents") * recipe.getint("train", "horizon")
    epochs = args.steps // batch
    if args.steps <= 0 or args.checkpoints <= 0 or epochs < args.checkpoints or (not args.stock and args.steps % (batch * args.checkpoints)):
        parser.error(f"--steps must be a positive multiple of rollout batch ({batch}) × --checkpoints")
    completed_steps = epochs * batch
    checkpoint_interval = (epochs + args.checkpoints - 1) // args.checkpoints
    checkpoint_steps = sorted({e * batch for e in range(checkpoint_interval, epochs + 1, checkpoint_interval)} | {completed_steps})
    if args.eval_games <= 0 or args.timeout <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive game/time limits and unique training seeds")
    eval_seeds = {args.eval_seed + i for i in range(len(args.seeds))}
    if set(args.seeds) & eval_seeds:
        parser.error("Training and evaluation seed lists must not overlap")
    os.chdir(ROOT)
    (ROOT / "build/connect4cnn").mkdir(parents=True, exist_ok=True)
    prefix = ("confirm-canary." if args.canary else "confirm.") if args.confirmation else "compare."
    out = Path(tempfile.mkdtemp(prefix=prefix, dir=ROOT / "build/connect4cnn"))
    commands = out / "commands.jsonl"
    print(f"Comparison: {out}", flush=True)
    sources = ["build.sh", "config/default.ini", "config/connect4.ini", "config/connect4cnn.ini"]
    sources += [str(p.relative_to(ROOT)) for p in (ROOT / "src").glob("*") if p.is_file()]
    sources += ["ocean/connect4/connect4.h"]
    sources += [str(p.relative_to(ROOT)) for p in HERE.glob("*") if p.is_file()]
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    with (out / "recipe.ini").open("w") as f:
        recipe.write(f)
    protocol = {**vars(args), "recipe": "config/default.ini + config/connect4.ini" if args.stock else str(args.recipe),
                "precision": "float32", "variants": variants, "completed_steps": completed_steps,
                "environment_rules": "connect4-full-board-draw-v2", "evaluation_protocol": "pooled-v1",
                "checkpoint_steps": checkpoint_steps,
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {p: sha256(ROOT / p) for p in sources},
                "build_arch": os.environ.get("NVCC_ARCH", "native"), "binary_sha256": {},
                "recipe_sha256": sha256(out / "recipe.ini"), "started_utc": datetime.now(timezone.utc).isoformat()}
    if args.confirmation:
        protocol["confirmation_manifest"] = manifest
        protocol["confirmation_manifest_sha256"] = sha256(HERE / "confirmation.json")
        protocol["order"] = [list(variants)[i % len(variants):] + list(variants)[:i % len(variants)] for i in range(len(args.seeds))]
    # Freeze the effective defaults in CLI overrides before any training starts.
    if args.confirmation:
        frozen = read_ini(ROOT / "config/default.ini")
        frozen.read([ROOT / "config/connect4cnn.ini", args.recipe])
        for section in list(frozen.sections()):
            if section.startswith("sweep."):
                frozen.remove_section(section)
        frozen.set("sweep", "downsample", "25")
        with (out / "effective.ini").open("w") as f:
            frozen.write(f)
        protocol["effective_sha256"] = sha256(out / "effective.ini")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    smi = find_nvidia_smi()
    if args.require_idle_gpu:
        require_idle_gpu(smi)
    run([smi], out / "gpu.txt", 20, commands)
    host = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "python": sys.version, "nvidia_smi": smi}
    if hasattr(os, "sched_getaffinity"):
        host["available_cpus"] = len(os.sched_getaffinity(0))
    (out / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc") if os.environ.get("CUDA_HOME") else shutil.which("nvcc")
    if nvcc:
        run([nvcc, "--version"], out / "cuda-compiler.txt", 20, commands)
    if shutil.which("lscpu"):
        run(["lscpu"], out / "cpu.txt", 20, commands)
    binaries = {}
    if os.environ.get("NVCC_PREPEND_FLAGS"):
        parser.error("Unset NVCC_PREPEND_FLAGS so reference build selection is controlled")
    # Always build from the current source. PufferLib's ccache handles reuse.
    for variant, env in variants.items():
        if variant.startswith("flex_") and "flex_quality" in binaries:
            binaries[variant] = binaries["flex_quality"]
            protocol["binary_sha256"][variant] = sha256(binaries[variant])
            continue
        binaries[variant] = str(out / variant)
        build = ["bash", "build.sh", env, binaries[variant], "--float"]
        if variant in ("nature_cnn", "impala_cnn", "impoola_cnn"):
            build = ["env", "NVCC_PREPEND_FLAGS=" + os.environ.get("NVCC_PREPEND_FLAGS", "") + " -DC4_" + variant.upper(), *build]
        run(build, out / f"build-{variant}.log", 300, commands)
        protocol["binary_sha256"][variant] = sha256(binaries[variant])
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed during build: {name}; start a fresh campaign")
    rows, jobs = [], []
    settings = [] if args.stock else overrides(frozen if args.confirmation else recipe)
    effective = None
    for index, seed in enumerate(args.seeds):
        order = protocol["order"][index] if args.confirmation else (list(variants) if index % 2 == 0 else list(reversed(variants)))
        for variant in order:
            env = variants[variant]
            trial = out / f"{variant}-s{seed}"
            trial.mkdir()
            policy = manifest["policies"].get(variant, {"encoder": 0})
            policy_args = [f"--policy.{k}={v}" for k, v in policy.items()] if env == "connect4cnn" else []
            common = settings + policy_args + ["--headless", f"--base.seed={seed}", "--base.run_id=trial",
                f"--train.total_timesteps={args.steps}",
                "--base.eval_episodes=0",
                f"--base.checkpoint_interval={checkpoint_interval}",
                f"--base.checkpoint_dir={trial}/checkpoints", f"--base.log_dir={trial}/metrics"]
            job = {"variant": variant, "seed": seed, "status": "running"}
            runtime_cwd = ROOT
            if args.confirmation or variant.startswith("flex_"):
                trial_config = copy.deepcopy(frozen) if args.confirmation else read_ini(ROOT / "config/default.ini")
                if not args.confirmation:
                    trial_config.read([ROOT / "config/connect4cnn.ini", args.recipe])
                for key, value in policy.items():
                    trial_config.set("policy", key, str(value))
                (trial / "config").mkdir()
                with (trial / "config/default.ini").open("w") as f:
                    trial_config.write(f)
                (trial / f"config/{env}.ini").write_text("# Frozen effective settings are in default.ini.\n")
                job["config_sha256"] = sha256(trial / "config/default.ini")
                runtime_cwd = trial
            jobs.append(job)
            try:
                if args.require_idle_gpu:
                    require_idle_gpu(smi)
                run([smi], trial / "gpu-before.txt", 20, commands)
                start_wall = time.time()
                elapsed = run([binaries[variant], "train", *common], trial / "train.log", args.timeout, commands, runtime_cwd)
                job["train_process_wall_s"] = elapsed
                resolved = read_ini(trial / "metrics" / env / "trial.ini")
                metrics = resolved["metrics"]
                native_uptime = float(metrics["uptime"].split(",")[-1])
                if native_uptime <= 0 or int(float(metrics["agent_steps"].split(",")[-1])) != completed_steps:
                    raise ValueError("Invalid native timing or step count")
                timing = dict(process_sps=completed_steps / elapsed, native_uptime_s=native_uptime,
                              native_avg_sps=completed_steps / native_uptime,
                              native_last_sps=float(metrics["sps"].split(",")[-1]),
                              vram_last_gb=float(metrics["util/vram_used_gb"].split(",")[-1]))
                # Enforce matched effective settings, including inherited defaults.
                ignore = {"env_name", "seed", "run_id", "checkpoint_dir", "log_dir"}
                current = {s: dict(resolved[s]) for s in resolved if s != "metrics"}
                current["base"] = {k: v for k, v in current["base"].items() if k not in ignore}
                if env == "connect4cnn":
                    for key, value in policy.items():
                        if resolved.getfloat("policy", key, fallback=0) != value:
                            raise ValueError(f"Wrong frozen encoder setting: {variant} {key}")
                    # These construction settings are inactive for reference encoders.
                    current["policy"] = {k: v for k, v in current["policy"].items()
                                         if k != "encoder" and not k.startswith("cnn_")}
                if effective is None:
                    effective = current
                if current != effective:
                    raise ValueError("Effective configurations differ; comparison is not matched")
                if args.stock:
                    for section in ("train", "vec", "policy", "env", "selfplay"):
                        for key, value in recipe[section].items():
                            if float(resolved[section][key]) != float(value):
                                raise ValueError(f"Stock configuration changed: {section}.{key}")
                    if resolved.getint("base", "async") != recipe.getint("base", "async"):
                        raise ValueError("Stock async setting changed")
                paths = sorted((trial / "checkpoints" / env / "trial").glob("*.bin"))
                if [int(p.stem) for p in paths] != checkpoint_steps:
                    raise ValueError("Missing or unexpected checkpoint schedule")
                for checkpoint in paths:
                    if variant in manifest["parameters"]:
                        import numpy as np
                        weights = np.fromfile(checkpoint, dtype=np.float32)
                        if weights.size != manifest["parameters"][variant] or not np.isfinite(weights).all():
                            raise ValueError(f"Malformed/nonfinite checkpoint: {checkpoint}")
                    checkpoint_wall = checkpoint.stat().st_mtime - start_wall
                    log = trial / f"eval-{checkpoint.stem}.log"
                    eval_seed = args.eval_seed + index
                    eval_args = common + [f"--base.seed={eval_seed}", f"--base.eval_episodes={args.eval_games}",
                                          f"--base.load_model_path={checkpoint}"]
                    if args.stock:
                        eval_args += ["--vec.total_agents=64", "--vec.num_buffers=1", "--vec.num_threads=2", "--base.async=0"]
                    eval_wall = run([binaries[variant], "eval", *eval_args], log, min(args.timeout, 60), commands, runtime_cwd)
                    match = EVAL.search(log.read_text())
                    if not match or match[1] != env or (args.confirmation and int(match[5]) != manifest["parameters"][variant]):
                        raise ValueError(f"Missing native evaluation result in {log}")
                    rows.append(dict(variant=variant, seed=seed, eval_seed=eval_seed,
                        steps=int(checkpoint.stem), win_rate=float(match[3]), score=float(match[2]),
                        games=int(match[4]), params=int(match[5]), checkpoint_wall_s=checkpoint_wall,
                        train_process_wall_s=elapsed, eval_wall_s=eval_wall,
                        **timing, checkpoint=str(checkpoint.relative_to(out))))
                job["status"] = "ok"
                job["checkpoint_sha256"] = {p.name: sha256(p) for p in paths}
                print(f"{variant} seed={seed}: {rows[-1]['win_rate']:.2%} wins, {elapsed:.3f}s training wall, {timing['process_sps']:,.0f} process SPS", flush=True)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                job.update(status="failed", error=str(exc))
                print(f"FAILED {variant} seed={seed}: {exc}; see {trial}", flush=True)
            report(out, rows, jobs, protocol)
            if job["status"] == "ok" and args.wandb != "disabled":
                sidecar = [sys.executable, str(out / "source/ocean/connect4cnn/compare_sidecar.py"), str(out),
                           "--mode", args.wandb, "--project", args.project, "--entity", args.entity]
                try:
                    run(sidecar, trial / "sidecar.log", 180, commands)
                    job["wandb_status"] = "ok"
                except (subprocess.SubprocessError, OSError) as exc:
                    job.update(wandb_status="failed", wandb_error=str(exc))
                    print(f"W&B upload failed; native results preserved: {trial}", flush=True)
                report(out, rows, jobs, protocol)
    print(f"Report: {out / 'REPORT.md'}", flush=True)
    append_history(out, rows, jobs, protocol)
    (out / "finished.json").write_text(json.dumps({"status": "ok" if all(j["status"] == "ok" for j in jobs) else "failed",
        "jobs": len(jobs), "evaluations": len(rows), "logging_failures": sum(j.get("wandb_status") == "failed" for j in jobs)}, indent=2) + "\n")
    if any(j["status"] != "ok" for j in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
