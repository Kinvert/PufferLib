"""Prepare isolated native PROTEIN campaigns with a fixed encoder family."""
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

from wandb_sidecar import read_ini, trials, display_name

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIMENSIONS = {"policy.cnn_channels", "policy.cnn_blocks", "policy.cnn_global_pool", "train.total_timesteps"}
LIMITS = {
    1: {"policy.cnn_channels": ("uniform_pow2", {8, 16, 32}),
        "policy.cnn_blocks": ("int_uniform", {0, 1, 2}),
        "policy.cnn_global_pool": ("int_uniform", {0, 1})},
    2: {},
    3: {"policy.cnn_channels": ("uniform_pow2", {8, 16, 32}),
        "policy.cnn_depth": ("int_uniform", {1, 2, 3}),
        "policy.cnn_stride": ("uniform_pow2", {2, 4}),
        "policy.cnn_projection": ("uniform_pow2", {32, 64, 128})},
}
LIMITS[4] = {"policy.cnn_depth": ("int_uniform", {1, 2, 3}),
             "policy.cnn_projection": ("uniform_pow2", {16, 32, 64, 128}),
             "policy.cnn_global_pool": ("int_uniform", {0, 1})}
for stage in range(1, 4):
    for key, bounds in {
        "channels": ("uniform_pow2", {8, 16, 32}),
        "kernel": ("int_uniform", set(range(1, 9 if stage == 1 else 6))),
        "stride": ("uniform_pow2", {4, 8} if stage == 1 else {1, 2, 4}),
        "pool": ("int_uniform", {0, 1, 2}),
        "residual": ("int_uniform", {0, 1}),
    }.items():
        LIMITS[4][f"policy.cnn_{key}_{stage}"] = bounds

LIMITS[5] = {"policy.cnn_depth": ("int_uniform", {1, 2, 3, 4}),
             "policy.cnn_projection": ("uniform_pow2", {16, 32, 64, 128}),
             "policy.cnn_readout": ("int_uniform", {0, 1, 2, 3}),
             "policy.cnn_projection_activation": ("int_uniform", {0, 1, 2, 3, 4})}
for stage in range(1, 5):
    for key, bounds in {
        "channels": ("uniform_pow2", {8, 16, 32, 64}),
        "kernel": ("int_uniform", set(range(1, 9 if stage == 1 else 6))),
        "stride": ("uniform_pow2", {1, 2, 4, 8} if stage == 1 else {1, 2, 4}),
        "dilation": ("int_uniform", {1, 2, 3, 4}),
        "activation": ("int_uniform", {0, 1, 2, 3, 4}),
        "pool": ("int_uniform", {0, 1, 2}),
        "residual": ("int_uniform", {0, 1, 2}),
    }.items():
        LIMITS[5][f"policy.cnn_{key}_{stage}"] = bounds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out, recipe, max_runs, depth=None, environment="connect4cnn", appearance_seed=None):
    if environment not in ("connect4cnn", "pongcnn", "flappycnn", "breakoutcnn", "snakecnn", "mazecnn"):
        raise ValueError("Unsupported pixel environment")
    ini = read_ini(ROOT / "config/default.ini")
    env_dir = ROOT / "ocean" / environment
    ini.read([ROOT / "config" / f"{environment}.ini", env_dir / "compare.ini"])
    for section in list(ini.sections()):
        if section.startswith("sweep."):
            ini.remove_section(section)
    ini.read(recipe)
    if appearance_seed is not None:
        if type(appearance_seed) is not int or not 0 <= appearance_seed <= 4294967295:
            raise ValueError("Appearance seed must be a uint32 integer")
        ini.set("env", "representation", "0")
        ini.set("env", "representation_mode", "1")
        ini.set("env", "representation_seed", str(appearance_seed))
        ini.set("env", "representation_mix_catalog", "1" if environment in ("pongcnn", "flappycnn") else "0")
    if depth is not None:
        if ini.getint("policy", "encoder") != 5:
            raise ValueError("--depth is for the expanded encoder 5")
        ini.set("policy", "cnn_depth", str(depth))
        ini.remove_section("sweep.policy.cnn_depth")
    if ini.getint("policy", "encoder") == 5 and not ini.has_section("sweep.policy.cnn_depth"):
        # An inactive coordinate must not consume GP search dimensions.
        active = ini.getint("policy", "cnn_depth")
        for stage in range(active + 1, 5):
            for name in ("channels", "kernel", "stride", "dilation", "activation", "pool", "residual"):
                ini.remove_section(f"sweep.policy.cnn_{name}_{stage}")
    dimensions = {s[6:] for s in ini.sections() if s.startswith("sweep.")}
    mode = ini.getfloat("env", "representation_mode", fallback=0)
    appearance_seed = ini.getfloat("env", "representation_seed", fallback=0)
    catalog = ini.getfloat("env", "representation_mix_catalog", fallback=0)
    if catalog not in (0, 1) or (environment not in ("pongcnn", "flappycnn") and catalog != 0):
        raise ValueError("Invalid appearance mix catalog (expanded catalogs: Pong/Flappy only)")
    legacy_count = {"pongcnn": 5, "flappycnn": 4}.get(environment)
    if legacy_count and mode == 1 and catalog == 0 and ini.getfloat("env", "representation") >= legacy_count:
        raise ValueError(f"Legacy mixed {environment} catalog requires representation below {legacy_count}")
    if mode not in (0, 1) or not (0 <= appearance_seed <= 4294967295 and appearance_seed.is_integer()):
        raise ValueError("Invalid fixed appearance mode/seed")
    if mode == 1 and "env.representation" in dimensions:
        raise ValueError("representation is inactive in mixed mode; do not sweep it")
    encoder = ini.getint("policy", "encoder")
    if encoder not in LIMITS or not dimensions or not dimensions <= set(LIMITS[encoder]) | {"train.total_timesteps", "env.representation"}:
        raise ValueError("Fix encoder to 1/2/3/4/5 and sweep a nonempty subset of architecture/budget/representation options")
    for key in list(ini["policy"]):
        if key.startswith("cnn_") and "policy." + key not in LIMITS[encoder]:
            ini.remove_option("policy", key)
    if ini.getint("train", "gpus") != 1 or ini.getint("sweep", "gpus") != 1 or ini.getint("selfplay", "enabled"):
        raise ValueError("This runner requires one GPU and no selfplay")
    if ini.getint("vec", "num_policies") != 1 or ini.getint("base", "eval_episodes") != 0:
        raise ValueError("Use one policy and training metrics for the discovery sweep")
    if max_runs is not None:
        ini.set("sweep", "max_runs", str(max_runs))
    if ini.getint("sweep", "max_runs") < 1:
        raise ValueError("max-runs must be positive")
    if ini.get("sweep", "metric") != "perf":
        raise ValueError("Use sweep.metric=perf (environment-specific training performance)")
    appearances = {"connect4cnn": 10, "pongcnn": 7, "flappycnn": 7, "breakoutcnn": 5, "snakecnn": 6, "mazecnn": 6}
    limits = {**LIMITS[encoder], "env.representation": ("int_uniform", set(range(appearances[environment])))}
    for key, (distribution, allowed) in limits.items():
        section = "sweep." + key
        base_section, name = key.split(".")
        if not ini.has_option(base_section, name):
            stage = name.rsplit("_", 1)[-1]
            if (encoder == 5 and stage.isdigit() and
                    int(stage) > ini.getint("policy", "cnn_depth") and
                    not ini.has_section(section)):
                continue
            raise ValueError(f"Missing fixed architecture setting: {key}")
        value = ini.getfloat(base_section, name)
        if value not in allowed:
            raise ValueError(f"Unsupported architecture default: {key}")
        if not ini.has_section(section):
            continue
        lo, hi = ini.getfloat(section, "min"), ini.getfloat(section, "max")
        if ini.get(section, "distribution") != distribution or lo not in allowed or hi not in allowed or not lo < hi or not lo <= value <= hi:
            raise ValueError(f"Unsupported architecture range/default: {key}")
    batch = ini.getint("vec", "total_agents") * ini.getint("train", "horizon")
    budget = "sweep.train.total_timesteps"
    default_steps = ini.getfloat("train", "total_timesteps")
    if default_steps < batch:
        raise ValueError("Training budget must cover at least one rollout batch")
    if ini.has_section(budget):
        lo, hi = ini.getfloat(budget, "min"), ini.getfloat(budget, "max")
        if not batch <= lo < hi or not lo <= default_steps <= hi:
            raise ValueError("Budget range must cover the default and at least one rollout batch")
    if os.environ.get("NVCC_PREPEND_FLAGS"):
        raise ValueError("Unset NVCC_PREPEND_FLAGS for this experimental-only sweep")
    ini.set("base", "checkpoint_dir", str(out / "checkpoints"))
    ini.set("base", "log_dir", str(out / "metrics"))
    (out / "config").mkdir()
    with (out / "config/default.ini").open("w") as f:
        ini.write(f)
    (out / "config" / f"{environment}.ini").write_text("# All effective settings are in this campaign's default.ini.\n")
    (out / "environment.txt").write_text(environment + "\n")
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
    environment = (out / "environment.txt").read_text().strip() if (out / "environment.txt").exists() else "connect4cnn"
    labels = {"connect4cnn": "Training wins", "pongcnn": "Training point fraction", "flappycnn": "Training perf (clipped pipes/20)", "breakoutcnn": "Training normalized score", "snakecnn": "Training perf (clipped ending length/120)", "mazecnn": "Training native logged goal reward"}
    fields = ["index", "run_id", "name", "family", "representation", "representation_mode", "representation_seed", "representation_mix_catalog", "channels", "depth", "stride", "projection", "blocks", "global_pool", "architecture_json", "steps", "score", "cost", "native_avg_sps", "params", "random", "gp_obs", "architecture_sha256", "pareto"]
    def appearance(row):
        mode = row.get("representation_mode", 0)
        return (mode, row.get("representation_seed", 0) if mode else row.get("representation", 0),
                row.get("representation_mix_catalog", 0) if mode else 0)
    flat = []
    for row in rows:
        dominated = any(appearance(r) == appearance(row) and
                        r["cost"] <= row["cost"] and r["score"] >= row["score"] and
                        (r["cost"] < row["cost"] or r["score"] > row["score"]) for r in rows)
        shape = row["architecture"]
        flat.append({**{k: row[k] for k in fields if k in row},
                     **{k: shape.get("policy.cnn_" + k, "") for k in ("channels", "depth", "stride", "projection", "blocks", "global_pool")},
                     "name": display_name(out, row["index"]), "family": shape["version"],
                     "representation": row.get("representation", 0),
                     "representation_mode": row.get("representation_mode", 0),
                     "representation_seed": row.get("representation_seed", 0),
                     "representation_mix_catalog": row.get("representation_mix_catalog", 0),
                     "architecture_json": json.dumps(shape, sort_keys=True), "pareto": not dominated})
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)
    lines = ["# Experimental CNN native PROTEIN sweep", "", f"Status: {status}. Sweep process wall: {wall:.3f} seconds.",
             f"Native {environment}, image dimensions 1x36x44; representation, architecture and budget follow the saved config. Learner/core recipe is locked.",
             "Training metrics, not held-out evaluation. Cost is native adjusted uptime rounded by PROTEIN stdout; SPS = actual decisions / that cost.",
             "Training observations guide discovery; held-out evaluation and repeated seeds are required to confirm finalists. Report Pareto flags use rounded final observations within each appearance assignment (fixed ID or mixed seed); native PROTEIN still optimizes the joint search objective. CSV retains mode and seed.", "",
             f"| Run | Family | Representation | Architecture settings | Steps | {labels[environment]} | Cost s | Native avg SPS | Parameters | GP proposal | Pareto |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for r in flat:
        shape = ", ".join(f"{k.removeprefix('policy.cnn_')}={v}" for k, v in json.loads(r["architecture_json"]).items() if k.startswith("policy.cnn_")) or "fixed Nature"
        appearance_label = f"mixed seed {r['representation_seed']}, catalog {r['representation_mix_catalog']}" if r['representation_mode'] else str(r['representation'])
        lines.append(f"| {r['name']} | {r['family']} | {appearance_label} | {shape} | {r['steps']:,} | {r['score']:.2%} | {r['cost']:.2f} | {r['native_avg_sps']:,.0f} | {r['params']:,} | {bool(r['gp_obs'])} | {r['pareto']} |")
    lines += ["", "Source/config/binary hashes: protocol.json. Effective frozen config: config/default.ini.",
              "Per-trial INIs and checkpoints retain exact shapes; sidecar JSON joins these with final native observations.",
              "Whole sweep wall includes PROTEIN search and worker startup, but excludes compilation and final W&B synchronization. Online logging runs concurrently on CPU.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=HERE / "sweep.ini")
    parser.add_argument("--environment", choices=("connect4cnn", "pongcnn", "flappycnn", "breakoutcnn", "snakecnn", "mazecnn"), default="connect4cnn")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3, 4), help="Fix encoder 5 depth and omit inactive sweep dimensions")
    parser.add_argument("--appearance-seed", type=int, help="Use deterministic per-slot mixtures of all current drawings; not a sweep dimension")
    parser.add_argument("--timeout", type=int, default=600, help="Hard deadline for the entire sweep and its worker process group")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="offline")
    parser.add_argument("--project", default="puffer-cnn")
    parser.add_argument("--entity")
    parser.add_argument("--canary", action="store_true", help="Keep the chosen family/ranges but use short 16K–64K training budgets")
    parser.add_argument("--prepare-only", action="store_true", help="Save resolved INIs and source receipts without building or accessing a GPU")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    build_dir = ROOT / "build" / args.environment
    build_dir.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="sweep.", dir=build_dir))
    shutil.copy2(args.recipe.resolve(), out / "recipe.ini")
    ini = prepare(out, out / "recipe.ini", args.max_runs, args.depth, args.environment, args.appearance_seed)
    if args.canary:
        ini.set("train", "total_timesteps", "32768")
        if ini.has_section("sweep.train.total_timesteps"):
            ini.set("sweep.train.total_timesteps", "min", "16384")
            ini.set("sweep.train.total_timesteps", "max", "65536")
        with (out / "config/default.ini").open("w") as f:
            ini.write(f)
    print(f"Sweep: {out}", flush=True)
    sources = [ROOT / "build.sh", ROOT / "config/default.ini", ROOT / "config/connect4cnn.ini",
               *sorted((ROOT / "src").glob("*")), *sorted(HERE.glob("*"))]
    if args.environment != "connect4cnn":
        sources += [ROOT / "config" / f"{args.environment}.ini", *sorted((ROOT / "ocean" / args.environment).glob("*"))]
        sources += sorted((ROOT / "ocean" / args.environment / "tests").glob("*"))
    sources = [p for p in sources if p.is_file()]
    sources += sorted((HERE / "tests").glob("*.cu")) + sorted((HERE / "tests").glob("*.py"))
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    protocol = dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    started_utc=datetime.now(timezone.utc).isoformat(), precision="float32", build_arch=os.environ.get("NVCC_ARCH", "native"),
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                    recipe_requested=str(args.recipe.resolve()), recipe_sha256=sha(out / "recipe.ini"),
                    config_sha256=sha(out / "config/default.ini"),
                    dimensions=sorted(s[6:] for s in ini.sections() if s.startswith("sweep.")),
                    max_runs=ini.getint("sweep", "max_runs"), timeout=args.timeout)
    protocol["environment"] = args.environment
    protocol["appearance"] = dict(mode=int(ini.getfloat("env", "representation_mode", fallback=0)),
        seed=int(ini.getfloat("env", "representation_seed", fallback=0)),
        catalog=int(ini.getfloat("env", "representation_mix_catalog", fallback=0)),
        fixed_fallback=int(ini.getfloat("env", "representation")),
        assignment="native-persistent-slot" if ini.getfloat("env", "representation_mode", fallback=0) else "fixed",
        native_vector_initialization_validated=False)
    if args.environment == "snakecnn":
        protocol["rules"] = "snake-local-episodic-v1"
    if args.environment == "mazecnn":
        protocol["rules"] = "maze-native-v1"
    binary = out / "cnn"
    build = ["bash", "build.sh", args.environment, str(binary), "--float"]
    command = [str(binary), "sweep", "--headless"]
    protocol.update(build_command=build, sweep_command=command, sweep_cwd=str(out))
    protocol["status"] = "prepared"
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared only: {out / 'protocol.json'}", flush=True)
        return
    execute(["bash", "-c", 'source ocean/connect4cnn/runtime_env.sh; smi=$(puffer_find_nvidia_smi) && puffer_require_idle_gpu "$smi" && "$smi"'], ROOT, out / "gpu.txt", 20)
    execute(build, ROOT, out / "build.log", 300)
    protocol["binary_sha256"] = sha(binary)
    protocol.update(wandb_mode=args.wandb, wandb_project=args.project, wandb_entity=args.entity,
                    wandb_concurrent=args.wandb == "online")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sidecar = [sys.executable, str(HERE / "wandb_sidecar.py"), str(out), "--mode", args.wandb, "--project", args.project]
    if args.entity:
        sidecar += ["--entity", args.entity]
    follower = None
    if args.wandb == "online":
        with (out / "sidecar.log").open("w") as log:
            follower = subprocess.Popen(sidecar + ["--follow"], cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
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
    if follower is not None:
        try:
            code = follower.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(follower.pid, signal.SIGKILL)
            follower.wait()
            raise RuntimeError(f"W&B synchronization timed out; training results saved in {out}")
        if code:
            raise RuntimeError(f"W&B sidecar failed; training results saved in {out}; see sidecar.log")
    else:
        execute(sidecar, ROOT, out / "sidecar.log", 300)
    print(f"{status}: {len(rows)} trials, {len({r['architecture_sha256'] for r in rows})} architectures; report: {out / 'REPORT.md'}", flush=True)
    if status != "ok":
        raise SystemExit(error)


if __name__ == "__main__":
    main()
