"""Audit/report Bash-owned native Pong confirmation receipts; no training logic."""
import argparse
import configparser
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

MODELS = {"flex_quality": (4, 160224), "nature_cnn": (2, 138016),
          "impala_cnn": (0, 269984), "impoola_cnn": (0, 151200)}
KEYS = "learning_rate ent_coef gamma gae_lambda replay_ratio clip_coef vf_coef max_grad_norm".split()
RECIPES = dict(zip("ab", [dict(zip(KEYS, values.split())) for values in (
    "0.0164826009 3.94817544e-05 0.969679892 0.979004204 1.91617525 0.297549665 1.75164807 0.538654029",
    "0.0840475857 7.57243834e-05 0.989680648 0.903716803 3.57716227 0.193612725 3.41555738 1.25941002")]))
EVAL = re.compile(r"^CUDA_EVAL env=pongcnn score=([-+\d.eE]+) perf=([-+\d.eE]+) games=(\d+) params=(\d+)$", re.M)
ANALYSIS = {
    "primary": "Ours versus Nature within each shared recipe at 0.95 evaluated episode-averaged point fraction",
    "target": .95, "bootstrap_seed": 20260915, "bootstrap_replicates": 10000,
    "bootstrap_unit": "Whole paired training-seed blocks, shared across all models, checkpoints and recipes",
    "interval": "Pointwise percentile 95%; not simultaneous or frontier-wide dominance",
    "crossing": "First observed qualifying checkpoint; report previous checkpoint time and later declines separately",
    "censoring": "Retain non-achievers and missing evaluations; no success-only crossing-time means",
    "failure_bounds": "For missing final point fraction, report [0,1] identification bounds, not imputed point estimates",
    "frontier": "All observed per-seed sets and complete-case paired mean curves separately by shared recipe; no interpolation",
    "replication": "Fixed five seeds; no powered superiority guarantee, adaptive additions, or tuning against these results",
}

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_ini(*paths):
    ini = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    for path in paths:
        with Path(path).open() as stream:
            ini.read_file(stream)
    return ini

def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def jobs(root):
    with (root / "jobs.tsv").open() as stream:
        return [dict(id=r[0], recipe=r[1], variant=r[2], seed=int(r[3]), eval_seed=int(r[4]))
                for r in csv.reader(stream, delimiter="\t")]

def budget(root):
    return dict(zip(("steps", "checkpoints", "interval", "games", "train_cap", "eval_cap"),
                    map(int, (root / "budget.tsv").read_text().split())))

def normalized(value):
    value = value.strip().strip("\"'")
    try:
        return float(value)
    except ValueError:
        return value

def assert_same(actual, expected, label):
    a, b = normalized(actual), normalized(expected)
    if isinstance(b, float):
        assert isinstance(a, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-10), (label, a, b)
    else:
        assert a == b, (label, a, b)

def validate_config(root, job):
    b = budget(root)
    directory = root / job["id"]
    ini = read_ini(directory / "config/default.ini", directory / "config/pongcnn.ini")
    expected = read_ini(root / "source/config/default.ini", root / "source/ocean/pongcnn/compare.ini")
    for section in list(expected.sections()):
        if section.startswith("sweep."):
            expected.remove_section(section)
    assert not any(s.startswith("sweep.") or s == "metrics" for s in ini.sections())
    for key, value in RECIPES[job["recipe"]].items():
        expected.set("train", key, value)
    expected.set("policy", "encoder", str(MODELS[job["variant"]][0]))
    expected.set("train", "total_timesteps", str(b["steps"]))
    for key, value in {"seed": job["seed"], "run_id": "trial", "result_fd": 0,
                       "checkpoint_interval": b["interval"], "checkpoint_dir": str(directory / "checkpoints"),
                       "log_dir": str(directory / "metrics")}.items():
        expected.set("base", key, str(value))
    assert set(ini.sections()) == set(expected.sections())
    for section in expected.sections():
        assert dict(ini[section]) == dict(expected[section]), (job["id"], section, "configuration drift")
    # Independent locks prevent changing both a common recipe and a generated config.
    locks = {"policy": {"hidden_size": 128, "num_layers": 1, "cnn_depth": 1, "cnn_channels_1": 16,
                         "cnn_kernel_1": 7, "cnn_stride_1": 4, "cnn_pool_1": 0, "cnn_residual_1": 0,
                         "cnn_global_pool": 0, "cnn_projection": 64},
             "train": {"gpus": 1, "horizon": 32, "minibatch_size": 2048, "anneal_lr": 1,
                       "min_lr_ratio": 0, "vf_clip_coef": .2, "anneal_ent_coef": 0, "momentum": .5, "vtrace": 0},
             "vec": {"total_agents": 64, "num_threads": 2, "num_buffers": 1, "num_policies": 1, "hist_policy_percent": 0},
             "base": {"async": 0, "cudagraphs": 1, "eval_episodes": 0, "reset_every_horizon": 0},
             "env": {"frameskip": 8, "continuous": 0}, "selfplay": {"enabled": 0}}
    for section, values in locks.items():
        for key, value in values.items():
            assert ini.getfloat(section, key) == value, (section, key)
    assert normalized(ini.get("base", "load_model_path")) == "None"
    assert ini.get("sweep", "metric") == "score"
    for step in range(b["steps"] // b["checkpoints"], b["steps"] + 1, b["steps"] // b["checkpoints"]):
        evaluation = directory / "evaluations" / f"{step:016d}"
        ev = read_ini(evaluation / "config/default.ini", evaluation / "config/pongcnn.ini")
        for section in ini.sections():
            for key, value in ini[section].items():
                wanted = value
                if section == "base":
                    wanted = {"seed": str(job["eval_seed"]), "eval_episodes": str(b["games"]),
                              "load_model_path": str(directory / f"checkpoints/pongcnn/trial/{step:016d}.bin")}.get(key, value)
                assert ev.get(section, key) == wanted, (job["id"], step, section, key)
    return ini

def prepare(root):
    b = budget(root)
    canary = (root / "mode.txt").read_text().strip() == "canary"
    expected = (65536, 4, 8, 32, 120, 120) if canary else (4194304, 8, 256, 512, 1800, 300)
    assert tuple(b.values()) == expected
    panel = jobs(root)
    seeds = [51001] if canary else list(range(31001, 31006))
    assert len(panel) == len(seeds) * 8
    assert len({j["id"] for j in panel}) == len(panel)
    assert {(j["recipe"], j["variant"], j["seed"], j["eval_seed"]) for j in panel} == {
        (r, v, s, s+10000) for r in RECIPES for v in MODELS for s in seeds}
    for job in panel:
        validate_config(root, job)
    header = (root / "source/ocean/pongcnn/pongcnn.h").read_text()
    for contract in ("#define ACT_SIZES {3}", "#define OBS_CHANNELS 1", "#define OBS_HEIGHT 36", "#define OBS_WIDTH 44"):
        assert contract in header
    sources = {line.split(maxsplit=1)[1].strip().removeprefix("./"): line.split()[0]
               for line in (root / "source.sha256").read_text().splitlines()}
    configs = {str(p.relative_to(root)): sha(p) for job in panel for p in (root / job["id"]).rglob("*.ini")}
    inputs = {p: sha(root / p) for p in ("jobs.tsv", "budget.tsv", "mode.txt", "runtime-5090.sh", "working.patch", "git-status.txt", "revision.txt")}
    protocol = {"version": "pong-5090-replication-v1", "mode": "canary" if canary else "full",
                "revision": (root / "revision.txt").read_text().strip(), "precision": "float32", "build_arch": "sm_120",
                "task": "Native PufferLib PongCNN; three actions; CHW 1x36x44; score bars",
                "metric": "Episode-averaged fraction of points won (not match win rate)", "budget": b,
                "training_seeds": seeds, "evaluation_seeds": [s+10000 for s in seeds], "order": panel,
                "models": MODELS, "recipes": RECIPES, "analysis": ANALYSIS,
                "native_update_semantics": {"expression": "int total_minibatches = replay_ratio * batch_size / minibatch_size",
                    "source": "src/pufferl.cu:1504,1605", "batch_size": 2048, "minibatch_size": 2048,
                    "updates_per_rollout": {"a": 1, "b": 3},
                    "total_updates": {"a": b["steps"]//2048, "b": 3*b["steps"]//2048}},
                "source_sha256": sources, "config_sha256": configs, "input_sha256": inputs,
                "prepared_at": float((root / "prepared-at.txt").read_text()),
                "wandb": "disabled: SDK unavailable in reporting venv; no dependency installation required"}
    save(root / "protocol.prepared.json", protocol)
    save(root / "jobs.json", [{**j, "status": "pending"} for j in panel])
    print(f"Prepared {len(panel)} locked jobs, {len(panel)*b['checkpoints']} evaluation configs")

def verify(root, protocol=None):
    if protocol is None:
        protocol = json.loads((root / "protocol.json").read_text())
        assert sha(root / "protocol.json") == (root / "protocol.sha256").read_text().split()[0], "Protocol hash changed"
    for group, prefix in (("source_sha256", root / "source"), ("config_sha256", root), ("input_sha256", root), ("binary_sha256", root), ("hardware_sha256", root)):
        for name, expected in protocol.get(group, {}).items():
            assert sha(prefix / name) == expected, (group, name, "hash changed")
    return protocol

def freeze(root):
    assert not (root / "protocol.json").exists(), "Protocol already frozen"
    protocol = json.loads((root / "protocol.prepared.json").read_text())
    protocol["binary_sha256"] = {v + ".bin": sha(root / (v + ".bin")) for v in MODELS}
    protocol["frozen_at"] = time.time()
    protocol["hardware_sha256"] = {p: sha(root / p) for p in ("gpu.txt", "cpu.txt", "host.txt", "cuda-compiler.txt", "compilers.txt", "runtime.txt", "dependencies.sha256")}
    verify(root, protocol)
    save(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "  protocol.json\n")
    print("Frozen protocol:", sha(root / "protocol.json"))

def attempt_status(directory):
    receipt = directory / "exit-code.txt"
    if not receipt.exists():
        return "interrupted" if (directory / "started.txt").exists() else "pending"
    code = int(receipt.read_text())
    return "ok" if code == 0 else "timeout" if code in (124, 137) else "failed"

def wall(directory):
    path = directory / "wall.txt"
    if not path.exists():
        return None
    # GNU time writes a diagnostic before its numeric elapsed time on failure.
    return float(path.read_text().splitlines()[-1])

def audit_job(root, job):
    ini = validate_config(root, job)
    directory = root / job["id"]
    assert attempt_status(directory / "training") == "ok"
    metrics_path = directory / "metrics/pongcnn/trial.ini"
    resolved = read_ini(metrics_path)
    for section in ini.sections():
        for key, value in ini[section].items():
            if (section, key) == ("base", "env_name"):
                assert resolved.get(section, key) == "pongcnn"
            else:
                assert_same(resolved.get(section, key), value, (job["id"], section, key))
    metrics = {key: np.array([float(v) for v in value.split(",")]) for key, value in resolved["metrics"].items()}
    assert all(np.isfinite(a).all() for a in metrics.values()), "Nonfinite native metric"
    b = budget(root)
    assert int(metrics["agent_steps"][-1]) == b["steps"]
    uptime = float(metrics["uptime"][-1])
    elapsed = wall(directory / "training")
    assert uptime > 0 and elapsed > 0 and uptime <= elapsed + .2
    checkpoints = sorted((directory / "checkpoints/pongcnn/trial").glob("*.bin"))
    assert [int(p.stem) for p in checkpoints] == list(range(b["steps"]//b["checkpoints"], b["steps"]+1, b["steps"]//b["checkpoints"]))
    started = float((directory / "training/started.txt").read_text())
    hashes, times = {}, {}
    for path in checkpoints:
        weights = np.fromfile(path, dtype=np.float32)
        assert weights.size == MODELS[job["variant"]][1] and np.isfinite(weights).all(), path
        hashes[path.name] = sha(path)
        times[path.stem] = path.stat().st_mtime - started
        assert 0 <= times[path.stem] <= elapsed + .2
    audit = {"status": "ok", "params": MODELS[job["variant"]][1], "train_process_wall_s": elapsed,
             "native_uptime_s": uptime, "process_sps": b["steps"]/elapsed, "native_avg_sps": b["steps"]/uptime,
             "native_last_sps": float(metrics["sps"][-1]), "vram_last_gb": float(metrics["util/vram_used_gb"][-1]),
             "checkpoint_sha256": hashes, "checkpoint_wall_s": times, "resolved_config_sha256": sha(metrics_path)}
    save(directory / "training-audit.json", audit)
    return audit

def audit_eval(root, job, checkpoint):
    evaluation = root / job["id"] / "evaluations" / checkpoint
    status = attempt_status(evaluation)
    assert status != "failed", (job["id"], checkpoint, "native evaluation failed")
    result = {"status": status, "eval_wall_s": wall(evaluation)}
    if status == "ok":
        matches = EVAL.findall((evaluation / "output.log").read_text())
        assert len(matches) == 1, "Missing or nonfinite native Pong evaluation output"
        score, perf, games, params = matches[0]
        score, perf, games, params = float(score), float(perf), int(games), int(params)
        assert math.isfinite(score) and -21 <= score <= 21 and 0 <= perf <= 1
        assert games >= budget(root)["games"] and params == MODELS[job["variant"]][1]
        result.update(point_fraction=perf, score=score, games=games, params=params)
    save(evaluation / "audit.json", result)
    return result

FIELDS = ["id", "recipe", "variant", "seed", "eval_seed", "steps", "status", "point_fraction", "score", "games", "params",
          "checkpoint_wall_s", "train_process_wall_s", "process_sps", "native_uptime_s", "native_avg_sps", "native_last_sps",
          "vram_last_gb", "eval_wall_s", "checkpoint_sha256"]

def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def collect(root):
    b = budget(root)
    panel, rows = [], []
    for job in jobs(root):
        directory = root / job["id"]
        status = attempt_status(directory / "training")
        audit = json.loads((directory / "training-audit.json").read_text()) if (directory / "training-audit.json").exists() else {}
        evaluations = []
        for step in range(b["steps"]//b["checkpoints"], b["steps"]+1, b["steps"]//b["checkpoints"]):
            checkpoint = f"{step:016d}"
            evpath = directory / "evaluations" / checkpoint / "audit.json"
            ev = json.loads(evpath.read_text()) if evpath.exists() else {"status": "pending" if status in ("ok", "pending") else "training_"+status}
            row = {key: None for key in FIELDS}
            row.update(job, steps=step, status=ev["status"], params=MODELS[job["variant"]][1])
            for key in FIELDS:
                if key in audit and key != "status" and not isinstance(audit[key], dict): row[key] = audit[key]
                if key in ev: row[key] = ev[key]
            row["checkpoint_wall_s"] = audit.get("checkpoint_wall_s", {}).get(checkpoint)
            row["checkpoint_sha256"] = audit.get("checkpoint_sha256", {}).get(checkpoint+".bin")
            rows.append(row)
            evaluations.append(ev["status"])
        overall = status
        if status == "ok":
            overall = "ok" if all(e == "ok" for e in evaluations) else "eval_pending" if "pending" in evaluations else "completed_with_eval_failures"
        panel.append({**job, "status": overall, "training_status": status, "train_process_wall_s": wall(directory / "training"), "evaluation_statuses": evaluations})
    save(root / "jobs.json", panel)
    write_csv(root / "results.csv", rows, FIELDS)
    return panel, rows

def pareto(points, x="time", y="score"):
    return [not any(q[x] <= p[x] and q[y] >= p[y] and (q[x] < p[x] or q[y] > p[y]) for q in points) for p in points]

def interval(values, indices):
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(axis=1)
    return [float(values.mean()), *map(float, np.quantile(draws, [.025, .975]))]

def analyze(root):
    protocol = verify(root)
    panel, rows = collect(root)
    seeds = protocol["training_seeds"]
    n = len(seeds)
    indices = np.random.default_rng(ANALYSIS["bootstrap_seed"]).integers(0, n, (ANALYSIS["bootstrap_replicates"], n))
    b = budget(root)
    lookup = {(r["recipe"], r["variant"], r["seed"], r["steps"]): r for r in rows}
    finals, means, targets, paired = [], [], [], []
    def values(recipe, variant, step, key):
        return [lookup[recipe, variant, seed, step][key] for seed in seeds]
    for recipe in RECIPES:
        for variant in MODELS:
            for step in range(b["steps"]//b["checkpoints"], b["steps"]+1, b["steps"]//b["checkpoints"]):
                score = values(recipe, variant, step, "point_fraction")
                times = values(recipe, variant, step, "checkpoint_wall_s")
                if all(v is not None for v in score+times):
                    s, lo, hi = interval(score, indices)
                    t, tl, th = interval(times, indices)
                    means.append(dict(recipe=recipe, variant=variant, steps=step, time=t, time_low=tl, time_high=th,
                                      score=s, score_low=lo, score_high=hi))
            score = values(recipe, variant, b["steps"], "point_fraction")
            present = sum(v is not None for v in score)
            bounds = [sum(v or 0 for v in score)/n, (sum(v or 0 for v in score)+n-present)/n]
            stats = interval(score, indices) if present == n else [None, None, None]
            record = dict(recipe=recipe, variant=variant, observed_seeds=present, total_seeds=n,
                          point_fraction=stats[0], ci_low=stats[1], ci_high=stats[2], missing_bound_low=bounds[0], missing_bound_high=bounds[1])
            for key in ("score", "train_process_wall_s", "process_sps", "native_avg_sps", "vram_last_gb"):
                v = values(recipe, variant, b["steps"], key)
                record[key] = float(np.mean(v)) if all(x is not None for x in v) else None
            finals.append(record)
            for seed in seeds:
                curve = [r for r in rows if r["recipe"] == recipe and r["variant"] == variant and r["seed"] == seed]
                observed = [r for r in curve if r["status"] == "ok"]
                crossing = next((r for r in observed if r["point_fraction"] >= .95), None)
                earlier = [r for r in observed if crossing and r["steps"] < crossing["steps"]]
                later = [r for r in curve if crossing and r["steps"] > crossing["steps"]]
                targets.append(dict(recipe=recipe, variant=variant, seed=seed, reached=crossing is not None,
                    first_observed_time=crossing["checkpoint_wall_s"] if crossing else None,
                    first_observed_steps=crossing["steps"] if crossing else None,
                    previous_observed_time=earlier[-1]["checkpoint_wall_s"] if earlier else 0,
                    last_observed_time=observed[-1]["checkpoint_wall_s"] if observed else None,
                    missing_evaluations=len(curve)-len(observed),
                    later_decline=any(r["status"] == "ok" and r["point_fraction"] < .95 for r in later),
                    sustained_observed=bool(crossing and all(r["status"] == "ok" and r["point_fraction"] >= .95 for r in later))))
        ours = values(recipe, "flex_quality", b["steps"], "point_fraction")
        nature = values(recipe, "nature_cnn", b["steps"], "point_fraction")
        delta = interval(np.array(ours)-np.array(nature), indices) if all(v is not None for v in ours+nature) else [None]*3
        reached = {v: [float(next(t for t in targets if t["recipe"] == recipe and t["variant"] == v and t["seed"] == s)["reached"]) for s in seeds] for v in ("flex_quality", "nature_cnn")}
        attainment = interval(np.array(reached["flex_quality"])-np.array(reached["nature_cnn"]), indices)
        paired.append(dict(recipe=recipe, final_point_fraction_difference=delta, observed_attainment_difference=attainment))
    # Fronts use all observed checkpoints, separately by recipe and seed.
    fronts = []
    for recipe in RECIPES:
        for seed in seeds:
            points = [dict(recipe=r["recipe"], seed=r["seed"], variant=r["variant"], steps=r["steps"], time=r["checkpoint_wall_s"], score=r["point_fraction"])
                      for r in rows if r["recipe"] == recipe and r["seed"] == seed and r["status"] == "ok"]
            for p, tf, sf in zip(points, pareto(points), pareto(points, x="steps")):
                fronts.append(dict(p, time_pareto=tf, steps_pareto=sf))
        points = [p for p in means if p["recipe"] == recipe]
        for p, tf, sf in zip(points, pareto(points), pareto(points, x="steps")):
            p.update(time_pareto=tf, steps_pareto=sf)
    # Whole-seed bootstrap frontier membership; pointwise descriptive stability.
    for recipe in RECIPES:
        points = [p for p in means if p["recipe"] == recipe]
        if not points: continue
        score_draws = np.array([np.array(values(recipe, p["variant"], p["steps"], "point_fraction"))[indices].mean(axis=1) for p in points])
        time_draws = np.array([np.array(values(recipe, p["variant"], p["steps"], "checkpoint_wall_s"))[indices].mean(axis=1) for p in points])
        for i, p in enumerate(points):
            dominated = ((time_draws <= time_draws[i]) & (score_draws >= score_draws[i]) &
                         ((time_draws < time_draws[i]) | (score_draws > score_draws[i]))).any(axis=0)
            p["bootstrap_time_front_fraction"] = float((~dominated).mean())
    for name, data in (("final-summary", finals), ("mean-curves", means), ("seed-frontiers", fronts), ("target-attainment", targets)):
        save(root / (name+".json"), data)
        if data: write_csv(root / (name+".csv"), data)
    save(root / "paired-comparisons.json", paired)
    train_sum = sum(j["train_process_wall_s"] or 0 for j in panel)
    eval_sum = sum(r["eval_wall_s"] or 0 for r in rows)
    build_sum = sum(wall(p) or 0 for p in (root / "builds").iterdir())
    elapsed = float((root / "campaign-ended.txt").read_text()) - float((root / "campaign-started.txt").read_text())
    active = sum(float((p / "ended.txt").read_text())-float((p / "started.txt").read_text()) for p in root.glob("segment.*") if (p / "ended.txt").exists())
    complete = sum(j["training_status"] == "ok" for j in panel)
    eval_ok = sum(r["status"] == "ok" for r in rows)
    receipt = dict(status="ok" if complete == len(panel) and eval_ok == len(rows) else "completed_with_failures",
                   training_completed=complete, training_planned=len(panel), evaluations_completed=eval_ok,
                   evaluations_planned=len(rows), train_process_sum_s=train_sum, evaluation_sum_s=eval_sum,
                   build_sum_s=build_sum, overall_elapsed_s=elapsed, active_execution_segments_s=active,
                   protocol_sha256=sha(root / "protocol.json"), source_files=len(protocol["source_sha256"]))
    save(root / "finished.json", receipt)
    save(root / "audit.json", {**receipt, "completed_training_weights_audited": complete, "successful_evaluation_outputs_audited": eval_ok, "missing_evaluations": [
        {k:r[k] for k in ("id", "steps", "status")} for r in rows if r["status"] != "ok"]})
    def pct(x): return "missing" if x is None else f"{100*x:.2f}%"
    def number(x): return "missing" if x is None else f"{x:,.2f}"
    lines = ["# Locked-architecture Pong comparison on RTX 5090", "",
        f"{complete}/{len(panel)} training runs and {eval_ok}/{len(rows)} checkpoint evaluations completed. Status: **{receipt['status']}**.",
        "Native PufferLib PongCNN, not Atari/ALE. Point fraction is episode-averaged fraction of points won, not match win rate.", "",
        f"Training-process sum: **{train_sum:.2f} s**; evaluation sum: **{eval_sum:.2f} s**; builds: **{build_sum:.2f} s**; complete campaign elapsed: **{elapsed:.2f} s** (includes any resume downtime, excludes preparation and this final analysis/package).",
        "Full-process training includes initialization and checkpoint writes; native uptime is the trainer's adjusted timer. Checkpoint times use file timestamps relative to train launch and are approximate. Evaluation runs only after uninterrupted training.", "",
        "## Final checkpoints, paired training-seed uncertainty", "",
        "| Recipe | Model | Parameters | Observed seeds | Mean point fraction [pointwise 95% CI] | Mean point-score difference | Mean train s | Process SPS | Native avg SPS |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in finals:
        lines.append(f"| {r['recipe'].upper()} | {r['variant']} | {MODELS[r['variant']][1]:,} | {r['observed_seeds']}/{n} | {pct(r['point_fraction'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}] | {number(r['score'])} | {number(r['train_process_wall_s'])} | {number(r['process_sps'])} | {number(r['native_avg_sps'])} |")
        if r["observed_seeds"] < n:
            lines.append(f"\nMissing-result identification bounds for {r['recipe']}/{r['variant']}: [{pct(r['missing_bound_low'])}, {pct(r['missing_bound_high'])}]. No available-case mean replaces the full panel.\n")
    lines += ["", "## Predeclared primary target: 95% point fraction", "",
        "| Recipe | Model | Seeds reaching target [pointwise 95% CI] | Sustained at all subsequent observed checkpoints | Later declines |",
        "|---|---|---|---:|---:|"]
    for recipe in RECIPES:
        for variant in MODELS:
            group = [t for t in targets if t["recipe"] == recipe and t["variant"] == variant]
            _, lo, hi = interval([float(t['reached']) for t in group], indices)
            lines.append(f"| {recipe.upper()} | {variant} | {sum(t['reached'] for t in group)}/{n} [{pct(lo)}, {pct(hi)}] | {sum(t['sustained_observed'] for t in group)}/{n} | {sum(t['later_decline'] for t in group)} |")
    lines += ["", "First crossings, previous/last observed checkpoint times, missing evaluations, non-achievement and later declines are retained for every seed in `target-attainment.csv`. Crossing resolution is 524,288 decisions for full runs (16,384 for canaries). The bracket between the prior observed checkpoint and first observed crossing is sampling resolution, not proof of monotonic learning or a continuously observed first passage. No success-only time average is used.", ""]
    for p in paired:
        mean, low, high = p["final_point_fraction_difference"]
        reach, rl, rh = p["observed_attainment_difference"]
        lines.append(f"Recipe {p['recipe'].upper()}, ours minus Nature: final point fraction {pct(mean)} [{pct(low)}, {pct(high)}]; observed target-attainment difference {pct(reach)} [{pct(rl)}, {pct(rh)}]. These are paired, pointwise 95% bootstrap intervals; failed evaluations can leave attainment unknown.")
    lines += ["", "Individual first observed crossing times in seconds (`censored` means the target was not observed; missing evaluations are listed separately):", "",
              "| Recipe | Model | Seed | First observed crossing s | Previous observed checkpoint s | Missing evaluations |",
              "|---|---|---:|---|---:|---:|"]
    for t in targets:
        crossing = number(t['first_observed_time']) if t['reached'] else 'censored'
        lines.append(f"| {t['recipe'].upper()} | {t['variant']} | {t['seed']} | {crossing} | {number(t['previous_observed_time'])} | {t['missing_evaluations']} |")
    lines += ["", "## Failures and missing observations", ""]
    failures = [r for r in rows if r['status'] != 'ok']
    lines += [f"- {r['id']} at {r['steps']:,} decisions: {r['status']} (score missing)." for r in failures] or ["No failed or missing checkpoint evaluations."]
    lines += ["", "## Complete observed frontiers", "",
        "`results.csv` retains every planned checkpoint, including failures with blank scores. `seed-frontiers.csv` contains every valid per-seed observation with wall-time and decision-count Pareto flags. `mean-curves.csv` contains all checkpoint means requiring all paired seeds, pointwise score/time intervals, both Pareto flags, and descriptive bootstrap time-front membership. `curves.html` plots all raw seed and mean curves without interpolation beyond connecting observed points; use the raw points for frontier claims.", "",
        "Bootstrap uses 10,000 resamples of whole paired training-seed blocks, with the same draws across both recipes, all models and checkpoints. Five seeds estimate reliability/variance and do not provide a pre-established power guarantee. Pointwise intervals and frontier-membership frequencies do not establish frontier-wide dominance. Missing evaluations are neither zero scores nor independent training replications. A two-recipe envelope, if computed later, is descriptive and secondary to the separate shared-recipe results.", "",
        "## Reproduction and provenance", "",
        f"Revision: `{protocol['revision']}`. Protocol SHA256: `{sha(root / 'protocol.json')}`.",
        "The saved source manifest includes untracked runner files, native core, encoders, environment, vendor files and exact configs. `working.patch`, `git-status.txt`, binaries/dependency hashes, runtime helper, GPU/CPU/compiler records, and per-process command/start/end/wall/exit receipts identify the measured implementation.", "",
        "```bash", "source build/connect4cnn/runtime-5090.sh", "bash ocean/pongcnn/confirm.sh --full",
        f"bash ocean/pongcnn/confirm.sh --resume {root}",
        f".venv/bin/python {root}/source/ocean/pongcnn/confirm_report.py {root} analyze", "```", "",
        "Resume skips every started training attempt and every started evaluation; completed training with pending evaluations continues without retraining. Interrupted attempts remain censored. Numerical/configuration failures create `fatal.txt` and require investigation. No automatic longer-cap retries occur.", "",
        "## Scope and limitations", "",
        "The four architectures and H128/L1 core were fixed. Recipe A/B learner values were selected from G240 development trials 5/7 and crossed with every model. Original floating replay ratios are preserved; native integer conversion yields one/three minibatch updates per 2,048-decision rollout respectively. Schedules anneal over the complete fixed budget, so earlier checkpoints are not independently tuned shorter runs.", "",
        "G240 development used one seed with unequal coverage: 24 trials each for ours/Nature and eight each for IMPALA/Impoola, retaining two failed evaluations. The original common .001-LR Pong pilot failed to learn; alternative development learners worked. This is architecture transfer to a game excluded from architecture selection, but Pong has already informed learner development and is not an untouched-task test. Backend efficiency and baseline tuning limitations remain; no SOTA or universal architectural claim follows.", "",
        "W&B is disabled because its SDK is absent from the existing reporting venv. Local native receipts are authoritative; no training or environment package was installed.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    render_curves(root, rows, means)
    print(json.dumps(receipt, indent=2))

def render_curves(root, rows, means):
    # Standalone SVG embedded in HTML: no plotting dependency/network asset.
    import html
    colors = dict(zip(MODELS, ("#2563eb", "#d97706", "#059669", "#9333ea")))
    chunks = ["<!doctype html><meta charset='utf-8'><title>Pong checkpoint curves</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}svg{width:100%;background:#fafafa}text{font:13px system-ui}section{margin-bottom:32px}</style><h1>Observed Pong checkpoint curves</h1><p>Point fraction, not match win rate. Faint curves are individual seeds; bold curves are paired-seed means. Markers are measured checkpoints. Lines only connect observations; missing points break curves.</p>"]
    chunks.append("<p>" + " · ".join(f"<span style='color:{c}'>{v}</span>" for v,c in colors.items()) + "</p>")
    for recipe in RECIPES:
        for axis in ("checkpoint_wall_s", "steps"):
            valid = [r for r in rows if r["recipe"] == recipe and r["status"] == "ok"]
            xmax = max((r[axis] for r in valid), default=1)
            chunks.append(f"<section><h2>Recipe {recipe.upper()} — {'training wall seconds' if axis == 'checkpoint_wall_s' else 'agent decisions'}</h2><svg viewBox='0 0 1000 430'>")
            for tick in range(6):
                y = 370-tick*64
                chunks.append(f"<path d='M60 {y}H960' stroke='#ddd'/><text x='20' y='{y+5}'>{tick/5:.1f}</text>")
                x=60+tick*180
                chunks.append(f"<text x='{x-20}' y='402'>{xmax*tick/5:,.0f}</text>")
            for variant, color in colors.items():
                seeds = sorted({r["seed"] for r in rows})
                for seed in seeds + [None]:
                    if seed is None:
                        curve = [dict(p, point_fraction=p["score"], checkpoint_wall_s=p["time"], status="ok") for p in means if p["recipe"] == recipe and p["variant"] == variant]
                    else:
                        curve = [r for r in rows if r["recipe"] == recipe and r["variant"] == variant and r["seed"] == seed]
                    previous = None
                    for p in curve:
                        if p["status"] != "ok": previous=None; continue
                        x,y=60+900*p[axis]/xmax,370-320*p["point_fraction"]
                        opacity,width=(1,3) if seed is None else (.25,1)
                        if previous is not None:
                            chunks.append(f"<path d='M{previous[0]:.2f} {previous[1]:.2f}L{x:.2f} {y:.2f}' stroke='{color}' opacity='{opacity}' stroke-width='{width}'/>")
                        title=html.escape(f"{variant} seed={seed or 'mean'} steps={p['steps']} point_fraction={p['point_fraction']:.6f} time={p['checkpoint_wall_s']:.3f}")
                        chunks.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{3 if seed is None else 2}' fill='{color}' opacity='{opacity}'><title>{title}</title></circle>")
                        previous=(x,y)
            chunks.append("</svg></section>")
    (root / "curves.html").write_text("\n".join(chunks))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("action", choices=("prepare", "freeze", "verify", "audit-job", "audit-eval", "collect", "analyze", "require-success"))
    parser.add_argument("--job")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action in ("prepare", "freeze", "verify", "collect", "analyze"):
        globals()[args.action](root)
    elif args.action == "require-success":
        receipt = json.loads((root / "finished.json").read_text())
        assert receipt["status"] == "ok", receipt
    else:
        job = next(j for j in jobs(root) if j["id"] == args.job)
        if args.action == "audit-job": audit_job(root, job)
        else: audit_eval(root, job, args.checkpoint)

if __name__ == "__main__":
    main()
