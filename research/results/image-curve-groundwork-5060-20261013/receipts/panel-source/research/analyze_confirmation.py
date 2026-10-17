"""Audit frozen confirmation receipts and summarize complete paired-seed curves."""
import argparse
import configparser
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"flex_quality": "Ours quality", "flex_fast": "Ours fast", "flex_small": "Ours small",
          "nature_cnn": "Nature", "impala_cnn": "IMPALA", "impoola_cnn": "Impoola"}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ini(path):
    c = configparser.ConfigParser(interpolation=None)
    c.read(path)
    return c


def front(points, axis):
    return sorted([p for p in points if not any(q[axis] <= p[axis] and q["wins"] >= p["wins"]
        and (q[axis] < p[axis] or q["wins"] > p["wins"]) for q in points)], key=lambda p: p[axis])


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = protocol["confirmation_manifest"]
    jobs = json.loads((root / "jobs.json").read_text())
    finished = json.loads((root / "finished.json").read_text())
    seeds, steps, variants = protocol["seeds"], protocol["checkpoint_steps"], list(protocol["variants"])
    assert len(seeds) == 5 and protocol["confirmation"] and not protocol["canary"]
    assert finished == {"status": "ok", "jobs": 30, "evaluations": 390, "logging_failures": 0}
    for name, expected in protocol["source_sha256"].items():
        assert sha(root / "source" / name) == expected, name
    for variant, expected in protocol["binary_sha256"].items():
        binary = root / variant
        if not binary.exists() and variant.startswith("flex_"):
            binary = root / "flex_quality"
        assert sha(binary) == expected, variant
    with (root / "results.csv").open() as f:
        raw = list(csv.DictReader(f))
    points, seen, common, gpu_flags = [], set(), None, []
    assert len(jobs) == 30 and len(raw) == 390
    lookup = {(j["variant"], j["seed"]): j for j in jobs}
    assert set(lookup) == set(itertools.product(variants, seeds))
    for key, job in lookup.items():
        variant, seed = key
        trial = root / f"{variant}-s{seed}"
        assert job["status"] == "ok" and job["wandb_status"] == "ok"
        assert sha(trial / "config/default.ini") == job["config_sha256"]
        c = ini(trial / "metrics/connect4cnn/trial.ini")
        assert c.getint("base", "seed") == seed
        assert c.getint("train", "total_timesteps") == manifest["steps"]
        for k, v in manifest["policies"].get(variant, {"encoder": 0}).items():
            assert c.getfloat("policy", k) == v
        effective = {s: dict(c[s]) for s in c if s != "metrics"}
        for k in ("env_name", "seed", "run_id", "checkpoint_dir", "log_dir"):
            effective["base"].pop(k, None)
        effective["policy"] = {k: v for k, v in effective["policy"].items() if k != "encoder" and not k.startswith("cnn_")}
        if common is None:
            common = effective
        assert effective == common, key
        gpu = (trial / "gpu-before.txt").read_text()
        if re.search(r"\|\s+\d+\s+\S+\s+\S+\s+\d+\s+C", gpu):
            gpu_flags.append(str(trial.relative_to(root)))
    for r in raw:
        variant, seed, step = r["variant"], int(r["seed"]), int(r["steps"])
        key = (variant, seed, step)
        assert key not in seen
        seen.add(key)
        assert int(r["eval_seed"]) == protocol["eval_seed"] + seeds.index(seed)
        assert int(r["games"]) >= protocol["eval_games"]
        checkpoint = root / r["checkpoint"]
        assert sha(checkpoint) == lookup[variant, seed]["checkpoint_sha256"][checkpoint.name]
        weights = np.fromfile(checkpoint, dtype=np.float32)
        assert weights.size == int(r["params"]) == manifest["parameters"][variant]
        assert np.isfinite(weights).all()
        point = dict(variant=variant, family=LABELS[variant], seed=seed, steps=step,
            wins=float(r["win_rate"]), seconds=float(r["checkpoint_wall_s"]),
            process_seconds=float(r["train_process_wall_s"]), native_sps=float(r["native_avg_sps"]),
            params=int(r["params"]), games=int(r["games"]))
        assert 0 <= point["wins"] <= 1 and 0 < point["seconds"] <= point["process_seconds"]
        points.append(point)
    assert seen == set(itertools.product(variants, seeds, steps))
    for variant, seed in lookup:
        group = sorted((p for p in points if p["variant"] == variant and p["seed"] == seed), key=lambda p: p["steps"])
        assert all(a["seconds"] < b["seconds"] for a, b in zip(group, group[1:]))
    # All 5^5 ordered bootstrap resamples. Same seed indices for paired differences.
    resamples = np.array(list(itertools.product(range(len(seeds)), repeat=len(seeds))))

    def interval(values):
        values = np.array(values)
        lo, hi = np.quantile(values[resamples].mean(axis=1), [.025, .975])
        return float(lo), float(hi)

    means, finals, differences, thresholds = [], [], [], []
    for variant in variants:
        for step in steps:
            group = sorted((p for p in points if p["variant"] == variant and p["steps"] == step), key=lambda p: p["seed"])
            lo, hi = interval([p["wins"] for p in group])
            means.append(dict(variant=variant, family=LABELS[variant], seed="mean", steps=step,
                wins=float(np.mean([p["wins"] for p in group])), low=lo, high=hi,
                seconds=float(np.mean([p["seconds"] for p in group])), params=group[0]["params"],
                native_sps=float(np.mean([p["native_sps"] for p in group])),
                min_wins=min(p["wins"] for p in group), max_wins=max(p["wins"] for p in group)))
        group = [p for p in points if p["variant"] == variant and p["steps"] == steps[-1]]
        finals.append({**means[-1], "process_seconds": float(np.mean([p["process_seconds"] for p in group]))})
        for threshold in (.5, .75, .9, .95):
            for seed in seeds:
                reached = sorted((p for p in points if p["variant"] == variant and p["seed"] == seed
                                  and p["wins"] >= threshold), key=lambda p: p["steps"])
                thresholds.append(dict(variant=variant, seed=seed, target=threshold, reached=bool(reached),
                    first_seconds=reached[0]["seconds"] if reached else None,
                    first_steps=reached[0]["steps"] if reached else None))
    for variant in ("flex_quality", "flex_fast", "flex_small", "impala_cnn", "impoola_cnn"):
        for step in steps:
            delta = []
            for seed in seeds:
                a = next(p for p in points if p["variant"] == variant and p["seed"] == seed and p["steps"] == step)
                b = next(p for p in points if p["variant"] == "nature_cnn" and p["seed"] == seed and p["steps"] == step)
                delta.append(a["wins"] - b["wins"])
            lo, hi = interval(delta)
            differences.append(dict(variant=variant, reference="nature_cnn", steps=step,
                mean_delta=float(np.mean(delta)), low=lo, high=hi, seeds_higher=sum(d > 0 for d in delta)))
    fronts = []
    for view, group in [("mean", means)] + [(str(s), [p for p in points if p["seed"] == s]) for s in seeds]:
        for axis in ("seconds", "steps"):
            for family in ["Combined", "Ours"] + list(LABELS.values()):
                if family == "Combined":
                    chosen = group
                elif family == "Ours":
                    chosen = [p for p in group if p["variant"].startswith("flex_")]
                else:
                    chosen = [p for p in group if p["family"] == family]
                fronts += [dict(view=view, axis=axis, frontier_family=family, **p) for p in front(chosen, axis)]
    audit = dict(status="ok", jobs=30, checkpoints_verified=390, source_hashes_verified=len(protocol["source_sha256"]),
        matched_configs=True, gpu_snapshot_flags=gpu_flags, bootstrap_resamples=len(resamples),
        total_decisions=30 * manifest["steps"], train_process_seconds=sum(j["train_process_wall_s"] for j in jobs),
        evaluation_seconds=sum(float(r["eval_wall_s"]) for r in raw),
        csv_sha256=sha(root / "results.csv"), protocol_sha256=sha(root / "protocol.json"))
    data = dict(campaign=root.name, points=points, means=means, finals=finals, differences=differences,
                frontiers=fronts, thresholds=thresholds, audit=audit)
    (out / "confirmation.json").write_text(json.dumps(data, indent=2) + "\n")
    for name, rows in (("observations", points), ("means", means), ("finals", finals), ("paired_differences", differences),
                       ("frontiers", fronts), ("thresholds", thresholds)):
        write_csv(out / (name + ".csv"), rows)
    template = (ROOT / "research/confirmation_frontier.html").read_text()
    (out / "frontier.html").write_text(template.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    lines = ["# Frozen confirmation: full numerical results", "",
        "All 30 jobs and 390 checkpoints audited. Five fresh seeds per fixed architecture; same 13 checkpoints and learner settings.",
        "[Interactive chart](frontier.html) · [Every observation](observations.csv) · [All per-seed/family/time/step frontiers](frontiers.csv)", "",
        "## Final checkpoint means", "",
        "| Model | Wins | Pointwise 95% interval | Seed range | Whole-process seconds | Native SPS |",
        "|---|---:|---:|---:|---:|---:|"]
    for p in finals:
        lines.append(f"| {p['family']} | {p['wins']:.2%} | {p['low']:.2%}–{p['high']:.2%} | {p['min_wins']:.2%}–{p['max_wins']:.2%} | {p['process_seconds']:.2f} | {p['native_sps']:,.0f} |")
    lines += ["", "## Entire combined frontier of checkpoint means", "",
        "Includes near-zero early points. Cost is mean launch-to-checkpoint wall time, not whole-process time.", "",
        "| Model | Decisions | Wall seconds | Wins | Pointwise 95% interval |", "|---|---:|---:|---:|---:|"]
    for p in fronts:
        if p["view"] == "mean" and p["axis"] == "seconds" and p["frontier_family"] == "Combined":
            lines.append(f"| {p['family']} | {p['steps']:,} | {p['seconds']:.2f} | {p['wins']:.2%} | {p['low']:.2%}–{p['high']:.2%} |")
    lines += ["", "## Final paired differences versus Nature", "",
        "Same training/evaluation seed pairs at the full step budget. Units are percentage points.", "",
        "| Model | Mean difference | Pointwise 95% interval | Higher-scoring seeds |", "|---|---:|---:|---:|"]
    for d in differences:
        if d["steps"] == steps[-1]:
            lines.append(f"| {LABELS[d['variant']]} | {d['mean_delta']*100:+.2f} | {d['low']*100:+.2f} to {d['high']*100:+.2f} | {d['seeds_higher']}/5 |")
    lines += ["", "Intervals enumerate all 3,125 ordered bootstrap resamples of the five seeds, pairing seed indices for differences. They are pointwise, not simultaneous or multiplicity-adjusted, and five seeds provide limited statistical resolution.",
        "Exploratory first-observed threshold crossings and failures to reach them are retained in thresholds.csv. They are not independently trained shorter-budget runs or guarantees of sustained performance.",
        "No GPU compute processes appeared in pre-job snapshots, but there was no continuous exclusive-resource audit. Brief CPU-only development work occurred during training. Native source snapshots/binaries/checkpoints remain in the original local campaign directory.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"audit": audit, "finals": finals, "final_differences": [d for d in differences if d["steps"] == steps[-1]],
        "mean_time_frontier": [p for p in fronts if p["view"] == "mean" and p["axis"] == "seconds" and p["frontier_family"] == "Combined"]}, indent=2))


if __name__ == "__main__":
    main()
