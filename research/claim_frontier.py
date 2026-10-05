"""Whole-seed frontier analysis; candidate joint bands are NOT yet calibrated.

No model execution. Bootstrap blocks retain all models/checkpoints and their
time/score correlation. A descriptive envelope is not a deployment selector.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def frontier(time, score):
    time, score = np.asarray(time), np.asarray(score)
    dominates = ((time[:, None] <= time[None, :]) & (score[:, None] >= score[None, :])
                 & ((time[:, None] < time[None, :]) | (score[:, None] > score[None, :])))
    return ~dominates.any(axis=0)


def paired_bands(values, draws=2000, seed=93017, alpha=.05):
    """Experimental max standardized bootstrap error over BOTH coordinates.

    values: [paired training seed, fixed model/checkpoint point, (time, score)].
    Original sample standard errors are used, not studentized resample errors.
    Zero-variance cells get the entire feasible domain rather than false certainty.
    This is an asymptotic candidate procedure, not a finite-sample coverage claim.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != 2 or not np.isfinite(x).all():
        raise ValueError("Need a finite complete paired seed/point/time-score array")
    if np.any(x[:, :, 0] <= 0) or np.any((x[:, :, 1] < 0) | (x[:, :, 1] > 1)):
        raise ValueError("Invalid time or bounded score")
    if draws < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid resampling settings")
    mean = x.mean(axis=0)
    se = x.std(axis=0, ddof=1)/np.sqrt(len(x))
    usable = se > 1e-12
    rng = np.random.default_rng(seed)
    maxima, memberships = [], np.zeros(x.shape[1], dtype=int)
    for _ in range(draws):
        sample = x[rng.integers(0, len(x), len(x))].mean(axis=0)
        maxima.append(float(np.max(np.abs(sample-mean)[usable]/se[usable])) if usable.any() else 0.)
        memberships += frontier(sample[:, 0], sample[:, 1])
    critical = float(np.quantile(maxima, 1-alpha, method="higher"))
    low, high = mean-critical*se, mean+critical*se
    low[:, 0] = np.maximum(low[:, 0], 0)
    low[:, 1], high[:, 1] = np.maximum(low[:, 1], 0), np.minimum(high[:, 1], 1)
    low[~usable] = 0
    high[:, 0][~usable[:, 0]] = np.inf
    high[:, 1][~usable[:, 1]] = 1
    return dict(mean=mean, low=low, high=high, critical=critical,
                degenerate_cells=int((~usable).sum()), membership=memberships/draws)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(out, protocol, points, failures, costs=None):
    expected = [(m, s, k) for m in protocol["models"] for s in protocol["training_seeds"] for k in protocol["checkpoint_steps"]]
    lookup = {(p["model"], p["seed"], p["steps"]): p for p in points}
    if len(lookup) != len(points) or set(lookup)-set(expected):
        raise ValueError("Duplicate/unexpected analysis points")
    missing = [dict(model=m, seed=s, steps=k) for m, s, k in expected if (m, s, k) not in lookup]
    means, cells = [], []
    for model in protocol["models"]:
        for step in protocol["checkpoint_steps"]:
            group = [lookup.get((model, seed, step)) for seed in protocol["training_seeds"]]
            if any(p is None for p in group):
                continue  # Missing cells stay in missing.csv, never drop bad seeds.
            cells.append((model, step))
            means.append(dict(model=model, steps=step, seeds=len(group),
                              seconds=float(np.mean([p["seconds"] for p in group])),
                              win_rate=float(np.mean([p["win_rate"] for p in group]))))
    complete = not missing and not failures
    bands = None
    if complete and len(protocol["training_seeds"]) >= 2:
        array = np.array([[[lookup[m, s, k]["seconds"], lookup[m, s, k]["win_rate"]]
                           for m, k in cells] for s in protocol["training_seeds"]])
        bands = paired_bands(array)
        for index, mean in enumerate(means):
            mean.update(time_low=float(bands["low"][index, 0]),
                        time_high=float(bands["high"][index, 0]) if np.isfinite(bands["high"][index, 0]) else None,
                        score_low=float(bands["low"][index, 1]), score_high=float(bands["high"][index, 1]),
                        bootstrap_frontier_fraction=float(bands["membership"][index]))
    if means:
        for axis in ("seconds", "steps"):
            flags = frontier([m[axis] for m in means], [m["win_rate"] for m in means])
            for row, flag in zip(means, flags):
                row["frontier_"+axis] = bool(flag)
    result = dict(status="complete_descriptive" if complete else "incomplete",
                  claim_status="exploratory_only_inference_and_runtime_gates_pending",
                  purpose=protocol["purpose"], points=points, means=means, missing=missing, failures=failures,
                  environment_rules=protocol.get("environment_rules", "legacy-unversioned-see-source"),
                  bootstrap=dict(method="paired whole-seed maximum standardized error", draws=2000, seed=93017,
                                 nominal_joint_coverage=.95, calibrated=False,
                                 degenerate_cells=bands["degenerate_cells"] if bands else None),
                  timing=protocol["timing"], costs=costs or {})
    (out/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    write_csv(out/"observations.csv", points, ["model", "seed", "steps", "seconds", "win_rate", "games", "parameters", "process_sps", "native_sps", "train_seconds"])
    write_csv(out/"means.csv", means, ["model", "steps", "seeds", "seconds", "win_rate", "time_low", "time_high", "score_low", "score_high", "bootstrap_frontier_fraction", "frontier_seconds", "frontier_steps"])
    write_csv(out/"missing.csv", missing, ["model", "seed", "steps"])
    lines = ["# Complete observed curves — descriptive analysis", "",
             f"Status: **{result['status']}**. Purpose: `{protocol['purpose']}`.", "",
             f"Environment rules: `{result['environment_rules']}`. Do not pool different rules revisions.", "",
             "No superiority/SOTA inference is authorized. Candidate simultaneous bands are uncalibrated.",
             "A canary validates measurement plumbing; it is not a learning comparison.", "",
             f"Observed cells: {len(points)}/{len(expected)}; missing: {len(missing)}; failures: {len(failures)}.",
             "[All curves](curves.html) · [Every observation](observations.csv) · [Means](means.csv) · [Missing](missing.csv)", "",
             "## Full observed frontier of complete checkpoint means", "",
             "| Model | Decisions | Mean completion seconds | Mean wins |", "|---|---:|---:|---:|"]
    for row in sorted(means, key=lambda r: r["seconds"]):
        if row["frontier_seconds"]:
            lines.append(f"| {row['model']} | {row['steps']} | {row['seconds']:.6f} | {row['win_rate']:.2%} |")
    lines += ["", "All chronological points and seed curves remain available, including declines and dominated points.",
              "Means require every assigned training seed at that checkpoint. Incomplete panels cannot establish dominance.",
              "Bootstrap frontier frequency is descriptive stability, not a probability of being truly optimal.",
              "Checkpoint selection for deployment requires independent validation; this envelope uses assessment scores.",
              "Process SPS includes startup and writes; native SPS uses the native adjusted timer. Evaluation/build costs are separate.",
              "See analysis.json for all failure records and inference limitations."]
    if costs:
        lines += ["", "## Accounted process costs", "", "Includes failed subprocesses when their elapsed time was captured.", "",
                  "| Phase | Seconds | Timed processes | Missing end/start receipts |", "|---|---:|---:|---:|"]
        for phase, cost in costs.items():
            lines.append(f"| {phase} | {cost['seconds']:.3f} | {cost['timed_processes']} | {cost['missing_timing']} |")
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    template = Path(__file__).with_suffix(".html").read_text()
    (out/"curves.html").write_text(template.replace("__DATA__", json.dumps(result, allow_nan=False).replace("<", "\\u003c")))
    return result


def calibrate(repetitions, seeds, draws):
    """Synthetic statistics only; no game, policy or GPU is invoked."""
    rng = np.random.default_rng(8831)
    rows = []
    for scenario in ("null", "crossing", "ties", "declining"):
        steps = np.arange(1, 6)
        times = np.tile(steps*10., (4, 1))
        scores = np.tile(.15+.1*steps, (4, 1))
        if scenario in ("crossing", "declining"):
            times *= np.array([.8, 1, 2, 2.2])[:, None]
            scores += np.array([.03, 0, .12, .05])[:, None]
            scores[0, -2:] -= .12
        if scenario == "declining":
            scores[:, -1] -= .2
        truth = np.stack([times.ravel(), scores.ravel()], axis=-1)
        covered, false_dominance = 0, 0
        for repeat in range(repetitions):
            shared = rng.uniform(-1, 1, (seeds, 1))
            q = scores.ravel()+.06*shared+.04*rng.uniform(-1, 1, (seeds, 20))
            performance = rng.binomial(128, q)/128
            timing = times.ravel()*np.exp(.12*rng.normal(size=(seeds, 20))-.12**2/2)
            if scenario == "ties":
                # Exact duplicate architectures retain perfectly paired curves.
                performance[:, 5:10] = performance[:, :5]
                timing[:, 5:10] = timing[:, :5]
            band = paired_bands(np.stack([timing, performance], axis=-1), draws, repeat+371)
            low, high = band["low"], band["high"]
            covered += bool(np.all((truth >= low) & (truth <= high)))
            certified = (high[:, 0, None] < low[None, :, 0]) & (low[:, 1, None] > high[None, :, 1])
            actual = (truth[:, 0, None] < truth[None, :, 0]) & (truth[:, 1, None] > truth[None, :, 1])
            false_dominance += bool(np.any(certified & ~actual))
        rate = covered/repetitions
        rows.append(dict(scenario=scenario, repetitions=repetitions, training_seeds=seeds, bootstrap_draws=draws,
                         joint_coverage=rate, coverage_monte_carlo_se=float(np.sqrt(rate*(1-rate)/repetitions)),
                         any_false_dominance_rate=false_dominance/repetitions))
    return dict(status="development_simulation_not_an_acceptance_certificate", scenarios=rows,
                limitation="These four populations do not cover RL failure mixtures, arbitrary hardware drift or missing outcomes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.repetitions < 1 or args.seeds < 2 or args.draws < 1:
        parser.error("Use a fresh output and positive simulation settings (at least two seeds)")
    args.out.write_text(json.dumps(calibrate(args.repetitions, args.seeds, args.draws), indent=2)+"\n")
