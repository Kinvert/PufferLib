import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "ocean" / "dogfight" / "analyze_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_sweep", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_native_log(
    experiment,
    run_id,
    *,
    perf,
    score,
    pool_score,
    frontier_fraction,
    steps=671_088_640,
):
    log_dir = experiment / "logs" / "dogfight"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{run_id}.ini").write_text(
        f"""
[base]
run_id = {run_id}
seed = 42

[env]
native_frontier_fraction = {frontier_fraction}

[train]
total_timesteps = {steps}

[metrics]
agent_steps = 1,{steps}
SPS = 1000000,2500000
uptime = 1.0,240.0
env/perf = 0.0,{perf}
env/score = 0.5,{score}
selfplay/pool_score = {pool_score},{pool_score}
env/avg_stage = 0.0,5.0
env/avg_abs_bias = 2.0,3.0
env/avg_signed_bias = 1.0,-1.5
env/timeouts = 0.4,0.1
""".strip()
        + "\n"
    )


def write_fixed_result(
    experiment,
    run_id,
    *,
    highest_stage,
    min_perf,
    mean_perf,
    bias,
):
    fixed_dir = experiment / "fixed_eval"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "format": "dogfight-fixed-stage-screen-v1",
        "run_id": run_id,
        "checkpoint": f"/checkpoints/{run_id}/final.bin",
        "checkpoint_sha256": f"sha-{run_id}",
        "highest_contiguous_stage": highest_stage,
        "min_perf": min_perf,
        "mean_perf": mean_perf,
        "max_abs_signed_bias": bias,
        "roll_gate_passed": True,
        "max_abs_roll_common_mode": 0.01,
        "evaluated_cells": 6,
        "stopped_after_stage": max(highest_stage + 1, 0),
        "all_mastered": highest_stage >= 10,
    }
    (fixed_dir / f"{run_id}-sha.json").write_text(json.dumps(result))


def make_experiment(tmp_path, fixed_means):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "manifest.json").write_text(
        json.dumps(
            {
                "format": "dogfight-native-protein-sweep-v1",
                "native_sweep_dimensions": ["env.native_frontier_fraction"],
            }
        )
    )
    for index, fixed_mean in enumerate(fixed_means):
        run_id = f"sweep_{index:04d}"
        native_perf = 0.1 * (index + 1)
        write_native_log(
            experiment,
            run_id,
            perf=native_perf,
            score=0.4 + native_perf,
            pool_score=0.4 + native_perf,
            frontier_fraction=0.45 + 0.05 * index,
        )
        write_fixed_result(
            experiment,
            run_id,
            highest_stage=index - 1,
            min_perf=max(0.0, fixed_mean - 0.05),
            mean_perf=fixed_mean,
            bias=5.0 - index,
        )
    return experiment


def test_analysis_ranks_mastery_and_measures_native_objective(tmp_path):
    module = load_module()
    experiment = make_experiment(tmp_path, [0.2, 0.5, 0.9])

    analysis = module.analyze_experiment(
        experiment, minimum_candidates=3, minimum_spearman=0.25
    )

    assert [row["run_id"] for row in analysis["leaderboard"]] == [
        "sweep_0002",
        "sweep_0001",
        "sweep_0000",
    ]
    assert analysis["correlations"]["pool_score_vs_fixed_mean"]["spearman"] == 1.0
    assert analysis["correlations"]["pool_score_vs_fixed_mean"]["pearson"] > 0.99
    assert analysis["protein_gate"]["status"] == "pass"
    assert analysis["leaderboard"][0]["parameters"] == {
        "env.native_frontier_fraction": 0.55
    }


def test_analysis_rejects_anti_correlated_native_objective(tmp_path):
    module = load_module()
    experiment = make_experiment(tmp_path, [0.9, 0.5, 0.2])

    analysis = module.analyze_experiment(
        experiment, minimum_candidates=3, minimum_spearman=0.25
    )

    assert analysis["correlations"]["pool_score_vs_fixed_mean"]["spearman"] == -1.0
    assert analysis["protein_gate"]["status"] == "fail"


def test_analysis_writes_json_csv_and_markdown_evidence(tmp_path):
    module = load_module()
    experiment = make_experiment(tmp_path, [0.2, 0.5, 0.9])

    analysis = module.analyze_experiment(
        experiment, minimum_candidates=16, minimum_spearman=0.25
    )
    output_dir = module.write_analysis(experiment, analysis)

    assert analysis["protein_gate"]["status"] == "insufficient"
    assert (output_dir / "calibration_summary.json").is_file()
    assert (output_dir / "calibration_leaderboard.csv").is_file()
    report = (output_dir / "CALIBRATION_REPORT.md").read_text()
    assert "Protein gate: `insufficient`" in report
    assert "sweep_0002" in report
