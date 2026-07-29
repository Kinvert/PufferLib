import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "ocean" / "dogfight" / "eval_stage_matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_stage_matrix", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_output(perf, *, score=0.5, abs_bias=4.0, signed_bias=1.0):
    return (
        "bot_eval=128/128 perf=0.000000 score=0.000000\n"
        "dogfight_eval_controls "
        f"avg_abs_bias={abs_bias:.6f} avg_signed_bias={signed_bias:.6f} "
        "az_neg_mean_aileron=0.100000 az_pos_mean_aileron=-0.100000 "
        "az_neg_steps=64.0 az_pos_steps=64.0\n"
        "dogfight_eval_outcomes "
        f"perf={perf:.6f} score={score:.6f} timeouts=2.000000 "
        "player_ground_hits=1.000000 opponent_ground_hits=3.000000\n"
    )


def test_parse_eval_output_uses_dogfight_contract():
    module = load_module()

    sample = module.parse_eval_output(
        eval_output(0.9375),
        stage=3,
        seed=42,
        mirror=1,
        requested_episodes=128,
    )

    assert sample.stage == 3
    assert sample.seed == 42
    assert sample.mirror == 1
    assert sample.episodes == 128
    assert sample.perf == 0.9375
    assert sample.score == 0.5
    assert sample.timeouts == 2.0
    assert sample.player_ground_hits == 1.0
    assert sample.opponent_ground_hits == 3.0
    assert sample.avg_abs_bias == 4.0
    assert sample.avg_signed_bias == 1.0


def test_parse_eval_output_accepts_progress_line_concatenation_and_overshoot():
    module = load_module()
    output = eval_output(0.25).replace(
        "bot_eval=128/128 perf=0.000000 score=0.000000\n"
        "dogfight_eval_controls ",
        "bot_eval=129/128 perf=0.250000 score=-0.187"
        "dogfight_eval_controls ",
    )

    sample = module.parse_eval_output(
        output,
        stage=0,
        seed=42,
        mirror=0,
        requested_episodes=128,
    )

    assert sample.episodes == 129
    assert sample.perf == 0.25


def test_summary_requires_every_seed_and_mirror_cell_to_master_stage():
    module = load_module()
    samples = []
    for seed in (41, 42):
        for mirror in (0, 1):
            samples.append(
                module.parse_eval_output(
                    eval_output(0.91),
                    stage=0,
                    seed=seed,
                    mirror=mirror,
                    requested_episodes=128,
                )
            )
            samples.append(
                module.parse_eval_output(
                    eval_output(0.89 if (seed, mirror) == (42, 1) else 0.95),
                    stage=1,
                    seed=seed,
                    mirror=mirror,
                    requested_episodes=128,
                )
            )

    summary = module.summarize(samples, stages=(0, 1), threshold=0.90)

    assert summary["stages"]["0"]["mastered"] is True
    assert summary["stages"]["1"]["mastered"] is False
    assert summary["stages"]["1"]["min_perf"] == 0.89
    assert summary["highest_contiguous_stage"] == 0
    assert summary["evaluated_cells"] == 8
    assert summary["max_abs_signed_bias"] == 1.0


def test_evaluation_stops_after_complete_stage_zero_matrix_fails():
    module = load_module()
    calls = []

    def runner(stage, seed, mirror, episodes):
        calls.append((stage, seed, mirror, episodes))
        return eval_output(0.75 if stage == 0 else 1.0)

    samples, summary = module.evaluate_checkpoint(
        checkpoint=Path("/tmp/model.bin"),
        stages=(0, 1, 2),
        seeds=(41, 42),
        mirrors=(0, 1),
        episodes=128,
        threshold=0.90,
        runner=runner,
    )

    assert len(samples) == 4
    assert calls == [
        (0, 41, 0, 128),
        (0, 41, 1, 128),
        (0, 42, 0, 128),
        (0, 42, 1, 128),
    ]
    assert summary["highest_contiguous_stage"] == -1
    assert summary["stopped_after_stage"] == 0


def test_subprocess_runner_surfaces_evaluator_output(monkeypatch):
    module = load_module()

    def failed_run(*args, **kwargs):
        del args, kwargs
        return subprocess.CompletedProcess(
            args=["eval"], returncode=134, stdout="cudaHostAlloc failed\n"
        )

    monkeypatch.setattr(module.subprocess, "run", failed_run)
    runner = module._subprocess_runner(
        Path("/tmp/eval_checkpoint.sh"), Path("/tmp/model.bin")
    )

    with pytest.raises(RuntimeError, match="cudaHostAlloc failed"):
        runner(0, 42, 0, 128)


def test_evaluation_surfaces_output_when_contract_is_missing():
    module = load_module()

    def runner(stage, seed, mirror, episodes):
        del stage, seed, mirror, episodes
        return "Loaded weights\ncontract was not printed\n"

    with pytest.raises(RuntimeError, match="contract was not printed"):
        module.evaluate_checkpoint(
            checkpoint=Path("/tmp/model.bin"),
            stages=(0,),
            seeds=(42,),
            mirrors=(0, 1),
            episodes=128,
            threshold=0.90,
            runner=runner,
        )
