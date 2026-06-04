import math
import importlib
import sys
import tempfile
from pathlib import Path


DOGFIGHT_DIR = Path(__file__).resolve().parents[1]


def load_sweep_hypers():
    sys.path.insert(0, str(DOGFIGHT_DIR))
    try:
        return importlib.import_module("sweep_hypers")
    finally:
        sys.path.pop(0)


def parse_text(text: str):
    sweep_hypers = load_sweep_hypers()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.log"
        path.write_text(text)
        return sweep_hypers.parse_log(path)


def test_sweep_hypers_scores_unsaturated_training_log():
    row = parse_text(
        """
        [CURRICULUM] target 1.90 -> 2.90 kill_rate=0.95 episodes=2048
        SPS 1.2M curriculum_target 2.90 base_stage_kill_rate 0.80 entropy 1.4
        action_sat_elevator 0.30 action_sat_aileron 0.45 action_sat_rudder 0.60
        action_sat_trigger 1.00
        """
    )

    expected_surface = (0.30 + 0.45 + 0.60) / 3.0
    expected_score = 2.90 + 0.25 * 0.80 - 2.0 * expected_surface

    assert row["rejected"] is False
    assert math.isclose(row["surface_saturation"], expected_surface, abs_tol=1e-9)
    assert math.isclose(row["selection_score"], expected_score, abs_tol=1e-9)
    assert math.isclose(row["final_SPS"], 1_200_000.0, abs_tol=1e-9)


def test_sweep_hypers_rejects_saturated_or_incomplete_action_metrics():
    saturated = parse_text(
        """
        SPS 1.2M curriculum_target 1.90 base_stage_kill_rate 0.55
        action_sat_elevator 0.80 action_sat_aileron 0.90 action_sat_rudder 0.70
        action_sat_trigger 0.00
        """
    )

    assert saturated["rejected"] is True
    assert math.isclose(saturated["surface_saturation"], 0.80, abs_tol=1e-9)

    incomplete = parse_text(
        """
        SPS 1.2M curriculum_target 1.90 base_stage_kill_rate 0.55
        action_sat_elevator 0.20 action_sat_aileron 0.30 action_sat_trigger 1.00
        """
    )

    assert incomplete["rejected"] is True
    assert incomplete["surface_saturation"] == ""
    assert incomplete["selection_score"] == ""


def test_sweep_hypers_derives_base_stage_kill_rate_from_counts():
    row = parse_text(
        """
        [CURRICULUM] epoch=4 target 0.90 -> 1.90 kill_rate=0.910 episodes=2754
        base_stage_kills 0.10 base_stage_eps 0.50
        action_sat_elevator 0.30 action_sat_aileron 0.45 action_sat_rudder 0.60
        base_stage_kills 0.40 base_stage_eps 0.50
        action_sat_elevator 0.20 action_sat_aileron 0.30 action_sat_rudder 0.40
        """
    )

    expected_surface = (0.20 + 0.30 + 0.40) / 3.0
    expected_score = 1.90 + 0.25 * 0.80 - 2.0 * expected_surface

    assert math.isclose(row["final_base_stage_kill_rate"], 0.80, abs_tol=1e-9)
    assert math.isclose(row["best_base_stage_kill_rate"], 0.80, abs_tol=1e-9)
    assert math.isclose(row["selection_score"], expected_score, abs_tol=1e-9)


def test_sweep_hypers_parses_kl_metrics_without_name_collisions():
    old_only = parse_text(
        """
        old_kl 1150.7 clipfrac 0.898 entropy 0.1
        action_sat_elevator 0.30 action_sat_aileron 0.40 action_sat_rudder 0.50
        """
    )

    assert math.isclose(old_only["final_old_kl"], 1150.7, abs_tol=1e-9)
    assert old_only["final_kl"] == ""
    assert math.isclose(old_only["final_clipfrac"], 0.898, abs_tol=1e-9)

    both = parse_text(
        """
        old_kl 1150.7 kl 1149.9 clipfrac 0.898 entropy 0.1
        action_sat_elevator 0.30 action_sat_aileron 0.40 action_sat_rudder 0.50
        """
    )

    assert math.isclose(both["final_old_kl"], 1150.7, abs_tol=1e-9)
    assert math.isclose(both["final_kl"], 1149.9, abs_tol=1e-9)
    assert math.isclose(both["final_clipfrac"], 0.898, abs_tol=1e-9)


def test_sweep_hypers_random_trials_include_entropy_annealing_space():
    sweep_hypers = load_sweep_hypers()
    trials = dict(sweep_hypers.random_trials(max_runs=3, seed=37))

    apricot = dict(sweep_hypers.TRIALS)["apricot_control"]
    assert apricot["train.ent-coef"] == "0.1604920157441345"

    random_rows = [overrides for name, overrides in trials.items() if name.startswith("random_")]
    assert len(random_rows) == 2
    for overrides in random_rows:
        assert overrides["train.anneal-ent-coef"] == "1"
        assert "train.min-ent-coef-ratio" in overrides
        assert 0.0002 <= float(overrides["train.ent-coef"]) <= 0.04


def test_sweep_hypers_random_trial_overrides_parse_as_dogfight_cli(monkeypatch):
    import pufferlib.pufferl as pufferl

    sweep_hypers = load_sweep_hypers()
    overrides = dict(sweep_hypers.random_trials(max_runs=2, seed=37))["random_0001"]
    argv = ["pufferl.py"]
    for key, value in overrides.items():
        argv.extend([f"--{key}", value])

    monkeypatch.setattr(sys, "argv", argv)
    args = pufferl.load_config("dogfight")

    assert args["train"]["anneal_ent_coef"] == 1
    assert isinstance(args["train"]["vf_coef"], float)
    assert isinstance(args["train"]["prio_beta0"], float)


def test_sweep_hypers_trials_from_summary_selects_top_non_rejected(tmp_path):
    sweep_hypers = load_sweep_hypers()
    path = tmp_path / "summary.csv"
    path.write_text(
        "trial,status,selection_score,rejected,overrides\n"
        'bad,parsed,9.0,True,"{""train.ent-coef"": ""0.1""}"\n'
        'low,failed:2,8.0,False,"{""train.ent-coef"": ""0.2""}"\n'
        'second,ok,1.5,False,"{""train.ent-coef"": ""0.3""}"\n'
        'first,parsed,2.0,False,"{""train.ent-coef"": ""0.4""}"\n'
    )

    assert sweep_hypers.trials_from_summary(path, top_k=2) == [
        ("first", {"train.ent-coef": "0.4"}),
        ("second", {"train.ent-coef": "0.3"}),
    ]
