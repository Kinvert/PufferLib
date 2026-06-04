import math
import sys
from copy import deepcopy


def test_dogfight_soft_quality_is_derived_from_curriculum_and_action_telemetry():
    import pufferlib.pufferl as pufferl

    logs = {
        "env/curriculum_target": 9.0,
        "env/action_sat_elevator": 0.0,
        "env/action_sat_aileron": 0.5,
        "env/action_sat_rudder": 1.0,
    }

    pufferl.add_derived_sweep_metrics(
        {
            "env_name": "dogfight",
            "curriculum": {"max_target": 18.0},
        },
        logs,
    )

    assert math.isclose(logs["env/curriculum_soft_quality"], 0.375)


def test_dogfight_mastery_quality_weights_curriculum_by_kill_rate():
    import pufferlib.pufferl as pufferl

    low_kill_logs = {
        "env/curriculum_target": 9.0,
        "env/base_stage_kill_rate": 0.25,
        "env/action_sat_elevator": 0.0,
        "env/action_sat_aileron": 0.5,
        "env/action_sat_rudder": 1.0,
    }
    high_kill_logs = dict(low_kill_logs)
    high_kill_logs["env/base_stage_kill_rate"] = 0.75

    args = {
        "env_name": "dogfight",
        "curriculum": {"max_target": 18.0},
    }

    pufferl.add_derived_sweep_metrics(args, low_kill_logs)
    pufferl.add_derived_sweep_metrics(args, high_kill_logs)

    assert math.isclose(low_kill_logs["env/curriculum_mastery_quality"], 0.09375)
    assert math.isclose(high_kill_logs["env/curriculum_mastery_quality"], 0.28125)


def test_dogfight_configured_sweep_metric_can_be_derived_from_logs(monkeypatch):
    import pufferlib.pufferl as pufferl

    monkeypatch.setattr(
        sys,
        "argv",
        ["puffer", "--train.total-timesteps", "4096"],
    )
    args = pufferl.load_config("dogfight")
    logs = {
        "env/curriculum_target": args["curriculum"]["initial_target"],
        "env/action_sat_elevator": 0.0,
        "env/action_sat_aileron": 0.0,
        "env/action_sat_rudder": 0.0,
    }

    pufferl.add_derived_sweep_metrics(args, logs)

    assert args["sweep"]["metric"] == "curriculum_soft_quality"
    assert f'env/{args["sweep"]["metric"]}' in logs


def test_dogfight_sweep_config_builds_puffer5_hyperparameter_space(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep

    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("dogfight")

    hypers = sweep.Hyperparameters(args["sweep"], verbose=False)

    assert "train/total_timesteps" in hypers.flat_spaces
    assert "early_stop_min_steps" not in hypers.flat_spaces


def test_dogfight_sweep_config_constructs_protein_verbose_sampler(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep
    import torch

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = pufferl.load_config("dogfight")

    protein = sweep.Protein(args["sweep"], use_gpu=False)

    assert protein.hyperparameters.get_flat_idx("train/total_timesteps") is not None


def test_dogfight_sweep_hyperparameters_honor_configured_means(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep

    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("dogfight")
    hypers = sweep.Hyperparameters(args["sweep"], verbose=False)

    for key in (
        "train/learning_rate",
        "train/horizon",
        "train/clip_coef",
        "policy/num_layers",
        "train/total_timesteps",
    ):
        space = hypers.flat_spaces[key]
        section = args["sweep"]
        for part in key.split("/"):
            section = section[part]
        assert math.isclose(space.unnormalize(space.norm_mean), section["mean"])


def test_protein_first_suggestion_uses_configured_search_center(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep
    import torch

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = pufferl.load_config("dogfight")
    protein = sweep.Protein(args["sweep"], use_gpu=False)

    suggested, _ = protein.suggest(deepcopy(args))

    assert suggested["train"]["total_timesteps"] == args["sweep"]["train"]["total_timesteps"]["mean"]
    assert math.isclose(
        suggested["train"]["learning_rate"],
        args["sweep"]["train"]["learning_rate"]["mean"],
    )
    assert suggested["train"]["horizon"] == args["sweep"]["train"]["horizon"]["mean"]
    assert math.isclose(
        suggested["train"]["clip_coef"],
        args["sweep"]["train"]["clip_coef"]["mean"],
    )
    assert math.isclose(
        suggested["policy"]["num_layers"],
        args["sweep"]["policy"]["num_layers"]["mean"],
    )


def test_dogfight_sweep_samples_discrete_topology_as_integers(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep
    import torch

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = pufferl.load_config("dogfight")

    hypers = sweep.Hyperparameters(args["sweep"], verbose=False)
    for key in (
        "policy/num_layers",
        "vec/num_buffers",
    ):
        space = hypers.flat_spaces[key]
        assert space.is_integer, key

    protein = sweep.Protein(args["sweep"], use_gpu=False)
    suggested, _ = protein.suggest(deepcopy(args))

    assert isinstance(suggested["policy"]["num_layers"], int)
    assert suggested["policy"]["num_layers"] == args["sweep"]["policy"]["num_layers"]["mean"]
    assert isinstance(suggested["vec"]["num_buffers"], int)
    assert suggested["vec"]["num_buffers"] == args["sweep"]["vec"]["num_buffers"]["mean"]


def test_dogfight_single_gpu_sweep_runs_trial_inline_to_avoid_cuda_ipc(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep

    class FakeProtein:
        def __init__(self, config):
            self.observations = []

        def suggest(self, args, fixed_total_timesteps=None):
            return args, {}

        def observe(self, hypers, score, cost, is_failure=False):
            self.observations.append((score, cost, is_failure))

    train_gpus = []

    def fake_train(env_name, args, gpus=None, **kwargs):
        train_gpus.append(gpus)
        result_queue = kwargs["result_queue"]
        assert not result_queue.__class__.__module__.startswith("multiprocessing")
        result_queue.put((
            args["gpu_id"],
            [1.0],
            [1.0],
            [args["train"]["total_timesteps"]],
        ))

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(sweep, "Protein", FakeProtein)
    monkeypatch.setattr(pufferl, "train", fake_train)
    args = pufferl.load_config("dogfight")
    args["sweep"]["gpus"] = 1
    args["train"]["gpus"] = 1
    args["sweep"]["max_runs"] = 1

    pufferl.sweep("dogfight", args=args)

    assert train_gpus == [None]


def test_sweep_asks_protein_for_second_trial_after_one_baseline(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep

    class FakeProtein:
        def __init__(self, config):
            self.suggest_calls = 0
            self.seed_with_search_center = True

        def suggest(self, args, fixed_total_timesteps=None):
            self.suggest_calls += 1
            assert self.seed_with_search_center is False
            args["train"]["learning_rate"] = 0.123

        def observe(self, hypers, score, cost, is_failure=False):
            pass

    learning_rates = []

    def fake_train(env_name, args, gpus=None, **kwargs):
        learning_rates.append(args["train"]["learning_rate"])
        kwargs["result_queue"].put((
            args["gpu_id"],
            [1.0],
            [1.0],
            [args["train"]["total_timesteps"]],
        ))

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(sweep, "Protein", FakeProtein)
    monkeypatch.setattr(pufferl, "train", fake_train)
    args = pufferl.load_config("dogfight")
    args["sweep"]["gpus"] = 1
    args["train"]["gpus"] = 1
    args["sweep"]["max_runs"] = 2

    pufferl.sweep("dogfight", args=args)

    assert learning_rates == [0.00045, 0.123]


def test_sweep_baseline_uses_configured_total_timesteps_mean(monkeypatch):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep

    class FakeProtein:
        def __init__(self, config):
            pass

        def suggest(self, args, fixed_total_timesteps=None):
            raise AssertionError("first baseline trial should not ask for a suggestion")

        def observe(self, hypers, score, cost, is_failure=False):
            pass

    trial_timesteps = []

    def fake_train(env_name, args, gpus=None, **kwargs):
        trial_timesteps.append(args["train"]["total_timesteps"])
        kwargs["result_queue"].put((
            args["gpu_id"],
            [1.0],
            [1.0],
            [args["train"]["total_timesteps"]],
        ))

    monkeypatch.setattr(sys, "argv", ["puffer"])
    monkeypatch.setattr(sweep, "Protein", FakeProtein)
    monkeypatch.setattr(pufferl, "train", fake_train)
    args = pufferl.load_config("dogfight")
    args["sweep"]["gpus"] = 1
    args["train"]["gpus"] = 1
    args["sweep"]["max_runs"] = 1

    pufferl.sweep("dogfight", args=args)

    assert trial_timesteps == [args["sweep"]["train"]["total_timesteps"]["mean"]]
    assert trial_timesteps[0] < args["train"]["total_timesteps"]


def test_inline_train_forwards_sweep_kwargs(monkeypatch):
    import pufferlib.pufferl as pufferl

    received = {}

    def fake_train(env_name, args, verbose=False, **kwargs):
        received["verbose"] = verbose
        received["kwargs"] = kwargs

    args = {
        "train": {"gpus": 1, "total_timesteps": 1024},
    }
    sweep_obj = object()
    result_queue = object()

    monkeypatch.setattr(pufferl, "validate_config", lambda args: None)
    monkeypatch.setattr(pufferl, "_train", fake_train)

    pufferl.train(
        "dogfight",
        args=args,
        gpus=None,
        sweep_obj=sweep_obj,
        result_queue=result_queue,
    )

    assert received["verbose"] is True
    assert received["kwargs"]["sweep_obj"] is sweep_obj
    assert received["kwargs"]["result_queue"] is result_queue


def test_dogfight_sweep_narrows_total_agents_to_measured_gpu_shape(monkeypatch):
    import pufferlib.pufferl as pufferl

    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("dogfight")
    space = args["sweep"]["vec"]["total_agents"]

    assert space["distribution"] == "uniform_pow2"
    assert space["min"] == 2048
    assert space["max"] == 4096
    assert args["vec"]["total_agents"] == 4096


def test_dogfight_wandb_payload_is_capped_to_high_signal_metrics():
    import pufferlib.pufferl as pufferl

    expected = [
        "SPS",
        "uptime",
        "util/gpu_percent",
        "util/vram_used_gb",
        "env/score",
        "env/match_score",
        "env/curriculum_soft_quality",
        "env/curriculum_mastery_quality",
        "env/curriculum_target",
        "env/mastery_stage",
        "env/stage9_bank_deg",
        "env/base_stage_kill_rate",
        "env/base_stage_timeout_rate",
        "env/base_stage_ground_rate",
        "env/base_stage_episode_length",
        "env/base_stage_window_kill_rate",
        "env/base_stage_action_saturation",
        "env/base_stage_action_sat_elevator",
        "env/base_stage_action_sat_aileron",
        "env/base_stage_action_sat_rudder",
        "env/base_stage_signed_bias_elevator",
        "env/base_stage_signed_bias_aileron",
        "env/base_stage_signed_bias_rudder",
        "env/base_stage_side_standard_kill_rate",
        "env/base_stage_side_energy_kill_rate",
        "loss/total",
        "loss/policy",
        "loss/value",
        "loss/entropy",
        "loss/kl",
        "loss/clipfrac",
    ]
    logs = {key: idx for idx, key in enumerate(expected)}
    logs.update(
        {
            "agent_steps": 123,
            "env/base_stage_action_sat_elevator_sum": 999,
            "env/base_stage_side_standard_action_saturation_sum": 999,
            "loss/old_kl": 999,
            "perf/train": 999,
        }
    )

    payload = pufferl.wandb_log_payload({"env_name": "dogfight"}, logs)

    assert list(payload) == expected
    assert len(payload) == 31
    assert "agent_steps" not in payload
    assert "env/base_stage_action_sat_elevator_sum" not in payload
    assert "env/base_stage_side_standard_action_saturation_sum" not in payload
    assert "agent_steps" in logs
