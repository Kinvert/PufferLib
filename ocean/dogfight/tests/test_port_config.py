from configparser import ConfigParser
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys


REQUIRED_ENV_KEYS = {
    "max_steps",
    "obs_scheme",
    "curriculum_enabled",
    "curriculum_randomize",
    "reward_aim_scale",
    "reward_closing_scale",
    "penalty_neg_g",
    "control_rate_penalty",
    "low_altitude_threshold",
    "low_altitude_penalty",
    "speed_min",
    "aim_decay_stage",
    "shaping_decay_start",
    "shaping_decay_end",
    "energy_gain_scale",
    "energy_loss_scale",
    "energy_advantage_scale",
}

REQUIRED_CURRICULUM_KEYS = {
    "warmup_steps",
    "eval_interval",
    "window_decay",
}

REQUIRED_TRAIN_KEYS = {
    "learning_rate",
    "momentum",
    "eps",
    "replay_ratio",
    "clip_coef",
    "vf_coef",
    "vf_clip_coef",
    "max_grad_norm",
}

DOGFIGHT3_TRAIN_BASELINE = {
    "learning_rate": 0.00045,
    "horizon": 64,
    "gamma": 0.99,
    "gae_lambda": 0.999,
    "ent_coef": 0.0024,
    "clip_coef": 0.11,
    "vf_coef": 2.9,
    "vf_clip_coef": 1.5,
    "max_grad_norm": 1.8,
    "momentum": 0.9896,
    "eps": 1.49e-08,
    "prio_alpha": 0.99,
    "prio_beta0": 0.99,
    "vtrace_rho_clip": 2.79,
    "vtrace_c_clip": 3.5,
    "replay_ratio": 1.0,
}

DOGFIGHT3_SHORT_SWEEP_PROFILE = {
    "policy.hidden_size": 128,
    "policy.num_layers": 1,
    "train.horizon": 64,
    "train.learning_rate": 0.00045,
    "train.ent_coef": 0.0024,
    "train.clip_coef": 0.11,
}


def load_sweep_hypers():
    dogfight_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "dogfight_sweep_hypers_for_config",
        dogfight_dir / "sweep_hypers.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dogfight_config_exposes_port_baseline_sections_and_env_keys():
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "dogfight.ini"

    parser = ConfigParser()
    parser.read(config_path)

    assert config_path.is_file()
    assert parser.get("base", "env_name") == "dogfight"
    assert parser.has_section("vec")
    assert parser.has_section("policy")
    assert parser.has_section("env")
    assert parser.has_section("train")
    assert parser.has_section("curriculum")
    assert REQUIRED_ENV_KEYS <= set(parser["env"])
    assert REQUIRED_CURRICULUM_KEYS <= set(parser["curriculum"])
    assert REQUIRED_TRAIN_KEYS <= set(parser["train"])
    assert parser.getint("env", "curriculum_enabled") == 1


def test_dogfight_config_matches_dogfight3_env_config_baseline():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert parser.getint("env", "max_steps") == 300
    assert parser.getint("env", "obs_scheme") == 1
    assert parser.getfloat("env", "reward_aim_scale") == 0.001695
    assert parser.getfloat("env", "reward_closing_scale") == 0.0001
    assert parser.getfloat("env", "penalty_neg_g") == 0.035
    assert parser.getfloat("env", "control_rate_penalty") == 0.002
    assert parser.getfloat("env", "low_altitude_threshold") == 1200.0
    assert parser.getfloat("env", "low_altitude_penalty") == 0.005
    assert parser.getfloat("env", "aim_decay_stage") == 30.0
    assert parser.getfloat("env", "domain_randomization") == 0.05
    assert parser.getfloat("env", "vertical_spawn_prob") == 0.02
    assert parser.getfloat("env", "recovery_trigger_prob") == 0.067

    assert parser.getint("curriculum", "warmup_steps") == 1_000_000
    assert parser.getint("curriculum", "eval_interval") == 60_000
    assert parser.getint("curriculum", "min_episodes") == 50



def test_dogfight_train_config_uses_real_training_horizon():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert parser.getint("train", "total_timesteps") == 400_000_000


def test_dogfight_sweep_time_budget_targets_fast_stage_discovery():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert parser.getint("sweep.train.total_timesteps", "min") >= 50_000_000
    assert parser.getint("sweep.train.total_timesteps", "mean") == 75_000_000
    assert parser.getint("sweep.train.total_timesteps", "max") <= 125_000_000


def test_dogfight_sweep_space_centers_reference_short_stage_discovery_profile():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    for dotted_key, expected in DOGFIGHT3_SHORT_SWEEP_PROFILE.items():
        section = f"sweep.{dotted_key}"
        assert parser.getfloat(section, "min") <= expected
        assert parser.getfloat(section, "max") >= expected
        assert math.isclose(parser.getfloat(section, "mean"), expected)


def test_dogfight_train_keeps_state_memory_disabled_by_default():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert not parser.has_option("train", "state_buffer_size")
    assert not parser.has_option("train", "cl_frac")


def test_dogfight_sweep_can_search_state_curriculum_knobs():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert parser.get("sweep.train.state_buffer_size", "distribution") == "int_uniform"
    assert parser.getint("sweep.train.state_buffer_size", "min") >= 512
    assert parser.getint("sweep.train.state_buffer_size", "mean") == 8192
    assert parser.getint("sweep.train.state_buffer_size", "max") <= 20_000

    assert parser.get("sweep.train.cl_frac", "distribution") == "uniform"
    assert 0.0 < parser.getfloat("sweep.train.cl_frac", "min") <= 0.1
    assert math.isclose(parser.getfloat("sweep.train.cl_frac", "mean"), 0.35)
    assert parser.getfloat("sweep.train.cl_frac", "max") <= 0.8


def test_dogfight_state_curriculum_sweep_space_roundtrips_through_protein(monkeypatch):
    from pufferlib import pufferl
    import pufferlib.sweep

    monkeypatch.setattr(sys, "argv", ["puffer"])
    args = pufferl.load_config("dogfight")
    sweep_config = deepcopy(args["sweep"])
    method = sweep_config.pop("method")
    sweep_config["use_gpu"] = False

    sweep = getattr(pufferlib.sweep, method)(sweep_config)
    sweep.suggest(args)
    pufferl.validate_config(args)

    assert args["train"]["state_buffer_size"] == 8192
    assert isinstance(args["train"]["state_buffer_size"], int)
    assert math.isclose(args["train"]["cl_frac"], 0.35)
    assert args["train"]["state_buffer_size"] > args["train"]["warmup_states"]


def test_dogfight_train_config_uses_puffer5_optimizer_keys():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")

    assert parser.has_option("train", "momentum")
    assert math.isclose(parser.getfloat("train", "momentum"), DOGFIGHT3_TRAIN_BASELINE["momentum"])
    assert not parser.has_option("train", "beta1")
    assert not parser.has_option("train", "beta2")
    assert parser.has_section("sweep.train.momentum")
    assert not parser.has_section("sweep.train.beta1")
    assert not parser.has_section("sweep.train.beta2")


def test_dogfight_train_config_ports_dogfight3_training_profile_to_puffer5_keys():
    repo_root = Path(__file__).resolve().parents[3]
    parser = ConfigParser()
    parser.read(repo_root / "config" / "dogfight.ini")
    sweep_hypers = load_sweep_hypers()

    for key, expected in DOGFIGHT3_TRAIN_BASELINE.items():
        assert math.isclose(parser.getfloat("train", key), expected)

    allowed_replay = {
        float(value)
        for value in sweep_hypers.DISCRETE_RANDOM_SPACE["train.replay-ratio"]
    }

    assert parser.getfloat("train", "replay_ratio") in allowed_replay
    # Short sweeps stay centered on the Dogfight3 trainer profile and only
    # broaden around it enough to find early-stage progress quickly.
    short_sweep_train_keys = {
        key.split(".", 1)[1]
        for key in DOGFIGHT3_SHORT_SWEEP_PROFILE
        if key.startswith("train.")
    }

    for key in (
        "gamma",
        "gae_lambda",
        "vtrace_rho_clip",
        "vtrace_c_clip",
        "vf_clip_coef",
        "vf_coef",
        "max_grad_norm",
        "momentum",
        "eps",
        "prio_alpha",
        "prio_beta0",
        "replay_ratio",
    ):
        assert key not in short_sweep_train_keys
        assert math.isclose(
            parser.getfloat(f"sweep.train.{key}", "mean"),
            parser.getfloat("train", key),
        )
    assert parser.getfloat("sweep.train.replay_ratio", "max") <= max(allowed_replay)
