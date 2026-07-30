import configparser
import math
import struct
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOGFIGHT_CONFIG = REPO_ROOT / "config" / "dogfight.ini"
BENCHMARK_PROFILE = REPO_ROOT / "ocean" / "dogfight" / "native_sweep_profile.ini"
DIRECT_LAUNCHER = REPO_ROOT / "ocean" / "dogfight" / "sweep_robocode_style.sh"


def read_ini(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    return config


def test_direct_launcher_is_a_plain_native_puffer_sweep():
    source = DIRECT_LAUNCHER.read_text(encoding="ascii")

    assert "native_sweep.py" not in source
    assert "exec ./puffer sweep dogfight" in source
    assert "--wandb" not in source
    assert "--wandb-project" not in source
    assert '"$@"' in source


def test_active_config_has_only_explicit_ranges_without_sweep_only():
    config = read_ini(DOGFIGHT_CONFIG)
    ranges = {
        section
        for section in config.sections()
        if section.startswith("sweep.")
    }

    assert not config.has_option("sweep", "sweep_only")
    assert ranges == {
        "sweep.train.total_timesteps",
        "sweep.policy.hidden_size",
        "sweep.policy.num_layers",
        "sweep.vec.total_agents",
        "sweep.train.horizon",
        "sweep.train.minibatch_size",
        "sweep.train.learning_rate",
        "sweep.train.ent_coef",
        "sweep.train.gamma",
        "sweep.train.gae_lambda",
        "sweep.train.replay_ratio",
        "sweep.train.clip_coef",
        "sweep.train.vf_clip_coef",
        "sweep.train.vf_coef",
        "sweep.train.max_grad_norm",
        "sweep.train.momentum",
        "sweep.train.prio_alpha",
        "sweep.train.prio_beta0",
        "sweep.env.native_spawn_total_steps",
        "sweep.env.native_frontier_fraction",
        "sweep.env.native_lateral_width_scale",
        "sweep.env.native_roll_recovery_fraction",
        "sweep.env.native_roll_discipline_scale",
        "sweep.env.native_bank_guidance_scale",
        "sweep.env.native_roll_guidance_scale",
        "sweep.vec.frozen_bank_pct",
    }
    for section in ranges:
        minimum = config.getfloat(section, "min")
        maximum = config.getfloat(section, "max")
        assert minimum < maximum


def test_every_effective_default_is_inside_its_sweep_range():
    config = configparser.ConfigParser(interpolation=None)
    config.read([REPO_ROOT / "config" / "default.ini", DOGFIGHT_CONFIG])
    outside = []

    for sweep_section in config.sections():
        if not sweep_section.startswith("sweep."):
            continue
        target = sweep_section.removeprefix("sweep.")
        section, key = target.rsplit(".", 1)
        value = config.getfloat(section, key)
        minimum = config.getfloat(sweep_section, "min")
        maximum = config.getfloat(sweep_section, "max")
        if not minimum <= value <= maximum:
            outside.append((target, value, minimum, maximum))

    assert outside == []


def test_sweep_ranges_remain_distinct_at_native_float_precision():
    config = read_ini(DOGFIGHT_CONFIG)

    for section in config.sections():
        if not section.startswith("sweep."):
            continue
        minimum = config.getfloat(section, "min")
        maximum = config.getfloat(section, "max")
        assert struct.pack("f", minimum) != struct.pack("f", maximum), section


def test_checkpoint_structural_dimensions_are_fixed_and_compatible():
    config = configparser.ConfigParser(interpolation=None)
    config.read([REPO_ROOT / "config" / "default.ini", DOGFIGHT_CONFIG])
    structural = (
        ("policy", "hidden_size"),
        ("policy", "num_layers"),
        ("vec", "total_agents"),
        ("train", "horizon"),
    )

    for section, key in structural:
        value = config.getint(section, key)
        sweep = config[f"sweep.{section}.{key}"]
        endpoints = (float(sweep["min"]), float(sweep["max"]))
        if sweep["distribution"] == "uniform_pow2":
            sampled = {2 ** round(math.log2(point)) for point in endpoints}
        elif sweep["distribution"] == "int_uniform":
            sampled = {round(point) for point in endpoints}
        else:
            raise AssertionError(
                f"{section}.{key} is not an integer sweep distribution"
            )
        assert sampled == {value}

    assert config.getint("vec", "frozen_bank_hidden_size") == config.getint(
        "policy", "hidden_size"
    )
    assert config.getint("vec", "frozen_bank_num_layers") == config.getint(
        "policy", "num_layers"
    )


def test_minibatch_sweep_explores_valid_optimizer_density():
    config = configparser.ConfigParser(interpolation=None)
    config.read([REPO_ROOT / "config" / "default.ini", DOGFIGHT_CONFIG])
    sweep = config["sweep.train.minibatch_size"]
    minimum = math.ceil(math.log2(float(sweep["min"])))
    maximum = math.floor(math.log2(float(sweep["max"])))
    candidates = {2**power for power in range(minimum, maximum + 1)}
    batch_size = (
        config.getint("vec", "total_agents")
        * config.getint("train", "horizon")
    )
    default = config.getint("train", "minibatch_size")

    assert sweep["distribution"] == "uniform_pow2"
    assert len(candidates) >= 3
    assert default in candidates
    assert min(candidates) < default
    assert all(batch_size % candidate == 0 for candidate in candidates)


def test_optimizer_sweep_excludes_known_catastrophic_extremes():
    config = configparser.ConfigParser(interpolation=None)
    config.read([REPO_ROOT / "config" / "default.ini", DOGFIGHT_CONFIG])

    learning_rate = config["sweep.train.learning_rate"]
    replay_ratio = config["sweep.train.replay_ratio"]
    entropy = config["sweep.train.ent_coef"]
    gamma = config["sweep.train.gamma"]

    assert learning_rate.getfloat("min") >= 0.001
    assert learning_rate.getfloat("max") <= 0.02
    assert replay_ratio.getfloat("min") >= 0.5
    assert replay_ratio.getfloat("max") <= 2.0
    assert entropy.getfloat("min") >= 0.00001
    assert entropy.getfloat("max") <= 0.001
    assert gamma.getfloat("min") >= 0.95


def test_benchmark_profile_preserves_its_private_allowlist():
    profile = read_ini(BENCHMARK_PROFILE)

    assert profile["sweep"]["sweep_only"] == (
        "env.native_spawn_total_steps, env.native_frontier_fraction, "
        "vec.frozen_bank_pct"
    )
    assert profile.has_section("fixed_eval")
