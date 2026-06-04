from configparser import ConfigParser
import math
from pathlib import Path


DOGFIGHT3_RECURRENT_EFFECTIVE_ACTION_INIT_SCALE = 1.0
DOGFIGHT3_VALUE_INIT_SCALE = 1.0
PUFFER5_STAGE_CLIMB_ACTION_INIT_SCALE = 1.0


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_dogfight_policy_config_ports_dogfight3_effective_recurrent_decoder_init_scales():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "dogfight.ini")

    assert math.isclose(
        parser.getfloat("policy", "action_init_scale"),
        DOGFIGHT3_RECURRENT_EFFECTIVE_ACTION_INIT_SCALE,
    )
    assert math.isclose(
        parser.getfloat("policy", "value_init_scale"),
        DOGFIGHT3_VALUE_INIT_SCALE,
    )


def test_dogfight_policy_config_starts_from_dogfight3_recurrent_depth():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "dogfight.ini")

    assert parser.getint("policy", "hidden_size") == 128
    assert parser.getint("policy", "num_layers") == 1
    assert parser.getint("sweep.policy.num_layers", "min") == 1
    assert parser.getint("sweep.policy.num_layers", "mean") == 2
    assert parser.getint("sweep.policy.num_layers", "max") >= 3


def test_dogfight_sweep_centers_on_measured_native_action_head_init_scale():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "dogfight.ini")

    assert parser.get("sweep.policy.action_init_scale", "distribution") == "log_normal"
    assert parser.getfloat("sweep.policy.action_init_scale", "min") <= DOGFIGHT3_RECURRENT_EFFECTIVE_ACTION_INIT_SCALE
    assert math.isclose(
        parser.getfloat("sweep.policy.action_init_scale", "mean"),
        PUFFER5_STAGE_CLIMB_ACTION_INIT_SCALE,
    )
    assert parser.getfloat("sweep.policy.action_init_scale", "max") >= PUFFER5_STAGE_CLIMB_ACTION_INIT_SCALE


def test_dogfight_sweep_keeps_dogfight3_action_init_reachable():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "dogfight.ini")

    # Dogfight3 config set the continuous action head to 0.01, then the
    # recurrent wrapper reinitialized wrapped policy weights with scale 1.0.
    assert parser.getfloat("sweep.policy.action_init_scale", "min") <= DOGFIGHT3_RECURRENT_EFFECTIVE_ACTION_INIT_SCALE
    assert parser.getfloat("sweep.policy.action_init_scale", "max") >= DOGFIGHT3_RECURRENT_EFFECTIVE_ACTION_INIT_SCALE


def test_dogfight_sweep_keeps_state_curriculum_disabled_by_default_but_searchable():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "dogfight.ini")

    assert not parser.has_option("train", "state_buffer_size")
    assert not parser.has_option("train", "cl_frac")
    assert parser.has_section("sweep.train.state_buffer_size")
    assert parser.has_section("sweep.train.cl_frac")


def test_default_policy_config_keeps_existing_native_init_scale_defaults():
    parser = ConfigParser()
    parser.read(repo_root() / "config" / "default.ini")

    assert math.isclose(parser.getfloat("policy", "action_init_scale"), 1.0)
    assert math.isclose(parser.getfloat("policy", "value_init_scale"), 1.0)


def test_native_policy_uses_separate_action_and_value_decoder_init_scales():
    root = repo_root()
    models_cu = (root / "src" / "models.cu").read_text()
    pufferlib_cu = (root / "src" / "pufferlib.cu").read_text()
    bindings_cu = (root / "src" / "bindings.cu").read_text()

    assert "action_init_scale" in models_cu
    assert "value_init_scale" in models_cu
    assert "dw->action_init_scale" in models_cu
    assert "dw->value_init_scale" in models_cu
    assert "action_wt" in models_cu
    assert "value_wt" in models_cu

    assert "action_init_scale" in pufferlib_cu
    assert "value_init_scale" in pufferlib_cu
    assert 'get_config(policy_kwargs, "action_init_scale")' in bindings_cu
    assert 'get_config(policy_kwargs, "value_init_scale")' in bindings_cu
