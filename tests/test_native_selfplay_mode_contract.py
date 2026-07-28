import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dogfight_defaults_to_official_native_selfplay():
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "config" / "dogfight.ini")

    assert config["selfplay"]["mode"] == "native"
    assert config.getint("selfplay", "enabled") == 1
    assert config.getint("vec", "num_frozen_banks") == 1
    assert config.getfloat("vec", "frozen_bank_pct") > 0
    assert config.getint("vec", "frozen_bank_hidden_size") == config.getint(
        "policy", "hidden_size"
    )
    assert config.getint("vec", "frozen_bank_num_layers") == config.getint(
        "policy", "num_layers"
    )


def test_native_runtime_has_explicit_mode_contract():
    source = (ROOT / "src" / "pufferl.cu").read_text()

    assert "resolve_selfplay_mode" in source
    assert 'strcmp(mode, "off")' in source
    assert 'strcmp(mode, "native")' in source
    assert 'strcmp(mode, "coordinator")' in source
    assert "Unknown selfplay.mode" in source


def test_coordinator_mode_pins_external_opponent_and_owns_evaluation():
    source = (ROOT / "src" / "pufferl.cu").read_text()

    assert "coordinator mode requires base.load_enemy_model_path" in source
    assert 'puf_ini_put(ini, "selfplay.opp_timeout_steps", "0")' in source
    assert 'strcmp(mode, "coordinator") == 0 && eval_games != 0' in source


def test_standard_eval_disables_training_only_frozen_banks():
    source = (ROOT / "src" / "pufferl.cu").read_text()

    assert "disable_selfplay_for_standard_eval" in source
