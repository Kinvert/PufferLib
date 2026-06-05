import importlib.util
from pathlib import Path


def load_obs_compat():
    dogfight_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "dogfight_obs_compat_for_tests",
        dogfight_dir / "obs_compat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_df36_known_good_run_uses_opponent_aware_obs_family():
    compat = load_obs_compat()

    df36_scheme = compat.DF36_OBS_SCHEMES[2]
    current_scheme = compat.DOGFIGHT5_OBS_SCHEMES[1]

    assert df36_scheme.name == "OBS_OPPONENT_AWARE"
    assert df36_scheme.size == 26
    assert current_scheme.name == "OBS_OPPONENT_AWARE"
    assert current_scheme.size == 26
    assert compat.dogfight5_scheme_for_df36_scheme(2) == 1
    assert compat.df36_scheme_for_dogfight5_scheme(1) == 2


def test_dogfight5_config_uses_df36_known_good_observation_family():
    compat = load_obs_compat()
    repo_root = Path(__file__).resolve().parents[3]
    config_text = (repo_root / "config" / "dogfight.ini").read_text()

    assert "obs_scheme = 1" in config_text
    assert compat.dogfight5_scheme_for_df36_scheme(
        compat.DF36_KNOWN_GOOD_OBS_SCHEME
    ) == compat.DOGFIGHT5_DEFAULT_OBS_SCHEME
