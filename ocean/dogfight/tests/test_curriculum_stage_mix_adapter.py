import configparser
from pathlib import Path


def test_stage_mix_is_disabled_by_default_and_wired_before_target_assignment():
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "config" / "dogfight.ini"
    adapter_path = repo_root / "ocean" / "dogfight" / "dogfight_puffer.h"

    config = configparser.ConfigParser()
    config.read(config_path)
    assert config.getint("env", "rehearsal_stage") == -1
    assert config.getint("env", "rehearsal_stride") == 0

    adapter = adapter_path.read_text()
    config_index = adapter.index("int rehearsal_stage")
    select_index = adapter.index("dogfight_select_fixed_stage(")
    target_index = adapter.index("set_curriculum_target(env, curriculum_target)")
    mix_config = adapter[config_index:select_index]
    assert select_index < target_index
    assert '"rehearsal_stage", -1' in mix_config
    assert '"rehearsal_stride", 0' in mix_config
    assert "(uint32_t)env->rng" in adapter
