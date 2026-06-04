from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_dogfight_native_encoder_restores_dogfight3_gelu_observation_path():
    dogfight3_models = Path("/home/claude/dogfight3/pufferlib/models.py").read_text()
    ocean_cu = (repo_root() / "src" / "ocean.cu").read_text()

    assert "nn.Linear(num_obs, hidden_size)" in dogfight3_models
    assert "nn.GELU()" in dogfight3_models

    assert 'env_name == "dogfight"' in ocean_cu
    assert "dogfight_gelu_encoder_forward" in ocean_cu
    assert "dogfight_gelu_encoder_backward" in ocean_cu
    assert "dogfight_gelu_encoder_init_weights" in ocean_cu
    assert "dogfight_gelu_bias_grad_kernel" in ocean_cu


def test_dogfight_native_binding_uses_dogfight3_default_scheme1_width():
    binding_c = (repo_root() / "ocean" / "dogfight" / "binding.c").read_text()

    assert "#define OBS_SIZE 26" in binding_c
    assert "if (obs_scheme != OBS_PILOT) obs_scheme = OBS_PILOT;" not in binding_c
