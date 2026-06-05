from configparser import ConfigParser
import importlib.util
from pathlib import Path

import torch


def repo_root():
    return Path(__file__).resolve().parents[3]


def load_dogfight_torch():
    dogfight_torch_path = repo_root() / "ocean" / "dogfight" / "torch.py"
    spec = importlib.util.spec_from_file_location(
        "dogfight_torch_for_tests",
        dogfight_torch_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dogfight_slowly_backend_config_uses_local_df36_policy_components():
    root = repo_root()
    parser = ConfigParser()
    parser.read(root / "config" / "dogfight.ini")

    assert parser.get("torch", "encoder") == "DogfightEncoder"
    assert parser.get("torch", "decoder") == "DogfightDecoder"

    dogfight_torch = (root / "ocean" / "dogfight" / "torch.py").read_text()
    assert "class DogfightEncoder" in dogfight_torch
    assert "class DogfightDecoder" in dogfight_torch
    assert "nn.Linear(obs_size, hidden_size)" in dogfight_torch
    assert "nn.GELU()" in dogfight_torch
    assert "nn.init.orthogonal_" in dogfight_torch


def test_torch_backend_can_resolve_dogfight_local_policy_components():
    torch_pufferl = (repo_root() / "pufferlib" / "torch_pufferl.py").read_text()

    assert "def _resolve_model_component" in torch_pufferl
    assert 'f"ocean.{args[\'env_name\']}.torch"' in torch_pufferl
    assert "_resolve_model_component(args, args['torch']['encoder'])" in torch_pufferl
    assert "_resolve_model_component(args, args['torch']['decoder'])" in torch_pufferl


def test_dogfight_diagnostic_decoder_orthogonalizes_df36_scale_one_layers():
    dogfight_torch = load_dogfight_torch()
    decoder = dogfight_torch.DogfightDecoder(
        (1, 1, 1),
        hidden_size=4,
        action_init_scale=1.0,
        value_init_scale=1.0,
    )

    action_rows = decoder.decoder_mean.weight @ decoder.decoder_mean.weight.T
    assert torch.allclose(action_rows, torch.eye(3), atol=1e-5)
    assert torch.allclose(decoder.decoder_mean.bias, torch.zeros(3))

    value_norm = decoder.value_function.weight.norm(dim=1)
    assert torch.allclose(value_norm, torch.ones(1), atol=1e-5)
    assert torch.allclose(decoder.value_function.bias, torch.zeros(1))
