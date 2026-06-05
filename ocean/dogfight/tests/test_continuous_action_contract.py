import importlib.util
from pathlib import Path

import torch


def load_action_contract():
    dogfight_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "dogfight_action_contract_for_tests",
        dogfight_dir / "action_contract.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_continuous_policy_logprob_uses_unsquashed_normal_actions():
    from pufferlib.models import DefaultDecoder

    contract = load_action_contract()
    decoder = DefaultDecoder((1, 1, 1, 1, 1), hidden_size=4)
    hidden = torch.zeros(2, 4)
    logits, _ = decoder(hidden)
    torch_pufferl = (repo_root() / "pufferlib" / "torch_pufferl.py").read_text()

    assert contract.CONTINUOUS_ACTION_DISTRIBUTION == "unsquashed_normal"
    assert isinstance(logits, torch.distributions.Normal)
    assert not isinstance(logits, torch.distributions.TransformedDistribution)
    assert "logits.log_prob(action.view(batch, -1)).sum(1)" in torch_pufferl
    assert "torch.tanh(action" not in torch_pufferl


def test_dogfight_native_env_clamps_unsquashed_actions_before_physics():
    contract = load_action_contract()
    dogfight_h = (repo_root() / "ocean" / "dogfight" / "dogfight.h").read_text()

    assert contract.DOGFIGHT_ENV_CLAMPS_ACTIONS_BEFORE_STEP
    assert "env->actions[i] = clampf(env->actions[i], -1.0f, 1.0f);" in dogfight_h
    assert dogfight_h.index("env->actions[i] = clampf") < dogfight_h.index(
        "step_plane_with_params(&env->player"
    )
