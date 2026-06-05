import importlib.util
import subprocess
from pathlib import Path


DF36_COMMIT = "171482a9b9889bebaa427ab658c95c0fc391c031"


def repo_root():
    return Path(__file__).resolve().parents[3]


def load_learner_compat():
    dogfight_dir = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "dogfight_learner_compat_for_tests",
        dogfight_dir / "learner_compat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def df36_file(path):
    return subprocess.check_output(
        [
            "git",
            "-C",
            "/home/claude/dogfight3",
            "show",
            f"{DF36_COMMIT}:{path}",
        ],
        text=True,
    )


def test_df36_known_good_dogfight_used_python_lstm_policy_names():
    compat = load_learner_compat()

    df36_config = df36_file("pufferlib/config/ocean/dogfight.ini")
    df36_ocean_torch = df36_file("pufferlib/ocean/torch.py")

    assert f"policy_name = {compat.DF36_POLICY_NAME}" in df36_config
    assert f"rnn_name = {compat.DF36_RNN_NAME}" in df36_config
    assert "class DogfightPolicy(Policy)" in df36_ocean_torch
    assert "class DogfightRecurrent(Recurrent)" in df36_ocean_torch
    assert "pufferlib.models.LSTMWrapper" in df36_file("pufferlib/ocean/torch.py")
    assert compat.DF36_RECURRENT_TYPE == "LSTMWrapper"


def test_dogfight5_normal_native_learner_is_mingru_not_df36_lstm():
    compat = load_learner_compat()
    root = repo_root()
    pufferlib_cu = (root / "src" / "pufferlib.cu").read_text()
    models_cu = (root / "src" / "models.cu").read_text()
    ocean_cu = (root / "src" / "ocean.cu").read_text()

    assert compat.DOGFIGHT5_NATIVE_NETWORK == "MinGRU"
    assert compat.DOGFIGHT5_NATIVE_LEARNER_MATCHES_DF36 is False
    assert ".forward = mingru_forward" in pufferlib_cu
    assert ".forward_train = mingru_forward_train" in pufferlib_cu
    assert ".backward = mingru_backward" in pufferlib_cu
    assert "puf_kaiming_init(&w2d, 1.0f" in models_cu
    assert "puf_kaiming_init(&wt, std::sqrt(2.0f)" in ocean_cu


def test_existing_lstm_path_is_pytorch_slowly_diagnostic_only():
    compat = load_learner_compat()
    root = repo_root()
    pufferl_py = (root / "pufferlib" / "pufferl.py").read_text()
    torch_pufferl_py = (root / "pufferlib" / "torch_pufferl.py").read_text()
    models_py = (root / "pufferlib" / "models.py").read_text()

    assert compat.DOGFIGHT5_LSTM_DIAGNOSTIC_BACKEND_FLAG == "--slowly"
    assert compat.DOGFIGHT5_LSTM_DIAGNOSTIC_NETWORK == "LSTM"
    assert "parser.add_argument('--slowly'" in pufferl_py
    assert "from pufferlib.torch_pufferl import PuffeRL" in pufferl_py
    assert "def _resolve_model_component" in torch_pufferl_py
    assert "_resolve_model_component(args, args['torch']['network'])" in torch_pufferl_py
    assert "class LSTM(nn.Module)" in models_py
