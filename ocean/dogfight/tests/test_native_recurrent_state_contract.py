from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[3]


def test_native_rollouts_store_initial_recurrent_state_for_training_minibatches():
    pufferlib_cu = (repo_root() / "src" / "pufferlib.cu").read_text()

    assert "PrecisionTensor states;" in pufferlib_cu
    assert ".states" in pufferlib_cu
    assert "__global__ void copy_state_rows" in pufferlib_cu
    assert "__device__ __forceinline__ void copy_state_row" in pufferlib_cu
    assert "copy_state_rows<<<" in pufferlib_cu
    assert "copy_state_row(rollouts.states.data, graph.mb_state.data" in pufferlib_cu
    assert "graph.mb_state.data" in pufferlib_cu
    assert "policy_forward_train(&pufferl.policy" in pufferlib_cu
