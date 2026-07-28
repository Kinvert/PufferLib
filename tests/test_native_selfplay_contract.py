from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "pufferl.cu").read_text()


def test_selfplay_preflight_runs_before_native_runtime_creation():
    run_train = SOURCE[SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)") :]

    assert run_train.index("validate_native_selfplay_config(ini, use_selfplay)") < (
        run_train.index("create_pufferl(ini, ctx)")
    )


def test_selfplay_preflight_rejects_async_and_topology_mismatch():
    start = SOURCE.index(
        "static void validate_native_selfplay_config(Ini* ini, int use_selfplay)"
    )
    end = SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)", start)
    validator = SOURCE[start:end]

    assert 'puf_ini_get_int(ini, "base", "async")' in validator
    assert "native self-play requires base.async=0" in validator
    assert 'puf_ini_get_int(ini, "policy", "hidden_size")' in validator
    assert 'puf_ini_get_int(ini, "policy", "num_layers")' in validator
    assert 'puf_ini_get_int(ini, "vec", "frozen_bank_hidden_size")' in validator
    assert 'puf_ini_get_int(ini, "vec", "frozen_bank_num_layers")' in validator
    assert "frozen-bank topology must match the primary policy" in validator
