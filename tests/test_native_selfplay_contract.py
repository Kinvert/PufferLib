from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "pufferl.cu").read_text()


def test_selfplay_preflight_runs_before_native_runtime_creation():
    run_train = SOURCE[SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)") :]

    assert run_train.index(
        "validate_native_selfplay_config(ini, use_selfplay, ctx->world_size)"
    ) < (
        run_train.index("create_pufferl(ini, ctx)")
    )


def test_selfplay_preflight_rejects_async_and_topology_mismatch():
    start = SOURCE.index(
        "static void validate_native_selfplay_config(\n"
        "        Ini* ini, int use_selfplay, int world_size)"
    )
    end = SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)", start)
    validator = SOURCE[start:end]

    assert 'puf_ini_get_int(ini, "base", "async")' in validator
    assert "native self-play requires base.async=0" in validator
    assert "native self-play requires exactly one GPU/process" in validator
    assert "native self-play currently requires exactly one frozen bank" in validator
    assert "native self-play requires selfplay.eval_games=0" in validator
    assert 'puf_ini_get_int(ini, "policy", "hidden_size")' in validator
    assert 'puf_ini_get_int(ini, "policy", "num_layers")' in validator
    assert 'puf_ini_get_int(ini, "vec", "frozen_bank_hidden_size")' in validator
    assert 'puf_ini_get_int(ini, "vec", "frozen_bank_num_layers")' in validator
    assert "frozen-bank topology must match the primary policy" in validator


def test_opponent_rotation_is_an_atomic_between_rollout_barrier():
    helper_start = SOURCE.index("static long selfplay_atomic_rotate(")
    helper_end = SOURCE.index("#endif", helper_start)
    helper = SOURCE[helper_start:helper_end]

    reset = helper.index("puf_reset(env)")
    load = helper.index("pufferl_load_frozen_bank(")
    clear = helper.index("clear_primary_recurrent_row(")
    publish = helper.index("cudaMemcpyAsync(")
    assert reset < load < clear < publish
    assert "BUF_WAITING" in helper
    assert "cudaStreamSynchronize(pufferl->default_stream)" in helper

    run_train = SOURCE[SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)") :]
    rotate = run_train.index("selfplay_atomic_rotate(")
    throttle = run_train.index(
        "wall_clock() < pufferl->last_log_time + 0.6"
    )
    assert rotate < throttle
    assert "pending_path" not in run_train
    assert "pool/swap_truncations" in run_train


def test_frozen_recurrent_state_is_cleared_on_every_bank_load():
    start = SOURCE.index("void pufferl_load_frozen_bank(")
    end = SOURCE.index("// Die on OOM", start)
    loader = SOURCE[start:end]

    assert "bank->buffer_states[i]" in loader
    assert "cudaMemsetAsync(state->data, 0" in loader
    assert "cudaDeviceSynchronize()" in loader
