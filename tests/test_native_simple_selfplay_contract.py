from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "pufferl.cu").read_text()


def _row_accounting(total_rows, agents_per_battle, frozen_battle_pct):
    battles = total_rows // agents_per_battle
    frozen_battles = int(frozen_battle_pct * battles)
    frozen_rows = frozen_battles
    current_rows = total_rows - frozen_rows
    return battles, frozen_battles, current_rows, frozen_rows


def test_modest_history_fraction_matches_pinned_allocator_semantics():
    battles, history_battles, current_rows, frozen_rows = _row_accounting(
        total_rows=4096,
        agents_per_battle=2,
        frozen_battle_pct=0.10,
    )

    assert battles == 2048
    assert history_battles == 204
    assert current_rows == 3892
    assert frozen_rows == 204


def test_training_logs_both_selfplay_cohorts_before_clearing_env_logs():
    train_start = SOURCE.index("TrainResult run_train(Ini* ini, TrainContext* ctx)")
    train = SOURCE[train_start:]
    current = train.index(
        'vec_log_tagged(pufferl->vec, &new_log, 0,\n'
        '                    "selfplay/current_vs_current")'
    )
    history = train.index(
        'vec_log_tagged(pufferl->vec, &new_log, 1,\n'
        '                    "selfplay/current_vs_history")'
    )
    clear = train.index("vec_log(pufferl->vec, &new_log, 1)")

    assert current < clear
    assert history < clear


def test_phase6_logs_battle_and_trainable_row_accounting():
    assert '"pool/current_vs_current_battles"' in SOURCE
    assert '"pool/current_vs_history_battles"' in SOURCE
    assert '"pool/current_trainable_rows"' in SOURCE
    assert '"pool/frozen_rows"' in SOURCE
    assert "selfplay.current_envs++" in SOURCE


def test_cohort_metrics_have_a_non_dashboard_evidence_sink():
    assert "void print_selfplay_cohort_metrics(Dict* log)" in SOURCE
    assert '"selfplay/metrics "' in SOURCE
    assert '"selfplay/current_vs_current/slot_0_score"' in SOURCE
    assert '"selfplay/current_vs_history/slot_0_score"' in SOURCE
    assert "print_selfplay_cohort_metrics(&new_log)" in SOURCE
