import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGO_SOURCE = (ROOT / "src" / "algo.cu").read_text()
PUFFERL_SOURCE = (ROOT / "src" / "pufferl.cu").read_text()
DOGFIGHT_SOURCE = (
    ROOT / "ocean" / "dogfight" / "dogfight_puffer.h"
).read_text()


def _physical_row(compact_row, rows_per_group, selected_per_group):
    group = compact_row // selected_per_group
    relative_row = compact_row % selected_per_group
    return group * rows_per_group + relative_row


def _primary_weights(advantages, rows_per_group, selected_per_group, alpha):
    weights = []
    for group_start in range(0, len(advantages), rows_per_group):
        for relative_row in range(selected_per_group):
            row = advantages[group_start + relative_row]
            weights.append(sum(abs(value) for value in row) ** alpha)
    return weights


def test_alpha_zero_and_one_never_map_to_frozen_rows():
    rows_per_group = 11
    selected_per_group = 6
    groups = 7
    compact_rows = groups * selected_per_group
    advantages = [
        [float(row + 1), float((row % 5) + 1)]
        for row in range(rows_per_group * groups)
    ]

    for alpha in (0.0, 1.0):
        weights = _primary_weights(
            advantages, rows_per_group, selected_per_group, alpha
        )
        draws = random.Random(1701 + int(alpha)).choices(
            range(compact_rows), weights=weights, k=500_000
        )
        assert all(
            _physical_row(draw, rows_per_group, selected_per_group)
            % rows_per_group
            < selected_per_group
            for draw in draws
        )


def test_priorities_are_invariant_to_arbitrary_frozen_contents():
    rows_per_group = 5
    selected_per_group = 2
    baseline = [
        [1.0, -2.0],
        [3.0, -4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [-1.0, 2.0],
        [-3.0, 4.0],
        [-5.0, -6.0],
        [-7.0, -8.0],
        [-9.0, -10.0],
    ]
    perturbed = [row[:] for row in baseline]
    for group_start in range(0, len(perturbed), rows_per_group):
        for relative_row in range(selected_per_group, rows_per_group):
            perturbed[group_start + relative_row] = [
                1.0e30 * (relative_row + 1),
                -1.0e30 * (relative_row + 2),
            ]

    for alpha in (0.0, 1.0):
        assert _primary_weights(
            baseline, rows_per_group, selected_per_group, alpha
        ) == _primary_weights(
            perturbed, rows_per_group, selected_per_group, alpha
        )


def test_native_pipeline_uses_explicit_trainable_rows_and_loss_mask():
    assert "zero_frozen_advantages_kernel" not in PUFFERL_SOURCE
    assert "int prio_trainable_rows = prio_primary_per_buffer * num_buffers;" in (
        PUFFERL_SOURCE
    )
    assert (
        "register_prio_buffers(pufferl->prio_bufs,\n"
        "        acts, prio_trainable_rows, minibatch_segments);"
    ) in PUFFERL_SOURCE
    assert "prio_build_grouped_cdf_cuda(" in PUFFERL_SOURCE
    assert "prio_sample_grouped_cuda(" in PUFFERL_SOURCE
    assert "graph.mb_trainable_mask.data" in PUFFERL_SOURCE

    assert "compute_grouped_prio_adv_reduction" in ALGO_SOURCE
    assert "map_grouped_prio_indices_and_mask" in ALGO_SOURCE
    assert "PrecisionTensor mb_trainable_mask;" in ALGO_SOURCE
    assert "float trainable = to_float(g.trainable_mask[n]);" in ALGO_SOURCE
    assert "float dL = inv_NT * trainable;" in ALGO_SOURCE


def test_dogfight_exposes_slot_one_as_the_frozen_policy_candidate():
    assert "env->agents[0].policy = 0;" in DOGFIGHT_SOURCE
    assert "env->agents[1].policy = 1;" in DOGFIGHT_SOURCE
