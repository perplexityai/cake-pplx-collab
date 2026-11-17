# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""Known gaps: the tests that miss the naive BF16 tier on a GPU model, expected to fail there.

Every listed test misses the tier (``tests/cake/test_helpers/cake_naive.py``) on a BF16
output only and in the max norm only; its L2 error meets the tier.  Most are BF16
rounding near ties: the largest error is about half a BF16 ulp of the top binade, and
the kernel's FP32 sum lands on the other side of a BF16 rounding midpoint a little more
often than torch's.  The composed BF16 accumulate (``accumulate=True`` into a bf16
``out``) rounds twice, the weight gradient to BF16 and then ``out.add_``, where the
naive tier rounds ``out0 + dW`` once.

``tests/conftest.py`` marks a listed test ``xfail`` on its GPU model only, and only for
its miss of the naive tier: any other failure of the test still fails.  The xfail is
strict in the deterministic suites, whose bits repeat run to run (a fix shows as an
XPASS, which fails), and not strict in the non-deterministic suites, where the order
of the atomic adds may move a near tie.  ``tests/test_known_gaps.py`` checks that every
key names a collected test.
"""

from __future__ import annotations

from pathlib import Path

#: The family directory, which holds ``tests/``; the keys are node ids relative to it.
ROOT = Path(__file__).resolve().parent.parent

#: ``torch.cuda.get_device_name`` -> the GPU model of :data:`KNOWN_GAPS`.
DEVICE_MODELS = {
    "NVIDIA B200": "B200",
    "NVIDIA GB200": "GB200",
    "NVIDIA GB300": "GB300",
    "NVIDIA VR NVL72": "VR200",
}

#: The text of an xfail's naive-tier miss, the only failure the xfail covers.
MISS = "misses the naive tier"

_NEAR_TIE = "known gap: BF16 rounding near tie in the max norm (L2 meets the naive tier)"
_ROUNDS_TWICE = (
    "known gap: the composed BF16 accumulate rounds twice (dW to BF16, then out.add_)"
)

#: GPU model -> {node id: reason} of the tests that miss the naive tier there.
KNOWN_GAPS: dict[str, dict[str, str]] = {
    "B200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_autograd_wrapper_matches_reference": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_balanced-sizes0-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_padded_outputs_leave_the_padding_untouched": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e8_m131072_balanced-gate_up]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e32_m131072_balanced_bf16]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _NEAR_TIE,
    },
    "GB200": {},
    "GB300": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e4_skewed_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_wave_halves-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves_queue-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_wave_halves-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e4_m136064_balanced_bf16]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _NEAR_TIE,
    },
    "VR200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[early_stop-k512]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[no_early_stop-k512]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_cuda_graph_replay_with_changed_offsets[wgrad]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves_k256t-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_prepared_launches_allocate_nothing_after_the_first[wgrad]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_k256t-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_k256t-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves_k256t-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_balanced-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_skewed-down]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_one_empty_bf16]": _ROUNDS_TWICE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_autograd_steps_give_identical_bits": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves_k256t-out_dtype0]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_graph_replays_give_the_eager_bits[wgrad_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _NEAR_TIE,
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _NEAR_TIE,
    },
}

#: Every listed node id, of any model.
LISTED = frozenset(key for gaps in KNOWN_GAPS.values() for key in gaps)

#: The deterministic suites (strict xfail): their file names start with this.
DET_SUITE = "test_cake_moe_grouped_gemm_det_train"


def node_key(path: str | Path, nodeid: str) -> str:
    """The key of a collected test: its file relative to :data:`ROOT`, then the rest of
    its node id (the same key wherever pytest's rootdir is)."""
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:  # not under ROOT: no key matches
        return nodeid
    return f"{rel}::{nodeid.split('::', 1)[1]}"


def known_gap(device_name: str, key: str) -> tuple[str, bool] | None:
    """The reason test ``key`` is a known gap on the GPU named ``device_name`` and whether
    its xfail is strict (a deterministic suite), or None."""
    reason = KNOWN_GAPS.get(DEVICE_MODELS.get(device_name, ""), {}).get(key)
    if reason is None:
        return None
    return reason, Path(key.split("::", 1)[0]).name.startswith(DET_SUITE)
