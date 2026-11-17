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

A gap is keyed by the GPU model and the test's node id, and names the outputs that miss
the tier there (as the TIER lines name them).  ``tests/conftest.py`` marks a listed test
``xfail`` on its GPU model only, and the xfail covers its known miss only
(:func:`covers`): the test fails with a ``NaiveTierMiss`` (its misses of the naive tier
are its only failures, ``tests/cake/test_helpers/cake_moe_judge.py``), and every output
it missed is one the gap names, meets the tier in the L2 norm, misses it in the max norm
only and has no draw where the tier's error is 0 and its own is not.  Any other failure
of the test still fails: a broken contract or exact output, a bit, rest-state or refusal
check (a test's judgment runs all of its other checks before it raises), a miss of
another output, a miss in the L2 norm or by the zero rule.

The xfail is strict in the deterministic suites, whose bits repeat run to run (a fix
shows as an XPASS, which fails).  It is not strict in the non-deterministic suites, by
the package's policy: their contract allows launches whose order varies, so a fix is
reported and does not fail.  Every launch of this release runs in a fixed order and
repeats bit for bit.  ``tests/test_known_gaps.py`` checks that every key names a
collected test, the xfail rule on synthetic failures, and that no test checks anything
after its judgment.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

from tests.cake.test_helpers.cake_moe_judge import NaiveTierMiss

#: The family directory, which holds ``tests/``; the keys are node ids relative to it.
ROOT = Path(__file__).resolve().parent.parent

#: ``torch.cuda.get_device_name`` -> the GPU model of :data:`KNOWN_GAPS`.
DEVICE_MODELS = {
    "NVIDIA B200": "B200",
    "NVIDIA GB200": "GB200",
    "NVIDIA GB300": "GB300",
    "NVIDIA VR NVL72": "VR200",
}


class Gap(NamedTuple):
    """A known gap of a test on a GPU model: the outputs that miss the naive tier there
    (by their TIER-line names) and why."""

    outputs: frozenset[str]
    reason: str


_NEAR_TIE = "known gap: BF16 rounding near tie in the max norm (L2 meets the naive tier)"
_ROUNDS_TWICE = (
    "known gap: the composed BF16 accumulate rounds twice (dW to BF16, then out.add_)"
)


def _near_tie(*outputs: str) -> Gap:
    return Gap(frozenset(outputs), _NEAR_TIE)


def _rounds_twice(*outputs: str) -> Gap:
    return Gap(frozenset(outputs), _ROUNDS_TWICE)


#: GPU model -> {node id: gap} of the tests that miss the naive tier there.
KNOWN_GAPS: dict[str, dict[str, Gap]] = {
    "B200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_autograd_wrapper_matches_reference": _near_tie("dW"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_balanced-sizes0-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie("dW(4864x1536)[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_padded_outputs_leave_the_padding_untouched": _near_tie("dW"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e8_m131072_balanced-gate_up]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e32_m131072_balanced_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie("dW[sampled]"),
    },
    "GB200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e4_skewed_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie("dW(4864x1536)[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_wave_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves_queue-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_wave_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e4_m136064_balanced_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie("dW[sampled]"),
    },
    "GB300": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e4_skewed_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie("dW(4864x1536)[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_wave_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves_queue-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_wave_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e4_m136064_balanced_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie("dW[sampled]"),
    },
    "VR200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie("dW[sample]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie("dW(4864x1536)[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[early_stop-k512]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[no_early_stop-k512]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_cuda_graph_replay_with_changed_offsets[wgrad]": _near_tie("dW(first)[sampled]", "dW(second)[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves_k256t-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_prepared_launches_allocate_nothing_after_the_first[wgrad]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_k256t-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_k256t-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves_k256t-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_balanced-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_skewed-down]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _rounds_twice("out[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_one_empty_bf16]": _rounds_twice("out[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_autograd_steps_give_identical_bits": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves_k256t-out_dtype0]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_graph_replays_give_the_eager_bits[wgrad_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie("dW[spot]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie("dW[sampled]"),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie("dW[sampled]"),
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


def known_gap(device_name: str, key: str) -> tuple[str, bool, frozenset[str]] | None:
    """The reason test ``key`` is a known gap on the GPU named ``device_name``, whether its
    xfail is strict (a deterministic suite), and the outputs that miss the naive tier
    there; or None."""
    gap = KNOWN_GAPS.get(DEVICE_MODELS.get(device_name, ""), {}).get(key)
    if gap is None:
        return None
    return gap.reason, Path(key.split("::", 1)[0]).name.startswith(DET_SUITE), gap.outputs


def covers(error: BaseException | None, outputs: Collection[str]) -> bool:
    """Whether a known gap's xfail covers the failure ``error`` of its test: ``error`` is a
    :class:`NaiveTierMiss` (misses of the naive tier are the test's only failures), and
    every output it missed is one of ``outputs`` (the gap's), meets the tier in the L2
    norm, misses it in the max norm only, and has no draw where the tier's error is 0 and
    its own is not."""
    if not isinstance(error, NaiveTierMiss) or not error.missed:
        return False
    return all(
        miss.output in outputs and miss.met["l2"] and not miss.met["max"] and not miss.zero
        for miss in error.missed
    )
