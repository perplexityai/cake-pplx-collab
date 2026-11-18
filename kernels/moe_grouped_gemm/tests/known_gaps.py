# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""Known gaps: the tests that miss the naive BF16 tier on a GPU model, expected to fail there.

Every listed test misses the tier (``tests/cake/test_helpers/cake_naive.py``) on a BF16
output only and in the max norm only; its L2 error meets the tier.  They are BF16
rounding near ties: the largest error is about half a BF16 ulp of the top binade, and
the kernel's FP32 sum lands on the other side of a BF16 rounding midpoint a little more
often than torch's.  The composed BF16 accumulate (``accumulate=True`` into a bf16
``out``) rounds as the naive tier does, the weight gradient to BF16 and then
``out0 + dW`` to BF16; its misses are the same near ties: the weight gradient rounds
to the other BF16 neighbour, and the second rounding carries that into ``out``.

A gap is keyed by the GPU model and the test's node id, and names the outputs that miss
the tier there (as the TIER lines name them), each with two caps.  An output's max-norm
miss ratios are its median and its largest max-norm error over the draws, each over the
tier's (``Missed.ratios["max"]``, the two ratios the judge holds to ``NEAR_TIE``).  Each
ratio has its own cap: the largest value of that ratio the validation runs of this
release measured for the output on that GPU model, rounded up to four decimals (1.11417
gives 1.1142); or ``NEAR_TIE`` where that ratio met the tier in every run, so that it
must still meet it.
``tests/conftest.py`` marks a listed test ``xfail`` on its GPU model only, and the xfail
covers its known miss only (:func:`covers`): the test fails with a ``NaiveTierMiss``
(its misses of the naive tier are its only failures,
``tests/cake/test_helpers/cake_moe_judge.py``), and every output it missed is one the
gap names, meets the tier in the L2 norm, misses it in the max norm only, has no draw
where the tier's error is 0 and its own is not, and has each max-norm miss ratio at most
its cap.  Any other failure of the test still fails: a broken contract or exact output,
a bit, rest-state or refusal check (a test's judgment runs all of its other checks
before it raises), a miss of another output, a miss in the L2 norm or by the zero rule,
a max-norm miss ratio above its cap.  The report of such a failure says why the xfail
does not cover it (:func:`uncovered`): for a ratio above its cap, the output, the ratio
and its cap.

The xfail is strict in the deterministic suites, whose bits repeat run to run (a fix
shows as an XPASS, which fails).  It is not strict in the non-deterministic suites, by
the package's policy: their contract allows launches whose order varies, so a fix is
reported and does not fail.  Every launch of this release runs in a fixed order and
repeats bit for bit.  ``tests/test_known_gaps.py`` checks that every key names a
collected test and caps both max-norm miss ratios of each of its outputs, the selection
of the gaps and caps by GPU model, the xfail rule on synthetic failures (also through the
conftest's fixture and hook, in a pytest run of synthetic suites), and that no test
checks anything after its judgment.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from tests.cake.test_helpers.cake_moe_judge import NaiveTierMiss
from tests.cake.test_helpers.cake_naive import NEAR_TIE

#: The family directory, which holds ``tests/``; the keys are node ids relative to it.
ROOT = Path(__file__).resolve().parent.parent

#: ``torch.cuda.get_device_name`` -> the GPU model of :data:`KNOWN_GAPS`.
DEVICE_MODELS = {
    "NVIDIA B200": "B200",
    "NVIDIA GB200": "GB200",
    "NVIDIA GB300": "GB300",
    "NVIDIA VR NVL72": "VR200",
}


#: The names of an output's two max-norm miss ratios, in the order of ``Missed.ratios``.
STATISTICS = ("median", "largest")


class Gap(NamedTuple):
    """A known gap of a test on a GPU model: the outputs that miss the naive tier there
    (by their TIER-line names), each with the caps of its median and its largest
    max-norm miss ratio (module docstring), and why."""

    caps: Mapping[str, tuple[float, float]]
    reason: str


_REASON = "known gap: BF16 rounding near tie in the max norm (L2 meets the naive tier)"


def _near_tie(caps: Mapping[str, tuple[float, float]]) -> Gap:
    return Gap(dict(caps), _REASON)


#: GPU model -> {node id: gap} of the tests that miss the naive tier there, each output
#: with the caps of its median and its largest max-norm miss ratio (module docstring).
KNOWN_GAPS: dict[str, dict[str, Gap]] = {
    "B200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0115, 1.0232)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0199, 1.0265)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0116, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0084)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0107, 1.0177)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0069)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0052, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0056)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0114, 1.0204)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (1.0161, 1.0249)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0531, 1.0304)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0155)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0082, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0075)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0167, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0078, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0066, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0102)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0063)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_autograd_wrapper_matches_reference": _near_tie({"dW": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0293)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0356)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_balanced-sizes0-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie({"dW[sampled]": (1.0156, 1.0213)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie({"dW[sampled]": (1.0105, 1.0137)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0133)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie({"dW[sampled]": (1.0189, 1.0295)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie({"dW[sampled]": (1.0126, 1.0234)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0064)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie({"dW[sampled]": (1.0051, 1.0071)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie({"dW[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie({"dW(4864x1536)[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_padded_outputs_leave_the_padding_untouched": _near_tie({"dW": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie({"dW[sampled]": (1.0135, 1.0173)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0135, 1.0173)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0135, 1.0173)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie({"dW[sampled]": (1.0138, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0138, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0138, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0132)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e8_m131072_balanced-gate_up]": _near_tie({"out[sampled]": (NEAR_TIE, 1.1143)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e32_m131072_balanced_bf16]": _near_tie({"out[spot]": (1.0101, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _near_tie({"out[spot]": (1.0280, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0082, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0075)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0167, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0078, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0066, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0102)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0063)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0063)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0060)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0067)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0139, 1.0224)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie({"dW[sampled]": (1.0057, 1.0107)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0097)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0200, 1.0274)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie({"dW[sampled]": (1.0124, 1.0180)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie({"dW[sampled]": (1.0055, 1.0071)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0074)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0076)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0064)}),
    },
    "GB200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0203, 1.0207)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0181, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0115, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0092)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0121, 1.0193)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0051, 1.0085)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0122, 1.0207)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (1.0161, 1.0051)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e4_skewed_down_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0524, 1.0352)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0232)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0058)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0069, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0101)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0492)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie({"dW[sampled]": (1.0140, 1.0199)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie({"dW[sampled]": (1.0064, 1.0112)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0091)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie({"dW[sampled]": (1.0224, 1.0280)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie({"dW[sampled]": (1.0120, 1.0224)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0062)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie({"dW[sampled]": (1.0056, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie({"dW[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie({"dW(4864x1536)[sampled]": (1.0059, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave-out_dtype0]": _near_tie({"dW[sampled]": (1.0161, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0161, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave-out_dtype0]": _near_tie({"dW[sampled]": (1.0157, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0157, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_wave_halves-out_dtype1]": _near_tie({"out[sampled]": (1.0061, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0077)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0079)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0114)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves_queue-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_wave_halves-out_dtype1]": _near_tie({"out[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e4_m136064_balanced_bf16]": _near_tie({"out[spot]": (1.0082, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0058)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0069, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0064)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0076)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0083)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0059)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0113)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0144, 1.0186)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie({"dW[sampled]": (1.0065, 1.0142)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0087)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0214, 1.0285)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie({"dW[sampled]": (1.0131, 1.0246)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie({"dW[sampled]": (1.0051, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0069)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie({"dW[sampled]": (1.0051, 1.0065)}),
    },
    "GB300": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0203, 1.0207)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0181, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0115, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0092)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0121, 1.0193)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0051, 1.0085)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_down_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0122, 1.0207)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (1.0161, 1.0051)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e4_skewed_down_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0524, 1.0352)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0232)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0058)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0069, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0101)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0492)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie({"dW[sampled]": (1.0140, 1.0199)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie({"dW[sampled]": (1.0064, 1.0112)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0091)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie({"dW[sampled]": (1.0224, 1.0280)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie({"dW[sampled]": (1.0120, 1.0224)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0062)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie({"dW[sampled]": (1.0056, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie({"dW[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie({"dW(4864x1536)[sampled]": (1.0059, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0121, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave-out_dtype0]": _near_tie({"dW[sampled]": (1.0161, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_wave_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0161, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_queue-out_dtype0]": _near_tie({"dW[sampled]": (1.0119, 1.0211)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave-out_dtype0]": _near_tie({"dW[sampled]": (1.0157, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_wave_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0157, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_wave_halves-out_dtype1]": _near_tie({"out[sampled]": (1.0061, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0077)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0079)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0114)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves_queue-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_wave_halves-out_dtype1]": _near_tie({"out[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_down_e4_m136064_balanced_bf16]": _near_tie({"out[spot]": (1.0082, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0058)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0062, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0069, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0064)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0076)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0083)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0068)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0059)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0113)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0144, 1.0186)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie({"dW[sampled]": (1.0065, 1.0142)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0087)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0214, 1.0285)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie({"dW[sampled]": (1.0131, 1.0246)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie({"dW[sampled]": (1.0051, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0069)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie({"dW[sampled]": (1.0051, 1.0065)}),
    },
    "VR200": {
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0151, 1.0205)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0200, 1.0281)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m112576_down_wgrad_bf16]": _near_tie({"dW[sample]": (1.0100, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0143)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e4_m136064_skewed_gate_up_wgrad_bf16]": _near_tie({"dW[sample]": (1.0122, 1.0179)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_error_vs_fp64_per_output[e8_down_wgrad_bf16]": _near_tie({"dW[sample]": (NEAR_TIE, 1.0078)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e1_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0065)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0126, 1.0240)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e32_hot_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (1.0166, 1.0244)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_down_wgrad_bf16]": _near_tie({"dW[spot]": (1.0501, 1.0146)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_acceptance.py::test_five_repeats_under_copy_streams[e8_edges_gate_up_wgrad_bf16]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0300)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0090, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0053, 1.0086)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0110, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0075, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0063)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0074, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0105)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_det_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0072)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m112576_balanced-sizes9-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0306)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e1_m136064_balanced-sizes13-out_dtype0]": _near_tie({"dW[sampled]": (1.0238, 1.0362)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_one_empty-sizes2-out_dtype0]": _near_tie({"dW[sampled]": (1.0152, 1.0196)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m112576_skewed-sizes1-out_dtype0]": _near_tie({"dW[sampled]": (1.0106, 1.0135)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_balanced-sizes3-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0108)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_one_empty-sizes5-out_dtype0]": _near_tie({"dW[sampled]": (1.0213, 1.0313)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e4_m136064_skewed-sizes4-out_dtype0]": _near_tie({"dW[sampled]": (1.0143, 1.0164)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m112576_balanced-sizes10-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0056)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m131072_balanced-sizes6-out_dtype0]": _near_tie({"dW[sampled]": (1.0054, 1.0069)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_bench_wgrad_rows_match_reference[down-e8_m136064_balanced-sizes14-out_dtype0]": _near_tie({"dW[sampled]": (1.0060, 1.0072)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_changed_shapes_reuse_and_garbage_filled_scratch": _near_tie({"dW(4864x1536)[sampled]": (1.0057, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[early_stop-k512]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_checkpoint_early_stop_skips_the_forward_launch[no_early_stop-k512]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_cuda_graph_replay_with_changed_offsets[wgrad]": _near_tie({"dW(first)[sampled]": (NEAR_TIE, 1.0053), "dW(second)[sampled]": (NEAR_TIE, 1.0059)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_ordered_tail_forms_match_reference[main_halves_k256t-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_prepared_launches_allocate_nothing_after_the_first[wgrad]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0071)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main-out_dtype0]": _near_tie({"dW[sampled]": (1.0134, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0134, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_short_groups_on_two_concurrent_streams[main_halves_k256t-out_dtype0]": _near_tie({"dW[sampled]": (1.0117, 1.0141)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main-out_dtype0]": _near_tie({"dW[sampled]": (1.0134, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves-out_dtype0]": _near_tie({"dW[sampled]": (1.0134, 1.0166)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train.py::test_wgrad_short_groups_never_include_the_next_group[mixed-main_halves_k256t-out_dtype0]": _near_tie({"dW[sampled]": (1.0122, 1.0157)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves-out_dtype1]": _near_tie({"out[sampled]": (1.0055, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_accumulate_on_two_concurrent_streams[main_halves_k256t-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0075)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_balanced-down]": _near_tie({"out[sampled]": (1.0069, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0081)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m112576_skewed-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0060)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_one_empty-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0086)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_composed_bf16_accumulate_on_the_benchmark_rows[e4_m136064_skewed-down]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0083)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_acc.py::test_short_groups_accumulate_their_own_rows[mixed-main_halves-out_dtype1]": _near_tie({"out[sampled]": (NEAR_TIE, 1.0092)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_balanced_bf16]": _near_tie({"out[spot]": (1.0124, 1.0815)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_accumulate_sequences_give_identical_bits[wgrad_acc_gate_up_e4_m136064_one_empty_bf16]": _near_tie({"out[spot]": (NEAR_TIE, 1.0228)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_autograd_steps_give_identical_bits": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_forced_forms_give_identical_bits[main_halves_k256t-out_dtype0]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0057)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_graph_replays_give_the_eager_bits[wgrad_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0053)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0090, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0053, 1.0086)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (1.0110, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e4_m136064_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0075, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_down_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0063)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m112576_one_empty_bf16-planned]": _near_tie({"dW[spot]": (1.0074, NEAR_TIE)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0105)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e4_m136064_skewed_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0052)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_determinism.py::test_repeat_launches_give_identical_bits[wgrad_gate_up_e8_m131072_balanced_bf16-planned]": _near_tie({"dW[spot]": (NEAR_TIE, 1.0072)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0064)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e16_m131072_skewed-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0056)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_queue_rows_meet_the_naive_tier[e8_m131072_one_empty-bfloat16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0100)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0073)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e16_m131072_skewed_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0061)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0101)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0157, 1.0191)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m112576_skewed_bf16]": _near_tie({"dW[sampled]": (1.0072, 1.0137)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_balanced_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0123)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_one_empty_bf16]": _near_tie({"dW[sampled]": (1.0190, 1.0293)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e4_m136064_skewed_bf16]": _near_tie({"dW[sampled]": (1.0132, 1.0190)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_balanced_bf16]": _near_tie({"dW[sampled]": (1.0054, 1.0091)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_one_empty_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0117)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_down_e8_m131072_skewed_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0070)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e2_long_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0054)}),
        "tests/experimental/test_cake_moe_grouped_gemm_nondet_train_reproducibility.py::test_repeat_launches_and_naive_tier[wgrad_edge_e3_long_empty_down_bf16]": _near_tie({"dW[sampled]": (NEAR_TIE, 1.0070)}),
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


def known_gap(
    device_name: str, key: str
) -> tuple[str, bool, Mapping[str, tuple[float, float]]] | None:
    """The reason test ``key`` is a known gap on the GPU named ``device_name``, whether its
    xfail is strict (a deterministic suite), and the outputs that miss the naive tier
    there with their caps; or None."""
    gap = KNOWN_GAPS.get(DEVICE_MODELS.get(device_name, ""), {}).get(key)
    if gap is None:
        return None
    return gap.reason, Path(key.split("::", 1)[0]).name.startswith(DET_SUITE), gap.caps


def uncovered(
    error: BaseException | None, caps: Mapping[str, tuple[float, float]]
) -> list[str]:
    """Why a known gap's xfail does not cover the failure ``error`` of its test, one line
    per reason; empty when it covers it.  ``error`` must be a :class:`NaiveTierMiss`
    (misses of the naive tier are the test's only failures), and every output it missed
    must be one of ``caps`` (the gap's), meet the tier in the L2 norm, miss it in the max
    norm, have no draw where the tier's error is 0 and its own is not, and have its median
    and its largest max-norm miss ratio (module docstring) each at most its cap."""
    if not isinstance(error, NaiveTierMiss) or not error.missed:
        return ["the failure is not a miss of the naive tier alone (a NaiveTierMiss)"]
    reasons = []
    for miss in error.missed:
        name = miss.output
        if name not in caps:
            reasons.append(f"{name}: not an output of this known gap")
            continue
        if not miss.met["l2"]:
            reasons.append(f"{name}: misses the naive tier in the L2 norm")
        if miss.met["max"]:
            reasons.append(f"{name}: meets the naive tier in the max norm")
        if miss.zero:
            reasons.append(f"{name}: error not 0 on draws {list(miss.zero)}, where the tier's is 0")
        ratio = miss.ratios["max"]
        if ratio is None:
            reasons.append(f"{name}: no max-norm miss ratio (the tier's error is 0 or not finite)")
        else:
            for statistic, value, cap in zip(STATISTICS, ratio, caps[name], strict=True):
                if value > cap:
                    reasons.append(
                        f"{name}: {statistic} max-norm miss ratio {value:.9g} "
                        f"is above its cap {cap!r}"
                    )
    return reasons


def covers(error: BaseException | None, caps: Mapping[str, tuple[float, float]]) -> bool:
    """Whether a known gap's xfail covers the failure ``error`` of its test: no reason of
    :func:`uncovered` (``caps``: the gap's outputs and their caps)."""
    return not uncovered(error, caps)
