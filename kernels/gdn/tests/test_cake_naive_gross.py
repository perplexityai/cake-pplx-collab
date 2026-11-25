"""
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The gross-deviation bound of the naive-tier judge, on the CPU (:mod:`tests.cake_naive`): an output whose largest error
over the draws, in the L2 or the max norm, is more than :data:`tests.cake_naive.GROSS` times the tier's largest error
fails its test (:meth:`tests.cake_naive.TierJudge.failures`); a missed tier within the bound stays record-only.
"""

from __future__ import annotations

import pytest
import torch

from . import cake_naive as cn

pytestmark = pytest.mark.cpu_only


def _judge_case(corrupt=None):
    """Judge one output ``out`` over :data:`tests.cake_naive.DRAWS` draws with :func:`tests.cake_naive.judge_case`: the
    oracle a random float64 tensor, the tier its BF16 rounding, and the output that rounding as FP32 (a tie with the
    tier) or ``corrupt(oracle, j)`` on draw ``j`` where that is not None."""
    oracles = {}

    def run(j, s):
        expected = oracles[j] = torch.randn(64, 32, dtype=torch.float64)
        actual = expected.to(torch.bfloat16).float()
        if corrupt is not None and (bad := corrupt(expected, j)) is not None:
            actual = bad.float()
        return {"out": actual}, {"out": expected}

    def naive(j, tier):
        return {"out": oracles[j].to(cn.TIER_DTYPES[tier])}

    return cn.judge_case(run, naive, seed=17)


def _gross(failures):
    return [message for message in failures if "gross deviation" in message]


def _judge_of(ours, theirs, draws=cn.DRAWS):
    """A judge holding output ``out`` with errors ``ours`` and tier errors ``theirs`` (per draw, in both norms)."""
    judge = cn.TierJudge(draws)
    for draw in range(draws):
        for norm in cn.NORMS:
            entry = {"ours": ours[draw], "bf16": theirs[draw]}
            judge.errors.setdefault(("out", norm), {})[draw] = entry
        judge.prints.setdefault(draw, {})["out"] = (float(draw), 0.0)
        judge.effective.setdefault("out", {})[draw] = 2
    return judge


def test_the_bound_is_the_owner_ruling():
    assert cn.GROSS == 10.0


def test_a_tie_with_the_tier_passes():
    judge, failures = _judge_case()
    assert failures == []
    assert judge.verdicts()["out"]["tier"] == "bf16"


def test_an_output_of_zeros_fails_in_both_norms():
    # the mutation: every draw's output zeroed; finite, so its contract holds, and only the bound fails it
    judge, failures = _judge_case(lambda expected, j: torch.zeros_like(expected))
    assert judge.verdicts()["out"]["tier"] is None
    gross = _gross(failures)
    assert len(gross) == len(failures) == 2, failures
    assert [message.split()[2] for message in gross] == ["l2", "max"]


def test_one_draw_of_zeros_fails():
    def zero_draw_5(expected, j):
        return torch.zeros_like(expected) if j == 5 else None

    _, failures = _judge_case(zero_draw_5)
    gross = _gross(failures)
    assert gross and len(gross) == len(failures), failures


def test_ten_times_the_tier_passes_and_more_fails():
    theirs = [1.0] * cn.DRAWS
    at = _judge_of([1.0] * (cn.DRAWS - 1) + [cn.GROSS], theirs)
    assert _gross(at.failures()) == [] and at.misses()
    above = _judge_of([1.0] * (cn.DRAWS - 1) + [cn.GROSS * 1.0001], theirs)
    assert len(_gross(above.failures())) == 2


def test_no_bound_where_the_tier_is_not_finite_or_zero():
    # the tier gives no bound: a draw where the naive value is not finite (None), or a largest error of 0 (the zero
    # rule records that miss)
    not_finite = _judge_of([100.0] * cn.DRAWS, [1.0] * (cn.DRAWS - 1) + [None])
    assert _gross(not_finite.failures()) == []
    exact = _judge_of([100.0] * cn.DRAWS, [0.0] * cn.DRAWS)
    assert _gross(exact.failures()) == [] and exact.verdicts()["out"]["tier"] is None
