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

The verdict logic of the KDA rule on the CPU (``tests/cake_naive.py`` ``TierJudge`` and
``test_helpers/cake_kda_fla.py``): an output passes when it meets the naive BF16 tier or FLA under the same rule
(:data:`NEAR_TIE`, the max norm draw by draw, the zero rule), and fails only when it misses both; an output without an
FLA value (no FLA counterpart, or FLA unavailable) is judged against the naive tier only.

Each judged output here has two distinct values per draw (``[1, 2]`` scaled by a power of two per draw, so the draws
differ and the max norm is the norm judged draw by draw), and each side's error on a draw is its offset (one number for
both elements, or one per element) times that scale: the ratios of the errors are the ratios of the offsets.  The
offsets are dyadic, so every value is exact in FP32 and the naive values are exact in BF16.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from tests.cake_naive import DRAWS, NO_FLA_COUNTERPART, REFERENCE, TierJudge, tier_line
from tests.experimental.test_helpers import cake_kda_fla
from tests.experimental.test_helpers import cake_kda_known_gaps as known_gaps
from tests.experimental.test_helpers.cake_kda_fla import (
    UNAVAILABLE,
    FlaReference,
    FlaSteps,
    fla_line,
    fla_of,
)

pytestmark = pytest.mark.cpu_only

BASE = torch.tensor([1.0, 2.0], dtype=torch.float64)


def _scale(j: int) -> float:
    return 2.0 ** (j - DRAWS // 2)


def _offset(x) -> torch.Tensor:
    return torch.tensor(x if isinstance(x, tuple) else (x, x), dtype=torch.float64)


def _judge(ours, tier, fla, name: str = "out") -> TierJudge:
    """A judge of output ``name`` over :data:`DRAWS` draws: on draw ``j`` the offsets ``ours(j)``, ``tier(j)`` and
    ``fla(j)`` (a float or a pair: FLA's offset; a string or None: FLA's reason, passed as it is) times the draw's
    scale."""
    judge = TierJudge(DRAWS)
    for j in range(DRAWS):
        s = _scale(j)
        want = BASE * s
        actual = (want + _offset(ours(j)) * s).to(torch.float32)
        naive = (want + _offset(tier(j)) * s).to(torch.bfloat16)
        assert torch.equal(naive.double(), want + _offset(tier(j)) * s), "the naive values are exact in BF16"
        other = fla(j)
        if isinstance(other, (float, tuple)):
            other = (want + _offset(other) * s).to(torch.float32)
        judge.add(name, j, actual, want, {"bf16": naive}, fla=other)
    return judge


def _const(x):
    return lambda j: x


def test_meets_the_naive_tier_only() -> None:
    judge = _judge(_const(0.0625), _const(0.125), _const(0.03125))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] == "bf16"
    assert verdict["met"] == {"bf16": True, REFERENCE: False}
    assert verdict["wins"]["max"] == DRAWS and verdict["fla_wins"]["max"] == 0
    assert judge.failures() == []
    line = tier_line("t", verdict)
    assert line.startswith("TIER t out bf16 met=bf16 bf16=meets fla=misses draws=32 paired max 32/32 (fla 0/32) |")
    assert " bf16 " in line and " fla " in line and "(ours x0.5 / x0.5)" in line and "(ours x2 / x2)" in line


def test_meets_fla_only() -> None:
    judge = _judge(_const(0.1875), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] == REFERENCE
    assert verdict["met"] == {"bf16": False, REFERENCE: True}
    assert judge.failures() == []
    line = tier_line("t", verdict)
    assert line.startswith("TIER t out fla met=fla bf16=misses fla=meets draws=32 paired max 0/32 (fla 32/32) |")
    assert "(ours x1.5 / x1.5)" in line and "(ours x0.75 / x0.75)" in line


def test_misses_both_fails_and_shows_both_references() -> None:
    judge = _judge(_const(0.375), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None
    assert verdict["met"] == {"bf16": False, REFERENCE: False}
    (failure,) = judge.failures()
    assert failure.startswith("out: misses the naive tier and FLA over 32 inputs (draw by draw: ours within 1.005 x")
    assert "the tier on max 0 and within 1.005 x FLA on max 0 of 32 draws, at least 16 needed" in failure
    assert "bf16 " in failure and "fla " in failure
    assert "(ours x3 / x3)" in failure and "(ours x1.5 / x1.5)" in failure
    assert tier_line("t", verdict).startswith("TIER t out FAIL met=none bf16=misses fla=misses ")


def test_the_near_tie_is_the_rule_bound() -> None:
    # within 1.004 x the tier meets it; 1.006 x misses it (and ours equal to FLA's error meets FLA)
    met = _judge(_const(0.125 * 1.004), _const(0.125), _const(f"{UNAVAILABLE}: test"))
    assert next(iter(met.verdicts().values()))["tier"] == "bf16"
    missed = _judge(_const(0.125 * 1.006), _const(0.125), _const(f"{UNAVAILABLE}: test"))
    assert next(iter(missed.verdicts().values()))["tier"] is None
    equal = _judge(_const(0.125 * 1.006), _const(0.125), _const(0.125 * 1.006))
    assert next(iter(equal.verdicts().values()))["tier"] == REFERENCE


# ours and FLA with the same L2 error on every draw, ours ahead in the max norm on draws ``WIN`` and behind on the
# others: FLA is met when ours is within 1.005 x FLA's max error on at least half of the draws
P, Q = (0.1875, 0.25), (0.0, 0.3125)  # |P| = |Q| = 0.3125 (L2); max(P) = 0.25 < 0.3125 = max(Q) / 1.25


def _wins(win) -> TierJudge:
    return _judge(lambda j: P if win(j) else Q, _const(0.0625), lambda j: Q if win(j) else P)


@pytest.mark.parametrize("count", [15, 16, 17, 32])
def test_the_max_norm_is_judged_draw_by_draw_on_half_of_the_draws(count) -> None:
    assert abs(sum(x * x for x in P) - sum(x * x for x in Q)) < 1e-12
    judge = _wins(lambda j: j < count)
    (verdict,) = judge.verdicts().values()
    assert verdict["fla_wins"]["max"] == count and verdict["wins"]["max"] == 0
    assert verdict["tier"] == (REFERENCE if 2 * count >= DRAWS else None)


def test_fla_unavailable_judges_the_naive_tier_only() -> None:
    reason = f"{UNAVAILABLE}: fla does not import (ModuleNotFoundError: No module named 'fla')"
    passed = _judge(_const(0.0625), _const(0.125), _const(reason))
    (verdict,) = passed.verdicts().values()
    assert verdict["tier"] == "bf16" and verdict["met"] == {"bf16": True, REFERENCE: None}
    assert verdict["fla_reason"] == reason and REFERENCE not in verdict["norms"]["l2"]
    assert tier_line("t", verdict).endswith(f"| fla: {reason}, naive tier only")
    failed = _judge(_const(0.1875), _const(0.125), _const(reason))
    (verdict,) = failed.verdicts().values()
    assert verdict["tier"] is None
    (failure,) = failed.failures()
    assert failure.startswith("out: misses the naive tier over 32 inputs (draw by draw: ours within 1.005 x the tier")
    assert failure.endswith(f"fla: {reason}, so judged against the naive tier only")


def test_an_output_without_an_fla_counterpart_is_judged_against_the_naive_tier_only() -> None:
    passed = _judge(_const(0.0625), _const(0.125), _const(None), name="aqk_int")
    (verdict,) = passed.verdicts().values()
    assert verdict["tier"] == "bf16" and verdict["fla_reason"] == NO_FLA_COUNTERPART
    assert tier_line("t", verdict).endswith(f"| fla: {NO_FLA_COUNTERPART}, naive tier only")
    failed = _judge(_const(0.1875), _const(0.125), _const(None), name="aqk_int")
    assert next(iter(failed.verdicts().values()))["tier"] is None
    (failure,) = failed.failures()
    assert failure.startswith("aqk_int: misses the naive tier over 32 inputs")
    assert NO_FLA_COUNTERPART in failure


def test_fla_on_some_draws_only_is_not_a_reference() -> None:
    judge = _judge(_const(0.1875), _const(0.125), lambda j: 0.25 if j else f"{UNAVAILABLE}: raised on draw 0")
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None and verdict["met"][REFERENCE] is None
    assert verdict["fla_reason"] == f"{UNAVAILABLE}: raised on draw 0 (no FLA value on draws [0] of 32)"


def test_the_zero_rule_holds_for_each_reference() -> None:
    # the tier exact on draw 3 (ours not): the tier is missed whatever the other draws say; FLA meets
    judge = _judge(_const(0.0625), lambda j: 0.0 if j == 3 else 0.125, _const(0.25))
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [3], REFERENCE: []}
    assert verdict["met"] == {"bf16": False, REFERENCE: True} and verdict["tier"] == REFERENCE
    # FLA exact on draw 5 (ours not) and the tier missed: both missed, the message names both zero rules
    judge = _judge(_const(0.1875), lambda j: 0.0 if j == 7 else 0.125, lambda j: 0.0 if j == 5 else 0.25)
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [7], REFERENCE: [5]} and verdict["tier"] is None
    (failure,) = judge.failures()
    assert "tier bf16: error 0 on draws [7], ours not" in failure
    assert "fla: error 0 on draws [5], ours not" in failure
    # ours exact too on the draws where a reference is exact: no zero-rule miss
    judge = _judge(lambda j: 0.0 if j == 5 else 0.0625, _const(0.125), lambda j: 0.0 if j == 5 else 0.03125)
    (verdict,) = judge.verdicts().values()
    assert verdict["zero"] == {"bf16": [], REFERENCE: []} and verdict["tier"] == "bf16"


def test_a_non_finite_fla_value_is_not_met() -> None:
    def fla(j):
        return float("nan") if j == 2 else 0.25

    judge = _judge(_const(0.1875), _const(0.125), fla)
    (verdict,) = judge.verdicts().values()
    assert verdict["norms"]["l2"][REFERENCE] is None and verdict["met"][REFERENCE] is False
    assert verdict["tier"] is None


def test_verdicts_survive_json_and_print() -> None:
    # the KCP judge writes rank 0's verdicts as JSON; the TIER line of the read verdict is the same
    judge = _judge(_const(0.1875), _const(0.125), _const(0.25))
    (verdict,) = judge.verdicts().values()
    back = json.loads(json.dumps(verdict))
    assert tier_line("t", back) == tier_line("t", verdict)


def test_settle_passes_an_fla_met_output_and_fails_a_double_miss() -> None:
    met = _judge(_const(0.1875), _const(0.125), _const(0.25))
    known_gaps.settle(met.verdicts(), met.failures(), lengths=(64,))
    missed = _judge(_const(0.375), _const(0.125), _const(0.25))
    with pytest.raises(AssertionError, match="out: misses the naive tier and FLA over 32 inputs"):
        known_gaps.settle(missed.verdicts(), missed.failures(), lengths=(64,))


def test_fla_reference_values_and_reasons() -> None:
    ddt = torch.arange(4 * 128, dtype=torch.float32)
    ref = FlaReference({"out": torch.zeros(3, 2, 128), "ddt_bias": ddt}, None, 0.5)
    assert ref.value("ddt_bias", torch.empty(4, 128)).shape == (4, 128)
    assert ref.value("out", torch.empty(3, 2, 128)).shape == (3, 2, 128)
    assert ref.value("aqk_int", torch.empty(3, 2, 128)) == "no FLA counterpart: FLA gives no aqk_int"
    gone = FlaReference(None, f"{UNAVAILABLE}: test")
    assert gone.value("out", torch.empty(1)) == f"{UNAVAILABLE}: test"
    steps = FlaSteps()
    steps.add(ref)
    steps.add(gone)
    steps.add(gone)
    assert steps.summary() == {"calls": 3, "seconds": 0.5, "reasons": {f"{UNAVAILABLE}: test": 2}}
    assert fla_line("t", steps.summary()) == f"FLA t calls=3 seconds=0.5 | {UNAVAILABLE}: test x2"


def _operands(device: str = "cpu", g=None) -> dict:
    S, H, D = 5, 2, 128
    return {
        "q": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "k": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "v": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "g": torch.full((S, H, D), -1.0, device=device) if g is None else g,
        "beta": torch.zeros(S, H, dtype=torch.bfloat16, device=device),
        "dout": torch.zeros(S, H, D, dtype=torch.bfloat16, device=device),
        "cu_seqlens": torch.tensor([0, S], dtype=torch.int32, device=device),
    }


def test_fla_of_gives_reasons_where_fla_does_not_run(monkeypatch) -> None:
    # no FLA in the environment
    monkeypatch.setattr(cake_kda_fla, "_loaded", (None, f"{UNAVAILABLE}: fla does not import (test)"))
    got = fla_of(_operands(), 128**-0.5, -5.0, False)
    assert got.values is None and got.reason == f"{UNAVAILABLE}: fla does not import (test)"
    # FLA present: operands on the CPU, and an activated gate outside FLA's safe gate range
    monkeypatch.setattr(cake_kda_fla, "_loaded", (object(), None))
    got = fla_of(_operands(), 128**-0.5, -5.0, False)
    assert got.values is None and got.reason == f"{UNAVAILABLE}: the operands are on cpu, FLA runs on CUDA devices"
    assert cake_kda_fla._check(_operands()) == got.reason
    g = torch.full((5, 2, 128), -1.0)
    g[2, 1, 7] = -6.0
    if torch.cuda.is_available():
        reason = cake_kda_fla._check(_operands("cuda", g.cuda()))
        assert reason.startswith(f"{UNAVAILABLE}: the activated gate is outside FLA's safe gate range [-5.0, 0.0]")


def test_fla_version_and_env(monkeypatch) -> None:
    # the reference is FLA 0.6.0's Triton chunk_kda; the environment of its import and calls is restored afterwards
    assert cake_kda_fla.FLA_VERSION == "0.6.0"
    assert cake_kda_fla.FLA_ENV == {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
    monkeypatch.setenv("FLA_FLASH_KDA", "1")
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    with cake_kda_fla._environment(cake_kda_fla.FLA_ENV):
        assert os.environ["FLA_FLASH_KDA"] == "0" and os.environ["FLA_TILELANG"] == "0"
    assert os.environ["FLA_FLASH_KDA"] == "1" and "FLA_TILELANG" not in os.environ
