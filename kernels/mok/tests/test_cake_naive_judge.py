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

The accuracy judge on the CPU (:mod:`tests.cake_naive`): the max norm's two-stage rule by wins (16 or more of 32
meet, 11 or fewer miss, 12 to 15 go on to 128 draws where 56 or more meet), the L2 median and largest rule and the zero
rule, the draws of stage 2 in :func:`tests.cake_naive.judge_case`, and the known-gap caps
(:mod:`tests.experimental._mok_gaps`).
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from tests import cake_naive as cn
from tests.experimental import _mok_gaps as gaps


def _record(judge, name, draw, *, max_win=True, l2=(0.5, 0.5), zero=False):
    """Record draw ``draw`` of output ``name`` straight into ``judge``: the max norm a win (ours = the tier's error) or
    a loss (ours = 1.25 x the tier's), the L2 errors ``l2`` = (ours, tier); ``zero`` makes the tier's errors 0."""
    tier = 0.0 if zero else 0.5
    ours_max = 0.5 if max_win else 0.625
    for norm, ours, theirs in (("l2", l2[0], 0.0 if zero else l2[1]), ("max", ours_max, tier)):
        judge.errors.setdefault((name, norm), {})[draw] = {"ours": ours, "bf16": theirs}
    judge.prints.setdefault(draw, {})[name] = (float(draw), 0.0)


def _fill(judge, name, wins, start=0, stop=32, **kwargs):
    """Draws ``start`` .. ``stop`` - 1 of ``name``: the first ``wins`` of them max-norm wins, the rest losses."""
    for draw in range(start, stop):
        _record(judge, name, draw, max_win=draw - start < wins, **kwargs)


def test_the_constants_are_the_owner_rule():
    assert (cn.DRAWS, cn.PAIRED_NORMS, cn.NEAR_TIE) == (32, ("max",), 1.005)
    assert math.ceil(cn.PAIRED_MEETS * 32) == 16
    assert cn.PAIRED_STAGE_TWO * 32 == 12
    assert cn.STAGE_TWO_DRAWS * 32 == 128
    assert math.ceil(cn.STAGE_TWO_MEETS * 128) == 56


def test_the_two_stages_pass_a_true_tie_as_the_ruling_states():
    def passes(p):
        def pmf(n, k):
            return math.comb(n, k) * p**k * (1 - p) ** (n - k)

        total = sum(pmf(32, w) for w in range(16, 33))
        for w in range(12, 16):
            total += pmf(32, w) * sum(pmf(96, v) for v in range(56 - w, 97))
        return total

    measured = {p: round(passes(p), 3) for p in (0.5, 0.45, 0.4, 0.375, 0.35)}
    assert measured == {0.5: 0.907, 0.45: 0.65, 0.4: 0.28, 0.375: 0.149, 0.35: 0.072}


@pytest.mark.parametrize(
    "wins,verdict",
    [(32, "meets"), (16, "meets"), (15, "undecided"), (12, "undecided"), (11, "misses"), (0, "misses")],
)
def test_the_first_stage_cuts_at_16_and_11_of_32(wins, verdict):
    judge = cn.TierJudge()
    _fill(judge, "out", wins)
    (found,) = judge.verdicts().values()
    assert found["wins"]["max"] == wins and found["counted"]["max"] == 32 and found["needed"]["max"] == 16
    assert found["tier"] == ("bf16" if verdict == "meets" else None)
    assert found["undecided"] == (("max",) if verdict == "undecided" else ())
    assert judge.stage_two() == (range(32, 128) if verdict == "undecided" else range(0))


@pytest.mark.parametrize("more,meets", [(42, True), (41, False), (96, True), (0, False)])
def test_stage_two_meets_at_56_of_128(more, meets):
    judge = cn.TierJudge()
    _fill(judge, "out", 14)
    added = judge.stage_two()
    assert added == range(32, 128) and judge.draws == 128 and judge.case_draws == 32
    _fill(judge, "out", more, start=32, stop=128)
    (found,) = judge.verdicts().values()
    assert (found["wins"]["max"], found["counted"]["max"], found["needed"]["max"]) == (14 + more, 128, 56)
    assert found["draws"] == 128 and found["undecided"] == ()
    assert found["tier"] == ("bf16" if meets else None)
    assert judge.stage_two() == range(0), "a judge at stage 2 asks for no further draws"
    assert bool(judge.failures()) != meets


def test_stage_two_keeps_the_l2_rule_on_the_case_draws():
    judge = cn.TierJudge()
    _fill(judge, "out", 13)
    judge.stage_two()
    # Stage 2's draws: every max norm a win, and an L2 error far over the tier's that the L2 rule never reads.
    _fill(judge, "out", 96, start=32, stop=128, l2=(9.0, 0.5))
    (found,) = judge.verdicts().values()
    assert found["tier"] == "bf16"
    assert found["counted"] == {"l2": 32, "max": 128}
    assert found["norms"]["l2"]["ours"] == [0.5, 0.5]


def test_stage_two_holds_the_zero_rule_on_its_draws():
    judge = cn.TierJudge()
    _fill(judge, "out", 13)
    judge.stage_two()
    _fill(judge, "out", 96, start=32, stop=128)
    _record(judge, "out", 100, zero=True)
    (found,) = judge.verdicts().values()
    assert found["tier"] is None and found["zero"] == {"bf16": [100]}


def test_an_output_the_case_draws_decide_keeps_its_verdict():
    judge = cn.TierJudge()
    _fill(judge, "decided", 20)
    _fill(judge, "undecided", 13)
    assert judge.stage_two() == range(32, 128)
    # Stage 2 would sink the decided output; its verdict stays on the case's 32 draws.
    _fill(judge, "decided", 0, start=32, stop=128)
    _fill(judge, "undecided", 50, start=32, stop=128)
    verdicts = judge.verdicts()
    assert verdicts["decided"]["tier"] == "bf16" and verdicts["decided"]["counted"]["max"] == 32
    assert verdicts["undecided"]["tier"] == "bf16" and verdicts["undecided"]["counted"]["max"] == 128


def test_an_output_that_misses_another_norm_still_takes_its_undecided_norm_to_stage_two():
    # 13 wins of 32 in the max norm and an L2 miss (ours 1.2 x the tier's), which no further draw changes: stage 2
    # still decides the max norm, which a known gap capped in L2 alone needs.
    judge = cn.TierJudge()
    _fill(judge, "out", 13, l2=(0.6, 0.5))
    (found,) = judge.verdicts().values()
    assert found["tier"] is None and found["undecided"] == ("max",) and found["met"] == {"l2": False, "max": None}
    assert judge.stage_two() == range(32, 128)


def test_an_output_the_zero_rule_misses_takes_no_stage_two():
    # 12 wins of 32 in the max norm, but the tier is exact on draw 7 and the output is not: no draw undoes that miss.
    judge = cn.TierJudge()
    _fill(judge, "out", 13)
    _record(judge, "out", 7, max_win=False, zero=True)
    (found,) = judge.verdicts().values()
    assert found["tier"] is None and found["undecided"] == () and found["zero"] == {"bf16": [7]}
    assert found["wins"]["max"] == 12 and found["met"]["max"] is None
    assert judge.stage_two() == range(0)


def test_reported_judges_take_the_draws_and_never_ask_for_them():
    main, alt = cn.TierJudge(), cn.TierJudge()
    _fill(main, "out", 20)
    _fill(alt, "out", 13)
    assert cn.stage_two(main, reported=(alt,)) == range(0) and alt.draws == 32
    _fill(main, "other", 14)
    assert cn.stage_two(main, reported=(alt,)) == range(32, 128)
    assert (main.draws, alt.draws) == (128, 128)
    with pytest.raises(ValueError, match="different numbers of draws"):
        cn.stage_two(cn.TierJudge(), reported=(cn.TierJudge(64),))


def test_the_failure_and_the_tier_line_name_the_wins():
    judge = cn.TierJudge()
    _fill(judge, "missed", 11)
    _fill(judge, "pending", 14)
    failures = judge.failures()
    assert failures[0].startswith(
        "missed: misses the naive tier over 32 inputs (draw by draw: ours within 1.005 x the tier on max 11 of 32 "
        "draws, at least 16 needed; error against the oracle, median / largest): "
    )
    assert failures[1].startswith("pending: misses the naive tier over 32 inputs")
    assert "max undecided on the case's 32 draws: stage 2 (128 draws, stage_two) was not run" in failures[1]
    lines = judge.lines("t")
    assert lines[0].startswith("TIER t missed FAIL draws=32 paired max 11/32 | l2 ours 0.5 / 0.5 bf16 0.5 / 0.5 | ")
    assert lines[1].endswith(" | undecided without stage 2")
    judge.stage_two()
    _fill(judge, "missed", 96, start=32, stop=128)
    _fill(judge, "pending", 41, start=32, stop=128)
    failures = judge.failures()
    assert len(failures) == 2 and "on max 11 of 32 draws, at least 16 needed" in failures[0]
    assert "on max 55 of 128 draws, at least 56 needed" in failures[1]
    assert "draws=128 paired max 55/128 |" in judge.lines("t")[1]


def _tensors(draw, win):
    """One draw's oracle, kernel value and naive BF16 value, exact in every dtype: the tier's error is 0.5 in the max
    norm and sqrt(0.5) in L2; the kernel's 0.5 (a win) or 0.625 (a loss) in the max norm, and below the tier's in L2."""
    expected = torch.tensor([1.0, 1.0, float(draw)], dtype=torch.float64)
    naive = torch.tensor([1.5, 1.5, float(draw)]).to(torch.bfloat16)
    actual = torch.tensor([1.5 if win else 1.625, 1.0, float(draw)])
    return expected, naive, actual


@pytest.mark.parametrize("stage_two_wins,meets", [(42, True), (41, False)])
def test_judge_case_runs_stage_two_and_restores_the_random_state(stage_two_wins, meets):
    calls, naive_values = [], {}
    alt = cn.TierJudge()

    def run(j, seed):
        assert seed == cn.draw_seed(5, j)
        calls.append(j)
        win = j < 14 if j < 32 else j - 32 < stage_two_wins
        expected, naive, actual = _tensors(j, win)
        naive_values[j] = {"out": naive}
        alt.add("out", j, actual, expected, {"bf16": naive})
        return {"out": actual}, {"out": expected}

    random.seed(1234)
    torch.manual_seed(1234)
    before = (random.random(), torch.rand(1).item())
    random.seed(1234)
    torch.manual_seed(1234)
    judge, failures = cn.judge_case(run, lambda j, tier: naive_values.pop(j), seed=5, reported=(alt,))
    assert (random.random(), torch.rand(1).item()) == before
    assert calls == list(range(128)) and judge.draws == alt.draws == 128
    (found,) = judge.verdicts().values()
    assert (found["wins"]["max"], found["counted"]["max"]) == (14 + stage_two_wins, 128)
    assert found["tier"] == ("bf16" if meets else None) and bool(failures) != meets
    assert alt.verdicts()["out"]["counted"]["max"] == 128


def test_judge_case_without_an_undecided_output_makes_32_draws():
    calls, naive_values = [], {}

    def run(j, seed):
        calls.append(j)
        expected, naive, actual = _tensors(j, j < 20)
        naive_values[j] = {"out": naive}
        return {"out": actual}, {"out": expected}

    judge, failures = cn.judge_case(run, lambda j, tier: naive_values.pop(j))
    assert calls == list(range(32)) and judge.draws == 32 and failures == []


def test_the_miss_ratios_read_the_norms_the_output_does_not_meet():
    judge = cn.TierJudge()
    _fill(judge, "max_only", 5)
    _fill(judge, "both", 0, l2=(0.6, 0.5))
    _fill(judge, "zero", 32)
    _record(judge, "zero", 3, zero=True)
    verdicts = judge.verdicts()
    assert verdicts["max_only"]["met"] == {"l2": True, "max": False}
    assert gaps.miss_ratios(verdicts["max_only"]) == {"max": [1.25, 1.25]}
    assert gaps.miss_ratios(verdicts["both"]) == {"l2": [pytest.approx(1.2), pytest.approx(1.2)], "max": [1.25, 1.25]}
    assert gaps.miss_ratios(verdicts["zero"]) == {}
    assert gaps.beyond_caps(verdicts["zero"], {"max": 1.3}) == ["tier bf16: error 0 on draws [3], ours not"]
    assert gaps.beyond_caps(verdicts["max_only"], {"max": 1.25}) == []
    assert gaps.beyond_caps(verdicts["both"], {"max": 1.25}) == ["l2: misses the tier, and the known gap has no cap in it"]
    assert gaps.cap_of(1.25) == 1.25
    assert gaps.cap_of(1.07201) == 1.0721
    assert gaps.cap_of(1.0) == cn.NEAR_TIE
    for ratio in (1.072, 1.0720000001, 1.0719999999, 1.8361234567):
        assert ratio <= gaps.cap_of(ratio) < ratio + 1.001e-4


def _gap_item(monkeypatch, caps, part="out"):
    """Judge known-gap item ``part`` of a case whose output misses the max norm at ratio 1.25 (5 wins of 32), under
    ``caps`` (``{arch: {norm: cap}}``) on an sm_103a GPU."""
    judge = cn.TierJudge()
    _fill(judge, "out", 5)
    _fill(judge, "fine", 32)
    item = "test_x[case-out]"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/test_m.py::{item} (call)")
    monkeypatch.setattr(gaps, "CAPS", {item: caps} if caps is not None else {})
    gaps.arch.cache_clear()
    monkeypatch.setattr(gaps.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gaps.torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    recorded = []
    try:
        gaps.check(lambda k, v: recorded.append((k, v)), part, list(judge.verdicts().values()), judge.failures(), ["out"])
    finally:
        gaps.arch.cache_clear()
    return recorded


def test_a_known_gap_within_its_caps_is_the_expected_miss(monkeypatch):
    with pytest.raises(AssertionError, match="out: misses the naive tier"):
        _gap_item(monkeypatch, {"sm_103a": {"max": 1.25}, "sm_100a": {"l2": 1.1}})


@pytest.mark.parametrize(
    "caps,why",
    [
        ({"sm_103a": {"max": 1.2499}}, "max median ours x1.25 the tier's, above the cap x1.2499"),
        ({"sm_103a": {"l2": 1.3}}, "max: misses the tier, and the known gap has no cap in it"),
        ({"sm_100a": {"max": 1.3}}, "no caps on this architecture"),
        (None, "no caps on this architecture"),
    ],
)
def test_a_known_gap_beyond_its_caps_or_without_them_fails(monkeypatch, caps, why):
    with pytest.raises(gaps.GapCapError, match="out: misses the naive tier on sm_103a beyond its known gap: ") as info:
        _gap_item(monkeypatch, caps)
    assert why in str(info.value)


@pytest.mark.parametrize(
    "caps",
    [{}, {"max": 1.25001}, {"max": 1.0}, {"max": math.inf}, {"max": 2}, {"l3": 1.1}],
    ids=["empty", "five-decimals", "below-near-tie", "inf", "int", "unknown-norm"],
)
def test_a_cap_of_another_form_is_refused(monkeypatch, caps):
    monkeypatch.setattr(gaps, "CAPS", {"test_x[a-out]": {"sm_103a": caps}})
    with pytest.raises(ValueError, match="test_x\\[a-out\\]: caps sm_103a"):
        gaps.bf16_straddle_dw("test_x[a-out]")
    monkeypatch.setattr(gaps, "CAPS", {"test_x[a-out]": {"sm_999a": {"max": 1.1}}})
    with pytest.raises(ValueError):
        gaps.bf16_straddle_dw("test_x[a-out]")


def test_the_main_item_leaves_the_known_gap_to_its_own_item(monkeypatch):
    recorded = _gap_item(monkeypatch, {"sm_103a": {"max": 1.25}}, part=gaps.OUTPUTS)
    assert [verdict["output"] for key, verdict in recorded] == ["fine"]


@pytest.mark.parametrize("further,meets", [(43, True), (42, False)])
def test_a_known_gap_capped_in_l2_takes_its_tied_max_norm_to_stage_two(monkeypatch, further, meets):
    # The output misses L2 at ratio 1.2, a miss its cap holds, and wins the max norm on 13 of the case's 32 draws and on
    # `further` of stage 2's 96: with 56 of 128 the max norm meets the tier and the item is the expected miss of its L2
    # cap; with 55 it misses the max norm too, which has no cap, and the item fails.
    judge = cn.TierJudge()
    _fill(judge, "out", 13, l2=(0.6, 0.5))
    assert judge.stage_two() == range(32, 128)
    _fill(judge, "out", further, start=32, stop=128, l2=(0.6, 0.5))
    (verdict,) = judge.verdicts().values()
    assert verdict["undecided"] == () and verdict["wins"]["max"] == 13 + further
    assert verdict["met"] == {"l2": False, "max": meets} and verdict["counted"] == {"l2": 32, "max": 128}
    item = "test_x[case-out]"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"tests/test_m.py::{item} (call)")
    monkeypatch.setattr(gaps, "CAPS", {item: {"sm_103a": {"l2": 1.2}}})
    monkeypatch.setattr(gaps, "arch", lambda: "sm_103a")
    seen = []
    expected = AssertionError if meets else gaps.GapCapError
    with pytest.raises(expected, match="out: misses the naive tier") as info:
        gaps.check(lambda k, v: seen.append((k, v)), "out", [verdict], judge.failures(), ["out"])
    assert type(info.value) is expected
    ((_, gap),) = [(key, value) for key, value in seen if key == "known_gap"]
    if meets:
        assert gap["ratios"] == {"l2": [1.2, 1.2]} and gap["beyond"] == []
    else:
        assert gap["ratios"] == {"l2": [1.2, 1.2], "max": [1.25, 1.25]}
        assert gap["beyond"] == ["max: misses the tier, and the known gap has no cap in it"]


def test_a_known_gap_item_records_its_ratios_and_caps(monkeypatch):
    judge = cn.TierJudge()
    _fill(judge, "out", 5)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_m.py::test_x[case-out] (call)")
    monkeypatch.setattr(gaps, "CAPS", {"test_x[case-out]": {"sm_103a": {"max": 1.25}}})
    monkeypatch.setattr(gaps, "arch", lambda: "sm_103a")
    seen = []
    with pytest.raises(AssertionError):
        gaps.check(lambda k, v: seen.append((k, v)), "out", list(judge.verdicts().values()), judge.failures(), ["out"])
    assert [key for key, _ in seen] == ["naive_tier", "known_gap"]
    assert seen[1][1] == {
        "item": "test_x[case-out]",
        "arch": "sm_103a",
        "caps": {"max": 1.25},
        "ratios": {"max": [1.25, 1.25]},
        "beyond": [],
    }


def test_a_marker_is_active_on_the_architectures_its_caps_list(monkeypatch):
    monkeypatch.setattr(gaps, "CAPS", {"test_x[a-out]": {"sm_100a": {"max": 1.1}, "sm_107a": {"l2": 1.2, "max": 1.3}}})
    gaps.arch.cache_clear()
    monkeypatch.setattr(gaps.torch.cuda, "is_available", lambda: True)
    try:
        for capability, active in (((10, 0), True), ((10, 3), False), ((10, 7), True)):
            monkeypatch.setattr(gaps.torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
            gaps.arch.cache_clear()
            mark = gaps.bf16_straddle_dw("test_x[a-out]").mark
            assert mark.args == (active,) and mark.kwargs["strict"] and mark.kwargs["raises"] is AssertionError
            assert mark.kwargs["reason"].startswith("known gap on sm_100a, sm_107a: ")
            assert "sm_100a max 1.1000; sm_107a l2 1.2000, max 1.3000" in mark.kwargs["reason"]
    finally:
        gaps.arch.cache_clear()


def test_every_known_gap_has_caps_of_the_cap_form():
    """Each item of :data:`tests.experimental._mok_gaps.CAPS` is a test item id (``<test function>[<id>]``) whose
    output ends its id, and each of its architectures has a nonempty ``{norm: cap}`` of caps of the cap form (the
    marker refuses any other at collection)."""
    for item, caps in gaps.CAPS.items():
        function, _, rest = item.partition("[")
        assert function.startswith("test_") and rest.endswith("]"), item
        assert rest[:-1].endswith(("d_scores", "dw_routed_gate", "dw_routed_up", "dw_routed_down")), item
        assert caps and gaps._checked(item, caps) is caps, item
