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

The owner-accepted misses on the CPU: ``tests/cake_naive.py`` (:func:`accepted_check`, the ``TIER`` line) with its
table ``tests/cake_naive_accepted.json``, applied by ``test_helpers/cake_kda_known_gaps.py`` (``settle``).  An output
that misses the naive BF16 tier and FLA passes only when its test and output are listed and, against the listed
reference, each part of the rule is met or within its cap; beyond any cap, or not listed, it fails.  An entry with
a ``temporary`` field is accepted the same way until the fix it names removes it, and its ``TIER`` line says so; the
shipped table has no such entry and no class 3 entry (the fix of the in-kernel L2-norm backward removed them).

The judged outputs here are those of ``test_cake_kda_naive_fla_rule.py``: two distinct values per draw scaled by a
power of two per draw, each side's error its dyadic offset times that scale, so the ratios of the errors are the ratios
of the offsets and every value is exact.
"""

from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

import pytest
import torch

from tests import cake_naive
from tests.cake_naive import (
    ACCEPTED,
    DRAWS,
    NEAR_TIE,
    REFERENCE,
    TierJudge,
    accepted_cap,
    accepted_check,
    accepted_entries,
    tier_line,
)
from tests.experimental.test_helpers import cake_kda_known_gaps as known_gaps
from tests.experimental.test_helpers.cake_kda_fla import UNAVAILABLE

pytestmark = pytest.mark.cpu_only

BASE = torch.tensor([1.0, 2.0], dtype=torch.float64)
TEST = "tests/experimental/test_cake_kda_example.py::test_example[p]"
# ours, the tier and FLA: ours x2.5 the tier's and x1.25 FLA's, in the L2 norm and on every draw in the max norm
OURS, TIER, FLA = 0.3125, 0.125, 0.25
# equal L2 errors, ours ahead of the other side in the max norm when ours is P and the other side Q
P, Q = (0.1875, 0.25), (0.0, 0.3125)


def _scale(j: int) -> float:
    return 2.0 ** (j - DRAWS // 2)


def _offset(x) -> torch.Tensor:
    return torch.tensor(x if isinstance(x, tuple) else (x, x), dtype=torch.float64)


def _judge(ours, tier, fla, name: str = "out") -> TierJudge:
    """A judge of output ``name`` over :data:`DRAWS` draws: on draw ``j`` the offsets ``ours(j)``, ``tier(j)`` and
    ``fla(j)`` (a float or a pair: FLA's offset; a string: why FLA has no value) times the draw's scale."""
    judge = TierJudge(DRAWS)
    for j in range(DRAWS):
        s = _scale(j)
        want = BASE * s
        actual = (want + _offset(ours(j)) * s).to(torch.float32)
        naive = (want + _offset(tier(j)) * s).to(torch.bfloat16)
        other = fla(j)
        if isinstance(other, (float, tuple)):
            other = (want + _offset(other) * s).to(torch.float32)
        judge.add(name, j, actual, want, {"bf16": naive}, fla=other)
    return judge


def _const(x):
    return lambda j: x


def _missed(**kw) -> TierJudge:
    judge = _judge(_const(OURS), _const(TIER), _const(FLA), **kw)
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None and verdict["met"] == {
        "bf16": False,
        REFERENCE: False,
    }
    assert verdict["wins"]["max"] == 0 and verdict["fla_wins"]["max"] == 0
    return judge


def _entry(**kw) -> dict:
    entry = {
        "test": TEST,
        "output": "out",
        "class": 1,
        "reference": REFERENCE,
        "l2_median": 1.25,
        "l2_largest": 1.25,
        "max_paired": 0,
        "draws": DRAWS,
        "measured": "test",
    }
    entry.update(kw)
    return entry


@pytest.fixture
def table(monkeypatch):
    """Install the given entries as the owner's table (the shipped one is read by the tests of the real table)."""
    monkeypatch.setattr(known_gaps, "_records", {})

    def install(*entries):
        monkeypatch.setattr(cake_naive, "_accepted", {(e["test"], e["output"]): e for e in entries})

    return install


def _settle(judge: TierJudge, test: str = TEST) -> None:
    with known_gaps.running(test):
        known_gaps.settle(judge.verdicts(), judge.failures(), lengths=(64,))


@pytest.mark.parametrize(
    "test",
    [TEST, f"kernels/kda/{TEST}", f"/abs/path/{TEST}"],
    ids=["node_id", "from_the_repo_root", "absolute"],
)
def test_a_listed_miss_within_its_caps_passes_and_says_so(table, test) -> None:
    table(_entry())
    judge = _missed()
    assert judge.failures(), "the judge itself still reports the miss; settling applies the caps"
    _settle(judge, test)
    (verdict,) = judge.verdicts().values()
    assert accepted_check(test, verdict) == (_entry(), [])
    line = tier_line(test, verdict)
    assert line.startswith(
        f"TIER {test} out {ACCEPTED} met={ACCEPTED} cap=fla,l2<=1.25/1.25,max>=0/32 bf16=misses fla=misses draws=32 "
    )
    # a passing call with an accepted miss stays passed: no xfail
    report = SimpleNamespace(when="call", passed=True, outcome="passed")
    known_gaps.finish(known_gaps.canonical(test), report)
    assert report.outcome == "passed" and not hasattr(report, "wasxfail")


def test_a_listed_miss_against_the_naive_tier_within_its_caps_passes(table) -> None:
    table(_entry(reference="bf16", l2_median=2.5, l2_largest=2.5))
    _settle(_missed())
    table(_entry(reference="bf16", l2_median=2.5, l2_largest=2.4999))
    with pytest.raises(AssertionError, match=r"l2 largest: ours x2.5 bf16's, above the cap x2.4999"):
        _settle(_missed())


@pytest.mark.parametrize(
    "caps, reason",
    [
        ({"l2_median": 1.2499}, r"l2 median: ours x1.25 fla's, above the cap x1.2499"),
        (
            {"l2_largest": 1.2499},
            r"l2 largest: ours x1.25 fla's, above the cap x1.2499",
        ),
        (
            {"max_paired": 1},
            r"max: ours within 1.005 x fla on 0 of 32 draws, below the cap 1",
        ),
    ],
)
def test_a_listed_miss_beyond_any_cap_fails(table, caps, reason) -> None:
    entry = _entry(**caps)
    table(entry)
    judge = _missed()
    with pytest.raises(AssertionError, match=reason) as raised:
        _settle(judge)
    text = str(raised.value)
    assert f"out: an owner-accepted miss beyond its cap {accepted_cap(entry)}: " in text
    assert "out: misses the naive tier and FLA over 32 inputs" in text
    (verdict,) = judge.verdicts().values()
    line = tier_line(TEST, verdict)
    assert line.startswith(f"TIER {TEST} out FAIL met=none bf16=misses fla=misses draws=32 ")
    assert f"| beyond the accepted cap {accepted_cap(entry)}: " in line


def test_an_unlisted_miss_fails(table) -> None:
    table(_entry())
    # another test, another parametrization, another output: the judge's own failure
    for test, name in (
        (TEST.replace("[p]", "[q]"), "out"),
        (TEST.replace("test_example", "test_other"), "out"),
        (TEST, "p0.out"),
        (TEST, "dk"),
    ):
        judge = _missed(name=name)
        with pytest.raises(
            AssertionError,
            match=rf"{name}: misses the naive tier and FLA over 32 inputs",
        ) as raised:
            _settle(judge, test)
        assert "owner-accepted" not in str(raised.value)
        (verdict,) = judge.verdicts().values()
        assert accepted_check(test, verdict) == (None, [])
        assert tier_line(test, verdict).startswith(f"TIER {test} {name} FAIL met=none ")


def test_a_part_the_output_meets_keeps_the_ordinary_bound(table) -> None:
    # equal L2 errors (ratio 1 <= 1.005), the max norm within 1.005 x FLA on 8 draws: FLA is missed in the max norm only
    judge = _judge(lambda j: P if j < 8 else Q, _const(0.0625), lambda j: Q if j < 8 else P)
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None and verdict["fla_wins"]["max"] == 8
    assert verdict["norms"]["l2"]["ours"] == verdict["norms"]["l2"][REFERENCE]
    # L2 caps below the ordinary bound do not tighten it; the count cap is the measured 8
    table(_entry(l2_median=0.97, l2_largest=0.97, max_paired=8))
    _settle(judge)
    table(_entry(l2_median=0.97, l2_largest=0.97, max_paired=9))
    with pytest.raises(
        AssertionError,
        match=r"max: ours within 1.005 x fla on 8 of 32 draws, below the cap 9",
    ):
        _settle(judge)


def test_no_cap_covers_the_zero_rule_missing_fla_or_another_draw_count(table) -> None:
    table(_entry())
    # FLA exact on draw 3, ours not
    judge = _judge(_const(OURS), _const(TIER), lambda j: 0.0 if j == 3 else FLA)
    zero = r"fla: error 0 on draws \[3\], ours not \(the zero rule; no cap covers it\)"
    with pytest.raises(AssertionError, match=zero):
        _settle(judge)
    # no FLA values: an FLA cap does not apply
    judge = _judge(_const(OURS), _const(TIER), _const(f"{UNAVAILABLE}: test"))
    with pytest.raises(
        AssertionError,
        match=r"no FLA values \(fla unavailable: test\): an entry accepts only a miss of both references",
    ):
        _settle(judge)
    # another draw count, or the L2 norm judged draw by draw (one effective element)
    (verdict,) = _missed().verdicts().values()
    other = json.loads(json.dumps(verdict))
    other["draws"] = 64
    assert accepted_check(TEST, other)[1] == ["judged over 64 draws, the caps over 32"]
    other = json.loads(json.dumps(verdict))
    other["by_draw"] = ["l2", "max"]
    assert accepted_check(TEST, other)[1] == ["judged draw by draw in ('l2', 'max'), the caps in ('max',)"]


def test_an_entry_against_the_naive_tier_does_not_accept_a_miss_without_fla_values(table) -> None:
    # judged against the naive tier only (FLA unavailable): within the bf16 caps, but FLA was not judged, so the output
    # did not miss both references and no entry accepts it
    entry = _entry(reference="bf16", l2_median=2.5, l2_largest=2.5)
    table(entry)
    judge = _judge(_const(OURS), _const(TIER), _const(f"{UNAVAILABLE}: test"))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] is None and verdict["met"] == {"bf16": False, REFERENCE: None}
    why = "no FLA values (fla unavailable: test): an entry accepts only a miss of both references"
    assert accepted_check(TEST, verdict) == (entry, [why])
    with pytest.raises(AssertionError, match=re.escape(why)):
        _settle(judge)
    line = tier_line(TEST, verdict)
    assert line.startswith(f"TIER {TEST} out FAIL met=none bf16=misses fla=none draws=32 ")
    assert f"| beyond the accepted cap {accepted_cap(entry)}: {why}" in line
    # the same miss with FLA judged and missed is accepted within these caps
    _settle(_missed())


def test_a_temporary_entry_says_so_in_its_tier_line(table) -> None:
    why = "class 3: a test miss; removed by its fix"
    table(_entry(temporary=why))
    judge = _missed()
    _settle(judge)
    (verdict,) = judge.verdicts().values()
    line = tier_line(TEST, verdict)
    assert line.startswith(f"TIER {TEST} out {ACCEPTED} met={ACCEPTED} cap=fla,l2<=1.25/1.25,max>=0/32 ")
    assert line.endswith(f" | temporary: {why}")
    # beyond a cap it fails as any listed miss does, and still names the reason it was listed
    table(_entry(temporary=why, l2_median=1.2499))
    with pytest.raises(AssertionError, match=r"l2 median: ours x1.25 fla's, above the cap x1.2499"):
        _settle(judge)
    line = tier_line(TEST, verdict)
    assert line.startswith(f"TIER {TEST} out FAIL met=none ")
    assert line.endswith(f" | temporary: {why}") and "| beyond the accepted cap " in line
    # an entry without the field adds nothing
    table(_entry())
    assert "temporary" not in tier_line(TEST, verdict)


def test_a_met_verdict_is_not_checked(table) -> None:
    table(_entry())
    judge = _judge(_const(0.0625), _const(TIER), _const(FLA))
    (verdict,) = judge.verdicts().values()
    assert verdict["tier"] == "bf16"
    with pytest.raises(ValueError, match="met bf16, not a miss"):
        accepted_check(TEST, verdict)
    assert tier_line(TEST, verdict).startswith(f"TIER {TEST} out bf16 met=bf16 ")
    _settle(judge)


# the shipped table


def _nine(x: float) -> float:
    return float(f"{x:.9g}")


def _next_nine(x: float) -> float:
    """The next value above ``x`` at nine significant digits."""
    return _nine(_nine(x) + 10 ** (math.floor(math.log10(abs(x))) - 8))


def _verdict(entry: dict, med: float, largest: float, wins: int) -> dict:
    """A JSON-form verdict that missed both references, with errors ``med`` / ``largest`` over a reference error of 1 in
    the L2 norm and ``wins`` draws within NEAR_TIE of the entry's reference in the max norm."""
    ref = entry["reference"]
    other = "bf16" if ref == REFERENCE else REFERENCE
    count = {ref: wins, other: 0}
    return {
        "output": entry["output"],
        "tier": None,
        "draws": entry["draws"],
        "norms": {
            "l2": {"ours": [med, largest], ref: [1.0, 1.0], other: [0.5, 0.5]},
            "max": {"ours": [1.0, 1.0], ref: [1.0, 1.0], other: [0.5, 0.5]},
        },
        "zero": {"bf16": [], REFERENCE: []},
        "wins": {"l2": 0, "max": count["bf16"]},
        "by_draw": ["max"],
        "met": {"bf16": False, REFERENCE: False},
        "fla_wins": {"l2": 0, "max": count[REFERENCE]},
        "fla_reason": None,
    }


def test_the_shipped_table_is_well_formed() -> None:
    raw = json.loads(cake_naive.ACCEPTED_FILE.read_text(encoding="utf-8"))
    entries = raw["entries"]
    assert set(raw) == {"about", "fields", "entries"} and entries
    assert len({(e["test"], e["output"]) for e in entries}) == len(entries) == len(accepted_entries())
    for e in entries:
        assert set(e) | {"temporary", "accepted"} == set(raw["fields"]), e
        assert e["test"].startswith("tests/experimental/test_") and "::test_" in e["test"], e
        assert e["class"] in (1, 2, 4, 5) and e["reference"] in ("bf16", REFERENCE) and e["draws"] == DRAWS, e
        # no entry is temporary (class 3, the temporary class, has no entry)
        assert "temporary" not in e, e
        # an entry the owner added after the others names the date it was accepted
        assert "accepted" not in e or re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["accepted"]), e
        assert type(e["max_paired"]) is int and 0 <= e["max_paired"] <= DRAWS, e
        for cap in (e["l2_median"], e["l2_largest"]):
            assert type(cap) is float and math.isfinite(cap) and cap > 0 and _nine(cap) == cap, e
        # every entry relaxes a part of the rule: the measured miss
        assert e["l2_median"] > NEAR_TIE or e["l2_largest"] > NEAR_TIE or 2 * e["max_paired"] < e["draws"], e
        assert not known_gaps.KNOWN_GAP_ITEMS.get(e["test"], {}).get(e["output"]), e


def _is_kcp_l2_norm_dk_of_the_default_entry(test: str, output: str) -> bool:
    """dk of the s1450-h2 case of the default-entry KCP tests with the in-kernel Q/K L2 norm."""
    return (
        "test_cake_kda_nondet_train_kcp.py::test_gpu_default_" in test
        and test.endswith(("[g-l2n]", "[raw-l2n]"))
        and output.startswith("s1450-h2.")
        and output.endswith(".dk")
    )


def test_the_kcp_l2_norm_dk_of_the_default_entry_is_not_accepted() -> None:
    # dk of the s1450-h2 case of the default-entry tests with the in-kernel Q/K L2 norm: accepted until the fix of the
    # in-kernel L2-norm backward (class 3, temporary), which removed every such entry; a miss of both references there
    # now fails like any unlisted miss
    entries = accepted_entries()
    assert not any("temporary" in e or e["class"] == 3 for e in entries.values())
    assert not any(_is_kcp_l2_norm_dk_of_the_default_entry(test, output) for test, output in entries)
    test = "tests/experimental/test_cake_kda_nondet_train_kcp.py::test_gpu_default_entry_launches_match_held[g-l2n]"
    assert _is_kcp_l2_norm_dk_of_the_default_entry(test, "s1450-h2.default.dk")
    # a miss of both references there: listed nowhere, so it fails (no cap, no temporary note)
    miss = {"output": "s1450-h2.default.dk", "reference": "bf16", "draws": DRAWS}
    verdict = _verdict(miss, 1.2, 1.2, 0)
    assert accepted_check(test, verdict) == (None, [])
    line = tier_line(test, verdict)
    assert line.startswith(f"TIER {test} s1450-h2.default.dk FAIL met=none ") and "temporary" not in line


def test_every_shipped_entry_is_accepted_at_its_caps_and_not_beyond() -> None:
    for entry in accepted_entries().values():
        # at the cap, or at the ordinary bound for a part the measured misses met
        med = max(entry["l2_median"], NEAR_TIE)
        largest = max(entry["l2_largest"], NEAR_TIE)
        half = -(-entry["draws"] // 2)
        wins = min(entry["max_paired"], half)
        test = entry["test"]
        assert accepted_check(test, _verdict(entry, med, largest, wins)) == (
            entry,
            [],
        ), entry
        assert accepted_check(f"tests/cross_family/kda/{test}", _verdict(entry, med, largest, wins))[1] == []
        beyond = [
            _verdict(entry, _next_nine(med), largest, wins),
            _verdict(entry, med, _next_nine(largest), wins),
        ]
        if wins > 0 and wins < half:
            beyond.append(_verdict(entry, med, largest, wins - 1))
        for verdict in beyond:
            found, why = accepted_check(test, verdict)
            assert found == entry and len(why) == 1, (entry, why)
