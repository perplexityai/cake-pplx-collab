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

Accuracy of a training kernel's outputs, judged against a naive implementation.

A training kernel's outputs have no tolerance.  Each floating output is judged
by its error against a float64 reference (the oracle), beside the error of a
naive implementation against the same oracle, over :data:`DRAWS` independent
random inputs of each case.  The naive implementation has one tier,
``"bf16"``: PyTorch on the same BF16 inputs, the same ops in BF16 with torch's
own internal accumulation.  Its result is cast to the dtype of the kernel's
output: an FP32 output gets the BF16 result widened exactly.

On each draw, the error of the output and of the tier is taken in the L2 and
in the max norm over the elements where the oracle is finite: the absolute
error ``||value - oracle||``, not divided by the oracle's norm.  The output meets
the tier when, in the L2 norm, the median of its errors over the draws is at
most :data:`NEAR_TIE` (1.005) times the tier's median and its largest error at
most :data:`NEAR_TIE` times the tier's largest; in the max norm
(:data:`PAIRED_NORMS`), by its wins, the draws where its error is at most
:data:`NEAR_TIE` times the tier's, in two stages (:data:`PAIRED_MEETS`): with
wins on 16 or more of the case's 32 draws it meets the tier, with 11 or fewer it
misses it, and with 12 to 15 stage 2 judges the case on 128 draws in all (the
96 further draws made as the case's are), where 56 or more wins meet the tier;
and its error is 0 on every draw where the tier's error is 0.  It passes
when it meets the tier and fails when it misses it.  A tier whose naive value
is not finite where the oracle is finite is not met.  Each test records every
output's verdict with pytest's ``record_property``
(:meth:`TierJudge.record`), and the family's ``conftest.py`` prints one
``TIER`` line per recorded output in the terminal summary, for passing tests
too::

    from .cake_naive import summarize_tiers


    def pytest_terminal_summary(terminalreporter, exitstatus, config):
        summarize_tiers(terminalreporter)

Every output is also held to its contract on every draw (:func:`contract`):
its shape is the oracle's; it is finite where the oracle is finite and holds
the oracle's inf or a NaN where the oracle holds one; an integer, index or
table output equals the oracle exactly.  A case whose inputs do not change
with the seed (all zeros, a fixed pattern of infs and NaNs) is held to its
contract only (:func:`contract`), as one input judges no accuracy.

The tests never compare a deterministic kernel with a non-deterministic one,
and never compare two versions of a kernel bit for bit.  A deterministic
kernel's own checks are run to run: the same call on the same inputs gives
the same bytes.

The comparisons work in blocks of about :data:`BLOCK` elements along dim 0, so
the float64 copies of a large output stay bounded.  Only torch is imported.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction

import torch

#: Elements per block of the float64 comparisons (blocks are whole rows of dim 0).
BLOCK = 1 << 26
#: The norms every judged output is measured in.
NORMS = ("l2", "max")
#: The naive tier, PyTorch BF16, and the dtype it returns.
TIERS = ("bf16",)
TIER_DTYPES = {"bf16": torch.bfloat16}
#: The independent random inputs each case is judged over, at least.
DRAWS = 32
#: How far an output's error may exceed the naive tier's and still meet it: in the L2 norm, its median and its largest
#: error over the draws at most this times the tier's; in a norm of :data:`PAIRED_NORMS`, its error at most this times
#: the tier's on enough of the draws (:data:`PAIRED_MEETS`; the owner's rulings). A near tie, never a tolerance:
#: where the tier's error is 0 the bound stays 0, and every draw where the tier's error is 0 still needs 0.
NEAR_TIE = 1.005
#: The norms every output is judged in draw by draw (:class:`TierJudge`): the max norm reads one element per draw, so
#: its median and largest over the draws only say how a few roundings fell (the owner's ruling). In such a norm an
#: output is judged by its wins, the draws where its error is at most :data:`NEAR_TIE` times the tier's
#: (:data:`PAIRED_MEETS`); in the other norm by its median and largest.
PAIRED_NORMS = ("max",)
#: When a norm judged draw by draw meets the tier, by its wins (:class:`TierJudge`; the owner's ruling), as fractions
#: of the case's draws N. With at least :data:`PAIRED_MEETS` N wins it meets the tier, and with fewer than
#: :data:`PAIRED_STAGE_TWO` N it misses it. Between the two, stage 2 judges the case on :data:`STAGE_TWO_DRAWS` N draws
#: in all, the further draws made as the case's are, and the norm meets the tier with at least :data:`STAGE_TWO_MEETS`
#: of them as wins. At N = 32: 16 wins meet, 11 miss, and 12 to 15 go on to 128 draws, where 56 wins meet. An output
#: that wins each draw with probability 1/2, a true tie, fails a cut at 16 of 32 alone with probability 0.43; the two
#: stages pass it with about 0.91, and an output that wins 40% of the draws with about 0.28. An output the case's N
#: draws decide keeps that verdict, read on those draws. Of an output stage 2 decides, the L2 median and largest stay
#: on the case's N draws, and the zero rule holds on every draw stage 2 adds.
PAIRED_MEETS = Fraction(1, 2)
PAIRED_STAGE_TWO = Fraction(3, 8)
STAGE_TWO_DRAWS = 4
STAGE_TWO_MEETS = Fraction(7, 16)
#: The seed of draw ``j`` of a case of seed ``base`` is ``base + j * DRAW_SEED_STRIDE``.
DRAW_SEED_STRIDE = 1_000_003


def draw_seed(base: int, draw: int) -> int:
    """The seed of draw ``draw`` of a case whose own seed is ``base``: ``base``
    for draw 0, far apart for the others."""
    return base + draw * DRAW_SEED_STRIDE


def seed_all(seed: int) -> None:
    """Seed the Python and torch (CPU and every CUDA device) random state."""
    random.seed(seed)
    torch.manual_seed(seed)


def _blocks(*tensors: torch.Tensor) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
    """Blocks of rows of same-shaped ``tensors``, about :data:`BLOCK` elements
    each, with the index of each block's first row; a 0-d tensor is one block."""
    head = tensors[0]
    if not head.dim():
        yield 0, tensors
        return
    rows = head.shape[0]
    step = max(1, BLOCK // max(1, head.numel() // max(1, rows)))
    for i in range(0, rows, step):
        yield i, tuple(t[i : i + step] for t in tensors)


def contract(
    name: str, actual: torch.Tensor, expected: torch.Tensor, exact: bool = False
) -> str | None:
    """Why ``actual`` breaks its contract against the oracle ``expected``, or
    None.  Its shape is the oracle's.  An ``exact`` output (integer, index or
    table) equals the oracle, in float64 where either side is floating (a NaN
    matching a NaN), else as integers.  Any other output is floating, finite
    where the oracle is finite, and holds the oracle's inf or a NaN where the
    oracle holds one."""
    if actual.shape != expected.shape:
        return f"{name}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    floating = actual.dtype.is_floating_point or expected.dtype.is_floating_point
    if not exact and not actual.dtype.is_floating_point:
        return f"{name}: a {actual.dtype} output is checked exactly (exact=True)"
    count, first, shown = 0, None, None
    for row, (a, e) in _blocks(actual, expected):
        e = e.to(actual.device)
        if exact:
            wide = torch.float64 if floating else torch.int64
            a, e = a.to(wide), e.to(wide)
            bad = ~((a == e) | (a.isnan() & e.isnan())) if floating else a != e
        else:
            bad = torch.where(
                e.isfinite(), ~a.isfinite(), ~((a == e) | (a.isnan() & e.isnan()))
            )
        n = int(bad.sum())
        if n and first is None:
            local = tuple(int(j) for j in bad.nonzero()[0]) if bad.dim() else ()
            first = (local[0] + row, *local[1:]) if local else local
            shown = (a[local].item(), e[local].item())
        count += n
    if not count:
        return None
    what = (
        "differ from the oracle"
        if exact
        else "are not finite where the oracle is, or miss its inf or NaN"
    )
    return (
        f"{name}: {count} of {actual.numel()} elements {what}; first at {first}: "
        f"{shown[0]}, expected {shown[1]}"
    )


def norm_errors(
    expected: torch.Tensor, tensors: Sequence[torch.Tensor], norm: str
) -> tuple[list[float], float]:
    """``||t - expected||`` for each of ``tensors``, and ``||expected||``, in
    ``norm`` (``"l2"`` or ``"max"``) over the elements where ``expected`` is
    finite, in float64 on the first tensor's device.  A non-finite difference
    counts as inf."""
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
    inf = float("inf")
    device = tensors[0].device
    sums = torch.zeros(len(tensors) + 1, dtype=torch.float64, device=device)
    for _, (e, *parts) in _blocks(expected, *tensors):
        e = e.to(device=device, dtype=torch.float64)
        keep = e.isfinite()
        e = torch.where(keep, e, 0.0)
        errors = [
            torch.where(
                keep,
                (x.to(device=device, dtype=torch.float64) - e)
                .abs()
                .nan_to_num(nan=inf, posinf=inf),
                0.0,
            )
            for x in parts
        ]
        terms = torch.stack([*errors, e.abs()]).reshape(len(tensors) + 1, -1)
        if norm == "max":
            sums = torch.maximum(sums, terms.max(dim=1).values if terms.shape[1] else sums)
        else:
            sums = sums + terms.square().sum(dim=1)
    values = (sums.sqrt() if norm == "l2" else sums).tolist()
    return values[:-1], values[-1]


def _fingerprint(expected: torch.Tensor) -> tuple[float, float]:
    sums = torch.zeros(2, dtype=torch.float64, device=expected.device)
    for _, (e,) in _blocks(expected):
        e = e.to(torch.float64)
        e = torch.where(e.isfinite(), e, 0.0)
        sums += torch.stack([e.sum(), e.square().sum()])
    return tuple(sums.tolist())


def _ratios(ours: list[float], theirs: list[float] | None) -> str:
    """The output's median and largest error over the tier's, to nine digits: an excess the rounded errors hide."""
    if theirs is None or not (theirs[0] > 0 and theirs[1] > 0):
        return ""
    return f" (ours x{ours[0] / theirs[0]:.9g} / x{ours[1] / theirs[1]:.9g})"


def _pair(errors: list[float] | None) -> str:
    return "none (not finite)" if errors is None else f"{errors[0]:.6g} / {errors[1]:.6g}"


class TierJudge:
    """The accuracy of a case's floating outputs over ``draws`` independent
    inputs, judged against the naive BF16 tier.

    Call :meth:`add` once per output and draw, then :meth:`stage_two` (or
    :func:`stage_two` of the judges of one draw loop) and :meth:`add` on each
    draw it returns, then :meth:`failures` (empty when every output met the
    tier and every draw was judged), and record the verdicts with
    :meth:`record` or :meth:`lines`.

    :meth:`verdicts` gives, for each output, whether it meets the tier over the
    case's ``draws``: in the L2 norm the median of the output's errors at most
    :data:`NEAR_TIE` times the tier's median and the largest at most
    :data:`NEAR_TIE` times the tier's largest; in each norm of
    :data:`PAIRED_NORMS` (the max norm, whose median and largest only say how a
    few roundings fell, the owner's ruling) by its wins, the draws where the
    output's error is at most :data:`NEAR_TIE` times the tier's; and the
    output's error 0 on every draw where the tier's is 0.

    A norm judged by its wins meets the tier with wins on at least
    :data:`PAIRED_MEETS` of the case's draws, misses it with wins on fewer than
    :data:`PAIRED_STAGE_TWO` of them, and is undecided between the two; an
    output that meets the rest of the tier with an undecided norm is undecided.
    An undecided output asks for stage 2 (:meth:`stage_two`; the owner's
    ruling): :data:`STAGE_TWO_DRAWS` times the case's draws in all.  On them,
    an undecided output is held to the zero rule, and each of its undecided
    norms meets the tier with wins on at least :data:`STAGE_TWO_MEETS` of them;
    the rest of its verdict, and the verdict of each output the case's draws
    decide, stays on the case's draws.  Without stage 2, an undecided output
    misses the tier.

    The verdict reports, per norm, the draws its rule counts (``counted``) and
    the count of them where the output's error is at most :data:`NEAR_TIE`
    times the tier's (``wins``); the norms judged by their wins (``by_draw``),
    with the wins each needs to meet the tier (``needed``); and the norms that
    stage 2 would decide where it did not run (``undecided``)."""

    def __init__(self, draws: int = DRAWS) -> None:
        if draws < DRAWS:
            raise ValueError(f"a case is judged over at least {DRAWS} draws, got {draws}")
        # the case's draws, which every verdict but an undecided output's stage 2 reads
        self.case_draws = draws
        # the draws the judge records: the case's, or stage 2's in all (stage_two)
        self.draws = draws
        # (output, norm) -> draw -> {"ours": error, tier: error or None}
        self.errors: dict[tuple[str, str], dict[int, dict[str, float | None]]] = {}
        self.prints: dict[int, dict[str, tuple[float, float]]] = {}

    def add(
        self,
        name: str,
        draw: int,
        actual: torch.Tensor,
        expected: torch.Tensor,
        naive: Mapping[str, torch.Tensor],
    ) -> None:
        """Record draw ``draw`` of output ``name``: the kernel's ``actual``,
        the oracle ``expected`` and ``naive[tier]`` for the tier of
        :data:`TIERS`, in the tier's dtype (:data:`TIER_DTYPES`).  Each naive
        value is cast to ``actual``'s dtype before it is measured.  A second
        launch on a recorded draw keeps the larger of the kernel's errors."""
        if not 0 <= draw < self.draws:
            raise ValueError(f"{name}: draw {draw} of a judge over {self.draws} draws")
        if actual.shape != expected.shape:
            raise ValueError(
                f"{name}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
            )
        tiers = {}
        for tier in TIERS:
            if tier not in naive:
                raise ValueError(f"{name}: no naive value for tier {tier}")
            value = naive[tier]
            if value.dtype != TIER_DTYPES[tier]:
                raise ValueError(
                    f"{name}: naive tier {tier} is {value.dtype}, not {TIER_DTYPES[tier]}"
                )
            if value.shape != expected.shape:
                raise ValueError(
                    f"{name}: naive tier {tier} has shape {tuple(value.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )
            tiers[tier] = value.to(actual.dtype)
        for norm in NORMS:
            values, _ = norm_errors(expected, (actual, *tiers.values()), norm)
            ours, theirs = values[0], dict(zip(tiers, values[1:], strict=True))
            entry = self.errors.setdefault((name, norm), {}).get(draw)
            if entry is None:
                self.errors[(name, norm)][draw] = {
                    "ours": ours,
                    **{t: (e if e < float("inf") else None) for t, e in theirs.items()},
                }
            else:
                entry["ours"] = max(entry["ours"], ours)
        self.prints.setdefault(draw, {})[name] = _fingerprint(expected)

    def stage_two(self) -> range:
        """Stage 2 of this judge alone (:func:`stage_two`): the draws to add, or an empty range."""
        return stage_two(self)

    def verdicts(self) -> dict[str, dict]:
        """Each output's verdict: the tier when it meets it (None when it
        misses it), the most draws it reads, per norm the median and largest
        error of the output and of the tier over the draws counted in that norm
        (None for a tier without an error on every one of them), the draws
        where the tier's error is 0 and the output's is not (``zero``), per
        norm the draws counted (``counted``), per norm judged by its wins
        (``by_draw``) the draws within :data:`NEAR_TIE` of the tier (``wins``)
        and the wins that meet it (``needed``), and the norms left undecided
        without stage 2 (``undecided``)."""
        found = {}
        (tier,) = TIERS
        for name in dict.fromkeys(name for name, _ in self.errors):
            norms, wins, counted, needed, met = {}, {}, {}, {}, {}
            for norm in NORMS:
                norms[norm], wins[norm], counted[norm] = self._tally(name, norm, self.case_draws)
                ours, bound = norms[norm]["ours"], norms[norm][tier]
                if norm not in PAIRED_NORMS:
                    met[norm] = (
                        bound is not None and ours[0] <= NEAR_TIE * bound[0] and ours[1] <= NEAR_TIE * bound[1]
                    )
                    continue
                needed[norm] = math.ceil(PAIRED_MEETS * counted[norm])
                if bound is None or wins[norm] < PAIRED_STAGE_TWO * counted[norm]:
                    met[norm] = False
                else:
                    # None: undecided, a count that stage 2 decides.
                    met[norm] = True if wins[norm] >= needed[norm] else None
            zero = self._zero(name, self.case_draws)
            undecided = (
                () if zero or False in met.values() else tuple(norm for norm in PAIRED_NORMS if met[norm] is None)
            )
            if undecided and self.draws > self.case_draws:
                zero = self._zero(name, self.draws)
                for norm in undecided:
                    norms[norm], wins[norm], counted[norm] = self._tally(name, norm, self.draws)
                    needed[norm] = math.ceil(STAGE_TWO_MEETS * counted[norm])
                    met[norm] = norms[norm][tier] is not None and wins[norm] >= needed[norm]
                undecided = ()
            meets = not zero and all(met.values())
            found[name] = {
                "output": name,
                "tier": tier if meets else None,
                "draws": max(counted.values()),
                "norms": norms,
                "zero": {tier: zero},
                "wins": wins,
                "counted": counted,
                "needed": needed,
                "by_draw": PAIRED_NORMS,
                "undecided": undecided,
            }
        return found

    def _tally(self, name: str, norm: str, draws: int) -> tuple[dict, int, int]:
        """Over the output's recorded draws under ``draws``, in ``norm``: the
        median and largest error of the output and of the tier (None without an
        error on every one of them), the draws where the output's error is at
        most :data:`NEAR_TIE` times the tier's, and how many draws there are."""
        (tier,) = TIERS
        entries = [entry for draw, entry in self.errors[(name, norm)].items() if draw < draws]
        ours = [entry["ours"] for entry in entries]
        theirs = [entry.get(tier) for entry in entries]
        errors = {
            "ours": [statistics.median(ours), max(ours)],
            tier: None if None in theirs else [statistics.median(theirs), max(theirs)],
        }
        wins = sum(1 for entry in entries if entry.get(tier) is not None and entry["ours"] <= NEAR_TIE * entry[tier])
        return errors, wins, len(entries)

    def _zero(self, name: str, draws: int) -> list[int]:
        """The output's recorded draws under ``draws`` where the tier's error is 0 and the output's is not."""
        (tier,) = TIERS
        return sorted(
            {
                draw
                for norm in NORMS
                for draw, entry in self.errors[(name, norm)].items()
                if draw < draws and entry.get(tier) == 0 and entry["ours"] > 0
            }
        )

    def failures(self) -> list[str]:
        """One message per output that missed the tier or was not judged on every
        draw, and one per pair of draws whose oracle values are the same."""
        found = []
        for name, verdict in self.verdicts().items():
            recorded = sorted(self.errors[(name, NORMS[0])])
            if recorded != list(range(self.draws)):
                found.append(f"{name}: judged on {len(recorded)} of {self.draws} draws")
            if verdict["tier"] is not None:
                continue
            parts = []
            for norm, errors in verdict["norms"].items():
                tiers = ", ".join(
                    f"{tier} {_pair(errors[tier])}{_ratios(errors['ours'], errors[tier])}" for tier in TIERS
                )
                parts.append(f"{norm}: ours {_pair(errors['ours'])}, {tiers}")
            parts += [
                f"tier {tier}: error 0 on draws {draws}, ours not"
                for tier, draws in verdict["zero"].items()
                if draws
            ]
            if verdict["undecided"]:
                parts.append(
                    f"{', '.join(verdict['undecided'])} undecided on the case's {self.case_draws} draws: stage 2 "
                    f"({STAGE_TWO_DRAWS * self.case_draws} draws, stage_two) was not run"
                )
            # The wins of each norm judged by them, grouped by the draws counted and the wins needed.
            counts: dict[tuple[int, int], list[str]] = {}
            for norm in verdict["by_draw"]:
                counts.setdefault((verdict["counted"][norm], verdict["needed"][norm]), []).append(
                    f"{norm} {verdict['wins'][norm]}"
                )
            wins = ", ".join(
                f"{', '.join(won)} of {count} draws, at least {need} needed" for (count, need), won in counts.items()
            )
            how = f"draw by draw: ours within {NEAR_TIE} x the tier on {wins}; "
            found.append(
                f"{name}: misses the naive tier over {verdict['draws']} inputs ({how}error "
                "against the oracle, median / largest): "
                + "; ".join(parts)
            )
        seen: dict[tuple, int] = {}
        for draw, prints in sorted(self.prints.items()):
            key = tuple(sorted(prints.items()))
            if key in seen:
                found.append(
                    f"draws {seen[key]} and {draw} have the same oracle values: the "
                    "inputs do not change with the seed"
                )
            else:
                seen[key] = draw
        return found

    def lines(self, test_id: str) -> list[str]:
        """One :func:`tier_line` per output."""
        return [tier_line(test_id, verdict) for verdict in self.verdicts().values()]

    def record(self, record_property: Callable[[str, object], None]) -> None:
        """Record each output's verdict with pytest's ``record_property``
        (``naive_tier``), which :func:`summarize_tiers` prints."""
        for verdict in self.verdicts().values():
            record_property("naive_tier", verdict)


def tier_line(test_id: str, verdict: Mapping) -> str:
    """``TIER <test> <output> bf16|FAIL``, the most draws the verdict reads,
    per norm judged draw by draw the draws where the output is within
    :data:`NEAR_TIE` of the tier over the draws counted in that norm
    (``paired max <n>/<draws> |``, 32 draws or stage 2's 128), per norm the
    median and largest error of the output and of the tier over those draws,
    and ``| undecided without stage 2`` for a count that stage 2 did not
    decide."""
    norms = " | ".join(
        f"{norm} ours {_pair(errors['ours'])} "
        + " ".join(f"{tier} {_pair(errors[tier])}" for tier in TIERS)
        for norm, errors in verdict["norms"].items()
    )
    if by_draw := verdict.get("by_draw"):
        wins = " ".join(f"{norm} {verdict['wins'][norm]}/{verdict['counted'][norm]}" for norm in by_draw)
        norms = f"paired {wins} | {norms}"
    if verdict.get("undecided"):
        norms += " | undecided without stage 2"
    return (
        f"TIER {test_id} {verdict['output']} {verdict['tier'] or 'FAIL'} "
        f"draws={verdict['draws']} {norms}"
    )


def stage_two(*judges: TierJudge, reported: Sequence[TierJudge] = ()) -> range:
    """Stage 2 of ``judges``, which record the same draws of one case (the
    owner's ruling): when an output of any of them is undecided on the case's
    draws (:meth:`TierJudge.verdicts`), each judge, and each of ``reported``
    (judges the same draw loop records in whose verdicts are reported, never
    judged: they never ask for stage 2), takes :data:`STAGE_TWO_DRAWS` times
    the case's draws in all, and the draws to add are returned, each to be made
    and recorded as the case's further draws are; else, and on judges already
    at stage 2, an empty range.

    A norm that wins each draw with probability 1/2, a true tie, fails a cut at
    half of 32 draws alone with probability 0.43; the two stages pass it with
    about 0.91, and a norm that wins 40% of the draws with about 0.28."""
    every = (*judges, *reported)
    if len(counts := {judge.case_draws for judge in every}) > 1:
        raise ValueError(f"judges of one draw loop over different numbers of draws: {sorted(counts)}")
    if any(judge.draws > judge.case_draws for judge in every) or not any(
        verdict["undecided"] for judge in judges for verdict in judge.verdicts().values()
    ):
        return range(0)
    (draws,) = counts
    added = range(draws, STAGE_TWO_DRAWS * draws)
    for judge in every:
        judge.draws = added.stop
    return added


def tier_lines(reports) -> Iterator[str]:
    """One :func:`tier_line` per verdict a call report recorded
    (``naive_tier`` in its user properties)."""
    for report in reports:
        if getattr(report, "when", None) != "call":
            continue
        for key, verdict in report.user_properties:
            if key == "naive_tier":
                yield tier_line(report.nodeid, verdict)


def summarize_tiers(terminalreporter) -> None:
    """The terminal summary's ``TIER`` lines: call it from the family
    conftest's ``pytest_terminal_summary``.  pytest captures a test's output,
    so the lines come from the recorded verdicts, passing tests included."""
    lines = list(
        tier_lines(
            report for reports in terminalreporter.stats.values() for report in reports
        )
    )
    if lines:
        terminalreporter.section("training accuracy: each output against the naive bf16 tier")
        for line in lines:
            terminalreporter.write_line(line)


def judge_case(
    run: Callable[[int, int], tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]],
    naive: Callable[[int, str], Mapping[str, torch.Tensor]],
    *,
    seed: int = 0,
    exact: frozenset[str] = frozenset(),
    draws: int = DRAWS,
    reported: Sequence[TierJudge] = (),
) -> tuple[TierJudge, list[str]]:
    """Judge one case over ``draws`` inputs, and over stage 2's where an output
    is undecided on them (:func:`stage_two`).

    For each draw ``j``, :func:`seed_all` seeds the random state with
    :func:`draw_seed` of ``seed`` and ``j``; ``run(j, s)`` (``s`` that seed)
    makes the draw's inputs, launches the kernel once and returns
    ``(outputs, oracle)`` by name; ``naive(j, tier)`` returns the tier's value
    of every judged output on the same inputs.  Every output is held to its
    :func:`contract` (exact for the names in ``exact``), and each other output
    is recorded in the judge.  After draws 0 .. ``draws`` - 1, the draws of
    stage 2, ``draws`` .. :data:`STAGE_TWO_DRAWS` x ``draws`` - 1, are made
    the same way when an output asks for them.  ``reported`` are the judges
    ``run`` also records in (verdicts reported, never judged): they take stage
    2's draws with the case's judge.  The Python, torch CPU and every CUDA
    device's random state are restored after the last draw.  Returns the judge
    and every failure: the contract's, then the judge's; record the verdicts
    with ``judge.record(record_property)``."""
    judge = TierJudge(draws)
    failures = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())

    def one(j: int) -> None:
        s = draw_seed(seed, j)
        seed_all(s)
        outputs, oracle = run(j, s)
        tiers = {tier: dict(naive(j, tier)) for tier in TIERS}
        for name, actual in outputs.items():
            want = oracle[name]
            if message := contract(name, actual, want, name in exact):
                failures.append(f"draw {j}: {message}")
                if actual.shape != want.shape:
                    continue
            if name in exact:
                continue
            judge.add(name, j, actual, want, {tier: tiers[tier][name] for tier in TIERS})

    try:
        for j in range(draws):
            one(j)
        for j in stage_two(judge, reported=reported):
            one(j)
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    return judge, failures + judge.failures()
