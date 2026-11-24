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
the tier when, in the L2 norm, the median of its errors over the draws is at most
:data:`NEAR_TIE` (1.005) times the tier's median and its largest error at most
:data:`NEAR_TIE` times the tier's largest; in the max norm (:data:`PAIRED_NORMS`,
one element per draw), its error is at most :data:`NEAR_TIE` times the tier's on
at least half of the draws; and its error is 0 on every draw where the tier's
error is 0.  An output whose oracle holds at most :data:`PAIRED_EFFECTIVE`
distinct nonzero finite value on every draw (one effective element: a single
token's scalar, one value repeated over heads) is judged draw by draw in the L2
norm too.  A tier whose naive value is not finite where the oracle is finite
is not met.

An output can have a second reference (:data:`REFERENCE`, ``"fla"``): the
value of flash-linear-attention (FLA) for the same output on the same inputs,
given to :meth:`TierJudge.add` as ``fla``.  FLA's error is taken against the
same oracle in the same norms on the same draws, on FLA's value as it is (not
cast), and the output meets FLA by the same rule: :data:`NEAR_TIE`, the norms
judged draw by draw, the zero rule.  The output passes when it meets the naive
tier or FLA, and fails only when it misses both.  An output without an FLA
value on every draw (it has no FLA counterpart, or FLA does not run the case)
is judged against the naive tier only: it passes
when it meets the tier and fails when it misses it, and its verdict names the
reason.

An output that misses both references passes only when the owner accepted it
(:func:`accepted_check`): its test and output are listed in
``cake_naive_accepted.json`` beside this file, and against the listed
reference (the naive tier or FLA) it is within the caps listed for it.  The
owner reviewed the validation of this version on B200, GB300 and VR200 (FLA's
autotuned kernels at their first config) and accepted these measured misses,
each capped at its measured values: the largest ratio of its median and of its
largest L2 error over the reference's (as the ``TIER`` line prints them, nine
significant digits), and the smallest count of draws within :data:`NEAR_TIE`
of the reference in the max norm.  A
part of the rule the output meets keeps its ordinary bound; the zero rule,
finite errors of the reference and the norms judged draw by draw still hold,
and every entry needs FLA's values: an output judged against the naive tier
only is never accepted, whatever its listed reference.  Any other output that
misses both references fails, and a listed output beyond a cap fails with the
reason.
The settling of a test's verdicts (the family's
``experimental/test_helpers/cake_kda_known_gaps.py``) applies the caps, and
the ``TIER`` line of an accepted output reads ``accepted met=accepted
cap=...``.  An entry with a ``temporary`` field is accepted only until the
fix it names lands, and the ``TIER`` line of its output adds ``| temporary:
<the field>``.

Each test records every
output's verdict with pytest's ``record_property``
(:meth:`TierJudge.record`), and the family's ``conftest.py`` prints one
``TIER`` line per recorded output in the terminal summary, for passing tests
too (:func:`tier_line`: the reference met, ``bf16``, ``fla`` or none, and the
errors and ratios against both references)::

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
the float64 copies of a large output stay bounded.  Only torch and the
standard library are imported.  This file and ``cake_naive_accepted.json``
are the same in the family's own tests and in its cross-family tests.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import torch

#: Elements per block of the float64 comparisons (blocks are whole rows of dim 0).
BLOCK = 1 << 26
#: The norms every judged output is measured in.
NORMS = ("l2", "max")
#: The naive tier, PyTorch BF16, and the dtype it returns.
TIERS = ("bf16",)
TIER_DTYPES = {"bf16": torch.bfloat16}
#: The second reference an output may meet instead of the naive tier: FLA on the same inputs (the KDA rule).
REFERENCE = "fla"
#: The reason an output is judged against the naive tier only when no FLA value is given for it.
NO_FLA_COUNTERPART = "no FLA counterpart"
#: The independent random inputs each case is judged over, at least.
DRAWS = 32
#: How far an output's error may exceed the naive tier's and still meet it: in the L2 norm, its median and its largest
#: error over the draws at most this times the tier's; in a norm of :data:`PAIRED_NORMS`, its error at most this
#: times the tier's on at least half of the draws (the owner's rulings). A near tie, never a tolerance: where the
#: tier's error is 0 the bound stays 0, and every draw where the tier's error is 0 still needs 0.
NEAR_TIE = 1.005
#: The norms every output is judged in draw by draw: the max norm reads one element per draw, so its median and
#: largest over the draws only say how a few roundings fell (the owner's ruling).
PAIRED_NORMS = ("max",)
#: The most distinct nonzero finite values the oracle of an output may hold on every draw for the output to be judged
#: draw by draw in every norm (:func:`effective_elements`): one effective element, whose median and largest error
#: only say how one rounding fell on each draw (the owner's ruling).
PAIRED_EFFECTIVE = 1
#: The seed of draw ``j`` of a case of seed ``base`` is ``base + j * DRAW_SEED_STRIDE``.
DRAW_SEED_STRIDE = 1_000_003
#: The verdict of an output that missed both references within the caps the owner accepted for it
#: (:func:`accepted_check`).
ACCEPTED = "accepted"
#: The owner-accepted misses and their caps (:func:`accepted_entries`).
ACCEPTED_FILE = Path(__file__).with_name("cake_naive_accepted.json")
_accepted: dict[tuple[str, str], dict] | None = None


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


def effective_elements(expected: torch.Tensor) -> int:
    """How many distinct nonzero finite values ``expected`` holds, counted up
    to 2: 0, 1, or 2 for two or more."""
    first = None
    for _, (e,) in _blocks(expected):
        e = e.to(torch.float64)
        values = e[e.isfinite() & (e != 0)]
        if not values.numel():
            continue
        if first is None:
            first = values[0]
        if bool((values != first).any()):
            return 2
    return 0 if first is None else 1


def _no_fla_reason(missing: Mapping[int, str], draws: int) -> str:
    """Why an output is judged against the naive tier only: the reasons of its draws without an FLA value."""
    text = "; ".join(dict.fromkeys(missing.values()))
    if len(missing) < draws:
        text += f" (no FLA value on draws {sorted(missing)} of {draws})"
    return text


class TierJudge:
    """The accuracy of a case's floating outputs over ``draws`` independent
    inputs, judged against the naive BF16 tier and, where its value is given,
    FLA (:data:`REFERENCE`): an output passes when it meets either.

    Call :meth:`add` once per output and draw, then :meth:`failures` (empty
    when every output met the tier or FLA and every draw was judged), and
    record the verdicts with :meth:`record` or :meth:`lines`."""

    def __init__(self, draws: int = DRAWS) -> None:
        if draws < DRAWS:
            raise ValueError(f"a case is judged over at least {DRAWS} draws, got {draws}")
        self.draws = draws
        # (output, norm) -> draw -> {"ours": error, tier: error or None, and with FLA's value "fla": error or None}
        self.errors: dict[tuple[str, str], dict[int, dict[str, float | None]]] = {}
        self.prints: dict[int, dict[str, tuple[float, float]]] = {}
        # output -> draw -> the oracle's distinct nonzero finite values, counted up to 2 (effective_elements)
        self.effective: dict[str, dict[int, int]] = {}
        # output -> draw -> why the draw has no FLA value (NO_FLA_COUNTERPART, or why FLA is unavailable)
        self.no_fla: dict[str, dict[int, str]] = {}

    def add(
        self,
        name: str,
        draw: int,
        actual: torch.Tensor,
        expected: torch.Tensor,
        naive: Mapping[str, torch.Tensor],
        fla: torch.Tensor | str | None = None,
    ) -> None:
        """Record draw ``draw`` of output ``name``: the kernel's ``actual``,
        the oracle ``expected``, ``naive[tier]`` for the tier of
        :data:`TIERS`, in the tier's dtype (:data:`TIER_DTYPES`), and ``fla``:
        FLA's value of the output on the same inputs, measured as it is, or
        the reason there is none (a string, such as why FLA is unavailable;
        None is :data:`NO_FLA_COUNTERPART`).  Each naive value is cast to
        ``actual``'s dtype before it is measured.  A second launch on a
        recorded draw keeps the larger of the kernel's errors."""
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
        if isinstance(fla, torch.Tensor):
            if fla.shape != expected.shape:
                raise ValueError(
                    f"{name}: FLA's value has shape {tuple(fla.shape)}, expected {tuple(expected.shape)}"
                )
            tiers[REFERENCE] = fla
        else:
            self.no_fla.setdefault(name, {})[draw] = fla or NO_FLA_COUNTERPART
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
        self.effective.setdefault(name, {})[draw] = effective_elements(expected)
        self.prints.setdefault(draw, {})[name] = _fingerprint(expected)

    def _against(self, name: str, ref: str, by_draw: tuple[str, ...]) -> tuple[dict, list[int], dict, bool]:
        """Output ``name`` against the reference ``ref`` (the tier, or :data:`REFERENCE`): per norm the reference's
        median and largest error (None when one is not finite), the draws where its error is 0 and ours is not, per
        norm the draws within :data:`NEAR_TIE` of it, and whether the output meets it."""
        bounds, zero, wins = {}, [], {}
        draws = len(self.errors[(name, NORMS[0])])
        meets = True
        for norm in NORMS:
            entries = self.errors[(name, norm)]
            ours = [entry["ours"] for entry in entries.values()]
            theirs = [entry.get(ref) for entry in entries.values()]
            bounds[norm] = None if None in theirs else [statistics.median(theirs), max(theirs)]
            wins[norm] = sum(
                1
                for entry in entries.values()
                if entry.get(ref) is not None and entry["ours"] <= NEAR_TIE * entry[ref]
            )
            for draw, entry in sorted(entries.items()):
                if entry.get(ref) == 0 and entry["ours"] > 0 and draw not in zero:
                    zero.append(draw)
            bound = bounds[norm]
            meets = meets and (
                bound is not None
                and (
                    2 * wins[norm] >= draws
                    if norm in by_draw
                    else statistics.median(ours) <= NEAR_TIE * bound[0] and max(ours) <= NEAR_TIE * bound[1]
                )
            )
        return bounds, zero, wins, meets and not zero

    def verdicts(self) -> dict[str, dict]:
        """Each output's verdict: the reference it meets (``tier``: the naive
        tier's name when it meets the tier, else :data:`REFERENCE` when it
        meets FLA, None when it misses both or misses the tier and has no FLA
        value), the number of draws, per norm the median and largest error of
        the output, of the tier and (when judged) of FLA, per reference the
        draws where its error was 0 and the output's was not (``zero``), per
        norm the draws within :data:`NEAR_TIE` of the tier (``wins``) and of
        FLA (``fla_wins``), the norms judged draw by draw (``by_draw``),
        whether it met each reference (``met``: None for FLA when not judged)
        and why FLA was not judged (``fla_reason``)."""
        found = {}
        (tier,) = TIERS
        for name in dict.fromkeys(name for name, _ in self.errors):
            paired = max(self.effective[name].values()) <= PAIRED_EFFECTIVE
            by_draw = NORMS if paired else tuple(n for n in NORMS if n in PAIRED_NORMS)
            norms = {}
            for norm in NORMS:
                ours = [entry["ours"] for entry in self.errors[(name, norm)].values()]
                norms[norm] = {"ours": [statistics.median(ours), max(ours)]}
            bounds, zero, wins, meets = self._against(name, tier, by_draw)
            for norm in NORMS:
                norms[norm][tier] = bounds[norm]
            zeros = {tier: zero}
            missing = self.no_fla.get(name)
            draws = len(self.errors[(name, NORMS[0])])
            fla_meets, fla_wins, fla_reason = None, None, None
            if missing:
                fla_reason = _no_fla_reason(missing, draws)
            else:
                bounds, zeros[REFERENCE], fla_wins, fla_meets = self._against(name, REFERENCE, by_draw)
                for norm in NORMS:
                    norms[norm][REFERENCE] = bounds[norm]
            found[name] = {
                "output": name,
                "tier": tier if meets else REFERENCE if fla_meets else None,
                "draws": draws,
                "norms": norms,
                "zero": zeros,
                "wins": wins,
                "by_draw": by_draw,
                "met": {tier: meets, REFERENCE: fla_meets},
                "fla_wins": fla_wins,
                "fla_reason": fla_reason,
            }
        return found

    def failures(self) -> list[str]:
        """One message per output that missed the tier and FLA (or missed the
        tier and has no FLA value) or was not judged on every draw, and one
        per pair of draws whose oracle values are the same."""
        found = []
        for name, verdict in self.verdicts().items():
            recorded = sorted(self.errors[(name, NORMS[0])])
            if recorded != list(range(self.draws)):
                found.append(f"{name}: judged on {len(recorded)} of {self.draws} draws")
            if verdict["tier"] is not None:
                continue
            judged = verdict["met"][REFERENCE] is not None
            refs = (*TIERS, REFERENCE) if judged else TIERS
            parts = []
            for norm, errors in verdict["norms"].items():
                shown = ", ".join(
                    f"{ref} {_pair(errors[ref])}{_ratios(errors['ours'], errors[ref])}" for ref in refs
                )
                parts.append(f"{norm}: ours {_pair(errors['ours'])}, {shown}")
            parts += [
                f"{'tier ' if ref in TIERS else ''}{ref}: error 0 on draws {draws}, ours not"
                for ref, draws in verdict["zero"].items()
                if draws
            ]
            counts = ", ".join(f"{norm} {verdict['wins'][norm]}" for norm in verdict["by_draw"])
            if judged:
                fla_counts = ", ".join(f"{norm} {verdict['fla_wins'][norm]}" for norm in verdict["by_draw"])
                what = "the naive tier and FLA"
                within = (
                    f"ours within {NEAR_TIE} x the tier on {counts} and within {NEAR_TIE} x FLA on {fla_counts} "
                    f"of {verdict['draws']} draws"
                )
            else:
                what = "the naive tier"
                within = f"ours within {NEAR_TIE} x the tier on {counts} of {verdict['draws']} draws"
                parts.append(f"fla: {verdict['fla_reason']}, so judged against the naive tier only")
            found.append(
                f"{name}: misses {what} over {verdict['draws']} inputs (draw by draw: {within}, "
                f"at least {-(-verdict['draws'] // 2)} needed; error against the oracle, median / largest): "
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


def node_key(test_id: str) -> str:
    """``test_id`` (pytest's node id) from its ``tests/experimental/`` directory on: the node id depends on the
    rootdir."""
    marker = "tests/experimental/"
    at = test_id.rfind(marker)
    return test_id[at:] if at >= 0 else test_id


def accepted_entries() -> dict[tuple[str, str], dict]:
    """The owner-accepted misses of :data:`ACCEPTED_FILE` by (test, output), read once."""
    global _accepted
    if _accepted is None:
        entries = json.loads(ACCEPTED_FILE.read_text(encoding="utf-8"))["entries"]
        _accepted = {(entry["test"], entry["output"]): entry for entry in entries}
    return _accepted


def accepted_cap(entry: Mapping) -> str:
    """An entry's caps as one token: ``<reference>,l2<=<median>/<largest>,max>=<draws within>/<draws>``."""
    return (
        f"{entry['reference']},l2<={entry['l2_median']!r}/{entry['l2_largest']!r},"
        f"max>={entry['max_paired']}/{entry['draws']}"
    )


def _l2_within(ours: float, theirs: float, cap: float) -> bool:
    """An L2 statistic of the output meets the reference's (at most :data:`NEAR_TIE` times it), or its ratio to it, to
    nine significant digits as the ``TIER`` line prints it, is at most ``cap``."""
    return ours <= NEAR_TIE * theirs or (theirs > 0 and float(f"{ours / theirs:.9g}") <= cap)


def accepted_check(test_id: str, verdict: Mapping) -> tuple[Mapping | None, list[str]]:
    """The owner's entry for the output of ``verdict`` (:meth:`TierJudge.verdicts`, or its JSON form) in the test
    ``test_id`` (pytest's node id), None when it is not listed, and why the verdict is beyond the entry's caps, one
    reason each (empty: the output is accepted).  Call it for a verdict that met no reference (``tier`` None): a miss of
    both references, or a miss of the naive tier where FLA has no values, which no entry accepts.

    Against the entry's reference (``"bf16"``, the tier, or ``"fla"``): the median and the largest L2 error over the
    draws each meet the reference's or their ratio to it is at most the entry's cap (``l2_median``, ``l2_largest``);
    in the max norm, judged draw by draw, the output is within :data:`NEAR_TIE` of the reference on at least half of
    the draws or on at least ``max_paired`` of them.  No cap covers the zero rule, an error of the reference that is
    not finite, another draw count or other norms judged draw by draw, and every entry, whatever its reference, needs
    FLA's values: it accepts only a miss of both references."""
    if verdict.get("tier") is not None:
        raise ValueError(f"{verdict['output']}: met {verdict['tier']}, not a miss to check against the owner's caps")
    entry = accepted_entries().get((node_key(test_id), verdict["output"]))
    if entry is None:
        return None, []
    ref, draws = entry["reference"], verdict["draws"]
    why = []
    if draws != entry["draws"]:
        why.append(f"judged over {draws} draws, the caps over {entry['draws']}")
    by_draw = tuple(verdict.get("by_draw") or ())
    if by_draw != PAIRED_NORMS:
        why.append(f"judged draw by draw in {by_draw}, the caps in {PAIRED_NORMS}")
    if (verdict.get("met") or {}).get(REFERENCE) is None:
        why.append(f"no FLA values ({verdict.get('fla_reason')}): an entry accepts only a miss of both references")
        return entry, why
    if verdict["zero"].get(ref):
        why.append(f"{ref}: error 0 on draws {verdict['zero'][ref]}, ours not (the zero rule; no cap covers it)")
    norms = verdict["norms"]
    if any(norms[norm].get(ref) is None for norm in NORMS):
        why.append(f"{ref}: an error that is not finite on a draw (no cap covers it)")
    else:
        ours, theirs = norms["l2"]["ours"], norms["l2"][ref]
        for i, (statistic, cap) in enumerate((("median", entry["l2_median"]), ("largest", entry["l2_largest"]))):
            if not _l2_within(ours[i], theirs[i], cap):
                ratio = ours[i] / theirs[i] if theirs[i] > 0 else float("inf")
                why.append(f"l2 {statistic}: ours x{ratio:.9g} {ref}'s, above the cap x{cap!r}")
    wins = (verdict["wins"] if ref in TIERS else verdict["fla_wins"])["max"]
    if 2 * wins < draws and wins < entry["max_paired"]:
        why.append(
            f"max: ours within {NEAR_TIE} x {ref} on {wins} of {draws} draws, below the cap {entry['max_paired']}"
        )
    return entry, why


def tier_line(test_id: str, verdict: Mapping) -> str:
    """``TIER <test> <output> bf16|fla|accepted|FAIL met=bf16|fla|accepted|none``
    (``accepted``: a miss of both references within the owner's caps for it,
    :func:`accepted_check`, followed by ``cap=<its caps>``), whether the
    output met each reference (``bf16=meets|misses fla=meets|misses|none``),
    the number of draws, per norm judged draw by draw the draws within
    :data:`NEAR_TIE` of the tier and of FLA (``paired max <n>/<draws> (fla
    <n>/<draws>) |``), and per norm the median and largest error of the output
    and of each reference with the output's ratios to them.  A listed miss
    beyond its caps adds ``| beyond the accepted cap <caps>: <reasons>``, and
    a listed miss of a temporary entry ``| temporary: <why>``.  An output
    judged against the naive tier only ends with ``| fla: <reason>, naive tier
    only``."""
    verdict_word, cap, beyond = verdict["tier"], "", ""
    if verdict_word is None:
        entry, why = accepted_check(test_id, verdict)
        if entry is not None and not why:
            verdict_word, cap = ACCEPTED, f" cap={accepted_cap(entry)}"
        elif entry is not None:
            beyond = f" | beyond the accepted cap {accepted_cap(entry)}: {'; '.join(why)}"
        if entry is not None and entry.get("temporary"):
            beyond += f" | temporary: {entry['temporary']}"
    met = verdict.get("met") or {}
    judged = met.get(REFERENCE) is not None
    refs = (*TIERS, REFERENCE) if judged else TIERS
    norms = " | ".join(
        f"{norm} ours {_pair(errors['ours'])} "
        + " ".join(f"{ref} {_pair(errors[ref])}{_ratios(errors['ours'], errors[ref])}" for ref in refs)
        for norm, errors in verdict["norms"].items()
    )
    if by_draw := verdict.get("by_draw"):
        wins = " ".join(
            f"{norm} {verdict['wins'][norm]}/{verdict['draws']}"
            + (f" (fla {verdict['fla_wins'][norm]}/{verdict['draws']})" if judged else "")
            for norm in by_draw
        )
        norms = f"paired {wins} | {norms}"
    shown = " ".join(
        f"{ref}={'none' if met.get(ref) is None else 'meets' if met[ref] else 'misses'}" for ref in (*TIERS, REFERENCE)
    )
    tail = "" if judged else f" | fla: {verdict.get('fla_reason') or NO_FLA_COUNTERPART}, naive tier only"
    return (
        f"TIER {test_id} {verdict['output']} {verdict_word or 'FAIL'} met={verdict_word or 'none'}{cap} "
        f"{shown} draws={verdict['draws']} {norms}{beyond}{tail}"
    )


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
        terminalreporter.section("training accuracy: each output against the naive bf16 tier and FLA")
        for line in lines:
            terminalreporter.write_line(line)


def judge_case(
    run: Callable[[int, int], tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]],
    naive: Callable[[int, str], Mapping[str, torch.Tensor]],
    *,
    seed: int = 0,
    exact: frozenset[str] = frozenset(),
    draws: int = DRAWS,
) -> tuple[TierJudge, list[str]]:
    """Judge one case over ``draws`` inputs.

    For each draw ``j``, :func:`seed_all` seeds the random state with
    :func:`draw_seed` of ``seed`` and ``j``; ``run(j, s)`` (``s`` that seed)
    makes the draw's inputs, launches the kernel once and returns
    ``(outputs, oracle)`` by name; ``naive(j, tier)`` returns the tier's value
    of every judged output on the same inputs.  Every output is held to its
    :func:`contract` (exact for the names in ``exact``), and each other output
    is recorded in the judge.  The Python, torch CPU and every CUDA device's
    random state are restored after the last draw.  No FLA value is given,
    so every output is judged against the naive tier only.  Returns the judge
    and every failure: the contract's, then the judge's; record the verdicts
    with ``judge.record(record_property)``."""
    judge = TierJudge(draws)
    failures = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(draws):
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
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    return judge, failures + judge.failures()
