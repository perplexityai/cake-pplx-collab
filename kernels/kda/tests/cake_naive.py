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
norm too.  It passes
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
This file is the same in every family that uses it.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence

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


class TierJudge:
    """The accuracy of a case's floating outputs over ``draws`` independent
    inputs, judged against the naive BF16 tier.

    Call :meth:`add` once per output and draw, then :meth:`failures` (empty
    when every output met the tier and every draw was judged), and record the
    verdicts with :meth:`record` or :meth:`lines`."""

    def __init__(self, draws: int = DRAWS) -> None:
        if draws < DRAWS:
            raise ValueError(f"a case is judged over at least {DRAWS} draws, got {draws}")
        self.draws = draws
        # (output, norm) -> draw -> {"ours": error, tier: error or None}
        self.errors: dict[tuple[str, str], dict[int, dict[str, float | None]]] = {}
        self.prints: dict[int, dict[str, tuple[float, float]]] = {}
        # output -> draw -> the oracle's distinct nonzero finite values, counted up to 2 (effective_elements)
        self.effective: dict[str, dict[int, int]] = {}

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
        self.effective.setdefault(name, {})[draw] = effective_elements(expected)
        self.prints.setdefault(draw, {})[name] = _fingerprint(expected)

    def verdicts(self) -> dict[str, dict]:
        """Each output's verdict: the tier when it meets it (None when it
        misses it), the number of draws, per norm the median and largest error
        of the output and of the tier, the draws where the tier was exact and
        the output was not, per norm the draws within :data:`NEAR_TIE` of the
        tier (``wins``), and the norms judged draw by draw (``by_draw``)."""
        found = {}
        (tier,) = TIERS
        for name in dict.fromkeys(name for name, _ in self.errors):
            norms = {}
            for norm in NORMS:
                by_draw = self.errors[(name, norm)]
                ours = [entry["ours"] for entry in by_draw.values()]
                theirs = [entry.get(tier) for entry in by_draw.values()]
                norms[norm] = {
                    "ours": [statistics.median(ours), max(ours)],
                    tier: None if None in theirs else [statistics.median(theirs), max(theirs)],
                }
            zero = {tier: []}
            for norm in NORMS:
                for draw, entry in sorted(self.errors[(name, norm)].items()):
                    if entry.get(tier) == 0 and entry["ours"] > 0 and draw not in zero[tier]:
                        zero[tier].append(draw)
            draws = len(self.errors[(name, NORMS[0])])
            wins = {
                norm: sum(
                    1
                    for entry in self.errors[(name, norm)].values()
                    if entry.get(tier) is not None and entry["ours"] <= NEAR_TIE * entry[tier]
                )
                for norm in NORMS
            }
            paired = max(self.effective[name].values()) <= PAIRED_EFFECTIVE
            by_draw = NORMS if paired else tuple(n for n in NORMS if n in PAIRED_NORMS)
            meets = not zero[tier] and all(
                (bound := norms[norm][tier]) is not None
                and (
                    2 * wins[norm] >= draws
                    if norm in by_draw
                    else norms[norm]["ours"][0] <= NEAR_TIE * bound[0]
                    and norms[norm]["ours"][1] <= NEAR_TIE * bound[1]
                )
                for norm in NORMS
            )
            met = tier if meets else None
            found[name] = {
                "output": name,
                "tier": met,
                "draws": draws,
                "norms": norms,
                "zero": zero,
                "wins": wins,
                "by_draw": by_draw,
            }
        return found

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
            counts = ", ".join(f"{norm} {verdict['wins'][norm]}" for norm in verdict["by_draw"])
            found.append(
                f"{name}: misses the naive tier over {verdict['draws']} inputs (draw by draw: "
                f"ours within {NEAR_TIE} x the tier on {counts} of {verdict['draws']} draws, "
                f"at least {-(-verdict['draws'] // 2)} needed; error against the oracle, "
                "median / largest): "
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
    """``TIER <test> <output> bf16|FAIL``, the number of draws, per norm
    judged draw by draw the draws within :data:`NEAR_TIE` of the tier
    (``paired max <n>/<draws> |``), and per norm the median and largest error
    of the output and of the tier."""
    norms = " | ".join(
        f"{norm} ours {_pair(errors['ours'])} "
        + " ".join(f"{tier} {_pair(errors[tier])}" for tier in TIERS)
        for norm, errors in verdict["norms"].items()
    )
    if by_draw := verdict.get("by_draw"):
        wins = " ".join(f"{norm} {verdict['wins'][norm]}/{verdict['draws']}" for norm in by_draw)
        norms = f"paired {wins} | {norms}"
    return (
        f"TIER {test_id} {verdict['output']} {verdict['tier'] or 'FAIL'} "
        f"draws={verdict['draws']} {norms}"
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
) -> tuple[TierJudge, list[str]]:
    """Judge one case over ``draws`` inputs.

    For each draw ``j``, :func:`seed_all` seeds the random state with
    :func:`draw_seed` of ``seed`` and ``j``; ``run(j, s)`` (``s`` that seed)
    makes the draw's inputs, launches the kernel once and returns
    ``(outputs, oracle)`` by name; ``naive(j, tier)`` returns the tier's value
    of every judged output on the same inputs.  Every output is held to its
    :func:`contract` (exact for the names in ``exact``), and each other output
    is recorded in the judge.  The Python, torch CPU and every CUDA device's
    random state are restored after the last draw.  Returns the judge and
    every failure: the contract's, then the judge's; record the verdicts with
    ``judge.record(record_property)``."""
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
