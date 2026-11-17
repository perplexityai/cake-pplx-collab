# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""Accuracy of the ragged MoE grouped GEMM's outputs: this family's use of :mod:`cake_naive`.

A GPU test judges every floating output it checks with :func:`judge_draws`, over
:data:`~cake_naive.DRAWS` independent inputs, against the naive BF16 tier (``cake_naive``):
in the L2 and in the max norm, the median and the largest of the output's error against
the float64 oracle over the draws are each at most :data:`~cake_naive.NEAR_TIE` times
the naive tier's, and the output's error is 0 on every draw where the tier's is.  There
is no tolerance.

* The oracle of an op (:func:`oracle`) is the float64 product of the BF16 operands of
  each group, unrounded: the forward ``Y[s:t] = X[s:t] @ W[e].T``, the data gradient
  ``dX[s:t] = G[s:t] @ W[e]`` and the weight gradient ``dW[e] = G[s:t].T @ X[s:t]``
  (zero for an empty group).  An accumulating weight gradient (``out[e] += dW[e]``)
  starts from ``start``, the output as stored before its first add: its oracle is
  ``start + dW`` (``adds`` times) in float64.
* The naive tier (:func:`naive`) is the same per-group product by torch's matmul of the
  BF16 operands (cuBLAS, its own FP32 accumulation, one rounding to BF16).  An
  accumulating weight gradient adds that product to ``start`` as stored (an FP32
  ``start`` in FP32, a BF16 one in BF16), ``adds`` times, and rounds the sum to BF16
  once.  ``cake_naive`` widens the tier's BF16 value exactly to an FP32 output.

Draw ``j`` of a case of seed ``seed`` draws its inputs from :func:`cake_naive.draw_seed`
``(seed, j)``; draw 0 is the case's own inputs.  :func:`judge_draws` runs draws 1, 2, ...
and draw 0 last, so the buffers and prepared launches a test refills hold the case's own
inputs and outputs when it returns, and the test's later checks (bits against another
route, rest states) run on them.  An output's contract (shape; finite where the oracle is
finite, its inf or a NaN where the oracle holds one) is checked on every draw; an exact
output (an empty group's zeros, a counter) is compared exactly with :meth:`Draw.exact`.

The float64 oracle runs on the device; GB300 multiplies float64 at about 1.1 TFLOPS.
Where an op's whole oracle takes more than :data:`FULL_ORACLE_MACS` multiply-adds (the
projection rows of the MoE benchmark), :meth:`Draw.op` judges the output's sampled rows
(:func:`sampled_rows`): the forward's and the data gradient's output rows, the weight
gradient's ``N`` rows of every group, each with all of its columns.  The sample takes
both edge rows of every group (of ``N``) and one row of every :data:`SAMPLE_STRIDE`
rows at an offset that moves from block to block, so every output tile has sampled rows.
The output is named ``<name>[sampled]`` then, and its whole tensor is still held to its
contract (shape, finite where the oracle is finite).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import torch

from .cake_naive import DRAWS, TIERS, TierJudge, contract, draw_seed

#: The output names of the ops, as the TIER lines print them.
OUTPUT_NAMES = {"fwd": "Y", "dgrad": "dX", "wgrad": "dW"}
#: Multiply-adds of an op's whole float64 oracle above which :meth:`Draw.op` judges sampled
#: rows (module docstring); ``CAKE_PPLX_MOE_FULL_ORACLE_MACS`` sets another bound.
FULL_ORACLE_MACS = int(os.environ.get("CAKE_PPLX_MOE_FULL_ORACLE_MACS", 1 << 37))
#: One sampled row per this many rows (:func:`sampled_rows`).
SAMPLE_STRIDE = 64
#: How far the sampled row moves from one block of SAMPLE_STRIDE rows to the next.
SAMPLE_SKEW = 37


def bounds(sizes: Sequence[int]) -> list[tuple[int, int]]:
    """``(start, end)`` rows of each group."""
    ends, total = [], 0
    for size in sizes:
        total += int(size)
        ends.append(total)
    return list(zip([0, *ends[:-1]], ends, strict=True))


def product(
    op: str, a: torch.Tensor, b: torch.Tensor, sizes: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    """The per-group product of ``op`` computed in ``dtype`` from the operands cast to it:
    ``fwd`` (``a`` = X ``[sum_m, K]``, ``b`` = W ``[E, N, K]``), ``dgrad`` (G ``[sum_m, N]``,
    W ``[E, N, K]``) or ``wgrad`` (G ``[sum_m, N]``, X ``[sum_m, K]``).  An empty group's
    weight gradient is zero."""
    if op == "fwd":
        out = torch.zeros(a.shape[0], b.shape[1], dtype=dtype, device=a.device)
        for e, (s, t) in enumerate(bounds(sizes)):
            if t > s:
                out[s:t] = a[s:t].to(dtype) @ b[e].to(dtype).T
    elif op == "dgrad":
        out = torch.zeros(a.shape[0], b.shape[2], dtype=dtype, device=a.device)
        for e, (s, t) in enumerate(bounds(sizes)):
            if t > s:
                out[s:t] = a[s:t].to(dtype) @ b[e].to(dtype)
    elif op == "wgrad":
        out = torch.zeros(len(sizes), a.shape[1], b.shape[1], dtype=dtype, device=a.device)
        for e, (s, t) in enumerate(bounds(sizes)):
            if t > s:
                out[e] = a[s:t].to(dtype).T @ b[s:t].to(dtype)
    else:
        raise ValueError(f"op must be fwd, dgrad or wgrad, got {op!r}")
    return out


def macs(op: str, a: torch.Tensor, b: torch.Tensor) -> int:
    """Multiply-adds of ``op``'s whole product: ``sum_m * N * K``."""
    if op == "fwd":
        return a.shape[0] * b.shape[1] * b.shape[2]
    if op == "dgrad":
        return a.shape[0] * b.shape[1] * b.shape[2]
    return a.shape[0] * a.shape[1] * b.shape[1]


def _strided(start: int, end: int) -> list[int]:
    """Both edge rows of ``[start, end)`` and one row of every SAMPLE_STRIDE rows, at an
    offset that moves by SAMPLE_SKEW from block to block."""
    rows = {start, end - 1} if end > start else set()
    for i, base in enumerate(range(start, end, SAMPLE_STRIDE)):
        row = base + (i * SAMPLE_SKEW) % SAMPLE_STRIDE
        if row < end:
            rows.add(row)
    return sorted(rows)


def sampled_rows(op: str, a: torch.Tensor, b: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
    """The sampled rows of ``op``'s output (module docstring), ascending: rows of
    ``[sum_m, *]`` for the forward and the data gradient (:func:`_strided` of each group),
    rows of ``N`` of each group's ``[N, K]`` for the weight gradient."""
    if op == "wgrad":
        rows = _strided(0, a.shape[1])
    else:
        rows = [row for s, t in bounds(sizes) for row in _strided(s, t)]
    return torch.tensor(rows, dtype=torch.long, device=a.device)


def _sample(op: str, a: torch.Tensor, b: torch.Tensor, sizes: Sequence[int], rows: torch.Tensor):
    """The operands and group sizes whose product is the sampled rows of ``op``'s output."""
    if op == "wgrad":
        return a[:, rows], b, list(sizes)
    counts = [int(((rows >= s) & (rows < t)).sum()) for s, t in bounds(sizes)]
    return a[rows], b, counts


def oracle(
    op: str,
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: Sequence[int],
    start: torch.Tensor | None = None,
    adds: int = 1,
) -> torch.Tensor:
    """The float64 oracle of ``op`` (module docstring); with ``start``, the accumulating
    weight gradient's ``start + dW``, ``adds`` times."""
    value = product(op, a, b, sizes, torch.float64)
    return value if start is None else oracle_sum(start, [value] * adds)


def naive(
    op: str,
    a: torch.Tensor,
    b: torch.Tensor,
    sizes: Sequence[int],
    start: torch.Tensor | None = None,
    adds: int = 1,
) -> torch.Tensor:
    """The naive BF16 tier of ``op`` (module docstring), in bfloat16; with ``start``, the
    accumulating weight gradient's add to ``start`` as stored, rounded to BF16 once."""
    value = product(op, a, b, sizes, torch.bfloat16)
    return value if start is None else naive_sum(start, [value] * adds)


def oracle_sum(start: torch.Tensor, values: Sequence[torch.Tensor]) -> torch.Tensor:
    """The float64 oracle of ``start`` (as stored) plus each float64 product of ``values``."""
    out = start.to(torch.float64, copy=True)
    for value in values:
        out += value
    return out


def naive_sum(start: torch.Tensor, values: Sequence[torch.Tensor]) -> torch.Tensor:
    """The naive tier of ``start`` plus each BF16 product of ``values`` in turn: the adds in
    ``start``'s dtype as stored (FP32 or BF16), the sum rounded to BF16 once."""
    out = start.to(torch.promote_types(start.dtype, torch.bfloat16), copy=True)
    for value in values:
        out += value.to(out.dtype)
    return out.to(torch.bfloat16)


class Draw:
    """One draw of :func:`judge_draws`: the outputs a test reports for it."""

    def __init__(self, judge: TierJudge, index: int, seed: int, failures: list[str]) -> None:
        self.judge = judge
        self.index = index
        self.seed = seed
        self.failures = failures
        # the last references :meth:`op` computed: (key, sampled rows, oracle, naive)
        self._kept: tuple | None = None

    def output(
        self, name: str, actual: torch.Tensor, expected: torch.Tensor, naive_value: torch.Tensor
    ) -> None:
        """Hold ``actual`` to its contract against the oracle ``expected`` and record its
        error beside the naive tier's ``naive_value`` (bfloat16).  A second report of
        ``name`` on this draw (another launch on the same inputs) keeps the larger of the
        output's errors."""
        if message := contract(name, actual, expected):
            self.failures.append(f"draw {self.index}: {message}")
            if actual.shape != expected.shape:
                return
        self.judge.add(name, self.index, actual, expected, {tier: naive_value for tier in TIERS})

    def op(
        self,
        name: str,
        actual: torch.Tensor,
        op: str,
        a: torch.Tensor,
        b: torch.Tensor,
        sizes: Sequence[int],
        start: torch.Tensor | None = None,
        adds: int = 1,
    ) -> None:
        """:meth:`output` with the oracle and the naive tier of ``op`` on ``a``, ``b``
        (``start``, ``adds``: an accumulating weight gradient).  Above
        :data:`FULL_ORACLE_MACS` the sampled rows are judged (module docstring).  Reports of
        several launches on the same operands reuse the references."""
        tensors = (a, b) if start is None else (a, b, start)
        key = (op, tuple(sizes), adds, tuple((id(t), t.data_ptr(), t._version) for t in tensors))
        if self._kept is None or self._kept[0] != key:
            rows = sampled_rows(op, a, b, sizes) if macs(op, a, b) > FULL_ORACLE_MACS else None
            if rows is None:
                sa, sb, ssizes, sstart = a, b, sizes, start
            else:
                sa, sb, ssizes = _sample(op, a, b, sizes, rows)
                sstart = None if start is None else start[:, rows]
            self._kept = (
                key,
                rows,
                oracle(op, sa, sb, ssizes, sstart, adds),
                naive(op, sa, sb, ssizes, sstart, adds),
            )
        _, rows, expected, naive_value = self._kept
        if rows is None:
            self.output(name, actual, expected, naive_value)
            return
        # the whole output's contract: its shape, and finite where the oracle is finite (the
        # BF16 operands are finite, so is their product; a start's inf or NaN stays)
        if start is None:
            if op == "fwd":
                shape = (a.shape[0], b.shape[1])
            elif op == "dgrad":
                shape = (a.shape[0], b.shape[2])
            else:
                shape = (len(sizes), a.shape[1], b.shape[1])
            bad = int((~actual.isfinite()).sum())
            if tuple(actual.shape) != shape or bad:
                self.failures.append(
                    f"draw {self.index}: {name}: shape {tuple(actual.shape)}, {bad} elements not finite"
                )
        elif message := contract(name, actual, torch.where(start.isfinite(), 0.0, start.to(torch.float64))):
            self.failures.append(f"draw {self.index}: {message}")
        part = actual[:, rows] if op == "wgrad" else actual[rows]
        self.output(f"{name}[sampled]", part, expected, naive_value)

    def exact(self, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        """Hold ``actual`` to ``expected`` exactly (``cake_naive.contract`` with ``exact``)."""
        if message := contract(name, actual, expected, exact=True):
            self.failures.append(f"draw {self.index}: {message}")


def judge_draws(
    record_property: Callable[[str, object], None] | None,
    draw: Callable[[Draw], None],
    *,
    seed: int,
    draws: int = DRAWS,
) -> TierJudge:
    """Judge the outputs ``draw`` reports over ``draws`` independent inputs and fail on any
    miss or broken contract.

    ``draw(d)`` makes the inputs of draw ``d.index`` from the seed ``d.seed``
    (``draw_seed(seed, d.index)``), launches, and reports each output with
    :meth:`Draw.output`, :meth:`Draw.op` or :meth:`Draw.exact`.  Draws ``1 .. draws - 1``
    run first and draw 0, the case's own inputs, last.  The verdicts are recorded with
    ``record_property`` (``TierJudge.record``) for the suite's TIER lines.  Returns the
    judge."""
    judge = TierJudge(draws)
    failures: list[str] = []
    for index in [*range(1, draws), 0]:
        draw(Draw(judge, index, draw_seed(seed, index), failures))
    if record_property is not None:
        judge.record(record_property)
    failures += judge.failures()
    assert not failures, "\n".join(failures)
    return judge
