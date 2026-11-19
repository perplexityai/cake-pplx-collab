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

The accuracy judge of the family's ratio-2 gated KV compression (width 512): the FP64 oracle, the naive BF16 tier and
the inputs of each judged case, for the judge of ``tests/cake_naive.py``; also the packs, the host-side pair rows and
the bitwise helpers the other suites share.

Operands: ``C`` and ``Z`` FP32 ``[T, 512]`` (two contiguous tensors, or the column halves of one ``[T, 1024]``
tensor), ``cu_seqlens`` int32 ``[S + 1]``, ``dout`` BF16 ``[G, 512]``; outputs ``out`` BF16 ``[G, 512]``, ``dC`` and
``dZ`` FP32 ``[T, 512]``.  Pair ``k`` of document ``s`` reads ``t0 = cu_seqlens[s] + 2k`` and ``t1 = t0 + 1``.

:func:`make_inputs` draws one ``[T, 1024]`` FP32 ``randn`` tensor (``C | Z``) and a BF16 ``randn`` ``dout`` from a CPU
generator seeded with the draw's seed (every device sees the same inputs), optionally with close pairs
(``C[t1] = C[t0] * (1 + 2^-k)`` in FP32) or gate extremes on fixed channel blocks of every pair (logit gaps of 100,
-100, 2000 and -2000, exact ties, and ties at a large logit).

Each judged output is one function evaluated in one dtype: float64 is the oracle (the reference of the feature
request, ``forward_unrounded`` / ``backward_pairwise``, on the kernel's own FP32 ``C`` and ``Z`` and BF16 ``dout``;
``out`` is compared unrounded), bfloat16 the naive tier (PyTorch, the same ops on the stored inputs, :func:`stored`:
FP32 ``C`` and ``Z`` stay FP32 and BF16 ``dout`` stays BF16, so torch's type promotion computes in FP32; each result is
cast to BF16, the tier's dtype).

The pairs come from the Python list of document lengths of the case (:func:`pair_starts`), never from a device read of
``cu_seqlens``: every test knows its pack.
"""

from __future__ import annotations

import math
import random
import types
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple, Optional

import torch

from tests.cake_naive import TIER_DTYPES, TIERS, judge_case

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
ORACLE_DTYPE = torch.float64
WIDTH = 512
LAYOUTS = ("contiguous", "halves")
# The integer dtype of each floating dtype's bits (bitwise comparisons compare these).
BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float64: torch.int64}


class Pack(NamedTuple):
    """One pack of documents: their lengths, in order (a length may be 0)."""

    lengths: tuple[int, ...]
    label: Optional[str] = None

    @property
    def T(self) -> int:
        return sum(self.lengths)

    @property
    def S(self) -> int:
        return len(self.lengths)

    @property
    def G(self) -> int:
        return sum(n // 2 for n in self.lengths)

    @property
    def id(self) -> str:
        if self.label is not None:
            return self.label
        if len(self.lengths) <= 8:
            return "L" + "-".join(str(n) for n in self.lengths) if self.lengths else "L"
        return f"T{self.T}-S{self.S}"


def lognormal_lengths(T: int, S: int, seed: int) -> tuple[int, ...]:
    """``S`` document lengths, each at least 1, summing to ``T``: weights from a seeded log-normal(0, 1) draw
    (Python's ``random``), the floor of each share, the remainder added to the longest document."""
    rng = random.Random(seed)
    weights = [rng.lognormvariate(0.0, 1.0) for _ in range(S)]
    total = sum(weights)
    lengths = [max(1, math.floor(w / total * T)) for w in weights]
    lengths[lengths.index(max(lengths))] += T - sum(lengths)
    if min(lengths) < 1 or sum(lengths) != T:
        raise ValueError(f"cannot pack T = {T} into S = {S} documents of at least one token")
    return tuple(lengths)


# The packs of the performance rows: T = 16384 in 27 documents and T = 65536 in 68 documents.
PERF_PACKS = (
    Pack(lognormal_lengths(16384, 27, 16384), "T16384-S27"),
    Pack(lognormal_lengths(65536, 68, 65536), "T65536-S68"),
)


def bounds(lengths: Sequence[int]) -> list[int]:
    """``[0, b_1, ..., b_S]``: the prefix sums of ``lengths``."""
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


def cu_seqlens_of(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """Int32 ``[S + 1]`` document bounds on ``device``."""
    return torch.tensor(bounds(lengths), dtype=torch.int32, device=device)


def pair_table_of(lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """Int32 ``[S + 1]``: the exclusive prefix of the pair counts ``floor(length / 2)``."""
    return torch.tensor(bounds([n // 2 for n in lengths]), dtype=torch.int32, device=device)


def pair_starts(lengths: Sequence[int]) -> list[int]:
    """``t0`` of every pair, in output-row order."""
    b = bounds(lengths)
    return [b[s] + 2 * k for s, n in enumerate(lengths) for k in range(n // 2)]


def tail_rows(lengths: Sequence[int]) -> list[int]:
    """The token row of every odd-length document's last token."""
    b = bounds(lengths)
    return [b[s + 1] - 1 for s, n in enumerate(lengths) if n % 2]


def stored(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x`` as a computation in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is (an
    FP32 tensor stays FP32 in the BF16 tier and is widened in the float64 oracle)."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def _gate_extremes(z: torch.Tensor, t0: torch.Tensor) -> None:
    """Fixed channel blocks of every pair (in place, FP32): logit gaps of -100 and +100 around a random logit, of
    -2000 and +2000, exact ties at a random logit and ties at 100 (where ``exp`` of the logit itself overflows FP32)."""
    t1 = t0 + 1
    z0 = z[t0]
    z1 = z[t1]
    z1[:, 0:64] = z0[:, 0:64] - 100.0
    z1[:, 64:128] = z0[:, 64:128] + 100.0
    z0[:, 128:192], z1[:, 128:192] = 1000.0, -1000.0
    z0[:, 192:256], z1[:, 192:256] = -1000.0, 1000.0
    z1[:, 256:320] = z0[:, 256:320]
    z0[:, 320:384], z1[:, 320:384] = 100.0, 100.0
    z[t0], z[t1] = z0, z1


def make_inputs(
    pack: Pack,
    *,
    seed: int,
    device: torch.device,
    layout: str = "contiguous",
    close: Optional[int] = None,
    gates: bool = False,
    rows: Optional[int] = None,
) -> types.SimpleNamespace:
    """The operands of one call on ``pack`` (module docstring).  ``layout`` ``"halves"`` makes ``c`` and ``z`` the
    column halves of ``cz [T, 1024]``; ``"contiguous"`` makes them two contiguous copies.  ``close``: every pair has
    ``C[t1] = C[t0] * (1 + 2^-close)``.  ``gates``: the gate extremes of :func:`_gate_extremes`.  ``rows``: the call's
    ``G`` (default the pack's pair count; ``dout`` has that many rows)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    generator = torch.Generator().manual_seed(seed)
    T = pack.T
    G = pack.G if rows is None else int(rows)
    cz = torch.randn(T, 2 * WIDTH, generator=generator, dtype=torch.float32)
    t0 = torch.tensor(pair_starts(pack.lengths), dtype=torch.long)
    if t0.numel():
        if close is not None:
            c = cz[:, :WIDTH]
            c[t0 + 1] = c[t0] * (1.0 + 2.0**-close)
        if gates:
            _gate_extremes(cz[:, WIDTH:], t0)
    dout = torch.randn(G, WIDTH, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    cz = cz.to(device)
    if layout == "halves":
        c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    else:
        c, z = cz[:, :WIDTH].contiguous(), cz[:, WIDTH:].contiguous()
    return types.SimpleNamespace(
        pack=pack,
        lengths=pack.lengths,
        T=T,
        S=pack.S,
        G=G,
        cz=cz,
        c=c,
        z=z,
        cu=cu_seqlens_of(pack.lengths, device),
        dout=dout.to(device),
        t0=t0.to(device),
        layout=layout,
    )


def nan_like_rows(rows: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """A ``[rows, 512]`` tensor of NaN (every element the kernel must write is poisoned)."""
    return torch.full((rows, WIDTH), math.nan, dtype=dtype, device=device)


def nan_halves(rows: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(buffer, left, right)``: a NaN FP32 ``[rows, 1024]`` buffer and its two column halves."""
    buffer = torch.full((rows, 2 * WIDTH), math.nan, dtype=torch.float32, device=device)
    return buffer, buffer[:, :WIDTH], buffer[:, WIDTH:]


# ----------------------------------------------------------------------------------------------------------------
# Oracle and tier


def _probabilities(c: torch.Tensor, z: torch.Tensor, t0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(p0, p1) = softmax(Z[t0], Z[t1])`` per pair and channel, as the reference spells it."""
    z0, z1 = z[t0], z[t0 + 1]
    m = torch.maximum(z0, z1)
    e0, e1 = torch.exp(z0 - m), torch.exp(z1 - m)
    denom = e0 + e1
    return e0 / denom, e1 / denom


def forward_values(x: types.SimpleNamespace, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """``out`` of the forward of ``x`` in ``dtype`` (float64: the oracle, unrounded; bfloat16: the naive tier): the
    first ``G`` pair rows, and zero rows past the pair count when ``G`` is larger."""
    c, z = stored(x.c, dtype), stored(x.z, dtype)
    t0 = x.t0
    pairs = int(t0.numel())
    if pairs:
        p0, p1 = _probabilities(c, z, t0)
        y = p0 * c[t0] + p1 * c[t0 + 1]
    else:
        y = torch.zeros(0, WIDTH, dtype=c.dtype, device=c.device)
    out = torch.zeros(x.G, WIDTH, dtype=y.dtype, device=c.device)
    n = min(pairs, x.G)
    out[:n] = y[:n]
    return {"out": out.to(dtype)}


def backward_values(x: types.SimpleNamespace, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """``dc`` and ``dz`` of the pairwise backward of ``x`` in ``dtype``: ``dC[t0] = p0 g``, ``dC[t1] = p1 g``,
    ``dZ[t0] = p0 p1 (C[t0] - C[t1]) g``, ``dZ[t1] = -dZ[t0]`` with ``g = dout[r]``; zero elsewhere (the tail rows).
    ``G`` is at least the pair count."""
    c, z = stored(x.c, dtype), stored(x.z, dtype)
    t0 = x.t0
    pairs = int(t0.numel())
    if x.G < pairs:
        raise ValueError(f"the backward needs G >= the pair count, got G = {x.G} for {pairs} pairs")
    if not pairs:
        zeros = torch.zeros(x.T, WIDTH, dtype=dtype, device=c.device)
        return {"dc": zeros, "dz": zeros.clone()}
    g = stored(x.dout[:pairs], dtype)
    p0, p1 = _probabilities(c, z, t0)
    dz0 = p0 * p1 * (c[t0] - c[t0 + 1]) * g
    dc = torch.zeros(x.T, WIDTH, dtype=dz0.dtype, device=c.device)
    dz = torch.zeros_like(dc)
    dc[t0], dc[t0 + 1] = p0 * g, p1 * g
    dz[t0], dz[t0 + 1] = dz0, -dz0
    return {"dc": dc.to(dtype), "dz": dz.to(dtype)}


def oracle_values(x: types.SimpleNamespace, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """The float64 oracle of the named outputs (``out``, ``dc``, ``dz``)."""
    found = {}
    if "out" in names:
        found.update(forward_values(x, ORACLE_DTYPE))
    if "dc" in names or "dz" in names:
        found.update(backward_values(x, ORACLE_DTYPE))
    return {name: found[name] for name in names}


def naive_values(x: types.SimpleNamespace, names: Sequence[str]) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of the named outputs."""
    found = {}
    if "out" in names:
        found.update(forward_values(x, NAIVE_DTYPE))
    if "dc" in names or "dz" in names:
        found.update(backward_values(x, NAIVE_DTYPE))
    return {name: found[name] for name in names}


# ----------------------------------------------------------------------------------------------------------------
# Contracts, bitwise checks and the judge


def bits(t: torch.Tensor) -> torch.Tensor:
    """The bits of a floating tensor (an integer view of the same elements): ``+0.0`` and ``-0.0`` differ, a NaN
    equals the same NaN."""
    return t.view(BITS[t.dtype])


def assert_bitwise(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """``got`` and ``want`` have the same shape, dtype and bits."""
    assert got.shape == want.shape and got.dtype == want.dtype, (what, tuple(got.shape), tuple(want.shape))
    diff = bits(got) != bits(want)
    count = int(diff.sum())
    if count:
        first = tuple(int(i) for i in diff.nonzero()[0])
        raise AssertionError(
            f"{what}: {count} of {got.numel()} elements differ bit for bit; first at {first}: "
            f"{got[first].item()} vs {want[first].item()}"
        )


def assert_zero_rows(t: torch.Tensor, rows: Sequence[int], what: str) -> None:
    """Every element of the rows ``rows`` of ``t`` is +0.0 (all bytes zero)."""
    if not rows:
        return
    index = torch.tensor(list(rows), dtype=torch.long, device=t.device)
    nonzero = int((bits(t[index]) != 0).sum())
    assert nonzero == 0, f"{what}: {nonzero} elements of the tail rows {list(rows)[:8]} are not +0.0"


def assert_forward_contract(label: str, x: types.SimpleNamespace, out: torch.Tensor) -> None:
    """BF16 ``[G, 512]`` on the operands' device, contiguous, finite."""
    found = (tuple(out.shape), out.dtype, out.device)
    assert found == ((x.G, WIDTH), torch.bfloat16, x.c.device), (label, found)
    assert out.is_contiguous(), (label, tuple(out.stride()))
    assert bool(torch.isfinite(out).all()), f"{label}: non-finite out"


def assert_backward_contract(label: str, x: types.SimpleNamespace, dc: torch.Tensor, dz: torch.Tensor) -> None:
    """FP32 ``[T, 512]`` on the operands' device, finite, the tail rows exactly +0.0."""
    for name, value in (("dc", dc), ("dz", dz)):
        found = (tuple(value.shape), value.dtype, value.device)
        assert found == ((x.T, WIDTH), torch.float32, x.c.device), (label, name, found)
        assert bool(torch.isfinite(value).all()), f"{label}: non-finite {name}"
        assert_zero_rows(value, tail_rows(x.lengths), f"{label}: {name}")


def judged(x: types.SimpleNamespace, runs: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw for :func:`judge_draws`: the calls ``runs`` (label -> outputs by name,
    any of ``out``, ``dc``, ``dz``) on ``x`` beside the oracle and the naive tier of ``x``.  With several calls each
    output is named ``"<label> <output>"``."""
    names = sorted({name for got in runs.values() for name in got})
    ref = oracle_values(x, names)
    low = naive_values(x, names)
    found, oracle, naive = {}, {}, {}
    for label, got in runs.items():
        prefix = f"{label} " if len(runs) > 1 else ""
        for name, value in got.items():
            found[prefix + name], oracle[prefix + name], naive[prefix + name] = value, ref[name], low[name]
    return found, oracle, naive


def judge_draws(record_property: Callable, case: Callable[[int], tuple[dict, dict, dict]], *, seed: int = 0) -> None:
    """Judge one case with :func:`judge_case` over its draws: ``case(s)`` makes the draw's inputs from the seed ``s``,
    launches the package and returns the ``(outputs, oracle, naive)`` of :func:`judged`.  Every verdict is recorded
    (``TIER`` lines), and any contract or tier failure fails the test with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        found, oracle, naive = case(s)
        held.clear()
        held[j] = naive
        return found, oracle

    judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed)
    judge.record(record_property)
    assert not failures, "\n".join(failures)


# ----------------------------------------------------------------------------------------------------------------
# Measures of the feature request's acceptance criteria


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per element, the number of BF16 values from ``a`` to ``b`` (both BF16): 0 when equal (``+0.0`` and ``-0.0``
    included), 1 for neighbours."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        i = x.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    """``||got - want|| / ||want||`` in float64 (the reference's measure; the denominator at least 1e-300)."""
    want = want.double()
    return float((got.double() - want).norm() / want.norm().clamp_min(1e-300))
