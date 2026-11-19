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

The accuracy judge of the family's routed-expert clamped SwiGLU: the inputs of each case, the FP64 oracle and the
naive BF16 tier, for the judge of ``tests/cake/test_helpers/cake_naive.py``.

Per row ``r < R`` and column ``c < 2304``, from the BF16 gate ``g`` and up ``u``, the row's FP32 route weight ``w``
(1 for an unweighted call) and the FP32 limit ``L``::

    g' = min(g, L),  u' = clamp(u, -L, L),  s = sigmoid(g'),  a = g' s u',  h = bf16(w a)
    d_gate = bf16(dh w u' s (1 + g' (1 - s)) [g <= L]),  d_up = bf16(dh w g' s [-L <= u <= L])
    d_weight = sum over c of dh a   (FP32, the unweighted a)

:func:`make_inputs` draws one case from a seed.  ``layout``: ``"split"`` (``gate`` and ``up`` two contiguous
``[R, 2304]`` tensors), ``"merged"`` (one contiguous ``[R, 4608]`` tensor, gate columns ``[0, 2304)``, up
``[2304, 4608)``), ``"strided"`` (``gate``, ``up`` and ``dh`` column slices of wider buffers, read in place) or
``"copied"`` (row strides that are not multiples of 8 elements: the host copies them once).  ``w``: ``"random"``
(``3 x U[0, 1)``, every third row from row 1 exactly 0), ``"zeros"`` or ``"none"`` (an unweighted call).  ``data``
shapes the gate:
``"normal"`` (``6 x`` unit normals for ``g`` and ``u``: both clamps hold on about a tenth of the elements),
``"boundary"`` (every fourth column of ``g`` and of ``u`` from ``L``, ``-L`` and their BF16 neighbours),
``"deep_negative"`` (every gate in ``[-100, -80]``), ``"newton_edge"`` (every gate in ``[-86.7, -80]``),
``"sigmoid_edge"`` (every gate in ``[-89, -86.5]``: FP32 ``exp(-g)`` overflows from -88.72 and the FP32 sigmoid is
subnormal below -87.34), ``"mixed"`` (every third row ``"sigmoid_edge"``, the others ``"normal"``), ``"slow_mixed"``
(``"boundary"`` with a gate of -88 at columns 0, 768 and 1536 of every row: one in each third of the row, beside
ordinary gates, gates at, above and next to ``L``, and ups at and beyond ``+-L``) or ``"all_gates"`` (every finite
BF16 bit pattern as a gate, in order, 0 for the others).  ``dh`` is a unit normal.  Every value is finite.

The oracle (:func:`oracle`) evaluates each step in float64 from the stored inputs and the FP32 value of ``L``, the
inclusive masks as factors, and returns every output unrounded.  The naive tier (:func:`naive`) is the same graph in
PyTorch BF16: the clamps written as selects (``torch.where``), so torch.autograd gives the inclusive masks whatever
torch's tie rule of ``clamp``; ``silu`` and the product on the BF16 tensors; the FP32 weight kept FP32 (a weight of
ones for an unweighted call), as in BF16 user code that reads it, and the weighted product stored to BF16;
torch.autograd through it gives the gradients, ``d_weight`` returned as BF16 (the tier's dtype).
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F

from tests.cake.test_helpers.cake_naive import TIER_DTYPES, TIERS, judge_case

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
WIDTH = 2304
SCALE = 6.0
LAYOUTS = ("split", "merged", "strided", "copied")
WEIGHTS = ("random", "zeros", "none")
DATA = ("normal", "boundary", "deep_negative", "newton_edge", "sigmoid_edge", "mixed", "slow_mixed", "all_gates")
# The gate of ``"slow_mixed"`` (exact in BF16) and its columns, the first column of each third of the row.
SLOW_GATE = -88.0
SLOW_COLUMNS = (0, WIDTH // 3, 2 * WIDTH // 3)
# Rows of one float64 block of the oracle.
ORACLE_ROWS = 4096


class Case(NamedTuple):
    """One judged case: ``R`` rows, a layout, a weight kind, a data kind and a limit."""

    R: int
    layout: str = "split"
    w: str = "random"
    data: str = "normal"
    L: float = 10.0

    @property
    def id(self) -> str:
        parts = [f"R{self.R}", self.layout, f"w_{self.w}", self.data]
        if self.L != 10.0:
            parts.append(f"L{self.L:g}")
        return "-".join(parts)


def fp32(value: float) -> float:
    """``value`` rounded to FP32, as a Python float (the kernels' ``L``)."""
    return float(torch.tensor(value, dtype=torch.float32))


def bf16_edges(limit: float, device: torch.device) -> torch.Tensor:
    """``L`` and ``-L`` as BF16 and the BF16 values on each side of them, as FP32."""
    bits = torch.tensor([limit, -limit], device=device).bfloat16().view(torch.int16)
    return torch.stack([bits - 1, bits, bits + 1]).view(torch.bfloat16).flatten().float()


def _uniform(R: int, low: float, high: float, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    return low + (high - low) * torch.rand(R, WIDTH, device=device, generator=gen)


def _gates(data: str, R: int, limit: float, gen: torch.Generator, device: torch.device):
    g = SCALE * torch.randn(R, WIDTH, device=device, generator=gen)
    u = SCALE * torch.randn(R, WIDTH, device=device, generator=gen)
    if data in ("boundary", "slow_mixed"):
        edges = bf16_edges(limit, device)
        g[:, 0::4] = edges[torch.randint(0, len(edges), (R, WIDTH // 4), device=device, generator=gen)]
        u[:, 0::4] = edges[torch.randint(0, len(edges), (R, WIDTH // 4), device=device, generator=gen)]
        if data == "slow_mixed":
            g[:, list(SLOW_COLUMNS)] = SLOW_GATE
    elif data == "deep_negative":
        g = _uniform(R, -100.0, -80.0, gen, device)
    elif data == "newton_edge":
        g = _uniform(R, -86.7, -80.0, gen, device)
    elif data == "sigmoid_edge":
        g = _uniform(R, -89.0, -86.5, gen, device)
    elif data == "mixed":
        g[0::3] = _uniform(R, -89.0, -86.5, gen, device)[0::3]
    elif data == "all_gates":
        bits = (torch.arange(R * WIDTH, device=device, dtype=torch.int32) % 65536).to(torch.int16)
        g = bits.view(torch.bfloat16).float().reshape(R, WIDTH)
        g = torch.where(g.isfinite(), g, 0.0)
    elif data != "normal":
        raise ValueError(f"data is one of {DATA}; got {data!r}")
    return g, u


def _place(values: torch.Tensor, *, pad: int) -> torch.Tensor:
    """``values`` (``[R, C]`` BF16) at column ``pad`` of a wider buffer whose row stride is ``C + 2 * pad``."""
    R, C = values.shape
    buffer = torch.full((R, C + 2 * pad), float("nan"), dtype=values.dtype, device=values.device)
    buffer[:, pad : pad + C] = values
    return buffer[:, pad : pad + C]


def make_inputs(case: Case, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """The inputs of ``case`` drawn from ``seed``: ``gate``, ``up`` (views of ``gate_up`` for a merged layout),
    ``weight`` (None for an unweighted case), ``dh`` and the FP32 limit ``L``; ``first`` and ``second`` are the
    entries' first two arguments (``gate_up, None`` or ``gate, up``)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    limit = fp32(case.L)
    g, u = _gates(case.data, case.R, limit, gen, device)
    t = types.SimpleNamespace(case=case, R=case.R, L=limit, layout=case.layout)
    if case.layout == "merged":
        t.gate_up = torch.cat([g, u], dim=1).bfloat16()
        t.gate, t.up = t.gate_up[:, :WIDTH], t.gate_up[:, WIDTH:]
        t.first, t.second = t.gate_up, None
    elif case.layout in ("split", "strided", "copied"):
        gate, up = g.bfloat16(), u.bfloat16()
        if case.layout == "strided":
            # row strides 2304 + 64 and 2304 + 16 elements, bases 64 and 16 bytes past an aligned allocation
            gate, up = _place(gate, pad=32), _place(up, pad=8)
        elif case.layout == "copied":
            # row strides 2304 + 6 and 2304 + 2: not multiples of 8 elements, copied once by the host
            gate, up = _place(gate, pad=3), _place(up, pad=1)
        t.gate, t.up = gate, up
        t.first, t.second = gate, up
    else:
        raise ValueError(f"layout is one of {LAYOUTS}; got {case.layout!r}")
    if case.w == "random":
        t.weight = 3.0 * torch.rand(case.R, device=device, generator=gen)
        t.weight[1::3] = 0.0
    elif case.w == "zeros":
        t.weight = torch.zeros(case.R, device=device)
    elif case.w == "none":
        t.weight = None
    else:
        raise ValueError(f"w is one of {WEIGHTS}; got {case.w!r}")
    dh = torch.randn(case.R, WIDTH, device=device, generator=gen).bfloat16()
    if case.layout == "strided":
        t.dh = _place(dh, pad=16)  # row stride 2304 + 32, base 32 bytes past an aligned allocation
    elif case.layout == "copied":
        t.dh = _place(dh, pad=5)  # row stride 2304 + 10, base 10 bytes past an aligned allocation
    else:
        t.dh = dh
    return t


def oracle(t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The float64 ``h``, ``d_gate``, ``d_up`` and (weighted) ``d_weight`` of the inputs ``t``, in blocks of rows."""
    R, L = t.R, t.L
    device = t.gate.device
    out = {
        "h": torch.empty(R, WIDTH, dtype=torch.float64, device=device),
        "d_gate": torch.empty(R, WIDTH, dtype=torch.float64, device=device),
        "d_up": torch.empty(R, WIDTH, dtype=torch.float64, device=device),
    }
    if t.weight is not None:
        out["d_weight"] = torch.empty(R, dtype=torch.float64, device=device)
    for i in range(0, R, ORACLE_ROWS):
        rows = slice(i, min(R, i + ORACLE_ROWS))
        g, u, d = t.gate[rows].double(), t.up[rows].double(), t.dh[rows].double()
        gc, uc = g.clamp(max=L), u.clamp(-L, L)
        s = torch.sigmoid(gc)
        silu = gc * s
        a = silu * uc
        w = 1.0 if t.weight is None else t.weight[rows].double()[:, None]
        out["h"][rows] = w * a
        out["d_gate"][rows] = d * w * uc * (s * (1.0 + gc * (1.0 - s))) * (g <= L).double()
        out["d_up"][rows] = d * w * silu * ((u >= -L) & (u <= L)).double()
        if t.weight is not None:
            out["d_weight"][rows] = (d * a).sum(dim=1)
    return out


def naive(t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of every judged output on the inputs ``t`` (see the module)."""
    L = t.L
    with torch.enable_grad():
        g = t.gate.detach().clone().requires_grad_(True)
        u = t.up.detach().clone().requires_grad_(True)
        ones = torch.ones(t.R, dtype=torch.float32, device=t.gate.device)
        w = (ones if t.weight is None else t.weight).detach().clone().requires_grad_(True)
        gc = torch.where(g <= L, g, L)
        uc = torch.where(u > L, L, torch.where(u < -L, -L, u))
        a = F.silu(gc) * uc
        h = (w[:, None] * a).to(NAIVE_DTYPE)
        h.backward(t.dh)
    found = {"h": h.detach(), "d_gate": g.grad, "d_up": u.grad}
    if t.weight is not None:
        found["d_weight"] = w.grad.to(NAIVE_DTYPE)
    return found


def outputs(fwd: Callable, bwd: Callable, t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """``h``, ``d_gate``, ``d_up`` and (weighted) ``d_weight`` of the forward ``fwd`` and backward ``bwd`` steps on
    the inputs ``t``; a merged layout's gradient is split into its halves."""
    h = fwd(t.first, t.second, t.weight, limit=t.L)
    d_first, d_up, d_weight = bwd(t.dh, t.first, t.second, t.weight, limit=t.L)
    if t.second is None:
        assert d_up is None and tuple(d_first.shape) == (t.R, 2 * WIDTH)
        d_gate, d_up = d_first[:, :WIDTH], d_first[:, WIDTH:]
    else:
        d_gate = d_first
    found = {"h": h, "d_gate": d_gate, "d_up": d_up}
    if t.weight is None:
        assert d_weight is None
    else:
        found["d_weight"] = d_weight
    return found


def judge(
    record_property: Callable,
    case: Case,
    fwd: Callable,
    bwd: Callable,
    *,
    device: torch.device,
    seed: int = 0,
) -> None:
    """Judge ``case`` over the draws of :func:`judge_case`: each draw's inputs from its seed, the forward and the
    backward launched once, every output held to its contract and to the naive tier.  Every verdict is recorded
    (``TIER`` lines), and any failure fails the test with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        t = make_inputs(case, seed=s, device=device)
        found = outputs(fwd, bwd, t)
        held.clear()
        held[j] = naive(t)
        return found, oracle(t)

    found_judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed)
    found_judge.record(record_property)
    assert not failures, "\n".join(failures)
