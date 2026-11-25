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

The accuracy judge of the family's SiTU-GLU: the inputs of each case, the FP64 oracle and the naive BF16 tier, for
the judge of ``tests/cake/test_helpers/cake_naive.py``.

Per row ``r < R`` and column ``c < I``, from the BF16 gate ``g`` and up ``u`` and the FP32 scalars ``beta`` and
``lb``::

    s = sigmoid(g),  tg = tanh(g / beta),  tu = tanh(u / lb),  a = beta tg s,  u' = lb tu,  h = bf16(a u')
    d_gate = bf16(dh u' ((1 - tg^2) s + a (1 - s))),  d_up = bf16(dh a (1 - tu^2))

:func:`make_inputs` draws one case from a seed.  ``layout``: ``"split"`` (``gate`` and ``up`` two contiguous
``[R, I]`` tensors), ``"merged"`` (one contiguous ``[R, 2I]`` tensor, gate columns ``[0, I)``, up ``[I, 2I)``),
``"sliced"`` (``gate``, ``up`` and ``dh`` row slices of wider buffers, read in place) or ``"copied"`` (row strides
that are not multiples of 8 elements: the host copies them once).  ``data`` shapes the inputs:

- ``"normal"``: ``2 x`` unit normals for ``g`` and ``u``, the scale of the issue's yardstick;
- ``"wide"``: ``6 x`` and ``40 x`` unit normals for ``g`` and ``u``;
- ``"saturated"``: ``g`` uniform in ``[-200, 200]`` and ``u`` in ``[-1000, 1000]``;
- ``"deep_negative"``: every gate in ``[-100, -80]``, where the sigmoid is subnormal or under FP32;
- ``"small"``: ``g`` and ``u`` uniform in ``[-0.5, 0.5]``, where ``1 - E`` and ``1 - W`` are small;
- ``"tiny_gate"`` and ``"tiny_up"``: every gate ``|g|`` (or up ``|u|``) in ``beta 2^-band [1, 2)`` (``lb 2^-band
  [1, 2)``) with a random sign, the other input normal.  ``1 - E`` (``1 - W``) is then in ``2^(1 - band) [1, 2)`` for
  every element, so the L2 and max norms of one band weigh each element by its relative error;
- ``"zero_gate"`` and ``"zero_up"``: a random half of the gates, or of the ups, is 0, so ``h`` is 0 there;
- ``"all_gates"``: every finite BF16 bit pattern as a gate, in order, and 0 for the others.

``dh`` is a unit normal.  Every value is finite.

The oracle (:func:`oracle`) evaluates the operation in float64 from the stored inputs and the FP32 values of ``beta``
and ``lb``.  It takes ``1 - tg^2`` as ``1 / cosh(g / beta)^2`` and ``1 - s`` as ``sigmoid(-g)``, so a saturated
gate or up keeps its float64 gradient, where ``1 - tg tg`` would cancel to 0.  It returns every output unrounded.
The naive tier (:func:`naive`) is the same formula in PyTorch BF16 on the BF16 tensors, and torch.autograd through it
gives the gradients.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import NamedTuple

import torch

from tests.cake.test_helpers.cake_naive import TIER_DTYPES, TIERS, judge_case

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
# The issue's widths.
WIDTHS = (3072, 6144, 33792)
LAYOUTS = ("split", "merged", "sliced", "copied")
DATA = (
    "normal",
    "wide",
    "saturated",
    "deep_negative",
    "small",
    "tiny_gate",
    "tiny_up",
    "zero_gate",
    "zero_up",
    "all_gates",
)
# The model's scalars, the defaults of every entry point.
BETA = 4.0
LB = 25.0
# Elements of one float64 block of the oracle (whole rows).
ORACLE_ELEMENTS = 1 << 24
# Column pads of the wider buffers of ``"sliced"`` (multiples of 8: aligned bases and row strides) and ``"copied"``
# (not multiples of 8), for the gate, the up and ``dh``.
PADS = {"sliced": (32, 8, 16), "copied": (3, 1, 5)}


class Case(NamedTuple):
    """One judged case: ``R`` rows of width ``I``, a layout, a data kind, the scalars and the outputs it judges."""

    R: int
    I: int = 3072  # noqa: E741 - the issue's name of the width
    layout: str = "split"
    data: str = "normal"
    beta: float = BETA
    lb: float = LB
    band: int = 20
    judged: tuple[str, ...] = ("h", "d_gate", "d_up")

    @property
    def id(self) -> str:
        parts = [f"R{self.R}", f"I{self.I}", self.layout, self.data]
        if self.data in ("tiny_gate", "tiny_up"):
            parts.append(f"band{self.band}")
        if self.beta != BETA or self.lb != LB:
            parts.append(f"beta{self.beta:g}-lb{self.lb:g}")
        if self.judged != ("h", "d_gate", "d_up"):
            parts.append("+".join(self.judged))
        return "-".join(parts)


def fp32(value: float) -> float:
    """``value`` rounded to FP32, as a Python float (the kernels' ``beta`` and ``lb``)."""
    return float(torch.tensor(value, dtype=torch.float32))


def _uniform(rows: int, width: int, low: float, high: float, gen: torch.Generator, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(rows, width, device=device, generator=gen)


def draw_values(
    data: str,
    rows: int,
    width: int,
    gen: torch.Generator,
    device,
    *,
    beta: float = BETA,
    lb: float = LB,
    band: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The FP32 gate and up values of ``data`` (rounded to BF16 by the caller)."""
    g = 2.0 * torch.randn(rows, width, device=device, generator=gen)
    u = 2.0 * torch.randn(rows, width, device=device, generator=gen)
    if data == "wide":
        g, u = 3.0 * g, 20.0 * u
    elif data == "saturated":
        g, u = _uniform(rows, width, -200.0, 200.0, gen, device), _uniform(rows, width, -1000.0, 1000.0, gen, device)
    elif data == "deep_negative":
        g = _uniform(rows, width, -100.0, -80.0, gen, device)
    elif data == "small":
        g, u = _uniform(rows, width, -0.5, 0.5, gen, device), _uniform(rows, width, -0.5, 0.5, gen, device)
    elif data in ("tiny_gate", "tiny_up"):
        tiny = (beta if data == "tiny_gate" else lb) * 2.0**-band * _uniform(rows, width, 1.0, 2.0, gen, device)
        tiny = torch.where(torch.rand(rows, width, device=device, generator=gen) < 0.5, -tiny, tiny)
        g, u = (tiny, u) if data == "tiny_gate" else (g, tiny)
    elif data == "zero_gate":
        g = torch.where(torch.rand(rows, width, device=device, generator=gen) < 0.5, 0.0, g)
    elif data == "zero_up":
        u = torch.where(torch.rand(rows, width, device=device, generator=gen) < 0.5, 0.0, u)
    elif data == "all_gates":
        bits = (torch.arange(rows * width, device=device, dtype=torch.int32) % 65536).to(torch.int16)
        g = bits.view(torch.bfloat16).float().reshape(rows, width)
        g = torch.where(g.isfinite(), g, 0.0)
    elif data != "normal":
        raise ValueError(f"data is one of {DATA}; got {data!r}")
    return g, u


def place(values: torch.Tensor, pad: int) -> torch.Tensor:
    """``values`` (``[R, C]``) at column ``pad`` of a wider buffer whose row stride is ``C + 2 pad``, the rest NaN."""
    rows, cols = values.shape
    buffer = torch.full((rows, cols + 2 * pad), float("nan"), dtype=values.dtype, device=values.device)
    buffer[:, pad : pad + cols] = values
    return buffer[:, pad : pad + cols]


def arrange(gate: torch.Tensor, up: torch.Tensor, dh: torch.Tensor, layout: str, **fields) -> types.SimpleNamespace:
    """The BF16 values ``gate``, ``up`` and ``dh`` in ``layout``: ``gate``, ``up`` and ``dh`` as the kernels see them,
    ``first`` and ``second`` the entries' first two arguments (``gate_up, None`` or ``gate, up``), and ``fields``."""
    t = types.SimpleNamespace(R=int(gate.shape[0]), I=int(gate.shape[1]), layout=layout, **fields)
    gate, up, dh = gate.contiguous(), up.contiguous(), dh.contiguous()
    if layout == "merged":
        t.gate_up = torch.cat([gate, up], dim=1)
        t.gate, t.up = t.gate_up[:, : t.I], t.gate_up[:, t.I :]
        t.first, t.second, t.dh = t.gate_up, None, dh
    elif layout == "split":
        t.gate, t.up, t.dh = gate, up, dh
        t.first, t.second = gate, up
    elif layout in PADS:
        pg, pu, pd = PADS[layout]
        t.gate, t.up, t.dh = place(gate, pg), place(up, pu), place(dh, pd)
        t.first, t.second = t.gate, t.up
    else:
        raise ValueError(f"layout is one of {LAYOUTS}; got {layout!r}")
    return t


def make_inputs(case: Case, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """The inputs of ``case`` drawn from ``seed`` (:func:`arrange`), with the FP32 ``beta`` and ``lb``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    g, u = draw_values(case.data, case.R, case.I, gen, device, beta=fp32(case.beta), lb=fp32(case.lb), band=case.band)
    dh = torch.randn(case.R, case.I, device=device, generator=gen).bfloat16()
    return arrange(g.bfloat16(), u.bfloat16(), dh, case.layout, case=case, beta=fp32(case.beta), lb=fp32(case.lb))


def oracle(t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The float64 ``h``, ``d_gate`` and ``d_up`` of the inputs ``t``, in blocks of rows."""
    rows, width, beta, lb = t.R, t.I, t.beta, t.lb
    device = t.gate.device
    out = {name: torch.empty(rows, width, dtype=torch.float64, device=device) for name in ("h", "d_gate", "d_up")}
    step = max(1, ORACLE_ELEMENTS // max(1, width))
    for i in range(0, rows, step):
        span = slice(i, min(rows, i + step))
        g, u, d = t.gate[span].double(), t.up[span].double(), t.dh[span].double()
        s = torch.sigmoid(g)
        a = beta * torch.tanh(g / beta) * s
        up = lb * torch.tanh(u / lb)
        out["h"][span] = a * up
        out["d_gate"][span] = d * up * (s / torch.cosh(g / beta) ** 2 + a * torch.sigmoid(-g))
        out["d_up"][span] = d * a / torch.cosh(u / lb) ** 2
    return out


def naive(t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of every judged output on the inputs ``t`` (see the module)."""
    with torch.enable_grad():
        g = t.gate.detach().clone().requires_grad_(True)
        u = t.up.detach().clone().requires_grad_(True)
        h = t.beta * torch.tanh(g / t.beta) * torch.sigmoid(g) * (t.lb * torch.tanh(u / t.lb))
        h.backward(t.dh)
    return {"h": h.detach().to(NAIVE_DTYPE), "d_gate": g.grad, "d_up": u.grad}


def outputs(fwd: Callable, bwd: Callable, t: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """``h``, ``d_gate`` and ``d_up`` of the forward ``fwd`` and backward ``bwd`` steps on the inputs ``t``; a merged
    layout's gradient is split into its halves."""
    h = fwd(t.first, t.second, beta=t.beta, lb=t.lb)
    d_first, d_up = bwd(t.dh, t.first, t.second, beta=t.beta, lb=t.lb)
    if t.second is None:
        assert d_up is None and tuple(d_first.shape) == (t.R, 2 * t.I)
        d_gate, d_up = d_first[:, : t.I], d_first[:, t.I :]
    else:
        d_gate = d_first
    return {"h": h, "d_gate": d_gate, "d_up": d_up}


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
    backward launched once, every output of ``case.judged`` held to its contract and to the naive tier.  Every verdict
    is recorded (``TIER`` lines), and any failure fails the test with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        t = make_inputs(case, seed=s, device=device)
        found = outputs(fwd, bwd, t)
        held.clear()
        held[j] = naive(t)
        return {name: found[name] for name in case.judged}, oracle(t)

    found_judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed)
    found_judge.record(record_property)
    assert not failures, "\n".join(failures)
