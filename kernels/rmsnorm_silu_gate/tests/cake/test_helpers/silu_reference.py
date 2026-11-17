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

The SiLU-gated RMSNorm's float64 oracle, its PyTorch BF16 naive
implementation and the judge of a case's training outputs against them
(:mod:`tests.cake.test_helpers.cake_naive`).

The oracle is the operation in float64 on the BF16 inputs, with no FP32
rounding anywhere: ``rstd = (mean(x^2) + eps)^-1/2``, ``s = sigmoid(z)``,
``g = z s``, ``n = x rstd``; ``y = n w g``; ``dn = dy w g``;
``dx = rstd dn - x rstd^3 sum(dn x) / 128``; ``dz = dy n w s (1 + z (1 - s))``;
``dw`` the sum of ``dy g n`` over every row, plus the starting value of an
accumulated weight gradient.  The naive implementation is the same operation
in PyTorch with every op in BF16 (no FP32 cast, so each op's result rounds to
BF16), its gradients from torch.autograd; an accumulated weight gradient is
the BF16 sum of the starting value and the gradient.

:func:`judge` runs a case over :data:`DRAWS` independent random inputs and
holds every output to the naive implementation (:class:`TierJudge`): in the
L2 and in the max norm, the median and the largest error of the output over
the draws at most :data:`NEAR_TIE` times the naive implementation's, and 0
on every draw where the naive implementation's is 0.  A launch that runs
again on a draw's inputs (a weight gradient whose last bits may vary from
run to run) is judged on every launch: the judge keeps the larger error.
Every output is also held to its contract (:func:`contract`).

Only torch is imported.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch

from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TIERS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)

D = 128
#: Rows of the float64 oracle computed at once (bounds its temporaries).
ORACLE_ROWS = 1 << 18
#: The quantities of the operation: the forward's output and the backward's
#: three gradients.
KINDS = ("y", "dx", "dz", "dw")


def oracle(
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor | None,
    eps: float,
    kinds: Iterable[str] = KINDS,
) -> dict[str, torch.Tensor]:
    """The float64 oracle of ``kinds`` on the BF16 inputs: ``y``, ``dx`` and
    ``dz`` in ``x``'s shape, ``dw`` ``[128]`` (the sum over every row).
    ``dy`` may be ``None`` when ``kinds`` is ``("y",)``."""
    kinds = set(kinds)
    shape = x.shape
    rows = x.numel() // D
    device = x.device
    xs, zs = x.reshape(rows, D), z.reshape(rows, D)
    gs = None if dy is None else dy.reshape(rows, D)
    wd = w.double().to(device)
    out = {
        kind: torch.empty(rows, D, dtype=torch.float64, device=device)
        for kind in ("y", "dx", "dz")
        if kind in kinds
    }
    dw = torch.zeros(D, dtype=torch.float64, device=device)
    for start in range(0, rows, ORACLE_ROWS):
        part = slice(start, start + ORACLE_ROWS)
        xd, zd = xs[part].double(), zs[part].double()
        rstd = torch.rsqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
        s = torch.sigmoid(zd)
        g = zd * s
        n = xd * rstd
        if "y" in out:
            out["y"][part] = n * wd * g
        if kinds & {"dx", "dz", "dw"}:
            gd = gs[part].double()
            if "dx" in out:
                dn = gd * wd * g
                dot = (dn * xd).sum(dim=-1, keepdim=True)
                out["dx"][part] = rstd * dn - xd * rstd.pow(3) * dot / D
            if "dz" in out:
                out["dz"][part] = gd * n * wd * s * (1.0 + zd * (1.0 - s))
            if "dw" in kinds:
                dw += (gd * g * n).sum(dim=0)
    found = {kind: value.reshape(shape) for kind, value in out.items()}
    if "dw" in kinds:
        found["dw"] = dw
    return found


def naive_forward(
    x: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """The operation with every op in BF16: no FP32 cast, so each op's result
    rounds to BF16."""
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return ((x * rstd) * w) * torch.nn.functional.silu(z)


def naive(
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor | None,
    eps: float,
    kinds: Iterable[str] = KINDS,
) -> dict[str, torch.Tensor]:
    """The PyTorch BF16 naive implementation's ``kinds``, all BF16: ``y``,
    ``dx`` and ``dz`` in ``x``'s shape, ``dw`` ``[128]``; the gradients from
    torch.autograd through :func:`naive_forward` on copies of the inputs, which
    stay unwritten."""
    kinds = set(kinds)
    grads = bool(kinds - {"y"})
    xr, zr, wr = (t.detach().clone().requires_grad_(grads) for t in (x, z, w))
    with torch.enable_grad():
        y = naive_forward(xr, zr, wr, eps)
        if grads:
            y.backward(dy)
    found = {"y": y.detach()}
    if grads:
        found.update(dx=xr.grad, dz=zr.grad, dw=wr.grad)
    return {kind: found[kind] for kind in kinds}


class Draw:
    """One draw of a case: ``launches``, the kernel's outputs by name, one
    mapping per launch on the draw's inputs; ``expected``, the oracle of every
    name; ``naive()``, the BF16 naive value of every name."""

    def __init__(
        self,
        launches: Sequence[Mapping[str, torch.Tensor]],
        expected: Mapping[str, torch.Tensor],
        naive: Callable[[], Mapping[str, torch.Tensor]],
    ) -> None:
        self.launches = launches
        self.expected = expected
        self.naive = naive


def silu_draw(
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    dy: torch.Tensor | None,
    eps: float,
    launches: Sequence[Mapping[str, torch.Tensor]],
    kinds: Mapping[str, str],
    *,
    starts: Mapping[str, torch.Tensor] | None = None,
) -> Draw:
    """The :class:`Draw` of the outputs ``launches`` of the operation on the
    BF16 inputs: ``kinds`` maps each output name to its quantity (``y``,
    ``dx``, ``dz`` or ``dw``), ``starts`` the name of an accumulated weight
    gradient to its starting value (a copy taken before the call: the oracle
    adds it in float64, the naive implementation in BF16)."""
    starts = dict(starts or {})
    used = set(kinds.values())
    reference = oracle(x, z, w, dy, eps, used)
    expected = {}
    for name, kind in kinds.items():
        value = reference[kind]
        if name in starts:
            value = starts[name].to(device=value.device, dtype=torch.float64) + value
        expected[name] = value

    def naive_values() -> dict[str, torch.Tensor]:
        values = naive(x, z, w, dy, eps, used)
        found = {}
        for name, kind in kinds.items():
            value = values[kind]
            if name in starts:
                value = starts[name].to(device=value.device, dtype=torch.bfloat16) + value
            found[name] = value
        return found

    return Draw(launches, expected, naive_values)


def judge(
    record_property: Callable[[str, object], None],
    run: Callable[[int, int], Draw],
    *,
    seed: int = 0,
    draws: int = DRAWS,
) -> None:
    """Judge a case over ``draws`` random inputs and record each output's
    verdict (``record_property``, printed as ``TIER`` lines by the family's
    ``conftest.py``).

    For draw ``j`` the Python and torch random state is seeded with
    :func:`draw_seed` of ``seed`` and ``j``, and ``run(j, s)`` (``s`` that
    seed, which the case's inputs take) makes the draw's inputs, launches the
    kernels and returns its :class:`Draw`.  Every output of every launch is
    held to its :func:`contract` and recorded in the judge (a second launch on a draw keeps the larger error).
    The random state is restored after the last draw.  Fails with every
    contract failure and every output that missed the naive tier."""
    tiers = TierJudge(draws)
    failures = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(draws):
            s = draw_seed(seed, j)
            seed_all(s)
            draw = run(j, s)
            values = None
            for outputs in draw.launches:
                for name, actual in outputs.items():
                    want = draw.expected[name]
                    if message := contract(name, actual, want):
                        failures.append(f"draw {j}: {message}")
                        if actual.shape != want.shape:
                            continue
                    if values is None:
                        values = draw.naive()
                    tiers.add(
                        name, j, actual, want, {tier: values[name] for tier in TIERS}
                    )
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    tiers.record(record_property)
    failures += tiers.failures()
    assert not failures, "\n".join(failures)
