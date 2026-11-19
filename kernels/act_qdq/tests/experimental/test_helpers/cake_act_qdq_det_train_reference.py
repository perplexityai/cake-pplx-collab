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

The FP64 rule of the activation quantize-dequantize, and the inputs of its
tests.  Plain torch only.

:func:`qdq` evaluates the rule in float64 on the input's device: per group of
``G`` consecutive elements, ``amax`` of the group (with the format's floor),
the scale ``s``, ``u = clamp(x / s, -Q, Q)``, ``q`` the element format's
round-to-nearest-even value of ``u`` with the sign of zero kept, and
``y = bf16(q * s)``.  Its outputs are exact in BF16 (``y``) and FP32 (``s``),
so the tests compare bit patterns (:func:`y_bits`, :func:`scale_bits`): a
``-0.0`` and a ``+0.0`` differ.

Inputs:

- :func:`randn_input`: ``randn`` scaled per group by ``2^e`` with ``e``
  uniform in ``[-lo, hi]``, a few groups all zero, and some elements set to
  ``+-0.0``.
- :func:`sweep_input`: every finite BF16 value under every scale of the
  format.  Each row is one group; its first element is the group's ``amax``
  (sign alternating), chosen to give one scale, and its other elements walk
  every finite BF16 value of magnitude at most ``amax``.  The E4M3 scale takes
  every positive E4M3 value from 2^-9 to 448, at two ``amax``: ``6 * s`` and the
  largest BF16 that still rounds to ``s`` (so the largest elements clamp),
  plus the largest BF16 (saturation).  An E8M0 scale ``2^k`` takes every ``k``
  the format reaches, at ``amax = Q * 2^k`` (the top of its range) and the
  next BF16 above (the bottom of the next one), plus groups under the floor.
- :func:`edge_input`: edge rows (E2M1 levels and midpoints, E4M3 levels and
  midpoints, the non-power-of-two quotient tie, signed zeros, the floors, the
  2^-32 tie, saturation and the top-of-range infinities).
"""

from __future__ import annotations

import torch

FORMATS = ("fp4_e4m3_g16", "mxfp4_e8m0_g32", "mxfp8_e8m0_g32")
GROUP = {"fp4_e4m3_g16": 16, "mxfp4_e8m0_g32": 32, "mxfp8_e8m0_g32": 32}
QMAX = {"fp4_e4m3_g16": 6.0, "mxfp4_e8m0_g32": 6.0, "mxfp8_e8m0_g32": 448.0}
AMAX_FLOOR = {
    "fp4_e4m3_g16": 6.0 * 2.0**-9,
    "mxfp4_e8m0_g32": 6.0 * 2.0**-126,
    "mxfp8_e8m0_g32": float(torch.tensor(1e-4, dtype=torch.float32).item()),
}
# The E8M0 exponents a format's scale reaches from finite BF16 inputs.
E8M0_RANGE = {"mxfp4_e8m0_g32": (-126, 126), "mxfp8_e8m0_g32": (-22, 120)}
# E2M1 rounding: (midpoint, strict, value).  A midpoint rounds up to ``value``
# when ``value``'s code is even (``>=``) and down otherwise (``>``).
E2M1_THRESHOLDS = (
    (0.25, True, 0.5),
    (0.75, False, 1.0),
    (1.25, True, 1.5),
    (1.75, False, 2.0),
    (2.50, True, 3.0),
    (3.50, False, 4.0),
    (5.00, True, 6.0),
)
# Every non-negative finite E4M3 value (codes 0 .. 0x7E), ascending.
E4M3_POS = torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
E4M3_MIDS = 0.5 * (E4M3_POS[:-1] + E4M3_POS[1:])
# Host lists: the rounding loop reads them without a device round trip.
_E4M3_MID_LIST = E4M3_MIDS.tolist()
# A midpoint above an even code rounds down (``>``), above an odd one up (``>=``).
_E4M3_EVEN_LOW = [k % 2 == 0 for k in range(126)]


def e2m1_rne(u: torch.Tensor) -> torch.Tensor:
    """``u`` rounded to E2M1 (nearest, ties to even, saturating at 6), with the sign of zero kept."""
    a, mag = u.abs().clamp(max=6.0), torch.zeros_like(u)
    for th, strict, val in E2M1_THRESHOLDS:
        mag = torch.where(a > th if strict else a >= th, torch.full_like(mag, val), mag)
    return torch.copysign(mag, u)


def e4m3_rne_sat(u: torch.Tensor) -> torch.Tensor:
    """``u`` rounded to E4M3 (nearest, ties to even, saturating at 448), with the sign of zero kept."""
    grid = E4M3_POS.to(u.device)
    a, idx = u.abs().clamp(max=448.0), torch.zeros(u.shape, dtype=torch.long, device=u.device)
    for k in range(126):
        mid = _E4M3_MID_LIST[k]
        hit = (a > mid) if _E4M3_EVEN_LOW[k] else (a >= mid)
        idx = torch.where(hit, torch.full_like(idx, k + 1), idx)
    return torch.copysign(grid[idx], u)


def ceil_pow2(v: torch.Tensor) -> torch.Tensor:
    """``2^ceil(log2(v))`` for positive ``v``."""
    mant, exp = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), torch.where(mant == 0.5, exp - 1, exp))


def qdq(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The rule in float64: ``(y, s)``, ``y`` of ``x``'s shape and ``s`` of ``[..., D / G]``, both float64."""
    G, qmax = GROUP[fmt], QMAX[fmt]
    g = x.double().reshape(*x.shape[:-1], x.shape[-1] // G, G)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(AMAX_FLOOR[fmt])
    s = e4m3_rne_sat(amax / 6.0) if fmt == "fp4_e4m3_g16" else ceil_pow2(amax / qmax)
    u = (g / s).clamp(-qmax, qmax)
    q = e4m3_rne_sat(u) if fmt == "mxfp8_e8m0_g32" else e2m1_rne(u)
    y = (q * s).to(torch.bfloat16).double()
    return y.reshape(x.shape), s.squeeze(-1)


def y_bits(t: torch.Tensor) -> torch.Tensor:
    """The BF16 bit patterns of ``t`` (exact BF16 values), as int16."""
    return t.to(torch.bfloat16).contiguous().view(torch.int16)


def scale_bits(t: torch.Tensor) -> torch.Tensor:
    """The FP32 bit patterns of ``t`` (exact FP32 values), as int32."""
    return t.to(torch.float32).contiguous().view(torch.int32)


def finite_bf16(device: str | torch.device = "cuda") -> torch.Tensor:
    """Every finite BF16 value, ``+0.0`` and ``-0.0`` included, ascending by magnitude, as float64."""
    bits = torch.arange(0, 0x7F81, dtype=torch.int32, device=device)  # +0 .. 0x7F7F, the largest finite magnitude
    pos = (bits[:-1] << 16).view(torch.float32).double()
    return torch.stack([pos, -pos], 1).reshape(-1)


def _bf16_up(v: torch.Tensor) -> torch.Tensor:
    """The next BF16 value above each (positive, finite) value of ``v``, as float64."""
    b = v.to(torch.bfloat16)
    return torch.nextafter(b, torch.full_like(b, float("inf"))).double()


def _largest_rounding_to(s: torch.Tensor) -> torch.Tensor:
    """For each positive E4M3 scale ``s``, the largest finite BF16 ``amax`` with ``e4m3_rne_sat(amax / 6) == s``."""
    vals = finite_bf16(s.device)
    vals = vals[vals > 0]
    t = e4m3_rne_sat(vals / 6.0)
    # vals ascend, so the last value whose scale is at most s is the largest that rounds to s or below.
    idx = torch.searchsorted(t.contiguous(), s.contiguous(), right=True) - 1
    return vals[idx]


def _sweep_amax(fmt: str, device: str | torch.device) -> torch.Tensor:
    """The group maxima of :func:`sweep_input`, positive float64."""
    if fmt == "fp4_e4m3_g16":
        scales = E4M3_POS[E4M3_POS >= 2.0**-9].to(device)
        top = torch.tensor([torch.finfo(torch.bfloat16).max], dtype=torch.float64, device=device)
        return torch.cat([6.0 * scales, _largest_rounding_to(scales), top]).unique()
    lo, hi = E8M0_RANGE[fmt]
    k = torch.arange(lo, hi + 1, dtype=torch.float64, device=device)
    tops = QMAX[fmt] * torch.exp2(k)
    tops = tops[tops <= torch.finfo(torch.bfloat16).max]
    under = torch.tensor([AMAX_FLOOR[fmt] / 2.0, 2.0**-133, 0.0], dtype=torch.float64, device=device)
    bf16_max = torch.tensor([torch.finfo(torch.bfloat16).max], dtype=torch.float64, device=device)
    amax = torch.cat([tops, _bf16_up(tops), under, bf16_max]).to(torch.bfloat16).double()
    return amax[amax <= torch.finfo(torch.bfloat16).max].unique()


def sweep_input(fmt: str, device: str | torch.device = "cuda") -> torch.Tensor:
    """Every finite BF16 value under every scale of ``fmt``: BF16 ``[rows, G]`` (see the module docstring)."""
    G = GROUP[fmt]
    vals = finite_bf16(device)
    mags = vals.abs()
    rows = []
    for j, amax in enumerate(_sweep_amax(fmt, device).tolist()):
        body = vals[mags <= amax]
        pad = (-body.numel()) % (G - 1)
        body = torch.cat([body, body.new_zeros(pad)]).reshape(-1, G - 1)
        head = body.new_full((body.shape[0], 1), amax if j % 2 == 0 else -amax)
        rows.append(torch.cat([head, body], 1))
    return torch.cat(rows).to(torch.bfloat16).contiguous()


def edge_input(fmt: str, device: str | torch.device = "cuda") -> torch.Tensor:
    """The edge rows of ``fmt``: BF16 ``[rows, G]``, each row one group whose first element is its ``amax``."""
    G = GROUP[fmt]
    f64 = dict(dtype=torch.float64, device=device)
    rows = []

    def row(head: float, values: list[float] | torch.Tensor) -> None:
        values = torch.as_tensor(values, **f64)
        for start in range(0, values.numel(), G - 1):
            r = torch.zeros(G, **f64)
            r[0] = head
            chunk = values[start : start + G - 1]
            r[1 : 1 + chunk.numel()] = chunk
            rows.append(r)

    if fmt != "mxfp8_e8m0_g32":
        levels = [6, -6, 4, -4, 3, -3, 2, -2, 1.5, -1.5, 1, -1, 0.5, -0.5, 0.0, -0.0]
        mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0]
        row(6.0, levels)
        row(6.0, mids)
        row(-6.0, [x / 4 for x in mids])
        if fmt == "fp4_e4m3_g16":
            row(6 * 15 * 2.0**-7, [75 * 2.0**-9, -(75 * 2.0**-9)])  # x / s = 1.25 exactly at s = 15 * 2^-7
            row(4096.0, [-4096.0, 6.25, 2.0**-20])  # saturated scale: the outputs clamp to +-2688
            row(6.25, [-6.25])
            row(2.0**-20, [-(2.0**-20), 0.0, -0.0])  # under the floor: signed zeros
        else:
            row(2.0**-130, [-(2.0**-130), 0.0, -0.0])
            row(2.0**-127, [-(2.0**-127), 2.0**-128, -(2.0**-128)])
    else:
        pos = E4M3_POS.to(device)
        row(448.0, torch.cat([pos, -pos]))
        row(448.0, torch.cat([E4M3_MIDS.to(device), -E4M3_MIDS.to(device)]))
        tie = 2.0**-32
        above = float(_bf16_up(torch.tensor([tie], **f64))[0])
        row(1e-5, [tie, -tie, above, -above, 0.0, -0.0])
        row(-0.0, [0.0, -0.0])
    for top in (torch.finfo(torch.bfloat16).max, 248 * 2.0**120, 224 * 2.0**120):
        row(top, [-top, top / 2, -top / 3])
    return torch.stack(rows).to(torch.bfloat16).contiguous()


def randn_input(
    shape: tuple[int, ...],
    fmt: str,
    *,
    lo: int = 20,
    hi: int = 20,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """``randn`` of ``shape`` (BF16, contiguous), each group of ``fmt`` scaled by ``2^e`` with ``e`` uniform in
    ``[-lo, hi]``; about 1/64 of the groups are all zero and about 1/32 of the elements are ``+-0.0``."""
    G = GROUP[fmt]
    if shape[-1] % G:
        raise ValueError(f"the last dimension {shape[-1]} must be a multiple of the group size {G}")
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, generator=gen, dtype=torch.float32)
    if x.numel():
        groups = x.view(-1, G)
        e = torch.randint(-lo, hi + 1, (groups.shape[0], 1), device=device, generator=gen)
        groups.mul_(torch.exp2(e.float()))
        zero_groups = torch.rand(groups.shape[0], 1, device=device, generator=gen) < 1 / 64
        groups.masked_fill_(zero_groups, 0.0)
        zeros = torch.rand(groups.shape, device=device, generator=gen) < 1 / 32
        groups.copy_(torch.where(zeros, torch.copysign(torch.zeros_like(groups), groups), groups))
    return x.bfloat16().contiguous()
