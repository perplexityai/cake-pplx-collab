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

The accuracy judges of the family's attention residual (AttnRes; values of width 7168): the FP64 oracle, the naive
BF16 tier and the inputs of each judged form, for the judge of ``tests/cake_naive.py``; and the yardstick of issue
#262 (the issue's own reference, FP32 math on the BF16 values with one BF16 rounding of ``out`` and each ``dv_j``,
on the CPU), against which the weight gradient ``dw`` is judged.

Per token ``t`` (``H = 7168``), values ``v_0 .. v_{n-1}`` and the weight ``w``::

    r_j = rsqrt(mean_c v_j[c]^2 + eps)   d_j = v_j . w   p = softmax_j(r_j d_j)   out = sum_j p_j v_j
    dp_j = dout . v_j   ds_j = p_j (dp_j - sum_i p_i dp_i)
    dv_j = p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / H) v_j      dw = sum over t and j of ds_j r_j v_j

:func:`attn_res_outputs` evaluates it in one dtype: float64 is the oracle, bfloat16 the naive tier (PyTorch, the same
ops in BF16 on the kernel's own inputs; the FP32 ``w`` stays FP32 in the tier, as in BF16 user code that reads it).
The outputs are ``out``, ``dv0`` .. ``dv{n-1}`` and ``dw``, and the forward's state ``p``, ``r`` and ``d`` (``[T, n]``).

:func:`released` and :func:`issue_backward` are the issue's reference (``released`` is the model's ``_apply_attn_res``
with the values stacked, as released), unchanged but for formatting and the ``eps`` argument.
"""

from __future__ import annotations

import statistics
import types
from collections.abc import Mapping, Sequence
from typing import Optional

import torch

from tests.cake_naive import NEAR_TIE, NORMS, TIER_DTYPES, TIERS, judge_case, norm_errors

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
ORACLE_DTYPE = torch.float64
HIDDEN = 7168
MAX_VALUES = 9
EPS = 1e-5
#: The input forms of :func:`make_inputs`.
DATA = ("scaled", "peaked", "zero_rows")
#: The weight forms of :func:`make_inputs`.
WEIGHTS = ("small", "normproj")


def output_names(n: int, *, state: bool = False) -> tuple[str, ...]:
    """The judged outputs of a call with ``n`` values: ``out``, ``dv0`` .. ``dv{n-1}``, ``dw`` (and ``p``, ``r``,
    ``d`` with ``state``)."""
    names = ("out", *(f"dv{j}" for j in range(n)), "dw")
    return (*names, "p", "r", "d") if state else names


# ----------------------------------------------------------------------------------------------------------------
# The issue's reference


def released(values: Sequence[torch.Tensor], norm_w: torch.Tensor, proj_w: torch.Tensor, eps: float = EPS):
    """_apply_attn_res with the blocks and the running sum stacked, as released."""
    v = torch.stack(list(values), dim=1)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    p = (k * (norm_w * proj_w)).sum(-1).softmax(-1)
    return torch.matmul(p.unsqueeze(1), v).squeeze(1)


def issue_backward(dout: torch.Tensor, values: Sequence[torch.Tensor], w: torch.Tensor, eps: float = EPS):
    """d values and d w for out = sum_j p_j v_j. H is the hidden size."""
    H = values[0].shape[-1]
    r = torch.stack([torch.rsqrt(v.pow(2).mean(-1) + eps) for v in values], 1)  # [T, n]
    dot = torch.stack([v @ w for v in values], 1)  # [T, n]
    p = (r * dot).softmax(1)
    dp = torch.stack([(dout * v).sum(-1) for v in values], 1)
    ds = p * (dp - (p * dp).sum(1, keepdim=True))  # softmax backward
    dv = [
        p[:, j : j + 1] * dout + (ds * r)[:, j : j + 1] * w - (ds * dot * r.pow(3) / H)[:, j : j + 1] * v
        for j, v in enumerate(values)
    ]
    dw = sum(v.t() @ (ds * r)[:, j] for j, v in enumerate(values))  # FP32 sum over tokens in a kernel
    return dv, dw


# ----------------------------------------------------------------------------------------------------------------
# The oracle and the naive tier


def stored(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t`` as a computation in ``dtype`` reads it: cast to ``dtype`` when ``t`` is stored no wider, else as is (the
    FP32 ``w`` stays FP32 in the BF16 tier and is widened in the float64 oracle)."""
    return t.to(dtype) if t.dtype.itemsize <= dtype.itemsize else t


def attn_res_outputs(
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    dout: Optional[torch.Tensor],
    dtype: torch.dtype,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """``out``, ``p``, ``r``, ``d`` (and, with ``dout``, ``dv0`` .. ``dv{n-1}`` and ``dw``) of the module docstring's
    formula evaluated in ``dtype`` (float64: the oracle; bfloat16: the naive tier), each returned in ``dtype``."""
    vs = [v.to(dtype) for v in values]
    wd = stored(w, dtype)
    H = vs[0].shape[-1]
    r = torch.stack([torch.rsqrt(v.square().mean(-1) + eps) for v in vs], 1)
    d = torch.stack([(v * wd).sum(-1) for v in vs], 1)
    p = (r * d).softmax(1)
    out = sum(p[:, j : j + 1] * v for j, v in enumerate(vs))
    found = {"out": out.to(dtype), "p": p.to(dtype), "r": r.to(dtype), "d": d.to(dtype)}
    if dout is None:
        return found
    g = dout.to(dtype)
    dp = torch.stack([(g * v).sum(-1) for v in vs], 1)
    ds = p * (dp - (p * dp).sum(1, keepdim=True))
    a = ds * r
    b = ds * d * r.pow(3) / H
    for j, v in enumerate(vs):
        found[f"dv{j}"] = (p[:, j : j + 1] * g + a[:, j : j + 1] * wd - b[:, j : j + 1] * v).to(dtype)
    found["dw"] = sum((a[:, j : j + 1] * v).sum(0) for j, v in enumerate(vs)).to(dtype)
    return found


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def make_inputs(
    T: int,
    n: int,
    *,
    seed: int,
    device: torch.device,
    data: str = "scaled",
    weights: str = "small",
) -> types.SimpleNamespace:
    """The operands of one call, drawn on ``device`` from ``seed``: ``n`` BF16 ``[T, 7168]`` values, ``w`` FP32
    ``[7168]`` and ``dout`` BF16 ``[T, 7168]`` (``randn``).

    ``data``: ``"scaled"``, value ``j`` is ``randn`` times ``linspace(0.5, 4, n)[j]`` (the running sum, last, is the
    largest, as residual sums grow); ``"peaked"``, the same, and in every token one value (drawn per token) carries a
    component along ``w`` that gives it almost all the softmax weight (its score ``r_j d_j`` about 20 to 60 plus three
    standard deviations of the other scores above them); ``"zero_rows"``, the scaled values with every third token
    row of the first value all zero (with ``n >= 2`` a closed block state; ``r = 1 / sqrt(eps)``, ``d = 0``).
    ``weights``: ``"small"``, ``w = 0.05 randn`` (the issue's yardstick); ``"normproj"``, ``w = (1 + 0.1 randn) *
    0.3 randn``, the product of a norm and a projection weight (the issue's gradient check)."""
    if data not in DATA:
        raise ValueError(f"data must be one of {DATA}, got {data!r}")
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device)

    if weights == "small":
        w = 0.05 * randn(HIDDEN)
    else:
        w = (1.0 + 0.1 * randn(HIDDEN)) * (0.3 * randn(HIDDEN))
    scales = torch.linspace(0.5, 4.0, n).tolist()
    values = [randn(T, HIDDEN) * c for c in scales]
    if data == "peaked" and T:
        # u = w / rms(w): adding gamma u to a randn row raises its score r_j d_j by about gamma * H * rms(w); the
        # scores of plain randn rows are about N(0, sigma), sigma = sqrt(H) rms(w).
        rms_w = w.square().mean().sqrt()
        u = w / rms_w
        sigma = HIDDEN**0.5 * rms_w
        chosen = torch.randint(0, n, (T,), generator=generator, device=device)
        shift = 20.0 + 40.0 * torch.rand(T, 1, generator=generator, device=device) + 3.0 * sigma
        gamma = shift / (HIDDEN * rms_w)
        for j in range(n):
            rows = (chosen == j).unsqueeze(1)
            values[j] = torch.where(rows, values[j] + scales[j] * gamma * u, values[j])
    if data == "zero_rows" and T:
        values[0][::3] = 0.0
    ns = types.SimpleNamespace(T=T, n=n, data=data, weights=weights)
    ns.values = [v.bfloat16() for v in values]
    ns.w = w.float()
    ns.dout = randn(T, HIDDEN).bfloat16()
    return ns


def oracle(ns: types.SimpleNamespace, *, eps: float = EPS) -> dict[str, torch.Tensor]:
    """The FP64 oracle of every output of ``ns``."""
    return attn_res_outputs(ns.values, ns.w, ns.dout, ORACLE_DTYPE, eps=eps)


def naive(ns: types.SimpleNamespace, *, eps: float = EPS) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of every output of ``ns``."""
    return attn_res_outputs(ns.values, ns.w, ns.dout, NAIVE_DTYPE, eps=eps)


def judge_draws(run, make, *, seed: int, draws: Optional[int] = None):
    """Judge ``run(ns) -> {output: tensor}`` over the draws of ``make(seed) -> ns`` against the FP64 oracle and the
    BF16 tier of :func:`attn_res_outputs` (only the outputs ``run`` returns are judged).  Returns the judge and its
    failures (``tests.cake_naive.judge_case``)."""
    cache: dict[int, types.SimpleNamespace] = {}

    def run_draw(j: int, s: int):
        ns = make(s)
        cache[j] = ns
        outputs = run(ns)
        want = oracle(ns)
        return outputs, {name: want[name] for name in outputs}

    def naive_draw(j: int, tier: str):
        return naive(cache.pop(j))

    kwargs = {} if draws is None else {"draws": draws}
    return judge_case(run_draw, naive_draw, seed=seed, **kwargs)


# ----------------------------------------------------------------------------------------------------------------
# The issue's yardstick: FP32 math on the BF16 values (CPU), one BF16 rounding of out and each dv_j


def issue_fp32(ns: types.SimpleNamespace, *, eps: float = EPS) -> dict[str, torch.Tensor]:
    """The issue's FP32 evaluation of every output (its yardstick script, on the CPU): ``out`` from :func:`released`
    and ``dv_j`` from :func:`issue_backward` in FP32, each rounded once to BF16; ``dw`` the FP32 sum of
    :func:`issue_backward`."""
    v32 = [v.cpu().float() for v in ns.values]
    w32 = ns.w.cpu().float()
    found = {"out": released(v32, w32, torch.ones_like(w32), eps).bfloat16()}
    dv, dw = issue_backward(ns.dout.cpu().float(), v32, w32, eps)
    for j, g in enumerate(dv):
        found[f"dv{j}"] = g.bfloat16()
    found["dw"] = dw
    return found


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``||actual - expected|| / ||expected||`` in FP64 (0 when both are 0)."""
    (num,), denom = norm_errors(expected, (actual.to(expected.device),), "l2")
    return num / denom if denom else num


class Fp32SumJudge:
    """The issue's bar for ``dw``: no worse than an FP32 sum.  Per draw, the error of the kernel's ``dw`` and of the
    yardstick's FP32 ``dw`` (:func:`issue_fp32`) against the FP64 oracle, in the L2 and the max norm; the kernel meets
    the bar when, in both norms, the median of its errors over the draws is at most :data:`tests.cake_naive.NEAR_TIE`
    times the yardstick's median and its largest error at most that times the yardstick's largest."""

    def __init__(self) -> None:
        self.errors: dict[str, dict[str, list[float]]] = {norm: {"ours": [], "fp32": []} for norm in NORMS}

    def add(self, actual: torch.Tensor, fp32: torch.Tensor, expected: torch.Tensor) -> None:
        for norm in NORMS:
            (ours, theirs), _ = norm_errors(expected, (actual.to(expected.device), fp32.to(expected.device)), norm)
            self.errors[norm]["ours"].append(ours)
            self.errors[norm]["fp32"].append(theirs)

    def rows(self) -> list[dict[str, object]]:
        found = []
        for norm, by in self.errors.items():
            ours, theirs = sorted(by["ours"]), sorted(by["fp32"])
            med = (statistics.median(ours), statistics.median(theirs))
            top = (ours[-1], theirs[-1])
            ok = med[0] <= NEAR_TIE * med[1] and top[0] <= NEAR_TIE * top[1]
            found.append(
                {"norm": norm, "draws": len(ours), "ours": (med[0], top[0]), "fp32": (med[1], top[1]), "ok": ok}
            )
        return found

    def failures(self) -> list[str]:
        return [
            f"dw ({row['norm']}): ours median/largest {row['ours'][0]:.6g}/{row['ours'][1]:.6g} against the FP32 sum's "
            f"{row['fp32'][0]:.6g}/{row['fp32'][1]:.6g}"
            for row in self.rows()
            if not row["ok"]
        ]


def yardstick_rows(ours: Mapping[str, list[float]], fp32: Mapping[str, list[float]]) -> list[dict[str, object]]:
    """One row per output: the median and largest relative L2 error against FP64 of the kernel and of the issue's
    FP32 evaluation over the draws (reported beside the tier verdicts; the bar of ``out`` and ``dv_j`` is the tier)."""
    rows = []
    for name in ours:
        a, b = sorted(ours[name]), sorted(fp32[name])
        rows.append(
            {"output": name, "n": len(a), "ours": (statistics.median(a), a[-1]), "fp32": (statistics.median(b), b[-1])}
        )
    return rows
