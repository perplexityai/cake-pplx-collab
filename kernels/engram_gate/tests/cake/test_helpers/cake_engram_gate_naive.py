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

The accuracy judges of the family's Engram context gate (four streams of width 5120): the FP64 oracle, the naive BF16
tier and the inputs of each judged form, for the judge of ``tests/cake_naive.py``; and the acceptance measure of issue
#82 (relative L2 error against FP64 beside the FP32 evaluation of the issue's reference).

Per token ``t`` and stream ``s`` (``H = 5120``)::

    h = x[t, s]   k = key[t, s]   v = value[t]   w = q_weight[s] * k_weight[s]   g = dout[t, s]
    r_h = rsqrt(mean(h^2) + eps)   r_k = rsqrt(mean(k^2) + eps)   P = sum_d w h k   dot = P r_h r_k / sqrt(H)
    gate = sigmoid(copysign(sqrt(max(|dot|, clamp)), dot))  (0 on a masked token)     out = h + gate v
    dvalue = sum_s gate g    c = (sum_d g v) gate (1 - gate) du r_h r_k / sqrt(H),  du = 1 / (2 sqrt(|dot|)) if
    |dot| > clamp else 0 (and c = 0 on a masked token)
    dx = g + c (w k - (P r_h^2 / H) h)    dkey = c (w h - (P r_k^2 / H) k)    dw = sum_t c h k
    dq_weight = dw * k_weight    dk_weight = dw * q_weight

:func:`gate_outputs` evaluates it in one dtype: float64 is the oracle, bfloat16 the naive tier (PyTorch, the same ops
in BF16 on the kernel's own inputs; an FP32 weight stays FP32 in the tier, as in BF16 user code that reads it).

The issue's acceptance measure (:func:`rel_l2`, :func:`issue_fp32_random`, :func:`issue_fp32_near`) is the relative
L2 error against the FP64 reference; the bound of an output is twice the error of the FP32 evaluation of the issue's
reference on the same draws: PyTorch's FP32 autograd of :func:`engram_gate_ref` for random draws, and the FP32
formula with an accurate (FP64-summed, FP32-rounded) dot product for the near-cancelling draws, as the issue's script
computes them.  :func:`engram_gate_ref`, :func:`engram_gate_vjp_fp64` and :func:`make_near_cancel_keys` are the
issue's reference, unchanged but for formatting.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Optional

import torch

from tests.cake_naive import TIER_DTYPES, TIERS, judge_case

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
ORACLE_DTYPE = torch.float64
STREAMS = 4
HIDDEN = 5120
KV_WIDTH = (STREAMS + 1) * HIDDEN
EPS = 1e-20
CLAMP = 1e-6
#: The stream scales of the issue's random draws: on ``x``, and reversed on ``key``.
SCALES = (0.25, 1.0, 2.0, 4.0)
#: The judged outputs, in the order of :func:`gate_outputs`.
OUTPUTS = ("out", "dx", "dkey", "dvalue", "dq", "dk")


# ----------------------------------------------------------------------------------------------------------------
# The issue's reference (CPU, FP64)


def signed_sqrt_gate(dot: torch.Tensor, clamp_value: float = CLAMP) -> torch.Tensor:
    return torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))


def engram_gate_ref(x, key, value, q_weight, k_weight, token_mask=None, eps: float = EPS, clamp_value: float = CLAMP):
    orig_dtype = x.dtype
    acc = torch.float64 if orig_dtype == torch.float64 else torch.float32
    h, k, v = x.to(acc), key.to(acc), value.to(acc)
    w = q_weight.to(acc) * k_weight.to(acc)
    H = h.shape[-1]
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * w * k).sum(-1) * rstd * (H**-0.5)
    gate = signed_sqrt_gate(dot, clamp_value)
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    return (h + gate.unsqueeze(-1) * v.unsqueeze(-2)).to(orig_dtype)


def engram_gate_vjp_fp64(x, key, value, q_weight, k_weight, dout, token_mask=None, eps=EPS, clamp_value=CLAMP):
    h, k, v = x.double(), key.double(), value.double()
    qw, kw, g = q_weight.double(), k_weight.double(), dout.double()
    T, S, H = h.shape
    if T == 0:
        zw = torch.zeros_like(qw)
        return h.clone(), k.clone(), v.clone(), zw, zw.clone()
    w = qw * kw
    rh = torch.rsqrt(h.square().mean(-1) + eps)
    rk = torch.rsqrt(k.square().mean(-1) + eps)
    prod_sum = (h * w * k).sum(-1)
    inv_sqrt_h = H**-0.5
    dot = prod_sum * rh * rk * inv_sqrt_h
    gate = signed_sqrt_gate(dot, clamp_value)
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    dv = (gate.unsqueeze(-1) * g).sum(dim=1)
    dgate = (g * v.unsqueeze(1)).sum(dim=-1)
    if token_mask is not None:
        dgate = dgate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    du_ddot = torch.where(dot.abs() > clamp_value, 0.5 / dot.abs().clamp_min(clamp_value).sqrt(), 0.0)
    c = (dgate * gate * (1.0 - gate) * du_ddot * rh * rk * inv_sqrt_h).unsqueeze(-1)
    ps = prod_sum.unsqueeze(-1)
    dx = g + c * (w * k - (ps * rh.square().unsqueeze(-1) / H) * h)
    dk = c * (w * h - (ps * rk.square().unsqueeze(-1) / H) * k)
    dw = (c * h * k).sum(dim=0)
    return dx, dk, dv, dw * kw, dw * qw


def make_near_cancel_keys(x: torch.Tensor, k_init: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """BF16 keys near ``k_init`` with ``2e-6 <= |dot| <= 2e-5`` against ``x`` and ``w`` (CPU tensors)."""
    H = x.shape[-1]
    h, hw = x.double(), x.double() * w.double()
    mag = 2.2e-6 * (8.5 ** torch.rand(k_init.shape[:2], dtype=torch.float64))
    target = torch.where(torch.rand_like(mag) < 0.5, -mag, mag)
    rh = torch.rsqrt(h.square().mean(-1) + EPS)

    def resid(kd: torch.Tensor) -> torch.Tensor:
        return target * (H**0.5) / (rh * torch.rsqrt(kd.square().mean(-1) + EPS)) - (hw * kd).sum(-1)

    kd = k_init.double()
    kd = (kd + (resid(kd) / hw.square().sum(-1)).unsqueeze(-1) * hw).bfloat16().double()
    for _ in range(3):
        left = resid(kd).unsqueeze(-1)
        step = left / hw
        upd = (kd + step).bfloat16().double()
        err = torch.where(step.abs() < 2.0, (left - hw * (upd - kd)).abs(), torch.inf)
        idx = err.argmin(-1, keepdim=True)
        kd = kd.scatter(-1, idx, upd.gather(-1, idx))
    dot = ((hw * kd).sum(-1) * rh * torch.rsqrt(kd.square().mean(-1) + EPS) * (H**-0.5)).abs()
    assert bool((dot >= 2e-6).all() and (dot <= 2e-5).all())
    return kd.bfloat16()


# ----------------------------------------------------------------------------------------------------------------
# The oracle and the naive tier


def stored(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``t`` as a computation in ``dtype`` reads it: cast to ``dtype`` when ``t`` is stored no wider, else as is (an
    FP32 weight stays FP32 in the BF16 tier and is widened in the float64 oracle)."""
    return t.to(dtype) if t.dtype.itemsize <= dtype.itemsize else t


def gate_outputs(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    dout: Optional[torch.Tensor],
    dtype: torch.dtype,
    *,
    eps: float = EPS,
    clamp_value: float = CLAMP,
) -> dict[str, torch.Tensor]:
    """``out`` (and, with ``dout``, ``dx``, ``dkey``, ``dvalue``, ``dq``, ``dk``) of the module docstring's formula
    evaluated in ``dtype`` (float64: the oracle; bfloat16: the naive tier), each returned in ``dtype``."""
    h, k, v = x.to(dtype), key.to(dtype), value.to(dtype)
    qw, kw = stored(q_weight, dtype), stored(k_weight, dtype)
    w = qw * kw
    H = h.shape[-1]
    rh = torch.rsqrt(h.square().mean(-1) + eps)
    rk = torch.rsqrt(k.square().mean(-1) + eps)
    P = (h * w * k).sum(-1)
    dot = P * rh * rk * H**-0.5
    gate = signed_sqrt_gate(dot, clamp_value)
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    found = {"out": (h + gate.unsqueeze(-1) * v.unsqueeze(-2)).to(dtype)}
    if dout is None:
        return found
    g = dout.to(dtype)
    dgate = (g * v.unsqueeze(1)).sum(-1)
    if token_mask is not None:
        dgate = dgate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    du = torch.where(dot.abs() > clamp_value, 0.5 / dot.abs().clamp_min(clamp_value).sqrt(), 0.0)
    c = (dgate * gate * (1.0 - gate) * du * rh * rk * H**-0.5).unsqueeze(-1)
    Pc = P.unsqueeze(-1)
    dw = (c * h * k).sum(0)
    found.update(
        dx=(g + c * (w * k - (Pc * rh.square().unsqueeze(-1) / H) * h)).to(dtype),
        dkey=(c * (w * h - (Pc * rk.square().unsqueeze(-1) / H) * k)).to(dtype),
        dvalue=(gate.unsqueeze(-1) * g).sum(1).to(dtype),
        dq=(dw * kw).to(dtype),
        dk=(dw * qw).to(dtype),
    )
    return found


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def make_inputs(
    T: int,
    *,
    seed: int,
    device: torch.device,
    weights: str = "fp32",
    mask: str = "none",
    layout: str = "contiguous",
    near_cancel: bool = False,
    scaled: bool = True,
) -> types.SimpleNamespace:
    """The operands of one call: ``x`` ``randn`` times the stream scales (0.25, 1, 2, 4), ``key`` ``randn`` times the
    reversed scales (``near_cancel``: moved to ``2e-6 <= |dot| <= 2e-5`` by :func:`make_near_cancel_keys`),
    ``value`` and ``dout`` ``randn``, all BF16; ``q_weight`` / ``k_weight`` ``1 + 0.1 randn`` rounded to BF16 and
    stored as ``weights`` (``"fp32"``, ``"bf16"`` or ``"mixed"``: BF16 ``q_weight``, FP32 ``k_weight``);
    ``token_mask`` None (``mask="none"``) or every fourth token False (``"some"``); ``key`` and ``value`` contiguous
    (``layout="contiguous"``) or the views of one packed ``kv [T, 25600]`` (``"kv"``).  Drawn on the CPU from
    ``seed`` (the draws do not depend on the device) and moved to ``device``."""
    generator = torch.Generator().manual_seed(seed)
    scales = torch.tensor(SCALES if scaled else (1.0,) * STREAMS).view(1, STREAMS, 1)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    x = (randn(T, STREAMS, HIDDEN) * scales).bfloat16()
    k_init = randn(T, STREAMS, HIDDEN) * scales.flip(1)
    value = randn(T, HIDDEN).bfloat16()
    dout = randn(T, STREAMS, HIDDEN).bfloat16()
    qw = (1.0 + 0.1 * randn(STREAMS, HIDDEN)).bfloat16()
    kw = (1.0 + 0.1 * randn(STREAMS, HIDDEN)).bfloat16()
    if near_cancel:
        # make_near_cancel_keys draws from the global generator: seed it from the draw.
        torch.manual_seed(seed)
        key = make_near_cancel_keys(x, k_init, qw.float() * kw.float()) if T else k_init.bfloat16()
    else:
        key = k_init.bfloat16()
    if weights == "fp32":
        qw, kw = qw.float(), kw.float()
    elif weights == "mixed":
        kw = kw.float()
    elif weights != "bf16":
        raise ValueError(f"weights must be fp32, bf16 or mixed, got {weights!r}")
    token_mask = None
    if mask == "some":
        token_mask = torch.ones(T, dtype=torch.bool)
        token_mask[::4] = False
    elif mask != "none":
        raise ValueError(f"mask must be none or some, got {mask!r}")
    ns = types.SimpleNamespace(T=T, kv=None)
    ns.x, ns.dout, ns.q_weight, ns.k_weight = (t.to(device) for t in (x, dout, qw, kw))
    ns.token_mask = None if token_mask is None else token_mask.to(device)
    if layout == "kv":
        kv = torch.cat((key.reshape(T, STREAMS * HIDDEN), value), dim=1).to(device)
        ns.kv = kv
        ns.key = kv[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
        ns.value = kv[:, STREAMS * HIDDEN :]
    elif layout == "contiguous":
        ns.key, ns.value = key.to(device), value.to(device)
    else:
        raise ValueError(f"layout must be contiguous or kv, got {layout!r}")
    return ns


def oracle(ns: types.SimpleNamespace, *, eps: float = EPS, clamp_value: float = CLAMP) -> dict[str, torch.Tensor]:
    """The FP64 oracle of every output of ``ns``."""
    return gate_outputs(
        ns.x,
        ns.key,
        ns.value,
        ns.q_weight,
        ns.k_weight,
        ns.token_mask,
        ns.dout,
        ORACLE_DTYPE,
        eps=eps,
        clamp_value=clamp_value,
    )


def naive(ns: types.SimpleNamespace, *, eps: float = EPS, clamp_value: float = CLAMP) -> dict[str, torch.Tensor]:
    """The naive BF16 tier of every output of ``ns``."""
    return gate_outputs(
        ns.x,
        ns.key,
        ns.value,
        ns.q_weight,
        ns.k_weight,
        ns.token_mask,
        ns.dout,
        NAIVE_DTYPE,
        eps=eps,
        clamp_value=clamp_value,
    )


def judge_draws(run, make, *, seed: int, draws: Optional[int] = None):
    """Judge ``run(ns) -> {output: tensor}`` over the draws of ``make(seed) -> ns`` against the FP64 oracle and the
    BF16 tier of :func:`gate_outputs` (only the outputs ``run`` returns are judged).  Returns the judge and its
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
# The issue's acceptance measure


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``||actual - expected|| / ||expected||`` in FP64 (0 when both are 0)."""
    a, e = actual.double().cpu(), expected.double().cpu()
    denom = e.norm().item()
    num = (a - e).norm().item()
    return num / denom if denom else num


def issue_fp64(ns: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The issue's FP64 reference of every output (CPU)."""
    cpu = [t.cpu() for t in (ns.x, ns.key, ns.value, ns.q_weight, ns.k_weight)]
    mask = None if ns.token_mask is None else ns.token_mask.cpu()
    out = engram_gate_ref(*(t.double() for t in cpu), mask)
    dx, dk, dv, dq, dkw = engram_gate_vjp_fp64(*cpu, ns.dout.cpu(), mask)
    return {"out": out, "dx": dx, "dkey": dk, "dvalue": dv, "dq": dq, "dk": dkw}


def issue_fp32_random(ns: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The FP32 evaluation of the issue's random draws: PyTorch's FP32 autograd of :func:`engram_gate_ref` on the
    CPU, the BF16 operands as BF16 leaves and the weights as FP32 leaves."""
    leaves = [t.detach().cpu().clone() for t in (ns.x, ns.key, ns.value)]
    leaves += [t.detach().cpu().float() for t in (ns.q_weight, ns.k_weight)]
    leaves = [t.requires_grad_(True) for t in leaves]
    mask = None if ns.token_mask is None else ns.token_mask.cpu()
    out = engram_gate_ref(*leaves, mask)
    out.backward(ns.dout.cpu())
    x, k, v, qw, kw = leaves
    return {"out": out.detach(), "dx": x.grad, "dkey": k.grad, "dvalue": v.grad, "dq": qw.grad, "dk": kw.grad}


def issue_fp32_near(ns: types.SimpleNamespace) -> dict[str, torch.Tensor]:
    """The FP32 evaluation of the issue's near-cancelling draws: the FP32 formula with the dot product summed in
    FP64 and rounded to FP32 once (the issue script's accurate-dot FP32 values of ``dx``, ``dkey`` and ``dq``; ``dk``
    likewise)."""
    X, Kn, V, G = (t.cpu() for t in (ns.x, ns.key, ns.value, ns.dout))
    qw, kw = ns.q_weight.cpu().float(), ns.k_weight.cpu().float()
    H = HIDDEN
    hf, kf, vf, gf, wf = X.float(), Kn.float(), V.float(), G.float(), qw * kw
    rhf, rkf = torch.rsqrt(hf.square().mean(-1) + EPS), torch.rsqrt(kf.square().mean(-1) + EPS)
    psf = (X.double() * wf.double() * Kn.double()).sum(-1).float()
    dotf = psf * rhf * rkf * (H**-0.5)
    gatef = signed_sqrt_gate(dotf)
    if ns.token_mask is not None:
        gatef = gatef.masked_fill(~ns.token_mask.cpu().unsqueeze(-1), 0.0)
    duf = torch.where(dotf.abs() > CLAMP, 0.5 / dotf.abs().clamp_min(CLAMP).sqrt(), 0.0)
    cf = ((gf * vf.unsqueeze(1)).sum(-1) * gatef * (1.0 - gatef) * duf * rhf * rkf * (H**-0.5)).unsqueeze(-1)
    dxf = (gf + cf * (wf * kf - (psf.unsqueeze(-1) * rhf.square().unsqueeze(-1) / H) * hf)).bfloat16()
    dkf = (cf * (wf * hf - (psf.unsqueeze(-1) * rkf.square().unsqueeze(-1) / H) * kf)).bfloat16()
    dwf = (cf * hf * kf).sum(0)
    return {"dx": dxf, "dkey": dkf, "dq": dwf * kw, "dk": dwf * qw}


def median(values: list[float]) -> float:
    """The issue script's median: the upper middle element."""
    return sorted(values)[len(values) // 2]


def acceptance_rows(
    ours: Mapping[str, list[float]], fp32: Mapping[str, list[float]], factor: float = 2.0
) -> list[dict[str, object]]:
    """One row per output: the median and maximum relative L2 error of the kernel and of the FP32 evaluation, the
    bounds (``factor`` times the FP32 evaluation's) and whether both are met."""
    rows = []
    for name in ours:
        a, b = ours[name], fp32[name]
        bound = (factor * median(b), factor * max(b))
        rows.append(
            {
                "output": name,
                "n": len(a),
                "ours": (median(a), max(a)),
                "fp32": (median(b), max(b)),
                "bound": bound,
                "ok": median(a) <= bound[0] and max(a) <= bound[1],
            }
        )
    return rows
