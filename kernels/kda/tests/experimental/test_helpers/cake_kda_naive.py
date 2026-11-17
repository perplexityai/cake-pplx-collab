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

The naive tier of the KDA training accuracy tests (``test_cake_kda_*_naive.py``), for the judge of
``tests/cake_naive.py``: PyTorch BF16.

:func:`naive_with_grads` is the FP64 token recurrence of ``cake_kda_det_train_reference.py``
(:func:`kda_exact_with_grads`, :func:`kda_exact_raw_with_grads` and their in-kernel Q/K L2 norm forms in
``cake_kda_det_train_reference_l2n.py``) and its autograd on the same inputs, with every op in BF16:

- an input stored no wider than BF16 (``q``, ``k``, ``v``, ``beta``, ``dout``, the raw gate) is read as BF16; one
  stored wider (the FP32 activated gate ``g``, ``A_log``, ``dt_bias``) keeps its storage dtype, so an elementwise op
  of it runs in FP32 by torch's type promotion, as in BF16 user code that reads the FP32 tensor;
- the products ``k^T S`` and ``q^T S`` take BF16 operands (the state is cast to BF16 where a product takes it), and
  the outer product ``(beta k) (x) (v - k^T S)`` is a product of BF16 operands;
- with the in-kernel Q/K L2 norm the raw ``q`` / ``k`` rows are normalized in BF16 (``x / sqrt(sum x^2 + 1e-6)``);
- every result (the output and every gradient) is rounded to BF16.

Per document and head, with the state ``S`` ``[K, V]`` starting from zero::

    S = S * exp(g_t)[:, None];  S = S + (beta_t k_t) (x) (v_t - k_t^T S);  o_t = scale q_t^T S

In the raw gate mode the gate is activated in the same graph: ``g = lb * sigmoid(exp(A_log)[h] * (raw +
dt_bias[h]))``.  The recurrence is checkpointed in 64-token blocks, as the reference is, so the backward of a few
thousand tokens fits in memory.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

D = 128
L2NORM_EPS = 1e-6  # the in-kernel Q/K L2 norm: x / sqrt(sum x^2 + 1e-6)
NAIVE_DTYPE = torch.bfloat16
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")


def stored(x: torch.Tensor, dtype: torch.dtype = NAIVE_DTYPE) -> torch.Tensor:
    """``x`` as the naive tier in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


def _block(S0, q, k, v, g, beta, scale, dtype):
    # S0 [H, K, V]; q, k, v [T, H, D] in dtype; g [T, H, D] (FP32, or dtype); beta [T, H] in dtype
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = torch.einsum("hk,hkv->hv", kt, St.to(dtype))
        St = St + torch.einsum("hk,hv->hkv", beta[t][:, None] * kt, v[t] - pred)
        outs.append(scale * torch.einsum("hk,hkv->hv", q[t], St.to(dtype)))
    return torch.stack(outs, 0), St


def naive_forward(q, k, v, g, beta, cu_seqlens, scale, block=64, dtype=NAIVE_DTYPE):
    """``out [S, H, D]`` of the per-document recurrence on ``q, k, v, g [S, H, D]``, ``beta [S, H]`` (the tensors as
    the naive tier reads them, see :func:`stored`) with every product in ``dtype``; the output is in ``dtype``."""
    S, H, K = q.shape
    V = v.shape[-1]
    pieces = []
    cu = [int(x) for x in cu_seqlens.tolist()]
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        St = torch.zeros(H, K, V, dtype=dtype, device=q.device)
        for b0 in range(s0, s1, block):
            b1 = min(b0 + block, s1)
            ob, St = checkpoint(
                _block,
                St,
                q[b0:b1],
                k[b0:b1],
                v[b0:b1],
                g[b0:b1],
                beta[b0:b1],
                scale,
                dtype,
                use_reentrant=False,
            )
            pieces.append(ob)
    if not pieces:
        return torch.zeros(S, H, V, dtype=dtype, device=q.device)
    return torch.cat(pieces, 0)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + L2NORM_EPS)


def naive_with_grads(
    q,
    k,
    v,
    g,
    beta,
    dout,
    cu_seqlens,
    scale,
    *,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    l2n: bool = False,
    dtype=NAIVE_DTYPE,
) -> dict[str, torch.Tensor]:
    """The naive tier of the reference's ``out`` and gradients, each rounded to ``dtype``: ``out``, ``dq``, ``dk``,
    ``dv``, ``dg``, ``dbeta`` and with ``A_log`` (the raw gate mode: ``g`` is the raw BF16 gate, ``A_log`` FP32
    ``[H]``, ``dt_bias`` FP32 ``[H * 128]`` or ``[H, 128]``, ``lower_bound`` the gate's lower bound) also ``dA_log``
    and ``ddt_bias``.  ``l2n``: ``q`` / ``k`` are the raw rows, normalized inside the graph."""
    raw = A_log is not None
    srcs = (q, k, v, g, beta) + ((A_log, dt_bias) if raw else ())
    leaves = [stored(x.detach(), dtype).clone().requires_grad_(True) for x in srcs]
    qn, kn, vn, gn, bn = leaves[:5]
    if l2n:
        qn, kn = _normalize(qn), _normalize(kn)
    if raw:
        a_log, bias = leaves[5], leaves[6]
        H = gn.shape[-2]
        gn = lower_bound * torch.sigmoid(
            torch.exp(a_log).reshape(H, 1) * (gn + bias.reshape(H, D))
        )
    o = naive_forward(qn, kn, vn, gn, bn, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, stored(dout.detach(), dtype).to(o.dtype))
    names = RAW_GRAD_NAMES if raw else GRAD_NAMES
    out = {"out": o.detach()}
    out.update(dict(zip(names, (x.detach() for x in grads), strict=True)))
    return {name: t.to(dtype) for name, t in out.items()}


def draw_case(
    lengths, H: int, gate: str, l2n: bool, seed: int, device="cuda"
) -> dict[str, torch.Tensor]:
    """The operands of one draw of a judged case, every tensor a function of ``seed``: the reference helper's
    :func:`make_inputs` (``gate="g"``: the activated FP32 gate) or :func:`make_raw_inputs` (``"raw"``: the BF16 raw
    gate, FP32 ``A_log`` / ``dt_bias``), and with ``l2n`` raw BF16 ``q`` / ``k`` rows of a few magnitudes per token
    (the caller's projections) in place of the normalized ones."""
    from .cake_kda_det_train_reference import make_inputs, make_raw_inputs

    make = {"g": make_inputs, "raw": make_raw_inputs}[gate]
    inputs = make(lengths, H, device, seed=seed)
    if l2n:
        S = int(sum(lengths))
        gen = torch.Generator().manual_seed(int(seed) + 7817)
        mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
        inputs["q"] = (
            (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16).to(device)
        )
        inputs["k"] = (
            (torch.randn(S, H, D, generator=gen) * mag.flip(0))
            .to(torch.bfloat16)
            .to(device)
        )
    return inputs


def oracle_with_grads(
    inputs: dict, scale: float, lower_bound: float, l2n: bool
) -> dict[str, torch.Tensor]:
    """The FP64 reference of :func:`draw_case`'s operands (``out`` and every gradient by name), from the reference
    helpers unchanged."""
    from .cake_kda_det_train_reference import (
        kda_exact_raw_with_grads,
        kda_exact_with_grads,
    )
    from .cake_kda_det_train_reference_l2n import (
        kda_exact_l2n_with_grads,
        kda_exact_raw_l2n_with_grads,
    )

    a = tuple(inputs[n] for n in ("q", "k", "v", "g", "beta"))
    if "A_log" in inputs:
        fn = kda_exact_raw_l2n_with_grads if l2n else kda_exact_raw_with_grads
        o, grads = fn(
            *a,
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["dout"],
            inputs["cu_seqlens"],
            scale,
            lower_bound,
        )
    else:
        fn = kda_exact_l2n_with_grads if l2n else kda_exact_with_grads
        o, grads = fn(*a, inputs["dout"], inputs["cu_seqlens"], scale)
    return {"out": o, **grads}


def naive_of(
    inputs: dict, scale: float, lower_bound: float, l2n: bool
) -> dict[str, torch.Tensor]:
    """:func:`naive_with_grads` of :func:`draw_case`'s operands, keyed as :func:`oracle_with_grads`."""
    raw = {}
    if "A_log" in inputs:
        raw = {
            "A_log": inputs["A_log"],
            "dt_bias": inputs["dt_bias"],
            "lower_bound": lower_bound,
        }
    return naive_with_grads(
        *(inputs[n] for n in ("q", "k", "v", "g", "beta", "dout", "cu_seqlens")),
        scale,
        l2n=l2n,
        **raw,
    )


def cake_step(
    entry, inputs: dict, lower_bound: float, l2n: bool, *, ckpt: bool = False
) -> dict[str, torch.Tensor]:
    """``out`` and the gradients of every leaf (``q, k, v, g, beta`` and in the raw gate mode ``A_log``,
    ``dt_bias``) of one training step through the autograd entry ``entry`` (``cake_kda_det`` or
    ``cake_kda_nondet``), keyed as :func:`oracle_with_grads`; ``ckpt``: under ``torch.utils.checkpoint``."""
    raw = "A_log" in inputs
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    cu = inputs["cu_seqlens"]

    def fn(*a):
        kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
        if raw:
            kw.update(
                use_gate_in_kernel=True,
                A_log=a[5],
                dt_bias=a[6],
                lower_bound=lower_bound,
            )
        return entry(*a[:5], cu, **kw)

    o = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
    grads = torch.autograd.grad(o, leaves, inputs["dout"])
    out = {"out": o.detach()}
    out.update(dict(zip(RAW_GRAD_NAMES if raw else GRAD_NAMES, grads, strict=True)))
    return out


def judge_single_rank(
    entry,
    case: tuple,
    seed: int,
    device,
    scale: float,
    lower_bound: float,
    *,
    det: bool,
):
    """Judge one single-rank case ``(lengths, H, gate, l2n, ckpt)`` of the autograd entry ``entry`` over the judge's
    draws (``tests/cake_naive.py``, :func:`judge_case`): every output of :func:`cake_step` against
    :func:`oracle_with_grads` beside :func:`naive_of`, and with ``ckpt`` the same step under
    ``torch.utils.checkpoint`` (outputs ``ckpt.<name>``).  ``det``: on the first draw the step runs twice and every
    output must repeat bit for bit (the deterministic package's run-to-run contract).  Returns ``(judge,
    failures)``."""
    from tests.cake_naive import judge_case

    lengths, H, gate, l2n, ckpt = case
    seen = {}

    def run(j, s):
        inputs = draw_case(lengths, H, gate, l2n, s, device)
        seen["inputs"] = inputs
        got = cake_step(entry, inputs, lower_bound, l2n)
        if det and j == 0:
            again = cake_step(entry, inputs, lower_bound, l2n)
            differ = [n for n in got if not torch.equal(got[n], again[n])]
            assert not differ, (
                f"run to run: {differ} differ bitwise between two identical calls"
            )
        ref = oracle_with_grads(inputs, scale, lower_bound, l2n)
        outputs, oracle = dict(got), dict(ref)
        if ckpt:
            for name, t in cake_step(
                entry, inputs, lower_bound, l2n, ckpt=True
            ).items():
                outputs[f"ckpt.{name}"] = t
                oracle[f"ckpt.{name}"] = ref[name]
        return outputs, oracle

    def naive(j, tier):
        nv = naive_of(seen["inputs"], scale, lower_bound, l2n)
        if ckpt:
            nv.update({f"ckpt.{name}": t for name, t in list(nv.items())})
        return nv

    return judge_case(run, naive, seed=seed)


__all__ = [
    "GRAD_NAMES",
    "L2NORM_EPS",
    "NAIVE_DTYPE",
    "RAW_GRAD_NAMES",
    "cake_step",
    "draw_case",
    "judge_single_rank",
    "naive_forward",
    "naive_of",
    "naive_with_grads",
    "oracle_with_grads",
    "stored",
]
