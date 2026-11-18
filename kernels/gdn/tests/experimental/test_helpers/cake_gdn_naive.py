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

The naive tier of the single-rank GDN training suites (``test_cake_gdn_det_train.py``,
``test_cake_gdn_nondet_train.py``), for the judge of ``tests/cake_naive.py``: PyTorch BF16.

:func:`token_recurrence` is those suites' FP64 token recurrence (FLA's gated delta rule, the suites' ``_reference``)
and its autograd on the same inputs with every op in BF16:

- an input stored no wider than BF16 (``q``, ``k``, ``v``, ``do``, a BF16 ``beta``) is read as BF16; one stored wider
  (the FP32 ``g``, an FP32 ``beta``, the FP32 initial state and final-state cotangent) keeps its storage dtype, so an
  elementwise op of it runs in FP32 by torch's type promotion, as in BF16 user code that reads the FP32 tensor;
- the products ``S^T k`` and ``S^T q`` take BF16 operands (an FP32 state is cast to BF16 where a product takes it);
- every result (the output, the final state and every gradient) is rounded to BF16.

Per sequence and value head ``hv`` (key head ``hv // (HV / H)``), with the state ``S`` ``[K, V]``::

    S = S * exp(g_t);  u_t = beta_t (v_t - S^T k_t);  S = S + k_t u_t^T;  o_t = scale S^T q_t
"""

from __future__ import annotations

import math

import torch

HEAD_DIM = 128
L2NORM_EPS = 1e-6  # the prep stage's q / k normalization: x / sqrt(sum x^2 + 1e-6)
NAIVE_DTYPE = torch.bfloat16


def stored(x: torch.Tensor, dtype: torch.dtype = NAIVE_DTYPE) -> torch.Tensor:
    """``x`` as the naive tier in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


def token_recurrence(
    inputs: dict,
    seq_lens,
    *,
    initial_state: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    normalize: bool = True,
    dtype: torch.dtype = NAIVE_DTYPE,
) -> dict[str, torch.Tensor]:
    """The suites' token recurrence on ``inputs`` (``q``, ``k`` ``[B, T, H, 128]``, ``v``, ``do`` ``[B, T, HV, 128]``,
    ``g``, ``beta`` ``[B, T, HV]``) with every op in ``dtype`` (see the module docstring): ``o``, ``dq``, ``dk``,
    ``dv``, ``dg``, ``dbeta``, ``final_state`` and (with ``initial_state``) ``dh0``, each rounded to ``dtype``.
    ``normalize``: the in-kernel L2 norm of ``q`` / ``k`` first."""
    leaves = {
        name: stored(inputs[name].detach(), dtype).clone().requires_grad_(True)
        for name in ("q", "k", "v", "g", "beta")
    }
    h0 = None
    if initial_state is not None:
        h0 = stored(initial_state.detach(), dtype).clone().requires_grad_(True)
    q, k, v, g, beta = (leaves[n].flatten(0, 1) for n in ("q", "k", "v", "g", "beta"))
    num_heads, num_v_heads = q.shape[1], v.shape[1]
    group = num_v_heads // num_heads
    if normalize:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + L2NORM_EPS)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + L2NORM_EPS)
    q = q.repeat_interleave(group, dim=1) * (1.0 / math.sqrt(HEAD_DIM))
    k = k.repeat_interleave(group, dim=1)
    outs, finals = [], []
    start = 0
    for i, n in enumerate(seq_lens):
        S = (
            h0[i]
            if h0 is not None
            else torch.zeros(
                num_v_heads, HEAD_DIM, HEAD_DIM, dtype=dtype, device=q.device
            )
        )
        for t in range(start, start + int(n)):
            S = S * torch.exp(g[t])[:, None, None]
            u = beta[t][:, None] * (
                v[t] - torch.einsum("hk,hkv->hv", k[t].to(dtype), S.to(dtype))
            )
            S = S + k[t][:, :, None] * u[:, None, :]
            outs.append(torch.einsum("hk,hkv->hv", q[t].to(dtype), S.to(dtype)))
        finals.append(S)
        start += int(n)
    o = torch.stack(outs, 0).reshape(inputs["v"].shape)
    final = torch.stack(finals, 0)
    loss = (o * stored(inputs["do"].detach(), dtype)).sum()
    if dht is not None:
        loss = loss + (final * stored(dht.detach(), dtype)).sum()
    loss.backward()
    res = {
        "o": o.detach(),
        "dq": leaves["q"].grad,
        "dk": leaves["k"].grad,
        "dv": leaves["v"].grad,
        "dg": leaves["g"].grad,
        "dbeta": leaves["beta"].grad,
        "final_state": final.detach(),
    }
    if h0 is not None:
        res["dh0"] = h0.grad
    return {name: t.to(dtype) for name, t in res.items()}
