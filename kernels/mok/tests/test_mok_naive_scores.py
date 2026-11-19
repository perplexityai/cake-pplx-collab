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

The score gradient of the MoK oracle and naive implementation on the CPU (:mod:`tests.experimental._mok_naive`):
``d_scores = (dY * y_e).sum(-1)``, ``dY`` the unweighted upstream gradient and ``y_e`` the routed expert's output as
the backward holds it.  With MXFP8 routed experts ``y_e`` is the SwiGLU of the gate and up the backward reads (through
the recipe in the naive implementation, unquantized in the oracle), quantized along K in the naive implementation, by
the backward's copy of the down weight; not the forward output ``y_route``, which reads the forward's copy.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F

from tests.experimental import _mok_naive as naive
from tests.experimental._mok_reference import mxfp8_quantize

ROWS, VALID, HIDDEN, INTERMEDIATE = 256, 240, 128, 256


@pytest.fixture(autouse=True)
def _cpu_matmul(monkeypatch):
    # The CPU's matrix products: the cuBLASLt switch of the GPU runs is a no-op here.
    monkeypatch.setattr(naive, "cublaslt", contextlib.nullcontext)


def _problem(seed=5):
    """One routed expert's padded segment: 240 routes and 16 padding rows, BF16 ``x``, ``dY`` and weights, FP32
    scores, and the stored MXFP8 weights as the package's quantizer returns them."""
    torch.manual_seed(seed)
    x = torch.randn(VALID, HIDDEN).bfloat16()
    dy = (torch.randn(VALID, HIDDEN) * 0.125).bfloat16()
    scores = torch.rand(VALID) + 0.1
    weights = [
        (torch.randn(1, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(1, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(1, HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    # One large element of Wd.  A power-of-two block scale leaves the E4M3 rounding of a normal value unchanged, so the
    # forward copy (blocks along I) and the backward copy (blocks along H) of Wd agree except where a block's scale
    # makes a small element an E4M3 subnormal: the blocks around this element along I and along H, which hold
    # different elements.
    weights[2][0, 5, 7] = 1024.0
    q = [mxfp8_quantize(w) for w in weights]
    mx = dict(forward=[(qq[0], qq[1]) for qq in q], backward=(q[0], q[1], (q[2][2], q[2][3])))
    routes = torch.cat((torch.arange(VALID, dtype=torch.int32), torch.full((ROWS - VALID,), -1, dtype=torch.int32)))
    return x, dy, scores, weights, mx, [naive.Segment(0, routes, 0)]


def _expected(x, dy, mx, limit, dtype, *, down="backward", quantize_gate_up=True):
    """``(dY * y_e).sum(-1)`` of the valid routes in ``dtype``, ``y_e`` from the ``down`` copy of ``Wd``."""
    oracle = dtype == naive.ORACLE
    rows = torch.cat((x, torch.zeros(ROWS - VALID, HIDDEN, dtype=x.dtype)))
    wg, wu = (naive.dequantized(pair)[0].to(dtype) for pair in mx["forward"][:2])
    if down == "backward":
        wd = naive.dequantized(mx["backward"][2], transposed=True)[0].to(dtype)
    else:
        wd = naive.dequantized(mx["forward"][2])[0].to(dtype)
    xn = naive.qdq(rows).to(dtype)
    gate, up = xn @ wg.T, xn @ wu.T
    g, u = (gate, up) if oracle or not quantize_gate_up else (naive.qdq(gate), naive.qdq(up))
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    hidden = F.silu(g) * u
    y_e = (hidden if oracle else naive.qdq(hidden)) @ wd.T
    return (dy.to(dtype) * y_e[:VALID].to(dtype)).sum(-1)


@pytest.mark.parametrize("limit", [None, 0.75], ids=["plain", "clamped"])
@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_the_mxfp8_score_gradient_reads_the_expert_output_the_backward_holds(limit, dtype):
    x, dy, scores, weights, mx, segments = _problem()
    r = naive.routed(x, dy, scores, segments, weights, limit, dtype, mx=mx)
    want = _expected(x, dy, mx, limit, dtype)
    assert r["d_scores"].dtype == dtype and torch.equal(r["d_scores"], want)
    # Not the forward output's form: the forward's copy of Wd gives other values.
    assert not torch.equal(r["d_scores"], _expected(x, dy, mx, limit, dtype, down="forward"))
    if dtype == naive.NAIVE:
        # The naive implementation reads the gate and up through the recipe, as the backward does.
        assert not torch.equal(r["d_scores"], _expected(x, dy, mx, limit, dtype, quantize_gate_up=False))


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_the_bf16_score_gradient_reads_the_routed_output(dtype):
    x, dy, scores, weights, _, segments = _problem()
    r = naive.routed(x, dy, scores, segments, weights, None, dtype)
    assert torch.equal(r["d_scores"], (dy.to(dtype) * r["y_route"]).sum(-1))
