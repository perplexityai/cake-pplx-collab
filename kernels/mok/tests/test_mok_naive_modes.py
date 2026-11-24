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

The explicit modes of the MoK oracle and naive implementation on the CPU (:mod:`tests.experimental._mok_naive`): the
per-token shared-expert scale ``y = s S(x) + sum_k w_k E_k(x)`` with ``d_s = (dY * S(x)).sum(-1)``, the layer without a
shared expert, zero route weights with their defined score gradient (the direct form, no division), and SiTU-GLU
``beta tanh(g / beta) sigmoid(g) lb tanh(u / lb)`` whose float64 oracle takes its gradients in the stable forms
``1 / cosh^2`` and ``sigmoid(-g)``.
"""

from __future__ import annotations

import contextlib
from decimal import Decimal, getcontext

import pytest
import torch

from tests.experimental import _mok_naive as naive

TOKENS, TOPK, EXPERTS, HIDDEN, INTERMEDIATE = 64, 4, 4, 128, 256


@pytest.fixture(autouse=True)
def _cpu_matmul(monkeypatch):
    # The CPU's matrix products: the cuBLASLt switch of the GPU runs is a no-op here.
    monkeypatch.setattr(naive, "cublaslt", contextlib.nullcontext)


def _problem(seed=17, zero_every=0):
    """A one-rank layer: BF16 ``x``, ``dY`` and weights, FP32 scores (every ``zero_every``-th one exactly 0), each
    token's routes on ``TOPK`` distinct experts, one padded segment per expert in route order."""
    torch.manual_seed(seed)
    x = torch.randn(TOKENS, HIDDEN).bfloat16()
    dy = (torch.randn(TOKENS, HIDDEN) * 0.125).bfloat16()
    scores = torch.rand(TOKENS, TOPK) + 0.1
    if zero_every:
        scores.view(-1)[::zero_every] = 0.0
    shared = [
        (torch.randn(INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    routed = [
        (torch.randn(EXPERTS, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(EXPERTS, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    expert = (torch.arange(TOKENS)[:, None] + torch.arange(TOPK)[None, :]) % EXPERTS
    route = torch.arange(TOKENS * TOPK).view(TOKENS, TOPK)
    segments = []
    for e in range(EXPERTS):
        rows = route[expert == e].to(torch.int32)
        pad = -rows.numel() % 256
        rows = torch.cat((rows, torch.full((pad,), -1, dtype=torch.int32)))
        segments.append(naive.Segment(e, rows, 0))
    return x, dy, scores, shared, routed, segments


def _layer(problem, dtype, **kwargs):
    x, dy, scores, shared, routed, segments = problem
    shared = kwargs.pop("shared_w", shared)
    return naive.layer(x, dy, scores, shared, routed, segments, None, dtype, **kwargs)


def _same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_shared_scale_one_is_the_unscaled_layer(dtype):
    """``s = 1`` gives the unscaled layer bit for bit in both evaluations: ``BF16(dY * 1) = dY`` and ``1 * S(x)`` is
    exact; ``d_shared_scale`` is ``(dY * S(x)).sum(-1)``."""
    problem = _problem()
    plain = _layer(problem, dtype)
    scaled = _layer(problem, dtype, shared_scale=torch.ones(TOKENS))
    for name in ("y", "dx", "d_scores", "dw"):
        assert _same(plain[name], scaled[name]), name
    x, dy, _, shared, _, _ = problem
    s_x = naive.shared(x, dy, shared, None, dtype)["y"]
    assert _same(scaled["d_shared_scale"], (dy.to(dtype) * s_x).sum(-1))


def test_shared_scale_oracle_follows_the_formula():
    """The oracle with ``s`` in (0, 2) and a quarter of the scales exactly 0: ``y = s S(x) + sum_k w_k E_k(x)``,
    ``dx`` adds ``s`` times the unscaled shared input gradient, the shared weight gradients are those of ``s dY`` and
    ``d_s = dot(dY, S(x))``, nonzero also where ``s = 0``; all in float64 from the formula's own terms."""
    problem = _problem(seed=23)
    x, dy, scores, shared, _, _ = problem
    s = torch.rand(TOKENS) * 2
    s[::4] = 0.0
    out = _layer(problem, naive.ORACLE, shared_scale=s)
    routed_only = _layer(problem, naive.ORACLE, shared_w=None)  # 0 + the routed sums: the routed sums exactly
    unscaled = naive.shared(x, dy, shared, None, naive.ORACLE)
    s64 = s.double()[:, None]
    assert torch.equal(out["y"], s64 * unscaled["y"] + routed_only["y"])
    scaled = naive.mlp(x, shared, dy.double() * s64, None, naive.ORACLE)
    assert torch.equal(out["dx"], scaled["dx"] + routed_only["dx"])
    for got, want in zip(out["dw"][3:], scaled["dw"], strict=True):
        assert torch.equal(got[0], want)
    d_s = (dy.double() * unscaled["y"]).sum(-1)
    assert torch.equal(out["d_shared_scale"], d_s)
    assert bool((d_s[::4] != 0).all())


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_shared_scale_zero_removes_the_shared_term_and_keeps_its_gradient(dtype):
    """``s = 0`` on every token: ``y`` and ``dx`` are the routed sums alone (the layer without a shared expert), the
    shared weight gradients are exactly 0, and ``d_shared_scale`` is the defined ``(dY * S(x)).sum(-1)``."""
    problem = _problem(seed=29)
    out = _layer(problem, dtype, shared_scale=torch.zeros(TOKENS))
    alone = _layer(problem, dtype, shared_w=None)
    assert _same(out["y"], alone["y"]) and _same(out["dx"], alone["dx"])
    assert len(alone["dw"]) == 3 and _same(out["dw"][:3], alone["dw"])
    for grad in out["dw"][3:]:
        assert torch.count_nonzero(grad).item() == 0
    assert torch.count_nonzero(out["d_shared_scale"]).item() > 0


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_zero_route_weights_have_the_defined_score_gradient(dtype):
    """Every 5th route weight exactly 0: the judged ``d_scores = (dY * y_e).sum(-1)`` is nonzero there (defined at
    zero), the route adds nothing to ``y``, ``dx`` or the routed weight gradients (the layer with those routes' expert
    outputs dropped), and in the naive implementation the quotient alternative is 0 there while the direct alternative
    ``(dh' * hidden).sum(-1)`` (unscaled ``dh'``, no division) is not."""
    problem = _problem(seed=31, zero_every=5)
    out = _layer(problem, dtype, direct=True)
    zero = problem[2].view(-1) == 0
    d_scores = out["d_scores"].view(-1)
    assert bool((d_scores[zero] != 0).all())
    x, dy, scores, shared, routed, segments = problem
    kept = [naive.Segment(seg.expert, torch.where(zero[seg.routes.clamp(min=0).long()] & (seg.routes >= 0),
                                                  torch.full_like(seg.routes, -1), seg.routes), 0)
            for seg in segments]  # fmt: skip
    dropped = naive.layer(x, dy, scores, shared, routed, kept, None, dtype)
    assert _same(out["y"], dropped["y"]) and _same(out["dx"], dropped["dx"]) and _same(out["dw"], dropped["dw"])
    if dtype == naive.NAIVE:
        assert torch.count_nonzero(out["alt_d_scores"].view(-1)[zero]).item() == 0
        assert bool((out["alt_d_scores_direct"].view(-1)[zero] != 0).all())


def test_no_shared_expert_layer_is_the_routed_sum():
    """``shared_w=None``: ``y`` and ``dx`` are the routed sums alone and ``dw`` holds the three routed gradients; a
    shared-expert scale without a shared expert is refused."""
    problem = _problem(seed=37)
    out = _layer(problem, naive.ORACLE, shared_w=None)
    with_shared = _layer(problem, naive.ORACLE)
    x, dy, _, shared, _, _ = problem
    s_out = naive.shared(x, dy, shared, None, naive.ORACLE)
    assert torch.equal(with_shared["y"], s_out["y"] + out["y"])
    assert torch.equal(with_shared["dx"], s_out["dx"] + out["dx"])
    assert len(out["dw"]) == 3 and all(torch.equal(a, b) for a, b in zip(out["dw"], with_shared["dw"][:3]))
    with pytest.raises(ValueError, match="needs the shared expert"):
        _layer(problem, naive.ORACLE, shared_w=None, shared_scale=torch.ones(TOKENS))


BETA, LB = 4.0, 25.0
#: Gate and up values: saturated (|g| up to 200, |u| up to 1000), deep negative gates, small values and zeros.
SITU_POINTS = [
    (-200.0, 3.0), (-150.0, -700.0), (-95.0, 1000.0), (-84.0, 0.5), (-40.0, 12.0), (-3.0, -0.0625), (-0.03125, 2.0),
    (0.0, 5.0), (2.0, 0.0), (0.25, 0.001), (5.5, -40.0), (16.0, 80.0), (45.0, -260.0), (75.0, 18.0), (200.0, -1000.0),
]  # fmt: skip


def _exact(g, u, beta, lb):
    """SiTU-GLU's value and its two partial derivatives at ``(g, u)``, in 60-digit decimal arithmetic from ``exp``."""
    getcontext().prec = 60
    g, u, beta, lb = (Decimal(repr(v)) for v in (g, u, beta, lb))

    def tanh(z):
        e = (-2 * abs(z)).exp()
        t = (1 - e) / (1 + e)
        return t if z >= 0 else -t

    def sech2(z):
        e = (-2 * abs(z)).exp()
        return 4 * e / (1 + e) ** 2

    def sigmoid(z):
        e = (-abs(z)).exp()
        return 1 / (1 + e) if z >= 0 else e / (1 + e)

    a = beta * tanh(g / beta) * sigmoid(g)
    up = lb * tanh(u / lb)
    d_a = sech2(g / beta) * sigmoid(g) + a * sigmoid(-g)
    return a * up, up * d_a, a * sech2(u / lb)


def test_situ_oracle_matches_exact_arithmetic_after_one_bf16_rounding():
    """The float64 SiTU-GLU oracle's value, gate gradient and up gradient (unit upstream gradient) round to the same
    BF16 value as 60-digit decimal arithmetic at saturated, deep-negative, small and zero inputs; the forms
    ``1 - tanh * tanh`` and ``1 - sigmoid`` would give 0 for the saturated gates, where the oracle's gradient is a normal
    nonzero value (shown at the positive saturated gate, where both forms cancel)."""
    g = torch.tensor([p[0] for p in SITU_POINTS], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([p[1] for p in SITU_POINTS], dtype=torch.float64, requires_grad=True)
    h = naive.situ_glu(g, u, BETA, LB)
    h.backward(torch.ones_like(h))
    exact = [_exact(gi, ui, BETA, LB) for gi, ui in SITU_POINTS]
    for i, column in enumerate((h.detach(), g.grad, u.grad)):
        want = torch.tensor([float(e[i]) for e in exact], dtype=torch.float64)
        assert torch.equal(column.bfloat16(), want.bfloat16()), (i, column, want)
    saturated = torch.tensor([abs(p[0]) > 37 and p[1] != 0 for p in SITU_POINTS])
    assert bool((g.grad[saturated] != 0).all())
    gate, up = g.detach(), u.detach()
    tg, s = torch.tanh(gate / BETA), torch.sigmoid(gate)
    cancelled = LB * torch.tanh(up / LB) * ((1 - tg * tg) * s + BETA * tg * s * (1 - s))
    big = gate > 120  # tanh and sigmoid both round to 1 in float64
    assert torch.count_nonzero(cancelled[big]).item() == 0 and bool((g.grad[big & saturated] != 0).all())


def test_situ_naive_is_the_pytorch_bf16_graph_and_zero_inputs_give_zero():
    """The naive SiTU-GLU expert is the PyTorch spelling on the BF16 tensors with autograd in BF16; a zero gate or a
    zero up gives an exactly zero activation in both evaluations."""
    torch.manual_seed(41)
    x = torch.randn(256, HIDDEN).bfloat16()
    dy = (torch.randn(256, HIDDEN) * 0.125).bfloat16()
    weights = [
        (torch.randn(INTERMEDIATE, HIDDEN) / HIDDEN**0.5 * 8).bfloat16(),
        (torch.randn(INTERMEDIATE, HIDDEN) / HIDDEN**0.5 * 8).bfloat16(),
        (torch.randn(HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    out = naive.mlp(x, weights, dy, None, naive.NAIVE, activation=("situ_glu", BETA, LB))
    with torch.enable_grad():
        xx = x.detach().requires_grad_()
        gate, up = xx @ weights[0].T, xx @ weights[1].T
        hidden = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate) * (LB * torch.tanh(up / LB))
        y = hidden @ weights[2].T
        y.backward(dy)
    assert torch.equal(out["hidden"], hidden.detach()) and torch.equal(out["y"], y.detach())
    assert torch.equal(out["dx"], xx.grad)
    zeros = torch.zeros(8, dtype=torch.float64)
    values = torch.linspace(-50, 50, 8, dtype=torch.float64)
    for dtype in (torch.float64, torch.bfloat16):
        assert torch.count_nonzero(naive.situ_glu(zeros.to(dtype), values.to(dtype), BETA, LB)).item() == 0
        assert torch.count_nonzero(naive.situ_glu(values.to(dtype), zeros.to(dtype), BETA, LB)).item() == 0
    with pytest.raises(ValueError, match="without a limit"):
        naive.mlp(x, weights, dy, 1.0, naive.NAIVE, activation=("situ_glu", BETA, LB))
