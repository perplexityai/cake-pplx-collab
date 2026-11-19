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

The router weight before the down projection in the MoK oracle and naive implementation, on the CPU
(:mod:`tests.experimental._mok_naive`, ``weight_before_down=True``): ``y_route = BF16(s * a) Wd^T`` (float64 in the
oracle), ``d_scores = (dh * a).sum(-1)`` with ``dh`` the gradient of the down projection's input, the activation's
gradient ``s * dh``, the inclusive clamp masks at the limit, the unweighted combine, and the requester's input
distribution (:func:`tests.experimental._mok_reference.production_inputs`).  Every check is exact.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F

from tests.experimental import _mok_naive as naive
from tests.experimental._mok_reference import production_inputs

ROWS, VALID, HIDDEN, INTERMEDIATE = 256, 240, 128, 256
LIMIT = 0.75


@pytest.fixture(autouse=True)
def _cpu_matmul(monkeypatch):
    # The CPU's matrix products: the cuBLASLt switch of the GPU runs is a no-op here.
    monkeypatch.setattr(naive, "cublaslt", contextlib.nullcontext)


def _problem(seed=11):
    """One routed expert's padded segment: 240 routes and 16 padding rows, BF16 ``x``, ``dY`` and weights, FP32
    scores in (0, 1.5) with every 7th one exactly 0."""
    torch.manual_seed(seed)
    x = torch.randn(VALID, HIDDEN).bfloat16()
    dy = (torch.randn(VALID, HIDDEN) * 0.125).bfloat16()
    scores = torch.rand(VALID) * 1.5
    scores[::7] = 0.0
    weights = [
        (torch.randn(1, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(1, INTERMEDIATE, HIDDEN) / HIDDEN**0.5).bfloat16(),
        (torch.randn(1, HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    routes = torch.cat((torch.arange(VALID, dtype=torch.int32), torch.full((ROWS - VALID,), -1, dtype=torch.int32)))
    return x, dy, scores, weights, [naive.Segment(0, routes, 0)]


def _routed(x, dy, scores, weights, segments, dtype):
    return naive.routed(x, dy, scores, segments, weights, LIMIT, dtype, weight_before_down=True)


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_before_down_score_gradient_does_not_depend_on_the_score(dtype):
    """``d_scores = (dh * a).sum(-1)`` reads neither the score nor any value it scales: the same bits for other scores,
    zero scores included; a zero-score route has a zero output and a zero input gradient."""
    x, dy, scores, weights, segments = _problem()
    r = _routed(x, dy, scores, weights, segments, dtype)
    other = torch.rand(VALID) * 1.5
    other[1::5] = 0.0
    q = _routed(x, dy, other, weights, segments, dtype)
    assert r["d_scores"].dtype == dtype and torch.equal(r["d_scores"], q["d_scores"])
    assert torch.count_nonzero(r["d_scores"][scores == 0]).item() > 0  # defined at a zero score, no division
    for result, s in ((r, scores), (q, other)):
        zero = s == 0
        assert torch.count_nonzero(result["y_route"][zero]).item() == 0
        assert torch.count_nonzero(result["dx_route"][zero]).item() == 0
        assert torch.count_nonzero(result["y_route"][~zero].abs().sum(-1) == 0).item() == 0
    assert "alt_d_scores" not in r


def test_before_down_oracle_follows_the_formula():
    """The float64 oracle on the segment's own rows: ``y_route = (s * a) Wd^T`` and ``d_scores = ((dY Wd) * a).sum(-1)``
    with ``a = silu(min(g, L)) * clamp(u, -L, L)``, the same operations in the same order; doubling every score
    doubles ``y_route`` and ``dx_route`` exactly and keeps ``d_scores``."""
    x, dy, scores, weights, segments = _problem()
    r = _routed(x, dy, scores, weights, segments, naive.ORACLE)
    routes = segments[0].routes
    xs = naive._rows(routes, x).double()
    s = naive._rows(routes, scores.float()).double()
    dys = naive._rows(routes, dy).double()
    wg, wu, wd = (w[0].double() for w in weights)
    gate, up = xs @ wg.T, xs @ wu.T
    a = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    y = (a * s[:, None]) @ wd.T
    d_scores = ((dys @ wd) * a).sum(-1)
    assert torch.equal(r["y_route"], y[:VALID])
    assert torch.equal(r["d_scores"], d_scores[:VALID])
    doubled = _routed(x, dy, scores * 2, weights, segments, naive.ORACLE)
    assert torch.equal(doubled["y_route"], 2 * r["y_route"])
    assert torch.equal(doubled["dx_route"], 2 * r["dx_route"])
    assert torch.equal(doubled["d_scores"], r["d_scores"])


def test_before_down_naive_casts_the_weighted_activation_once():
    """The naive down projection reads ``(a * s).to(bfloat16)``: the BF16 activation times the FP32 score in FP32,
    cast to BF16 once, not the product of the score rounded to BF16."""
    x, dy, scores, weights, segments = _problem()
    rows = naive._rows(segments[0].routes, x)
    s = naive._rows(segments[0].routes, scores.float())
    out = naive.mlp(rows, [w[0] for w in weights], naive._rows(segments[0].routes, dy), LIMIT, naive.NAIVE, s)
    assert out["a"].dtype == out["hidden"].dtype == torch.bfloat16
    assert torch.equal(out["hidden"], (out["a"].float() * s[:, None]).bfloat16())
    assert not torch.equal(out["hidden"], out["a"] * s.bfloat16()[:, None])
    assert torch.equal(out["y"], out["hidden"] @ weights[2][0].T)
    assert torch.equal(out["d_route"], (out["dh"] * out["a"]).sum(-1))


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_before_down_masks_pass_the_gradient_at_the_limit(dtype):
    """With the score before the down projection, the clamped SwiGLU's gradient still passes at ``g = L`` and
    ``u = +-L`` exactly and is zero beyond: one-hot ``x`` rows make gate and up the weights' own entries."""
    limit = 10.0
    torch.manual_seed(3)
    x = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    x[torch.arange(ROWS), torch.arange(ROWS) % HIDDEN] = 1.0
    weights = [
        (torch.randn(INTERMEDIATE, HIDDEN)).bfloat16(),
        (torch.randn(INTERMEDIATE, HIDDEN)).bfloat16(),
        (torch.randn(HIDDEN, INTERMEDIATE) / INTERMEDIATE**0.5).bfloat16(),
    ]
    beyond = 10.0625  # the next BF16 value above 10
    weights[0][0, 0], weights[0][1, 0] = limit, beyond
    weights[1][0, 0], weights[1][1, 0], weights[1][2, 0] = limit, -limit, -beyond
    dy = torch.randn(ROWS, HIDDEN).bfloat16()
    s = torch.full((ROWS,), 0.75)
    out = naive.mlp(x, weights, dy, limit, dtype, route_weight=s.double() if dtype == naive.ORACLE else s)
    gate, up = out["d_gate"][0], out["d_up"][0]
    # g = L passes; g beyond L does not (its up is -L, so the gradient would be nonzero without the mask)
    assert gate[0].item() != 0 and gate[1].item() == 0
    assert up[0].item() != 0 and up[1].item() != 0 and up[2].item() == 0  # u = +-L pass, u beyond -L does not


@pytest.mark.parametrize("dtype", [naive.ORACLE, naive.NAIVE], ids=["oracle", "naive"])
def test_before_down_combine_sums_the_routed_rows_unweighted(dtype):
    """``y = shared + sum_k y_route_k``: the routed rows carry their scores, so the scores are not read."""
    t, topk, hidden = 8, 6, 16
    torch.manual_seed(5)
    shared_y = torch.randn(t, hidden).to(dtype)
    route = dict(y_route=torch.randn(t * topk, hidden).to(dtype), dx_route=torch.randn(t * topk, hidden).to(dtype))
    y, dx = naive.combine(shared_y, shared_y, route, torch.zeros(t, topk), dtype, weight_before_down=True)
    assert torch.equal(y, shared_y + route["y_route"].view(t, topk, hidden).sum(1))
    assert torch.equal(dx, shared_y + route["dx_route"].view(t, topk, hidden).sum(1))


def test_before_down_has_no_mxfp8_form():
    x, dy, scores, weights, segments = _problem()
    with pytest.raises(ValueError, match="no MXFP8 form"):
        naive.routed(x, dy, scores, segments, weights, LIMIT, naive.NAIVE, mx={}, weight_before_down=True)


def test_production_inputs_follow_the_requested_distribution():
    """Unit-RMS BF16 rows; distinct experts; nonnegative weights up to 1.5 with exact zeros (every 8th token's smallest
    and token 2's ``(1.5, 0, ...)``); a pinned token whose gate is exactly ``+L`` and ups exactly ``+L`` and ``-L``;
    gates beyond the limit on some elements."""
    tokens, hidden, intermediate, experts, topk, limit = 64, 512, 256, 16, 6, 10.0
    p = production_inputs(tokens, hidden, intermediate, experts, topk, limit, device="cpu", seed=3)
    x, w, ids = p["x"], p["weights"], p["ids"]
    assert x.dtype == torch.bfloat16 and w.dtype == torch.float32 and ids.dtype == torch.int64
    assert all(len(set(row)) == topk for row in ids.tolist())
    assert (w >= 0).all() and (w <= 1.5).all()
    assert (w[7::8] == 0).any(-1).all() and torch.count_nonzero(w[2]).item() == 1 and w[2, 0].item() == 1.5
    token, expert = p["pinned"]
    assert (x[token].abs() == 1).all()
    row = x[token].double()
    assert (row @ p["routed"][0][expert, 0].double()).item() == limit
    assert (row @ p["routed"][1][expert, 0].double()).item() == limit
    assert (row @ p["routed"][1][expert, 1].double()).item() == -limit
    gate = x.double() @ p["routed"][0][0].double().T
    assert torch.count_nonzero(gate > limit).item() > 0
