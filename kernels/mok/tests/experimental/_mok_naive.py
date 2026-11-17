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

The float64 oracle and the naive PyTorch BF16 implementation of the MoK layer: every definition the accuracy
checks use lives in this module.

The layer.  One expert is the SwiGLU MLP ``E(x) = (silu(clamp(x Wg^T, max=L)) * clamp(x Wu^T, -L, L)) Wd^T`` (no clamp
without a limit ``L``); the clamp's gradient passes at the limit itself (``gate <= L``, ``-L <= up <= L``).  A token's
output is ``y = S(x) + sum_k s_k E_{e_k}(x)``: the shared expert ``S`` plus its ``topk`` routed experts, each weighted
by its FP32 score.  The kernels return, per route, the routed expert's
unweighted output (``y_route``), the gradient of ``x`` through that route (``dx_route``) and the score gradient
(``d_scores``); per weight, the weight gradient (BF16, or added into the caller's FP32 accumulator).

* The oracle (:data:`ORACLE`, float64) evaluates the layer and its gradients in float64 on the kernel's own inputs
  (BF16 ``x``, ``dY`` and weights, FP32 scores).  With MXFP8 routed experts the routed GEMMs read ``x`` and the
  routed weights through the MX recipe (the forward GEMMs ``x`` along K, the weight-gradient GEMMs ``x`` along the
  rows; ``x``'s E4M3 codes are themselves an exactly checked output) and the caller-prequantized weights of each
  pass; no other value is quantized.  A computed value the kernel stores as E4M3 codes is judged as the value
  ``code * 2**(byte - 127)`` against the oracle's float64 value before quantization.
* The naive implementation (:data:`NAIVE`, bfloat16) is the standard PyTorch form of the same layer: the ops of
  autograd in BF16 with torch's own internal accumulation (one cuBLAS BF16 GEMM per expert and weight gradient,
  ``silu`` and its backward as torch computes them on BF16, the weighted sum promoted by the FP32 scores) and
  ``d_scores = (dY * y_route).sum(-1)`` in BF16.  With MXFP8 routed experts it applies the MX recipe's quantization
  points and nothing else: ``x`` (both orientations), the SwiGLU activation (along K for the down GEMM, along the rows
  for its weight gradient), the saved gate and up the backward reads, the score-scaled ``dY`` and ``d_gate`` /
  ``d_up`` (both orientations).  An FP32 output gets the BF16 result widened exactly.
* :data:`ALTERNATIVE`: the kernel-shaped variants of the naive implementation, measured and reported beside the judged
  tier and never judged: BF16 weight gradients accumulated in BF16 per macrobatch (the kernel's order), and the score
  gradient as ``(dh * hidden).sum(-1) / s`` (the kernel's fused form).

Each routed expert's rows are given in the kernel's schedule order (a padded segment: ``-1`` rows are padding), since
the MX recipe's blocks along the rows and the macrobatches follow that order.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.experimental._mok_reference import mxfp8_dequantize, mxfp8_quantize

ORACLE = torch.float64
NAIVE = torch.bfloat16
#: The kernel-shaped alternatives, by output name: what they change in the naive implementation.
ALTERNATIVE = {
    "dw_routed_gate": "BF16 accumulation per macrobatch",
    "dw_routed_up": "BF16 accumulation per macrobatch",
    "dw_routed_down": "BF16 accumulation per macrobatch",
    "d_scores": "(dh * hidden).sum(-1) / s",
}


@contextlib.contextmanager
def cublaslt():
    """GEMMs through cuBLASLt: the cuBLAS 13.1 default path on SM103 drops the last rows of some BF16 ``A^T B``
    reductions (for example K = 9475), which would weaken the naive weight gradients."""
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def qdq(value: torch.Tensor, transposed: bool = False) -> torch.Tensor:
    """A BF16 ``[R, K]`` value through the MX recipe and back: E4M3 codes and UE8M0 scales of 32-element blocks along K
    (along R when ``transposed``), dequantized exactly, as BF16 ``[R, K]`` (an E4M3 value times a power of two is a
    BF16 value).  R and K are multiples of 128."""
    value = value.bfloat16().contiguous()
    if transposed:
        q = mxfp8_quantize(value, normal=False)
        return mxfp8_dequantize(q[2], q[3]).transpose(-1, -2).bfloat16()
    q = mxfp8_quantize(value, transposed=False)
    return mxfp8_dequantize(q[0], q[1]).bfloat16()


def dequantized(pair, transposed: bool = False) -> torch.Tensor:
    """FP32 values of a stored ``(E4M3 codes, tile-layout UE8M0 scales)`` pair; ``transposed`` pairs (blocks along the
    rows, stored ``[K, R]``) are returned in ``[R, K]`` orientation."""
    data, scales = pair
    values = mxfp8_dequantize(data.contiguous().view(torch.float8_e4m3fn), scales.contiguous())
    return values.transpose(-1, -2) if transposed else values


class _Clamp(torch.autograd.Function):
    """``value.clamp(low, high)`` whose gradient passes at the limits themselves (``low <= value <= high``): the
    clamped SwiGLU's definition.  torch's autograd of ``clamp`` gives 0 at the limits, which BF16 and E4M3 gate and up
    values hit exactly."""

    @staticmethod
    def forward(ctx, value, low, high):
        ctx.save_for_backward(value)
        ctx.limits = (low, high)
        return value.clamp(low, high)

    @staticmethod
    def backward(ctx, grad):
        (value,) = ctx.saved_tensors
        low, high = ctx.limits
        keep = value <= high
        if low is not None:
            keep = keep & (value >= low)
        return torch.where(keep, grad, torch.zeros_like(grad)), None, None


def _clamped(gate, up, limit):
    if limit is None:
        return gate, up
    return _Clamp.apply(gate, None, limit), _Clamp.apply(up, -limit, limit)


def mlp(x, weights, dy, limit, dtype):
    """One SwiGLU MLP and its backward in ``dtype``, by autograd: ``y``, ``dx``, ``d_gate``, ``d_up``, ``dh`` (the
    gradient of the activation), the activation and the three weight gradients."""
    with torch.enable_grad(), cublaslt():
        x = x.to(dtype).detach().requires_grad_()
        w = [t.to(dtype).detach().requires_grad_() for t in weights]
        gate, up = x @ w[0].T, x @ w[1].T
        gate.retain_grad()
        up.retain_grad()
        gc, uc = _clamped(gate, up, limit)
        hidden = F.silu(gc) * uc
        hidden.retain_grad()
        y = hidden @ w[2].T
        y.backward(dy.to(dtype))
    return dict(
        y=y.detach(),
        dx=x.grad,
        d_gate=gate.grad,
        d_up=up.grad,
        dh=hidden.grad,
        hidden=hidden.detach(),
        dw=[t.grad for t in w],
    )


@dataclass
class Segment:
    """One routed expert's rows in schedule order: ``expert`` (index into the routed weights), ``routes`` (route index
    of each row, ``-1`` for padding; a multiple of 256 rows) and ``offset`` (the first row's index in its rank's
    routed slab, for the macrobatch boundaries)."""

    expert: int
    routes: torch.Tensor
    offset: int


def _rows(routes, values):
    """``values[routes]`` with zero rows where ``routes`` is ``-1``."""
    valid = routes >= 0
    picked = values[routes.clamp(min=0).long()]
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    return torch.where(valid.view(-1, *([1] * (values.dim() - 1))), picked, zero)


def _macro_sum(product, rows, offset, macro, dtype):
    """Kernel-shaped weight gradient: the per-macrobatch BF16 products of rows ``[0, rows)`` (slab rows
    ``offset + i``) accumulated in BF16 in macrobatch order."""
    total = None
    start = 0
    while start < rows:
        end = min(rows, (offset + start) // macro * macro + macro - offset)
        part = product(start, end).to(dtype)
        total = part if total is None else (total.float() + part.float()).to(dtype)
        start = end
    return total


def routed(
    x_routes,
    dy_routes,
    s_routes,
    segments,
    weights,
    limit,
    dtype,
    *,
    mx=None,
    macro=None,
):
    """The routed experts on per-route inputs (``x_routes``, ``dy_routes``: ``[routes, hidden]`` BF16, the source row
    of each route; ``s_routes``: ``[routes]`` FP32), in ``dtype`` (:data:`ORACLE` or :data:`NAIVE`).

    ``weights``: the BF16 ``(gate, up, down)`` ``[experts, ...]``, or with ``mx`` the stored MXFP8 weights
    ``dict(forward=(gate, up, down) pairs, backward=(gate, up, down) pairs)`` as the package's quantizer returns them.
    Returns per route ``y_route``, ``dx_route``, ``d_scores`` (zero on routes no segment holds), the weight gradients
    ``dw`` (three ``[experts, ...]``) and, with ``macro``, the kernel-shaped alternatives ``alt_dw`` and
    ``alt_d_scores`` (:data:`ALTERNATIVE`; naive only).  With ``mx`` also the per-row values the kernel stores as
    E4M3 codes, per segment (``mx_rows``: ``(offset, routes, dict(gate=, up=, hidden=))``, the values before
    quantization; :func:`slab` places them)."""
    n, hidden_size = x_routes.shape
    experts = (weights[0] if mx is None else mx["forward"][0][0]).shape[0]
    device = x_routes.device
    y_route = torch.zeros(n, hidden_size, dtype=dtype, device=device)
    dx_route = torch.zeros_like(y_route)
    d_scores = torch.zeros(n, dtype=dtype, device=device)
    alt_d_scores = torch.zeros_like(d_scores) if dtype == NAIVE else None
    if mx is None:
        shapes = [w.shape for w in weights]
    else:
        fwd = [dequantized(p) for p in mx["forward"]]
        # The backward GEMMs read gate and up quantized along the intermediate dimension and down along hidden (the
        # transposed pairs), here in the weights' own orientation.
        gate_b, up_b, down_b = mx["backward"]
        bwd = [
            dequantized(gate_b[2:], transposed=True),
            dequantized(up_b[2:], transposed=True),
            dequantized(down_b, transposed=True),
        ]
        shapes = [w.shape for w in fwd]
    dw = [torch.zeros(s, dtype=dtype, device=device) for s in shapes]
    alt_dw = [torch.zeros(s, dtype=dtype, device=device) for s in shapes] if macro and dtype == NAIVE else None
    mx_rows = []
    with cublaslt():
        for seg in segments:
            routes = seg.routes.to(device)
            valid = routes >= 0
            if not bool(valid.any()):
                continue
            e = seg.expert
            x = _rows(routes, x_routes)
            s = _rows(routes, s_routes.float())
            dy = _rows(routes, dy_routes)
            if dtype == ORACLE:
                dys = dy.double() * s.double()[:, None]
            else:
                dys = (dy.float() * s[:, None]).bfloat16()
            if mx is None:
                out = mlp(x, [w[e] for w in weights], dys, limit, dtype)
                y, dx, dws = out["y"], out["dx"], out["dw"]
                hidden_value, dh = out["hidden"], out["dh"]
                d_gate, d_up, x_t = out["d_gate"], out["d_up"], x.to(dtype)
            else:
                y, dx, dws, extra = _mx_expert(x, dys, [w[e] for w in fwd], [w[e] for w in bwd], limit, dtype)
                hidden_value, dh = extra["hidden_ds"], extra["dh"]
                d_gate, d_up, x_t = extra["d_gate_t"], extra["d_up_t"], extra["x_t"]
                mx_rows.append((seg.offset, routes, extra["rows"]))
            index = routes[valid].long()
            y_route[index] = y[valid]
            dx_route[index] = dx[valid]
            # d_scores of the standard form: (dY * y_route).sum(-1), dY the unweighted upstream gradient.
            d_scores[index] = (dy.to(dtype) * y.to(dtype)).sum(-1)[valid]
            for dest, grad in zip(dw, dws, strict=True):
                dest[e] = grad
            if alt_d_scores is not None:
                inverse = torch.where(s > 0, 1 / s, torch.zeros_like(s))
                alt = ((dh * hidden_value).sum(-1).float() * inverse).to(dtype)
                alt_d_scores[index] = alt[valid]
            if alt_dw is not None:
                rows = x.shape[0]
                dys_t = extra["dys_t"] if mx is not None else dys
                hid_t = extra["hidden_t"] if mx is not None else hidden_value
                products = (
                    lambda a, b: (d_gate[a:b].T @ x_t[a:b]),
                    lambda a, b: (d_up[a:b].T @ x_t[a:b]),
                    lambda a, b: (dys_t[a:b].T @ hid_t[a:b]),
                )
                for dest, product in zip(alt_dw, products, strict=True):
                    dest[e] = _macro_sum(product, rows, seg.offset, macro, dtype)
    result = dict(y_route=y_route, dx_route=dx_route, d_scores=d_scores, dw=dw)
    if alt_dw is not None:
        result["alt_dw"] = alt_dw
    if alt_d_scores is not None:
        result["alt_d_scores"] = alt_d_scores
    if mx is not None:
        result["mx_rows"] = mx_rows
    return result


def _mx_expert(x, dys, fwd, bwd, limit, dtype):
    """One MXFP8 routed expert on its padded segment (rows in schedule order): forward and backward in ``dtype``.

    ``fwd``: the dequantized forward weights ``(gate, up, down)`` (blocks along K); ``bwd``: the dequantized backward
    weights in the same orientation (gate and up quantized along the intermediate dimension, down along hidden),
    FP32 values."""
    oracle = dtype == ORACLE
    x_n, x_t = qdq(x), qdq(x, transposed=True)
    xn = x_n.to(dtype)
    xt = x_t.to(dtype)
    wg, wu, wd = (w.to(dtype) for w in fwd)
    wg_b, wu_b, wd_b = (w.to(dtype) for w in bwd)
    gate, up = xn @ wg.T, xn @ wu.T
    gc, uc = _clamped(gate, up, limit)
    hidden = F.silu(gc) * uc
    if oracle:
        y = hidden @ wd.T
        dh = dys @ wd_b
        g, u = gate, up
    else:
        y = qdq(hidden) @ wd.T
        dh = qdq(dys) @ wd_b
        # The backward reads the saved gate and up through the recipe.
        g, u = qdq(gate), qdq(up)
    g_c, u_c = _clamped(g, u, limit)
    d_silu = dh * u_c
    silu = F.silu(g_c)
    d_up = dh * silu
    d_gate = torch.ops.aten.silu_backward(d_silu, g_c)
    if limit is not None:
        d_gate = d_gate * (g <= limit).to(dtype)
        d_up = d_up * ((u >= -limit) & (u <= limit)).to(dtype)
    if oracle:
        dx = d_gate @ wg_b + d_up @ wu_b
        dg_t, du_t, dys_t, hidden_t = d_gate, d_up, dys, hidden
    else:
        dx = qdq(d_gate) @ wg_b + qdq(d_up) @ wu_b
        dg_t, du_t = qdq(d_gate, transposed=True), qdq(d_up, transposed=True)
        dys_t = qdq(dys, transposed=True)
        hidden_t = qdq(hidden, transposed=True)
    dws = [dg_t.T @ xt, du_t.T @ xt, dys_t.T @ hidden_t]
    extra = dict(
        dh=dh,
        hidden_ds=(silu * u_c),
        d_gate_t=dg_t,
        d_up_t=du_t,
        x_t=xt,
        dys_t=dys_t,
        hidden_t=hidden_t,
        rows=dict(gate=gate, up=up, hidden=hidden),
    )
    return y, dx, dws, extra


def slab(mx_rows, name, rows):
    """The per-row values ``name`` (``gate``, ``up`` or ``hidden``) of :func:`routed`'s ``mx_rows`` placed at their
    slab rows: ``[rows, ...]``, zero where no segment lies."""
    _, _, first = mx_rows[0]
    out = torch.zeros(rows, first[name].shape[-1], dtype=first[name].dtype, device=first[name].device)
    for offset, _, values in mx_rows:
        out[offset : offset + values[name].shape[0]] = values[name]
    return out


def shared(x, dy, weights, limit, dtype):
    """The BF16 shared expert over every local token in ``dtype``: :func:`mlp` on the unweighted ``dY``."""
    return mlp(x, weights, dy, limit, dtype)


def combine(shared_y, shared_dx, route, scores, dtype):
    """The layer's ``y`` and ``dx`` from the shared expert's ``y`` / ``dx`` (``[T, hidden]``) and the routed experts'
    per-route values of :func:`routed` (``y_route``, ``dx_route``), with the FP32 ``scores`` ``[T, topk]``: ``y = shared
    + sum_k s_k y_route_k`` (BF16 routed outputs times FP32 scores promote to FP32; the sum is rounded to BF16 once)
    and ``dx = shared_dx + sum_k dx_route_k`` (BF16)."""
    t, topk = scores.shape
    if dtype == ORACLE:
        y = shared_y + (route["y_route"].view(t, topk, -1) * scores.double()[..., None]).sum(1)
    else:
        y = (shared_y + (route["y_route"].view(t, topk, -1) * scores[..., None]).sum(1)).to(dtype)
    dx = shared_dx + route["dx_route"].view(t, topk, -1).sum(1)
    return y, dx


def layer(x, dy, scores, shared_w, routed_w, segments, limit, dtype, *, sections=None, mx=None, macro=None):
    """The whole layer over the tokens of a group: ``x``, ``dy`` ``[T, hidden]``, ``scores`` ``[T, topk]``,
    ``segments`` per routed expert of the group (route index ``t * topk + k``; every route in one segment).  The
    shared expert runs per source rank (``sections``: the token count of each rank, default one rank), so its weight
    gradients are per rank, stacked ``[ranks, ...]``.  Returns ``y``, ``dx``, ``d_scores``, ``dw`` (routed gate, up,
    down over every expert, then the stacked shared gate, up, down), the per-route values and the alternatives of
    :func:`routed`."""
    t, topk = scores.shape
    sections = [t] if sections is None else list(sections)
    r = routed(
        x.repeat_interleave(topk, 0), dy.repeat_interleave(topk, 0), scores.flatten(), segments, routed_w, limit,
        dtype, mx=mx, macro=macro,
    )  # fmt: skip
    parts, start = [], 0
    for n in sections:
        parts.append(shared(x[start : start + n], dy[start : start + n], shared_w, limit, dtype))
        start += n
    y, dx = combine(
        torch.cat([p["y"] for p in parts]), torch.cat([p["dx"] for p in parts]), r, scores, dtype
    )
    out = dict(
        y=y,
        dx=dx,
        d_scores=r["d_scores"].view(t, topk),
        dw=[*r["dw"], *(torch.stack([p["dw"][i] for p in parts]) for i in range(3))],
    )
    if "alt_dw" in r:
        out["alt_dw"] = r["alt_dw"]
    if "alt_d_scores" in r:
        out["alt_d_scores"] = r["alt_d_scores"].view(t, topk)
    return out


def epilogue_forward(shared_y, routed_y, scores, dtype):
    """The forward epilogue ``y = shared + sum_k s_k routed_k`` (``routed_y``: ``[T * topk, hidden]``)."""
    t, topk = scores.shape
    if dtype == ORACLE:
        return shared_y.double() + (routed_y.double().view(t, topk, -1) * scores.double()[..., None]).sum(1)
    return (shared_y + (routed_y.view(t, topk, -1) * scores[..., None]).sum(1)).to(dtype)
