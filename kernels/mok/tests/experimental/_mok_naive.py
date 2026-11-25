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
  ``d_scores = (dY * y_e).sum(-1)`` in BF16.  With MXFP8 routed experts it applies the MX recipe's quantization
  points and nothing else: ``x`` (both orientations), the SwiGLU activation (along K for the down GEMM, along the rows
  for its weight gradient), the saved gate and up the backward reads, the score-scaled ``dY`` and ``d_gate`` /
  ``d_up`` (both orientations).  An FP32 output gets the BF16 result widened exactly.
* The score gradient ``d_scores = (dY * y_e).sum(-1)`` (``dY`` the unweighted upstream gradient) reads ``y_e``, the
  routed expert's output as the backward holds it.  With BF16 routed experts that is ``y_route``.  With MXFP8 routed
  experts it is the SwiGLU of the gate and up the backward reads (in the naive implementation through the recipe, in
  the oracle unquantized), quantized along K in the naive implementation, times the backward's copy of the down weight
  (the pair quantized along the hidden dimension, which ``dh = dY_s @ Wd`` reads), in both evaluations; ``y_route``
  reads the forward's copy.  Both evaluations share this one formula.
* :data:`ALTERNATIVE`: the kernel-shaped variants of the naive implementation, measured and reported beside the judged
  tier and never judged: BF16 weight gradients accumulated in BF16 per macrobatch (the kernel's order), and the score
  gradient as ``(dh * hidden).sum(-1) / s`` (the kernel's fused form).

Each routed expert's rows are given in the kernel's schedule order (a padded segment: ``-1`` rows are padding), since
the MX recipe's blocks along the rows and the macrobatches follow that order.

The router weight before the down projection (``weight_before_down``, the package's
``router_weight_position="before_down"``): a token's output is ``y = S(x) + sum_k E'_{e_k}(x)`` with
``E'(x) = BF16(s * a) Wd^T``, ``a`` the SwiGLU activation: the routed experts' outputs (``y_route``) carry their
scores and are summed unweighted, ``dY`` reaches the down projection unscaled, the activation's gradient is
``s * dh`` (``dh`` the gradient of the down projection's input) and ``d_scores = (dh * a).sum(-1)``, defined at a zero
score.  The naive implementation forms ``s * a`` as PyTorch does, the FP32 score promoting the BF16 activation and the
product cast to BF16 once, and takes ``(dh * a).sum(-1)`` of the two BF16 tensors in BF16 (the oracle all in
float64); it has no MXFP8 form and no kernel-shaped score alternative.

The per-token shared-expert scale (the package's ``shared_expert="scaled"``): ``y = s S(x) + sum_k w_k E_{e_k}(x)``
with an FP32 ``s`` per token.  The shared expert is one more weighted route: its backward reads ``BF16(dY * s)`` in the
naive implementation (the FP32 scale promoting the BF16 ``dY``, cast once, as autograd of ``s * S(x)`` gives it) and
``dY * s`` in the oracle, ``s S(x)`` is formed in FP32 with the routed terms before the one cast, and the scale's
gradient is ``d_s = (dY * S(x)).sum(-1)`` (BF16 in the naive implementation, as ``d_scores``), defined at ``s = 0``.
Without a shared expert (``shared_w=None``) ``y`` and ``dx`` are the routed sums alone.

The direct router-weight gradient (the package's ``router_weight_gradient="direct"``) has the same exact value as
``d_scores`` above; its kernel-shaped alternative ``(dh' * hidden).sum(-1)`` with ``dh' = dY Wd`` from the unscaled
``dY`` (no division) is reported beside the judged tier with ``direct=True`` and never judged.

SiTU-GLU (``activation=("situ_glu", beta, lb)``): the activation
``a = beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb)`` in place of SwiGLU.  The naive implementation is that
PyTorch spelling on the BF16 tensors with autograd in BF16.  The oracle evaluates it in float64 with the derivatives in
their stable forms, ``1 - tanh^2(z) = 1 / cosh^2(z)`` and ``1 - sigmoid(g) = sigmoid(-g)``: the forms
``1 - tanh * tanh`` and ``1 - sigmoid`` cancel to 0 in float64 for saturated gates (``|g|`` above about 37), where the
true gradient is a normal BF16 value.
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
    "d_scores_direct": "(dh' * hidden).sum(-1), dh' from the unscaled dY",
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


class _SiTUOracle(torch.autograd.Function):
    """The float64 SiTU-GLU ``beta tanh(g / beta) sigmoid(g) lb tanh(u / lb)`` whose gradients use the stable forms
    ``1 / cosh^2`` and ``sigmoid(-g)`` (module docstring); ``cosh`` overflowing to ``inf`` gives the exact limit 0."""

    @staticmethod
    def forward(ctx, gate, up, beta, lb):
        ctx.save_for_backward(gate, up)
        ctx.scalars = (beta, lb)
        return (
            beta
            * torch.tanh(gate / beta)
            * torch.sigmoid(gate)
            * (lb * torch.tanh(up / lb))
        )

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        beta, lb = ctx.scalars
        a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        u = lb * torch.tanh(up / lb)
        d_a = torch.cosh(gate / beta).pow(-2) * torch.sigmoid(gate) + a * torch.sigmoid(
            -gate
        )
        d_u = torch.cosh(up / lb).pow(-2)
        return grad * u * d_a, grad * a * d_u, None, None


def situ_glu(gate, up, beta, lb):
    """SiTU-GLU of ``gate`` and ``up`` in their dtype: the float64 oracle form (:class:`_SiTUOracle`) or, in any other
    dtype, the PyTorch spelling ``beta * tanh(gate / beta) * sigmoid(gate) * (lb * tanh(up / lb))`` with autograd."""
    if gate.dtype == ORACLE:
        return _SiTUOracle.apply(gate, up, float(beta), float(lb))
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * (lb * torch.tanh(up / lb))
    )


def _activation(gate, up, limit, activation):
    """The experts' activation: SwiGLU (clamped with ``limit``) or, with ``activation=("situ_glu", beta, lb)``,
    :func:`situ_glu`."""
    if activation is None:
        gc, uc = _clamped(gate, up, limit)
        return F.silu(gc) * uc
    name, beta, lb = activation
    if name != "situ_glu" or limit is not None:
        raise ValueError(
            "activation is None (SwiGLU) or ('situ_glu', beta, lb) without a limit"
        )
    return situ_glu(gate, up, beta, lb)


def mlp(x, weights, dy, limit, dtype, route_weight=None, activation=None):
    """One SwiGLU MLP and its backward in ``dtype``, by autograd: ``y``, ``dx``, ``d_gate``, ``d_up``, ``dh`` (the
    gradient of the down projection's input), ``hidden`` (that input) and the three weight gradients.  ``activation``
    (:func:`_activation`) replaces SwiGLU by SiTU-GLU.

    ``route_weight`` (``[rows]``; FP32 in the naive implementation, float64 in the oracle) multiplies the activation
    ``a`` before the down projection: ``hidden = (a * w).to(dtype)``, the FP32 weight promoting the BF16 ``a`` and one
    cast; autograd gives ``a`` the gradient ``w * dh`` cast once.  Then also ``a`` and the route weight's gradient
    ``d_route = (dh * a).sum(-1)`` in ``dtype``."""
    with torch.enable_grad(), cublaslt():
        x = x.to(dtype).detach().requires_grad_()
        w = [t.to(dtype).detach().requires_grad_() for t in weights]
        gate, up = x @ w[0].T, x @ w[1].T
        gate.retain_grad()
        up.retain_grad()
        hidden = _activation(gate, up, limit, activation)
        hidden.retain_grad()
        down_in = hidden
        if route_weight is not None:
            down_in = (hidden * route_weight.detach()[:, None]).to(dtype)
            down_in.retain_grad()
        y = down_in @ w[2].T
        y.backward(dy.to(dtype))
    out = dict(
        y=y.detach(),
        dx=x.grad,
        d_gate=gate.grad,
        d_up=up.grad,
        dh=down_in.grad,
        hidden=down_in.detach(),
        dw=[t.grad for t in w],
    )
    if route_weight is not None:
        out["a"] = hidden.detach()
        out["d_route"] = (down_in.grad * out["a"]).sum(-1)
    return out


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
    weight_before_down=False,
    activation=None,
    direct=False,
):
    """The routed experts on per-route inputs (``x_routes``, ``dy_routes``: ``[routes, hidden]`` BF16, the source row
    of each route; ``s_routes``: ``[routes]`` FP32), in ``dtype`` (:data:`ORACLE` or :data:`NAIVE`).

    ``weights``: the BF16 ``(gate, up, down)`` ``[experts, ...]``, or with ``mx`` the stored MXFP8 weights
    ``dict(forward=(gate, up, down) pairs, backward=(gate, up, down) pairs)`` as the package's quantizer returns them.
    Returns per route ``y_route``, ``dx_route``, ``d_scores`` (zero on routes no segment holds), the weight gradients
    ``dw`` (three ``[experts, ...]``) and, with ``macro``, the kernel-shaped alternatives ``alt_dw`` and
    ``alt_d_scores`` (:data:`ALTERNATIVE`; naive only).  With ``mx`` also the per-row values the kernel stores as
    E4M3 codes, per segment (``mx_rows``: ``(offset, routes, dict(gate=, up=, hidden=))``, the values before
    quantization; :func:`slab` places them).

    ``weight_before_down``: the score multiplies the activation before the down projection (module docstring):
    ``y_route`` is the weighted expert output, ``d_scores`` is ``(dh * a).sum(-1)`` and there is no ``alt_d_scores``;
    BF16 routed weights only.

    ``activation``: SiTU-GLU in place of SwiGLU (:func:`_activation`; BF16 routed weights, ``"after_down"``).
    ``direct``: the naive ``alt_d_scores_direct`` (:data:`ALTERNATIVE`) beside ``alt_d_scores``."""
    if weight_before_down and mx is not None:
        raise ValueError("The router weight before the down projection has no MXFP8 form")
    if activation is not None and (mx is not None or weight_before_down):
        raise ValueError("SiTU-GLU has BF16 routed weights and the router weight after the down projection only")
    if direct and (mx is not None or weight_before_down):
        raise ValueError("The direct score-gradient alternative has BF16 routed weights and 'after_down' only")
    n, hidden_size = x_routes.shape
    experts = (weights[0] if mx is None else mx["forward"][0][0]).shape[0]
    device = x_routes.device
    y_route = torch.zeros(n, hidden_size, dtype=dtype, device=device)
    dx_route = torch.zeros_like(y_route)
    d_scores = torch.zeros(n, dtype=dtype, device=device)
    alt_d_scores = torch.zeros_like(d_scores) if dtype == NAIVE and not weight_before_down else None
    alt_direct = torch.zeros_like(d_scores) if dtype == NAIVE and direct else None
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
            if weight_before_down:
                dys = dy.to(dtype)
            elif dtype == ORACLE:
                dys = dy.double() * s.double()[:, None]
            else:
                dys = (dy.float() * s[:, None]).bfloat16()
            if weight_before_down:
                route_weight = s.double() if dtype == ORACLE else s
                out = mlp(x, [w[e] for w in weights], dys, limit, dtype, route_weight=route_weight)
                y, dx, dws = out["y"], out["dx"], out["dw"]
                hidden_value, dh = out["hidden"], out["dh"]
                d_gate, d_up, x_t = out["d_gate"], out["d_up"], x.to(dtype)
            elif mx is None:
                out = mlp(x, [w[e] for w in weights], dys, limit, dtype, activation=activation)
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
            if weight_before_down:
                # The score multiplies a before the down projection: d_scores = (dh * a).sum(-1).
                d_scores[index] = out["d_route"][valid]
            else:
                # d_scores of the standard form: (dY * y_e).sum(-1), dY the unweighted upstream gradient and y_e the
                # expert's output as the backward holds it (with MXFP8 routed experts the backward's copy of Wd,
                # _mx_expert).
                y_e = y if mx is None else extra["y_e"]
                d_scores[index] = (dy.to(dtype) * y_e.to(dtype)).sum(-1)[valid]
            for dest, grad in zip(dw, dws, strict=True):
                dest[e] = grad
            if alt_d_scores is not None:
                inverse = torch.where(s > 0, 1 / s, torch.zeros_like(s))
                alt = ((dh * hidden_value).sum(-1).float() * inverse).to(dtype)
                alt_d_scores[index] = alt[valid]
            if alt_direct is not None:
                # dh' = BF16(dY Wd) from the unscaled dY, then the FP32 dot with hidden, no division.
                dh_direct = dy.to(dtype) @ weights[2][e].to(dtype)
                alt = (dh_direct.float() * hidden_value.float()).sum(-1).to(dtype)
                alt_direct[index] = alt[valid]
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
    if alt_direct is not None:
        result["alt_d_scores_direct"] = alt_direct
    if mx is not None:
        result["mx_rows"] = mx_rows
    return result


def _mx_expert(x, dys, fwd, bwd, limit, dtype):
    """One MXFP8 routed expert on its padded segment (rows in schedule order): forward and backward in ``dtype``.

    ``fwd``: the dequantized forward weights ``(gate, up, down)`` (blocks along K); ``bwd``: the dequantized backward
    weights in the same orientation (gate and up quantized along the intermediate dimension, down along hidden),
    FP32 values.  ``extra["y_e"]`` is the expert's output the score gradient reads: the SwiGLU of the gate and up the
    backward reads, quantized along K in the naive implementation, by the backward's ``Wd``."""
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
    hidden_ds = silu * u_c
    # The score gradient's y_e: the SwiGLU of the gate and up the backward reads by the backward's copy of Wd.
    y_e = (hidden_ds if oracle else qdq(hidden_ds)) @ wd_b.T
    extra = dict(
        dh=dh,
        hidden_ds=hidden_ds,
        y_e=y_e,
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


def shared(x, dy, weights, limit, dtype, scale=None):
    """The BF16 shared expert over every local token in ``dtype``: :func:`mlp` on the unweighted ``dY``, or with the
    FP32 per-token ``scale`` ``[T]`` on ``dY * scale`` (``BF16(dY * scale)`` in the naive implementation, as autograd of
    ``scale * S(x)`` with an FP32 ``scale`` gives it); ``y`` stays the unscaled ``S(x)``."""
    if scale is None:
        return mlp(x, weights, dy, limit, dtype)
    if dtype == ORACLE:
        dys = dy.double() * scale.double()[:, None]
    else:
        dys = (dy.float() * scale.float()[:, None]).to(dtype)
    return mlp(x, weights, dys, limit, dtype)


def combine(
    shared_y,
    shared_dx,
    route,
    scores,
    dtype,
    *,
    weight_before_down=False,
    shared_scale=None,
):
    """The layer's ``y`` and ``dx`` from the shared expert's ``y`` / ``dx`` (``[T, hidden]``) and the routed experts'
    per-route values of :func:`routed` (``y_route``, ``dx_route``), with the FP32 ``scores`` ``[T, topk]``: ``y = shared
    + sum_k s_k y_route_k`` (BF16 routed outputs times FP32 scores promote to FP32; the sum is rounded to BF16 once)
    and ``dx = shared_dx + sum_k dx_route_k`` (BF16).  With ``weight_before_down`` the routed outputs carry their
    scores: ``y = shared + sum_k y_route_k``, in the form of ``dx``.  With the FP32 ``shared_scale`` ``[T]`` the
    shared term is ``shared_scale * shared`` (FP32 in the naive implementation, before the one cast); ``shared_dx`` is
    then already the scaled expert's (:func:`shared` with ``scale``)."""
    t, topk = scores.shape
    if shared_scale is not None:
        if weight_before_down:
            raise ValueError("The shared-expert scale runs with the router weight after the down projection")
        scale = shared_scale.double() if dtype == ORACLE else shared_scale.float()
        shared_y = shared_y.to(scale.dtype) * scale[:, None]
    if weight_before_down:
        y = shared_y + route["y_route"].view(t, topk, -1).sum(1)
    elif dtype == ORACLE:
        y = shared_y + (route["y_route"].view(t, topk, -1) * scores.double()[..., None]).sum(1)
    else:
        y = (shared_y + (route["y_route"].view(t, topk, -1) * scores[..., None]).sum(1)).to(dtype)
    dx = shared_dx + route["dx_route"].view(t, topk, -1).sum(1)
    return y, dx


def layer(
    x, dy, scores, shared_w, routed_w, segments, limit, dtype, *, sections=None, mx=None, macro=None,
    weight_before_down=False, shared_scale=None, activation=None, direct=False,
):  # fmt: skip
    """The whole layer over the tokens of a group: ``x``, ``dy`` ``[T, hidden]``, ``scores`` ``[T, topk]``,
    ``segments`` per routed expert of the group (route index ``t * topk + k``; every route in one segment).  The
    shared expert runs per source rank (``sections``: the token count of each rank, default one rank), so its weight
    gradients are per rank, stacked ``[ranks, ...]``.  Returns ``y``, ``dx``, ``d_scores``, ``dw`` (routed gate, up,
    down over every expert, then the stacked shared gate, up, down), the per-route values and the alternatives of
    :func:`routed`.

    ``shared_scale`` (FP32 ``[T]``): the per-token shared-expert scale (module docstring); also returns
    ``d_shared_scale = (dY * S(x)).sum(-1)``.  ``shared_w=None``: no shared expert (``dw`` holds the routed three).
    ``activation`` (:func:`_activation`) and ``direct`` as in :func:`routed`."""
    t, topk = scores.shape
    sections = [t] if sections is None else list(sections)
    if shared_w is None and shared_scale is not None:
        raise ValueError("A shared-expert scale needs the shared expert")
    r = routed(
        x.repeat_interleave(topk, 0), dy.repeat_interleave(topk, 0), scores.flatten(), segments, routed_w, limit,
        dtype, mx=mx, macro=macro, weight_before_down=weight_before_down, activation=activation, direct=direct,
    )  # fmt: skip
    parts, start = [], 0
    if shared_w is not None:
        for n in sections:
            scale = None if shared_scale is None else shared_scale[start : start + n]
            parts.append(
                shared(
                    x[start : start + n],
                    dy[start : start + n],
                    shared_w,
                    limit,
                    dtype,
                    scale,
                )
            )
            start += n
        shared_y, shared_dx = (
            torch.cat([p["y"] for p in parts]),
            torch.cat([p["dx"] for p in parts]),
        )
    else:
        shared_y = shared_dx = torch.zeros(t, x.shape[1], dtype=dtype, device=x.device)
    y, dx = combine(
        shared_y, shared_dx, r, scores, dtype, weight_before_down=weight_before_down, shared_scale=shared_scale,
    )  # fmt: skip
    out = dict(
        y=y,
        dx=dx,
        d_scores=r["d_scores"].view(t, topk),
        dw=[
            *r["dw"],
            *(torch.stack([p["dw"][i] for p in parts]) for i in range(3) if parts),
        ],
    )
    if shared_scale is not None:
        out["d_shared_scale"] = (dy.to(dtype) * shared_y.to(dtype)).sum(-1)
    if "alt_dw" in r:
        out["alt_dw"] = r["alt_dw"]
    if "alt_d_scores" in r:
        out["alt_d_scores"] = r["alt_d_scores"].view(t, topk)
    if "alt_d_scores_direct" in r:
        out["alt_d_scores_direct"] = r["alt_d_scores_direct"].view(t, topk)
    return out


def epilogue_forward(shared_y, routed_y, scores, dtype, shared_scale=None):
    """The forward epilogue ``y = shared + sum_k s_k routed_k`` (``routed_y``: ``[T * topk, hidden]``), with the FP32
    ``shared_scale`` ``[T]`` ``y = shared_scale * shared + sum_k s_k routed_k``."""
    t, topk = scores.shape
    if dtype == ORACLE:
        base = shared_y.double()
        if shared_scale is not None:
            base = base * shared_scale.double()[:, None]
        return base + (
            routed_y.double().view(t, topk, -1) * scores.double()[..., None]
        ).sum(1)
    base = (
        shared_y
        if shared_scale is None
        else shared_y.float() * shared_scale.float()[:, None]
    )
    return (base + (routed_y.view(t, topk, -1) * scores[..., None]).sum(1)).to(dtype)
