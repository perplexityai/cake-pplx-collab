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

Inputs, oracles and package calls of the pre-handoff suites of the two mHC training packages
(``tests/experimental/test_cake_mhc_nondet_train_handoff*.py`` and ``tests/mhc/test_cake_mhc_det_train_handoff.py``).

Every case runs at the DeepSeek-V4.1-Flash constants: hidden :data:`HIDDEN` 5120, ``rms_eps`` 1e-20, ``hc_eps`` 1e-6
and 20 Sinkhorn iterations (:data:`CONSTANTS`, passed to the package and to the oracle alike).

* :func:`make_inputs`: one sublayer's inputs, ``X, fn, scale, base, B, dY, dU`` as the family's generator makes them
  plus ``pre_in`` (``sigmoid(N(0, 1)) + 1e-6``, or one-hot on stream 0) and ``dpre_next`` (``sqrt(H) N(0, 1)``, the
  size of a real ``dpre_in``), each gradient optionally zero.
* :func:`collapse_oracle`: the standalone collapse ``U = sum_i pre_in_i X_i`` and its VJP in a dtype.
* :func:`chain_oracle`: sublayers from the one-hot start through the final collapse, the issue's reference structure
  (branch ``tanh(U W)`` per sublayer, distinct parameters), and the gradients of an upstream ``G`` on the final ``U``
  by autograd, in a dtype (float64: the oracle; BF16: the naive tier).
* :class:`Fp32Gate`: the ``dpre_in`` criterion: the relative L2 error against the oracle is at most
  :data:`DPRE_IN_FACTOR` x that of the IEEE-FP32 evaluation of the reference ``(dU X_i).sum(-1)`` on the CPU (median
  and largest over the draws).
* :class:`Api`: one package's entry points under their mode's names.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import torch

from tests.cake.test_helpers.cake_mhc_naive import (
    BF16,
    F32,
    F64,
    MAPPING_ROWS,
    NAIVE_DTYPE,
    STREAMS,
    _sinkhorn,
)

HIDDEN = 5120
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-20, 1e-6, 20
CONSTANTS = {"rms_eps": RMS_EPS, "hc_eps": HC_EPS, "sinkhorn_iters": SINKHORN_ITERS}
#: The ``dpre_in`` criterion of the issue: at most this factor x the relative L2 error of the IEEE-FP32 evaluation.
DPRE_IN_FACTOR = 2.0
#: The device of the IEEE-FP32 evaluation of the ``dpre_in`` reference (pinned: torch's CPU sum).
DPRE_IN_REFERENCE_DEVICE = "cpu"
BLOCK_TOKENS = 8192
#: The judged outputs of one handoff sublayer (the forward, the statistics, the gradients).
SUBLAYER_OUTPUTS = (
    "U", "Y", "dB", "dX", "r", "z", "pre", "post", "C", "dpost", "dC", "dfn", "dscale", "dbase", "pre_next", "dpre_in",
)  # fmt: skip
COLLAPSE_OUTPUTS = ("U", "dX", "dpre_in")
PRE_COLUMNS = (25, 29)


# ----------------------------------------------------------------------------------------------------------------
# The package


class Api:
    """The entry points of one package (``mode`` ``"nondet"`` or ``"det"``) under their names
    ``cake_mhc_<mode>_train_<name>`` (``api.handoff_prepare`` is ``cake_mhc_<mode>_train_handoff_prepare``)."""

    def __init__(self, pkg: ModuleType, loader: ModuleType, mode: str) -> None:
        self.pkg, self.loader, self.mode = pkg, loader, mode

    def __getattr__(self, name: str):
        return getattr(self.pkg, f"cake_mhc_{self.mode}_train_{name}")


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def make_inputs(
    T: int,
    *,
    device: torch.device,
    seed: int,
    mode: str = "unit",
    pre_in: str = "random",
    dpre_next: str = "random",
    dY: str = "random",
    dU: str = "random",
    fn_scale: float = 1.0,
    zero_rows: Sequence[int] = (),
    hidden: int = HIDDEN,
) -> dict:
    """One sublayer's inputs: ``X, B, dY, dU ~ N(0, 1)`` BF16; ``fn ~ 1e-2 N(0, 1)`` (``1e-4`` in the ``tail`` and
    ``cancel`` regimes, times ``fn_scale``); ``scale ~ 1 + 0.1 N``; ``base ~ 0.1 N`` (``base[:8] = 20`` in ``tail``:
    saturated ``pre`` / ``post`` sigmoids); ``pre_in`` FP32 ``sigmoid(N(0, 1)) + 1e-6`` (``"one_hot"``: ``[1, 0, 0,
    0]``); ``dpre_next`` FP32 ``sqrt(H) N(0, 1)``; ``dY`` / ``dU`` / ``dpre_next`` ``"zero"``: zeros; the tokens of
    ``zero_rows`` zeroed in ``X``.  In ``cancel`` ``dU := BF16(-dX_0 / pre_in_0)`` with ``dX_0`` the FP64 ``dX`` of
    stream 0 at ``dU = 0``, so the collapse term of stream 0 cancels the residual-mixing and coefficient terms (exact in
    the handoff: ``dU`` enters ``dX`` through ``pre_in_i dU`` only).  The case carries :data:`CONSTANTS`."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=F32).to(dtype)

    fs = {"unit": 1e-2, "tail": 1e-4, "cancel": 1e-4}[mode]
    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = (randn(MAPPING_ROWS, STREAMS * hidden) * (fs * fn_scale)).contiguous()
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    if mode == "tail":
        base[:8] = 20.0
    B = randn(T, hidden, dtype=BF16)
    dYv = randn(T, STREAMS, hidden, dtype=BF16)
    dUv = randn(T, hidden, dtype=BF16)
    pin = torch.sigmoid(randn(T, STREAMS)) + 1e-6
    dpn = randn(T, STREAMS) * math.sqrt(hidden)
    if pre_in == "one_hot":
        pin = torch.zeros(T, STREAMS, dtype=F32, device=device)
        pin[:, 0] = 1.0
    if dpre_next == "zero":
        dpn = torch.zeros_like(dpn)
    if dY == "zero":
        dYv = torch.zeros_like(dYv)
    if dU == "zero":
        dUv = torch.zeros_like(dUv)
    if zero_rows:
        X[list(zero_rows)] = 0
    i = {
        "X": X, "fn": fn, "scale": scale, "base": base, "B": B, "dY": dYv, "dU": dUv,
        "pre_in": pin.contiguous(), "dpre_next": dpn.contiguous(), "mode": mode, **CONSTANTS,
    }  # fmt: skip
    if mode == "cancel" and T > 0:
        from tests.cake.test_helpers.cake_mhc_naive import case_oracle

        dX0 = case_oracle({**i, "dU": torch.zeros_like(dUv)})["dX"][:, 0, :]
        i["dU"] = (-dX0 / pin.double()[:, 0:1]).to(BF16).contiguous()
    return i


def one_hot(T: int, device: torch.device) -> torch.Tensor:
    pre = torch.zeros(T, STREAMS, dtype=F32, device=device)
    pre[:, 0] = 1.0
    return pre


# ----------------------------------------------------------------------------------------------------------------
# Oracles


def collapse_oracle(
    X: torch.Tensor, pre_in: torch.Tensor, dU: torch.Tensor, *, dtype: torch.dtype
) -> dict:
    """The standalone collapse and its VJP in ``dtype``: ``U = sum_i pre_in_i X_i``, ``dX_i = pre_in_i dU``,
    ``dpre_in_i = sum_d dU_d X_id`` (values before any cast)."""
    out = {k: [] for k in COLLAPSE_OUTPUTS}
    for lo in range(0, max(int(X.shape[0]), 1), BLOCK_TOKENS):
        x = X[lo : lo + BLOCK_TOKENS].to(dtype)
        p = pre_in[lo : lo + BLOCK_TOKENS].to(dtype)
        u = dU[lo : lo + BLOCK_TOKENS].to(dtype)
        out["U"].append((p.unsqueeze(-1) * x).sum(dim=1))
        out["dX"].append(p.unsqueeze(-1) * u.unsqueeze(1))
        out["dpre_in"].append((u.unsqueeze(1) * x).sum(dim=-1))
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def collapse_expected_and_naive(i: Mapping) -> tuple[dict, dict]:
    want = collapse_oracle(i["X"], i["pre_in"], i["dU"], dtype=F64)
    tier = collapse_oracle(i["X"], i["pre_in"], i["dU"], dtype=NAIVE_DTYPE)
    return want, {k: v.to(NAIVE_DTYPE) for k, v in tier.items()}


def _coefficients(X, fn, scale, base, *, rms_eps, hc_eps, iters):
    """``(pre, post, C)`` of ``X [T, 4, H]`` in ``X``'s dtype, the family's op order ``((u r) fn^T)``."""
    T, S, H = X.shape
    u = X.reshape(T, S * H)
    r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)
    z = (u * r) @ fn.T
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn((z[:, 8:] * scale[2] + base[8:]).reshape(T, S, S), hc_eps, iters)
    return pre, post, C


def chain_inputs(
    T: int, *, device: torch.device, seed: int, sublayers: int = 2, hidden: int = HIDDEN
) -> dict:
    """The inputs of a chain of ``sublayers`` sublayers: ``X0 ~ N(0, 1)`` BF16 ``[T, 4, H]``; per sublayer distinct
    ``fn ~ 1e-2 N``, ``scale ~ 1 + 0.1 N``, ``base ~ 0.1 N`` (FP32) and a branch weight ``W ~ N(0, 1 / H)`` BF16
    ``[H, H]``; the upstream gradient ``G ~ N(0, 1)`` BF16 ``[T, H]`` of the final collapse."""
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=F32).to(dtype)

    params, weights = [], []
    for _ in range(sublayers):
        params.append(
            (
                (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
                1.0 + 0.1 * randn(3),
                0.1 * randn(MAPPING_ROWS),
            )
        )
        weights.append((randn(hidden, hidden) / math.sqrt(hidden)).to(BF16))
    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16),
        "params": params,
        "W": weights,
        "G": randn(T, hidden, dtype=BF16),
    }


def chain_names(sublayers: int) -> tuple[str, ...]:
    """The judged outputs of a chain: the final ``U``, each sublayer's ``pre_next``, ``dX0`` and each sublayer's
    parameter gradients."""
    pres = tuple(f"pre{k + 1}" for k in range(sublayers))
    grads = tuple(
        f"{n}{k + 1}" for k in range(sublayers) for n in ("dfn", "dscale", "dbase")
    )
    return ("U_final", *pres, "dX0", *grads)


def chain_oracle(i: Mapping, *, dtype: torch.dtype) -> dict:
    """The chain of :func:`chain_inputs` in ``dtype`` from the one-hot start: per sublayer ``(pre_next, post, C)`` of
    its input, ``U = sum_i pre_in_i X_i``, ``B = tanh(U W)``, ``Y = post B + sum_i C_ij X_i``, the next sublayer's
    ``pre_in = pre_next``; then ``U_final = sum_i pre_last_i X_last_i`` and the gradients of ``<U_final, G>`` with
    respect to ``X0`` and every parameter (:func:`chain_names`)."""
    T = int(i["X"].shape[0])
    with torch.enable_grad():
        X0 = i["X"].detach().to(dtype).requires_grad_(True)
        params = [
            [p.detach().to(dtype).requires_grad_(True) for p in ps]
            for ps in i["params"]
        ]
        X, pre_in = X0, one_hot(T, X0.device).to(dtype)
        pres = []
        for ps, W in zip(params, i["W"], strict=True):
            pre_next, post, C = _coefficients(
                X, *ps, rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=SINKHORN_ITERS
            )
            U = (pre_in.unsqueeze(-1) * X).sum(dim=1)
            B = torch.tanh(U @ W.to(dtype))
            X = post.unsqueeze(-1) * B.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, X)
            pre_in = pre_next
            pres.append(pre_next)
        U_final = (pre_in.unsqueeze(-1) * X).sum(dim=1)
        leaves = [X0, *(p for ps in params for p in ps)]
        grads = torch.autograd.grad(U_final, leaves, i["G"].to(dtype))
    names = chain_names(len(params))
    values = [U_final.detach(), *(p.detach() for p in pres), *grads]
    return dict(zip(names, values, strict=True))


def chain_expected_and_naive(i: Mapping) -> tuple[dict, dict]:
    want = chain_oracle(i, dtype=F64)
    tier = chain_oracle(i, dtype=NAIVE_DTYPE)
    return want, {k: v.to(NAIVE_DTYPE) for k, v in tier.items()}


# ----------------------------------------------------------------------------------------------------------------
# The dpre_in criterion


def dpre_in_fp32(X: torch.Tensor, dU: torch.Tensor) -> torch.Tensor:
    """The IEEE-FP32 evaluation of the issue's ``dpre_in = (dU.unsqueeze(1) * X).sum(-1)`` on
    :data:`DPRE_IN_REFERENCE_DEVICE` (in blocks of :data:`BLOCK_TOKENS` tokens)."""
    parts = []
    for lo in range(0, max(int(X.shape[0]), 1), BLOCK_TOKENS):
        x = X[lo : lo + BLOCK_TOKENS].to(DPRE_IN_REFERENCE_DEVICE, F32)
        u = dU[lo : lo + BLOCK_TOKENS].to(DPRE_IN_REFERENCE_DEVICE, F32)
        parts.append((u.unsqueeze(1) * x).sum(dim=-1))
    return torch.cat(parts, dim=0)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``||actual - expected|| / ||expected||`` in float64 (0 when both are zero, inf when only ``expected`` is)."""
    a = actual.detach().to("cpu", F64)
    e = expected.detach().to("cpu", F64)
    den = float(e.norm())
    num = float((a - e).norm())
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


class Fp32Gate:
    """The ``dpre_in`` criterion over the draws of a case: per draw the relative L2 error of the kernel's ``dpre_in``
    (the largest over the draw's launches) and of :func:`dpre_in_fp32`, both against the float64 oracle; the median
    and the largest of the kernel's errors are at most :data:`DPRE_IN_FACTOR` x those of the FP32 evaluation."""

    def __init__(self, name: str = "dpre_in") -> None:
        self.name = name
        self.ours: dict[int, float] = {}
        self.fp32: dict[int, float] = {}

    def add(
        self,
        draw: int,
        actual: torch.Tensor,
        X: torch.Tensor,
        dU: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        ours = relative_l2(actual, expected)
        self.ours[draw] = max(self.ours.get(draw, 0.0), ours)
        if draw not in self.fp32:
            self.fp32[draw] = relative_l2(dpre_in_fp32(X, dU), expected)

    def verdict(self) -> dict:
        ours, fp32 = list(self.ours.values()), list(self.fp32.values())
        return {
            "output": self.name,
            "draws": len(ours),
            "ours": [statistics.median(ours), max(ours)],
            "fp32": [statistics.median(fp32), max(fp32)],
            "factor": DPRE_IN_FACTOR,
        }

    def check(
        self, record_property: Callable[[str, object], None] | None, draws: int
    ) -> None:
        v = self.verdict()
        if record_property is not None:
            record_property("dpre_in_fp32_gate", v)
        assert v["draws"] == draws, (
            f"{self.name}: gated on {v['draws']} of {draws} draws"
        )
        (om, ox), (fm, fx) = v["ours"], v["fp32"]
        assert om <= DPRE_IN_FACTOR * fm and ox <= DPRE_IN_FACTOR * fx, (
            f"{self.name}: relative L2 error median / largest {om:.3e} / {ox:.3e} exceeds {DPRE_IN_FACTOR} x "
            f"the IEEE-FP32 evaluation's {fm:.3e} / {fx:.3e} over {v['draws']} draws"
        )


# ----------------------------------------------------------------------------------------------------------------
# Package calls


def sublayer_forward_backward(
    api: Api, i: Mapping, *, fwd_ws=None, bwd_ws=None, outputs: Mapping | None = None
) -> dict:
    """One handoff sublayer through the low-level entry points: ``handoff_forward_prepare`` (``U``, ``pre_next``,
    ``stats``), ``forward_finish`` (``Y``), ``backward_prepare`` (``dB``, ``gstats``) and ``handoff_backward_finish``
    (``dX``, ``dpre_in``, ``dfn``, ``dscale``, ``dbase``); ``outputs`` (caller buffers of those names) are passed as
    the ``out`` / ``pre_next_out`` / ``stats_out`` and gradient buffers."""
    o = dict(outputs or {})
    fwd = {
        "out": o.get("U"),
        "pre_next_out": o.get("pre_next"),
        "stats_out": o.get("stats"),
    }
    U, pre_next, stats = api.handoff_forward_prepare(
        i["X"], i["pre_in"], i["fn"], i["scale"], i["base"], **CONSTANTS, workspace=fwd_ws,
        **{k: v for k, v in fwd.items() if v is not None},
    )  # fmt: skip
    Y = api.forward_finish(i["X"], i["B"], stats)
    dB, gstats = api.backward_prepare(i["dY"], i["B"], i["X"], stats)
    bwd = {
        k: o[k]
        for k in ("dX", "dpre_in", "dfn", "dscale", "dbase")
        if o.get(k) is not None
    }
    dX, dpre_in, dfn, dscale, dbase = api.handoff_backward_finish(
        i["X"], i["dY"], i["dU"], i["pre_in"], i["dpre_next"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
        **CONSTANTS, workspace=bwd_ws, **bwd,
    )  # fmt: skip
    return {
        "U": U, "pre_next": pre_next, "stats": stats, "Y": Y, "dB": dB, "gstats": gstats,
        "dX": dX, "dpre_in": dpre_in, "dfn": dfn, "dscale": dscale, "dbase": dbase,
    }  # fmt: skip


def nan_outputs(T: int, device: torch.device, hidden: int = HIDDEN) -> dict:
    """Caller buffers holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")

    def full(shape, dtype):
        return torch.full(shape, nan, dtype=dtype, device=device)

    return {
        "U": full((T, hidden), BF16), "pre_next": full((T, STREAMS), F32), "stats": full((T, 64), F32),
        "dX": full((T, STREAMS, hidden), BF16), "dpre_in": full((T, STREAMS), F32),
        "dfn": full((MAPPING_ROWS, STREAMS * hidden), F32), "dscale": full((3,), F32),
        "dbase": full((MAPPING_ROWS,), F32),
    }  # fmt: skip


def autograd_sublayer(
    api: Api, i: Mapping, *, save_stats: bool = True, leaves: bool = True
) -> dict:
    """One handoff sublayer through the autograd entry points, ``U`` and ``B`` taken as given (the branch outside):
    ``handoff_prepare`` and ``finish``, then ``autograd.backward([U, Y, pre_next], [dU, dY, dpre_next])``: the
    forward outputs and the gradients of ``X``, ``pre_in``, ``fn``, ``scale``, ``base`` and ``B``."""
    X = i["X"].detach().clone().requires_grad_(True)
    pre_in = i["pre_in"].detach().clone().requires_grad_(True)
    params = [
        i[k].detach().clone().requires_grad_(True) for k in ("fn", "scale", "base")
    ]
    B = i["B"].detach().clone().requires_grad_(True)
    U, pre_next, handle = api.handoff_prepare(
        X, pre_in, *params, save_stats=save_stats, **CONSTANTS
    )
    Y = api.finish(X, B, handle)
    torch.autograd.backward([U, Y, pre_next], [i["dU"], i["dY"], i["dpre_next"]])
    return {
        "U": U.detach(), "Y": Y.detach(), "pre_next": pre_next.detach(), "dX": X.grad, "dpre_in": pre_in.grad,
        "dfn": params[0].grad, "dscale": params[1].grad, "dbase": params[2].grad, "dB": B.grad,
    }  # fmt: skip


class _ZeroGrad(torch.autograd.Function):
    """Identity forward, zero backward: the consumer's ``dpre_in`` never reaches the producer (a coefficient VJP that
    ignores ``dpre_next``)."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return torch.zeros_like(g)


def _torch_pre(X, fn, scale, base):
    """A differentiable FP32 torch ``pre`` of ``X`` (the own-``pre`` mutation hands it on)."""
    pre, _post, _C = _coefficients(
        X.float(), fn, scale, base, rms_eps=RMS_EPS, hc_eps=HC_EPS, iters=1
    )  # ``pre`` does not depend on the Sinkhorn count
    return pre


MUTATIONS = ("own_pre", "detached", "ignores_dpre_next")


def chain_package(
    api: Api,
    i: Mapping,
    *,
    save_stats: bool = True,
    checkpointed: bool = False,
    region: int = 1,
    mutation: str | None = None,
) -> dict:
    """The chain of :func:`chain_inputs` through the package's autograd entry points: per sublayer
    ``handoff_prepare`` (``pre_in`` the previous ``pre_next``, one-hot first), the branch ``tanh(U W)`` and ``finish``;
    then ``collapse`` of the last streams with the last ``pre_next`` and ``U_final.backward(G)``.  With
    ``checkpointed`` each run of ``region`` consecutive sublayers is one non-reentrant checkpoint returning ``(Y,
    pre_next)``.  ``mutation`` (:data:`MUTATIONS`): ``own_pre`` collapses each sublayer with its own ``pre`` (the
    standard prepare) and hands on a torch ``pre``; ``detached`` detaches every handoff; ``ignores_dpre_next`` passes
    every handoff through a node whose backward is zero.  Returns the outputs of :func:`chain_names` and, under
    ``"leaves"`` / ``"pres"``, the leaves and the handed-on ``pre_next`` tensors."""
    from torch.utils.checkpoint import checkpoint

    T = int(i["X"].shape[0])
    X0 = i["X"].detach().clone().requires_grad_(True)
    params = [
        [p.detach().clone().requires_grad_(True) for p in ps] for ps in i["params"]
    ]

    def sublayer(X, pre_in, fn, scale, base, W):
        if mutation == "own_pre":
            U, handle = api.prepare(
                X, fn, scale, base, save_stats=save_stats, **CONSTANTS
            )
            pre_next = _torch_pre(X, fn, scale, base)
        else:
            U, pre_next, handle = api.handoff_prepare(
                X, pre_in, fn, scale, base, save_stats=save_stats, **CONSTANTS
            )
        Y = api.finish(X, torch.tanh(U @ W), handle)
        if mutation == "detached":
            pre_next = pre_next.detach()
        elif mutation == "ignores_dpre_next":
            pre_next = _ZeroGrad.apply(pre_next)
        return Y, pre_next

    def run(X, pre_in, *flat):
        for k in range(0, len(flat), 4):
            X, pre_in = sublayer(X, pre_in, *flat[k : k + 4])
        return X, pre_in

    X, pre_in = X0, one_hot(T, X0.device)
    pres = []
    for lo in range(0, len(params), region):
        flat = [
            t
            for k in range(lo, min(lo + region, len(params)))
            for t in (*params[k], i["W"][k])
        ]
        if checkpointed:
            X, pre_in = checkpoint(run, X, pre_in, *flat, use_reentrant=False)
        else:
            X, pre_in = run(X, pre_in, *flat)
        pres.append(pre_in)
    U_final = api.collapse(X, pre_in)
    U_final.backward(i["G"])
    out = {"U_final": U_final.detach()}
    if region == 1:
        out.update({f"pre{k + 1}": p.detach() for k, p in enumerate(pres)})
    out["dX0"] = X0.grad
    for k, ps in enumerate(params):
        for n, p in zip(("dfn", "dscale", "dbase"), ps, strict=True):
            out[f"{n}{k + 1}"] = p.grad
    out["leaves"] = [X0, *(p for ps in params for p in ps)]
    out["pres"] = pres
    return out


def assert_equal(got: Mapping, want: Mapping, names: Sequence[str], label: str) -> None:
    """``got[k]`` and ``want[k]`` hold the same bits for every ``k`` of ``names``."""
    for k in names:
        a, b = got[k], want[k]
        assert a.dtype == b.dtype and a.shape == b.shape, (label, k)
        assert torch.equal(
            a.view(torch.uint8) if a.numel() else a,
            b.view(torch.uint8) if b.numel() else b,
        ), f"{label}: {k} differs bitwise"
