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

The float64 oracle and the naive BF16 tier of the four-stream mHC training
function, and the loop that judges a package's outputs with
``tests/cake_naive.py`` over the draws of a case.

:func:`oracle` is the whole function in plain torch (``N = 4 H``)::

    u[t, :]      = X[t].reshape(N)
    r[t]         = rsqrt(mean_k u[t, k]^2 + rms_eps)
    z[t, n]      = sum_k (u[t, k] r[t]) fn[n, k]                          # [T, 24]
    a[t, n]      = z[t, n] scale[g(n)] + base[n]                           # g = 0 for n < 4, 1 for n < 8, else 2
    pre[t, i]    = sigmoid(a[t, i]) + hc_eps;   post[t, j] = 2 sigmoid(a[t, 4 + j])
    C            = row_softmax(a[t, 8:].reshape(4, 4)) + hc_eps; one column normalisation; (iters - 1) x (row, column)
    U = sum_i pre_i X_i;   Y[t, j] = post_j B + sum_i C_ij X_i

and its backward by autograd from ``dU`` and ``dY`` (``dX``, ``dfn``,
``dscale``, ``dbase``, ``dB``), plus the gradient statistics ``dpost_j =
<dY_j, B>`` and ``dC_ij = <X_i, dY_j>``.

The pre handoff (``pre_in`` given, FP32 ``[T, 4]``): the collapse takes the
incoming ``pre_in`` instead of this sublayer's ``pre``, ``U = sum_i pre_in_i
X_i``; this sublayer's ``pre`` is handed on as ``pre_next``, and the backward
runs autograd of ``[U, Y, pre]`` with ``[dU, dY, dpre_next]`` (``dpre_next``
``None``: zeros), which adds the gradient of ``pre_in`` (``dpre_in``) to the
outputs (:data:`HANDOFF_OUTPUTS`).  The naive tier rounds ``pre_in`` and
``dpre_next`` to BF16 like every other input.  Without ``pre_in`` the
function and its op sequence are the standard ones.  Evaluated in float64 it is the
oracle; evaluated in BF16 on the same BF16 inputs (``fn``, ``scale`` and
``base`` rounded to BF16, the same ops in BF16 with torch's own internal
accumulation) it is the naive tier ``"bf16"``.  The BF16 outputs ``U``, ``Y``,
``dB`` and ``dX`` are the values before their cast: the oracle never rounds an
output to the output's format.

Rows are evaluated in blocks of :data:`BLOCK_TOKENS` tokens; the per-token
values are concatenated and the parameter gradients are added in block order.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping

import torch

from tests.cake_naive import DRAWS, TIERS, TierJudge, contract, draw_seed, seed_all

STREAMS, MAPPING_ROWS = 4, 24
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
BLOCK_TOKENS = 8192
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
NAIVE_DTYPE = BF16
#: Every output :func:`oracle` holds, in the order the judge records them.
OUTPUTS = (
    "U", "Y", "dB", "dX", "r", "z", "pre", "post", "C", "dpost", "dC", "dfn", "dscale", "dbase",
)  # fmt: skip
#: The forward statistics record ``stats [T, 64]``: r | z[24] | pre[4] | post[4] | C[16] (i-major) | pad[15].
STATS_COLUMNS = {
    "r": (0, 1),
    "z": (1, 25),
    "pre": (25, 29),
    "post": (29, 33),
    "C": (33, 49),
}
#: The gradient statistics record ``gstats [T, 32]``: dpost[4] | dC[16] (i-major) | zeros[12].
GSTATS_COLUMNS = {"dpost": (0, 4), "dC": (4, 20)}
#: The outputs of the pre handoff (``pre_in`` given): :data:`OUTPUTS`, the ``pre`` handed on and the gradient of
#: ``pre_in``.
HANDOFF_OUTPUTS = (*OUTPUTS, "pre_next", "dpre_in")
PARAMETERS = ("dfn", "dscale", "dbase")
_PER_TOKEN = ("U", "Y", "dB", "dX", "r", "z", "pre", "post", "C", "dpost", "dC")
_PER_TOKEN_HANDOFF = (*_PER_TOKEN, "pre_next", "dpre_in")


def _sinkhorn(logits: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    """``softmax`` over j, ``+ hc_eps``, one column normalisation, then ``iters - 1`` x (row, column)."""
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _block(
    X,
    fn,
    scale,
    base,
    B,
    dY,
    dU,
    *,
    dtype,
    rms_eps,
    hc_eps,
    iters,
    pre_in=None,
    dpre_next=None,
) -> dict:
    """One row block of the function in ``dtype``: the forward, then the backward by autograd (the pre handoff with
    ``pre_in``; its gradient ``dpre_next``, ``None`` for zeros)."""
    T, S, H = X.shape
    with torch.enable_grad():
        Xd = X.detach().to(dtype).requires_grad_(True)
        fnd = fn.detach().to(dtype).requires_grad_(True)
        scaled = scale.detach().to(dtype).requires_grad_(True)
        based = base.detach().to(dtype).requires_grad_(True)
        Bd = B.detach().to(dtype).requires_grad_(True)
        u = Xd.reshape(T, S * H)
        r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + rms_eps)  # [T, 1]
        z = (u * r) @ fnd.T  # [T, 24]
        a_pre = z[:, 0:4] * scaled[0] + based[0:4]
        a_post = z[:, 4:8] * scaled[1] + based[4:8]
        a_logit = (z[:, 8:] * scaled[2] + based[8:]).reshape(T, STREAMS, STREAMS)
        pre = torch.sigmoid(a_pre) + hc_eps
        post = 2 * torch.sigmoid(a_post)
        C = _sinkhorn(a_logit, hc_eps, iters)
        if pre_in is None:
            U_pre = (pre.unsqueeze(-1) * Xd).sum(dim=1)
        else:
            pind = pre_in.detach().to(dtype).requires_grad_(True)
            U_pre = (pind.unsqueeze(-1) * Xd).sum(dim=1)
        Y_pre = post.unsqueeze(-1) * Bd.unsqueeze(1) + torch.einsum(
            "tij,tid->tjd", C, Xd
        )
        dUd = dU.detach().to(dtype)
        dYd = dY.detach().to(dtype)
        if pre_in is None:
            dX, dfn, dscale, dbase, dB = torch.autograd.grad(
                [U_pre, Y_pre], [Xd, fnd, scaled, based, Bd], grad_outputs=[dUd, dYd]
            )
        else:
            dpn = (
                torch.zeros_like(pre)
                if dpre_next is None
                else dpre_next.detach().to(dtype)
            )
            dX, dfn, dscale, dbase, dB, dpin = torch.autograd.grad(
                [U_pre, Y_pre, pre],
                [Xd, fnd, scaled, based, Bd, pind],
                grad_outputs=[dUd, dYd, dpn],
            )
    with torch.no_grad():
        Xd, Bd = Xd.detach(), Bd.detach()
        out = {
            "U": U_pre.detach(), "Y": Y_pre.detach(), "dB": dB, "dX": dX,
            "r": r.detach().reshape(T), "z": z.detach(), "pre": pre.detach(), "post": post.detach(), "C": C.detach(),
            "dpost": torch.einsum("tjd,td->tj", dYd, Bd), "dC": torch.einsum("tid,tjd->tij", Xd, dYd),
            "dfn": dfn, "dscale": dscale, "dbase": dbase,
        }  # fmt: skip
        if pre_in is not None:
            out["pre_next"], out["dpre_in"] = pre.detach(), dpin
        return out


def oracle(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    B: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    *,
    dtype: torch.dtype = F64,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    iters: int = SINKHORN_ITERS,
    block_tokens: int | None = BLOCK_TOKENS,
    pre_in: torch.Tensor | None = None,
    dpre_next: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Every output of :data:`OUTPUTS` of the function in ``dtype`` (float64: the oracle; bfloat16: the naive tier)
    on ``X [T, 4, H]``, ``fn [24, 4 H]``, ``scale [3]``, ``base [24]``, ``B [T, H]``, ``dY [T, 4, H]``, ``dU [T, H]``.
    ``U``, ``Y``, ``dB`` and ``dX`` are their values before the BF16 cast.  With ``pre_in [T, 4]`` the pre handoff
    (``dpre_next [T, 4]``, ``None`` for zeros): every output of :data:`HANDOFF_OUTPUTS`."""
    T, S, H = X.shape
    if T == 0:
        e = lambda *shape: torch.empty(shape, dtype=dtype, device=X.device)  # noqa: E731
        z = lambda *shape: torch.zeros(shape, dtype=dtype, device=X.device)  # noqa: E731
        return {
            "U": e(0, H), "Y": e(0, S, H), "dB": e(0, H), "dX": e(0, S, H), "r": e(0), "z": e(0, MAPPING_ROWS),
            "pre": e(0, S), "post": e(0, S), "C": e(0, S, S), "dpost": e(0, S), "dC": e(0, S, S),
            "dfn": z(MAPPING_ROWS, S * H), "dscale": z(3), "dbase": z(MAPPING_ROWS),
            **({"pre_next": e(0, S), "dpre_in": e(0, S)} if pre_in is not None else {}),
        }  # fmt: skip
    bt = T if not block_tokens else max(1, min(int(block_tokens), T))
    eps = {"rms_eps": rms_eps, "hc_eps": hc_eps, "iters": iters}

    def handoff(lo: int) -> dict:
        if pre_in is None:
            return {}
        return {
            "pre_in": pre_in[lo : lo + bt],
            "dpre_next": None if dpre_next is None else dpre_next[lo : lo + bt],
        }

    blocks = [
        _block(X[lo : lo + bt], fn, scale, base, B[lo : lo + bt], dY[lo : lo + bt], dU[lo : lo + bt], dtype=dtype,
               **eps, **handoff(lo))
        for lo in range(0, T, bt)
    ]  # fmt: skip
    if len(blocks) == 1:
        return blocks[0]
    per_token = _PER_TOKEN if pre_in is None else _PER_TOKEN_HANDOFF
    out = {k: torch.cat([b[k] for b in blocks], dim=0) for k in per_token}
    for k in PARAMETERS:
        acc = blocks[0][k].clone()
        for b in blocks[1:]:
            acc += b[k]
        out[k] = acc
    return out


def case_oracle(
    i: Mapping, *, dtype: torch.dtype = F64, iters: int | None = None
) -> dict[str, torch.Tensor]:
    """:func:`oracle` of the inputs ``i`` of a test case (``X, fn, scale, base, B, dY, dU``; its ``rms_eps``,
    ``hc_eps`` and ``sinkhorn_iters`` where it carries them, else the model constants; ``iters`` overrides the
    Sinkhorn count; its ``pre_in`` and ``dpre_next`` where it carries them: the pre handoff)."""
    count = iters if iters is not None else i.get("sinkhorn_iters", SINKHORN_ITERS)
    return oracle(
        i["X"], i["fn"], i["scale"], i["base"], i["B"], i["dY"], i["dU"], dtype=dtype,
        rms_eps=i.get("rms_eps", RMS_EPS), hc_eps=i.get("hc_eps", HC_EPS), iters=count,
        pre_in=i.get("pre_in"), dpre_next=i.get("dpre_next"),
    )  # fmt: skip


def stats_outputs(stats: torch.Tensor) -> dict[str, torch.Tensor]:
    """The per-token statistics of a ``stats [T, 64]`` record: ``r [T]``, ``z [T, 24]``, ``pre [T, 4]``,
    ``post [T, 4]``, ``C [T, 4, 4]``."""
    T = stats.shape[0]
    out = {
        name: stats[:, lo:hi].contiguous() for name, (lo, hi) in STATS_COLUMNS.items()
    }
    out["r"] = out["r"].reshape(T)
    out["C"] = out["C"].reshape(T, STREAMS, STREAMS)
    return out


def gstats_outputs(gstats: torch.Tensor) -> dict[str, torch.Tensor]:
    """The gradient statistics of a ``gstats [T, 32]`` record: ``dpost [T, 4]`` and ``dC [T, 4, 4]``."""
    T = gstats.shape[0]
    return {
        "dpost": gstats[:, 0:4].contiguous(),
        "dC": gstats[:, 4:20].contiguous().reshape(T, STREAMS, STREAMS),
    }


def package_outputs(res: Mapping) -> dict[str, torch.Tensor]:
    """The judged outputs of a package run: every name of :data:`HANDOFF_OUTPUTS` ``res`` holds (``U``, ``Y``,
    ``dB``, ``dX``, ``dfn``, ``dscale``, ``dbase``, ``pre_next``, ``dpre_in`` as returned; the columns of ``stats``
    and ``gstats``)."""
    out = {k: res[k] for k in HANDOFF_OUTPUTS if k in res and res[k] is not None}
    if res.get("stats") is not None:
        out.update(stats_outputs(res["stats"]))
    if res.get("gstats") is not None:
        out.update(gstats_outputs(res["gstats"]))
    return {k: out[k] for k in HANDOFF_OUTPUTS if k in out}


def expected_and_naive(
    i: Mapping, names: Iterable[str], *, iters: int | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The oracle's (float64) and the naive tier's (BF16) values of the outputs ``names`` on the inputs ``i``."""
    names = tuple(names)
    want = case_oracle(i, dtype=F64, iters=iters)
    tier = case_oracle(i, dtype=NAIVE_DTYPE, iters=iters)
    return {k: want[k] for k in names}, {k: tier[k].to(NAIVE_DTYPE) for k in names}


def judge_draws(
    case: Callable[
        [int, int], tuple | Iterable[tuple[Iterable[Mapping], Mapping, Mapping]]
    ],
    record_property: Callable[[str, object], None] | None,
    *,
    seed: int,
    draws: int = DRAWS,
    prefix: str = "",
) -> TierJudge:
    """Judge one case over ``draws`` inputs with ``tests/cake_naive.py``.

    For each draw ``j``, :func:`~tests.cake_naive.seed_all` seeds the random state with
    :func:`~tests.cake_naive.draw_seed` of ``seed`` and ``j``; ``case(j, s)`` (``s`` that seed) makes the draw's
    inputs, launches the package and returns ``(launches, expected, naive)``: the outputs of every launch on those
    inputs (a list of name -> tensor), the oracle's value of each judged output and the naive tier's (BF16).  Every
    launch's every output is held to its contract and recorded in the judge (a second launch of a draw keeps the
    larger of the errors).  ``case`` may instead yield such triples one after another (the calls of a sequence on one
    draw): each is judged before the next is made, so the float64 values of one call at a time are held.  The random
    state is restored after the last draw, the verdicts are recorded with ``record_property`` (one ``TIER`` line each
    in the terminal summary) and every failure is asserted.  Output names take ``prefix``."""
    judge = TierJudge(draws)
    failures: list[str] = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(draws):
            s = draw_seed(seed, j)
            seed_all(s)
            result = case(j, s)
            for launches, expected, naive in (
                [result] if isinstance(result, tuple) else result
            ):
                for launch in launches:
                    for name, actual in launch.items():
                        want = expected[name]
                        if message := contract(prefix + name, actual, want):
                            failures.append(f"draw {j}: {message}")
                            if actual.shape != want.shape:
                                continue
                        judge.add(
                            prefix + name,
                            j,
                            actual,
                            want,
                            {tier: naive[name] for tier in TIERS},
                        )
                del launches, expected, naive
            # A returned triple would otherwise stay held while the next draw makes its own.
            del result
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    if record_property is not None:
        judge.record(record_property)
    assert not failures, "\n".join(failures)
    return judge


def judge_package(
    make: Callable[[int], Mapping],
    run: Callable[[Mapping], Mapping | list],
    record_property: Callable[[str, object], None] | None,
    *,
    seed: int,
    names: Iterable[str] | None = None,
    draws: int = DRAWS,
    prefix: str = "",
    check: Callable[[Mapping, list, Mapping], None] | None = None,
    iters: int | None = None,
) -> TierJudge:
    """:func:`judge_draws` of the usual case: ``make(s)`` makes the inputs of the draw of seed ``s``, ``run(i)``
    launches the package on them (a result dict, or a list of them for repeated launches), and the judged outputs
    (``names``, default every name of :data:`OUTPUTS` the first launch holds, :func:`package_outputs`) are compared with
    :func:`expected_and_naive` of the inputs.  ``check(i, results, expected)``, when given, runs on every draw
    before the judge (the case's own exact checks; ``expected`` holds the oracle's value of every name of
    :data:`OUTPUTS`)."""

    def case(j: int, s: int):
        i = make(s)
        results = run(i)
        results = results if isinstance(results, list) else [results]
        launches = [package_outputs(res) for res in results]
        wanted = tuple(names) if names is not None else tuple(launches[0])
        launches = [{k: launch[k] for k in wanted} for launch in launches]
        expected, naive = expected_and_naive(
            i, HANDOFF_OUTPUTS if "pre_in" in i else OUTPUTS, iters=iters
        )
        if check is not None:
            check(i, results, expected)
        return launches, {k: expected[k] for k in wanted}, {k: naive[k] for k in wanted}

    return judge_draws(case, record_property, seed=seed, draws=draws, prefix=prefix)
