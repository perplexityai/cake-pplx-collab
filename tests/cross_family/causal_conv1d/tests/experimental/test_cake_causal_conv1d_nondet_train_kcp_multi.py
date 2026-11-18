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

Single-GPU tests of the non-deterministic ``cake_causal_conv1d_nondet_kcp_multi`` (several tensors of one layer with
one halo exchange per direction), on the deterministic multi suite's rows.  The ``y``, ``dx`` and ``dweight`` of every
tensor are judged by the naive judge (``tests/cake/test_helpers/cake_naive.py``; 32 random draws, each tensor against
the float64 oracle of its own unpartitioned problem and PyTorch's BF16 naive convolution) -- through the explicit
functions (also into caller-owned outputs), one single call per tensor, the autograd entry, the facade, a checkpoint
with and without a recompute handle (2 x forward + backward), gradient accumulation, graph replay and torch's
deterministic mode.  No bit rule holds the package: its outputs are never compared bit for bit with another launch
and never with the deterministic package.  The NCCL rows are in
``test_cake_causal_conv1d_nondet_train_kcp_multi_dist.py``.  Every test skips until the exporter has rendered the
non-deterministic package.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import cake_pplx._backends.cake_causal_conv1d_nondet_train as pkg
from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_causal_conv1d_oracle import (
    assert_judged,
    naive_bf16,
    oracle,
    problem,
)
from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)
from tests.test_cake_causal_conv1d_det_train_kcp import WIDTH, _context
from tests.test_cake_causal_conv1d_det_train_kcp_multi import MROWS, _problem

# The seed of the multi suite's problems (``_problem``'s default): draw 0 is its problem.
SEED = 7815

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)


def _global(lens, cuts, device):
    """``(l2g, cu)``: the global token of each local token of the one-rank layout and the documents' ``cu_seqlens``."""

    _, S, cu_global, frags = _context(lens, cuts, device)
    l2g = torch.tensor(
        [cu_global[d] + off + t for d, off, ln in frags for t in range(ln)],
        dtype=torch.int64,
        device=device,
    )
    assert l2g.numel() == S and int(torch.unique(l2g).numel()) == S
    return l2g, torch.tensor(cu_global, dtype=torch.int32, device=device)


def _unpartitioned(local: torch.Tensor, l2g: torch.Tensor) -> torch.Tensor:
    """The rows of the rank's local layout ``local`` in global order."""

    out = torch.empty_like(local)
    out[l2g] = local
    return out


def judge_multi(lens, cuts, Cs, run, *, label: str, seed: int = SEED) -> None:
    """``run(ctx, xs, ws, dys)`` on each draw's multi problem (``_problem`` of the draw's seed) returns one entry per
    tensor: its outputs by name (``y`` / ``dx`` in the rank's local order, as ``[S, C]`` or ``[1, S, C]``,
    ``dweight``; a value or a list of several launches' values).  Each tensor is judged against the naive tier of its
    own unpartitioned problem."""

    device = torch.device("cuda")
    l2g, cu = _global(lens, cuts, device)

    def launch(_draw: int, draw_seed: int):
        ctx, xs, ws, dys = _problem(lens, cuts, Cs, device, seed=draw_seed)
        found = run(ctx, xs, ws, dys)
        assert len(found) == len(Cs), (label, len(found))
        return [
            (
                {f"{name}[{i}]": value for name, value in outputs.items()},
                problem(
                    _unpartitioned(xs[i], l2g),
                    ws[i],
                    cu,
                    None,
                    _unpartitioned(dys[i], l2g),
                    rows=l2g,
                ),
            )
            for i, outputs in enumerate(found)
        ]

    assert_judged(launch, seed=seed, label=label)


def _singles(ctx, xs, ws, dys):
    """One explicit single forward / backward per tensor."""

    out = []
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
        dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist)
        out.append({"y": y, "hist": hist, "dx": dx, "dweight": dw})
    return out


def _by_tensor(*runs: list[dict]) -> list[dict]:
    """Per tensor, the ``y`` / ``dx`` / ``dweight`` of every run (each run a list with one dict per tensor)."""

    return [
        {name: [r[i][name] for r in runs] for name in ("y", "dx", "dweight")}
        for i in range(len(runs[0]))
    ]


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
def test_multi_and_single_calls_match_reference(name, lens, cuts, Cs) -> None:
    """The explicit multi functions (into new outputs and into caller-owned ones) and one single call per tensor:
    every tensor's ``y``, ``dx`` and ``dweight`` judged against the naive tier."""

    def run(ctx, xs, ws, dys):
        single = _singles(ctx, xs, ws, dys)
        ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
        dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs, ws, ctx, dys, hists
        )
        assert len(ys) == len(hists) == len(dxs) == len(dws) == len(Cs)
        outs = [torch.empty_like(x) for x in xs]
        ys2, hists2 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
            xs, ws, ctx, outs=outs
        )
        dx_o = [torch.full_like(x, float("nan")) for x in xs]
        dw_o = [torch.full_like(w, float("nan")) for w in ws]
        dxs2, dws2 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs, ws, ctx, dys, hists2, dxs=dx_o, dweights=dw_o
        )
        torch.cuda.synchronize()
        for i in range(len(Cs)):
            assert ys2[i] is outs[i] and dxs2[i] is dx_o[i] and dws2[i] is dw_o[i]
        multi = [{"y": ys[i], "dx": dxs[i], "dweight": dws[i]} for i in range(len(Cs))]
        owned = [{"y": outs[i], "dx": dx_o[i], "dweight": dw_o[i]} for i in range(len(Cs))]
        return _by_tensor(multi, owned, single)

    judge_multi(lens, cuts, Cs, run, label=name)


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
@pytest.mark.parametrize("batched", [False, True], ids=["2d", "3d"])
def test_multi_autograd_and_facade_match_reference(
    name, lens, cuts, Cs, batched
) -> None:
    """The facade's single entry per tensor and two passes through its multi entry (``[S, C]`` or ``[1, S, C]``
    tensors): every tensor's ``y``, ``dx`` and ``dweight`` judged against the naive tier."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    def run(ctx, xs, ws, dys):
        if batched:
            xs = [x.unsqueeze(0) for x in xs]
            dys = [d.unsqueeze(0) for d in dys]

        def once(multi: bool):
            xl = [x.clone().requires_grad_(True) for x in xs]
            wl = [w.clone().requires_grad_(True) for w in ws]
            if multi:
                ys = facade.cake_causal_conv1d_nondet_kcp_multi(xl, wl, ctx)
            else:
                ys = [
                    facade.cake_causal_conv1d_nondet_kcp(x, w, ctx)
                    for x, w in zip(xl, wl, strict=True)
                ]
            torch.autograd.backward(list(ys), list(dys))
            for y, x in zip(ys, xs, strict=True):
                assert y.shape == x.shape
            return [
                {"y": y.detach(), "dx": x.grad, "dweight": w.grad}
                for y, x, w in zip(ys, xl, wl, strict=True)
            ]

        return _by_tensor(once(False), once(True), once(True))

    judge_multi(lens, cuts, Cs, run, label=f"{name} {'3d' if batched else '2d'}")


def _ckpt(fn, args, *, reentrant):
    from torch.utils.checkpoint import checkpoint

    if reentrant:
        return checkpoint(fn, *args, use_reentrant=True)
    return checkpoint(fn, *args, use_reentrant=False)


HROWS = MROWS[:3] + MROWS[-3:-1]


@pytest.mark.parametrize("name,lens,cuts,Cs", HROWS, ids=[r[0] for r in HROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_multi_checkpoint_and_recompute_handle(name, lens, cuts, Cs, reentrant) -> None:
    """The multi entry directly, checkpointed and checkpointed with a recompute handle (which keeps one history
    pool per tensor): every tensor's ``y``, ``dx`` and ``dweight`` of each judged against the naive tier."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    n = len(Cs)

    def run(ctx, xs, ws, dys):
        def once(handle, *, ckpt=True):
            xl = [x.clone().requires_grad_(True) for x in xs]
            wl = [w.clone().requires_grad_(True) for w in ws]

            def fn(*args):
                return facade.cake_causal_conv1d_nondet_kcp_multi(
                    args[:n], args[n : 2 * n], ctx, recompute_handle=args[2 * n]
                )

            args = (*xl, *wl, handle)
            ys = _ckpt(fn, args, reentrant=reentrant) if ckpt else fn(*args)
            torch.autograd.backward(list(ys), list(dys))
            return [
                {"y": y.detach(), "dx": x.grad, "dweight": w.grad}
                for y, x, w in zip(ys, xl, wl, strict=True)
            ]

        direct = once(None, ckpt=False)
        plain = once(None)
        h = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
        conv_kcp.LAST_CALL.pop("recompute_handle", None)
        handled = once(h)
        assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay", "n": n}
        assert len(h.entries[conv_kcp.HANDLE_KEY]["hists"]) == n
        assert h.nbytes() == sum(int(ctx.halo.n_hist_rows) * C * 2 for C in Cs)
        return _by_tensor(direct, plain, handled)

    judge_multi(
        lens, cuts, Cs, run, label=f"{name} {'reentrant' if reentrant else 'nonreentrant'}"
    )


def test_multi_gradient_accumulation_with_retain_graph() -> None:
    """A checkpointed multi call with a recompute handle, ``retain_graph`` and a second backward of the same
    ``dy``: every tensor's accumulated ``dx`` and ``dweight`` judged against the naive tier's two-pass sums (the
    float64 oracle's sum of the two passes, twice one pass)."""

    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    device = torch.device("cuda")
    lens, cuts, Cs = (5000, 1003), (1, 2, 4997, 1003), (512, 512, 256)
    l2g, cu = _global(lens, cuts, device)
    judge = TierJudge(DRAWS)
    failures = []
    for j in range(DRAWS):
        seed = draw_seed(SEED, j)
        seed_all(seed)
        ctx, xs, ws, dys = _problem(lens, cuts, Cs, device, seed=seed)

        def fn(xq, xk, xv, wq, wk, wv, h, ctx=ctx):
            return facade.cake_causal_conv1d_nondet_kcp_multi(
                (xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h
            )

        a = [x.clone().requires_grad_(True) for x in xs] + [
            w.clone().requires_grad_(True) for w in ws
        ]
        ys = checkpoint(
            fn, *a, conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), use_reentrant=False
        )
        torch.autograd.backward(list(ys), list(dys), retain_graph=True)
        torch.autograd.backward(list(ys), list(dys))
        for i in range(len(Cs)):
            x_g, dy_g = _unpartitioned(xs[i], l2g), _unpartitioned(dys[i], l2g)
            want = oracle(x_g, ws[i], cu, None, dy_g, outputs=("dx", "dweight"))
            naive = naive_bf16(x_g, ws[i], cu, None, dy_g, outputs=("dx", "dweight"))
            for out, actual, rows in (
                ("dx", a[i].grad, l2g),
                ("dweight", a[len(Cs) + i].grad, None),
            ):
                expected, tier = want[out] + want[out], naive[out] + naive[out]
                if rows is not None:
                    expected, tier = expected[rows], tier[rows]
                name = f"{out}[{i}]"
                if message := contract(name, actual, expected):
                    failures.append(f"draw {j}: {message}")
                    continue
                judge.add(name, j, actual, expected, {"bf16": tier})
    failures += judge.failures()
    assert not failures, "\n".join(failures)


def test_multi_empty_rank_returns_empty_outputs() -> None:
    device = torch.device("cuda")
    ctx, _, _, _ = _context((0,), (0,), device)
    Cs = (512, 256, 512)
    xs = [torch.empty(0, C, dtype=torch.bfloat16, device=device) for C in Cs]
    ws = [torch.randn(C, WIDTH, device=device) for C in Cs]
    ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
    dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
        xs, ws, ctx, [torch.empty_like(x) for x in xs], hists
    )
    for y, dx, dw, C in zip(ys, dxs, dws, Cs, strict=True):
        assert y.shape == (0, C) and dx.shape == (0, C)
        assert torch.equal(dw, torch.zeros_like(dw))


def test_multi_graph_replay() -> None:
    """Graph replay of the multi calls on each draw's inputs (the graphs of the previous draw dropped first): the
    single entry's event sequence; every call's ``y``, ``dx`` and ``dweight`` judged against the naive tier."""

    def run(ctx, xs, ws, dys):
        conv_kcp._GRAPHS.clear()
        found = [{"y": [], "dx": [], "dweight": []} for _ in xs]
        events = []
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                xs, ws, ctx, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            for i, y in enumerate(ys):
                found[i]["y"].append(y.clone())
            dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            torch.cuda.synchronize()
            for i, (dx, dw) in enumerate(zip(dxs, dws, strict=True)):
                found[i]["dx"].append(dx.clone())
                found[i]["dweight"].append(dw.clone())
        assert events == [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ], events
        return found

    try:
        judge_multi((5000, 1003), (1, 2, 4997, 1003), (512, 512, 256), run, label="graph")
    finally:
        conv_kcp._GRAPHS.clear()


def test_multi_one_rank_issues_no_exchange(monkeypatch) -> None:
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 512), device
    )
    calls = []
    original = conv_kcp._exchange
    monkeypatch.setattr(
        conv_kcp, "_exchange", lambda *a, **kw: calls.append(1) or original(*a, **kw)
    )
    ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(xs, ws, ctx, dys, hists)
    assert calls == []


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool = False):
    """Torch's deterministic mode (``warn_only`` as given) inside the block, the previous mode restored after it (the
    oracle and the naive tier run outside the block)."""

    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def test_multi_torch_deterministic_mode_alerts() -> None:
    """Under torch's deterministic mode the multi backward raises; ``warn_only``: it warns and runs (every tensor's
    ``dx`` and ``dweight`` judged against the naive tier)."""

    device = torch.device("cuda")
    lens, cuts, Cs = (65,), (64, 1), (256, 128)
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    _, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
    with _torch_deterministic(), pytest.raises(
        RuntimeError, match="cake_causal_conv1d_nondet_train_kcp_multi_backward"
    ):
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs, ws, ctx, dys, hists
        )

    def warned(ctx, xs, ws, dys):
        _, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
        with _torch_deterministic(warn_only=True), pytest.warns(
            UserWarning, match="cake_causal_conv1d_nondet_train_kcp_multi_backward"
        ):
            dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists
            )
        return [{"dx": dx, "dweight": dw} for dx, dw in zip(dxs, dws, strict=True)]

    judge_multi(lens, cuts, Cs, warned, label="warn_only")
