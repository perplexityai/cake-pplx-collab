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

Single-GPU tests of the opt-in FP32 (hi, lo) pair ``dweight`` of the non-deterministic KCP convolution
(``dweight_sum="fp32_pair"``: the overlapped backward's main program from route ``kcp_bwd_dyn_pair``).  One process,
world size 1, on the deterministic KCP suite's rows and contexts.  Every output is judged by the naive judge as in
``test_cake_causal_conv1d_nondet_train_kcp.py`` (32 draws, L2 and max norm, median and largest error at most 1.005
times the naive BF16 tier's); no bit rule holds the package.

(a) ``REPEATS`` pair backward launches per draw, every launch judged, the accumulators zero after every launch.
(b) The pair call launches the ``kcp_bwd_dyn_pair`` main program where the default launches ``kcp_bwd_dyn``.
(c) The autograd entry, the facade, the multi entries, checkpointing with a recompute handle and graph replay with
    ``dweight_sum="fp32_pair"``: every output judged.
(d) Refusals: ``bwd_reserve_sms >= 1``, the one-program backward (test knob), more than ``DYN_MAX_CHANNELS``
    channels and an unknown ``dweight_sum`` raise ``ValueError``; nothing falls back to another program.
"""

from __future__ import annotations

import pytest
import torch

import cake_pplx._backends.cake_causal_conv1d_nondet_train as pkg
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp as conv_kcp
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp_jit
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp_multi import judge_multi
from tests.test_cake_causal_conv1d_det_train_kcp import ROWS, judge_kcp
from tests.test_cake_causal_conv1d_det_train_kcp_multi import MROWS

PAIR = "fp32_pair"
REPEATS = 5
WIDE_ROWS = [("wide-qkv", (2000, 1003), (1, 2, 1997, 1003), 10240)]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)


def _fwd(x, w, ctx, **kw):
    return conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx, **kw)


def _bwd(x, w, ctx, dy, hist, **kw):
    return conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist, **kw)


def _stream_workspace() -> torch.Tensor | None:
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


@pytest.mark.parametrize("name,lens,cuts,C", ROWS + WIDE_ROWS, ids=[r[0] for r in ROWS + WIDE_ROWS])
def test_pair_backward_repeats_are_judged(name, lens, cuts, C):
    """``REPEATS`` pair backward launches on each draw's inputs: every launch's ``dx`` and ``dweight`` judged, the
    accumulator workspace zero after every launch."""

    def run(p):
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        found = {"dx": [], "dweight": []}
        for repeat in range(REPEATS):
            dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=PAIR)
            torch.cuda.synchronize()
            held = _stream_workspace()
            assert held is None or int(torch.count_nonzero(held)) == 0, f"accumulators not zero after {repeat}"
            assert dw.dtype == torch.float32 and tuple(dw.shape) == tuple(p.w.shape)
            found["dx"].append(dx)
            found["dweight"].append(dw)
        return found

    judge_kcp(lens, cuts, C, run, label=f"pair {name}")


def _main_programs(monkeypatch) -> list[str]:
    """The roles of the programs the KCP calls plan for a launch, in planning order (the host cache cleared first, so
    every call plans afresh)."""

    roles: list[str] = []
    plan = conv_kcp._launch_plan

    def recording_plan(name, grid, device_index):
        roles.append(kcp_jit.MODULES[name]["role"] if name in kcp_jit.MODULES else "?")
        return plan(name, grid, device_index)

    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_plan)
    return roles


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_pair_call_launches_the_pair_main_program(monkeypatch, name, lens, cuts, C):
    """The pair backward plans the ``kcp_bwd_dyn_pair`` main program (and the shared tail-row program); the default
    plans ``kcp_bwd_dyn``; the outputs are judged."""

    roles = _main_programs(monkeypatch)

    def run(p):
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        roles.clear()
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=PAIR)
        assert "kcp_bwd_dyn_pair" in roles and "kcp_bwd_dyn" not in roles, roles
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        roles.clear()
        _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        assert "kcp_bwd_dyn" in roles and "kcp_bwd_dyn_pair" not in roles, roles
        return {"dx": dx, "dweight": dw}

    judge_kcp(lens, cuts, C, run, label=f"pair launch {name}")


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_pair_autograd_facade_and_direct_calls(name, lens, cuts, C):
    """The backend's autograd entry, the facade and the direct calls with ``dweight_sum="fp32_pair"``: every pass's
    ``y``, ``dx`` and ``dweight`` judged."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    def run(p):
        found = {"y": [], "dx": [], "dweight": []}
        for entry in (conv_kcp.cake_causal_conv1d_nondet_kcp, facade.cake_causal_conv1d_nondet_kcp):
            x = p.x_local.clone().requires_grad_(True)
            w = p.w.clone().requires_grad_(True)
            y = entry(x, w, p.ctx, dweight_sum=PAIR)
            y.backward(p.dy_local)
            for out, value in (("y", y.detach()), ("dx", x.grad), ("dweight", w.grad)):
                found[out].append(value)
        y, hist = _fwd(p.x_local, p.w, p.ctx)
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=PAIR)
        for out, value in (("y", y), ("dx", dx), ("dweight", dw)):
            found[out].append(value)
        return found

    judge_kcp(lens, cuts, C, run, label=f"pair autograd {name}")


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:2], ids=[r[0] for r in ROWS[:2]])
def test_pair_checkpoint_with_recompute_handle(name, lens, cuts, C):
    """Non-reentrant checkpointing (2 x forward + backward) through the facade with and without a recompute handle,
    ``dweight_sum="fp32_pair"``: ``y``, ``dx`` and ``dweight`` judged."""

    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    def once(p, handle):
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)

        def fn(x_, w_, h_):
            return facade.cake_causal_conv1d_nondet_kcp(x_, w_, p.ctx, recompute_handle=h_, dweight_sum=PAIR)

        y = checkpoint(fn, x, w, handle, use_reentrant=False)
        y.backward(p.dy_local)
        return y.detach(), x.grad, w.grad

    def run(p):
        found = {"y": [], "dx": [], "dweight": []}
        for handle in (None, conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()):
            for out, value in zip(("y", "dx", "dweight"), once(p, handle), strict=True):
                found[out].append(value)
        return found

    judge_kcp(lens, cuts, C, run, label=f"pair checkpoint {name}")


def test_pair_graph_replay() -> None:
    """Graph replay of the KCP calls with ``graph="replay"`` and ``dweight_sum="fp32_pair"`` (the pair call keys its
    own graph): the event sequence of the shipped calls, and every call's outputs judged."""

    def run(p):
        conv_kcp._GRAPHS.clear()
        x, w, ctx, dy = p.x_local, p.w, p.ctx, p.dy_local
        found = {"y": [], "dx": [], "dweight": []}
        events = []
        for _ in range(4):
            y, hist = _fwd(x, w, ctx, graph="replay")
            events.append(conv_kcp.LAST_CALL["fwd"]["graph_event"])
            found["y"].append(y.clone())
            dx, dw = _bwd(x, w, ctx, dy, hist, graph="replay", dweight_sum=PAIR)
            events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            torch.cuda.synchronize()
            found["dx"].append(dx.clone())
            found["dweight"].append(dw.clone())
        assert events == ["eager", "eager", "capture", "eager", "replay", "capture", "replay", "replay"], events
        return found

    try:
        judge_kcp((5000, 1003), (1, 2, 4997, 1003), 512, run, label="pair graph")
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
    finally:
        conv_kcp._GRAPHS.clear()


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS[-3:], ids=[r[0] for r in MROWS[-3:]])
def test_pair_multi_entries(name, lens, cuts, Cs) -> None:
    """The multi backward and the facade's multi autograd entry with ``dweight_sum="fp32_pair"``: every tensor's
    outputs judged."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    def run(ctx, xs, ws, dys):
        ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
        dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(xs, ws, ctx, dys, hists, dweight_sum=PAIR)
        xl = [x.clone().requires_grad_(True) for x in xs]
        wl = [w.clone().requires_grad_(True) for w in ws]
        yl = facade.cake_causal_conv1d_nondet_kcp_multi(xl, wl, ctx, dweight_sum=PAIR)
        torch.autograd.backward(list(yl), list(dys))
        return [
            {"y": [ys[i], yl[i].detach()], "dx": [dxs[i], xl[i].grad], "dweight": [dws[i], wl[i].grad]}
            for i in range(len(Cs))
        ]

    judge_multi(lens, cuts, Cs, run, label=f"pair multi {name}")


def test_pair_refusals() -> None:
    """``ValueError`` and no fallback: an unknown ``dweight_sum``, ``"fp32_pair"`` with ``bwd_reserve_sms >= 1`` (at
    the autograd entry's forward call and at the backward), with the one-program backward of the test knob, and
    beyond ``DYN_MAX_CHANNELS`` channels."""

    from tests.test_cake_causal_conv1d_det_train_kcp import _problem

    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, torch.device("cuda"), seed=3)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    for bad, error in (("fp64", ValueError), ("pair", ValueError), ("", ValueError), ("FP32_PAIR", ValueError),
                       (1, TypeError), (True, TypeError)):
        with pytest.raises(error, match="dweight_sum must be None or"):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=bad)
        with pytest.raises(error, match="dweight_sum must be None or"):
            conv_kcp.cake_causal_conv1d_nondet_kcp(p.x_local, p.w, p.ctx, dweight_sum=bad)
    # A reserve is refused first where the architecture does not support one (its own message).
    arch = kcp_jit.device_arch(p.x_local.device.index)
    reserve_refusal = "is not supported on" if arch in conv_kcp.RESERVE_UNSUPPORTED else "runs on the overlapped backward"
    with pytest.raises(ValueError, match=reserve_refusal):
        _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=PAIR, bwd_reserve_sms=1)
    with pytest.raises(ValueError, match=reserve_refusal):
        conv_kcp.cake_causal_conv1d_nondet_kcp(p.x_local, p.w, p.ctx, dweight_sum=PAIR, bwd_reserve_sms=1)
    conv_kcp._dweight_sum(None, 0, "entry")
    with pytest.raises(ValueError, match="runs on the overlapped backward"):
        conv_kcp._dweight_sum(PAIR, 1, "entry")
    try:
        conv_kcp._set_test_knobs({"bwd_route": "single"})
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        with pytest.raises(ValueError, match="runs on the overlapped backward"):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight_sum=PAIR)
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    limit = kcp_jit.DYN_MAX_CHANNELS
    conv_kcp._pair_fits(True, limit, 100, 0, "entry")
    conv_kcp._pair_fits(False, limit + 8, 100, 0, "entry")
    with pytest.raises(ValueError, match="runs on the overlapped backward"):
        conv_kcp._pair_fits(True, limit + 8, 100, 0, "entry")
    assert conv_kcp.DWEIGHT_SUMS == ("fp32", "fp32_pair")
    for arch in kcp_jit.ROUTES:
        (name,) = kcp_jit.route_modules(arch, "kcp_bwd_dyn_pair")
        assert kcp_jit.MODULES[name]["role"] == "kcp_bwd_dyn_pair"
