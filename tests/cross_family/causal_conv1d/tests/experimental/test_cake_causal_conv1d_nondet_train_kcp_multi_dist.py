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

Multi-GPU rows of the non-deterministic ``cake_causal_conv1d_nondet_kcp_multi`` (the q / k / v convolutions of a
layer with one halo exchange per direction) on the deterministic multi suite's partitions, W = 2 / 4 / 8 over NCCL,
plus W = 16 rows (``-k w16``) on zigzag and per-document zigzag partitions (eager, and graph replay on the
per-document one).  The outputs are judged by the naive judge (``tests/cake/test_helpers/cake_naive.py``): every rank
draws the whole pack of every tensor from each of 32 seeds (the same on every rank), runs the calls on its tokens and
judges, per tensor, its ``y`` / ``dx`` rows and the sum of the ranks' ``dweight`` partials (all-gathered and added in
rank order) against the float64 oracle of the whole pack and PyTorch's BF16 naive convolution.  No bit rule holds the
package: its outputs are never compared bit for bit with another launch and never with the deterministic package.
Per row and rank, for every tensor:

- the single calls and the multi call -- direct, plain checkpoint, checkpoint with one handle, ``use_reentrant``
  False and True, nested checkpoints -- judged; ``retain_graph`` accumulation judged as the gradients of ``2 dy``
  (the convolution is linear in ``dy`` and doubling a BF16 ``dy`` is exact);
- the collective counts per phase of the deterministic suite: multi 1/0/1 direct, 1/1/1 plain checkpoint, 1/0/1
  with a handle; single calls n/0/n;
- the two-launch forward (knobs ``fwd_main`` x ``fwd_overlap``, the same on every rank): ``LAST_CALL["fwd_multi"]``
  names the form the rank ran, and every tensor's ``y`` of every form and of the one-launch single calls is judged;
- ``graph="replay"`` of the multi forward / backward (the graphs of the previous draw dropped first): every call's
  outputs judged, the deterministic suite's event sequence; and of the single backward of the first tensor (its head
  program and exchange issue on the side stream inside the capture): outputs judged, events eager, capture, replay,
  replay;
- a main-program launch that raises after the fork (single and multi backward) still joins the side stream: the
  error is the injected one and the current stream waits for work queued on the side stream before the call;
- the overlapped backward's tail rows (single and multi backward, eager): every main-program launch passes the
  context's ``tail_row_off`` as its ``skip_row_off`` when the rank receives tail rows (else the all ``-1`` table) and
  every tail-row launch runs off the caller's stream (the side stream, or a stream forked from it).

Launch forms as the deterministic suite.  Skipped until the exporter renders the non-deterministic package.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_det_train_kcp import WIDTH
from tests.test_cake_causal_conv1d_det_train_kcp_dist import _free_port, _local
from tests.test_cake_causal_conv1d_det_train_kcp_multi_dist import (
    GRAPH_ROWS,
    ROWS,
    _report,
    _row_id,
    _setup,
)
from tests.experimental.test_helpers.kcp_collective_shims import (
    CollectiveCounter,
    RecomputePhase,
)

# The seed of the packs (``_setup``'s): draw 0 of the judges.
SEED = 7815
# W = 16 (16-way context parallelism) on the deterministic rows' packs, for the layouts any world size partitions;
# the per-document zigzag row also runs under graph replay.
ROWS_W16 = [
    ((3000,), 16, "zigzag", (256, 256, 256)),
    ((2048, 5, 3000, 1), 16, "zigzag_per_document", (384, 384, 384)),
]
GRAPH_ROWS_W16 = [ROWS_W16[1]]


class _WorldJudge:
    """The judges of one rank: the packs drawn from a seed (``_setup``'s order of draws) and :meth:`judge`."""

    def __init__(self, W: int, row, rank: int, part, cu, device, flags: dict, info: dict) -> None:
        self.W, self.row, self.rank, self.part, self.cu = W, row, rank, part, cu
        self.device, self.flags, self.info = device, flags, info
        self.cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
        self.rows = _local(part, rank, cu, torch.arange(cu[-1])).to(device)

    def pack(self, seed: int) -> types.SimpleNamespace:
        """Every tensor's whole pack drawn from ``seed`` in ``_setup``'s order, and the rank's tokens of it."""

        g = torch.Generator(device="cpu").manual_seed(seed)
        T, dev = self.cu[-1], self.device
        p = types.SimpleNamespace(xs_all=[], dys_all=[], xs=[], ws=[], dys=[])
        for C in self.row[3]:
            x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
            w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
            dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
            p.xs_all.append(x_all.to(dev))
            p.dys_all.append(dy_all.to(dev))
            p.xs.append(_local(self.part, self.rank, self.cu, x_all).to(dev))
            p.ws.append(w_all.to(dev))
            p.dys.append(_local(self.part, self.rank, self.cu, dy_all).to(dev))
        return p

    def world_dweight(self, dw: torch.Tensor) -> torch.Tensor:
        """The ranks' ``dweight`` partials all-gathered and added in rank order (the same order on every rank)."""

        partials = [torch.empty_like(dw) for _ in range(self.W)]
        dist.all_gather(partials, dw.contiguous())
        total = partials[0].clone()
        for partial in partials[1:]:
            total += partial
        return total

    def judge(self, label: str, call, *, dy_scale: int = 1) -> None:
        """``call(p)`` on each draw's pack returns one entry per tensor, its outputs by name (a value or a list;
        ``dweight`` the rank's partial(s), summed over the ranks here); every rank judges its rows.  The verdict is
        the flag ``judged:<label>``."""

        from tests.cake.test_helpers.cake_causal_conv1d_oracle import judge_draws, problem

        def launch(_draw: int, seed: int):
            p = self.pack(seed)
            found = call(p)
            assert len(found) == len(self.row[3]), (label, len(found))
            pairs = []
            for i, outputs in enumerate(found):
                outputs = dict(outputs)
                if "dweight" in outputs:
                    values = outputs["dweight"]
                    outputs["dweight"] = (
                        self.world_dweight(values)
                        if isinstance(values, torch.Tensor)
                        else [self.world_dweight(v) for v in values]
                    )
                if not self.rows.numel():
                    # a rank without tokens: its y / dx are empty and judge nothing (every draw's would look alike)
                    outputs = {name: v for name, v in outputs.items() if name == "dweight"}
                pairs.append(
                    (
                        {f"{name}[{i}]": value for name, value in outputs.items()},
                        problem(
                            p.xs_all[i],
                            p.ws[i],
                            self.cu_dev,
                            None,
                            p.dys_all[i] * dy_scale,
                            rows=self.rows,
                        ),
                    )
                )
            return pairs

        failures = judge_draws(launch, seed=SEED, label=f"{_row_id(self.row)} {label}")
        self.info[f"judge_failures:{label}"] = failures[:8]
        self.flags[f"judged:{label}"] = not failures


def _by_tensor(runs: list[list[dict]]) -> list[dict]:
    """Per tensor, the ``y`` / ``dx`` / ``dweight`` of every run (each run a list with one dict per tensor)."""

    return [
        {name: [r[i][name] for r in runs] for name in runs[0][i]}
        for i in range(len(runs[0]))
    ]


def _run_rank(rank: int, W: int, row, port: int, result_dir, local_id: int) -> dict:
    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as kcp_api
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )

    Cs = row[3]
    n = len(Cs)
    device, part, cu, ctx, _xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    first = types.SimpleNamespace(xs=xs, ws=ws, dys=dys)

    def layer(multi: bool):
        def fn(*args):
            a_x, a_w, a_h = args[:n], args[n : 2 * n], args[2 * n :]
            if multi:
                return kcp_api.cake_causal_conv1d_nondet_kcp_multi(
                    a_x, a_w, ctx, recompute_handle=a_h[0]
                )
            return tuple(
                kcp_api.cake_causal_conv1d_nondet_kcp(x, w, ctx, recompute_handle=h)
                for x, w, h in zip(a_x, a_w, a_h, strict=True)
            )

        return fn

    def run(multi: bool, mode: str, handles=None, *, twice=False, reentrant=False, p=first):
        fn = layer(multi)
        if handles is None:
            handles = [None] if multi else [None] * n
        xl = [x.detach().clone().requires_grad_(True) for x in p.xs]
        wl = [w.detach().clone().requires_grad_(True) for w in p.ws]
        with CollectiveCounter() as cc:
            ctx_fn = lambda: (contextlib.nullcontext(), RecomputePhase(cc))  # noqa: E731
            if mode == "direct":
                ys = fn(*xl, *wl, *handles)
            elif mode == "ckpt":
                kw = (
                    {"use_reentrant": True}
                    if reentrant
                    else {"use_reentrant": False, "context_fn": ctx_fn}
                )
                ys = checkpoint(fn, *xl, *wl, *handles, **kw)
            else:

                def outer(*args):
                    hh = (
                        [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]
                        if mode == "nested_in"
                        else list(args[2 * n :])
                    )
                    yi = checkpoint(fn, *args[: 2 * n], *hh, use_reentrant=False)
                    return tuple(y * torch.ones_like(y) for y in yi)

                ys = checkpoint(
                    outer, *xl, *wl, *handles, use_reentrant=False, context_fn=ctx_fn
                )
            cc.phase = "bwd"
            torch.autograd.backward(list(ys), list(p.dys), retain_graph=twice)
            if twice:
                torch.autograd.backward(list(ys), list(p.dys))
        out = [
            {"y": y.detach(), "dx": x.grad, "dweight": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]
        rec = cc.record()
        counts = {
            t: {ph: rec[t].get(ph, 0) for ph in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def handle():
        return [conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()]

    def table(o, r, b):
        return {"orig": o, "recompute": r, "bwd": b}

    flags: dict[str, bool] = {}
    info: dict[str, object] = {
        "n_hist_rows": int(ctx.halo.n_hist_rows),
        "T_r": int(xs[0].shape[0]),
    }
    world = _WorldJudge(W, row, rank, part, cu, device, flags, info)
    # The first nondet calls of the context (the single calls) also run the rank-agreement check: one
    # all_gather_into_tensor per phase; every later call issues only the halo exchanges.
    _, cs_direct = run(False, "direct")
    _, cm_direct = run(True, "direct")
    _, cm_plain = run(True, "ckpt")
    _, cm_handle = run(True, "ckpt", handle())
    _, cm_handle_re = run(True, "ckpt", handle(), reentrant=True)
    _, cm_twice = run(True, "ckpt", handle(), twice=True)
    _, cm_nest_out = run(True, "nested_out", handle())
    _, cm_nest_in = run(True, "nested_in", handle())
    info["counts"] = {
        "single_direct": cs_direct, "multi_direct": cm_direct, "multi_plain": cm_plain, "multi_handle": cm_handle,
        "multi_handle_reentrant": cm_handle_re, "multi_twice": cm_twice, "nested_out": cm_nest_out,
        "nested_in": cm_nest_in,
    }  # fmt: skip
    flags["counts"] = (
        cs_direct["calls"] == table(n + 1, 0, n + 1)
        and cm_direct["calls"] == table(1, 0, 1)
        and cm_plain["calls"] == table(1, 1, 1)
        and cm_handle["calls"] == table(1, 0, 1)
        and cm_handle_re["calls"] == table(1, 0, 1)
        and cm_twice["calls"] == table(1, 0, 2)
        and cm_nest_out["calls"] == table(1, 0, 1)
        and cm_nest_in["calls"] == table(1, 1, 1)
    )

    def calls(p):
        return _by_tensor(
            [
                run(False, "direct", p=p)[0],
                run(True, "direct", p=p)[0],
                run(True, "ckpt", p=p)[0],
                run(True, "ckpt", handle(), p=p)[0],
                run(True, "ckpt", reentrant=True, p=p)[0],
                run(True, "ckpt", handle(), reentrant=True, p=p)[0],
                run(True, "nested_out", handle(), p=p)[0],
                run(True, "nested_in", handle(), p=p)[0],
            ]
        )

    world.judge("single and multi calls", calls)

    def accumulated(p):
        out, _ = run(True, "ckpt", handle(), twice=True, p=p)
        return [{"dx": o["dx"], "dweight": o["dweight"]} for o in out]

    world.judge("retain_graph accumulation", accumulated, dy_scale=2)
    # The overlapped backward's tail rows: the main programs leave them to the tail-row programs on the side stream.
    _, hist0 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(xs[0], ws[0], ctx)
    flags["skip_path_single"] = _skip_path(
        conv_kcp,
        ctx,
        device,
        lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
            xs[0], ws[0], ctx, dys[0], hist0
        ),
    )
    _, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
    flags["skip_path_multi"] = _skip_path(
        conv_kcp,
        ctx,
        device,
        lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs, ws, ctx, dys, hists
        ),
    )
    # The two-launch forward: every (fwd_main, fwd_overlap) form, the same on every rank, records its form in
    # LAST_CALL["fwd_multi"] (the first tensor's main program before the exchange's wait on a rank with tokens; the
    # history-row program when the rank receives history rows and the form has two launches); every tensor's y of
    # every form and of the one-launch single calls judged.
    n_hist = int(ctx.halo.n_hist_rows)
    fwd_recorded = [True]

    def forms(p):
        found = [[] for _ in range(n)]
        conv_kcp._set_test_knobs({"fwd_main": 0, "fwd_overlap": 0})
        try:
            for i, (x, w) in enumerate(zip(p.xs, p.ws, strict=True)):
                y, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
                found[i].append(y)
            torch.cuda.synchronize(device)
        finally:
            conv_kcp._set_test_knobs(None)
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        for fwd_main in (0, 1):
            for fwd_overlap in (0, 1):
                conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})
                try:
                    for _ in range(2):
                        ys, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                            p.xs, p.ws, ctx
                        )
                        torch.cuda.synchronize(device)
                        want = (
                            {
                                "main": "sr" if fwd_main else "f",
                                "form": "pair" if fwd_overlap else "wait",
                                "repair": bool(fwd_main or fwd_overlap) and n_hist > 0,
                            }
                            if p.xs[0].shape[0]
                            else {"main": None, "form": "wait", "repair": False}
                        )
                        fwd_recorded[0] = (
                            fwd_recorded[0] and conv_kcp.LAST_CALL["fwd_multi"] == want
                        )
                        for i, y in enumerate(ys):
                            found[i].append(y)
                finally:
                    conv_kcp._set_test_knobs(None)
                    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        return [{"y": values} for values in found]

    world.judge("forward forms", forms)
    flags["fwd_forms_recorded"] = fwd_recorded[0]
    return _report(W, row, flags, info, result_dir, rank, "nondet-multi")


def _skip_path(conv_kcp, ctx, device, call) -> bool:
    """``call`` (a backward) with its main-program and tail-row launches recorded: True when every main-program launch
    passed the context's ``tail_row_off`` as ``skip_row_off`` if the rank receives tail rows (else the all ``-1``
    table) and the tail-row launches, one per tensor when it receives them and none otherwise, ran off the caller's
    stream; or, for a call that takes no side stream (one rank) or a rank without tokens, when it launched neither."""
    tokens = int(ctx.T_r) > 0
    if conv_kcp._side_stream(ctx, device, True) is None or not tokens:
        real = conv_kcp._launch_dyn
        hits: list[int] = []

        def counted(d, **kwargs):
            hits.append(1)
            return real(d, **kwargs)

        conv_kcp._launch_dyn = counted
        try:
            call()
            torch.cuda.synchronize(device)
        finally:
            conv_kcp._launch_dyn = real
        return tokens or not hits
    caller = torch.cuda.current_stream(device).cuda_stream
    tail = ctx.halo_tables["tail_row_off"]
    receives = int(ctx.halo.n_tail_rows) > 0
    skips: list[bool] = []
    streams: list[int] = []
    real_dyn, real_halo = conv_kcp._launch_dyn, conv_kcp._launch_halo

    def dyn(d, **kwargs):
        skip = kwargs["skip"]
        skips.append(skip is tail if receives else bool((skip < 0).all()))
        return real_dyn(d, **kwargs)

    def halo(d, **kwargs):
        if d["halo"] is not None:
            streams.append(torch.cuda.current_stream(device).cuda_stream)
        return real_halo(d, **kwargs)

    conv_kcp._launch_dyn, conv_kcp._launch_halo = dyn, halo
    try:
        call()
        torch.cuda.synchronize(device)
    finally:
        conv_kcp._launch_dyn, conv_kcp._launch_halo = real_dyn, real_halo
    if not skips or not all(skips):
        return False
    if not receives:
        return not streams
    return len(streams) == len(skips) and caller not in streams


def _join_after_failed_launch(conv_kcp, ctx, device, call) -> bool:
    """``call`` with the main-program launch raising, while the side stream sleeps from before the call: True when
    the call raised the injected error and an event recorded on the current stream right after it is still pending
    (the current stream waits for the side stream), or when the call launched no main program (no error)."""
    if conv_kcp._side_stream(ctx, device, True) is None:
        return True
    side = conv_kcp._device_stream(device, 0)
    torch.cuda.synchronize(device)
    hits = []

    def boom(*args, **kwargs):
        hits.append(1)
        raise RuntimeError("injected main-launch failure")

    real = conv_kcp._launch_dyn
    conv_kcp._launch_dyn = boom
    err = None
    try:
        with torch.cuda.stream(side):
            torch.cuda._sleep(1 << 30)
        call()
    except RuntimeError as exc:
        err = exc
    finally:
        conv_kcp._launch_dyn = real
    done = torch.cuda.Event()
    done.record(torch.cuda.current_stream(device))
    pending = not done.query()
    torch.cuda.synchronize(device)
    if not hits:
        return err is None
    return err is not None and "injected main-launch failure" in str(err) and pending


def _run_rank_graph(
    rank: int, W: int, row, port: int, result_dir, local_id: int
) -> dict:
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )

    device, part, cu, ctx, _xs_all, xs, ws, dys = _setup(rank, W, row, port, local_id)
    n = len(row[3])
    tokens = int(xs[0].shape[0])
    if tokens == 0:
        single_expected = ["eager"] * 4
        expected = ["eager"] * 8
    elif int(ctx.halo.n_hist_rows) == 0:
        single_expected = ["eager", "capture", "replay", "replay"]
        expected = ["eager", "eager", "capture", "capture"] + ["replay"] * 4
    else:
        single_expected = ["eager", "capture", "replay", "replay"]
        expected = [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ]
    flags: dict[str, bool] = {}
    info: dict[str, object] = {"T_r": tokens}
    world = _WorldJudge(W, row, rank, part, cu, device, flags, info)
    seen = {"events": [], "single_events": []}

    def replays(p):
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
        found = [{"y": [], "dx": [], "dweight": []} for _ in range(n)]
        events, single_events = [], []
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(
                p.xs, p.ws, ctx, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                p.xs, p.ws, ctx, p.dys, hists, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            torch.cuda.synchronize(device)
            for i in range(n):
                found[i]["y"].append(ys[i].clone())
                found[i]["dx"].append(dxs[i].clone())
                found[i]["dweight"].append(dws[i].clone())
        _, hist0 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
            p.xs[0], p.ws[0], ctx
        )
        for _ in range(4):
            dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                p.xs[0], p.ws[0], ctx, p.dys[0], hist0, graph="replay"
            )
            single_events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            torch.cuda.synchronize(device)
            found[0]["dx"].append(dx.clone())
            found[0]["dweight"].append(dw.clone())
        seen["events"].append(events)
        seen["single_events"].append(single_events)
        return found

    try:
        world.judge("graph replay", replays)
        _, hist0 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(xs[0], ws[0], ctx)
        _, hists = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx)
        join_single = _join_after_failed_launch(
            conv_kcp,
            ctx,
            device,
            lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                xs[0], ws[0], ctx, dys[0], hist0
            ),
        )
        join_multi = _join_after_failed_launch(
            conv_kcp,
            ctx,
            device,
            lambda: conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                xs, ws, ctx, dys, hists
            ),
        )
    finally:
        torch.cuda.synchronize(device)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    flags.update(
        {
            "graph_events": all(e == expected for e in seen["events"]),
            "graph_single_events": all(e == single_expected for e in seen["single_events"]),
            "launch_error_joins_side_single": join_single,
            "launch_error_joins_side_multi": join_multi,
        }
    )
    info["events"] = seen["events"][0]
    info["single_events"] = seen["single_events"][0]
    return _report(W, row, flags, info, result_dir, rank, "nondet-graph")


def _spawn_entry(
    rank: int, W: int, row, port: int, result_dir: str, graph: bool
) -> None:
    try:
        fn = _run_rank_graph if graph else _run_rank
        report = fn(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        is_cake_causal_conv1d_nondet_train_supported,
    )

    return is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda"))


def _launch(row, graph: bool) -> None:
    import json

    W = row[1]
    tag = "nondet-graph" if graph else "nondet-multi"
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip(
                "the non-deterministic conv package is not rendered or does not support this device"
            )
        fn = _run_rank_graph if graph else _run_rank
        report = fn(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    if not _supported():
        pytest.skip(
            "the non-deterministic conv package is not rendered or does not support this device"
        )
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_multi_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir, graph),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, f"{tag}-{_row_id(row)}.json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


@pytest.mark.parametrize("row", ROWS + ROWS_W16, ids=_row_id)
def test_nondeterministic_multi_rows(row) -> None:
    _launch(row, graph=False)


@pytest.mark.parametrize("row", GRAPH_ROWS + GRAPH_ROWS_W16, ids=_row_id)
def test_nondeterministic_multi_graph_replay_rows(row) -> None:
    _launch(row, graph=True)
