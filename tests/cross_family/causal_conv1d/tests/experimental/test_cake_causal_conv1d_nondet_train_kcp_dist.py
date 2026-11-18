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

Multi-GPU rows of the non-deterministic ``cake_causal_conv1d_nondet_kcp``: ``W`` ranks over NCCL (W = 2 / 4 / 8) on the
deterministic KCP suite's partitions (contiguous, zigzag, per-document zigzag, ragged tables with a zero-token rank,
one-token fragments, a fragment that reads its history from three predecessor ranks), plus W = 16 rows (``-k w16``) on
the layouts any world size partitions: contiguous, zigzag and per-document zigzag (the ragged tables are written for at
most eight ranks).  The outputs are judged by the naive judge (``tests/cake/test_helpers/cake_naive.py``): every
rank draws the whole pack from each of 32 seeds (the same on every rank), runs the call on its tokens and judges its
``y`` / ``dx`` rows and the sum of the ranks' ``dweight`` partials (all-gathered and added in rank order) against the
float64 oracle of the whole pack and PyTorch's BF16 naive convolution.  No bit rule holds the package: its outputs
are never compared bit for bit with another launch and never with the deterministic package.  Per row and rank:

- ``REPEATS`` launches of the direct pass on each draw, every launch judged;
- checkpoints with and without a recompute handle (``use_reentrant`` False and True; 2 x forward + backward) and
  nested checkpoints, judged; the per-phase collective counts of the deterministic suite (one halo exchange per
  direction, one more in a plain recompute, none in a recompute with a handle); ``retain_graph`` accumulation
  (judged as the gradients of ``2 dy``);
- the handle retains exactly the history pool;
- the two-launch forward (knobs ``fwd_main`` x ``fwd_overlap``, the same on every rank): ``LAST_CALL["fwd"]`` names
  the form the rank ran (the overlapped form on a rank with tokens, the history-row program when the rank receives
  history rows) and every ``y`` is judged;
- a 1-thread kernel resident on another stream does not stall the backward (the
  main backward is an ordinary launch; ``busy <= 2 x alone``), and the outputs of backward calls launched beside it
  are judged;
- the rank-agreement check runs once per phase and refuses a mismatched call on every rank.

The CUDA Graph replay cache (``graph="replay"``, W = 2): more keys than ``GRAPH_CACHE_MAX`` captured and replayed in
turn; the cache keeps at most ``GRAPH_CACHE_MAX`` graphs, the package's pre-zeroed buffers those graphs hold stay
bounded by the live graphs (an evicted graph's buffers return to the package), and the replays of a key are judged.

Launch forms as the deterministic suite: ``torch.multiprocessing`` (skipped with fewer than ``W`` visible devices)
or one process per Slurm task (``SLURM_NTASKS == W``).  Skipped until the exporter renders the package.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import statistics
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_det_train_kcp_dist import (
    ROWS,
    WIDTH,
    _Counter,
    _free_port,
    _local,
    _partition,
    _requested,
    _row_id,
)

# The seed of every rank's pack (draw 0 of the judges).
SEED = 7815
REPEATS = 20
# W = 16 (16-way context parallelism) on the deterministic rows' packs, for the layouts any world size partitions.
ROWS_W16 = [
    ((100, 640, 77, 1000, 129), 16, "contiguous", 512),
    ((3000,), 16, "zigzag", 256),
    ((2048, 5, 3000, 1), 16, "zigzag_per_document", 384),
]


def _run_rank(
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
) -> dict:
    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as kcp_api
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )
    from cake_pplx._backends.cake_kda_det_train_kcp import (
        cake_kcp_plan as plan,
    )
    from tests.cake.test_helpers.cake_causal_conv1d_oracle import judge_draws, problem

    lengths, _w, layout, C = row
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD
    part, cu = _partition(plan.cake_kda_det_train_kcp_partition, lengths, W, layout)
    ctx = plan.cake_kda_det_train_kcp_build_context(
        part, group, rank, device, conv_kernel_size=WIDTH
    )
    T = cu[-1]
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    rows = _local(part, rank, cu, torch.arange(T)).to(device)
    n_hist = int(ctx.halo.n_hist_rows)

    def pack(seed: int) -> types.SimpleNamespace:
        """The whole pack drawn from ``seed`` (the same on every rank) and the rank's tokens of it."""

        g = torch.Generator(device="cpu").manual_seed(seed)
        x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
        dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        return types.SimpleNamespace(
            x=_local(part, rank, cu, x_all).to(device),
            w=w_all.to(device),
            dy=_local(part, rank, cu, dy_all).to(device),
            x_all=x_all.to(device),
            dy_all=dy_all.to(device),
        )

    # The pack of the collective-count, retention and side-kernel checks: draw 0 of every judge below.
    first = pack(SEED)
    x, w, dy = first.x, first.w, first.dy

    def fn(x_, w_, h_):
        return kcp_api.cake_causal_conv1d_nondet_kcp(x_, w_, ctx, recompute_handle=h_)

    def run(mode: str, handle=None, *, twice: bool = False, reentrant: bool = False, p=first):
        xl = p.x.detach().clone().requires_grad_(True)
        wl = p.w.detach().clone().requires_grad_(True)
        with _Counter() as cc:
            if mode == "direct":
                y = fn(xl, wl, handle)
            elif mode == "ckpt":
                kw = (
                    {"use_reentrant": True}
                    if reentrant
                    else {
                        "use_reentrant": False,
                        "context_fn": lambda: (
                            contextlib.nullcontext(),
                            cc.recompute_cm(),
                        ),
                    }
                )
                y = checkpoint(fn, xl, wl, handle, **kw)
            else:

                def outer(x_, w_, h_):
                    hh = (
                        conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
                        if mode == "nested_in"
                        else h_
                    )
                    yi = checkpoint(fn, x_, w_, hh, use_reentrant=False)
                    return yi * torch.ones_like(yi)

                y = checkpoint(
                    outer, xl, wl, handle, use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )  # fmt: skip
            cc.phase = "bwd"
            y.backward(p.dy, retain_graph=twice)
            if twice:
                y.backward(p.dy)
        out = {"y": y.detach(), "dx": xl.grad, "dweight": wl.grad}
        counts = {
            t: {ph: cc.get(t, ph) for ph in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def world_dweight(dw: torch.Tensor) -> torch.Tensor:
        """The ranks' ``dweight`` partials all-gathered and added in rank order (the same order on every rank)."""

        partials = [torch.empty_like(dw) for _ in range(W)]
        dist.all_gather(partials, dw.contiguous())
        total = partials[0].clone()
        for partial in partials[1:]:
            total += partial
        return total

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"n_hist_rows": n_hist, "T_r": int(x.shape[0])}

    def judged(label: str, call, *, dy_scale: int = 1) -> None:
        """Every rank judges ``call(p)`` -- the outputs by name, ``dweight`` the rank's partial(s), summed over the
        ranks here -- on its rows over the draws, against the naive tier of the whole pack (``dy`` times
        ``dy_scale``).  The verdict is the flag ``judged:<label>``."""

        def launch(_draw: int, seed: int):
            p = pack(seed)
            outputs = dict(call(p))
            if not rows.numel():
                # a rank without tokens: its y / dx are empty and judge nothing (every draw's would look alike)
                outputs = {name: v for name, v in outputs.items() if name == "dweight"}
            if "dweight" in outputs:
                values = outputs["dweight"]
                outputs["dweight"] = (
                    world_dweight(values)
                    if isinstance(values, torch.Tensor)
                    else [world_dweight(v) for v in values]
                )
            return outputs, problem(
                p.x_all, p.w, cu_dev, None, p.dy_all * dy_scale, rows=rows
            )

        failures = judge_draws(launch, seed=SEED, label=f"{_row_id(row)} {label}")
        info[f"judge_failures:{label}"] = failures[:8]
        flags[f"judged:{label}"] = not failures

    # The first call of the context runs the rank-agreement check (one all_gather_into_tensor per phase, before that
    # phase's exchange); every later call issues only the halo exchanges.
    _, c_first = run("direct")
    flags["agreement_check_once_per_phase"] = c_first["calls"] == {
        "orig": 2,
        "recompute": 0,
        "bwd": 2,
    }
    _, c_direct = run("direct")
    _, c_plain = run("ckpt")
    hd = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    _, c_h = run("ckpt", hd)
    _, c_plain_re = run("ckpt", reentrant=True)
    hd_re = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    _, c_h_re = run("ckpt", hd_re, reentrant=True)
    info["counts"] = {
        "direct": c_direct, "plain": c_plain, "handle": c_h, "plain_reentrant": c_plain_re, "handle_reentrant": c_h_re,
    }  # fmt: skip
    for t in ("calls", "waits"):
        flags[f"positive_control_{t}"] = c_direct[t] == {"orig": 1, "recompute": 0, "bwd": 1} and c_plain[t] == {
            "orig": 1, "recompute": 1, "bwd": 1,
        }  # fmt: skip
        flags[f"handle_counts_{t}"] = c_h[t] == {"orig": 1, "recompute": 0, "bwd": 1}
        flags[f"reentrant_counts_{t}"] = c_plain_re[t] == {"orig": 1, "recompute": 0, "bwd": 2} and c_h_re[t] == {
            "orig": 1, "recompute": 0, "bwd": 1,
        }  # fmt: skip
    flags["handle_bytes"] = hd.nbytes() == n_hist * C * 2 == hd_re.nbytes()

    # Repeated launches of the direct pass on each draw: every launch judged; the number of distinct dweight bit
    # patterns of the first draw is recorded (no bound is placed on it).
    patterns: set[bytes] = set()
    draws: list[int] = []

    def repeats(p):
        found = {"y": [], "dx": [], "dweight": []}
        for _ in range(REPEATS):
            out, _ = run("direct", p=p)
            for name, value in out.items():
                found[name].append(value)
            if not draws:
                patterns.add(bytes(out["dweight"].contiguous().view(torch.uint8).cpu().numpy()))
        draws.append(len(draws))
        return found

    judged("direct repeats", repeats)
    info["repeats"] = {"n": REPEATS, "distinct_dw_patterns_first_draw": len(patterns)}

    # Checkpoints with and without a recompute handle, reentrant or not, and nested checkpoints.
    def checkpoints(p):
        found = {"y": [], "dx": [], "dweight": []}
        for mode, handle, reentrant in (
            ("ckpt", None, False),
            ("ckpt", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), False),
            ("ckpt", None, True),
            ("ckpt", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), True),
            ("nested_out", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), False),
            ("nested_in", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), False),
        ):
            out, _ = run(mode, handle, reentrant=reentrant, p=p)
            for name, value in out.items():
                found[name].append(value)
        return found

    judged("checkpoints", checkpoints)

    # retain_graph accumulation: the gradients of two passes of dy are those of 2 dy (the convolution is linear in
    # dy and doubling a BF16 dy is exact), judged as such.
    twice, c_twice = run(
        "ckpt", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), twice=True
    )
    info["counts"]["handle_twice"] = c_twice
    flags["twice_counts"] = c_twice["calls"] == {"orig": 1, "recompute": 0, "bwd": 2}

    def accumulated(p):
        out, _ = run(
            "ckpt", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), twice=True, p=p
        )
        return {"dx": out["dx"], "dweight": out["dweight"]}

    judged("retain_graph accumulation", accumulated, dy_scale=2)
    _, c_nest_out = run(
        "nested_out", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    )
    _, c_nest_in = run(
        "nested_in", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    )
    info["counts"]["nested_out"], info["counts"]["nested_in"] = c_nest_out, c_nest_in
    flags["nested_out_counts"] = c_nest_out["calls"] == {
        "orig": 1,
        "recompute": 0,
        "bwd": 1,
    }
    flags["nested_in_counts"] = c_nest_in["calls"] == {
        "orig": 1,
        "recompute": 1,
        "bwd": 1,
    }

    # The two-launch forward: every (fwd_main, fwd_overlap) form, the same on every rank, records its form in
    # LAST_CALL["fwd"] (the overlapped form with the exchange in flight on a rank with tokens; the history-row program
    # when the rank receives history rows and the form has two launches); every call's y judged.
    fwd_recorded = [True]

    def forms(p):
        found = []
        for fwd_main in (0, 1):
            for fwd_overlap in (0, 1):
                conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})
                try:
                    for _ in range(2):
                        y_f, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
                            p.x, p.w, ctx
                        )
                        torch.cuda.synchronize(device)
                        want = (
                            {
                                "main": "sr" if fwd_main else "f",
                                "form": "pair" if fwd_overlap else "wait",
                                "repair": bool(fwd_main or fwd_overlap) and n_hist > 0,
                            }
                            if p.x.shape[0]
                            else {"main": None, "form": "wait", "repair": False}
                        )
                        fwd_recorded[0] = fwd_recorded[0] and conv_kcp.LAST_CALL["fwd"] == want
                        found.append(y_f)
                finally:
                    conv_kcp._set_test_knobs(None)
                    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        return {"y": found}

    judged("forward forms", forms)
    flags["fwd_forms_recorded"] = fwd_recorded[0]

    retained = {}
    for name, h in (
        ("plain", None),
        ("handle", conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()),
    ):
        xl = x.detach().clone().requires_grad_(True)
        wl = w.detach().clone().requires_grad_(True)
        torch.cuda.synchronize(device)
        gc.collect()
        before = _requested(device)
        y = checkpoint(fn, xl, wl, h, use_reentrant=False)
        torch.cuda.synchronize(device)
        retained[name] = _requested(device) - before - y.numel() * y.element_size()
        y.backward(dy)
        del y, xl, wl
    info["retained_bytes"] = retained
    flags["retained"] = retained["plain"] == 0 and retained["handle"] == n_hist * C * 2
    # A kernel resident on another stream of every rank.
    side = torch.cuda.Stream(device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    _, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        dist.barrier()
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        start.record()
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    def side_kernel_ms() -> float:
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            start.record()
            torch.cuda._sleep(10_000_000)
            end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    busy = statistics.median(timed(True) for _ in range(7))
    sleep = side_kernel_ms()
    info["side_kernel_ms"] = {"alone": alone, "busy": busy, "side": sleep}
    # A stall waits for the side kernel, so busy reaches its sleep (milliseconds).
    # These rows are tiny: one more wave of CTAs beside the SM that the side kernel
    # holds can exceed 2x alone (about 0.1 ms) without a stall.
    flags["side_kernel_no_stall"] = busy <= max(2.0 * alone, 0.5 * sleep)

    def beside(p):
        _, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(p.x, p.w, ctx)
        torch.cuda.synchronize(device)
        dist.barrier()
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)
        dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
            p.x, p.w, ctx, p.dy, hist
        )
        torch.cuda.synchronize(device)
        return {"dx": dx, "dweight": dw}

    judged("beside the side kernel", beside)
    # Rank agreement: on a fresh context of the same partition, rank 0 calls with C channels and every other
    # rank with C + 8.  The first forward's check raises the same ValueError on every rank, before any exchange; a
    # matching call on that context then runs (the failed check recorded nothing).
    ctx2 = plan.cake_kda_det_train_kcp_build_context(
        part, group, rank, device, conv_kernel_size=WIDTH
    )
    C2 = C if rank == 0 else C + 8
    x2 = torch.zeros(x.shape[0], C2, dtype=torch.bfloat16, device=device)
    w2 = torch.zeros(C2, WIDTH, dtype=torch.float32, device=device)
    try:
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x2, w2, ctx2)
        flags["agreement_mismatch_raises"] = False
    except ValueError as exc:
        flags["agreement_mismatch_raises"] = "rank agreement" in str(exc)
        info["agreement_error"] = str(exc)[:300]
    y2, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
        x2[:, :C].contiguous(), w2[:C].contiguous(), ctx2
    )
    flags["agreement_matching_call_runs"] = bool(torch.all(y2 == 0))
    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "torch": torch.__version__,
        "flags": {k: all(f[k] for f in flag_list) for k in flags},
        "failing_ranks": {
            k: [r for r, f in enumerate(flag_list) if not f[k]] for k in flags
        },
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values())
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "nondet-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


# The row of the replay-cache test, and the keys it captures beyond the cache's capacity.
GRAPH_CACHE_ROW = ((3000,), 2, "zigzag", 256)
GRAPH_CACHE_EXTRA = 24


def _run_rank_graph_cache(rank: int, W: int, port: int, result_dir: str | None, local_id: int) -> dict:
    """One rank of :func:`test_kcp_graph_replay_cache_eviction_bounded`: ``GRAPH_CACHE_MAX + GRAPH_CACHE_EXTRA`` keys
    (pointer-distinct inputs, the same on every rank), each run four times with ``graph="replay"`` (forward and
    backward: eager, capture, replay); after every key the cache holds at most ``GRAPH_CACHE_MAX`` entries and the
    package holds at most a few pre-zeroed buffers beyond one per cached graph; then a fixed key's calls (eager,
    capture, replays) are judged over the draws."""

    import cake_pplx._backends.cake_causal_conv1d_nondet_train as backend
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )
    from cake_pplx._backends.cake_kda_det_train_kcp import (
        cake_kcp_plan as plan,
    )
    from tests.cake.test_helpers.cake_causal_conv1d_oracle import judge_draws, problem

    lengths, _w, layout, C = GRAPH_CACHE_ROW
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    part, cu = _partition(plan.cake_kda_det_train_kcp_partition, lengths, W, layout)
    ctx = plan.cake_kda_det_train_kcp_build_context(
        part, dist.group.WORLD, rank, device, conv_kernel_size=WIDTH
    )
    T = cu[-1]
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    rows = _local(part, rank, cu, torch.arange(T)).to(device)

    def pack(seed: int) -> types.SimpleNamespace:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
        dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
        return types.SimpleNamespace(
            x=_local(part, rank, cu, x_all).to(device),
            w=w_all.to(device),
            dy=_local(part, rank, cu, dy_all).to(device),
            x_all=x_all.to(device),
            dy_all=dy_all.to(device),
        )

    def step(x, w, dy):
        y, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx, graph="replay")
        dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist, graph="replay")
        return y, dx, dw

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"T_r": None}
    first = pack(SEED)
    info["T_r"] = int(first.x.shape[0])
    stats = dict(conv_kcp.GRAPH_STATS)
    keys = []  # the keyed inputs stay alive, so no later key reuses their addresses
    held_ok, cached_ok, worst = True, True, 0
    for _ in range(conv_kcp.GRAPH_CACHE_MAX + GRAPH_CACHE_EXTRA):
        keys.append(first.x.clone())
        for _ in range(4):
            step(keys[-1], first.w, first.dy)
        torch.cuda.synchronize(device)
        live = sum(e.graph is not None for e in conv_kcp._GRAPHS.values())
        held = len(backend._GRAPH_HELD)
        worst = max(worst, held - live)
        cached_ok = cached_ok and len(conv_kcp._GRAPHS) <= conv_kcp.GRAPH_CACHE_MAX
        held_ok = held_ok and held <= live + 4
    captures = conv_kcp.GRAPH_STATS["capture"] - stats["capture"]
    replays = conv_kcp.GRAPH_STATS["replay"] - stats["replay"]
    info["graph_stats"] = {"captures": captures, "replays": replays, "held_beyond_live_max": worst}
    n_keys = conv_kcp.GRAPH_CACHE_MAX + GRAPH_CACHE_EXTRA
    flags["captured_every_key"] = captures >= 2 * n_keys and replays >= n_keys
    flags["cache_bounded"] = cached_ok
    flags["held_buffers_bounded"] = held_ok
    flags["idle_pool_bounded"] = all(len(p) <= backend._POOL_DEPTH for p in backend._ZEROED_POOL.values())
    del keys

    x_key, w_key, dy_key = first.x.clone(), first.w.clone(), first.dy.clone()

    def launch(_draw: int, seed: int):
        p = pack(seed)
        x_key.copy_(p.x)
        w_key.copy_(p.w)
        dy_key.copy_(p.dy)
        y, dx, dw = step(x_key, w_key, dy_key)
        outputs = {"y": y.clone(), "dx": dx.clone(), "dweight": dw.clone()}
        partials = [torch.empty_like(outputs["dweight"]) for _ in range(W)]
        dist.all_gather(partials, outputs["dweight"].contiguous())
        total = partials[0].clone()
        for partial in partials[1:]:
            total += partial
        outputs["dweight"] = total
        if not rows.numel():
            outputs = {"dweight": total}
        return outputs, problem(p.x_all, p.w, cu_dev, None, p.dy_all, rows=rows)

    failures = judge_draws(launch, seed=SEED, label="graph replay cache")
    info["judge_failures"] = failures[:8]
    flags["judged:graph replay cache"] = not failures
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    torch.cuda.synchronize(device)
    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "flags": {k: all(f[k] for f in flag_list) for k in flags},
        "failing_ranks": {k: [r for r, f in enumerate(flag_list) if not f[k]] for k in flags},
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values())
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "nondet-graph-cache.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_graph_cache_entry(rank: int, W: int, port: int, result_dir: str) -> None:
    try:
        report = _run_rank_graph_cache(rank, W, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, port, result_dir, rank)
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


@pytest.mark.parametrize("row", ROWS + ROWS_W16, ids=_row_id)
def test_nondeterministic_kcp_rows(row) -> None:
    W = row[1]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip(
                "the non-deterministic conv package is not rendered or does not support this device"
            )
        report = _run_rank(
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
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, "nondet-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


def test_kcp_graph_replay_cache_eviction_bounded() -> None:
    W = GRAPH_CACHE_ROW[1]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip(
                "the non-deterministic conv package is not rendered or does not support this device"
            )
        report = _run_rank_graph_cache(
            int(os.environ["SLURM_PROCID"]),
            W,
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
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_graph_cache_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_graph_cache_entry,
            args=(W, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, "nondet-graph-cache.json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
