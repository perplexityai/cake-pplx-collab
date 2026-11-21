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

Single-GPU tests of the non-deterministic KCP convolution (``cake_causal_conv1d_nondet_kcp``): one process, world
size 1, no process group (the halo exchange of one rank is the identity), on the deterministic KCP suite's rows and
contexts, plus a C = 10240 row for (a) and (b).  The outputs are judged by the naive judge
(``tests/cake/test_helpers/cake_naive.py``): over 32 random draws, the error of ``y`` / ``dx`` (in the rank's local
order) and of the FP32 ``dweight`` partial against the float64 oracle of the unpartitioned problem, in the L2 and the
max norm, has its median and its maximum within ``NEAR_TIE`` x the error of PyTorch's BF16 naive convolution (zero
where the naive error is zero).  No bit rule holds the package: its outputs are never compared bit for bit with
another launch and never with the deterministic package.

(a) ``y``, ``dx`` and ``dweight`` judged on every row.
(b) ``REPEATS`` launches on each draw's inputs: every launch judged (the worst launch of a draw counts); the
    package's accumulator workspace reads back all zero after every launch.
(c) The autograd entry and the direct calls, activation checkpointing with and without a recompute handle
    (2 x forward + backward; one exchange per direction, none in a recompute with a handle), gradient
    accumulation, graph replay, the legacy channel-tile plan, a side kernel (the backward is an ordinary launch: no
    stall beside a resident kernel), the grid that gives up one SM when the last channel-tile rank is partial and
    torch's deterministic mode (the backward alerts, through the package's entries and through the
    non-deterministic KCP package ``cake_pplx.cake_causal_conv1d_nondet_train_kcp``, whose entries take no
    ``deterministic`` keyword): every output judged.

Every test skips until the exporter has rendered the non-deterministic package.
"""

from __future__ import annotations

import contextlib
import statistics

import pytest
import torch

import cake_pplx._backends.cake_causal_conv1d_nondet_train as pkg
from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_causal_conv1d_oracle import naive_bf16, oracle
from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    ROWS,
    WIDTH,
    _context,
    _ExchangeCounter,
    _problem,
    judge_kcp,
)

REPEATS = 20
# Beside the deterministic suite's rows: C = 10240 (a packed q / k / v projection width), a fragment cut inside each
# of two documents.
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
    return conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
        x, w, ctx, dy, hist, **kw
    )


def _stream_workspace() -> torch.Tensor | None:
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + WIDE_ROWS, ids=[r[0] for r in ROWS + WIDE_ROWS]
)
def test_forward_matches_reference(name, lens, cuts, C):
    """``y`` judged against the naive tier on every row."""

    judge_kcp(
        lens, cuts, C, lambda p: {"y": _fwd(p.x_local, p.w, p.ctx)[0]}, label=name
    )


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + WIDE_ROWS, ids=[r[0] for r in ROWS + WIDE_ROWS]
)
def test_backward_repeats_are_judged(name, lens, cuts, C, record_property):
    """``REPEATS`` backward launches on each draw's inputs: every launch's ``dx`` and ``dweight`` judged against the
    naive tier (the worst launch of a draw counts), the accumulator workspace zero after every launch; the number of
    distinct ``dweight`` bit patterns of the first draw is recorded (the order of the cross-CTA sums may vary; no
    bound is placed on that count)."""

    patterns = set()
    draws = []

    def run(p):
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        found = {"dx": [], "dweight": []}
        for repeat in range(REPEATS):
            dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
            torch.cuda.synchronize()
            held = _stream_workspace()
            assert held is None or int(torch.count_nonzero(held)) == 0, (
                f"accumulators not zero after repeat {repeat}"
            )
            found["dx"].append(dx)
            found["dweight"].append(dw)
            if not draws:
                patterns.add(bytes(dw.contiguous().view(torch.uint8).cpu().numpy()))
        draws.append(len(draws))
        return found

    judge_kcp(lens, cuts, C, run, label=name)
    print(f"{name}: {REPEATS} launches per draw, {len(patterns)} distinct dweight bit patterns")
    record_property("repeats", REPEATS)
    record_property("distinct_dweight_patterns", len(patterns))


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_autograd_function_and_the_direct_calls(name, lens, cuts, C):
    """Two passes through the autograd entry and one through the direct calls: every pass's ``y``, ``dx`` and
    ``dweight`` judged against the naive tier."""

    def run(p):
        found = {"y": [], "dx": [], "dweight": []}
        for _ in range(2):
            x = p.x_local.clone().requires_grad_(True)
            w = p.w.clone().requires_grad_(True)
            y = conv_kcp.cake_causal_conv1d_nondet_kcp(x, w, p.ctx)
            y.backward(p.dy_local)
            for out, value in (("y", y.detach()), ("dx", x.grad), ("dweight", w.grad)):
                found[out].append(value)
        y, hist = _fwd(p.x_local, p.w, p.ctx)
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        for out, value in (("y", y), ("dx", dx), ("dweight", dw)):
            found[out].append(value)
        return found

    judge_kcp(lens, cuts, C, run, label=name)


def test_empty_rank_returns_empty_outputs():
    device = torch.device("cuda")
    C = 512
    ctx, _, _, _ = _context((0,), (0,), device)
    x = torch.empty(0, C, dtype=torch.bfloat16, device=device)
    w = torch.randn(C, WIDTH, device=device)
    y, hist = _fwd(x, w, ctx)
    dx, dw = _bwd(x, w, ctx, torch.empty_like(x), hist)
    assert y.shape == (0, C) and dx.shape == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_rejects_mismatched_context_and_keeps_the_kcp_contract():
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError):
        _fwd(p.x_local[:-1].contiguous(), p.w, p.ctx)
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        _fwd(p.x_local, p.w.to(torch.bfloat16), p.ctx)
    with pytest.raises(TypeError):
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
            p.x_local, p.w, p.ctx, bias=torch.zeros(p.C, device=device)
        )


def test_fp32_dweight_must_be_16_byte_aligned():
    """The KCP backward stores one channel's four FP32 ``dweight`` taps with one 16-byte store: a ``dweight`` off
    the 16-byte grid is refused (``ValueError``) before anything launches; a 16-byte aligned view inside a larger
    buffer is written (``dx`` and the view judged against the naive tier)."""

    device = torch.device("cuda")
    lens, cuts, C = (5000, 1003), (1, 2, 4997, 1003), 512
    p = _problem(lens, cuts, C, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    n = p.C * WIDTH
    buf = torch.zeros(n + 4, dtype=torch.float32, device=device)
    for offset in (1, 2, 3):
        misaligned = buf[offset : offset + n].view(p.C, WIDTH)
        assert misaligned.data_ptr() % 16 != 0
        with pytest.raises(ValueError, match="16-byte aligned"):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight=misaligned)
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(buf)) == 0
    held = _stream_workspace()
    assert held is None or int(torch.count_nonzero(held)) == 0

    def run(p):
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        buf = torch.zeros(n + 4, dtype=torch.float32, device=device)
        aligned = buf[4:].view(p.C, WIDTH)
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dweight=aligned)
        assert dw is aligned
        return {"dx": dx, "dweight": dw}

    judge_kcp(lens, cuts, C, run, label="16-byte aligned dweight view")


def test_tokens_and_taps_must_be_16_byte_aligned():
    """The KCP programs move the token rows (``x_local``, ``dy``, ``out``, ``dx``) as 16-byte vectors and one
    channel's four FP32 taps as one 16-byte vector: a tensor off the 16-byte grid is refused (``ValueError``) by the
    forward, the backward and the autograd entry of a one-rank context before anything launches or exchanges (the
    buffers stay zero)."""

    device = torch.device("cuda")
    lens, cuts, C = (5000, 1003), (1, 2, 4997, 1003), 512
    p = _problem(lens, cuts, C, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    shape, n = tuple(p.x_local.shape), p.x_local.numel()
    flat = torch.zeros(n + 1, dtype=torch.bfloat16, device=device)
    v = flat[1:].view(shape)
    taps = torch.zeros(p.C * WIDTH + 1, dtype=torch.float32, device=device)
    taps = taps[1:].view(p.C, WIDTH)
    assert v.data_ptr() % 16 != 0 and taps.data_ptr() % 16 != 0
    for name, call in (
        ("x_local", lambda: _fwd(v, p.w, p.ctx)),
        ("x_local", lambda: conv_kcp.cake_causal_conv1d_nondet_kcp(v, p.w, p.ctx)),
        ("x_local", lambda: _bwd(v, p.w, p.ctx, p.dy_local, hist)),
        ("out", lambda: _fwd(p.x_local, p.w, p.ctx, out=v)),
        ("dy", lambda: _bwd(p.x_local, p.w, p.ctx, v, hist)),
        ("dx", lambda: _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist, dx=v)),
        ("weight", lambda: _fwd(p.x_local, taps, p.ctx)),
        ("weight", lambda: _bwd(p.x_local, taps, p.ctx, p.dy_local, hist)),
    ):
        with pytest.raises(ValueError, match=f"{name} must be 16-byte aligned"):
            call()
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(flat)) == 0 and int(torch.count_nonzero(taps)) == 0
    held = _stream_workspace()
    assert held is None or int(torch.count_nonzero(held)) == 0


def test_kcp_plans_with_the_legacy_tile_rule(monkeypatch):
    """The KCP programs plan their launches with ``cake_jit.plan(..., legacy=True)``: the single backward program
    (the ``bwd_route`` test knob ``"single"``; the overlapped pair plans with its own work plan) and the one-launch
    forward (knobs ``fwd_main`` 0, ``fwd_overlap`` 0), while the shipped forward's single-rank main program
    (``FWD_MAIN_ARCHS``) plans with the ``fwd`` route's rule for the device's architecture.  ``y``, ``dx`` and
    ``dweight`` are judged against the naive tier either way."""
    calls = []
    original = conv_kcp.plan

    def recording_plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    monkeypatch.setattr(conv_kcp, "plan", recording_plan)

    def run(p):
        found = {"y": [], "dx": [], "dweight": []}
        for knobs in ({"fwd_main": 0, "fwd_overlap": 0}, {}):
            calls.clear()
            # the trimmed host path caches its plans per context: each form plans afresh
            conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
            conv_kcp._set_test_knobs(knobs)
            try:
                y, hist = _fwd(p.x_local, p.w, p.ctx)
                conv_kcp._set_test_knobs({**knobs, "bwd_route": "single"})
                dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
            finally:
                conv_kcp._set_test_knobs(None)
            assert len(calls) == 2 and calls[1].get("legacy") is True, (knobs, calls)
            if not knobs and arch in conv_kcp.FWD_MAIN_ARCHS:
                assert calls[0].get("legacy") is not True, calls
                assert (calls[0].get("route"), calls[0].get("arch")) == ("fwd", arch), calls
            else:
                assert calls[0].get("legacy") is True, (knobs, calls)
            for out, value in (("y", y), ("dx", dx), ("dweight", dw)):
                found[out].append(value)
        return found

    try:
        judge_kcp((5000, 1003), (1, 2, 4997, 1003), 1280, run, label="legacy plan")
    finally:
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()


def _one_sm_rows(device: torch.device) -> list[tuple[int, tuple[int, int, int]]]:
    """``(C, expected main backward grid)`` of the first ``C`` (a multiple of 1024 up to 16384) whose single-wave grid
    gives up one SM's CTAs on this device and of the first whose grid stays whole: ``ctas_per_sm x SM
    count`` less ``ctas_per_sm`` when ``grid % n_ch_tiles >= ctas_per_sm``, from the record and the legacy split."""

    arch = conv_kcp._loader.device_arch(device.index)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows: dict[bool, tuple[int, tuple[int, int, int]]] = {}
    for C in range(1024, 16385, 1024):
        (name,) = conv_kcp.selected_modules(arch, conv_kcp.KCP_BWD, C=C)
        rule = conv_kcp._loader.MODULES[name]["grid_rule"]
        k, mult = int(rule["ctas_per_sm"]), int(rule["grid_mult"][arch])
        full = mult * k * sms
        n = conv_kcp.plan(1, C, 1, geometry=rule["geometry"], legacy=True)["n_ch_tiles"]
        trims = mult == 1 and full % n >= k
        rows.setdefault(trims, (C, (full - k if trims else full, 1, 1)))
    return [rows[t] for t in (True, False) if t in rows]


def test_backward_grid_gives_up_one_sm_when_the_last_rank_is_partial(
    monkeypatch,
) -> None:
    """The single backward program (the ``bwd_route`` test knob ``"single"``): it launches
    ``ctas_per_sm x SM count`` CTAs, less ``ctas_per_sm`` when
    the last channel-tile rank of that grid holds at least ``ctas_per_sm`` CTAs (so a kernel on another stream that
    keeps one SM's shared memory leaves the launch one wave), else the whole grid, on both host paths; ``dx`` and
    ``dweight`` judged against the naive tier and the workspace reads back zero either way."""

    device = torch.device("cuda")
    rows = _one_sm_rows(device)
    assert len(rows) == 2, rows
    seen = []
    launch_plan, run_module = conv_kcp._launch_plan, conv_kcp.run_module

    def main(name: str) -> bool:
        return conv_kcp._loader.MODULES[name]["role"] == conv_kcp.KCP_BWD

    def recording_plan(name, grid, device_index):
        if main(name):
            seen.append(tuple(grid))
        return launch_plan(name, grid, device_index)

    def recording_run(name, values, *, device_index, grid=None):
        if main(name):
            seen.append(None if grid is None else tuple(grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_plan)
    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    def run_for(C, expected):
        def run(p):
            conv_kcp._set_test_knobs(None)
            _, hist = _fwd(p.x_local, p.w, p.ctx)
            found = {"dx": [], "dweight": []}
            for host_trim in (1, 0):
                conv_kcp._set_test_knobs(
                    {"host_trim": host_trim, "bwd_route": "single"}
                )
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
                seen.clear()
                dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
                torch.cuda.synchronize()
                label = f"C={C} host_trim={host_trim}"
                assert seen == [expected], (label, seen)
                held = _stream_workspace()
                assert held is None or int(torch.count_nonzero(held)) == 0, label
                found["dx"].append(dx)
                found["dweight"].append(dw)
            return found

        return run

    try:
        for C, expected in rows:
            judge_kcp(
                (5000, 1003), (1, 2, 4997, 1003), C, run_for(C, expected), label=f"C={C}"
            )
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()


def _forward_record(device: torch.device) -> tuple[str, dict, tuple[int, int, int]]:
    """``(arch, the forward's record, its full co-resident grid)`` on ``device``."""

    arch = conv_kcp._loader.device_arch(device.index)
    (name,) = conv_kcp.route_modules(arch, conv_kcp.KCP_FWD)
    record = conv_kcp._loader.MODULES[name]
    return arch, record, conv_kcp.launch_grid(record, device_index=device.index)


def _record_forward_grids(monkeypatch) -> list:
    """The grid of every forward launch from here on (the trimmed path's argument plan or the untrimmed path's
    ``run_module`` call), in order."""

    seen = []
    launch_plan, run_module = conv_kcp._launch_plan, conv_kcp.run_module

    def recording_plan(name, grid, device_index):
        if conv_kcp._loader.MODULES[name]["role"] == conv_kcp.KCP_FWD:
            seen.append(tuple(grid))
        return launch_plan(name, grid, device_index)

    def recording_run(name, values, *, device_index, grid=None):
        if conv_kcp._loader.MODULES[name]["role"] == conv_kcp.KCP_FWD:
            seen.append(None if grid is None else tuple(grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_plan)
    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    return seen


def _check_forward_grid(seen: list, expected: tuple, Cs: tuple, arch: str) -> None:
    """For each ``C`` of ``Cs`` and both host paths, the one-launch forward (knobs ``fwd_main`` 0, ``fwd_overlap`` 0:
    the ``kcp_fwd`` program): one forward launch of grid ``expected``, ``y`` judged against the naive tier."""

    def run_for(C):
        def run(p):
            found = []
            for host_trim in (1, 0):
                conv_kcp._set_test_knobs(
                    {"host_trim": host_trim, "fwd_main": 0, "fwd_overlap": 0}
                )
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
                seen.clear()
                y, _ = _fwd(p.x_local, p.w, p.ctx)
                torch.cuda.synchronize()
                label = f"{arch} C={C} host_trim={host_trim}"
                assert seen == [expected], (label, seen)
                found.append(y)
            return {"y": found}

        return run

    try:
        for C in Cs:
            judge_kcp((5000, 1003), (1, 2, 4997, 1003), C, run_for(C), label=f"{arch} C={C}")
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()


def test_forward_grid_per_architecture(monkeypatch) -> None:
    """The one-launch forward (knobs ``fwd_main`` 0, ``fwd_overlap`` 0: the ``kcp_fwd`` program) launches the
    record's co-resident grid less ``FWD_RESERVE_SMS[arch]`` SMs of CTAs (sm_107a: one SM, the deterministic KCP
    forward's grid; sm_100a / sm_103a: the full grid), on both host paths; ``y`` is judged against the naive tier
    either way."""

    arch, record, full = _forward_record(torch.device("cuda"))
    assert conv_kcp.FWD_RESERVE_SMS == {"sm_107a": 1}
    reserved = conv_kcp.FWD_RESERVE_SMS.get(arch, 0) * int(
        record["grid_rule"]["ctas_per_sm"]
    )
    seen = _record_forward_grids(monkeypatch)
    _check_forward_grid(
        seen, (full[0] - reserved, full[1], full[2]), (1024, 10240), arch
    )


def test_forward_grid_reserve_falls_back_to_the_full_grid(monkeypatch) -> None:
    """A reserve that leaves fewer CTAs than the call's channel tiles launches the full grid; one that leaves exactly
    as many launches the shorter grid; more channel tiles than the full grid holds raise ``ValueError``.  The
    fallback launch (C = 10240, one SM of CTAs left: fewer CTAs than its channel tiles) has its ``y`` judged against
    the naive tier on both host paths."""

    device = torch.device("cuda")
    arch, record, full = _forward_record(device)
    per_sm = int(record["grid_rule"]["ctas_per_sm"])
    monkeypatch.setattr(conv_kcp, "FWD_RESERVE_SMS", {arch: full[0] // per_sm - 1})

    def grid(n_ch_tiles):
        return conv_kcp._forward_grid(
            record, arch=arch, device_index=device.index, C=10240, n_ch_tiles=n_ch_tiles
        )

    assert grid(per_sm) == (per_sm, full[1], full[2])
    assert grid(per_sm + 1) == full
    assert grid(full[0]) == full
    with pytest.raises(ValueError):
        grid(full[0] + 1)
    n_ch_tiles = conv_kcp.plan(
        5000, 10240, 2, geometry=record["grid_rule"]["geometry"], legacy=True
    )["n_ch_tiles"]
    assert n_ch_tiles > per_sm, (n_ch_tiles, per_sm)
    seen = _record_forward_grids(monkeypatch)
    _check_forward_grid(seen, full, (10240,), arch)


@pytest.mark.parametrize("form", ["knob", "keyword"])
def test_graph_replay(monkeypatch, form) -> None:
    """Graph replay of the KCP calls on each draw's inputs (the graphs of the previous draw dropped first): the
    deterministic suite's event sequence; ``y``, ``dx`` and ``dweight`` of every call judged against the naive tier
    (the captured launch zeroes nothing itself: the accumulators it replays on are left zero by every launch)."""

    kw = {}
    if form == "knob":
        conv_kcp._set_test_knobs({"graph": "replay"})
    else:
        kw = {"graph": "replay"}

    def run(p):
        conv_kcp._GRAPHS.clear()
        x, w, ctx, dy = p.x_local, p.w, p.ctx, p.dy_local
        found = {"y": [], "dx": [], "dweight": []}
        events = []
        for _ in range(4):
            y, hist = _fwd(x, w, ctx, **kw)
            events.append(conv_kcp.LAST_CALL["fwd"]["graph_event"])
            found["y"].append(y.clone())
            dx, dw = _bwd(x, w, ctx, dy, hist, **kw)
            events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            torch.cuda.synchronize()
            found["dx"].append(dx.clone())
            found["dweight"].append(dw.clone())
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
        judge_kcp((5000, 1003), (1, 2, 4997, 1003), 512, run, label=f"graph {form}")
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()


# ----------------------------------------------------------------------------------------------------------------
# Checkpointing and the recompute handle.


def _ckpt_run(p, *, handle=None, reentrant=False, ckpt=True, counter=None, twice=False):
    import contextlib

    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return facade.cake_causal_conv1d_nondet_kcp(x_, w_, p.ctx, recompute_handle=h_)

    if not ckpt:
        y = fn(x, w, handle)
    elif reentrant:
        y = checkpoint(fn, x, w, handle, use_reentrant=True)
    else:
        cm = counter.recompute_cm if counter is not None else contextlib.nullcontext
        y = checkpoint(
            fn,
            x,
            w,
            handle,
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), cm()),
        )
    if counter is not None:
        counter.phase = "bwd"
    y.backward(p.dy_local, retain_graph=twice)
    if twice:
        y.backward(p.dy_local)
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}


def _outputs(*runs: dict) -> dict:
    """The ``y`` / ``dx`` / ``dweight`` of the runs of :func:`_ckpt_run`, by output name."""

    return {
        "y": [r["y"] for r in runs],
        "dx": [r["dx"] for r in runs],
        "dweight": [r["dw"] for r in runs],
    }


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_checkpoint_and_recompute_handle(
    monkeypatch, name, lens, cuts, C, reentrant
) -> None:
    """Checkpointed (2 x forward + backward) with and without a recompute handle, and the direct pass: ``y``,
    ``dx`` and ``dweight`` of each judged against the naive tier; the exchange counts of the deterministic suite (one
    per direction, none in a recompute with a handle)."""

    def run(p):
        direct = _ckpt_run(p, ckpt=False)
        plain_cc = _ExchangeCounter(monkeypatch, conv_kcp)
        plain = _ckpt_run(p, reentrant=reentrant, counter=plain_cc)
        monkeypatch.undo()
        handle = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
        cc = _ExchangeCounter(monkeypatch, conv_kcp)
        conv_kcp.LAST_CALL.pop("recompute_handle", None)
        got = _ckpt_run(p, handle=handle, reentrant=reentrant, counter=cc)
        monkeypatch.undo()
        assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
        if reentrant:
            assert plain_cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
            assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
        else:
            assert plain_cc.table() == {"orig": 1, "recompute": 1, "bwd": 1}
            assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
        assert handle.nbytes() == int(p.ctx.halo.n_hist_rows) * C * 2
        handle.clear()
        return _outputs(direct, plain, got)

    judge_kcp(lens, cuts, C, run, label=f"{name} {'reentrant' if reentrant else 'nonreentrant'}")


def test_gradient_accumulation_and_retain_graph(monkeypatch) -> None:
    """``retain_graph`` + a second backward of the same ``dy``: the accumulated ``dx`` and ``dweight`` judged against
    the naive tier's two-pass sums (the float64 oracle's sum of the two passes, twice one pass); the recompute
    exchanges nothing."""

    device = torch.device("cuda")
    lens, cuts, C = (5000, 1003), (1, 2, 4997, 1003), 512
    judge = TierJudge(DRAWS)
    failures = []
    for j in range(DRAWS):
        seed = draw_seed(0, j)
        seed_all(seed)
        p = _problem(lens, cuts, C, device, seed=seed)
        cc = _ExchangeCounter(monkeypatch, conv_kcp)
        twice = _ckpt_run(
            p,
            handle=conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(),
            counter=cc,
            twice=True,
        )
        monkeypatch.undo()
        assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
        want = oracle(p.x, p.w, p.cu, None, p.dy, outputs=("dx", "dweight"))
        naive = naive_bf16(p.x, p.w, p.cu, None, p.dy, outputs=("dx", "dweight"))
        for name, actual, rows in (
            ("dx", twice["dx"], p.l2g),
            ("dweight", twice["dw"], None),
        ):
            expected, tier = want[name] + want[name], naive[name] + naive[name]
            if rows is not None:
                expected, tier = expected[rows], tier[rows]
            if message := contract(name, actual, expected):
                failures.append(f"draw {j}: {message}")
                continue
            judge.add(name, j, actual, expected, {"bf16": tier})
    failures += judge.failures()
    assert not failures, "\n".join(failures)


def test_recompute_handle_misuse_raises() -> None:
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    handle = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    conv_kcp.cake_causal_conv1d_nondet_kcp(
        p.x_local, p.w, p.ctx, recompute_handle=handle
    )
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "capture"}
    with pytest.raises(RuntimeError, match="per call and micro-step"):
        conv_kcp.cake_causal_conv1d_nondet_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=handle
        )
    with pytest.raises(ValueError, match="graph='off'"):
        conv_kcp.cake_causal_conv1d_nondet_kcp(
            p.x_local,
            p.w,
            p.ctx,
            graph="replay",
            recompute_handle=conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(),
        )
    with pytest.raises(TypeError, match="entries"):
        conv_kcp.cake_causal_conv1d_nondet_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=object()
        )


# ----------------------------------------------------------------------------------------------------------------
# Streams, torch's deterministic mode.


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """The main KCP backward is an ordinary launch, so one 1-thread kernel resident
    on another stream does not stall it (``busy <= 2 x alone``, medians of 7 after 3 warm-ups); the ``dx`` and
    ``dweight`` of backward calls launched beside it are judged against the naive tier."""

    device = torch.device("cuda")
    lens, cuts, C = (150000, 111393), (1, 2, 149997, 111393), 512
    p = _problem(lens, cuts, C, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    side = torch.cuda.Stream(device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    out = {}

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        start.record()
        out["dx"], out["dw"] = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"KCP backward alone {alone:.3f} ms, beside a resident kernel {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("kcp_backward_alone_ms", f"{alone:.4f}")
    record_property("kcp_backward_beside_side_kernel_ms", f"{busy:.4f}")
    assert busy <= 2.0 * alone, (
        f"the KCP backward stalled beside a resident kernel: {busy:.3f} vs {alone:.3f} ms"
    )

    def beside(p):
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        torch.cuda.synchronize(device)
        return {"dx": dx, "dweight": dw}

    judge_kcp(lens, cuts, C, beside, label="beside the side kernel")


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


def test_torch_deterministic_mode_alerts_on_the_backward() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` the forward runs (``y`` judged against the naive tier) and
    the backward raises, through the direct call and through autograd; ``warn_only=True``: the backward warns and
    runs (``dx`` and ``dweight`` judged against the naive tier)."""

    device = torch.device("cuda")
    lens, cuts, C = (65,), (64, 1), 256
    p = _problem(lens, cuts, C, device)
    _, hist = _fwd(p.x_local, p.w, p.ctx)
    with _torch_deterministic():
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_kcp_backward"
        ):
            _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        x = p.x_local.clone().requires_grad_(True)
        y = conv_kcp.cake_causal_conv1d_nondet_kcp(x, p.w, p.ctx)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_kcp_backward"
        ):
            y.backward(p.dy_local)

    def forward(p):
        with _torch_deterministic():
            y, _ = _fwd(p.x_local, p.w, p.ctx)
        return {"y": y}

    def warned(p):
        _, hist = _fwd(p.x_local, p.w, p.ctx)
        with _torch_deterministic(warn_only=True), pytest.warns(
            UserWarning, match="cake_causal_conv1d_nondet_train_kcp_backward"
        ):
            dx, dw = _bwd(p.x_local, p.w, p.ctx, p.dy_local, hist)
        return {"dx": dx, "dweight": dw}

    judge_kcp(lens, cuts, C, forward, label="deterministic mode forward")
    judge_kcp(lens, cuts, C, warned, label="warn_only")


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_kcp_facade_under_torch_deterministic_mode(multi) -> None:
    """``cake_pplx.cake_causal_conv1d_nondet_train_kcp`` on a device: its entries take no ``deterministic`` keyword
    (the ordinary ``TypeError``); under torch's deterministic mode the forward runs and the backward raises (strict,
    also when the mode is set between the forward and the backward: the mode is read when the backward runs) or
    warns and runs (``warn_only``).  ``y``, ``dx`` and ``dweight`` of a pass without the mode and of a
    ``warn_only`` pass are judged against the naive tier."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as facade

    device = torch.device("cuda")
    lens, cuts, C = (5000, 1003), (1, 2, 4997, 1003), 512
    p = _problem(lens, cuts, C, device)

    def call(x, w, ctx, **kw):
        if multi:
            (y,) = facade.cake_causal_conv1d_nondet_kcp_multi((x,), (w,), ctx, **kw)
            return y
        return facade.cake_causal_conv1d_nondet_kcp(x, w, ctx, **kw)

    def run(p, before_backward=None):
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)
        y = call(x, w, p.ctx)
        if before_backward is not None:
            before_backward()
        y.backward(p.dy_local)
        return {"y": y.detach(), "dx": x.grad, "dweight": w.grad}

    for value in (False, True):
        with pytest.raises(TypeError, match="deterministic"):
            call(p.x_local, p.w, p.ctx, deterministic=value)

    alert = "cake_causal_conv1d_nondet_train_kcp_(multi_)?backward does not have a deterministic"
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match=alert):
            run(p)
        torch.use_deterministic_algorithms(False)
        with pytest.raises(RuntimeError, match=alert):
            run(p, before_backward=lambda: torch.use_deterministic_algorithms(True))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)

    def judged(p):
        plain = run(p)
        with _torch_deterministic(warn_only=True), pytest.warns(UserWarning, match=alert):
            got = run(p)
        return {name: [plain[name], got[name]] for name in plain}

    judge_kcp(lens, cuts, C, judged, label="multi" if multi else "single")
