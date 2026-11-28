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

Multi-GPU rows of the non-deterministic KCP backward's L1o route (``kcp_jit.L1O_ROUTE``, :func:`kcp._l1o_host`): on a
key, the overlapped backward launches the one-program backward (kind ``kcp_bwd``) before the halo exchange's wait with
every ``tail_row_off`` entry ``-1``, and the tail-row program on the side stream waits for it
(``side.wait_stream(caller)``) as well as for the exchange.  ``W`` ranks over NCCL (W = 2 / 4 / 8):

- forced rows (any device): ``kcp.l1o_route`` is replaced by a function that returns the running architecture's
  ``kcp_jit.L1O_RESERVE_SMS`` for the row's forced channel counts (else ``None``), so small packs take the route.
  Single calls (contiguous, zigzag and per-document zigzag; W 2, 4 and 8) and multi calls (one width, so the tail-row
  programs read in place, and mixed widths with a forced and an unforced tensor), each eager, with checkpoint
  recompute (with a handle, and reentrant without one), under ``graph="replay"`` (eager, capture, replay, replay),
  the single call's previous host path (test knob ``host_trim`` 0) and a ``dweight_sum="fp32_pair"`` call (never
  routed);
- key rows (a measured part only: the device's (architecture, SM count) is a key of ``kcp_jit.L1O_ROUTE``): the
  shipped route on a real key (C 512 at the smallest listed token count of the world size; W 2 and 4, and 8 on
  B200), single and multi calls (one width at W 2, a key tensor beside a non-key tensor at W 4), against the same
  calls with the route off (``kcp.l1o_route`` returning ``None``).

Every draw runs each form with the route and without it.  ``y``, ``dx`` and the checkpointed ``dx`` of every form
with the route must equal the call without it bit for bit (``dx`` does not depend on the route); ``dweight`` is
judged with every output by the naive judge (``tests/cake/test_helpers/cake_naive.py``: the whole pack from each of
32 seeds, the sum of the ranks' partials) against the float64 oracle and PyTorch's BF16 naive convolution.  A spy on
``kcp._launch_dyn`` records each main-program launch's state: the L1o state on every routed tensor of a form with the
route (never on an ``fp32_pair`` call), the overlapped pair's on every other.  Before each L1o launch the spy queues a
``torch.cuda._sleep`` on the caller's stream, so the one-program backward starts after the exchange has arrived: a
tail-row program that did not wait for the caller's stream would write its rows first and the one-program backward
would overwrite them (a ``dx`` bit mismatch).

Launch forms as the deterministic suite: ``torch.multiprocessing`` (skipped with fewer than ``W`` visible devices)
or one process per Slurm task (``SLURM_NTASKS == W``).  Skipped until the exporter renders the package.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_det_train_kcp_dist import _free_port
from tests.test_cake_causal_conv1d_det_train_kcp_multi_dist import (
    _report,
    _row_id,
    _same,
    _setup,
)
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp_multi_dist import (
    _WorldJudge,
)

# Cycles of the spy's sleep before each L1o main-program launch (about 0.5 ms; the small rows' exchanges take tens
# of microseconds).
DELAY_CYCLES = 1_000_000
# (document lengths, W, layout, channel counts of the tensors), and the channel counts the route is forced on.
FORCED_ROWS = [
    (((1500, 2600, 900), 2, "contiguous", (512,)), (512,)),
    (((2048, 5, 3000, 1), 2, "zigzag_per_document", (384,)), (384,)),
    (((6000,), 4, "zigzag", (256,)), (256,)),
    (((100, 640, 77, 1000, 129, 4000), 4, "contiguous", (512,)), (512,)),
    (((3000, 2000, 3000), 8, "contiguous", (256,)), (256,)),
]
FORCED_MULTI_ROWS = [
    (((1500, 2600, 900), 2, "contiguous", (512, 512, 512)), (512,)),
    (((6000,), 4, "zigzag", (512, 256, 512)), (512,)),
    (((2048, 5, 3000, 1), 2, "zigzag_per_document", (384, 768)), (768,)),
]
# Key rows: (W, layout, channel counts); the first tensor's C is 512, the key's token count is chosen per device.
KEY_ROWS = {
    "w2": (2, "contiguous", (512, 512)),
    "w4": (4, "zigzag", (512, 256)),
    "w8": (8, "contiguous", (512,)),
}


class _Route:
    """The route switch and the main-program spy of one rank.  :meth:`use` selects ``"off"`` (``kcp.l1o_route``
    returns ``None``), ``"real"`` (the shipped table) or ``"forced"`` (the architecture's reserve for the forced
    channel counts) and drops the host and graph caches, whose keys do not name the route.  The spy records
    ``(form, C, S, l1o)`` per ``kcp._launch_dyn`` call and delays each L1o launch on the caller's stream."""

    def __init__(self, conv_kcp, kcp_jit, device, W: int, forced: tuple[int, ...]) -> None:
        self.kcp, self.jit, self.W, self.forced = conv_kcp, kcp_jit, W, forced
        self.arch = conv_kcp._device_arch(device)
        self.sms = int(kcp_jit.sm_count(device.index))
        self.real_route, self.real_dyn = conv_kcp.l1o_route, conv_kcp._launch_dyn
        self.mode, self._form = "real", ""
        self.seen: list[tuple[str, int, int, bool]] = []
        self.forms_run: list[str] = []

        def spy(d, **kwargs):
            l1o = bool(d.get("l1o"))
            self.seen.append((self.form, int(kwargs["C"]), int(kwargs["S"]), l1o))
            if l1o:
                torch.cuda._sleep(DELAY_CYCLES)
            return self.real_dyn(d, **kwargs)

        conv_kcp._launch_dyn = spy

    @property
    def form(self) -> str:
        return self._form

    @form.setter
    def form(self, name: str) -> None:
        self._form = name
        if name:
            self.forms_run.append(name)

    def use(self, mode: str) -> None:
        reserve = self.jit.L1O_RESERVE_SMS[self.arch]
        forced = set(self.forced)

        def off(arch, sms, world_size, S, C):
            return None

        def force(arch, sms, world_size, S, C):
            ok = arch is not None and world_size is not None and int(C) in forced
            return reserve if ok else None

        self.kcp.l1o_route = {"off": off, "real": self.real_route, "forced": force}[mode]
        self.kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        self.kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
        self.mode, self._form = mode, ""
        self.seen.clear()
        self.forms_run.clear()

    def expected(self, form: str, C: int, S: int) -> bool:
        """Whether a main-program launch of ``form`` takes the L1o state in the current mode."""

        if form == "pair" or self.mode == "off":
            return False
        if self.mode == "forced":
            return C in self.forced
        return self.real_route(self.arch, self.sms, self.W, S, C) is not None

    def check(self) -> tuple[bool, list]:
        """Every recorded launch took the expected state, and every form run launched a main program (every rank of
        these rows has tokens): with the route, at least one in the L1o state in each form but ``pair``."""

        bad = [s for s in self.seen if s[3] != self.expected(s[0], s[1], s[2])]
        launched = {s[0] for s in self.seen}
        routed = {s[0] for s in self.seen if s[3]}
        missing = [f for f in self.forms_run if f not in launched]
        if self.mode != "off":
            missing += [f"{f}:no-l1o" for f in self.forms_run if f != "pair" and f not in routed]
        return not bad and not missing, bad[:4] + missing[:4]

    def restore(self) -> None:
        self.kcp.l1o_route, self.kcp._launch_dyn = self.real_route, self.real_dyn
        self.kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        self.kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()


def _single_forms(conv_kcp, kcp_api, ctx, device, route: _Route, x, w, dy, *, trim0: bool = True) -> dict:
    """One tensor's forms (eager, checkpoints, graph replay, previous host path, ``fp32_pair``): their outputs."""

    from torch.utils.checkpoint import checkpoint

    def leaf(t):
        return t.detach().clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return kcp_api.cake_causal_conv1d_nondet_kcp(x_, w_, ctx, recompute_handle=h_)

    out: dict[str, object] = {}
    route.form = "eager"
    xl, wl = leaf(x), leaf(w)
    y = fn(xl, wl, None)
    y.backward(dy)
    out.update(y=y.detach(), dx=xl.grad, dw=wl.grad)
    route.form = "ckpt"
    xl, wl = leaf(x), leaf(w)
    y = checkpoint(fn, xl, wl, conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle(), use_reentrant=False)
    y.backward(dy)
    out.update(ckpt_y=y.detach(), ckpt_dx=xl.grad, ckpt_dw=wl.grad)
    route.form = "ckpt_reentrant"
    xl, wl = leaf(x), leaf(w)
    y = checkpoint(fn, xl, wl, None, use_reentrant=True)
    y.backward(dy)
    out.update(ckpt2_dx=xl.grad, ckpt2_dw=wl.grad)
    if trim0:
        route.form = "host_trim0"
        conv_kcp._set_test_knobs({"host_trim": 0})
        try:
            _, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx)
            dx0, dw0 = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hist)
        finally:
            conv_kcp._set_test_knobs(None)
        out.update(trim0_dx=dx0, trim0_dw=dw0)
    route.form = "graph"
    gy, gdx, gdw, events = [], [], [], []
    for _ in range(4):
        yg, hg = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(x, w, ctx, graph="replay")
        dxg, dwg = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(x, w, ctx, dy, hg, graph="replay")
        events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
        torch.cuda.synchronize(device)
        gy.append(yg.clone())
        gdx.append(dxg.clone())
        gdw.append(dwg.clone())
    out.update(graph_y=gy, graph_dx=gdx, graph_dw=gdw, graph_events=events)
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    route.form = "pair"
    xl, wl = leaf(x), leaf(w)
    y = kcp_api.cake_causal_conv1d_nondet_kcp(xl, wl, ctx, dweight_sum="fp32_pair")
    y.backward(dy)
    out.update(pair_dx=xl.grad, pair_dw=wl.grad)
    torch.cuda.synchronize(device)
    return out


def _multi_forms(conv_kcp, kcp_api, ctx, device, route: _Route, xs, ws, dys) -> list[dict]:
    """The multi call's forms (eager, checkpoint with one handle, graph replay): per tensor, their outputs."""

    from torch.utils.checkpoint import checkpoint

    n = len(xs)

    def leaves(ts):
        return [t.detach().clone().requires_grad_(True) for t in ts]

    def fn(*args):
        return kcp_api.cake_causal_conv1d_nondet_kcp_multi(args[:n], args[n : 2 * n], ctx, recompute_handle=args[2 * n])

    out = [{} for _ in range(n)]
    route.form = "multi_eager"
    xl, wl = leaves(xs), leaves(ws)
    ys = fn(*xl, *wl, None)
    torch.autograd.backward(list(ys), list(dys))
    for i in range(n):
        out[i].update(y=ys[i].detach(), dx=xl[i].grad, dw=wl[i].grad)
    route.form = "multi_ckpt"
    xl, wl = leaves(xs), leaves(ws)
    h = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    ys = checkpoint(fn, *xl, *wl, h, use_reentrant=False)
    torch.autograd.backward(list(ys), list(dys))
    for i in range(n):
        out[i].update(ckpt_y=ys[i].detach(), ckpt_dx=xl[i].grad, ckpt_dw=wl[i].grad)
    route.form = "multi_graph"
    runs = []
    for _ in range(4):
        yg, hg = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_forward(xs, ws, ctx, graph="replay")
        dxg, dwg = conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(xs, ws, ctx, dys, hg, graph="replay")
        torch.cuda.synchronize(device)
        runs.append(([t.clone() for t in yg], [t.clone() for t in dxg], [t.clone() for t in dwg]))
    for i in range(n):
        out[i].update(
            graph_y=[r[0][i] for r in runs],
            graph_dx=[r[1][i] for r in runs],
            graph_dw=[r[2][i] for r in runs],
        )
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()
    torch.cuda.synchronize(device)
    return out


# The outputs compared bit for bit with the call without the route, and those judged, per form.
_BITS = ("y", "dx", "ckpt_y", "ckpt_dx", "ckpt2_dx", "trim0_dx", "graph_y", "graph_dx", "pair_dx")
_JUDGED = {
    "y": ("y", "ckpt_y", "graph_y"),
    "dx": ("dx", "ckpt_dx", "ckpt2_dx", "trim0_dx", "graph_dx", "pair_dx"),
    "dweight": ("dw", "ckpt_dw", "ckpt2_dw", "trim0_dw", "graph_dw", "pair_dw"),
}


def _listed(value) -> list:
    return list(value) if isinstance(value, list) else [value]


def _run_rank(rank: int, W: int, row, forced, kind: str, port: int, result_dir, local_id: int) -> dict:
    """One rank of a forced row (``kind`` ``"single"`` / ``"multi"``: the route ``"forced"`` against ``"off"``) or a
    key row (``kind`` ``"key"``: ``"real"`` against ``"off"``, the single forms of the first tensor and the multi
    forms of all tensors)."""

    import cake_pplx.cake_causal_conv1d_nondet_train_kcp as kcp_api
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp_jit,
    )

    device, part, cu, ctx, _xs_all, _xs, _ws, _dys = _setup(rank, W, row, port, local_id)
    flags: dict[str, bool] = {}
    info: dict[str, object] = {"T_r": int(ctx.T_r), "tail_rows": int(ctx.halo.n_tail_rows)}
    world = _WorldJudge(W, row, rank, part, cu, device, flags, info)
    route = _Route(conv_kcp, kcp_jit, device, W, tuple(forced))
    on = "real" if kind == "key" else "forced"
    info.update(arch=route.arch, sms=route.sms, mode=on)
    state = {"bits": True, "route_on": True, "route_off": True, "replayed": True}
    detail: dict[str, list] = {"bits": [], "route": []}

    def forms(p, mode):
        route.use(mode)
        found = []
        if kind in ("single", "key"):
            found.append(_single_forms(conv_kcp, kcp_api, ctx, device, route, p.xs[0], p.ws[0], p.dys[0]))
        if kind in ("multi", "key"):
            found.extend(_multi_forms(conv_kcp, kcp_api, ctx, device, route, p.xs, p.ws, p.dys))
        ok, why = route.check()
        key = "route_on" if mode == on else "route_off"
        state[key] = state[key] and ok
        if not ok:
            detail["route"].append((mode, why))
        return found

    def call(p):
        ref = forms(p, "off")
        got = forms(p, on)
        for j, (a, b) in enumerate(zip(got, ref, strict=True)):
            for name in _BITS:
                if name not in a:
                    continue
                same = all(_same(u, v) for u, v in zip(_listed(a[name]), _listed(b[name]), strict=True))
                if not same:
                    state["bits"] = False
                    detail["bits"].append((j, name))
            events = a.get("graph_events")
            if events is not None and "replay" not in events:
                state["replayed"] = False
        # the judged outputs: every form with the route, per tensor (single forms first, then the multi forms)
        judged = []
        for a in got:
            judged.append(
                {out: [v for name in names if name in a for v in _listed(a[name])] for out, names in _JUDGED.items()}
            )
        if kind == "key":
            # the single forms of tensor 0 and the multi forms of every tensor: one entry per tensor
            first = {k: judged[0][k] + judged[1][k] for k in judged[0]}
            judged = [first, *judged[2:]]
        return judged

    try:
        world.judge("l1o route", call)
    finally:
        torch.cuda.synchronize(device)
        route.restore()
    flags["bitwise_y_dx_ckpt_dx_vs_without_route"] = state["bits"]
    flags["route_taken_with_route"] = state["route_on"]
    flags["route_not_taken_without_route"] = state["route_off"]
    flags["graph_replayed"] = state["replayed"]
    info["bit_mismatches"] = detail["bits"][:8]
    info["route_mismatches"] = detail["route"][:4]
    return _report(W, row, flags, info, result_dir, rank, f"l1o-{kind}")


def _spawn_entry(rank: int, W: int, row, forced, kind: str, port: int, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, forced, kind, port, result_dir, rank)
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


def _launch(row, forced, kind: str) -> None:
    import json

    W = row[1]
    tag = f"l1o-{kind}"
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            forced,
            kind,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_l1o_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, forced, kind, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, f"{tag}-{_row_id(row)}.json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


def _key_row(name: str):
    """The key row ``name`` on the running device: three documents over ``W x S`` tokens, ``S`` the smallest token
    count of a key at the row's world size and C 512 (``None`` off the measured parts or without such a key)."""

    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp as conv_kcp,
    )
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        kcp_jit,
    )

    W, layout, widths = KEY_ROWS[name]
    device = torch.device("cuda", 0)
    keys = kcp_jit.L1O_ROUTE.get((conv_kcp._device_arch(device), int(kcp_jit.sm_count(0))))
    tokens = sorted(S for w, C, S in keys or () if w == W and C == widths[0])
    if not tokens:
        return None
    T = W * tokens[0]
    lengths = (T * 3 // 10, T * 45 // 100, T - T * 3 // 10 - T * 45 // 100)
    return (lengths, W, layout, widths)


@pytest.mark.parametrize("case", FORCED_ROWS, ids=lambda case: _row_id(case[0]))
def test_l1o_route_forced_single_call(case) -> None:
    if not _supported():
        pytest.skip("the non-deterministic conv package is not rendered or does not support this device")
    row, forced = case
    _launch(row, forced, "single")


@pytest.mark.parametrize("case", FORCED_MULTI_ROWS, ids=lambda case: _row_id(case[0]))
def test_l1o_route_forced_multi_call(case) -> None:
    if not _supported():
        pytest.skip("the non-deterministic conv package is not rendered or does not support this device")
    row, forced = case
    _launch(row, forced, "multi")


@pytest.mark.parametrize("name", sorted(KEY_ROWS))
def test_l1o_route_on_a_measured_key(name) -> None:
    if not _supported():
        pytest.skip("the non-deterministic conv package is not rendered or does not support this device")
    if torch.cuda.device_count() < KEY_ROWS[name][0]:
        pytest.skip(f"{KEY_ROWS[name][0]} CUDA devices required, {torch.cuda.device_count()} visible")
    row = _key_row(name)
    if row is None:
        pytest.skip("the running device has no L1o key at this world size and C 512 (not a measured part)")
    _launch(row, (), "key")
