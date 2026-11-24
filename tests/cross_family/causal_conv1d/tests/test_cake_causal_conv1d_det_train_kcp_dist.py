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

Multi-GPU rows of ``cake_causal_conv1d_det_kcp`` under ``torch.utils.checkpoint`` with and without a recompute handle:
``W`` ranks over NCCL, each holding the fragments of one partition of a packed
batch (contiguous, zigzag, per-document zigzag, ragged tables with a zero-token rank, one-token fragments and a
fragment that reads its history from three predecessor ranks).  Per row and rank:

- ``y`` is bitwise the single-rank zero-history convolution of the whole pack on the rank's tokens (anchor);
- ``y`` / ``dx`` / ``dweight`` of the checkpoint with a handle (``use_reentrant`` False and True), of the plain
  checkpoint and of the direct pass are bitwise equal;
- the ``torch.distributed`` collectives issued (and the waits on their work handles) are counted per phase by
  test-local shims: the original forward and the backward issue one halo exchange each (positive control, > 0),
  the plain checkpoint's recompute issues one more, the recompute with a handle issues none;
- ``retain_graph`` + a second backward replays the entry again (no collective) and accumulates exactly twice;
- nested checkpoints: a handle created outside the outer region makes both recomputes communication-free; a handle
  created inside the outer region is empty in the outer recompute, which then exchanges once (still bitwise);
- the handle retains exactly the history pool: ``nbytes() == n_hist_rows * C * 2``, the bytes still allocated after
  the checkpointed forward are that pool (0 for the plain checkpoint);
- every output repeats bit for bit over ``REPEATS`` explicit forward + backward calls per launch form (every rank
  passing the same ``cooperative``): ``y``, ``hist``, ``dx``, ``dweight`` and the halo rows of both exchanges (the
  rows the rank sends, among them the backward's ``dpre`` head rows, the halo adjoints, and the pools it receives);
  ``y`` / ``hist`` / ``dx`` / halo rows are the same in the two forms, and the autograd entry's default is the
  explicit call of ``COOPERATIVE_DEFAULT``;
- the two-launch forward (knobs ``fwd_main`` x ``fwd_overlap``, the same on every rank): ``LAST_CALL["fwd"]`` names
  the form the rank ran (the overlapped form on a rank with tokens, the history-row program when the rank receives
  history rows) and ``y`` / ``hist`` are bitwise the one-launch forward's;
- the direct pass on 32 independent draws of the whole pack, judged against the naive BF16 tier of the whole pack
  (``tests.cake.test_helpers.cake_naive``): ``y`` / ``dx`` on the rank's rows and the ranks' ``dweight`` partials
  added in rank order, each by its error against the float64 oracle in the L2 and the max norm (median and largest
  over the draws at most 1.005 times the tier's, 0 wherever the tier's is 0);
- a steady explicit forward + backward of each form issues no host-to-device copy, no synchronizing operation and
  no device allocation (``tests/experimental/test_helpers/cake_host_sync.py``), bitwise the first call;
- per-document zigzag rows: the longest document is a chain of ``2 W`` fragments over every rank.

``ROWS`` are W = 2 / 4 / 8; ``F_ROWS`` are W = 2 / 4 at C = 528 and 568 around the forward's token floor of the
channel count on the running architecture (``cake_jit.NARROW_ROW_FLOORS["fwd"]``), run with the forward's small-S rows
removed (:func:`_small_s_rows_removed`): its ``floor:`` rows take the ranks' token counts from that table at run time
(a rank at the floor and one a token below it), its other rows fixed counts (the comment above ``F_ROWS``);
``S_ROWS`` are W = 2 / 4 at C = 576 with the ranks' token counts on both sides of the ends of the ``fwd`` route's
static, direct and pair direct ranges (the forward's small-S rows, read from ``cake_jit``), so the ranks' single-rank
main programs are the static, the pair direct, the direct and the dynamic program; ``ROWS_SCALE`` are per-document
zigzag at W = 16 / 32 / 64 (at W = 64 a document of 128 fragments, and documents of one- and two-token fragments
whose history comes from up to three predecessor ranks).
Two launch forms (as ``tests/experimental/test_cake_kda_det_train_kcp_dist.py``):
``torch.multiprocessing`` (``pytest``: skipped when fewer than ``W`` CUDA devices are visible) and one process per Slurm
task (``SLURM_NTASKS == W``; the rendezvous from ``MASTER_ADDR`` / ``MASTER_PORT``; rows of another ``W`` are skipped
under a multi-task step).  The ``ROWS_SCALE`` rows run in a multi-node Slurm step of ``W`` tasks.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WIDTH = 4

# (document lengths, W, layout, C)
ROWS = [
    ((100, 640, 77, 1000, 129), 2, "contiguous", 512),
    ((3000,), 2, "zigzag", 256),
    ((100, 640, 77, 1000, 129), 4, "contiguous", 512),
    ((3000,), 4, "zigzag_per_document", 256),
    ((2048, 5, 3000, 1), 4, "zigzag_per_document", 384),
    ((1200 + 900 + 900,), 4, "ragged:1200,0,900,900", 256),
    ((5000, 1003), 4, "ragged:1,2,4997,1003", 512),
    ((3000,), 8, "zigzag", 256),
    ((2048, 5, 3000, 1), 8, "zigzag_per_document", 384),
    ((6, 1, 2, 3, 4000), 8, "ragged:1,1,1,3,0,2,4000,4", 256),
]
# Rows of C = 528 and 568 around the forward's token floor of the channel count on the running architecture
# (``cake_jit.NARROW_ROW_FLOORS["fwd"]``; the nondet package's ``FORWARD_NARROW_ROW_FLOORS`` holds the same floors):
# from its floor the single-rank main program of a rank runs the narrower tiles of the row, below it the single tile.
# The rows run with ``FALLBACK_ROWS``, ``DIRECT_ROWS`` and ``PAIR_ROWS`` empty (:func:`_small_s_rows_removed`; in nondet
# also the backward keys of ``FALLBACK_ROWS`` and ``DIRECT_ROWS``, which no KCP call reads): the forward's small-S rows
# cover most of these token counts, where a rank runs the static, the direct or the pair direct program instead
# (``S_ROWS`` runs those programs).
# The ``floor:`` rows (C = 528, W = 2 and 4) take their ranks' token counts from that table at run time
# (:func:`_floor_row`; ``f`` is the floor): a rank at the floor and one a token below it, so on every architecture one
# rank runs the narrower tiles and another the single tile; the last document takes the tokens the others leave.  A
# row of fixed counts crosses the floor of an architecture where the floor lies above its smallest count and at most
# its largest (C = 528: 26000 / 10003 and 24576 / 24575 / 1 / 9000 tokens; C = 568: 20000 / 6003 and 16384 / 12288 /
# 12287 / 3044 tokens).  Kept out of ``ROWS`` (other suites import it).
F_ROWS = [
    ((20000, 10003, 6000), 2, "ragged:26000,10003", 528),
    ((30000, 19151, 9001), 4, "ragged:24576,24575,1,9000", 528),
    ((9000, 11000, 6003), 2, "ragged:20000,6003", 568),
    ((20000, 15000, 9000, 3), 4, "ragged:16384,12288,12287,3044", 568),
    ((20000, 10003, "rest"), 2, "floor:f,f-1", 528),
    ((30000, 19000, "rest"), 4, "floor:f,f-1,1,3", 528),
]


def _small_s_rows() -> list:
    """Rows of C = 576 per architecture whose ``fwd`` route has a static and a direct range there
    (``cake_jit.FALLBACK_ROWS`` / ``DIRECT_ROWS``): at W = 2 the ranks hold the static range's last token count
    ``S_x`` and one more, at W = 4 ``S_x``, ``S_x + 1``, the direct range's last count ``S_hi`` and one more (the
    static, direct and dynamic programs; the pair direct program where a ``PAIR_ROWS`` range holds the count); and
    where it has a pair direct range (``cake_jit.PAIR_ROWS``), at W = 2 the ranks hold one token below the range's
    first count ``P_lo`` and ``P_lo``, at W = 4 ``P_lo - 1``, ``P_lo``, the range's last count ``P_hi`` and one more
    (static, pair direct, pair direct, and the static or the direct program); a first document of three quarters of
    the tokens spans every rank."""

    from cake_pplx._backends.cake_causal_conv1d_det_train import cake_jit

    rows = []
    for arch in cake_jit.ARCHES:
        static = cake_jit.fallback_row_range(arch, "fwd", 576)
        direct = cake_jit.direct_row_range(arch, "fwd", 576)
        pair = cake_jit.pair_row_range(arch, "fwd", 576)
        sides = []
        if static is not None and direct is not None:
            s_x, s_hi = static[1], direct[1]
            sides += [(s_x, s_x + 1), (s_x, s_x + 1, s_hi, s_hi + 1)]
        if pair is not None:
            p_lo, p_hi = pair
            sides += [(p_lo - 1, p_lo), (p_lo - 1, p_lo, p_hi, p_hi + 1)]
        for counts in sides:
            total = sum(counts)
            lengths = (total - total // 4, total // 4)
            row = (lengths, len(counts), "ragged:" + ",".join(map(str, counts)), 576)
            if row not in rows:
                rows.append(row)
    return rows


# Kept out of ``ROWS`` (other suites import it).
S_ROWS = _small_s_rows()
# Per-document zigzag at W = 16 / 32 / 64: every document is cut into 2 W fragments, so the longest is a chain over
# every rank (128 fragments at W = 64); the short documents give one- and two-token fragments and empty ones.  Kept
# out of ``ROWS`` (other suites import it).
ROWS_SCALE = [
    ((2048, 5, 3000, 1), 16, "zigzag_per_document", 384),
    ((6000, 70, 2, 1000), 32, "zigzag_per_document", 256),
    ((12800, 200, 100, 3), 64, "zigzag_per_document", 256),
]
TENSORS = ("y", "dx", "dw")
# explicit forward + backward calls per launch form of the determinism check
REPEATS = 5


def _row_id(row) -> str:
    lengths, W, layout, C = row
    return "s" + "_".join(map(str, lengths)) + f"-w{W}-{layout.split(':')[0]}-c{C}"


def _floor_row(row, arch: str):
    """``row`` with the token counts of a ``floor:`` layout on ``arch``, as :func:`_partition` takes it (other rows
    unchanged).  The layout lists the ranks' token counts: ``f`` is the forward's token floor of ``C`` on ``arch``
    (``cake_jit.NARROW_ROW_FLOORS["fwd"]``, checked equal to the nondet package's ``FORWARD_NARROW_ROW_FLOORS``),
    ``f-N`` a count ``N`` tokens below it, a number that count; the last document (``"rest"``) takes the tokens the
    other documents leave."""

    lengths, W, layout, C = row
    if not layout.startswith("floor:"):
        return row
    from cake_pplx._backends.cake_causal_conv1d_det_train import cake_jit as det_jit
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        cake_jit as nondet_jit,
    )

    # the floors are keyed by channel vectors of 8 BF16 channels
    floor = det_jit.NARROW_ROW_FLOORS["fwd"][arch][C // 8]
    assert nondet_jit.FORWARD_NARROW_ROW_FLOORS[arch][C // 8] == floor, (arch, C)
    t_r = [
        floor + int(t[1:] or 0) if t.startswith("f") else int(t)
        for t in layout.split(":")[1].split(",")
    ]
    *docs, rest = lengths
    docs.append(sum(t_r) - sum(docs))
    # a rank at or above the floor and one below it; the last document holds tokens
    assert rest == "rest" and docs[-1] > 0 and min(t_r) < floor <= max(t_r), row
    return tuple(docs), W, "ragged:" + ",".join(map(str, t_r)), C


@contextlib.contextmanager
def _small_s_rows_removed(row):
    """While a row of :data:`F_ROWS` runs, ``FALLBACK_ROWS``, ``DIRECT_ROWS`` and ``PAIR_ROWS`` of both packages'
    ``cake_jit`` are empty (in nondet also the backward keys of ``FALLBACK_ROWS`` and ``DIRECT_ROWS``, which no KCP
    call reads), so each rank's single-rank main program is the one the token floor decides: the narrower tiles from
    the floor, the single tile below it.  Every other row runs with the rows of record."""

    if row not in F_ROWS:
        yield
        return
    from cake_pplx._backends.cake_causal_conv1d_det_train import cake_jit as det_jit
    from cake_pplx._backends.cake_causal_conv1d_nondet_train import (
        cake_jit as nondet_jit,
    )

    with pytest.MonkeyPatch.context() as m:
        for jit in (det_jit, nondet_jit):
            m.setattr(jit, "FALLBACK_ROWS", {})
            m.setattr(jit, "DIRECT_ROWS", {})
            m.setattr(jit, "PAIR_ROWS", {})
        yield


def _partition(partition, lengths, W: int, layout: str):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if not layout.startswith("ragged:"):
        return partition(cu, W, layout=layout), cu
    t_r = [int(x) for x in layout.split(":")[1].split(",")]
    assert len(t_r) == W and sum(t_r) == cu[-1]
    # contiguous ragged split of the global buffer, cut at document boundaries, in local order
    nested = [[] for _ in range(W)]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return partition(cu, W, fragments=nested), cu


def _local(part, rank: int, cu: list[int], t: torch.Tensor) -> torch.Tensor:
    parts = [
        t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in part.fragments[rank]
    ]
    return torch.cat(parts, 0).contiguous() if parts else t[:0].clone()


class _Counter:
    """Counts the ``torch.distributed`` collectives issued and the waits on their work handles, by phase (``orig``,
    ``recompute`` -- set by the checkpoint's ``context_fn``, re-entered per recompute -- and ``bwd``).  The shims
    replace the module attributes the conv package calls while the counter is entered; no tensor is read."""

    NAMES = (
        "all_to_all_single",
        "batch_isend_irecv",
        "all_gather_object",
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "broadcast",
        "reduce_scatter_tensor",
        "isend",
        "irecv",
        "send",
        "recv",
    )

    def __init__(self) -> None:
        self.phase = "orig"
        self.calls: dict[str, int] = {}
        self.waits: dict[str, int] = {}
        self._saved: dict[str, object] = {}
        self._stack: list[str] = []

    def _bump(self, table: dict[str, int]) -> None:
        table[self.phase] = table.get(self.phase, 0) + 1

    def _work(self, w):
        counter = self

        class _Work:
            def __init__(self, inner) -> None:
                self._inner = inner

            def wait(self, *a, **kw):
                counter._bump(counter.waits)
                return self._inner.wait(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Work(w) if hasattr(w, "wait") else w

    def __enter__(self) -> "_Counter":
        for name in self.NAMES:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._saved[name] = fn

            def shim(*a, _fn=fn, **kw):
                self._bump(self.calls)
                r = _fn(*a, **kw)
                if isinstance(r, list):
                    return [self._work(x) for x in r]
                return self._work(r)

            setattr(dist, name, shim)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(dist, name, fn)
        self._saved.clear()

    def recompute_cm(self):
        counter = self

        class _Phase:
            def __enter__(self):
                counter._stack.append(counter.phase)
                counter.phase = "recompute"
                return self

            def __exit__(self, *exc):
                counter.phase = counter._stack.pop()
                return False

        return _Phase()

    def get(self, table: str, phase: str) -> int:
        return getattr(self, table).get(phase, 0)


def _requested(device) -> int:
    return int(torch.cuda.memory_stats(device).get("requested_bytes.all.current", 0))


def _run_rank(
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
) -> dict:
    from torch.utils.checkpoint import checkpoint

    import cake_pplx.cake_causal_conv1d_det_train as conv
    import cake_pplx.cake_causal_conv1d_det_train_kcp as kcp_api
    from cake_pplx._backends.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )
    from cake_pplx._backends.cake_kda_det_train_kcp import (
        cake_kcp_plan as plan,
    )

    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    lengths, _w, layout, C = _floor_row(row, conv_kcp._device_arch(device))
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
    g = torch.Generator(device="cpu").manual_seed(7815)
    x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
    dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    x = _local(part, rank, cu, x_all).to(device)
    dy = _local(part, rank, cu, dy_all).to(device)
    w = w_all.to(device)
    n_hist = int(ctx.halo.n_hist_rows)

    def leaves():
        return x.detach().clone().requires_grad_(
            True
        ), w.detach().clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return kcp_api.cake_causal_conv1d_det_kcp(x_, w_, ctx, recompute_handle=h_)

    def run(mode: str, handle=None, *, twice: bool = False, reentrant: bool = False):
        """mode: direct | ckpt | nested_out (handle outside both regions) | nested_in (handle made in the outer)."""
        xl, wl = leaves()
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
                    # the product with ones saves a tensor of the outer region (so the outer recompute runs) and
                    # changes no bit of y or of its gradient
                    hh = (
                        conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
                        if mode == "nested_in"
                        else h_
                    )
                    yi = checkpoint(fn, x_, w_, hh, use_reentrant=False)
                    return yi * torch.ones_like(yi)

                y = checkpoint(
                    outer,
                    xl,
                    wl,
                    handle,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )
            cc.phase = "bwd"
            y.backward(dy, retain_graph=twice)
            if twice:
                y.backward(dy)
        out = {"y": y.detach(), "dx": xl.grad, "dw": wl.grad}
        counts = {
            t: {p: cc.get(t, p) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def bits(t: torch.Tensor) -> torch.Tensor:
        # bit view: -0.0 vs 0.0 and NaN payloads count as differences
        return t.contiguous().view(
            torch.int16 if t.element_size() == 2 else torch.int32
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
        )

    def eq(a, b) -> bool:
        return all(same(a[n], b[n]) for n in TENSORS)

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"n_hist_rows": n_hist, "T_r": int(x.shape[0])}
    direct, c_direct = run("direct")
    plain, c_plain = run("ckpt")
    hd = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    withh, c_h = run("ckpt", hd)
    plain_re, c_plain_re = run("ckpt", reentrant=True)
    hd_re = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    withh_re, c_h_re = run("ckpt", hd_re, reentrant=True)
    info["counts"] = {
        "direct": c_direct,
        "plain": c_plain,
        "handle": c_h,
        "plain_reentrant": c_plain_re,
        "handle_reentrant": c_h_re,
    }
    # anchor: y is the single-rank zero-history convolution of the whole pack on this rank's tokens
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    y_single = conv.cake_causal_conv1d_det(x_all.to(device), w, cu_dev)
    flags["y_eq_single_rank"] = same(direct["y"], _local(part, rank, cu, y_single))
    flags["plain_eq_direct"] = eq(plain, direct)
    flags["handle_eq_plain"] = eq(withh, plain)
    flags["handle_eq_direct"] = eq(withh, direct)
    flags["reentrant_plain_eq_direct"] = eq(plain_re, direct)
    flags["reentrant_handle_eq_direct"] = eq(withh_re, direct)
    for t in ("calls", "waits"):
        # positive control: one halo exchange (and its wait) in the forward and one in the backward on every rank
        flags[f"positive_control_{t}"] = c_direct[t] == {
            "orig": 1,
            "recompute": 0,
            "bwd": 1,
        } and c_plain[t] == {"orig": 1, "recompute": 1, "bwd": 1}
        flags[f"handle_counts_{t}"] = c_h[t] == {"orig": 1, "recompute": 0, "bwd": 1}
        # reentrant: the recompute runs inside the backward (no context_fn): plain = exchange + recompute exchange
        flags[f"reentrant_counts_{t}"] = c_plain_re[t] == {
            "orig": 1,
            "recompute": 0,
            "bwd": 2,
        } and c_h_re[t] == {"orig": 1, "recompute": 0, "bwd": 1}
    flags["handle_bytes"] = (
        hd.nbytes() == n_hist * C * 2 == hd_re.nbytes()
        and hd.entries["conv"]["hist"].shape == (n_hist, C)
    )
    # retain_graph: the entry is replayed again, no collective; exactly twice the gradients
    hd2 = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    twice, c_twice = run("ckpt", hd2, twice=True)
    info["counts"]["handle_twice"] = c_twice
    flags["twice_accumulates"] = (
        same(twice["dx"], direct["dx"] + direct["dx"])
        and same(twice["dw"], direct["dw"] + direct["dw"])
        and c_twice["calls"] == {"orig": 1, "recompute": 0, "bwd": 2}
    )
    # nested checkpoints
    nest_out, c_nest_out = run(
        "nested_out", conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    )
    nest_in, c_nest_in = run(
        "nested_in", conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    )
    info["counts"]["nested_out"], info["counts"]["nested_in"] = c_nest_out, c_nest_in
    flags["nested_out_bitwise"] = eq(nest_out, direct)
    flags["nested_in_bitwise"] = eq(nest_in, direct)
    # handle made outside both regions: the outer recompute (recompute phase) and the inner recompute (counted in
    # the backward phase, outside the outer's context) replay the entry; only the backward exchange remains
    flags["nested_out_counts"] = c_nest_out["calls"] == {
        "orig": 1,
        "recompute": 0,
        "bwd": 1,
    }
    # handle made inside the outer region: the outer recompute's inner forward is an original forward (exchanges)
    flags["nested_in_counts"] = c_nest_in["calls"] == {
        "orig": 1,
        "recompute": 1,
        "bwd": 1,
    }
    # retained bytes after a checkpointed forward (after the warm-up above): the history pool with a handle, 0 without
    retained = {}
    for name, h in (
        ("plain", None),
        ("handle", conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()),
    ):
        xl, wl = leaves()
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

    # Every output repeats bit for bit: REPEATS explicit forward + backward calls per launch form, the same form on
    # every rank.  The halo rows of both exchanges are recorded: a copy of the rows this rank sends (history rows,
    # then the dpre head rows -- the halo adjoints), taken before the exchange, and the pool it receives.
    def explicit(coop: bool) -> dict:
        rows: list[torch.Tensor] = []
        exchange = conv_kcp._exchange

        def recording(send, **kw):
            sent = send.detach().clone()
            recv, work = exchange(send, **kw)
            rows.extend((sent, recv))
            return recv, work

        conv_kcp._exchange = recording
        try:
            y_e, hist_e = conv_kcp.cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
            dx_e, dw_e = conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
                x, w, ctx, dy, hist_e, cooperative=coop
            )
        finally:
            conv_kcp._exchange = exchange
        torch.cuda.synchronize(device)
        return {"y": y_e, "hist": hist_e, "dx": dx_e, "dw": dw_e, "halo": rows}

    def same_halo(a: dict, b: dict) -> bool:
        # one exchange per direction: (sent, received) of the forward, then of the backward
        return len(a["halo"]) == len(b["halo"]) == 4 and all(
            same(s, t) for s, t in zip(a["halo"], b["halo"], strict=True)
        )

    forms = {}
    for coop in (True, False):
        ref = forms[coop] = explicit(coop)
        ok = True
        for _ in range(REPEATS - 1):
            again = explicit(coop)
            ok = (
                ok
                and same_halo(again, ref)
                and all(same(again[k], ref[k]) for k in ("y", "hist", "dx", "dw"))
            )
        flags["repeats_bitwise_" + ("cooperative" if coop else "split")] = ok
    flags["forms_agree_except_dweight"] = same_halo(forms[True], forms[False]) and all(
        same(forms[True][k], forms[False][k]) for k in ("y", "hist", "dx")
    )
    flags["autograd_default_eq_explicit_default"] = eq(
        direct, forms[conv_kcp.COOPERATIVE_DEFAULT]
    )

    # The two-launch forward: every (fwd_main, fwd_overlap) form, the same on every rank, records its form in
    # LAST_CALL["fwd"] (the overlapped form with the exchange in flight on a rank with tokens; the history-row program
    # when the rank receives history rows and the form has two launches) and gives y / hist bitwise the one-launch
    # forward's.
    fwd_bitwise, fwd_recorded = True, True
    for fwd_main in (0, 1):
        for fwd_overlap in (0, 1):
            conv_kcp._set_test_knobs({"fwd_main": fwd_main, "fwd_overlap": fwd_overlap})
            try:
                for _ in range(2):
                    y_f, hist_f = conv_kcp.cake_causal_conv1d_det_train_kcp_forward(
                        x, w, ctx
                    )
                    torch.cuda.synchronize(device)
                    want = (
                        {
                            "main": "sr" if fwd_main else "f",
                            "form": "pair" if fwd_overlap else "wait",
                            "repair": bool(fwd_main or fwd_overlap) and n_hist > 0,
                        }
                        if x.shape[0]
                        else {"main": None, "form": "wait", "repair": False}
                    )
                    fwd_recorded = fwd_recorded and conv_kcp.LAST_CALL["fwd"] == want
                    fwd_bitwise = (
                        fwd_bitwise
                        and same(y_f, forms[True]["y"])
                        and same(hist_f, forms[True]["hist"])
                    )
            finally:
                conv_kcp._set_test_knobs(None)
                conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()
    flags["fwd_forms_bitwise"] = fwd_bitwise
    flags["fwd_forms_recorded"] = fwd_recorded

    # Against the naive tier of the whole pack over the draws: every rank draws the same pack from the draw's seed (draw
    # 0 is the pack above), runs the direct pass on its tokens and judges ``y`` / ``dx`` on its rows and the ranks'
    # dweight partials, all-gathered and added in rank order (the same order on every rank).
    from tests.cake.test_helpers.cake_causal_conv1d_oracle import judge_draws, problem

    rows = _local(part, rank, cu, torch.arange(T)).to(device)

    def launch(_draw: int, seed: int):
        gd = torch.Generator(device="cpu").manual_seed(seed)
        xa = torch.randn(T, C, generator=gd).to(torch.bfloat16)
        wa = (torch.randn(C, WIDTH, generator=gd) * 0.5).contiguous()
        dya = torch.randn(T, C, generator=gd).to(torch.bfloat16)
        xl = _local(part, rank, cu, xa).to(device).requires_grad_(True)
        wl = wa.to(device).requires_grad_(True)
        y_d = kcp_api.cake_causal_conv1d_det_kcp(xl, wl, ctx)
        y_d.backward(_local(part, rank, cu, dya).to(device))
        partials = [torch.empty_like(wl.grad) for _ in range(W)]
        dist.all_gather(partials, wl.grad.contiguous())
        dw_sum = partials[0].clone()
        for partial in partials[1:]:
            dw_sum += partial
        outputs = {"y": y_d.detach(), "dx": xl.grad, "dweight": dw_sum}
        return outputs, problem(
            xa.to(device), wa.to(device), cu_dev, None, dya.to(device), rows=rows
        )

    judged = judge_draws(launch, seed=7815, label=_row_id(row))
    info["judge_failures"] = judged[:8]
    flags["judged_against_the_naive_tier"] = not judged

    # Steady state: an explicit forward + backward of each form after the calls above issues no host-to-device copy,
    # no synchronizing operation (the probe queued before it still runs when the host returns) and no device
    # allocation or free; bitwise the first calls.
    from tests.experimental.test_helpers.cake_host_sync import (
        allocator_counts,
        census,
        nonblocking,
        summary,
    )

    alloc_ok = allocator_counts(device)

    def steady() -> list:
        y_s, hist_s = conv_kcp.cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
        out = [y_s, hist_s]
        for coop in (True, False):
            out += conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
                x, w, ctx, dy, hist_s, cooperative=coop
            )
        return out

    # after allocator_counts' empty_cache: fill the allocator's cache again
    for _ in range(2):
        steady()
    c = census(steady, device)
    info["steady_census"] = summary(c)
    expected = [forms[True]["y"], forms[True]["hist"], forms[True]["dx"]]
    expected += [forms[True]["dw"], forms[False]["dx"], forms[False]["dw"]]
    flags["steady_nonblocking"] = nonblocking(c) and c["pinned"] == 0
    flags["steady_no_device_alloc"] = not alloc_ok or (
        (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (0, 0, 0)
        and c["prof_malloc"] in (None, 0)
    )
    flags["steady_bitwise"] = all(
        same(a, b) for a, b in zip(c["out"], expected, strict=True)
    )
    info["allocator_counts"] = alloc_ok

    # Per-document zigzag: the longest document is a chain of 2 W fragments over every rank (128 at W = 64).
    if layout == "zigzag_per_document":
        chain = max(
            sum(1 for frs in part.fragments for f in frs if f.doc == d and f.length)
            for d in range(len(lengths))
        )
        info["longest_chain"] = chain
        flags["chain_of_2w_fragments"] = chain == 2 * W
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
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        with _small_s_rows_removed(row):
            report = _run_rank(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from cake_pplx.cake_causal_conv1d_det_train import (
        is_cake_causal_conv1d_det_train_supported,
    )

    return is_cake_causal_conv1d_det_train_supported(torch.device("cuda"))


def _skip_without_allocator_counts(report: dict) -> None:
    # every other flag passed; the no-device-allocation flag is unmeasured on a rank whose allocator does not count
    if not all(info["allocator_counts"] for info in report["info_by_rank"]):
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


@pytest.mark.parametrize("row", ROWS + F_ROWS + S_ROWS + ROWS_SCALE, ids=_row_id)
def test_recompute_handle_rows(row) -> None:
    """Bitwise and per-phase collective counts of the checkpoint with / without a recompute handle, repeats per launch
    form, the direct pass judged against the naive tier and the steady-state census (module docstring)."""
    W = row[1]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip("the conv package does not support this device")
        with _small_s_rows_removed(row):
            report = _run_rank(
                int(os.environ["SLURM_PROCID"]),
                W,
                row,
                int(os.environ.get("MASTER_PORT", "29500")),
                os.environ.get("KCP_TEST_RESULT_DIR"),
                int(os.environ.get("SLURM_LOCALID", "0")),
            )
        assert report["pass"], report
        _skip_without_allocator_counts(report)
        return
    if not _supported():
        pytest.skip("cake_causal_conv1d_det_train needs a supported CUDA device")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
    _skip_without_allocator_counts(report)
