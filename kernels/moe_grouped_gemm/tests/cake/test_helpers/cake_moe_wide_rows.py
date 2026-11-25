# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""Wide outputs of the ragged MoE grouped GEMM packages: the checks the deterministic and the
non-deterministic suites share (``tests/experimental/test_cake_moe_grouped_gemm_{det,nondet}_train_wide_rows.py``).

A wide output is a forward or data-gradient ``out`` whose span ``sum_m * out.stride(0)`` passes
``int32_elems`` (2^31 - 1) elements, the kernels' 32-bit element addressing.  On the architectures of
``wide_rows_archs`` the host launches the stage selected on the whole call once per row chunk
(``cake_backend.row_chunks``; ``cake_backend`` module docstring, wide outputs), each chunk with its row
table computed on the device; on the other architectures it refuses the call.

On the CPU: the chunk arithmetic (:func:`check_row_chunks`), the row tables a chunk computes against the
rows of each group in the chunk (:func:`check_row_tables`), the plan and binding of a call within the span
(one launch, bound as a call that never chunks: :func:`check_within_span`), the binding of a wide call on
meta tensors (:func:`check_wide_binding`) and the refusals (:func:`check_refusals`).

On a GPU (an architecture of ``wide_rows_archs``, but :func:`check_forced_chunks`, which runs on every
architecture):

* :func:`check_forced_chunks`: ``int32_elems`` lowered so that a small call takes several chunks; every
  forward and data-gradient stage the route serves, ``[E]`` and ``m_indptr`` tables, a chunk of no row and
  a first group past the first chunk give the bits of the same call in one launch;
* :func:`check_issue_rows`: the issue's shapes (4096 output columns over a reduction of 1024: the down
  projection's forward and the gate/up projection's data gradient) at 655,360 rows over 64 or 32 skewed
  groups (empty groups, one group of 90 % of the rows) and at the boundary row counts 524,287 (one
  launch), 524,288, 524,289 and 1,048,576 (2^32 elements): judged against the naive BF16 tier over the
  family's draws (:func:`cake_moe_judge.judge_draws`; the whole output poisoned with :data:`SENTINEL`
  before each launch, so an unwritten row fails its contract), and windows of rows (the first and last
  rows, each chunk boundary, row 524,288) bitwise against an independent call within the span over the
  window's rows alone (:func:`check_windows`); at 524,287 rows the one launch gives the bits of the same
  call in forced chunks;
* :func:`check_slice`: an ``m_indptr`` device slice (first entry above 0, a group across a chunk
  boundary) into ``out`` as a slice of a taller and wider buffer: bitwise the ``[E]`` call over the whole
  table on the slice's rows, every other element of the buffer untouched;
* :func:`check_dynamic_rows`: one prepared launch over device routings of 300,000, 524,288 and 655,360
  used rows, eager and replayed from one CUDA graph, under ``torch.cuda.set_sync_debug_mode("error")``:
  bitwise a fresh preparation per routing, the rows past the routing untouched, no program loaded
  after the first launch;
* :func:`check_checkpoint_block`: an MoE expert block (gate/up, ``silu(a) * b``, down) through the
  autograd entry under ``torch.utils.checkpoint(..., use_reentrant=False)``, bitwise the same block
  without the checkpoint (its data gradients and forward wide, its weight gradients reading operands of
  more than 2^31 elements); :func:`check_wgrad_operands` judges such a weight gradient against the
  naive BF16 tier;
* :func:`check_repeat`: three launches into outputs prefilled with zero words and with
  :data:`SENTINEL`, and a fresh preparation, give the same bits.

Each GPU check skips when the device has less free memory than it needs (:func:`require_memory`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers import cake_moe_device_slice as ds
from tests.cake.test_helpers.cake_moe_judge import OUTPUT_NAMES, judge_draws

ROW_OPS = ("fwd", "dgrad")
#: The output columns and the reduction of the issue's shapes: the down projection's forward
#: (``w [E, 4096, 1024]``) and the gate/up projection's data gradient (``w [E, 1024, 4096]``).
OUT_COLS = 4096
REDUCTION = 1024
#: The receive buffer of the issue, the most rows whose span fits at 4096 columns, and the boundary
#: row counts: that one (one launch), the next two and 2^20 (2^32 elements).
ISSUE_ROWS = 655_360
LAST_FIT = (2**31 - 1) // OUT_COLS
BOUNDARY_ROWS = (LAST_FIT, LAST_FIT + 1, LAST_FIT + 2, 1 << 20)
#: The group counts of the issue (64 and 32 local experts).
ISSUE_GROUPS = (64, 32)
#: The share of the rows the dominant group of a skewed distribution holds.
DOMINANT_FRACTION = 0.9
#: Rows on each side of a window centre (:func:`check_windows`).
WINDOW = 512
#: The forced chunks (:func:`check_forced_chunks`): groups with empty and one-row groups, rows of no
#: group after them (a whole chunk of them), the columns and the rows per chunk.
FORCED_SIZES = (0, 700, 1, 2900, 0, 1500, 333, 1200)
FORCED_TAIL = 1000
FORCED_OUT_COLS = 1024
FORCED_REDUCTION = 512
FORCED_CHUNK_ROWS = 1000
#: The routings of :func:`check_dynamic_rows`: rows used and the seed of their skewed groups.
DYNAMIC_ROUTINGS = ((300_000, 1), (LAST_FIT + 1, 2), (ISSUE_ROWS, 3))
GIB = 1 << 30
BF16 = torch.bfloat16


# ---------------------------------------------------------------------------
# Shapes, groups and operands
# ---------------------------------------------------------------------------


def dims(
    op: str, out_cols: int = OUT_COLS, reduction: int = REDUCTION
) -> tuple[int, int, int]:
    """``(N, K, columns of the row operand)`` of ``op`` with ``out_cols`` output columns over a
    reduction of ``reduction`` (``w [E, N, K]``; the forward's row operand is ``x [rows, K]``, the data
    gradient's ``g [rows, N]``)."""
    if op == "fwd":
        return out_cols, reduction, reduction
    return reduction, out_cols, reduction


def sizes(rows: int, groups: int, dist: str = "skewed", seed: int = 260) -> list[int]:
    """``groups`` group sizes summing to ``rows``: ``"balanced"`` (equal, the first ``rows % groups`` one
    row more) or ``"skewed"``: one group of ``round(0.9 * rows)`` rows, ``max(2, groups // 8)`` empty
    groups and the rest ragged (log-normal weights, at least one row each), the indices drawn from
    ``seed``."""
    if dist == "balanced":
        q, r = divmod(rows, groups)
        return [q + (1 if e < r else 0) for e in range(groups)]
    rng = random.Random(seed * 1_000_003 + rows * 131 + groups)
    order = list(range(groups))
    rng.shuffle(order)
    dominant, empty = order[0], set(order[1 : 1 + max(2, groups // 8)])
    rest = [e for e in range(groups) if e != dominant and e not in empty]
    big = round(DOMINANT_FRACTION * rows)
    left = rows - big
    out = [0] * groups
    out[dominant] = big
    weights = [math.exp(rng.gauss(0.0, 1.5)) for _ in rest]
    total = sum(weights)
    parts = [1 + int((left - len(rest)) * x / total) for x in weights]
    for _ in range(left - sum(parts)):
        parts[rng.randrange(len(rest))] += 1
    for e, part in zip(rest, parts, strict=True):
        out[e] = part
    assert sum(out) == rows and min(out) >= 0
    return out


def bounds(group_sizes: Sequence[int], first: int = 0) -> list[tuple[int, int]]:
    """``(start, end)`` rows of each group, the first from row ``first``."""
    out, row = [], first
    for size in group_sizes:
        out.append((row, row + size))
        row += size
    return out


def ends(
    group_sizes: Sequence[int], device, *, first: int | None = None
) -> torch.Tensor:
    """The device row table of ``group_sizes``: ``offs[E]`` (cumulative ends), or with ``first``
    the ``m_indptr[E + 1]`` whose groups start at row ``first``."""
    table = [] if first is None else [first]
    row = 0 if first is None else first
    for size in group_sizes:
        row += size
        table.append(row)
    return torch.tensor(table, dtype=torch.int32, device=device)


def fill_operands(op: str, a: torch.Tensor, w: torch.Tensor, seed: int) -> None:
    """Draw the row operand ``a`` and ``w [E, N, K]`` in place from ``seed`` (BF16 normal; ``x`` and ``w``
    scaled by ``1 / sqrt(K)`` as in the sibling suites)."""
    gen = torch.Generator(device=a.device).manual_seed(seed)
    scale = 1.0 / math.sqrt(w.shape[2])
    a.normal_(generator=gen)
    w.normal_(generator=gen).mul_(scale)
    if op == "fwd":
        a.mul_(scale)


def operands(
    op: str,
    rows: int,
    groups: int,
    device,
    seed: int,
    *,
    out_cols: int = OUT_COLS,
    reduction: int = REDUCTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a, w)``: the row operand ``[rows, cols]`` and ``w [groups, N, K]`` of ``op``
    (:func:`dims`), drawn from ``seed`` (:func:`fill_operands`)."""
    n, k, cols = dims(op, out_cols, reduction)
    a = torch.empty(rows, cols, dtype=BF16, device=device)
    w = torch.empty(groups, n, k, dtype=BF16, device=device)
    fill_operands(op, a, w, seed)
    return a, w


def out_cols_of(op: str, w: torch.Tensor) -> int:
    """The output columns of ``op`` with weights ``w [E, N, K]``."""
    return int(w.shape[1] if op == "fwd" else w.shape[2])


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


def require_memory(device, gib: float) -> None:
    """Skip unless ``device`` has ``gib`` GiB free (after torch returns its cached blocks)."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(device)
    if free < gib * GIB:
        pytest.skip(
            f"needs {gib:.0f} GiB of free device memory, {free / GIB:.1f} GiB free"
        )


def require_wide(P: ds.Package, *ops: str) -> tuple[torch.device, str]:
    """The CUDA device and its architecture, after skipping unless the package serves ``ops`` there
    (:func:`cake_moe_device_slice.require`) and lists the architecture in ``wide_rows_archs``."""
    device = ds.require(P, *ops)
    arch = ds.arch_of(P, device)
    if arch not in P.constants["wide_rows_archs"]:
        pytest.skip(
            f"wide outputs are refused on {arch} (not in wide_rows_archs): their row chunks are not "
            "validated there"
        )
    return device, arch


def wide_record(launch) -> dict[str, Any]:
    """:func:`cake_moe_device_slice.launch_record` with the row chunks, for ``record_property``."""
    record = ds.launch_record(launch)
    record["launches"] = launch.plan["launches"]
    if "row_chunks" in launch.plan:
        record["row_chunks"] = [list(c) for c in launch.plan["row_chunks"]]
    return record


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def check_constants(P: ds.Package) -> None:
    """The shipped constants chunk on sm_100a, sm_103a and sm_107a, at the 32-bit element limit."""
    c = P.constants
    assert c["wide_rows_archs"] == ["sm_100a", "sm_103a", "sm_107a"], c[
        "wide_rows_archs"
    ]
    assert int(c["int32_elems"]) == 2**31 - 1


def check_row_chunks(P: ds.Package) -> None:
    """``row_chunks``: the issue's row counts, and over a grid of rows and pitches the chunks tile the
    rows in order, each fits the span, there are ``ceil(sum_m / R)`` of them, all but the last of
    ``ceil(sum_m / n)`` rows; one chunk when the span fits; a pitch past the span is refused."""
    row_chunks = P.cb.row_chunks
    span = int(P.constants["int32_elems"])
    assert row_chunks(ISSUE_ROWS, OUT_COLS, span) == ((0, 327_680), (327_680, 327_680))
    assert row_chunks(LAST_FIT, OUT_COLS, span) == ((0, LAST_FIT),)
    assert row_chunks(LAST_FIT + 1, OUT_COLS, span) == (
        (0, 262_144),
        (262_144, 262_144),
    )
    assert row_chunks(LAST_FIT + 2, OUT_COLS, span) == (
        (0, 262_145),
        (262_145, 262_144),
    )
    assert row_chunks(1 << 20, OUT_COLS, span) == (
        (0, 349_526),
        (349_526, 349_526),
        (699_052, 349_524),
    )
    assert row_chunks(699_060, OUT_COLS + 64, span) == (
        (0, 349_530),
        (349_530, 349_530),
    )
    for ldc in (1, 7, 256, 4096, 4160, 7168, 1 << 20):
        most = span // ldc
        for sum_m in (
            0,
            1,
            most - 1,
            most,
            most + 1,
            2 * most,
            2 * most + 1,
            5 * most + 3,
        ):
            chunks = row_chunks(sum_m, ldc, span)
            what = (sum_m, ldc, chunks[:3])
            count = max(1, -(-sum_m // most))
            assert len(chunks) == count, what
            rows = -(-sum_m // count)
            row = 0
            for i, (start, r) in enumerate(chunks):
                assert start == row and r * ldc <= span, what
                assert r == rows or (i == count - 1 and r <= rows), what
                row += r
            assert row == sum_m, what
    with pytest.raises(ValueError, match="row stride exceeds"):
        row_chunks(2, span + 1, span)


def _group_rows(lo: int, hi: int, start: int, rows: int) -> int:
    """Rows of the group ``[lo, hi)`` inside the chunk ``[start, start + rows)``."""
    return max(0, min(hi, start + rows) - max(lo, start))


def check_row_tables(P: ds.Package) -> None:
    """``_RowChunkTable.fill`` on host tensors: for ``[E]`` and ``m_indptr`` tables (empty groups, a first
    entry above 0, rows of no group after the last group) and every chunk of several chunkings, group
    ``e`` of the chunk's table holds exactly the rows of group ``e`` inside the chunk, from its first
    row less the chunk's start; the caller's table is unchanged."""
    cb = P.cb
    rng = random.Random(7)
    layouts = [
        ([0, 700, 1, 2900, 0, 1500, 333, 1200], 0, 1000),
        ([5, 0, 0, 9000, 17, 0], 2500, 300),
        ([4000], 0, 0),
        ([0, 0, 0], 100, 50),
        ([2, 0, 3], 1, 2),
        ([rng.randrange(0, 900) for _ in range(40)], 1234, 777),
    ]
    for group_sizes, first, tail in layouts:
        total = first + sum(group_sizes) + tail
        for lead in (False, True):
            if not lead and first:
                continue
            if lead:
                table = ends(group_sizes, "cpu", first=first)
                spans = bounds(group_sizes, first)
            else:
                table = ends(group_sizes, "cpu")
                spans = bounds(group_sizes)
            keep = table.clone()
            for most in (1, 7, 333, 1000, max(1, total // 2), total, total + 1):
                if -(-total // most) > 64:
                    continue
                for start, rows in cb.row_chunks(total, 1, most):
                    buffer = torch.full_like(table, -12345)
                    cb._RowChunkTable(table, start, rows, buffer).fill()
                    got = buffer.tolist()
                    what = (group_sizes, first, tail, lead, most, start, rows, got)
                    assert all(0 <= v <= rows for v in got), what
                    for e, (lo, hi) in enumerate(spans):
                        if lead:
                            c_lo, c_hi = got[e], got[e + 1]
                        else:
                            c_lo, c_hi = (got[e - 1] if e else 0), got[e]
                        count = _group_rows(lo, hi, start, rows)
                        assert c_hi - c_lo == count, (e, what)
                        if count:
                            assert c_lo == max(lo, start) - start, (e, what)
            assert torch.equal(table, keep)


def meta_call(
    P: ds.Package,
    op: str,
    rows: int,
    groups: int,
    *,
    lead: bool = False,
    out_cols: int = OUT_COLS,
    reduction: int = REDUCTION,
    row_pad: int = 0,
):
    """Prepare ``op`` on meta tensors (under :func:`cake_moe_device_slice.fake_device`): ``rows`` rows,
    ``groups`` groups, an ``[E]`` table or (``lead``) an ``m_indptr`` view into a larger table, ``out``
    rows of ``out_cols + row_pad`` elements.  Returns ``(launch, a, w, table, out)``."""
    meta = torch.device("meta")
    n, k, cols = dims(op, out_cols, reduction)
    a = torch.empty(rows, cols, dtype=BF16, device=meta)
    w = torch.empty(groups, n, k, dtype=BF16, device=meta)
    if lead:
        table = torch.empty(groups + 3, dtype=torch.int32, device=meta)[1 : groups + 2]
    else:
        table = torch.empty(groups, dtype=torch.int32, device=meta)
    out = torch.empty(rows, out_cols + row_pad, dtype=BF16, device=meta)[:, :out_cols]
    return P.prepare[op](a, w, table, out=out), a, w, table, out


def _plan(
    P: ds.Package, op: str, rows: int, groups: int, sms: int, arch: str, n: int, k: int
):
    """``(stage, plan)`` of one launch of ``op`` over ``rows`` rows (the host's choice)."""
    cb, c = P.cb, P.constants
    name = P.jit.select_module(op, arch)
    stage, tile_n = cb.select_row_stage(
        name, op, n, k, arch, c, sum_m=rows, num_groups=groups
    )
    plan = cb.plan_fwd_dgrad(
        op,
        sum_m=rows,
        num_groups=groups,
        n=n,
        k=k,
        sm_count=sms,
        tile_n=tile_n,
        constants=c,
        arch=arch,
    )
    return stage, plan


def check_within_span(monkeypatch, P: ds.Package, op: str) -> None:
    """A call whose span fits (up to 524,287 rows of 4096 columns, every architecture) plans and binds
    one launch as a call that never chunks: the plan of ``plan_fwd_dgrad`` (no ``row_chunks``), one
    stage, no row table and no buffer, the operands, ``out`` and the caller's table themselves."""
    ds.ensure_lead_registry(monkeypatch, P)
    n, k, _cols = dims(op)
    for arch in ds.ARCHES:
        for sms in (148, 152, 216):
            with monkeypatch.context() as scope:
                ds.fake_device(scope, P, arch, sms)
                binds = ds.record_binds(scope, P)
                for rows in (1, 300_000, LAST_FIT):
                    for groups in ISSUE_GROUPS:
                        for lead in (False, True):
                            binds.clear()
                            launch, a, w, table, out = meta_call(
                                P, op, rows, groups, lead=lead
                            )
                            what = (arch, sms, op, rows, groups, lead)
                            stage, plan = _plan(P, op, rows, groups, sms, arch, n, k)
                            assert launch.plan == plan, (what, launch.plan, plan)
                            assert launch.stages == (stage,), what
                            assert (
                                launch.row_tables == () and launch.workspaces == ()
                            ), what
                            assert len(binds) == 1, what
                            kwargs = binds[0][2]
                            assert kwargs["A"] is a and kwargs["B"] is w, what
                            assert kwargs["C"] is out, what
                            assert kwargs["sum_m"] == rows, what
                            assert kwargs["ldc"] == out.stride(0), what
                            assert kwargs["grid"] == plan["grid"], what
                            assert (kwargs["c0"], kwargs["c1"]) == (
                                plan["c0"],
                                plan["c1"],
                            )
                            if lead:
                                ds.assert_m_begin_view(binds, table, str(what))
                            else:
                                assert (
                                    kwargs["offs"] is table and "m_begin" not in kwargs
                                )


#: The wide calls of :func:`check_wide_binding`: rows and the padding of each row of ``out``.
WIDE_BINDING_CASES = (
    (LAST_FIT + 1, 0),
    (ISSUE_ROWS, 0),
    (1 << 20, 0),
    (ISSUE_ROWS, 64),
)


def _assert_wide_binds(
    P: ds.Package, op, arch, sms, rows, groups, lead, row_pad, binds
):
    """One case of :func:`check_wide_binding` (``binds`` empty before it)."""
    span = int(P.constants["int32_elems"])
    n, k, cols = dims(op)
    launch, a, w, table, out = meta_call(
        P, op, rows, groups, lead=lead, row_pad=row_pad
    )
    what = (arch, sms, op, rows, row_pad, groups, lead)
    ldc = out.stride(0)
    chunks = P.cb.row_chunks(rows, ldc, span)
    assert len(chunks) >= 2, what
    stage, plan = _plan(P, op, rows, groups, sms, arch, n, k)
    plans = [_plan(P, op, r, groups, sms, arch, n, k)[1] for _s, r in chunks]
    grids = tuple(p["grid"] for p in plans)
    want = dict(plan, launches=len(chunks), row_chunks=chunks, grids=grids)
    assert launch.plan == want, (what, launch.plan)
    assert launch.stages == (stage,) * len(chunks), what
    assert len(binds) == len(chunks) == len(launch.row_tables), what
    name = P.jit.select_module(op, arch)
    route = P.jit.lead_route(name) if lead else name
    assert launch.bound_routes == (route,) * len(chunks), what
    buffers = tuple(rt.buffer for rt in launch.row_tables)
    assert launch.workspaces == buffers, what
    for (start, r), cplan, (bname, bstage, kw), rt in zip(
        chunks, plans, binds, launch.row_tables, strict=True
    ):
        where = (what, start, r)
        assert bname == route and bstage == stage, where
        assert kw["A"].shape == (r, cols), where
        assert kw["A"].storage_offset() == start * a.stride(0), where
        assert kw["B"] is w, where
        # a dense out's rows, else the flat span of the rows
        span_shape = ((r - 1) * ldc + OUT_COLS,) if row_pad else (r, OUT_COLS)
        assert kw["C"].shape == span_shape, where
        assert kw["C"].storage_offset() == start * ldc, where
        assert kw["sum_m"] == r and kw["ldc"] == ldc, where
        assert kw["grid"] == cplan["grid"], where
        assert (kw["c0"], kw["c1"]) == (cplan["c0"], cplan["c1"]), where
        assert rt.table is table and (rt.start, rt.rows) == (start, r), where
        buf = rt.buffer
        assert buf.shape == table.shape and buf.dtype == torch.int32, where
        if lead:
            assert kw["m_begin"]._base is buf and kw["offs"]._base is buf, where
            assert kw["m_begin"].storage_offset() == 0, where
            assert kw["offs"].storage_offset() == 1, where
            assert kw["offs"].shape == (groups,), where
        else:
            assert kw["offs"] is buf and "m_begin" not in kw, where


def check_wide_binding(monkeypatch, P: ds.Package, op: str) -> None:
    """A wide call on the architectures of ``wide_rows_archs`` (meta tensors; :data:`WIDE_BINDING_CASES`:
    524,288, 655,360 and 1,048,576 rows of 4096 columns, and 655,360 rows at a padded pitch): the whole
    call's plan and stage with ``launches``, ``row_chunks`` and ``grids``; per chunk one bind of that
    stage on the route (the lead twin for an ``m_indptr``) over its rows of the row operand and of
    ``out``, ``sum_m`` its rows, the grid and queue split planned for its rows, and its row table: a
    buffer of the caller's table's shape (``offs``, or ``m_begin`` and the ends), retained by the launch
    and filled from the caller's table by ``launch()``."""
    ds.ensure_lead_registry(monkeypatch, P)
    for arch in P.constants["wide_rows_archs"]:
        for sms in (148, 152, 216):
            with monkeypatch.context() as scope:
                ds.fake_device(scope, P, arch, sms)
                binds = ds.record_binds(scope, P)
                for rows, row_pad in WIDE_BINDING_CASES:
                    for groups in ISSUE_GROUPS:
                        for lead in (False, True):
                            binds.clear()
                            _assert_wide_binds(
                                P, op, arch, sms, rows, groups, lead, row_pad, binds
                            )


def check_refusals(monkeypatch, P: ds.Package, op: str) -> None:
    """A wide call on an architecture outside ``wide_rows_archs`` (every architecture is in the shipped
    constants, so each one is taken out of them in turn) is refused with ``ValueError`` before anything
    is bound, while a call within the span there binds; an ``out`` whose row stride alone passes the
    span is refused on sm_100a too."""
    ds.ensure_lead_registry(monkeypatch, P)
    c = P.constants
    outside = [a for a in ds.ARCHES if a not in c["wide_rows_archs"]]
    assert outside == [], outside
    span = int(c["int32_elems"])
    for arch in ds.ARCHES:
        with monkeypatch.context() as scope:
            scope.setitem(
                c, "wide_rows_archs", [a for a in c["wide_rows_archs"] if a != arch]
            )
            ds.fake_device(scope, P, arch, 216)
            binds = ds.record_binds(scope, P)
            for rows in (LAST_FIT + 1, ISSUE_ROWS):
                for lead in (False, True):
                    with pytest.raises(
                        ValueError,
                        match=f"exceeds the 32-bit element addressing.*not yet validated on {arch}",
                    ):
                        meta_call(P, op, rows, 64, lead=lead)
            assert binds == []
            launch = meta_call(P, op, LAST_FIT, 64)[0]
            assert launch.plan["launches"] == 1 and len(binds) == 1
    with monkeypatch.context() as scope:
        ds.fake_device(scope, P, "sm_100a", 148)
        binds = ds.record_binds(scope, P)
        meta = torch.device("meta")
        n, k, cols = dims(op)
        storage = torch.empty(span + 1 + OUT_COLS, dtype=BF16, device=meta)
        out = storage.as_strided((2, OUT_COLS), (span + 1, 1))
        a = torch.empty(2, cols, dtype=BF16, device=meta)
        w = torch.empty(1, n, k, dtype=BF16, device=meta)
        table = torch.empty(1, dtype=torch.int32, device=meta)
        with pytest.raises(ValueError, match="row stride exceeds"):
            P.prepare[op](a, w, table, out=out)
        assert binds == []


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def check_forced_chunks(
    monkeypatch, P: ds.Package, op: str, form: str, record_property
) -> None:
    """Stage ``form`` forced (:func:`cake_moe_device_slice.force_row_form`) and ``int32_elems`` lowered
    to :data:`FORCED_CHUNK_ROWS` rows of the output: the chunked call gives the bits of the same call
    in one launch (both into :data:`SENTINEL`, so the rows of no group stay unwritten in both), for
    the ``[E]`` table from row 0 (rows of no group fill the last chunk) and for an ``m_indptr`` slice
    whose first group starts past the first chunks (chunks of no row).  The device's architecture is
    added to ``wide_rows_archs`` for the test when it is not listed, so the chunking itself is checked
    on every architecture."""
    device = ds.require(P, op)
    arch = ds.arch_of(P, device)
    c = P.constants
    if not ds.force_row_form(monkeypatch, P, op, form, arch):
        pytest.skip(f"no {form} {op} form on {arch}")
    if not P.jit.stage_served(P.jit.select_module(op, arch), form, arch):
        pytest.skip(f"the {op} route serves no {form} stage on {arch}")
    if arch not in c["wide_rows_archs"]:
        monkeypatch.setitem(c, "wide_rows_archs", [*c["wide_rows_archs"], arch])
    rows = sum(FORCED_SIZES) + FORCED_TAIL
    groups = len(FORCED_SIZES)
    a, w = operands(
        op,
        rows,
        groups,
        device,
        seed=31,
        out_cols=FORCED_OUT_COLS,
        reduction=FORCED_REDUCTION,
    )
    table = ends(FORCED_SIZES, device, first=0)
    e0 = 4  # the slice's first group starts at row 3601, past the first three chunks
    cases = (("[E]", w, table[1:]), ("m_indptr", w[e0:], table[e0:]))
    limit = FORCED_CHUNK_ROWS * FORCED_OUT_COLS
    record = {}
    for label, wc, tc in cases:
        what = f"{op} {form} {label}"
        one = P.prepare[op](
            a, wc, tc, out=ds.sentinel((rows, FORCED_OUT_COLS), BF16, device)
        )
        with monkeypatch.context() as scope:
            scope.setitem(c, "int32_elems", limit)
            chunked = P.prepare[op](
                a, wc, tc, out=ds.sentinel((rows, FORCED_OUT_COLS), BF16, device)
            )
        chunks = P.cb.row_chunks(rows, FORCED_OUT_COLS, limit)
        assert one.plan["launches"] == 1 and "row_chunks" not in one.plan, what
        assert chunked.plan["row_chunks"] == chunks and len(chunks) >= 7, what
        assert one.stages == (form,) and chunked.stages == (form,) * len(chunks), what
        if label == "m_indptr":
            first = sum(FORCED_SIZES[:e0])
            assert first >= 3 * chunks[0][1], (what, first, chunks[:3])
        else:
            assert chunks[-1][0] >= sum(FORCED_SIZES), (what, chunks[-1])
        one.launch()
        chunked.launch()
        torch.cuda.synchronize(device)
        ds.assert_same_bits(chunked.out, one.out, what)
        start = 0 if label == "[E]" else sum(FORCED_SIZES[:e0])
        ds.assert_untouched(one.out[:start], f"{what}: rows before the groups")
        ds.assert_untouched(
            one.out[sum(FORCED_SIZES) :], f"{what}: rows after the groups"
        )
        record[label] = wide_record(chunked)
    record_property("bound", record)


def windows(total: int, chunks: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The windows of :func:`check_windows`: the first and the last :data:`WINDOW` * 2 rows, and
    :data:`WINDOW` rows on each side of every chunk start and of row ``LAST_FIT + 1`` (524,288)."""
    centres = {start for start, _rows in chunks[1:]}
    if LAST_FIT + 1 < total:
        centres.add(LAST_FIT + 1)
    spans = {(0, min(total, 2 * WINDOW)), (max(0, total - 2 * WINDOW), total)}
    for centre in centres:
        spans.add((max(0, centre - WINDOW), min(total, centre + WINDOW)))
    return sorted(spans)


def check_windows(
    P: ds.Package,
    op: str,
    a: torch.Tensor,
    w: torch.Tensor,
    group_sizes: Sequence[int],
    out: torch.Tensor,
    chunks: Sequence[tuple[int, int]],
    what: str,
) -> None:
    """Each window of :func:`windows` of ``out`` (the output of a call over ``group_sizes`` from row 0,
    every row in a group) holds the bits of an independent call within the span over the window's rows
    alone (``a[lo:hi]``, each group clipped to the window): every stage of a route gives the same bits,
    and a row's bits do not depend on its position."""
    total = sum(group_sizes)
    spans = bounds(group_sizes)
    for lo, hi in windows(total, chunks):
        clipped = [_group_rows(s, t, lo, hi - lo) for s, t in spans]
        sub = P.prepare[op](a[lo:hi], w, ends(clipped, a.device))
        assert sub.plan["launches"] == 1, (what, lo, hi)
        sub.launch()
        torch.cuda.synchronize(a.device)
        ds.assert_same_bits(out[lo:hi], sub.out, f"{what}: rows [{lo}, {hi})")


def check_issue_rows(
    monkeypatch,
    P: ds.Package,
    op: str,
    rows: int,
    groups: int,
    record_property,
    *,
    dist: str = "skewed",
) -> None:
    """``op`` at the issue's shape over ``rows`` rows in ``groups`` groups of ``dist``: the plan chunks
    as ``row_chunks`` says (one launch at 524,287 rows), every output judged against the naive BF16 tier
    over the family's draws with the whole output poisoned before each launch, then (on draw 0's
    inputs) the windows of :func:`check_windows`, and for a call within the span the bits of the same
    call in two forced chunks."""
    device, _arch = require_wide(P, op)
    _n, _k, cols = dims(op)
    out_cols = OUT_COLS
    need = rows * (cols * 2 + out_cols * 6) * 1.3 / GIB + 4
    require_memory(device, need)
    group_sizes = sizes(rows, groups, dist)
    table = ends(group_sizes, device)
    seed = 260_000 + rows % 1000 + groups
    a, w = operands(op, rows, groups, device, seed)
    out = ds.sentinel((rows, out_cols), BF16, device)
    launch = P.prepare[op](a, w, table, out=out)
    span = int(P.constants["int32_elems"])
    chunks = P.cb.row_chunks(rows, out_cols, span)
    what = f"{op} rows={rows} E={groups} {dist}"
    if len(chunks) == 1:
        assert launch.plan["launches"] == 1 and "row_chunks" not in launch.plan, what
    else:
        assert launch.plan["row_chunks"] == chunks, (what, launch.plan)
    record_property("bound", wide_record(launch))
    record_property("group_sizes", group_sizes)

    def draw(d) -> None:
        fill_operands(op, a, w, d.seed)
        ds.fill_word(out, ds.SENTINEL[BF16])
        launch.launch()
        d.op(OUTPUT_NAMES[op], out, op, a, w, group_sizes)

    def then() -> None:
        torch.cuda.synchronize(device)
        check_windows(P, op, a, w, group_sizes, out, chunks, what)
        if len(chunks) == 1:
            with monkeypatch.context() as scope:
                scope.setitem(P.constants, "int32_elems", (rows // 2 + 1) * out_cols)
                forced = P.prepare[op](
                    a, w, table, out=ds.sentinel((rows, out_cols), BF16, device)
                )
            assert (
                forced.plan["launches"] == 2 and forced.stages == launch.stages * 2
            ), what
            forced.launch()
            torch.cuda.synchronize(device)
            ds.assert_same_bits(forced.out, out, f"{what}: forced chunks")

    judge_draws(record_property, draw, seed=seed, then=then)


def _straddling_sizes(rows: int, groups: int) -> list[int]:
    """Skewed sizes with the dominant group in the middle and the first and last groups not empty, so
    the slice ``[1, groups - 1)`` starts above row 0, ends before the last row and holds the dominant
    group, which spans the middle row."""
    s = sizes(rows, groups, "skewed", seed=261)
    dom = s.index(max(s))
    mid = groups // 2
    s[dom], s[mid] = s[mid], s[dom]
    for edge in (0, groups - 1):
        if s[edge] == 0:
            donor = next(e for e in range(1, groups - 1) if e != mid and s[e] > 1)
            s[edge], s[donor] = 1, s[donor] - 1
    assert sum(s) == rows and s[0] and s[-1]
    return s


def check_slice(P: ds.Package, op: str, record_property) -> None:
    """An ``m_indptr`` device slice of groups ``[1, E - 1)`` (first entry above 0, the dominant group
    across the chunk boundary) at 655,360 rows into ``out`` = rows ``[64, 64 + rows)`` and the first
    4096 columns of a :data:`SENTINEL` buffer 128 rows taller and 64 columns wider (row pitch 4160):
    the lead twin binds every chunk, the slice's rows hold the bits of the ``[E]`` call over the
    whole table into a dense ``out`` (whose windows are checked as in :func:`check_issue_rows`), and
    every other element of the buffer keeps :data:`SENTINEL`."""
    device, _arch = require_wide(P, op)
    rows, groups, guard, pad = ISSUE_ROWS, 64, 64, 64
    _n, _k, cols = dims(op)
    require_memory(device, rows * (cols * 2 + OUT_COLS * 6) * 1.3 / GIB + 4)
    group_sizes = _straddling_sizes(rows, groups)
    table = ends(group_sizes, device, first=0)
    a, w = operands(op, rows, groups, device, seed=262)
    buffer = ds.sentinel((rows + 2 * guard, OUT_COLS + pad), BF16, device)
    out = buffer[guard : guard + rows, :OUT_COLS]
    e0, e1 = 1, groups - 1
    lead = P.prepare[op](a, w[e0:e1], table[e0 : e1 + 1], out=out)
    assert all(P.jit.is_lead_route(r) for r in lead.bound_routes), lead.bound_routes
    chunks = lead.plan["row_chunks"]
    spans = bounds(group_sizes)
    r0, r1 = spans[e0][0], spans[e1 - 1][1]
    assert r0 > 0 and r1 < rows, (r0, r1)
    assert any(s < start < t for s, t in spans[e0:e1] for start, _r in chunks[1:]), (
        chunks,
        r0,
        r1,
    )
    whole = P.prepare[op](
        a, w, table[1:], out=ds.sentinel((rows, OUT_COLS), BF16, device)
    )
    lead.launch()
    whole.launch()
    torch.cuda.synchronize(device)
    what = f"{op} slice [{e0}, {e1}) rows [{r0}, {r1})"
    ds.assert_same_bits(out[r0:r1], whole.out[r0:r1], what)
    ds.assert_untouched(buffer[: guard + r0], f"{what}: rows before the slice")
    ds.assert_untouched(buffer[guard + r1 :], f"{what}: rows after the slice")
    ds.assert_untouched(buffer[guard + r0 : guard + r1, OUT_COLS:], f"{what}: padding")
    check_windows(P, op, a, w, group_sizes, whole.out, whole.plan["row_chunks"], what)
    record_property("bound", {"slice": wide_record(lead), "whole": wide_record(whole)})


def check_dynamic_rows(P: ds.Package, op: str, record_property) -> None:
    """One launch prepared over a 655,360-row buffer (64 groups, ``offs[E]`` on the device): for each
    routing of :data:`DYNAMIC_ROUTINGS` copied into ``offs`` on the device, a ``launch()`` and a replay
    of one CUDA graph that captured it (each into ``out`` refilled with :data:`SENTINEL`, under
    ``torch.cuda.set_sync_debug_mode("error")``) give the bits of a fresh preparation over that
    routing, leave the rows past the routing untouched, and load no program after the first launch
    (``cake_jit._load_module``)."""
    device, _arch = require_wide(P, op)
    rows, groups = ISSUE_ROWS, 64
    _n, _k, cols = dims(op)
    require_memory(device, rows * (cols * 2 + OUT_COLS * 6) * 1.3 / GIB + 4)
    a, w = operands(op, rows, groups, device, seed=263)
    routings = []
    for used, seed in DYNAMIC_ROUTINGS:
        routings.append((used, ends(sizes(used, groups, "skewed", seed=seed), device)))
    offs = routings[-1][1].clone()
    out = ds.sentinel((rows, OUT_COLS), BF16, device)
    with ds.no_host_sync():
        launch = P.prepare[op](a, w, offs, out=out)
        launch.launch()  # stream workspaces (and the queue probe) before the capture
    torch.cuda.synchronize(device)
    assert launch.plan["launches"] == 2, launch.plan
    loads = P.jit._load_module.cache_info().misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch.launch()
    torch.cuda.synchronize(device)
    reference = ds.sentinel((rows, OUT_COLS), BF16, device)
    for used, routed in routings:
        fresh = P.prepare[op](
            a, w, routed, out=ds.fill_word(reference, ds.SENTINEL[BF16])
        )
        fresh.launch()
        for mode in ("eager", "graph"):
            with ds.no_host_sync():
                offs.copy_(routed)
                ds.fill_word(out, ds.SENTINEL[BF16])
                if mode == "eager":
                    launch.launch()
                else:
                    graph.replay()
            torch.cuda.synchronize(device)
            what = f"{op} {mode} over {used} rows"
            ds.assert_same_bits(out[:used], reference[:used], what)
            ds.assert_untouched(out[used:], f"{what}: rows past the routing")
    assert P.jit._load_module.cache_info().misses == loads
    record_property("bound", wide_record(launch))


def check_checkpoint_block(P: ds.Package, record_property) -> None:
    """An MoE expert block through the autograd entry at 655,360 rows over 64 skewed groups (hidden
    4096, expert width 1024): ``h = mm(x, w1)`` (gate/up, ``w1 [E, 2048, 4096]``), ``silu(a) * b`` on
    its halves, ``y = mm(., w2)`` (down, ``w2 [E, 4096, 1024]``: a wide forward), and its backward
    with ``dy`` (a wide data gradient of gate/up; weight gradients whose operands ``dy`` and ``x``
    pass 2^31 elements): under ``torch.utils.checkpoint(..., use_reentrant=False)`` ``y`` and the
    gradients of ``x``, ``w1`` and ``w2`` are the bits of the block without the checkpoint, and
    finite."""
    device, _arch = require_wide(P, "fwd", "dgrad", "wgrad")
    rows, groups, hidden, inter = ISSUE_ROWS, 64, OUT_COLS, REDUCTION
    require_memory(device, 72)
    group_sizes = sizes(rows, groups, "skewed", seed=264)
    offs = ends(group_sizes, device)
    gen = torch.Generator(device=device).manual_seed(264)
    x0 = torch.empty(rows, hidden, dtype=BF16, device=device).normal_(generator=gen)
    x0.mul_(1.0 / math.sqrt(hidden))
    w1 = torch.empty(groups, 2 * inter, hidden, dtype=BF16, device=device)
    w1.normal_(generator=gen).mul_(1.0 / math.sqrt(hidden))
    w2 = torch.empty(groups, hidden, inter, dtype=BF16, device=device)
    w2.normal_(generator=gen).mul_(1.0 / math.sqrt(inter))
    dy = torch.empty(rows, hidden, dtype=BF16, device=device).normal_(generator=gen)

    def block(x, w_1, w_2, table):
        a, b = P.gemm(x, w_1, table).chunk(2, -1)
        return P.gemm(torch.nn.functional.silu(a) * b, w_2, table)

    def run(checkpoint: bool) -> list[torch.Tensor]:
        leaves = [t.detach().requires_grad_(True) for t in (x0, w1, w2)]
        if checkpoint:
            y = torch.utils.checkpoint.checkpoint(
                block, *leaves, offs, use_reentrant=False
            )
        else:
            y = block(*leaves, offs)
        y.backward(dy)
        return [y.detach()] + [leaf.grad for leaf in leaves]

    plain = run(False)
    torch.cuda.synchronize(device)
    ckpt = run(True)
    torch.cuda.synchronize(device)
    for name, got, want in zip(
        ("y", "x.grad", "w1.grad", "w2.grad"), ckpt, plain, strict=True
    ):
        ds.assert_same_bits(got, want, f"checkpointed block {name}")
        bad = int((~want.isfinite()).sum())
        assert bad == 0, f"block {name}: {bad} elements not finite"
    record_property("group_sizes", group_sizes)


def check_wgrad_operands(P: ds.Package, record_property) -> None:
    """A weight gradient whose operands pass 2^31 elements (``g [655,360, 4096]``, ``x [655,360,
    1024]``, 64 skewed groups, an fp32 ``dW``): judged against the naive BF16 tier over the family's
    draws.  The weight gradient's own output ``[E, N, K]`` is far within the span; its operands are
    read through 64-bit TMA descriptors."""
    device, _arch = require_wide(P, "wgrad")
    rows, groups = ISSUE_ROWS, 64
    require_memory(device, rows * (OUT_COLS + REDUCTION) * 2 * 1.5 / GIB + 8)
    group_sizes = sizes(rows, groups, "skewed", seed=265)
    table = ends(group_sizes, device)
    g = torch.empty(rows, OUT_COLS, dtype=BF16, device=device)
    x = torch.empty(rows, REDUCTION, dtype=BF16, device=device)
    out = torch.empty(groups, OUT_COLS, REDUCTION, dtype=torch.float32, device=device)
    launch = P.prepare["wgrad"](g, x, table, out=out)
    record_property("bound", ds.launch_record(launch))

    def draw(d) -> None:
        gen = torch.Generator(device=device).manual_seed(d.seed)
        g.normal_(generator=gen)
        x.normal_(generator=gen).mul_(1.0 / math.sqrt(REDUCTION))
        ds.fill_word(out, ds.SENTINEL[torch.float32])
        launch.launch()
        d.op(OUTPUT_NAMES["wgrad"], out, "wgrad", g, x, group_sizes)

    judge_draws(record_property, draw, seed=265_000)


def check_repeat(P: ds.Package, op: str, record_property) -> None:
    """At 655,360 rows over 64 skewed groups: three launches of one prepared launch into ``out``
    prefilled with zero words, :data:`SENTINEL` and zero words again, and a fresh preparation, give
    the same bits."""
    device, _arch = require_wide(P, op)
    rows, groups = ISSUE_ROWS, 64
    _n, _k, cols = dims(op)
    require_memory(device, rows * (cols * 2 + OUT_COLS * 6) * 1.3 / GIB + 4)
    group_sizes = sizes(rows, groups, "skewed", seed=266)
    table = ends(group_sizes, device)
    a, w = operands(op, rows, groups, device, seed=266)
    out = torch.empty(rows, OUT_COLS, dtype=BF16, device=device)
    launch = P.prepare[op](a, w, table, out=out)
    first = None
    for word in (ds.POS_ZERO, ds.SENTINEL[BF16], ds.POS_ZERO):
        ds.fill_word(out, word)
        launch.launch()
        torch.cuda.synchronize(device)
        if first is None:
            first = out.clone()
        else:
            ds.assert_same_bits(out, first, f"{op} repeat into {word:#x}")
    fresh = P.prepare[op](a, w, table, out=ds.sentinel((rows, OUT_COLS), BF16, device))
    fresh.launch()
    torch.cuda.synchronize(device)
    ds.assert_same_bits(fresh.out, first, f"{op} fresh preparation")
    record_property("bound", wide_record(launch))
