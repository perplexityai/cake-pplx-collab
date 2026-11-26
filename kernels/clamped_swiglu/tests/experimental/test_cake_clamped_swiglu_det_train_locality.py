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

Tests of the ``locality="domain"`` route of the Cake routed-expert clamped SwiGLU training kernels: the
locality-domain twin of the forward on SM107 (R200), the memory whose chunks alternate between the two dies
(``cake_clamped_swiglu_det_train_domain_empty``), the route's plans
(``cake_clamped_swiglu_det_train_forward_domain_plan``, ``..._backward_domain_plan``) and
``cake_clamped_swiglu_det_train_domain_release``.  The backward stages have no twin: with ``"domain"`` they run their
``"off"`` programs (reason ``arch``).

The ``cpu_only`` tests check the pure parts: the twin record serves its role and never enters the row-count route or the
fine window, with the measured first rows per SM, and the backward stages have no twin record; the item decode covers
every row exactly once against the kernel's formulas (``cake_jit.domain_items``, ``domain_item_rows``), and at
``DOMAIN_MAX_ROWS`` rows from the last slot of a chunk gives a pure-Python reference's item counts and rows with every
index in 32 bits; the grid (and the backend's grid of the twin against it); where a tensor lies in a range of that
memory and the row frame (and ``slot_of``, the place of a contiguous tensor as one integer, against it; a range whose
chunk is not whole slots places nothing); the route's reasons in their order and the two classes (with fake tensors and
a fake registry), and the forward step's own routing of a common call against the route and the plan, and the twin
launch it keeps in a checkpoint's recomputation (the common call's arguments, its entry bound and its counters claimed
when it runs, its stream's workspace dropped when it raises); the first row counts a forward call can reach before its
placement is read (``_floors``), a call below them that reads neither the registry nor the SM table, takes no claim
counters and launches what the ``"off"`` step launches, and a grid of row counts, classes and memory on which the plan
keeps its reasons and the step and the full checks' path launch what it names; the plan's keys;
``locality`` validation (a non-``str`` is a ``TypeError``, another string a ``ValueError``, before any other check);
``"off"`` imports no domain module; the signatures (``locality="off"`` the last keyword of the steps and the autograd
entry; each plan's signature its step's without ``locality``); and, against a fake driver, the unwind of a mapping that
fails, the error without cuda-python, a range queued while the module's lock is held (a garbage collection inside an
allocation) freed when the lock is released, the registry entry removed before the range is unmapped, the claim
counters' first mapping, the per-thread stream key, the dies' rows of a workspace and the counters a launch passes,
``release`` and the SM table's checks; and the SM table itself: its packing (each SM's rank, its die's SM count, its die
and its supplemental flag), the rows it refuses, and the package's probe with a fake probe library (a table that covers
every SM, a launch that misses an SM taken again, a probe that never covers every SM or names a die outside 0 and 1 kept
as no table, so a ``"domain"`` call runs the ``"off"`` program with reason ``sm_table``).

The GPU tests (every architecture) check that ``"domain"`` on ordinary memory runs the ``"off"`` programs with the
``"off"`` bits, steps and autograd, and that on SM100 and SM103 the reason is ``arch`` and the helper raises.  The SM107
tests (skipped elsewhere; an SM107 whose driver reports fewer than two locality domains fails them) check the helper's
chunks lie in their dies, the refusal and the deferred free during a capture, the twin's bytes against the ``"off"``
bytes, merged and split, at four phases (row 0 at the start of a die-0 chunk, inside a chunk, in a die-1 chunk, a row
slice), on grids of 1 and 3 CTAs and the host's grid, in both classes; the backward, weighted and unweighted, on that
memory (its ``"off"`` program, reason ``arch``); each step running what its plan names, common calls and others, at
every reason of the route and in both classes; with the shipped first rows, at the ``homed`` first row count, the
autograd entry (plain and checkpointed) and a forward without ``out`` reading no registry and running the ``"off"``
program, and a forward whose ``out`` lies in the anchor's row frame running its twin; the claim counters left zero, a
launch that raises (in the step's own frame, in a call the step checks in full and in a held launch), two streams, CUDA
graph replay, autograd and a non-reentrant checkpoint, a saved-tensors hook that moves the saved tensors to the host on
a side stream, steady-state calls without host synchronization and the SM table's fields.

This suite names the backend package once, through ``tests.cake.test_helpers.cake_package``.
"""

from __future__ import annotations

import collections
import contextlib
import enum
import gc
import inspect
import random
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package

PACKAGE = "cake_clamped_swiglu_det_train"
pkg = cake_package(PACKAGE)
cake_jit = cake_package(PACKAGE, "cake_jit")
memory = cake_package(PACKAGE, "cake_domain")
locality = cake_package(PACKAGE, "cake_locality")

forward = pkg.cake_clamped_swiglu_det_train_forward
backward = pkg.cake_clamped_swiglu_det_train_backward
clamped_swiglu = pkg.cake_clamped_swiglu_det
domain_empty = pkg.cake_clamped_swiglu_det_train_domain_empty
forward_plan = pkg.cake_clamped_swiglu_det_train_forward_domain_plan
backward_plan = pkg.cake_clamped_swiglu_det_train_backward_domain_plan
domain_release = pkg.cake_clamped_swiglu_det_train_domain_release

ARCH = "sm_107a"
I = 2304
L = 10.0
STAGES = ("fwd", "bwd", "bwd_unweighted")
# The twin and its build: the forward's (a row a warp, 4 rows an item, 5 CTAs an SM).  The backward stages have none:
# with ``"domain"`` they run their ``"off"`` programs.
TWINS = {"fwd": {"block": 128, "item_rows": 4, "ctas_per_sm": 5}}
NO_TWIN = ("bwd", "bwd_unweighted")
# The measured first rows per SM of the twin, per class (a literal copy: a record that drifts fails here).
FIRST_ROWS = {"fwd": {ARCH: {"homed": 112, "anchor": 128}}}
# The chunk of each helper width, the slots of a chunk and of a row.
CHUNK = {I: 2097152, 2 * I: 4194304}
SLOTS = 4096
ROW_SLOTS = 9
REASONS = (
    "placed",
    "arch",
    "size",
    "sm_table",
    "input_memory",
    "input_layout",
    "overlap",
    "input_phase",
    "out_memory",
    "no_out",
)
PLAN_KEYS = {
    "stage",
    "route",
    "reason",
    "class",
    "program",
    "first_slot",
    "first_die",
    "chunk_shift",
    "item_rows",
    "grid",
    "homed",
    "outputs",
}


def _twin(stage: str) -> str:
    return cake_jit.PROGRAM_OF[stage + cake_jit.DOMAIN][ARCH]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _assert_same(what: str, got, want) -> None:
    if want is None:
        assert got is None, what
        return
    assert got.shape == want.shape and got.dtype == want.dtype, what
    if not torch.equal(_bits(got), _bits(want)):
        bad = (_bits(got) != _bits(want)).nonzero()
        raise AssertionError(f"{what}: {bad.shape[0]} of {got.numel()} values differ; first at {bad[0].tolist()}")


def _clear_twin_caches() -> None:
    """Drop the backend's twins and the first row counts the forward step keeps of each (``_FLOORS``): after a change
    of the first rows per SM, the architecture or the SM count, the next call makes them again."""
    pkg._twin.cache_clear()
    pkg._FLOORS.clear()


# --------------------------------------------------------------------------- CPU-only: the registry and the decode


@pytest.mark.cpu_only
def test_twin_records_serve_their_roles() -> None:
    """The forward's twin, on SM107 only, in the forward's locality-domain role, with the forward's lane layout, its
    grid rule, its claim workspace (one ``uint32`` word a die, re-armed by the kernel) and the measured first rows per
    SM; the backward stages have no twin.  ``route`` and ``fine`` never name the twin, so the ``"off"`` route is the
    0.1.10 route."""
    assert cake_jit.DOMAIN == "_domain"
    assert tuple(cake_jit.DOMAIN_REASONS) == REASONS
    assert tuple(cake_jit.DOMAIN_CLASSES) == ("homed", "anchor")
    assert cake_jit.DOMAIN_MAX_ROWS == 2**27
    twins = {name: record for name, record in cake_jit.MODULES.items() if record["role"].endswith("_domain")}
    assert {record["stage"] for record in twins.values()} == set(TWINS)
    for stage in NO_TWIN:
        assert cake_jit.PROGRAM_OF[stage + "_domain"] == {} and stage + "_domain" not in cake_jit.BUILD_OF, stage
        assert all(cake_jit.domain_program(stage, arch) is None for arch in cake_jit.ARCHES), stage
    for stage, facts in TWINS.items():
        name = _twin(stage)
        record = cake_jit.MODULES[name]
        assert cake_jit.PROGRAM_OF[stage + "_domain"] == {ARCH: name}
        assert record["arches"] == [ARCH] and record["role"] == stage + "_domain" and record["stage"] == stage
        assert tuple(record["launch"]["block"]) == (facts["block"], 1, 1)
        assert record["launch"]["dynamic_smem_bytes"] == 0 and record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False and "rows_per_sm_windows" not in record
        assert record["counters"] == {"dtype": "uint32", "rearm": "kernel", "words": 1}
        assert record["grid_rule"] == {
            "chunk_shift": 12,
            "ctas_per_sm": {ARCH: facts["ctas_per_sm"]},
            "item_rows": facts["item_rows"],
            "kind": "domain_rows",
            "max_chunk_shift": 27,
            "max_rows": 2**27,
            "row_slots": 9,
        }
        assert record["first_rows_per_sm"] == FIRST_ROWS[stage]
        assert cake_jit.domain_rule(name, ARCH) == (12, facts["item_rows"], facts["ctas_per_sm"])
        for arch in ("sm_100a", "sm_103a"):
            with pytest.raises(RuntimeError, match="vocabulary"):
                cake_jit.domain_rule(name, arch)
        assert cake_jit.domain_program(stage, ARCH) == name
        for arch in ("sm_100a", "sm_103a"):
            assert cake_jit.domain_program(stage, arch) is None
        base = {"lanes": 32, "rows_per_thread": 1, "width_bits": 128}
        assert cake_jit.BUILD_OF[stage + "_domain"] == base
        with pytest.raises(ValueError, match="twin"):
            cake_jit.grid(name, 1)
    assert cake_jit.DOMAIN_FIRST_ROWS_PER_SM == FIRST_ROWS
    for stage in STAGES:
        for arch in cake_jit.ARCHES:
            for sm_count in (148, 152, 208, 212):
                names = set(cake_jit.route(stage, arch, sm_count)[1]) | {cake_jit.fine(stage, arch, sm_count)[1]}
                assert not names & set(twins), (stage, arch, sm_count)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "change",
    [
        {"grid_rule": {"kind": "tiles"}},
        {"grid_rule": {"row_slots": 8}},
        {"grid_rule": {"chunk_shift": 28}},
        {"grid_rule": {"item_rows": 0}},
        {"grid_rule": {"max_rows": 2**26}},
        {"grid_rule": {"ctas_per_sm": {ARCH: 0}}},
        {"grid_rule": {"ctas_per_sm": 5}},
        {"grid_rule": {"ctas_per_sm": {ARCH: 5, "sm_100a": 5}}},
        {"counters": {"rearm": "host"}},
        {"counters": {"words": 0}},
        {"first_rows_per_sm": {"sm_100a": {"homed": 1, "anchor": None}}},
        {"first_rows_per_sm": {ARCH: {"homed": 0, "anchor": None}}},
        {"first_rows_per_sm": {ARCH: {"homed": 1.5, "anchor": None}}},
        {"first_rows_per_sm": {ARCH: {"homed": 1}}},
    ],
)
def test_a_twin_record_outside_the_vocabulary_is_refused(change, monkeypatch) -> None:
    name = _twin("fwd")
    record = {key: (dict(value) if isinstance(value, dict) else value) for key, value in cake_jit.MODULES[name].items()}
    for key, value in change.items():
        record[key] = {**record[key], **value} if key != "first_rows_per_sm" else value
    monkeypatch.setitem(cake_jit.MODULES, name, record)
    with pytest.raises(RuntimeError, match="vocabulary"):
        cake_jit.domain_rule(name, ARCH)


def _rows_of_items(rows: int, first_slot: int, first_die: int, shift: int, item_rows: int) -> list[int]:
    """Every row of every item the dies hold (``domain_items``), decoded as the kernel decodes a claim word."""
    items = cake_jit.domain_items(rows, first_slot, first_die, shift, item_rows)
    found = []
    for die, count in enumerate(items):
        for item in range(count):
            lo, hi = cake_jit.domain_item_rows(item * 2 + die, rows, first_slot, first_die, shift, item_rows)
            assert hi - lo <= item_rows, (rows, first_slot, first_die, shift, item_rows, die, item)
            found.extend(range(lo, hi))
            # an item's rows start in chunks of its die
            for row in range(lo, hi):
                chunk = (first_slot + ROW_SLOTS * row) >> shift
                assert first_die ^ (chunk & 1) == die, (rows, first_slot, first_die, shift, item_rows, row)
    return found


@pytest.mark.cpu_only
@pytest.mark.parametrize("shift", [0, 1, 2, 3, 4, 5, 12])
@pytest.mark.parametrize("item_rows", [1, 2, 3, 4, 7])
def test_the_items_cover_every_row_once(shift: int, item_rows: int) -> None:
    """For row counts from 1 to past two chunks, first slots at the start, inside and at the end of a chunk and
    both first dies, the dies' items hold every row exactly once, each item at most ``item_rows`` rows that start in a
    chunk of its die (the kernels' decode: ``domain_items`` and ``domain_item_rows``)."""
    span = 1 << shift
    firsts = sorted({0, 1, span // 2, span - 1, max(span - 9, 0)} & set(range(span)))
    counts = sorted({1, 2, 3, 8, 9, 10, 455, 456, 457, 911, 912, 2000, (3 * span) // ROW_SLOTS + 1})
    for rows in counts:
        if shift == 12 and rows > 2000:
            continue
        for first_slot in firsts:
            for first_die in (0, 1):
                found = _rows_of_items(rows, first_slot, first_die, shift, item_rows)
                assert sorted(found) == list(range(rows)), (rows, first_slot, first_die, shift, item_rows)


@pytest.mark.cpu_only
def test_the_items_of_the_production_chunk() -> None:
    """At ``chunk_shift`` 12 a chunk holds 455 or 456 row starts, ``P = ceil(456 / 4) = 114`` items of 4 rows:
    98,304 rows from phase (0, 0) span 216 chunks, 108 a die, 12,312 items each."""
    assert cake_jit.domain_chunk_items(12, 4) == 114
    assert cake_jit.domain_chunk_items(12, 2) == 228
    assert cake_jit.domain_items(98304, 0, 0, 12, 4) == (12312, 12312)
    assert cake_jit.domain_items(1, 4095, 1, 12, 4) == (0, 1)
    assert cake_jit.domain_items(1, 0, 0, 12, 4) == (1, 0)


@pytest.mark.cpu_only
def test_the_grid_is_the_items_at_most_the_resident_ctas() -> None:
    """``domain_grid`` is ``min(items0 + items1, ctas_per_sm * SMs)`` CTAs, at least one; it refuses no rows, more
    than ``DOMAIN_MAX_ROWS`` rows and a phase outside a chunk."""
    for stage, facts in TWINS.items():
        name = _twin(stage)
        for rows, first_slot, first_die, sm_count in ((1, 0, 0, 212), (98304, 0, 0, 212), (5000, 4000, 1, 104)):
            items = sum(cake_jit.domain_items(rows, first_slot, first_die, 12, facts["item_rows"]))
            want = max(1, min(items, facts["ctas_per_sm"] * sm_count))
            assert cake_jit.domain_grid(name, ARCH, rows, first_slot, first_die, sm_count) == (want, 1, 1)
        assert cake_jit.domain_grid(name, ARCH, 2**27, 0, 0, 212)[0] == facts["ctas_per_sm"] * 212
        for rows, first_slot, first_die in ((0, 0, 0), (2**27 + 1, 0, 0), (1, 4096, 0), (1, -1, 0), (1, 0, 2)):
            with pytest.raises(ValueError):
                cake_jit.domain_grid(name, ARCH, rows, first_slot, first_die, 212)


# The decode at the most rows: production's chunk shift, a 15-bit chunk and the largest (27); 1 to 16 rows an item.
MOST_ROWS_SHIFTS = (12, 15, 27)
MOST_ROWS_ITEM_ROWS = (1, 2, 4, 8, 16)
I32 = 2**31 - 1


def _ceil(a: int, b: int) -> int:
    return -(a // -b)


def _reference_row_start(chunk: int, first_slot: int, shift: int, rows: int) -> int:
    """The first of ``rows`` rows whose first slot ``first_slot + 9 r`` lies in chunk ``chunk`` or later (``rows`` when
    none does), by bisection on a row's chunk: a reference that shares no arithmetic with the package's decode."""
    lo, hi = 0, rows
    while lo < hi:
        mid = (lo + hi) // 2
        if (first_slot + ROW_SLOTS * mid) >> shift >= chunk:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _reference_items(rows: int, first_slot: int, first_die: int, shift: int, item_rows: int) -> tuple[int, int]:
    """Each die's item count: die ``d`` holds chunks ``d ^ first_die``, then every second one up to the last row's
    chunk; ``P = ceil(ceil(2^shift / 9) / item_rows)`` items each, but its last chunk only the items of the rows that
    start there."""
    per_chunk = _ceil(_ceil(1 << shift, ROW_SLOTS), item_rows)
    last = (first_slot + ROW_SLOTS * (rows - 1)) >> shift
    counts = []
    for die in (0, 1):
        chunks = range(die ^ first_die, last + 1, 2)
        if not chunks:
            counts.append(0)
            continue
        first = _reference_row_start(chunks[-1], first_slot, shift, rows)
        tail = _reference_row_start(chunks[-1] + 1, first_slot, shift, rows) - first
        counts.append((len(chunks) - 1) * per_chunk + _ceil(tail, item_rows))
    return counts[0], counts[1]


@pytest.mark.cpu_only
@pytest.mark.parametrize("item_rows", MOST_ROWS_ITEM_ROWS)
@pytest.mark.parametrize("shift", MOST_ROWS_SHIFTS)
def test_the_items_of_the_most_rows_stay_in_32_bits(shift: int, item_rows: int) -> None:
    """At ``DOMAIN_MAX_ROWS`` rows from the last slot of a chunk (``first_slot = 2^shift - 1``), both first dies:
    ``domain_chunk_items`` and ``domain_items`` give a pure-Python reference's counts (:func:`_reference_items`), and a
    chunk starts at most ``P * item_rows`` rows; every slot, ``row_start`` numerator, claim word ``item * 2 + die`` and
    walk limit (a die's items plus a grid of at most every item, less one) is an i32, as the kernel computes them;
    edge and drawn rows decode (``domain_item_rows``) into the one item of their chunk and offset and into neither
    neighbouring item; each die's last item ends at the end of its last chunk's rows, clipped at ``R``."""
    R, first_slot = cake_jit.DOMAIN_MAX_ROWS, (1 << shift) - 1
    assert R == 2**27
    per_chunk = _ceil(_ceil(1 << shift, ROW_SLOTS), item_rows)
    assert cake_jit.domain_chunk_items(shift, item_rows) == per_chunk

    def start(chunk: int) -> int:
        return _reference_row_start(chunk, first_slot, shift, R)

    last = (first_slot + ROW_SLOTS * (R - 1)) >> shift
    # the rows chunks start repeat every 9 chunks from chunk 1
    assert max(start(k + 1) - start(k) for k in range(19)) <= per_chunk * item_rows
    assert first_slot + ROW_SLOTS * (R - 1) <= I32
    assert (max(last + 1, 2) << shift) + ROW_SLOTS <= I32
    assert (last + 1) << shift <= first_slot + ROW_SLOTS * (R - 1) + (1 << shift) <= 2 * 2**27 + 9 * 2**27 < I32
    rng = random.Random(shift * 1024 + item_rows)
    for first_die in (0, 1):
        case = (R, first_slot, first_die, shift, item_rows)
        counts = _reference_items(*case)
        assert cake_jit.domain_items(*case) == counts, case
        assert min(counts) > 0 and 2 * max(counts) + 1 <= I32 and max(counts) + sum(counts) <= I32, (case, counts)
        edges = {0, 1, 2, R - 2, R - 1, start(1) - 1, start(1), start(2), start(last - 1), start(last) - 1, start(last)}
        for row in sorted(edges | {rng.randrange(R) for _ in range(400)}):
            chunk = (first_slot + ROW_SLOTS * row) >> shift
            die, offset = first_die ^ (chunk & 1), (row - start(chunk)) // item_rows
            item = (chunk >> 1) * per_chunk + offset
            assert offset < per_chunk and item < counts[die], (case, row)
            for other in (item - 1, item, item + 1):
                if 0 <= other < counts[die]:
                    lo, hi = cake_jit.domain_item_rows(other * 2 + die, *case)
                    assert (lo <= row < hi) == (other == item), (case, row, other, lo, hi)
        for die, count in enumerate(counts):
            final = range(die ^ first_die, last + 1, 2)[-1]
            first_row, end = start(final), min(start(final + 1), R)
            lo, hi = cake_jit.domain_item_rows((count - 1) * 2 + die, *case)
            assert (lo, hi) == (first_row + (_ceil(end - first_row, item_rows) - 1) * item_rows, end), (case, die)
    name = _twin("fwd")
    twin_shift, twin_rows, _ = cake_jit.domain_rule(name, ARCH)
    assert (twin_shift, twin_rows) == (12, TWINS["fwd"]["item_rows"])
    if (shift, item_rows) == (twin_shift, twin_rows):
        # the forward twin's own geometry at the most rows: its grid keeps the walk limit an i32
        for first_die in (0, 1):
            grid = cake_jit.domain_grid(name, ARCH, R, (1 << twin_shift) - 1, first_die, 212)[0]
            items = cake_jit.domain_items(R, (1 << twin_shift) - 1, first_die, twin_shift, twin_rows)
            assert max(items) + grid - 1 <= I32 and grid == TWINS["fwd"]["ctas_per_sm"] * 212


# --------------------------------------------------------------------------- CPU-only: placement and the route


class FakeTensor:
    """The metadata the route reads of a 2-D BF16 (or 1-D FP32) CUDA tensor, with no memory."""

    def __init__(self, address: int, shape: tuple[int, ...], stride=None, size: int = 2, device: int = 0) -> None:
        self.address, self.shape, self.size, self.device = address, tuple(shape), size, device
        self.strides = tuple(stride) if stride is not None else (tuple(shape[1:]) + (1,) if len(shape) == 2 else (1,))

    def data_ptr(self) -> int:
        return self.address

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def element_size(self) -> int:
        return self.size

    def get_device(self) -> int:
        return self.device


BASE = 1 << 40


@pytest.mark.cpu_only
def test_chunk_place_and_the_row_frame() -> None:
    """A tensor wholly in a range, on a slot of its chunk, with a row pitch of 9 slots, is placed at its first slot
    and its chunk's die (``start_die ^ (k & 1)``); one past the range is ``MEMORY``, one off a slot or with another
    pitch ``LAYOUT`` (a single row's pitch is not read).  The row frame: a tensor whose row 0 starts 0 to 8 slots after
    the anchor's (a chunk later only in the other die) is in it."""
    for width, chunk in CHUNK.items():
        slot = chunk // SLOTS
        entry = (BASE, 8 * chunk, chunk, 0)
        rows = 3 * 455
        assert memory.chunk_place(BASE, rows, width, width, entry) == (0, 0)
        assert memory.chunk_place(BASE + 456 * 9 * slot, 10, width, width, entry) == (456 * 9 - SLOTS, 1)
        assert memory.chunk_place(BASE + 3 * chunk + 7 * slot, 1, width, 12345, entry) == (7, 1)
        assert memory.chunk_place(BASE + 2 * chunk, 1, width, width, (BASE, 8 * chunk, chunk, 1)) == (0, 1)
        assert memory.chunk_place(BASE - slot, 1, width, width, entry) == memory.MEMORY
        assert memory.chunk_place(BASE + 8 * chunk - slot, 2, width, width, entry) == memory.MEMORY
        assert memory.chunk_place(BASE + 16, 4, width, width, entry) == memory.LAYOUT
        assert memory.chunk_place(BASE, 4, width, width + 8, entry) == memory.LAYOUT
        if width == 2 * I:
            # the gate half of a merged tensor keeps the merged pitch; its up half starts inside a slot
            assert memory.chunk_place(BASE, rows, I, 2 * I, entry) == (0, 0)
            assert memory.chunk_place(BASE + 2 * I, rows, I, 2 * I, entry) == memory.LAYOUT
    frame = memory.in_frame
    assert frame((0, 0), (0, 0)) and frame((100, 1), (108, 1)) and not frame((100, 1), (109, 1))
    assert not frame((100, 0), (99, 0)) and not frame((100, 0), (100, 1))
    assert frame((4094, 0), (1, 1)) and not frame((4094, 0), (1, 0))
    assert frame((4094, 1), (1, 0)) and not frame((4094, 1), (1, 1))


def _code(place) -> int:
    """A ``place`` as ``slot_of`` gives it: ``slot + (die << 12)``, -1 for ``MEMORY``, -2 for ``LAYOUT``."""
    if place == memory.MEMORY:
        return -1
    if place == memory.LAYOUT:
        return -2
    return place[0] + (place[1] << 12)


@pytest.mark.cpu_only
def test_slot_of_is_the_place_of_a_contiguous_tensor(monkeypatch) -> None:
    """``slot_of`` of a contiguous tensor is its ``place`` as one integer, in ranges of both widths starting in either
    die and in a range whose chunk is not a whole number of slots (where both place nothing), before, inside and after
    each range, on and off a slot and with either row pitch; the registry it reads follows each registration and
    removal.  Two placed tensors lie in one row frame (``in_frame``) exactly when
    ``(other - anchor) % FRAME_PERIOD < 9``."""
    monkeypatch.setattr(memory, "_RANGES", {})
    assert memory.ranges(0) == ((), (), ()) and memory.registered(0) == ()
    assert memory.slot_of((), (), BASE, 1, I) == -1
    odd = CHUNK[I] + 2048  # 512-byte slots, 4100 of them: the range places nothing (LAYOUT)
    kinds = [(I, CHUNK[I], 0), (2 * I, CHUNK[2 * I], 0), (I, CHUNK[I], 1), (2 * I, CHUNK[2 * I], 1), (I, odd, 1)]
    for k, (_, chunk, start_die) in enumerate(kinds):
        memory._register(0, BASE * (k + 1), 8 * chunk, chunk, start_die)
    bases, entries, frames = memory.ranges(0)
    assert entries == memory.registered(0) and bases == tuple(entry[0] for entry in entries) == tuple(
        BASE * (k + 1) for k in range(len(kinds))
    )
    assert [frame[4] is None for frame in frames] == [False, False, False, False, True]
    regular: set = set()
    for k, (width, chunk, start_die) in enumerate(kinds):
        base, slot = BASE * (k + 1), chunk >> 12
        starts = (0, 1, 8, 200, 455, 456, 457, 911, 1500)
        cases = [(base + row * 9 * slot, rows, width) for row in starts for rows in (1, 2, 455)]
        cases += [(base + n * slot, 1, cols) for n in (0, 1, 8, 9, 4095, 4096, 4097, 8191, 8192) for cols in (I, 2 * I)]
        cases += [
            (base + 16, 4, width),  # off a slot
            (base + 9 * slot, 4, 3 * I - width),  # the other width's row pitch
            (base + 8 * chunk - 9 * slot, 2, width),  # its last row past the range
            (base - slot, 1, width),  # before the range
            (base + 8 * chunk, 1, width),  # just after it
        ]
        for address, rows, cols in cases:
            got = memory.slot_of(bases, frames, address, rows, cols)
            assert got == _code(memory.place(FakeTensor(address, (rows, cols)))), (k, address - base, rows, cols)
            if got >= 0 and k < 4:
                regular.add(got)
    assert len(regular) == 20  # the distinct places of the cases above in the four whole-slot ranges
    for anchor in regular:
        for other in regular:
            want = memory.in_frame((anchor & 4095, anchor >> 12), (other & 4095, other >> 12))
            assert ((other - anchor) % memory.FRAME_PERIOD < ROW_SLOTS) == want, (anchor, other)
    memory._unregister(0, BASE * 2)
    bases, entries, frames = memory.ranges(0)
    assert entries == memory.registered(0) and BASE * 2 not in bases and len(frames) == len(kinds) - 1
    for address in (BASE * 2, BASE * 2 + 9 * (CHUNK[2 * I] >> 12)):
        assert (
            memory.slot_of(bases, frames, address, 2, 2 * I)
            == -1
            == _code(memory.place(FakeTensor(address, (2, 2 * I))))
        )
    assert memory.slot_of(bases, frames, BASE * 3, 2, I) == 1 << 12


@pytest.fixture
def fake_r200(monkeypatch):
    """A CPU stand-in of an SM107 device 0 of 212 SMs with an SM table and a registry of fake ranges: ``add(width,
    chunks)`` registers a range of that width's chunk and returns its base.  The backend's twin cache is cleared before
    and after."""
    properties = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(pkg, "device_arch", lambda index: ARCH)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(memory, "_RANGES", {})
    monkeypatch.setattr(memory, "_TABLES", {0: torch.zeros(256, dtype=torch.int32)})
    _clear_twin_caches()
    bases = []

    def add(width: int, chunks: int) -> int:
        base = BASE * (len(bases) + 1)
        bases.append(base)
        memory._register(0, base, chunks * CHUNK[width], CHUNK[width], 0)
        return base

    yield SimpleNamespace(add=add, sm_count=212)
    _clear_twin_caches()


def _route(stage, rows, anchor, others=(), outputs=(), inputs=None, callers=()):
    anchor = ("gate_up" if anchor.shape[1] == 2 * I else "gate", anchor)
    if inputs is None:
        inputs = (anchor[1], *(t for _, t in others))
    return pkg._domain_route(stage, 0, rows, anchor, tuple(others), tuple(outputs), tuple(inputs), tuple(callers))


@pytest.mark.cpu_only
def test_the_route_reasons_in_order(fake_r200, monkeypatch) -> None:
    """Each reason of the route in its order, each the first that holds: ``arch``, ``size`` (below the lower first row
    count, or past ``DOMAIN_MAX_ROWS``), ``sm_table``, ``input_memory``, ``input_layout``, ``overlap``, then for a
    class without a first row count the first operand outside the anchor's row frame (``input_phase``,
    ``out_memory``, ``no_out``), then ``size`` for the class, else ``placed``."""
    sms = fake_r200.sm_count
    R = 128 * sms
    merged = fake_r200.add(2 * I, 600)
    rows2304 = fake_r200.add(I, 600)
    other = fake_r200.add(I, 600)
    anchor = FakeTensor(merged, (R, 2 * I))
    h = FakeTensor(rows2304, (R, I))
    # placed, both classes of the forward
    twin, reason, klass, place, homed = _route("fwd", R, anchor, outputs=(("h", h),), callers=(h,))
    assert (reason, klass, place, homed) == ("placed", "homed", (0, 0), {"gate_up": True, "h": True})
    assert twin.name == _twin("fwd")
    twin, reason, klass, _, homed = _route("fwd", R, anchor, outputs=(("h", None),))
    assert (twin.name, reason, klass, homed) == (_twin("fwd"), "placed", "anchor", {"gate_up": True, "h": False})
    # the class's own first row count: 128 rows per SM for anchor, 112 for homed
    assert _route("fwd", R - 1, anchor, outputs=(("h", None),))[1:3] == ("size", "anchor")
    assert _route("fwd", 112 * sms, anchor, outputs=(("h", h),), callers=(h,))[1:3] == ("placed", "homed")
    # below the lower of the two, and past the most rows
    assert _route("fwd", 112 * sms - 1, anchor, outputs=(("h", h),), callers=(h,))[1:3] == ("size", None)
    assert _route("fwd", 2**27 + 1, anchor, outputs=(("h", h),), callers=(h,))[1] == "size"
    # arch: another architecture, and the backward stages, which have no twin
    for stage in NO_TWIN:
        assert _route(stage, R, anchor)[1] == "arch"
    monkeypatch.setattr(pkg, "device_arch", lambda index: "sm_100a")
    _clear_twin_caches()
    assert _route("fwd", R, anchor)[1] == "arch"
    monkeypatch.setattr(pkg, "device_arch", lambda index: ARCH)
    _clear_twin_caches()
    # sm_table
    monkeypatch.setattr(memory, "_TABLES", {})
    assert _route("fwd", R, anchor)[1] == "sm_table"
    monkeypatch.setattr(memory, "_TABLES", {0: None})
    assert _route("fwd", R, anchor)[1] == "sm_table"
    monkeypatch.setattr(memory, "_TABLES", {0: torch.zeros(256, dtype=torch.int32)})
    # input_memory, input_layout
    assert _route("fwd", R, FakeTensor(7 * BASE, (R, 2 * I)))[1] == "input_memory"
    assert _route("fwd", R, FakeTensor(merged + 16, (R, 2 * I)))[1] == "input_layout"
    assert _route("fwd", R, FakeTensor(merged, (R, 2 * I), stride=(2 * I + 8, 1)))[1] == "input_layout"
    # overlap: a caller's output that meets an input (another tensor) comes before the class's reasons
    gate = FakeTensor(rows2304, (R, I))
    up = FakeTensor(other, (R, I))
    out = FakeTensor(other + 9 * 512 * 7, (R - 7, I))
    found = _route("fwd", R - 7, gate, others=(("up", up),), outputs=(("h", out),), callers=(out,))
    assert found[1] == "overlap"
    # a class without a first row count (here the forward's anchor class): its first operand outside the anchor's row
    # frame names the reason
    monkeypatch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM["fwd"], ARCH, {"homed": 112, "anchor": None})
    _clear_twin_caches()
    R2 = 112 * sms
    gate2 = FakeTensor(rows2304, (R2, I))
    up2 = FakeTensor(other, (R2, I))
    h2 = FakeTensor(fake_r200.add(I, 600), (R2, I))
    homed2 = (("h", h2),)
    assert _route("fwd", R2, gate2, others=(("up", up2),), outputs=homed2, callers=(h2,))[1:3] == ("placed", "homed")
    torch_up = FakeTensor(9 * BASE, (R2, I))
    found = _route("fwd", R2, gate2, others=(("up", torch_up),), outputs=homed2, callers=(h2,))
    assert found[1:3] == ("input_phase", "anchor") and found[4] == {"gate": True, "up": False, "h": True}
    # an up of another phase (a row later) is outside the frame too
    late = FakeTensor(other + 9 * 512, (R2, I))
    assert _route("fwd", R2, gate2, others=(("up", late),), outputs=homed2, callers=(h2,))[1] == "input_phase"
    torch_out = FakeTensor(9 * BASE, (R2, I))
    found = _route("fwd", R2, gate2, others=(("up", up2),), outputs=(("h", torch_out),), callers=(torch_out,))
    assert found[1:3] == ("out_memory", "anchor")
    assert _route("fwd", R2, gate2, others=(("up", up2),), outputs=(("h", None),))[1:3] == ("no_out", "anchor")
    assert _route("fwd", R2 - 1, gate2, others=(("up", up2),), outputs=homed2, callers=(h2,))[1:3] == ("size", None)


@pytest.mark.cpu_only
def test_the_twin_grid_is_the_domain_grid(fake_r200) -> None:
    """``_twin_grid`` (the resident CTAs from ``saturated = resident * item_rows`` rows on, with no item count) is
    ``cake_jit.domain_grid``'s grid at every row count around ``saturated`` and from every kind of anchor place."""
    sms = fake_r200.sm_count
    for stage, facts in TWINS.items():
        twin = pkg._twin(stage, 0)
        assert twin.resident == facts["ctas_per_sm"] * sms
        assert twin.saturated == twin.resident * facts["item_rows"]
        top, step = twin.saturated, facts["item_rows"]
        counts = {1, 2, 3, twin.resident - 1, twin.resident, twin.resident + 1, top - step - 1, top - step}
        counts |= {top - 2, top - 1, top, top + 1, 3 * top, cake_jit.DOMAIN_MAX_ROWS}
        for rows in sorted(counts):
            for place in ((0, 0), (0, 1), (1, 0), (8, 1), (2048, 0), (4087, 1), (4095, 0), (4095, 1)):
                want = cake_jit.domain_grid(twin.name, ARCH, rows, place[0], place[1], sms)[0]
                assert pkg._twin_grid(twin, rows, place) == want, (stage, rows, place)


@pytest.mark.cpu_only
def test_a_range_whose_chunk_is_not_whole_slots_places_nothing(fake_r200) -> None:
    """A registered range whose chunk is not ``CHUNK_SLOTS`` whole slots places no tensor: ``chunk_place``, ``place``
    and ``slot_of`` give ``LAYOUT`` (-2) for every tensor wholly in it, also at a slot count of ``CHUNK_SLOTS`` or more
    within a chunk (where one integer would name the next chunk's die), and ``MEMORY`` (-1) past it.  So an anchor
    there routes ``input_layout`` and another operand there lies outside every anchor's row frame."""
    odd = CHUNK[I] + 2048  # 512-byte slots, 4100 of them
    base = BASE * 50
    entry = (base, 600 * odd, odd, 0)
    memory._register(0, *entry)
    bases, _, frames = memory.ranges(0)
    (frame,) = [f for f in frames if f[0] == base]
    assert frame[4] is None
    slot = odd >> 12
    for n in (0, 1, 9, 4095, 4096, 4097, 4099, 4100, 4101, 8199, 8200):
        for rows in (1, 2, 455):
            address = base + n * slot
            assert memory.chunk_place(address, rows, I, I, entry) == memory.LAYOUT, (n, rows)
            assert memory.place(FakeTensor(address, (rows, I))) == memory.LAYOUT, (n, rows)
            assert memory.slot_of(bases, frames, address, rows, I) == -2, (n, rows)
    end = base + 600 * odd
    assert memory.place(FakeTensor(end - 9 * slot, (2, I))) == memory.MEMORY
    assert memory.slot_of(bases, frames, end - 9 * slot, 2, I) == -1
    sms = fake_r200.sm_count
    R = 128 * sms
    assert _route("fwd", R, FakeTensor(base, (R, I)))[1] == "input_layout"
    anchor = FakeTensor(fake_r200.add(2 * I, 600), (R, 2 * I))
    found = _route("fwd", R, anchor, outputs=(("h", FakeTensor(base, (R, I))),), callers=())
    assert found[1:3] == ("placed", "anchor") and found[4] == {"gate_up": True, "h": False}


class CudaFake(FakeTensor):
    """A :class:`FakeTensor` the steps' checks take for a plain contiguous CUDA tensor of ``dtype`` while the backend's
    ``torch.Tensor`` is this class (:func:`cpu_forward`)."""

    is_cuda = True

    def __init__(self, address: int, shape: tuple[int, ...], dtype=torch.bfloat16) -> None:
        super().__init__(address, shape, size=2 if dtype is torch.bfloat16 else 4)
        self.dtype = dtype

    def is_contiguous(self) -> bool:
        return True


# Ordinary memory of :func:`cpu_forward`'s tensors: no registered range lies there.
ORDINARY = BASE * 100


class _Torch:
    """The backend's ``torch`` in :func:`cpu_forward`: ``Tensor`` is :class:`CudaFake` and ``empty`` makes one in
    ordinary memory; every other name is torch's."""

    Tensor = CudaFake

    def __init__(self) -> None:
        self.made = 0

    def empty(self, *shape, dtype=torch.bfloat16, device=None) -> CudaFake:
        self.made += 1
        return CudaFake(ORDINARY + self.made * (1 << 36), shape, dtype)

    def __getattr__(self, name: str):
        return getattr(torch, name)


class _FakeLauncher:
    """The forward's launcher in :func:`cpu_forward`: one program, ``"off"``, whose entry is ``off`` (a row a task and
    a task a CTA: the grid is the row count); a call (the launch of a call checked in full) records ``(rows,
    *arguments)`` in ``calls``."""

    def __init__(self, off) -> None:
        self.device = None
        self.weight = CudaFake(ORDINARY, (1,), torch.float32)
        self.route = (0, (), ("off", None, 1, 1))
        self.entries = {"off": off}
        self.calls: list = []

    def ignored_weight(self):
        return self.weight

    def __call__(self, rows, *arguments) -> None:
        self.calls.append((rows, *arguments))


@pytest.fixture
def cpu_forward(fake_r200, monkeypatch):
    """The forward step's ``"domain"`` path without a GPU, on :func:`fake_r200`'s device: the backend's ``torch``
    takes :class:`CudaFake` for a CUDA tensor, the device is an SM107, and the forward's launcher is a stand-in
    (:class:`_FakeLauncher`) whose ``"off"`` program is ``off``, which records the arguments of each launch in
    ``launches``.  ``run(first, up, weight, out)`` calls the step with a held launch and returns ``(launch,
    arguments)``, what it would launch; the plain route of a plan names ``"off"``."""
    monkeypatch.setattr(pkg, "torch", _Torch())
    monkeypatch.setattr(pkg, "_arch_of_index", lambda index: ARCH)
    launches: list = []

    def off(*args) -> None:
        launches.append(args)

    launcher = _FakeLauncher(off)
    monkeypatch.setattr(pkg, "_launcher", lambda stage, index: launcher)
    monkeypatch.setattr(pkg, "_plain_plan", lambda stage, index, rows: ("off", (1, 1, 1)))

    def run(first, up, weight, out):
        held = pkg._HeldLaunch()
        h = pkg._forward_step_domain(first, up, weight, L, out, held)
        assert out is None or h is out
        (launch,) = held
        return launch

    return SimpleNamespace(
        run=run,
        off=off,
        launches=launches,
        launcher=launcher,
        add=fake_r200.add,
        sm_count=fake_r200.sm_count,
    )


@pytest.mark.cpu_only
def test_the_forward_step_routes_as_the_route_and_the_plan(cpu_forward, monkeypatch) -> None:
    """The forward step's own routing of a common call (plain contiguous operands, checked and routed in its frame)
    makes the decision of ``_domain_route`` and of the plan for every reason and class, without a GPU (a fake
    registry and fake tensors): the twin, at the route's place, exactly when the route places the call, else the
    ``"off"`` program.  The first rows per SM are set per class (each alone, both, the shipped ones), so a step that
    took the other class would launch otherwise."""
    sms = cpu_forward.sm_count
    merged_base, gate_base, up_base, h_base = (cpu_forward.add(width, 600) for width in (2 * I, I, I, I))
    odd = CHUNK[I] + 2048
    odd_base = BASE * 60
    memory._register(0, odd_base, 600 * odd, odd, 0)
    seen: collections.Counter = collections.Counter()

    def check(what, rows, first, up, weight, out):
        label = "gate_up" if up is None else "gate"
        others = () if up is None else (("up", up),)
        route = pkg._domain_route("fwd", 0, rows, (label, first), others, (("h", out),), (first, up, weight), (out,))
        plan = pkg.cake_clamped_swiglu_det_train_forward_domain_plan(first, up, weight, limit=L, out=out)
        assert (plan["reason"], plan["class"]) == route[1:3], (what, plan, route)
        launch, arguments = cpu_forward.run(first, up, weight, out)
        if route[0] is None:
            assert plan["route"] == "plain" and launch is cpu_forward.off, (what, route[1:3])
        else:
            assert (plan["first_slot"], plan["first_die"]) == route[3], what
            assert launch is pkg._launch_held_twin, (what, route[1:3])
            assert arguments[0] is route[0] and arguments[11:13] == route[3] and arguments[6] == rows, what
        seen[route[1]] += 1
        seen[f"{route[1]} {route[2]}"] += 1

    def cases(rows, row0):
        """``(what, first, up, weight, out)`` of each layout: the operands at row ``row0`` of their ranges."""
        weight = CudaFake(ORDINARY + (1 << 40), (rows,), torch.float32)
        merged = CudaFake(merged_base + row0 * 9 * 1024, (rows, 2 * I))
        gate = CudaFake(gate_base + row0 * 9 * 512, (rows, I))
        up = CudaFake(up_base + row0 * 9 * 512, (rows, I))
        h = CudaFake(h_base + row0 * 9 * 512, (rows, I))
        ordinary = CudaFake(ORDINARY + (2 << 40), (rows, I))
        for name, first, second in (("merged", merged, None), ("split", gate, up)):
            yield f"{name} homed", first, second, weight, h
            yield f"{name} unweighted", first, second, None, h
            yield f"{name} new h", first, second, weight, None
            yield f"{name} ordinary h", first, second, weight, ordinary
            yield f"{name} h in an odd range", first, second, weight, CudaFake(odd_base, (rows, I))
            yield f"{name} ordinary first", CudaFake(ORDINARY, first.shape), second, weight, h
            yield f"{name} off a slot", CudaFake(first.address + 16, first.shape), second, weight, h
            yield f"{name} first in an odd range", CudaFake(odd_base, first.shape), second, weight, h
        yield "split up a row later", gate, CudaFake(up.address + 9 * 512, (rows, I)), weight, h
        yield "split ordinary up", gate, ordinary, weight, h
        yield "merged h over the input", merged, None, weight, CudaFake(merged.address, (rows, I))
        yield "split h over the up", gate, up, weight, CudaFake(up.address, (rows, I))
        yield "split h over the weight", gate, up, weight, CudaFake(weight.address, (rows, I))

    settings = (
        ({"homed": 112, "anchor": 128}, (112 * sms - 1, 112 * sms, 128 * sms - 1, 128 * sms)),
        ({"homed": 1, "anchor": None}, (sms - 1, 2000)),
        ({"homed": None, "anchor": 1}, (2000,)),
        ({"homed": 2, "anchor": 3}, (2 * sms - 1, 2 * sms, 3 * sms - 1, 3 * sms)),
    )
    for firsts, counts in settings:
        monkeypatch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM["fwd"], ARCH, firsts)
        _clear_twin_caches()
        for rows in counts:
            for row0 in (0, 456):
                for what, first, up, weight, out in cases(rows, row0):
                    check(f"{firsts} {rows} rows at row {row0}: {what}", rows, first, up, weight, out)
    monkeypatch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM["fwd"], ARCH, FIRST_ROWS["fwd"][ARCH])
    _clear_twin_caches()
    rows = 128 * sms
    first, up, weight, out = next(cases(rows, 0))[1:]
    most = cake_jit.DOMAIN_MAX_ROWS + 1
    big = [None if t is None else CudaFake(t.address, (most, *t.shape[1:]), t.dtype) for t in (first, up, weight, out)]
    check("past the most rows", most, *big)
    with monkeypatch.context() as patch:
        patch.setattr(memory, "_TABLES", {})
        check("no SM table", rows, first, up, weight, out)
    with monkeypatch.context() as patch:
        patch.setattr(pkg, "device_arch", lambda index: "sm_100a")
        _clear_twin_caches()
        check("another architecture", rows, first, up, weight, out)
    _clear_twin_caches()
    assert set(REASONS) <= set(seen), seen
    assert {"placed homed", "placed anchor", "size homed", "size anchor", "size None"} <= set(seen), seen


@pytest.fixture
def cpu_claims(monkeypatch) -> list:
    """``cake_domain.claim`` without a GPU: every stream is handle ``0x77``, nothing captures, and the stream's eager
    workspace is a CPU block; returns the ``(index, words)`` of each claim, in order."""
    monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(memory, "_current_stream_handle", lambda index: 0x77)
    monkeypatch.setattr(memory, "_stream_capturing", lambda index: False)
    block = torch.zeros(memory.DIES, memory.SLOT_WORDS, dtype=torch.int32)
    monkeypatch.setattr(memory, "_eager_workspace", lambda index: memory._Workspace(block))
    claims: list = []
    claim = memory.claim

    def counted(index: int, words: int):
        claims.append((index, words))
        return claim(index, words)

    monkeypatch.setattr(memory, "claim", counted)
    return claims


@pytest.mark.cpu_only
def test_a_held_twin_launch_passes_the_common_calls_arguments(cpu_forward, cpu_claims, monkeypatch) -> None:
    """The twin launch a common call keeps in a checkpoint's recomputation (``_launch_held_twin``, run after ``apply``
    returns) binds the twin's entry and claims the counters only when it runs, then calls the entry with the arguments
    of the common call's own launch: the same tensors, row count, strides, limit, counters of the stream's workspace,
    first slot and die, twin geometry, SM table, column offset and grid.  Merged and split, weighted and unweighted,
    both classes, at a row in each die."""
    twin = pkg._twin("fwd", 0)
    calls: list = []
    bound: list = []

    def entry_of(name, arch):
        bound.append(name)
        return lambda *args: calls.append(args)

    monkeypatch.setattr(pkg, "_entry", entry_of)
    rows = 128 * cpu_forward.sm_count
    merged_base, gate_base, up_base, h_base = (cpu_forward.add(width, 600) for width in (2 * I, I, I, I))
    for row0 in (0, 456):
        weight = CudaFake(ORDINARY + (1 << 40), (rows,), torch.float32)
        merged = CudaFake(merged_base + row0 * 9 * 1024, (rows, 2 * I))
        gate = CudaFake(gate_base + row0 * 9 * 512, (rows, I))
        up = CudaFake(up_base + row0 * 9 * 512, (rows, I))
        h = CudaFake(h_base + row0 * 9 * 512, (rows, I))
        for what, first, second, w, out, klass in (
            ("merged homed", merged, None, weight, h, "homed"),
            ("merged new h", merged, None, weight, None, "anchor"),
            ("merged unweighted", merged, None, None, h, "homed"),
            ("split homed", gate, up, weight, h, "homed"),
            ("split unweighted new h", gate, up, None, None, "anchor"),
        ):
            what = f"{what} at row {row0}"
            plan = pkg.cake_clamped_swiglu_det_train_forward_domain_plan(first, second, w, limit=L, out=out)
            assert (plan["route"], plan["class"]) == ("domain", klass), (what, plan)
            calls.clear()
            cpu_claims.clear()
            launch, arguments = cpu_forward.run(first, second, w, out)
            assert launch is pkg._launch_held_twin and arguments[0] is twin, what
            # held: nothing bound, claimed or launched before it runs
            assert not cpu_claims and not calls, what
            if row0 == 0 and what.startswith("merged homed"):
                assert twin.entry is None and not bound, what
            launch(*arguments)
            assert cpu_claims == [(0, twin.words)] and bound == [twin.name], what
            (held,) = calls
            calls.clear()
            # the common call's own launch, into the held launch's h (a new h: its class stays anchor)
            h_held = arguments[5]
            assert pkg._forward_step_domain(first, second, w, L, h_held) is h_held, what
            (common,) = calls
            assert len(held) == len(common) == len(pkg._DOMAIN_ARGUMENTS["fwd"]) + 3, what
            for i, (a, b) in enumerate(zip(held, common)):
                assert (type(a) is type(b) and a == b) if type(a) in (int, float) else a is b, (what, i)
            assert held[3] is h_held and (held[12], held[13]) == (plan["first_slot"], plan["first_die"]), what
            assert held[-3:] == plan["grid"] and held[16] is memory.sm_table(0), what
            counters = memory.claim(0, twin.words)
            assert held[10] is counters[0] and held[11] is counters[1], what


@pytest.mark.cpu_only
def test_a_raising_held_twin_launch_drops_its_workspace(cpu_forward, cpu_claims, monkeypatch) -> None:
    """A held twin launch whose entry raises drops the stream's workspace it claimed (``cake_domain.drop``), so the
    next launch claims a new one; the error reaches the caller."""
    twin = pkg._twin("fwd", 0)
    rows = 128 * cpu_forward.sm_count
    first = CudaFake(cpu_forward.add(2 * I, 600), (rows, 2 * I))
    weight = CudaFake(ORDINARY + (1 << 40), (rows,), torch.float32)
    seen: list = []

    def boom(*args):
        seen.append(dict(memory._STREAM_WORKSPACES))
        raise RuntimeError("launch refused")

    monkeypatch.setattr(twin, "entry", boom)
    launch, arguments = cpu_forward.run(first, None, weight, None)
    assert launch is pkg._launch_held_twin and not cpu_claims
    with pytest.raises(RuntimeError, match="refused"):
        launch(*arguments)
    (during,) = seen
    assert list(during) == [(0, 0x77)] and cpu_claims == [(0, twin.words)]
    assert memory._STREAM_WORKSPACES == {}
    monkeypatch.setattr(twin, "entry", lambda *args: seen.append(dict(memory._STREAM_WORKSPACES)))
    launch, arguments = cpu_forward.run(first, None, weight, None)
    launch(*arguments)
    assert len(seen) == 2 and seen[1][(0, 0x77)] is not during[(0, 0x77)]
    assert memory._STREAM_WORKSPACES == seen[1]


def _firsts(sms: int) -> tuple[int, int]:
    """The shipped first row counts of the forward's twin, ``(homed, anchor)``, on a device of ``sms`` SMs."""
    return FIRST_ROWS["fwd"][ARCH]["homed"] * sms, FIRST_ROWS["fwd"][ARCH]["anchor"] * sms


@pytest.fixture
def cpu_domain_reads(cpu_forward, cpu_claims, monkeypatch) -> SimpleNamespace:
    """:func:`cpu_forward` with every way into the helper's registry and SM table counted in ``reads``: the calls of
    the backend's ``_domain`` (each placement read and each twin launch starts there) and of ``cake_domain``'s
    ``sm_table``, ``ranges``, ``slot_of`` and ``place``; ``claims`` lists the claim counters taken
    (:func:`cpu_claims`).  Every twin entry records its arguments in ``twin_launches``; a plain plan names the
    ``"off"`` program with the stand-in launcher's grid (``rows`` CTAs)."""
    reads: collections.Counter = collections.Counter()

    def counted(name, function):
        def wrapper(*args, **kwargs):
            reads[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(pkg, "_domain", counted("_domain", pkg._domain))
    for name in ("sm_table", "ranges", "slot_of", "place"):
        monkeypatch.setattr(memory, name, counted(name, getattr(memory, name)))
    twin_launches: list = []
    monkeypatch.setattr(pkg, "_entry", lambda name, arch: lambda *args: twin_launches.append((name, args)))
    monkeypatch.setattr(pkg, "_plain_plan", lambda stage, index, rows: ("off", (rows, 1, 1)))
    return SimpleNamespace(reads=reads, claims=cpu_claims, twin_launches=twin_launches, **vars(cpu_forward))


def _same_launch(what: str, got: tuple, want: tuple, new_at) -> None:
    """Two launches (``(entry, arguments)`` or the arguments alone) are the same: the same entry and the same
    arguments, the same objects but at position ``new_at`` (None: none), a new ``h`` each call allocates, compared by
    its shape and dtype."""
    assert len(got) == len(want), what
    for i, (a, b) in enumerate(zip(got, want)):
        if i == new_at:
            assert type(a) is type(b) is CudaFake and (a.shape, a.dtype) == (b.shape, b.dtype) and a is not b, (what, i)
        elif type(a) in (int, float, tuple):
            assert type(a) is type(b) and a == b, (what, i)
        else:
            assert a is b, (what, i)


@pytest.mark.cpu_only
def test_below_its_first_rows_a_domain_forward_runs_the_off_step_without_the_registry(cpu_domain_reads) -> None:
    """A ``"domain"`` forward call with fewer rows than the first row count its class can reach before its placement
    is read (the ``anchor`` class's for a new ``h``, as in the autograd forward, whose ``h`` is new; the lower of the
    two with a caller's ``out``) reads neither the registry nor the SM table, takes no claim counters and launches
    what the ``"off"`` step launches: a common call (checked in the step's own frame) as :func:`_forward_step`, held
    (a checkpoint's recomputation) or not, and a call checked in full (``_forward_domain``) as ``_forward``.  Every
    operand lies in helper memory in the anchor's row frame, merged and split, weighted and unweighted, at row counts
    from 1 to one below each first row count.  The first row counts are those of the twin."""
    c = cpu_domain_reads
    sms = c.sm_count
    homed, anchor = _firsts(sms)
    merged_base, gate_base, up_base, h_base = (c.add(width, 600) for width in (2 * I, I, I, I))
    cases = 0
    for rows, outs in ((1, ("h", None)), (4096, ("h", None)), (homed - 1, ("h", None)), (anchor - 1, (None,))):
        for merged in (True, False):
            for weighted in (True, False):
                first = CudaFake(merged_base, (rows, 2 * I)) if merged else CudaFake(gate_base, (rows, I))
                up = None if merged else CudaFake(up_base, (rows, I))
                weight = CudaFake(ORDINARY + (1 << 40), (rows,), torch.float32) if weighted else None
                for kind in outs:
                    out = None if kind is None else CudaFake(h_base, (rows, I))
                    what = f"{rows} rows merged={merged} weighted={weighted} out={kind}"
                    plan = forward_plan(first, up, weight, limit=L, out=out)
                    assert plan["route"] == "plain" and plan["reason"] == "size", (what, plan)
                    c.reads.clear()
                    # held: the step's item against the "off" step's, and the full checks' against ``_forward``'s
                    for domain, plain, h_at in (
                        (pkg._forward_step_domain, pkg._forward_step, 3),
                        (pkg._forward_domain, pkg._forward, 4),
                    ):
                        got, want = pkg._HeldLaunch(), pkg._HeldLaunch()
                        h = domain(first, up, weight, L, out, got)
                        assert out is None or h is out, what
                        plain(first, up, weight, L, out, want)
                        ((got_launch, got_args),) = got
                        ((want_launch, want_args),) = want
                        assert got_launch is want_launch, (what, domain.__name__)
                        _same_launch(f"{what} {domain.__name__}", got_args, want_args, h_at if out is None else None)
                    # not held: the "off" entry and the stand-in launcher record the launches
                    c.launches.clear()
                    c.launcher.calls.clear()
                    pkg._forward_step_domain(first, up, weight, L, out)
                    pkg._forward_step(first, up, weight, L, out)
                    pkg._forward_domain(first, up, weight, L, out)
                    pkg._forward(first, up, weight, L, out)
                    assert len(c.launches) == len(c.launcher.calls) == 2, what
                    new = out is None
                    _same_launch(f"{what} step", c.launches[0], c.launches[1], 3 if new else None)
                    _same_launch(f"{what} full checks", c.launcher.calls[0], c.launcher.calls[1], 4 if new else None)
                    assert not c.reads and not c.claims and not c.twin_launches, (what, dict(c.reads))
                    cases += 1
    assert cases == 28
    assert pkg._FLOORS[0] == (homed, anchor, pkg._twin("fwd", 0))


@pytest.mark.cpu_only
def test_the_plan_and_the_call_agree_on_a_grid_of_rows_classes_and_memory(cpu_domain_reads, monkeypatch) -> None:
    """For a grid of row counts (1, 4096, one below, at and one above each first row count of the twin, twice the
    ``anchor`` one, past ``DOMAIN_MAX_ROWS``), classes (``homed``: the caller's ``out`` and the up of two tensors in the
    anchor's row frame; ``anchor``: a new ``h`` as in the autograd forward, a caller's ``out`` in ordinary memory, or
    an up in ordinary memory) and memory (the anchor in helper memory, in ordinary memory, or no SM table), merged and
    split, weighted and unweighted: the plan keeps its reasons (``size`` without a class below the lower first row
    count and past the most rows, then ``sm_table``, ``input_memory``, ``size`` with the class below the class's own
    first row count, else ``placed``), and the common step and the full checks' path, each held and not, launch the
    twin exactly when the plan routes the call to it, at its place and grid, else the ``"off"`` program on its grid.
    A call reads the registry exactly when its row count reaches the first row count its class can reach before its
    placement is read (``anchor``'s for a new ``h``, else the lower of the two) and is at most ``DOMAIN_MAX_ROWS``;
    only a twin launch takes claim counters (when it runs)."""
    c = cpu_domain_reads
    sms = c.sm_count
    homed, anchor = _firsts(sms)
    top = cake_jit.DOMAIN_MAX_ROWS
    merged_base, gate_base, up_base, h_base = (c.add(width, 600) for width in (2 * I, I, I, I))
    twin = pkg._twin("fwd", 0)
    seen: collections.Counter = collections.Counter()

    def expect(rows: int, klass: str, where: str) -> tuple:
        if rows < homed or rows > top:
            return "plain", "size", None
        if where == "no SM table":
            return "plain", "sm_table", None
        if where == "ordinary first":
            return "plain", "input_memory", None
        if rows < (homed if klass == "homed" else anchor):
            return "plain", "size", klass
        return "domain", "placed", klass

    def cases(rows: int, merged: bool, weighted: bool, where: str):
        """``(what, class, first, up, weight, out)`` of each class."""
        weight = CudaFake(ORDINARY + (1 << 40), (rows,), torch.float32) if weighted else None
        if where == "ordinary first":
            first = CudaFake(ORDINARY + (3 << 40), (rows, 2 * I if merged else I))
        else:
            first = CudaFake(merged_base, (rows, 2 * I)) if merged else CudaFake(gate_base, (rows, I))
        up = None if merged else CudaFake(up_base, (rows, I))
        h = CudaFake(h_base, (rows, I))
        yield "homed", "homed", first, up, weight, h
        yield "new h", "anchor", first, up, weight, None
        yield "ordinary out", "anchor", first, up, weight, CudaFake(ORDINARY + (2 << 40), (rows, I))
        if not merged:
            yield "ordinary up", "anchor", first, CudaFake(ORDINARY + (4 << 40), (rows, I)), weight, h

    def check(what: str, klass: str, where: str, rows: int, first, up, weight, out) -> None:
        plan = forward_plan(first, up, weight, limit=L, out=out)
        route, reason, plan_class = expect(rows, klass, where)
        assert (plan["route"], plan["reason"], plan["class"]) == (route, reason, plan_class), (what, plan)
        if route == "domain":
            assert plan["program"] == twin.name and plan["grid"][0] == pkg._twin_grid(twin, rows, (0, 0)), what
        else:
            assert (plan["program"], plan["grid"]) == ("off", (rows, 1, 1)), (what, plan)
        reads = (anchor if out is None else homed) <= rows <= top
        for path, step in (("step", pkg._forward_step_domain), ("full checks", pkg._forward_domain)):
            for held in (pkg._HeldLaunch(), None):
                for log in (c.reads, c.claims, c.twin_launches, c.launches, c.launcher.calls):
                    log.clear()
                where_ = f"{what}: {path}{' held' if held is not None else ''}"
                h = step(first, up, weight, L, out, held)
                assert out is None or h is out, where_
                assert bool(c.reads) == reads, (where_, dict(c.reads))
                if held is not None:
                    ((launch, arguments),) = held
                    assert not (c.claims or c.twin_launches or c.launches or c.launcher.calls), where_
                    if route == "domain" and path == "step":
                        assert launch is pkg._launch_held_twin and arguments[0] is twin, where_
                        assert arguments[11:13] == (0, 0) and (arguments[15], 1, 1) == plan["grid"], where_
                    elif route == "domain":
                        assert launch is pkg._launch_twin and arguments[0] is twin and arguments[3] == (0, 0), where_
                    elif path == "step":
                        assert launch is c.off and arguments[-3:] == plan["grid"], where_
                    else:
                        assert launch is c.launcher and arguments[0] == rows, where_
                elif route == "domain":
                    ((name, arguments),) = c.twin_launches
                    assert name == twin.name and arguments[-3:] == plan["grid"], where_
                    assert c.claims == [(0, twin.words)] and not (c.launches or c.launcher.calls), where_
                else:
                    assert not (c.claims or c.twin_launches), where_
                    if path == "step":
                        assert [args[-3:] for args in c.launches] == [plan["grid"]] and not c.launcher.calls, where_
                    else:
                        assert [args[0] for args in c.launcher.calls] == [rows] and not c.launches, where_
                seen[f"{path} {route} {reason} {plan_class} reads={reads}"] += 1

    counts = (1, 4096, homed - 1, homed, homed + 1, anchor - 1, anchor, anchor + 1, 2 * anchor, top + 1)
    for rows in counts:
        for merged in (True, False):
            for weighted in (True, False):
                for where in ("helper", "ordinary first", "no SM table"):
                    with monkeypatch.context() as patch:
                        if where == "no SM table":
                            patch.setattr(memory, "_TABLES", {})
                        for what, klass, first, up, weight, out in cases(rows, merged, weighted, where):
                            label = f"{rows} rows merged={merged} weighted={weighted} {where}: {what}"
                            check(label, klass, where, rows, first, up, weight, out)
    for path in ("step", "full checks"):
        # every reason, both classes, and each side of the first row count of a new h's class
        for want in (
            "plain size None reads=False",
            "plain size anchor reads=True",
            "plain sm_table None reads=True",
            "plain input_memory None reads=True",
            "domain placed homed reads=True",
            "domain placed anchor reads=True",
        ):
            assert seen[f"{path} {want}"], (path, want, seen)
        # from the lower first row count to one below the anchor's, a new h reads nothing: the plan places its anchor
        # (class anchor, too few rows), finds no SM table or the anchor in ordinary memory
        for want in ("plain size anchor", "plain sm_table None", "plain input_memory None"):
            assert seen[f"{path} {want} reads=False"], (path, want, seen)
        assert not seen[f"{path} plain size None reads=True"], seen


@pytest.mark.cpu_only
def test_the_floors_are_the_twins_first_row_counts(fake_r200, monkeypatch) -> None:
    """``_floors`` keeps per device ``(the lower of the two first row counts, the anchor class's, the twin)``, each
    ``rows_per_sm * SMs`` of the device as the twin computes it, ``MAX_ROWS`` for a class without one, and
    ``(MAX_ROWS, MAX_ROWS, None)`` on an architecture without a twin."""
    most = cake_jit.MAX_ROWS
    for sms in (212, 148, 7):
        properties = SimpleNamespace(multi_processor_count=sms)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index, p=properties: p)
        for firsts, want in (
            (FIRST_ROWS["fwd"][ARCH], (112 * sms, 128 * sms)),
            ({"homed": 1, "anchor": None}, (sms, most)),
            ({"homed": None, "anchor": 3}, (3 * sms, 3 * sms)),
            ({"homed": None, "anchor": None}, (most, most)),
        ):
            monkeypatch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM["fwd"], ARCH, firsts)
            _clear_twin_caches()
            twin = pkg._twin("fwd", 0)
            assert pkg._floors(0) == (*want, twin) and pkg._FLOORS[0] == (*want, twin), (sms, firsts)
            assert want[0] == (most if twin.lowest is None else twin.lowest), (sms, firsts)
    monkeypatch.setattr(pkg, "device_arch", lambda index: "sm_100a")
    _clear_twin_caches()
    assert pkg._floors(0) == (most, most, None)
    _clear_twin_caches()


@pytest.mark.cpu_only
def test_the_plan_of_each_route() -> None:
    """A plan holds the keys of the contract: the twin's launch on the domain route, the ``"off"`` program and its
    grid on the plain route (the device's route), nothing for no rows."""
    empty = pkg._route_plan("fwd", 0, 0, None, {"h": "empty"})
    assert set(empty) == PLAN_KEYS
    assert empty["route"] == "plain" and empty["reason"] == "empty" and empty["program"] is None
    assert empty["grid"] is None and empty["outputs"] == {"h": "empty"}


@pytest.mark.cpu_only
@pytest.mark.parametrize("bad", [None, 1, b"off", True, ("off",)])
def test_locality_must_be_a_string(bad) -> None:
    """A ``locality`` that is not a ``str`` is a ``TypeError``, checked before any other argument."""
    x = torch.zeros(2, 2 * I, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="locality"):
        forward(x, limit=L, locality=bad)
    with pytest.raises(TypeError, match="locality"):
        backward(x[:, :I], x, limit=L, locality=bad)
    with pytest.raises(TypeError, match="locality"):
        clamped_swiglu(x, limit=L, locality=bad)


@pytest.mark.cpu_only
@pytest.mark.parametrize("bad", ["", "Domain", "on", "domain ", "default"])
def test_locality_is_off_or_domain(bad) -> None:
    x = torch.zeros(2, 2 * I, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="locality"):
        forward(x, limit=L, locality=bad)
    with pytest.raises(ValueError, match="locality"):
        backward(x[:, :I], x, limit=L, locality=bad)
    with pytest.raises(ValueError, match="locality"):
        clamped_swiglu(x, limit=L, locality=bad)
    # "domain" passes the check and then refuses the CPU tensor as "off" does
    for locality in ("off", "domain"):
        with pytest.raises(ValueError, match="CUDA"):
            forward(x, limit=L, locality=locality)
        with pytest.raises(ValueError, match="CUDA"):
            backward(x[:, :I], x, limit=L, locality=locality)


@pytest.mark.cpu_only
def test_signatures_end_with_locality_and_the_plans_mirror_the_steps() -> None:
    """The forward step, the backward step and the autograd entry end with the keyword ``locality="off"``; each plan
    takes its step's parameters without it; the helpers take ``(shape, *, dtype=torch.bfloat16, device=None)`` and
    ``(device=None)``."""
    for name in (
        "cake_clamped_swiglu_det",
        "cake_clamped_swiglu_det_train_forward",
        "cake_clamped_swiglu_det_train_backward",
    ):
        last = list(inspect.signature(getattr(pkg, name)).parameters.values())[-1]
        assert last.name == "locality" and last.default == "off" and last.kind is last.KEYWORD_ONLY, name
    for step in ("forward", "backward"):
        want = list(inspect.signature(getattr(pkg, f"cake_clamped_swiglu_det_train_{step}")).parameters.values())
        got = list(
            inspect.signature(getattr(pkg, f"cake_clamped_swiglu_det_train_{step}_domain_plan")).parameters.values()
        )
        assert got == want[:-1], step
    helper = inspect.signature(domain_empty).parameters
    assert list(helper) == ["shape", "dtype", "device"]
    assert helper["dtype"].default is torch.bfloat16 and helper["device"].default is None
    release = inspect.signature(domain_release).parameters
    assert list(release) == ["device"] and release["device"].default is None


# A fresh interpreter: import the package, call the forward step, the backward step and the autograd entry with
# ``"off"`` and ``"domain"`` on CPU tensors (each refused, as a CPU tensor is), then list the domain modules loaded.
FRESH = """
import importlib, sys
import torch
backend = importlib.import_module({package!r})
x = torch.zeros(2, 4608, dtype=torch.bfloat16)
for locality in ("off", "domain"):
    for call in (
        lambda: backend.cake_clamped_swiglu_det_train_forward(x, limit=10.0, locality=locality),
        lambda: backend.cake_clamped_swiglu_det_train_backward(x[:, :2304], x, limit=10.0, locality=locality),
    ):
        try:
            call()
        except ValueError:
            pass
loaded = sorted(m for m in sys.modules if m.endswith((".cake_domain", ".cake_locality")))
print("loaded", loaded)
"""


@pytest.mark.cpu_only
def test_off_imports_no_domain_module() -> None:
    """In a fresh interpreter, importing the package and calling its steps with ``"off"`` and ``"domain"`` (refused
    for a CPU tensor) imports neither ``cake_domain`` nor ``cake_locality``."""
    r = subprocess.run(
        [sys.executable, "-c", FRESH.format(package=pkg.__name__)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "loaded []", (r.stdout, r.stderr[-2000:])


# --------------------------------------------------------------------------- CPU-only: the helper against a fake driver

CUresult = enum.Enum("CUresult", ("CUDA_SUCCESS", "CUDA_ERROR_OUT_OF_MEMORY", "CUDA_ERROR_INVALID_VALUE"))
OK = CUresult.CUDA_SUCCESS
GRANULE = 2097152


class _FakeDriver:
    """The driver calls of ``cake_domain._map`` without a GPU.  ``call`` returns ``status`` at chunk ``k`` of every
    attempt (``cuMemSetAccess`` at any chunk); ``log`` records each reservation, created handle, mapped and unmapped
    chunk, released handle and freed reservation as ``(what, value)``; ``ids`` the location id of each created
    chunk."""

    CUresult = CUresult
    CUdeviceptr = int
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=1, CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN=2)
    CUmemAllocationGranularity_flags = SimpleNamespace(CU_MEM_ALLOC_GRANULARITY_MINIMUM=0)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    def __init__(self, call: str, k: int, status: CUresult, chunk: int = GRANULE) -> None:
        self.call, self.k, self.status, self.chunk = call, k, status, chunk
        self.log: list[tuple[str, int]] = []
        self.ids: list[int] = []
        self.base = 0
        self.created = 0
        self.handles = 0
        self.contexts = 0

    def _status(self, call: str, k: int) -> CUresult:
        return self.status if call == self.call and (k == self.k or call == "cuMemSetAccess") else OK

    @staticmethod
    def CUmemAllocationProp() -> SimpleNamespace:
        return SimpleNamespace(type=None, location=SimpleNamespace(type=None, id=None))

    @staticmethod
    def CUmemAccessDesc() -> SimpleNamespace:
        return SimpleNamespace(location=SimpleNamespace(type=None, id=None), flags=None)

    def cuInit(self, flags):
        return (OK,)

    def cuDeviceGet(self, index):
        return (OK, index)

    def cuDevicePrimaryCtxRetain(self, device):
        self.contexts += 1
        return (OK, "context")

    def cuDevicePrimaryCtxRelease(self, device):
        self.contexts -= 1
        return (OK,)

    def cuCtxPushCurrent(self, context):
        return (OK,)

    def cuCtxPopCurrent(self):
        return (OK, "context")

    def cuMemGetAllocationGranularity(self, prop, flags):
        return (OK, GRANULE)

    def cuMemAddressReserve(self, size, alignment, address, flags):
        self.base += 1 << 40
        self.created = 0
        self.log.append(("reserve", self.base))
        return (OK, self.base)

    def cuMemCreate(self, size, prop, flags):
        status = self._status("cuMemCreate", self.created)
        self.created += 1
        if status != OK:
            return (status, None)
        self.handles += 1
        self.ids.append(prop.location.id)
        self.log.append(("create", self.handles))
        return (OK, self.handles)

    def cuMemMap(self, address, size, offset, handle, flags):
        k = (address - self.base) // self.chunk
        status = self._status("cuMemMap", k)
        if status == OK:
            self.log.append(("map", k))
        return (status,)

    def cuMemRelease(self, handle):
        self.log.append(("release", handle))
        return (OK,)

    def cuMemSetAccess(self, address, size, descriptors, count):
        return (self._status("cuMemSetAccess", 0),)

    def cuMemUnmap(self, address, size):
        self.log.append(("unmap", (address - self.base) // self.chunk))
        return (OK,)

    def cuMemAddressFree(self, address, size):
        self.log.append(("free", address))
        return (OK,)


def _attempts(fake: _FakeDriver) -> list[dict[str, list[int]]]:
    attempts: list[dict[str, list[int]]] = []
    for what, value in fake.log:
        if what == "reserve":
            attempts.append({kind: [] for kind in ("reserve", "create", "map", "release", "unmap", "free")})
        attempts[-1][what].append(value)
    return attempts


def _fake_driver(monkeypatch, call: str, k: int, status: CUresult, chunk: int = GRANULE):
    fake = _FakeDriver(call, k, status, chunk)
    monkeypatch.setattr(memory, "_driver", lambda: fake)
    emptied: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    return fake, emptied


@pytest.mark.cpu_only
@pytest.mark.parametrize("chunk", [CHUNK[I], CHUNK[2 * I]])
def test_a_mapping_alternates_the_dies_chunk_by_chunk(chunk: int, monkeypatch) -> None:
    """Chunk ``k`` of a range is created in die ``k & 1`` of the device (location id ``device | die << 8``), in chunks
    of 2 MB or 4 MB (two granules)."""
    fake, _ = _fake_driver(monkeypatch, "none", -1, OK, chunk)
    _, size = memory._map_or_out_of_memory(fake, 3, chunk, 5)
    assert size == 5 * chunk and fake.ids == [3 | (k & 1) << 8 for k in range(5)]
    assert _attempts(fake)[0]["map"] == list(range(5)) and fake.contexts == 0


@pytest.mark.cpu_only
def test_out_of_memory_unwinds_each_attempt(monkeypatch) -> None:
    k, chunks = 3, 8
    fake, emptied = _fake_driver(monkeypatch, "cuMemCreate", k, CUresult.CUDA_ERROR_OUT_OF_MEMORY)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="no room"):
        memory._map_or_out_of_memory(fake, 0, GRANULE, chunks)
    assert len(emptied) == 1
    attempts = _attempts(fake)
    assert len(attempts) == 2
    for attempt in attempts:
        assert len(attempt["create"]) == k
        assert attempt["map"] == list(range(k)) and attempt["unmap"] == list(range(k))
        assert sorted(attempt["release"]) == attempt["create"]
        assert attempt["free"] == attempt["reserve"]
    assert fake.contexts == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("call", ("cuMemMap", "cuMemSetAccess"))
def test_a_failed_mapping_unwinds_without_retry(call: str, monkeypatch) -> None:
    k, chunks = 3, 8
    fake, emptied = _fake_driver(monkeypatch, call, k, CUresult.CUDA_ERROR_INVALID_VALUE)
    with pytest.raises(memory.DriverError, match=call) as raised:
        memory._map_or_out_of_memory(fake, 0, GRANULE, chunks)
    assert raised.value.status is CUresult.CUDA_ERROR_INVALID_VALUE
    assert emptied == []
    (attempt,) = _attempts(fake)
    mapped = k if call == "cuMemMap" else chunks
    assert sorted(attempt["release"]) == attempt["create"]
    assert attempt["map"] == list(range(mapped)) and attempt["unmap"] == list(range(mapped))
    assert attempt["free"] == attempt["reserve"]
    assert fake.contexts == 0


@pytest.mark.cpu_only
def test_driver_needs_cuda_python(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cuda.bindings", None)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", None)
    with pytest.raises(RuntimeError, match="cuda-python"):
        memory._driver.__wrapped__()
    bindings = ModuleType("cuda.bindings")
    bindings.driver = ModuleType("cuda.bindings.driver")
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    with pytest.raises(RuntimeError, match="does not name"):
        memory._driver.__wrapped__()


def _stub_allocation(monkeypatch) -> list:
    """The stubs of :func:`cake_domain.domain_empty` on the CPU: an SM107 device with two dies, no capture, a probed
    SM table, no claim counters to map; ``_free`` records what it frees (the list returned) and the registry starts
    empty."""
    freed: list = []
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    monkeypatch.setattr(memory, "_RANGES", {})
    monkeypatch.setattr(memory, "_free", freed.append)
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    monkeypatch.setattr(memory, "_driver", lambda: SimpleNamespace())
    monkeypatch.setattr(memory, "domain_count", lambda index: memory.DIES)
    monkeypatch.setattr(memory, "_make_graph_workspaces", lambda index: None)
    monkeypatch.setattr(memory, "_probe_once", lambda index, arch: None)
    monkeypatch.setattr(memory, "_arch", lambda index: ARCH)
    return freed


@pytest.mark.cpu_only
def test_a_range_queued_under_the_lock_is_freed_when_the_lock_is_released(monkeypatch) -> None:
    """A finalizer that finds the module's lock taken (another thread, or a garbage collection inside an allocation
    on this thread) only queues its range and never waits; the call that holds the lock frees it as it releases it:
    here :func:`domain_empty`, whose mapping fails after the finalizer ran.  A pass that can free nothing (during a
    capture) leaves the range queued for a later call."""
    freed = _stub_allocation(monkeypatch)
    entry = memory._Mapped(0, 1 << 30, GRANULE, GRANULE)
    queued: list = []

    def map_after_a_finalizer(driver, index, chunk, chunks):
        memory._release(entry)  # the finalizer of a range whose last view died while domain_empty holds the lock
        queued.append((list(freed), list(memory._PENDING)))
        raise RuntimeError("mapping refused")

    monkeypatch.setattr(memory, "_map_or_out_of_memory", map_after_a_finalizer)
    with pytest.raises(RuntimeError, match="mapping refused"):
        memory.domain_empty((4, I), device="cuda:0")
    assert queued == [([], [entry])], queued
    assert freed == [entry] and not memory._PENDING
    monkeypatch.setattr(memory, "_capturing", lambda: True)
    memory._release(entry)
    assert freed == [entry] and list(memory._PENDING) == [entry]
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    memory._drain()
    assert freed == [entry, entry] and not memory._PENDING


@pytest.mark.cpu_only
def test_a_garbage_collection_inside_an_allocation_only_queues(monkeypatch) -> None:
    """A range whose owner a garbage collection collects while this thread holds the module's lock (inside an
    allocation): its finalizer runs ``gc.collect()``'s way, under the lock, only queues the range and returns; the
    range is freed once the lock is released."""
    freed = _stub_allocation(monkeypatch)
    owner = memory._Range()
    entry = memory._Mapped(0, 1 << 31, GRANULE, GRANULE)
    finalizer = memory.weakref.finalize(owner, memory._release, entry)
    finalizer.atexit = False
    cycle = [owner]
    cycle.append(cycle)
    del owner
    with memory._LOCK:
        del cycle
        gc.collect()
        assert list(memory._PENDING) == [entry] and freed == []
    memory._drain()
    assert freed == [entry] and not memory._PENDING


@pytest.mark.cpu_only
def test_the_registry_entry_goes_before_the_range_is_unmapped(monkeypatch) -> None:
    """A pending range leaves the registry before its free runs, so an address the driver hands out again is never
    taken for that memory; a free that fails keeps the range queued, unregistered."""
    seen: list = []
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    monkeypatch.setattr(memory, "_RANGES", {})
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    memory._register(0, 1 << 30, 4 * GRANULE, GRANULE, 0)

    def free(entry):
        seen.append(memory.registered(0))
        if len(seen) == 1:
            raise RuntimeError("unmap refused")

    monkeypatch.setattr(memory, "_free", free)
    entry = memory._Mapped(0, 1 << 30, 4 * GRANULE, GRANULE)
    memory._release(entry)
    assert seen == [()] and list(memory._PENDING) == [entry] and memory.registered(0) == ()
    memory._drain()
    assert seen == [(), ()] and not memory._PENDING


@pytest.mark.cpu_only
def test_release_frees_the_devices_queue(monkeypatch) -> None:
    """``release(index)`` frees that device's queued ranges, then (as every call that takes the module's lock) the
    rest of the queue; with none of the device's ranges queued it does nothing, and with some it refuses a capture."""
    freed: list = []
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    monkeypatch.setattr(memory, "_RANGES", {})
    monkeypatch.setattr(memory, "_free", freed.append)
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    a, b = memory._Mapped(0, 1 << 30, GRANULE, GRANULE), memory._Mapped(1, 1 << 31, GRANULE, GRANULE)
    memory._PENDING.extend([a, b])
    assert memory.pending() == 2 and memory.pending(1) == 1
    assert memory.release(1) == 1
    assert freed == [b, a] and not memory._PENDING
    assert memory.release(1) == 0
    memory._PENDING.append(a)
    monkeypatch.setattr(memory, "_capturing", lambda: True)
    assert memory.release(1) == 0
    with pytest.raises(RuntimeError, match="capture"):
        memory.release(0)
    assert list(memory._PENDING) == [a] and freed == [b, a]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("shape", "dtype", "error"),
    [
        ((4, I), torch.float32, ValueError),
        ((4, I), "bfloat16", TypeError),
        ((4, 1024), torch.bfloat16, ValueError),
        ((4, I, 1), torch.bfloat16, ValueError),
        ((-1, I), torch.bfloat16, ValueError),
        (7, torch.bfloat16, ValueError),
    ],
)
def test_domain_empty_refuses_other_shapes_and_dtypes(shape, dtype, error, monkeypatch) -> None:
    _stub_allocation(monkeypatch)
    with pytest.raises(error):
        memory.domain_empty(shape, dtype=dtype, device="cuda:0")


@pytest.mark.cpu_only
def test_domain_empty_refuses_another_architecture_before_the_driver(monkeypatch) -> None:
    """On SM100 and SM103 the helper raises ``RuntimeError`` before it imports cuda-python."""
    for arch in ("sm_100a", "sm_103a"):
        monkeypatch.setattr(memory, "_arch", lambda index, arch=arch: arch)
        monkeypatch.setattr(memory, "_driver", lambda: pytest.fail("the driver was reached"))
        with pytest.raises(RuntimeError, match="SM107"):
            memory.domain_empty((4, I), device="cuda:0")


@pytest.mark.cpu_only
@pytest.mark.parametrize("width", [I, 2 * I])
def test_domain_empty_registers_its_range(width: int, monkeypatch) -> None:
    """A ``(R, width)`` allocation maps ``ceil(2 R width / chunk)`` chunks of its width's chunk and registers the
    range with its chunk and die 0 first."""
    _stub_allocation(monkeypatch)
    mapped: list = []

    def fake_map(driver, index, chunk, chunks):
        mapped.append((index, chunk, chunks))
        return 1 << 33, chunk * chunks

    exported: list = []

    def fake_dlpack(producer):
        exported.append(producer)  # the producer keeps the range's owner alive, as a real view does
        return torch.zeros(producer.size, dtype=torch.uint8)

    monkeypatch.setattr(memory, "_map_or_out_of_memory", fake_map)
    monkeypatch.setattr(torch, "from_dlpack", fake_dlpack)
    rows = 1000
    t = memory.domain_empty((rows, width), device="cuda:0")
    chunk = CHUNK[width]
    chunks = -(-rows * width * 2 // chunk)
    assert mapped == [(0, chunk, chunks)]
    assert memory.registered(0) == ((1 << 33, chunks * chunk, chunk, 0),)
    assert tuple(t.shape) == (rows, width) and t.dtype == torch.bfloat16
    assert [(p.base, p.size, p.index) for p in exported] == [(1 << 33, chunks * chunk, 0)]


@pytest.mark.cpu_only
def test_the_sm_table_is_probed_once_and_checked(monkeypatch) -> None:
    """The device's first allocation probes its SM table once (a failed probe is kept as None); a table that is not a
    contiguous CPU int32 tensor of 256 words is no table."""
    calls: list = []
    good = torch.arange(256, dtype=torch.int32)
    tables = {
        0: SimpleNamespace(entries=good),
        1: None,
        2: SimpleNamespace(entries=good.to(torch.int64)),
        3: SimpleNamespace(entries=good[:128]),
        4: SimpleNamespace(entries=torch.arange(512, dtype=torch.int32)[::2]),
    }
    fake = ModuleType("cake_locality")

    def probe_table(index, arch):
        calls.append((index, arch))
        return tables[index]

    fake.probe_table = probe_table
    name = f"{pkg.__name__}.cake_locality"
    monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(pkg, "cake_locality", fake, raising=False)
    monkeypatch.setattr(memory, "_TABLES", {})
    for index in tables:
        memory._probe_once(index, ARCH)
        memory._probe_once(index, ARCH)
    assert calls == [(index, ARCH) for index in tables]
    assert memory.sm_table(0) is good
    assert all(memory.sm_table(index) is None for index in (1, 2, 3, 4))
    assert memory.sm_table(5) is None


@pytest.mark.cpu_only
def test_the_claim_counters_are_zeroed_before_they_are_published(monkeypatch) -> None:
    """The device's first allocation maps one counter chunk in each die (chunk ``d`` in die ``d``), zeroes it and waits
    for the fill before it records the workspaces, once per device; the eager slots follow the graph slots."""
    fake, _ = _fake_driver(monkeypatch, "none", -1, OK)
    monkeypatch.setattr(memory, "_GRAPH_WORKSPACES", {})
    backing = torch.full((memory.DIES * memory.COUNTER_CHUNK_BYTES,), 0x5A, dtype=torch.uint8)
    monkeypatch.setattr(torch, "from_dlpack", lambda producer: backing)
    seen: list = []

    def synchronize() -> None:
        seen.append((int(backing.count_nonzero()), 0 in memory._GRAPH_WORKSPACES))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: SimpleNamespace(synchronize=synchronize))
    memory._make_graph_workspaces(0)
    assert seen == [(0, False)]
    held = memory._GRAPH_WORKSPACES[0]
    assert held["taken"] == 0 and held["eager"] == memory.GRAPH_WORKSPACES
    assert held["slots"] == memory.COUNTER_CHUNK_BYTES // (4 * memory.SLOT_WORDS)
    assert tuple(held["block"].shape) == (memory.DIES, memory.COUNTER_CHUNK_BYTES // 4)
    assert fake.ids == [0, 1 << 8]
    memory._make_graph_workspaces(0)
    assert len(_attempts(fake)) == 1 and len(seen) == 1
    for k in (memory.GRAPH_WORKSPACES, memory.GRAPH_WORKSPACES + 1):
        words = memory._eager_workspace(0).words
        assert words.data_ptr() == held["block"][0, k * memory.SLOT_WORDS].data_ptr()
        assert words.stride() == held["block"].stride() and tuple(words.shape) == (memory.DIES, memory.SLOT_WORDS)


@pytest.mark.cpu_only
def test_a_failed_counter_fill_frees_the_counters(monkeypatch) -> None:
    _fake_driver(monkeypatch, "none", -1, OK)
    monkeypatch.setattr(memory, "_GRAPH_WORKSPACES", {})
    freed: list = []
    monkeypatch.setattr(memory, "_free", freed.append)

    def refuse(producer):
        raise RuntimeError("export refused")

    monkeypatch.setattr(torch, "from_dlpack", refuse)
    with pytest.raises(RuntimeError, match="export refused"):
        memory._make_graph_workspaces(0)
    chunk = memory.COUNTER_CHUNK_BYTES
    assert [(e.index, e.size, e.chunk) for e in freed] == [(0, memory.DIES * chunk, chunk)]
    assert memory._GRAPH_WORKSPACES == {}


@pytest.mark.cpu_only
def test_a_capture_takes_one_graph_workspace_for_good(monkeypatch) -> None:
    """The calls of one capture on one stream share one graph workspace; another capture takes the next; a call whose
    capture id the driver does not give takes one of its own.  Every one is taken for good."""
    block = torch.zeros(memory.DIES, 64 * memory.SLOT_WORDS, dtype=torch.int32)
    workspaces = {0: {"block": block, "taken": 0, "eager": memory.GRAPH_WORKSPACES, "slots": 64}}
    monkeypatch.setattr(memory, "_GRAPH_WORKSPACES", workspaces)
    monkeypatch.setattr(memory, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    ids = iter([7, 7, 8, None, None])
    monkeypatch.setattr(memory, "_capture_id", lambda stream: next(ids))
    key = (0, 0x55)
    first = memory._capture_workspace(key)
    assert memory._capture_workspace(key) is first
    taken = [first, memory._capture_workspace(key), memory._capture_workspace(key), memory._capture_workspace(key)]
    assert [w.words.data_ptr() for w in taken] == [block[0, k * memory.SLOT_WORDS].data_ptr() for k in range(4)]
    assert memory.graph_workspaces_left(0) == memory.GRAPH_WORKSPACES - 4


@pytest.mark.cpu_only
def test_the_per_thread_default_stream_has_a_key_per_thread() -> None:
    keys: list = []
    for _ in range(2):
        thread = threading.Thread(target=lambda: keys.append(memory._stream_key(0, memory.PER_THREAD_STREAM)))
        thread.start()
        thread.join()
    here = memory._stream_key(0, memory.PER_THREAD_STREAM)
    assert here == memory._stream_key(0, memory.PER_THREAD_STREAM)
    assert len({here, *keys}) == 3, (here, keys)
    assert memory._stream_key(0, 0) == (0, 0)


@pytest.mark.cpu_only
def test_a_workspace_gives_each_die_its_row() -> None:
    block = torch.arange(memory.DIES * 4 * memory.SLOT_WORDS, dtype=torch.int32).view(memory.DIES, -1)
    words = block[:, memory.SLOT_WORDS : 2 * memory.SLOT_WORDS]
    workspace = memory._Workspace(words)
    counters = workspace.counters(1)
    assert workspace.counters(1) is counters and len(counters) == memory.DIES
    for d, row in enumerate(counters):
        assert row.dtype == torch.uint32 and row.shape == (1,) and row.data_ptr() == words[d].data_ptr()


@pytest.mark.cpu_only
def test_a_launch_passes_the_counters_between_the_base_arguments_and_the_tail(monkeypatch) -> None:
    """``cake_domain.launch`` calls the entry with the base program's arguments, the two dies' counters of the
    stream's workspace (``claim``'s, the same objects at every call), then the rest; a launch that raises drops the
    stream's workspace."""
    monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(memory, "_current_stream_handle", lambda index: 0x77)
    monkeypatch.setattr(memory, "_stream_capturing", lambda index: False)
    block = torch.zeros(memory.DIES, memory.SLOT_WORDS, dtype=torch.int32)
    monkeypatch.setattr(memory, "_eager_workspace", lambda index: memory._Workspace(block))
    calls: list = []
    memory.launch(lambda *args: calls.append(args), 1, 0, ("a", "b"), ("c", 1, 1, 1))
    ((a, b, c0, c1, c, *grid),) = calls
    assert (a, b, c, grid) == ("a", "b", "c", [1, 1, 1])
    # the counters as the entry takes them: views of the workspace's two rows, made once per count
    views = [torch.from_dlpack(counters) for counters in (c0, c1)]
    assert [view.data_ptr() for view in views] == [block[0].data_ptr(), block[1].data_ptr()]
    assert all(view.dtype == torch.uint32 and tuple(view.shape) == (1,) for view in views)
    again = memory.claim(0, 1)
    assert again[0] is c0 and again[1] is c1 and again[2] == (0, 0x77)

    def boom(*args):
        raise RuntimeError("launch refused")

    with pytest.raises(RuntimeError, match="refused"):
        memory.launch(boom, 1, 0, (), ())
    assert memory._STREAM_WORKSPACES == {}


# --------------------------------------------------------------------------- CPU-only: the SM table and its probe

# The table word the twins read (a literal copy of the layout: a drift fails here): field -> (shift, bits).
ENTRY_FIELDS = {"rank": (0, 10), "count": (10, 10), "die": (20, 4), "supplemental": (24, 1)}
SM_TABLE_LENGTH = 256
PROBE_ATTEMPTS = 4
# The opt-in shared memory of the stand-in device's SM (the probe asks each CTA for all of it).
SMEM_OPTIN = 232448
# The dies of the stand-in device's 212 SMs: two SM ids on die 0, two on die 1, and so on (106 each); and an uneven
# device (112 and 100).
EVEN_DIES = [(sm >> 1) & 1 for sm in range(212)]
UNEVEN_DIES = [(sm >> 1) & 1 if sm < 200 else 0 for sm in range(212)]


def _probe_rows(dies: list[int], order=None, nsmid=None) -> list[list[int]]:
    """The probe's rows ``(smid, domain, supplemental, nsmid)``, CTA ``i`` on SM ``order[i]`` (default: the SM ids in
    reverse, so a table that followed the rows' order would be wrong); SM ``sm`` is on die ``dies[sm]``, every 37th
    SM from SM 5 is supplemental."""
    order = list(reversed(range(len(dies)))) if order is None else order
    top = len(dies) if nsmid is None else nsmid
    return [[sm, dies[sm], int(sm % 37 == 5), top] for sm in order]


def _missing_one(dies: list[int], missing: int = 18, twice: int = 17) -> list[list[int]]:
    """A launch that misses SM ``missing``: two CTAs ran on SM ``twice`` (another context held an SM)."""
    order = [twice if sm == missing else sm for sm in reversed(range(len(dies)))]
    return _probe_rows(dies, order)


def _table_words(rows: list[list[int]]) -> list[int]:
    """The table of ``rows`` restated from :data:`ENTRY_FIELDS`: SM ``sm``'s rank among its die's SM ids, its die's SM
    count, its die and its flag; 0 at an SM id no row names."""
    members = {die: sorted(row[0] for row in rows if row[1] == die) for die in (0, 1)}
    words = [0] * SM_TABLE_LENGTH
    for sm, die, supplemental, _ in rows:
        fields = {"rank": members[die].index(sm), "count": len(members[die]), "die": die, "supplemental": supplemental}
        words[sm] = sum(value << ENTRY_FIELDS[field][0] for field, value in fields.items())
    return words


@pytest.mark.cpu_only
def test_the_sm_table_packs_each_sms_rank_count_die_and_flag() -> None:
    """``pack_rows`` packs one word per SM id: its rank among its die's SM ids, its die's SM count, its die and its
    supplemental flag, in the twins' layout; the counts are the dies' SMs; an SM id no row names (a gap) stays 0 and
    the ranks skip it.  The rows' order does not matter."""
    fields = {
        "rank": (locality.ENTRY_RANK_SHIFT, locality.ENTRY_BITS),
        "count": (locality.ENTRY_COUNT_SHIFT, locality.ENTRY_BITS),
        "die": (locality.ENTRY_DOMAIN_SHIFT, locality.ENTRY_DOMAIN_BITS),
        "supplemental": (locality.ENTRY_SUPPLEMENTAL_SHIFT, 1),
    }
    assert fields == ENTRY_FIELDS and locality.SM_TABLE_LENGTH == memory.SM_TABLE_WORDS == SM_TABLE_LENGTH
    for dies in (EVEN_DIES, UNEVEN_DIES):
        for rows in (_probe_rows(dies), _probe_rows(dies, order=list(range(len(dies))))):
            counts, words = locality.pack_rows(rows)
            assert counts == (dies.count(0), dies.count(1)) and words == _table_words(rows)
    # SM 5 (die 0, the 4th of its die's SM ids 0, 1, 4, 5, ...) is supplemental; SM 6 is the 3rd of die 1's (2, 3, 6)
    words = locality.pack_rows(_probe_rows(EVEN_DIES))[1]
    assert words[5] == 3 | 106 << 10 | 0 << 20 | 1 << 24 and words[6] == 2 | 106 << 10 | 1 << 20
    # SM ids 0 to 212 without SM 5: its word is 0, and SM 8 is the 4th of die 0's SM ids (0, 1, 4, 8)
    gap = [[sm, (sm >> 1) & 1, 0, 213] for sm in range(213) if sm != 5]
    counts, words = locality.pack_rows(gap)
    assert counts == (106, 106) and words == _table_words(gap)
    assert words[5] == 0 and words[4] == 2 | 106 << 10 and words[8] == 3 | 106 << 10


def _changed(rows: list[list[int]], index: int, column: int, value: int) -> list[list[int]]:
    rows = [list(row) for row in rows]
    rows[index][column] = value
    return rows


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows",
    [
        _changed(_probe_rows(EVEN_DIES), 10, 0, _probe_rows(EVEN_DIES)[11][0]),
        _changed(_probe_rows(EVEN_DIES), 10, 1, 2),
        _changed(_probe_rows(EVEN_DIES), 10, 1, -1),
        _probe_rows([0] * 212),
        _probe_rows([1] * 212),
        _changed(_probe_rows(EVEN_DIES), 0, 0, 256),
        _changed(_probe_rows(EVEN_DIES), 0, 0, -1),
        _probe_rows(EVEN_DIES, nsmid=257),
    ],
    ids=["duplicate_sm", "die_2", "die_minus_1", "die_0_only", "die_1_only", "sm_id_256", "sm_id_minus_1", "nsmid_257"],
)
def test_the_sm_table_refuses_rows_it_cannot_pack(rows) -> None:
    """``pack_rows`` gives no table for two rows of one SM id, a die outside 0 and 1, one die only, an SM id outside
    ``[0, 256)`` or an SM-id bound above 256."""
    assert locality.pack_rows(rows) is None


class _FakeProbeLibrary:
    """The package's probe library without a GPU: the driver is CUDA 13.5's, and launch ``i`` of ``probe(out, ctas,
    smem)`` writes the rows ``attempts[i]`` (the last again past the end) into ``out``; ``calls`` holds ``(ctas,
    smem)`` of each launch."""

    def __init__(self, attempts: list[list[list[int]]], driver: int = 13050) -> None:
        self.attempts, self.driver, self.calls = attempts, driver, []

    def driver_version(self) -> int:
        return self.driver

    def probe(self, out: torch.Tensor, ctas: int, smem: int) -> None:
        rows = self.attempts[min(len(self.calls), len(self.attempts) - 1)]
        self.calls.append((ctas, smem))
        out.copy_(torch.tensor(rows, dtype=torch.int32))


@pytest.fixture
def dynamo_loaded() -> None:
    """Dynamo loaded before any device stand-in: the probe runs under ``torch.compiler.disable``, and Dynamo's device
    interfaces keep the ``torch.cuda`` objects they find when they load."""
    torch.compiler.disable(lambda: None)


@pytest.fixture
def fake_probe(dynamo_loaded, fake_r200, monkeypatch):
    """The package's own probe path on ``fake_r200``'s CPU stand-in of an SM107 device 0: ``cake_locality``'s
    ``probe_table`` (its capture check, up to ``PROBE_ATTEMPTS`` launches until the rows name every SM, ``pack_rows``,
    its cache) under ``cake_domain``'s ``_probe_once`` (the checks of ``_probe_table``, the device's cache that
    ``sm_table`` and the route read).  ``load(attempts)`` installs a :class:`_FakeProbeLibrary` and returns it; only
    the device is a stand-in (its properties, the output buffer on the CPU, the stream and its synchronization).  Both
    modules' caches start empty."""
    import tvm_ffi

    properties = SimpleNamespace(multi_processor_count=fake_r200.sm_count, shared_memory_per_block_optin=SMEM_OPTIN)
    zeros = torch.zeros

    def cpu_zeros(*args, device=None, **kwargs):
        if device is not None and torch.device(device).type == "cuda":
            device = None
        return zeros(*args, device=device, **kwargs)

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(tvm_ffi, "use_torch_stream", contextlib.nullcontext)
    for cache in ("_TABLES", "_PROBED"):
        monkeypatch.setattr(locality, cache, {})
    monkeypatch.setattr(locality, "_PROBE_MODULE", [])
    monkeypatch.setattr(memory, "_TABLES", {})

    def load(attempts: list[list[list[int]]], driver: int = 13050) -> _FakeProbeLibrary:
        library = _FakeProbeLibrary(attempts, driver)
        monkeypatch.setattr(locality, "_probe_library", lambda: library)
        return library

    return load


def _placed_call(fake_r200) -> tuple:
    """The route of a forward call whose merged input lies in helper memory, 128 rows per SM (anchor class)."""
    rows = 128 * fake_r200.sm_count
    anchor = FakeTensor(fake_r200.add(2 * I, 600), (rows, 2 * I))
    return _route("fwd", rows, anchor, outputs=(("h", None),))


@pytest.mark.cpu_only
@pytest.mark.parametrize("dies", [EVEN_DIES, UNEVEN_DIES], ids=["even", "uneven"])
def test_the_probe_keeps_a_table_that_covers_every_sm(dies, fake_probe, fake_r200) -> None:
    """A probe whose first launch names every SM gives the device's table once: one launch of a CTA per SM with the
    SM's whole opt-in shared memory; ``sm_table`` is the packed table as a contiguous CPU int32 tensor of 256 words,
    the probe's cache holds it with the dies' counts, a second allocation does not probe again, and the route places a
    call it would refuse with reason ``sm_table`` before."""
    assert locality.PROBE_ATTEMPTS == PROBE_ATTEMPTS
    rows = _probe_rows(dies)
    library = fake_probe([rows])
    assert _placed_call(fake_r200)[1] == "sm_table"
    memory._probe_once(0, ARCH)
    assert library.calls == [(fake_r200.sm_count, SMEM_OPTIN)]
    table = memory.sm_table(0)
    assert isinstance(table, torch.Tensor) and table.device.type == "cpu" and table.dtype == torch.int32
    assert tuple(table.shape) == (SM_TABLE_LENGTH,) and table.is_contiguous()
    assert table.tolist() == _table_words(rows)
    probed = locality.probe_table(0, ARCH)
    assert probed.counts == (dies.count(0), dies.count(1)) and probed.entries is table
    memory._probe_once(0, ARCH)
    assert len(library.calls) == 1
    assert _placed_call(fake_r200)[1:3] == ("placed", "anchor")


@pytest.mark.cpu_only
@pytest.mark.parametrize("misses", [1, PROBE_ATTEMPTS - 1])
def test_a_probe_launch_that_misses_an_sm_is_taken_again(misses: int, fake_probe) -> None:
    """A launch whose rows miss an SM (two CTAs on one SM) is taken again, up to ``PROBE_ATTEMPTS`` launches in all:
    the first launch that names every SM gives the table."""
    rows = _probe_rows(EVEN_DIES)
    library = fake_probe([_missing_one(EVEN_DIES)] * misses + [rows])
    memory._probe_once(0, ARCH)
    assert len(library.calls) == misses + 1
    assert memory.sm_table(0).tolist() == _table_words(rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "launch", [_missing_one(EVEN_DIES), _missing_one(EVEN_DIES, missing=0, twice=211)], ids=["sm_18", "sm_0"]
)
def test_a_probe_that_never_covers_every_sm_gives_no_table(launch, fake_probe, fake_r200) -> None:
    """A probe whose ``PROBE_ATTEMPTS`` launches each miss an SM gives no table, kept for the device: ``sm_table`` is
    None, a later allocation does not probe again, and a ``"domain"`` call runs the ``"off"`` program with reason
    ``sm_table``."""
    library = fake_probe([launch])
    memory._probe_once(0, ARCH)
    assert len(library.calls) == PROBE_ATTEMPTS
    assert memory.sm_table(0) is None and memory._TABLES == {0: None} and locality._PROBED == {(0, ARCH): None}
    memory._probe_once(0, ARCH)
    assert locality.probe_table(0, ARCH) is None and len(library.calls) == PROBE_ATTEMPTS
    assert _placed_call(fake_r200)[1] == "sm_table"


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "rows",
    [
        _changed(_probe_rows(EVEN_DIES), 40, 1, 2),
        _changed(_probe_rows(EVEN_DIES), 40, 1, -1),
        _changed(_probe_rows(EVEN_DIES), 40, 1, 15),
        _probe_rows([0] * 212),
    ],
    ids=["die_2", "die_minus_1", "die_15", "die_0_only"],
)
def test_a_probe_with_a_die_outside_0_and_1_gives_no_table(rows, fake_probe, fake_r200) -> None:
    """A launch that names every SM but a die outside 0 and 1 (or one die only) gives no table, at once (no second
    launch) and kept: a ``"domain"`` call runs the ``"off"`` program with reason ``sm_table``."""
    library = fake_probe([rows])
    memory._probe_once(0, ARCH)
    assert len(library.calls) == 1
    assert memory._TABLES == {0: None} and locality._PROBED == {(0, ARCH): None}
    assert _placed_call(fake_r200)[1] == "sm_table"


# --------------------------------------------------------------------------- GPU: every architecture


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not pkg.is_cake_clamped_swiglu_det_train_supported(torch.device("cuda")):
        pytest.skip(f"the package ships {cake_jit.ARCHES}; this device is {torch.cuda.get_device_capability()}")
    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _inputs(rows: int, *, seed: int, device, merged: bool = True, weighted: bool = True) -> SimpleNamespace:
    g = torch.Generator(device=device).manual_seed(seed)
    gate_up = (6.0 * torch.randn(rows, 2 * I, device=device, generator=g)).bfloat16()
    t = SimpleNamespace(rows=rows, merged=merged)
    t.weight = 1.5 * torch.rand(rows, device=device, generator=g) if weighted else None
    t.dh = torch.randn(rows, I, device=device, generator=g).bfloat16()
    if merged:
        t.first, t.second = gate_up, None
    else:
        t.first, t.second = gate_up[:, :I].contiguous(), gate_up[:, I:].contiguous()
    return t


def _off(t: SimpleNamespace) -> tuple:
    h = forward(t.first, t.second, t.weight, limit=L)
    grads = backward(t.dh, t.first, t.second, t.weight, limit=L)
    return (h, *grads)


def _check_outputs(what: str, got: tuple, want: tuple) -> None:
    for name, a, b in zip(("h", "d_first", "d_up", "d_weight"), got, want, strict=True):
        _assert_same(f"{what}: {name}", a, b)


@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
def test_domain_on_ordinary_memory_runs_the_off_program(merged: bool, weighted: bool, device, monkeypatch) -> None:
    """On every architecture, ``"domain"`` on ordinary memory gives the bits of ``"off"`` (steps and autograd) and
    launches the ``"off"`` programs; the plans say why: ``arch`` off SM107 and for the backward, else ``size``
    below the twin's first row count and ``sm_table`` or ``input_memory`` above it."""
    arch = pkg.device_arch(device.index)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    for rows in (1, 129, 4097, 120 * sms):
        t = _inputs(rows, seed=rows, device=device, merged=merged, weighted=weighted)
        want = _off(t)
        h = forward(t.first, t.second, t.weight, limit=L, locality="domain")
        grads = backward(t.dh, t.first, t.second, t.weight, limit=L, locality="domain")
        _check_outputs(f"{rows} rows", (h, *grads), want)
        fplan = forward_plan(t.first, t.second, t.weight, limit=L)
        bplan = backward_plan(t.dh, t.first, t.second, t.weight, limit=L)
        assert set(fplan) == set(bplan) == PLAN_KEYS
        for plan in (fplan, bplan):
            assert plan["route"] == "plain" and plan["program"] in cake_jit.MODULES
            assert not cake_jit.MODULES[plan["program"]]["role"].endswith("_domain")
        assert fplan["outputs"] == {"h": "empty"}
        assert bplan["stage"] == ("bwd" if weighted else "bwd_unweighted")
        assert bplan["reason"] == "arch", bplan
        if arch != ARCH:
            assert fplan["reason"] == "arch"
        else:
            assert fplan["reason"] in (("size",) if rows < 112 * sms else ("sm_table", "input_memory"))
        leaves = [None if x is None else x.detach().clone().requires_grad_(True) for x in (t.first, t.second, t.weight)]
        h2 = clamped_swiglu(*leaves, limit=L, locality="domain")
        h2.backward(t.dh)
        _assert_same("autograd h", h2.detach(), want[0])
        _assert_same("autograd d_first", leaves[0].grad, want[1])
        if not merged:
            _assert_same("autograd d_up", leaves[1].grad, want[2])
        if weighted:
            _assert_same("autograd d_weight", leaves[2].grad, want[3])


def test_domain_empty_refuses_other_architectures(device) -> None:
    if pkg.device_arch(device.index) == ARCH:
        pytest.skip("SM107 allocates")
    with pytest.raises(RuntimeError, match="SM107"):
        domain_empty((4, I), device=device)


def test_off_calls_load_no_domain_state(device) -> None:
    """In a fresh interpreter on the device, ``"off"`` steps and autograd calls import neither ``cake_domain`` nor
    ``cake_locality``."""
    code = (
        "import importlib, sys, torch\n"
        f"b = importlib.import_module({pkg.__name__!r})\n"
        f"x = torch.randn(300, 4608, device='cuda:{device.index}').bfloat16().requires_grad_()\n"
        "h = b.cake_clamped_swiglu_det(x, limit=10.0)\n"
        "h.backward(torch.ones_like(h))\n"
        "b.cake_clamped_swiglu_det_train_forward(x.detach(), limit=10.0, locality='off')\n"
        "torch.cuda.synchronize()\n"
        "print('loaded', sorted(m for m in sys.modules if m.endswith(('.cake_domain', '.cake_locality'))))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "loaded []", (r.stdout, r.stderr[-2000:])


# --------------------------------------------------------------------------- SM107 (R200)


@pytest.fixture(scope="module")
def r200(device) -> int:
    if pkg.device_arch(device.index) != ARCH:
        pytest.skip("the locality-domain twins run on SM107 (R200) only")
    count = memory.domain_count(device.index)
    assert count >= memory.DIES, f"an SM107 whose driver reports {count} locality domain(s)"
    return device.index


@pytest.fixture
def low_firsts(monkeypatch, r200):
    """The twin's first rows per SM set to 1 for both classes (so small row counts take it), the twin cache cleared
    before and after; returns the SM count."""
    for stage in TWINS:
        monkeypatch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM[stage], ARCH, {"homed": 1, "anchor": 1})
    _clear_twin_caches()
    yield torch.cuda.get_device_properties(r200).multi_processor_count
    _clear_twin_caches()


@pytest.fixture
def twins_run(monkeypatch) -> list:
    """The program of every twin launch, in order (each twin's entry wrapped as it is bound; other programs'
    entries are not wrapped)."""
    found: list = []
    entry = pkg._entry

    def spy(name, arch):
        run = entry(name, arch)

        if not cake_jit.MODULES[name]["role"].endswith(cake_jit.DOMAIN):
            return run

        def recorded(*args):
            found.append(name)
            return run(*args)

        return recorded

    monkeypatch.setattr(pkg, "_entry", spy)
    _clear_twin_caches()
    yield found
    _clear_twin_caches()


@pytest.mark.parametrize("width", [I, 2 * I])
def test_domain_empty_alternates_dies(r200, width: int) -> None:
    """Chunk ``k`` of a helper range lies in die ``k & 1``, at its first and last byte, and the range is registered
    with its width's chunk; the tensor is a zero-copy view (its data pointer is the range's base).  Each die's claim
    counters lie in that die; the range is unregistered once its last view is gone."""
    rows = 2000
    t = domain_empty((rows, width), device=r200)
    chunk = CHUNK[width]
    (entry,) = [e for e in memory.registered(r200) if e[0] == t.data_ptr()]
    assert entry[2:] == (chunk, 0) and entry[1] % chunk == 0
    for k in range(entry[1] // chunk):
        for address in (entry[0] + k * chunk, entry[0] + (k + 1) * chunk - 1):
            assert memory.domain_of(address) == k & 1, (width, k, address)
    assert memory.sm_table(r200) is not None
    # the claim counters: chunk d of the device's counter range in die d, at its first and last word
    block = memory._GRAPH_WORKSPACES[r200]["block"]
    for d in range(memory.DIES):
        assert [memory.domain_of(int(at)) for at in (block[d].data_ptr(), block[d, -1].data_ptr())] == [d, d]
    del t
    gc.collect()
    assert all(e[0] != entry[0] for e in memory.registered(r200))


def test_domain_empty_refuses_capture_and_defers_frees(r200) -> None:
    """During a capture the helper refuses to allocate, and a range whose last view dies there is freed by the next
    helper call (or ``domain_release``) after it."""
    keep = domain_empty((10, I), device=r200)
    base = keep.data_ptr()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(r200)
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="capture"):
            domain_empty((10, I), device=r200)
        del keep
        gc.collect()
    assert any(e[0] == base for e in memory.registered(r200)) and memory.pending(r200) >= 1
    domain_release(r200)
    assert all(e[0] != base for e in memory.registered(r200)) and memory.pending(r200) == 0


def _place(t: SimpleNamespace, r200: int, offset_rows: int, rows: int, homed_outputs: bool) -> SimpleNamespace:
    """``t``'s inputs copied into helper memory of ``offset_rows + rows`` rows, as the row slice from ``offset_rows``;
    with ``homed_outputs``, output buffers from the same slice of helper memory."""
    total = offset_rows + rows

    def slab(width: int, value=None):
        whole = domain_empty((total, width), device=r200)
        view = whole[offset_rows:]
        if value is not None:
            view.copy_(value)
        return view

    d = SimpleNamespace(rows=rows, weight=t.weight, merged=t.merged)
    d.first = slab(t.first.shape[1], t.first)
    d.second = None if t.second is None else slab(I, t.second)
    d.dh = slab(I, t.dh)
    d.h = slab(I) if homed_outputs else None
    d.d_first = slab(t.first.shape[1]) if homed_outputs else None
    d.d_second = slab(I) if homed_outputs and t.second is not None else None
    torch.cuda.synchronize(r200)
    return d


def _domain_step(d: SimpleNamespace) -> tuple:
    h = forward(d.first, d.second, d.weight, limit=L, out=d.h, locality="domain")
    grads = backward(
        d.dh,
        d.first,
        d.second,
        d.weight,
        limit=L,
        grad_gate_up_or_gate=d.d_first,
        grad_up=d.d_second,
        locality="domain",
    )
    return (h, *grads)


# Row slices of the helper's memory: row 0 at the start of a die-0 chunk, inside a chunk, in a die-1 chunk (one chunk
# later) and a later row slice.
PHASES = {"chunk_start": 0, "inside": 200, "die1": 456, "slice": 1500}


@pytest.mark.parametrize("homed", [True, False], ids=["homed", "anchor"])
@pytest.mark.parametrize("grid", ["host", 1, 3])
@pytest.mark.parametrize("phase", sorted(PHASES))
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
def test_the_twin_gives_the_off_bytes(
    merged: bool, phase: str, grid, homed: bool, low_firsts, twins_run, monkeypatch, r200
) -> None:
    """The forward's twin, on helper memory at each phase, on grids of 1 and 3 CTAs and the host's, with an output
    buffer in the anchor's frame (class ``homed``) or a new ``h`` (``anchor``), gives the bits of ``"off"`` on ordinary
    memory; the plan names the twin with the anchor's first slot and die.  The backward on the same memory runs its
    ``"off"`` program (reason ``arch``) with the ``"off"`` bits."""
    if grid != "host":
        monkeypatch.setattr(pkg, "_twin_grid", lambda twin, rows, place, grid=grid: grid)
    rows = 1237
    t = _inputs(rows, seed=PHASES[phase] + rows, device=torch.device("cuda", r200), merged=merged)
    want = _off(t)
    d = _place(t, r200, PHASES[phase], rows, homed)
    slot = (ROW_SLOTS * PHASES[phase]) % SLOTS
    die = ((ROW_SLOTS * PHASES[phase]) // SLOTS) & 1
    fplan = forward_plan(d.first, d.second, d.weight, limit=L, out=d.h)
    bplan = backward_plan(
        d.dh, d.first, d.second, d.weight, limit=L, grad_gate_up_or_gate=d.d_first, grad_up=d.d_second
    )
    assert fplan["route"] == "domain" and fplan["reason"] == "placed", fplan
    assert fplan["class"] == ("homed" if homed else "anchor") and fplan["program"] == _twin("fwd")
    assert (fplan["first_slot"], fplan["first_die"], fplan["chunk_shift"]) == (slot, die, 12)
    assert fplan["item_rows"] == TWINS["fwd"]["item_rows"]
    if grid == "host":
        assert fplan["grid"] == cake_jit.domain_grid(_twin("fwd"), ARCH, rows, slot, die, low_firsts)
    assert (bplan["route"], bplan["reason"], bplan["class"]) == ("plain", "arch", None), bplan
    got = _domain_step(d)
    torch.cuda.synchronize(r200)
    assert twins_run == [_twin("fwd")]
    _check_outputs(f"{phase} grid {grid}", got, want)
    # run to run: the same bits again
    _check_outputs(f"{phase} grid {grid} again", _domain_step(d), want)


@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
def test_the_backward_on_helper_memory_runs_its_off_program(merged: bool, weighted: bool, low_firsts, twins_run, r200):
    """The backward, weighted and unweighted, with every operand and output buffer in the anchor's row frame (the
    forward's ``homed`` class) runs the ``"off"`` program of its row count (reason ``arch``: no backward stage has a
    twin) with the ``"off"`` bits; the forward on the same memory runs its twin."""
    rows = 999
    t = _inputs(rows, seed=5, device=torch.device("cuda", r200), merged=merged, weighted=weighted)
    want = _off(t)
    d = _place(t, r200, 0, rows, True)
    plan = backward_plan(d.dh, d.first, d.second, d.weight, limit=L, grad_gate_up_or_gate=d.d_first, grad_up=d.d_second)
    stage = "bwd" if weighted else "bwd_unweighted"
    assert (plan["stage"], plan["route"], plan["reason"], plan["class"]) == (stage, "plain", "arch", None)
    assert (plan["program"], plan["grid"]) == pkg._plain_plan(stage, r200, rows), plan
    got = _domain_step(d)
    torch.cuda.synchronize(r200)
    assert twins_run == [_twin("fwd")]
    _check_outputs(stage, got, want)


def test_the_measured_first_rows_route_the_twin(twins_run, r200) -> None:
    """With the shipped first rows per SM: the forward's twin from 112 rows per SM (homed) and 128 (anchor); the
    backward runs its ``"off"`` program (reason ``arch``) also with every operand in the anchor's row frame; the
    bits of ``"off"``."""
    sms = torch.cuda.get_device_properties(r200).multi_processor_count
    homed, anchor = 112 * sms, 128 * sms
    rows = anchor
    t = _inputs(rows, seed=1, device=torch.device("cuda", r200))
    want = _off(t)
    d = _place(t, r200, 0, rows, True)

    def fplan(count, out):
        return forward_plan(d.first[:count], None, d.weight[:count], limit=L, out=None if out is None else out[:count])

    assert (fplan(anchor, None)["reason"], fplan(anchor, None)["class"]) == ("placed", "anchor")
    assert (fplan(anchor - 1, None)["reason"], fplan(anchor - 1, None)["class"]) == ("size", "anchor")
    assert (fplan(homed, d.h)["reason"], fplan(homed, d.h)["class"]) == ("placed", "homed")
    assert (fplan(homed - 1, d.h)["reason"], fplan(homed - 1, d.h)["class"]) == ("size", None)
    plan = backward_plan(d.dh, d.first, None, d.weight, limit=L, grad_gate_up_or_gate=d.d_first)
    assert (plan["route"], plan["reason"], plan["class"]) == ("plain", "arch", None), plan
    got = _domain_step(d)
    torch.cuda.synchronize(r200)
    assert twins_run == [_twin("fwd")]
    _check_outputs("measured first rows", got, want)


def test_below_the_anchor_rows_a_new_h_reads_no_registry(twins_run, monkeypatch, r200) -> None:
    """With the shipped first rows per SM, at the ``homed`` first row count (below the ``anchor`` one): the autograd
    entry with ``"domain"`` on leaves in helper memory (its ``h`` is new: class ``anchor``), plain and under a
    non-reentrant checkpoint with the down projection, and a forward step without ``out`` read neither the registry
    nor the SM table, take no claim counters and run the ``"off"`` program with the ``"off"`` bits; a forward step at
    the same row count whose ``out`` lies in the anchor's row frame (class ``homed``) runs its twin, as planned."""
    sms = torch.cuda.get_device_properties(r200).multi_processor_count
    rows = 112 * sms
    dev = torch.device("cuda", r200)
    t = _inputs(rows, seed=17, device=dev)
    want = _off(t)
    d = _place(t, r200, 0, rows, True)
    w_down = (torch.randn(512, I, device=dev) / I**0.5).bfloat16()
    reads: collections.Counter = collections.Counter()

    def counted(name, function):
        def wrapper(*args, **kwargs):
            reads[name] += 1
            return function(*args, **kwargs)

        return wrapper

    for name in ("sm_table", "ranges", "slot_of", "place", "claim"):
        monkeypatch.setattr(memory, name, counted(name, getattr(memory, name)))
    plan = forward_plan(d.first, None, d.weight, limit=L)
    assert (plan["route"], plan["reason"], plan["class"]) == ("plain", "size", "anchor"), plan
    plan = forward_plan(d.first, None, d.weight, limit=L, out=d.h)
    assert (plan["route"], plan["reason"], plan["class"]) == ("domain", "placed", "homed"), plan

    def region(first, weight, locality):
        return torch.nn.functional.linear(clamped_swiglu(first, None, weight, limit=L, locality=locality), w_down)

    def run(locality, checkpoint):
        first = domain_empty((rows, 2 * I), device=dev) if locality == "domain" else torch.empty_like(t.first)
        first.copy_(t.first)
        first.requires_grad_(True)
        weight = t.weight.clone().requires_grad_(True)
        torch.cuda.synchronize(r200)
        reads.clear()
        twins_run.clear()
        if checkpoint:
            y = torch.utils.checkpoint.checkpoint(region, first, weight, locality, use_reentrant=False)
        else:
            y = region(first, weight, locality)
        y.backward(torch.ones_like(y))
        torch.cuda.synchronize(r200)
        return y.detach(), first.grad, weight.grad

    for checkpoint in (False, True):
        off = run("off", checkpoint)
        got = run("domain", checkpoint)
        assert not reads and not twins_run, (checkpoint, dict(reads), twins_run)
        for name, a, b in zip(("y", "d_gate_up", "d_weight"), got, off):
            _assert_same(f"checkpoint {checkpoint}: {name}", a, b)
    reads.clear()
    twins_run.clear()
    h = forward(d.first, None, d.weight, limit=L, locality="domain")
    torch.cuda.synchronize(r200)
    assert not reads and not twins_run, (dict(reads), twins_run)
    _assert_same("a new h", h, want[0])
    h = forward(d.first, None, d.weight, limit=L, out=d.h, locality="domain")
    torch.cuda.synchronize(r200)
    assert reads["slot_of"] and reads["claim"] == 1, dict(reads)
    assert twins_run == [_twin("fwd")], twins_run
    _assert_same("the homed out", h, want[0])


def _but(d: SimpleNamespace, **changes) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(d), **changes})


def test_each_step_runs_its_plan(low_firsts, twins_run, monkeypatch, r200) -> None:
    """Each step runs what its plan names, a common call (plain contiguous operands: checked and routed in the forward
    step's own frame) as any other call (the full checks): the forward's twin, from the plan's first slot and die on
    the plan's grid, exactly when the plan routes the call to it, else the ``"off"`` program, with the bits of
    ``"off"``; the backward runs its ``"off"`` program (reason ``arch``) for every call.  The calls meet both classes
    and every reason of the route, the class reasons with the forward's class ``anchor`` given no first row count."""
    grids: list = []
    general: list = []
    twin_grid, forward_domain = pkg._twin_grid, pkg._forward_domain

    def grid_of(twin, rows, place):
        grid = twin_grid(twin, rows, place)
        grids.append((twin.name, place, grid))
        return grid

    def forward_general(*args, **kwargs):
        general.append("fwd")
        return forward_domain(*args, **kwargs)

    monkeypatch.setattr(pkg, "_twin_grid", grid_of)
    monkeypatch.setattr(pkg, "_forward_domain", forward_general)
    seen: collections.Counter = collections.Counter()

    def check(what: str, f: SimpleNamespace, b: SimpleNamespace, want, common: bool = True) -> tuple:
        fplan = forward_plan(f.first, f.second, f.weight, limit=L, out=f.h)
        bplan = backward_plan(
            b.dh, b.first, b.second, b.weight, limit=L, grad_gate_up_or_gate=b.d_first, grad_up=b.d_second
        )
        grids.clear()
        general.clear()
        twins_run.clear()
        h = forward(f.first, f.second, f.weight, limit=L, out=f.h, locality="domain")
        grads = backward(
            b.dh,
            b.first,
            b.second,
            b.weight,
            limit=L,
            grad_gate_up_or_gate=b.d_first,
            grad_up=b.d_second,
            locality="domain",
        )
        torch.cuda.synchronize(r200)
        assert (bplan["route"], bplan["reason"], bplan["class"]) == ("plain", "arch", None), (what, bplan)
        assert not cake_jit.MODULES[bplan["program"]]["role"].endswith(cake_jit.DOMAIN), (what, bplan)
        twins = [fplan] if fplan["route"] == "domain" else []
        assert twins_run == [plan["program"] for plan in twins], (what, fplan["reason"])
        assert grids == [(p["program"], (p["first_slot"], p["first_die"]), p["grid"][0]) for p in twins], what
        assert general == ([] if common else ["fwd"]), what
        seen.update((fplan["reason"], bplan["reason"]))
        seen.update(f"{plan['stage']} {plan['class']}" for plan in twins)
        if want is not None:
            _check_outputs(what, (h, *grads), want)
        return fplan["reason"], fplan["class"]

    def shifted(value: torch.Tensor) -> torch.Tensor:
        """``value`` in helper memory 16 bytes after a row start: contiguous and aligned, off every slot."""
        rows_, cols = value.shape
        whole = domain_empty((rows_ + 1, cols), device=r200)
        view = whole.view(-1)[8 : 8 + rows_ * cols].view(rows_, cols)
        view.copy_(value)
        return view

    dev = torch.device("cuda", r200)
    rows, small = 1237, low_firsts - 1
    for merged in (True, False):
        m = f"merged={merged}"
        t = _inputs(rows, seed=31 + merged, device=dev, merged=merged)
        want = _off(t)
        d = _place(t, r200, 200, rows, True)
        assert check(f"homed {m}", d, d, want) == ("placed", "homed")
        new = _place(t, r200, 200, rows, False)
        assert check(f"new outputs {m}", new, new, want) == ("placed", "anchor")
        ordinary = _but(
            d,
            h=torch.empty_like(t.dh),
            d_first=torch.empty_like(t.first),
            d_second=None if merged else torch.empty_like(t.second),
        )
        assert check(f"ordinary buffers {m}", ordinary, ordinary, want) == ("placed", "anchor")
        later = _place(t, r200, 201, rows, True)
        f = d if merged else _but(d, second=later.second)
        assert check(f"a row later {m}", f, _but(d, dh=later.dh), want) == ("placed", "homed" if merged else "anchor")
        first = _but(d, first=t.first, second=t.second)
        assert check(f"ordinary first {m}", first, first, want) == ("input_memory", None)
        off_slot = _but(d, first=shifted(t.first))
        assert check(f"off a slot {m}", off_slot, off_slot, want) == ("input_layout", None)
        with monkeypatch.context() as patch:
            patch.setattr(memory, "_TABLES", {})
            assert check(f"no SM table {m}", d, d, want) == ("sm_table", None)
        part = _but(
            d,
            **{
                key: None if value is None else value[:small]
                for key, value in vars(d).items()
                if isinstance(value, torch.Tensor) or value is None
            },
        )
        cut = tuple(None if value is None else value[:small] for value in want)
        assert check(f"{small} rows {m}", part, part, cut) == ("size", None)
        tu = _inputs(rows, seed=41 + merged, device=dev, merged=merged, weighted=False)
        du = _place(tu, r200, 200, rows, True)
        assert check(f"unweighted {m}", du, du, _off(tu)) == ("placed", "homed")
        # not a common call: the two halves of a merged helper tensor as the two tensors (the up half off a slot),
        # or a merged input with a padded row
        if merged:
            padded = torch.empty(rows, 2 * I + 64, dtype=torch.bfloat16, device=dev)[:, : 2 * I]
            padded.copy_(t.first)
            g = _but(new, first=padded)
        else:
            whole = domain_empty((rows, 2 * I), device=r200)
            whole.copy_(torch.cat((t.first, t.second), dim=1))
            g = _but(new, first=whole[:, :I], second=whole[:, I:])
        assert check(f"general {m}", g, g, want, common=False) == (
            ("input_memory", None) if merged else ("placed", "anchor")
        )
        # a caller's buffer that meets an input (the bits of such a call are not compared)
        o = _place(t, r200, 200, rows, True)
        out = o.first.view(-1)[: rows * I].view(rows, I) if merged else o.second
        assert check(f"overlap {m}", _but(o, h=out), _but(o, d_first=o.first), None) == ("overlap", None)
        # the forward's class anchor without a first row count: its first operand outside the frame names the reason
        with monkeypatch.context() as patch:
            patch.setitem(cake_jit.DOMAIN_FIRST_ROWS_PER_SM["fwd"], ARCH, {"homed": 1, "anchor": None})
            _clear_twin_caches()
            assert check(f"no anchor rows: homed {m}", d, d, want) == ("placed", "homed")
            assert check(f"no anchor rows: new outputs {m}", new, new, want) == ("no_out", "anchor")
            assert check(f"no anchor rows: ordinary buffers {m}", ordinary, ordinary, want) == (
                "out_memory",
                "anchor",
            )
            if not merged:
                assert check(f"no anchor rows: a row later {m}", f, d, want) == ("input_phase", "anchor")
                assert check(f"no anchor rows: general {m}", g, g, want, common=False) == ("input_phase", "anchor")
        _clear_twin_caches()
    assert set(REASONS) <= set(seen) and {"fwd homed", "fwd anchor"} <= set(seen), seen


def test_counters_are_zero_after_calls_and_a_raising_launch_drops_its_workspace(low_firsts, monkeypatch, r200) -> None:
    rows = 777
    t = _inputs(rows, seed=3, device=torch.device("cuda", r200))
    d = _place(t, r200, 0, rows, True)
    stream = memory._current_stream_handle(r200)
    for _ in range(3):
        _domain_step(d)
    torch.cuda.synchronize(r200)
    words = memory.stream_workspace(r200, stream)
    assert words is not None and int(words.abs().sum()) == 0
    twin = pkg._twin("fwd", r200)
    monkeypatch.setattr(twin, "entry", lambda *args: (_ for _ in ()).throw(RuntimeError("launch refused")))
    with pytest.raises(RuntimeError, match="refused"):
        forward(d.first, None, d.weight, limit=L, out=d.h, locality="domain")
    assert memory.stream_workspace(r200, stream) is None
    monkeypatch.setattr(twin, "entry", None)
    want = _off(t)
    _check_outputs("after a raising launch", _domain_step(d), want)


def test_a_raising_twin_launch_of_the_general_and_held_paths_drops_its_workspace(low_firsts, monkeypatch, r200) -> None:
    """A twin launch that raises through ``_launch_twin`` (a call the forward step checks in full, and the held launch
    of a checkpoint's recomputation) drops its stream's workspace; the next call takes a new workspace, leaves its
    counters zero and gives the ``"off"`` bits."""
    rows = 777
    dev = torch.device("cuda", r200)
    t = _inputs(rows, seed=7, device=dev)
    want = _off(t)
    whole = domain_empty((rows, 2 * I), device=r200)
    whole.copy_(t.first)
    # not a common call: the two halves of a merged helper tensor as the two tensors
    gate, up = whole[:, :I], whole[:, I:]
    plan = forward_plan(gate, up, t.weight, limit=L)
    assert (plan["route"], plan["class"]) == ("domain", "anchor"), plan
    general: list = []
    forward_domain = pkg._forward_domain

    def forward_general(*args, **kwargs):
        general.append(len(args))
        return forward_domain(*args, **kwargs)

    monkeypatch.setattr(pkg, "_forward_domain", forward_general)
    stream = memory._current_stream_handle(r200)
    twin = pkg._twin("fwd", r200)
    for path in ("general", "held"):
        _assert_same(f"{path}: h", forward(gate, up, t.weight, limit=L, locality="domain"), want[0])
        torch.cuda.synchronize(dev)
        assert memory.stream_workspace(r200, stream) is not None
        monkeypatch.setattr(twin, "entry", lambda *args: (_ for _ in ()).throw(RuntimeError("launch refused")))
        general.clear()
        with pytest.raises(RuntimeError, match="refused"):
            if path == "general":
                forward(gate, up, t.weight, limit=L, locality="domain")
            else:
                pkg._held_domain(gate, up, t.weight, L)
        assert general, path
        assert memory.stream_workspace(r200, stream) is None, path
        monkeypatch.setattr(twin, "entry", None)
        if path == "general":
            h = forward(gate, up, t.weight, limit=L, locality="domain")
        else:
            h = pkg._held_domain(gate, up, t.weight, L)
        torch.cuda.synchronize(dev)
        words = memory.stream_workspace(r200, stream)
        assert words is not None and int(words.abs().sum()) == 0, path
        _assert_same(f"{path}: h after a raising launch", h, want[0])


def test_two_streams_take_two_workspaces(low_firsts, r200) -> None:
    rows = 900
    dev = torch.device("cuda", r200)
    a = _inputs(rows, seed=11, device=dev)
    b = _inputs(rows, seed=12, device=dev, merged=False)
    want_a, want_b = _off(a), _off(b)
    da, db = _place(a, r200, 0, rows, True), _place(b, r200, 0, rows, True)
    streams = [torch.cuda.Stream(dev), torch.cuda.Stream(dev)]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream(dev))
    got = []
    for s, d in zip(streams, (da, db)):
        with torch.cuda.stream(s):
            got.append(_domain_step(d))
    for s in streams:
        torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    keys = [memory.stream_workspace(r200, s.cuda_stream) for s in streams]
    assert keys[0] is not None and keys[1] is not None and keys[0].data_ptr() != keys[1].data_ptr()
    _check_outputs("stream a", got[0], want_a)
    _check_outputs("stream b", got[1], want_b)


def test_cuda_graph_replays_match_eager(low_firsts, twins_run, r200) -> None:
    """A captured domain step (after one eager call) replays with the eager bits, on new inputs copied into the
    captured buffers; the capture takes a graph workspace."""
    rows = 1100
    dev = torch.device("cuda", r200)
    a = _inputs(rows, seed=21, device=dev)
    d = _place(a, r200, 0, rows, True)
    _domain_step(d)
    torch.cuda.synchronize(dev)
    left = memory.graph_workspaces_left(r200)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _domain_step(d)
    assert memory.graph_workspaces_left(r200) == left - 1
    for seed in (22, 23):
        b = _inputs(rows, seed=seed, device=dev)
        d.first.copy_(b.first)
        d.dh.copy_(b.dh)
        d.weight.copy_(b.weight)
        graph.replay()
        torch.cuda.synchronize(dev)
        _check_outputs(f"replay {seed}", (d.h, d.d_first, None, None), (*_off(b)[:2], None, None))


def test_autograd_and_checkpoint_give_the_off_bits(low_firsts, twins_run, r200) -> None:
    """The autograd entry with ``locality="domain"`` on leaves in helper memory, plain and under a non-reentrant
    checkpoint with the down projection, gives the bits of ``"off"``; with lowered first rows its forward runs its
    twin (class ``anchor``) and its backward the ``"off"`` program."""
    rows = 1300
    dev = torch.device("cuda", r200)
    t = _inputs(rows, seed=31, device=dev)
    w_down = (torch.randn(512, I, device=dev) / I**0.5).bfloat16()

    def region(first, weight, locality):
        return torch.nn.functional.linear(clamped_swiglu(first, None, weight, limit=L, locality=locality), w_down)

    def run(locality, checkpoint):
        first = domain_empty((rows, 2 * I), device=dev) if locality == "domain" else torch.empty_like(t.first)
        first.copy_(t.first)
        first.requires_grad_(True)
        weight = t.weight.clone().requires_grad_(True)
        if checkpoint:
            y = torch.utils.checkpoint.checkpoint(region, first, weight, locality, use_reentrant=False)
        else:
            y = region(first, weight, locality)
        y.backward(torch.ones_like(y))
        return y.detach(), first.grad, weight.grad

    for checkpoint in (False, True):
        want = run("off", checkpoint)
        twins_run.clear()
        got = run("domain", checkpoint)
        assert twins_run and set(twins_run) == {_twin("fwd")}, twins_run
        for name, a, b in zip(("y", "d_gate_up", "d_weight"), got, want):
            _assert_same(f"checkpoint {checkpoint}: {name}", a, b)


def test_offloaded_saved_tensors_give_the_off_gradients(low_firsts, r200) -> None:
    """Autograd with ``"domain"`` under a saved-tensors hook that copies each saved tensor to the host on a side stream
    (``record_stream`` on the original), with a side-stream consumer of the gradients, over several steps: gradients
    bitwise equal to ``"off"``."""
    rows = 1000
    dev = torch.device("cuda", r200)
    side = torch.cuda.Stream(dev)

    def pack(x):
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
            host.copy_(x, non_blocking=True)
        x.record_stream(side)
        return (host, x.device)

    def unpack(saved):
        host, where = saved
        torch.cuda.current_stream(dev).wait_stream(side)
        return host.to(where, non_blocking=True)

    for step in range(3):
        t = _inputs(rows, seed=40 + step, device=dev)
        want = _off(t)
        first = domain_empty((rows, 2 * I), device=dev)
        first.copy_(t.first)
        first.requires_grad_(True)
        weight = t.weight.clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            h = clamped_swiglu(first, None, weight, limit=L, locality="domain")
        h.backward(t.dh)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            total = first.grad.float().sum()
            first.grad.record_stream(side)
            expected = want[1].float().sum()
        torch.cuda.synchronize(dev)
        _assert_same(f"step {step}: a side-stream sum of d_gate_up", total, expected)
        _assert_same(f"step {step}: h", h.detach(), want[0])
        _assert_same(f"step {step}: d_gate_up", first.grad, want[1])
        _assert_same(f"step {step}: d_weight", weight.grad, want[3])


def test_steady_state_domain_calls_do_not_synchronize(low_firsts, r200) -> None:
    rows = 800
    dev = torch.device("cuda", r200)
    t = _inputs(rows, seed=51, device=dev)
    d = _place(t, r200, 0, rows, True)
    want = _off(t)
    _domain_step(d)
    torch.cuda.synchronize(dev)
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3):
            got = _domain_step(d)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize(dev)
    _check_outputs("steady state", got, want)


def test_the_sm_table_names_every_sm_and_both_dies(r200) -> None:
    """The probed table holds a word for each SM id below the device's SM count, each die 0 or 1, the dies' counts
    adding to the SM count and each SM's rank below its die's count (the table's field layout)."""
    domain_empty((1, I), device=r200)
    table = memory.sm_table(r200)
    assert table is not None and table.dtype == torch.int32 and tuple(table.shape) == (256,)
    sms = torch.cuda.get_device_properties(r200).multi_processor_count
    locality = cake_package(PACKAGE, "cake_locality")
    bits = (1 << locality.ENTRY_BITS) - 1
    dies = collections.Counter()
    counts = {}
    for sm in range(sms):
        word = int(table[sm])
        die = (word >> locality.ENTRY_DOMAIN_SHIFT) & ((1 << locality.ENTRY_DOMAIN_BITS) - 1)
        rank = (word >> locality.ENTRY_RANK_SHIFT) & bits
        count = (word >> locality.ENTRY_COUNT_SHIFT) & bits
        assert die in (0, 1) and rank < count, (sm, word)
        dies[die] += 1
        assert counts.setdefault(die, count) == count
    assert dies[0] + dies[1] == sms and dict(dies) == counts
