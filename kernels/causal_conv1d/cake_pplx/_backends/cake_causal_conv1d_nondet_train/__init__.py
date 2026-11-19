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

Non-deterministic backend of the Cake packed width-4 depthwise causal
convolution + SiLU training forward / backward (SM100 / SM103 / SM107), one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  It
imports nothing from other packages of this repository.  The stable entry points and
the user-facing documentation live in the non-deterministic package's
facades :mod:`cake_pplx.cake_causal_conv1d_nondet_train` and
:mod:`cake_pplx.cake_causal_conv1d_nondet_train_kcp` (the KCP entry points of
the :mod:`.kcp` module here).

Numerical contract against the deterministic package: the forward runs this
package's copies of the det forward programs, so ``y`` is the det ``y`` bit
for bit.  The backward is ONE ordinary launch of a program built from the det
backward body of the same route and channel rule, or, at the rows of
``cake_jit.FALLBACK_ROWS``, from the route's fallback body, or, at the rows
of ``cake_jit.SM_WIDE_ROWS``, from the det SM-wide row-group body of the
route (one CTA per SM for the whole launch): every per-thread and per-CTA
FP32 step is that det program's and ``dx`` is that det program's ``dx`` bit
for bit; only the cross-CTA reduction of ``dweight`` / ``dbias`` differs.  Each
CTA adds its CTA row into an accumulator with unordered global reductions in
plain FP32 words, as the det reduce program sums on every route (the
``bwd_fp32g`` and ``bwd_bias_fp32g`` routes, the
``silu_grad_preactivation="fp32"`` ones, spread the adds over sixteen
copies of the accumulator, added in copy order at the end, and the SM-wide
program of the longest sm_107a rows over two) -- and the last CTA of each
channel tile stores the gradients and zeroes its accumulators.
The order of that cross-CTA sum depends on the CTAs' timing, so ``dweight`` /
``dbias`` are not bitwise reproducible run to run; that is the only relaxed
property (the accumulation dtypes, plain FP32 words, and rounding boundaries
are the det package's).  No reduce program, no host synchronization, no cooperative
launch, no grid barrier.

Numerical mode: the package is chosen by its module name and no entry point
takes a mode keyword.  This package has no deterministic programs (they are the
deterministic package's, ``cake_pplx.cake_causal_conv1d_det_train[_kcp]``;
nothing here routes to them).  With ``torch.use_deterministic_algorithms(True)``
the entry points that launch the unordered reduction (the backward, hence the
autograd backward) raise PyTorch's alert for an operation without a
deterministic implementation (a ``RuntimeError``, or a ``UserWarning`` under
``warn_only=True``, after which they run), reading the setting when the
backward runs, and never fall back.  The forward is fixed-order (the det
programs' bits) and does not alert.

Host design (the knobs are keywords; no environment variable):

- Taps ``weight[C, 4]`` are FP32 or BF16; an optional per-channel ``bias[C]``
  is FP32 or BF16, independently of the taps.  ``dweight`` / ``dbias`` default
  to the parameter's dtype and may be preallocated as FP32 or BF16 (the
  ``flags`` bit mask tells the kernels which).
- ``silu_grad_preactivation``: ``"bf16"`` (default) or ``"fp32"`` (SiLU' at
  the unrounded FP32 preactivation; sixteen FP32 accumulator copies).
- ``dweight_sum``: ``"fp32"`` (default: one plain FP32 word per ``dweight`` /
  ``dbias`` sum) or ``"fp32_pair"`` (opt-in, with ``silu_grad_preactivation=
  "fp32"`` only: the pair routes ``cake_jit.PAIR_ROUTES``, every accumulator
  word an FP32 (hi, lo) pair -- the CTA's fold Kahan-compensated, each add's
  rounding error kept in lo, one final rounding; FP32 operations only).  The
  default never selects a pair route; ``"fp32_pair"`` without the FP32 SiLU'
  policy is a ``ValueError``.
- A call's arguments select its route (``cake_jit.forward_route`` /
  ``backward_route``) and the channel count the route's program, the det
  package's rules.
- The backward workspace holds the accumulators ``acc`` and the per channel
  tile ticket counters ``work`` (``cake_jit.accumulator_layout``): zero before
  the first launch and left zero by every launch.  A caller-owned buffer
  (``workspace=``, from
  :func:`cake_causal_conv1d_nondet_train_backward_workspace`, which returns it
  zeroed) serves every later backward call of one stream;
  when omitted the package keeps one zeroed buffer per (device, current
  stream), grown as needed and never zeroed again.  Inside a CUDA Graph
  capture with a nonzero CUDA capture id (``cudaStreamGetCaptureInfo``) an
  omitted workspace is a buffer of that capture and stream that the package
  zeroed before the capture, on an eager call, and saw complete: the graph
  records no fill node for it, and the capture's later calls on that stream
  reuse it (a call that needs more bytes takes a larger one).  No eager call
  and no other capture shares it.  The graph holds a reference to the buffer
  (a CUDA user object); after the graph and its instantiations are destroyed
  and their launches have completed, the buffer, zero again, returns to the
  package at its next refill (below).  The replays of one graph must not
  overlap (they share the graph's buffers, as they share its other memory).
  Only eager calls that omit ``workspace=`` and take counters zero these
  buffers, the forward's and the backward's separately: per device and kind up
  to ``_POOL_DEPTH`` buffers of the largest need of such a call so far,
  refilled by the next such call after a capture took one or when a call needs
  more (a forward at a row of ``cake_jit.FALLBACK_ROWS`` or
  ``cake_jit.DIRECT_ROWS`` takes no counters and zeroes none).  A captured call
  finds none ready when no such eager call of its kind ran before the capture,
  when the captures since the last one took them all, when it needs more than
  that eager need, or when their zero fills have not completed yet (a
  ``CUDAGraph.capture_begin`` right after such an eager call, with no
  synchronization; ``torch.cuda.graph`` synchronizes first); it then zeroes a
  buffer of its own (one fill node per
  call; so does every captured call when the capture id is 0, also when
  ``cuda.bindings`` is not importable), which the package keeps no reference
  to after the call.  Inside a capture the forward's omitted workspace (int32
  counters) follows the same rules with buffers of its own kind.  The package
  allocates the per-stream buffers and the pre-zeroed ones from a memory pool
  of its own per device (``torch.cuda.MemPool``; without it the package keeps
  no pre-zeroed buffers), so a CUDA Graph tree's warm-up does not take them
  into the tree's private pool.  Memory the package frees stays cached in that
  pool (a replaced per-stream buffer, a pool buffer it drops) and is never
  returned to the caching allocator's other pools, so the pool keeps the
  largest footprint these buffers reached at once.  Per device and kind the
  live pre-zeroed buffers are at most ``_POOL_DEPTH`` idle ones of the largest
  eager need, those live graphs hold, and those that graphs released since the
  device's last refill.  A kind turned off (a refill that failed other than
  out of memory, or a device without its memory pool) refills and hands out
  no more: its idle buffers stay allocated for the life of the process, and a
  buffer its graphs held returns, once released, only at a refill of the
  device's other kind (while that kind still refills), which keeps it idle
  (when fewer than ``_POOL_DEPTH`` are idle and it holds the kind's last pool
  size) or drops it into the package's memory pool.
  Its size depends on ``C`` (and the route), not on ``S``.  Two backward
  launches that may run at once (two streams, two graphs replayed
  concurrently) need two buffers: never share one between them.
- ``bwd_grid_mult`` (keyword-only, default 1, one of ``BWD_GRID_MULTS``; the
  deterministic package's option and validation): an opt-in finer static
  partition of the backward's token axis.  The backward launches
  ``bwd_grid_mult`` times its default untrimmed grid; the programs are the
  same, as every program reads its partition from the launch (``gridDim.x``:
  the CTAs of a channel tile split its tokens evenly, and its ticket counter
  wraps after that many CTAs), so ``dx`` keeps its bits and more CTAs add into
  the same accumulators.  The workspace does not grow with it (one accumulator
  set and one counter per channel tile, whatever the grid).
- Backward grid beside a concurrent kernel: on the
  ``bwd`` and ``bwd_fp32g`` routes at ``bwd_grid_mult`` 1, a launch of exactly
  ``ctas_per_sm x SM count`` CTAs whose last channel-tile rank holds at least
  ``ctas_per_sm`` CTAs launches ``ctas_per_sm`` fewer (:func:`_one_sm_trim`),
  so a kernel on another stream that keeps one SM's shared memory does not
  push the top CTAs, each with a whole token stream, into a second wave.
  ``dx`` keeps its bits (the partition is read from ``gridDim.x``, as above).
- The forward plans its channel tiles by the det package's forward rule
  (its route, ``S`` and the architecture: ``cake_jit.channel_split``) from
  the row's token floor (``cake_jit.forward_narrow_min_s``: at C = 520 to
  568 the row's ``FORWARD_NARROW_ROW_FLOORS`` floor on the architecture;
  else the architecture's ``FORWARD_NARROW_MIN_S`` floor, or the
  ``FORWARD_NARROW_ROW_MIN_S`` floor of a route's rows of a tap dtype from a
  channel-vector count upward: on sm_103a for the ``fwd_bias`` route at C =
  712 to 768 20480 with FP32 taps and 12288 with BF16 taps, which keep the
  single tile in the ``FORWARD_SINGLE_TILE_ROWS`` token ranges), so C = 520
  to 768 run the det forward's narrow tiles from that floor; below it (and
  on an architecture without one) they keep 0.2.0's single tile.  The
  forward workspace (int32 token-chunk counters) follows the det package's
  rules and size, which cover every plan; at C = 520 to 568 it is larger
  than 0.2.3's (6 to 12 words instead of 2), so a buffer sized by 0.2.3 for
  those rows is refused (``ValueError``) from the row's token floor.
- Below the forward token ceilings at C = 576 to 768 the forward runs two
  other det programs of its route: the static program in the forward ranges
  of ``cake_jit.FALLBACK_ROWS`` (each CTA a fixed token range of its channel
  tile, on the route's plan above) and the direct program in the ranges of
  ``cake_jit.DIRECT_ROWS`` (each CTA the next token blocks of its channel
  tile, on the route's tiles of ``cake_jit.DIRECT_PLANS``, at most one CTA per
  token block).  Neither takes counters: a passed ``workspace`` is checked
  and not passed to the program, and the workspace size is unchanged.
  ``y`` is the same bits on every forward program and plan.
- Programmatic dependent launch: a program whose record's
  ``launch.use_pdl`` is ``"runtime"`` (every single-rank forward and
  backward program but those of the pair routes) takes the ``enable_pdl``
  launch scalar, ``cake_jit.pdl_enabled`` of the architecture and the mode:
  ``PDL_EAGER`` outside a CUDA Graph capture, ``PDL_GRAPH`` inside one;
  but a backward call at a row of ``cake_jit.PDL_OFF_ROWS`` (by its route,
  ``C`` and ``S``: the ``bwd`` route at C = 1000 to 1008 from S = 187755 and
  at C = 1016 from S = 155648 on sm_100a, at C = 1000 to 1008 from
  S = 131073 and at C = 1016 from S = 125171 on sm_103a) passes 0 in both
  modes.  The forward passes the tables' value at every row.
  With 1 the launch may begin while the stream's preceding kernel finishes;
  every program waits for that kernel (``griddepcontrol.wait``) before it
  reads or writes global memory, so the outputs do not depend on it.  The
  KCP forward launches its single-rank program with 0.
- Host plan: the first call of a row (architecture, route, ``C``, device,
  ``bwd_grid_mult``; the forward: the taps' dtype) selects its program,
  splits ``C`` into channel tiles, reads the row's ``TILE_OVERRIDES`` or
  ``NARROW_TILES`` entry (the forward: its ``forward_narrow_min_s`` and
  ``FORWARD_WIDTH_FLOORS`` token floors and its single-tile range)
  and sizes the grid (``cake_jit.plan_row``); later calls of the row only
  compare ``S`` with the entry's token ranges (and size the grid where
  ``launch_grid`` trims it by ``S``), so they launch the values a per-call
  plan gives.  A row a ``cake_jit.SM_WIDE_ROWS`` or ``cake_jit.FALLBACK_ROWS``
  key covers is planned for every side of its token ranges at its first call
  (``cake_jit.row_sides``: an SM-wide segment's program, the route's fallback
  program inside the fallback range, the rules' program outside them); a
  later call picks its side with a few compares of ``S``.  A row whose
  program selection reads ``S`` through a token floor of the selection rules
  plans every call (on that side).
  :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` drops the rows.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import sys
import threading
import warnings
import weakref
from typing import Callable, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    CHANNEL_MULTIPLE,
    FLAG_B_BF16,
    FLAG_DB_BF16,
    FLAG_DW_BF16,
    FLAG_W_BF16,
    INT32_MAX,
    REQUIRED_KINDS,
    WIDTH,
    accumulator_layout,
    arch_for_capability,
    backward_route,
    backward_workspace_bytes,
    counter_words,
    forward_route,
    forward_workspace_words,
    launch_grid,
    pdl_enabled,
    plan,
    route_applies,
    run_module,
    selected_modules,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The dtypes of the taps, the bias and their gradients (FP32 or BF16, any combination).
PARAMETER_DTYPES = (torch.float32, torch.bfloat16)
# The two SiLU'-evaluation policies of the weight gradient (``silu_grad_preactivation``).
SILU_GRAD_PREACTIVATIONS = ("bf16", "fp32")
# The sum forms of ``dweight`` / ``dbias`` (``dweight_sum``): one FP32 word per sum (the default) or an FP32 (hi, lo)
# pair (the opt-in pair routes).
DWEIGHT_SUMS: tuple[str, ...] = tuple(_loader.DWEIGHT_SUMS)
# The backward grid multiples a call may choose (``bwd_grid_mult``): 1 is the default grid; 2, 3 and 4 launch that
# many times its untrimmed CTA count.  The deterministic package's ``BWD_GRID_MULTS`` (a test checks they agree).
BWD_GRID_MULTS: tuple[int, ...] = (1, 2, 3, 4)
# The forward workspaces of the calls that pass none: (device index, CUDA stream
# handle) -> a zeroed int32 buffer, grown as needed.  The launches of one stream
# run in order and each leaves its counters zero, so the stream's buffer serves
# every later forward call on it.
_FORWARD_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
# The backward workspaces of the calls that pass none (single-rank and KCP):
# (device index, CUDA stream handle) -> a zeroed uint8 buffer of accumulators and
# ticket counters, grown as needed.  Every launch leaves its accumulators and
# counters zero, so the stream's buffer serves every later backward call on it.
_BACKWARD_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
# The backward workspaces of the calls that pass none inside a CUDA Graph capture:
# (device index, capture stream handle) -> (CUDA capture id, pre-zeroed uint8 buffer).
# The capture's first such call on the stream takes a pre-zeroed buffer of the
# package (``_ZEROED_POOL``: no fill node); every launch leaves it zero and the
# stream orders the launches, so the later calls of the same capture and
# stream reuse it.  A call that needs more bytes takes a larger one that
# replaces the entry; another capture (another id) replaces the entry with its
# own.  An entry is dropped when its buffer returns to the package.  A captured
# call that finds no pre-zeroed buffer ready, or whose capture id is 0 (no
# ``cuda.bindings``), zeroes a buffer of its own (a fill node) and adds no entry:
# that buffer is memory of the capture's pool, and the package keeps no
# reference to it after the call.
_CAPTURE_WORKSPACES: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
# The forward's twin of ``_CAPTURE_WORKSPACES`` (int32 token-chunk counters): only
# pre-zeroed buffers enter it; a captured forward call that zeroes a buffer of its
# own (a fill node) keeps it to itself.
_CAPTURE_FORWARD_WORKSPACES: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
# The pre-zeroed capture buffers.  An eager call that omits ``workspace=`` keeps, per
# (device index, kind) -- kind ``"fwd"`` (int32 counter words) or ``"bwd"`` (uint8
# accumulator and counter bytes) -- up to ``_POOL_DEPTH`` idle zeroed buffers of
# ``_POOL_SIZE`` elements (``_ZEROED_POOL``), the size set by the device's last
# refill of that kind that did not run out of memory (a turned-off kind's idle
# buffers keep their earlier size).  Each idle buffer carries (``_PoolBuffer``) a
# slot of a pinned host flag array that a device-to-host copy, queued on the eager
# call's stream behind the buffer's zero fill, sets to 1, and a CUDA user object,
# made on the eager call (outside any capture), whose destructor posts the buffer's
# POSIX semaphore.  A capture call takes an idle buffer whose flag reads 1 (a host load:
# its zero fill has completed) and moves the user object's reference into the graph
# being captured (``cudaGraphRetainUserObject``, the only CUDA call besides
# ``cudaStreamGetCaptureInfo`` a capture makes here): the graph and each of its
# instantiations then hold it (``_GRAPH_HELD``) until they are destroyed and their
# launches have completed, when CUDA posts the semaphore; a later eager call
# (``_pool_refill``) returns it, zero again (every launch leaves its bytes zero),
# to the pool with a new user object, or frees it.  Pool buffers come from the
# package's own memory pool (``_mem_pool``, a ``torch.cuda.MemPool`` per device) on
# the eager call's stream, never from a capture; idle buffers take at most
# ``_POOL_DEPTH`` x ``_POOL_SIZE`` elements per (device, kind); a taken buffer lives as
# long as the graphs that hold it and then until the device's next refill returns it.
# Without ``torch.cuda.MemPool`` or ``cuda.bindings`` the device has no pool: every
# captured call zeroes a buffer of its own (a fill node).  A failure that turns a
# kind off (``_pool_off``) frees nothing itself: the kind's idle buffers and the user
# objects, semaphores and flag slots of an interrupted refill stay allocated for the
# life of the process.  The buffers its graphs hold then wait, once released, in
# ``_GRAPH_HELD`` (or in ``uo.released``, where a refill on another device moves the
# released buffers of every other device) until a refill of the device's other kind
# takes them (``_pool_drain`` returns the released buffers of every kind of its
# device): it keeps each one idle in the dead kind's pool (at most ``_POOL_DEPTH``,
# at least the kind's ``_POOL_SIZE``, which the failed refill already raised) or
# drops it; with no such refill (the other kind off too, or the device without its
# memory pool) they stay.  A dropped buffer's memory stays cached in the package's
# memory pool.  ``_POOL_READY[kind][device index]`` is the need up to which an eager
# call does not refill: the pool's size after a refill that succeeded, 0 (or absent:
# the next eager call refills) once a capture took a buffer or found none, the call's
# need after a refill that ran out of memory (that refill keeps no new buffer and
# keeps the pool's size, but its drain has already run) and ``sys.maxsize`` once
# ``_pool_off`` turned the kind off: the one lookup of an eager call while the pool
# needs nothing.  ``_POOL_LOCK`` (a plain ``threading.Lock``: not reentrant) guards
# the take (``_pool_take``), the refill and its drain (``_pool_refill``), every entry
# into the package's memory pool (the refill's and ``_package_zeros``': two threads
# never route their allocations to one pool at once) and the writes and drops of the
# ``_CAPTURE_*WORKSPACES`` entries.  Nothing that runs under it may call a function
# that takes it (``_pool_take``, ``_pool_refill``, ``_package_zeros``,
# ``_capture_buffer``): the thread would wait for itself.
_POOL_DEPTH = 4
# torch's out-of-memory error: a refill that meets it keeps no new buffer, frees no
# idle buffer and keeps the pool's size.
_OUT_OF_MEMORY = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", MemoryError)
_POOL_DTYPES: dict[str, torch.dtype] = {"fwd": torch.int32, "bwd": torch.uint8}
_ZEROED_POOL: dict[tuple[int, str], list] = {}
_POOL_SIZE: dict[tuple[int, str], int] = {}
_POOL_READY: dict[str, dict[int, int]] = {"fwd": {}, "bwd": {}}
_GRAPH_HELD: list = []
_POOL_LOCK = threading.Lock()
# The captured calls (with no ``workspace=``) that zeroed a buffer of their own, each one fill node in its graph:
# (device index, kind) -> count, kind ``"fwd"`` or ``"bwd"``; whatever the reason (capture id 0 or no
# ``cuda.bindings``, a user-object error, no ready pre-zeroed buffer, none large enough).  A key is absent until its
# first such call; eager calls never change it.
_CAPTURE_FILL_FALLBACKS: dict[tuple[int, str], int] = {}
# The byte layout of a backward launch per accumulator rule, C and channel tile count (a pure function of them),
# computed once: it is on the host path of every backward call before its launch.
_LAYOUTS: dict[tuple, dict] = {}
# The ``(acc, work)`` tensors of each backward workspace per layout, built once and reused while the workspace keeps
# its storage: retyping two slices of the workspace before every launch costs about 3 us of host time in front of
# the launch.  They alias the workspace's storage, not the tensor (``Tensor.set_``, no view base), so an entry dies
# with its workspace; an entry whose workspace's storage moved (``resize_`` / ``set_``) is rebuilt.
# ``id(workspace) -> (weak reference to the workspace, {layout key: (data_ptr, acc, work)})``: a lookup is one dict
# hit and one weak-reference call (a ``WeakIdKeyDictionary`` builds a key object and runs Python-level hash and
# equality on every lookup); an entry only serves the workspace its weak reference still names, and the reference's
# callback drops the entry when the workspace dies.
_ACCUMULATOR_VIEWS: dict[int, tuple[weakref.ref, dict]] = {}
# The host plan of each call row, planned by the row's first call: ``(arch, route kind, C, device index,
# bwd_grid_mult, bf16_taps) -> ((s_lo, s_hi, cake_jit.RowPlan or None, {n_ch_tiles: the forward's counter words or the
# backward's accumulator layout}), ...)``, one entry per side of ``cake_jit.row_sides`` (ascending token ranges from
# 0 up; one side unless a ``cake_jit.SM_WIDE_ROWS`` or ``cake_jit.FALLBACK_ROWS`` key covers the row, whose ranges
# then cut the token axis).  Selecting the route's program, splitting C into channel tiles, looking
# up the TILE_OVERRIDES / NARROW_TILES row (the forward: its token floors) and sizing the grid cost several us of host
# time in front of every launch; with the row's plan a call only compares ``S`` with the sides' ranges and the side's
# token ranges (``cake_jit.row_launch``).  A side whose program selection reads ``S`` through a token floor
# (``cake_jit.tokens_select``) holds ``None`` and plans every call.  At most ``_ROW_PLANS_MAX`` rows (then emptied);
# :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties it (needed after a change of the registry tables,
# the SM count or the architecture a device reports, which the rows were planned from).
_ROW_PLANS: dict[tuple, tuple[tuple[int, int, Optional[_loader.RowPlan], dict], ...]] = {}
_ROW_PLANS_MAX = 1024
# The torch dtype of each accumulator / counter dtype name of a workspace rule.
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "uint32": torch.uint32,
}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device index, queried once per process."""

    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(
            f"Cake causal conv1d training requires CUDA tensors, got {device}"
        )
    arch = _arch_of_index(_device_index(device))
    if arch is None:
        raise ValueError(
            "Cake causal conv1d training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got "
            f"{tuple(torch.cuda.get_device_capability(device))}"
        )
    return arch


def _alert_nondeterministic(entry: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic
    algorithms are enabled: a ``RuntimeError``, or a ``UserWarning`` under
    ``torch.use_deterministic_algorithms(True, warn_only=True)``; nothing otherwise.  Called by the entry points that
    launch the unordered reduction (the single-rank backward, the KCP backward and the KCP multi backward), before
    they launch or exchange anything (output tensors the caller did not pass may already be allocated).  The alert is raised here, not through a private PyTorch hook
    (PyTorch exposes no public Python entry for it)."""

    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            "cake_pplx.cake_causal_conv1d_det_train is the deterministic package",
            stacklevel=3,
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'; cake_pplx.cake_causal_conv1d_det_train is "
        "the deterministic package, or use the 'warn_only=True' option if that is acceptable for your "
        "application"
    )


def is_cake_causal_conv1d_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` has the exported routes every architecture ships (``fwd`` and ``bwd``) and the package
    is rendered (its ``build.json`` exists).

    The optional routes (bias, BF16 taps / gradients, FP32 SiLU') are checked
    per call: a call that needs one the architecture lacks raises a
    ``ValueError`` naming it.
    """

    device = torch.device(device)
    if device.type != "cuda" or not _loader.is_rendered():
        return False
    capability = tuple(torch.cuda.get_device_capability(device))
    return all(
        route_applies(device_capability=capability, kind=kind)
        for kind in REQUIRED_KINDS
    )


def _require_tensor(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    return tensor


def _check_placement(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | tuple[torch.dtype, ...],
) -> None:
    """On ``device``, of one of the ``dtype``\\ s, contiguous (``ValueError`` otherwise)."""

    dtypes = (dtype,) if isinstance(dtype, torch.dtype) else tuple(dtype)
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype not in dtypes:
        wanted = " or ".join(str(d) for d in dtypes)
        raise ValueError(f"{name} must be {wanted}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


# The largest token count ``S`` a call takes.  The kernels split a channel
# tile's tokens over its token streams with the 32-bit ``(S + streams - 1) /
# streams``.  A tile has at most ``grid_mult x ctas_per_sm`` streams per SM, a
# few thousand on any supported device and far below MAX_TOKEN_STREAMS, so
# ``S <= INT32_MAX - MAX_TOKEN_STREAMS + 1`` keeps that sum in range.
MAX_TOKEN_STREAMS = 1 << 16
MAX_TOKENS = INT32_MAX - MAX_TOKEN_STREAMS + 1


def _tokens(
    tensor: object,
    name: str,
    *,
    device: torch.device,
    shape: Optional[tuple[int, ...]] = None,
) -> tuple[int, int, torch.Tensor]:
    """Validate a BF16 ``[S, C]`` or ``[1, S, C]`` token tensor; return ``(S, C, [S, C] view)``.

    ``shape`` pins the exact shape another token tensor of the call has (``dy``
    against ``x``).  Both kernels read the packed rows as 16-byte vectors of eight
    channels, so ``C`` is a multiple of eight; ``S`` and ``C`` are 32-bit kernel
    scalars (``S x C`` itself may exceed 2^31: the kernels address with 64-bit
    offsets), and ``S`` is at most :data:`MAX_TOKENS`.
    """

    t = _require_tensor(tensor, name)
    if t.ndim == 3 and t.shape[0] == 1:
        S, C = int(t.shape[1]), int(t.shape[2])
    elif t.ndim == 2:
        S, C = int(t.shape[0]), int(t.shape[1])
    else:
        raise ValueError(
            f"{name} must be a [S, C] or [1, S, C] tensor, got shape {tuple(t.shape)}"
        )
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    _check_placement(t, name, device=device, dtype=torch.bfloat16)
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"{name} must have a channel count C that is a positive multiple of "
            f"{CHANNEL_MULTIPLE}, got C={C}"
        )
    if S > MAX_TOKENS or C > INT32_MAX:
        raise ValueError(
            f"{name}: S={S} must be at most {MAX_TOKENS} and C={C} must fit a "
            "32-bit kernel scalar (the kernels split S over at most "
            f"{MAX_TOKEN_STREAMS} token streams with the 32-bit S + streams - 1)"
        )
    return S, C, t.view(S, C)


def _weight(
    weight: object,
    C: int,
    *,
    device: torch.device,
    name: str = "weight",
    dtypes: tuple[torch.dtype, ...] = (torch.float32,),
) -> torch.Tensor:
    """Validate the ``[C, 4]`` taps of one of ``dtypes`` (the training API passes FP32 and BF16; the KCP path
    keeps the default, FP32 only)."""

    w = _require_tensor(weight, name)
    if w.ndim != 2 or int(w.shape[1]) != WIDTH:
        raise ValueError(
            f"{name} must be a [C, {WIDTH}] tensor of a width-{WIDTH} convolution "
            f"(tap {WIDTH - 1} multiplies the current token), got shape {tuple(w.shape)}"
        )
    if int(w.shape[0]) != C:
        raise ValueError(
            f"{name} must have one row per channel ([{C}, {WIDTH}]), got shape {tuple(w.shape)}"
        )
    _check_placement(w, name, device=device, dtype=dtypes)
    return w


def _bias(bias: object, C: int, *, device: torch.device) -> torch.Tensor:
    """Validate the per-channel ``[C]`` bias (FP32 or BF16, independently of the taps)."""

    b = _require_tensor(bias, "bias")
    if b.ndim != 1 or int(b.shape[0]) != C:
        raise ValueError(
            f"bias must be a [C] tensor with one value per channel ([{C}]), got shape {tuple(b.shape)}"
        )
    _check_placement(b, "bias", device=device, dtype=PARAMETER_DTYPES)
    return b


def _cu_seqlens(
    cu_seqlens: object, *, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Validate the ``int32 [num_docs + 1]`` device prefix sums; return ``(cu_seqlens, num_docs)``.

    The host reads the tensor's metadata only: no element is copied to the host
    and no stream is synchronized, so the call is CUDA Graph capturable.
    """

    cu = _require_tensor(cu_seqlens, "cu_seqlens")
    if cu.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be a 1-D [num_docs + 1] tensor, got shape {tuple(cu.shape)}"
        )
    _check_placement(cu, "cu_seqlens", device=device, dtype=torch.int32)
    num_docs = int(cu.numel()) - 1
    if num_docs < 1:
        raise ValueError(
            "cu_seqlens must hold at least two prefix sums (one document), got "
            f"{int(cu.numel())} elements"
        )
    return cu, num_docs


def _output(
    tensor: Optional[object],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dtypes: Optional[tuple[torch.dtype, ...]] = None,
) -> torch.Tensor:
    """A caller-provided output (exact shape, contiguous, of ``dtype`` or of one of ``dtypes`` when given) or a
    fresh ``dtype`` one."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    out = _require_tensor(tensor, name)
    if tuple(out.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(out.shape)}"
        )
    _check_placement(out, name, device=device, dtype=dtypes or dtype)
    return out


# The backward programs store one channel's four FP32 ``dweight`` taps with one 16-byte store.
DWEIGHT_FP32_ALIGN = 16


def _dweight_aligned(dw_out: torch.Tensor) -> torch.Tensor:
    """An FP32 ``dweight`` output whose address is :data:`DWEIGHT_FP32_ALIGN`-byte aligned (``ValueError``
    otherwise; a fresh tensor always is).  BF16 ``dweight`` and ``dbias`` are stored per element."""

    misaligned = dw_out.data_ptr() % DWEIGHT_FP32_ALIGN
    if dw_out.dtype == torch.float32 and misaligned:
        raise ValueError(
            f"an FP32 dweight must be {DWEIGHT_FP32_ALIGN}-byte aligned (each channel's {WIDTH} taps are stored "
            f"with one {DWEIGHT_FP32_ALIGN}-byte store), got data_ptr() % {DWEIGHT_FP32_ALIGN} == {misaligned}"
        )
    return dw_out


def _silu_grad_preactivation(value: object) -> str:
    if value not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(
            f"silu_grad_preactivation must be one of {SILU_GRAD_PREACTIVATIONS}, got {value!r}"
        )
    return str(value)


def _dweight_sum(value: object, policy: str) -> str:
    """Validate ``dweight_sum``: a ``str`` (``TypeError`` otherwise), one of ``DWEIGHT_SUMS`` (``ValueError``
    otherwise); ``"fp32_pair"`` needs the ``"fp32"`` SiLU' policy (``ValueError`` otherwise: only the FP32-SiLU'
    routes have a pair form)."""

    if not isinstance(value, str):
        raise TypeError(f"dweight_sum must be a str, one of {DWEIGHT_SUMS}, got {type(value).__name__}")
    if value not in DWEIGHT_SUMS:
        raise ValueError(f"dweight_sum must be one of {DWEIGHT_SUMS}, got {value!r}")
    if value == "fp32_pair" and policy != "fp32":
        raise ValueError(
            'dweight_sum="fp32_pair" needs silu_grad_preactivation="fp32": only the FP32-SiLU\' '
            f"routes have an FP32 (hi, lo) pair form, got silu_grad_preactivation={policy!r}"
        )
    return str(value)


def _bwd_grid_mult(value: object) -> int:
    """Validate ``bwd_grid_mult``: an ``int`` (``TypeError`` otherwise, ``bool`` included) in ``BWD_GRID_MULTS``
    (``ValueError`` otherwise)."""

    if type(value) is not int:
        raise TypeError(
            f"bwd_grid_mult must be an int, one of {BWD_GRID_MULTS}, got {type(value).__name__} {value!r}"
        )
    if value not in BWD_GRID_MULTS:
        raise ValueError(f"bwd_grid_mult must be one of {BWD_GRID_MULTS}, got {value}")
    return value


def _scalars(scalars: dict[str, int]) -> dict[str, int]:
    """The launch scalars :func:`cake_jit.plan` produced (``flags`` is set beside them by the caller)."""

    return {key: int(scalars[key]) for key in _loader.PLAN_SCALARS}


def _flags(
    w: torch.Tensor,
    dw: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    db: Optional[torch.Tensor] = None,
) -> int:
    """The ``flags`` bit mask of a ``dtype_flags`` program from the ACTUAL tensors of the call."""

    flags = 0
    for tensor, bit in (
        (w, FLAG_W_BF16),
        (dw, FLAG_DW_BF16),
        (b, FLAG_B_BF16),
        (db, FLAG_DB_BF16),
    ):
        if tensor is not None and tensor.dtype == torch.bfloat16:
            flags |= bit
    return flags


def _route(arch: str, kind: str, *, needs: str, C: int, S: int) -> tuple[str, ...]:
    """The programs ``kind`` launches on ``arch`` for the call (``ValueError`` naming the route when the
    architecture's export lacks it)."""

    if kind not in _loader.ROUTES.get(arch, {}):
        raise ValueError(
            f"{needs} needs the {kind!r} route, which this package does not export for "
            f"{arch}; its routes: {sorted(_loader.ROUTES.get(arch, {}))}"
        )
    return selected_modules(arch, kind, C=C, S=S)


def _launch(
    record: dict,
    S: int,
    C: int,
    num_docs: int,
    *,
    device_index: int,
    arch: str,
    bwd_grid_mult: int = 1,
    bf16_taps: bool = False,
) -> tuple[dict[str, int], tuple[int, int, int]]:
    """The kernel scalars and the grid of one launch of ``record`` (the host plan of ``cake_jit``).

    CTA ``b`` owns channel tile ``b % n_ch_tiles``, so every channel tile needs
    a CTA of the launched program's grid on the device (for the backward also
    so that every tile has a last CTA that stores its gradients): a ``C``
    beyond ``grid x threads x vec`` channels is refused rather than left partly
    unwritten.  The caller launches this grid, so a call plans it once.  A
    backward ``bwd_grid_mult`` above 1 launches that many times the record's
    untrimmed grid (``launch_grid`` without the row), never trimmed to whole
    channel-tile ranks; the scalars do not depend on the grid.  A record's
    plan takes the device's architecture ``arch`` and the record's route, so a
    backward row of ``cake_jit.TILE_OVERRIDES`` or ``cake_jit.NARROW_TILES``
    and a forward narrow row (``cake_jit.channel_split``, of the call's taps:
    ``bf16_taps``) run their narrower channel tiles.  At ``bwd_grid_mult`` 1
    the grid is :func:`_one_sm_trim` of the row's ``launch_grid``.
    """

    scalars = plan(
        S,
        C,
        num_docs,
        geometry=record["grid_rule"]["geometry"],
        role=record["role"],
        arch=arch,
        route=_loader.record_route(record, arch),
        bf16_taps=bf16_taps,
    )
    if bwd_grid_mult == 1:
        grid = _one_sm_trim(
            record,
            scalars["n_ch_tiles"],
            launch_grid(record, device_index=device_index, C=C, S=S),
            device_index=device_index,
            arch=arch,
            bwd_grid_mult=1,
        )
    else:
        grid = (launch_grid(record, device_index=device_index)[0] * bwd_grid_mult, 1, 1)
    _check_tiles(C, scalars["n_ch_tiles"], grid)
    return scalars, grid


# The backward routes whose single-wave grid gives up ``ctas_per_sm`` CTAs when its last channel-tile rank holds at
# least that many (:func:`_one_sm_trim`); the other routes launch the grid of ``launch_grid``.
_ONE_SM_TRIM_ROUTES: tuple[str, ...] = ("bwd", "bwd_fp32g")


def _one_sm_trim(
    record: dict,
    n_ch_tiles: int,
    grid: tuple[int, int, int],
    *,
    device_index: int,
    arch: str,
    bwd_grid_mult: int,
) -> tuple[int, int, int]:
    """``grid`` less ``ctas_per_sm`` CTAs for a backward of a route of ``_ONE_SM_TRIM_ROUTES`` at ``bwd_grid_mult``
    1 when ``grid`` is the record's untrimmed single wave (``ctas_per_sm x SM count``, ``grid_mult`` 1) and its last
    channel-tile rank holds at least ``ctas_per_sm`` CTAs (``grid % n_ch_tiles >= ctas_per_sm``); else ``grid``.
    The grid then fits on one SM fewer: a kernel on another stream that keeps one SM's shared memory leaves the
    launch one wave.  A grid ``launch_grid`` already trimmed to whole ranks is unchanged."""

    rule = record["grid_rule"]
    if (
        bwd_grid_mult != 1
        or rule["kind"] != "occupancy"
        or _loader.grid_mult(record, arch) != 1
        or _loader.record_route(record, arch) not in _ONE_SM_TRIM_ROUTES
    ):
        return grid
    k = int(rule["ctas_per_sm"])
    full = k * _loader.sm_count(device_index)
    if int(grid[0]) != full or full % int(n_ch_tiles) < k:
        return grid
    return (full - k, int(grid[1]), int(grid[2]))


def _check_tiles(C: int, n_ch_tiles: int, grid: tuple[int, int, int]) -> None:
    """Refuse a launch whose ``grid`` has fewer CTAs than ``C``'s ``n_ch_tiles`` channel tiles (``ValueError``)."""

    if n_ch_tiles > grid[0]:
        raise ValueError(
            f"C={C} spans {n_ch_tiles} channel tiles but the launch grid on this "
            f"device is {grid[0]} CTAs; every channel tile needs a CTA"
        )


def _route_needs(policy: Optional[str]) -> str:
    """What a call needs, for the error of a route the architecture lacks: the forward's (``policy`` ``None``) or
    the backward's."""

    if policy is None:
        return "this forward (bias / BF16 taps)"
    return f"this backward (bias / BF16 taps or dweight / silu_grad_preactivation={policy!r})"


def _call_plan(
    arch: str,
    kind: str,
    S: int,
    C: int,
    num_docs: int,
    *,
    device_index: int,
    policy: Optional[str] = None,
    bwd_grid_mult: int = 1,
    bf16_taps: bool = False,
) -> tuple[tuple[str, ...], dict, dict[str, int], tuple[int, int, int], dict]:
    """``(names, record, scalars, grid, per_tiles)`` of one call of route ``kind``: the programs :func:`_route`
    selects, the first one's record, the :data:`cake_jit.PLAN_SCALARS` and the grid :func:`_launch` plans (same
    values, same errors), from the call's ``_ROW_PLANS`` row; ``per_tiles`` is the side's dict for the host values
    of each channel tile count (empty and not kept for a row planned per call).  ``policy`` is the backward's
    ``silu_grad_preactivation`` (``None`` for the forward) and only names the route in an error; ``bf16_taps`` (the
    forward's taps are BF16; ``False`` for every backward call) is part of the row.  A row a
    :data:`cake_jit.SM_WIDE_ROWS`, :data:`cake_jit.DIRECT_ROWS` or :data:`cake_jit.FALLBACK_ROWS` key covers is
    planned for every side of :func:`cake_jit.row_sides` by its first call; the call's side is a few compares of
    ``S`` with the sides' ranges."""

    key = (arch, kind, C, device_index, bwd_grid_mult, bf16_taps)
    sides = _ROW_PLANS.get(key)
    if sides is None:
        # A missing route or a registry error raises as in a per-call plan.
        _route(arch, kind, needs=_route_needs(policy), C=C, S=S)
        sides = tuple(
            (
                s_lo,
                s_hi,
                _loader.plan_row(
                    arch,
                    kind,
                    C,
                    device_index=device_index,
                    bwd_grid_mult=bwd_grid_mult,
                    at=s_lo,
                    bf16_taps=bf16_taps,
                ),
                {},
            )
            for s_lo, s_hi in _loader.row_sides(arch, kind, C)
        )
        if len(_ROW_PLANS) >= _ROW_PLANS_MAX:
            _ROW_PLANS.clear()
        _ROW_PLANS[key] = sides
    for s_lo, s_hi, row, per_tiles in sides:
        if s_lo <= S <= s_hi:
            break
    if row is None:
        names = _route(arch, kind, needs=_route_needs(policy), C=C, S=S)
        record = _loader.MODULES[names[0]]
        scalars, grid = _launch(
            record,
            S,
            C,
            num_docs,
            device_index=device_index,
            arch=arch,
            bwd_grid_mult=bwd_grid_mult,
            bf16_taps=bf16_taps,
        )
        return names, record, _scalars(scalars), grid, {}
    scalars, grid = _loader.row_launch(row, S, num_docs, device_index=device_index)
    grid = _one_sm_trim(
        row.record,
        scalars["n_ch_tiles"],
        grid,
        device_index=device_index,
        arch=arch,
        bwd_grid_mult=bwd_grid_mult,
    )
    _check_tiles(C, scalars["n_ch_tiles"], grid)
    return row.names, row.record, scalars, grid, per_tiles


def cake_causal_conv1d_nondet_train_plan_cache_clear() -> int:
    """Drop the host plans of the single-rank calls (``_ROW_PLANS``), planned again by the next call of each row;
    returns the number of rows dropped.  Call it after changing what the rows were planned from: the registry
    tables of ``cake_jit`` (``MODULES``, ``ROUTES``, ``TILE_OVERRIDES``, ``NARROW_TILES``, ``FALLBACK_ROWS``,
    ``SM_WIDE_ROWS``, ``DIRECT_ROWS``, ``DIRECT_PLANS``, the forward's ``FORWARD_NARROW_MIN_S``,
    ``FORWARD_NARROW_ROW_MIN_S``, ``FORWARD_NARROW_ROW_FLOORS``, ``FORWARD_SINGLE_TILE_ROWS``,
    ``FORWARD_NARROW_WIDTHS`` and ``FORWARD_WIDTH_FLOORS``, the grid-trim tables), its ``sm_count`` or
    ``device_arch``.  The programmatic dependent launch policy (``PDL_EAGER`` / ``PDL_GRAPH`` / ``PDL_OFF_ROWS``) is
    read by every call, not planned."""

    n = len(_ROW_PLANS)
    _ROW_PLANS.clear()
    return n


def _check_shape_args(S: int, C: int) -> tuple[int, int]:
    S, C = int(S), int(C)
    if S < 0:
        raise ValueError("S must be non-negative")
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"C must be a positive multiple of {CHANNEL_MULTIPLE}, got {C}"
        )
    return S, C


def cake_causal_conv1d_nondet_train_backward_workspace_bytes(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> int:
    """Bytes of the caller-owned backward workspace for ``C`` channels on ``device``.

    The workspace holds the backward's accumulators and per channel tile ticket
    counters (``cake_jit.accumulator_layout``): ``slots x taps x C`` FP32 words
    (``slots`` 16 on the ``_fp32g`` routes, else 1; ``taps`` 5 on the bias
    routes) and one uint32 counter per channel tile, sized by the largest
    over every backward program of every backward route of the architecture
    (``cake_jit.backward_workspace_bytes``), so one buffer serves every route.
    It depends on ``C`` and the architecture only (``S`` is checked and
    otherwise unused): one buffer serves every token count.  ``bwd_grid_mult``
    (one of ``BWD_GRID_MULTS``, default 1; ``TypeError`` / ``ValueError`` as in
    the deterministic package) is validated and does not change the size: the
    accumulators and counters are per channel tile, whatever the grid, so one
    buffer serves the calls of every ``bwd_grid_mult``.
    """

    _bwd_grid_mult(bwd_grid_mult)
    device = torch.device(device)
    S, C = _check_shape_args(S, C)
    return backward_workspace_bytes(_device_arch(device), C)


def cake_causal_conv1d_nondet_train_backward_workspace(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> torch.Tensor:
    """Allocate a ZEROED ``uint8`` backward workspace (see ``..._workspace_bytes``).

    Every launch leaves it zero, so one buffer serves every later backward call
    of one stream, CUDA Graph replays and every ``bwd_grid_mult`` included.
    Two launches that may run at once need two buffers.
    """

    nbytes = cake_causal_conv1d_nondet_train_backward_workspace_bytes(
        S, C, device, bwd_grid_mult=bwd_grid_mult
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_causal_conv1d_nondet_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Bytes of the caller-owned forward workspace for ``C`` channels on ``device``.

    The forward's programs take their token chunks from int32 counters, two
    per channel tile, sized by the most over every forward program of every
    forward route of the architecture (``cake_jit.forward_workspace_words``):
    the det package's rule over the same programs.  The buffer must be zero
    before its first launch; every launch leaves it zero.  Two launches that
    may run at once need two buffers.
    """

    device = torch.device(device)
    S, C = _check_shape_args(S, C)
    return 4 * forward_workspace_words(_device_arch(device), C)


def cake_causal_conv1d_nondet_train_forward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a zeroed int32 forward workspace (see ``..._workspace_bytes``); keep it for every later forward call
    of one stream."""

    nbytes = cake_causal_conv1d_nondet_train_forward_workspace_bytes(S, C, device)
    return torch.zeros(nbytes // 4, dtype=torch.int32, device=torch.device(device))


# torch's getter of the raw current-stream handle (``torch.cuda.current_stream(i).cuda_stream`` without building a
# ``torch.cuda.Stream``); ``None`` on a torch build without it.
_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream_handle(device_index: int) -> int:
    """The handle of the device's current CUDA stream, the key of the package's per-stream workspaces: the value of
    ``torch.cuda.current_stream(device_index).cuda_stream``, read without building a ``torch.cuda.Stream`` object
    (about 1 us of host time in front of every launch that uses a package workspace)."""

    if _CURRENT_RAW_STREAM is not None:
        return int(_CURRENT_RAW_STREAM(device_index))
    return int(torch.cuda.current_stream(device_index).cuda_stream)


def _capture_info(stream: int) -> tuple[int, object]:
    """``(capture id, graph)`` of the CUDA Graph capture the raw stream ``stream`` takes part in: CUDA's id of the
    capture (two captures have distinct ids even on the same stream) and the ``cudaGraph_t`` being captured, both
    read by one ``cudaStreamGetCaptureInfo``; ``(0, None)`` when the stream is not capturing or when the CUDA Python
    bindings are not importable (then every captured backward call zeroes a buffer of its own).  The graph is
    ``None`` unless the result's fourth item is a ``cudaGraph_t`` (the layout ``(error, status, id, graph, ...)``);
    without it a capture takes no pre-zeroed buffer."""

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        return 0, None
    info = cudart.cudaStreamGetCaptureInfo(stream)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        return 0, None
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0, None
    graph = info[3] if len(info) > 3 and isinstance(info[3], cudart.cudaGraph_t) else None
    return int(info[2]), graph


class _PoolBuffer:
    """One pre-zeroed capture buffer: the zeroed ``tensor`` of ``kind`` on device ``device_index``, the address of
    its pinned readiness flag (``flag``: 1 once its zero fill completed, ``0`` while it is in a graph), the POSIX
    semaphore ``sem`` its user object's destructor posts and the user object ``obj`` (our reference while idle)."""

    __slots__ = ("tensor", "kind", "device_index", "flag", "sem", "obj")

    def __init__(self, tensor: torch.Tensor, kind: str, device_index: int, flag: int, sem: int, obj: object):
        self.tensor, self.kind, self.device_index = tensor, kind, device_index
        self.flag, self.sem, self.obj = flag, sem, obj


class _UserObjects:
    """The CUDA user-object calls of the pre-zeroed capture buffers (``cuda.bindings`` runtime API) and their
    destructor: libc's ``sem_post`` on the buffer's POSIX semaphore.  CUDA runs a destructor on an internal thread,
    after the graph that held the object and its instantiations are destroyed and their launches have completed;
    it must not call CUDA or block, which ``sem_post`` (async-signal-safe) does not.  ``sem_post`` has the type
    ``int (sem_t *)`` and CUDA calls it as ``void (*)(void *)``: on the LP64 Linux ABIs (x86-64, AArch64) the pointer
    is passed in the first integer register and the ``int`` result is ignored.  The semaphores live in host memory
    from libc's ``malloc`` that is never freed (a destructor may post one at any time, also after the interpreter
    has torn this module down); a semaphore is handed out again only after ``sem_trywait`` consumed its post."""

    SEM_BYTES = 64  # at least sizeof(sem_t) (32 on 64-bit glibc), one per 64-byte line
    SEM_CHUNK = 256

    def __init__(self) -> None:
        from cuda.bindings import runtime as cudart

        libc = ctypes.CDLL(None, use_errno=True)
        libc.sem_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
        libc.sem_init.restype = ctypes.c_int
        libc.sem_trywait.argtypes = [ctypes.c_void_p]
        libc.sem_trywait.restype = ctypes.c_int
        libc.malloc.argtypes = [ctypes.c_size_t]
        libc.malloc.restype = ctypes.c_void_p
        self.cudart = cudart
        self.libc = libc
        self.sem_post = ctypes.cast(libc.sem_post, ctypes.c_void_p).value
        self.success = cudart.cudaError_t.cudaSuccess
        self.create_flags = int(cudart.cudaUserObjectFlags.cudaUserObjectNoDestructorSync)
        self.move = int(cudart.cudaUserObjectRetainFlags.cudaGraphUserObjectMove)
        self.free_sems: list[int] = []
        self.retired: list[int] = []  # semaphores of released idle objects, until their post is consumed
        self.flags: list[torch.Tensor] = []  # pinned int32 readiness flags, never freed
        self.free_flags: list[tuple[torch.Tensor, int, int]] = []
        self.pending_flags: list[tuple[torch.Tensor, int, int]] = []  # returned before their copy was seen
        self.released: list[_PoolBuffer] = []  # held buffers their graphs released, for their device's refill

    def semaphore(self) -> int:
        if not self.free_sems:
            chunk = self.libc.malloc(self.SEM_BYTES * (self.SEM_CHUNK + 1))  # never freed
            if not chunk:
                raise MemoryError("malloc of the semaphores failed")
            base = (chunk + self.SEM_BYTES - 1) // self.SEM_BYTES * self.SEM_BYTES
            self.free_sems.extend(base + self.SEM_BYTES * i for i in range(self.SEM_CHUNK))
        sem = self.free_sems.pop()
        if self.libc.sem_init(sem, 0, 0) != 0:
            self.free_sems.append(sem)
            raise OSError(ctypes.get_errno(), "sem_init")
        return sem

    def posted(self, sem: int) -> bool:
        return self.libc.sem_trywait(sem) == 0

    def create(self, sem: int) -> object:
        """A user object (one reference, ours) whose destructor posts ``sem``; outside any capture."""

        res = self.cudart.cudaUserObjectCreate(sem, self.sem_post, 1, self.create_flags)
        if res[0] != self.success:
            raise RuntimeError(f"cudaUserObjectCreate: {res[0]}")
        return res[1]

    def retain(self, graph: object, obj: object) -> bool:
        """Move our reference of ``obj`` into ``graph`` (the graph being captured); False when CUDA refuses."""

        try:
            res = self.cudart.cudaGraphRetainUserObject(graph, obj, 1, self.move)
        except Exception:
            return False
        return (res[0] if isinstance(res, tuple) else res) == self.success

    def release(self, obj: object, sem: int) -> None:
        """Drop our reference of an idle ``obj``; its semaphore returns once the destructor posted it."""

        self.cudart.cudaUserObjectRelease(obj, 1)
        self.retired.append(sem)

    def flag(self) -> tuple[torch.Tensor, int, int]:
        """A free readiness flag: ``(pinned tensor, index, address)``."""

        if not self.free_flags:
            flags = torch.zeros(64, dtype=torch.int32, pin_memory=True)
            self.flags.append(flags)
            base = flags.data_ptr()
            self.free_flags.extend((flags, i, base + 4 * i) for i in range(64))
        return self.free_flags.pop()

    def free_flag(self, flag: tuple[torch.Tensor, int, int]) -> None:
        """Return a flag; one whose device-to-host copy has not been seen yet (it reads 0) waits in
        ``pending_flags``, so that a late copy never marks another buffer ready."""

        if ctypes.c_int32.from_address(flag[2]).value == 1:
            self.free_flags.append(flag)
        else:
            self.pending_flags.append(flag)


# The package's own memory pool per device index (``torch.cuda.MemPool``), made on first use, or ``None`` on a torch
# without ``torch.cuda.MemPool`` / ``torch.cuda.use_mem_pool`` or when making it failed.  The package allocates its
# pre-zeroed capture buffers and its per-(device, stream) workspaces inside ``torch.cuda.use_mem_pool`` of it, so they
# never land in another pool the calling thread's allocations are routed to (a CUDA Graph tree's warm-up routes the
# thread's allocations to the tree's private pool, which must hold only the graph's own tensors).  Its memory is not
# shared with the caching allocator's other pools: freed blocks stay cached in it for the package's later
# allocations.
_MEM_POOLS: dict[int, object] = {}


def _mem_pool(device_index: int) -> object:
    pool = _MEM_POOLS.get(device_index, False)
    if pool is False:
        make = getattr(torch.cuda, "MemPool", None)
        pool = None
        if make is not None and hasattr(torch.cuda, "use_mem_pool"):
            try:
                with torch.cuda.device(device_index):
                    pool = make()
            except Exception:
                pool = None
        _MEM_POOLS[device_index] = pool
    return pool


def _package_zeros(numel: int, dtype: torch.dtype, device: torch.device, device_index: int) -> torch.Tensor:
    """A zeroed tensor of the package's own memory pool on the device (:func:`_mem_pool`), or of the caching
    allocator's current pool where the package has none.  Only a failure to enter the pool (an unusable
    ``torch.cuda.use_mem_pool``) takes the device off it; an allocation error (an out-of-memory error) propagates.
    Holds ``_POOL_LOCK`` (never taken by a caller of this function)."""

    # under _POOL_LOCK, as the refill: two threads never route their allocations to one pool at once
    with _POOL_LOCK:
        pool = _mem_pool(device_index)
        if pool is not None:
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(torch.cuda.use_mem_pool(pool, device=device))
                except Exception:  # the API is unusable here: the device leaves its pool for good
                    _MEM_POOLS[device_index] = None
                else:
                    # an allocation error (an OOM) propagates and the device keeps its pool
                    return torch.zeros(numel, dtype=dtype, device=device)
    return torch.zeros(numel, dtype=dtype, device=device)


# ``None`` until the first refill probes the user-object calls; then a ``_UserObjects``, or ``False`` when they are
# unavailable (no ``cuda.bindings``, no libc semaphores, a CUDA error), and the captures zero buffers of their own.
_USER_OBJECTS: object = None


def _user_objects() -> object:
    global _USER_OBJECTS
    if _USER_OBJECTS is None:
        try:
            _USER_OBJECTS = _UserObjects()
        except Exception:
            _USER_OBJECTS = False
    return _USER_OBJECTS


def _flag_ready(buf: _PoolBuffer) -> bool:
    return buf.flag != 0 and ctypes.c_int32.from_address(buf.flag[2]).value == 1


def _pool_off(kind: str, device_index: int) -> None:
    """No pre-zeroed buffers of ``kind`` on the device: its eager calls never refill again and its captures take
    none.  Its idle buffers (at most ``_POOL_DEPTH``) stay allocated; a buffer its graphs hold returns, once
    released, only at a refill of the device's other kind, which keeps it idle or drops it (the pool comment above
    ``_POOL_DEPTH``); a dropped buffer's memory stays cached in the package's memory pool."""

    _POOL_READY[kind][device_index] = sys.maxsize


def _drop_capture_entries(tensor: torch.Tensor) -> None:
    """Drop the ``_CAPTURE_*WORKSPACES`` entries naming ``tensor`` (its graph released it)."""

    for table in (_CAPTURE_WORKSPACES, _CAPTURE_FORWARD_WORKSPACES):
        for key in [k for k, (_, t) in list(table.items()) if t is tensor]:
            table.pop(key, None)


def _pool_drain(uo: _UserObjects, device_index: int) -> list[_PoolBuffer]:
    """The buffers of device ``device_index`` whose graphs released them (their semaphore was posted), removed from
    ``_GRAPH_HELD`` (those of other devices wait in ``uo.released`` for their device's refill); the semaphores of
    released idle objects and the pending flags whose copy completed are returned.  Host calls only."""

    held = []
    for buf in _GRAPH_HELD:
        (uo.released if uo.posted(buf.sem) else held).append(buf)
    _GRAPH_HELD[:] = held
    mine = [buf for buf in uo.released if buf.device_index == device_index]
    uo.released[:] = [buf for buf in uo.released if buf.device_index != device_index]
    retired = []
    for sem in uo.retired:
        (uo.free_sems if uo.posted(sem) else retired).append(sem)
    uo.retired[:] = retired
    pending, uo.pending_flags[:] = list(uo.pending_flags), []
    for flag in pending:
        uo.free_flag(flag)
    return mine


def _pool_refill(device_index: int, kind: str, need: int, device: torch.device) -> None:
    """On an eager call that omits its workspace and needs ``need`` elements of ``kind`` beyond
    ``_POOL_READY[kind][device_index]``: return the buffers released by their graphs (to the pool, with a new user
    object, or freed), raise the device's pool size to ``need``, free idle buffers below it and zero new buffers on
    the current stream until the pool holds ``_POOL_DEPTH``; each new buffer's flag is set by a device-to-host copy,
    queued behind its zero fill, of a 1 this refill writes on the same stream.  An out-of-memory error keeps no new
    buffer, frees no idle buffer and keeps the pool's size (the drain before it has already returned or dropped the
    released buffers) and sets ``_POOL_READY[kind][device_index]`` to ``need`` (the next refill comes at a larger
    need or after the next take); any other failure leaves the device without pre-zeroed buffers of ``kind``."""

    uo = _user_objects()
    if not uo:
        _pool_off(kind, device_index)
        return
    with _POOL_LOCK:
        mem_pool = _mem_pool(device_index)
        if mem_pool is None:
            _pool_off(kind, device_index)
            return
        key = (device_index, kind)
        old_size = _POOL_SIZE.get(key, 0)
        size = max(old_size, int(need))
        _POOL_SIZE[key] = size
        try:
            with torch.cuda.device(device_index):
                for buf in _pool_drain(uo, device_index):
                    _drop_capture_entries(buf.tensor)
                    bkey = (buf.device_index, buf.kind)
                    pool = _ZEROED_POOL.setdefault(bkey, [])
                    if len(pool) < _POOL_DEPTH and buf.tensor.numel() >= _POOL_SIZE.get(bkey, 0):
                        # its graphs' launches have completed and left it zero: ready at once
                        buf.obj = uo.create(buf.sem)
                        buf.flag = uo.flag()
                        ctypes.c_int32.from_address(buf.flag[2]).value = 1
                        pool.append(buf)
                    else:
                        uo.free_sems.append(buf.sem)
                pool = _ZEROED_POOL.setdefault(key, [])
                keep = [b for b in pool if b.tensor.numel() >= size]
                fresh: list[_PoolBuffer] = []
                if len(keep) < _POOL_DEPTH:
                    try:
                        with torch.cuda.use_mem_pool(mem_pool, device=device):
                            # the flags' source, written on this stream before the copies that read it
                            one = torch.ones(1, dtype=torch.int32, device=device)
                            tensors = [
                                torch.zeros(size, dtype=_POOL_DTYPES[kind], device=device)
                                for _ in range(_POOL_DEPTH - len(keep))
                            ]
                    except _OUT_OF_MEMORY:
                        _POOL_SIZE[key] = old_size
                        _POOL_READY[kind][device_index] = int(need)
                        return
                    for tensor in tensors:
                        flag = uo.flag()
                        ctypes.c_int32.from_address(flag[2]).value = 0
                        flag[0][flag[1] : flag[1] + 1].copy_(one, non_blocking=True)
                        sem = uo.semaphore()
                        fresh.append(_PoolBuffer(tensor, kind, device_index, flag, sem, uo.create(sem)))
                for buf in pool:
                    if buf not in keep:
                        uo.release(buf.obj, buf.sem)
                        uo.free_flag(buf.flag)
                pool[:] = keep + fresh
        except Exception:
            _pool_off(kind, device_index)
            return
        _POOL_READY[kind][device_index] = size


def _pool_take(device_index: int, kind: str, need: int, graph: object) -> Optional[torch.Tensor]:
    """Inside a capture of ``graph``: an idle pre-zeroed buffer of ``kind`` with at least ``need`` elements whose
    zero fill has completed (its flag reads 1), its user object's reference moved into ``graph``; ``None`` when the
    device has none (or CUDA refuses the retain).  Either way the next eager call refills the pool.  Makes no CUDA
    call but ``cudaGraphRetainUserObject``."""

    uo = _USER_OBJECTS
    if not uo or graph is None:
        return None
    with _POOL_LOCK:
        if _POOL_READY[kind].get(device_index) == sys.maxsize:
            return None  # the device has no pool (``_pool_off``)
        _POOL_READY[kind][device_index] = 0
        # buffers released meanwhile wait in _GRAPH_HELD for the next eager refill (it makes their new user objects)
        for buf in _ZEROED_POOL.get((device_index, kind), ()):
            if buf.tensor.numel() >= need and _flag_ready(buf):
                if not uo.retain(graph, buf.obj):
                    return None
                _ZEROED_POOL[(device_index, kind)].remove(buf)
                uo.free_flags.append(buf.flag)  # it read 1: its copy is done
                buf.flag, buf.obj = 0, None
                _GRAPH_HELD.append(buf)
                return buf.tensor
    return None


def _capture_buffer(
    table: dict, device_index: int, kind: str, need: int, device: torch.device
) -> torch.Tensor:
    """The zeroed buffer of ``need`` elements of ``kind`` of a captured call that omits its workspace: the
    capture's pre-zeroed buffer of the stream (``table``) when it holds one large enough, else a pre-zeroed buffer
    of the package (``_pool_take``, kept in ``table``), else a buffer this call zeroes itself (a fill node, counted
    in ``_CAPTURE_FILL_FALLBACKS``).  No table keeps that last one: it comes from the capture's memory pool, and a
    reference kept after the call would outlive the capture (a CUDA Graph tree's pool, which its graphs share, must
    hold only the tensors the tree tracks); the graph's fill node zeroes it on every replay."""

    stream = _current_stream_handle(device_index)
    capture, graph = _capture_info(stream)
    held = table.get((device_index, stream))
    if capture and held is not None and held[0] == capture and held[1].numel() >= need:
        return held[1]
    ws = _pool_take(device_index, kind, need, graph) if capture else None
    if ws is None:
        key = (device_index, kind)
        _CAPTURE_FILL_FALLBACKS[key] = _CAPTURE_FILL_FALLBACKS.get(key, 0) + 1
        return torch.zeros(need, dtype=_POOL_DTYPES[kind], device=device)
    with _POOL_LOCK:  # _drop_capture_entries reads the tables under it
        table[(device_index, stream)] = (capture, ws)
    return ws


def _forward_work(
    workspace: object,
    words: int,
    *,
    device: torch.device,
    device_index: int,
    capturing: Optional[bool] = None,
) -> Optional[torch.Tensor]:
    """The ``work`` counters of one forward launch needing ``words`` int32 words.

    A caller-owned ``workspace`` (1-D int32, contiguous, on ``device``, at least
    ``words`` words, zero before its first launch) is used as is; a program
    that takes no counters (``words`` 0: the forward's static and direct
    programs) gets none, so such a workspace is checked and not passed on (the
    record's argument plan binds no ``work``; the launch is the one without a
    workspace).  Without one:
    inside a CUDA Graph capture the capture's pre-zeroed buffer of the stream,
    else a fresh zeroed buffer (its zeroing is captured with the launch); the
    graph shares no counters with eager calls or other graphs
    (:func:`_capture_buffer`).  Else the zeroed buffer this module keeps for the
    device and the current stream, replaced by a larger zeroed one when it is
    too small (the stream-ordered allocator frees the old one after the
    launches queued on the stream); such an eager call also keeps the device's
    pre-zeroed capture buffers (:func:`_pool_refill`).  ``capturing`` is
    ``torch.cuda.is_current_stream_capturing()`` when the caller read it.
    ``None`` when the program takes no counters.
    """

    if workspace is not None:
        ws = _require_tensor(workspace, "workspace")
        if ws.ndim != 1:
            raise ValueError(
                f"workspace must be a 1-D int32 CUDA tensor, got shape {tuple(ws.shape)}"
            )
        _check_placement(ws, "workspace", device=device, dtype=torch.int32)
        if ws.numel() < words:
            raise ValueError(
                f"workspace holds {ws.numel()} int32 words; this call needs at least "
                f"{words}; allocate cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)"
            )
        return ws if words else None
    if words == 0:
        return None
    if capturing is None:
        capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        return _capture_buffer(_CAPTURE_FORWARD_WORKSPACES, device_index, "fwd", words, device)
    if words > _POOL_READY["fwd"].get(device_index, 0):
        _pool_refill(device_index, "fwd", words, device)
    key = (device_index, _current_stream_handle(device_index))
    held = _FORWARD_WORKSPACES.get(key)
    if held is None or held.numel() < words:
        held = _package_zeros(words, torch.int32, device, device_index)
        _FORWARD_WORKSPACES[key] = held
    return held


def _accumulator_layout(record: dict, C: int, n_ch_tiles: int) -> dict:
    """:func:`cake_jit.accumulator_layout` of ``record``, computed once per accumulator rule, ``C`` and
    ``n_ch_tiles`` (the returned mapping is shared: read it, never change it)."""

    rule = record["workspace_rule"]
    if rule is None:
        return accumulator_layout(record, C, n_ch_tiles)  # raises naming the record
    key = (
        rule["kind"],
        rule["acc"],
        rule["taps"],
        rule.get("slots"),  # a rule without it misses here and is refused below
        rule["work"],
        C,
        n_ch_tiles,
    )
    layout = _LAYOUTS.get(key)
    if layout is None:
        layout = _LAYOUTS[key] = accumulator_layout(record, C, n_ch_tiles)
    return layout


def _storage_view(
    ws: torch.Tensor, byte_offset: int, count: int, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """``count`` ``dtype`` elements at ``byte_offset`` of the uint8 tensor ``ws``, aliasing its storage without
    referencing ``ws`` itself; ``None`` when that offset is not a whole number of elements of the storage."""

    start = ws.storage_offset() + int(byte_offset)
    if start % dtype.itemsize:
        return None
    return torch.empty(0, dtype=dtype, device=ws.device).set_(
        ws.untyped_storage(), start // dtype.itemsize, (int(count),), (1,)
    )


def _drop_accumulator_views(ident: int) -> Callable[[weakref.ref], None]:
    """The weak-reference callback of the ``_ACCUMULATOR_VIEWS`` entry of the workspace with ``id`` ``ident``: it
    drops that entry, unless a later workspace with the same ``id`` already holds it."""

    def drop(ref: weakref.ref) -> None:
        hit = _ACCUMULATOR_VIEWS.get(ident)
        if hit is not None and hit[0] is ref:
            del _ACCUMULATOR_VIEWS[ident]

    return drop


def _accumulator_views(
    ws: torch.Tensor, layout: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(acc, work)`` tensors of ``layout`` inside the validated uint8 workspace ``ws``: cached per workspace
    and layout (see ``_ACCUMULATOR_VIEWS``)."""

    key = (
        layout["acc_dtype"],
        layout["acc_elems"],
        layout["work_dtype"],
        layout["work_offset"],
        layout["work_words"],
    )
    ptr = ws.data_ptr()
    ident = id(ws)
    hit = _ACCUMULATOR_VIEWS.get(ident)
    if hit is None or hit[0]() is not ws:
        per_layout: dict = {}
        _ACCUMULATOR_VIEWS[ident] = (
            weakref.ref(ws, _drop_accumulator_views(ident)),
            per_layout,
        )
    else:
        per_layout = hit[1]
        held = per_layout.get(key)
        if held is not None and held[0] == ptr:
            return held[1], held[2]
    acc_dtype = _TORCH_DTYPES[layout["acc_dtype"]]
    work_dtype = _TORCH_DTYPES[layout["work_dtype"]]
    acc = _storage_view(ws, 0, layout["acc_elems"], acc_dtype)
    work = _storage_view(ws, layout["work_offset"], layout["work_words"], work_dtype)
    if acc is None or work is None:
        # a storage offset off the element grid: plain views, not cached
        work_offset = int(layout["work_offset"])
        acc = ws[: int(layout["acc_bytes"])].view(acc_dtype)
        work = ws[work_offset : work_offset + 4 * int(layout["work_words"])].view(
            work_dtype
        )
        return acc, work
    per_layout[key] = (ptr, acc, work)
    return acc, work


def _backward_accumulators(
    workspace: object,
    layout: dict,
    *,
    device: torch.device,
    device_index: int,
    entry: str,
    capturing: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(acc, work)`` views of one backward launch with the byte ``layout`` of
    :func:`cake_jit.accumulator_layout` (shared by the single-rank and the KCP backward).

    A caller-owned ``workspace`` (1-D uint8, contiguous, 16-byte aligned, on
    ``device``, at least ``layout["total_bytes"]`` bytes, ZERO before its first
    launch: allocate it with :func:`cake_causal_conv1d_nondet_train_backward_workspace`)
    is used as is.  Without one: inside a CUDA Graph capture with a nonzero
    capture id the pre-zeroed buffer of that capture and stream
    (``_CAPTURE_WORKSPACES``, :func:`_capture_buffer`: a buffer the package
    zeroed before the capture; the graph shares no accumulators with eager
    calls or other graphs, and the capture's later calls on the stream reuse
    it; a larger one replaces it when it is too small), or, when the package
    has none ready, a fresh zeroed buffer of this call (a fill node), as inside
    a capture with capture id 0; else the zeroed buffer this module keeps for
    the device and the current
    stream, replaced by a larger zeroed one when it is too small (such an eager
    call also keeps the device's pre-zeroed capture buffers,
    :func:`_pool_refill`).
    Every launch leaves the bytes it uses zero, so a zeroed buffer stays zero
    whatever ``C`` or route the next call has.  ``acc`` is a ``float32`` view
    (the rule's dtype) of ``layout["acc_elems"]`` words at offset 0, ``work`` a
    ``uint32`` view of ``layout["work_words"]`` counters at
    ``layout["work_offset"]``; both are built once per workspace and layout
    (``_ACCUMULATOR_VIEWS``), except for the buffers of a capture.
    ``capturing`` is ``torch.cuda.is_current_stream_capturing()`` when the
    caller read it.
    """

    nbytes = int(layout["total_bytes"])
    if workspace is None and capturing is None:
        capturing = torch.cuda.is_current_stream_capturing()
    if workspace is not None:
        ws = _require_tensor(workspace, "workspace")
        if ws.ndim != 1:
            raise ValueError(
                f"workspace must be a 1-D uint8 CUDA tensor, got shape {tuple(ws.shape)}"
            )
        _check_placement(ws, "workspace", device=device, dtype=torch.uint8)
        if ws.data_ptr() % _loader.WORKSPACE_ALIGN != 0:
            raise ValueError(
                f"workspace must be {_loader.WORKSPACE_ALIGN}-byte aligned"
            )
        if ws.numel() < nbytes:
            raise ValueError(
                f"workspace holds {ws.numel()} bytes; {entry} needs at least {nbytes}; allocate "
                "cake_causal_conv1d_nondet_train_backward_workspace(S, C, device) (a zeroed buffer)"
            )
    elif capturing:
        # the buffer of this capture and stream: plain views, not cached per layout
        ws = _capture_buffer(_CAPTURE_WORKSPACES, device_index, "bwd", nbytes, device)
        work_offset = int(layout["work_offset"])
        acc = ws[: int(layout["acc_bytes"])].view(_TORCH_DTYPES[layout["acc_dtype"]])
        work = ws[work_offset : work_offset + 4 * int(layout["work_words"])].view(
            _TORCH_DTYPES[layout["work_dtype"]]
        )
        return acc, work
    else:
        if nbytes > _POOL_READY["bwd"].get(device_index, 0):
            _pool_refill(device_index, "bwd", nbytes, device)
        key = (device_index, _current_stream_handle(device_index))
        ws = _BACKWARD_WORKSPACES.get(key)
        if ws is None or ws.numel() < nbytes:
            ws = _package_zeros(nbytes, torch.uint8, device, device_index)
            _BACKWARD_WORKSPACES[key] = ws
    return _accumulator_views(ws, layout)


def cake_causal_conv1d_nondet_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    The forward has no cross-CTA reduction: this package's programs are copies
    of the det forward programs, so ``y`` is bitwise the det
    ``cake_causal_conv1d_det_train_forward``'s (the forward does not alert
    under torch's deterministic mode).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the
        current token, ``weight[c, 0]`` the token three positions back.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths (``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == S``; a repeated
        value is an empty document).  The host reads its metadata only.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias (its dtype is
        independent of ``weight``'s), added after the four taps.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned int32 buffer from
        :func:`cake_causal_conv1d_nondet_train_forward_workspace` (the
        token-chunk counters, zero before its first launch; every launch leaves
        it zero, so one buffer serves every later call of one stream).  Two
        launches that may run at once need two buffers.  When omitted the
        package uses a zeroed buffer it keeps per (device, current stream), or
        inside a CUDA Graph capture one buffer per capture and stream that an
        earlier eager forward call without ``workspace=`` zeroed (no fill node;
        the graph holds it until the graph is destroyed), or, when none is
        ready, a fresh zeroed one per call (one fill node each).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c] + bias[c]))`` with zero history before each document's
        first token.  ``S == 0`` returns the empty output without launching.
    """

    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device, dtypes=PARAMETER_DTYPES)
    b = None if bias is None else _bias(bias, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    if S == 0:
        return y
    bf16_taps = w.dtype == torch.bfloat16
    kind = forward_route(bias=b is not None, bf16_taps=bf16_taps)
    device_index = _device_index(device)
    names, _, scalars, grid, per_tiles = _call_plan(
        arch, kind, S, C, num_docs, device_index=device_index, bf16_taps=bf16_taps
    )
    capturing = torch.cuda.is_current_stream_capturing()
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "y": y.view(S, C),
        **scalars,
        "flags": _flags(w, b=b),
        "enable_pdl": pdl_enabled(arch, capturing=capturing),
    }
    if b is not None:
        values["b"] = b
    n_ch_tiles = scalars["n_ch_tiles"]
    words = per_tiles.get(n_ch_tiles)
    if words is None:
        words = per_tiles[n_ch_tiles] = max(
            counter_words(_loader.MODULES[name], n_ch_tiles) for name in names
        )
    work = _forward_work(
        workspace, words, device=device, device_index=device_index, capturing=capturing
    )
    if work is not None:
        values["work"] = work
    for name in names:
        run_module(
            name,
            values,
            device_index=device_index,
            grid=grid if name == names[0] else None,
        )
    return y


def cake_causal_conv1d_nondet_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    silu_grad_preactivation: str = "bf16",
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    dbias: Optional[torch.Tensor] = None,
    bwd_grid_mult: int = 1,
    dweight_sum: str = "fp32",
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward, ONE ordinary launch with unordered
    cross-CTA ``dweight`` / ``dbias`` reductions.

    Under ``torch.use_deterministic_algorithms(True)`` (read when the call runs) the call raises a
    ``RuntimeError`` (a ``UserWarning``, then it runs, with ``warn_only=True``) after its arguments are validated
    and its outputs are prepared (an output not passed is already allocated), before its kernel launches; it never
    falls back to the deterministic package.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_nondet_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        Where the SiLU derivative of the weight gradient is evaluated:
        ``"bf16"`` at the BF16-rounded recomputed preactivation with an FP32
        ``weight`` and at the unrounded FP32 preactivation with a BF16
        ``weight``, ``"fp32"`` at
        the unrounded FP32 preactivation; both accumulate in plain FP32 words.
        ``dx`` is within one BF16 ulp of exact either way, and bitwise the det
        package's.
    workspace : torch.Tensor, optional
        Caller-owned ZEROED ``uint8`` buffer from
        :func:`cake_causal_conv1d_nondet_train_backward_workspace` (the
        accumulators and ticket counters; every launch leaves it zero, so one
        buffer serves every later call of one stream; never share one between
        two launches that may run at once).  When omitted the package uses a zeroed buffer it keeps per
        (device, current stream), or inside a CUDA Graph capture one buffer per
        capture and stream, shared by the capture's backward calls on that
        stream, that an earlier eager backward call without ``workspace=``
        zeroed (no fill node; the graph holds it until the graph is destroyed
        and its launches have completed), or, when none is ready (no such
        eager call before the capture, all taken by earlier captures, too
        small, or their zero fill not complete yet: a
        ``CUDAGraph.capture_begin`` right after that eager call with no
        synchronization), a fresh zeroed one per call (one fill node each), as
        with capture id 0 (also without ``cuda.bindings``).
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted), contiguous.
        An FP32 ``dweight`` must be 16-byte aligned (each channel's four taps
        are stored with one 16-byte store; ``ValueError`` otherwise).
        ``dbias`` needs ``bias``.  Every element is overwritten.
    bwd_grid_mult : {1, 2, 3, 4}
        Opt-in finer static partition of the token axis (default 1, the
        default grid): the program launches ``bwd_grid_mult`` times its
        untrimmed grid, so each CTA streams that much fewer tokens.  ``dx`` is
        bitwise the default's; more CTAs add into the same accumulators, so
        ``dweight`` / ``dbias`` are summed the same way.  The workspace size does
        not depend on it.  ``TypeError`` for a non-int, ``ValueError`` for
        another value.
    dweight_sum : {"fp32", "fp32_pair"}
        The sum form of ``dweight`` / ``dbias``: ``"fp32"`` (default) one
        plain FP32 word per sum; ``"fp32_pair"`` (opt-in, with
        ``silu_grad_preactivation="fp32"`` only, ``ValueError`` otherwise; a
        non-``str`` is a ``TypeError``, another ``str`` a ``ValueError``) an
        FP32 (hi, lo) pair per sum -- the pair route of the call's ``_fp32g``
        route: the same ``dx``, the CTA's fold Kahan-compensated and every
        accumulator word a pair that keeps each add's rounding error, rounded
        once (the CTAs' order still varies, so the last bits are not bitwise
        reproducible run to run).  The default workspace serves it.

    Returns
    -------
    ``(dx, dweight)``, or ``(dx, dweight, dbias)`` when ``bias`` is given: BF16
    ``dx`` of ``x``'s shape (the det program's bits) and the ``[C, 4]`` weight
    gradient (and ``[C]`` bias gradient), accumulated across CTAs in an
    unordered order in plain FP32 words and rounded once into the
    requested dtype.  ``S == 0`` returns the empty ``dx`` and zero gradients
    without launching.
    """

    policy = _silu_grad_preactivation(silu_grad_preactivation)
    sum_form = _dweight_sum(dweight_sum, policy)
    bwd_grid_mult = _bwd_grid_mult(bwd_grid_mult)
    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device, dtypes=PARAMETER_DTYPES)
    b = None if bias is None else _bias(bias, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _dweight_aligned(
        _output(
            dweight,
            "dweight",
            shape=(C, WIDTH),
            dtype=w.dtype,
            device=device,
            dtypes=PARAMETER_DTYPES,
        )
    )
    if b is None:
        if dbias is not None:
            raise ValueError(
                "dbias is the bias gradient's output; pass bias to produce it"
            )
        db_out = None
    else:
        db_out = _output(
            dbias,
            "dbias",
            shape=(C,),
            dtype=b.dtype,
            device=device,
            dtypes=PARAMETER_DTYPES,
        )
    # Every argument is valid: a call under torch's deterministic mode is refused (or warned about) here, before
    # the kernel launches (outputs the caller did not pass are already allocated).
    _alert_nondeterministic("cake_causal_conv1d_nondet_train_backward")
    if S == 0:
        dw_out.zero_()
        if db_out is None:
            return dx_out, dw_out
        db_out.zero_()
        return dx_out, dw_out, db_out
    kind = backward_route(
        bias=b is not None,
        fp32_grad=policy == "fp32",
        bf16_taps=w.dtype == torch.bfloat16,
        bf16_dweight=dw_out.dtype == torch.bfloat16,
        pair=sum_form == "fp32_pair",
    )
    device_index = _device_index(device)
    names, record, scalars, grid, per_tiles = _call_plan(
        arch,
        kind,
        S,
        C,
        num_docs,
        device_index=device_index,
        policy=policy,
        bwd_grid_mult=bwd_grid_mult,
    )
    n_ch_tiles = scalars["n_ch_tiles"]
    layout = per_tiles.get(n_ch_tiles)
    if layout is None:
        layout = per_tiles[n_ch_tiles] = _accumulator_layout(record, C, n_ch_tiles)
    capturing = torch.cuda.is_current_stream_capturing()
    acc, work = _backward_accumulators(
        workspace,
        layout,
        device=device,
        device_index=device_index,
        entry="cake_causal_conv1d_nondet_train_backward",
        capturing=capturing,
    )
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "acc": acc,
        "work": work,
        **scalars,
        "flags": _flags(w, dw_out, b, db_out),
        # the backward passes its row: a row of ``PDL_OFF_ROWS`` launches without PDL in both modes
        "enable_pdl": pdl_enabled(arch, capturing=capturing, kind=kind, C=C, S=S),
    }
    if b is not None:
        values["b"] = b
        values["db"] = db_out
    run_module(names[0], values, device_index=device_index, grid=grid)
    if db_out is None:
        return dx_out, dw_out
    return dx_out, dw_out, db_out


class CakeCausalConv1dNondetFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x, weight, cu_seqlens, bias)`` for the backward.

    ``forward(x, weight, cu_seqlens, bias=None, silu_grad_preactivation="bf16", bwd_grid_mult=1,
    dweight_sum="fp32")`` returns ``y``;
    the backward runs the one-launch ``dx`` + unordered ``dweight`` (+ ``dbias``) program of
    :func:`cake_causal_conv1d_nondet_train_backward` (with ``bwd_grid_mult``; it alerts under torch's deterministic
    mode) on the incoming gradient (materialized as a contiguous BF16 tensor when autograd hands over another layout
    or dtype) and returns the gradients of ``weight`` and ``bias`` in the parameters' dtypes.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        cu_seqlens,
        bias=None,
        silu_grad_preactivation="bf16",
        bwd_grid_mult=1,
        dweight_sum="fp32",
    ):
        policy = _silu_grad_preactivation(silu_grad_preactivation)
        sum_form = _dweight_sum(dweight_sum, policy)
        bwd_grid_mult = _bwd_grid_mult(bwd_grid_mult)
        y = cake_causal_conv1d_nondet_train_forward(x, weight, cu_seqlens, bias=bias)
        ctx.save_for_backward(x, weight, cu_seqlens, bias)
        ctx.silu_grad_preactivation = policy
        ctx.bwd_grid_mult = bwd_grid_mult
        ctx.dweight_sum = sum_form
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, cu_seqlens, bias = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x)
        dy = grad_y.to(torch.bfloat16).contiguous()
        grads = cake_causal_conv1d_nondet_train_backward(
            x,
            weight,
            cu_seqlens,
            dy,
            bias=bias,
            silu_grad_preactivation=ctx.silu_grad_preactivation,
            bwd_grid_mult=ctx.bwd_grid_mult,
            dweight_sum=ctx.dweight_sum,
        )
        dbias = grads[2] if bias is not None else None
        return grads[0], grads[1], None, dbias, None, None, None


def cake_causal_conv1d_nondet(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
    bwd_grid_mult: int = 1,
    dweight_sum: str = "fp32",
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU, non-deterministic backward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations on a
        supported CUDA device.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias.
    silu_grad_preactivation : {"bf16", "fp32"}
        The weight gradient's SiLU' policy (see
        :func:`cake_causal_conv1d_nondet_train_backward`).
    bwd_grid_mult : {1, 2, 3, 4}
        The backward's opt-in finer partition (default 1; see
        :func:`cake_causal_conv1d_nondet_train_backward`); checked here, used by
        the backward.
    dweight_sum : {"fp32", "fp32_pair"}
        The sum form of ``dweight`` / ``dbias`` (default ``"fp32"``; the opt-in
        ``"fp32_pair"`` needs ``silu_grad_preactivation="fp32"``, see
        :func:`cake_causal_conv1d_nondet_train_backward`); checked here, used by
        the backward.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns BF16 ``dx`` and the ``dweight``
        (and ``dbias``) in the parameters' dtypes; under torch's deterministic
        mode it alerts like :func:`cake_causal_conv1d_nondet_train_backward`.
    """

    return CakeCausalConv1dNondetFunction.apply(
        x, weight, cu_seqlens, bias, silu_grad_preactivation, bwd_grid_mult, dweight_sum
    )


__all__ = [
    "ARCHES",
    "BWD_GRID_MULTS",
    "DWEIGHT_SUMS",
    "PARAMETER_DTYPES",
    "SILU_GRAD_PREACTIVATIONS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dNondetFunction",
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_plan_cache_clear",
    "is_cake_causal_conv1d_nondet_train_supported",
]
