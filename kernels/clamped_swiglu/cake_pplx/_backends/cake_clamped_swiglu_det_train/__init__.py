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

Backend of :mod:`cake_pplx.cake_clamped_swiglu_det_train`: the Cake clamped
SwiGLU of a routed MoE expert in training, forward and backward
(SM100 / SM103 / SM107), as one self-contained package (host API bodies, the
``cake_jit`` registry, the pinned ``cake_build`` loader and the generated CUDA
sources under ``csrc/``).  The ``@cake_api``-decorated entry points and the
user-facing documentation live in :mod:`cake_pplx.cake_clamped_swiglu_det_train`;
import that module.

Host design (every control is an argument; only a JIT build reads the shared
JIT variables):

- Operands.  ``gate`` and ``up`` are BF16 ``[R, 2304]``, as two tensors or as
  the two halves of one ``[R, 4608]`` tensor (gate columns ``[0, 2304)``, up
  ``[2304, 4608)``; pass it as the first argument with ``up=None``).  The
  kernels take every BF16 tensor's row stride in elements, so a merged tensor
  is read in place (the launch passes it as both the gate and the up, with the
  up's column offset 2304: the binding moves the up's pointer that many
  columns, so the host makes no view) and so are row slices of wider
  tensors.  An input the kernels cannot read (not 2-D with a unit last
  stride, a row stride that is not a multiple of 8 elements, a base address
  that is not 16-byte aligned) is copied once to a contiguous tensor.
  ``weight`` is FP32 ``[R]`` (copied once when not contiguous), or ``None``
  for an unweighted expert (``w = 1``; the forward kernels then get a cached
  one-element FP32 buffer of the device, set to 1, and read its one element
  and ignore it, and the backward runs the unweighted program, which takes no
  weight).  ``limit`` is ``L``, a positive finite Python number rounded to
  FP32; it is a launch argument, as are ``R``, the weight flag, the strides
  and the column offsets, so one library per program and architecture serves
  every call (the row count picks the program, and in the backward so does
  whether a weight is given).
- Checks.  An entry point checks every argument (the limit, the operands'
  types, dtypes, shapes and devices, the output buffers, the ``grad_up`` and
  ``grad_weight`` rules, ``R`` and the device's architecture) before it
  copies an input or allocates an output, so a call refused for its arguments
  copies, allocates and launches nothing.  The common call (plain contiguous CUDA tensors of the
  exact dtype and shape on one device, the BF16 ones 16-byte aligned,
  ``0 < R < 2**31 - 256``, output buffers None or plain contiguous tensors) is
  accepted in one pass over each tensor's metadata and launched by the step
  itself (:func:`_forward_step`, :func:`_backward_step`; in a checkpoint's
  recomputation the autograd forward keeps that launch for :func:`_held`, see
  Checkpointing): the first operand,
  the weight and ``dh`` are checked in the step's own frame, the ``up`` of two
  tensors and each output buffer by one :func:`_plain` call each, in the order
  of the full checks (the operands and ``dh``, then ``limit``, then the output
  buffers and their rules); its row strides are not read (a contiguous tensor
  of more than one row has the stride of its row width).  Any other call takes
  the full checks, which raise every error and mark the inputs to copy.  Both
  give the same launch.  A ``limit`` that is a
  Python float, or an int that a float holds exactly, equal to the float value
  of the last accepted one reuses its FP32 rounding (one cached pair); any
  other ``limit`` takes the full limit checks.  The
  autograd function's backward reuses the forward's FP32 ``L`` and checks the
  saved operands again (a saved-tensors hook or a ``.data`` assignment can
  change them after the forward) and the upstream gradient; it has no output
  buffer to check.
- Outputs.  ``h`` is a new contiguous BF16 ``[R, 2304]`` tensor, or the
  caller's ``out``.  The backward's ``d_gate`` and ``d_up`` have the layout of
  the inputs: two new contiguous ``[R, 2304]`` tensors, or one new contiguous
  ``[R, 4608]`` gradient for a merged input (passed as both ``d_gate`` and
  ``d_up``, with the column offset 2304).  ``d_weight`` is FP32 ``[R]`` for a
  weighted call; an unweighted backward has none, and its program writes none,
  so it allocates none.  A caller's output buffer must already be a layout the
  kernels write (2-D, unit last stride, a row stride that is a multiple of 8
  elements and at least the row width, 16-byte aligned; FP32 ``[R]``
  contiguous for the weight gradient): it is refused with a ``ValueError``
  otherwise, never copied.  An output must not overlap any input or another
  output (not checked).  A new output is ``torch.empty`` on the operands'
  device (one ``torch.device`` per device index for the process): a contiguous
  plain tensor.
- Launches.  One kernel per call: the forward runs stage ``fwd``, a weighted
  backward ``bwd`` and an unweighted backward ``bwd_unweighted``
  (``cake_jit.MODULES``).  Each stage has a small-R program and a fine
  program, ``fwd`` and ``bwd`` also have a large-R program, and ``bwd`` also
  has a relay program (role ``bwd_relay``).  The fine program runs when ``R <=
  hi * SMs`` for its record's one window ``[0, hi]`` of rows per SM on the
  device's architecture (``cake_jit.fine``).  At more rows a windowed program
  (the large-R or the relay one) runs when ``R`` lies in one of its record's
  windows, ``lo * SMs <= R <= hi * SMs`` for a window ``[lo, hi]`` (the
  windows of a stage's windowed programs are disjoint; ``cake_jit.route``),
  and the small-R program runs at every other ``R``.  The architecture is read
  once per device and the SM count once per stage and device; an architecture
  without a fine, a large-R or a relay program runs the small-R one in its
  place.  The programs of a stage give the same bits for every input, so the
  route changes no output bit, and the unweighted programs give the bits of
  ``bwd`` with the weight flag 0.  In the small-R and large-R ``fwd`` and
  ``bwd`` programs, CTAs of ``threads / lanes`` rows (32 lanes a row; 128
  threads and 4 rows, but 256 threads and 8 rows in the large-R forward on
  SM100 and SM103), one CTA per block of rows on the grid's x axis; in the
  small-R ``bwd_unweighted`` program, a warp per (row, run of 768 columns)
  task, 4 tasks a CTA of 128 threads, ``ceil(3 R / 4)`` CTAs.  The fine
  programs run a row as 9 tasks, one load of 256 columns of each input a task:
  a warp per task, 4 tasks a CTA of 128 threads, ``ceil(9 R / 4)`` CTAs, in
  ``fwd`` and ``bwd_unweighted``, and a CTA of 9 warps per row, ``R`` CTAs, in
  ``bwd``.  The relay program of ``bwd`` runs a row on a CTA of 3 warps (96
  threads), a warp per (row, run of 768 columns) task, ``R`` CTAs
  (``cake_jit.grid``).  ``R == 0`` launches nothing and loads no library; a
  program's library loads at its first launch.  Every launch is a plain
  stream-ordered launch on the current torch stream of the operands' device (a
  tvm-ffi call whose arguments include PyTorch CUDA tensors runs on that
  stream, a CUDA graph's capture stream during capture; the binding makes the
  device current): nothing cooperative, no PDL, no dynamic shared memory.  The
  entry points launch through one launcher per stage and device
  (``_launcher``), made at the stage's first launch on the device: the route
  above, the kernel parameters and the column offsets of ``up`` (and ``d_up``)
  by position in the order every program of the stage takes them (checked
  once, when the launcher is made), each program's FFI entry, bound at its
  first launch, the device of the new outputs and, in ``fwd``, the ignored
  weight, made by the first unweighted forward on the device.  Nothing here
  reads device memory, reads an environment variable or synchronizes; the
  checks read dtypes, shapes, strides, devices and data pointers only.  The
  host keeps no state between calls but caches of facts a call cannot change
  (a device's architecture, SM count and launchers, the libraries, the FP32
  rounding of the last accepted ``limit``, the forward's ignored weight).
  Under ``torch.use_deterministic_algorithms(True)`` with
  ``torch.utils.deterministic.fill_uninitialized_memory`` (on by default),
  PyTorch fills every tensor it allocates here with NaN before the kernel
  writes it: one more fill launch per new output (``h``; ``d_gate_up``, or
  ``d_gate`` and ``d_up``; a weighted call's ``d_weight``) and per copied
  input.  Output buffers allocated once and passed to every call avoid the
  fills of the outputs.  The kernels write every element of every output, so a
  fill never reaches a result.
- Numerics.  ``g' = min(g, L)`` and ``u' = clamp(u, -L, L)`` from the BF16
  values widened to FP32; ``s = 1 / (1 + 2^(-g' log2 e))`` with the
  approximate base-2 exponential (``ex2.approx.ftz``) and the approximate
  reciprocal plus one round-to-nearest Newton step (the IEEE ``rcp.rn`` of
  every ``1 <= 1 + e < 2^126``); a warp with a clamped gate at or below -87
  in a run of its loads computes that run with a scaled reciprocal that keeps
  the subnormal sigmoid (``RN(1 / (1 + e))`` where it is normal, within one
  subnormal ulp of it where it is subnormal, +0 for an infinite ``e``).
  ``silu = g' s`` and ``a = silu u'``; ``h = bf16(w a)``, one rounding.
  The backward: ``d_gate = bf16(((dh w) u') (silu (1 - s) + s) [g <= L])``
  and ``d_up = bf16((dh w) silu [-L <= u <= L])``, the masks inclusive and
  taken on the raw inputs, each output rounded once from FP32 (the unweighted
  program takes ``dh`` itself, the same value as ``dh 1``);
  ``d_weight = sum over c of dh a`` in FP32, from the unweighted ``a`` (no
  division by ``w``, so a zero-weight row keeps its sum).  Every product and
  FMA rounds to nearest without flushing subnormals; nothing is FP64.  A NaN
  input gives a NaN where the float64 operation has one; finite inputs give
  finite outputs wherever the float64 values are finite.
- Accuracy: the tests hold ``h``, ``d_gate``, ``d_up`` and ``d_weight`` to
  the PyTorch BF16 naive implementation of the same operation (the clamps as
  selects so that autograd gives the inclusive masks, the FP32 weight kept
  FP32, the gradients from PyTorch autograd), each against a float64
  reference computed from the BF16 inputs: over 32 random inputs of a case, in
  the L2 and in the max norm of the error, the median and the largest error of
  the output are each at most 1.005 times the naive implementation's, and 0 on
  every input where the naive implementation's is 0
  (``tests/cake/test_helpers/cake_naive.py``).  No output has an error budget
  of its own.
- Workspace: none with ``locality="off"`` (a ``"domain"`` forward's claim
  counters are the package's own, ``cake_domain``).  Retained state: the autograd function saves only its
  inputs (``gate`` and ``up`` or the merged tensor, and ``weight``) through
  ``save_for_backward``, references to the caller's tensors with no copy; the
  backward recomputes the activation.
  ``torch.utils.checkpoint(..., use_reentrant=False)`` keeps the same inputs;
  the function adds no saved state of its own and no activation.  The
  gradients have no derivative of their own: when the autograd engine records
  them (``create_graph=True``), differentiating them raises ``RuntimeError``.
- Checkpointing.  The backward of a non-reentrant checkpoint recomputes its
  region with saved-tensors hooks set.  There (a backward running on the
  thread, ``torch._C._current_graph_task_id() != -1``, and saved-tensors hooks
  set), :func:`cake_clamped_swiglu_det` applies :class:`_HeldFunction`, whose
  ``forward`` checks the arguments and allocates ``h`` but keeps the launch in
  its fifth argument (a :class:`_HeldLaunch`); :func:`_held` launches the
  forward kernel after ``apply`` returns.  So autograd saves the inputs (runs
  the recomputation's pack hooks) before the kernel runs, as for a built-in
  op: PyTorch packs a custom function's saved tensors only after its
  ``forward`` returns.  With early stop on (the default), the recomputation
  stops once it has saved again every tensor the region saved.  It ends
  before the launch when these inputs are the region's last saved tensors
  (the op alone, or the gate/up projection and the op, with the down
  projection outside the region), and launches the forward when a later
  operation of the region saves a tensor (the down projection saves ``h``; so
  does the whole expert block).  Anywhere else (no hooks, the checkpoint's own
  forward pass, activation offloading outside a backward)
  :class:`CakeClampedSwiGLUDetFunction` launches inside ``forward``, as in
  0.1.7.  A reentrant checkpoint (it sets no hooks), early stop off (the
  recomputation runs to its end) and a direct
  ``CakeClampedSwiGLUDetFunction.apply`` recompute the forward in every
  region.  A call that Dynamo or ``torch.jit.trace`` records launches inside
  ``forward``; a checkpoint's recomputation runs with Dynamo disabled.  Not
  tested under ``torch.compile``.  The launch, its arguments and its bits are
  the same in every case.  Checkpointing the op alone saves no memory: it
  keeps only its inputs anyway, and the down projection keeps ``h`` for its
  weight gradient.  A region of the gate/up projection and the op does not
  keep the projection's ``[R, 4608]`` output (the op's input) from the forward
  to the backward; one that also holds the down projection does not keep
  ``h``.
- Locality domains (``locality="domain"``, opt-in; ``"off"`` is the default
  and runs everything above unchanged).  An R200 (``sm_107a``) has two dies,
  each with its own SMs and memory; default memory alternates between them every
  4 KB.  :func:`cake_clamped_swiglu_det_train_domain_empty` allocates
  ``[R, 2304]`` (2 MB chunks) or merged ``[R, 4608]`` (4 MB chunks) BF16 tensors
  whose chunks alternate between the dies, row 0 at the start of a die-0 chunk,
  so a chunk holds 455.1 rows of either width and every such tensor of one row
  count changes die at the same rows (``cake_domain``).  A ``"domain"`` call of
  the forward runs the forward's twin (``cake_jit``: ``fwd_domain``) when the
  merged input or the gate (the anchor) lies in that memory on a slot of its
  chunk, the device has its SM table, no caller output overlaps an input, and
  ``R`` is at least the twin's first rows per SM times the SMs for the call's
  class: ``homed`` when every BF16 operand (the inputs and ``h``) lies in the
  anchor's row frame, else ``anchor`` (``cake_jit.DOMAIN_FIRST_ROWS_PER_SM``; a
  class without a value never runs the twin).  Every other call runs the
  ``"off"`` program with the same arguments, and never raises for its placement.
  The backward, weighted or unweighted, has no twin: with ``"domain"`` it runs
  the ``"off"`` program (reason ``"arch"``).  A new ``h`` is ``torch.empty``
  (ordinary memory), so a forward without ``out`` is of class ``anchor``; the
  autograd entry with ``locality="domain"`` allocates its outputs so.  A call
  with fewer rows than the first row count its class can reach before its
  placement is read (``anchor``'s for a new ``h``, the lower of the two with a
  caller's ``out``) runs the ``"off"`` program without reading the registry
  (:data:`_FLOORS`).  The twin
  stores the bits of the ``"off"`` programs for every input, grid and placement.
  A twin launch takes per-die claim counters from the package's workspaces
  (``cake_domain``), which every launch leaves zero, and the SM table probed by
  the device's first helper call.  The plans
  (:func:`cake_clamped_swiglu_det_train_forward_domain_plan`,
  :func:`cake_clamped_swiglu_det_train_backward_domain_plan`) report the route,
  its reason, the class and the launch without launching; the route reads the
  package's registry of the helper's ranges, with no driver call.
  Costs, measured on R200s of 212 SMs (this package's README lists them): where
  the twin runs, its route, claims and launch add 2.0 to 3.6 us of host time to
  an autograd step and 9.0 to 11.3 us to a checkpointed one, so an eager step
  whose host time exceeds its device time took longer than with ``"off"`` (an
  autograd step up to 4.5 % at 27,136 rows; a checkpointed one up to 6.6 % from
  27,136 rows until 49,152 with a merged input or 65,536 with two tensors), while
  its device time was shorter.  A forward between two GEMMs on inputs with a
  gate of -88 in each third of every row took up to 1.7 % longer near the first
  row counts, and up to 0.9 % longer at 98,304 rows than with ``"off"`` on
  default memory.  Below the first row counts a forward call takes up to about
  0.2 us more host time than with ``"off"``.
- Mode: the mode is part of the name (``det``).  Every output is elementwise
  or a sum within one row in one fixed order, so every launch gives the same
  bits for the same inputs on the same architecture, whatever the row count
  (and so the program it selects), the layout, the stream or a CUDA graph;
  ``torch.use_deterministic_algorithms`` changes no bit of any output (it
  adds only PyTorch's fills above), and no entry point takes a
  ``deterministic`` keyword (passing one is a ``TypeError``).  The family
  has no non-deterministic module.
"""

from __future__ import annotations

import functools
import math
import numbers
import operator
import struct
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch

from . import cake_jit
from .cake_jit import (
    ALIGNMENT_BYTES,
    ARCH_BY_CAPABILITY,
    ARCHES,
    MAX_ROWS,
    VECTOR_ELEMENTS,
    WIDTH,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))
# I: the intermediate width of the expert.
INTERMEDIATE_SIZE: int = WIDTH

_WHAT = "Cake clamped SwiGLU training"

# Whether an entry point first tries the one-pass checks of the common call (:func:`_forward_step`,
# :func:`_backward_step`, :func:`_fast_operands`, :func:`_plain`); off, every call takes the full checks.  The launches
# are the same either way.
_FAST_CHECKS = True


# ----------------------------------------------------------------------------------------------------------------
# Devices


def arch_for_capability(capability: tuple[int, int]) -> Optional[str]:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device is not an exported target); the
    capability is read once per device index."""
    arch = _arch_of_index(int(device_index))
    if arch is None:
        capability = tuple(torch.cuda.get_device_capability(device_index))
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported {_WHAT} target; the exported "
            f"architectures are {ARCHES} (capabilities {list(SUPPORTED_COMPUTE_CAPABILITIES)})"
        )
    return arch


def is_cake_clamped_swiglu_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _arch_of_index(int(index)) is not None


# ----------------------------------------------------------------------------------------------------------------
# Records, libraries and the launch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = cake_jit.MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"{_WHAT} record {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a builder
    compiled it, else the JIT build of the same sources and flags (``cake_build.load_library``)."""
    from . import cake_build  # noqa: PLC0415 - the loader, imported at the first load

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up builds, before a CUDA graph capture)."""
    from . import cake_build  # noqa: PLC0415 - the loader

    return cake_build.jit_specs(arch)


@functools.cache
def parameters(name: str) -> tuple[str, ...]:
    """The parameter names of record ``name`` in ``run`` order (its argument plan without the grid): the kernel's
    parameters, then the column offsets its binding takes (``up_col``, and ``d_up_col`` in the backward stages)."""
    names: list[str] = []
    axes: list[str] = []
    for kind, key in cake_jit.MODULES[name]["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        elif axes:
            raise ValueError(f"record {name}: parameter {key!r} follows the grid arguments")
        else:
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}")
    return tuple(names)


@functools.cache
def _entry(name: str, arch: str) -> Any:
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process)."""
    return getattr(load(name, arch), cake_jit.MODULES[name]["ffi_entry"])


@functools.cache
def _plan(name: str) -> tuple[Callable[[dict[str, Any]], tuple[Any, ...]], int, int]:
    """``(arguments, per_row, per_cta)`` of record ``name``, made once per record: ``arguments(values)`` is the tuple of
    its parameters in ``run`` order (:func:`parameters`: the kernel's, then the column offsets), a row is ``per_row``
    tasks and one CTA owns ``per_cta`` tasks (``cake_jit.cta_tasks``), so a launch of ``rows`` rows has
    ``ceil(rows * per_row / per_cta)`` CTAs (``cake_jit.grid``)."""
    per_cta, per_row = cake_jit.cta_tasks(name)
    return operator.itemgetter(*parameters(name)), per_row, per_cta


# A program a launch may run: ``(name, arguments, per_row, per_cta)`` (:func:`_plan`).
_Plan = tuple[str, Callable[[dict[str, Any]], tuple[Any, ...]], int, int]


@functools.cache
def _programs(stage: str, device_index: int) -> tuple[int, tuple[int, ...], tuple[_Plan, ...], _Plan]:
    """``(last, edges, runs, fine)`` of ``stage`` on device ``device_index``, read once per stage and device from the
    device's architecture and SM count: the last row count of the fine window and the fine program's plan
    (``cake_jit.fine``; 0 and the small-R plan without a fine program on the architecture), and the edges of the
    windows of the stage's windowed programs with the plan of each run between them (``cake_jit.route``: ``runs[0]``
    is the small-R plan and serves the row counts up to ``edges[0]``; for ``0 < i < len(edges)``, ``runs[i]`` serves
    those above ``edges[i - 1]`` up to ``edges[i]``; ``runs[-1]`` serves those above ``edges[-1]``.  Without a
    windowed program ``edges`` is empty and ``runs`` holds the small-R plan alone).  Each program has one plan, shared
    by its runs.  No library is loaded here: a program's library loads at its first launch."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    arch = device_arch(device_index)
    edges, names = cake_jit.route(stage, arch, sm_count)
    last, fine = cake_jit.fine(stage, arch, sm_count)
    plans = {name: (name, *_plan(name)) for name in dict.fromkeys((*names, fine))}
    return last, edges, tuple(plans[name] for name in names), plans[fine]


@functools.cache
def _route(stage: str, device_index: int) -> tuple[int, tuple[_Plan, ...], _Plan]:
    """``(limit, plans, tail)`` of ``stage`` on device ``device_index`` (:func:`_programs`), made once per stage and
    device.  A launch of ``rows < limit`` rows runs the plan ``plans[rows]``: the fine plan up to the fine window's
    last row count, then the plan of a windowed program in its windows and the small-R plan at every other row count.
    A launch of more rows runs ``tail``: the plan of the run above the last edge (a windowed program's when the last
    window has no upper bound, else the small-R one).  ``limit`` is one above the larger of the fine window's last row
    count and the last edge of ``cake_jit.route``.  So a launch compares the row count with one integer and reads one
    entry of a tuple, and a launch of more rows makes the comparison only.  The tuple holds a reference per row count
    below ``limit`` (8 bytes each: 0.79 MB for ``bwd`` on 152 SMs, the most of the delivered GPUs).  On Grace, a search of the edges per launch
    measured 1-3 % more host time per call."""
    last, edges, runs, fine = _programs(stage, device_index)
    limit = max(edges[-1] + 1 if edges else 0, last + 1)
    plans = [fine] * (last + 1) + [runs[0]] * (limit - last - 1)
    for below, top, plan in zip(edges, edges[1:], runs[1:]):
        plans[below + 1 : top + 1] = [plan] * (top - below)
    return limit, tuple(plans), runs[-1]


def run_module(stage: str, values: dict[str, Any], *, rows: int, device_index: int) -> None:
    """Launch the program of ``stage`` for ``rows`` rows (:func:`_route`: the fine program up to its last row count,
    a windowed program in its windows, the small-R one at every other row count) with its parameters (the kernel's,
    then the column offsets) named in ``values``, on the current torch stream of device ``device_index``.

    The first argument of every record is a CUDA tensor of that device: tvm-ffi runs the call on PyTorch's current
    stream of its device, and the binding makes that device current for the launch.  The grid is the uncapped
    ``ceil(rows * per_row / per_cta)`` CTAs of the program ``name`` that runs (:func:`_plan`).  It equals
    ``cake_jit.grid(name, rows)`` at every routed row count (``cake_jit.fine`` refuses a fine window whose grid passes
    ``cake_jit.MAX_GRID``).  The two differ only when ``fwd_fine`` or ``bwd_unweighted_fine`` is forced past its
    window to more than about 954 M rows: ``cake_jit.grid`` caps there, and the binding refuses the uncapped grid.  The
    entry points launch through :func:`_launcher`, which runs the same program with the same arguments and grid, passed
    by position."""
    limit, plans, tail = _route(stage, device_index)
    name, arguments, per_row, per_cta = plans[rows] if rows < limit else tail
    try:
        args = arguments(values)
    except KeyError:
        for key in parameters(name):
            if key not in values:
                raise ValueError(f"record {name} needs argument {key!r}") from None
        raise
    if rows < 1:
        cake_jit.grid(name, rows)  # raises: a launch needs at least one row
    _entry(name, device_arch(device_index))(*args, -(-rows * per_row // per_cta), 1, 1)


# The parameters the entry points pass to the launcher of each stage, in this order (:func:`_launcher`): the kernel's,
# then the column offsets of ``up`` (and ``d_up``), which the binding adds to those pointers in BF16 elements.
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "fwd": ("gate", "up", "w", "h", "rows", "gate_stride", "up_stride", "h_stride", "limit", "weighted", "up_col"),
    "bwd": (
        "gate",
        "up",
        "dh",
        "w",
        "d_gate",
        "d_up",
        "d_weight",
        "rows",
        "gate_stride",
        "up_stride",
        "dh_stride",
        "d_gate_stride",
        "d_up_stride",
        "limit",
        "weighted",
        "up_col",
        "d_up_col",
    ),
    "bwd_unweighted": (
        "gate",
        "up",
        "dh",
        "d_gate",
        "d_up",
        "rows",
        "gate_stride",
        "up_stride",
        "dh_stride",
        "d_gate_stride",
        "d_up_stride",
        "limit",
        "up_col",
        "d_up_col",
    ),
}


class _Launcher:
    """The launch of one stage on one device (:func:`_launcher`): ``launch(rows, *arguments)`` runs, for ``rows >= 1``
    rows, the program :func:`run_module` runs (:func:`_route`) with ``arguments``, the kernel parameters in the order
    of ``_ARGUMENTS[stage]``, then the uncapped grid of :func:`run_module`.  It holds what every launch of the stage on
    the device reads: ``device``, the device of the new outputs (one ``torch.device`` per device for the process:
    ``torch.empty`` on it costs about a third of the host time of ``Tensor.new_empty``); ``route``, :func:`_route` of
    the stage and device; ``entries``, each program's FFI entry, bound through :func:`_entry` at the program's first
    launch (:meth:`entry`; no library loads when the launcher is made); and ``weight``, the ignored weight of the
    unweighted forward launches on the device (:meth:`ignored_weight`; ``fwd`` only, None until the first one).  The
    programs of the stage are checked to take their parameters in that order when the launcher is made, so a launch
    builds no dict of the arguments.  The common calls of the entry points read these fields and launch in their own
    frame (:func:`_forward_step`, :func:`_backward_step`), as ``launch`` does; the autograd forward of a checkpoint's
    recomputation keeps its launch for :func:`_held` instead (:class:`_HeldLaunch`)."""

    __slots__ = ("arch", "device", "entries", "route", "weight")

    def __init__(self, stage: str, device_index: int) -> None:
        self.route = _route(stage, device_index)
        _, _, runs, fine = _programs(stage, device_index)
        for name in dict.fromkeys((*(run[0] for run in runs), fine[0])):
            if parameters(name) != _ARGUMENTS[stage]:
                raise RuntimeError(
                    f"record {name}: parameters {parameters(name)} are not in the order the host passes them, "
                    f"{_ARGUMENTS[stage]}"
                )
        self.arch = device_arch(device_index)
        self.device = _device(device_index)
        self.entries: dict[str, Any] = {}
        self.weight: Optional[torch.Tensor] = None

    def entry(self, name: str) -> Any:
        """The FFI entry of program ``name`` on the device's architecture, looked up once (:func:`_entry`)."""
        entry = self.entries.get(name)
        if entry is None:
            entry = self.entries[name] = _entry(name, self.arch)
        return entry

    def ignored_weight(self) -> torch.Tensor:
        """The ``w`` of an unweighted forward launch (``weighted`` 0: the kernels read its one element and ignore it):
        one one-element FP32 buffer per device for the process, set to 1, made by the first unweighted forward on the
        device and by no other call (a weighted call or a backward fills no buffer).  That call may be a CUDA graph
        capture: the buffer then lives in the graph's memory pool for the process, and its value is never used.  An
        unweighted backward runs ``bwd_unweighted``, which takes no weight."""
        weight = self.weight
        if weight is None:
            weight = self.weight = torch.ones(1, dtype=torch.float32, device=self.device)
        return weight

    def __call__(self, rows: int, *arguments: Any) -> None:
        limit, plans, tail = self.route
        name, _, per_row, per_cta = plans[rows] if rows < limit else tail
        entry = self.entries.get(name)
        if entry is None:
            entry = self.entry(name)
        entry(*arguments, -(-rows * per_row // per_cta), 1, 1)


@functools.cache
def _launcher(stage: str, device_index: int) -> _Launcher:
    """The launcher of ``stage`` on device ``device_index`` (:class:`_Launcher`), made once per stage and device at
    the stage's first launch on the device.  The entry points check the device's architecture at every call before
    they allocate; a launcher is made only for an exported architecture."""
    return _Launcher(stage, device_index)


@functools.cache
def _device(device_index: int) -> torch.device:
    """CUDA device ``device_index``, one object per device for the process: the device of the new outputs of a call
    (``_Launcher.device``)."""
    return torch.device("cuda", device_index)


# ----------------------------------------------------------------------------------------------------------------
# Checks: metadata only (dtypes, shapes, strides, devices, data pointers), nothing reads tensor contents.  Each
# check returns what it found; the entry points copy and allocate only after every check passed.


# The last limit :func:`_limit` accepted, as a Python float, and its FP32 rounding (NaN: none yet).  One pair, replaced
# whole, so a thread reads a value and its own rounding.
_last_limit: tuple[float, float] = (math.nan, math.nan)


def _limit(limit: Any) -> float:
    """``limit`` rounded to FP32 (the kernels' ``L``): a positive finite Python number.  A tensor is refused, as
    reading it would synchronize the host.  A ``float`` or ``int`` (exact types: no bool, no subclass) equal to the
    Python float value of the last accepted limit returns its rounding: Python compares an int and a float exactly, so
    an equal value has that float value and the same rounding.  An int that a float cannot hold exactly (such as
    ``2**53 + 1``) equals no float, so it takes the checks below at every call, as does any other value."""
    global _last_limit
    if type(limit) is float or type(limit) is int:
        value, rounded = _last_limit
        if limit == value:
            return rounded
    if isinstance(limit, (bool, torch.Tensor)) or not isinstance(limit, numbers.Real):
        raise TypeError(
            f"limit must be a Python float (a run-time FP32 scalar of the launch), got {type(limit).__name__}"
        )
    try:
        value = float(limit)
    except OverflowError:
        # An integer or a fraction past every Python float (its text is not shown: it can exceed the int-to-str limit).
        raise ValueError("limit is past the FP32 range (it has no Python float value)") from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"limit must be finite and positive, got {value!r}")
    try:
        # The standard "<f" format refuses a value past FP32 (OverflowError); the native "f" format would make it
        # infinite without an error.
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        raise ValueError(f"limit {value!r} is past the FP32 range") from None
    if rounded <= 0.0:
        raise ValueError(f"limit {value!r} rounds to 0 in FP32")
    _last_limit = (value, rounded)
    return rounded


def _check_tensor(tensor: Any, name: str, dtype: torch.dtype, device: Optional[torch.device]) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.device.type != "cuda":
        raise ValueError(f"{_WHAT} requires CUDA tensors; {name} is on {tensor.device}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    return tensor


def _plain(tensor: Any, dtype: torch.dtype, shape: tuple[int, ...], index: int) -> bool:
    """Whether ``tensor`` is a plain ``torch.Tensor`` (no subclass) of ``dtype`` and ``shape`` on CUDA device
    ``index``, contiguous, a BF16 one 16-byte aligned: one pass over its metadata.  Every full check accepts such a
    tensor as it is (no copy), so a ``True`` here gives the launch of the full checks."""
    return (
        type(tensor) is torch.Tensor
        and tensor.is_cuda
        and tensor.dtype is dtype
        and tensor.shape == shape
        and tensor.is_contiguous()
        and tensor.get_device() == index
        and (dtype is not torch.bfloat16 or tensor.data_ptr() % ALIGNMENT_BYTES == 0)
    )


def _readable(tensor: torch.Tensor) -> bool:
    """Whether the kernels read the 2-D BF16 ``tensor`` in place: unit last stride, a row stride that is a multiple
    of 8 elements (a single row's is never used), a 16-byte aligned base."""
    rows = tensor.shape[0]
    return (
        tensor.stride(1) == 1
        and (rows <= 1 or tensor.stride(0) % VECTOR_ELEMENTS == 0)
        and tensor.data_ptr() % ALIGNMENT_BYTES == 0
    )


def _input(
    tensor: Any, name: str, *, rows: Optional[int], cols: int, device: Optional[torch.device]
) -> tuple[torch.Tensor, bool]:
    """A BF16 ``[rows, cols]`` input, checked, and whether the kernels need it copied (it is not :func:`_readable`);
    the copy waits until every argument is checked (:func:`_copied`)."""
    tensor = _check_tensor(tensor, name, torch.bfloat16, device)
    if tensor.dim() != 2 or tensor.shape[1] != cols or (rows is not None and tensor.shape[0] != rows):
        want = f"[{'R' if rows is None else rows}, {cols}]"
        raise ValueError(f"{name} must have shape {want}, got {tuple(tensor.shape)}")
    return tensor, not (tensor.shape[0] == 0 or _readable(tensor))


def _copied(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous copy of an input the kernels cannot read in place."""
    return tensor.clone(memory_format=torch.contiguous_format)


def _output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    rows: int,
    cols: int,
    first: torch.Tensor,
    index: int,
) -> Optional[torch.Tensor]:
    """A caller's BF16 ``[rows, cols]`` output, checked: the kernels write it in place (rows that do not overlap), it
    is never copied; it must live on the device of ``first``, the call's checked first operand (on CUDA device
    ``index``).  None (a new contiguous output, allocated once every argument is checked) stays None; the entry
    points skip the call for None."""
    if tensor is None or (_FAST_CHECKS and _plain(tensor, torch.bfloat16, (rows, cols), index)):
        return tensor
    tensor = _check_tensor(tensor, name, torch.bfloat16, first.device)
    if tensor.dim() != 2 or tuple(tensor.shape) != (rows, cols):
        raise ValueError(f"{name} must have shape [{rows}, {cols}], got {tuple(tensor.shape)}")
    if rows and not (_readable(tensor) and (rows <= 1 or tensor.stride(0) >= cols)):
        raise ValueError(
            f"{name} must have a unit last stride, a row stride that is a multiple of {VECTOR_ELEMENTS} elements and "
            f"at least {cols}, and a {ALIGNMENT_BYTES}-byte aligned base; got strides {tuple(tensor.stride())} at "
            f"address {tensor.data_ptr():#x}"
        )
    return tensor


def _weight(weight: Any, *, rows: int, device: torch.device) -> tuple[Optional[torch.Tensor], bool]:
    """``weight`` checked (FP32 ``[rows]``; ``None`` stays) and whether the kernels need it copied (not contiguous)."""
    if weight is None:
        return None, False
    weight = _check_tensor(weight, "weight", torch.float32, device)
    if tuple(weight.shape) != (rows,):
        raise ValueError(f"weight must have shape [{rows}] (one route weight per row), got {tuple(weight.shape)}")
    return weight, not weight.is_contiguous()


def _grad_weight(
    tensor: Optional[torch.Tensor], *, rows: int, first: torch.Tensor, index: int
) -> Optional[torch.Tensor]:
    """A caller's FP32 ``[rows]`` weight gradient, checked (contiguous; never copied; on the device of ``first``, as
    for :func:`_output`).  None (a new one, allocated once every argument is checked) stays None."""
    if tensor is None or (_FAST_CHECKS and _plain(tensor, torch.float32, (rows,), index)):
        return tensor
    tensor = _check_tensor(tensor, "grad_weight", torch.float32, first.device)
    if tuple(tensor.shape) != (rows,) or not tensor.is_contiguous():
        raise ValueError(
            f"grad_weight must be a contiguous FP32 [{rows}] tensor, got shape {tuple(tensor.shape)} and strides "
            f"{tuple(tensor.stride())}"
        )
    return tensor


def _fast_operands(
    first: Any, up: Any, weight: Any
) -> Optional[tuple[int, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """``(index, R, first, up, weight)`` of the common call, found in one pass over each operand's metadata: plain
    contiguous tensors on CUDA device ``index`` (the merged BF16 ``[R, 4608]`` input when ``up`` is None, else two
    BF16 ``[R, 2304]`` tensors, both 16-byte aligned; ``weight`` FP32 ``[R]`` or None), ``0 < R < MAX_ROWS``.
    :func:`_operands` accepts such operands with no copy and gives the same tensors.  Any other call (an error, an
    input to copy, ``R = 0``, a tensor subclass) gives None, and :func:`_operands` runs.  No ``torch.device`` is
    read from a tensor: the caller takes the device's own (:func:`_device`)."""
    if type(first) is not torch.Tensor or not first.is_cuda or first.dtype is not torch.bfloat16:
        return None
    shape = first.shape
    if len(shape) != 2 or shape[1] != (2 * WIDTH if up is None else WIDTH):
        return None
    rows = shape[0]
    if not 0 < rows < MAX_ROWS or not first.is_contiguous() or first.data_ptr() % ALIGNMENT_BYTES:
        return None
    index = first.get_device()
    if up is not None and not _plain(up, torch.bfloat16, shape, index):
        return None
    if weight is not None and not _plain(weight, torch.float32, (rows,), index):
        return None
    return index, rows, first, up, weight


def _operands(
    gate_up_or_gate: Any, up: Any, weight: Any
) -> tuple[
    torch.device,
    int,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[tuple[bool, bool, bool]],
]:
    """``(device, R, first, up, weight, copies)`` of a call, checked: ``first`` the merged ``[R, 4608]`` input (``up``
    None) or the ``[R, 2304]`` gate, ``up`` the ``[R, 2304]`` up, ``weight`` FP32 ``[R]`` or None; ``copies`` marks
    which of the three the kernels cannot read in place (None: none), copied by :func:`_read` once every argument is
    checked."""
    first = _check_tensor(gate_up_or_gate, "gate_up_or_gate", torch.bfloat16, None)
    device = first.device
    if up is None:
        first, copy_first = _input(
            first, "gate_up_or_gate (the merged [R, 4608] gate and up)", rows=None, cols=2 * WIDTH, device=device
        )
        copy_up = False
    else:
        first, copy_first = _input(first, "gate_up_or_gate (the gate)", rows=None, cols=WIDTH, device=device)
        up, copy_up = _input(up, "up", rows=int(first.shape[0]), cols=WIDTH, device=device)
    rows = int(first.shape[0])
    if rows >= MAX_ROWS:
        raise ValueError(f"R = {rows} rows exceeds the kernels' 32-bit row count (fewer than {MAX_ROWS})")
    weight, copy_weight = _weight(weight, rows=rows, device=device)
    copies = (copy_first, copy_up, copy_weight) if copy_first or copy_up or copy_weight else None
    return device, rows, first, up, weight, copies


def _read(
    first: torch.Tensor,
    up: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    copies: tuple[bool, bool, bool],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The operands as the kernels read them: each one ``copies`` marks copied once to a contiguous tensor."""
    copy_first, copy_up, copy_weight = copies
    return (
        _copied(first) if copy_first else first,
        _copied(up) if copy_up else up,
        _copied(weight) if copy_weight else weight,
    )


# ----------------------------------------------------------------------------------------------------------------
# Entry points


def _forward(
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    out: Optional[torch.Tensor],
    held: Optional[_HeldLaunch] = None,
) -> tuple[torch.Tensor, float]:
    """The forward step and its FP32 ``L``: every argument checked, then the copies of the inputs the kernels cannot
    read, the new output and the launch (with ``held``, a :class:`_HeldLaunch`, appended there as ``(launcher,
    arguments)`` for :func:`_held` to run)."""
    fast = _fast_operands(gate_up_or_gate, up, weight) if _FAST_CHECKS else None
    if fast is None:
        device, rows, first, second, w, copies = _operands(gate_up_or_gate, up, weight)
        index = int(device.index)
    else:
        index, rows, first, second, w = fast
        device, copies = _device(index), None
    L = _limit(limit)
    h = out if out is None else _output(out, "out", rows=rows, cols=WIDTH, first=first, index=index)
    if _arch_of_index(index) is None:
        device_arch(index)  # raises: not an exported architecture
    if copies:
        first, second, w = _read(first, second, w, copies)
    if h is None:
        h = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
    if rows == 0:
        return h, L
    # A merged input is both the gate (its first 2304 columns, at its own row stride) and the up, from column 2304 of
    # each row on: the binding adds the up's column offset to its pointer, so no view is made.  A single row's strides
    # are 0 (never dereferenced).
    many = rows > 1
    gate_stride = first.stride(0) if many else 0
    if second is None:
        up_tensor, up_stride, up_col = first, gate_stride, WIDTH
    else:
        up_tensor, up_stride, up_col = second, second.stride(0) if many else 0, 0
    launch = _launcher("fwd", index)
    arguments = (
        rows,
        first,
        up_tensor,
        launch.ignored_weight() if w is None else w,
        h,
        rows,
        gate_stride,
        up_stride,
        h.stride(0) if many else 0,
        L,
        0 if w is None else 1,
        up_col,
    )
    if held is None:
        launch(*arguments)
    else:
        held.append((launch, arguments))
    return h, L


def _backward(
    grad_out: Any,
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    grad_gate_up_or_gate: Optional[torch.Tensor],
    grad_up: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    *,
    rounded: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The backward step: every argument checked, then the copies of the inputs the kernels cannot read, the new
    outputs and the launch.  ``rounded``: ``limit`` is already the FP32 ``L`` of a checked forward (the autograd
    function's backward), used as it is."""
    fast = _fast_operands(gate_up_or_gate, up, weight) if _FAST_CHECKS else None
    if fast is None:
        device, rows, first, second, w, copies = _operands(gate_up_or_gate, up, weight)
        index = int(device.index)
    else:
        index, rows, first, second, w = fast
        device, copies = _device(index), None
    if _FAST_CHECKS and _plain(grad_out, torch.bfloat16, (rows, WIDTH), index):
        dh, copy_dh = grad_out, False
    else:
        dh, copy_dh = _input(grad_out, "grad_out", rows=rows, cols=WIDTH, device=device)
    L = limit if rounded else _limit(limit)
    cols = 2 * WIDTH if second is None else WIDTH
    if second is None and grad_up is not None:
        raise ValueError("grad_up must be None for a merged input: grad_gate_up_or_gate receives both halves")
    d_first = grad_gate_up_or_gate
    if d_first is not None:
        d_first = _output(d_first, "grad_gate_up_or_gate", rows=rows, cols=cols, first=first, index=index)
    d_second = grad_up
    if d_second is not None:
        d_second = _output(d_second, "grad_up", rows=rows, cols=WIDTH, first=first, index=index)
    d_weight = grad_weight
    if d_weight is not None:
        if w is None:
            raise ValueError("grad_weight needs a weight: an unweighted call has no weight gradient")
        d_weight = _grad_weight(d_weight, rows=rows, first=first, index=index)
    if _arch_of_index(index) is None:
        device_arch(index)  # raises: not an exported architecture
    if copies:
        first, second, w = _read(first, second, w, copies)
    if copy_dh:
        dh = _copied(dh)
    if d_first is None:
        d_first = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
    if second is not None and d_second is None:
        d_second = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
    if w is not None and d_weight is None:  # an unweighted call has no weight gradient: none is allocated
        d_weight = torch.empty(rows, dtype=torch.float32, device=device)
    result = (d_first, d_second, d_weight)
    if rows == 0:
        return result
    # A merged input and its gradient are both the gate and the up, and both d_gate and d_up, the up halves from column
    # 2304 of each row on: the binding adds the column offsets to the up and d_up pointers, so no view is made.  A
    # single row's strides are 0 (never dereferenced).
    many = rows > 1
    gate_stride = first.stride(0) if many else 0
    d_gate_stride = d_first.stride(0) if many else 0
    if second is None:
        up_tensor, up_stride, up_col = first, gate_stride, WIDTH
        d_up, d_up_stride, d_up_col = d_first, d_gate_stride, WIDTH
    else:
        up_tensor, up_stride, up_col = second, second.stride(0) if many else 0, 0
        d_up, d_up_stride, d_up_col = d_second, d_second.stride(0) if many else 0, 0
    dh_stride = dh.stride(0) if many else 0
    if w is None:
        # The unweighted program: no weight, no weight flag and no weight gradient; the bits of ``bwd`` at ``w = 1``.
        _launcher("bwd_unweighted", index)(
            rows,
            first,
            up_tensor,
            dh,
            d_first,
            d_up,
            rows,
            gate_stride,
            up_stride,
            dh_stride,
            d_gate_stride,
            d_up_stride,
            L,
            up_col,
            d_up_col,
        )
        return result
    _launcher("bwd", index)(
        rows,
        first,
        up_tensor,
        dh,
        w,
        d_first,
        d_up,
        d_weight,
        rows,
        gate_stride,
        up_stride,
        dh_stride,
        d_gate_stride,
        d_up_stride,
        L,
        1,
        up_col,
        d_up_col,
    )
    return result


# The common call of a step, checked and launched by the step itself (:func:`_forward_step`, :func:`_backward_step`;
# the autograd forward of a checkpoint's recomputation keeps the launch for :func:`_held`):
# plain contiguous operands of the exact dtypes and shapes on one CUDA device (the BF16 ones 16-byte aligned), the
# merged BF16 ``[R, 4608]`` input or two BF16 ``[R, 2304]`` tensors, ``0 < R < MAX_ROWS``, and outputs that are None
# or plain contiguous tensors of their shapes: what :func:`_fast_operands` and :func:`_plain` accept.  The first
# operand, the weight and ``dh`` are checked in the step's own frame, the ``up`` of two tensors and each output buffer
# by one :func:`_plain` call each, so only a merged input with new outputs, whose limit is a Python float, or an int
# that a float holds exactly, equal to the float value of the last accepted limit, is checked and launched in exactly
# one Python frame per step once the host caches are filled (any other limit adds one :func:`_limit` call; a call that
# fills a host cache, such as the first launch of a program on a device, adds the calls that fill it).  The checks
# keep the order of the full checks: the operands and ``dh``, then ``limit``, then the output buffers and their rules,
# then the device's architecture; an output buffer that is not plain sends the call, its limit checked, to the full
# checks.  A contiguous tensor of more than one row has the row stride of its row width, so its row stride is that
# constant and is not read; a single row's strides are 0 (never dereferenced).  Any other call runs :func:`_forward`
# or :func:`_backward`, whose checks raise every error and mark the inputs to copy; both give the same launch.
_BF16 = torch.bfloat16
_FP32 = torch.float32
_MERGED_WIDTH = 2 * WIDTH


def _forward_step(
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    out: Optional[torch.Tensor],
    held: Optional[_HeldLaunch] = None,
) -> torch.Tensor:
    """The forward step's ``h``: the common call checked and launched here, any other call through :func:`_forward`.
    The checks and the launch are those of :func:`_forward`: the operands, then ``limit``, then ``out`` (one that is
    not plain goes, with the limit checked, to :func:`_forward`), then the device's architecture, then the new output
    and the launch.  With ``held`` (the autograd forward of a checkpoint's recomputation, :class:`_HeldFunction`), the
    launch is appended there as ``(entry, arguments)`` instead, for :func:`_held` to run."""
    first = gate_up_or_gate
    if _FAST_CHECKS and type(first) is torch.Tensor and first.is_cuda and first.dtype is _BF16:
        shape = first.shape
        if len(shape) == 2 and shape[1] == (_MERGED_WIDTH if up is None else WIDTH):
            rows = shape[0]
            if 0 < rows < MAX_ROWS and first.is_contiguous() and not first.data_ptr() % ALIGNMENT_BYTES:
                index = first.get_device()
                if (up is None or _plain(up, _BF16, shape, index)) and (
                    weight is None
                    or (
                        type(weight) is torch.Tensor
                        and weight.is_cuda
                        and weight.dtype is _FP32
                        and weight.shape == (rows,)
                        and weight.is_contiguous()
                        and weight.get_device() == index
                    )
                ):
                    if type(limit) is float or type(limit) is int:  # :func:`_limit`'s cached rounding, inlined
                        value, L = _last_limit
                        if limit != value:
                            L = _limit(limit)
                    else:
                        L = _limit(limit)
                    if out is not None and not _plain(out, _BF16, (rows, WIDTH), index):
                        # ``out`` after ``limit``, as in :func:`_forward`, which checks this ``out`` in full.
                        return _forward(gate_up_or_gate, up, weight, limit, out, held)[0]
                    if _arch_of_index(index) is None:
                        device_arch(index)  # raises: not an exported architecture
                    launch = _launcher("fwd", index)
                    if out is None:
                        out = torch.empty(rows, WIDTH, dtype=_BF16, device=launch.device)
                    if weight is None:
                        weight, weighted = launch.weight, 0
                        if weight is None:
                            weight = launch.ignored_weight()
                    else:
                        weighted = 1
                    # The gate and the up share the row stride: the merged tensor's (passed as both, the up from
                    # column 2304) or the two tensors' 2304.
                    stride, out_stride = (shape[1], WIDTH) if rows > 1 else (0, 0)
                    limit_rows, plans, tail = launch.route
                    name, _, per_row, per_cta = plans[rows] if rows < limit_rows else tail
                    entry = launch.entries.get(name)
                    if entry is None:
                        entry = launch.entry(name)
                    if up is None:
                        up, up_col = first, WIDTH
                    else:
                        up_col = 0
                    if held is not None:  # the same arguments as the launch below, kept for :func:`_held`
                        held.append(
                            (
                                entry,
                                (
                                    first,
                                    up,
                                    weight,
                                    out,
                                    rows,
                                    stride,
                                    stride,
                                    out_stride,
                                    L,
                                    weighted,
                                    up_col,
                                    -(-rows * per_row // per_cta),
                                    1,
                                    1,
                                ),
                            )
                        )
                        return out
                    entry(
                        first,
                        up,
                        weight,
                        out,
                        rows,
                        stride,
                        stride,
                        out_stride,
                        L,
                        weighted,
                        up_col,
                        -(-rows * per_row // per_cta),
                        1,
                        1,
                    )
                    return out
    return _forward(gate_up_or_gate, up, weight, limit, out, held)[0]


def _backward_step(
    grad_out: Any,
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    grad_gate_up_or_gate: Optional[torch.Tensor],
    grad_up: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    rounded: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The backward step: the common call checked and launched here, any other call through :func:`_backward`
    (``rounded`` as there).  The checks and the launch are those of :func:`_backward`: the operands and ``grad_out``,
    then ``limit``, then the output buffers and their rules in its order (a buffer that is not plain, or a broken rule,
    goes with the limit checked to :func:`_backward`), then the device's architecture, then the new outputs and the
    launch (``bwd`` for a weighted call, ``bwd_unweighted`` for an unweighted one)."""
    first, dh = gate_up_or_gate, grad_out
    if _FAST_CHECKS and type(first) is torch.Tensor and first.is_cuda and first.dtype is _BF16:
        shape = first.shape
        if len(shape) == 2 and shape[1] == (_MERGED_WIDTH if up is None else WIDTH):
            rows, cols = shape
            if 0 < rows < MAX_ROWS and first.is_contiguous() and not first.data_ptr() % ALIGNMENT_BYTES:
                index = first.get_device()
                if (
                    (up is None or _plain(up, _BF16, shape, index))
                    and (
                        weight is None
                        or (
                            type(weight) is torch.Tensor
                            and weight.is_cuda
                            and weight.dtype is _FP32
                            and weight.shape == (rows,)
                            and weight.is_contiguous()
                            and weight.get_device() == index
                        )
                    )
                    and type(dh) is torch.Tensor
                    and dh.is_cuda
                    and dh.dtype is _BF16
                    and dh.shape == (rows, WIDTH)
                    and dh.is_contiguous()
                    and dh.get_device() == index
                    and not dh.data_ptr() % ALIGNMENT_BYTES
                ):
                    if rounded:
                        L = limit
                    elif type(limit) is float or type(limit) is int:  # :func:`_limit`'s cached rounding, inlined
                        value, L = _last_limit
                        if limit != value:
                            L = _limit(limit)
                    else:
                        L = _limit(limit)
                    d_first, d_up, d_weight = grad_gate_up_or_gate, grad_up, grad_weight
                    # In the order of :func:`_backward`: the ``grad_up`` rule, then ``grad_gate_up_or_gate``, then
                    # ``grad_up``, then the ``grad_weight`` rule, then ``grad_weight``.
                    if (
                        (d_up is None or up is not None)
                        and (d_first is None or _plain(d_first, _BF16, shape, index))
                        and (d_up is None or _plain(d_up, _BF16, shape, index))
                        and (d_weight is None or (weight is not None and _plain(d_weight, _FP32, (rows,), index)))
                    ):
                        if _arch_of_index(index) is None:
                            device_arch(index)  # raises: not an exported architecture
                        launch = _launcher("bwd_unweighted" if weight is None else "bwd", index)
                        device = launch.device
                        if d_first is None:
                            d_first = torch.empty(rows, cols, dtype=_BF16, device=device)
                        if up is not None and d_up is None:
                            d_up = torch.empty(rows, WIDTH, dtype=_BF16, device=device)
                        if weight is not None and d_weight is None:
                            d_weight = torch.empty(rows, dtype=_FP32, device=device)
                        result = (d_first, d_up, d_weight)
                        # The gate, the up and their gradients share the row stride: the merged tensors' (each
                        # passed as both halves, the up halves from column 2304) or the two tensors' 2304.
                        stride, dh_stride = (cols, WIDTH) if rows > 1 else (0, 0)
                        if up is None:
                            up, d_up, col = first, d_first, WIDTH
                        else:
                            col = 0
                        limit_rows, plans, tail = launch.route
                        name, _, per_row, per_cta = plans[rows] if rows < limit_rows else tail
                        entry = launch.entries.get(name)
                        if entry is None:
                            entry = launch.entry(name)
                        blocks = -(-rows * per_row // per_cta)
                        if weight is None:
                            # The unweighted program: no weight, no weight flag and no weight gradient; the bits of
                            # ``bwd`` at ``w = 1``.
                            entry(
                                first,
                                up,
                                dh,
                                d_first,
                                d_up,
                                rows,
                                stride,
                                stride,
                                dh_stride,
                                stride,
                                stride,
                                L,
                                col,
                                col,
                                blocks,
                                1,
                                1,
                            )
                        else:
                            entry(
                                first,
                                up,
                                dh,
                                weight,
                                d_first,
                                d_up,
                                d_weight,
                                rows,
                                stride,
                                stride,
                                dh_stride,
                                stride,
                                stride,
                                L,
                                1,
                                col,
                                col,
                                blocks,
                                1,
                                1,
                            )
                        return result
    return _backward(
        grad_out,
        gate_up_or_gate,
        up,
        weight,
        limit,
        grad_gate_up_or_gate,
        grad_up,
        grad_weight,
        rounded=rounded,
    )


# ----------------------------------------------------------------------------------------------------------------
# Locality domains (``locality="domain"``): the forward's twin (``cake_jit``) on memory whose chunks alternate between
# the two dies of an R200.  Nothing here runs for ``locality="off"``: the off steps above are not edited, and the entry
# points compare ``locality`` with ``"off"`` once before them.

# The values of ``locality``.
_LOCALITIES = ("off", "domain")
# The slots of a row of every tensor of the helper's memory (``cake_domain.slot_of``'s row frame).
_ROW_SLOTS = cake_jit.DOMAIN_ROW_SLOTS


def _domain_requested(locality: object) -> bool:
    """Whether ``locality`` asks for the locality-domain route: ``"domain"``; ``"off"`` does not; a non-``str`` is a
    ``TypeError`` and any other string a ``ValueError``."""
    if not isinstance(locality, str):
        raise TypeError(f"locality must be a str, one of {_LOCALITIES}, got {type(locality).__name__}")
    if locality not in _LOCALITIES:
        raise ValueError(f"locality must be one of {_LOCALITIES}, got {locality!r}")
    return locality == "domain"


@functools.cache
def _domain() -> Any:
    """The module of the memory that alternates dies, the SM tables and the claim workspaces (``cake_domain``),
    imported on the first ``"domain"`` call that reads a placement and on the first helper call: an ``"off"`` call, and
    a ``"domain"`` call refused for its architecture or its row count, import nothing of it."""
    from . import cake_domain  # noqa: PLC0415 - only the locality-domain route needs it

    return cake_domain


# The parameters of a stage's twin (only ``fwd`` has one), in launch order: its base program's kernel parameters, the
# claim counters of both dies, the twin's geometry, the SM table and the column offsets.  ``counters0`` and
# ``counters1`` come from the launch's claim workspace (``cake_domain.launch``).
_DOMAIN_ARGUMENTS: dict[str, tuple[str, ...]] = {
    stage: (
        *(key for key in _ARGUMENTS[stage] if key not in ("up_col", "d_up_col")),
        "counters0",
        "counters1",
        "first_slot",
        "first_die",
        "chunk_shift",
        "item_rows",
        "sm_table",
        *(key for key in _ARGUMENTS[stage] if key in ("up_col", "d_up_col")),
    )
    for stage in _ARGUMENTS
}


class _Twin:
    """The twin of one stage on one device (:func:`_twin`): ``name``, its record's ``chunk_shift`` and ``item_rows``,
    ``resident`` (its CTAs per SM times the device's SMs, the largest grid), ``saturated`` (``resident * item_rows``:
    from that many rows on the grid is ``resident``, :func:`_twin_grid`), ``words`` (its claim counter words a die),
    ``homed`` and ``anchor`` (the first row count of each route class on the device, ``rows_per_sm * SMs``, or None
    where the class never runs the twin), ``lowest`` (the smaller of the two, or None) and ``entry``, its FFI entry,
    bound at its first launch.  Its parameters are checked to be in :data:`_DOMAIN_ARGUMENTS` order when it is made."""

    __slots__ = (
        "anchor",
        "arch",
        "chunk_shift",
        "entry",
        "homed",
        "item_rows",
        "lowest",
        "name",
        "resident",
        "saturated",
        "words",
    )

    def __init__(self, stage: str, device_index: int, name: str) -> None:
        if parameters(name) != _DOMAIN_ARGUMENTS[stage]:
            raise RuntimeError(
                f"record {name}: parameters {parameters(name)} are not in the order the host passes them, "
                f"{_DOMAIN_ARGUMENTS[stage]}"
            )
        self.arch = device_arch(device_index)
        self.name = name
        self.chunk_shift, self.item_rows, ctas = cake_jit.domain_rule(name, self.arch)
        sm_count = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
        firsts = cake_jit.DOMAIN_FIRST_ROWS_PER_SM[stage][self.arch]
        self.homed = None if firsts["homed"] is None else firsts["homed"] * sm_count
        self.anchor = None if firsts["anchor"] is None else firsts["anchor"] * sm_count
        lows = [value for value in (self.homed, self.anchor) if value is not None]
        self.lowest = min(lows) if lows else None
        self.resident = ctas * sm_count
        self.saturated = self.resident * self.item_rows
        self.words = int(cake_jit.MODULES[name]["counters"]["words"])
        self.entry: Any = None

    def bound(self) -> Any:
        """The twin's FFI entry, looked up once (:func:`_entry`)."""
        entry = self.entry
        if entry is None:
            entry = self.entry = _entry(self.name, self.arch)
        return entry


@functools.cache
def _twin(stage: str, device_index: int) -> Optional[_Twin]:
    """The twin of ``stage`` on device ``device_index`` (:class:`_Twin`), or None when the device's architecture has
    none (``cake_jit.domain_program``): made once per stage and device, at the stage's first ``"domain"`` call on the
    device.  No library loads here."""
    name = cake_jit.domain_program(stage, device_arch(device_index))
    return None if name is None else _Twin(stage, device_index, name)


# Per device, the row counts from which a ``"domain"`` forward call can take the twin, known before any placement is
# read (:func:`_floors`): ``(caller, new, twin)``.  ``caller`` is the lower of the twin's two first row counts (a call
# with the caller's ``out``, of either class); ``new`` is the ``anchor`` class's (a call that allocates ``h``, which is
# ordinary memory, so the call is of class ``anchor``: the autograd forward and a step without ``out``); each is
# ``MAX_ROWS`` (more rows than any call has) where no class the call can be of runs the twin; ``twin`` is the forward's
# :func:`_twin`, None on an architecture without one.  A call with fewer rows runs the ``"off"`` program without a
# registry read, as :func:`_domain_route` and the plans route it.
_FLOORS: dict[int, tuple[int, int, Optional[_Twin]]] = {}


def _floors(device_index: int) -> tuple[int, int, Optional[_Twin]]:
    """``_FLOORS[device_index]`` from the forward's twin on the device (its first row counts, ``rows_per_sm * SMs``),
    made at the device's first ``"domain"`` forward call, after the call's architecture check."""
    twin = _twin("fwd", device_index)
    if twin is None:
        floors: tuple[int, int, Optional[_Twin]] = (MAX_ROWS, MAX_ROWS, None)
    else:
        floors = (
            MAX_ROWS if twin.lowest is None else twin.lowest,
            MAX_ROWS if twin.anchor is None else twin.anchor,
            twin,
        )
    _FLOORS[device_index] = floors
    return floors


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    """The byte interval ``[start, end)`` the elements of ``tensor`` (non-negative strides) lie in."""
    start = tensor.data_ptr()
    extent = sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    return start, start + (extent + 1) * tensor.element_size()


def _overlaps(outputs: tuple[Optional[torch.Tensor], ...], inputs: tuple[Optional[torch.Tensor], ...]) -> bool:
    """Whether the byte span of a caller's output meets the byte span of an input (each output against each input
    tensor; the two halves of one merged tensor are one tensor and never compared with each other)."""
    spans = [_span(tensor) for tensor in inputs if tensor is not None]
    for tensor in outputs:
        if tensor is not None:
            start, end = _span(tensor)
            if any(start < top and low < end for low, top in spans):
                return True
    return False


# The result of :func:`_domain_route`: ``(twin, reason, klass, place, homed)``.
_Route = tuple[Optional[_Twin], str, Optional[str], Optional[tuple[int, int]], Optional[dict[str, bool]]]


def _domain_route(
    stage: str,
    device_index: int,
    rows: int,
    anchor: tuple[str, torch.Tensor],
    others: tuple[tuple[str, torch.Tensor], ...],
    outputs: tuple[tuple[str, Optional[torch.Tensor]], ...],
    inputs: tuple[Optional[torch.Tensor], ...],
    callers: tuple[Optional[torch.Tensor], ...],
    copied: frozenset[str] = frozenset(),
) -> _Route:
    """The route of a ``locality="domain"`` call of ``stage`` with ``rows >= 1`` rows on device ``device_index``:
    ``(twin, "placed", class, place, homed)`` for the twin, else ``(None, reason, class, place, homed)`` for the
    program the call runs with ``locality="off"`` (the same bits).  ``anchor`` is the merged input or the gate,
    ``others`` the other BF16 inputs (the up of two tensors) and ``outputs`` the BF16 outputs (None: a new
    output the call allocates with ``torch.empty``), each as ``(label, tensor)``; ``inputs`` are every input tensor and
    ``callers`` the caller's output buffers, for the overlap check.  A label in ``copied`` is an operand the call copies
    first (the copy is ordinary memory).  The reasons, in this order:

    1. ``"arch"``: the architecture has no twin of the stage;
    2. ``"size"``: ``rows`` is below the lower of the twin's first row counts, or above ``cake_jit.DOMAIN_MAX_ROWS``;
    3. ``"sm_table"``: the device has no SM table (no :func:`cake_clamped_swiglu_det_train_domain_empty` call ran on
       it, or its probe failed);
    4. ``"input_memory"``: the anchor does not lie in one range of that helper;
    5. ``"input_layout"``: its first byte is not on a slot of its range, or its row pitch is not 9 slots;
    6. ``"overlap"``: a caller's output meets an input;
    7. the call is of class ``anchor`` (some BF16 operand not in the anchor's row frame) and the stage's twin has no
       first row count for that class: the first such operand names the reason, ``"input_phase"`` for an input,
       ``"out_memory"`` for a caller's output and ``"no_out"`` for a new output;
    8. ``"size"``: ``rows`` is below the first row count of the call's class.

    ``place`` is the anchor's ``(first_slot, first_die)`` from reason 6 on; ``homed`` maps each BF16 operand's label to
    whether it lies in the anchor's row frame (``cake_domain.in_frame``), from reason 6 on.  Nothing here launches,
    allocates or calls the driver; a fallback never raises."""
    twin = _twin(stage, device_index)
    if twin is None:
        return None, "arch", None, None, None
    if twin.lowest is None or rows < twin.lowest or rows > cake_jit.DOMAIN_MAX_ROWS:
        return None, "size", None, None, None
    memory = _domain()
    if memory.sm_table(device_index) is None:
        return None, "sm_table", None, None, None
    label, tensor = anchor
    place = memory.MEMORY if label in copied else memory.place(tensor)
    if place == memory.MEMORY:
        return None, "input_memory", None, None, None
    if place == memory.LAYOUT:
        return None, "input_layout", None, None, None
    homed = {label: True}
    reason = None
    for label, tensor in others:
        where = memory.MEMORY if label in copied else memory.place(tensor)
        homed[label] = ok = type(where) is tuple and memory.in_frame(place, where)
        if not ok and reason is None:
            reason = "input_phase"
    for label, tensor in outputs:
        if tensor is None:
            homed[label] = False
            reason = reason or "no_out"
            continue
        where = memory.place(tensor)
        homed[label] = ok = type(where) is tuple and memory.in_frame(place, where)
        if not ok and reason is None:
            reason = "out_memory"
    if _overlaps(callers, inputs):
        return None, "overlap", None, place, homed
    klass = "anchor" if reason else "homed"
    first = twin.anchor if reason else twin.homed
    if first is None:
        return None, reason or "size", klass, place, homed
    if rows < first:
        return None, "size", klass, place, homed
    return twin, "placed", klass, place, homed


def _twin_grid(twin: _Twin, rows: int, place: tuple[int, int]) -> int:
    """The grid of ``twin`` for ``rows`` rows from the anchor's ``place``: ``cake_jit.domain_grid``'s, inline (each
    die's items, at most the twin's resident CTAs, at least one).  Every row lies in exactly one item and an item holds
    at most ``item_rows`` rows, so ``rows >= twin.saturated`` (``resident * item_rows``) rows make at least ``resident``
    items: the grid is ``resident`` with no count."""
    if rows >= twin.saturated:
        return twin.resident
    items0, items1 = cake_jit.domain_items(rows, place[0], place[1], twin.chunk_shift, twin.item_rows)
    return max(1, min(items0 + items1, twin.resident))


def _launch_twin(
    twin: _Twin, device_index: int, head: tuple[Any, ...], place: tuple[int, int], tail: tuple[Any, ...], rows: int
) -> None:
    """Launch ``twin`` on the current stream of device ``device_index``: its base program's arguments ``head``, the
    claim counters of a workspace no launch in flight shares (``cake_domain.launch``), the anchor's ``place``, the
    twin's ``chunk_shift`` and ``item_rows``, the device's SM table, the column offsets ``tail`` and the grid.  The
    launch of a call :func:`_forward_domain` checks in full; a common call's twin launch is the step's own (held:
    :func:`_launch_held_twin`)."""
    memory = _domain()
    grid = _twin_grid(twin, rows, place)
    memory.launch(
        twin.bound(),
        twin.words,
        device_index,
        head,
        (
            place[0],
            place[1],
            twin.chunk_shift,
            twin.item_rows,
            memory.sm_table(device_index),
            *tail,
            grid,
            1,
            1,
        ),
    )


def _launch_held_twin(
    twin: _Twin,
    device_index: int,
    first: torch.Tensor,
    up: torch.Tensor,
    weight: torch.Tensor,
    h: torch.Tensor,
    rows: int,
    stride: int,
    h_stride: int,
    limit: float,
    weighted: int,
    first_slot: int,
    first_die: int,
    table: torch.Tensor,
    up_col: int,
    grid: int,
) -> None:
    """The twin launch of a common forward call that :func:`_forward_step_domain` kept for :func:`_held_domain` (a
    checkpoint's recomputation), with the arguments the step built: the step's own launch, run after ``apply``
    returns.  The entry is bound and the claim counters are taken now, at the launch (``cake_domain.claim``); a launch
    that raises drops its stream's workspace (``cake_domain.drop``)."""
    entry = twin.entry
    if entry is None:
        entry = twin.bound()
    memory = _domain()
    counters0, counters1, key = memory.claim(device_index, twin.words)
    try:
        entry(
            first,
            up,
            weight,
            h,
            rows,
            stride,
            stride,
            h_stride,
            limit,
            weighted,
            counters0,
            counters1,
            first_slot,
            first_die,
            twin.chunk_shift,
            twin.item_rows,
            table,
            up_col,
            grid,
            1,
            1,
        )
    except BaseException:
        memory.drop(key)
        raise


def _forward_checked(
    gate_up_or_gate: Any, up: Any, weight: Any, limit: Any, out: Optional[torch.Tensor]
) -> tuple[int, torch.device, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Any, float, Any]:
    """The checks of :func:`_forward`, in its order: ``(index, device, R, first, up, weight, copies, L, out)``.  No
    copy, allocation or launch."""
    fast = _fast_operands(gate_up_or_gate, up, weight) if _FAST_CHECKS else None
    if fast is None:
        device, rows, first, second, w, copies = _operands(gate_up_or_gate, up, weight)
        index = int(device.index)
    else:
        index, rows, first, second, w = fast
        device, copies = _device(index), None
    L = _limit(limit)
    h = out if out is None else _output(out, "out", rows=rows, cols=WIDTH, first=first, index=index)
    if _arch_of_index(index) is None:
        device_arch(index)  # raises: not an exported architecture
    return index, device, rows, first, second, w, copies, L, h


# The common call of the ``"domain"`` forward step (:func:`_forward_step_domain`): the common call of the
# ``"off"`` step (see :func:`_forward_step`), checked in the same order, with the same outputs, then routed in
# the step's own frame.  A call with fewer rows than the first row count its class can reach before its placement is
# read (:data:`_FLOORS`: ``anchor``'s for a new ``h``, else the lower of the two) launches the ``"off"`` program as
# :func:`_forward_step` does, with no registry read.  Every operand of the common call is contiguous, so its bytes and
# its row pitch follow from its shape: the route reads each operand's data pointer once and finds its place with one
# ``cake_domain.slot_of`` call (the anchor's, then each other BF16 operand's until one lies outside the anchor's row
# frame), and compares the byte spans of the caller's output buffers with the inputs' (C12) only for a call whose class
# can take the twin.  The decision is :func:`_domain_route`'s: the twin runs exactly when every reason there holds (the
# reasons' order decides only the plan's reason).  Any other call runs :func:`_forward_domain`, which checks in full,
# copies the inputs the kernels cannot read and routes through :func:`_domain_route`; both give the same launch.


def _forward_step_domain(
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    out: Optional[torch.Tensor],
    held: Optional[_HeldLaunch] = None,
) -> torch.Tensor:
    """The forward step with ``locality="domain"``: the common call checked, routed and launched here, any other call
    through :func:`_forward_domain`; the checks, ``h``, the route and the launch of :func:`_forward_domain`.  With
    ``held``, the launch is appended there for :func:`_held_domain` to run: ``(_launch_held_twin, arguments)`` for the
    twin (the arguments of the launch here, the grid and the SM table included), ``(entry, arguments)`` for the
    ``"off"`` program (:func:`_forward_step`'s)."""
    first = gate_up_or_gate
    if _FAST_CHECKS and type(first) is torch.Tensor and first.is_cuda and first.dtype is _BF16:
        shape = first.shape
        if len(shape) == 2 and shape[1] == (_MERGED_WIDTH if up is None else WIDTH):
            rows = shape[0]  # two reads: a ``torch.Size`` is a tuple subclass, which unpacks through an iterator
            cols = shape[1]
            if 0 < rows < MAX_ROWS and first.is_contiguous() and not (address := first.data_ptr()) % ALIGNMENT_BYTES:
                index = first.get_device()
                # :func:`_plain` of the up of two tensors, inline, its data pointer kept for the route
                if (
                    up is None
                    or (
                        type(up) is torch.Tensor
                        and up.is_cuda
                        and up.dtype is _BF16
                        and up.shape == shape
                        and up.is_contiguous()
                        and up.get_device() == index
                        and not (up_address := up.data_ptr()) % ALIGNMENT_BYTES
                    )
                ) and (
                    weight is None
                    or (
                        type(weight) is torch.Tensor
                        and weight.is_cuda
                        and weight.dtype is _FP32
                        and weight.shape == (rows,)
                        and weight.is_contiguous()
                        and weight.get_device() == index
                    )
                ):
                    if type(limit) is float or type(limit) is int:  # :func:`_limit`'s cached rounding, inlined
                        value, L = _last_limit
                        if limit != value:
                            L = _limit(limit)
                    else:
                        L = _limit(limit)
                    if out is not None and not (
                        type(out) is torch.Tensor
                        and out.is_cuda
                        and out.dtype is _BF16
                        and out.shape == (rows, WIDTH)
                        and out.is_contiguous()
                        and out.get_device() == index
                        and not (out_address := out.data_ptr()) % ALIGNMENT_BYTES
                    ):
                        # ``out`` after ``limit``, as in :func:`_forward`, which checks this ``out`` in full.
                        return _forward_domain(gate_up_or_gate, up, weight, limit, out, held)
                    if _arch_of_index(index) is None:
                        device_arch(index)  # raises: not an exported architecture
                    launch = _launcher("fwd", index)
                    caller = out
                    if out is None:
                        out = torch.empty(rows, WIDTH, dtype=_BF16, device=launch.device)
                    if weight is None:
                        w, weighted = launch.weight, 0
                        if w is None:
                            w = launch.ignored_weight()
                    else:
                        w, weighted = weight, 1
                    stride, out_stride = (cols, WIDTH) if rows > 1 else (0, 0)
                    if up is None:
                        up_tensor, up_col = first, WIDTH
                    else:
                        up_tensor, up_col = up, 0
                    # Below the first row count the call's class can reach (a new ``h``: ``anchor``'s), the ``"off"``
                    # launch below, with no registry read.
                    floors = _FLOORS.get(index) or _floors(index)
                    if floors[1 if caller is None else 0] <= rows <= cake_jit.DOMAIN_MAX_ROWS:
                        twin = floors[2]
                        memory = _domain()
                        table = memory.sm_table(index)
                        if table is not None:
                            bases, _, frames = memory.ranges(index)
                            slot_of = memory.slot_of
                            at = slot_of(bases, frames, address, rows, cols)
                            if at >= 0:
                                # The class: ``homed`` when the caller's ``out`` and the up of two tensors lie in the
                                # anchor's row frame (a new ``h`` is ordinary memory: class ``anchor``).
                                homed = caller is not None
                                if homed:
                                    period = memory.FRAME_PERIOD
                                    other = slot_of(bases, frames, out_address, rows, WIDTH)
                                    homed = other >= 0 and (other - at) % period < _ROW_SLOTS
                                    if homed and up is not None:
                                        other = slot_of(bases, frames, up_address, rows, WIDTH)
                                        homed = other >= 0 and (other - at) % period < _ROW_SLOTS
                                first_rows = twin.homed if homed else twin.anchor
                                if first_rows is not None and rows >= first_rows:
                                    # C12: the caller's ``out`` against the bytes of every input (an absent input
                                    # spans nothing)
                                    if caller is not None:
                                        end = out_address + 2 * rows * WIDTH
                                        if up is None:
                                            up_address = up_end = 0
                                        else:
                                            up_end = up_address + 2 * rows * WIDTH
                                        if weight is None:
                                            w_address = w_end = 0
                                        else:
                                            w_address = weight.data_ptr()
                                            w_end = w_address + 4 * rows
                                        meets = (
                                            (out_address < address + 2 * rows * cols and address < end)
                                            or (out_address < up_end and up_address < end)
                                            or (out_address < w_end and w_address < end)
                                        )
                                    else:
                                        meets = False
                                    if not meets:
                                        die, slot = divmod(at, memory.CHUNK_SLOTS)
                                        place = (slot, die)
                                        if held is not None:
                                            # the launch below, its arguments kept (:func:`_launch_held_twin`)
                                            arguments = (
                                                twin,
                                                index,
                                                first,
                                                up_tensor,
                                                w,
                                                out,
                                                rows,
                                                stride,
                                                out_stride,
                                                L,
                                                weighted,
                                                slot,
                                                die,
                                                table,
                                                up_col,
                                                _twin_grid(twin, rows, place),
                                            )
                                            held.append((_launch_held_twin, arguments))
                                            return out
                                        entry = twin.entry
                                        if entry is None:
                                            entry = twin.bound()
                                        grid = _twin_grid(twin, rows, place)
                                        # ``cake_domain.launch``'s, with the arguments in one tuple
                                        counters0, counters1, key = memory.claim(index, twin.words)
                                        try:
                                            entry(
                                                first,
                                                up_tensor,
                                                w,
                                                out,
                                                rows,
                                                stride,
                                                stride,
                                                out_stride,
                                                L,
                                                weighted,
                                                counters0,
                                                counters1,
                                                slot,
                                                die,
                                                twin.chunk_shift,
                                                twin.item_rows,
                                                table,
                                                up_col,
                                                grid,
                                                1,
                                                1,
                                            )
                                        except BaseException:
                                            memory.drop(key)
                                            raise
                                        return out
                    limit_rows, plans, tail = launch.route
                    name, _, per_row, per_cta = plans[rows] if rows < limit_rows else tail
                    entry = launch.entries.get(name)
                    if entry is None:
                        entry = launch.entry(name)
                    if held is not None:  # the same arguments as the launch below, kept for :func:`_held_domain`
                        held.append(
                            (
                                entry,
                                (
                                    first,
                                    up_tensor,
                                    w,
                                    out,
                                    rows,
                                    stride,
                                    stride,
                                    out_stride,
                                    L,
                                    weighted,
                                    up_col,
                                    -(-rows * per_row // per_cta),
                                    1,
                                    1,
                                ),
                            )
                        )
                        return out
                    entry(
                        first,
                        up_tensor,
                        w,
                        out,
                        rows,
                        stride,
                        stride,
                        out_stride,
                        L,
                        weighted,
                        up_col,
                        -(-rows * per_row // per_cta),
                        1,
                        1,
                    )
                    return out
    return _forward_domain(gate_up_or_gate, up, weight, limit, out, held)


def _forward_domain(
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    out: Optional[torch.Tensor],
    held: Optional[_HeldLaunch] = None,
) -> torch.Tensor:
    """The forward step with ``locality="domain"`` for any call: the checks of :func:`_forward`, the copies of the
    inputs the kernels cannot read, ``h`` (the caller's ``out``, else ``torch.empty``), then the twin where
    :func:`_domain_route` places the call, else the program of ``"off"``, with the same arguments and bits.  With
    ``held``, the launch is appended there as ``(launch, arguments)`` for :func:`_held_domain` to run."""
    index, device, rows, first, second, w, copies, L, h = _forward_checked(gate_up_or_gate, up, weight, limit, out)
    if copies:
        first, second, w = _read(first, second, w, copies)
    caller = h
    if h is None:
        h = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
    if rows == 0:
        return h
    many = rows > 1
    gate_stride = first.stride(0) if many else 0
    if second is None:
        up_tensor, up_stride, up_col = first, gate_stride, WIDTH
    else:
        up_tensor, up_stride, up_col = second, second.stride(0) if many else 0, 0
    launch = _launcher("fwd", index)
    head = (
        first,
        up_tensor,
        launch.ignored_weight() if w is None else w,
        h,
        rows,
        gate_stride,
        up_stride,
        h.stride(0) if many else 0,
        L,
        0 if w is None else 1,
    )
    # Below the first row count the call's class can reach (:data:`_FLOORS`), the ``"off"`` program with no registry
    # read: the program :func:`_domain_route` gives such a call.
    floors = _FLOORS.get(index) or _floors(index)
    if rows < floors[1 if caller is None else 0]:
        twin = place = None
    else:
        twin, _, _, place, _ = _domain_route(
            "fwd",
            index,
            rows,
            ("gate_up" if second is None else "gate", first),
            () if second is None else (("up", second),),
            (("h", caller),),
            (first, second, w),
            (caller,),
        )
    if twin is None:
        arguments: tuple[Any, ...] = (rows, *head, up_col)
        if held is None:
            launch(*arguments)
        else:
            held.append((launch, arguments))
        return h
    arguments = (twin, index, head, place, (up_col,), rows)
    if held is None:
        _launch_twin(*arguments)
    else:
        held.append((_launch_twin, arguments))
    return h


def _backward_checked(
    grad_out: Any,
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    grad_gate_up_or_gate: Optional[torch.Tensor],
    grad_up: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    rounded: bool,
) -> tuple[Any, ...]:
    """The checks of :func:`_backward`, in its order: ``(index, device, R, first, up, weight, copies, dh, copy_dh, L,
    cols, d_first, d_up, d_weight)``, the output buffers as the caller gave them (None: new).  No copy, allocation or
    launch."""
    fast = _fast_operands(gate_up_or_gate, up, weight) if _FAST_CHECKS else None
    if fast is None:
        device, rows, first, second, w, copies = _operands(gate_up_or_gate, up, weight)
        index = int(device.index)
    else:
        index, rows, first, second, w = fast
        device, copies = _device(index), None
    if _FAST_CHECKS and _plain(grad_out, torch.bfloat16, (rows, WIDTH), index):
        dh, copy_dh = grad_out, False
    else:
        dh, copy_dh = _input(grad_out, "grad_out", rows=rows, cols=WIDTH, device=device)
    L = limit if rounded else _limit(limit)
    cols = 2 * WIDTH if second is None else WIDTH
    if second is None and grad_up is not None:
        raise ValueError("grad_up must be None for a merged input: grad_gate_up_or_gate receives both halves")
    d_first = grad_gate_up_or_gate
    if d_first is not None:
        d_first = _output(d_first, "grad_gate_up_or_gate", rows=rows, cols=cols, first=first, index=index)
    d_second = grad_up
    if d_second is not None:
        d_second = _output(d_second, "grad_up", rows=rows, cols=WIDTH, first=first, index=index)
    d_weight = grad_weight
    if d_weight is not None:
        if w is None:
            raise ValueError("grad_weight needs a weight: an unweighted call has no weight gradient")
        d_weight = _grad_weight(d_weight, rows=rows, first=first, index=index)
    if _arch_of_index(index) is None:
        device_arch(index)  # raises: not an exported architecture
    return index, device, rows, first, second, w, copies, dh, copy_dh, L, cols, d_first, d_second, d_weight


def _forward_locality(
    gate_up_or_gate: Any, up: Any, weight: Any, limit: Any, out: Optional[torch.Tensor], locality: Any
) -> torch.Tensor:
    """The forward step for a ``locality`` other than ``"off"``: checked (:func:`_domain_requested`; the ``str``
    ``"domain"`` itself passes with no call), then :func:`_forward_step_domain` for ``"domain"``."""
    if (type(locality) is str and locality == "domain") or _domain_requested(locality):
        return _forward_step_domain(gate_up_or_gate, up, weight, limit, out)
    return _forward_step(gate_up_or_gate, up, weight, limit, out)


def _backward_locality(
    grad_out: Any,
    gate_up_or_gate: Any,
    up: Any,
    weight: Any,
    limit: Any,
    grad_gate_up_or_gate: Optional[torch.Tensor],
    grad_up: Optional[torch.Tensor],
    grad_weight: Optional[torch.Tensor],
    locality: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The backward step for a ``locality`` other than ``"off"``: checked (:func:`_domain_requested`; the ``str``
    ``"domain"`` itself passes with no call), then the ``"off"`` step: the backward has no twin, so a ``"domain"``
    call runs the ``"off"`` program (reason ``"arch"``)."""
    if not (type(locality) is str and locality == "domain"):
        _domain_requested(locality)
    return _backward_step(
        grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight, False
    )


def _plain_plan(stage: str, device_index: int, rows: int) -> tuple[str, tuple[int, int, int]]:
    """The program and grid of the ``"off"`` route of ``stage`` for ``rows >= 1`` rows on the device (:func:`_route`)."""
    limit, plans, tail = _route(stage, device_index)
    name, _, per_row, per_cta = plans[rows] if rows < limit else tail
    return name, (-(-rows * per_row // per_cta), 1, 1)


def _route_plan(
    stage: str,
    device_index: int,
    rows: int,
    route: Optional[_Route],
    outputs: dict[str, str],
) -> dict[str, Any]:
    """The plan of a ``locality="domain"`` call (see :func:`cake_clamped_swiglu_det_train_forward_domain_plan`)."""
    if route is None:
        return {
            "stage": stage,
            "route": "plain",
            "reason": "empty",
            "class": None,
            "program": None,
            "first_slot": None,
            "first_die": None,
            "chunk_shift": None,
            "item_rows": None,
            "grid": None,
            "homed": None,
            "outputs": outputs,
        }
    twin, reason, klass, place, homed = route
    plan = {
        "stage": stage,
        "route": "plain" if twin is None else "domain",
        "reason": reason,
        "class": klass,
        "program": None,
        "first_slot": None,
        "first_die": None,
        "chunk_shift": None,
        "item_rows": None,
        "grid": None,
        "homed": None if homed is None else dict(homed),
        "outputs": outputs,
    }
    if twin is None:
        plan["program"], plan["grid"] = _plain_plan(stage, device_index, rows)
    else:
        plan.update(
            program=twin.name,
            first_slot=place[0],
            first_die=place[1],
            chunk_shift=twin.chunk_shift,
            item_rows=twin.item_rows,
            grid=(_twin_grid(twin, rows, place), 1, 1),
        )
    return plan


def _copied_labels(labels: tuple[str, ...], copies: Any) -> frozenset[str]:
    """The labels of the operands a call copies first (``copies``: the first operand, the up, the weight)."""
    if not copies:
        return frozenset()
    return frozenset(label for label, copy in zip(labels, copies) if copy)


def cake_clamped_swiglu_det_train_forward(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    out: Optional[torch.Tensor] = None,
    locality: str = "off",
) -> torch.Tensor:
    """Forward step: ``h = bf16(weight * silu(min(gate, L)) * clamp(up, -L, L))``, one rounding (see the module and
    the facade's documentation).  ``locality="domain"`` runs the forward's twin where
    :func:`cake_clamped_swiglu_det_train_forward_domain_plan` says so, the same bits either way."""
    if locality == "off":
        return _forward_step(gate_up_or_gate, up, weight, limit, out)
    return _forward_locality(gate_up_or_gate, up, weight, limit, out, locality)


def cake_clamped_swiglu_det_train_backward(
    grad_out: torch.Tensor,
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    grad_gate_up_or_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
    grad_weight: Optional[torch.Tensor] = None,
    locality: str = "off",
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward step: ``(d_gate_up, None, d_weight)`` for a merged input, ``(d_gate, d_up, d_weight)`` for two
    tensors; ``d_weight`` is None without a weight (see the module and the facade's documentation).
    ``locality="domain"`` runs the ``"off"`` program too: the backward has no twin
    (:func:`cake_clamped_swiglu_det_train_backward_domain_plan`: reason ``"arch"``)."""
    if locality == "off":
        return _backward_step(
            grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight, False
        )
    return _backward_locality(
        grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight, locality
    )


def cake_clamped_swiglu_det_train_domain_empty(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[Union[torch.device, str, int]] = None,
) -> torch.Tensor:
    """An uninitialized contiguous BF16 tensor of ``shape`` ``(R, 2304)`` or ``(R, 4608)`` in memory whose chunks
    (2 MB, or 4 MB for the merged width) alternate between the two dies of an SM107 (R200) device, row 0 at the
    start of a die-0 chunk: an operand or output of a ``locality="domain"`` call (``cake_domain.domain_empty``).
    ``device`` defaults to the current CUDA device.  ``TypeError`` for a ``dtype`` that is not a ``torch.dtype``,
    ``ValueError`` for another dtype or shape or a device that is not CUDA, ``torch.cuda.OutOfMemoryError`` when the
    device has no room, ``RuntimeError`` on another architecture, during a CUDA graph capture on any device in this
    thread, with fewer than two locality domains or without cuda-python.  Freeing the last view synchronizes the
    device."""
    return _domain().domain_empty(shape, dtype=dtype, device=device)


def cake_clamped_swiglu_det_train_domain_release(device: Optional[Union[torch.device, str, int]] = None) -> None:
    """Free the ranges of :func:`cake_clamped_swiglu_det_train_domain_empty` memory of ``device`` (the current CUDA
    device by default) whose last view is gone and that still wait to be freed (their last view died while another
    call held the module's lock, or during a CUDA graph capture).  ``RuntimeError`` during a CUDA graph capture on the
    device's current stream; ``ValueError`` for a device that is not CUDA."""
    index = _domain()._cuda_index(device)
    with torch.cuda.device(index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cake_clamped_swiglu_det_train_domain_release runs outside a CUDA graph capture only")
    _domain().release(index)


def cake_clamped_swiglu_det_train_forward_domain_plan(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    out: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """What ``cake_clamped_swiglu_det_train_forward(..., locality="domain")`` runs for these arguments, checked as the
    forward checks them, without a launch, an allocation or a copy: ``stage``; ``route`` (``"domain"`` for the twin,
    else ``"plain"``); ``reason`` (``"placed"`` for the twin, else a reason of ``cake_jit.DOMAIN_REASONS``, or
    ``"empty"`` for ``R = 0``); ``class`` (``"homed"`` or ``"anchor"``, once the anchor is placed); ``program`` (the
    ``cake_jit.MODULES`` record that runs; None for no launch); ``first_slot``, ``first_die``, ``chunk_shift`` and
    ``item_rows`` (the twin's, None on the plain route); ``grid``; ``homed`` (each BF16 operand: whether it lies in the
    anchor's row frame; None before the anchor is placed); ``outputs`` (each output: ``"caller"`` for a caller's
    buffer, ``"empty"`` for a new ``torch.empty`` tensor)."""
    index, _, rows, first, second, w, copies, _, h = _forward_checked(gate_up_or_gate, up, weight, limit, out)
    outputs = {"h": "empty" if h is None else "caller"}
    if rows == 0:
        return _route_plan("fwd", index, rows, None, outputs)
    anchor = "gate_up" if second is None else "gate"
    copied = _copied_labels((anchor, "up", "weight"), copies)
    route = _domain_route(
        "fwd",
        index,
        rows,
        (anchor, first),
        () if second is None else (("up", second),),
        (("h", h),),
        tuple(
            None if label in copied else tensor for label, tensor in ((anchor, first), ("up", second), ("weight", w))
        ),
        (h,),
        copied,
    )
    return _route_plan("fwd", index, rows, route, outputs)


def cake_clamped_swiglu_det_train_backward_domain_plan(
    grad_out: torch.Tensor,
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    grad_gate_up_or_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
    grad_weight: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """What ``cake_clamped_swiglu_det_train_backward(..., locality="domain")`` runs for these arguments, checked as the
    backward checks them, without a launch, an allocation or a copy: the keys of
    :func:`cake_clamped_swiglu_det_train_forward_domain_plan`, ``stage`` ``"bwd"`` for a weighted call and
    ``"bwd_unweighted"`` for an unweighted one, and ``outputs`` naming ``d_gate_up`` (merged) or ``d_gate`` and
    ``d_up``, and ``d_weight`` for a weighted call.  The backward has no twin: the route is ``"plain"`` with reason
    ``"arch"`` (``"empty"`` for ``R = 0``), the ``"off"`` program and its grid."""
    (index, _, rows, _, second, w, _, _, _, _, _, d_first, d_second, d_weight) = _backward_checked(
        grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight, False
    )
    stage = "bwd_unweighted" if w is None else "bwd"
    if second is None:
        labels: tuple[tuple[str, Optional[torch.Tensor]], ...] = (("d_gate_up", d_first),)
    else:
        labels = (("d_gate", d_first), ("d_up", d_second))
    outputs = {label: "empty" if tensor is None else "caller" for label, tensor in labels}
    if w is not None:
        outputs["d_weight"] = "empty" if d_weight is None else "caller"
    if rows == 0:
        return _route_plan(stage, index, rows, None, outputs)
    # No backward stage has a twin (:func:`_backward_locality`): the ``"off"`` program, reason ``"arch"``.
    return _route_plan(stage, index, rows, (None, "arch", None, None, None), outputs)


class _Gradients(torch.autograd.Function):
    """The backward step of :class:`CakeClampedSwiGLUDetFunction` as an autograd function whose own backward raises.
    The autograd function's backward runs through it when the engine records the gradients (``create_graph=True``),
    so differentiating them raises ``RuntimeError`` instead of dropping this operation's second-order terms; the
    gradients are the step's (its forward runs the step, without autograd)."""

    @staticmethod
    def forward(ctx, grad_h, gate_up_or_gate, up, weight, limit):
        return _backward_step(grad_h, gate_up_or_gate, up, weight, limit, None, None, None, True)

    @staticmethod
    def backward(ctx, *grads):
        raise RuntimeError(
            f"{_WHAT} has no double backward: the gradients of cake_clamped_swiglu_det are not differentiable"
        )


class _HeldLaunch(list):
    """The forward launch of one autograd call in a checkpoint's recomputation, kept from :class:`_HeldFunction`'s
    ``forward`` until ``apply`` returns: the step appends one ``(launcher, arguments)`` item, which :func:`_held` then
    runs (no item when the step launched nothing, ``R == 0``).  A list with no Python method: keeping and running the
    launch are C list operations."""

    __slots__ = ()


class CakeClampedSwiGLUDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_clamped_swiglu_det`: saves only its inputs (``gate_up_or_gate``, ``up`` and ``weight``;
    no activation, no FP32 intermediate) and recomputes the activation in the backward, so
    ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as saved state.  The gradient of a merged
    ``[R, 4608]`` input is one ``[R, 4608]`` tensor; ``limit`` gets no gradient.  ``limit`` is checked in the forward,
    by the forward step; the autograd forward then reads its FP32 rounding through :func:`_limit` again (the cached pair
    for a Python float, or an int that a float holds exactly; for any other number the limit checks run a second time,
    on a value that passed them), and the backward reuses that FP32 value.  The backward has no output buffer,
    ``grad_up`` or ``grad_weight`` to check; it checks the upstream gradient and, in the same one pass as a step, the
    saved operands again: a saved-tensors hook (activation offloading) or a ``.data`` assignment can change their
    device, shape or layout after the forward without the autograd version check seeing it.  No double backward:
    differentiating gradients computed with ``create_graph=True`` raises ``RuntimeError``.

    ``apply(gate_up_or_gate, up, weight, limit)`` launches the forward kernel inside ``forward``.  PyTorch packs a
    custom function's saved tensors (its saved-tensors hooks run) only after ``forward`` returns, so a checkpoint's
    recomputation of a direct ``apply`` runs that launch even when its output is dropped.  In a checkpoint's
    recomputation :func:`cake_clamped_swiglu_det` applies :class:`_HeldFunction` instead, which launches after the
    inputs are saved."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit):
        h = _forward_step(gate_up_or_gate, up, weight, limit, None)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        # The FP32 rounding the step used (its cached pair for a float, or an int that a float holds exactly).
        ctx.limit = _limit(limit)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        gate_up_or_gate, up, weight = ctx.saved_tensors
        if torch.is_grad_enabled():  # create_graph=True: the gradients' own backward raises
            grads = _Gradients.apply(grad_h, gate_up_or_gate, up, weight, ctx.limit)
        else:
            grads = _backward_step(grad_h, gate_up_or_gate, up, weight, ctx.limit, None, None, None, True)
        d_first, d_up, d_weight = grads
        need_first, need_up, need_weight = ctx.needs_input_grad[:3]
        return (
            d_first if need_first else None,
            d_up if need_up else None,
            d_weight if need_weight else None,
            None,
        )


class _HeldFunction(CakeClampedSwiGLUDetFunction):
    """:class:`CakeClampedSwiGLUDetFunction` whose ``forward`` checks the arguments and allocates ``h`` but keeps the
    launch in its fifth argument, a :class:`_HeldLaunch`, for :func:`_held` to run after ``apply`` returns (after
    autograd saved the inputs); the same saved state and backward (no gradient for the fifth argument)."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit, held):
        h = _forward_step(gate_up_or_gate, up, weight, limit, None, held)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        ctx.limit = _limit(limit)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        return (*CakeClampedSwiGLUDetFunction.backward(ctx, grad_h), None)


_apply = CakeClampedSwiGLUDetFunction.apply
_held_apply = _HeldFunction.apply
# The id of the backward graph task running on this thread, -1 outside a backward (torch's checkpoint reads it the
# same way), and the innermost saved-tensors hooks, a (pack, unpack) pair or None.  Without either function (another
# torch) its stand-in answers "in a backward" / "hooks set", so the calls it decides take the held launch.
_graph_task = getattr(torch._C, "_current_graph_task_id", None) or (lambda: 0)
_hooks = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None) or (lambda ignore_is_tracing: ())
_is_compiling = torch.compiler.is_compiling
_is_tracing = torch._C._is_tracing


def _held(gate_up_or_gate: Any, up: Any, weight: Any, limit: Any) -> torch.Tensor:
    """:func:`cake_clamped_swiglu_det` with positional arguments in a backward with saved-tensors hooks set (a
    non-reentrant checkpoint's recomputation): :class:`_HeldFunction` applied with a :class:`_HeldLaunch`, then the
    kept forward launch.  ``apply`` saves the inputs (runs their pack hooks) before it returns, so a recomputation with
    early stop on (the default) ends inside ``apply``, before the launch, when these inputs are the last tensors its
    region saves: the recomputed ``h`` would be dropped.  When a later operation of the region saves a tensor (the down
    projection saves ``h``), the recomputation continues and launches it.  A call that Dynamo or ``torch.jit.trace``
    records applies :class:`CakeClampedSwiGLUDetFunction` (the launch inside ``forward``: a trace records ``apply``
    only)."""
    if _is_compiling() or _is_tracing():
        return _apply(gate_up_or_gate, up, weight, limit)
    held = _HeldLaunch()
    h = _held_apply(gate_up_or_gate, up, weight, limit, held)
    if held:
        launch, arguments = held[0]
        launch(*arguments)
    return h


# ----------------------------------------------------------------------------------------------------------------
# Autograd with ``locality="domain"``: the classes above with the domain forward step (the backward has no twin).  The
# ``"off"`` classes and their steps are not edited.


class _DomainFunction(CakeClampedSwiGLUDetFunction):
    """:class:`CakeClampedSwiGLUDetFunction` with ``locality="domain"``: the forward step runs its twin where the domain
    route places the call (:func:`_forward_step_domain`); the backward is :class:`CakeClampedSwiGLUDetFunction`'s, since
    the backward has no twin.  The same saved state (the inputs only), the same checks and the same bits.  ``h`` is a
    new ``torch.empty`` tensor, so the forward runs its twin as a call of class ``anchor``."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit):
        h = _forward_step_domain(gate_up_or_gate, up, weight, limit, None)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        ctx.limit = _limit(limit)
        return h


class _HeldDomainFunction(_DomainFunction):
    """:class:`_DomainFunction` whose ``forward`` keeps the launch in its fifth argument, a :class:`_HeldLaunch`, for
    :func:`_held_domain` to run after ``apply`` returns (:class:`_HeldFunction`'s rule)."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit, held):
        h = _forward_step_domain(gate_up_or_gate, up, weight, limit, None, held)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        ctx.limit = _limit(limit)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        return (*_DomainFunction.backward(ctx, grad_h), None)


_domain_apply = _DomainFunction.apply
_held_domain_apply = _HeldDomainFunction.apply


def _held_domain(gate_up_or_gate: Any, up: Any, weight: Any, limit: Any) -> torch.Tensor:
    """:func:`_held` with ``locality="domain"``."""
    if _is_compiling() or _is_tracing():
        return _domain_apply(gate_up_or_gate, up, weight, limit)
    held = _HeldLaunch()
    h = _held_domain_apply(gate_up_or_gate, up, weight, limit, held)
    if held:
        launch, arguments = held[0]
        launch(*arguments)
    return h


def _autograd_locality(gate_up_or_gate: Any, up: Any, weight: Any, limit: Any, locality: Any) -> torch.Tensor:
    """:func:`cake_clamped_swiglu_det` for a ``locality`` other than ``"off"``: checked (:func:`_domain_requested`; the
    ``str`` ``"domain"`` itself passes with no call), then the domain classes, applied as the ``"off"`` ones are."""
    if not ((type(locality) is str and locality == "domain") or _domain_requested(locality)):
        return cake_clamped_swiglu_det(gate_up_or_gate, up, weight, limit=limit)
    if _graph_task() == -1 or _hooks(False) is None:
        return _domain_apply(gate_up_or_gate, up, weight, limit)
    return _held_domain(gate_up_or_gate, up, weight, limit)


def cake_clamped_swiglu_det(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    locality: str = "off",
) -> torch.Tensor:
    """Differentiable routed-expert clamped SwiGLU: :class:`CakeClampedSwiGLUDetFunction` applied to the arguments
    (the forward launched inside ``forward``); in a backward with saved-tensors hooks set (a non-reentrant
    checkpoint's recomputation), :func:`_held`: the forward kernel launched after autograd saved the inputs.  With
    ``locality="domain"``, the same with the domain classes (:func:`_autograd_locality`): the forward step runs its
    twin where the domain route places it and the backward runs the ``"off"`` program, the same bits either way."""
    if locality != "off":
        return _autograd_locality(gate_up_or_gate, up, weight, limit, locality)
    if _graph_task() == -1 or _hooks(False) is None:
        return _apply(gate_up_or_gate, up, weight, limit)
    return _held(gate_up_or_gate, up, weight, limit)


__all__ = [
    "ARCHES",
    "INTERMEDIATE_SIZE",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeClampedSwiGLUDetFunction",
    "cake_clamped_swiglu_det",
    "cake_clamped_swiglu_det_train_backward",
    "cake_clamped_swiglu_det_train_backward_domain_plan",
    "cake_clamped_swiglu_det_train_domain_empty",
    "cake_clamped_swiglu_det_train_domain_release",
    "cake_clamped_swiglu_det_train_forward",
    "cake_clamped_swiglu_det_train_forward_domain_plan",
    "is_cake_clamped_swiglu_det_train_supported",
]
