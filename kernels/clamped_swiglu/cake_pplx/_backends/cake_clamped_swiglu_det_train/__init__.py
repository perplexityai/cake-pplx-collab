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
  is read in place (the kernels get it as the gate, its first 2304 columns,
  and a view of its second half as the up) and so are row slices of wider
  tensors.  An input the kernels cannot read (not 2-D with a unit last
  stride, a row stride that is not a multiple of 8 elements, a base address
  that is not 16-byte aligned) is copied once to a contiguous tensor.
  ``weight`` is FP32 ``[R]`` (copied once when not contiguous), or ``None``
  for an unweighted expert (``w = 1``; the kernels then get a cached
  one-element FP32 buffer of the device, set to 1, and read its one element
  and ignore it).  ``limit`` is ``L``, a positive finite Python number rounded to FP32;
  it is a launch argument, as are ``R``, the weight flag and the strides, so
  one library per program and architecture serves every call (the row count
  picks the program).
- Checks.  An entry point checks every argument (the limit, the operands'
  types, dtypes, shapes and devices, the output buffers, the ``grad_up`` and
  ``grad_weight`` rules, ``R`` and the device's architecture) before it
  copies an input or allocates an output, so a call refused for its arguments
  copies, allocates and launches nothing.  The common call (plain contiguous CUDA tensors of the
  exact dtype and shape on one device, the BF16 ones 16-byte aligned,
  ``0 < R < 2**31 - 256``) is accepted in one pass over each tensor's
  metadata; any other call takes the full checks, which raise every error and
  mark the inputs to copy.  Both give the same launch.  A ``limit`` that is a
  Python float or int equal to the last accepted one reuses its FP32 rounding
  (one cached pair); any other ``limit`` takes the full limit checks.  The
  autograd function's backward reuses the forward's FP32 ``L`` and checks the
  saved operands again (a saved-tensors hook or a ``.data`` assignment can
  change them after the forward) and the upstream gradient; it has no output
  buffer to check.
- Outputs.  ``h`` is a new contiguous BF16 ``[R, 2304]`` tensor, or the
  caller's ``out``.  The backward's ``d_gate`` and ``d_up`` have the layout
  of the inputs: two new contiguous ``[R, 2304]`` tensors, or one new
  contiguous ``[R, 4608]`` gradient for a merged input.  ``d_weight`` is FP32
  ``[R]``; the backward kernel writes it for every call, so an unweighted
  backward allocates it and drops it.  A caller's output buffer must already
  be a layout the kernels write (2-D, unit last stride, a row stride that is a
  multiple of 8 elements and at least the row width, 16-byte aligned; FP32
  ``[R]`` contiguous for the weight gradient): it is refused with a
  ``ValueError`` otherwise, never copied.  An output must not overlap any
  input or another output (not checked).  A new output is ``torch.empty`` on
  the operands' device (one ``torch.device`` per device index for the
  process): a contiguous plain tensor.
- Launches.  One kernel per call.  Each stage has two programs
  (``cake_jit.MODULES``): a small-R program and a large-R program, which runs
  when ``R`` lies in one of its record's windows of rows per SM on the
  device's architecture, ``lo * SMs <= R <= hi * SMs`` for a window
  ``[lo, hi]`` (``cake_jit.route``; the architecture is read once per device
  and the SM count once per stage and device, and an architecture without a
  large-R program runs the small-R one at every ``R``).  Both programs of a stage give the same bits for every
  input, so the route changes no output bit.  CTAs of ``threads / lanes``
  rows (32 lanes a row; 128 threads and 4 rows, but 256 threads and 8 rows in
  the large-R forward on SM100 and SM103), one CTA per block of rows on the
  grid's x axis (``cake_jit.grid``).  ``R == 0`` launches nothing and loads
  no library; a program's library loads at its first launch.  Every launch
  is a plain stream-ordered launch on the current torch stream of the
  operands' device (a tvm-ffi call whose arguments include PyTorch CUDA
  tensors runs on that stream, a CUDA graph's capture stream during capture;
  the binding makes the device current): nothing cooperative, no PDL, no
  dynamic shared memory.  The entry points launch through one launcher per
  stage and device (``_launcher``), made at the stage's first launch on the
  device: the route above, the kernel parameters by position in the order
  every program of the stage takes them (checked once, when the launcher is
  made), and each program's FFI entry, bound at its first launch.  Nothing
  here reads device memory, reads an environment variable or synchronizes; the
  checks read dtypes, shapes, strides, devices and data pointers only.  The
  host keeps no state between calls but caches of facts a call cannot change
  (a device's architecture, SM count and launchers, the libraries, the FP32
  rounding of the last accepted ``limit``, the ignored weight below).
  Under ``torch.use_deterministic_algorithms(True)`` with
  ``torch.utils.deterministic.fill_uninitialized_memory`` (on by default),
  PyTorch fills every tensor it allocates here with NaN before the kernel
  writes it: one more fill launch per new output (``h``; ``d_gate_up``, or
  ``d_gate`` and ``d_up``; ``d_weight``, also the one an unweighted backward
  drops) and per copied input.  Output buffers allocated once and passed to
  every call avoid the fills of the outputs.  The kernels write every element
  of every output, so a fill never reaches a result.
- Numerics.  ``g' = min(g, L)`` and ``u' = clamp(u, -L, L)`` from the BF16
  values widened to FP32; ``s = 1 / (1 + 2^(-g' log2 e))`` with the
  approximate base-2 exponential (``ex2.approx.ftz``) and the approximate
  reciprocal plus one round-to-nearest Newton step (the IEEE ``rcp.rn`` of
  every ``1 <= 1 + e < 2^126``); a warp with a clamped gate at or below -87
  in a run of its loads computes that run with a branch-free exact reciprocal
  that keeps the subnormal sigmoid (``RN(1 / (1 + e))`` for every ``e``, +0 for an infinite
  one).  ``silu = g' s`` and ``a = silu u'``; ``h = bf16(w a)``, one rounding.
  The backward: ``d_gate = bf16(((dh w) u') (silu (1 - s) + s) [g <= L])``
  and ``d_up = bf16((dh w) silu [-L <= u <= L])``, the masks inclusive and
  taken on the raw inputs, each output rounded once from FP32;
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
- Workspace: none.  Retained state: the autograd function saves only its
  inputs (``gate`` and ``up`` or the merged tensor, and ``weight``) through
  ``save_for_backward``, references to the caller's tensors with no copy; the
  backward recomputes the activation.
  ``torch.utils.checkpoint(..., use_reentrant=False)`` keeps the same inputs;
  the function adds no saved state of its own and no activation.  The
  gradients have no derivative of their own: when the autograd engine records
  them (``create_graph=True``), differentiating them raises ``RuntimeError``.
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
from collections.abc import Callable
from typing import Any, Optional

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

# Whether an entry point first tries the one-pass checks of the common call (:func:`_fast_operands`, :func:`_plain`);
# off, every call takes the full checks.  The launches are the same either way.
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
    """The kernel parameter names of record ``name`` in ``run`` order (its argument plan without the grid)."""
    names: list[str] = []
    axes: list[str] = []
    for kind, key in cake_jit.MODULES[name]["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        elif axes:
            raise ValueError(f"record {name}: kernel parameter {key!r} follows the grid arguments")
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
def _plan(name: str) -> tuple[Callable[[dict[str, Any]], tuple[Any, ...]], int]:
    """``(arguments, rows_per_cta)`` of record ``name``, made once per record: ``arguments(values)`` is the tuple of
    its kernel parameters in ``run`` order (:func:`parameters`), and one CTA owns ``rows_per_cta`` rows
    (``cake_jit.rows_per_cta``)."""
    return operator.itemgetter(*parameters(name)), cake_jit.rows_per_cta(name)


# A program a launch may run: ``(name, arguments, rows_per_cta)`` (:func:`_plan`).
_Plan = tuple[str, Callable[[dict[str, Any]], tuple[Any, ...]], int]


@functools.cache
def _route(stage: str, device_index: int) -> tuple[int, bytes, _Plan, _Plan, _Plan]:
    """``(limit, table, small, large, tail)`` of ``stage`` on device ``device_index``, made once per stage and device
    from the device's architecture and SM count (``cake_jit.route``).  A launch of ``rows < limit`` rows runs the plan
    ``large`` when ``table[rows]`` is 1 (``rows`` in a window of the large-R program) and the plan ``small`` when it
    is 0; a launch of more rows runs ``tail``: ``large`` when the last window has no upper bound, else ``small``.
    ``limit`` is one above the last edge of ``cake_jit.route`` (0 without a window).  The launch looks up one byte
    instead of searching the edges: on Grace, a search per launch measured 1-3 % more host time per call.  No library
    is loaded here: a program's library loads at its first launch."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    edges, small, large = cake_jit.route(stage, device_arch(device_index), sm_count)
    limit = edges[-1] + 1 if edges else 0
    table = bytearray(limit)
    for below, last in zip(edges[0::2], edges[1::2]):
        table[below + 1 : last + 1] = bytes([1]) * (last - below)
    small, large = (small, *_plan(small)), (large, *_plan(large))
    return limit, bytes(table), small, large, large if len(edges) % 2 else small


def run_module(stage: str, values: dict[str, Any], *, rows: int, device_index: int) -> None:
    """Launch the program of ``stage`` for ``rows`` rows (:func:`_route`: the large-R program in its windows, the
    small-R one at every other row count) with its kernel parameters named in ``values``, on the current torch stream
    of device ``device_index``.

    The first argument of every record is a CUDA tensor of that device: tvm-ffi runs the call on PyTorch's current
    stream of its device, and the binding makes that device current for the launch.  The grid is
    ``cake_jit.grid(name, rows)`` of the program ``name`` that runs.  The entry points launch through
    :func:`_launcher`, which runs the same program with the same arguments and grid, passed by position."""
    limit, table, small, large, tail = _route(stage, device_index)
    name, arguments, per_cta = (large if table[rows] else small) if rows < limit else tail
    try:
        args = arguments(values)
    except KeyError:
        for key in parameters(name):
            if key not in values:
                raise ValueError(f"record {name} needs argument {key!r}") from None
        raise
    if rows < 1:
        cake_jit.grid(name, rows)  # raises: a launch needs at least one row
    _entry(name, device_arch(device_index))(*args, -(-rows // per_cta), 1, 1)


# The kernel parameters the entry points pass to the launcher of each stage, in this order (:func:`_launcher`).
_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "fwd": ("gate", "up", "w", "h", "rows", "gate_stride", "up_stride", "h_stride", "limit", "weighted"),
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
    ),
}


@functools.cache
def _launcher(stage: str, device_index: int) -> Callable[..., None]:
    """``launch(rows, *arguments)`` of ``stage`` on device ``device_index``, made once per stage and device: for
    ``rows >= 1`` rows it runs the program :func:`run_module` runs (:func:`_route`) with ``arguments``, the kernel
    parameters in the order of ``_ARGUMENTS[stage]``, then the grid ``cake_jit.grid(name, rows)``.  Both programs of
    the stage are checked to take their parameters in that order when the launcher is made, so a launch builds no dict
    of the arguments, and it looks up only its program's FFI entry, bound by the launcher at the program's first
    launch (no library loads when the launcher is made).  The entry points check the device's architecture at every
    call before they allocate; a launcher is made only for an exported architecture."""
    limit, table, small, large, tail = _route(stage, device_index)
    for name in dict.fromkeys((small[0], large[0])):
        if parameters(name) != _ARGUMENTS[stage]:
            raise RuntimeError(
                f"record {name}: kernel parameters {parameters(name)} are not in the order the host passes them, "
                f"{_ARGUMENTS[stage]}"
            )
    arch = device_arch(device_index)
    entries: dict[str, Any] = {}

    def launch(rows: int, *arguments: Any) -> None:
        name, _, per_cta = (large if table[rows] else small) if rows < limit else tail
        entry = entries.get(name)
        if entry is None:
            entry = entries[name] = _entry(name, arch)
        entry(*arguments, -(-rows // per_cta), 1, 1)

    return launch


@functools.cache
def _device(device_index: int) -> torch.device:
    """CUDA device ``device_index``, one object per device for the process: the device of the new outputs of a common
    call.  ``torch.empty`` on it costs about a third of the host time of ``Tensor.new_empty``."""
    return torch.device("cuda", device_index)


@functools.cache
def _ignored_weight(device_index: int) -> torch.Tensor:
    """The ``w`` of an unweighted launch (``weighted`` 0: the kernels read its one element and ignore it): one
    one-element FP32 buffer per device for the process, set to 1, made by the first unweighted call on the device.
    That call may be a CUDA graph capture: the buffer then lives in the graph's memory pool for the process, and its
    value is never used.  An unweighted call pays one cache lookup for it."""
    return torch.ones(1, dtype=torch.float32, device=torch.device("cuda", device_index))


# ----------------------------------------------------------------------------------------------------------------
# Checks: metadata only (dtypes, shapes, strides, devices, data pointers), nothing reads tensor contents.  Each
# check returns what it found; the entry points copy and allocate only after every check passed.


# The last limit :func:`_limit` accepted, as a Python float, and its FP32 rounding (NaN: none yet).  One pair, replaced
# whole, so a thread reads a value and its own rounding.
_last_limit: tuple[float, float] = (math.nan, math.nan)


def _limit(limit: Any) -> float:
    """``limit`` rounded to FP32 (the kernels' ``L``): a positive finite Python number.  A tensor is refused, as
    reading it would synchronize the host.  A ``float`` or ``int`` (exact types: no bool, no subclass) equal to the
    last accepted limit returns its rounding: an equal float or int has the same Python float value (Python compares an
    int and a float exactly), so the same rounding.  Any other value takes the checks below."""
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
    gate_up_or_gate: Any, up: Any, weight: Any, limit: Any, out: Optional[torch.Tensor]
) -> tuple[torch.Tensor, float]:
    """The forward step and its FP32 ``L``: every argument checked, then the copies of the inputs the kernels cannot
    read, the new output and the launch."""
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
    # A merged input is the gate itself (its first 2304 columns, at its own row stride) and a view of its second half.
    # A single row's strides are 0 (never dereferenced).
    many = rows > 1
    gate_stride = first.stride(0) if many else 0
    if second is None:
        up_view, up_stride = first.narrow(1, WIDTH, WIDTH), gate_stride
    else:
        up_view, up_stride = second, second.stride(0) if many else 0
    _launcher("fwd", index)(
        rows,
        first,
        up_view,
        _ignored_weight(index) if w is None else w,
        h,
        rows,
        gate_stride,
        up_stride,
        h.stride(0) if many else 0,
        L,
        0 if w is None else 1,
    )
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
    if d_weight is None:  # an unweighted call drops it
        d_weight = torch.empty(rows, dtype=torch.float32, device=device)
    result = (d_first, d_second, None if w is None else d_weight)
    if rows == 0:
        return result
    # A merged input and its gradient are the gate and d_gate themselves, and views of their second halves.  A single
    # row's strides are 0 (never dereferenced).
    many = rows > 1
    gate_stride = first.stride(0) if many else 0
    d_gate_stride = d_first.stride(0) if many else 0
    if second is None:
        up_view, up_stride = first.narrow(1, WIDTH, WIDTH), gate_stride
        d_up, d_up_stride = d_first.narrow(1, WIDTH, WIDTH), d_gate_stride
    else:
        up_view, up_stride = second, second.stride(0) if many else 0
        d_up, d_up_stride = d_second, d_second.stride(0) if many else 0
    _launcher("bwd", index)(
        rows,
        first,
        up_view,
        dh,
        _ignored_weight(index) if w is None else w,
        d_first,
        d_up,
        d_weight,
        rows,
        gate_stride,
        up_stride,
        dh.stride(0) if many else 0,
        d_gate_stride,
        d_up_stride,
        L,
        0 if w is None else 1,
    )
    return result


def cake_clamped_swiglu_det_train_forward(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward step: ``h = bf16(weight * silu(min(gate, L)) * clamp(up, -L, L))``, one rounding (see the module and
    the facade's documentation)."""
    return _forward(gate_up_or_gate, up, weight, limit, out)[0]


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
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward step: ``(d_gate_up, None, d_weight)`` for a merged input, ``(d_gate, d_up, d_weight)`` for two
    tensors; ``d_weight`` is None without a weight (see the module and the facade's documentation)."""
    return _backward(grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight)


class _Gradients(torch.autograd.Function):
    """The backward step of :class:`CakeClampedSwiGLUDetFunction` as an autograd function whose own backward raises.
    The autograd function's backward runs through it when the engine records the gradients (``create_graph=True``),
    so differentiating them raises ``RuntimeError`` instead of dropping this operation's second-order terms; the
    gradients are the step's (its forward runs the step, without autograd)."""

    @staticmethod
    def forward(ctx, grad_h, gate_up_or_gate, up, weight, limit):
        return _backward(grad_h, gate_up_or_gate, up, weight, limit, None, None, None, rounded=True)

    @staticmethod
    def backward(ctx, *grads):
        raise RuntimeError(
            f"{_WHAT} has no double backward: the gradients of cake_clamped_swiglu_det are not differentiable"
        )


class CakeClampedSwiGLUDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_clamped_swiglu_det`: saves only its inputs (``gate_up_or_gate``, ``up`` and
    ``weight``; no activation, no FP32 intermediate) and recomputes the activation in the backward, so
    ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as saved state.  The gradient of a
    merged ``[R, 4608]`` input is one ``[R, 4608]`` tensor; ``limit`` gets no gradient.  ``limit`` is checked once,
    in the forward, and its FP32 value is reused by the backward.  The backward has no output buffer, ``grad_up`` or
    ``grad_weight`` to check; it checks the upstream gradient and, in the same one pass as a step, the saved operands
    again: a saved-tensors hook (activation offloading) or a ``.data`` assignment can change their device, shape or
    layout after the forward without the autograd version check seeing it.  No double backward: differentiating
    gradients computed with ``create_graph=True`` raises ``RuntimeError``."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit):
        h, L = _forward(gate_up_or_gate, up, weight, limit, None)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        ctx.limit = L
        return h

    @staticmethod
    def backward(ctx, grad_h):
        gate_up_or_gate, up, weight = ctx.saved_tensors
        if torch.is_grad_enabled():  # create_graph=True: the gradients' own backward raises
            grads = _Gradients.apply(grad_h, gate_up_or_gate, up, weight, ctx.limit)
        else:
            grads = _backward(grad_h, gate_up_or_gate, up, weight, ctx.limit, None, None, None, rounded=True)
        d_first, d_up, d_weight = grads
        need_first, need_up, need_weight = ctx.needs_input_grad[:3]
        return (
            d_first if need_first else None,
            d_up if need_up else None,
            d_weight if need_weight else None,
            None,
        )


def cake_clamped_swiglu_det(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
) -> torch.Tensor:
    """Differentiable routed-expert clamped SwiGLU: :class:`CakeClampedSwiGLUDetFunction` applied to the
    arguments."""
    return CakeClampedSwiGLUDetFunction.apply(gate_up_or_gate, up, weight, limit)


__all__ = [
    "ARCHES",
    "INTERMEDIATE_SIZE",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeClampedSwiGLUDetFunction",
    "cake_clamped_swiglu_det",
    "cake_clamped_swiglu_det_train_backward",
    "cake_clamped_swiglu_det_train_forward",
    "is_cake_clamped_swiglu_det_train_supported",
]
