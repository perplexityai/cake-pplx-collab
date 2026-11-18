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

Host design (every control is an argument; no environment variable):

- Operands.  ``gate`` and ``up`` are BF16 ``[R, 2304]``, as two tensors or as
  the two halves of one ``[R, 4608]`` tensor (gate columns ``[0, 2304)``, up
  ``[2304, 4608)``; pass it as the first argument with ``up=None``).  The
  kernels take every BF16 tensor's row stride in elements, so the halves of a
  merged tensor, and row slices of wider tensors, are read in place.  An
  input the kernels cannot read (not 2-D with a unit last stride, a row
  stride that is not a multiple of 8 elements, a base address that is not
  16-byte aligned) is copied once to a contiguous tensor.  ``weight`` is FP32
  ``[R]`` (copied once when not contiguous), or ``None`` for an unweighted
  expert (``w = 1``; the kernels then read no weight and get a cached
  one-element buffer of the device).  ``limit`` is ``L``, a positive finite
  Python number rounded to FP32; it is a launch argument, as are ``R``, the
  weight flag and the strides, so one build serves every call.
- Outputs.  ``h`` is a new contiguous BF16 ``[R, 2304]`` tensor, or the
  caller's ``out``.  The backward's ``d_gate`` and ``d_up`` have the layout
  of the inputs: two new contiguous ``[R, 2304]`` tensors, or one new
  contiguous ``[R, 4608]`` gradient for a merged input.  ``d_weight`` is FP32
  ``[R]``; the backward kernel writes it for every call, so an unweighted
  backward allocates it and drops it.  A caller's output buffer must already
  be a layout the kernels write (2-D, unit last stride, a row stride that is a
  multiple of 8 elements and at least the row width, 16-byte aligned; FP32
  ``[R]`` contiguous for the weight gradient): it is refused with a
  ``ValueError`` otherwise, never copied.  Outputs must not overlap the
  inputs (not checked).
- Launches.  One kernel per call (``cake_jit.MODULES``, the record of the
  stage through ``cake_jit.PROGRAM_OF``), CTAs of ``threads / lanes`` rows
  (256 threads and 32 lanes a row: 8 rows), one CTA per block of rows on the
  grid's x axis (``cake_jit.grid``).  ``R == 0`` launches nothing and loads no
  library.
  Every launch is a plain stream-ordered launch on the current torch stream
  of the operands' device (a tvm-ffi call whose arguments include PyTorch
  CUDA tensors runs on that stream, a CUDA graph's capture stream during
  capture; the binding makes the device current): nothing cooperative, no
  PDL, no dynamic shared memory.  Nothing here reads device memory, reads an
  environment variable or synchronizes; the checks read dtypes, shapes,
  strides, devices and data pointers only.
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
  backward recomputes the activation.  So a non-reentrant checkpointed
  forward keeps nothing but its output.
- Mode: the mode is part of the name (``det``).  Every output is elementwise
  or a sum within one row in one fixed order, so every launch gives the same
  bits for the same inputs on the same architecture, whatever the row count,
  the layout, the stream or a CUDA graph; ``torch.use_deterministic_algorithms``
  does not affect any call, and no entry point takes a ``deterministic``
  keyword (passing one is a ``TypeError``).  The family has no
  non-deterministic module.
"""

from __future__ import annotations

import functools
import math
import numbers
import struct
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


def run_module(stage: str, values: dict[str, Any], *, rows: int, device_index: int) -> None:
    """Launch the record of ``stage`` (``cake_jit.PROGRAM_OF``) with its kernel parameters named in ``values`` on the
    grid of ``rows`` rows, on the current torch stream of device ``device_index``.

    The first argument of every record is a CUDA tensor of that device: tvm-ffi runs the call on PyTorch's current
    stream of its device, and the binding makes that device current for the launch."""
    name = cake_jit.PROGRAM_OF[stage]
    args = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    grid = cake_jit.grid(stage, rows)
    _entry(name, device_arch(device_index))(*args, int(grid[0]), int(grid[1]), int(grid[2]))


@functools.cache
def _unread_weight(device_index: int) -> torch.Tensor:
    """The one-element FP32 buffer an unweighted launch passes as ``w`` (``weighted`` 0: the kernels never read
    it), one per device for the process."""
    return torch.empty(1, dtype=torch.float32, device=torch.device("cuda", device_index))


# ----------------------------------------------------------------------------------------------------------------
# Checks: metadata only (dtypes, shapes, strides, devices, data pointers), nothing reads tensor contents


def _limit(limit: Any) -> float:
    """``limit`` rounded to FP32 (the kernels' ``L``): a positive finite Python number.  A tensor is refused, as
    reading it would synchronize the host."""
    if isinstance(limit, (bool, torch.Tensor)) or not isinstance(limit, numbers.Real):
        raise TypeError(
            f"limit must be a Python float (a run-time FP32 scalar of the launch), got {type(limit).__name__}"
        )
    value = float(limit)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"limit must be finite and positive, got {value!r}")
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        raise ValueError(f"limit {value!r} is past the FP32 range") from None
    if rounded <= 0.0:
        raise ValueError(f"limit {value!r} rounds to 0 in FP32")
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


def _readable(tensor: torch.Tensor) -> bool:
    """Whether the kernels read the 2-D BF16 ``tensor`` in place: unit last stride, a row stride that is a multiple
    of 8 elements (a single row's is never used), a 16-byte aligned base."""
    rows = tensor.shape[0]
    return (
        tensor.stride(1) == 1
        and (rows <= 1 or tensor.stride(0) % VECTOR_ELEMENTS == 0)
        and tensor.data_ptr() % ALIGNMENT_BYTES == 0
    )


def _row_stride(tensor: torch.Tensor) -> int:
    """The row stride the kernels get: the tensor's, or 0 for a single row (never dereferenced)."""
    return int(tensor.stride(0)) if tensor.shape[0] > 1 else 0


def _input(
    tensor: Any, name: str, *, rows: Optional[int], cols: int, device: Optional[torch.device]
) -> torch.Tensor:
    """A BF16 ``[rows, cols]`` input as the kernels read it: in place when :func:`_readable`, else copied once."""
    tensor = _check_tensor(tensor, name, torch.bfloat16, device)
    if tensor.dim() != 2 or tensor.shape[1] != cols or (rows is not None and tensor.shape[0] != rows):
        want = f"[{'R' if rows is None else rows}, {cols}]"
        raise ValueError(f"{name} must have shape {want}, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 0 or _readable(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    rows: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor:
    """A BF16 ``[rows, cols]`` output: a new contiguous tensor, or the caller's when the kernels write it in place
    (rows that do not overlap); never a copy."""
    if tensor is None:
        return torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
    tensor = _check_tensor(tensor, name, torch.bfloat16, device)
    if tensor.dim() != 2 or tuple(tensor.shape) != (rows, cols):
        raise ValueError(f"{name} must have shape [{rows}, {cols}], got {tuple(tensor.shape)}")
    if rows and not (_readable(tensor) and (rows <= 1 or tensor.stride(0) >= cols)):
        raise ValueError(
            f"{name} must have a unit last stride, a row stride that is a multiple of {VECTOR_ELEMENTS} elements and "
            f"at least {cols}, and a {ALIGNMENT_BYTES}-byte aligned base; got strides {tuple(tensor.stride())} at "
            f"address {tensor.data_ptr():#x}"
        )
    return tensor


def _weight(weight: Any, *, rows: int, device: torch.device) -> Optional[torch.Tensor]:
    """``weight`` as the kernels read it: FP32 ``[rows]``, contiguous (copied once otherwise); ``None`` stays."""
    if weight is None:
        return None
    weight = _check_tensor(weight, "weight", torch.float32, device)
    if tuple(weight.shape) != (rows,):
        raise ValueError(f"weight must have shape [{rows}] (one route weight per row), got {tuple(weight.shape)}")
    if weight.is_contiguous():
        return weight
    return weight.clone(memory_format=torch.contiguous_format)


def _grad_weight(tensor: Optional[torch.Tensor], *, rows: int, device: torch.device) -> torch.Tensor:
    if tensor is None:
        return torch.empty(rows, dtype=torch.float32, device=device)
    tensor = _check_tensor(tensor, "grad_weight", torch.float32, device)
    if tuple(tensor.shape) != (rows,) or not tensor.is_contiguous():
        raise ValueError(
            f"grad_weight must be a contiguous FP32 [{rows}] tensor, got shape {tuple(tensor.shape)} and strides "
            f"{tuple(tensor.stride())}"
        )
    return tensor


def _operands(
    gate_up_or_gate: Any, up: Any, weight: Any
) -> tuple[torch.device, int, bool, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(device, R, merged, gate, up, weight)`` of a call: ``gate`` and ``up`` the ``[R, 2304]`` BF16 views the
    kernels read (the halves of the merged input when ``up`` is None), ``weight`` FP32 ``[R]`` or None."""
    first = _check_tensor(gate_up_or_gate, "gate_up_or_gate", torch.bfloat16, None)
    device = first.device
    merged = up is None
    if merged:
        gate_up = _input(first, "gate_up_or_gate (the merged [R, 4608] gate and up)", rows=None, cols=2 * WIDTH,
                         device=device)
        rows = int(gate_up.shape[0])
        gate, up_view = gate_up[:, :WIDTH], gate_up[:, WIDTH:]
    else:
        gate = _input(first, "gate_up_or_gate (the gate)", rows=None, cols=WIDTH, device=device)
        rows = int(gate.shape[0])
        up_view = _input(up, "up", rows=rows, cols=WIDTH, device=device)
    if rows >= MAX_ROWS:
        raise ValueError(f"R = {rows} rows exceeds the kernels' 32-bit row count (fewer than {MAX_ROWS})")
    return device, rows, merged, gate, up_view, _weight(weight, rows=rows, device=device)


# ----------------------------------------------------------------------------------------------------------------
# Entry points


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
    device, rows, _, gate, up_view, w = _operands(gate_up_or_gate, up, weight)
    L = _limit(limit)
    h = _output(out, "out", rows=rows, cols=WIDTH, device=device)
    index = int(device.index)
    device_arch(index)
    if rows == 0:
        return h
    values = {
        "gate": gate,
        "up": up_view,
        "w": _unread_weight(index) if w is None else w,
        "h": h,
        "rows": rows,
        "gate_stride": _row_stride(gate),
        "up_stride": _row_stride(up_view),
        "h_stride": _row_stride(h),
        "limit": L,
        "weighted": 0 if w is None else 1,
    }
    run_module("fwd", values, rows=rows, device_index=index)
    return h


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
    device, rows, merged, gate, up_view, w = _operands(gate_up_or_gate, up, weight)
    dh = _input(grad_out, "grad_out", rows=rows, cols=WIDTH, device=device)
    L = _limit(limit)
    if merged:
        if grad_up is not None:
            raise ValueError("grad_up must be None for a merged input: grad_gate_up_or_gate receives both halves")
        d_gate_up = _output(grad_gate_up_or_gate, "grad_gate_up_or_gate", rows=rows, cols=2 * WIDTH, device=device)
        d_gate, d_up = d_gate_up[:, :WIDTH], d_gate_up[:, WIDTH:]
        result: tuple[torch.Tensor, Optional[torch.Tensor]] = (d_gate_up, None)
    else:
        d_gate = _output(grad_gate_up_or_gate, "grad_gate_up_or_gate", rows=rows, cols=WIDTH, device=device)
        d_up = _output(grad_up, "grad_up", rows=rows, cols=WIDTH, device=device)
        result = (d_gate, d_up)
    if w is None and grad_weight is not None:
        raise ValueError("grad_weight needs a weight: an unweighted call has no weight gradient")
    d_weight = _grad_weight(grad_weight, rows=rows, device=device)
    index = int(device.index)
    device_arch(index)
    if rows == 0:
        return (*result, None if w is None else d_weight)
    values = {
        "gate": gate,
        "up": up_view,
        "dh": dh,
        "w": _unread_weight(index) if w is None else w,
        "d_gate": d_gate,
        "d_up": d_up,
        "d_weight": d_weight,
        "rows": rows,
        "gate_stride": _row_stride(gate),
        "up_stride": _row_stride(up_view),
        "dh_stride": _row_stride(dh),
        "d_gate_stride": _row_stride(d_gate),
        "d_up_stride": _row_stride(d_up),
        "limit": L,
        "weighted": 0 if w is None else 1,
    }
    run_module("bwd", values, rows=rows, device_index=index)
    return (*result, None if w is None else d_weight)


class CakeClampedSwiGLUDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_clamped_swiglu_det`: saves only its inputs (``gate_up_or_gate``, ``up`` and
    ``weight``; no activation, no FP32 intermediate) and recomputes the activation in the backward, so
    ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as saved state.  The gradient of a
    merged ``[R, 4608]`` input is one ``[R, 4608]`` tensor; ``limit`` gets no gradient.  No double backward."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, weight, limit):
        h = cake_clamped_swiglu_det_train_forward(gate_up_or_gate, up, weight, limit=limit)
        ctx.save_for_backward(gate_up_or_gate, up, weight)
        ctx.limit = _limit(limit)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        gate_up_or_gate, up, weight = ctx.saved_tensors
        d_first, d_up, d_weight = cake_clamped_swiglu_det_train_backward(
            grad_h, gate_up_or_gate, up, weight, limit=ctx.limit
        )
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
