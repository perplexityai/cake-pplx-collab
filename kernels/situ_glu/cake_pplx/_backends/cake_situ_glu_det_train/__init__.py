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

Backend of :mod:`cake_pplx.cake_situ_glu_det_train`: the Cake SiTU-GLU
activation of an MLP in training, forward and backward (SM100 / SM103), as one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  The
``@cake_api``-decorated entry points and the user-facing documentation live in
:mod:`cake_pplx.cake_situ_glu_det_train`; import that module.

Host design (every control is an argument; no environment variable):

- Operands.  ``gate`` and ``up`` are BF16 ``[R, I]``, as two tensors or as the
  two halves of one ``[R, 2I]`` tensor (gate columns ``[0, I)``, up
  ``[I, 2I)``; pass it as the first argument with ``up=None``).  ``I`` is any
  positive multiple of 256.  The kernels take every BF16 tensor's row stride
  in elements, so the halves of a merged tensor, and row slices of wider
  tensors, are read in place.  An input the kernels cannot read (a last stride
  that is not 1, a row stride that is not a multiple of 8 elements, a base
  address that is not 16-byte aligned) is copied once to a contiguous tensor.
  ``beta`` and ``lb`` are finite positive Python numbers rounded to FP32
  (defaults 4.0 and 25.0); they are launch arguments, as are ``R``, ``I`` and
  the strides, so one build serves every call.
- Outputs.  ``h`` is a new contiguous BF16 ``[R, I]`` tensor, or the caller's
  ``out``.  The backward's ``d_gate`` and ``d_up`` have the layout of the
  inputs: two new contiguous ``[R, I]`` tensors, or one new contiguous
  ``[R, 2I]`` gradient for a merged input.  A caller's output buffer must
  already be a layout the kernels write (2-D, a unit last stride, a row stride
  that is a multiple of 8 elements and at least the row width, a 16-byte
  aligned base): it is refused with a ``ValueError`` otherwise, never copied.
  An output must not overlap any input or another output (not checked).
- Checks.  Every argument is checked (types, dtypes, shapes, devices, the
  width, the scalars, the output buffers and the device's architecture) before
  an input is copied or an output allocated, so a refused call copies,
  allocates and launches nothing.
- Launches.  One kernel per call (``cake_jit.MODULES``, the record of the
  stage through ``cake_jit.PROGRAM_OF``): CTAs of 128 threads (the record's
  launch block), one CTA per row on the grid's x axis (``cake_jit.grid``); a
  CTA's four warps take the row's segments of 256 columns in turn.  ``R == 0``
  launches nothing and loads no library.  Every launch is a plain
  stream-ordered launch on the current torch stream of the operands' device
  (a tvm-ffi call whose arguments include PyTorch CUDA tensors runs on that
  stream, a CUDA graph's capture stream during capture; the binding makes the
  device current): nothing cooperative, no PDL, no dynamic shared memory.
  Nothing here reads device memory, reads an environment variable or
  synchronizes; the checks read dtypes, shapes, strides, devices and data
  pointers only.
- Numerics.  From the BF16 values widened to FP32, with
  ``E = 2^(-2 |g| log2(e) / beta)``, ``P = 2^(-|g| log2(e))`` and
  ``W = 2^(-2 |u| log2(e) / lb)``, all in ``[0, 1]`` so that no exponential
  overflows: ``tanh(|g| / beta) = (1 - E) / (1 + E)``, ``sigmoid(g) =
  1 / (1 + P)`` for ``g >= 0`` and ``P / (1 + P)`` for ``g < 0``, and
  ``1 - tanh^2 = 4 E / (1 + E)^2`` (no cancellation).  Below
  ``|g| / beta = 1/16`` (``|u| / lb`` for the up) the tanh is the odd
  polynomial ``z - z^3 / 3 + 2 z^5 / 15``, so ``g = 0`` or ``u = 0`` gives
  ``h = 0`` exactly.  ``beta = 4`` takes ``P = E E``; any other ``beta`` takes
  ``P`` from its own exponential, the same choice for every element of a call.
  The exponentials are the approximate base-2 exponential; the forward
  flushes a subnormal ``E`` or ``W`` (they only enter ``1 + E`` and ``1 + W``
  there) and keeps a subnormal ``P``, and the backward keeps every subnormal,
  so a saturated gate or up keeps a nonzero gradient where FP32 holds one.
  One reciprocal per element, of ``(1 + E)(1 + P)(1 + W)`` in ``[1, 8]``; the
  backward multiplies it back out into the three factors' reciprocals.
  ``h``, ``d_gate`` and ``d_up`` are each rounded once to BF16.  Every product
  and FMA rounds to nearest without flushing subnormals; nothing is FP64.
- Accuracy: the tests hold ``h``, ``d_gate`` and ``d_up`` to the PyTorch BF16
  naive implementation of the same operation (the formula on the BF16
  tensors, the gradients from PyTorch autograd), each against a float64
  reference computed from the BF16 inputs: over 32 random inputs of a case, in
  the L2 and in the max norm of the error, the median and the largest error of
  the output are each at most 1.005 times the naive implementation's, and 0 on
  every input where the naive implementation's is 0
  (``tests/cake/test_helpers/cake_naive.py``).  No output has an error budget
  of its own.
- Workspace: none.  Retained state: the autograd function saves only its
  inputs (``gate`` and ``up`` or the merged tensor) through
  ``save_for_backward``, references to the caller's tensors with no copy; the
  backward recomputes the activation.
  ``torch.utils.checkpoint(..., use_reentrant=False)`` keeps the same inputs;
  the function adds no saved state of its own and no activation.
- Mode: the mode is part of the name (``det``).  Every output is elementwise,
  so every launch gives the same bits for the same inputs on the same
  architecture, whatever the row count, the layout, the stream or a CUDA graph;
  ``torch.use_deterministic_algorithms`` does not affect any call, and no entry
  point takes a ``deterministic`` keyword (passing one is a ``TypeError``).
  The family has no non-deterministic module (there is no reduction to
  reorder) and no KCP module (rows are independent).
"""

from __future__ import annotations

import functools
import math
import numbers
import struct
from typing import Any, Optional

import torch
from torch.autograd.function import once_differentiable

from . import cake_jit
from .cake_jit import (
    ALIGNMENT_BYTES,
    ARCH_BY_CAPABILITY,
    ARCHES,
    MAX_ROWS,
    MAX_WIDTH,
    SEGMENT,
    VECTOR_ELEMENTS,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))
# I, the width of gate, up, h and the gradients, is any positive multiple of this.
WIDTH_MULTIPLE: int = SEGMENT

_WHAT = "Cake SiTU-GLU training"


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


def is_cake_situ_glu_det_train_supported(device: torch.device) -> bool:
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
        raise RuntimeError(f"{_WHAT} record {name} is not delivered for {arch}; its architectures: {record['arches']}")
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


# ----------------------------------------------------------------------------------------------------------------
# Checks: metadata only (dtypes, shapes, strides, devices, data pointers), nothing reads tensor contents


def _scalar(value: Any, name: str) -> float:
    """``value`` rounded to FP32 (the kernels' ``beta`` or ``lb``): a positive finite Python number.  A tensor is
    refused, as reading it would synchronize the host."""
    if isinstance(value, (bool, torch.Tensor)) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a Python float (a run-time FP32 scalar of the launch), got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {number!r}")
    try:
        # The standard "<f" format refuses a value past FP32 (OverflowError); the native "f" format would make it
        # infinite without an error.
        rounded = struct.unpack("<f", struct.pack("<f", number))[0]
    except OverflowError:
        raise ValueError(f"{name} {number!r} is past the FP32 range") from None
    if rounded <= 0.0:
        raise ValueError(f"{name} {number!r} rounds to 0 in FP32")
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


def _bf16_2d(
    tensor: Any, name: str, *, rows: Optional[int], cols: Optional[int], device: Optional[torch.device]
) -> torch.Tensor:
    """``tensor`` checked as a BF16 ``[rows, cols]`` tensor (``None``: any count) on ``device``."""
    tensor = _check_tensor(tensor, name, torch.bfloat16, device)
    if (
        tensor.dim() != 2
        or (cols is not None and tensor.shape[1] != cols)
        or (rows is not None and tensor.shape[0] != rows)
    ):
        want = f"[{'R' if rows is None else rows}, {'I' if cols is None else cols}]"
        raise ValueError(f"{name} must have shape {want}, got {tuple(tensor.shape)}")
    return tensor


def _width(cols: int, name: str, *, merged: bool) -> int:
    """The width ``I`` of a call whose first operand ``name`` has ``cols`` columns (``2 I`` when ``merged``)."""
    width = cols // 2 if merged else cols
    if (merged and cols % 2) or width < SEGMENT or width % SEGMENT or width > MAX_WIDTH:
        what = "half its columns" if merged else "its columns"
        raise ValueError(
            f"{name}: the width I ({what}) must be a positive multiple of {SEGMENT} up to {MAX_WIDTH}; got {cols} "
            "columns"
        )
    return width


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


def _read(tensor: torch.Tensor) -> torch.Tensor:
    """A checked input as the kernels read it: in place when :func:`_readable`, else copied once."""
    if tensor.shape[0] == 0 or _readable(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _check_output(tensor: Any, name: str, *, rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """A caller's BF16 ``[rows, cols]`` output buffer, checked: the kernels write it in place (rows that do not
    overlap); it is never copied."""
    tensor = _bf16_2d(tensor, name, rows=rows, cols=cols, device=device)
    if rows and not (_readable(tensor) and (rows <= 1 or tensor.stride(0) >= cols)):
        raise ValueError(
            f"{name} must have a unit last stride, a row stride that is a multiple of {VECTOR_ELEMENTS} elements and "
            f"at least {cols}, and a {ALIGNMENT_BYTES}-byte aligned base; got strides {tuple(tensor.stride())} at "
            f"address {tensor.data_ptr():#x}"
        )
    return tensor


def _operands(
    gate_up_or_gate: Any, up: Any
) -> tuple[torch.device, int, int, bool, torch.Tensor, Optional[torch.Tensor]]:
    """``(device, R, I, merged, first, up)`` of a call, checked (nothing copied yet): ``first`` the merged
    ``[R, 2I]`` tensor when ``up`` is None, else the gate."""
    first = _check_tensor(gate_up_or_gate, "gate_up_or_gate", torch.bfloat16, None)
    device = first.device
    merged = up is None
    if merged:
        what = "gate_up_or_gate (the merged [R, 2I] gate and up)"
        first = _bf16_2d(first, what, rows=None, cols=None, device=device)
        width = _width(int(first.shape[1]), what, merged=True)
        second = None
    else:
        what = "gate_up_or_gate (the gate)"
        first = _bf16_2d(first, what, rows=None, cols=None, device=device)
        width = _width(int(first.shape[1]), what, merged=False)
        second = _bf16_2d(up, "up", rows=int(first.shape[0]), cols=width, device=device)
    rows = int(first.shape[0])
    if rows > MAX_ROWS:
        raise ValueError(f"R = {rows} rows exceeds the kernels' row count (at most {MAX_ROWS})")
    return device, rows, width, merged, first, second


def _halves(first: torch.Tensor, second: Optional[torch.Tensor], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``[R, I]`` gate and up the kernels read: the halves of the merged input (copied once as a whole when the
    kernels cannot read it), or the two tensors (each copied once when the kernels cannot read it)."""
    if second is None:
        merged = _read(first)
        return merged[:, :width], merged[:, width:]
    return _read(first), _read(second)


# ----------------------------------------------------------------------------------------------------------------
# Entry points


def cake_situ_glu_det_train_forward(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward step: ``h = bf16(beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb))``, one rounding (see the
    module and the facade's documentation)."""
    device, rows, width, _, first, second = _operands(gate_up_or_gate, up)
    beta32, lb32 = _scalar(beta, "beta"), _scalar(lb, "lb")
    if out is not None:
        out = _check_output(out, "out", rows=rows, cols=width, device=device)
    index = int(device.index)
    device_arch(index)
    gate, up_view = _halves(first, second, width)
    h = torch.empty(rows, width, dtype=torch.bfloat16, device=device) if out is None else out
    if rows == 0:
        return h
    values = {
        "gate": gate,
        "up": up_view,
        "h": h,
        "rows": rows,
        "width": width,
        "gate_stride": _row_stride(gate),
        "up_stride": _row_stride(up_view),
        "h_stride": _row_stride(h),
        "beta": beta32,
        "lb": lb32,
    }
    run_module("fwd", values, rows=rows, device_index=index)
    return h


def cake_situ_glu_det_train_backward(
    grad_out: torch.Tensor,
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
    grad_gate_up_or_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Backward step: ``(d_gate_up, None)`` for a merged input, ``(d_gate, d_up)`` for two tensors (see the module
    and the facade's documentation)."""
    device, rows, width, merged, first, second = _operands(gate_up_or_gate, up)
    dh = _bf16_2d(grad_out, "grad_out", rows=rows, cols=width, device=device)
    beta32, lb32 = _scalar(beta, "beta"), _scalar(lb, "lb")
    if merged:
        if grad_up is not None:
            raise ValueError("grad_up must be None for a merged input: grad_gate_up_or_gate receives both halves")
        if grad_gate_up_or_gate is not None:
            grad_gate_up_or_gate = _check_output(
                grad_gate_up_or_gate, "grad_gate_up_or_gate", rows=rows, cols=2 * width, device=device
            )
    else:
        if grad_gate_up_or_gate is not None:
            grad_gate_up_or_gate = _check_output(
                grad_gate_up_or_gate, "grad_gate_up_or_gate", rows=rows, cols=width, device=device
            )
        if grad_up is not None:
            grad_up = _check_output(grad_up, "grad_up", rows=rows, cols=width, device=device)
    index = int(device.index)
    device_arch(index)
    gate, up_view = _halves(first, second, width)
    dh = _read(dh)
    if merged:
        d_gate_up = grad_gate_up_or_gate
        if d_gate_up is None:
            d_gate_up = torch.empty(rows, 2 * width, dtype=torch.bfloat16, device=device)
        d_gate, d_up = d_gate_up[:, :width], d_gate_up[:, width:]
        result: tuple[torch.Tensor, Optional[torch.Tensor]] = (d_gate_up, None)
    else:
        d_gate, d_up = grad_gate_up_or_gate, grad_up
        if d_gate is None:
            d_gate = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        if d_up is None:
            d_up = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        result = (d_gate, d_up)
    if rows == 0:
        return result
    values = {
        "gate": gate,
        "up": up_view,
        "dh": dh,
        "d_gate": d_gate,
        "d_up": d_up,
        "rows": rows,
        "width": width,
        "gate_stride": _row_stride(gate),
        "up_stride": _row_stride(up_view),
        "dh_stride": _row_stride(dh),
        "d_gate_stride": _row_stride(d_gate),
        "d_up_stride": _row_stride(d_up),
        "beta": beta32,
        "lb": lb32,
    }
    run_module("bwd", values, rows=rows, device_index=index)
    return result


class CakeSiTUGLUDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_situ_glu_det`: saves only its inputs (``gate_up_or_gate`` and ``up``; no activation,
    no FP32 intermediate) and recomputes the activation in the backward, so
    ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as saved state.  The gradient of a merged
    ``[R, 2I]`` input is one ``[R, 2I]`` tensor; ``beta`` and ``lb`` get no gradient.  No double backward
    (``once_differentiable``): with ``create_graph=True`` the gradients are constants of the graph, and differentiating
    them raises ``RuntimeError`` when the upstream gradient requires grad."""

    @staticmethod
    def forward(ctx, gate_up_or_gate, up, beta, lb):
        h = cake_situ_glu_det_train_forward(gate_up_or_gate, up, beta=beta, lb=lb)
        ctx.save_for_backward(gate_up_or_gate, up)
        ctx.beta = _scalar(beta, "beta")
        ctx.lb = _scalar(lb, "lb")
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_h):
        gate_up_or_gate, up = ctx.saved_tensors
        d_first, d_up = cake_situ_glu_det_train_backward(grad_h, gate_up_or_gate, up, beta=ctx.beta, lb=ctx.lb)
        need_first, need_up = ctx.needs_input_grad[:2]
        return (d_first if need_first else None, d_up if need_up else None, None, None)


def cake_situ_glu_det(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
) -> torch.Tensor:
    """Differentiable SiTU-GLU: :class:`CakeSiTUGLUDetFunction` applied to the arguments."""
    return CakeSiTUGLUDetFunction.apply(gate_up_or_gate, up, beta, lb)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "WIDTH_MULTIPLE",
    "CakeSiTUGLUDetFunction",
    "cake_situ_glu_det",
    "cake_situ_glu_det_train_backward",
    "cake_situ_glu_det_train_forward",
    "is_cake_situ_glu_det_train_supported",
]
