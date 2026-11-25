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

Backend of :mod:`cake_pplx.cake_route_weight_scale_det_train`: the Cake
route-weight row scaling of a routed MoE expert in training, forward and
backward (SM100 / SM103), as one self-contained package (host code,
the ``cake_jit`` registry, the pinned ``cake_build`` loader and the generated
CUDA sources under ``csrc/``).  The ``@cake_api``-decorated entry points and
the user-facing documentation live in
:mod:`cake_pplx.cake_route_weight_scale_det_train`; import that module.

Host design:

- Operands.  ``x`` (the down projection's output), ``dy``, ``y`` and ``dx``
  are BF16 ``[rows, H]`` with ``H`` in ``cake_jit.WIDTHS`` (4096, 3584).
  The kernels take every BF16 row stride in elements, so row slices of wider
  or longer buffers are read and written in place.  Every BF16 tensor of a
  call with ``rows >= 1`` must have a unit column stride, a row stride that is
  a multiple of 8 elements (a single row's stride is not read) and a 16-byte
  aligned base; an output's row stride must also be at least ``H``.  ``w`` and
  ``dw`` are FP32 ``[rows]``, contiguous.  Nothing is copied: an operand of
  another layout is refused with ``ValueError``.  The autograd backward is the
  one exception: it copies an upstream gradient the kernels cannot read (an
  expanded or misaligned ``dy`` that autograd hands over) once.
- Checks.  An entry point checks every argument (types, dtypes, shapes,
  strides, alignment, then devices, then the device's architecture) before it
  allocates an output, so a call refused for its arguments allocates and
  launches nothing.  The checks read metadata only: nothing reads tensor
  contents, nothing synchronizes.
- Launches.  One kernel per call, of the record and grid cap
  ``cake_jit.route`` picks from the band of ``cake_jit.ROUTES`` that holds
  ``rows``, for stage ``fwd`` or ``bwd`` at the width of ``x``
  (``<stage>_h<H>``): the forward at H = 4096 takes ``fwd_h4096_t32`` (one
  row a CTA of 32 threads) for few rows, the forward of each width takes its
  flat program (``fwd_h<H>_flat``) for many rows, the backward at H = 3584
  takes ``bwd_h3584_warp`` (one warp a row) in its band, all with the same
  bits.  The grid is the record's (``cake_jit.grid``): one CTA per block of
  its ``rows_per_cta`` rows (4 for ``fwd_h4096``, ``fwd_h3584`` and
  ``bwd_h3584_warp``, 1 for ``fwd_h4096_t32``, ``bwd_h4096`` and
  ``bwd_h3584``), or for a flat record one CTA per chunk of its
  ``elements_per_cta`` (2048) elements, capped in a band with a cap at that
  many CTAs per SM of the device (a persistent grid).  ``rows`` and the
  strides are launch arguments, so one library per record and architecture
  serves every call; it loads (and, without a prebuilt library, JIT-builds)
  at its first launch.  ``rows == 0`` launches nothing and loads no library.
  Every launch is a plain stream-ordered launch on the current torch stream
  of the operands' device (tvm-ffi runs the call on that stream; the binding
  makes the device current): nothing cooperative, no PDL, no dynamic shared
  memory.
- Numerics.  ``y = bf16(w x)`` and ``dx = bf16(w dy)``: the FP32 product,
  rounded once to BF16 (round to nearest, no flush), so both are bit for bit
  ``(w[:, None] * x.float()).bfloat16()`` and ``(w[:, None] *
  dy.float()).bfloat16()``.  ``dw = sum over c of dy x``: the products are
  exact in FP32 and the additions form a balanced binary tree in one fixed
  order (pairs within a load of 8 columns, the loads of a lane pairwise, the
  even and odd halves, then five XOR shuffles across 32 lanes; the
  backward's four warps each sum a subtree of it); one FP32 word per row,
  stored once.  ``dw`` does not divide by ``w``: a zero-weight row gets
  signed zeros in ``y`` and ``dx`` and keeps its ``dw``.  Nothing is FP64.
- Workspace: none.  The forward allocates ``y`` only, the backward ``dx`` and
  ``dw`` only (none when the caller passes them): no ``[rows, H]`` FP32
  buffer.  Retained state of the autograd function: ``x`` and ``w`` through
  ``save_for_backward`` (references, no copy).  Under
  ``torch.utils.checkpoint(..., use_reentrant=False)`` the recomputation
  launches the forward once.  No double backward: differentiating gradients
  computed with ``create_graph=True`` raises ``RuntimeError``.
- Mode: ``det``.  Nothing sums across rows and the one sum, ``dw`` within a
  row, has one fixed order, so every call gives the same bits for the same
  inputs on the same architecture, whatever ``rows``, the layout, the stream
  or a CUDA graph.  The family has no non-deterministic module and no KCP
  module.  Nothing here reads torch's deterministic-algorithms setting, and no
  entry point takes a ``deterministic`` keyword.
- Caches: a device's architecture and SM count, the loaded libraries and
  their FFI entries.  Nothing else is kept between calls.  Only the JIT shim
  reads environment variables, to build or find a library.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from . import cake_jit
from .cake_jit import ALIGNMENT_BYTES, ARCH_BY_CAPABILITY, ARCHES, MAX_ROWS, VECTOR_ELEMENTS, WIDTHS

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))

_WHAT = "Cake route-weight row scaling training"
_BF16 = torch.bfloat16
_FP32 = torch.float32

# The kernel parameters of each stage in ``run`` order (a record's argument plan without the grid): the order every
# launch passes them in.
_PARAMETERS: dict[str, tuple[str, ...]] = {
    "fwd": ("x", "w", "y", "rows", "x_stride", "y_stride"),
    "bwd": ("x", "dy", "w", "dx", "dw", "rows", "x_stride", "dy_stride", "dx_stride"),
}


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


@functools.cache
def _sm_count(device_index: int) -> int:
    """The SM count of one CUDA device, read once per device index (the grid cap of ``cake_jit.ROUTES``)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


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


def is_cake_route_weight_scale_det_train_supported(device: torch.device) -> bool:
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
            raise ValueError(f"record {name}: parameter {key!r} follows the grid arguments")
        else:
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}")
    return tuple(names)


@functools.cache
def _entry(name: str, arch: str) -> Any:
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process), after checking that the
    record takes its parameters in the order the host passes them (:data:`_PARAMETERS`)."""
    stage = cake_jit.MODULES[name]["stage"]
    if parameters(name) != _PARAMETERS[stage]:
        raise RuntimeError(
            f"record {name}: parameters {list(parameters(name))} are not in the order the host passes them, "
            f"{list(_PARAMETERS[stage])}"
        )
    return getattr(load(name, arch), cake_jit.MODULES[name]["ffi_entry"])


# The FFI entry of each (record, device index), bound at its first launch.
_ENTRIES: dict[tuple[str, int], Any] = {}


def _launch(stage: str, width: int, device_index: int, rows: int, *arguments: Any) -> None:
    """Launch the program of ``stage`` at hidden size ``width`` on device ``device_index`` for ``rows >= 1`` rows: the
    record and grid cap ``cake_jit.route(stage, width, rows)``, with ``arguments`` (the kernel parameters in the order
    of :data:`_PARAMETERS`) and then that record's grid ``cake_jit.grid(rows, name, ctas_per_sm, sm_count)`` for the
    device's SM count.  The one place a kernel is launched."""
    name, ctas_per_sm = cake_jit.route(stage, width, rows)
    key = (name, device_index)
    entry = _ENTRIES.get(key)
    if entry is None:
        entry = _ENTRIES[key] = _entry(name, device_arch(device_index))
    entry(*arguments, *cake_jit.grid(rows, name, ctas_per_sm, _sm_count(device_index)))


# ----------------------------------------------------------------------------------------------------------------
# Checks: metadata only (dtypes, shapes, strides, data pointers, devices); nothing reads tensor contents.


def _tensor(tensor: Any, name: str, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    return tensor


def _matrix(
    tensor: Any, name: str, *, rows: Optional[int] = None, width: Optional[int] = None, output: bool = False
) -> tuple[int, int]:
    """``(rows, H)`` of a BF16 ``[rows, H]`` operand the kernels read or write in place: ``H`` in ``WIDTHS`` (or equal
    to ``width``), ``rows`` equal to ``rows`` when given; with at least one row a unit column stride, a row stride that
    is a multiple of 8 elements (a single row's is not read; an output's at least ``H``) and a 16-byte aligned base."""
    tensor = _tensor(tensor, name, _BF16)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 2-D [rows, H], got shape {tuple(tensor.shape)}")
    n, h = (int(size) for size in tensor.shape)
    if width is None and h not in WIDTHS:
        raise ValueError(f"{name} must have H in {WIDTHS} columns, got shape {tuple(tensor.shape)}")
    if width is not None and h != width:
        raise ValueError(f"{name} must have H = {width} columns (x's), got shape {tuple(tensor.shape)}")
    if rows is not None and n != rows:
        raise ValueError(f"{name} must have {rows} rows (x's), got shape {tuple(tensor.shape)}")
    if n == 0:
        return n, h
    if tensor.stride(1) != 1:
        raise ValueError(f"{name} must have a unit column stride, got strides {tuple(tensor.stride())}")
    if n > 1 and tensor.stride(0) % VECTOR_ELEMENTS:
        raise ValueError(
            f"{name}'s row stride must be a multiple of {VECTOR_ELEMENTS} elements, got strides "
            f"{tuple(tensor.stride())}"
        )
    if output and n > 1 and tensor.stride(0) < h:
        raise ValueError(f"{name}'s rows must not overlap: row stride {tensor.stride(0)} is below H = {h}")
    if tensor.data_ptr() % ALIGNMENT_BYTES:
        raise ValueError(f"{name} must start on a {ALIGNMENT_BYTES}-byte boundary, got address {tensor.data_ptr():#x}")
    return n, h


def _vector(tensor: Any, name: str, rows: int) -> None:
    """An FP32 ``[rows]`` contiguous operand (``w`` or ``dw``)."""
    tensor = _tensor(tensor, name, _FP32)
    if tuple(tensor.shape) != (rows,):
        raise ValueError(f"{name} must have shape [{rows}] (one value per row of x), got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(tensor.stride())}")


def _device_index(x: torch.Tensor, *others: tuple[str, Optional[torch.Tensor]]) -> int:
    """The CUDA device index of ``x``, which every other operand given must share."""
    if x.device.type != "cuda":
        raise ValueError(f"{_WHAT} requires CUDA tensors; x is on {x.device}")
    for name, tensor in others:
        if tensor is not None and tensor.device != x.device:
            raise ValueError(f"{name} must live on x's device {x.device}, got {tensor.device}")
    return int(x.device.index)


def _check_rows(rows: int) -> None:
    if rows >= MAX_ROWS:
        raise ValueError(f"rows = {rows} exceeds the package's row bound (fewer than {MAX_ROWS})")


def _readable(dy: torch.Tensor) -> torch.Tensor:
    """An upstream gradient the kernels can read: ``dy`` itself when it has a unit column stride, a row stride that
    is a multiple of 8 elements and a 16-byte aligned base, else a contiguous copy (autograd may hand over an expanded
    or otherwise strided gradient)."""
    if not isinstance(dy, torch.Tensor) or dy.dim() != 2 or dy.shape[0] == 0:
        return dy
    if (
        dy.stride(1) == 1
        and (dy.shape[0] == 1 or dy.stride(0) % VECTOR_ELEMENTS == 0)
        and dy.data_ptr() % ALIGNMENT_BYTES == 0
    ):
        return dy
    return dy.clone(memory_format=torch.contiguous_format)


# ----------------------------------------------------------------------------------------------------------------
# Entry points


def cake_route_weight_scale_det_train_forward(
    x: torch.Tensor, w: torch.Tensor, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Forward step: ``y = bf16(w[:, None] * x)``, the FP32 product rounded once (see the module and the facade's
    documentation)."""
    rows, width = _matrix(x, "x")
    _vector(w, "w", rows)
    if out is not None:
        _matrix(out, "out", rows=rows, width=width, output=True)
    _check_rows(rows)
    index = _device_index(x, ("w", w), ("out", out))
    device_arch(index)  # RuntimeError on a device outside ARCHES, before the allocation
    y = torch.empty((rows, width), dtype=_BF16, device=x.device) if out is None else out
    if rows:
        many = rows > 1
        _launch("fwd", width, index, rows, x, w, y, rows, x.stride(0) if many else 0, y.stride(0) if many else 0)
    return y


def cake_route_weight_scale_det_train_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    w: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward step: ``(dx, dw)`` with ``dx = bf16(w[:, None] * dy)`` and ``dw = sum over c of dy * x`` (FP32, one
    fixed order per row), in one pass over ``x`` and ``dy`` (see the module and the facade's documentation)."""
    rows, width = _matrix(x, "x")
    _matrix(dy, "dy", rows=rows, width=width)
    _vector(w, "w", rows)
    if dx is not None:
        _matrix(dx, "dx", rows=rows, width=width, output=True)
    if dw is not None:
        _vector(dw, "dw", rows)
    _check_rows(rows)
    index = _device_index(x, ("dy", dy), ("w", w), ("dx", dx), ("dw", dw))
    device_arch(index)  # RuntimeError on a device outside ARCHES, before the allocation
    if dx is None:
        dx = torch.empty((rows, width), dtype=_BF16, device=x.device)
    if dw is None:
        dw = torch.empty((rows,), dtype=_FP32, device=x.device)
    if rows:
        many = rows > 1
        _launch(
            "bwd",
            width,
            index,
            rows,
            x,
            dy,
            w,
            dx,
            dw,
            rows,
            x.stride(0) if many else 0,
            dy.stride(0) if many else 0,
            dx.stride(0) if many else 0,
        )
    return dx, dw


class CakeRouteWeightScaleDetFunction(torch.autograd.Function):
    """Autograd of :func:`cake_route_weight_scale_det`: ``y = bf16(w[:, None] * x)``.  Saves only its inputs ``x`` and
    ``w`` (no FP32 intermediate), so ``torch.utils.checkpoint(..., use_reentrant=False)`` sees only the inputs as saved
    state.  The backward runs the backward step once (``dx`` and ``dw`` in one launch), copying an upstream gradient
    the kernels cannot read once, and checks the saved operands again (a saved-tensors hook can change them).  No
    double backward: differentiating gradients computed with ``create_graph=True`` raises ``RuntimeError``."""

    @staticmethod
    def forward(ctx, x, w):
        y = cake_route_weight_scale_det_train_forward(x, w)
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx, dw = cake_route_weight_scale_det_train_backward(x, _readable(dy), w)
        need_x, need_w = ctx.needs_input_grad
        return (dx if need_x else None, dw if need_w else None)


def cake_route_weight_scale_det(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Differentiable route-weight row scaling: :class:`CakeRouteWeightScaleDetFunction` applied to ``x`` and ``w``."""
    return CakeRouteWeightScaleDetFunction.apply(x, w)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "WIDTHS",
    "CakeRouteWeightScaleDetFunction",
    "cake_route_weight_scale_det",
    "cake_route_weight_scale_det_train_backward",
    "cake_route_weight_scale_det_train_forward",
    "is_cake_route_weight_scale_det_train_supported",
]
