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

Loader, checks and launch of the Cake Engram row lookup: rows of a table of
FP8 E4M3 values with one E8M0 scale per 32 columns, gathered by row id and
written as BF16 (SM100 / SM103 / SM107).

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds
the two programs (their ``library`` stems; the ``build.json`` entries are
``<library>_<arch>``), :data:`.cake_jit.SMALL_MAX_POSITIONS` the most looked-up
rows the small program serves and :func:`.cake_jit.grid` the launch grid.
:mod:`.cake_build` serves the library from a prebuilt ``aot/`` file when a
builder compiled it and otherwise through the package's JIT build shim
(``_cake_pplx_jit.py``) from the sources and flags of ``build.json``.

A call launches one kernel on the current torch stream of the operands' device
(the first call on a device two, below): the small program (record
``lookup_small``) for at most :data:`.cake_jit.SMALL_MAX_POSITIONS` looked-up
rows, the main program (record ``lookup``) above.  The two programs write the
same bytes, so the choice changes no output bit; a call captured in a CUDA
graph keeps the program its row count picked at capture.  The first call on a
device launches both programs there, on its own operands: first the program
its row count does not pick, then its own (both write the same bytes, so the
output is the same).  Before them it looks up both libraries (a prebuilt file
or a JIT build, once per process and architecture).  The first launch of a
kernel on a device loads the kernel's module there (CUDA's lazy module
loading, the default, loads it then), so after that call no call on the device
loads a library or a module, whichever program it launches: a later call
captured in a CUDA graph launches a kernel that is already loaded.  Make the
first call on each device outside a CUDA graph capture.  For each looked-up
row ``p`` with ``r = row_ids[p]`` and each column
``c``: ``out[p, c] = bf16(float(weight[r, c]) * float(scale[r, c // 32]))``.
The kernel decodes the E4M3 values and the E8M0 scale to FP32, multiplies in
FP32 without flushing subnormals and rounds the product once to BF16.  That is
the exact product rounded once: a finite E4M3 value has at most 4 significant
bits and is a multiple of 2^-9, and a finite E8M0 scale is a power of two, so
every product is exact in FP32 unless it passes the FP32 range (then it is
``±inf``, as in BF16).  The E8M0 byte ``0xFF`` is NaN and makes its 32
columns NaN; the byte ``0`` is 2^-127, an FP32 subnormal.  Byte offsets are
64-bit, so tables past 2^31 bytes and outputs past 2^31 elements are served.

Each program serves both table layouts and every ``row_ids`` shape: the
binding reads the row strides of ``weight`` and ``scale`` from the tensors
(two contiguous tensors, or two views into one packed ``[R, 264]`` byte
buffer), and the number of looked-up rows is a run-time value.  Nothing here
reads device memory or synchronizes, and only the JIT shim the loader calls
reads environment variables (to build or find the library); the grid
and the output size come from tensor shapes, and the output is the only
allocation.  The row ids must lie in ``[0, R)``.  An id outside the range (a
negative id included) stops the launch with a device-side trap before its row
is read; the error surfaces at the call or the next synchronizing CUDA call and
leaves the CUDA context unusable.  The host checks no id (that would
synchronize).

This is the deterministic package of its family, and the family has no
non-deterministic package: every output element is one product rounded once
and nothing is summed, so the output repeats bit for bit.  Nothing here reads
torch's deterministic-algorithms setting, and no call alerts.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GROUP",
    "GROUPS",
    "PACKED_ROW_BYTES",
    "RECORD",
    "RECORDS",
    "ROW_DIM",
    "SMALL_RECORD",
    "arch_for_capability",
    "cake_engram_lookup_det_train_forward",
    "cake_engram_lookup_det_train_split_packed",
    "device_arch",
    "is_prebuilt",
    "library_name",
    "load",
    "parameters",
    "record",
    "sm_count",
    "specs_for_arch",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 /
# GB300 (``sm_103a``) and R200 (``sm_107a``), each compiled with its exact
# ``-gencode`` flag set.  No other architecture has a build.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
ROW_DIM: int = cake_jit.ROW_DIM
GROUP: int = cake_jit.GROUP
GROUPS: int = cake_jit.GROUPS
PACKED_ROW_BYTES: int = cake_jit.PACKED_ROW_BYTES

# The registry records of the lookup (its only stage): the main program, and the small program a call launches for
# at most ``cake_jit.SMALL_MAX_POSITIONS`` looked-up rows.
RECORD = "lookup"
SMALL_RECORD = "lookup_small"
RECORDS = (RECORD, SMALL_RECORD)

# The operand dtypes.
_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_DTYPES = {"weight": _E4M3, "scale": _E8M0, "row_ids": torch.int64}

# The kernel loads 8 bytes of a row at a time: ``weight`` and ``scale`` start on an 8-byte boundary and their row
# strides are multiples of 8 elements (one byte each).
_ALIGNMENT = 8

# The number of looked-up rows is a 32-bit kernel parameter; the kernel's last step of ROWS_PER_WARP (4) rows may
# reach past the last row by up to ROWS_PER_WARP - 1 (3) before it clamps, so ``row_ids.numel()`` stays below
# 2^31 - 8, which leaves room for a step of up to 8 rows.
_MAX_POSITIONS = 2**31 - 8

# The kernel parameters of either lookup program, in launch order (the registry's argument plan without the grid).
_PARAMETERS = ("w", "s", "row_ids", "out", "P")


# ----------------------------------------------------------------------------------------------------------------
# Devices


def arch_for_capability(capability: tuple[int, int]) -> Optional[str]:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device is not an exported target); cached
    per device index."""
    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake Engram row lookup target; "
            f"the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """The SM count of one CUDA device (the launch grid's cap); cached per device index."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


# ----------------------------------------------------------------------------------------------------------------
# Records and libraries (``build.json``)


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        raise KeyError(f"{name!r} is not a record of the Cake Engram row lookup registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake Engram row lookup record {name} is not delivered for {arch}; its architectures: {found['arches']}"
        )
    return f"{found['library']}_{arch}"


def is_prebuilt(name: str, arch: str) -> bool:
    """Whether the library of record ``name`` for ``arch`` is served from ``aot/`` (a builder compiled it)."""
    return cake_build.prebuilt_path(library_name(name, arch)).is_file()


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a builder
    compiled it, else the JIT build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up builds)."""
    return cake_build.jit_specs(arch)


@functools.cache
def parameters(name: str) -> tuple[str, ...]:
    """The kernel parameter names of record ``name`` in ``run`` order (the argument plan without the grid)."""
    names: list[str] = []
    axes: list[str] = []
    for kind, key in record(name)["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        else:
            if axes:
                raise ValueError(f"record {name}: kernel parameter {key!r} follows the grid arguments")
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}")
    return tuple(names)


@functools.cache
def _entry(name: str, arch: str) -> Any:
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process), after checking that the
    registry's argument plan is the one :func:`_launch` passes."""
    names = parameters(name)
    if names != _PARAMETERS:
        raise RuntimeError(
            f"the Cake Engram row lookup program {name} takes the parameters {list(names)}; this host code passes "
            f"{list(_PARAMETERS)}"
        )
    return getattr(load(name, arch), record(name)["ffi_entry"])


# The FFI entry of each record on each device, by (record, device index): the first launch on a device fills it for
# every record (:func:`_first_launch`), and every launch takes its entry from here.
_ENTRIES: dict[tuple[str, int], Any] = {}


def _first_launch(arch: str, name: str, index: int, *args: Any) -> Any:
    """The FFI entry of record ``name`` for the first launch on device ``index``, after launching every other record
    with the same arguments ``args`` (the programs write the same bytes, so the output does not change): the entries
    of every record are looked up (:func:`_entry` loads each library once per process and architecture) and stored
    for ``index``.  With the launch of ``name`` that follows, the first call on a device launches each program once
    there, and so loads each kernel's module on the device under CUDA's lazy module loading: no later call on the
    device, a call captured in a CUDA graph included, loads a library or a module."""
    entries = {record: _entry(record, arch) for record in RECORDS}
    for record, entry in entries.items():
        if record != name:
            entry(*args)
    for record, entry in entries.items():
        _ENTRIES[record, index] = entry
    return entries[name]


def _record_for(positions: int) -> str:
    """The record a lookup of ``positions`` rows launches: :data:`SMALL_RECORD` for at most
    ``cake_jit.SMALL_MAX_POSITIONS`` rows, :data:`RECORD` above (read at each call)."""
    return SMALL_RECORD if positions <= cake_jit.SMALL_MAX_POSITIONS else RECORD


def _launch(
    arch: str,
    name: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    positions: int,
    grid_x: int,
    index: int,
) -> None:
    """Launch the program of record ``name`` on ``grid_x`` CTAs: ``out`` (BF16 ``[*row_ids.shape, 256]``, contiguous)
    from the rows ``row_ids`` (int64, contiguous, ``positions`` elements) of ``weight`` and ``scale``, all on device
    ``index``.  The one place the programs' argument list is spelled: the table operands go in as byte views of the
    same strides (the binding reads the row strides from them), then the ids, the output, the row count and the grid.
    tvm-ffi runs the call on the current torch stream of the device of its first CUDA tensor argument (every operand
    is on that device), and the binding makes that device current for the launch.  The first launch on a device
    launches the other program first, with the same arguments (:func:`_first_launch`).

    A one-row table goes in with its compact row strides (256 and 8) instead.  Its row stride changes no address: the
    kernel multiplies it by the row id, the only valid id is 0, and any other id traps before a row is read.  The
    stride of a size-1 dimension carries no layout information, so a DLPack exporter may report another value for it,
    and the binding requires row strides that are multiples of 8.  The compact strides guard against an exporter that
    keeps a compact view's strides and rewrites other size-1 strides; an exporter that writes 1 for every size-1
    stride would fail the check either way.  With torch 2.14.1 and apache-tvm-ffi 0.1.16 the binding accepts both a
    one-row view's own stride (264, say) and the compact strides."""
    w = weight.view(torch.uint8)
    s = scale.view(torch.uint8)
    if w.shape[0] == 1:
        w = w.as_strided((1, ROW_DIM), (ROW_DIM, 1))
        s = s.as_strided((1, GROUPS), (GROUPS, 1))
    try:
        entry = _ENTRIES[name, index]
    except KeyError:
        entry = _first_launch(arch, name, index, w, s, row_ids, out, positions, grid_x, 1, 1)
    entry(w, s, row_ids, out, positions, grid_x, 1, 1)


# ----------------------------------------------------------------------------------------------------------------
# Checks (dtypes, shapes, strides, alignment and devices only: nothing reads tensor contents)


def _check(weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor) -> int:
    """The number of looked-up rows ``row_ids.numel()`` of a valid call; ``ValueError`` otherwise.  The device check
    comes before anything that touches CUDA.  :func:`_accepted` repeats these checks for the hot path: a change here
    needs the same change there."""
    operands = (("weight", weight), ("scale", scale), ("row_ids", row_ids))
    for name, t in operands:
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    for name, t in operands:
        if t.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {t.device}")
    for name, t in operands[1:]:
        if t.device != weight.device:
            raise ValueError(f"{name} must be on weight's device {weight.device}, got {t.device}")
    for name, t in operands:
        if t.dtype != _DTYPES[name]:
            raise ValueError(f"{name} must be {_DTYPES[name]}, got {t.dtype}")
    for name, t, width in (("weight", weight, ROW_DIM), ("scale", scale, GROUPS)):
        if t.dim() != 2 or t.shape[1] != width:
            raise ValueError(f"{name} must be [R, {width}], got shape {tuple(t.shape)}")
    if scale.shape[0] != weight.shape[0]:
        raise ValueError(f"scale must have weight's R = {weight.shape[0]} rows, got {scale.shape[0]}")
    for name, t in (("weight", weight), ("scale", scale)):
        if t.stride(1) != 1:
            raise ValueError(f"{name} must have a unit column stride, got strides {t.stride()}")
        if t.stride(0) % _ALIGNMENT:
            raise ValueError(
                f"{name}'s row stride must be a multiple of {_ALIGNMENT} elements, got strides {t.stride()}"
            )
        if t.data_ptr() % _ALIGNMENT:
            raise ValueError(f"{name} must start on an {_ALIGNMENT}-byte boundary, got address {t.data_ptr():#x}")
    if not row_ids.is_contiguous():
        raise ValueError(
            f"row_ids must be contiguous, got strides {row_ids.stride()} for shape {tuple(row_ids.shape)}"
        )
    positions = row_ids.numel()
    if positions >= _MAX_POSITIONS:
        raise ValueError(f"row_ids must hold fewer than {_MAX_POSITIONS} elements, got {positions}")
    if positions and weight.shape[0] == 0:
        raise ValueError(f"the table has no rows (R = 0) to look {positions} row ids up in")
    return positions


def _accepted(weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor) -> Optional[int]:
    """``row_ids.numel()`` of a call that passes every check of :func:`_check`, found with the cheapest tests in one
    pass; ``None`` when any of them fails (or an operand is a ``torch.Tensor`` subclass), and then :func:`_check` runs
    and raises its error.  The entry also runs :func:`_check` when this function raises, so an invalid call gets the
    error of :func:`_check` in every case."""
    tensor = torch.Tensor
    if not (type(weight) is tensor and type(scale) is tensor and type(row_ids) is tensor):
        return None
    if not (weight.dtype is _E4M3 and scale.dtype is _E8M0 and row_ids.dtype is torch.int64):
        return None
    if not (weight.is_cuda and scale.is_cuda and row_ids.is_cuda):
        return None
    index = weight.get_device()
    if scale.get_device() != index or row_ids.get_device() != index:
        return None
    w_shape = weight.shape
    s_shape = scale.shape
    if len(w_shape) != 2 or len(s_shape) != 2 or w_shape[1] != ROW_DIM or s_shape[1] != GROUPS:
        return None
    if s_shape[0] != w_shape[0]:
        return None
    w_stride = weight.stride()
    s_stride = scale.stride()
    if w_stride[1] != 1 or s_stride[1] != 1 or w_stride[0] % _ALIGNMENT or s_stride[0] % _ALIGNMENT:
        return None
    if (weight.data_ptr() | scale.data_ptr()) % _ALIGNMENT:
        return None
    if not row_ids.is_contiguous():
        return None
    positions = row_ids.numel()
    if positions >= _MAX_POSITIONS or (positions and not w_shape[0]):
        return None
    return positions


# ----------------------------------------------------------------------------------------------------------------
# Entry points


def cake_engram_lookup_det_train_forward(
    weight: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    """The Engram row lookup: ``out[p, 32*g + j] = bf16(float(weight[r, 32*g + j]) * float(scale[r, g]))`` for
    ``r = row_ids[p]``, the exact product rounded once to BF16.

    ``weight``: ``float8_e4m3fn [R, 256]``; ``scale``: ``float8_e8m0fnu [R, 8]``; each with a unit column stride, a row
    stride that is a multiple of 8 elements and an 8-byte aligned start (two contiguous tensors, or the two views of a
    packed ``uint8 [R, 264]`` table from :func:`cake_engram_lookup_det_train_split_packed`).  ``row_ids``: contiguous
    int64 of any shape (``[T, 24]`` or ``[M]``), values in ``[0, R)`` (a caller error otherwise; see the module
    docstring).  All on one CUDA device.  Returns a
    new contiguous BF16 ``[*row_ids.shape, 256]`` on that device; no other allocation.  An empty ``row_ids`` returns an
    empty output and launches nothing, before the device's architecture is looked up.

    Every output element is one product rounded once, so the output repeats bit for bit.  The kernel runs on the
    current stream of the operands' device; nothing synchronizes.  At most ``cake_jit.SMALL_MAX_POSITIONS`` looked-up
    rows launch the small program, more the main one: both write the same bytes.  The first call on a device launches
    both programs (the module docstring), so that a CUDA graph capture after it loads nothing: make it outside a
    capture.
    """
    try:
        accepted = _accepted(weight, scale, row_ids)
    except Exception:  # an operand the one-pass check cannot read (a nested tensor's shape): _check gives the error
        accepted = None
    positions = _check(weight, scale, row_ids) if accepted is None else accepted
    shape = (*row_ids.shape, ROW_DIM)
    if positions == 0:
        # No row to look up: nothing to launch.
        return torch.empty(shape, dtype=torch.bfloat16, device=row_ids.device)
    index = row_ids.get_device()
    arch = device_arch(index)  # RuntimeError on a device outside ARCHES, before the allocation
    out = torch.empty(shape, dtype=torch.bfloat16, device=row_ids.device)
    grid_x = cake_jit.grid(positions, sm_count(index), arch)[0]
    _launch(arch, _record_for(positions), weight, scale, row_ids, out, positions, grid_x, index)
    return out


def cake_engram_lookup_det_train_split_packed(buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight, scale)``: the ``float8_e4m3fn [R, 256]`` and ``float8_e8m0fnu [R, 8]`` views of a packed table
    ``buf``, ``uint8 [R, 264]`` with a unit column stride (each row: its 256 value bytes, then its 8 scale bytes).  The
    views share ``buf``'s storage and keep its row stride (264 for a contiguous ``buf``); nothing is copied, and any
    device works.  ``ValueError`` for another dtype, rank, width or column stride."""
    if not isinstance(buf, torch.Tensor):
        raise ValueError(f"buf must be a torch.Tensor, got {type(buf).__name__}")
    if buf.dtype != torch.uint8:
        raise ValueError(f"buf must be torch.uint8, got {buf.dtype}")
    if buf.dim() != 2 or buf.shape[1] != PACKED_ROW_BYTES:
        raise ValueError(f"buf must be [R, {PACKED_ROW_BYTES}], got shape {tuple(buf.shape)}")
    if buf.stride(1) != 1:
        raise ValueError(f"buf must have a unit column stride, got strides {buf.stride()}")
    return buf[:, :ROW_DIM].view(_E4M3), buf[:, ROW_DIM:].view(_E8M0)
