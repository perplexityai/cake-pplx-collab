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

Loader, checks, launches and autograd of the Cake attention residual (AttnRes): a softmax-weighted sum of ``n`` BF16
values of width 7168, ``1 <= n <= 9`` (SM100 / SM103 / SM107), deterministic package.

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds one record per program (its ``library``
stem; the ``build.json`` entries are ``<library>_<arch>``), :func:`.cake_jit.record` names the record a stage launches
for ``n`` values and :func:`.cake_jit.grid` gives its grid.  :mod:`.cake_build` serves a library from a prebuilt
``aot/`` file when a builder compiled it and otherwise through the package's JIT build shim (``_cake_pplx_jit.py``)
from the sources and flags of ``build.json``.

Per token ``t`` (``H = 7168``), with every quantity in FP32 and each BF16 output rounded once, at its store::

    ss_j = sum_c v_j[c]^2   d_j = sum_c v_j[c] w[c]   r_j = 1 / sqrt(ss_j / H + eps)
    s_j = r_j d_j   p = softmax_j(s)   (the largest s_j subtracted first)
    out = bf16(sum_j p_j v_j)

and, given ``dout``::

    dp_j = sum_c dout[c] v_j[c]   ds_j = p_j (dp_j - sum_i p_i dp_i)
    dv_j = bf16(p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / H) v_j)
    dw   = sum over t and j of ds_j r_j v_j                       (FP32)

The kernels evaluate ``ds_j`` as ``p_j ((dp_j - dp_c) - sum_i p_i (dp_i - dp_c))``, ``c`` the value of the largest
``p_j`` (the first on ties): the same value, without the cancellation of ``dp_c - sum_i p_i dp_i`` when ``p_c`` rounds
to 1 in FP32 (a peaked softmax), which would drop the dominant value's term of ``dw``.

One forward call launches ``fwd`` alone: it writes ``out`` and the state the backward reads, ``p``, ``r`` and ``d``
(FP32 ``[T, n]`` each, row-major).  One backward call launches ``bwd`` and ``dw_reduce``, back to back: ``bwd`` writes
every ``dv_j`` from ``dout``, the values, ``w`` and that state (no forward sum is recomputed) and one FP32 row of
``dw`` partial sums per launch group of its grid into a workspace ``dw_rows [G, 7168]`` (every row written in full);
``dw_reduce`` adds the ``G`` rows in one fixed order into ``dw``.  The kernels take the values as nine pointers (this
module passes the first value in the slots past ``n``, which the kernels never read) and ``n``, the token count and
``eps`` as run-time scalars.  ``fwd`` and ``bwd`` ship one program per value bound ``nmax`` from 2 to 9 (``nmax`` values
held in registers); a call of ``n`` values runs the program of the smallest bound at least ``n``
(:func:`.cake_jit.nmax_for`).  Every program serves every token count, so the libraries are a fixed set: no shape
compiles a new one.

``fwd`` and ``bwd`` run a persistent grid: as many launch groups as the token count and as the device holds at once
(:func:`occupancy_groups`: ``cudaOccupancyMaxActiveClusters`` of the loaded program, asked once per program and device;
:func:`.cake_jit.resident_groups`, a table of measured counts and a fallback under them, when the query is not
reachable); any grid is correct.  Every launch is a plain stream-ordered launch on the current torch stream of the
operands' device: nothing cooperative.  Nothing here reads an environment variable, reads device memory or
synchronizes; grids and buffer sizes come from tensor shapes and the device's occupancy.

This is the deterministic package of its family: every sum has one fixed order, so every output repeats bit for bit
for identical inputs on one device (one grid per token count and device).  Nothing reads torch's deterministic mode:
there is nothing to alert about.
"""

from __future__ import annotations

import ctypes
import functools
import math
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HIDDEN",
    "MAX_VALUES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeAttnResDetFunction",
    "arch_for_capability",
    "cake_attn_res_det",
    "cake_attn_res_det_train_backward",
    "cake_attn_res_det_train_forward",
    "cake_attn_res_det_train_retained_bytes",
    "cake_attn_res_det_train_workspace_bytes",
    "device_arch",
    "is_cake_attn_res_det_train_supported",
    "is_prebuilt",
    "launch_bwd",
    "launch_dw_reduce",
    "launch_fwd",
    "library_name",
    "load",
    "occupancy_groups",
    "parameters",
    "record",
    "sm_count",
    "specs_for_arch",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 / GB300 (``sm_103a``) and R200 (``sm_107a``),
# each compiled with its exact ``-gencode`` flag set.  No other architecture has a build.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))
# The fixed operand geometry: values of width 7168, at most nine of them.
HIDDEN: int = cake_jit.HIDDEN
MAX_VALUES: int = cake_jit.MAX_VALUES

# Every BF16 operand and ``w`` start on a 16-byte boundary: the kernels move 16 bytes per access.
_ALIGNMENT = 16
# The kernels take the token count as a 32-bit integer (token offsets are 64-bit).
_MAX_TOKENS = 2**31 - 1
_BF16 = torch.bfloat16
_FP32 = torch.float32
# The record the weight-gradient sum launches, looked up once.
_DW_REDUCE_RECORD: str = cake_jit.record("dw_reduce")
_DW_REDUCE_GRID: tuple[int, int, int] = cake_jit.dw_reduce_grid()
# Bytes of one FP32 [7168] row (a partial row of the backward's weight gradient).
_ROW_BYTES = HIDDEN * 4

# The public module the entry points are documented under.
_MODULE = "cake_pplx.cake_attn_res_det_train"


# ----------------------------------------------------------------------------------------------------------------
# Devices


def arch_for_capability(capability: tuple[int, int]) -> Optional[str]:
    """The architecture of ``ARCHES`` built for one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """The architecture of one CUDA device (``RuntimeError`` when it is not one of ``ARCHES``); cached per device
    index."""
    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not a Cake AttnRes target; the built architectures "
            f"are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (fixed for the process)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


class _LaunchConfig(ctypes.Structure):
    """``cudaLaunchConfig_t``: grid, block, dynamic shared memory, stream, launch attributes."""

    _fields_ = [
        ("grid", ctypes.c_uint * 3),
        ("block", ctypes.c_uint * 3),
        ("dynamic_smem_bytes", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p),
        ("num_attrs", ctypes.c_uint),
    ]


@functools.cache
def occupancy_groups(name: str, device_index: int) -> Optional[int]:
    """The launch groups (clusters) of the ``fwd`` or ``bwd`` record ``name`` one CUDA device holds at once, as
    ``cudaOccupancyMaxActiveClusters`` reports them for the loaded program (its block, dynamic shared memory and
    compile-time cluster), looked up once per record and device; None when the query is not reachable (the library
    does not export the kernel symbol or the CUDA runtime query, or the query fails).  A host-only runtime call: it
    reads no device memory and does not synchronize."""
    found = record(name)
    try:
        arch = device_arch(device_index)
        load(name, arch)
        library = ctypes.CDLL(str(cake_build.library_path(library_name(name, arch))))
        kernel = ctypes.cast(getattr(library, found["kernel_symbol"]), ctypes.c_void_p)
        query = library.cudaOccupancyMaxActiveClusters
    except (AttributeError, OSError, RuntimeError):
        return None
    query.restype = ctypes.c_int
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.POINTER(_LaunchConfig)]
    launch = found["launch"]
    config = _LaunchConfig()
    config.grid[:] = (int(launch["cluster"][0]) * 1024, 1, 1)
    config.block[:] = tuple(int(x) for x in launch["block"])
    config.dynamic_smem_bytes = int(launch["dynamic_smem_bytes"])
    count = ctypes.c_int(0)
    with torch.cuda.device(device_index):
        status = query(ctypes.byref(count), kernel, ctypes.byref(config))
    return count.value if status == 0 and count.value >= 1 else None


@functools.cache
def _resident(name: str, device_index: int) -> int:
    """The resident launch groups of record ``name`` on one CUDA device, looked up once per record and device: the
    runtime count (:func:`occupancy_groups`) when the query is reachable, else :func:`.cake_jit.resident_groups` (a
    table of measured counts and a fallback under every measured count)."""
    found = occupancy_groups(name, device_index)
    if found is not None:
        return found
    return cake_jit.resident_groups(name, device_arch(device_index), sm_count(device_index))


def _groups(name: str, T: int, device_index: int) -> int:
    """:func:`.cake_jit.groups` of record ``name`` for ``T`` tokens on one CUDA device: the resident groups, never
    more than ``T``, at least one."""
    return max(1, min(T, _resident(name, device_index)))


def _device_index(device: Union[torch.device, int, str, None]) -> Optional[int]:
    """The CUDA device index of ``device`` (None: the current device), or None for a non-CUDA device."""
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return torch.cuda.current_device() if device.index is None else device.index


def is_cake_attn_res_det_train_supported(device: Union[torch.device, int, str, None] = None) -> bool:
    """Whether ``device`` (a ``torch.device``, a device string or index; None: the current CUDA device) is a CUDA
    device of one of ``ARCHES``.  False without CUDA."""
    if not torch.cuda.is_available():
        return False
    index = _device_index(device)
    if index is None or not 0 <= index < torch.cuda.device_count():
        return False
    return arch_for_capability(torch.cuda.get_device_capability(index)) is not None


# ----------------------------------------------------------------------------------------------------------------
# Records and libraries (``build.json``)


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        raise KeyError(f"{name!r} is not a record of the Cake AttnRes registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake AttnRes record {name} is not delivered for {arch}; its architectures: {found['arches']}"
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
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process)."""
    return getattr(load(name, arch), record(name)["ffi_entry"])


_VALUE_SLOTS = tuple(f"v{j}" for j in range(MAX_VALUES))
_GRAD_SLOTS = tuple(f"dv{j}" for j in range(MAX_VALUES))
# The kernel parameters each stage's launcher passes, in order (then the grid): every record of a stage takes these.
_STAGE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "fwd": (*_VALUE_SLOTS, "w", "out", "p", "r", "d", "T", "n", "eps"),
    "bwd": ("dout", *_VALUE_SLOTS, "w", "p", "r", "d", *_GRAD_SLOTS, "dw_rows", "T", "n"),
    "dw_reduce": ("dw_rows", "dw", "G"),
}


@functools.cache
def _plan(name: str, stage: str) -> None:
    """Check, once per record, that record ``name`` serves ``stage`` and that its argument plan takes the parameters
    :data:`_STAGE_PARAMETERS` lists for the stage, in that order: the launchers pass their arguments by position."""
    found = record(name)
    if found["stage"] != stage:
        raise ValueError(f"record {name} serves stage {found['stage']}, not {stage}")
    if parameters(name) != _STAGE_PARAMETERS[stage]:
        passed = list(_STAGE_PARAMETERS[stage])
        raise ValueError(f"record {name} takes {list(parameters(name))}; the {stage} launcher passes {passed}")


def _run(name: str, stage: str, device_index: int, args: tuple[Any, ...]) -> None:
    """Launch record ``name`` of ``stage`` with its kernel parameters ``args`` in plan order, then the grid.  tvm-ffi
    runs the call on the current torch stream of the device of its first CUDA tensor argument (every buffer is on
    CUDA device ``device_index``), and the binding makes that device current for the launch."""
    _plan(name, stage)
    _entry(name, device_arch(device_index))(*args)


def _slots(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """The nine pointer slots of the kernels: ``tensors`` then its first tensor repeated (never read past ``n``)."""
    return (*tensors, *(tensors[0],) * (MAX_VALUES - len(tensors)))


# ----------------------------------------------------------------------------------------------------------------
# Stage launchers: caller-owned buffers, the current stream, no allocation, no check of the operands


def launch_fwd(
    name: str,
    *,
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    out: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    eps: float,
    groups: int,
) -> None:
    """Launch the forward record ``name`` over ``T = out.shape[0] >= 1`` tokens on ``groups`` launch groups:
    ``out`` (BF16 ``[T, 7168]``) and ``p``, ``r``, ``d`` (FP32 ``[T, n]``) from the ``n = len(values)`` values (BF16
    ``[T, 7168]``, contiguous) and ``w`` (FP32 ``[7168]``)."""
    grid = cake_jit.cluster_ctas(name) * int(groups)
    args = (*_slots(values), w, out, p, r, d, int(out.shape[0]), len(values), float(eps))
    _run(name, "fwd", out.get_device(), (*args, grid, 1, 1))


def launch_bwd(
    name: str,
    *,
    dout: torch.Tensor,
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    dv: Sequence[torch.Tensor],
    dw_rows: torch.Tensor,
    groups: int,
) -> None:
    """Launch the backward record ``name`` over ``T = dout.shape[0] >= 1`` tokens on ``groups`` launch groups: every
    ``dv_j`` (BF16 ``[T, 7168]``, contiguous) written in full, and ``dw_rows`` (FP32 ``[groups, 7168]``): row ``g`` is
    group ``g``'s sum of the ``dw`` terms of its tokens, written in full (zero for a group without a token)."""
    grid = cake_jit.cluster_ctas(name) * int(groups)
    args = (dout, *_slots(values), w, p, r, d, *_slots(dv), dw_rows, int(dout.shape[0]), len(values))
    _run(name, "bwd", dout.get_device(), (*args, grid, 1, 1))


def launch_dw_reduce(name: str, *, dw_rows: torch.Tensor, dw: torch.Tensor) -> None:
    """Launch the finish record ``name``: ``dw`` (FP32 ``[7168]``) is the sum of the ``G`` rows of ``dw_rows`` (FP32
    ``[G, 7168]``) in one fixed order."""
    args = (dw_rows, dw, int(dw_rows.shape[0]), *cake_jit.dw_reduce_grid())
    _run(name, "dw_reduce", dw_rows.get_device(), args)


# ----------------------------------------------------------------------------------------------------------------
# Checks (types, devices, dtypes, shapes, strides and alignment only: nothing reads tensor contents)


def _tensor(name: str, t: Any) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    return t


def _on_device(name: str, t: torch.Tensor, device: torch.device) -> None:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be on a CUDA device, got {t.device}")
    if t.device != device:
        raise ValueError(f"{name} must be on the first value's device {device}, got {t.device}")


def _aligned(name: str, t: torch.Tensor) -> None:
    if t.numel() and t.data_ptr() % _ALIGNMENT:
        raise ValueError(f"{name} must start on a {_ALIGNMENT}-byte boundary, got address {t.data_ptr():#x}")


def _row_layout(name: str, t: torch.Tensor, T: int) -> None:
    """A BF16 ``[T, 7168]`` operand: contiguous, 16-byte aligned."""
    if t.dtype != _BF16:
        raise ValueError(f"{name} must be torch.bfloat16, got {t.dtype}")
    if t.shape != (T, HIDDEN):
        raise ValueError(f"{name} must be [T, {HIDDEN}] with T = {T}, got {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {t.stride()}")
    _aligned(name, t)


def _state_layout(name: str, t: Any, T: int, n: int, device: torch.device) -> None:
    """An FP32 ``[T, n]`` forward state (``p``, ``r`` or ``d``): contiguous, on ``device``."""
    t = _tensor(name, t)
    _on_device(name, t, device)
    if t.dtype != _FP32:
        raise ValueError(f"{name} must be torch.float32, got {t.dtype}")
    if t.shape != (T, n):
        raise ValueError(f"{name} must be [T, n] = [{T}, {n}], got {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {t.stride()}")


def _check_eps(eps: Any) -> float:
    if type(eps) is float and 0.0 <= eps < math.inf:
        return eps
    try:
        eps_f = float(eps)
    except (TypeError, ValueError):
        raise ValueError(f"eps must be a real number, got {eps!r}") from None
    if not (math.isfinite(eps_f) and eps_f >= 0.0):
        raise ValueError(f"eps must be finite and non-negative, got {eps}")
    return eps_f


def _valid_tokens(values: Any, w: Any) -> int:
    """The token count ``T`` of forward operands that pass every tensor check of :func:`_check_operands`, read in one
    pass; -1 for anything else (the caller then runs those checks, which name the first one that fails)."""
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_VALUES:
        return -1
    first = values[0]
    if not (isinstance(first, torch.Tensor) and first.is_cuda and first.dim() == 2 and isinstance(w, torch.Tensor)):
        return -1
    T, index = first.shape[0], first.get_device()
    if T > _MAX_TOKENS:
        return -1
    for v in values:
        if not (
            isinstance(v, torch.Tensor)
            and v.is_cuda
            and v.get_device() == index
            and v.dtype == _BF16
            and v.shape == (T, HIDDEN)
            and v.is_contiguous()
            and not v.data_ptr() % _ALIGNMENT
        ):
            return -1
    if not (w.is_cuda and w.get_device() == index and w.dtype == _FP32 and w.shape == (HIDDEN,)):
        return -1
    return T


def _check_operands(values: Any, w: Any, eps: Any) -> tuple[int, float]:
    """``(T, eps)`` of a valid set of forward operands; ``ValueError`` otherwise.  The device checks come before
    anything that touches CUDA."""
    T = _valid_tokens(values, w)
    if T >= 0:
        return T, _check_eps(eps)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"values must be a list or tuple of 1 to {MAX_VALUES} tensors, got {type(values).__name__}")
    if not 1 <= len(values) <= MAX_VALUES:
        raise ValueError(f"values must hold 1 to {MAX_VALUES} tensors, got {len(values)}")
    operands = [(f"values[{j}]", _tensor(f"values[{j}]", v)) for j, v in enumerate(values)]
    operands.append(("w", _tensor("w", w)))
    first = operands[0][1]
    index = first.get_device() if first.is_cuda else -1
    for name, t in operands:
        if index < 0 or not t.is_cuda or t.get_device() != index:
            _on_device(name, t, first.device)
    if first.dim() != 2:
        raise ValueError(f"values[0] must be [T, {HIDDEN}], got {tuple(first.shape)}")
    T = int(first.shape[0])
    if T > _MAX_TOKENS:
        raise ValueError(f"the token count must be at most {_MAX_TOKENS}, got {T}")
    for name, t in operands[:-1]:
        _row_layout(name, t, T)
    if w.dtype != _FP32:
        raise ValueError(f"w must be torch.float32, got {w.dtype}")
    if w.shape != (HIDDEN,):
        raise ValueError(f"w must be [{HIDDEN}], got {tuple(w.shape)}")
    return T, _check_eps(eps)


def _weight(w: torch.Tensor) -> torch.Tensor:
    """``w`` as the kernels read it: FP32 ``[7168]``, contiguous and 16-byte aligned (a copy otherwise)."""
    if w.requires_grad and torch.is_grad_enabled():
        w = w.detach()
    w = w.contiguous()
    if w.data_ptr() % _ALIGNMENT:
        w = w.clone()
    return w


def _check_outputs(name: str, ts: Any, T: int, n: int, device: torch.device) -> list[torch.Tensor]:
    """Caller-owned per-value BF16 ``[T, 7168]`` outputs: a list or tuple of ``n`` contiguous, aligned tensors."""
    if not isinstance(ts, (list, tuple)) or len(ts) != n:
        raise ValueError(f"{name} must be a list or tuple of n = {n} tensors, got {ts!r:.80}")
    found = []
    for j, t in enumerate(ts):
        label = f"{name}[{j}]"
        t = _tensor(label, t)
        _on_device(label, t, device)
        _row_layout(label, t, T)
        found.append(t)
    return found


def cake_attn_res_det_train_workspace_bytes(T: int, n: int, device: Union[torch.device, int, str, None] = None) -> int:
    """Bytes of device memory one backward call over ``T`` tokens and ``n`` values on ``device`` (None: the current
    CUDA device) allocates beside its operands and outputs: the ``[G, 7168]`` FP32 rows of ``dw`` partial sums, one per
    launch group of the backward grid, 28 KiB each (0 for ``T == 0``); ``n`` picks the backward program and so its
    resident groups.  A forward call allocates its outputs alone: ``out`` and the retained state
    (:func:`cake_attn_res_det_train_retained_bytes`)."""
    T, n = int(T), int(n)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if not 1 <= n <= MAX_VALUES:
        raise ValueError(f"n must be 1 to {MAX_VALUES}, got {n}")
    if T == 0:
        return 0
    index = _device_index(device)
    if index is None:
        raise ValueError(f"device must be a CUDA device, got {device}")
    device_arch(index)  # RuntimeError on a device outside ARCHES
    rows = _groups(cake_jit.record("bwd", n), T, index)
    return _ROW_BYTES * rows


def cake_attn_res_det_train_retained_bytes(T: int, n: int) -> int:
    """Bytes of the state one forward call keeps for the backward beside references to its inputs: ``p``, ``r`` and
    ``d``, FP32 ``[T, n]`` each, ``12 * T * n``."""
    T, n = int(T), int(n)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if not 1 <= n <= MAX_VALUES:
        raise ValueError(f"n must be 1 to {MAX_VALUES}, got {n}")
    return 3 * 4 * T * n


# ----------------------------------------------------------------------------------------------------------------
# Forward and backward (explicit, without autograd)


def _forward(
    values: Sequence[torch.Tensor], w: torch.Tensor, eps: Any, out: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """The checked forward: ``(out, p, r, d, eps)`` with ``eps`` as checked (a float)."""
    T, eps_f = _check_operands(values, w, eps)
    n = len(values)
    device = values[0].device
    if out is None:
        out = torch.empty(T, HIDDEN, dtype=_BF16, device=device)
    else:
        out = _tensor("out", out)
        _on_device("out", out, device)
        _row_layout("out", out, T)
    # One allocation for the three [T, n] states: p, r and d are its contiguous slices.
    state = torch.empty(3, T, n, dtype=_FP32, device=device)
    p, r, d = state.unbind(0)
    if T == 0:
        return out, p, r, d, eps_f
    index = device.index
    device_arch(index)  # RuntimeError on a device outside ARCHES
    name = cake_jit.record("fwd", n)
    groups = _groups(name, T, index)
    args = (*_slots(values), _weight(w), out, p, r, d, T, n, eps_f)
    _run(name, "fwd", index, (*args, cake_jit.cluster_ctas(name) * groups, 1, 1))
    return out, p, r, d, eps_f


def _launch_backward(
    dout: torch.Tensor,
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    T: int,
    dv: Optional[list[torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The backward of checked operands over ``T`` tokens: ``(dv, dw)``.  Every buffer is allocated and both argument
    tuples are built before the first launch, so the launches go out back to back."""
    device = dout.device
    n = len(values)
    if dv is None:
        dv = [torch.empty(T, HIDDEN, dtype=_BF16, device=device) for _ in range(n)]
    if T == 0:
        # No token: nothing to launch; a zero weight gradient.
        return dv, torch.zeros(HIDDEN, dtype=_FP32, device=device)
    index = device.index
    device_arch(index)  # RuntimeError on a device outside ARCHES
    name = cake_jit.record("bwd", n)
    groups = _groups(name, T, index)
    # One row per launch group, each written in full by its group.
    dw_rows = torch.empty(groups, HIDDEN, dtype=_FP32, device=device)
    dw = torch.empty(HIDDEN, dtype=_FP32, device=device)
    bwd = (dout, *_slots(values), _weight(w), p, r, d, *_slots(dv), dw_rows, T, n)
    _run(name, "bwd", index, (*bwd, cake_jit.cluster_ctas(name) * groups, 1, 1))
    _run(_DW_REDUCE_RECORD, "dw_reduce", index, (dw_rows, dw, groups, *_DW_REDUCE_GRID))
    return dv, dw


def _backward(
    dout: Any,
    values: Any,
    w: Any,
    p: Any,
    r: Any,
    d: Any,
    dv: Optional[Sequence[torch.Tensor]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The explicit backward: every operand checked."""
    T, _ = _check_operands(values, w, 1.0)
    n = len(values)
    device = values[0].device
    dout = _tensor("dout", dout)
    _on_device("dout", dout, device)
    _row_layout("dout", dout, T)
    for name, t in (("p", p), ("r", r), ("d", d)):
        _state_layout(name, t, T, n, device)
    if dv is not None:
        dv = _check_outputs("dv", dv, T, n, device)
    return _launch_backward(dout, values, w, p, r, d, T, dv)


def cake_attn_res_det_train_forward(
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    *,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The forward without autograd: ``(out, p, r, d)``, ``out`` BF16 ``[T, 7168]`` (``out`` given: written in place
    and returned) and the state the backward reads, FP32 ``[T, n]`` each.

    ``values`` is a list or tuple of ``n`` (1 to 9) BF16 ``[T, 7168]`` tensors, each contiguous and 16-byte aligned
    (separate tensors: nothing is stacked); ``w`` FP32 ``[7168]``; every tensor on one CUDA device.  ``eps`` is a
    finite, non-negative run-time float.  ``T == 0`` launches nothing.  The forward never consults
    ``torch.use_deterministic_algorithms``."""
    out, p, r, d, _ = _forward(values, w, eps, out)
    return out, p, r, d


def cake_attn_res_det_train_backward(
    dout: torch.Tensor,
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    *,
    dv: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The backward without autograd: ``(dv, dw)`` from ``dout`` (BF16 ``[T, 7168]``, contiguous), the forward's
    values and ``w`` and its state ``p``, ``r``, ``d`` (FP32 ``[T, n]``, contiguous); the forward's output is not
    needed.  ``dv`` is a list of ``n`` BF16 ``[T, 7168]`` gradients (``dv`` given: a list or tuple of ``n`` contiguous
    tensors written in place); ``dw`` is FP32 ``[7168]``.  ``T == 0`` returns a zero ``dw`` and launches nothing.
    Every output repeats bit for bit for identical inputs on one device; nothing consults
    ``torch.use_deterministic_algorithms``."""
    return _backward(dout, values, w, p, r, d, dv)


# ----------------------------------------------------------------------------------------------------------------
# Autograd


def _grad_out(dout: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """The gradient of ``out`` as the backward reads it: BF16, contiguous, 16-byte aligned, on ``like``'s device and
    of its shape."""
    if dout is None:
        return torch.zeros_like(like)
    if (
        dout.dtype == _BF16
        and dout.is_cuda
        and dout.get_device() == like.get_device()
        and dout.shape == like.shape
        and dout.is_contiguous()
        and not dout.data_ptr() % _ALIGNMENT
    ):
        return dout
    if dout.dtype != _BF16:
        dout = dout.to(_BF16)
    if not dout.is_contiguous():
        dout = dout.contiguous()
    elif dout.data_ptr() % _ALIGNMENT:
        dout = dout.clone()
    _on_device("dout", dout, like.device)
    _row_layout("dout", dout, int(like.shape[0]))
    return dout


class CakeAttnResDetFunction(torch.autograd.Function):
    """``out = attn_res(values, w)`` with the values as separate arguments: ``apply(w, eps, v_0, ..., v_{n-1})``.  The
    saved set is the inputs (``values`` and ``w``, references) and the forward's FP32 state ``p``, ``r``, ``d``
    (``[T, n]`` each); ``eps`` lives on ``ctx``.  The backward reads that state (no forward sum is recomputed); under
    ``torch.utils.checkpoint`` the state is recomputed with the forward.  The backward returns ``dw`` (FP32) and every
    ``dv_j`` (BF16)."""

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor, eps: float, *values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out, p, r, d, ctx.eps = _forward(values, w, eps)
        ctx.save_for_backward(w, p, r, d, *values)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
        w, p, r, d, *values = ctx.saved_tensors
        T = int(values[0].shape[0])
        dv, dw = _launch_backward(_grad_out(dout, values[0]), values, w, p, r, d, T, None)
        need = ctx.needs_input_grad
        return (dw if need[0] else None, None, *(g if need[2 + j] else None for j, g in enumerate(dv)))


def cake_attn_res_det(values: Sequence[torch.Tensor], w: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    """The attention residual with autograd (:class:`CakeAttnResDetFunction`): ``out`` BF16 ``[T, 7168]``;
    operands as in :func:`cake_attn_res_det_train_forward`.  Gradients go to every value (BF16) and to ``w``
    (FP32)."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"values must be a list or tuple of 1 to {MAX_VALUES} tensors, got {type(values).__name__}")
    return CakeAttnResDetFunction.apply(w, eps, *values)
