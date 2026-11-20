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

Loader, checks and launches of the Cake dense multi-head attention backward
(head dimension 64 or 128, non-causal or causal; SM100 / SM103 / SM107).

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds
one record per program (its ``library`` stem; the ``build.json`` entries are
``<library>_<arch>``), ``RECORD_OF`` the three records of each variant,
``SEQUENCES`` the host sequence of each variant (its ``library`` stem and
records) and :func:`.cake_jit.grid` the launch grids.  :mod:`.cake_build`
serves a library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through the package's JIT build shim (``_cake_pplx_jit.py``) from the
sources and flags of ``build.json``.

One backward call runs three kernels on the current torch stream of the
operands' device, in this order:

1. ``bwd_pre``: ``dPsum = rowsum(out * dout)`` and ``LSE_log2 = lse * log2(e)``
   (FP32 ``[B*H, Sq]``), and the FP32 dQ accumulator cleared.
2. ``bwd_main``: one CTA per 128 keys of one (head, batch): S, P, dP and dS
   against every 128-row query block, dK and dV summed in FP32 and stored once,
   the dQ tiles added into the FP32 accumulator by bulk reductions.
3. ``bwd_post``: ``dq = bf16(acc * softmax_scale)`` in the layout of ``q``.

A head-dimension 128 call whose key length is a multiple of 256 takes the pair
route on the architectures of ``cake_jit.PAIR_ROUTE`` (:func:`.cake_jit.variant`
picks the ``RECORD_OF`` variant): its main stage is the 2-CTA kernel, one CTA
pair per 256 keys, which adds the dQ tiles in the pair-row order, and its
postprocess is the build that reads that order.  The kernel parameters, grids
and workspace are the same on both routes.

The backward runs the three launches in one host call: the library of the
variant's host sequence compiles the three kernel units beside a host unit
that launches them in order, with the arguments and grids of the stage
launchers, and takes ``lse_log2``, ``dpsum`` and ``dq_accum`` from the tvm-ffi
environment allocator (PyTorch's caching allocator, on the current stream).
Without an environment allocator the backward allocates them and passes them
in.  The stage launchers ``launch_bwd_pre``, ``launch_bwd_main`` and
``launch_bwd_post`` launch one stage each.

Every launch is a plain stream-ordered launch: nothing cooperative, no PDL, no
co-residency requirement beyond the clusters of two CTAs of the pair route's
main kernel, which the launch itself schedules together.  Nothing here
reads an environment variable, reads device memory or synchronizes; grids and
buffer sizes come from tensor shapes.

This is the non-deterministic package of its family, and the family has no
deterministic package: the dQ tiles of the key blocks of one head reach the
FP32 accumulator in scheduling order, so ``dq`` may differ in its last bits
run to run.  ``dk``, ``dv`` and the preprocess outputs have one fixed order.
Every backward call that launches raises PyTorch's deterministic-algorithms
alert first (:func:`_alert_nondeterministic`).
"""

from __future__ import annotations

import functools
import inspect
import math
import struct
import warnings
from collections.abc import Mapping
from typing import Any, Optional

import torch

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BLOCK",
    "HEAD_DIMS",
    "arch_for_capability",
    "cake_fmha_nondet_train_backward",
    "device_arch",
    "is_prebuilt",
    "launch_bwd_main",
    "launch_bwd_post",
    "launch_bwd_pre",
    "library_name",
    "load",
    "parameters",
    "record",
    "specs_for_arch",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 /
# GB300 (``sm_103a``) and R200 (``sm_107a``), each compiled with its exact
# ``-gencode`` flag set.  No other architecture has a build.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
HEAD_DIMS: tuple[int, ...] = tuple(cake_jit.HEAD_DIMS)
BLOCK: int = cake_jit.BLOCK
LOG2_E = math.log2(math.e)

# The grid's y axis is the head and its z axis the batch (CUDA limits both to 65535); the kernels index with 32-bit
# offsets, so every operand holds fewer than 2^31 elements.
_MAX_GRID_YZ = 65535
_MAX_ELEMENTS = 2**31

# The entry named by the deterministic-algorithms alert.
_ENTRY = "cake_fmha_nondet_train_backward"

# This package's module prefix (the vendored name of a vendored copy).
_PACKAGE = __name__.rpartition(".")[0]


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
            f"device {device_index} (capability {capability}) is not an exported Cake dense attention backward "
            f"target; the exported architectures are {ARCHES}"
        )
    return arch


# ----------------------------------------------------------------------------------------------------------------
# Records and libraries (``build.json``)


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        raise KeyError(f"{name!r} is not a record of the Cake dense attention backward registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake dense attention backward record {name} is not delivered for {arch}; its architectures: "
            f"{found['arches']}"
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


@functools.cache
def _sequence(variant: tuple[int, bool, bool], arch: str) -> Optional[tuple[Any, Any, Any]]:
    """``(library, run, run_with_workspace)`` of the host sequence of ``variant`` (a ``RECORD_OF`` key ``(D, causal,
    pair)``, :func:`.cake_jit.variant`) built for ``arch`` (looked up once per process), or ``None`` when
    ``SEQUENCES`` has none for the variant on ``arch``; the backward then launches through the stage launchers.

    The library compiles the variant's three kernel units beside one host unit.  ``run(q, k, v, out, lse, dout, dq,
    dk, dv, softmax_scale, softmax_scale_log2)`` launches the records ``RECORD_OF`` names for the variant, in order,
    with the arguments and grids of their stage launchers, in one host call; ``lse_log2``, ``dpsum`` and ``dq_accum``
    come from the tvm-ffi environment allocator (PyTorch's caching allocator, on the current stream).  It returns
    ``False`` without launching when no environment allocator is set, and raises the builtin ``MemoryError``
    before any launch when an allocation fails.  ``run_with_workspace(q, k, v, out, lse, dout,
    dq, dk, dv, lse_log2, dpsum, dq_accum, softmax_scale, softmax_scale_log2)`` takes the three buffers from the
    caller."""
    found = cake_jit.SEQUENCES.get(variant)
    if found is None or arch not in found["arches"]:
        return None
    if tuple(found["records"]) != tuple(cake_jit.RECORD_OF[variant]):
        raise RuntimeError(
            f"the host sequence of variant {variant} runs {found['records']}, not the records "
            f"{cake_jit.RECORD_OF[variant]} of RECORD_OF"
        )
    library = cake_build.load_library(f"{found['library']}_{arch}")
    return library, library.run, library.run_with_workspace


def _launch(stage: str, name: str, values: Mapping[str, Any], dims: Mapping[str, int], device: torch.device) -> None:
    """Launch record ``name`` of ``stage`` with its kernel parameters named in ``values``, on the grid of ``dims``,
    on the current torch stream of ``device`` (made the current device for the launch)."""
    found = record(name)
    if found["stage"] != stage:
        raise ValueError(f"record {name} serves stage {found['stage']}, not {stage}")
    index = device.index
    arch = device_arch(index)
    args = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    grid = cake_jit.grid(stage, dims["B"], dims["H"], dims["Sq"], dims["Skv"])
    # tvm-ffi runs the call on the current torch stream of the device of its first CUDA tensor argument (every
    # buffer is on ``device``), and the binding makes that device current for the launch.
    _entry(name, arch)(*args, int(grid[0]), int(grid[1]), int(grid[2]))


def _fp32(value: float) -> float:
    """``value`` rounded to FP32 (round to nearest even), as a Python float."""
    return struct.unpack("f", struct.pack("f", value))[0]


# The kernel parameters of each stage in the order its stage launcher passes them positionally (``_positional``).
_POSITIONAL = {
    "bwd_pre": ("O", "dO", "LSE", "LSE_log2", "dPsum", "dQ_accum", "seqlen_q", "num_heads"),
    "bwd_main": (
        "dQ_accum",
        "LSE_log2_gmem",
        "dPsum_gmem",
        "dK_gmem",
        "dV_gmem",
        "Q",
        "K",
        "V",
        "dO",
        "dK",
        "dV",
        "seqlen_q",
        "seqlen_kv",
        "softmax_scale",
        "softmax_scale_log2",
        "num_m_blocks",
        "num_heads",
    ),
    "bwd_post": ("dQ_accum", "dQ", "seqlen_q", "softmax_scale", "num_heads"),
}


@functools.cache
def _positional(name: str, stage: str) -> bool:
    """Whether record ``name`` serves ``stage``, takes the parameters of ``_POSITIONAL[stage]`` in that order and
    :func:`.cake_jit.grid` gives ``stage`` the grid ``(S / BLOCK, H, B)``; its stage launcher then passes the
    arguments positionally to the FFI entry, and otherwise launches through :func:`_launch`.  An unknown record is
    the ``KeyError`` of :func:`record`."""
    if record(name)["stage"] != stage or parameters(name) != _POSITIONAL[stage]:
        return False
    # Distinct sizes, each one row past a multiple of BLOCK: a swapped axis, another tile size or a rounded-up tile
    # count gives another grid than the stage launchers' (S // BLOCK, H, B).
    rows = 11 if stage == "bwd_main" else 7
    return tuple(cake_jit.grid(stage, 3, 5, 7 * BLOCK + 1, 11 * BLOCK + 1)) == (rows, 5, 3)


# ----------------------------------------------------------------------------------------------------------------
# Stage launchers: caller-owned buffers, the current stream, no allocation, no check of the operands' contents


def launch_bwd_pre(
    name: str,
    *,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    lse_log2: torch.Tensor,
    dpsum: torch.Tensor,
    dq_accum: torch.Tensor,
) -> None:
    """Launch the preprocess record ``name``: from ``out`` and ``dout`` (BF16 ``[B, Sq, H, D]``) and ``lse`` (FP32
    natural log ``[B, H, Sq]``) it writes ``dpsum = rowsum(out * dout)`` and ``lse_log2 = lse * log2(e)`` (0 where
    ``lse`` is ``-inf``), FP32 ``[B*H, Sq]`` (row ``b*H + h``), and clears ``dq_accum`` (FP32, ``B*H*Sq*D``
    elements)."""
    B, Sq, H, _ = out.shape
    if _positional(name, "bwd_pre"):
        _entry(name, device_arch(out.device.index))(out, dout, lse, lse_log2, dpsum, dq_accum, Sq, H, Sq // BLOCK, H, B)
        return
    dims = {"B": int(B), "H": int(H), "Sq": int(Sq), "Skv": int(Sq)}
    values = {
        "O": out,
        "dO": dout,
        "LSE": lse,
        "LSE_log2": lse_log2,
        "dPsum": dpsum,
        "dQ_accum": dq_accum,
        "seqlen_q": dims["Sq"],
        "num_heads": dims["H"],
    }
    _launch("bwd_pre", name, values, dims, out.device)


def launch_bwd_main(
    name: str,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    lse_log2: torch.Tensor,
    dpsum: torch.Tensor,
    dq_accum: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Launch the main-stage record ``name``: from ``q``, ``dout`` (BF16 ``[B, Sq, H, D]``), ``k``, ``v`` (BF16
    ``[B, Skv, H, D]``) and the preprocess outputs ``lse_log2`` and ``dpsum`` (FP32 ``[B*H, Sq]``) it writes ``dk``
    and ``dv`` (BF16 ``[B, Skv, H, D]``) and adds the unscaled dQ tiles into ``dq_accum`` (FP32, ``B*H*Sq*D``
    elements, cleared by the preprocess)."""
    B, Sq, H, _ = q.shape
    Skv = k.shape[1]
    scale = float(softmax_scale)
    if _positional(name, "bwd_main"):
        # The parameters of _POSITIONAL["bwd_main"] in order, then the grid (Skv // BLOCK, H, B).
        _entry(name, device_arch(q.device.index))(
            dq_accum,
            lse_log2,
            dpsum,
            dk,
            dv,
            q,
            k,
            v,
            dout,
            dk,
            dv,
            Sq,
            Skv,
            scale,
            _fp32(scale * LOG2_E),
            Sq // BLOCK,
            H,
            Skv // BLOCK,
            H,
            B,
        )
        return
    dims = {"B": int(B), "H": int(H), "Sq": int(Sq), "Skv": int(Skv)}
    values = {
        "dQ_accum": dq_accum,
        "LSE_log2_gmem": lse_log2,
        "dPsum_gmem": dpsum,
        "dK_gmem": dk,
        "dV_gmem": dv,
        "Q": q,
        "K": k,
        "V": v,
        "dO": dout,
        "dK": dk,
        "dV": dv,
        "seqlen_q": dims["Sq"],
        "seqlen_kv": dims["Skv"],
        "softmax_scale": scale,
        "softmax_scale_log2": _fp32(scale * LOG2_E),
        "num_m_blocks": dims["Sq"] // BLOCK,
        "num_heads": dims["H"],
    }
    _launch("bwd_main", name, values, dims, q.device)


def launch_bwd_post(
    name: str,
    *,
    dq_accum: torch.Tensor,
    dq: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Launch the postprocess record ``name``: it writes ``dq = bf16(dq_accum * softmax_scale)``, BF16
    ``[B, Sq, H, D]``, from the FP32 accumulator of the main stage."""
    B, Sq, H, _ = dq.shape
    if _positional(name, "bwd_post"):
        _entry(name, device_arch(dq.device.index))(dq_accum, dq, Sq, float(softmax_scale), H, Sq // BLOCK, H, B)
        return
    dims = {"B": int(B), "H": int(H), "Sq": int(Sq), "Skv": int(Sq)}
    values = {
        "dQ_accum": dq_accum,
        "dQ": dq,
        "seqlen_q": dims["Sq"],
        "softmax_scale": float(softmax_scale),
        "num_heads": dims["H"],
    }
    _launch("bwd_post", name, values, dims, dq.device)


# ----------------------------------------------------------------------------------------------------------------
# torch.use_deterministic_algorithms(True)


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside this package and ``cake_pplx``, for a warning issued by the
    function that calls this one: the caller of the entry point, also through the facade and its decorators."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None:
            name = str(parent.f_globals.get("__name__", ""))
            if not (
                name == _PACKAGE
                or name.startswith(_PACKAGE + ".")
                or name == "cake_pplx"
                or name.startswith("cake_pplx.")
            ):
                return level
        frame = parent
    return 2


def _alert_nondeterministic(entry: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic algorithms
    are enabled: a ``RuntimeError``, or a ``UserWarning`` under ``torch.use_deterministic_algorithms(True,
    warn_only=True)`` after which the backward runs; nothing otherwise.  Called by every backward call that launches,
    after the checks and before it allocates or launches anything.  A backward with an empty batch launches nothing
    and does not alert.  There is no fallback: the family has no deterministic package.  The alert is raised here
    because PyTorch exposes no public Python entry for it."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'; use the 'warn_only=True' option if that is acceptable for your "
        "application"
    )


# ----------------------------------------------------------------------------------------------------------------
# Checks (dtypes, shapes, strides, alignment and devices only: nothing reads tensor contents)


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
) -> tuple[dict[str, int], float]:
    """The sizes ``B``, ``H``, ``Sq``, ``Skv``, ``D`` of a valid call and its ``softmax_scale``; ``ValueError``
    otherwise.  The device check comes before anything that touches CUDA.  :func:`_accepted` repeats these checks
    for the hot path: a change here needs the same change there."""
    operands = (("q", q), ("k", k), ("v", v), ("out", out), ("lse", lse), ("dout", dout))
    for name, t in operands:
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    for name, t in operands:
        if t.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {t.device}")
    for name, t in operands:
        if t.device != q.device:
            raise ValueError(f"{name} must be on q's device {q.device}, got {t.device}")
    for name, t in operands:
        wanted = torch.float32 if name == "lse" else torch.bfloat16
        if t.dtype != wanted:
            raise ValueError(f"{name} must be {wanted}, got {t.dtype}")
    for name, t in operands:
        rank = 3 if name == "lse" else 4
        if t.dim() != rank:
            raise ValueError(f"{name} must have {rank} dimensions, got {tuple(t.shape)}")
    B, Sq, H, D = (int(n) for n in q.shape)
    Skv = int(k.shape[1])
    for name, t in (("out", out), ("dout", dout)):
        if tuple(t.shape) != tuple(q.shape):
            raise ValueError(f"{name} must have q's shape {tuple(q.shape)} [B, Sq, H, D], got {tuple(t.shape)}")
    if tuple(k.shape) != (B, Skv, H, D):
        raise ValueError(
            f"k must be [B, Skv, H, D] with q's B = {B}, H = {H} and D = {D}, got {tuple(k.shape)}"
        )
    if tuple(v.shape) != tuple(k.shape):
        raise ValueError(f"v must have k's shape {tuple(k.shape)}, got {tuple(v.shape)}")
    if tuple(lse.shape) != (B, H, Sq):
        raise ValueError(f"lse must be [B, H, Sq] = {(B, H, Sq)}, got {tuple(lse.shape)}")
    for name, t in operands:
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got strides {t.stride()} for shape {tuple(t.shape)}")
    if D not in HEAD_DIMS:
        raise ValueError(f"the head dimension must be one of {sorted(HEAD_DIMS)}, got {D}")
    if H < 1:
        raise ValueError(f"the head count must be at least 1, got {H}")
    for name, length in (("Sq", Sq), ("Skv", Skv)):
        if length < BLOCK or length % BLOCK:
            raise ValueError(f"{name} must be a positive multiple of {BLOCK}, got {length}")
    if causal and Sq != Skv:
        raise ValueError(f"causal attention needs Sq == Skv, got Sq = {Sq} and Skv = {Skv}")
    if B > _MAX_GRID_YZ or H > _MAX_GRID_YZ:
        raise ValueError(f"B and H must each be at most {_MAX_GRID_YZ} (CUDA grid limit), got B = {B}, H = {H}")
    if B * H * max(Sq, Skv) * D >= _MAX_ELEMENTS:
        raise ValueError(
            f"B * H * max(Sq, Skv) * D must be below 2^31 (32-bit offsets), got {B} * {H} * {max(Sq, Skv)} * {D}"
        )
    for name, t in operands:
        alignment = 4 if name == "lse" else 16
        if t.data_ptr() % alignment:
            raise ValueError(f"{name} must start on a {alignment}-byte boundary, got address {t.data_ptr():#x}")
    scale = D**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError(f"softmax_scale must be finite, got {softmax_scale}")
    return {"B": B, "H": H, "Sq": Sq, "Skv": Skv, "D": D}, scale


def _accepted(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
) -> Optional[tuple[int, int, int, int, int, float]]:
    """``(B, H, Sq, Skv, D, softmax_scale)`` of a call that passes every check of :func:`_check`, found with the
    cheapest tests in one pass; ``None`` when any of them fails (or an operand is a ``torch.Tensor`` subclass), and then
    :func:`_check` runs and raises its error.  The backward also runs :func:`_check` when this function raises, so an
    invalid call gets the error of :func:`_check` in every case."""
    tensor = torch.Tensor
    if not (
        type(q) is tensor
        and type(k) is tensor
        and type(v) is tensor
        and type(out) is tensor
        and type(lse) is tensor
        and type(dout) is tensor
    ):
        return None
    bf16 = torch.bfloat16
    if not (
        q.dtype is bf16
        and k.dtype is bf16
        and v.dtype is bf16
        and out.dtype is bf16
        and dout.dtype is bf16
        and lse.dtype is torch.float32
    ):
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda and lse.is_cuda and dout.is_cuda):
        return None
    index = q.get_device()
    if not (
        k.get_device() == index
        and v.get_device() == index
        and out.get_device() == index
        and lse.get_device() == index
        and dout.get_device() == index
    ):
        return None
    shape = q.shape
    kv_shape = k.shape
    if len(shape) != 4 or len(kv_shape) != 4 or out.shape != shape or dout.shape != shape or v.shape != kv_shape:
        return None
    B, Sq, H, D = shape
    Skv = kv_shape[1]
    if kv_shape[0] != B or kv_shape[2] != H or kv_shape[3] != D or lse.shape != (B, H, Sq):
        return None
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and out.is_contiguous()
        and lse.is_contiguous()
        and dout.is_contiguous()
    ):
        return None
    if D not in HEAD_DIMS or H < 1 or Sq < BLOCK or Sq % BLOCK or Skv < BLOCK or Skv % BLOCK:
        return None
    if (causal and Sq != Skv) or B > _MAX_GRID_YZ or H > _MAX_GRID_YZ or B * H * max(Sq, Skv) * D >= _MAX_ELEMENTS:
        return None
    if (q.data_ptr() | k.data_ptr() | v.data_ptr() | out.data_ptr() | dout.data_ptr()) % 16 or lse.data_ptr() % 4:
        return None
    scale = D**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        return None
    return B, H, Sq, Skv, D, scale


# ----------------------------------------------------------------------------------------------------------------
# Backward


def cake_fmha_nondet_train_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of dense multi-head attention: ``(dq, dk, dv)`` from the forward's ``out`` and ``lse``.

    ``q``, ``out``, ``dout``: BF16 ``[B, Sq, H, D]``; ``k``, ``v``: BF16 ``[B, Skv, H, D]``; ``lse``: FP32
    ``[B, H, Sq]``, the natural-log ``logsumexp`` of ``softmax_scale * q k^T`` over each query row (the mask
    applied).  All contiguous, on one CUDA device, BF16 operands 16-byte aligned (``lse`` 4-byte aligned).  ``D`` is
    64 or 128, ``H >= 1``, ``Sq`` and ``Skv`` are positive multiples of 128; ``causal`` (top-left: query ``i`` sees
    keys ``j <= i``) needs ``Sq == Skv``.  ``softmax_scale`` defaults to ``D ** -0.5`` and must be the forward's.
    Returns contiguous BF16 ``dq [B, Sq, H, D]``, ``dk`` and ``dv [B, Skv, H, D]`` on the operands' device.

    The dQ tiles of the key blocks of one head are added into an FP32 accumulator in arrival order, so ``dq`` is not
    bitwise reproducible run to run; ``dk`` and ``dv`` have one fixed order.  Under
    ``torch.use_deterministic_algorithms(True)`` the call raises PyTorch's ``RuntimeError`` before it launches
    (``warn_only=True``: one ``UserWarning``, then it runs).  An empty batch returns empty gradients without an
    alert.  The kernels run on the current stream of the operands' device; nothing synchronizes.
    """
    causal = bool(causal)
    try:
        accepted = _accepted(q, k, v, out, lse, dout, softmax_scale, causal)
    except Exception:  # an operand the one-pass check cannot read (a nested tensor's shape): _check gives the error
        accepted = None
    if accepted is None:
        dims, scale = _check(q, k, v, out, lse, dout, softmax_scale, causal)
        B, H, Sq, Skv, D = dims["B"], dims["H"], dims["Sq"], dims["Skv"], dims["D"]
    else:
        B, H, Sq, Skv, D, scale = accepted
    device = q.device
    if B == 0:
        # No query row: nothing to launch and nothing to alert.
        dq = torch.empty(0, Sq, H, D, dtype=torch.bfloat16, device=device)
        dk = torch.empty(0, Skv, H, D, dtype=torch.bfloat16, device=device)
        return dq, dk, torch.empty_like(dk)
    arch = device_arch(device.index)  # RuntimeError on a device outside ARCHES, before the alert
    variant = cake_jit.variant(D, causal, Skv, arch)
    pre, main, post = cake_jit.RECORD_OF[variant]
    _alert_nondeterministic(_ENTRY)
    sequence = _sequence(variant, arch)
    if sequence is not None:
        # One host call runs the three records: the kernels, arguments and grids of the stage launchers below.
        _, run, run_with_workspace = sequence
        dq = torch.empty(B, Sq, H, D, dtype=torch.bfloat16, device=device)
        dk = torch.empty(B, Skv, H, D, dtype=torch.bfloat16, device=device)
        dv = torch.empty(B, Skv, H, D, dtype=torch.bfloat16, device=device)
        scale_log2 = _fp32(scale * LOG2_E)
        try:
            done = run(q, k, v, out, lse, dout, dq, dk, dv, scale, scale_log2)
        except MemoryError:
            # The environment allocator failed, before any launch (tvm-ffi raises the builtin MemoryError): the
            # workspace is allocated here instead, so a lasting shortage raises torch.OutOfMemoryError, as 0.1.3 did
            # (the failing allocation and its message can differ: 0.1.4 allocates dq, dk and dv first).  Such a call
            # reports two out-of-memory events to the caching allocator (``num_ooms`` of torch.cuda.memory_stats()
            # and the registered OOM observers), the environment allocator's and this one, where 0.1.3 reported
            # one.
            done = False
        if done:
            return dq, dk, dv
        # No tvm-ffi environment allocator, or it failed: the workspace is allocated here.
        lse_log2 = torch.empty(B * H, Sq, dtype=torch.float32, device=device)
        dpsum = torch.empty(B * H, Sq, dtype=torch.float32, device=device)
        dq_accum = torch.empty(B * H * Sq * D, dtype=torch.float32, device=device)  # cleared by bwd_pre
        run_with_workspace(q, k, v, out, lse, dout, dq, dk, dv, lse_log2, dpsum, dq_accum, scale, scale_log2)
        return dq, dk, dv
    lse_log2 = torch.empty(B * H, Sq, dtype=torch.float32, device=device)
    dpsum = torch.empty(B * H, Sq, dtype=torch.float32, device=device)
    dq_accum = torch.empty(B * H * Sq * D, dtype=torch.float32, device=device)  # cleared by bwd_pre
    dq = torch.empty(B, Sq, H, D, dtype=torch.bfloat16, device=device)
    dk = torch.empty(B, Skv, H, D, dtype=torch.bfloat16, device=device)
    dv = torch.empty(B, Skv, H, D, dtype=torch.bfloat16, device=device)
    launch_bwd_pre(pre, out=out, dout=dout, lse=lse, lse_log2=lse_log2, dpsum=dpsum, dq_accum=dq_accum)
    launch_bwd_main(
        main,
        q=q,
        k=k,
        v=v,
        dout=dout,
        lse_log2=lse_log2,
        dpsum=dpsum,
        dq_accum=dq_accum,
        dk=dk,
        dv=dv,
        softmax_scale=scale,
    )
    launch_bwd_post(post, dq_accum=dq_accum, dq=dq, softmax_scale=scale)
    return dq, dk, dv
