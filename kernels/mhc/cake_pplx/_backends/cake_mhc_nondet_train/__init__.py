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

Fused-reduction backend of the Cake four-stream mHC training forward /
backward (hidden sizes 4096, 5120 and 7168 per call, from ``X``; 4 streams;
SM100 / SM103 / SM107; the backward's column-slab pass folds
``dfn``, ``dscale`` and ``dbase`` itself), one self-contained
package (host API bodies, the ``cake_jit`` registry, the ``cake_build`` loader
and the generated CUDA sources under ``csrc/``).  It imports nothing from other
``cake_pplx`` packages and ships its own copy of every program it launches.  The
user-facing module and documentation are
:mod:`cake_pplx.cake_mhc_nondet_train`, which forwards every entry point
here.

Numerical contract: no bit promise.  Every sum of the current programs is a
plain FP32 sum in an order fixed by the token count and the device, but the
package does not promise that an output repeats bit for bit from run to run,
and its outputs are not compared bitwise with the deterministic package's.
The forward, the statistics replay, the residual mixing and
``backward_prepare`` run copies of the deterministic package's programs.
``backward_finish`` runs a copy of the deterministic token-owner pass, the
atomic fn split (beside the token-owner pass, as its programmatic dependent)
and the column slab (see :mod:`.cake_jit`; its dynamic-schedule form over the
deterministic package's token chunks: the host rule's two-size tail schedule
is calibrated for no device) and no final reduction kernel.  The column slab
folds ``dscale`` and ``dbase`` from the token-owner pass's parameter rows in a
fixed order.  It stores the FP32 ``dfn`` sum of every (column slab, token
chunk) unit in its chunk's slots of the backward workspace, and once every
unit has stored its sums the grid adds each element's ``P <= 32`` partials
``x_p`` in ascending chunk order as one FP32 sum from +0.0, the caller's
``dfn`` last under ``accumulate=True``: within ``gamma_n (|start| + sum
|x_p|)`` of the exact sum (``n`` the additions, ``gamma_n = n u / (1 - n u)``,
``u = 2^-24``), no addition flushing a subnormal.  By default
(``accumulation="fp32"``) every sum is one plain FP32 word: no FP64
accumulation or reduction and no compensation word (a few scalar epilogues
of the RMS scale, a finished FP32 sum times ``1/(4H)`` plus ``rms_eps`` or
times the RMS scale, compute in FP64 and round once to FP32).  The opt-in
``accumulation="fp32x2"`` (hidden 4096, ``sinkhorn_iters=20``) runs the two-word programs of the
mapping pass, the token-owner pass and the column slab with its folds
(``cake_jit.ACC_WORDS_PROGRAMS``): each of their sums is a compensated pair
of FP32 words stored as one FP32 word, not FP64.

Mode: the names keep the package's own prefix (``cake_mhc_nondet_train_*``,
``CakeMHCNondet*Function``); no entry point takes a ``deterministic`` keyword
and nothing falls back to the deterministic package
(``cake_pplx.cake_mhc_det_train``).  Every entry point runs under
``torch.use_deterministic_algorithms(True)`` without an alert: no launch adds
in a run-dependent order.

Pre handoff (hidden 5120 on SM100 / SM103 / SM107; ``cake_jit.HANDOFF_KINDS``): the
routes ``fwd_prepare_handoff`` (the standard forward with the collapse
``fwd_collapse__pi_pn``: ``U`` from ``pre_in``, ``pre_next`` written beside
it) and ``bwd_finish_handoff`` (the standard backward with the token-owner
forms ``bwd_token__<form>_ho``: ``dpre_next`` into the coefficient VJP,
``dpre_in`` out), and the standalone collapse routes ``collapse_forward`` /
``collapse_backward`` (no workspace; on SM107 the three collapse records are
that architecture's own forms: ``fwd_collapse__m2_pi_pn``, ``collapse_fwd__m2``,
``collapse_bwd__vjpwa_pfq2``; ``collapse_forward`` runs ``collapse_fwd__m2``
below 6656 tokens and ``collapse_fwd__m2_xhnone``, the same form without the L2
evict-first policy on ``X``, from 6656 tokens).  ``handoff_prepare`` is the
autograd node; ``dU`` / ``dpre_next`` from autograd are made contiguous, of their
dtype and 32-byte aligned before the launch.  No run-to-run bit promise.
"""

from __future__ import annotations

import functools
import math
import operator
import weakref
from typing import Any, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    GSTATS_COLS,
    HANDOFF_KINDS,
    HC_EPS,
    HIDDEN,
    ACCUMULATION,
    HIDDEN_SIZES,
    MAPPING_ROWS,
    RMS_EPS,
    SINKHORN_ITERS,
    SINKHORN_ITERS_MAX,
    STATS_COLS,
    STREAMS,
    arch_for_capability,
    handoff_hidden_sizes,
    route_applies,
    selected_modules,
    supported_hidden_sizes,
)
from .cake_jit import route_key as cake_mhc_nondet_train_route_key
from .cake_jit import route_modules as cake_mhc_nondet_train_route_modules

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The Sinkhorn iteration counts the delivered programs compile in: their chain is
# unrolled for the count of record, the default, which costs nothing at run time.
# Every other count from 1 to SINKHORN_ITERS_MAX runs too: the coefficient pass and
# the token-owner pass then launch their run-time-count programs, which take the
# count as a kernel argument (the other kernels never read it).  The epsilons are
# launch arguments of every program and never select one.
SHIPPED_SINKHORN_ITERS: tuple[int, ...] = (SINKHORN_ITERS,)
# Every kernel tensor is read with 256-bit vector loads or described to the
# tensor memory accelerator: contiguous storage, 32-byte aligned base.
TENSOR_ALIGNMENT_BYTES = 32
WORKSPACE_ALIGNMENT_BYTES = 256
# Bound of the per-token-count caches below (layouts, default-workspace views,
# launch plans); ``cake_jit.host_plan`` keeps the same bound.
_CACHE_ENTRIES = 4096
# The call paths of the per-call PDL argument (``cake_jit.PDL_PATHS``): the forward
# prepare, the backward from saved statistics, the backward after a replayed forward
# prefix (the autograd nodes know which; the public backward entry points are the
# backward path).
_PATH_FORWARD, _PATH_BACKWARD, _PATH_CHECKPOINT = _loader.PDL_PATHS
_BF16 = torch.bfloat16
_F32 = torch.float32
_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint32": torch.uint32,
}


def _require_hidden(hidden: Any, entry: str) -> None:
    """The hidden-size contract of the workspace entry points, checked before any device query: a ``hidden``
    keyword that is not an ``int`` of :data:`cake_jit.HIDDEN_SIZES` raises ``ValueError``.  The sizes of the
    device's architecture (``supported_hidden_sizes``) are checked with the device (:func:`_workspace_bytes`)."""

    if type(hidden) is not int or hidden not in HIDDEN_SIZES:
        raise ValueError(
            f"{entry} supports hidden sizes {HIDDEN_SIZES} (got {hidden!r}); "
            "each architecture's sizes are cake_jit.supported_hidden_sizes(arch)"
        )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


# The exported architecture of each CUDA device, resolved once per device
# index (the capability of a device never changes within a process).
_DEVICE_ARCH: dict[int, str] = {}


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Cake mHC training requires CUDA tensors, got {device}")
    index = _device_index(device)
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        capability = tuple(torch.cuda.get_device_capability(index))
        arch = arch_for_capability(capability)
        if arch is None:
            raise ValueError(
                f"Cake mHC training supports compute capability {sorted(ARCH_BY_CAPABILITY)} only, got {capability}"
            )
        _DEVICE_ARCH[index] = arch
    return arch


def is_cake_mhc_nondet_train_supported(
    device: torch.device, hidden: Optional[int] = None
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if hidden is None:
        return bool(supported_hidden_sizes(arch))
    return int(hidden) in supported_hidden_sizes(arch)


def is_cake_mhc_nondet_train_handoff_supported(
    device: torch.device, hidden: int | None = None
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has the pre-handoff routes and the standalone collapse."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if hidden is None:
        return bool(handoff_hidden_sizes(arch))
    return int(hidden) in handoff_hidden_sizes(arch)


# ----------------------------------------------------------------------------
# Validation


def _validate_tensor(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A kernel operand: exact shape, dtype and device, contiguous, 32-byte aligned storage.

    No layout is materialized silently: a view the kernels cannot read in place
    is a ``ValueError`` naming the argument (the caller decides whether to copy).
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        return tensor
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (strides {tuple(tensor.stride())} for shape {shape})"
        )
    if tensor.data_ptr() % TENSOR_ALIGNMENT_BYTES:
        raise ValueError(
            f"{name} must start at a {TENSOR_ALIGNMENT_BYTES}-byte aligned address"
        )
    return tensor


def _fast_tensor(
    tensor: Any, shape: tuple[int, ...], dtype: torch.dtype, index: int
) -> bool:
    """Whether :func:`_validate_tensor` accepts ``tensor`` for ``shape`` / ``dtype`` on
    CUDA device ``index`` (``shape`` non-empty).  ``False`` sends the call to the
    full validation, which raises the error of record."""

    return (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype is dtype
        and tensor.is_cuda
        and tensor.get_device() == index
        and tensor.shape == shape
        and tensor.is_contiguous()
        and tensor.data_ptr() % TENSOR_ALIGNMENT_BYTES == 0
    )


def _fast_rows(X: Any, kind: str) -> Optional[tuple[int, int, int, str]]:
    """``(rows, hidden, device index, architecture)`` when ``X`` passes
    :func:`_rows_of`, :func:`_device_arch` and :func:`_validate_tensor` with at
    least one token on a device whose architecture and ``kind`` route at that
    hidden size were already checked; else ``None`` (the full path validates and
    raises)."""

    if not (
        isinstance(X, torch.Tensor) and X.dtype is _BF16 and X.is_cuda and X.dim() == 3
    ):
        return None
    rows, streams, hidden = X.shape
    if not (
        rows > 0
        and streams == STREAMS
        and X.is_contiguous()
        and X.data_ptr() % TENSOR_ALIGNMENT_BYTES == 0
    ):
        return None
    index = X.get_device()
    arch = _DEVICE_ARCH.get(index)
    if arch is None or (arch, kind, hidden) not in _ROUTES_CHECKED:
        return None
    return rows, hidden, index, arch


def _fast_constants(rms_eps: Any, hc_eps: Any, sinkhorn_iters: Any) -> bool:
    """Whether :func:`_check_constants` accepts the three constants and the compiled
    programs run them: exact ``float`` epsilons and an ``int`` iteration count of
    :data:`SHIPPED_SINKHORN_ITERS`.  Any other type or count (a run-time count, an
    integral float, a ``bool``) takes the full check and the full path."""

    return (
        type(rms_eps) is float
        and type(hc_eps) is float
        and type(sinkhorn_iters) is int
        and 0.0 < rms_eps < math.inf
        and 0.0 < hc_eps < math.inf
        and sinkhorn_iters in SHIPPED_SINKHORN_ITERS
    )


def _one_word(accumulation: Any) -> bool:
    """Whether ``accumulation`` is the default mode (``"fp32"``): the fast paths serve it alone."""

    return type(accumulation) is str and accumulation == ACCUMULATION


def _rows_of(X: Any) -> tuple[int, int, torch.device]:
    """``(rows, hidden, device)`` of the residual streams ``X`` ``[T, 4, H]``."""

    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if X.ndim != 3 or X.shape[1] != STREAMS or X.shape[2] not in HIDDEN_SIZES:
        raise ValueError(
            f"X must be a [T, {STREAMS}, H] tensor with H in {HIDDEN_SIZES}, "
            f"got {tuple(X.shape)}"
        )
    return int(X.shape[0]), int(X.shape[2]), X.device


def _check_constants(rms_eps: float, hc_eps: float, sinkhorn_iters: int) -> None:
    """Validate the three model constants of a call.

    ``rms_eps`` and ``hc_eps`` are launch arguments of the exported kernels (every
    mapping-pass and token-owner program takes them as kernel parameters), so any
    finite positive value runs on the compiled programs without a rebuild; the
    defaults are the model's ``1e-5`` / ``1e-6``.  ``sinkhorn_iters`` is an integer
    from 1 to :data:`SINKHORN_ITERS_MAX`: the counts of
    :data:`SHIPPED_SINKHORN_ITERS` (the default ``20``) run the compiled programs,
    every other count the run-time-count programs of the coefficient and
    token-owner passes; a count above :data:`SINKHORN_ITERS_MAX` is beyond what
    those programs were built for (``NotImplementedError``).
    """

    if not (math.isfinite(float(rms_eps)) and float(rms_eps) > 0.0):
        raise ValueError(f"rms_eps must be a finite positive float, got {rms_eps!r}")
    if not (math.isfinite(float(hc_eps)) and float(hc_eps) > 0.0):
        raise ValueError(f"hc_eps must be a finite positive float, got {hc_eps!r}")
    if isinstance(sinkhorn_iters, bool):
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        )
    try:
        iters = int(sinkhorn_iters)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        ) from None
    if iters != sinkhorn_iters or iters < 1:
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        )
    if iters > SINKHORN_ITERS_MAX:
        raise NotImplementedError(
            f"the exported Cake mHC training programs run sinkhorn_iters from 1 to {SINKHORN_ITERS_MAX} "
            f"(the run-time-count programs are built for at most {SINKHORN_ITERS_MAX}); got sinkhorn_iters={sinkhorn_iters!r}"
        )


def _output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A caller-provided output (validated like an input) or a fresh contiguous buffer."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return _validate_tensor(tensor, name, shape=shape, device=device, dtype=dtype)


# The (architecture, route kind, hidden size) triples the registry has already
# been checked for; the check itself is a pure function of the registry.
_ROUTES_CHECKED: set[tuple[str, str, int]] = set()


def _check_route(device: torch.device, arch: str, kind: str, hidden: int) -> None:
    if (arch, kind, hidden) in _ROUTES_CHECKED:
        return
    if not route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        hidden=hidden,
        kind=kind,
    ):
        sizes = (
            handoff_hidden_sizes(arch)
            if kind in HANDOFF_KINDS
            else supported_hidden_sizes(arch)
        )
        raise ValueError(
            f"no exported Cake mHC training route {cake_mhc_nondet_train_route_key(hidden, kind)!r} on {arch}; "
            f"supported hidden sizes: {sizes}"
        )
    _ROUTES_CHECKED.add((arch, kind, hidden))


def _parameters(
    fn: Any, scale: Any, base: Any, device: torch.device, hidden: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = _validate_tensor(
        fn, "fn", shape=(MAPPING_ROWS, STREAMS * hidden), device=device, dtype=_F32
    )
    scale = _validate_tensor(scale, "scale", shape=(3,), device=device, dtype=_F32)
    base = _validate_tensor(
        base, "base", shape=(MAPPING_ROWS,), device=device, dtype=_F32
    )
    return fn, scale, base


# ----------------------------------------------------------------------------
# Workspaces


def _workspace_bytes(
    kind: str, rows: int, device: torch.device, hidden: int = HIDDEN
) -> int:
    device = torch.device(device)
    rows = int(rows)
    if rows < 0:
        raise ValueError("T must be non-negative")
    arch = _device_arch(device)
    if isinstance(hidden, bool) or hidden not in supported_hidden_sizes(arch):
        raise ValueError(
            f"hidden must be one of {supported_hidden_sizes(arch)} on {arch}, "
            f"got {hidden!r}"
        )
    return int(
        _loader.workspace_bytes_upto(
            kind, rows, _loader.sm_count(_device_index(device)), arch, int(hidden)
        )
    )


def cake_mhc_nondet_train_forward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens of hidden
    size ``hidden`` on ``device``.

    The forward routes (``forward_prepare`` and the ``forward_stats`` replay) hold
    one 256-byte ticket region (first, at the same offset for every token count),
    the BF16 pieces of ``fn`` in the mapping operand layout and the FP32
    per-K-unit partials of the mapping pass: the deterministic package's forward
    workspace, byte for byte.  The size is the maximum over every token count up
    to ``T``, so a buffer sized for the largest token count serves every smaller
    one (and CUDA Graph replays at a smaller ``T``).  The ticket words must be
    zero on first use (allocate with ``torch.zeros`` or
    :func:`cake_mhc_nondet_train_forward_workspace`); the kernels leave them zero.
    ``T == 0`` needs no workspace (nothing is launched).

    ``hidden`` (keyword, default ``4096``) is the hidden size ``H`` of the calls
    the buffer serves (``X.shape[2]``; a size of ``supported_hidden_sizes`` of
    the device's architecture, else ``ValueError``).  The ``fn`` pieces and the
    partials scale with ``4 H``; every call checks the buffer against its own
    ``H`` and ``T``, so a buffer sized for one ``H`` is refused (``ValueError``)
    by a call of a larger ``H`` it is too small for.
    """

    _require_hidden(hidden, "cake_mhc_nondet_train_forward_workspace_bytes")
    return _workspace_bytes("fwd", T, device, hidden)


def cake_mhc_nondet_train_backward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens of hidden
    size ``hidden`` on ``device``.

    The ``backward_finish`` route holds one 256-byte ticket region and the
    column slab's 512-byte arrival counts (first, at the same offsets for every
    token count: the first 768 bytes must be zero on first use -- allocate
    with ``torch.zeros`` or :func:`cake_mhc_nondet_train_backward_workspace`
    -- and the kernels leave them zero), the transposed BF16 pieces of ``fn``,
    the per-token records of the token-owner pass (``dz'`` pieces and the
    per-token scalars, padded to a multiple of 128 rows), the parameter rows
    of the token-owner pass (per-CTA partials, or 8-token group rows in the
    group-row band) and the column slab's per-chunk FP32 ``dfn`` slots
    (``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes).  The layout is this
    package's own: a deterministic backward workspace does not serve it.  The
    size is the maximum over every token count up to ``T``; ``hidden`` is the
    hidden size of the calls the buffer serves, as in
    :func:`cake_mhc_nondet_train_forward_workspace_bytes` (a buffer sized for one ``H``
    is refused by a call of a larger ``H`` it is too small for).
    """

    _require_hidden(hidden, "cake_mhc_nondet_train_backward_workspace_bytes")
    return _workspace_bytes("bwd", T, device, hidden)


def cake_mhc_nondet_train_forward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace for up to ``T`` tokens of hidden
    size ``hidden`` (see ``..._forward_workspace_bytes``)."""

    _require_hidden(hidden, "cake_mhc_nondet_train_forward_workspace")
    nbytes = _workspace_bytes("fwd", T, device, hidden)
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_mhc_nondet_train_backward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace for up to ``T`` tokens of hidden
    size ``hidden`` (see ``..._backward_workspace_bytes``)."""

    _require_hidden(hidden, "cake_mhc_nondet_train_backward_workspace")
    nbytes = _workspace_bytes("bwd", T, device, hidden)
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_mhc_nondet_train_reset_workspace(workspace: torch.Tensor) -> None:
    """Return a workspace to its first-use state after an interrupted launch.

    The kernels' unit tickets and the column slab's arrival counts self-reset at
    the end of every launch; a launch that did not complete (an asynchronous
    error, a cancelled stream capture) can leave them non-zero.  Zeroing the
    whole buffer restores the contract for every token count the buffer serves;
    no other region needs a value.
    """

    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
    ):
        raise ValueError("workspace must be a 1-D uint8 tensor")
    workspace.zero_()


# The byte layout of one (workspace kind, token count, SM count, architecture,
# hidden size) -- region dtypes, shapes, contiguous strides and element offsets
# -- is computed once;
# the region views of a caller workspace are then one ``as_strided`` per
# region on a typed view of the buffer.
_LAYOUTS: dict[tuple[str, int, int, str, int], dict[str, Any]] = {}


def _layout(
    kind: str, rows: int, sm_count: int, arch: str, hidden: int
) -> dict[str, Any]:
    key = (kind, rows, sm_count, arch, hidden)
    found = _LAYOUTS.get(key)
    if found is None:
        regions = _loader.workspace_regions(kind, rows, sm_count, arch, hidden)
        plan = []
        for name, region in regions["regions"].items():
            dtype = _TORCH_DTYPES[region["dtype"]]
            shape = tuple(int(dim) for dim in region["shape"])
            strides = []
            step = 1
            for dim in reversed(shape):
                strides.append(step)
                step *= dim
            element_offset = region["offset"] // dtype.itemsize
            plan.append((name, dtype, shape, tuple(reversed(strides)), element_offset))
        found = {"total_bytes": int(regions["total_bytes"]), "plan": tuple(plan)}
        if len(_LAYOUTS) >= _CACHE_ENTRIES:
            _LAYOUTS.clear()
        _LAYOUTS[key] = found
    return found


def _workspace_views(
    workspace: Any, kind: str, *, rows: int, device: torch.device, hidden: int
) -> dict[str, torch.Tensor]:
    """The region views of ``workspace`` for exactly ``rows`` tokens of hidden size
    ``hidden`` (validated against the layout)."""

    layout = _layout(
        kind,
        rows,
        _loader.sm_count(_device_index(device)),
        _device_arch(device),
        hidden,
    )
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    if (
        workspace.device != device
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
    ):
        raise ValueError(
            f"workspace must be a 1-D uint8 CUDA tensor on {device}, got "
            f"{workspace.dtype} {tuple(workspace.shape)} on {workspace.device}"
        )
    if (
        not workspace.is_contiguous()
        or workspace.data_ptr() % WORKSPACE_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"workspace must be contiguous and {WORKSPACE_ALIGNMENT_BYTES}-byte aligned"
        )
    total = layout["total_bytes"]
    if workspace.numel() < total:
        helper = (
            "cake_mhc_nondet_train_forward_workspace_bytes"
            if kind == "fwd"
            else "cake_mhc_nondet_train_backward_workspace_bytes"
        )
        raise ValueError(
            f"workspace holds {workspace.numel()} bytes; T={rows} at hidden={hidden} "
            f"needs {total} ({helper}(T, device, hidden={hidden}))"
        )
    base = workspace[:total]  # a multiple of 256 bytes: every typed view divides it
    typed: dict[torch.dtype, torch.Tensor] = {}
    views: dict[str, torch.Tensor] = {}
    for name, dtype, shape, strides, offset in layout["plan"]:
        flat = typed.get(dtype)
        if flat is None:
            flat = typed[dtype] = base.view(dtype)
        views[name] = flat.as_strided(shape, strides, flat.storage_offset() + offset)
    if kind == "bwd":
        views["dzp_rows"] = views["dzp"][:rows]
    return views


# Default workspaces of the convenience path (``workspace=None``): one buffer
# per (device, kind, hidden size, power-of-two token bucket), allocated on first
# use, so a caller that passes no workspace allocates at most log2(T_max)
# buffers per device and hidden size while the caller-held path never
# allocates.  Every stream and every graph replay on the device shares a
# bucket's buffer: calls that can run concurrently on different streams pass
# caller-owned workspaces.  No buffer is created during a CUDA graph capture
# (its zero fill would only be recorded).
_DEFAULT_WORKSPACES: dict[tuple[int, str, int, int], torch.Tensor] = {}


def _default_workspace(
    kind: str, rows: int, hidden: int, device: torch.device
) -> torch.Tensor:
    bucket = 1 << (max(int(rows), 1) - 1).bit_length()
    key = (_device_index(device), kind, hidden, bucket)
    found = _DEFAULT_WORKSPACES.get(key)
    if found is None:
        # The fill runs on the current stream of ``device`` (the stream the
        # call launches on), which need not be the current device's stream.
        with torch.cuda.device(key[0]):
            capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            raise ValueError(
                f"no default {kind} workspace exists for T={rows} at hidden={hidden} (token-count bucket "
                f"{bucket}) and none can be allocated during a CUDA graph capture: run the call once with "
                "the same token count and hidden size before the capture, or pass a caller-owned workspace"
            )
        found = torch.zeros(
            _workspace_bytes(kind, bucket, device, hidden),
            dtype=torch.uint8,
            device=device,
        )
        _DEFAULT_WORKSPACES[key] = found
    return found


# The region views of the default workspaces, per (device, kind, hidden size,
# token count): those buffers live for the whole process, so their views can
# too.
_DEFAULT_VIEWS: dict[tuple[int, str, int, int], dict[str, torch.Tensor]] = {}


def _workspace_for(
    workspace: Optional[torch.Tensor],
    kind: str,
    rows: int,
    hidden: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if workspace is None:
        key = (_device_index(device), kind, hidden, rows)
        views = _DEFAULT_VIEWS.get(key)
        if views is None:
            views = _workspace_views(
                _default_workspace(kind, rows, hidden, device),
                kind,
                rows=rows,
                device=device,
                hidden=hidden,
            )
            if len(_DEFAULT_VIEWS) >= _CACHE_ENTRIES:
                _DEFAULT_VIEWS.clear()
            _DEFAULT_VIEWS[key] = views
        return views
    return _caller_views(workspace, kind, rows, hidden, device)


# The region views of caller-owned workspaces, per workspace object and (kind,
# hidden size, token count, device): ``id(workspace) -> (weak reference, data
# pointer, {(kind, hidden, rows, device index): views})``.  The views of an
# entry alias the workspace's storage without referencing the workspace tensor,
# and the weak reference drops the entry when that tensor dies, so the cache
# never keeps a caller's buffer alive beyond the caller's own reference.  Every
# call still validates the workspace; a moved data pointer rebuilds the views.
_CALLER_VIEWS: dict[int, tuple[Any, int, dict[tuple[str, int, int, int], Any]]] = {}
# Views kept per workspace object (a caller that cycles through more token
# counts or hidden sizes on one buffer rebuilds them).
_CALLER_VIEW_ENTRIES = 64


def _forget_workspace(key: int, ref: Any) -> None:
    entry = _CALLER_VIEWS.get(key)
    if entry is not None and entry[0] is ref:
        del _CALLER_VIEWS[key]


def _caller_views(
    workspace: Any, kind: str, rows: int, hidden: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """:func:`_workspace_views` of a caller workspace, cached per workspace object."""

    if type(workspace) is not torch.Tensor:
        return _workspace_views(
            workspace, kind, rows=rows, device=device, hidden=hidden
        )
    index = _device_index(device)
    entry = _CALLER_VIEWS.get(id(workspace))
    if (
        entry is not None
        and entry[0]() is workspace
        and entry[1] == workspace.data_ptr()
    ):
        found = entry[2].get((kind, hidden, rows, index))
        if found is not None:
            views, total = found
            if (
                workspace.device == device
                and workspace.dtype == torch.uint8
                and workspace.ndim == 1
                and workspace.is_contiguous()
                and workspace.numel() >= total
            ):
                # The data pointer is the cached one, so it is still aligned.
                return views
            return _workspace_views(
                workspace, kind, rows=rows, device=device, hidden=hidden
            )
    views = _workspace_views(workspace, kind, rows=rows, device=device, hidden=hidden)
    total = _layout(kind, rows, _loader.sm_count(index), _device_arch(device), hidden)[
        "total_bytes"
    ]
    aliased = _aliased_views(workspace, views)
    if entry is None or entry[0]() is not workspace:
        key = id(workspace)
        entry = (
            weakref.ref(workspace, functools.partial(_forget_workspace, key)),
            workspace.data_ptr(),
            {},
        )
        _CALLER_VIEWS[key] = entry
    elif entry[1] != workspace.data_ptr():
        entry = (entry[0], workspace.data_ptr(), {})
        _CALLER_VIEWS[id(workspace)] = entry
    if len(entry[2]) >= _CALLER_VIEW_ENTRIES:
        entry[2].clear()
    entry[2][(kind, hidden, rows, index)] = (aliased, total)
    return aliased


def _aliased_views(
    workspace: torch.Tensor, views: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """``views`` rebuilt on ``workspace``'s storage without a reference to
    ``workspace`` itself: the same dtypes, shapes, strides and data pointers."""

    storage = workspace.untyped_storage()
    aliased: dict[str, torch.Tensor] = {}
    for name, view in views.items():
        alias = torch.empty((0,), dtype=view.dtype, device=view.device)
        alias.set_(storage, view.storage_offset(), view.shape, view.stride())
        aliased[name] = alias
    return aliased


# ----------------------------------------------------------------------------
# Launch plans
#
# The modules one public call launches, their grids and their argument order
# are pure functions of (architecture, route, token count, hidden size,
# device); the plan below resolves them once per such key and keeps the bound
# FFI entry of every module, so a call fills the tensor slots of each argument
# template and launches.  The route runs on the current torch stream of the tensors'
# device, set once for the whole route.

_TVM_FFI: Any = None
_TVM_DEVICES: dict[int, Any] = {}


def _tvm_ffi() -> Any:
    global _TVM_FFI
    if _TVM_FFI is None:
        import tvm_ffi

        _TVM_FFI = tvm_ffi
    return _TVM_FFI


def _tvm_device(device_index: int) -> Any:
    found = _TVM_DEVICES.get(device_index)
    if found is None:
        found = _TVM_DEVICES[device_index] = _tvm_ffi().device("cuda", device_index)
    return found


class _Launch:
    """One module launch of a route plan: the bound FFI entry, the argument
    template (grid and record-fixed scalars filled in) and the template
    positions a call fills from its logical values."""

    __slots__ = ("run", "template", "slots")

    def __init__(
        self, run: Any, template: list[Any], slots: tuple[tuple[int, str], ...]
    ) -> None:
        self.run, self.template, self.slots = run, template, slots


_PLANS: dict[tuple[str, str, int, int, int, str, bool, str], tuple[_Launch, ...]] = {}


def _route_plan(
    arch: str,
    kind: str,
    rows: int,
    hidden: int,
    device_index: int,
    path: str,
    sinkhorn_iters: int = SINKHORN_ITERS,
    accumulation: str = ACCUMULATION,
) -> tuple[_Launch, ...]:
    """The launches of one route for ``rows`` tokens of hidden size ``hidden`` on
    ``device_index`` along ``path`` (one of ``cake_jit.PDL_PATHS``: with the
    device's SM count it fixes the per-call PDL argument of the records whose
    binding takes one, :func:`cake_jit.enable_pdl`).  A ``sinkhorn_iters`` other
    than the count of record selects the run-time-count records; the count
    itself is a launch value, so one plan serves every such count.  A route that
    exports no run-time-count program raises ``NotImplementedError`` for such a
    count (``cake_jit.selected_modules``).  ``accumulation`` (``"fp32"`` or
    ``"fp32x2"``) is part of the plan's key: the two-word mode selects its own
    programs (``cake_jit.two_word_modules``)."""

    run_time = int(sinkhorn_iters) != SINKHORN_ITERS
    key = (arch, kind, rows, hidden, device_index, path, run_time, accumulation)
    plan = _PLANS.get(key)
    if plan is None:
        launches = []
        sm_count = _loader.sm_count(device_index)
        for name in selected_modules(
            arch,
            hidden,
            kind,
            rows=rows,
            sm_count=sm_count,
            sinkhorn_iters=sinkhorn_iters,
            accumulation=accumulation,
        ):
            record = _loader.MODULES[name]
            logical = record["logical"]
            fixed = _loader.RECORD_SCALARS.get(name, {})
            grid_x, grid_y, grid_z = _loader.launch_grid(
                record, rows, device_index=device_index
            )
            grid_values = {"grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z}
            template: list[Any] = []
            slots: list[tuple[int, str]] = []
            for position, (arg_kind, arg_key) in enumerate(record["arg_plan"]):
                if arg_kind == "grid":
                    template.append(grid_values[arg_key])
                    continue
                if arg_kind == "launch":
                    template.append(
                        _loader.enable_pdl(name, rows, arch, path, sm_count=sm_count)
                    )
                    continue
                logical_name = logical[arg_key]
                if logical_name in fixed:
                    template.append(fixed[logical_name])
                    continue
                template.append(None)
                slots.append((position, logical_name))
            module = _loader.load(name, arch)
            launches.append(
                _Launch(getattr(module, record["ffi_entry"]), template, tuple(slots))
            )
        if len(_PLANS) >= _CACHE_ENTRIES:
            _PLANS.clear()
        plan = _PLANS[key] = tuple(launches)
    return plan


# The raw handle of the current torch stream of a device (what
# ``torch.cuda.current_stream(index).cuda_stream`` returns, without building a
# ``torch.cuda.Stream``).
_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream(device_index: int) -> int:
    if _CURRENT_RAW_STREAM is not None:
        return _CURRENT_RAW_STREAM(device_index)
    return torch.cuda.current_stream(device_index).cuda_stream


# The per-call logical values of each route, in the order a call passes them to
# :func:`_launch_route`, then the workspace regions (fetched from the views with
# ``_WORKSPACE_VALUES``); the token count and the host-plan scalars of a route
# (and the column slab's ``lpt`` chunks of ``cake_jit.slab_plan``) are fixed per
# token count (``_HOST_VALUES``) and bound into the plan.
_CALL_VALUES: dict[str, tuple[str, ...]] = {
    "fwd_prepare": (
        "fn",
        "x",
        "x_flat",
        "scale",
        "base",
        "stats",
        "rms_eps",
        "hc_eps",
        "u",
    ),
    "fwd_recompute": (
        "fn",
        "x",
        "x_flat",
        "scale",
        "base",
        "stats",
        "rms_eps",
        "hc_eps",
    ),
    "fwd_finish": ("x", "b", "stats", "y"),
    "bwd_prepare": ("dy", "b", "x", "stats", "db", "gstats"),
    "bwd_finish": (
        "fn",
        "x",
        "du",
        "stats",
        "gstats",
        "scale",
        "base",
        "hc_eps",
        "x_flat",
        "dy_flat",
        "dx_flat",
        "dfn",
        "dscale",
        "dbase",
        "accumulate",
    ),
    # The pre handoff: the standard route's values, then the handoff operands.
    "fwd_prepare_handoff": (
        "fn",
        "x",
        "x_flat",
        "scale",
        "base",
        "stats",
        "rms_eps",
        "hc_eps",
        "u",
        "pre_in",
        "pre_next",
    ),
    "bwd_finish_handoff": (
        "fn",
        "x",
        "du",
        "stats",
        "gstats",
        "scale",
        "base",
        "hc_eps",
        "x_flat",
        "dy_flat",
        "dx_flat",
        "dfn",
        "dscale",
        "dbase",
        "accumulate",
        "pre_in",
        "dpre_next",
        "dpre_in",
    ),
    "collapse_forward": ("x", "pre_in", "u"),
    "collapse_backward": ("x", "pre_in", "du", "dx", "dpre_in"),
}
# Logical workspace values -> region names.  The forward fn split writes the
# forward operand layout only; its transposed-layout pointer is bound to the
# same region, exactly like the source launcher binds an unused parameter.  The
# backward's atomic fn split takes the transposed layout only; the column slab
# stores its per-chunk ``dfn`` sums in ``dfn_slot`` and counts the grid's
# finished units and CTAs in ``slab_cnt``.
_FWD_WORKSPACE_VALUES = (
    ("fnp", "fnp"),
    ("fnTp", "fnp"),
    ("spart", "spart"),
    ("qpart", "qpart"),
    ("ticket", "ticket"),
)
_BWD_WORKSPACE_VALUES = (
    ("fnTp", "fnTp"),
    ("dzp", "dzp"),
    ("tokp", "tokp"),
    ("param_partial", "param_partial"),
    ("dzp_rows", "dzp_rows"),
    ("ticket", "ticket"),
    ("dfn_slot", "dfn_slot"),
    ("slab_cnt", "slab_cnt"),
)
_WORKSPACE_VALUES: dict[str, tuple[tuple[str, str], ...]] = {
    "fwd_prepare": _FWD_WORKSPACE_VALUES,
    "fwd_recompute": _FWD_WORKSPACE_VALUES,
    "fwd_finish": (),
    "bwd_prepare": (),
    "bwd_finish": _BWD_WORKSPACE_VALUES,
    "fwd_prepare_handoff": _FWD_WORKSPACE_VALUES,
    "bwd_finish_handoff": _BWD_WORKSPACE_VALUES,
    "collapse_forward": (),
    "collapse_backward": (),
}
# The fast path runs the Sinkhorn count of record only, so the host plan's ``prow`` / ``param_rows`` (the token
# owner's group rows inside ``BWD_ROW_BANDS``) hold for every fast launch.
_BWD_HOST_VALUES = (
    "t_pad",
    "rows_per_cta",
    "parts",
    "rows_per_chunk",
    "chunks",
    "lpt_parts",
    "lpt_rows_per_chunk",
    "lpt_big_parts",
    "lpt_small_rows_per_chunk",
    "param_rows",
    "prow",
)
_HOST_VALUES: dict[str, tuple[str, ...]] = {
    "fwd_prepare": ("m_tiles", "log2_pk", "p_k", "t_pad"),
    "fwd_recompute": ("m_tiles", "log2_pk", "p_k", "t_pad"),
    "fwd_finish": (),
    "bwd_prepare": (),
    "bwd_finish": _BWD_HOST_VALUES,
    "fwd_prepare_handoff": ("m_tiles", "log2_pk", "p_k", "t_pad"),
    "bwd_finish_handoff": _BWD_HOST_VALUES,
    "collapse_forward": (),
    "collapse_backward": (),
}


class _FastRoute:
    """The launches of one route plan as ``(bound FFI entry, argument getter)``
    pairs over ``call values + workspace values + statics``: every argument
    position of every launch resolved once, so a call concatenates three tuples
    and launches."""

    __slots__ = ("launches", "statics", "workspace", "x_flat")

    def __init__(
        self,
        launches: tuple[tuple[Any, Any], ...],
        statics: tuple[Any, ...],
        workspace: Any,
        x_flat: tuple[int, int],
    ) -> None:
        self.launches, self.statics = launches, statics
        self.workspace, self.x_flat = workspace, x_flat


_FAST_ROUTES: dict[tuple[str, int, int, int, str], _FastRoute] = {}


def _fast_route(
    arch: str, kind: str, rows: int, hidden: int, device_index: int, path: str
) -> _FastRoute:
    key = (kind, hidden, rows, device_index, path)
    found = _FAST_ROUTES.get(key)
    if found is None:
        found = _build_fast_route(arch, kind, rows, hidden, device_index, path)
        if len(_FAST_ROUTES) >= _CACHE_ENTRIES:
            _FAST_ROUTES.clear()
        _FAST_ROUTES[key] = found
    return found


def _build_fast_route(
    arch: str, kind: str, rows: int, hidden: int, device_index: int, path: str
) -> _FastRoute:
    plan = _route_plan(arch, kind, rows, hidden, device_index, path)
    workspace_values = _WORKSPACE_VALUES[kind]
    names = _CALL_VALUES[kind] + tuple(name for name, _ in workspace_values)
    position_of = {name: position for position, name in enumerate(names)}
    host: dict[str, Any] = {"rows": rows}
    if _HOST_VALUES[kind]:
        sm = _loader.sm_count(device_index)
        host_plan = _loader.host_plan(rows, sm, arch, hidden=hidden)
        if kind in ("bwd_finish", "bwd_finish_handoff"):
            # the column slab's lpt chunks: slab_plan, read when the plan is built (not part of the cached host plan)
            host_plan = {
                **host_plan,
                **_loader.slab_plan(rows, sm, arch, hidden=hidden),
            }
        for name in _HOST_VALUES[kind]:
            host[name] = host_plan[name]
    statics: list[Any] = []
    launches = []
    for launch in plan:
        logical_at = dict(launch.slots)
        sources = []
        for position, value in enumerate(launch.template):
            logical_name = logical_at.get(position)
            if logical_name is None:
                sources.append(len(names) + len(statics))
                statics.append(value)
            elif logical_name in position_of:
                sources.append(position_of[logical_name])
            elif logical_name in host:
                sources.append(len(names) + len(statics))
                statics.append(host[logical_name])
            else:
                raise ValueError(
                    f"the {cake_mhc_nondet_train_route_key(hidden, kind)!r} route needs the logical value {logical_name!r}"
                )
        launches.append((launch.run, _tuple_getter(sources)))
    workspace_getter = (
        _tuple_getter([region for _, region in workspace_values])
        if workspace_values
        else None
    )
    return _FastRoute(
        tuple(launches), tuple(statics), workspace_getter, (rows, STREAMS * hidden)
    )


def _tuple_getter(keys: list[Any]) -> Any:
    """``values -> tuple(values[key] for key in keys)`` (``keys`` non-empty)."""

    if len(keys) > 1:
        return operator.itemgetter(*keys)
    key = keys[0]
    return lambda values: (values[key],)


def _launch_route(
    route: _FastRoute, values: tuple[Any, ...], device_index: int
) -> None:
    """Launch ``route`` with its call values (workspace values and statics appended)
    on the current torch stream of ``device_index``."""

    values += route.statics
    with _tvm_ffi().use_raw_stream(
        _tvm_device(device_index), _current_stream(device_index)
    ):
        for run, arguments in route.launches:
            run(*arguments(values))


def _run_route(
    arch: str,
    kind: str,
    values: dict[str, Any],
    *,
    rows: int,
    hidden: int,
    device: torch.device,
    path: str,
    sinkhorn_iters: int = SINKHORN_ITERS,
    accumulation: str = ACCUMULATION,
) -> None:
    device_index = _device_index(device)
    plan = _route_plan(
        arch, kind, rows, hidden, device_index, path, sinkhorn_iters, accumulation
    )
    stream = _current_stream(device_index)
    with _tvm_ffi().use_raw_stream(_tvm_device(device_index), stream):
        for launch in plan:
            args = launch.template.copy()
            try:
                for position, logical_name in launch.slots:
                    args[position] = values[logical_name]
            except KeyError as missing:
                raise ValueError(
                    f"the {cake_mhc_nondet_train_route_key(hidden, kind)!r} route needs the logical value {missing.args[0]!r}"
                ) from None
            launch.run(*args)


# ----------------------------------------------------------------------------
# Entry points
#
# Each entry point first tries a fast path: when every input passes the checks
# of the full path (inlined boolean forms of the same checks, on a device and
# route the full path has already resolved), it allocates the outputs, fetches
# the workspace views and launches the route's cached fast plan.  Any input the
# fast checks do not accept takes the full path, which validates in its own
# order and raises the error of record.  Both paths launch the same modules
# with the same arguments.

# ``fn`` / ``dfn`` are ``[24, 4 H]`` at each hidden size ``H``.
_FN_SHAPES: dict[int, tuple[int, int]] = {
    hidden: (MAPPING_ROWS, STREAMS * hidden) for hidden in HIDDEN_SIZES
}
_SCALE_SHAPE = (3,)
_BASE_SHAPE = (MAPPING_ROWS,)


def _empty(
    shape: tuple[int, ...], dtype: torch.dtype, like: torch.Tensor
) -> torch.Tensor:
    """A fresh contiguous buffer on ``like``'s device (what :func:`_output` allocates)."""

    return torch.empty(shape, dtype=dtype, device=like.device)


def _fast_forward(
    X: Any,
    fn: Any,
    scale: Any,
    base: Any,
    rms_eps: Any,
    hc_eps: Any,
    sinkhorn_iters: Any,
    kind: str,
) -> Optional[tuple[int, int, int, str]]:
    """``(rows, hidden, device index, architecture)`` when the forward inputs pass
    every check of the full path, else ``None``."""

    if not _fast_constants(rms_eps, hc_eps, sinkhorn_iters):
        return None
    found = _fast_rows(X, kind)
    if (
        found is not None
        and _fast_tensor(fn, _FN_SHAPES[found[1]], _F32, found[2])
        and _fast_tensor(scale, _SCALE_SHAPE, _F32, found[2])
        and _fast_tensor(base, _BASE_SHAPE, _F32, found[2])
    ):
        return found
    return None


def _fast_workspace(
    workspace: Optional[torch.Tensor],
    kind: str,
    rows: int,
    hidden: int,
    index: int,
    X: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if workspace is None:
        views = _DEFAULT_VIEWS.get((index, kind, hidden, rows))
        if views is not None:
            return views
    return _workspace_for(workspace, kind, rows, hidden, X.device)


def _forward_values(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    ws: dict[str, torch.Tensor],
    *,
    rows: int,
    hidden: int,
    device: torch.device,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> dict[str, Any]:
    plan = _loader.host_plan(
        rows,
        _loader.sm_count(_device_index(device)),
        _device_arch(device),
        hidden=hidden,
    )
    # The fn split writes the forward operand layout only (``want_t = 0`` is the
    # record's own scalar); its transposed-layout pointer is bound to the same
    # region, exactly like the source launcher binds an unused parameter.
    return {
        "fn": fn,
        "fnp": ws["fnp"],
        "fnTp": ws["fnp"],
        "x": X,
        "x_flat": X.view(rows, STREAMS * hidden),
        "spart": ws["spart"],
        "qpart": ws["qpart"],
        "ticket": ws["ticket"],
        "scale": scale,
        "base": base,
        "stats": stats,
        "rows": rows,
        "m_tiles": plan["m_tiles"],
        "log2_pk": plan["log2_pk"],
        "p_k": plan["p_k"],
        "t_pad": plan["t_pad"],
        "rms_eps": float(rms_eps),
        "hc_eps": float(hc_eps),
        "sinkhorn_iters": int(sinkhorn_iters),
    }


def cake_mhc_nondet_train_forward_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    out: Optional[torch.Tensor] = None,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``U`` and the per-token statistics.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (``T`` is a runtime
        scalar; ``T == 0`` returns empty outputs without launching).  The
        hidden size ``H`` of the call is ``X.shape[2]``: 4096, 5120 or 7168
        on SM100 / SM103 / SM107.
    fn : torch.Tensor
        FP32 ``[24, 4H]`` contiguous mapping matrix.
    scale : torch.Tensor
        FP32 ``[3]`` group scales of the pre / post / Sinkhorn logits.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets.
    rms_eps : float
        Normalization epsilon (default ``1e-5``): a launch argument of the exported
        kernels, so any finite positive value runs on the same compiled programs.
    hc_eps : float
        Coefficient epsilon (default ``1e-6``): a launch argument like ``rms_eps``.
    sinkhorn_iters : int
        Sinkhorn iteration count, ``1 .. SINKHORN_ITERS_MAX`` (64; default ``20``).
        The default runs the compiled programs, whose chain is unrolled for it;
        every other count runs the run-time-count programs of the coefficient and
        token-owner passes.  A larger count raises ``NotImplementedError``.
    accumulation : str
        ``"fp32"`` (default): every kernel keeps each sum in one plain FP32 word.
        ``"fp32x2"`` (opt-in): the passes of ``cake_jit.ACC_WORDS_ROLES`` (the
        mapping pass; in the backward also the token-owner pass and the column
        slab with its folds) run their two-word programs, which keep each sum
        as a compensated pair of FP32 words (the running sum and its rounding
        error) and store one FP32 word: a compensated FP32 sum, not FP64.  The
        workspaces and records are unchanged.  Hidden 4096 and
        ``sinkhorn_iters=20`` only (another hidden size or count raises
        ``NotImplementedError``).  One two-word program per pass serves every
        token count, in place of the token-count bands of the default.  A
        replay (``forward_stats``) and the backward of a forward take the
        forward's mode.
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_nondet_train_forward_workspace`
        (sized for this ``H``: ``hidden=X.shape[2]``); a per-device default
        buffer is used when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])`` and the
    FP32 ``[T, 64]`` per-token record ``r | z[24] | pre[4] | post[4] | C[16] | pad[15]``
    (256 bytes per token) that ``forward_finish``, ``backward_prepare`` and
    ``backward_finish`` consume.  No
    ``[T, 4H]`` or ``[T, 4, H]`` FP32 temporary is produced.

    The launches run on the current torch stream of the tensors' device.
    """

    fast = (
        _fast_forward(
            X, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters, "fwd_prepare"
        )
        if _one_word(accumulation)
        else None
    )
    if fast is not None:
        rows, hidden, index, arch = fast
        if (out is None or _fast_tensor(out, (rows, hidden), _BF16, index)) and (
            stats_out is None
            or _fast_tensor(stats_out, (rows, STATS_COLS), _F32, index)
        ):
            U = _empty((rows, hidden), _BF16, X) if out is None else out
            stats = (
                _empty((rows, STATS_COLS), _F32, X) if stats_out is None else stats_out
            )
            route = _fast_route(arch, "fwd_prepare", rows, hidden, index, _PATH_FORWARD)
            views = _fast_workspace(workspace, "fwd", rows, hidden, index, X)
            _launch_route(
                route,
                (fn, X, X.view(route.x_flat), scale, base, stats, rms_eps, hc_eps, U)
                + route.workspace(views),
                index,
            )
            return U, stats
    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, hidden, device = _rows_of(X)
    _loader.check_accumulation(accumulation, hidden, sinkhorn_iters)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    fn, scale, base = _parameters(fn, scale, base, device, hidden)
    _check_route(device, arch, "fwd_prepare", hidden)
    U = _output(out, "out", shape=(rows, hidden), device=device, dtype=_BF16)
    stats = _output(
        stats_out, "stats_out", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    if rows == 0:
        return U, stats
    ws = _workspace_for(workspace, "fwd", rows, hidden, device)
    values = _forward_values(
        X,
        fn,
        scale,
        base,
        stats,
        ws,
        rows=rows,
        hidden=hidden,
        device=device,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    values["u"] = U
    _run_route(
        arch,
        "fwd_prepare",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_FORWARD,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
    )
    return U, stats


def cake_mhc_nondet_train_forward_stats(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``"fp32"`` default,
        ``"fp32x2"`` opt-in): the mode of the forward whose record this replays.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_nondet_train_forward_workspace`).

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``: the record
    :func:`cake_mhc_nondet_train_forward_prepare` returns for the same inputs and mode on the
    same device, computed by the same programs (no bit promise): the route is
    the strict prefix of the forward (fn split, mapping pass, coefficients)
    under the same work partition, without the ``U`` write.

    The launches run on the current torch stream of the tensors' device.
    """

    fast = (
        _fast_forward(
            X, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters, "fwd_recompute"
        )
        if _one_word(accumulation)
        else None
    )
    if fast is not None:
        rows, hidden, index, arch = fast
        if stats_out is None or _fast_tensor(
            stats_out, (rows, STATS_COLS), _F32, index
        ):
            stats = (
                _empty((rows, STATS_COLS), _F32, X) if stats_out is None else stats_out
            )
            route = _fast_route(
                arch, "fwd_recompute", rows, hidden, index, _PATH_CHECKPOINT
            )
            views = _fast_workspace(workspace, "fwd", rows, hidden, index, X)
            _launch_route(
                route,
                (fn, X, X.view(route.x_flat), scale, base, stats, rms_eps, hc_eps)
                + route.workspace(views),
                index,
            )
            return stats
    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, hidden, device = _rows_of(X)
    _loader.check_accumulation(accumulation, hidden, sinkhorn_iters)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    fn, scale, base = _parameters(fn, scale, base, device, hidden)
    _check_route(device, arch, "fwd_recompute", hidden)
    stats = _output(
        stats_out, "stats_out", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    if rows == 0:
        return stats
    ws = _workspace_for(workspace, "fwd", rows, hidden, device)
    values = _forward_values(
        X,
        fn,
        scale,
        base,
        stats,
        ws,
        rows=rows,
        hidden=hidden,
        device=device,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    _run_route(
        arch,
        "fwd_recompute",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_CHECKPOINT,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
    )
    return stats


def cake_mhc_nondet_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (the ``forward_prepare``
        input; ``H = X.shape[2]``).
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record returned by :func:`cake_mhc_nondet_train_forward_prepare`
        (or :func:`cake_mhc_nondet_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``; the FP32 sum is rounded once.

    The launches run on the current torch stream of the tensors' device.
    """

    fast = _fast_rows(X, "fwd_finish")
    if fast is not None:
        rows, hidden, index, arch = fast
        if (
            _fast_tensor(B, (rows, hidden), _BF16, index)
            and _fast_tensor(stats, (rows, STATS_COLS), _F32, index)
            and (
                out is None or _fast_tensor(out, (rows, STREAMS, hidden), _BF16, index)
            )
        ):
            Y = _empty((rows, STREAMS, hidden), _BF16, X) if out is None else out
            _launch_route(
                _fast_route(arch, "fwd_finish", rows, hidden, index, _PATH_FORWARD),
                (X, B, stats, Y),
                index,
            )
            return Y
    rows, hidden, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    B = _validate_tensor(B, "B", shape=(rows, hidden), device=device, dtype=_BF16)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "fwd_finish", hidden)
    Y = _output(out, "out", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16)
    if rows == 0:
        return Y
    values = {"x": X, "b": B, "stats": stats, "y": Y, "rows": rows}
    _run_route(
        arch,
        "fwd_finish",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_FORWARD,
    )
    return Y


def cake_mhc_nondet_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics (before the branch backward).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    ``(dB, grad_stats)``: ``dB[t, d] = BF16(sum_j post[t, j] dY[t, j, d])`` (one
    rounding of the full FP32 sum) and the FP32 ``[T, 32]`` record
    ``dpost[4] | dC[16] | zeros[12]`` (``dpost[t, j] = sum_d dY B``,
    ``dC[t, i, j] = sum_d X[t, i, d] dY[t, j, d]``) that ``backward_finish`` consumes.

    The launches run on the current torch stream of the tensors' device.
    """

    return _backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out, path=_PATH_BACKWARD
    )


def _backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor],
    grad_stats_out: Optional[torch.Tensor],
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`cake_mhc_nondet_train_backward_prepare` along ``path`` (``_PATH_BACKWARD`` from
    saved statistics, ``_PATH_CHECKPOINT`` after the forward prefix was replayed: the
    autograd nodes know which): the path fixes the per-call PDL argument of the
    backward prepare's records (:func:`cake_jit.enable_pdl`)."""

    fast = _fast_rows(X, "bwd_prepare")
    if fast is not None:
        rows, hidden, index, arch = fast
        if (
            _fast_tensor(dY, (rows, STREAMS, hidden), _BF16, index)
            and _fast_tensor(B, (rows, hidden), _BF16, index)
            and _fast_tensor(stats, (rows, STATS_COLS), _F32, index)
            and (dB is None or _fast_tensor(dB, (rows, hidden), _BF16, index))
            and (
                grad_stats_out is None
                or _fast_tensor(grad_stats_out, (rows, GSTATS_COLS), _F32, index)
            )
        ):
            if dB is None:
                dB = _empty((rows, hidden), _BF16, X)
            gstats = (
                _empty((rows, GSTATS_COLS), _F32, X)
                if grad_stats_out is None
                else grad_stats_out
            )
            _launch_route(
                _fast_route(arch, "bwd_prepare", rows, hidden, index, path),
                (dY, B, X, stats, dB, gstats),
                index,
            )
            return dB, gstats
    rows, hidden, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    dY = _validate_tensor(
        dY, "dY", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    B = _validate_tensor(B, "B", shape=(rows, hidden), device=device, dtype=_BF16)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "bwd_prepare", hidden)
    dB = _output(dB, "dB", shape=(rows, hidden), device=device, dtype=_BF16)
    gstats = _output(
        grad_stats_out,
        "grad_stats_out",
        shape=(rows, GSTATS_COLS),
        device=device,
        dtype=_F32,
    )
    if rows == 0:
        return dB, gstats
    values = {
        "dy": dY,
        "b": B,
        "x": X,
        "stats": stats,
        "db": dB,
        "gstats": gstats,
        "rows": rows,
    }
    _run_route(
        arch, "bwd_prepare", values, rows=rows, hidden=hidden, device=device, path=path
    )
    return dB, gstats


def cake_mhc_nondet_train_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    accumulate: bool = False,
    workspace: Optional[torch.Tensor] = None,
    dX: Optional[torch.Tensor] = None,
    dfn: Optional[torch.Tensor] = None,
    dscale: Optional[torch.Tensor] = None,
    dbase: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients (after the branch backward produced ``dU``).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, H]`` branch output of the forward; accepted for the
        prepare / finish symmetry and validated when given, not read (its
        contribution is already in ``grad_stats``).
    fn, scale, base : torch.Tensor
        The parameters of the forward (FP32 ``[24, 4H]``, ``[3]``, ``[24]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved, or recomputed with
        :func:`cake_mhc_nondet_train_forward_stats`: the same programs, no bit promise).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record returned by :func:`cake_mhc_nondet_train_backward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``"fp32"`` default,
        ``"fp32x2"`` opt-in: the token-owner pass and the column slab with its
        parameter and ``dfn`` folds run their two-word programs, each fold a
        compensated two-word FP32 sum; pass the forward's mode).
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three; each fold adds the current value
        last (``dfn``: after its FP32 sum of the chunk slots; ``dscale`` /
        ``dbase``: after the fixed-order FP32 sums of the parameter rows).
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_nondet_train_backward_workspace`
        (sized for this ``H``: ``hidden=X.shape[2]``); a per-device default
        buffer is used when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dfn, dscale, dbase : torch.Tensor, optional
        Preallocated FP32 ``[24, 4H]``, ``[3]``, ``[24]`` contiguous outputs
        (required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dfn, dscale, dbase)``, with no run-to-run bit promise (the sums
    run in a fixed order today): the BF16 ``dX`` sums the mapping, collapse
    and residual-mixing contributions in FP32 and rounds once.  ``dscale`` / ``dbase`` fold the
    token-owner pass's parameter rows in a fixed order (per-CTA rows, or
    8-token group rows where the deterministic package runs them: hidden 4096
    above 14927 tokens on sm_100a and sm_103a; see :mod:`.cake_jit`), each sum
    one FP32 word.  ``dfn`` adds the per-chunk FP32 partials ``x_p`` of at
    most ``P <= 32`` token chunks (the deterministic package's chunks per
    token count and device) per element in ascending chunk order as one FP32
    sum from +0.0, the caller's ``dfn`` last under ``accumulate=True``: within
    ``gamma_n (|start| + sum |x_p|)`` of the exact sum (``n`` the additions,
    ``gamma_n = n u / (1 - n u)``, ``u = 2^-24``).
    With ``T == 0`` nothing is launched: ``dX`` is empty and the parameter
    gradients are zeroed (``accumulate=False``) or left untouched
    (``accumulate=True``).

    The launches run on the current torch stream of the tensors' device.
    """

    return _backward_finish(
        X,
        dY,
        dU,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
        path=_PATH_BACKWARD,
    )


def _backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    accumulation: str = "fp32",
    accumulate: bool,
    workspace: Optional[torch.Tensor],
    dX: Optional[torch.Tensor],
    dfn: Optional[torch.Tensor],
    dscale: Optional[torch.Tensor],
    dbase: Optional[torch.Tensor],
    path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`cake_mhc_nondet_train_backward_finish` along ``path`` (``_PATH_BACKWARD`` from
    saved statistics, ``_PATH_CHECKPOINT`` after the forward prefix was replayed: the
    autograd nodes know which): the path fixes the per-call PDL argument of the
    column-slab record (:func:`cake_jit.enable_pdl`)."""

    fast = (
        _fast_forward(X, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters, "bwd_finish")
        if _one_word(accumulation)
        else None
    )
    if fast is not None:
        rows, hidden, index, arch = fast
        fn_shape = _FN_SHAPES[hidden]
        if (
            _fast_tensor(dY, (rows, STREAMS, hidden), _BF16, index)
            and _fast_tensor(dU, (rows, hidden), _BF16, index)
            and (B is None or _fast_tensor(B, (rows, hidden), _BF16, index))
            and _fast_tensor(stats, (rows, STATS_COLS), _F32, index)
            and _fast_tensor(grad_stats, (rows, GSTATS_COLS), _F32, index)
            and not (accumulate and (dfn is None or dscale is None or dbase is None))
            and (dX is None or _fast_tensor(dX, (rows, STREAMS, hidden), _BF16, index))
            and (dfn is None or _fast_tensor(dfn, fn_shape, _F32, index))
            and (dscale is None or _fast_tensor(dscale, _SCALE_SHAPE, _F32, index))
            and (dbase is None or _fast_tensor(dbase, _BASE_SHAPE, _F32, index))
        ):
            if dX is None:
                dX = _empty((rows, STREAMS, hidden), _BF16, X)
            if dfn is None:
                dfn = _empty(fn_shape, _F32, X)
            if dscale is None:
                dscale = _empty(_SCALE_SHAPE, _F32, X)
            if dbase is None:
                dbase = _empty(_BASE_SHAPE, _F32, X)
            route = _fast_route(arch, "bwd_finish", rows, hidden, index, path)
            views = _fast_workspace(workspace, "bwd", rows, hidden, index, X)
            _launch_route(
                route,
                (
                    fn,
                    X,
                    dU,
                    stats,
                    grad_stats,
                    scale,
                    base,
                    hc_eps,
                    X.view(route.x_flat),
                    dY.view(route.x_flat),
                    dX.view(route.x_flat),
                    dfn,
                    dscale,
                    dbase,
                    1 if accumulate else 0,
                )
                + route.workspace(views),
                index,
            )
            return dX, dfn, dscale, dbase
    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, hidden, device = _rows_of(X)
    _loader.check_accumulation(accumulation, hidden, sinkhorn_iters)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    dY = _validate_tensor(
        dY, "dY", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    dU = _validate_tensor(dU, "dU", shape=(rows, hidden), device=device, dtype=_BF16)
    if B is not None:
        _validate_tensor(B, "B", shape=(rows, hidden), device=device, dtype=_BF16)
    fn, scale, base = _parameters(fn, scale, base, device, hidden)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    grad_stats = _validate_tensor(
        grad_stats, "grad_stats", shape=(rows, GSTATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "bwd_finish", hidden)
    if accumulate and (dfn is None or dscale is None or dbase is None):
        raise ValueError(
            "accumulate=True needs the dfn, dscale and dbase buffers to accumulate into"
        )
    dX = _output(dX, "dX", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16)
    dfn = _output(
        dfn, "dfn", shape=(MAPPING_ROWS, STREAMS * hidden), device=device, dtype=_F32
    )
    dscale = _output(dscale, "dscale", shape=(3,), device=device, dtype=_F32)
    dbase = _output(dbase, "dbase", shape=(MAPPING_ROWS,), device=device, dtype=_F32)
    if rows == 0:
        if not accumulate:
            dfn.zero_()
            dscale.zero_()
            dbase.zero_()
        return dX, dfn, dscale, dbase
    ws = _workspace_for(workspace, "bwd", rows, hidden, device)
    sm = _loader.sm_count(_device_index(device))
    plan = _loader.host_plan(rows, sm, arch, hidden=hidden)
    slab = _loader.slab_plan(rows, sm, arch, hidden=hidden)
    # The token owner writes the group rows inside BWD_ROW_BANDS at the count of
    # record; at a run-time Sinkhorn count its run-time-count record writes the
    # per-CTA rows, which the column slab then folds (prow 0, G6 rows).
    rows_mode = plan["group_rows"] and int(sinkhorn_iters) == SINKHORN_ITERS
    # The atomic fn split writes the transposed operand layout and, without
    # accumulate, zeroes dfn.  The column-slab pass (the dynr record over the
    # one-size chunks, or its slab-fastest _os copy where bwd_slab_order names
    # "slab"; no two-size lpt record ships)
    # describes the ``[T, 4, H]`` tensors as ``[T, 4H]``, reads the first
    # ``T`` rows of the padded ``dz'`` record, draws its work units from the
    # workspace's self-resetting ticket word, stores each unit's FP32
    # dfn sums in its chunk's slots, folds the slots into dfn across the grid once
    # every unit is done (the self-resetting slab_cnt words count the units and
    # the CTAs) and folds dscale / dbase from the token owner's rows.
    values: dict[str, Any] = {
        "fn": fn,
        "fnTp": ws["fnTp"],
        "x": X,
        "du": dU,
        "stats": stats,
        "gstats": grad_stats,
        "scale": scale,
        "base": base,
        "dzp": ws["dzp"],
        "tokp": ws["tokp"],
        "param_partial": ws["param_partial"],
        "rows": rows,
        "t_pad": plan["t_pad"],
        "rows_per_cta": plan["rows_per_cta"],
        "hc_eps": float(hc_eps),
        "sinkhorn_iters": int(sinkhorn_iters),
        "x_flat": X.view(rows, STREAMS * hidden),
        "dy_flat": dY.view(rows, STREAMS * hidden),
        "dzp_rows": ws["dzp_rows"],
        "dx_flat": dX.view(rows, STREAMS * hidden),
        "ticket": ws["ticket"],
        "parts": plan["parts"],
        "rows_per_chunk": plan["rows_per_chunk"],
        "lpt_parts": slab["lpt_parts"],
        "lpt_rows_per_chunk": slab["lpt_rows_per_chunk"],
        "lpt_big_parts": slab["lpt_big_parts"],
        "lpt_small_rows_per_chunk": slab["lpt_small_rows_per_chunk"],
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
        "chunks": plan["chunks"],
        "dfn_slot": ws["dfn_slot"],
        "slab_cnt": ws["slab_cnt"],
        "prow": 1 if rows_mode else 0,
        "param_rows": plan["groups"] if rows_mode else plan["chunks"],
        "accumulate": 1 if accumulate else 0,
    }
    _run_route(
        arch,
        "bwd_finish",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=path,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
    )
    return dX, dfn, dscale, dbase


# ----------------------------------------------------------------------------
# Pre handoff and standalone collapse
#
# A sublayer of the pre handoff collapses its streams with the ``pre`` of the
# previous sublayer (``pre_in``, FP32 ``[T, 4]``) instead of its own, and hands
# its own ``pre`` (the coefficient pass's, ``stats`` columns 25..28) on as
# ``pre_next``; its backward takes the gradient of ``pre_next`` from its
# consumer (``dpre_next``) for the coefficient VJP and returns the gradient of
# ``pre_in`` (``dpre_in``).  The routes are the standard routes with the
# collapse and the token-owner forms of the handoff (``cake_jit.HANDOFF_KINDS``);
# ``fwd_recompute``, ``fwd_finish`` and ``backward_prepare`` serve both
# contracts.  The standalone collapse runs ``(X, pre_in) -> U`` and its VJP
# alone (the final collapse before the model's head).


def _require_pre(
    tensor: Any, name: str, rows: int, device: torch.device
) -> torch.Tensor:
    return _validate_tensor(
        tensor, name, shape=(rows, STREAMS), device=device, dtype=_F32
    )


def cake_mhc_nondet_train_handoff_forward_prepare(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    out: torch.Tensor | None = None,
    pre_next_out: torch.Tensor | None = None,
    stats_out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Forward mapping and coefficients of ``X``, collapse with the incoming ``pre_in``: ``(U, pre_next, stats)``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (``T == 0`` returns empty
        outputs without launching; ``H`` one of :func:`cake_jit.handoff_hidden_sizes`
        of the device's architecture, else ``ValueError``).
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` contiguous ``pre`` of the previous sublayer (its
        ``pre_next``; a one-hot ``[1, 0, 0, 0]`` row for the first sublayer).
    fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters, accumulation : float, float, int, str
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``"fp32x2"`` serves
        hidden 4096 only, so it raises ``NotImplementedError`` here).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.
    pre_next_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 4]`` contiguous ``pre_next``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_nondet_train_forward_workspace`,
        ``hidden=X.shape[2]``); a per-device default when omitted.

    Returns
    -------
    ``(U, pre_next, stats)``: BF16 ``U[t, d] = BF16(sum_i pre_in[t, i] X[t, i, d])``
    (one FP32 sum, one rounding), FP32 ``pre_next [T, 4]`` (this sublayer's
    ``pre = sigmoid(.) + hc_eps``, the values of ``stats`` columns 25..28, written
    by the collapse) and the ``[T, 64]`` record of
    :func:`cake_mhc_nondet_train_forward_prepare` (it does not depend on
    ``pre_in``: ``forward_stats`` replays it, ``forward_finish`` and
    ``backward_prepare`` consume it).

    The launches run on the current torch stream of the tensors' device.
    """

    fast = (
        _fast_forward(
            X, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters, "fwd_prepare_handoff"
        )
        if _one_word(accumulation)
        else None
    )
    if fast is not None:
        rows, hidden, index, arch = fast
        pre_shape = (rows, STREAMS)
        if (
            _fast_tensor(pre_in, pre_shape, _F32, index)
            and (out is None or _fast_tensor(out, (rows, hidden), _BF16, index))
            and (
                pre_next_out is None
                or _fast_tensor(pre_next_out, pre_shape, _F32, index)
            )
            and (
                stats_out is None
                or _fast_tensor(stats_out, (rows, STATS_COLS), _F32, index)
            )
        ):
            U = _empty((rows, hidden), _BF16, X) if out is None else out
            pre_next = (
                _empty(pre_shape, _F32, X) if pre_next_out is None else pre_next_out
            )
            stats = (
                _empty((rows, STATS_COLS), _F32, X) if stats_out is None else stats_out
            )
            route = _fast_route(
                arch, "fwd_prepare_handoff", rows, hidden, index, _PATH_FORWARD
            )
            views = _fast_workspace(workspace, "fwd", rows, hidden, index, X)
            _launch_route(
                route,
                (
                    fn,
                    X,
                    X.view(route.x_flat),
                    scale,
                    base,
                    stats,
                    rms_eps,
                    hc_eps,
                    U,
                    pre_in,
                    pre_next,
                )
                + route.workspace(views),
                index,
            )
            return U, pre_next, stats
    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, hidden, device = _rows_of(X)
    _loader.check_accumulation(accumulation, hidden, sinkhorn_iters)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    pre_in = _require_pre(pre_in, "pre_in", rows, device)
    fn, scale, base = _parameters(fn, scale, base, device, hidden)
    _check_route(device, arch, "fwd_prepare_handoff", hidden)
    U = _output(out, "out", shape=(rows, hidden), device=device, dtype=_BF16)
    pre_next = _output(
        pre_next_out, "pre_next_out", shape=(rows, STREAMS), device=device, dtype=_F32
    )
    stats = _output(
        stats_out, "stats_out", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    if rows == 0:
        return U, pre_next, stats
    ws = _workspace_for(workspace, "fwd", rows, hidden, device)
    values = _forward_values(
        X,
        fn,
        scale,
        base,
        stats,
        ws,
        rows=rows,
        hidden=hidden,
        device=device,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    values.update(u=U, pre_in=pre_in, pre_next=pre_next)
    _run_route(
        arch,
        "fwd_prepare_handoff",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_FORWARD,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
    )
    return U, pre_next, stats


def cake_mhc_nondet_train_handoff_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    pre_in: torch.Tensor,
    dpre_next: torch.Tensor,
    B: torch.Tensor | None,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    accumulate: bool = False,
    workspace: torch.Tensor | None = None,
    dX: torch.Tensor | None = None,
    dpre_in: torch.Tensor | None = None,
    dfn: torch.Tensor | None = None,
    dscale: torch.Tensor | None = None,
    dbase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Backward of one pre-handoff sublayer: ``(dX, dpre_in, dfn, dscale, dbase)``.

    Parameters
    ----------
    X, dY, dU, B, fn, scale, base, stats, grad_stats : torch.Tensor
        As in :func:`cake_mhc_nondet_train_backward_finish` (``stats`` of
        :func:`cake_mhc_nondet_train_handoff_forward_prepare` or its replay
        ``forward_stats``; ``grad_stats`` of ``backward_prepare``).
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` contiguous ``pre_in`` of the forward.
    dpre_next : torch.Tensor
        FP32 ``[T, 4]`` contiguous gradient of ``pre_next`` from its consumer
        (zeros when ``pre_next`` reached no loss).
    rms_eps, hc_eps, sinkhorn_iters, accumulation, accumulate, workspace : as in
        :func:`cake_mhc_nondet_train_backward_finish`.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dpre_in : torch.Tensor, optional
        Preallocated FP32 ``[T, 4]`` contiguous output.
    dfn, dscale, dbase : torch.Tensor, optional
        As in :func:`cake_mhc_nondet_train_backward_finish`.

    Returns
    -------
    ``(dX, dpre_in, dfn, dscale, dbase)``, with no run-to-run bit promise.
    ``dX = BF16(dX_coef + pre_in_i dU + sum_j C_ij dY_j)``: one FP32 sum, one
    rounding; the coefficient VJP ``dX_coef`` and the parameter gradients take
    ``dpre_next`` (with ``dpost`` and ``dC`` of ``grad_stats``) and do not depend
    on ``dU``.  ``dpre_in[t, i] = sum_d dU[t, d] X[t, i, d]`` is one FP32 sum per
    element in a fixed order (no FP64, no atomics); it is written whether or not
    the caller needs it.  The parameter gradients follow
    :func:`cake_mhc_nondet_train_backward_finish` (``accumulate`` applies to them
    alone).  With ``T == 0`` nothing is launched: ``dX`` and ``dpre_in`` are
    empty and the parameter gradients are zeroed (``accumulate=False``) or left
    untouched.

    The launches run on the current torch stream of the tensors' device.
    """

    return _handoff_backward_finish(
        X,
        dY,
        dU,
        pre_in,
        dpre_next,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dpre_in=dpre_in,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
        path=_PATH_BACKWARD,
    )


def _handoff_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    pre_in: torch.Tensor,
    dpre_next: torch.Tensor,
    B: torch.Tensor | None,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    accumulation: str = "fp32",
    accumulate: bool,
    workspace: torch.Tensor | None,
    dX: torch.Tensor | None,
    dpre_in: torch.Tensor | None,
    dfn: torch.Tensor | None,
    dscale: torch.Tensor | None,
    dbase: torch.Tensor | None,
    path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`cake_mhc_nondet_train_handoff_backward_finish` along ``path`` (as
    :func:`_backward_finish`)."""

    fast = (
        _fast_forward(
            X, fn, scale, base, rms_eps, hc_eps, sinkhorn_iters, "bwd_finish_handoff"
        )
        if _one_word(accumulation)
        else None
    )
    if fast is not None:
        rows, hidden, index, arch = fast
        fn_shape = _FN_SHAPES[hidden]
        pre_shape = (rows, STREAMS)
        if (
            _fast_tensor(dY, (rows, STREAMS, hidden), _BF16, index)
            and _fast_tensor(dU, (rows, hidden), _BF16, index)
            and _fast_tensor(pre_in, pre_shape, _F32, index)
            and _fast_tensor(dpre_next, pre_shape, _F32, index)
            and (B is None or _fast_tensor(B, (rows, hidden), _BF16, index))
            and _fast_tensor(stats, (rows, STATS_COLS), _F32, index)
            and _fast_tensor(grad_stats, (rows, GSTATS_COLS), _F32, index)
            and not (accumulate and (dfn is None or dscale is None or dbase is None))
            and (dX is None or _fast_tensor(dX, (rows, STREAMS, hidden), _BF16, index))
            and (dpre_in is None or _fast_tensor(dpre_in, pre_shape, _F32, index))
            and (dfn is None or _fast_tensor(dfn, fn_shape, _F32, index))
            and (dscale is None or _fast_tensor(dscale, _SCALE_SHAPE, _F32, index))
            and (dbase is None or _fast_tensor(dbase, _BASE_SHAPE, _F32, index))
        ):
            if dX is None:
                dX = _empty((rows, STREAMS, hidden), _BF16, X)
            if dpre_in is None:
                dpre_in = _empty(pre_shape, _F32, X)
            if dfn is None:
                dfn = _empty(fn_shape, _F32, X)
            if dscale is None:
                dscale = _empty(_SCALE_SHAPE, _F32, X)
            if dbase is None:
                dbase = _empty(_BASE_SHAPE, _F32, X)
            route = _fast_route(arch, "bwd_finish_handoff", rows, hidden, index, path)
            views = _fast_workspace(workspace, "bwd", rows, hidden, index, X)
            _launch_route(
                route,
                (
                    fn,
                    X,
                    dU,
                    stats,
                    grad_stats,
                    scale,
                    base,
                    hc_eps,
                    X.view(route.x_flat),
                    dY.view(route.x_flat),
                    dX.view(route.x_flat),
                    dfn,
                    dscale,
                    dbase,
                    1 if accumulate else 0,
                    pre_in,
                    dpre_next,
                    dpre_in,
                )
                + route.workspace(views),
                index,
            )
            return dX, dpre_in, dfn, dscale, dbase
    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, hidden, device = _rows_of(X)
    _loader.check_accumulation(accumulation, hidden, sinkhorn_iters)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    dY = _validate_tensor(
        dY, "dY", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    dU = _validate_tensor(dU, "dU", shape=(rows, hidden), device=device, dtype=_BF16)
    pre_in = _require_pre(pre_in, "pre_in", rows, device)
    dpre_next = _require_pre(dpre_next, "dpre_next", rows, device)
    if B is not None:
        _validate_tensor(B, "B", shape=(rows, hidden), device=device, dtype=_BF16)
    fn, scale, base = _parameters(fn, scale, base, device, hidden)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    grad_stats = _validate_tensor(
        grad_stats, "grad_stats", shape=(rows, GSTATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "bwd_finish_handoff", hidden)
    if accumulate and (dfn is None or dscale is None or dbase is None):
        raise ValueError(
            "accumulate=True needs the dfn, dscale and dbase buffers to accumulate into"
        )
    dX = _output(dX, "dX", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16)
    dpre_in = _output(
        dpre_in, "dpre_in", shape=(rows, STREAMS), device=device, dtype=_F32
    )
    dfn = _output(
        dfn, "dfn", shape=(MAPPING_ROWS, STREAMS * hidden), device=device, dtype=_F32
    )
    dscale = _output(dscale, "dscale", shape=(3,), device=device, dtype=_F32)
    dbase = _output(dbase, "dbase", shape=(MAPPING_ROWS,), device=device, dtype=_F32)
    if rows == 0:
        if not accumulate:
            dfn.zero_()
            dscale.zero_()
            dbase.zero_()
        return dX, dpre_in, dfn, dscale, dbase
    ws = _workspace_for(workspace, "bwd", rows, hidden, device)
    sm = _loader.sm_count(_device_index(device))
    plan = _loader.host_plan(rows, sm, arch, hidden=hidden)
    slab = _loader.slab_plan(rows, sm, arch, hidden=hidden)
    # As in _backward_finish: the token owner's group rows at the count of record only.
    rows_mode = plan["group_rows"] and int(sinkhorn_iters) == SINKHORN_ITERS
    values: dict[str, Any] = {
        "fn": fn,
        "fnTp": ws["fnTp"],
        "x": X,
        "du": dU,
        "stats": stats,
        "gstats": grad_stats,
        "scale": scale,
        "base": base,
        "dzp": ws["dzp"],
        "tokp": ws["tokp"],
        "param_partial": ws["param_partial"],
        "rows": rows,
        "t_pad": plan["t_pad"],
        "rows_per_cta": plan["rows_per_cta"],
        "hc_eps": float(hc_eps),
        "sinkhorn_iters": int(sinkhorn_iters),
        "x_flat": X.view(rows, STREAMS * hidden),
        "dy_flat": dY.view(rows, STREAMS * hidden),
        "dzp_rows": ws["dzp_rows"],
        "dx_flat": dX.view(rows, STREAMS * hidden),
        "ticket": ws["ticket"],
        "parts": plan["parts"],
        "rows_per_chunk": plan["rows_per_chunk"],
        "lpt_parts": slab["lpt_parts"],
        "lpt_rows_per_chunk": slab["lpt_rows_per_chunk"],
        "lpt_big_parts": slab["lpt_big_parts"],
        "lpt_small_rows_per_chunk": slab["lpt_small_rows_per_chunk"],
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
        "chunks": plan["chunks"],
        "dfn_slot": ws["dfn_slot"],
        "slab_cnt": ws["slab_cnt"],
        "prow": 1 if rows_mode else 0,
        "param_rows": plan["groups"] if rows_mode else plan["chunks"],
        "accumulate": 1 if accumulate else 0,
        "pre_in": pre_in,
        "dpre_next": dpre_next,
        "dpre_in": dpre_in,
    }
    _run_route(
        arch,
        "bwd_finish_handoff",
        values,
        rows=rows,
        hidden=hidden,
        device=device,
        path=path,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
    )
    return dX, dpre_in, dfn, dscale, dbase


def cake_mhc_nondet_train_collapse_forward(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Standalone collapse: ``U[t, d] = BF16(sum_i pre_in[t, i] X[t, i, d])``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (``H`` one of
        :func:`cake_jit.handoff_hidden_sizes` of the device's architecture).
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` contiguous weights (the last sublayer's ``pre_next``).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.

    Returns
    -------
    BF16 ``U`` of shape ``[T, H]``: the collapse of
    :func:`cake_mhc_nondet_train_handoff_forward_prepare` (one FP32 sum, one
    rounding), without parameters or workspace.  ``T == 0`` launches nothing.
    """

    fast = _fast_rows(X, "collapse_forward")
    if fast is not None:
        rows, hidden, index, arch = fast
        if _fast_tensor(pre_in, (rows, STREAMS), _F32, index) and (
            out is None or _fast_tensor(out, (rows, hidden), _BF16, index)
        ):
            U = _empty((rows, hidden), _BF16, X) if out is None else out
            _launch_route(
                _fast_route(
                    arch, "collapse_forward", rows, hidden, index, _PATH_FORWARD
                ),
                (X, pre_in, U),
                index,
            )
            return U
    rows, hidden, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    pre_in = _require_pre(pre_in, "pre_in", rows, device)
    _check_route(device, arch, "collapse_forward", hidden)
    U = _output(out, "out", shape=(rows, hidden), device=device, dtype=_BF16)
    if rows == 0:
        return U
    _run_route(
        arch,
        "collapse_forward",
        {"x": X, "pre_in": pre_in, "u": U, "rows": rows},
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_FORWARD,
    )
    return U


def cake_mhc_nondet_train_collapse_backward(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    dU: torch.Tensor,
    *,
    dX: torch.Tensor | None = None,
    dpre_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""VJP of the standalone collapse: ``(dX, dpre_in)``.

    Parameters
    ----------
    X, pre_in : torch.Tensor
        The inputs of :func:`cake_mhc_nondet_train_collapse_forward`.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U``.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dpre_in : torch.Tensor, optional
        Preallocated FP32 ``[T, 4]`` contiguous output.

    Returns
    -------
    ``(dX, dpre_in)``: ``dX[t, i, d] = BF16(pre_in[t, i] dU[t, d])`` (one FP32
    product, one rounding) and ``dpre_in[t, i] = sum_d dU[t, d] X[t, i, d]``,
    one FP32 sum per element in a fixed order (no FP64, no atomics).  ``T == 0``
    launches nothing.
    """

    fast = _fast_rows(X, "collapse_backward")
    if fast is not None:
        rows, hidden, index, arch = fast
        pre_shape = (rows, STREAMS)
        if (
            _fast_tensor(pre_in, pre_shape, _F32, index)
            and _fast_tensor(dU, (rows, hidden), _BF16, index)
            and (dX is None or _fast_tensor(dX, (rows, STREAMS, hidden), _BF16, index))
            and (dpre_in is None or _fast_tensor(dpre_in, pre_shape, _F32, index))
        ):
            if dX is None:
                dX = _empty((rows, STREAMS, hidden), _BF16, X)
            if dpre_in is None:
                dpre_in = _empty(pre_shape, _F32, X)
            _launch_route(
                _fast_route(
                    arch, "collapse_backward", rows, hidden, index, _PATH_BACKWARD
                ),
                (X, pre_in, dU, dX, dpre_in),
                index,
            )
            return dX, dpre_in
    rows, hidden, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16
    )
    pre_in = _require_pre(pre_in, "pre_in", rows, device)
    dU = _validate_tensor(dU, "dU", shape=(rows, hidden), device=device, dtype=_BF16)
    _check_route(device, arch, "collapse_backward", hidden)
    dX = _output(dX, "dX", shape=(rows, STREAMS, hidden), device=device, dtype=_BF16)
    dpre_in = _output(
        dpre_in, "dpre_in", shape=(rows, STREAMS), device=device, dtype=_F32
    )
    if rows == 0:
        return dX, dpre_in
    _run_route(
        arch,
        "collapse_backward",
        {
            "x": X,
            "pre_in": pre_in,
            "du": dU,
            "dx": dX,
            "dpre_in": dpre_in,
            "rows": rows,
        },
        rows=rows,
        hidden=hidden,
        device=device,
        path=_PATH_BACKWARD,
    )
    return dX, dpre_in


# ----------------------------------------------------------------------------
# Autograd


class _MHCTrainState:
    """What the prepare and finish autograd nodes of one layer application share.

    The parameters (detached), the constants, the workspaces and, transiently,
    ``stats``: between the prepare and the finish forward.  Nothing the forward
    produced stays on the state once the finish forward has run: ``stats``
    travels through ``save_for_backward`` like the inputs, so an activation
    checkpoint (non-reentrant ``torch.utils.checkpoint``) keeps no tensor of the
    layer alive.

    The finish node's backward hands ``dY``, the gradient statistics and, under
    ``save_stats=False``, the statistics it replayed to the prepare node's
    backward on autograd edges (:class:`_MHCTrainHandle`), so the autograd engine
    releases them once the prepare node's backward has run, however long the
    graph stays alive.  ``dY``, ``grad_stats`` and ``stats`` carry them on the
    state only from a finish node without those edges whose prepare node is
    ``recorded`` (a finish forward rerun by a reentrant checkpoint region that
    starts after the prepare); the prepare node's backward empties them.
    """

    __slots__ = (
        "fn",
        "scale",
        "base",
        "stats",
        "grad_stats",
        "dY",
        "recorded",
        "save_stats",
        "rms_eps",
        "hc_eps",
        "sinkhorn_iters",
        "accumulation",
        "workspace",
        "backward_workspace",
    )

    def __init__(
        self,
        *,
        save_stats: bool = True,
        rms_eps: float = 1e-5,
        hc_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        accumulation: str = "fp32",
        workspace: Optional[torch.Tensor] = None,
        backward_workspace: Optional[torch.Tensor] = None,
    ) -> None:
        self.fn = self.scale = self.base = None
        self.stats = self.grad_stats = self.dY = None
        self.recorded = False
        self.save_stats = bool(save_stats)
        self.rms_eps, self.hc_eps, self.sinkhorn_iters = rms_eps, hc_eps, sinkhorn_iters
        self.accumulation = accumulation
        self.workspace, self.backward_workspace = workspace, backward_workspace

    def replay(self, X: torch.Tensor) -> torch.Tensor:
        """The replay of the forward prefix's ``stats`` (the same programs)."""

        return cake_mhc_nondet_train_forward_stats(
            X,
            self.fn,
            self.scale,
            self.base,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            accumulation=self.accumulation,
            workspace=self.workspace,
        )

    def take(
        self,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``(dY, grad_stats, stats)`` as a finish node without edges left them; the state lets go of them."""

        held = self.dY, self.grad_stats, self.stats
        self.dY = self.grad_stats = self.stats = None
        return held


class _MHCTrainHandle:
    """What :func:`cake_mhc_nondet_train_prepare` returns beside ``U``: the shared state and the prepare node's edges.

    The edges are extra outputs of the prepare node: one-element, zero-stride
    tensors shaped like what the finish node's backward hands the prepare node's
    backward (``dY``, the gradient statistics and, under ``save_stats=False``,
    the replayed statistics).  The finish forward takes them off the handle as
    inputs, so that hand-over runs on autograd edges.  The state never holds
    them: the prepare node's context holds the state, and the edges hold the
    prepare node.
    """

    __slots__ = ("state", "edges")

    def __init__(self, state: _MHCTrainState, edges: tuple[torch.Tensor, ...]) -> None:
        self.state, self.edges = state, edges


# The edge stand-ins per (device, token count, hidden size, save_stats).  No
# kernel reads or writes an edge, so the edges of every prepare call alias these
# one-element buffers; each call takes its own tensors (``detach``), so the
# edges of two connections are distinct autograd outputs, and allocates no
# memory.
_EDGES: dict[tuple[torch.device, int, int, bool], tuple[torch.Tensor, ...]] = {}


def _edges(
    rows: int, hidden: int, device: torch.device, save_stats: bool
) -> tuple[torch.Tensor, ...]:
    """One-element, zero-stride stand-ins of BF16 ``dY`` ``[T, 4, H]``, the FP32
    gradient statistics ``[T, 32]`` and, under ``save_stats=False``, the FP32
    statistics ``[T, 64]``: the prepare node's edge outputs."""

    key = (device, rows, hidden, save_stats)
    edges = _EDGES.get(key)
    if edges is None:
        shapes = [((rows, STREAMS, hidden), _BF16), ((rows, GSTATS_COLS), _F32)]
        if not save_stats:
            shapes.append(((rows, STATS_COLS), _F32))
        edges = tuple(
            torch.empty_strided(shape, (0,) * len(shape), dtype=dtype, device=device)
            for shape, dtype in shapes
        )
        if len(_EDGES) >= _CACHE_ENTRIES:
            _EDGES.clear()
        _EDGES[key] = edges
    if save_stats:
        return edges[0].detach(), edges[1].detach()
    return edges[0].detach(), edges[1].detach(), edges[2].detach()


def _as_bf16(
    grad: Optional[torch.Tensor], *, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    if grad is None:
        return torch.zeros(shape, dtype=_BF16, device=device)
    if grad.dtype is _BF16 and grad.is_contiguous():
        return grad  # what ``contiguous()`` and ``to()`` return for it
    return grad.contiguous().to(_BF16)


class CakeMHCNondetPrepareFunction(torch.autograd.Function):
    """Autograd node of ``forward_prepare``: ``forward(X, fn, scale, base, state) -> (U, *edges)``.

    Its backward is ``backward_finish``: it receives
    ``dU`` from the external branch and, as the gradients of the edges (outputs
    the finish node takes as inputs), ``dY`` and the gradient statistics from the
    finish node's backward; it returns the single-cast ``dX`` (mapping, collapse
    and residual-mixing contributions summed in FP32) and the FP32 parameter
    gradients (``dfn`` folded from its per-chunk slots in a fixed order), with
    no run-to-run bit promise; the finish node returns no gradient for ``X``.
    ``state`` is the private state :func:`cake_mhc_nondet_train_prepare` builds.
    """

    @staticmethod
    def forward(ctx, X, fn, scale, base, state):
        U, stats = cake_mhc_nondet_train_forward_prepare(
            X,
            fn,
            scale,
            base,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            accumulation=state.accumulation,
            workspace=state.workspace,
        )
        state.fn, state.scale, state.base = fn.detach(), scale.detach(), base.detach()
        state.stats = stats  # read and released by the finish forward
        if state.save_stats:
            ctx.save_for_backward(X, stats)
        else:
            ctx.save_for_backward(X)
        ctx.state = state
        # An output the backward did not reach arrives as ``None``, not as zeros.
        ctx.set_materialize_grads(False)
        return (U,) + _edges(X.shape[0], X.shape[2], X.device, state.save_stats)

    @staticmethod
    def backward(ctx, dU, dY, grad_stats, stats=None):
        X, *saved = ctx.saved_tensors
        state = ctx.state
        held = state.take()
        if dY is None:
            # No edge gradient: a finish node rerun by a reentrant checkpoint left
            # them on the state, or the finish node's backward never ran (``Y``
            # did not reach the loss: the residual-mixing path then carries no
            # gradient).
            dY, grad_stats, stats = held
        rows = int(X.shape[0])
        if saved:
            stats = saved[0]
        elif stats is None:
            stats = state.replay(X)
        if dY is None:
            dY = torch.zeros_like(X)
        if grad_stats is None:
            grad_stats = torch.zeros((rows, GSTATS_COLS), dtype=_F32, device=X.device)
        dX, dfn, dscale, dbase = _backward_finish(
            X,
            dY,
            _as_bf16(dU, shape=(rows, X.shape[2]), device=X.device),
            None,  # ``B``: validated when given, not read
            state.fn,
            state.scale,
            state.base,
            stats,
            grad_stats,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            accumulation=state.accumulation,
            accumulate=False,
            workspace=state.backward_workspace,
            dX=None,
            dfn=None,
            dscale=None,
            dbase=None,
            path=_PATH_BACKWARD if saved else _PATH_CHECKPOINT,
        )
        return dX, dfn, dscale, dbase, None


class CakeMHCNondetFinishFunction(torch.autograd.Function):
    """Autograd node of ``forward_finish``: ``forward(X, B, state, *edges) -> Y``.

    Its backward is ``backward_prepare``: it returns ``dB`` for the external
    branch and hands ``dY`` and the gradient statistics (under
    ``save_stats=False`` also the statistics it replayed) to the prepare node as
    the gradients of ``edges``, the prepare node's edge outputs, so the autograd
    engine releases them once the prepare node's backward has run.  When a
    checkpoint region that starts after the prepare reruns this forward, the
    statistics the state already released are replayed from ``X``; a
    reentrant region reruns it without the edges, and its backward leaves them
    on the state for the prepare node instead.
    """

    @staticmethod
    def forward(ctx, X, B, state, *edges):
        stats = state.stats
        if stats is None:
            if state.fn is None:
                raise RuntimeError(
                    "cake_mhc_nondet_train_finish needs the state of a preceding cake_mhc_nondet_train_prepare"
                )
            # Replay: a checkpoint region that starts after the prepare reruns the
            # finish on a state that already released its statistics; recompute
            # them from ``X``.
            stats = state.replay(X)
        Y = cake_mhc_nondet_train_forward_finish(X, B, stats)
        if state.save_stats:
            ctx.save_for_backward(X, B, stats)
        else:
            ctx.save_for_backward(X, B)
        state.stats = None  # nothing of the forward stays on the state
        ctx.state = state
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, B, *saved = ctx.saved_tensors
        state = ctx.state
        stats = saved[0] if saved else state.replay(X)
        dY = _as_bf16(dY, shape=(X.shape[0], STREAMS, X.shape[2]), device=X.device)
        dB, grad_stats = _backward_prepare(
            dY,
            B,
            X,
            stats,
            dB=None,
            grad_stats_out=None,
            path=_PATH_BACKWARD if saved else _PATH_CHECKPOINT,
        )
        handed = (dY, grad_stats) if saved else (dY, grad_stats, stats)
        edges = len(ctx.needs_input_grad) - 3
        if edges and ctx.needs_input_grad[3]:
            return (None, dB, None, *handed)
        if state.recorded:
            # No edge reaches the prepare node (a finish rerun by a reentrant
            # checkpoint): its backward takes them off the state.
            state.dY, state.grad_stats = dY, grad_stats
            if not saved:
                state.stats = stats
        return (None, dB, None) + (None,) * edges


def _as_operand(
    grad: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """A gradient autograd delivers, as a kernel operand: zeros for ``None``; else of ``dtype``, contiguous (a
    zero-stride expanded gradient is materialized) and 32-byte aligned (copied when it is not)."""

    if grad is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    if grad.dtype is not dtype or not grad.is_contiguous():
        grad = grad.contiguous().to(dtype)
    if grad.numel() and grad.data_ptr() % TENSOR_ALIGNMENT_BYTES:
        grad = grad.clone()
    return grad


class CakeMHCNondetHandoffPrepareFunction(torch.autograd.Function):
    """Autograd node of ``handoff_forward_prepare``: ``forward(X, pre_in, fn, scale, base, state) -> (U, pre_next,
    *edges)``.

    The pre-handoff counterpart of :class:`CakeMHCNondetPrepareFunction`: the same state, edges and handle (the
    finish is :class:`CakeMHCNondetFinishFunction`, unchanged).  Its backward is ``handoff_backward_finish``: it
    receives ``dU`` from the external branch, ``dpre_next`` from the consumer of ``pre_next`` (``None``: zeros) and,
    on the edges, ``dY`` and the gradient statistics from the finish node's backward; it returns ``dX``, ``dpre_in``
    (``None`` when ``pre_in`` needs no gradient) and the parameter gradients.  ``dU`` and ``dpre_next`` are
    normalized first (:func:`_as_operand`: dtype, contiguity, alignment).  It saves ``X``, ``pre_in`` (the previous
    node's ``pre_next``: no new memory) and, with ``save_stats``, ``stats``.
    """

    @staticmethod
    def forward(ctx, X, pre_in, fn, scale, base, state):
        U, pre_next, stats = cake_mhc_nondet_train_handoff_forward_prepare(
            X,
            pre_in,
            fn,
            scale,
            base,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            accumulation=state.accumulation,
            workspace=state.workspace,
        )
        state.fn, state.scale, state.base = fn.detach(), scale.detach(), base.detach()
        state.stats = stats  # read and released by the finish forward
        if state.save_stats:
            ctx.save_for_backward(X, pre_in, stats)
        else:
            ctx.save_for_backward(X, pre_in)
        ctx.state = state
        ctx.set_materialize_grads(False)
        return (U, pre_next) + _edges(
            X.shape[0], X.shape[2], X.device, state.save_stats
        )

    @staticmethod
    def backward(ctx, dU, dpre_next, dY, grad_stats, stats=None):
        X, pre_in, *saved = ctx.saved_tensors
        state = ctx.state
        held = state.take()
        if dY is None:
            dY, grad_stats, stats = held
        rows, hidden = int(X.shape[0]), int(X.shape[2])
        if saved:
            stats = saved[0]
        elif stats is None:
            stats = state.replay(X)
        if dY is None:
            dY = torch.zeros_like(X)
        if grad_stats is None:
            grad_stats = torch.zeros((rows, GSTATS_COLS), dtype=_F32, device=X.device)
        dX, dpre_in, dfn, dscale, dbase = _handoff_backward_finish(
            X,
            dY,
            _as_operand(dU, shape=(rows, hidden), dtype=_BF16, device=X.device),
            pre_in,
            _as_operand(dpre_next, shape=(rows, STREAMS), dtype=_F32, device=X.device),
            None,  # ``B``: validated when given, not read
            state.fn,
            state.scale,
            state.base,
            stats,
            grad_stats,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            accumulation=state.accumulation,
            accumulate=False,
            workspace=state.backward_workspace,
            dX=None,
            dpre_in=None,
            dfn=None,
            dscale=None,
            dbase=None,
            path=_PATH_BACKWARD if saved else _PATH_CHECKPOINT,
        )
        return (
            dX,
            dpre_in if ctx.needs_input_grad[1] else None,
            dfn,
            dscale,
            dbase,
            None,
        )


class CakeMHCNondetCollapseFunction(torch.autograd.Function):
    """Autograd node of the standalone collapse: ``forward(X, pre_in) -> U``, backward ``collapse_backward``.

    It saves ``X`` and ``pre_in``; ``dU`` is normalized like the handoff node's (:func:`_as_operand`), and a
    ``None`` ``dU`` (``U`` reached no loss) gives no gradient.
    """

    @staticmethod
    def forward(ctx, X, pre_in):
        U = cake_mhc_nondet_train_collapse_forward(X, pre_in)
        ctx.save_for_backward(X, pre_in)
        ctx.set_materialize_grads(False)
        return U

    @staticmethod
    def backward(ctx, dU):
        if dU is None:
            return None, None
        X, pre_in = ctx.saved_tensors
        dX, dpre_in = cake_mhc_nondet_train_collapse_backward(
            X,
            pre_in,
            _as_operand(
                dU, shape=(X.shape[0], X.shape[2]), dtype=_BF16, device=X.device
            ),
        )
        return (
            dX if ctx.needs_input_grad[0] else None,
            dpre_in if ctx.needs_input_grad[1] else None,
        )


def cake_mhc_nondet_train_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    save_stats: bool = True,
    workspace: Optional[torch.Tensor] = None,
    backward_workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, _MHCTrainHandle]:
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_nondet_train_finish`.

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``X`` and the parameters may require gradients).
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``"fp32"`` default,
        ``"fp32x2"`` opt-in); the forward, its replay and the backward of this
        node (and the finish node of its handle) run in this mode.
    save_stats : bool
        ``True`` (default): the per-token statistics are saved for the backward
        through the autograd saved tensors (an activation checkpoint frees and
        recomputes them with the inputs).  ``False``: they are dropped after the
        forward and recomputed from ``X`` in the backward (checkpoint replay)
        by the same programs as the saved record (no bit promise).
    workspace, backward_workspace : torch.Tensor, optional
        Caller-owned forward / backward workspaces sized for this ``H``
        (``hidden=X.shape[2]``); per-device defaults when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` and the handle of this prepare node for one
    :func:`cake_mhc_nondet_train_finish`.  The forward runs the deterministic programs'
    copies; the prepare node's backward runs ``backward_finish`` (see
    :class:`CakeMHCNondetPrepareFunction`).
    """

    state = _MHCTrainState(
        save_stats=save_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=_loader.check_accumulation(accumulation),
        workspace=workspace,
        backward_workspace=backward_workspace,
    )
    out = CakeMHCNondetPrepareFunction.apply(X, fn, scale, base, state)
    U = out[0]
    # ``U`` requires grad exactly when the prepare node was recorded.
    state.recorded = U.requires_grad
    return U, _MHCTrainHandle(state, out[1:] if state.recorded else ())


def cake_mhc_nondet_train_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    handle: _MHCTrainHandle,
) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, H]`` tensor passed to :func:`cake_mhc_nondet_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, H]`` output of the external branch applied to ``U`` (may require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_nondet_train_prepare`, for one finish
        (any other handle, such as a ``cake_mhc_det_train_prepare`` one, raises
        ``TypeError``).  A reentrant checkpoint region (``use_reentrant=True``)
        holds both calls or starts after the prepare; it does not end between
        them.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.
    """

    if not isinstance(handle, _MHCTrainHandle):
        raise TypeError(
            "handle must be the handle returned by cake_mhc_nondet_train_prepare (a cake_mhc_det_train_prepare "
            "handle goes to cake_pplx.cake_mhc_det_train.cake_mhc_det_train_finish)"
        )
    # The forward takes the edges off the handle: the handle then holds nothing of
    # the prepare node, and a checkpoint rerun of this call runs without them.
    edges, handle.edges = handle.edges, ()
    return CakeMHCNondetFinishFunction.apply(X, B, handle.state, *edges)


def cake_mhc_nondet_train_handoff_prepare(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    save_stats: bool = True,
    workspace: torch.Tensor | None = None,
    backward_workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, _MHCTrainHandle]:
    """Autograd-enabled ``handoff_forward_prepare``: ``(U, pre_next, handle)``; pass ``handle`` to
    :func:`cake_mhc_nondet_train_finish` and ``pre_next`` to the next sublayer (or the final collapse).

    Parameters
    ----------
    X, pre_in, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_handoff_forward_prepare` (``X``, ``pre_in`` and the parameters may
        require gradients).
    rms_eps, hc_eps, sinkhorn_iters, accumulation, save_stats, workspace, backward_workspace :
        As in :func:`cake_mhc_nondet_train_prepare`.

    Returns
    -------
    ``(U, pre_next, handle)``: BF16 ``U``, FP32 ``pre_next`` and the handle of this node for one
    :func:`cake_mhc_nondet_train_finish` (see :class:`CakeMHCNondetHandoffPrepareFunction`).
    """

    state = _MHCTrainState(
        save_stats=save_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=_loader.check_accumulation(accumulation),
        workspace=workspace,
        backward_workspace=backward_workspace,
    )
    out = CakeMHCNondetHandoffPrepareFunction.apply(X, pre_in, fn, scale, base, state)
    U, pre_next = out[0], out[1]
    state.recorded = U.requires_grad
    return U, pre_next, _MHCTrainHandle(state, out[2:] if state.recorded else ())


def cake_mhc_nondet_train_collapse(
    X: torch.Tensor, pre_in: torch.Tensor
) -> torch.Tensor:
    """Autograd-enabled standalone collapse: ``U = BF16(sum_i pre_in_i X_i)`` (:class:`CakeMHCNondetCollapseFunction`);
    its backward is :func:`cake_mhc_nondet_train_collapse_backward`."""

    return CakeMHCNondetCollapseFunction.apply(X, pre_in)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TENSOR_ALIGNMENT_BYTES",
    "WORKSPACE_ALIGNMENT_BYTES",
    "CakeMHCNondetCollapseFunction",
    "CakeMHCNondetFinishFunction",
    "CakeMHCNondetHandoffPrepareFunction",
    "CakeMHCNondetPrepareFunction",
    "cake_mhc_nondet_train_backward_finish",
    "cake_mhc_nondet_train_backward_prepare",
    "cake_mhc_nondet_train_backward_workspace",
    "cake_mhc_nondet_train_backward_workspace_bytes",
    "cake_mhc_nondet_train_collapse",
    "cake_mhc_nondet_train_collapse_backward",
    "cake_mhc_nondet_train_collapse_forward",
    "cake_mhc_nondet_train_finish",
    "cake_mhc_nondet_train_forward_finish",
    "cake_mhc_nondet_train_forward_prepare",
    "cake_mhc_nondet_train_forward_stats",
    "cake_mhc_nondet_train_forward_workspace",
    "cake_mhc_nondet_train_forward_workspace_bytes",
    "cake_mhc_nondet_train_handoff_backward_finish",
    "cake_mhc_nondet_train_handoff_forward_prepare",
    "cake_mhc_nondet_train_handoff_prepare",
    "cake_mhc_nondet_train_prepare",
    "cake_mhc_nondet_train_reset_workspace",
    "is_cake_mhc_nondet_train_handoff_supported",
    "is_cake_mhc_nondet_train_supported",
    "cake_mhc_nondet_train_route_key",
    "cake_mhc_nondet_train_route_modules",
]
