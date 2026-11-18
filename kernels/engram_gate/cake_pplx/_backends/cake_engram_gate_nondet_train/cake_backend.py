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

Loader, checks, launches and autograd of the Cake Engram context gate: four
residual streams of width 5120 (SM100 / SM103 / SM107), non-deterministic
package.

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds
one record per program (its ``library`` stem; the ``build.json`` entries are
``<library>_<arch>``), :func:`.cake_jit.record` names the record of a stage
and :func:`.cake_jit.grid` gives its grid.  :mod:`.cake_build` serves a
library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through the package's JIT build shim (``_cake_pplx_jit.py``) from
the sources and flags of ``build.json``.

Per token ``t`` and stream ``s`` (``H = 5120``), with every quantity in FP32
and each BF16 output rounded once, at its store::

    h = x[t, s]   k = key[t, s]   v = value[t]   w = q_weight[s] * k_weight[s]
    r_h = rsqrt(mean(h^2) + eps)   r_k = rsqrt(mean(k^2) + eps)
    P = sum_d w_d h_d k_d          dot = P r_h r_k / sqrt(H)
    gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))   (0 on a masked token)
    out[t, s] = bf16(h + gate v)

The host forms ``w`` (FP32 ``[4, 5120]``, one rounding of the FP32 product).
One forward call launches ``fwd``.  One backward call launches ``bwd``, which
recomputes ``r_h``, ``r_k`` and ``P`` from the inputs (the forward keeps
nothing beyond its inputs), writes ``dx``, ``dkey`` and ``dvalue`` and the FP32
weight-gradient partial sums ``dw = sum_t c h k``, then ``dw_reduce``, which
writes ``dq_weight = dw * k_weight`` and ``dk_weight = dw * q_weight`` in FP32.
``key`` and ``value`` (and ``dkey`` and ``dvalue``) may be views into one packed
``[T, 25600]`` buffer: the kernels read their token strides.  A masked token
(``token_mask[t]`` False) gives ``out[t] = x[t]`` and ``dx[t] = dout[t]``
exactly, zero ``dkey[t]`` and ``dvalue[t]``, and adds nothing to the weight
gradients.

``fwd`` and ``bwd`` run a persistent grid of four-CTA clusters sized from the
token count and the SM count; any grid is correct, so the token count is a
run-time value (no recompilation, no shape list).  Every launch is a plain
stream-ordered launch on the current torch stream of the operands' device:
nothing cooperative.  Nothing here reads an environment variable, reads
device memory or synchronizes; grids and buffer sizes come from tensor
shapes and the SM count.

This is the non-deterministic package of its family: ``bwd`` adds each
cluster's weight-gradient partial sums into one FP32 accumulator with float
atomics, in arrival order, so ``dq_weight`` and ``dk_weight`` may differ in
their last bits run to run; ``out``, ``dx``, ``dkey`` and ``dvalue`` have one
fixed order.  Every backward call that launches raises PyTorch's
deterministic-algorithms alert first (:func:`_alert_nondeterministic`); the
forward never does.
"""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from collections.abc import Mapping
from typing import Any, Optional, Union

import torch

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HIDDEN",
    "KV_WIDTH",
    "STREAMS",
    "CakeEngramGateNondetFunction",
    "CakeEngramGateNondetKvFunction",
    "arch_for_capability",
    "cake_engram_gate_nondet",
    "cake_engram_gate_nondet_kv",
    "cake_engram_gate_nondet_train_backward",
    "cake_engram_gate_nondet_train_forward",
    "cake_engram_gate_nondet_train_workspace_bytes",
    "device_arch",
    "is_cake_engram_gate_nondet_train_supported",
    "is_prebuilt",
    "launch_bwd",
    "launch_dw_reduce",
    "launch_fwd",
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
# The fixed operand geometry: four residual streams of width 5120.
STREAMS: int = cake_jit.STREAMS
HIDDEN: int = cake_jit.HIDDEN
# Columns of the packed projection output ``kv``: the four keys, then the value.
KV_WIDTH: int = (STREAMS + 1) * HIDDEN

# How the forward and the backward sum the dot product ``P = sum_d w_d h_d k_d``:
# the record suffix of :data:`.cake_jit.ACC_MODES` they launch.  ``"dot2"`` sums
# it as a compensated FP32 dot product, ``"f32"`` in one FP32 word.
_DOT_ACC = "dot2"

# Token strides of the strided operands are multiples of 8 elements and every
# BF16 operand starts on a 16-byte boundary: the kernels move 16 bytes per access.
_STRIDE_ELEMENTS = 8
_ALIGNMENT = 16
# The kernels take the token count as a 32-bit integer (token offsets are 64-bit).
_MAX_TOKENS = 2**31 - 1
# Bytes of one FP32 [4, 5120] row (the weight product, a weight copy, a partial row).
_ROW_BYTES = STREAMS * HIDDEN * 4

# This package's module prefix (the vendored name of a vendored copy).
_PACKAGE = __name__.rpartition(".")[0]
# The public module the entry points are documented under.
_MODULE = "cake_pplx.cake_engram_gate_nondet_train"
_DET_MODULE = "cake_pplx.cake_engram_gate_det_train"


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
            f"device {device_index} (capability {capability}) is not a Cake Engram gate target; the built "
            f"architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (fixed for the process)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


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


def is_cake_engram_gate_nondet_train_supported(device: Union[torch.device, int, str, None] = None) -> bool:
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
        raise KeyError(f"{name!r} is not a record of the Cake Engram gate registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake Engram gate record {name} is not delivered for {arch}; its architectures: {found['arches']}"
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


def _launch(name: str, stage: str, values: Mapping[str, Any], grid: tuple[int, int, int], device: torch.device) -> None:
    """Launch record ``name`` of ``stage`` with its kernel parameters named in ``values`` on ``grid``.  tvm-ffi runs
    the call on the current torch stream of the device of its first CUDA tensor argument (every buffer is on
    ``device``), and the binding makes that device current for the launch."""
    found = record(name)
    if found["stage"] != stage:
        raise ValueError(f"record {name} serves stage {found['stage']}, not {stage}")
    args = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    _entry(name, device_arch(device.index))(*args, int(grid[0]), int(grid[1]), int(grid[2]))


def _mask_values(name: str, mask: Optional[torch.Tensor], device: torch.device) -> dict[str, Any]:
    """The mask arguments of record ``name``: the mask (``None`` when absent: a nullable pointer) and, for a kernel
    that takes ``has_mask``, the flag and a one-byte stand-in for an absent mask (never read)."""
    if "has_mask" not in parameters(name):
        return {"mask": mask}
    if mask is None:
        return {"mask": torch.empty(1, dtype=torch.bool, device=device), "has_mask": 0}
    return {"mask": mask, "has_mask": 1}


# ----------------------------------------------------------------------------------------------------------------
# Stage launchers: caller-owned buffers, the current stream, no allocation (but the absent-mask stand-in), no check
# of the operands


def launch_fwd(
    name: str,
    *,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    w: torch.Tensor,
    mask: Optional[torch.Tensor],
    out: torch.Tensor,
    eps: float,
    clamp_value: float,
    clusters: int,
) -> None:
    """Launch the forward record ``name`` over ``T = x.shape[0] >= 1`` tokens on ``clusters`` four-CTA clusters:
    ``out = bf16(x + gate * value)`` from ``x`` (BF16 ``[T, 4, 5120]``, contiguous), ``key`` (BF16 ``[T, 4, 5120]``,
    token stride a multiple of 8), ``value`` (BF16 ``[T, 5120]``, same), ``w`` (FP32 ``[4, 5120]``, the weight
    product) and ``mask`` (bool ``[T]`` or None)."""
    T = int(x.shape[0])
    values = {
        "x": x,
        "key": key,
        "value": value,
        "w": w,
        "out": out,
        "T": T,
        "eps": float(eps),
        "clamp": float(clamp_value),
        **_mask_values(name, mask, x.device),
    }
    _launch(name, "fwd", values, (cake_jit.CLUSTER_CTAS * int(clusters), 1, 1), x.device)


def launch_bwd(
    name: str,
    *,
    dout: torch.Tensor,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    w: torch.Tensor,
    mask: Optional[torch.Tensor],
    dx: torch.Tensor,
    dkey: torch.Tensor,
    dvalue: torch.Tensor,
    dw: torch.Tensor,
    eps: float,
    clamp_value: float,
    clusters: int,
) -> None:
    """Launch the backward record ``name`` over ``T = x.shape[0] >= 1`` tokens on ``clusters`` four-CTA clusters:
    ``dx`` (BF16 ``[T, 4, 5120]``, contiguous), ``dkey`` and ``dvalue`` (BF16, laid out as ``key`` and ``value``)
    and the FP32 ``[1, 4, 5120]`` accumulator ``dw`` (zeroed by the caller), into which each cluster adds its sum of
    ``c h k`` over its tokens."""
    T = int(x.shape[0])
    values = {
        "dout": dout,
        "x": x,
        "key": key,
        "value": value,
        "w": w,
        "dx": dx,
        "dkey": dkey,
        "dvalue": dvalue,
        "dw_acc": dw,
        "T": T,
        "eps": float(eps),
        "clamp": float(clamp_value),
        **_mask_values(name, mask, x.device),
    }
    _launch(name, "bwd", values, (cake_jit.CLUSTER_CTAS * int(clusters), 1, 1), x.device)


def launch_dw_reduce(
    name: str,
    *,
    dw: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
) -> None:
    """Launch the finalize record ``name``: ``dq = (sum_p dw[p]) * kw`` and ``dk = (sum_p dw[p]) * qw``, FP32
    ``[4, 5120]``, from the ``NP`` partial rows ``dw`` (FP32 ``[NP, 4, 5120]``) summed in one fixed order; ``qw`` and
    ``kw`` are the FP32 weights (contiguous)."""
    values = {"dw_part": dw, "qw": qw, "kw": kw, "dq": dq, "dk": dk, "NP": int(dw.shape[0])}
    _launch(name, "dw_reduce", values, cake_jit.grid(name, 1, 1), dw.device)


# ----------------------------------------------------------------------------------------------------------------
# torch.use_deterministic_algorithms(True)


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside this package, ``cake_pplx`` and torch's autograd, for a
    warning issued by the function that calls this one: the caller of the entry point, also through the facade and
    autograd."""
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
                or name.startswith("torch.")
            ):
                return level
        frame = parent
    return 2


def _alert_nondeterministic(api: str, det_api: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic algorithms
    are enabled: a ``RuntimeError`` naming ``det_api``, the deterministic package's entry of the same role, or a
    ``UserWarning`` under ``torch.use_deterministic_algorithms(True, warn_only=True)`` after which this package's
    backward runs (never the deterministic package's); nothing otherwise.  Called by every backward call that
    launches, after the checks and before it allocates or launches anything.  A backward over no token launches
    nothing and does not alert; the forward never alerts."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        f"{_MODULE}.{api} adds the weight gradients with float atomics, so they are not bitwise reproducible run to "
        f"run, but torch.use_deterministic_algorithms(True) is in effect; call {_DET_MODULE}.{det_api}"
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(f"{message}; running this package's kernels", UserWarning, stacklevel=_caller_stacklevel())
        return
    raise RuntimeError(message)


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
        raise ValueError(f"{name} must be on x's device {device}, got {t.device}")


def _aligned(name: str, t: torch.Tensor) -> None:
    if t.numel() and t.data_ptr() % _ALIGNMENT:
        raise ValueError(f"{name} must start on a {_ALIGNMENT}-byte boundary, got address {t.data_ptr():#x}")


def _streams_layout(name: str, t: torch.Tensor, T: int, *, contiguous: bool, output: bool) -> None:
    """A BF16 ``[T, 4, 5120]`` operand: contiguous, or (``contiguous`` False) unit column stride, stream stride 5120
    and a token stride that is a multiple of 8 elements (an output's rows must not overlap); 16-byte aligned."""
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {t.dtype}")
    if tuple(t.shape) != (T, STREAMS, HIDDEN):
        raise ValueError(f"{name} must be [T, {STREAMS}, {HIDDEN}] with T = {T}, got {tuple(t.shape)}")
    if contiguous:
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got strides {t.stride()}")
    elif T:
        stride = t.stride()
        if stride[2] != 1 or stride[1] != HIDDEN:
            raise ValueError(
                f"{name} must have the column stride 1 and the stream stride {HIDDEN} (a [T, {STREAMS}, {HIDDEN}] "
                f"slice of a [T, n] buffer), got strides {stride}"
            )
        if T > 1 and stride[0] % _STRIDE_ELEMENTS:
            raise ValueError(f"{name} needs a token stride that is a multiple of {_STRIDE_ELEMENTS}, got {stride[0]}")
        if output and T > 1 and stride[0] < STREAMS * HIDDEN:
            raise ValueError(f"{name} is an output: its token rows must not overlap, got the token stride {stride[0]}")
    _aligned(name, t)


def _value_layout(name: str, t: torch.Tensor, T: int, *, output: bool) -> None:
    """A BF16 ``[T, 5120]`` operand: unit column stride and a token stride that is a multiple of 8 elements (an
    output's rows must not overlap); 16-byte aligned."""
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {t.dtype}")
    if tuple(t.shape) != (T, HIDDEN):
        raise ValueError(f"{name} must be [T, {HIDDEN}] with T = {T}, got {tuple(t.shape)}")
    if T:
        stride = t.stride()
        if stride[1] != 1:
            raise ValueError(f"{name} must have the column stride 1, got strides {stride}")
        if T > 1 and stride[0] % _STRIDE_ELEMENTS:
            raise ValueError(f"{name} needs a token stride that is a multiple of {_STRIDE_ELEMENTS}, got {stride[0]}")
        if output and T > 1 and stride[0] < HIDDEN:
            raise ValueError(f"{name} is an output: its token rows must not overlap, got the token stride {stride[0]}")
    _aligned(name, t)


def _check_scalars(eps: Any, clamp_value: Any) -> tuple[float, float]:
    try:
        eps_f, clamp_f = float(eps), float(clamp_value)
    except (TypeError, ValueError):
        raise ValueError(f"eps and clamp_value must be real numbers, got {eps!r} and {clamp_value!r}") from None
    if not (math.isfinite(eps_f) and eps_f >= 0.0):
        raise ValueError(f"eps must be finite and non-negative, got {eps}")
    if not (math.isfinite(clamp_f) and clamp_f >= 0.0):
        raise ValueError(f"clamp_value must be finite and non-negative, got {clamp_value}")
    return eps_f, clamp_f


def _check_operands(
    x: Any,
    key: Any,
    value: Any,
    q_weight: Any,
    k_weight: Any,
    token_mask: Any,
    eps: Any,
    clamp_value: Any,
) -> tuple[int, float, float]:
    """``(T, eps, clamp_value)`` of a valid set of forward operands; ``ValueError`` otherwise.  The device checks
    come before anything that touches CUDA."""
    x = _tensor("x", x)
    operands = [("x", x), ("key", _tensor("key", key)), ("value", _tensor("value", value))]
    operands += [("q_weight", _tensor("q_weight", q_weight)), ("k_weight", _tensor("k_weight", k_weight))]
    if token_mask is not None:
        operands.append(("token_mask", _tensor("token_mask", token_mask)))
    for name, t in operands:
        _on_device(name, t, x.device)
    if x.dim() != 3:
        raise ValueError(f"x must be [T, {STREAMS}, {HIDDEN}], got {tuple(x.shape)}")
    T = int(x.shape[0])
    if T > _MAX_TOKENS:
        raise ValueError(f"the token count must be at most {_MAX_TOKENS}, got {T}")
    _streams_layout("x", x, T, contiguous=True, output=False)
    _streams_layout("key", key, T, contiguous=False, output=False)
    _value_layout("value", value, T, output=False)
    for name, t in (("q_weight", q_weight), ("k_weight", k_weight)):
        if t.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be torch.bfloat16 or torch.float32, got {t.dtype}")
        if tuple(t.shape) != (STREAMS, HIDDEN):
            raise ValueError(f"{name} must be [{STREAMS}, {HIDDEN}], got {tuple(t.shape)}")
    if token_mask is not None:
        if token_mask.dtype != torch.bool:
            raise ValueError(f"token_mask must be torch.bool, got {token_mask.dtype}")
        if tuple(token_mask.shape) != (T,):
            raise ValueError(f"token_mask must be [T] with T = {T}, got {tuple(token_mask.shape)}")
    eps_f, clamp_f = _check_scalars(eps, clamp_value)
    return T, eps_f, clamp_f


def _check_output(name: str, t: Any, T: int, device: torch.device, kind: str) -> None:
    t = _tensor(name, t)
    _on_device(name, t, device)
    if kind == "value":
        _value_layout(name, t, T, output=True)
    else:
        _streams_layout(name, t, T, contiguous=kind == "contiguous", output=True)


def _split_kv(kv: Any, name: str = "kv") -> tuple[torch.Tensor, torch.Tensor]:
    """``(key, value)``: the ``[T, 4, 5120]`` and ``[T, 5120]`` views of a packed BF16 ``[T, 25600]`` buffer (the
    four keys, then the value, along the last axis; unit column stride)."""
    kv = _tensor(name, kv)
    if kv.dim() != 2 or kv.shape[1] != KV_WIDTH:
        raise ValueError(f"{name} must be [T, {KV_WIDTH}], got {tuple(kv.shape)}")
    if kv.shape[0] and kv.stride(1) != 1:
        raise ValueError(f"{name} must have the column stride 1, got strides {kv.stride()}")
    key = kv[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN))
    return key, kv[:, STREAMS * HIDDEN :]


# ----------------------------------------------------------------------------------------------------------------
# Host operands


def _fp32(t: torch.Tensor) -> torch.Tensor:
    """``t`` as a contiguous, 16-byte aligned FP32 tensor (``t`` itself when it already is one)."""
    t = t.detach().to(dtype=torch.float32, memory_format=torch.contiguous_format)
    return t if t.data_ptr() % _ALIGNMENT == 0 else t.clone()


def _mask(token_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The mask the kernels read: contiguous bool ``[T]``, or None."""
    return None if token_mask is None else token_mask.contiguous()


def _param_grad(grad: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """An FP32 weight gradient in its parameter's dtype."""
    return grad if param.dtype == torch.float32 else grad.to(param.dtype)


def cake_engram_gate_nondet_train_workspace_bytes(T: int, device: Union[torch.device, int, str, None] = None) -> int:
    """Bytes of device memory one backward call over ``T`` tokens on ``device`` (None: the current CUDA device)
    allocates beside its operands and outputs, with BF16 weights: the FP32 weight product, the two FP32 weight
    copies (none for contiguous FP32 weights) and the
    ``[4, 5120]`` FP32 weight-gradient accumulator,
    80 KiB each row (0 for ``T == 0``).  A forward call allocates the weight product (and an FP32 copy of a BF16
    ``q_weight``) only; the state kept for the backward is the inputs alone."""
    T = int(T)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if T == 0:
        return 0
    parts = 1
    return _ROW_BYTES * (3 + parts)


# ----------------------------------------------------------------------------------------------------------------
# Forward and backward (explicit, without autograd)


def _forward(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    eps: float,
    clamp_value: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    T, eps_f, clamp_f = _check_operands(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
    device = x.device
    if out is None:
        out = torch.empty((T, STREAMS, HIDDEN), dtype=torch.bfloat16, device=device)
    else:
        _check_output("out", out, T, device, "contiguous")
    if T == 0:
        return out
    index = device.index
    device_arch(index)  # RuntimeError on a device outside ARCHES
    # FP32(q_weight) * FP32(k_weight): a BF16 k_weight is widened exactly by type promotion.
    w = (_fp32(q_weight) * k_weight.detach()).contiguous()
    launch_fwd(
        cake_jit.record("fwd", _DOT_ACC),
        x=x,
        key=key,
        value=value,
        w=w,
        mask=_mask(token_mask),
        out=out,
        eps=eps_f,
        clamp_value=clamp_f,
        clusters=cake_jit.clusters("fwd", T, sm_count(index)),
    )
    return out


def _backward(
    dout: torch.Tensor,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    eps: float,
    clamp_value: float,
    dx: Optional[torch.Tensor],
    dkey: Optional[torch.Tensor],
    dvalue: Optional[torch.Tensor],
    api: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, eps_f, clamp_f = _check_operands(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
    device = x.device
    dout = _tensor("dout", dout)
    _on_device("dout", dout, device)
    _streams_layout("dout", dout, T, contiguous=True, output=False)
    for name, t, kind in (("dx", dx, "contiguous"), ("dkey", dkey, "strided"), ("dvalue", dvalue, "value")):
        if t is not None:
            _check_output(name, t, T, device, kind)
    if T == 0:
        # No token: nothing to launch and nothing to alert; zero weight gradients.
        empty = functools.partial(torch.empty, dtype=torch.bfloat16, device=device)
        dq = torch.zeros((STREAMS, HIDDEN), dtype=torch.float32, device=device)
        return (
            empty((0, STREAMS, HIDDEN)) if dx is None else dx,
            empty((0, STREAMS, HIDDEN)) if dkey is None else dkey,
            empty((0, HIDDEN)) if dvalue is None else dvalue,
            dq,
            torch.zeros_like(dq),
        )
    index = device.index
    device_arch(index)  # RuntimeError on a device outside ARCHES, before the alert
    _alert_nondeterministic(api, api.replace("nondet", "det"))
    if dx is None:
        dx = torch.empty((T, STREAMS, HIDDEN), dtype=torch.bfloat16, device=device)
    if dkey is None:
        dkey = torch.empty((T, STREAMS, HIDDEN), dtype=torch.bfloat16, device=device)
    if dvalue is None:
        dvalue = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=device)
    qw32, kw32 = _fp32(q_weight), _fp32(k_weight)
    w = qw32 * kw32
    clusters = cake_jit.clusters("bwd", T, sm_count(index))
    # The accumulator the clusters add into.
    dw = torch.zeros((1, STREAMS, HIDDEN), dtype=torch.float32, device=device)
    launch_bwd(
        cake_jit.record("bwd", _DOT_ACC),
        dout=dout,
        x=x,
        key=key,
        value=value,
        w=w,
        mask=_mask(token_mask),
        dx=dx,
        dkey=dkey,
        dvalue=dvalue,
        dw=dw,
        eps=eps_f,
        clamp_value=clamp_f,
        clusters=clusters,
    )
    dq = torch.empty((STREAMS, HIDDEN), dtype=torch.float32, device=device)
    dk = torch.empty((STREAMS, HIDDEN), dtype=torch.float32, device=device)
    launch_dw_reduce(cake_jit.record("dw_reduce", _DOT_ACC), dw=dw, qw=qw32, kw=kw32, dq=dq, dk=dk)
    return dx, dkey, dvalue, dq, dk


def cake_engram_gate_nondet_train_forward(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The forward without autograd: ``out`` BF16 ``[T, 4, 5120]`` (``out`` given: written in place and returned).

    ``x`` BF16 ``[T, 4, 5120]`` contiguous; ``key`` BF16 ``[T, 4, 5120]`` and ``value`` BF16 ``[T, 5120]``,
    contiguous or views into a packed ``[T, 25600]`` buffer (unit column stride, stream stride 5120, token stride a
    multiple of 8); ``q_weight`` and ``k_weight`` BF16 or FP32 ``[4, 5120]``; ``token_mask`` bool ``[T]`` or None;
    every tensor on one CUDA device, every BF16 operand 16-byte aligned.  ``T == 0`` launches nothing.  The forward
    never consults ``torch.use_deterministic_algorithms``."""
    return _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, out)


def cake_engram_gate_nondet_train_backward(
    dout: torch.Tensor,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    dx: Optional[torch.Tensor] = None,
    dkey: Optional[torch.Tensor] = None,
    dvalue: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward without autograd: ``(dx, dkey, dvalue, dq_weight, dk_weight)`` from the forward's inputs and
    ``dout`` (BF16 ``[T, 4, 5120]``, contiguous); the forward's own output is not needed.

    ``dx`` BF16 ``[T, 4, 5120]`` (contiguous), ``dkey`` and ``dvalue`` BF16, each laid out as ``key`` and ``value``
    may be (caller views into one ``[T, 25600]`` gradient buffer are written in place), allocated when not given;
    ``dq_weight`` and ``dk_weight`` FP32 ``[4, 5120]``.  ``T == 0`` returns zero weight gradients and launches
    nothing.
    ``dq_weight`` and ``dk_weight`` are summed with float atomics: under ``torch.use_deterministic_algorithms(True)``
    the call raises PyTorch's ``RuntimeError`` before it launches (``warn_only=True``: one ``UserWarning``, then it
    runs).
    """
    return _backward(
        dout,
        x,
        key,
        value,
        q_weight,
        k_weight,
        token_mask,
        eps,
        clamp_value,
        dx,
        dkey,
        dvalue,
        "cake_engram_gate_nondet_train_backward",
    )


# ----------------------------------------------------------------------------------------------------------------
# Autograd


def _grad_out(dout: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    if dout is None:
        return torch.zeros_like(x)
    if dout.dtype != torch.bfloat16:
        dout = dout.to(torch.bfloat16)
    if not dout.is_contiguous():
        return dout.contiguous()
    return dout if dout.data_ptr() % _ALIGNMENT == 0 else dout.clone()


class CakeEngramGateNondetFunction(torch.autograd.Function):
    """``out = engram_gate(x, key, value, q_weight, k_weight, token_mask)``.  The saved set is the inputs alone
    (``ctx.save_for_backward``; ``eps`` and ``clamp_value`` on ``ctx``): the backward recomputes the norms and the
    dot product, so a checkpointed forward keeps nothing alive.  The weight gradients are returned in each
    parameter's dtype.
    The backward raises PyTorch's deterministic-algorithms alert (naming
    ``cake_pplx.cake_engram_gate_det_train.cake_engram_gate_det``); the forward never does.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        eps: float,
        clamp_value: float,
    ) -> torch.Tensor:
        out = _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
        ctx.save_for_backward(x, key, value, q_weight, k_weight, token_mask)
        ctx.eps = eps
        ctx.clamp_value = clamp_value
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
        x, key, value, q_weight, k_weight, token_mask = ctx.saved_tensors
        dx, dkey, dvalue, dq, dk = _backward(
            _grad_out(dout, x),
            x,
            key,
            value,
            q_weight,
            k_weight,
            token_mask,
            ctx.eps,
            ctx.clamp_value,
            None,
            None,
            None,
            "cake_engram_gate_nondet",
        )
        need = ctx.needs_input_grad
        return (
            dx if need[0] else None,
            dkey if need[1] else None,
            dvalue if need[2] else None,
            _param_grad(dq, q_weight) if need[3] else None,
            _param_grad(dk, k_weight) if need[4] else None,
            None,
            None,
            None,
        )


class CakeEngramGateNondetKvFunction(torch.autograd.Function):
    """``out = engram_gate(x, key, value, q_weight, k_weight, token_mask)`` with ``key`` and ``value`` the views of
    one packed ``kv [T, 25600]``.  The backward writes dK and dV straight into one new ``dkv [T, 25600]`` (no split
    and no concatenation copy).  The saved set is the inputs alone, as in
    :class:`CakeEngramGateNondetFunction`."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        token_mask: Optional[torch.Tensor],
        eps: float,
        clamp_value: float,
    ) -> torch.Tensor:
        key, value = _split_kv(kv)
        out = _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
        ctx.save_for_backward(x, kv, q_weight, k_weight, token_mask)
        ctx.eps = eps
        ctx.clamp_value = clamp_value
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
        x, kv, q_weight, k_weight, token_mask = ctx.saved_tensors
        key, value = _split_kv(kv)
        dkv = torch.empty((kv.shape[0], KV_WIDTH), dtype=torch.bfloat16, device=kv.device)
        dkey, dvalue = _split_kv(dkv, "dkv")
        dx, _, _, dq, dk = _backward(
            _grad_out(dout, x),
            x,
            key,
            value,
            q_weight,
            k_weight,
            token_mask,
            ctx.eps,
            ctx.clamp_value,
            None,
            dkey,
            dvalue,
            "cake_engram_gate_nondet_kv",
        )
        need = ctx.needs_input_grad
        return (
            dx if need[0] else None,
            dkv if need[1] else None,
            _param_grad(dq, q_weight) if need[2] else None,
            _param_grad(dk, k_weight) if need[3] else None,
            None,
            None,
            None,
        )


def cake_engram_gate_nondet(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
) -> torch.Tensor:
    """The Engram context gate with autograd (:class:`CakeEngramGateNondetFunction`): ``out`` BF16
    ``[T, 4, 5120]``; operands as in :func:`cake_engram_gate_nondet_train_forward`."""
    return CakeEngramGateNondetFunction.apply(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)


def cake_engram_gate_nondet_kv(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
) -> torch.Tensor:
    """The Engram context gate with autograd on the packed ``kv`` BF16 ``[T, 25600]`` (the four keys, then the
    value; unit column stride, token stride a multiple of 8) (:class:`CakeEngramGateNondetKvFunction`): ``out`` BF16
    ``[T, 4, 5120]``; the gradient of ``kv`` is one ``[T, 25600]`` buffer written by the backward kernel."""
    return CakeEngramGateNondetKvFunction.apply(x, kv, q_weight, k_weight, token_mask, eps, clamp_value)
