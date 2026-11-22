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

One forward call launches ``fwd`` alone.  One backward call launches ``bwd``
and ``dw_reduce``, back to back.  Every kernel reads ``q_weight`` and
``k_weight`` as stored, BF16 or FP32, as the 32-bit words of each
(:func:`_words`) with a mode word (:func:`_wmode`): ``fwd`` and ``bwd`` form
the FP32 product ``w`` once per CTA (each BF16 value widened exactly, one
FP32 multiply per column, the bits of ``q_weight.float() * k_weight.float()``).
``bwd`` recomputes ``r_h``, ``r_k`` and ``P`` from the inputs (the forward
keeps nothing beyond its inputs) and writes ``dx``, ``dkey`` and ``dvalue``
and the FP32 weight-gradient partial sums ``dw = sum_t c h k``.  ``dw_reduce``
writes ``dq_weight = dw * k_weight`` and ``dk_weight = dw * q_weight``: in
FP32 for the explicit backward, in each parameter's dtype (rounded once, as
``.to(dtype)`` rounds) for autograd.
``key`` and ``value`` (and ``dkey`` and ``dvalue``) may be views into one packed
``[T, 25600]`` buffer: the kernels read their token strides.  A masked token
(``token_mask[t]`` False) gives ``out[t] = x[t]`` and ``dx[t] = dout[t]``
exactly, zero ``dkey[t]`` and ``dvalue[t]``, and adds nothing to the weight
gradients.

``dot_sum`` selects how ``fwd`` sums ``P``.  ``"fp32"`` (the default) launches
``fwd_f32``: one FP32 word.  ``"compensated"`` (opt-in) launches ``fwd_dot2``:
a compensated FP32 dot product (each product split with an FMA, each running
sum with an error-free two-sum, the error words added once at the end), slower,
which rounds fewer output elements away from the correctly rounded result where
the products nearly cancel.  The backward is the same for both: it recomputes
``P`` from the inputs as a compensated dot product (``bwd_dot2``, or
``bwd_dot2_stage`` with the same bits; :func:`.cake_jit.backward_record`), so
the gradients do not depend on the forward's choice.

``fwd`` and ``bwd`` run a persistent grid of four-CTA clusters: as many as the
token count and as the device holds at once (:func:`.cake_jit.clusters`, from
a table per architecture, SM count and record); any grid is correct, so the
token count is a run-time value (no recompilation, no shape list).  Every
launch is a plain stream-ordered launch on the current torch stream of the
operands' device: nothing cooperative.  Nothing here reads an environment
variable, reads device memory or synchronizes; grids and buffer sizes come
from tensor shapes, the architecture and the SM count.

This is the non-deterministic package of its family: ``out``, ``dx``,
``dkey`` and ``dvalue`` have one fixed order, and ``dq_weight`` and
``dk_weight`` promise no bits run to run.  It launches the deterministic
package's kernels under its own library names: each cluster writes its own
row of weight-gradient partial sums and ``dw_reduce`` adds the rows in one
fixed order.  Adding the clusters' sums into one cleared row with float
atomics instead was slower in eager calls and faster only in some CUDA-graph
calls with a token mask at 129 to 16384 tokens.  Nothing reads torch's
deterministic mode: no launch adds in an order that changes run to run, so
nothing alerts.
"""

from __future__ import annotations

import functools
import math
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

# The forward sums of the dot product ``P = sum_d w_d h_d k_d`` a caller selects with ``dot_sum``, each with the
# suffix of the ``fwd`` record it launches (:data:`.cake_jit.ACC_MODES`): ``"fp32"`` (the default) one FP32 word,
# ``"compensated"`` (opt-in) a compensated FP32 dot product.  Every backward launches ``bwd_dot2`` or, without a
# token mask on an architecture of :data:`.cake_jit.STAGE_BWD_ARCHES` and with at least
# :data:`.cake_jit.STAGE_BWD_MIN_TOKENS` tokens, ``bwd_dot2_stage`` (the same bits).
_DOT_SUMS: dict[str, str] = {"fp32": "f32", "compensated": "dot2"}

# Token strides of the strided operands are multiples of 8 elements and every
# BF16 operand starts on a 16-byte boundary: the kernels move 16 bytes per access.
_STRIDE_ELEMENTS = 8
_ALIGNMENT = 16
# The kernels take the token count as a 32-bit integer (token offsets are 64-bit).
_MAX_TOKENS = 2**31 - 1
# The operand dtypes: BF16 activations, BF16 or FP32 weights.
_BF16 = torch.bfloat16
_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)
# Bytes of one FP32 [4, 5120] row (a partial row of the backward's weight gradients).
_ROW_BYTES = STREAMS * HIDDEN * 4
# The records the launches use, looked up once: the forward record of each summation and ``dw_reduce`` with its grid
# (the same for every call).  The backward record follows the call (:func:`.cake_jit.backward_record`).
_FWD_RECORDS: dict[str, str] = {acc: cake_jit.record("fwd", acc) for acc in cake_jit.ACC_MODES["fwd"]}
_DW_REDUCE_RECORD: str = cake_jit.record("dw_reduce")
_DW_REDUCE_GRID: tuple[int, int, int] = cake_jit.dw_reduce_grid()

# This package's module prefix (the vendored name of a vendored copy).
_PACKAGE = __name__.rpartition(".")[0]
# The public module the entry points are documented under.
_MODULE = "cake_pplx.cake_engram_gate_nondet_train"


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


@functools.cache
def _resident(name: str, device_index: int) -> int:
    """The resident clusters of the ``fwd`` or ``bwd`` record ``name`` on one CUDA device
    (:func:`.cake_jit.resident_clusters`), looked up once per record and device."""
    return cake_jit.resident_clusters(name, device_arch(device_index), sm_count(device_index))


def _clusters(name: str, T: int, device_index: int) -> int:
    """:func:`.cake_jit.clusters` of record ``name`` for ``T`` tokens on one CUDA device: the resident clusters, never
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


# The kernel parameters each stage's launcher passes, in order (then the grid): every record of a stage takes these.
_STAGE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "fwd": ("x", "key", "value", "q_weight", "k_weight", "mask", "out", "T", "eps", "clamp", "wmode"),
    "bwd": (
        "dout",
        "x",
        "key",
        "value",
        "q_weight",
        "k_weight",
        "mask",
        "dx",
        "dkey",
        "dvalue",
        "dw_part",
        "T",
        "eps",
        "clamp",
        "wmode",
    ),
    "dw_reduce": ("dw_part", "q_weight", "k_weight", "dq", "dk", "NP", "wmode"),
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


# ----------------------------------------------------------------------------------------------------------------
# Stage launchers: caller-owned buffers, the current stream, no allocation (but the absent-mask stand-in), no check
# of the operands


def launch_fwd(
    name: str,
    *,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    wmode: int,
    mask: Optional[torch.Tensor],
    out: torch.Tensor,
    eps: float,
    clamp_value: float,
    clusters: int,
) -> None:
    """Launch the forward record ``name`` over ``T = x.shape[0] >= 1`` tokens on ``clusters`` four-CTA clusters:
    ``out = bf16(x + gate * value)`` from ``x`` (BF16 ``[T, 4, 5120]``, contiguous), ``key`` (BF16 ``[T, 4, 5120]``,
    token stride a multiple of 8), ``value`` (BF16 ``[T, 5120]``, same), the weights' 32-bit words ``q_weight`` and
    ``k_weight`` (:func:`_words`) with their mode ``wmode`` (:func:`_wmode`, bits 0 and 1) and ``mask`` (bool ``[T]``
    or None)."""
    grid = cake_jit.CLUSTER_CTAS * int(clusters)
    args = (x, key, value, q_weight, k_weight, mask, out, int(x.shape[0]), float(eps), float(clamp_value), int(wmode))
    _run(name, "fwd", x.get_device(), (*args, grid, 1, 1))


def launch_bwd(
    name: str,
    *,
    dout: torch.Tensor,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    wmode: int,
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
    and ``dw``, FP32 ``[clusters, 4, 5120]``: row ``c`` is cluster ``c``'s sum of ``c h k`` over its tokens, written
    in full (zero for a cluster without a token).  ``q_weight`` and ``k_weight`` are the weights' 32-bit words
    (:func:`_words`) and ``wmode`` their mode (:func:`_wmode`)."""
    grid = cake_jit.CLUSTER_CTAS * int(clusters)
    args = (dout, x, key, value, q_weight, k_weight, mask, dx, dkey, dvalue, dw)
    scalars = (int(x.shape[0]), float(eps), float(clamp_value), int(wmode))
    _run(name, "bwd", x.get_device(), (*args, *scalars, grid, 1, 1))


def launch_dw_reduce(
    name: str,
    *,
    dw: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    wmode: int,
) -> None:
    """Launch the finalize record ``name``: ``dq = (sum_p dw[p]) * k_weight`` and ``dk = (sum_p dw[p]) * q_weight``,
    ``[4, 5120]``, from the ``NP`` partial rows ``dw`` (FP32 ``[NP, 4, 5120]``) summed in one fixed order.
    ``q_weight``, ``k_weight``, ``dq`` and ``dk`` are 32-bit words (:func:`_words`): ``wmode`` bits 0 and 1 say which
    weight is BF16 (:func:`_wmode`), bits 2 and 3 whether ``dq`` and ``dk`` are BF16 (each rounded once) or FP32."""
    args = (dw, q_weight, k_weight, dq, dk, int(dw.shape[0]), int(wmode), *cake_jit.dw_reduce_grid())
    _run(name, "dw_reduce", dw.get_device(), args)


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
    if t.shape != (T, STREAMS, HIDDEN):
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
        if stride[0] % _STRIDE_ELEMENTS:
            raise ValueError(f"{name} needs a token stride that is a multiple of {_STRIDE_ELEMENTS}, got {stride[0]}")
        if output and T > 1 and stride[0] < STREAMS * HIDDEN:
            raise ValueError(f"{name} is an output: its token rows must not overlap, got the token stride {stride[0]}")
    _aligned(name, t)


def _value_layout(name: str, t: torch.Tensor, T: int, *, output: bool) -> None:
    """A BF16 ``[T, 5120]`` operand: unit column stride and a token stride that is a multiple of 8 elements (an
    output's rows must not overlap); 16-byte aligned."""
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {t.dtype}")
    if t.shape != (T, HIDDEN):
        raise ValueError(f"{name} must be [T, {HIDDEN}] with T = {T}, got {tuple(t.shape)}")
    if T:
        stride = t.stride()
        if stride[1] != 1:
            raise ValueError(f"{name} must have the column stride 1, got strides {stride}")
        if stride[0] % _STRIDE_ELEMENTS:
            raise ValueError(f"{name} needs a token stride that is a multiple of {_STRIDE_ELEMENTS}, got {stride[0]}")
        if output and T > 1 and stride[0] < HIDDEN:
            raise ValueError(f"{name} is an output: its token rows must not overlap, got the token stride {stride[0]}")
    _aligned(name, t)


def _dot_acc(dot_sum: Any) -> str:
    """The ``fwd`` record suffix of the ``dot_sum`` choice: ``TypeError`` for a non-``str``, ``ValueError`` for another
    ``str``; both name the choices."""
    if not isinstance(dot_sum, str):
        raise TypeError(f"dot_sum must be a str, one of {list(_DOT_SUMS)}, got {type(dot_sum).__name__}")
    acc = _DOT_SUMS.get(dot_sum)
    if acc is None:
        raise ValueError(f"dot_sum must be one of {list(_DOT_SUMS)}, got {dot_sum!r}")
    return acc


def _check_scalars(eps: Any, clamp_value: Any) -> tuple[float, float]:
    if type(eps) is float and type(clamp_value) is float and 0.0 <= eps < math.inf and 0.0 <= clamp_value < math.inf:
        return eps, clamp_value
    try:
        eps_f, clamp_f = float(eps), float(clamp_value)
    except (TypeError, ValueError):
        raise ValueError(f"eps and clamp_value must be real numbers, got {eps!r} and {clamp_value!r}") from None
    if not (math.isfinite(eps_f) and eps_f >= 0.0):
        raise ValueError(f"eps must be finite and non-negative, got {eps}")
    if not (math.isfinite(clamp_f) and clamp_f >= 0.0):
        raise ValueError(f"clamp_value must be finite and non-negative, got {clamp_value}")
    return eps_f, clamp_f


def _valid_tokens(x: Any, key: Any, value: Any, q_weight: Any, k_weight: Any, token_mask: Any) -> int:
    """The token count ``T`` of forward tensors that pass every tensor check of :func:`_check_operands`, read in one
    pass; -1 for anything else (the caller then runs those checks, which name the first one that fails)."""
    tensor = torch.Tensor
    if not (
        isinstance(x, tensor)
        and isinstance(key, tensor)
        and isinstance(value, tensor)
        and isinstance(q_weight, tensor)
        and isinstance(k_weight, tensor)
        and x.is_cuda
        and x.dtype == _BF16
        and x.dim() == 3
    ):
        return -1
    T, streams, hidden = x.shape
    index = x.get_device()
    if not (
        streams == STREAMS
        and hidden == HIDDEN
        and T <= _MAX_TOKENS
        and x.is_contiguous()
        and not x.data_ptr() % _ALIGNMENT
        and key.is_cuda
        and key.get_device() == index
        and key.dtype == _BF16
        and key.shape == (T, STREAMS, HIDDEN)
        and not key.data_ptr() % _ALIGNMENT
        and value.is_cuda
        and value.get_device() == index
        and value.dtype == _BF16
        and value.shape == (T, HIDDEN)
        and not value.data_ptr() % _ALIGNMENT
        and q_weight.is_cuda
        and q_weight.get_device() == index
        and q_weight.dtype in _WEIGHT_DTYPES
        and q_weight.shape == (STREAMS, HIDDEN)
        and k_weight.is_cuda
        and k_weight.get_device() == index
        and k_weight.dtype in _WEIGHT_DTYPES
        and k_weight.shape == (STREAMS, HIDDEN)
    ):
        return -1
    if T:
        key_stride, value_stride = key.stride(), value.stride()
        if not (
            key_stride[2] == 1
            and key_stride[1] == HIDDEN
            and not key_stride[0] % _STRIDE_ELEMENTS
            and value_stride[1] == 1
            and not value_stride[0] % _STRIDE_ELEMENTS
        ):
            return -1
    if token_mask is not None and not (
        isinstance(token_mask, tensor)
        and token_mask.is_cuda
        and token_mask.get_device() == index
        and token_mask.dtype == torch.bool
        and token_mask.shape == (T,)
    ):
        return -1
    return T


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
    T = _valid_tokens(x, key, value, q_weight, k_weight, token_mask)
    if T >= 0:
        eps_f, clamp_f = _check_scalars(eps, clamp_value)
        return T, eps_f, clamp_f
    x = _tensor("x", x)
    operands = [("x", x), ("key", _tensor("key", key)), ("value", _tensor("value", value))]
    operands += [("q_weight", _tensor("q_weight", q_weight)), ("k_weight", _tensor("k_weight", k_weight))]
    if token_mask is not None:
        operands.append(("token_mask", _tensor("token_mask", token_mask)))
    index = x.get_device() if x.is_cuda else -1
    for name, t in operands:
        if index < 0 or not t.is_cuda or t.get_device() != index:
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
        if t.shape != (STREAMS, HIDDEN):
            raise ValueError(f"{name} must be [{STREAMS}, {HIDDEN}], got {tuple(t.shape)}")
    if token_mask is not None:
        if token_mask.dtype != torch.bool:
            raise ValueError(f"token_mask must be torch.bool, got {token_mask.dtype}")
        if token_mask.shape != (T,):
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
    return _kv_views(kv)


def _kv_views(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(key, value)``: the views ``kv[:, :20480].unflatten(1, (4, 5120))`` and ``kv[:, 20480:]`` (the same sizes and
    offsets, the same strides on every dimension with more than one element, ``kv``'s row stride on the first) of a
    ``[T, 25600]`` buffer with the column stride 1, made directly (two views instead of three) and without checks:
    :func:`_split_kv` checks ``kv`` first; the backward's ``dkv`` is allocated so."""
    s0, offset = kv.stride(0), kv.storage_offset()
    T = kv.shape[0]
    key = kv.as_strided((T, STREAMS, HIDDEN), (s0, HIDDEN, 1), offset)
    return key, kv.as_strided((T, HIDDEN), (s0, 1), offset + STREAMS * HIDDEN)


# ----------------------------------------------------------------------------------------------------------------
# Host operands


def _mask(token_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The mask the kernels read: contiguous bool ``[T]``, or None."""
    return None if token_mask is None else token_mask.contiguous()


def _words(t: torch.Tensor) -> torch.Tensor:
    """The 32-bit words the kernels read of ``t``, a BF16 or FP32 ``[4, 5120]`` weight or weight gradient: an
    ``int32`` view (``[4, 2560]`` for BF16, two columns a word; ``[4, 5120]`` for FP32) of ``t`` itself when it is
    contiguous and 16-byte aligned, of a copy otherwise."""
    if t.requires_grad and torch.is_grad_enabled():
        t = t.detach()
    t = t.contiguous()
    if t.data_ptr() % _ALIGNMENT:
        t = t.clone()
    return t.view(torch.int32)


def _wmode(q: torch.dtype, k: torch.dtype, dq: torch.dtype = torch.float32, dk: torch.dtype = torch.float32) -> int:
    """The kernels' mode word: bit 0 ``q_weight`` is BF16, bit 1 ``k_weight``, bit 2 ``dq_weight``, bit 3
    ``dk_weight`` (each FP32 otherwise)."""
    bf16 = torch.bfloat16
    return int(q == bf16) | int(k == bf16) << 1 | int(dq == bf16) << 2 | int(dk == bf16) << 3


def cake_engram_gate_nondet_train_workspace_bytes(T: int, device: Union[torch.device, int, str, None] = None) -> int:
    """Bytes of device memory one backward call over ``T`` tokens on ``device`` (None: the current CUDA device)
    allocates beside its operands and outputs: the ``[clusters, 4, 5120]`` FP32 weight-gradient partial rows (one
    per cluster of the backward grid), 80 KiB each (0 for ``T == 0``), and a copy of a weight that is not contiguous
    or not 16-byte aligned (none otherwise: the kernels read the weights in place, in their dtypes).  A forward call
    allocates its output alone (and such a copy); the state kept for the backward is the inputs alone.  The backward,
    and so its workspace, is the same for either ``dot_sum`` of the forward."""
    T = int(T)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if T == 0:
        return 0
    index = _device_index(device)
    if index is None:
        raise ValueError(f"device must be a CUDA device, got {device}")
    parts = cake_jit.clusters(cake_jit.record("bwd"), T, device_arch(index), sm_count(index))
    return _ROW_BYTES * parts


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
    dot_sum: str,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float, float]:
    """The checked forward: ``(out, eps, clamp_value)`` with the scalars as checked (floats)."""
    acc = _dot_acc(dot_sum)
    T, eps_f, clamp_f = _check_operands(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
    if out is None:
        out = torch.empty(T, STREAMS, HIDDEN, dtype=torch.bfloat16, device=x.device)
    else:
        _check_output("out", out, T, x.device, "contiguous")
    if T == 0:
        return out, eps_f, clamp_f
    index = x.get_device()
    device_arch(index)  # RuntimeError on a device outside ARCHES
    name = _FWD_RECORDS[acc]
    clusters = _clusters(name, T, index)
    # The weights as stored (their 32-bit words), the product formed in the kernel: one launch.
    args = (x, key, value, _words(q_weight), _words(k_weight), _mask(token_mask), out, T, eps_f, clamp_f)
    wmode = _wmode(q_weight.dtype, k_weight.dtype)
    _run(name, "fwd", index, (*args, wmode, cake_jit.CLUSTER_CTAS * clusters, 1, 1))
    return out, eps_f, clamp_f


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
    """The explicit backward: every operand checked, FP32 weight gradients."""
    T, eps_f, clamp_f = _check_operands(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value)
    device = x.device
    dout = _tensor("dout", dout)
    _on_device("dout", dout, device)
    _streams_layout("dout", dout, T, contiguous=True, output=False)
    for name, t, kind in (("dx", dx, "contiguous"), ("dkey", dkey, "strided"), ("dvalue", dvalue, "value")):
        if t is not None:
            _check_output(name, t, T, device, kind)
    fp32 = torch.float32
    return _launch_backward(
        dout, x, key, value, q_weight, k_weight, token_mask, T, eps_f, clamp_f, dx, dkey, dvalue, api, fp32, fp32
    )


def _launch_backward(
    dout: torch.Tensor,
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    T: int,
    eps: float,
    clamp_value: float,
    dx: Optional[torch.Tensor],
    dkey: Optional[torch.Tensor],
    dvalue: Optional[torch.Tensor],
    api: str,
    dq_dtype: torch.dtype,
    dk_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward of checked operands over ``T`` tokens: ``(dx, dkey, dvalue, dq_weight, dk_weight)``, the weight
    gradients in ``dq_dtype`` and ``dk_dtype`` (FP32 or BF16).  Every buffer is allocated and both argument tuples are
    built before the first launch, so the launches go out back to back."""
    device = x.device
    if T == 0:
        # No token: nothing to launch; zero weight gradients.
        empty = functools.partial(torch.empty, dtype=torch.bfloat16, device=device)
        return (
            empty((0, STREAMS, HIDDEN)) if dx is None else dx,
            empty((0, STREAMS, HIDDEN)) if dkey is None else dkey,
            empty((0, HIDDEN)) if dvalue is None else dvalue,
            torch.zeros((STREAMS, HIDDEN), dtype=dq_dtype, device=device),
            torch.zeros((STREAMS, HIDDEN), dtype=dk_dtype, device=device),
        )
    index = device.index
    arch = device_arch(index)  # RuntimeError on a device outside ARCHES
    # (Sizes as separate arguments: torch parses them faster than one tuple.)
    if dx is None:
        dx = torch.empty(T, STREAMS, HIDDEN, dtype=torch.bfloat16, device=device)
    if dkey is None:
        dkey = torch.empty(T, STREAMS, HIDDEN, dtype=torch.bfloat16, device=device)
    if dvalue is None:
        dvalue = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=device)
    # One backward for either forward sum: it recomputes P as a compensated dot product, in the program that is the
    # faster one for the call (a token mask given or not, the tokens, the architecture); both give the same bits.
    name = cake_jit.backward_record(arch, token_mask is not None, T)
    clusters = _clusters(name, T, index)
    # One row per cluster, each written in full by its cluster.
    dw = torch.empty(clusters, STREAMS, HIDDEN, dtype=torch.float32, device=device)
    dq = torch.empty(STREAMS, HIDDEN, dtype=dq_dtype, device=device)
    dk = torch.empty(STREAMS, HIDDEN, dtype=dk_dtype, device=device)
    # The weights as stored (their 32-bit words), the product formed in the kernels.
    qw, kw = _words(q_weight), _words(k_weight)
    wmode = _wmode(q_weight.dtype, k_weight.dtype, dq_dtype, dk_dtype)
    grid = cake_jit.CLUSTER_CTAS * clusters
    bwd = (dout, x, key, value, qw, kw, _mask(token_mask), dx, dkey, dvalue, dw, T, eps, clamp_value, wmode & 3)
    finish = (dw, qw, kw, dq.view(torch.int32), dk.view(torch.int32), clusters, wmode)
    _run(name, "bwd", index, (*bwd, grid, 1, 1))
    _run(_DW_REDUCE_RECORD, "dw_reduce", index, (*finish, *_DW_REDUCE_GRID))
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
    dot_sum: str = "fp32",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The forward without autograd: ``out`` BF16 ``[T, 4, 5120]`` (``out`` given: written in place and returned).

    ``x`` BF16 ``[T, 4, 5120]`` contiguous; ``key`` BF16 ``[T, 4, 5120]`` and ``value`` BF16 ``[T, 5120]``,
    contiguous or views into a packed ``[T, 25600]`` buffer (unit column stride, stream stride 5120, token stride a
    multiple of 8); ``q_weight`` and ``k_weight`` BF16 or FP32 ``[4, 5120]``; ``token_mask`` bool ``[T]`` or None;
    every tensor on one CUDA device, every BF16 operand 16-byte aligned.  ``dot_sum``: ``"fp32"`` (default) or
    ``"compensated"`` (opt-in, slower), how ``P`` is summed.  ``T == 0`` launches nothing.  The forward never consults
    ``torch.use_deterministic_algorithms``."""
    return _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum, out)[0]


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
    ``dout`` (BF16 ``[T, 4, 5120]``, contiguous); the forward's own output is not needed.  It recomputes ``P`` as a
    compensated dot product: the same backward for either ``dot_sum`` of the forward.

    ``dx`` BF16 ``[T, 4, 5120]`` (contiguous), ``dkey`` and ``dvalue`` BF16, each laid out as ``key`` and ``value``
    may be (caller views into one ``[T, 25600]`` gradient buffer are written in place), allocated when not given;
    ``dq_weight`` and ``dk_weight`` FP32 ``[4, 5120]``.  ``T == 0`` returns zero weight gradients and launches
    nothing.
    ``dq_weight`` and ``dk_weight`` promise no bits run to run; nothing consults
    ``torch.use_deterministic_algorithms``.
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
    """The gradient of ``out`` as the backward reads it: BF16, contiguous, 16-byte aligned, on ``x``'s device and of
    its shape."""
    if dout is None:
        return torch.zeros_like(x)
    if (
        dout.dtype == torch.bfloat16
        and dout.is_cuda
        and dout.get_device() == x.get_device()
        and dout.shape == x.shape
        and dout.is_contiguous()
        and not dout.data_ptr() % _ALIGNMENT
    ):
        return dout
    if dout.dtype != torch.bfloat16:
        dout = dout.to(torch.bfloat16)
    if not dout.is_contiguous():
        dout = dout.contiguous()
    elif dout.data_ptr() % _ALIGNMENT:
        dout = dout.clone()
    _on_device("dout", dout, x.device)
    _streams_layout("dout", dout, int(x.shape[0]), contiguous=True, output=False)
    return dout


def _autograd_backward(
    dout: Optional[torch.Tensor],
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor],
    ctx: Any,
    dkey: Optional[torch.Tensor],
    dvalue: Optional[torch.Tensor],
    api: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward of an autograd entry: the saved operands were checked by its forward (autograd refuses a saved
    tensor changed in place since), so only ``dout`` is; the weight gradients come in each parameter's dtype."""
    return _launch_backward(
        _grad_out(dout, x),
        x,
        key,
        value,
        q_weight,
        k_weight,
        token_mask,
        int(x.shape[0]),
        ctx.eps,
        ctx.clamp_value,
        None,
        dkey,
        dvalue,
        api,
        q_weight.dtype,
        k_weight.dtype,
    )


class CakeEngramGateNondetFunction(torch.autograd.Function):
    """``out = engram_gate(x, key, value, q_weight, k_weight, token_mask)``, the forward's sum of ``P`` chosen by
    ``dot_sum``.  The saved set is the inputs alone (``ctx.save_for_backward``; ``eps`` and ``clamp_value`` on
    ``ctx``): the backward recomputes the norms and the dot product (compensated, whatever the forward's sum), so a
    checkpointed forward keeps nothing alive.  The weight gradients are returned in each parameter's dtype.
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
        dot_sum: str,
    ) -> torch.Tensor:
        checked = _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum)
        out, ctx.eps, ctx.clamp_value = checked
        ctx.save_for_backward(x, key, value, q_weight, k_weight, token_mask)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
        x, key, value, q_weight, k_weight, token_mask = ctx.saved_tensors
        dx, dkey, dvalue, dq, dk = _autograd_backward(
            dout, x, key, value, q_weight, k_weight, token_mask, ctx, None, None, "cake_engram_gate_nondet"
        )
        need = ctx.needs_input_grad
        return (
            dx if need[0] else None,
            dkey if need[1] else None,
            dvalue if need[2] else None,
            dq if need[3] else None,
            dk if need[4] else None,
            None,
            None,
            None,
            None,
        )


class CakeEngramGateNondetKvFunction(torch.autograd.Function):
    """``out = engram_gate(x, key, value, q_weight, k_weight, token_mask)`` with ``key`` and ``value`` the views of
    one packed ``kv [T, 25600]``.  The backward writes dK and dV straight into one new ``dkv [T, 25600]`` (no split
    and no concatenation copy).  The saved set is the inputs alone and the backward is the one for either ``dot_sum``,
    as in :class:`CakeEngramGateNondetFunction`."""

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
        dot_sum: str,
    ) -> torch.Tensor:
        key, value = _split_kv(kv)
        checked = _forward(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum)
        out, ctx.eps, ctx.clamp_value = checked
        ctx.save_for_backward(x, kv, q_weight, k_weight, token_mask)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:
        x, kv, q_weight, k_weight, token_mask = ctx.saved_tensors
        key, value = _kv_views(kv)
        dkv = torch.empty(kv.shape[0], KV_WIDTH, dtype=torch.bfloat16, device=kv.device)
        dkey, dvalue = _kv_views(dkv)
        dx, _, _, dq, dk = _autograd_backward(
            dout, x, key, value, q_weight, k_weight, token_mask, ctx, dkey, dvalue, "cake_engram_gate_nondet_kv"
        )
        need = ctx.needs_input_grad
        return (
            dx if need[0] else None,
            dkv if need[1] else None,
            dq if need[2] else None,
            dk if need[3] else None,
            None,
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
    dot_sum: str = "fp32",
) -> torch.Tensor:
    """The Engram context gate with autograd (:class:`CakeEngramGateNondetFunction`): ``out`` BF16
    ``[T, 4, 5120]``; operands and ``dot_sum`` as in :func:`cake_engram_gate_nondet_train_forward` (``dot_sum``
    selects the forward's sum of ``P``; the backward is the same for both)."""
    return CakeEngramGateNondetFunction.apply(x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum)


def cake_engram_gate_nondet_kv(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    dot_sum: str = "fp32",
) -> torch.Tensor:
    """The Engram context gate with autograd on the packed ``kv`` BF16 ``[T, 25600]`` (the four keys, then the
    value; unit column stride, token stride a multiple of 8) (:class:`CakeEngramGateNondetKvFunction`): ``out`` BF16
    ``[T, 4, 5120]``; the gradient of ``kv`` is one ``[T, 25600]`` buffer written by the backward kernel."""
    return CakeEngramGateNondetKvFunction.apply(x, kv, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum)
