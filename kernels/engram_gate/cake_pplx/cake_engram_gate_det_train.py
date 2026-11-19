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

Cake deterministic Engram context gate training kernels: four residual
streams of width 5120 (SM100 / SM103 / SM107).

Per token ``t`` and stream ``s`` (``H = 5120``), with every quantity in FP32
and each BF16 output rounded once, at its store::

    h = x[t, s]   k = key[t, s]   v = value[t]   w = q_weight[s] * k_weight[s]
    r_h = rsqrt(mean(h^2) + eps)   r_k = rsqrt(mean(k^2) + eps)
    P = sum_d w_d h_d k_d          dot = P r_h r_k / sqrt(H)
    gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))   (0 on a masked token)
    out[t, s] = bf16(h + gate v)

The derivative of the square root is 0 where ``|dot| <= clamp_value``; ``dot =
+0`` gives ``+sqrt(clamp_value)`` and ``-0`` gives ``-sqrt(clamp_value)``.  A
masked token (``token_mask[t]`` False) gives ``out[t] = x[t]`` and
``dx[t] = dout[t]`` exactly, zero ``dkey[t]`` and ``dvalue[t]``, and adds
nothing to the weight gradients.

The autograd entries and the explicit forward take ``dot_sum``, how the
forward sums ``P``.  The default, ``"compensated"``, is a compensated FP32 dot
product: each product split with an FMA, each running sum with an error-free
two-sum, the error words added once at the end.  ``"fp32"`` (opt-in) sums
``P`` in one FP32 word: the forward takes about 5-10% less time, depending
on the GPU (``T = 16384``, FP32 weights: 0.313 ms against 0.329 ms on
GB300, 0.197 ms against 0.218 ms on VR200) and, where the products nearly
cancel, rounds more output elements away from the correctly rounded result
(17802 of the 8.19 million output elements of 25 near-cancelling draws,
against 117; an FP32 evaluation of the formula: 21917).  The backward is the
same for both: it recomputes ``P`` as a compensated dot product, so the
gradients do not depend on ``dot_sum`` (the explicit backward does not take
it).  As measured on B200, GB200, GB300 and VR200, both choices meet every
accuracy bound of the acceptance tests: the median and the largest relative
L2 error against an FP64 reference, over random draws and over draws whose
dot products nearly cancel (``|dot|`` from 2e-6 to 2e-5), at most twice
those of an FP32 evaluation of the formula.  The ``out`` errors of the two
forwards agree to four digits (GB300, near-cancelling: 1.661e-3 median and
1.669e-3 largest, against bounds of 3.321e-3 and 3.338e-3).

Entries: :func:`cake_engram_gate_det` (with autograd; ``key`` and ``value``
contiguous or views into one packed ``[T, 25600]``),
:func:`cake_engram_gate_det_kv` (with autograd on the packed ``kv``: the
backward writes one ``dkv [T, 25600]`` with no split or concatenation copy),
:func:`cake_engram_gate_det_train_forward` /
:func:`cake_engram_gate_det_train_backward` (the two halves without
autograd), :func:`cake_engram_gate_det_train_workspace_bytes` and
:func:`is_cake_engram_gate_det_train_supported`.  The autograd classes
``CakeEngramGateDetFunction`` and ``CakeEngramGateDetKvFunction``
resolve to the backend's objects on first access.

The mode is part of the name: no entry takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  Every sum has one fixed order, so every
output is bitwise reproducible run to run on one device; nothing consults
``torch.use_deterministic_algorithms`` (there is nothing to alert about).

A stable API (``@cake_api``; no ``ExperimentalWarning``) of thin wrappers:
the checks, the launch plan, the autograd Functions and the generated kernels
live in the backend package ``cake_pplx._backends.cake_engram_gate_det_train``,
imported on first use; this module forwards every keyword.  Every switch is a
keyword argument; no environment variable is read.  The token count is a
run-time value: no recompilation, no device-to-host read, no host
synchronization; the calls are CUDA-graph capturable after one eager call
(of each ``dot_sum`` the graph uses).

Memory: the state kept for the backward is the inputs alone (the backward
recomputes the norms and the dot product).  A backward call allocates FP32
``[4, 5120]`` rows of weight-gradient partial sums beside its outputs and
reads the weights as stored:
:func:`cake_engram_gate_det_train_workspace_bytes`.

Usage::

    from cake_pplx.cake_engram_gate_det_train import cake_engram_gate_det_kv

    kv = projection(rows)                          # BF16 [T, 25600]: four keys, then the value
    out = cake_engram_gate_det_kv(x, kv, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)
    out.backward(dout)                             # x.grad, kv.grad [T, 25600], q_weight.grad, k_weight.grad
    # opt-in: the forward sums each dot product in one FP32 word (faster); the backward is the same
    out = cake_engram_gate_det_kv(x, kv, q_weight, k_weight, token_mask, dot_sum="fp32")
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ._cake_pplx_api_engram_gate import cake_api

if TYPE_CHECKING:  # resolved at run time by ``__getattr__`` (no module-level backend import)
    from ._backends.cake_engram_gate_det_train import (
        CakeEngramGateDetFunction,
        CakeEngramGateDetKvFunction,
    )

__all__ = [
    "ARCHES",
    "HIDDEN",
    "KV_WIDTH",
    "STREAMS",
    "CakeEngramGateDetFunction",
    "CakeEngramGateDetKvFunction",
    "cake_engram_gate_det",
    "cake_engram_gate_det_kv",
    "cake_engram_gate_det_train_backward",
    "cake_engram_gate_det_train_forward",
    "cake_engram_gate_det_train_workspace_bytes",
    "is_cake_engram_gate_det_train_supported",
]

# The built architectures and the fixed operand geometry (the backend's values).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STREAMS = 4
HIDDEN = 5120
KV_WIDTH = 25600

# Backend objects this module re-exports, resolved on first access (``__getattr__``).
_BACKEND_OBJECTS = frozenset({"CakeEngramGateDetFunction", "CakeEngramGateDetKvFunction"})


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use.
    from ._backends import cake_engram_gate_det_train  # noqa: PLC0415

    return cake_engram_gate_det_train


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_pkg(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BACKEND_OBJECTS)


@cake_api
def cake_engram_gate_det(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    dot_sum: str = "compensated",
) -> torch.Tensor:
    r"""The Engram context gate with autograd (``CakeEngramGateDetFunction``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, 4, 5120]``, contiguous: the four residual streams.
    key : torch.Tensor
        BF16 ``[T, 4, 5120]``: contiguous, or a view into a ``[T, n]`` buffer (unit column stride, stream stride
        5120, token stride a multiple of 8 elements), such as the first 20480 columns of the packed ``kv``.
    value : torch.Tensor
        BF16 ``[T, 5120]``: unit column stride, token stride a multiple of 8 elements (the last 5120 columns of
        ``kv`` qualify).
    q_weight, k_weight : torch.Tensor
        BF16 or FP32 ``[4, 5120]``, each independently; the kernels use the FP32 product
        ``FP32(q_weight) * FP32(k_weight)``.
    token_mask : torch.Tensor, optional
        Bool ``[T]``; ``False`` shuts the gate of the token.
    eps, clamp_value : float
        The RMS epsilon and the clamp of ``|dot|`` (run-time values; finite, non-negative).
    dot_sum : {"compensated", "fp32"}
        How the forward sums ``P = sum_d w_d h_d k_d``: ``"compensated"`` (default), a compensated FP32 dot product;
        ``"fp32"`` (opt-in), one FP32 word, faster, with more output elements rounded away from the correctly
        rounded result where the products nearly cancel.  Both meet the accuracy bounds as measured on B200, GB200,
        GB300 and VR200 (see the module notes); the backward is the same for both.

    Returns
    -------
    ``out`` BF16 ``[T, 4, 5120]``.  Its backward returns ``dx`` (BF16), ``dkey`` and ``dvalue`` (BF16, new
    contiguous tensors) and ``dq_weight`` / ``dk_weight`` in each parameter's dtype (summed in FP32).

    Raises
    ------
    TypeError
        ``dot_sum`` is not a ``str``.
    ValueError
        A tensor is not a CUDA tensor of the operands' device, or has another dtype, shape, layout or alignment
        than above; ``eps`` or ``clamp_value`` is not finite and non-negative; ``dot_sum`` is another value.
    RuntimeError
        The device is not one of ``ARCHES``.

    Notes
    -----
    Every output is bitwise reproducible run to run on one device; ``torch.use_deterministic_algorithms`` is not
    consulted.  ``T == 0`` launches nothing and gives zero weight gradients.
    """
    return _pkg().cake_engram_gate_det(
        x, key, value, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum
    )


@cake_api
def cake_engram_gate_det_kv(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    dot_sum: str = "compensated",
) -> torch.Tensor:
    r"""The Engram context gate with autograd on the packed projection output (``CakeEngramGateDetKvFunction``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, 4, 5120]``, contiguous.
    kv : torch.Tensor
        BF16 ``[T, 25600]``: the four keys (columns ``0:20480``), then the value (``20480:25600``); unit column
        stride, token stride a multiple of 8 elements.
    q_weight, k_weight, token_mask, eps, clamp_value, dot_sum
        As in :func:`cake_engram_gate_det`.

    Returns
    -------
    ``out`` BF16 ``[T, 4, 5120]``.  Its backward writes the key and value gradients straight into one new BF16
    ``dkv [T, 25600]`` (the gradient of ``kv``): no split and no concatenation copy.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As in :func:`cake_engram_gate_det`.

    Notes
    -----
    Bitwise reproducible run to run; ``torch.use_deterministic_algorithms`` is not consulted.
    """
    return _pkg().cake_engram_gate_det_kv(
        x, kv, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum
    )


@cake_api
def cake_engram_gate_det_train_forward(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
    dot_sum: str = "compensated",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The forward without autograd.

    Parameters
    ----------
    x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum
        As in :func:`cake_engram_gate_det`.
    out : torch.Tensor, optional
        BF16 ``[T, 4, 5120]``, contiguous, 16-byte aligned: written in place and returned.

    Returns
    -------
    ``out`` BF16 ``[T, 4, 5120]``.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As in :func:`cake_engram_gate_det`.

    Notes
    -----
    The forward never consults ``torch.use_deterministic_algorithms``; nothing is saved (the backward takes the
    inputs again).
    """
    return _pkg().cake_engram_gate_det_train_forward(
        x, key, value, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum, out=out
    )


@cake_api
def cake_engram_gate_det_train_backward(
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
    r"""The backward without autograd, from the forward's inputs (its output is not needed).

    Parameters
    ----------
    dout : torch.Tensor
        BF16 ``[T, 4, 5120]``, contiguous: the gradient of ``out``.
    x, key, value, q_weight, k_weight, token_mask, eps, clamp_value
        The forward's operands, as in :func:`cake_engram_gate_det`.  The backward recomputes ``P`` as a
        compensated dot product: it is the same after either ``dot_sum`` of the forward.
    dx : torch.Tensor, optional
        BF16 ``[T, 4, 5120]``, contiguous.
    dkey, dvalue : torch.Tensor, optional
        BF16 ``[T, 4, 5120]`` and ``[T, 5120]`` laid out as ``key`` and ``value`` may be: caller views into one
        ``[T, 25600]`` gradient buffer are written in place (rows must not overlap).

    Returns
    -------
    ``(dx, dkey, dvalue, dq_weight, dk_weight)``: the given or new BF16 gradients and FP32 ``[4, 5120]`` weight
    gradients (``dq_weight = dw * k_weight``, ``dk_weight = dw * q_weight``, ``dw = sum_t c h k`` in FP32).

    Raises
    ------
    ValueError, RuntimeError
        As in :func:`cake_engram_gate_det`.

    Notes
    -----
    Bitwise reproducible run to run on one device; ``torch.use_deterministic_algorithms`` is not consulted.
    ``T == 0`` launches nothing and returns zero weight gradients.
    """
    return _pkg().cake_engram_gate_det_train_backward(
        dout,
        x,
        key,
        value,
        q_weight,
        k_weight,
        token_mask,
        eps=eps,
        clamp_value=clamp_value,
        dx=dx,
        dkey=dkey,
        dvalue=dvalue,
    )


@cake_api
def cake_engram_gate_det_train_workspace_bytes(T: int, device: Union[torch.device, int, str, None] = None) -> int:
    r"""Bytes of device memory one backward call allocates beside its operands and outputs.

    Parameters
    ----------
    T : int
        The token count.
    device : torch.device, int or str, optional
        The CUDA device (None: the current one).
        The number of partial rows follows its architecture and SM count (``RuntimeError`` on a device outside
        ``ARCHES``).

    Returns
    -------
    80 KiB (``4 * 5120`` FP32) for each ``[4, 5120]`` partial row of the weight gradient (one per cluster of the
    backward grid); 0 for ``T == 0``.  The backward reads BF16 and FP32 weights as stored (a weight that is not
    contiguous or not 16-byte aligned is copied first).  A forward call reads them the same way and allocates its
    output alone, and the state kept for the backward is the inputs alone.

    Notes
    -----
    Workspace sizes do not depend on ``torch.use_deterministic_algorithms`` or on the forward's ``dot_sum`` (one
    backward serves both).
    """
    return _pkg().cake_engram_gate_det_train_workspace_bytes(T, device)


@cake_api
def is_cake_engram_gate_det_train_supported(device: Union[torch.device, int, str, None] = None) -> bool:
    r"""Whether ``device`` (None: the current CUDA device) is a CUDA device of one of ``ARCHES``.

    Parameters
    ----------
    device : torch.device, int or str, optional
        The device to ask about.

    Returns
    -------
    True on SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200); False otherwise and without CUDA.

    Notes
    -----
    Support does not depend on ``torch.use_deterministic_algorithms``.
    """
    return _pkg().is_cake_engram_gate_det_train_supported(device)
