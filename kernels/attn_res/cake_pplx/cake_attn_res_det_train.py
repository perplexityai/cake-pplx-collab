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

Cake deterministic attention-residual (AttnRes) training kernels: a
softmax-weighted sum of ``n`` BF16 values of width 7168, ``1 <= n <= 9``
(SM100 / SM103 / SM107).

Per token ``t`` (``H = 7168``), with every quantity in FP32 and each BF16
output rounded once, at its store::

    r_j = 1 / sqrt(mean_c v_j[c]^2 + eps)   d_j = sum_c v_j[c] w[c]
    p   = softmax over j of (r_j d_j)       (the largest term subtracted first)
    out = bf16(sum_j p_j v_j)

and, given ``dout``::

    dp_j = sum_c dout[c] v_j[c]   ds_j = p_j (dp_j - sum_i p_i dp_i)
    dv_j = bf16(p_j dout + ds_j r_j w - (ds_j d_j r_j^3 / H) v_j)
    dw   = sum over t and j of ds_j r_j v_j        (FP32)

The values are the closed block states and the running sum of a block
(Kimi-K3 attention residuals), each a separate BF16 ``[T, 7168]`` tensor:
nothing is stacked or copied.  ``w`` is FP32 ``[7168]`` (the caller's product
of the norm and projection weights); ``dw`` is its FP32 gradient, the one sum
across tokens.  The norms, dot products and the softmax are FP32; no sum uses
FP64.  An all-zero value row gives ``r_j = 1 / sqrt(eps)`` and ``d_j = 0``: a
finite result.

Entries: :func:`cake_attn_res_det` (with autograd),
:func:`cake_attn_res_det_train_forward` /
:func:`cake_attn_res_det_train_backward` (the two halves without
autograd), :func:`cake_attn_res_det_train_workspace_bytes`,
:func:`cake_attn_res_det_train_retained_bytes` and
:func:`is_cake_attn_res_det_train_supported`.  The autograd class
``CakeAttnResDetFunction`` resolves to the backend's object on first
access.

The mode is part of the name: no entry takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  Every sum has one fixed order: each launch
group of the backward writes its own FP32 row of ``dw`` partial sums and a
second kernel adds the rows in one order, so every output repeats bit for bit
for identical inputs on one device.  Nothing consults
``torch.use_deterministic_algorithms`` (there is nothing to alert about).

A stable API (``@cake_api``; no ``ExperimentalWarning``) of thin wrappers:
the checks, the launch plan, the autograd Function and the generated kernels
live in the backend package ``cake_pplx._backends.cake_attn_res_det_train``,
imported on first use; this module forwards every keyword.  Every switch is a
keyword argument; no environment variable is read.  The token count, ``n``
and ``eps`` are run-time values: one program per stage serves them all (no
recompilation), with no device-to-host read and no host synchronization; the
calls are CUDA-graph capturable after one eager call.

Memory: the state kept for the backward is references to the inputs and the
forward's FP32 ``p``, ``r`` and ``d`` (``[T, n]`` each, ``12 T n`` bytes:
:func:`cake_attn_res_det_train_retained_bytes`).  A backward call allocates
FP32 ``[7168]`` rows of ``dw`` partial sums beside its outputs, one per launch
group of its grid (:func:`cake_attn_res_det_train_workspace_bytes`).

Usage::

    from cake_pplx.cake_attn_res_det_train import cake_attn_res_det

    values = [*closed_blocks, running_sum]         # n BF16 [T, 7168] tensors
    w = norm.weight * proj.weight                  # FP32 [7168]
    h = cake_attn_res_det(values, w, eps=1e-5)  # BF16 [T, 7168]
    h.backward(dh)                                 # every value's .grad (BF16), w's grad (FP32)
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ._cake_pplx_api_attn_res import cake_api

if TYPE_CHECKING:  # resolved at run time by ``__getattr__`` (no module-level backend import)
    from ._backends.cake_attn_res_det_train import CakeAttnResDetFunction

__all__ = [
    "ARCHES",
    "HIDDEN",
    "MAX_VALUES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeAttnResDetFunction",
    "cake_attn_res_det",
    "cake_attn_res_det_train_backward",
    "cake_attn_res_det_train_forward",
    "cake_attn_res_det_train_retained_bytes",
    "cake_attn_res_det_train_workspace_bytes",
    "is_cake_attn_res_det_train_supported",
]

# The built architectures, their compute capabilities and the fixed operand geometry (the backend's values).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN = 7168
MAX_VALUES = 9

# Backend objects this module re-exports, resolved on first access (``__getattr__``).
_BACKEND_OBJECTS = frozenset({"CakeAttnResDetFunction"})


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use.
    from ._backends import cake_attn_res_det_train  # noqa: PLC0415

    return cake_attn_res_det_train


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_pkg(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BACKEND_OBJECTS)


@cake_api
def cake_attn_res_det(values: Sequence[torch.Tensor], w: torch.Tensor, *, eps: float = 1e-5) -> torch.Tensor:
    r"""The attention residual with autograd (``CakeAttnResDetFunction``).

    Parameters
    ----------
    values : list or tuple of torch.Tensor
        ``n`` (1 to 9) BF16 ``[T, 7168]`` tensors, each contiguous and 16-byte aligned, on one CUDA device: the closed
        block states, then the running sum.  They stay separate (nothing is stacked).
    w : torch.Tensor
        FP32 ``[7168]`` (the product of the norm and projection weights).
    eps : float
        The RMS epsilon (a run-time value; finite, non-negative; 1e-5 by default).

    Returns
    -------
    ``out`` BF16 ``[T, 7168]``.  Its backward returns every ``dv_j`` (BF16, new contiguous tensors) and ``dw`` (FP32).

    Raises
    ------
    ValueError
        ``values`` is not a list or tuple of 1 to 9 tensors, a tensor is not a CUDA tensor of the first value's device,
        or has another dtype, shape, layout or alignment than above; ``eps`` is not finite and non-negative.
    RuntimeError
        The device is not one of ``ARCHES``.

    Notes
    -----
    The autograd node saves references to ``values`` and ``w`` and the forward's FP32 ``p``, ``r``, ``d``
    (``[T, n]`` each).  Every output repeats bit for bit for identical inputs on one device;
    ``torch.use_deterministic_algorithms`` is not consulted.  ``T == 0`` launches nothing and gives a zero ``dw``.
    """
    return _pkg().cake_attn_res_det(values, w, eps=eps)


@cake_api
def cake_attn_res_det_train_forward(
    values: Sequence[torch.Tensor],
    w: torch.Tensor,
    *,
    eps: float = 1e-5,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""The forward without autograd.

    Parameters
    ----------
    values, w, eps
        As in :func:`cake_attn_res_det`.
    out : torch.Tensor, optional
        BF16 ``[T, 7168]``, contiguous, 16-byte aligned: written in place and returned.

    Returns
    -------
    ``(out, p, r, d)``: ``out`` BF16 ``[T, 7168]`` and the state the backward reads, FP32 ``[T, n]`` each (row-major):
    the softmax weights ``p``, the norms ``r_j = 1 / sqrt(mean(v_j^2) + eps)`` and the dot products ``d_j = v_j . w``.

    Raises
    ------
    ValueError, RuntimeError
        As in :func:`cake_attn_res_det`.

    Notes
    -----
    The forward never consults ``torch.use_deterministic_algorithms``.  ``T == 0`` launches nothing.
    """
    return _pkg().cake_attn_res_det_train_forward(values, w, eps=eps, out=out)


@cake_api
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
    r"""The backward without autograd, from the forward's inputs and state (its output is not needed).

    Parameters
    ----------
    dout : torch.Tensor
        BF16 ``[T, 7168]``, contiguous: the gradient of ``out``.
    values, w
        The forward's operands, as in :func:`cake_attn_res_det`.
    p, r, d : torch.Tensor
        The forward's state, FP32 ``[T, n]`` each, contiguous (:func:`cake_attn_res_det_train_forward`).
    dv : list or tuple of torch.Tensor, optional
        ``n`` BF16 ``[T, 7168]`` tensors, contiguous, 16-byte aligned: written in place and returned.

    Returns
    -------
    ``(dv, dw)``: a list of ``n`` BF16 ``[T, 7168]`` value gradients and the FP32 ``[7168]`` gradient of ``w``.

    Raises
    ------
    ValueError, RuntimeError
        As in :func:`cake_attn_res_det`.

    Notes
    -----
    Every output repeats bit for bit for identical inputs on one device; ``torch.use_deterministic_algorithms`` is not
    consulted.  ``T == 0`` launches nothing and returns a zero ``dw``.
    """
    return _pkg().cake_attn_res_det_train_backward(dout, values, w, p, r, d, dv=dv)


@cake_api
def cake_attn_res_det_train_workspace_bytes(T: int, n: int, device: Union[torch.device, int, str, None] = None) -> int:
    r"""Bytes of device memory one backward call allocates beside its operands and outputs.

    Parameters
    ----------
    T : int
        The token count.
    n : int
        The number of values (1 to 9).
    device : torch.device, int or str, optional
        The CUDA device (None: the current one).  The number of partial rows follows its architecture and SM count: a
        ``ValueError`` for a device that is not a CUDA device, a ``RuntimeError`` on a CUDA device outside ``ARCHES``.

    Returns
    -------
    28 KiB (``7168`` FP32) for each ``[7168]`` row of ``dw`` partial sums, one per launch group of the backward grid; 0
    for ``T == 0``.  It does not depend on ``n``.

    Notes
    -----
    The workspace does not depend on ``torch.use_deterministic_algorithms``.
    """
    return _pkg().cake_attn_res_det_train_workspace_bytes(T, n, device)


@cake_api
def cake_attn_res_det_train_retained_bytes(T: int, n: int) -> int:
    r"""Bytes of the state one forward call keeps for its backward beside references to its inputs.

    Parameters
    ----------
    T : int
        The token count.
    n : int
        The number of values (1 to 9).

    Returns
    -------
    ``12 * T * n``: ``p``, ``r`` and ``d``, FP32 ``[T, n]`` each.

    Notes
    -----
    The retained state does not depend on ``torch.use_deterministic_algorithms``.
    """
    return _pkg().cake_attn_res_det_train_retained_bytes(T, n)


@cake_api
def is_cake_attn_res_det_train_supported(device: Union[torch.device, int, str, None] = None) -> bool:
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
    return _pkg().is_cake_attn_res_det_train_supported(device)
