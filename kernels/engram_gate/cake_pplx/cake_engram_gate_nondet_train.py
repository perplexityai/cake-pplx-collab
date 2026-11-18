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

Cake non-deterministic Engram context gate training kernels: four residual
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

Every entry takes ``dot_sum``, how ``P`` is summed (the backward of an
autograd entry sums it as its forward did).  The default, ``"compensated"``,
is a compensated FP32 dot product: each product split with an FMA, each
running sum with an error-free two-sum, the error words added once at the
end.  It meets the accuracy bound of the acceptance tests on every output:
the median and the largest relative L2 error against an FP64 reference, over
random draws and over draws whose dot products nearly cancel (``|dot|`` from
2e-6 to 2e-5), at most twice those of an FP32 evaluation of the formula.
``"fp32"`` (opt-in) sums ``P`` in one FP32 word.  Where the products nearly
cancel, that sum loses most of its digits and the gate's derivative magnifies
the loss: measured on GB300, ``dx``, ``dkey``, ``dq_weight`` and
``dk_weight`` per stream are off by a relative L2 error of about 1e-2 (at most
2.6e-2), against bounds of 3.4e-3 (``dx``, ``dkey``) and 4.7e-7 (the weight
gradients); on random draws its largest weight-gradient error is 1.1e-4,
against 7.7e-5.  ``out``, ``dvalue`` and, on random draws, ``dx`` and
``dkey`` meet the bound with either sum.

Entries: :func:`cake_engram_gate_nondet` (with autograd; ``key`` and ``value``
contiguous or views into one packed ``[T, 25600]``),
:func:`cake_engram_gate_nondet_kv` (with autograd on the packed ``kv``: the
backward writes one ``dkv [T, 25600]`` with no split or concatenation copy),
:func:`cake_engram_gate_nondet_train_forward` /
:func:`cake_engram_gate_nondet_train_backward` (the two halves without
autograd), :func:`cake_engram_gate_nondet_train_workspace_bytes` and
:func:`is_cake_engram_gate_nondet_train_supported`.  The autograd classes
``CakeEngramGateNondetFunction`` and ``CakeEngramGateNondetKvFunction``
resolve to the backend's objects on first access.

The mode is part of the name: no entry takes a ``deterministic`` keyword
(passing one is a ``TypeError``).  The backward adds the weight-gradient
partial sums of the clusters with float atomics, so ``dq_weight`` and
``dk_weight`` may differ in their last bits run to run; under
``torch.use_deterministic_algorithms(True)`` every backward call that
launches raises PyTorch's ``RuntimeError`` naming the deterministic package's
entry (``warn_only=True``: one ``UserWarning``, then this package runs).  The
forward never consults the mode.

A stable API (``@cake_api``; no ``ExperimentalWarning``) of thin wrappers:
the checks, the launch plan, the autograd Functions and the generated kernels
live in the backend package ``cake_pplx._backends.cake_engram_gate_nondet_train``,
imported on first use; this module forwards every keyword.  Every switch is a
keyword argument; no environment variable is read.  The token count is a
run-time value: no recompilation, no device-to-host read, no host
synchronization; the calls are CUDA-graph capturable after one eager call
(of each ``dot_sum`` the graph uses).

Memory: the state kept for the backward is the inputs alone (the backward
recomputes the norms and the dot product).  A backward call allocates FP32
``[4, 5120]`` rows beside its outputs (the weight product, FP32 copies of BF16
weights and the weight-gradient partial sums):
:func:`cake_engram_gate_nondet_train_workspace_bytes`.

Usage::

    from cake_pplx.cake_engram_gate_nondet_train import cake_engram_gate_nondet_kv

    kv = projection(rows)                          # BF16 [T, 25600]: four keys, then the value
    out = cake_engram_gate_nondet_kv(x, kv, q_weight, k_weight, token_mask, eps=1e-20, clamp_value=1e-6)
    out.backward(dout)                             # x.grad, kv.grad [T, 25600], q_weight.grad, k_weight.grad
    # opt-in: one FP32 word per dot product (less accurate where the products nearly cancel)
    out = cake_engram_gate_nondet_kv(x, kv, q_weight, k_weight, token_mask, dot_sum="fp32")
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from ._cake_pplx_api_engram_gate import cake_api

if TYPE_CHECKING:  # resolved at run time by ``__getattr__`` (no module-level backend import)
    from ._backends.cake_engram_gate_nondet_train import (
        CakeEngramGateNondetFunction,
        CakeEngramGateNondetKvFunction,
    )

__all__ = [
    "ARCHES",
    "HIDDEN",
    "KV_WIDTH",
    "STREAMS",
    "CakeEngramGateNondetFunction",
    "CakeEngramGateNondetKvFunction",
    "cake_engram_gate_nondet",
    "cake_engram_gate_nondet_kv",
    "cake_engram_gate_nondet_train_backward",
    "cake_engram_gate_nondet_train_forward",
    "cake_engram_gate_nondet_train_workspace_bytes",
    "is_cake_engram_gate_nondet_train_supported",
]

# The built architectures and the fixed operand geometry (the backend's values).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STREAMS = 4
HIDDEN = 5120
KV_WIDTH = 25600

# Backend objects this module re-exports, resolved on first access (``__getattr__``).
_BACKEND_OBJECTS = frozenset({"CakeEngramGateNondetFunction", "CakeEngramGateNondetKvFunction"})


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use.
    from ._backends import cake_engram_gate_nondet_train  # noqa: PLC0415

    return cake_engram_gate_nondet_train


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_pkg(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BACKEND_OBJECTS)


@cake_api
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
    dot_sum: str = "compensated",
) -> torch.Tensor:
    r"""The Engram context gate with autograd (``CakeEngramGateNondetFunction``).

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
        How the forward and its backward sum ``P = sum_d w_d h_d k_d``: ``"compensated"`` (default), a
        compensated FP32 dot product that meets the accuracy bound on every output; ``"fp32"`` (opt-in), one FP32
        word, about 1e-2 off in ``dx``, ``dkey`` and the weight gradients where the products nearly cancel (see the
        module notes).

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
        In the backward under ``torch.use_deterministic_algorithms(True)`` (``warn_only=True``: a warning).

    Notes
    -----
    The weight gradients are summed with float atomics: under ``torch.use_deterministic_algorithms(True)`` the
    backward raises before it launches.  ``T == 0`` launches nothing, gives zero weight gradients and does not
    alert.
    """
    return _pkg().cake_engram_gate_nondet(
        x, key, value, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum
    )


@cake_api
def cake_engram_gate_nondet_kv(
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
    r"""The Engram context gate with autograd on the packed projection output (``CakeEngramGateNondetKvFunction``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, 4, 5120]``, contiguous.
    kv : torch.Tensor
        BF16 ``[T, 25600]``: the four keys (columns ``0:20480``), then the value (``20480:25600``); unit column
        stride, token stride a multiple of 8 elements.
    q_weight, k_weight, token_mask, eps, clamp_value, dot_sum
        As in :func:`cake_engram_gate_nondet`.

    Returns
    -------
    ``out`` BF16 ``[T, 4, 5120]``.  Its backward writes the key and value gradients straight into one new BF16
    ``dkv [T, 25600]`` (the gradient of ``kv``): no split and no concatenation copy.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As in :func:`cake_engram_gate_nondet`.

    Notes
    -----
    The backward raises PyTorch's alert under ``torch.use_deterministic_algorithms(True)``.
    """
    return _pkg().cake_engram_gate_nondet_kv(
        x, kv, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum
    )


@cake_api
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
    dot_sum: str = "compensated",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The forward without autograd.

    Parameters
    ----------
    x, key, value, q_weight, k_weight, token_mask, eps, clamp_value, dot_sum
        As in :func:`cake_engram_gate_nondet`.
    out : torch.Tensor, optional
        BF16 ``[T, 4, 5120]``, contiguous, 16-byte aligned: written in place and returned.

    Returns
    -------
    ``out`` BF16 ``[T, 4, 5120]``.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As in :func:`cake_engram_gate_nondet`.

    Notes
    -----
    The forward never consults ``torch.use_deterministic_algorithms``; nothing is saved (the backward takes the
    inputs again).
    """
    return _pkg().cake_engram_gate_nondet_train_forward(
        x, key, value, q_weight, k_weight, token_mask, eps=eps, clamp_value=clamp_value, dot_sum=dot_sum, out=out
    )


@cake_api
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
    dot_sum: str = "compensated",
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
        The forward's operands, as in :func:`cake_engram_gate_nondet`.
    dot_sum : {"compensated", "fp32"}
        How ``P`` is summed, as in :func:`cake_engram_gate_nondet`; pass the forward's choice.
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
    TypeError, ValueError, RuntimeError
        As in :func:`cake_engram_gate_nondet`.
        ``RuntimeError`` also under ``torch.use_deterministic_algorithms(True)`` (``warn_only=True``: a warning).

    Notes
    -----
    ``dq_weight`` and ``dk_weight`` are summed with float atomics; under ``torch.use_deterministic_algorithms(True)``
    the call raises before it allocates or launches anything.  ``T == 0`` launches nothing, returns zero weight
    gradients and does not alert.
    """
    return _pkg().cake_engram_gate_nondet_train_backward(
        dout,
        x,
        key,
        value,
        q_weight,
        k_weight,
        token_mask,
        eps=eps,
        clamp_value=clamp_value,
        dot_sum=dot_sum,
        dx=dx,
        dkey=dkey,
        dvalue=dvalue,
    )


@cake_api
def cake_engram_gate_nondet_train_workspace_bytes(
    T: int, device: Union[torch.device, int, str, None] = None, *, dot_sum: str = "compensated"
) -> int:
    r"""Bytes of device memory one backward call allocates beside its operands and outputs.

    Parameters
    ----------
    T : int
        The token count.
    device : torch.device, int or str, optional
        The CUDA device (None: the current one).
    dot_sum : {"compensated", "fp32"}
        The backward's sum of ``P``, as in :func:`cake_engram_gate_nondet`.
        Both sums use the same workspace.

    Returns
    -------
    With BF16 weights: 80 KiB (``4 * 5120`` FP32) for each of the weight product, the two FP32 weight copies and
    the ``[4, 5120]`` FP32 weight-gradient accumulator;
    contiguous FP32 weights need no copies (160 KiB less).  0 for ``T == 0``.  A forward call allocates the weight
    product (and an FP32 copy of a BF16 ``q_weight``) only, and the state kept for the backward is the inputs alone.

    Notes
    -----
    Workspace sizes do not depend on ``torch.use_deterministic_algorithms``.
    """
    return _pkg().cake_engram_gate_nondet_train_workspace_bytes(T, device, dot_sum=dot_sum)


@cake_api
def is_cake_engram_gate_nondet_train_supported(device: Union[torch.device, int, str, None] = None) -> bool:
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
    return _pkg().is_cake_engram_gate_nondet_train_supported(device)
