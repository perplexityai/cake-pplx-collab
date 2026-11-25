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

Cake SiTU-GLU training forward / backward of an MLP, at any width ``I`` that
is a multiple of 256 (SM100 / SM103 / SM107), deterministic.

Per row ``r < R`` and column ``c < I``, from the BF16 gate ``g`` and up ``u``
(two ``[R, I]`` tensors, or one ``[R, 2I]`` tensor with the gate in columns
``[0, I)`` and the up in ``[I, 2I)``) and the FP32 scalars ``beta`` and
``lb``::

    s = sigmoid(g);  tg = tanh(g / beta);  tu = tanh(u / lb)       # FP32
    a = beta tg s;  u' = lb tu
    h      = BF16(a u')                                         # the only rounding
    d_gate = BF16(dh u' ((1 - tg^2) s + a (1 - s)))
    d_up   = BF16(dh a (1 - tu^2))

There is no clamp: both factors saturate smoothly, so large ``|g|`` and
``|u|`` keep finite values, and ``g = 0`` or ``u = 0`` gives ``h = 0``
exactly.  The backward stores 0 for ``d_up`` where ``|u| > 43.67 lb`` and,
at ``beta >= 2.4``, for ``d_gate`` where ``|g| > 43.67 beta``: there the
exact values are under ``2^-124 |dh| beta`` and about ``2^-124 |dh| lb``
(the backend's numerics).  The activation is unweighted.
``R`` is a run-time value (any ``0 <= R <= 2**30``; ``R = 0`` returns empty
outputs and launches nothing), and so are ``I`` and the scalars: one build
serves every call, with no recompilation.  ``beta`` and ``lb`` default to 4.0
and 25.0, the values of Kimi-K3.1-Pro (``activation_situ_beta`` and
``activation_situ_linear_beta``).  They must be positive and finite in FP32,
with ``2.88539 / beta``, ``2.88539 / lb``, ``beta lb``, ``beta (1 + 2^-21)``
and ``lb (1 + 2^-21)`` finite in FP32 (``ValueError`` otherwise); the tests
cover ``beta`` in ``[2.5, 4]`` and ``lb`` in ``[10, 40]``.

The kernels take every exponential as a power of two of a negative argument
(the approximate base-2 exponential), every tanh as ``(1 - E) / (1 + E)`` of
such a power, and at most one reciprocal per element; every product rounds to
nearest without flushing subnormals, and nothing is FP64.  The tests hold ``h``,
``d_gate`` and ``d_up`` to the PyTorch BF16 naive implementation of the same
operation (the gradients from PyTorch autograd), each against a float64
reference computed from the BF16 inputs: over 32 random inputs of a case, in
the L2 and in the max norm of the error, the median and the largest error of
each output are at most 1.005 times the naive implementation's, and 0 on every
input where the naive implementation's is 0.  No output has an error budget of
its own.

Mode rules:

1. The mode is part of every name: every public name of this module and of
   its backend package ``cake_pplx._backends.cake_situ_glu_det_train``
   carries ``det``; the constants ``ARCHES``,
   ``SUPPORTED_COMPUTE_CAPABILITIES`` and ``WIDTH_MULTIPLE`` are exempt.  The
   family has no non-deterministic module and no KCP module: every output is
   elementwise, with no reduction to reorder, and rows are independent.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.
3. Every call gives the same bits for the same inputs on the same
   architecture, whatever ``R``, the layout, the stream or a CUDA graph, so
   under ``torch.use_deterministic_algorithms(True)`` no call alerts.  With
   ``torch.utils.deterministic.fill_uninitialized_memory`` (on by default),
   PyTorch then fills each new output and each copied input with NaN before
   the kernel writes it: one more launch each, and the bits do not change.
4. Autograd: ``CakeSiTUGLUDetFunction`` saves only its inputs (the gate and
   up, or the merged tensor; no activation) and recomputes the activation in
   the backward: retained state is the inputs.
   ``torch.utils.checkpoint(..., use_reentrant=False)`` keeps the same inputs;
   the function adds no saved state of its own and no activation.  No call
   uses a workspace.

No call synchronizes the host or the device or reads tensor contents on the
host; every kernel runs on the current torch stream of the operands' device.
Before capturing a CUDA graph, run one eager call (or build the libraries) so
that no JIT build happens during capture.  This module imports its backend
package on first use and reads no environment variable.

Usage::

    from cake_pplx.cake_situ_glu_det_train import cake_situ_glu_det

    h = cake_situ_glu_det(gate_up)                  # merged [R, 2I]; gate_up.grad is [R, 2I]
    h = cake_situ_glu_det(gate, up)                 # two [R, I] tensors
    h = cake_situ_glu_det(gate_up, beta=4.0, lb=25.0)

    # steps without autograd, for a caller's own torch.autograd.Function
    h = cake_situ_glu_det_train_forward(gate_up)
    d_gate_up, _ = cake_situ_glu_det_train_backward(dh, gate_up)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from ._cake_pplx_api_situ_glu import cake_api

# The entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.  The backend package is
# imported on first use.


@functools.cache
def _backend() -> Any:
    from ._backends import cake_situ_glu_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_situ_glu_det_train


# The backend package's own objects (the autograd class and the constants),
# resolved on first access by the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "WIDTH_MULTIPLE",
    "CakeSiTUGLUDetFunction",
)


@cake_api
def is_cake_situ_glu_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of an exported architecture.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` for a CUDA device of compute capability (10, 0), (10, 3) or (10, 7).
    """

    return _backend().is_cake_situ_glu_det_train_supported(device)


@cake_api
def cake_situ_glu_det_train_forward(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SiTU-GLU forward: ``h = BF16(beta tanh(gate / beta) sigmoid(gate) lb tanh(up / lb))``.

    The activation is FP32; ``h`` is rounded once.  One kernel launch (plus one copy of an input the kernels cannot
    read in place), no workspace.  The same inputs give the same
    bits at every call.

    Parameters
    ----------
    gate_up_or_gate : torch.Tensor
        BF16 ``[R, I]`` gate when ``up`` is given, else the merged BF16 ``[R, 2I]`` projection output (gate columns
        ``[0, I)``, up ``[I, 2I)``), read in place.  ``I`` is a positive multiple of 256.
    up : torch.Tensor, optional
        BF16 ``[R, I]`` up, on the gate's device.
    beta : float
        ``beta``, a finite positive Python number, rounded to FP32 (a run-time launch argument).  Default 4.0.
    lb : float
        ``lb``, a finite positive Python number, rounded to FP32 (a run-time launch argument).  Default 25.0.
    out : torch.Tensor, optional
        Preallocated BF16 ``[R, I]`` output: 2-D, unit last stride, a row stride that is a multiple of 8 and at least
        ``I``, a 16-byte aligned base.  It must not overlap any input (not checked).

    Returns
    -------
    torch.Tensor
        BF16 ``h [R, I]``, contiguous (``out`` when given).

    Raises
    ------
    TypeError
        A non-tensor operand; a ``beta`` or ``lb`` that is not a Python number (a tensor would need a host read).
    ValueError
        An operand off CUDA, off the gate's device, of another dtype or shape; a width that is not a positive multiple
        of 256; a non-finite or non-positive ``beta`` or ``lb``; ``R > 2**30``; an ``out`` of another layout.
    RuntimeError
        A device other than SM100 (``sm_100a``), SM103 (``sm_103a``) or SM107 (``sm_107a``), also with ``R = 0``.

    Notes
    -----
    An input with a layout the kernels cannot read (a non-unit last stride, a row stride that is not a multiple of
    8 elements, a base that is not 16-byte aligned) is copied once.  ``R = 0`` launches nothing.
    """

    return _backend().cake_situ_glu_det_train_forward(gate_up_or_gate, up, beta=beta, lb=lb, out=out)


@cake_api
def cake_situ_glu_det_train_backward(
    grad_out: torch.Tensor,
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
    grad_gate_up_or_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""SiTU-GLU backward: ``d_gate`` and ``d_up`` from ``gate``, ``up`` and ``dh`` (the activation is recomputed;
    nothing from the forward is needed).

    With ``s = sigmoid(g)``, ``tg = tanh(g / beta)``, ``tu = tanh(u / lb)``, ``a = beta tg s`` and ``u' = lb tu``::

        d_gate = BF16(dh u' ((1 - tg^2) s + a (1 - s)))
        d_up   = BF16(dh a (1 - tu^2))

    in FP32, each BF16 output rounded once.  One kernel launch (plus one copy of an input the kernels cannot read in
    place), no workspace.  The same inputs give the same bits at
    every call.

    Parameters
    ----------
    grad_out : torch.Tensor
        BF16 ``[R, I]`` upstream gradient ``dh`` (any layout; copied once when the kernels cannot read it).
    gate_up_or_gate, up : torch.Tensor
        The forward's inputs (see :func:`cake_situ_glu_det_train_forward`).
    beta, lb : float
        The forward's scalars.  Defaults 4.0 and 25.0.
    grad_gate_up_or_gate : torch.Tensor, optional
        Preallocated BF16 output: ``d_gate [R, I]`` for two input tensors, or the merged ``[R, 2I]`` gradient
        (``d_gate`` columns ``[0, I)``, ``d_up`` ``[I, 2I)``) for a merged input.  Same layout rules as ``out`` of the
        forward.  It must not overlap any input or another output (not checked).
    grad_up : torch.Tensor, optional
        Preallocated BF16 ``d_up [R, I]`` for two input tensors; must be None for a merged input.  It must not overlap
        any input or another output (not checked).

    Returns
    -------
    tuple
        ``(d_gate_up, None)`` for a merged input (``d_gate_up`` BF16 ``[R, 2I]``), ``(d_gate, d_up)`` for two tensors
        (BF16 ``[R, I]`` each); new tensors are contiguous.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As for the forward; also ``grad_up`` with a merged input, and output buffers of another shape or layout
        (outputs are never copied).
    """

    return _backend().cake_situ_glu_det_train_backward(
        grad_out,
        gate_up_or_gate,
        up,
        beta=beta,
        lb=lb,
        grad_gate_up_or_gate=grad_gate_up_or_gate,
        grad_up=grad_up,
    )


@cake_api
def cake_situ_glu_det(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    beta: float = 4.0,
    lb: float = 25.0,
) -> torch.Tensor:
    """Differentiable SiTU-GLU: :class:`CakeSiTUGLUDetFunction` applied to the arguments.

    Parameters
    ----------
    gate_up_or_gate : torch.Tensor
        BF16 ``[R, I]`` gate when ``up`` is given, else the merged BF16 ``[R, 2I]`` gate and up.  Its gradient has its
        shape: one ``[R, 2I]`` tensor (``d_gate`` then ``d_up``) for the merged form.
    up : torch.Tensor, optional
        BF16 ``[R, I]`` up.
    beta, lb : float
        Finite positive Python numbers (FP32 in the kernels), defaults 4.0 and 25.0; they get no gradient.

    Returns
    -------
    torch.Tensor
        BF16 ``h [R, I]``, contiguous: the down projection's input.

    Notes
    -----
    Saves only the inputs and recomputes the activation in the backward (no activation retained; works under
    ``torch.utils.checkpoint(..., use_reentrant=False)``).  No double backward.
    """

    return _backend().cake_situ_glu_det(gate_up_or_gate, up, beta=beta, lb=lb)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_backend(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKEND_OBJECTS))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "WIDTH_MULTIPLE",  # noqa: F822 - resolved by __getattr__
    "CakeSiTUGLUDetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_situ_glu_det",
    "cake_situ_glu_det_train_backward",
    "cake_situ_glu_det_train_forward",
    "is_cake_situ_glu_det_train_supported",
]
