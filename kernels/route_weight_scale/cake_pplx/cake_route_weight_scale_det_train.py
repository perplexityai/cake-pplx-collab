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

Cake route-weight row scaling of a routed MoE expert, training forward /
backward, hidden size ``H`` 4096 or 3584 (SM100 / SM103),
deterministic.

After the down projection, each received token copy (row ``r < rows``) is
scaled by its FP32 route weight ``w[r]`` before the combine.  From the BF16
row ``x`` (``[rows, H]``) and ``w`` (FP32 ``[rows]``)::

    y  = BF16(w * x)                  # the FP32 product, rounded once
    dx = BF16(w * dy)                 # the FP32 product, rounded once
    dw = sum over c of dy * x         # FP32 [rows], one word per row

``y`` and ``dx`` equal ``(w[:, None] * x.float()).bfloat16()`` and
``(w[:, None] * dy.float()).bfloat16()`` bit for bit.  The products of two
BF16 values in ``dw`` are exact in FP32; the additions form a balanced binary
tree in one fixed order, and ``dw`` is stored once.  No ``[rows, H]`` FP32
product is formed: the backward reads ``x`` and ``dy`` once and writes ``dx``
and ``dw`` once.  ``dw`` does not divide by ``w``: a zero-weight row gets
signed zeros in ``y`` and ``dx`` and keeps its ``dw``.  Nothing is FP64.

``rows`` is a run-time value (``0 <= rows < 2**31 - 256``; ``rows = 0``
returns empty outputs and launches nothing), and so are the row strides: one
library per program and architecture serves every call, with no
recompilation.  Row offsets are 64-bit, so one call may span more than 2^31
elements (1,048,576 rows at ``H = 4096``).  Every BF16 operand may be a row
slice of a wider or longer buffer: a unit column stride, a row stride that is a
multiple of 8 elements and a 16-byte aligned base.  ``w`` and ``dw`` are FP32
``[rows]``, contiguous.  An operand of another layout raises ``ValueError``;
nothing is copied.

Mode rules:

1. The mode is part of every name: every public name of this module and of
   its backend package ``cake_pplx._backends.cake_route_weight_scale_det_train``
   carries ``det``; the constants ``ARCHES``, ``SUPPORTED_COMPUTE_CAPABILITIES``
   and ``WIDTHS`` are exempt.  The family has one module, the deterministic
   one: nothing sums across rows, and the one sum, ``dw`` within a row, has
   one fixed order.  So no non-deterministic module and no KCP module apply.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.
3. Every call gives the same bits for the same inputs on the same
   architecture, whatever ``rows``, the layout, the stream or a CUDA graph, so
   under ``torch.use_deterministic_algorithms(True)`` no call alerts and no
   output bit changes.  With
   ``torch.utils.deterministic.fill_uninitialized_memory`` (on by default),
   PyTorch then fills each output a call allocates with NaN before the kernel
   writes it: one more fill launch per new output.  Output buffers passed to
   every call avoid the fills.
4. Autograd: ``CakeRouteWeightScaleDetFunction`` saves only its inputs ``x``
   and ``w`` (references, no copy) and runs one backward launch for ``dx`` and
   ``dw``.  Under ``torch.utils.checkpoint(..., use_reentrant=False)`` the
   recomputation launches the forward once.  An upstream gradient the kernels
   cannot read (an expanded gradient, say) is copied once.  No double
   backward: differentiating gradients computed with ``create_graph=True``
   raises ``RuntimeError``.  Not tested under ``torch.compile``.

No call synchronizes the host or the device or reads tensor contents on the
host; every kernel runs on the current torch stream of the operands' device.
Every argument is checked before a call allocates an output, so a call refused
for its arguments allocates and launches nothing.  Workspace: none; a call
allocates only the outputs it is not given.  A program's library loads (and,
without a prebuilt one, JIT-builds) at its first launch: before capturing a
CUDA graph, run an eager call of each width and stage the graph captures (or
build every library of the architecture with the backend's
``specs_for_arch``).  This module imports its backend package on first use and
reads no environment variable.

Usage::

    from cake_pplx.cake_route_weight_scale_det_train import cake_route_weight_scale_det

    y = cake_route_weight_scale_det(x, w)   # x BF16 [rows, 4096], w FP32 [rows]; x.grad BF16, w.grad FP32

    # steps without autograd, for a caller's own torch.autograd.Function
    y = cake_route_weight_scale_det_train_forward(x, w, out=y_rows)       # out: a row slice of a buffer
    dx, dw = cake_route_weight_scale_det_train_backward(x, dy, w)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from ._cake_pplx_api_route_weight_scale import cake_api

# The entry points are real ``def``s here (public-API tooling reads the decorated signatures of this module); their
# bodies live in the self-contained backend package next to the generated CUDA sources.  The backend package is
# imported on first use.


@functools.cache
def _backend() -> Any:
    from ._backends import cake_route_weight_scale_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_route_weight_scale_det_train


# The backend package's own objects (the autograd class and the constants), resolved on first access by the module
# ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "WIDTHS",
    "CakeRouteWeightScaleDetFunction",
)


@cake_api
def is_cake_route_weight_scale_det_train_supported(device: torch.device) -> bool:
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

    return _backend().is_cake_route_weight_scale_det_train_supported(device)


@cake_api
def cake_route_weight_scale_det_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Route-weight row scaling forward: ``y = BF16(w[:, None] * x)``, the FP32 product rounded once.

    One kernel launch, no workspace.  The same inputs give the same bits at every call.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[rows, H]``, ``H`` 4096 or 3584: a unit column stride, a row stride that is a multiple of 8 elements
        and a 16-byte aligned base (a row slice of a wider or longer buffer is read in place).
    w : torch.Tensor
        FP32 ``[rows]`` route weights, contiguous, on ``x``'s device.  Any value; zeros allowed.
    out : torch.Tensor, optional
        Preallocated BF16 ``[rows, H]`` output: the layout rules of ``x``, and a row stride of at least ``H``.  It
        must not overlap ``x`` or ``w`` (not checked).

    Returns
    -------
    torch.Tensor
        BF16 ``y [rows, H]``: ``out`` when given, else a new contiguous tensor.

    Raises
    ------
    TypeError
        A non-tensor operand.
    ValueError
        An operand of another dtype, shape or layout; an operand off CUDA or off ``x``'s device;
        ``rows >= 2**31 - 256``.
    RuntimeError
        A device other than SM100 (``sm_100a``) or SM103 (``sm_103a``), also with ``rows = 0``;
        a local CUDA toolkit below the library's floor when the call JIT-builds it.

    Notes
    -----
    ``rows = 0`` launches nothing.  Under ``torch.use_deterministic_algorithms(True)`` with
    ``torch.utils.deterministic.fill_uninitialized_memory``, PyTorch fills a new ``y`` with NaN first (one fill
    launch); an ``out`` passed to every call avoids the fill.
    """

    return _backend().cake_route_weight_scale_det_train_forward(x, w, out=out)


@cake_api
def cake_route_weight_scale_det_train_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    w: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Route-weight row scaling backward: ``dx = BF16(w[:, None] * dy)`` and ``dw = sum over c of dy * x``.

    One kernel launch reads ``x`` and ``dy`` once and writes ``dx`` and ``dw`` once; no workspace, no ``[rows, H]``
    FP32 buffer.  The same inputs give the same bits at every call.

    Parameters
    ----------
    x : torch.Tensor
        The forward's input (see :func:`cake_route_weight_scale_det_train_forward`).
    dy : torch.Tensor
        BF16 ``[rows, H]`` upstream gradient, with ``x``'s rows and ``H`` and the layout rules of ``x``.
    w : torch.Tensor
        The forward's FP32 ``[rows]`` route weights, contiguous.
    dx : torch.Tensor, optional
        Preallocated BF16 ``[rows, H]`` input gradient: the layout rules of ``out`` of the forward.  It must not
        overlap an input or ``dw`` (not checked).
    dw : torch.Tensor, optional
        Preallocated contiguous FP32 ``[rows]`` route-weight gradient.  It must not overlap an input or ``dx`` (not
        checked).

    Returns
    -------
    tuple
        ``(dx, dw)``: BF16 ``[rows, H]`` and FP32 ``[rows]`` (the given buffers, else new contiguous tensors).

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As for the forward; also ``dy``, ``dx`` or ``dw`` of another dtype, shape or layout.

    Notes
    -----
    ``dw`` is one FP32 word per row: the products ``dy * x`` are exact in FP32, and the additions form a balanced
    binary tree in one fixed order (pairs within a load of 8 columns, the loads of a lane pairwise, the even and odd
    halves, then five XOR shuffles across the row's 32 lanes).  It does not divide by ``w``, so a zero-weight row
    keeps its sum.  ``rows = 0`` launches nothing.
    """

    return _backend().cake_route_weight_scale_det_train_backward(x, dy, w, dx=dx, dw=dw)


@cake_api
def cake_route_weight_scale_det(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Differentiable route-weight row scaling: :class:`CakeRouteWeightScaleDetFunction` applied to ``x`` and ``w``.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[rows, H]``, ``H`` 4096 or 3584 (the layout rules of the forward step).  Its gradient is BF16
        ``[rows, H]``.
    w : torch.Tensor
        FP32 ``[rows]`` route weights, contiguous.  Its gradient is FP32 ``[rows]``.

    Returns
    -------
    torch.Tensor
        BF16 ``y [rows, H]``, contiguous.

    Notes
    -----
    Saves only ``x`` and ``w`` (no FP32 intermediate); works under ``torch.utils.checkpoint(...,
    use_reentrant=False)``, whose recomputation launches the forward once.  The backward is one launch for ``dx`` and
    ``dw``; an upstream gradient the kernels cannot read is copied once.  No double backward: differentiating
    gradients computed with ``create_graph=True`` raises ``RuntimeError``.
    """

    return _backend().cake_route_weight_scale_det(x, w)


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
    "WIDTHS",  # noqa: F822 - resolved by __getattr__
    "CakeRouteWeightScaleDetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_route_weight_scale_det",
    "cake_route_weight_scale_det_train_backward",
    "cake_route_weight_scale_det_train_forward",
    "is_cake_route_weight_scale_det_train_supported",
]
