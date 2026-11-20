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

Cake routed-expert clamped SwiGLU training forward / backward, intermediate
width ``I = 2304`` (SM100 / SM103 / SM107), deterministic.

Per row ``r < R`` and column ``c < I``, from the BF16 gate ``g`` and up ``u``
(two ``[R, I]`` tensors, or one ``[R, 2I]`` tensor with the gate in columns
``[0, I)`` and the up in ``[I, 2I)``), the row's FP32 route weight ``w``
(``[R]``, ``>= 0``, zeros allowed; ``None``: ``w = 1``, a shared expert) and
the FP32 limit ``L``::

    g' = min(g, L);  u' = clamp(u, -L, L);  s = sigmoid(g');  a = g' s u'   # FP32
    h        = BF16(w * a)                                       # the only rounding
    d_gate   = BF16(dh * w * u' * s * (1 + g' (1 - s)) * [g <= L])
    d_up     = BF16(dh * w * g' s * [-L <= u <= L])
    d_weight = sum over c of dh * a                              # FP32, the unweighted a

The route weight multiplies the FP32 activation before the one rounding of
``h``, as the released expert does.  The masks are inclusive: the gradient
passes at ``g = L`` and ``u = +-L``.  ``d_weight`` does not divide by ``w``,
so a zero-weight row gets exactly 0 in ``h``, ``d_gate`` and ``d_up`` and its
full ``d_weight``.  ``R`` is a run-time value (any ``R >= 0``; ``R = 0``
returns empty outputs and launches nothing), and so are ``L`` and the weight
flag: one library per program and architecture serves every call, with no
recompilation (the row count picks the program, and in the backward so does
whether a weight is given).  ``limit`` has no default: pass the model's
value (``swiglu_limit``, 10.0 for DeepSeek-V4.1-Flash) on every call.

The sigmoid uses the approximate FP32 base-2 exponential and the approximate
reciprocal with one round-to-nearest Newton step (below the flushed
reciprocal's range a scaled reciprocal, within one subnormal ulp); every
product rounds to nearest without flushing subnormals, and nothing is FP64.
The tests hold ``h``, ``d_gate``, ``d_up`` and ``d_weight`` to the PyTorch
BF16 naive implementation of the same operation (the clamps written as
selects, so autograd gives the inclusive masks; the FP32 weight kept FP32;
the gradients from PyTorch autograd), each against a float64 reference
computed from the BF16 inputs: over 32 random inputs of a case, in the L2
and in the max norm of the error, the median and the largest error of each
output are at most 1.005 times the naive implementation's, and 0 on every
input where the naive implementation's is 0.  No output has an error budget
of its own.

Mode rules:

1. The mode is part of every name: every public name of this module and of
   its backend package ``cake_pplx._backends.cake_clamped_swiglu_det_train``
   carries ``det``; the constants ``ARCHES``, ``SUPPORTED_COMPUTE_CAPABILITIES``
   and ``INTERMEDIATE_SIZE`` are exempt.  The family has no non-deterministic
   module: every output is elementwise or a sum within one row in one fixed
   order.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.
3. Every call gives the same bits for the same inputs on the same
   architecture, whatever ``R``, the layout, the stream or a CUDA graph, so
   under ``torch.use_deterministic_algorithms(True)`` no call alerts and no
   output bit changes.  With
   ``torch.utils.deterministic.fill_uninitialized_memory`` (on by default),
   PyTorch then fills each tensor a call allocates with NaN before the
   kernel writes it: one more fill launch per new output and per copied
   input.  Output buffers allocated once and passed to every call avoid the
   fills of the outputs.
4. Autograd: ``CakeClampedSwiGLUDetFunction`` saves only its inputs (the
   gate and up, or the merged tensor, and the weight; no activation) and
   recomputes the activation in the backward: retained state is the inputs.
   ``torch.utils.checkpoint(..., use_reentrant=False)`` keeps the same inputs;
   the function adds no saved state of its own and no activation.  So
   checkpointing ``cake_clamped_swiglu_det`` alone saves no memory: the down
   projection that reads ``h`` keeps ``h`` for its weight gradient.  A
   region of the gate/up projection and the op drops the projection's
   ``[R, 4608]`` output until the backward; one that also holds the down
   projection drops ``h``.  In the backward of a non-reentrant checkpoint,
   whose recomputation sets saved-tensors hooks, ``cake_clamped_swiglu_det``
   launches its forward kernel after autograd saved its inputs, as a
   built-in op does.  So with early stop on (the default) the recomputation
   skips that launch when nothing later in the region saves a tensor (the op
   alone, or the gate/up projection and the op, the down projection
   outside) and runs it when a later operation saves one (the down
   projection saves ``h``; so does the whole expert block).  A reentrant
   checkpoint, early stop off and a direct
   ``CakeClampedSwiGLUDetFunction.apply`` recompute the forward in every
   region, even when its output is dropped.  Anywhere else (no hooks, the
   checkpoint's own forward pass) ``forward`` launches, as in 0.1.7.  Not
   tested under ``torch.compile``.  No double
   backward: differentiating gradients computed with ``create_graph=True``
   raises ``RuntimeError``.  No call uses a workspace or keeps state between
   calls, but for the ignored weight of unweighted forward calls: one cached
   one-element FP32 buffer per device, set to 1, which the forward kernels
   read and ignore.  An unweighted backward runs a kernel that takes no
   weight and writes no weight gradient.

No call synchronizes the host or the device or reads tensor contents on the
host; every kernel runs on the current torch stream of the operands' device.
Every argument is checked before a call copies an input or allocates an
output, so a call refused for its arguments copies, allocates and launches
nothing.  The forward and the weighted backward each have a small-``R`` and
a large-``R`` program, chosen by ``R`` and the device's SM count; the
unweighted backward has one program, which stores the bits of the weighted
backward's programs at ``w = 1``.  A program's library loads (and, without a
prebuilt one, JIT-builds) at its first launch.  Before capturing a CUDA
graph, run an eager call of each kind the graph captures (weighted and
unweighted) at each row count it captures (or build every library of the
architecture with the backend's ``specs_for_arch``) so that no JIT build
happens during capture.  A JIT build needs a CUDA toolkit of at least 12.8
for ``sm_100a``, 12.9 for ``sm_103a`` and 13.5 for ``sm_107a``.  This module
imports its backend package on first use and reads no environment variable.

Usage::

    from cake_pplx.cake_clamped_swiglu_det_train import cake_clamped_swiglu_det

    h = cake_clamped_swiglu_det(gate_up, weight=w, limit=10.0)  # merged [R, 4608]; gate_up.grad is [R, 4608]
    h = cake_clamped_swiglu_det(gate, up, w, limit=10.0)        # two [R, 2304] tensors
    h = cake_clamped_swiglu_det(gate_up, limit=10.0)            # shared expert: no route weight

    # steps without autograd, for a caller's own torch.autograd.Function
    h = cake_clamped_swiglu_det_train_forward(gate_up, None, w, limit=10.0)
    d_gate_up, _, d_w = cake_clamped_swiglu_det_train_backward(dh, gate_up, None, w, limit=10.0)
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from ._cake_pplx_api_clamped_swiglu import cake_api

# The entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); their bodies live in the self-contained
# backend package next to the generated CUDA sources.  The backend package is
# imported on first use.  The steps and the autograd entry call the backend's
# own step functions, ``CakeClampedSwiGLUDetFunction.apply`` and, in a
# backward with saved-tensors hooks set, its held-launch entry ``_held``,
# which the backend functions of the same names call: the same work, one
# Python call fewer per call.  Each is bound to a global of this module at
# its first call (:func:`_step_forward`, :func:`_step_backward`,
# :func:`_apply`, :func:`_held`), so a later call reads no cache.


@functools.cache
def _backend() -> Any:
    from ._backends import cake_clamped_swiglu_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_clamped_swiglu_det_train


def _step_forward(*arguments: Any) -> torch.Tensor:
    """The backend's forward step; the first call imports the backend and binds this global to its own function."""
    global _step_forward
    _step_forward = _backend()._forward_step
    return _step_forward(*arguments)


def _step_backward(*arguments: Any) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The backend's backward step; the first call imports the backend and binds this global to its own function."""
    global _step_backward
    _step_backward = _backend()._backward_step
    return _step_backward(*arguments)


def _apply(*arguments: Any) -> torch.Tensor:
    """The backend's ``CakeClampedSwiGLUDetFunction.apply``; the first call imports the backend and binds this global
    to it."""
    global _apply
    _apply = _backend().CakeClampedSwiGLUDetFunction.apply
    return _apply(*arguments)


def _held(*arguments: Any) -> torch.Tensor:
    """The backend's held-launch autograd entry (its ``_HeldFunction.apply``, then the forward launch), for a call in a
    backward with saved-tensors hooks set; the first call imports the backend and binds this global to it."""
    global _held
    _held = _backend()._held
    return _held(*arguments)


# The id of the backward graph task running on this thread, -1 outside a backward (torch's checkpoint reads it the
# same way), and the innermost saved-tensors hooks, a (pack, unpack) pair or None, as the backend reads them.
# Without either function (another torch) its stand-in answers "in a backward" / "hooks set", so the calls it decides
# take the held launch.
_graph_task = getattr(torch._C, "_current_graph_task_id", None) or (lambda: 0)
_hooks = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None) or (lambda ignore_is_tracing: ())


# The backend package's own objects (the autograd class and the constants),
# resolved on first access by the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "INTERMEDIATE_SIZE",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeClampedSwiGLUDetFunction",
)


@cake_api
def is_cake_clamped_swiglu_det_train_supported(device: torch.device) -> bool:
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

    return _backend().is_cake_clamped_swiglu_det_train_supported(device)


@cake_api
def cake_clamped_swiglu_det_train_forward(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Clamped SwiGLU forward: ``h = BF16(weight * silu(min(gate, L)) * clamp(up, -L, L))``.

    The activation and the product with the FP32 route weight are FP32; ``h`` is rounded once.  One kernel launch,
    no workspace.  The same inputs give the same bits at every call.

    Parameters
    ----------
    gate_up_or_gate : torch.Tensor
        BF16 ``[R, 2304]`` gate when ``up`` is given, else the merged BF16 ``[R, 4608]`` projection output (gate
        columns ``[0, 2304)``, up ``[2304, 4608)``), read in place.
    up : torch.Tensor, optional
        BF16 ``[R, 2304]`` up, on the gate's device.
    weight : torch.Tensor, optional
        FP32 ``[R]`` route weight (``>= 0``, zeros allowed; not checked).  ``None``: an unweighted expert,
        ``h = BF16(a)``.
    limit : float
        ``L``, a finite positive Python number, rounded to FP32 (a run-time launch argument).  Required.
    out : torch.Tensor, optional
        Preallocated BF16 ``[R, 2304]`` output: 2-D, unit last stride, a row stride that is a multiple of 8 and at
        least 2304, a 16-byte aligned base.  It must not overlap any input (not checked).

    Returns
    -------
    torch.Tensor
        BF16 ``h [R, 2304]``, contiguous (``out`` when given).

    Raises
    ------
    TypeError
        A non-tensor operand; a ``limit`` that is not a Python number (a tensor would need a host read).
    ValueError
        An operand off CUDA, off the gate's device, of another dtype or shape; a non-finite or non-positive
        ``limit``, or one past FP32 (an integer or a fraction past every Python float too); ``R >= 2**31 - 256``; an
        ``out`` of another layout.
    RuntimeError
        A device other than SM100 (``sm_100a``), SM103 (``sm_103a``) or SM107 (``sm_107a``), also with ``R = 0``; a
        local CUDA toolkit below the library's floor when the call JIT-builds it (12.8 for ``sm_100a``, 12.9 for
        ``sm_103a``, 13.5 for ``sm_107a``).

    Notes
    -----
    An input with a layout the kernels cannot read (a non-unit last stride, a row stride that is not a multiple of
    8 elements, a base that is not 16-byte aligned; a non-contiguous ``weight``) is copied once, after every argument
    is checked: a call refused for its arguments copies and allocates nothing.  ``R = 0`` launches nothing.  Under
    ``torch.use_deterministic_algorithms(True)`` with ``torch.utils.deterministic.fill_uninitialized_memory``,
    PyTorch fills a new ``h`` and each copied input with NaN first (one fill launch each); an ``out`` allocated once
    and passed to every call avoids the fill of ``h``.
    """

    return _step_forward(gate_up_or_gate, up, weight, limit, out)


@cake_api
def cake_clamped_swiglu_det_train_backward(
    grad_out: torch.Tensor,
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
    grad_gate_up_or_gate: Optional[torch.Tensor] = None,
    grad_up: Optional[torch.Tensor] = None,
    grad_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    r"""Clamped SwiGLU backward: ``d_gate``, ``d_up`` and the route-weight gradient, from ``gate``, ``up``,
    ``weight`` and ``dh`` (the activation is recomputed; nothing from the forward is needed).

    With ``g' = min(g, L)``, ``u' = clamp(u, -L, L)``, ``s = sigmoid(g')`` and ``a = g' s u'``::

        d_gate   = BF16(dh * w * u' * s * (1 + g' (1 - s)) * [g <= L])
        d_up     = BF16(dh * w * g' s * [-L <= u <= L])
        d_weight = sum over c of dh * a        # FP32, one fixed order per row

    in FP32, each BF16 output rounded once; the masks are inclusive.  One kernel launch, no workspace.  The same
    inputs give the same bits at every call.

    Parameters
    ----------
    grad_out : torch.Tensor
        BF16 ``[R, 2304]`` upstream gradient ``dh`` (any layout; copied once when the kernels cannot read it).
    gate_up_or_gate, up, weight : torch.Tensor
        The forward's inputs (see :func:`cake_clamped_swiglu_det_train_forward`).
    limit : float
        The forward's ``L``.  Required.
    grad_gate_up_or_gate : torch.Tensor, optional
        Preallocated BF16 output: ``d_gate [R, 2304]`` for two input tensors, or the merged ``[R, 4608]`` gradient
        (``d_gate`` columns ``[0, 2304)``, ``d_up`` ``[2304, 4608)``) for a merged input.  Same layout rules as
        ``out`` of the forward.  It must not overlap any input or another output (not checked).
    grad_up : torch.Tensor, optional
        Preallocated BF16 ``d_up [R, 2304]`` for two input tensors; must be None for a merged input.  It must not
        overlap any input or another output (not checked).
    grad_weight : torch.Tensor, optional
        Preallocated contiguous FP32 ``[R]`` route-weight gradient (needs ``weight``).  It must not overlap any input
        or another output (not checked).

    Returns
    -------
    tuple
        ``(d_gate_up, None, d_weight)`` for a merged input (``d_gate_up`` BF16 ``[R, 4608]``), ``(d_gate, d_up,
        d_weight)`` for two tensors (BF16 ``[R, 2304]`` each); new tensors are contiguous.  ``d_weight`` is FP32
        ``[R]``, or None without ``weight``.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        As for the forward; also ``grad_up`` with a merged input, ``grad_weight`` without ``weight``, and output
        buffers of another shape or layout (outputs are never copied).

    Notes
    -----
    ``d_weight`` is one FP32 word per row: each lane's two FMA chains (even and odd columns) in load order, added,
    then XOR shuffles across the row's 32 lanes, in one fixed order.  It uses the unweighted ``a``, so a zero-weight
    row keeps its sum.  An unweighted call runs a kernel that stores ``d_gate`` and ``d_up`` only (the bits of a
    weighted call with ``w = 1``): it allocates and writes no weight gradient.  Every argument is checked before an
    input is copied or an output allocated.  Under ``torch.use_deterministic_algorithms(True)`` with
    ``torch.utils.deterministic.fill_uninitialized_memory``, PyTorch fills each new output and each copied input with
    NaN first, one fill launch each; preallocated outputs passed to every call avoid the fills of those outputs.
    """

    return _step_backward(
        grad_out, gate_up_or_gate, up, weight, limit, grad_gate_up_or_gate, grad_up, grad_weight, False
    )


@cake_api
def cake_clamped_swiglu_det(
    gate_up_or_gate: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    *,
    limit: float,
) -> torch.Tensor:
    """Differentiable routed-expert clamped SwiGLU: :class:`CakeClampedSwiGLUDetFunction` applied to the arguments.

    Parameters
    ----------
    gate_up_or_gate : torch.Tensor
        BF16 ``[R, 2304]`` gate when ``up`` is given, else the merged BF16 ``[R, 4608]`` gate and up.  Its gradient
        has its shape: one ``[R, 4608]`` tensor (``d_gate`` then ``d_up``) for the merged form.
    up : torch.Tensor, optional
        BF16 ``[R, 2304]`` up.
    weight : torch.Tensor, optional
        FP32 ``[R]`` route weight; ``None`` for an unweighted (shared) expert, which has no weight gradient.
    limit : float
        ``L``, a finite positive Python number (FP32 in the kernels).  Required; it gets no gradient.

    Returns
    -------
    torch.Tensor
        BF16 ``h [R, 2304]``, contiguous: the down projection's input.

    Notes
    -----
    Saves only the inputs and recomputes the activation in the backward (no activation retained; works under
    ``torch.utils.checkpoint(..., use_reentrant=False)``).  Checkpointing this op alone saves no memory (the down
    projection keeps ``h``); a region of the gate/up projection and this op drops the projection's output until the
    backward, and one that also holds the down projection drops ``h``.  In the backward of a non-reentrant checkpoint
    the forward kernel runs after autograd saved the inputs, so with early stop on (the default) the recomputation
    launches it only when a later operation of its region saves a tensor (the down projection saves ``h``); a
    reentrant checkpoint, early stop off and a direct ``CakeClampedSwiGLUDetFunction.apply`` launch it in every
    region.
    ``limit`` is checked in the forward (twice there for a number that is neither a Python float nor an int that a
    float holds exactly), not in the backward, which reuses its FP32 rounding.  No double backward: differentiating
    gradients computed with ``create_graph=True`` raises ``RuntimeError``.
    """

    if _graph_task() == -1 or _hooks(False) is None:
        return _apply(gate_up_or_gate, up, weight, limit)  # no recomputation to stop: the forward launches in place
    return _held(gate_up_or_gate, up, weight, limit)


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
    "INTERMEDIATE_SIZE",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "CakeClampedSwiGLUDetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_clamped_swiglu_det",
    "cake_clamped_swiglu_det_train_backward",
    "cake_clamped_swiglu_det_train_forward",
    "is_cake_clamped_swiglu_det_train_supported",
]
