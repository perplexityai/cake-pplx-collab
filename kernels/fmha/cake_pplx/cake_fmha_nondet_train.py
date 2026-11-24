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

Cake non-deterministic dense multi-head attention backward for training, head
dimension 64 or 128, non-causal or causal (SM100 / SM103 / SM107).

Entry: :func:`cake_fmha_nondet_train_backward`, the backward step without
autograd: ``(dq, dk, dv)`` from ``q``, ``k``, ``v``, the forward's ``out`` and
``lse``, and ``dout``.  The family has no forward kernel and no autograd
entry: the caller brings ``out`` and ``lse`` from its own forward (for
example FlashAttention's ``out`` and ``softmax_lse``) and calls the backward
from its own ``torch.autograd.Function``.

The mode is part of the name: the entry takes no ``deterministic`` keyword
(passing one is a ``TypeError``).  The family has no deterministic module:
on the three-kernel routes the dQ tiles of the key blocks of one head are
added into an FP32 accumulator in arrival order, so ``dq`` may differ in its
last bits run to run; ``dk`` and ``dv`` have one fixed order.  On the
one-launch route (few CTAs, without the mask or with it) ``dq``, ``dk`` and
``dv`` each have one fixed order; the module stays non-deterministic.  Under
``torch.use_deterministic_algorithms(True)`` the entry raises PyTorch's alert
before it launches.

A stable API (``@cake_api``; no ``ExperimentalWarning``) of a thin wrapper:
the checks, the launch plan and the generated kernels live in the backend
package ``cake_pplx._backends.cake_fmha_nondet_train``, imported on first
use; this module forwards every argument.  No environment variable is read.

Operands: ``q``, ``out``, ``dout`` BF16 ``[B, Sq, H, D]``; ``k``, ``v`` BF16
``[B, Skv, H, D]`` (the same ``H``: no grouped heads); ``lse`` FP32
``[B, H, Sq]``.  All contiguous on one CUDA device.  ``D`` is 64 or 128,
``H >= 1``, and ``Sq`` and ``Skv`` are positive multiples of 128.  The
gradients are contiguous BF16 in the operands' layouts.  ``B``, ``H``,
``Sq`` and ``Skv`` are run-time values: no recompilation, no device-to-host
read, no synchronization.

Usage::

    from cake_pplx.cake_fmha_nondet_train import cake_fmha_nondet_train_backward

    # out, lse from the forward: out BF16 [B, Sq, H, D], lse FP32 [B, H, Sq] (natural log)
    dq, dk, dv = cake_fmha_nondet_train_backward(q, k, v, out, lse, dout, causal=True)
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Optional

import torch

from ._cake_pplx_api_fmha import cake_api

__all__ = ["cake_fmha_nondet_train_backward"]


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use.
    from ._backends import cake_fmha_nondet_train  # noqa: PLC0415

    return cake_fmha_nondet_train


@cake_api
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
    r"""Backward of dense multi-head attention: ``(dq, dk, dv)``.

    With ``P = softmax(softmax_scale * q k^T)`` (the mask applied) and ``out = P v``:
    ``dv = P^T dout``, ``dS = P * (dout v^T - rowsum(out * dout))``, ``dq = softmax_scale * dS k`` and
    ``dk = softmax_scale * dS^T q``, per batch and head.

    Parameters
    ----------
    q : torch.Tensor
        Contiguous BF16 ``[B, Sq, H, D]``, ``D`` 64 or 128, ``Sq`` a positive multiple of 128.
    k, v : torch.Tensor
        Contiguous BF16 ``[B, Skv, H, D]`` (``q``'s ``B``, ``H`` and ``D``), ``Skv`` a positive multiple of 128.
    out : torch.Tensor
        The forward's contiguous BF16 ``[B, Sq, H, D]`` output.
    lse : torch.Tensor
        The forward's contiguous FP32 ``[B, H, Sq]`` natural-log ``logsumexp`` of ``softmax_scale * q k^T`` over
        each query row, the mask applied (FlashAttention's ``softmax_lse``).
    dout : torch.Tensor
        Contiguous BF16 ``[B, Sq, H, D]`` gradient of ``out``.
    softmax_scale : float, optional
        The forward's scale; default ``D ** -0.5``.  Must be finite.
    causal : bool
        Top-left causal mask: query row ``i`` sees the keys ``j <= i``.  Needs ``Sq == Skv`` (where top-left and
        bottom-right alignment agree).

    Returns
    -------
    ``(dq, dk, dv)``: contiguous BF16 ``dq [B, Sq, H, D]``, ``dk`` and ``dv [B, Skv, H, D]`` on the operands'
    device.  An empty batch (``B == 0``) returns empty gradients, launches nothing and raises no alert.

    Raises
    ------
    ValueError
        An operand off the CUDA device of ``q``, of another dtype, rank or shape, not contiguous, or not 16-byte
        aligned (``lse``: 4-byte); ``D`` not 64 or 128; ``H < 1``; ``Sq`` or ``Skv`` not a positive multiple of
        128; ``causal`` with ``Sq != Skv``; ``B`` or ``H`` above 65535; ``B * H * max(Sq, Skv) * D >= 2**31``; a
        non-finite ``softmax_scale``.
    RuntimeError
        A device other than SM100 (``sm_100a``), SM103 (``sm_103a``) or SM107 (``sm_107a``); a local CUDA toolkit
        older than the library's ``build.json`` minimum when it is built; under
        ``torch.use_deterministic_algorithms(True)``, the alert below.

    Notes
    -----
    Not bitwise reproducible: on the three-kernel routes the dQ tiles of the key blocks of one head are added into
    an FP32 accumulator in arrival order, so ``dq`` may differ in its last bits run to run.  ``dk`` and ``dv`` are
    summed in one fixed order.  A call on the one-launch route (few CTAs, without the mask or with it) runs one
    kernel in which ``dq``, ``dk`` and ``dv`` each have one fixed order; the module makes no bit promise either way.
    Under ``torch.use_deterministic_algorithms(True)`` the call raises PyTorch's ``RuntimeError`` (``... does not
    have a deterministic implementation ...``) before it allocates or launches; with ``warn_only=True`` it warns once
    (``UserWarning``) and runs.  There is no deterministic module to fall back to.  The kernels run on the current
    stream of the operands' device (made current for the launches); nothing synchronizes.
    """
    return _pkg().cake_fmha_nondet_train_backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale=softmax_scale,
        causal=causal,
    )
