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

Cake Engram row lookup for training (SM100 / SM103 / SM107): rows of a
quantized table, FP8 E4M3 values with one E8M0 scale per 32 columns, gathered
by row id and written as BF16.

Entry: :func:`cake_engram_lookup_det_train_forward`, the forward step without
autograd: ``out[p, 32*g + j] = bf16(float(weight[r, 32*g + j]) *
float(scale[r, g]))`` for ``r = row_ids[p]``, the exact product rounded once to
BF16.  The family has no backward and no autograd entry: the table is not
trained through this call (``out`` does not require a gradient).
:func:`cake_engram_lookup_det_train_split_packed` makes the ``weight`` and
``scale`` views of a packed ``uint8 [R, 264]`` table.

The mode is part of the name: the entry takes no ``deterministic`` keyword
(passing one is a ``TypeError``).  The family has a deterministic module only:
every output element is one product rounded once and nothing is summed across
threads, CTAs or calls, so the output repeats bit for bit and there is no
non-deterministic variant to offer.  No call alerts under
``torch.use_deterministic_algorithms(True)``.

A stable API (``@cake_api``; no ``ExperimentalWarning``) of a thin wrapper:
the checks, the launch plan and the generated kernel live in the backend
package ``cake_pplx._backends.cake_engram_lookup_det_train``, imported on
first use; this module forwards every argument.  No environment variable is
read.

Operands: ``weight`` ``float8_e4m3fn [R, 256]`` and ``scale``
``float8_e8m0fnu [R, 8]``, each with a unit column stride, a row stride that
is a multiple of 8 elements and an 8-byte aligned start: two contiguous
tensors, or two views into one packed ``[R, 264]`` byte buffer (row stride
264).  ``row_ids`` contiguous int64 of any shape, ``[T, 24]`` for a direct
lookup and ``[M]`` for deduplicated or routed rows.  All on one CUDA device.
``R``, ``T`` and ``M`` are run-time values: no recompilation, no
device-to-host read, no synchronization.

Usage::

    from cake_pplx.cake_engram_lookup_det_train import (
        cake_engram_lookup_det_train_forward,
        cake_engram_lookup_det_train_split_packed,
    )

    # a local shard: weight float8_e4m3fn [R, 256], scale float8_e8m0fnu [R, 8]
    out = cake_engram_lookup_det_train_forward(weight, scale, row_ids)  # row_ids int64 [T, 24] -> BF16 [T, 24, 256]

    # rows fetched from other ranks: one packed uint8 [M, 264] buffer
    weight_m, scale_m = cake_engram_lookup_det_train_split_packed(fetched)
    out_m = cake_engram_lookup_det_train_forward(weight_m, scale_m, ids)  # ids int64 [N] -> BF16 [N, 256]
"""

from __future__ import annotations

import functools
from types import ModuleType

import torch

from ._cake_pplx_api_engram_lookup import cake_api

__all__ = [
    "cake_engram_lookup_det_train_forward",
    "cake_engram_lookup_det_train_split_packed",
]


@functools.cache
def _pkg() -> ModuleType:
    # The backend package, imported on first use.
    from ._backends import cake_engram_lookup_det_train  # noqa: PLC0415

    return cake_engram_lookup_det_train


@cake_api
def cake_engram_lookup_det_train_forward(
    weight: torch.Tensor,
    scale: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    r"""The Engram row lookup: the rows ``row_ids`` of the quantized table, as BF16.

    For each looked-up row ``p`` with ``r = row_ids[p]``, each group ``g`` in ``[0, 8)`` and ``j`` in ``[0, 32)``:
    ``out[p, 32*g + j] = bf16(float(weight[r, 32*g + j]) * float(scale[r, g]))``.

    Parameters
    ----------
    weight : torch.Tensor
        ``float8_e4m3fn [R, 256]``: unit column stride, row stride a multiple of 8 elements, start on an 8-byte
        boundary.  A contiguous tensor or the value view of a packed ``[R, 264]`` table.
    scale : torch.Tensor
        ``float8_e8m0fnu [R, 8]``, one scale per 32 columns of the same row: the same stride and alignment rules.  A
        contiguous tensor or the scale view of the same packed table.
    row_ids : torch.Tensor
        Contiguous int64 of any shape (``[T, 24]`` or ``[M]``): the rows to look up, in ``[0, R)``; repeats are
        allowed.  <<ROW_ID_RANGE: Nothing checks them (a host check would synchronize): an id outside ``[0, R)``
        reads memory outside the table, and its output row is undefined.>>

    Returns
    -------
    ``out``: a new contiguous BF16 tensor of shape ``(*row_ids.shape, 256)`` on the operands' device.  Each element
    is the exact product rounded once to BF16 (round to nearest even): FP32-subnormal products are kept, products
    past the FP32 range are ``±inf``, and an E8M0 byte ``0xFF`` (NaN) makes its 32 columns NaN.  An empty
    ``row_ids`` (``T == 0`` or ``M == 0``) returns an empty ``out`` and launches nothing.

    Raises
    ------
    ValueError
        An operand that is not a tensor, not on a CUDA device or not on ``weight``'s device; another dtype, rank or
        shape (``weight [R, 256]``, ``scale [R, 8]`` with ``weight``'s ``R``); a column stride other than 1, a row
        stride that is not a multiple of 8 elements or a start off an 8-byte boundary (``weight``, ``scale``);
        ``row_ids`` not contiguous or with ``2**31 - 8`` or more elements; a non-empty ``row_ids`` with ``R == 0``.
    RuntimeError
        A device other than SM100 (``sm_100a``), SM103 (``sm_103a``) or SM107 (``sm_107a``) when ``row_ids`` is not
        empty; a local CUDA toolkit older than the library's ``build.json`` minimum when it is built.

    Notes
    -----
    Bitwise reproducible: the output depends only on the bytes of the looked-up rows, for both layouts and every
    ``row_ids`` shape.  The call never reads torch's deterministic-algorithms setting: under
    ``torch.use_deterministic_algorithms(True)``, also with ``warn_only=True``, it runs unchanged and raises no
    alert or warning.  Workspace: none; the call allocates only ``out`` (``row_ids.numel() * 512`` bytes) and
    retains no state.  The kernel runs on the current stream of the operands' device (made current for the
    launch); nothing synchronizes, and a call can be captured in a CUDA graph after one eager call has built the
    library.  The output is a fresh tensor that does not require a gradient, so the call works inside
    ``torch.utils.checkpoint(..., use_reentrant=False)``, which repeats it with the same bits.
    """
    return _pkg().cake_engram_lookup_det_train_forward(weight, scale, row_ids)


@cake_api
def cake_engram_lookup_det_train_split_packed(buf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(weight, scale)`` views of a packed table, for :func:`cake_engram_lookup_det_train_forward`.

    Parameters
    ----------
    buf : torch.Tensor
        ``uint8 [R, 264]`` with a unit column stride: each row holds its 256 E4M3 value bytes, then its 8 E8M0 scale
        bytes.  Any device.

    Returns
    -------
    ``(weight, scale)``: ``buf[:, :256]`` as ``float8_e4m3fn [R, 256]`` and ``buf[:, 256:]`` as
    ``float8_e8m0fnu [R, 8]``.  Views of ``buf``'s storage with its row stride (264 for a contiguous ``buf``):
    nothing is copied.

    Raises
    ------
    ValueError
        ``buf`` not a tensor, not ``uint8``, not ``[R, 264]`` or without a unit column stride.
    """
    return _pkg().cake_engram_lookup_det_train_split_packed(buf)
