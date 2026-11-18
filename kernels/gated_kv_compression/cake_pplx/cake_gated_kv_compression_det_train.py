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

Cake ratio-2 gated KV compression for training, deterministic: the pairs of
adjacent tokens of each packed document pooled into one row, mixed per channel
by a softmax over the two gate logits, width 512 (SM100 / SM103 / SM107).

For document ``s`` with bounds ``b_s = cu_seqlens[s]``, length ``L_s`` and
``g_s`` pairs before it, pair ``k < floor(L_s / 2)`` reads the tokens
``t0 = b_s + 2k`` and ``t1 = t0 + 1`` and writes row ``r = g_s + k``; per
channel ``d``::

    p0, p1    = softmax(Z[t0, d], Z[t1, d])                    # FP32
    out[r, d] = BF16(p0 * C[t0, d] + p1 * C[t1, d])            # the one rounding

    g         = float32(dout[r, d])
    dC[t0, d] = p0 * g;   dC[t1, d] = p1 * g
    dZ[t0, d] = p0 * p1 * (C[t0, d] - C[t1, d]) * g;   dZ[t1, d] = -dZ[t0, d]

Pairs never cross a document boundary.  The last token of an odd-length
document belongs to no pair: it changes no output, and its ``dC`` and ``dZ``
rows are exactly zero.  The backward treats the BF16 cast as the identity and
uses the pairwise form of ``dZ``, accurate when the two values of a pair are
close.  The kernels compute in FP32 only (no FP64); gate gaps of 100 and more
and exact ties give finite outputs.

Entries: the explicit steps :func:`cake_gated_kv_compression_det_train_forward`
and :func:`cake_gated_kv_compression_det_train_backward`, the autograd entries
:func:`cake_gated_kv_compression_det` (separate ``C`` and ``Z``) and
:func:`cake_gated_kv_compression_det_cz` (one ``[T, 1024]`` tensor ``C | Z``,
whose gradient is written into one ``[T, 1024]`` buffer by one launch), and
:func:`cake_gated_kv_compression_det_train_memory_bytes` (the workspace and the
state retained for the backward, reported separately).

Names and mode: this is the deterministic module of the family, and the
family's only module.  Every output element is one fixed chain of FP32
operations on its own pair or token: nothing is summed across threads, CTAs or
calls and no atomic is used, so every output repeats bit for bit for identical
inputs, also in the recompute of an activation checkpoint.  There is no
non-deterministic step to trade for speed, so no ``nondet`` module exists.
There is no KCP (context-parallel) module either: the kernel is single-rank,
and under context parallelism the caller moves a boundary token so that every
rank holds whole pairs.  Every public name carries the mode word ``det``
(``cake_gated_kv_compression_det_*``, ``CakeGatedKvCompressionDet*``,
``is_cake_gated_kv_compression_det_train_supported``) except the constants
``ARCHES`` and ``SUPPORTED_COMPUTE_CAPABILITIES``.  No entry takes a
``deterministic`` keyword (passing one is a ``TypeError``), and nothing alerts
under ``torch.use_deterministic_algorithms(True)``.

A stable API (``@cake_api``) of a thin wrapper: the checks, the launch plan and
the generated kernels live in the backend package
``cake_pplx._backends.cake_gated_kv_compression_det_train``, imported on the
first call or attribute access (importing this module imports no backend).
No environment variable is read.

Operands: ``c`` and ``z`` FP32 ``[T, 512]`` with a unit feature stride, a row
stride that is a multiple of 8 elements and a 32-byte aligned base (the column
halves of one ``[T, 1024]`` tensor qualify and are read in place);
``cu_seqlens`` int32 ``[S + 1]`` on the same device, ``0 = b_0 <= ... <= b_S =
T`` (the caller's contract: the host never reads it); ``out`` and ``dout`` BF16
``[G, 512]`` contiguous, 16-byte aligned; ``dc`` and ``dz`` FP32 ``[T, 512]``
like ``c`` (the halves of one caller buffer qualify).  ``G``, the sum over
documents of ``floor(L_s / 2)``, comes from the caller (``num_pairs`` or a
preallocated ``out``).  ``T``, ``S`` and ``G`` are run-time values: no shape
list, no recompilation, no device-to-host read, no synchronization; a call can
be captured in a CUDA graph once one eager call has loaded the programs.

Usage::

    from cake_pplx.cake_gated_kv_compression_det_train import (
        cake_gated_kv_compression_det_cz,
        cake_gated_kv_compression_det_train_backward,
        cake_gated_kv_compression_det_train_forward,
    )

    # cz: FP32 [T, 1024] from one projection GEMM (C | Z); G: the pair count of cu_seqlens
    out = cake_gated_kv_compression_det_cz(cz, cu_seqlens, G)        # autograd, BF16 [G, 512]

    c, z = cz[:, :512], cz[:, 512:]
    out = cake_gated_kv_compression_det_train_forward(c, z, cu_seqlens, G)
    grad = torch.empty_like(cz)
    cake_gated_kv_compression_det_train_backward(dout, c, z, cu_seqlens, dc=grad[:, :512], dz=grad[:, 512:])
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any, Optional

import torch

from ._cake_pplx_api_gated_kv_compression import cake_api


# The backend package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg() -> ModuleType:
    from ._backends import cake_gated_kv_compression_det_train  # noqa: PLC0415 - the backend, imported on first use

    return cake_gated_kv_compression_det_train


# The stable entry points are real ``def``s here (public-API tooling reads the decorated signatures of this module);
# their bodies live in the self-contained backend package next to the generated CUDA sources.


@cake_api
def cake_gated_kv_compression_det_train_forward(
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_pairs: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The pooled rows: ``out[r] = BF16(p0 * C[t0] + p1 * C[t1])`` with ``(p0, p1) = softmax(Z[t0], Z[t1])``.

    Two launches on the current stream: the pair table of ``cu_seqlens`` (int32 ``[S + 1]``, a per-call workspace),
    then one row of 512 columns per pair, in FP32 with one rounding to BF16.

    Parameters
    ----------
    c : torch.Tensor
        FP32 ``[T, 512]`` values ``C``: unit feature stride, row stride a multiple of 8 elements (at least 512), base
        32-byte aligned.  The left half of one ``[T, 1024]`` tensor qualifies.
    z : torch.Tensor
        FP32 ``[T, 512]`` gate logits ``Z``, the same layout rules (the right half of that tensor qualifies).
    cu_seqlens : torch.Tensor
        Contiguous int32 ``[S + 1]`` document bounds on the operands' device, ``0 = b_0 <= ... <= b_S = T``.  Never
        read on the host; the caller keeps this contract.
    num_pairs : int, optional
        ``G``, the sum over documents of ``floor(length / 2)``: a Python int, never a tensor.  Required unless
        ``out`` is given (then it must equal ``out``'s rows).  A larger ``G`` gives zero rows past the pair count; a
        smaller one gives the first ``G`` rows.
    out : torch.Tensor, optional
        Preallocated contiguous BF16 ``[G, 512]``, 16-byte aligned, overlapping no input (default ``None``: a new
        tensor).

    Returns
    -------
    BF16 ``out [G, 512]`` (the caller's tensor when given).  ``G == 0`` launches nothing; ``T == 0`` or ``S == 0``
    zero-fills ``out`` without a kernel launch.

    Raises
    ------
    TypeError
        A non-tensor operand, or ``num_pairs`` that is not an int (a tensor ``num_pairs`` would synchronize).
    ValueError
        An operand off the CUDA device of ``c``, of another dtype or shape, of a layout the kernels cannot read in
        place (feature stride, row stride, base alignment); ``out`` not contiguous, misaligned or overlapping an
        input; neither ``num_pairs`` nor ``out``, or the two disagreeing; ``G`` outside ``[0, 2^26]``; more than
        ``2^24`` documents.
    RuntimeError
        A device other than SM100 (``sm_100a``), SM103 (``sm_103a``) or SM107 (``sm_107a``); a local CUDA toolkit
        older than the library's ``build.json`` minimum when it is built.

    Notes
    -----
    Deterministic: ``out`` repeats bit for bit for identical inputs, whatever ``T``, ``S``, ``G`` or the device's
    SM count.  No alert under ``torch.use_deterministic_algorithms(True)``.  Nothing synchronizes.
    """
    return _pkg().cake_gated_kv_compression_det_train_forward(c, z, cu_seqlens, num_pairs, out=out)


@cake_api
def cake_gated_kv_compression_det_train_backward(
    dout: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    dc: Optional[torch.Tensor] = None,
    dz: Optional[torch.Tensor] = None,
    pair_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The gradients ``(dC, dZ)`` of the pooled rows: per pair ``dC[t0] = p0 g``, ``dC[t1] = p1 g``,
    ``dZ[t0] = p0 p1 (C[t0] - C[t1]) g`` and ``dZ[t1] = -dZ[t0]`` with ``g = float32(dout[r])``; the rows of an
    odd-length document's last token are exactly zero.

    One launch writes every ``dC`` and ``dZ`` row of a token below ``cu_seqlens[S]`` exactly once (plus the pair-table
    launch when ``pair_offsets`` is not given).  The BF16 cast of the forward counts as the identity.

    Parameters
    ----------
    dout : torch.Tensor
        Contiguous BF16 ``[G, 512]`` gradient of ``out``, 16-byte aligned.  ``G`` must be at least the pair count of
        ``cu_seqlens`` (its rows past the pair count are not read).
    c, z, cu_seqlens : torch.Tensor
        The forward's operands (see :func:`cake_gated_kv_compression_det_train_forward`).
    dc, dz : torch.Tensor, optional
        FP32 ``[T, 512]`` gradients to write, with the layout rules of ``c`` (the two halves of one ``[T, 1024]``
        buffer qualify), overlapping neither each other nor any input (default ``None``: new tensors).
    pair_offsets : torch.Tensor, optional
        The pair table of ``cu_seqlens``: contiguous int32 ``[S + 1]``, ``pair_offsets[0] = 0`` and
        ``pair_offsets[s + 1] = pair_offsets[s] + (cu_seqlens[s + 1] - cu_seqlens[s]) // 2``.  Given, the call skips
        the pair-table launch (the autograd backward passes its forward's table).

    Returns
    -------
    ``(dc, dz)``: FP32 ``[T, 512]`` each (the caller's tensors when given), every row written.  ``T == 0`` launches
    nothing; ``G == 0`` or ``S == 0`` (no pair: every token is a tail token) zero-fills both without a kernel launch.

    Raises
    ------
    TypeError
        A non-tensor operand.
    ValueError
        An operand off the CUDA device of ``dout``, of another dtype or shape, of a layout the kernels cannot read or
        write in place; ``dc`` and ``dz`` sharing an element or overlapping an input; ``pair_offsets`` not
        ``[S + 1]``.
    RuntimeError
        A device other than SM100, SM103 or SM107; a local CUDA toolkit older than the ``build.json`` minimum.

    Notes
    -----
    Deterministic: ``dc`` and ``dz`` repeat bit for bit for identical inputs.  No alert under
    ``torch.use_deterministic_algorithms(True)``.  Nothing synchronizes.
    """
    return _pkg().cake_gated_kv_compression_det_train_backward(
        dout, c, z, cu_seqlens, dc=dc, dz=dz, pair_offsets=pair_offsets
    )


@cake_api
def cake_gated_kv_compression_det(
    c: torch.Tensor, z: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int
) -> torch.Tensor:
    """Autograd-enabled :func:`cake_gated_kv_compression_det_train_forward` on separate ``c`` and ``z``.

    Parameters
    ----------
    c, z : torch.Tensor
        FP32 ``[T, 512]`` values and gate logits (may require gradients), the layout rules of the forward.
    cu_seqlens : torch.Tensor
        Int32 ``[S + 1]`` document bounds on the device.
    num_pairs : int
        ``G``, the pair count of ``cu_seqlens``, or more: a larger ``G`` gives zero rows past the pair count (their
        incoming gradient is not read).  Never less: unlike the explicit forward, an autograd entry does not support a
        smaller ``G``, since its backward would leave the gradient rows of the tokens of the pairs at or past ``G``
        uninitialized.

    Returns
    -------
    BF16 ``out [G, 512]``.  Its backward is :func:`cake_gated_kv_compression_det_train_backward` (one launch for
    ``dC`` and ``dZ``; the incoming gradient is copied once when it is not contiguous and 16-byte aligned).  The node
    saves ``c``, ``z``, ``cu_seqlens`` and the forward's int32 ``[S + 1]`` pair table only, and works under
    ``torch.utils.checkpoint(..., use_reentrant=False)`` (the recompute gives the same bits).
    """
    return _pkg().cake_gated_kv_compression_det(c, z, cu_seqlens, num_pairs)


@cake_api
def cake_gated_kv_compression_det_cz(cz: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int) -> torch.Tensor:
    """Autograd-enabled forward on one FP32 ``cz [T, 1024]`` that holds ``C`` in columns ``[0, 512)`` and ``Z`` in
    ``[512, 1024)`` (the output of one projection GEMM).

    Parameters
    ----------
    cz : torch.Tensor
        FP32 ``[T, 1024]``: unit feature stride, row stride a multiple of 8 elements (at least 1024), base 32-byte
        aligned (may require gradients).
    cu_seqlens : torch.Tensor
        Int32 ``[S + 1]`` document bounds on the device.
    num_pairs : int
        ``G``, the pair count of ``cu_seqlens``, or more: a larger ``G`` gives zero rows past the pair count (their
        incoming gradient is not read).  Never less: unlike the explicit forward, an autograd entry does not support a
        smaller ``G``, since its backward would leave the gradient rows of the tokens of the pairs at or past ``G``
        uninitialized.

    Returns
    -------
    BF16 ``out [G, 512]``.  The backward allocates one FP32 ``[T, 1024]`` gradient and one launch writes ``dC`` and
    ``dZ`` into its halves: no slice-backward zero fill or copy.  The node saves ``cz``, ``cu_seqlens`` and the
    forward's pair table only, and works under ``torch.utils.checkpoint(..., use_reentrant=False)``.
    """
    return _pkg().cake_gated_kv_compression_det_cz(cz, cu_seqlens, num_pairs)


@cake_api
def cake_gated_kv_compression_det_train_memory_bytes(num_tokens: int, num_docs: int) -> dict[str, int]:
    """The workspace and the state retained for the backward, in bytes, reported separately.

    Parameters
    ----------
    num_tokens : int
        ``T``.
    num_docs : int
        ``S``.

    Returns
    -------
    ``dict`` with

    * ``forward_workspace``: ``4 (S + 1)``, the pair table an explicit forward allocates and frees;
    * ``backward_workspace``: ``4 (S + 1)``, the pair table an explicit backward without ``pair_offsets`` allocates
      and frees.  The autograd backward allocates none, except a ``1024 G``-byte contiguous copy of an incoming
      gradient that is not contiguous and 16-byte aligned (for example the expanded gradient of
      ``out.sum().backward()``), freed on return; the report takes no ``G`` and leaves this copy out;
    * ``retained_for_backward``: ``4 (S + 1)``, the bytes the autograd node allocates in its forward and keeps until
      its backward (its pair table);
    * ``retained_inputs``: ``4096 T + 4 (S + 1)``, the caller's tensors the autograd node keeps referenced (``C`` and
      ``Z`` or ``cz``, and ``cu_seqlens``; no copy).

    The outputs (``out``, ``dC``, ``dZ``) are not counted.  A call that launches nothing allocates no workspace.
    """
    return _pkg().cake_gated_kv_compression_det_train_memory_bytes(num_tokens, num_docs)


# The backend objects this module serves under its own names, resolved on first access so that importing this module imports no
# backend.
_LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "is_cake_gated_kv_compression_det_train_supported",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(_pkg(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeGatedKvCompressionDetCzFunction",  # noqa: F822
    "CakeGatedKvCompressionDetFunction",  # noqa: F822
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
    "is_cake_gated_kv_compression_det_train_supported",  # noqa: F822
]
