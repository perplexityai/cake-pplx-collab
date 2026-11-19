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

Cake DeepSeek Sparse Attention (top-k sparse MLA, absorbed queries, 64 query
heads) training forward and backward on SM100 / SM103 / SM107, the
deterministic package: the stable entry points over the backend package
``cake_pplx._backends.cake_dsa_sparse_attention_det_train``.

Numerical mode: the module name chooses the package, and no entry point takes a
mode keyword.  Every output repeats bit for bit for the same inputs, device
architecture and package version: ``out``, ``lse`` and the derived row lengths
(one fixed-order forward kernel), ``dq`` (summed in a fixed tile order, written
once per row) and ``dkv`` -- the ``kv_latent`` / ``k_rope`` or ``kv``
gradients, or the rows added into ``dkv_acc``.  The dK / dV contributions are
added as integers on a power-of-two grid per destination row (integer addition
does not depend on the order), so the result does not depend on the SM count,
the launch grid, other work on the device, CUDA-graph or eager execution,
activation-checkpoint recompute, repeated destination rows of ``dkv_dst_map``
or the token order.  Bits are not promised across package versions, device
architectures or against the non-deterministic package
``cake_pplx.cake_dsa_sparse_attention_nondet_train``.  Nothing raises or warns
under ``torch.use_deterministic_algorithms(True)``.

Entry points (each an autograd call: forward now, backward through autograd)::

    cake_dsa_sparse_attention_det               flat, rope + latent scores
    cake_dsa_sparse_attention_det_varlen        packed documents, rope + latent scores
    cake_dsa_sparse_attention_det_nope          flat, latent scores only (NoPE)
    cake_dsa_sparse_attention_det_nope_varlen   packed documents, NoPE

Explicit steps (no autograd; the forward returns what the backward reads), for
the flat rope form, the flat NoPE form and the packed NoPE form::

    cake_dsa_sparse_attention_det_train_forward
    cake_dsa_sparse_attention_det_train_backward
    cake_dsa_sparse_attention_det_train_nope_forward
    cake_dsa_sparse_attention_det_train_nope_backward
    cake_dsa_sparse_attention_det_train_nope_varlen_forward
    cake_dsa_sparse_attention_det_train_nope_varlen_backward

The packed forward also returns the global rows and row lengths that its
backward reads.

The NoPE entries take the ``sink`` keyword of the non-deterministic package's
entries with ``None`` only: a sink raises ``NotImplementedError`` here.

Their autograd Functions, ``CakeDSASparseAttentionDetFunction`` (both rope
entries), ``CakeDSASparseAttentionDetNopeFunction`` and
``CakeDSASparseAttentionDetNopeVarlenFunction``, resolve from the backend
package on first access.  The signatures match the non-deterministic package's
entries; ``key_passes`` and ``dkv_direct`` are accepted there and validated
here.  The backward of either program may split its main stage into key-range
passes (as the non-deterministic package's policy does, or as ``key_passes``
asks; the NoPE policy keeps one pass unless ``key_passes`` asks for more).
Every backward accumulates in one form, and its dK / dV do not depend on the
pass count.  The flat rope entry and the rope step backward take
``num_segments``, the packed segment count of the key row (default 1): a
count above 1 plans one pass unless ``key_passes`` overrides, as the packed
entry does for the same rows.

Usage::

    from cake_pplx.cake_dsa_sparse_attention_det_train import (
        cake_dsa_sparse_attention_det,
    )

    out = cake_dsa_sparse_attention_det(q_latent, q_rope, kv_latent, k_rope, indices)
    out.float().square().sum().backward()  # every gradient repeats bit for bit
"""

import functools
import os
from typing import Any, Callable, Optional, Union

import torch

from ._cake_pplx_api_dsa_sparse_attention import cake_api

# Stable entry points (``@cake_api``) over the backend package; validation
# of the backend contract, the workspace, launch binding, autograd wrappers and
# JIT registration live in
# cake_pplx._backends.cake_dsa_sparse_attention_det_train, which this
# core module imports on first use only: importing this module loads nothing
# from the backend package.  The backend takes every switch as a keyword argument
# and reads no environment variable; the variables below are this module's
# knobs, resolved here and passed down.

__all__ = [
    "BINDING_CACHE_CAPACITY_ENV",
    "CHECK_DST_MAP_ENV",
    "CakeDSASparseAttentionDetFunction",  # noqa: F822 (resolved by __getattr__)
    "CakeDSASparseAttentionDetNopeFunction",  # noqa: F822
    "CakeDSASparseAttentionDetNopeVarlenFunction",  # noqa: F822
    "cake_dsa_sparse_attention_det",
    "cake_dsa_sparse_attention_det_nope",
    "cake_dsa_sparse_attention_det_nope_varlen",
    "cake_dsa_sparse_attention_det_varlen",
    "cake_dsa_sparse_attention_det_train_backward",
    "cake_dsa_sparse_attention_det_train_forward",
    "cake_dsa_sparse_attention_det_train_nope_backward",
    "cake_dsa_sparse_attention_det_train_nope_forward",
    "cake_dsa_sparse_attention_det_train_nope_varlen_backward",
    "cake_dsa_sparse_attention_det_train_nope_varlen_forward",
]

# The backend's autograd Functions, re-exported on first access (module ``__getattr__``).
_FROM_BACKEND = (
    "CakeDSASparseAttentionDetFunction",
    "CakeDSASparseAttentionDetNopeFunction",
    "CakeDSASparseAttentionDetNopeVarlenFunction",
)

# Unset, empty or ``0`` (default): the values of ``dkv_dst_map`` are not
# checked; anything else validates them on every backward (one device sync).
CHECK_DST_MAP_ENV = "CAKE_PPLX_DSA_CHECK_DST_MAP"
# A positive integer: the capacity of the backend's binding cache (64 by
# default).  Read when this module is imported, applied when it first reaches
# the backend.
BINDING_CACHE_CAPACITY_ENV = "CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY"


def _check_dst_map_from_environment() -> bool:
    """``CAKE_PPLX_DSA_CHECK_DST_MAP`` as the backend's ``check_dst_map``
    keyword: unset, empty or ``0`` is ``False`` (default), anything else
    ``True``."""
    return os.environ.get(CHECK_DST_MAP_ENV, "0") not in ("", "0")


def _binding_cache_capacity_from_environment() -> Optional[int]:
    """``CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY`` as a positive integer,
    or ``None`` when unset or blank (the backend's configured capacity stays).
    Anything else raises ``ValueError``."""
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
        )
    return capacity


@functools.cache
def _cake_backend():
    # The backend is imported on first use (experimental containment: core
    # reaches kernels and their dependencies through a deferred import) and
    # the binding-cache capacity read at import is applied to it once.
    from ._backends.cake_dsa_sparse_attention_det_train import cake_backend

    if _BINDING_CACHE_CAPACITY is not None:
        cake_backend.BINDING_CACHE.configure(capacity=_BINDING_CACHE_CAPACITY)
    return cake_backend


def _backend(backend: str):
    if backend != "cake":
        raise ValueError(
            "DSA sparse-attention training currently supports backend='cake'"
        )
    return _cake_backend()


def _check_num_segments(num_segments: Any) -> int:
    """``num_segments`` as the backend takes it: a Python ``int`` of at least 1.  ``bool``, a tensor (whose value
    would need a device synchronization), a float, a string or a count below 1 raises ``ValueError`` before the
    backend is reached."""
    if (
        isinstance(num_segments, bool)
        or not isinstance(num_segments, int)
        or num_segments < 1
    ):
        shown = (
            repr(num_segments)
            if isinstance(num_segments, (bool, int, float, str))
            else f"a {type(num_segments).__name__}"
        )
        raise ValueError(
            "num_segments must be a positive Python int (the packed segment count of the key row), "
            f"got {shown}"
        )
    return int(num_segments)


def _refuse_sink(sink: Optional[torch.Tensor]) -> None:
    """The NoPE entries take the ``sink`` keyword of the non-deterministic package's entries, ``None`` only: this
    package has no sink build."""
    if sink is not None:
        raise NotImplementedError(
            "a sink is not supported by the deterministic package: it has no sink build of the NoPE kernels; "
            "the non-deterministic package's flat NoPE entries take one"
        )


@cake_api
def cake_dsa_sparse_attention_det(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[Union[torch.Tensor, Callable]] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
    num_segments: int = 1,
):
    r"""Differentiable top-k sparse MLA attention with absorbed queries (DeepSeek
    Sparse Attention) for training, 64 query heads, SM100 / SM103 / SM107 only;
    every output repeats bit for bit.

    ``O = softmax(softmax_scale * (q_rope . k_rope^T + q_latent . kv_latent^T)) . kv_latent``
    over the keys a row's ``indices`` select (K = V = the latent).

    Strided inputs are consumed in place (no host copy): the kernels read the
    query operands through TMA descriptors encoded from the tensor's own head
    and token strides and gather the key operands by row stride.  Admitted
    layouts: unit innermost stride; head / token / key-row strides that are
    positive multiples of 8 elements (16 bytes, the TMA descriptor stride
    granule); head stride and key row stride at least the slice width; a
    16-byte-aligned base address.  Outputs equal the contiguous path bitwise.

    Parameters
    ----------
    q_latent : torch.Tensor
        BF16 ``[T, 64, 512]`` absorbed query latents; contiguous or a view of
        a packed ``[T, 64, 576]`` query (head stride 576).
    q_rope : torch.Tensor
        BF16 ``[T, 64, 64]`` query rope part; contiguous, a view of the packed
        ``[T, 64, 576]`` query, or the ``192:256`` channel slice of the
        pre-absorption ``[T, 64, 256]`` query (head stride 256, token stride
        16384, storage offset 192 elements).
    kv_latent : torch.Tensor
        BF16 ``[S, 512]`` key/value latent (shared K = V); columns ``0:512`` of
        a packed ``[S, 576]`` or ``[S, 704]`` row (a 128-channel indexer key
        stored alongside) are accepted.
    k_rope : torch.Tensor
        BF16 ``[S, 64]`` key rope part; columns ``512:576`` of the same packed
        row are accepted.
    indices : torch.Tensor
        int32 ``[T, topk]`` **global** key rows into ``kv_latent`` / ``k_rope``.
        ``-1`` or a value ``>= S`` marks an invalid slot; invalid slots may
        appear anywhere in the row.  Any positive ``topk``.
    topk_length : Optional[torch.Tensor]
        int32 ``[T]``; slots ``>= topk_length[t]`` are invalid regardless of
        their content.
    softmax_scale : Optional[float]
        Defaults to ``576 ** -0.5``.
    return_lse : bool
        Also return the natural-log logsumexp ``[T, 64]`` FP32 of the scaled
        scores over the valid keys (``-inf`` for fully masked rows).
    key_passes : Optional[int]
        Validated as in the non-deterministic package (``None`` or
        ``1 <= key_passes <= S``).  ``None`` lets that package's key-pass policy
        choose; a count above 1 runs the backward's main stage as that many
        key-range passes (a compaction and a main pass each, with their scratch
        in the backward's workspace).  dK / dV are bit for bit those of one
        pass; dQ is summed through an FP32 partial between passes.
    backend : str
        Only ``"cake"``.
    dkv_acc : Optional[torch.Tensor | Callable]
        Caller-owned FP32 ``[S_dst, >= 576]`` buffer (row stride a multiple of
        4 elements, 16-byte aligned) into which the backward adds the key
        gradients in place (the caller zeroes it): latent dimensions in columns
        ``0:512``, rope dimensions in ``512:576``, one FP32 addition per
        element (an element that receives no contribution keeps its bits).
        When given, the ``kv_latent`` / ``k_rope`` gradients of the autograd
        call are ``None``.  Instead of the tensor, a zero-argument callable
        that returns it -- or an ``(accumulator, dkv_dst_map)`` pair -- may be
        given; the backward calls it, so the accumulator need not exist at
        forward time (under activation checkpointing a tensor passed at
        forward time is alive for every layer at once).  The autograd entries
        save every tensor they produce for the backward through
        ``ctx.save_for_backward`` and retain nothing else of a checkpointed
        forward; a tensor ``dkv_acc`` stays caller-owned.
    dkv_dst_map : Optional[torch.Tensor]
        int32 ``[S]`` destination row of ``dkv_acc`` for every key row
        (default identity).  Repeated destinations are allowed (the rows of a
        context-parallel window that map onto one parameter row) and combine
        exactly, as integers, before the one FP32 addition.  Every value must
        lie in ``[0, S_dst)``; the kernels do not range-check the map (see
        ``check_dst_map``).
    dkv_direct : Optional[bool]
        Accepted for signature parity with the non-deterministic package
        (``None``, ``True`` or ``False``) and without effect: the accumulation
        into ``dkv_acc`` has one form here.
    check_dst_map : Optional[bool]
        Validate the values of ``dkv_dst_map`` (every destination row in
        ``[0, S_dst)``) on every backward, at the cost of one device
        synchronization, raising ``ValueError`` on a violation (skipped during
        CUDA graph capture).  ``None`` (default) reads
        ``CAKE_PPLX_DSA_CHECK_DST_MAP``: unset, empty or ``0`` (its
        default) means no check, anything else enables it.  ``True`` /
        ``False`` decide explicitly and ignore the variable.
    num_segments : int
        The packed segment count of the key row, as in the
        non-deterministic package: ``len(cu_seqlens_k) - 1`` when ``indices``
        hold the global rows of a packed batch (a Python ``int``, read on the
        host).  ``1`` (default) keeps the key-pass policy of a one-document
        row; a count above 1 plans one pass unless ``key_passes`` overrides,
        as :func:`cake_dsa_sparse_attention_det_varlen` does for the same rows.
        dK / dV do not depend on it; dQ is summed through an FP32 partial
        between passes, so its bits follow the plan.  ``bool``, a tensor or a
        value below 1 raises ``ValueError``.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, 64, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.

    Gradients flow to ``q_latent``, ``q_rope``, ``kv_latent`` and ``k_rope``.
    BF16 operands into the tensor cores, FP32 accumulation; the backward
    recomputes the scores from the BF16 inputs and forms ``delta`` in FP32
    from the BF16 output and its saved BF16 residual ``o_lo``.  The dK / dV contributions are
    rounded once to the row's integer grid and summed as integers; each
    gradient element is then formed with one more rounding (and, with
    ``dkv_acc``, one FP32 addition).  The backward reuses the storage of the
    saved ``o_lo`` as its integer accumulator when it fits (``S_dst`` up to
    about ``27.7 T`` rows) and allocates it otherwise; a second backward
    through the same graph (``retain_graph=True``) first recomputes the
    forward into the saved tensors.  The kernels apply no positional mask: the
    index rows define the key set.

    Host side: the inputs are validated and bound on the first call for an
    input geometry (shape, stride, dtype and alignment of every input plus the
    scale); later calls launch from the remembered plans with freshly allocated
    outputs
    (``cake_pplx._backends.cake_dsa_sparse_attention_det_train.cake_backend.BINDING_CACHE``, 64
    bindings by default, least recently used first out;
    ``CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY`` -- a positive integer,
    read when this module is imported -- sets the capacity through
    ``BINDING_CACHE.configure(capacity=...)``).  No host synchronization
    unless ``check_dst_map`` asks for one.  Every output repeats bit for bit
    for the same inputs, ``key_passes`` and ``num_segments``.
    """
    num_segments = _check_num_segments(num_segments)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_det(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=num_segments,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


@cake_api
def cake_dsa_sparse_attention_det_varlen(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    causal: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[Union[torch.Tensor, Callable]] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
):
    r"""Packed multi-document form of :func:`cake_dsa_sparse_attention_det` (SM100 /
    SM103 / SM107); every output repeats bit for bit.

    Documents are packed along the first dimension of the query tensors
    (``T = cu_seqlens_q[-1]``) and of the key tensors (``S = cu_seqlens_k[-1]``);
    ``cu_seqlens_q`` and ``cu_seqlens_k`` are independent and query and key
    lengths may differ per document: the query segment is the tail of its key
    prefix, so query ``local_q`` of document ``d`` sits at key position
    ``(seqlen_k[d] - seqlen_q[d]) + local_q`` (a context-parallel tail or a
    GLM-style packed batch).  ``gather_kv_indices`` ``[T, topk]`` int32 hold
    key positions relative to the row's document (``-1`` or ``>= seqlen_k[d]``
    invalid) and are offset by ``cu_seqlens_k`` on device in one launch right
    before the forward kernel; each row is compacted (valid keys first, in slot
    order, then ``-1``) so rows with holes cost their valid slots, and a caller
    ``topk_length`` is a position cut applied before compaction.  Zero-length
    query segments contribute no rows; a row whose every slot is invalid gives
    ``out = 0``, ``lse = -inf`` and zero gradients.  The kernels apply no
    positional mask of their own; the offset index rows define the key sets.
    ``max_seqlen_q`` / ``max_seqlen_k`` are accepted for signature parity with
    other varlen attention entry points and are not read on the host.  See
    :func:`cake_dsa_sparse_attention_det` for the other arguments and the
    returns; the strided query slices and packed key rows described there are
    accepted here too.

    Parameters
    ----------
    causal : bool
        With ``False`` (default) the index row is taken as is -- only ``-1`` /
        out-of-range slots are dropped.  With ``True`` a slot that selects a
        key after the query's own position (``idx > (seqlen_k[d] -
        seqlen_q[d]) + local_q``) is invalid too, so a top-k selector need not
        enforce causality itself.
    key_passes : Optional[int]
        As in :func:`cake_dsa_sparse_attention_det`: the key-range passes of
        the backward's main stage (``None``: the key-pass policy chooses);
        dK / dV are bit for bit those of one pass.
    dkv_direct : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`: accepted for parity,
        validated, without effect.
    check_dst_map : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`: ``None`` (default) reads
        ``CAKE_PPLX_DSA_CHECK_DST_MAP`` (unset, empty or ``0`` = no
        check); ``True`` validates the values of ``dkv_dst_map`` on every
        backward, ``False`` never does.
    """
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_det_varlen(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


@cake_api
def cake_dsa_sparse_attention_det_nope(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[Union[torch.Tensor, Callable]] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
):
    r"""Differentiable top-k sparse MLA attention without a rope part (NoPE:
    the whole score is the 512-dimensional latent product) for training, 64
    query heads, SM100 / SM103 / SM107 only; every output repeats bit for bit.

    ``O = softmax(softmax_scale * (q . kv^T)) . kv`` over the keys a row's
    ``indices`` select (K = V = the shared latent ``kv``).  The native kernels
    of this form read no rope operands and produce one combined gradient per
    key row, ``dkv = dK + dV``, natural layout, FP32.

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T, 64, 512]`` absorbed query latents; contiguous or a strided
        token / head view (unit innermost stride, head and token strides
        positive multiples of 8 elements, head stride at least 512,
        16-byte-aligned base), consumed in place.
    kv : torch.Tensor
        BF16 ``[S, 512]`` shared key/value latent, or its one-head form
        ``[S, 1, 512]`` (squeezed to ``[S, 512]`` on the host; the gradient
        reaches the ``[S, 1, 512]`` leaf through that view).
    indices : torch.Tensor
        int32 ``[T, K_slots]`` **global** key rows into ``kv``.  Any
        ``K_slots >= 1``; ``-1`` or a value ``>= S`` marks an invalid slot,
        anywhere in the row.  A row whose every slot is invalid gives
        ``out = 0``, ``lse = -inf`` and zero gradients.
    topk_length : Optional[torch.Tensor]
        int32 ``[T]``; slots ``>= topk_length[t]`` are invalid regardless of
        their content.  Without it the route follows the table size
        (``cake_backend.NOPE_CONVERT_SLOT_LIMIT`` slots): below the limit the
        per-row lengths (last valid slot + 1) are derived by the forward kernel
        while it runs and saved for the backward with the rows as given; at or
        above it the rows are compacted on device first (valid keys first in
        slot order, then ``-1``, the valid count as the length) and the
        compacted pair is saved.  Hole-free rows give the same bytes on either
        route.
    softmax_scale : Optional[float]
        Defaults to ``1 / 16``.
    return_lse : bool
        Also return the natural-log logsumexp ``[T, 64]`` FP32 of the scaled
        scores over the valid keys (``-inf`` for fully masked rows).
    key_passes : Optional[int]
        As in :func:`cake_dsa_sparse_attention_det`: the key-range passes of
        the backward's main stage.  ``None`` lets the NoPE key-pass policy
        choose (one pass); a count above 1 runs that many key-range passes (a
        compaction and a main pass each, with their scratch in the backward's
        workspace).  dK + dV is bit for bit that of one pass; dQ is summed
        through an FP32 partial between passes.
    backend : str
        Only ``"cake"``.
    dkv_acc : Optional[torch.Tensor | Callable]
        Caller-owned FP32 ``[S_dst, >= 512]`` buffer (row stride a multiple of
        4 elements and at least 512, 16-byte-aligned base) into which the
        backward adds ``dK + dV`` in place (columns ``0:512``, one FP32
        addition per element; the caller zeroes it).  When given, the ``kv``
        gradient of the autograd call is ``None``.  A zero-argument callable
        that returns it -- or an ``(accumulator, dkv_dst_map)`` pair -- may be
        given instead, as in :func:`cake_dsa_sparse_attention_det`.
    dkv_dst_map : Optional[torch.Tensor]
        int32 ``[S]`` destination row of ``dkv_acc`` for every key row
        (default identity).  Repeated destinations are allowed and combine
        exactly.  Every value must lie in ``[0, S_dst)``; the kernels do not
        range-check the map (see ``check_dst_map``).
    dkv_direct : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`: accepted for parity,
        validated, without effect.
    check_dst_map : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, 64, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.

    Gradients flow to ``q`` and ``kv``.  ``dkv`` is formed in FP32 from the
    integer sums and handed over in FP32: through ``dkv_acc`` when given,
    otherwise as the ``kv`` gradient -- which the autograd engine casts to the
    dtype of the ``kv`` leaf (BF16 for a BF16 leaf); call
    ``cake_pplx._backends.cake_dsa_sparse_attention_det_train.cake_backend.backward_nope``
    or pass ``dkv_acc`` to keep the FP32 values.  The backward reuses the
    storage of the saved ``o_lo`` as scratch (see
    :func:`cake_dsa_sparse_attention_det`).

    The incoming ``dout`` may be a head-major view (strides ``[512, T * 512,
    1]``, the layout an output projection hands back) or any strided
    ``[T, 64, 512]`` view under the layout rules of ``q``: it is read in place
    with no host copy.

    Host side as in :func:`cake_dsa_sparse_attention_det` (the same
    ``BINDING_CACHE``, under keys disjoint from the rope entries).
    """
    _refuse_sink(sink)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_det_nope(
        q,
        kv,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


@cake_api
def cake_dsa_sparse_attention_det_nope_varlen(
    q: torch.Tensor,
    kv: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    causal: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[Union[torch.Tensor, Callable]] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
    check_cu_seqlens: bool = False,
):
    r"""Packed multi-document form of :func:`cake_dsa_sparse_attention_det_nope` (SM100 /
    SM103 / SM107); every output repeats bit for bit.

    Documents are packed along the first dimension of ``q`` (``T =
    cu_seqlens_q[-1]``) and of ``kv`` (``S = cu_seqlens_k[-1]``; ``[S, 512]``
    or ``[S, 1, 512]``); query and key lengths may differ per document: the
    query segment is the tail of its key prefix, so query ``local_q`` of
    document ``d`` sits at key position ``(seqlen_k[d] - seqlen_q[d]) +
    local_q``.  ``gather_kv_indices`` ``[T, K_slots]`` int32 hold key positions
    relative to the row's document (``-1`` or ``>= seqlen_k[d]`` invalid; any
    ``K_slots >= 1``) and are offset by ``cu_seqlens_k`` on device.  The route
    follows the table size (``cake_backend.NOPE_CONVERT_SLOT_LIMIT`` slots):
    below the limit, without caller lengths, the forward kernel converts the
    rows itself in its single launch, position-preserving; at or above it, and
    always with a caller ``topk_length``, one launch right before the forward
    kernel compacts each row (valid keys first, in slot order, then ``-1``),
    and the caller's ``topk_length`` is a position cut applied before
    compaction.  See :func:`cake_dsa_sparse_attention_det_nope` for the other
    arguments, the ``1 / 16`` default scale, the FP32 ``dK + dV`` semantics and
    the returns.

    ``cu_seqlens_q`` and ``cu_seqlens_k`` are caller-owned prefix sums: int32
    ``[num_docs + 1]`` tensors of one length on the device of ``q``, starting
    at 0 and non-decreasing, ``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] ==
    S`` and ``seqlen_q <= seqlen_k`` for every document.  Their structure is
    checked on every call; their values only with ``check_cu_seqlens=True``,
    since that check synchronizes with the device.

    Parameters
    ----------
    check_cu_seqlens : bool
        ``True`` checks the values of ``cu_seqlens_q`` / ``cu_seqlens_k``
        against the invariants above at the cost of one blocking device-to-host
        copy per call, before any launch (skipped during CUDA-graph capture).
        ``False`` (default) keeps the structural checks only.
    causal : bool
        With ``True`` a slot that selects a key after the query's own position
        is invalid too; ``False`` (default) takes the index row as is.
    key_passes : Optional[int]
        As in :func:`cake_dsa_sparse_attention_det_nope`: the key-range passes
        of the backward's main stage (``None``: the NoPE key-pass policy, one
        pass); dK + dV is bit for bit that of one pass.
    dkv_direct : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`: accepted for parity,
        validated, without effect.
    check_dst_map : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_det`.
    """
    _refuse_sink(sink)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_det_nope_varlen(
        q,
        kv,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
        check_cu_seqlens=check_cu_seqlens,
    )


# ---------------------------------------------------------------------------
# Explicit (non-autograd) steps: the forward and the backward as two calls.
# ---------------------------------------------------------------------------


@cake_api
def cake_dsa_sparse_attention_det_train_forward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training forward of the rope form without autograd: ``(out, lse,
    o_lo)``.

    The inputs are those of :func:`cake_dsa_sparse_attention_det`. ``out`` is
    BF16 ``[T, 64, 512]`` (zero for a fully masked row), ``lse`` FP32 ``[T,
    64]`` and ``o_lo`` BF16 ``[T, 64, 512]``, the rounding residual of ``out``
    (``out + o_lo`` is the output before its BF16 rounding). Pass the three
    unchanged to :func:`cake_dsa_sparse_attention_det_train_backward`. Every
    output repeats bit for bit and is that of the autograd entry's forward on
    the same inputs. A call without query rows returns empty outputs without
    launching.
    """
    return _backend(backend).forward(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
    )


@cake_api
def cake_dsa_sparse_attention_det_train_backward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
    dkv_fp32: bool = False,
    num_segments: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Training backward of the rope form without autograd: ``(dq_latent,
    dq_rope, dkv_latent, dk_rope)``.

    ``out``, ``o_lo`` and ``lse`` are the outputs of
    :func:`cake_dsa_sparse_attention_det_train_forward` on the same inputs and
    ``topk_length`` / ``softmax_scale``; ``dout`` is the BF16 ``[T, 64, 512]``
    gradient of ``out``. The call leaves ``o_lo`` unchanged. ``dq_latent`` /
    ``dq_rope`` are BF16. ``dkv_latent`` / ``dk_rope`` are BF16 in the layout
    of ``kv_latent`` / ``k_rope``, or FP32 with ``dkv_fp32=True``. With a
    tensor ``dkv_acc`` (FP32 ``[S_dst, >= 576]``) the call adds ``dK + dV``
    into it instead (latent columns ``0:512``, rope ``512:576`` of row
    ``dkv_dst_map[s]``; the caller owns the zeroing) and returns ``None`` for
    both. ``key_passes``, ``check_dst_map`` and ``num_segments`` (the packed
    segment count of the key row, default 1) as in
    :func:`cake_dsa_sparse_attention_det`; ``dkv_direct``: accepted and
    validated, without effect. Every gradient repeats bit for bit and is that
    of the autograd entry's backward on the same inputs, ``key_passes`` and
    ``num_segments``. Nothing raises or warns under
    ``torch.use_deterministic_algorithms(True)``.
    """
    num_segments = _check_num_segments(num_segments)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).backward(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        dkv_fp32=dkv_fp32,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=num_segments,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


@cake_api
def cake_dsa_sparse_attention_det_train_nope_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training forward of the NoPE form without autograd: ``(out, lse,
    o_lo)``.

    The inputs are those of :func:`cake_dsa_sparse_attention_det_nope`, and the
    outputs those of :func:`cake_dsa_sparse_attention_det_train_forward`. Pass
    them unchanged to
    :func:`cake_dsa_sparse_attention_det_train_nope_backward`. The call reads
    the rows as given. Every output repeats bit for bit. It is that of the
    autograd entry's forward on the same inputs when the call passes
    ``topk_length``, the table has fewer than
    ``cake_backend.NOPE_CONVERT_SLOT_LIMIT`` slots, or the rows have no holes
    (at or above that limit, without ``topk_length``, the autograd entry
    compacts the rows first).
    """
    _refuse_sink(sink)
    return _backend(backend).forward_nope(
        q, kv, indices, topk_length=topk_length, softmax_scale=softmax_scale
    )


@cake_api
def cake_dsa_sparse_attention_det_train_nope_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Training backward of the NoPE form without autograd: ``(dq, dkv)``.

    ``out``, ``o_lo`` and ``lse`` are the outputs of
    :func:`cake_dsa_sparse_attention_det_train_nope_forward` on the same
    inputs. ``dq`` is BF16 ``[T, 64, 512]``; ``dkv = dK + dV`` is FP32 ``[S,
    512]``. With a tensor ``dkv_acc`` (FP32 ``[S_dst, >= 512]``) the call adds
    ``dkv`` into columns ``0:512`` of row ``dkv_dst_map[s]`` instead (the
    caller owns the zeroing) and returns ``(dq, None)``. The call leaves
    ``o_lo`` unchanged. ``key_passes`` and ``check_dst_map`` as in
    :func:`cake_dsa_sparse_attention_det_nope`; ``dkv_direct``: accepted and
    validated, without effect. Every gradient repeats bit for bit. It is that
    of the autograd entry's backward on the same inputs and ``key_passes``
    under the conditions of
    :func:`cake_dsa_sparse_attention_det_train_nope_forward`. Nothing raises or
    warns under ``torch.use_deterministic_algorithms(True)``.
    """
    _refuse_sink(sink)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).backward_nope(
        q,
        kv,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


@cake_api
def cake_dsa_sparse_attention_det_train_nope_varlen_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = False,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    backend: str = "cake",
    check_cu_seqlens: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training forward of the packed NoPE form without autograd: ``(out, lse,
    o_lo, indices, topk_length)``.

    The inputs and their checks are those of
    :func:`cake_dsa_sparse_attention_det_nope_varlen`. ``out``, ``lse`` and
    ``o_lo`` as in :func:`cake_dsa_sparse_attention_det_train_forward`;
    ``indices`` (int32 ``[T, K_slots]``, global key rows) and ``topk_length``
    (int32 ``[T]``) are the rows and row lengths the forward read. Pass all
    five unchanged to
    :func:`cake_dsa_sparse_attention_det_train_nope_varlen_backward`. Every
    output repeats bit for bit and is that of the autograd entry's forward on
    the same inputs.
    """
    _refuse_sink(sink)
    return _backend(backend).forward_nope_varlen(
        q,
        kv,
        gather_kv_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        causal=causal,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        check_cu_seqlens=check_cu_seqlens,
    )


@cake_api
def cake_dsa_sparse_attention_det_train_nope_varlen_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    topk_length: torch.Tensor,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    key_passes: Optional[int] = None,
    backend: str = "cake",
    dkv_acc: Optional[torch.Tensor] = None,
    dkv_dst_map: Optional[torch.Tensor] = None,
    dkv_direct: Optional[bool] = None,
    check_dst_map: Optional[bool] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Training backward of the packed NoPE form without autograd: ``(dq,
    dkv)``.

    ``out``, ``o_lo``, ``lse``, ``indices`` and ``topk_length`` are the outputs
    of :func:`cake_dsa_sparse_attention_det_train_nope_varlen_forward`;
    ``cu_seqlens_k`` is the forward's. The outputs and the other arguments as
    in :func:`cake_dsa_sparse_attention_det_train_nope_backward`. Every
    gradient repeats bit for bit and is that of the autograd entry's backward
    on the same inputs and ``key_passes``. Nothing raises or warns under
    ``torch.use_deterministic_algorithms(True)``.
    """
    _refuse_sink(sink)
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).backward_nope(
        q,
        kv,
        indices,
        out,
        o_lo,
        lse,
        dout,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        key_passes=key_passes,
        dkv_acc=dkv_acc,
        dkv_dst_map=dkv_dst_map,
        num_segments=int(cu_seqlens_k.numel()) - 1,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
    )


def __getattr__(name: str) -> Any:
    # The backend's autograd Functions resolve on first access, so that importing
    # this module imports no backend.
    if name in _FROM_BACKEND:
        return getattr(_cake_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FROM_BACKEND))


# Read once at import: a malformed value fails here, naming the variable, not
# on the first call.  Applied to the backend's cache by ``_cake_backend``.
_BINDING_CACHE_CAPACITY = _binding_cache_capacity_from_environment()
