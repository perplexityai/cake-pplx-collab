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

Non-deterministic Cake DSA indexer top-k for training (SM100 / SM103 / SM107).

Every public name carries the mode word ``nondet``; no entry point takes a
``deterministic=`` keyword.  The deterministic
counterpart is ``cake_pplx.cake_dsa_indexer_det_train``: it writes every row in
ascending id order.  This package implements the set-equality semantics: a
row's selected ids are a set, written in no specified slot order, and no output
is promised to repeat bit for bit between calls.

Thin entry points: validation of the backend contract, the host dispatch and
workspace policy, the launch binding, JIT registration and the generated
kernel sources live in the backend package
``cake_pplx._backends.cake_dsa_indexer_nondet_train``, which this
module imports on the first call (never at import time).
"""

import functools
from typing import Optional, Tuple

import torch

from ._cake_pplx_api_dsa_indexer import cake_api


@functools.cache
def _backend():
    from ._backends.cake_dsa_indexer_nondet_train import (  # noqa: PLC0415
        cake_backend,
    )

    return cake_backend


@cake_api
def cake_dsa_indexer_nondet_train_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 2048,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    return_scores: bool = True,
    k_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fused DSA indexer scoring and exact top-k selection with set-equality
    semantics (DeepSeek Sparse Attention indexer, training), SM100 / SM103 /
    SM107 only.

    For query ``t`` and key ``j`` of the same packed segment

    ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])``

    with FP32 products, accumulation and score arithmetic (no BF16 intermediate
    rounding; the reduction scheme is documented in
    ``cake_pplx/_backends/cake_dsa_indexer_nondet_train/README.md``).
    Every row selects the exact global top ``min(top_k, visible)`` keys of its
    visible prefix, ranked by score descending then key id descending
    (``+0.0 == -0.0`` for ranking), and returns them with aligned scores in
    no specified slot order.

    The FP8 form is selected by float8_e4m3fn ``q`` and ``k`` with a per-key
    FP32 ``k_scale``:

    ``s[t, j] = k_scale[j] * sum_h w[t, h] * relu(sum_d q[t, h, d] * k[j, d])``

    where ``w`` carries every other scale (``softmax_scale`` must be ``None``).
    The e4m3 products are exact and accumulate in FP32 on the tensor cores:
    four K = 32 steps on SM100 / SM103 and two K = 64 steps on SM107.  Every
    FP8 program of one architecture selects the same set; selection,
    visibility and outputs follow the BF16 form.

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T, 32, 128]`` indexer queries after RoPE (contiguous);
        float8_e4m3fn for the FP8 form.
    k : torch.Tensor
        BF16 ``[Tkv, 128]`` packed indexer keys after normalization and RoPE.  A
        row-strided view (for example ``packed[:, :128]`` of a ``[Tkv, 704]``
        tensor) is read in place; the row stride must be a multiple of 8
        elements.  The FP8 form: float8_e4m3fn (the same dtype as ``q``), a
        row stride that is a multiple of 16 elements.
    w : torch.Tensor
        FP32 ``[T, 32]`` signed head weights, already scaled by ``32 ** -0.5``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[S + 1]`` independent query and key segment boundaries on the
        device (``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] == Tkv``).  They
        are not read on the host.
    top_k : int
        Output slots per query, ``1 <= top_k <= 4096`` (default 2048).
    softmax_scale : Optional[float]
        Positive scale applied to the per-head logits; defaults to ``128 ** -0.5``.
        The FP8 form takes ``None`` (the scale is folded into ``w``).
    q_causal_offsets : Optional[torch.Tensor]
        int64 ``[S]`` per-segment query offsets on the device.  Key ``j`` is
        visible to the query at segment-local position ``u`` iff ``j < Lk`` and
        ``j < floor((offset + u + 1) / ratio)``; without offsets ``offset = Lk -
        Lq`` for ``ratio == 1`` (queries are the tail of their key prefix) and
        ``0`` for compressed keys.  Negative offsets, empty segments and rows
        without visible keys are defined (the row is all padding).
    ratio : int
        Key compression ratio, ``>= 1`` (normally 1).
    max_seqlen_q : Optional[int]
        Host mirror of the longest query segment; accepted for signature parity
        and not read by any launch decision.
    max_seqlen_k : Optional[int]
        The caller's bound on every key segment (``max(cu_seqlens_k[1:] -
        cu_seqlens_k[:-1])``); the selected sets are identical with and
        without it.  It lets the dispatch tell a packed call of equal key
        segments, whose scan then runs without the longest-first segment
        order; a value above ``Tkv`` or below ``ceil(Tkv / S)`` is rejected.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least :func:`cake_dsa_indexer_nondet_train_topk_workspace_size`
        bytes for the call's geometry (uint8, 8-byte aligned); allocated through
        the caching allocator when omitted.  A caller-owned buffer may be shared
        with other calls and kernels between runs: on the ticket architectures
        the backend then zeroes the buffer's ticket word before every run,
        while its own workspace is zeroed once at prepare.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    return_scores : bool
        ``False``: indices only; ``scores`` must be omitted and ``None`` is
        returned in its place (the scores are still computed internally).
    k_scale : Optional[torch.Tensor]
        FP32 ``[Tkv]`` per-key dequantization scale of the FP8 form
        (contiguous).  BF16 keys take none.

    Returns
    -------
    indices : torch.Tensor
        int32 ``[T, top_k]`` segment-local key ids, unique within the row;
        the ``min(top_k, visible)`` selected ids and ``-1`` in the other
        slots, in no specified slot order.
    scores : Optional[torch.Tensor]
        ``None`` with ``return_scores=False``; else FP32 ``[T, top_k]`` scores
        aligned with ``indices``; the slots of ``-1`` ids hold ``-inf``.

    Every row holds the exact selection above in its ``min(top_k, visible)``
    valid (non ``-1``) slots; the slot order within a row and the positions of
    the ``-1`` / ``-inf`` padding slots may vary between calls, and no output
    is promised to repeat bit for bit.  The internal query / key partition and
    the program the host dispatches for the geometry never change the selected
    ids.  Under ``torch.use_deterministic_algorithms(True)`` a call with
    queries and keys raises ``RuntimeError`` before it allocates or launches
    anything (with ``warn_only=True`` it warns and runs).  The call never
    materializes a ``[T, Tkv]`` matrix, and no call synchronizes with the
    host, so calls can be captured into a CUDA graph.  Finite inputs are the
    normal domain: with NaN / inf / overflowing inputs the operator returns
    with the output structure intact and the kernels never hang.
    """
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "cake_dsa_indexer_nondet_train_topk needs cu_seqlens_q and cu_seqlens_k"
        )
    if not return_scores and scores is not None:
        raise ValueError("scores given with return_scores=False")
    indices, cake_scores = _backend().cake_dsa_indexer_nondet_train_topk(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
        k_scale=k_scale,
    )
    return indices, cake_scores if return_scores else None


@cake_api
def cake_dsa_indexer_nondet_train_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = 2048,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    r"""Explicit workspace bound of :func:`cake_dsa_indexer_nondet_train_topk` in bytes for one call geometry.

    The host dispatch picks the scan program (unit geometry, key-range split,
    CTA pair) from ``T = num_queries``, ``Tkv = num_keys``, ``S =
    num_segments``, ``ratio`` and ``top_k`` together with the SM count of
    ``device``, and the bound follows that choice: ``grid x queries_per_unit
    x candidate_capacity(top_k) x 8`` bytes of persistent candidate buffers
    (for example about 74 MiB at ``top_k = 2048`` with four queries per unit
    on a 148-SM sm_100a device at its 16384-slot capacity floor, 148 MiB with
    eight) plus ``T x n_split x top_k x 8``
    bytes of staging when the dispatch splits the key ranges -- the bound of
    ``cake_pplx.cake_dsa_indexer_det_train`` for the same call.  Size a
    reused buffer for the largest bound over the geometries it serves.
    ``dtype`` is the q / k dtype: ``torch.float8_e4m3fn`` bounds the FP8
    form, whose programs differ.  Raises ``NotImplementedError`` while no
    generated program of the form is registered for the device's
    architecture.  The bound never consults
    ``torch.use_deterministic_algorithms``.

    Parameters
    ----------
    num_queries : int
        ``T``, the number of packed queries of the call.
    num_keys : int
        ``Tkv``, the number of packed keys of the call.
    num_segments : int
        ``S``, the number of segments (``len(cu_seqlens_q) - 1``).
    top_k : int
        Output slots per query, as passed to :func:`cake_dsa_indexer_nondet_train_topk`
        (default 2048).
    ratio : int
        Key compression ratio, as passed to :func:`cake_dsa_indexer_nondet_train_topk`
        (default 1).
    device : Optional[torch.device]
        The device whose SM count and architecture select the scan program;
        the current CUDA device when omitted.
    dtype : torch.dtype
        The q / k dtype: ``torch.bfloat16`` (default) or ``torch.float8_e4m3fn``.

    Returns
    -------
    int
        The workspace bound in bytes for this geometry.
    """
    return _backend().cake_dsa_indexer_nondet_train_topk_workspace_size(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
        dtype=dtype,
    )


@cake_api
def cake_dsa_indexer_nondet_train_topk_with_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 512,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    candidate_blocks: Optional[torch.Tensor] = None,
    return_scores: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    r""":func:`cake_dsa_indexer_nondet_train_topk` that also returns each
    query's candidate blocks, computed from the same scores as ``indices``;
    SM100 / SM103 only (BF16 ``q`` / ``k``, ``ratio == 1``).

    Block ``b`` holds the segment-local keys ``8 b .. 8 b + 7``.  For a query
    with ``n`` visible keys (the family's visibility rule) let ``nb = ceil(n /
    8)`` and ``S_b`` the largest score of the visible keys of block ``b``
    (future keys of the newest block do not count).  With ``nb <= 2048`` the
    candidates are blocks ``0 .. nb - 1``; otherwise block ``nb - 1`` (the
    newest, always kept) and the 2047 blocks of ``0 .. nb - 2`` with the
    largest ``S_b``, ties broken by the larger block id (``+0.0 == -0.0``),
    the tie rule of ``indices``.  With ``top_k <= 2047`` every id of
    ``indices`` lies in a block of its row.

    Parameters
    ----------
    q, k, w, cu_seqlens_q, cu_seqlens_k, softmax_scale, q_causal_offsets, max_seqlen_q
        As :func:`cake_dsa_indexer_nondet_train_topk` (BF16 ``q`` and ``k``).
    top_k : int
        Output slots per query, ``1 <= top_k <= 2047`` (default 512).
    ratio : int
        Must be 1 (blocks of 8 keys).
    max_seqlen_k : Optional[int]
        As :func:`cake_dsa_indexer_nondet_train_topk`; here it also sizes the
        workspace's block entries, so it must bound every key segment of the
        call (a row with more blocks than the workspace holds stops the kernel
        with a device-side trap).  ``Tkv`` when omitted.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least
        :func:`cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size`
        bytes for the call's geometry and ``max_seqlen_k`` (uint8, 8-byte
        aligned); allocated through the caching allocator when omitted.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    candidate_blocks : Optional[torch.Tensor]
        Caller-owned ``[T, 2048]`` int32 contiguous output on the device of
        ``q``; allocated when omitted.
    return_scores : bool
        ``False``: ``scores`` must be omitted and ``None`` is returned in its
        place (the scores are still computed internally).

    Returns
    -------
    indices, scores : torch.Tensor, Optional[torch.Tensor]
        As :func:`cake_dsa_indexer_nondet_train_topk`.
    candidate_blocks : torch.Tensor
        int32 ``[T, 2048]`` segment-local block ids: the selected blocks
        and ``-1`` in the other slots, in no specified slot order (all
        ``-1`` when ``n == 0``).
    Every row holds the exact sets above; the slot order within a row and
    the positions of the ``-1`` slots may vary between calls, and no
    output is promised to repeat bit for bit.  Under
    ``torch.use_deterministic_algorithms(True)`` a call with queries and
    keys raises ``RuntimeError`` before it allocates or launches anything
    (with ``warn_only=True`` it warns and runs).  The call never materializes a ``[T, Tkv]`` matrix and never
    synchronizes with the host.  Raises ``NotImplementedError`` on an
    architecture without a registered candidate-block program (SM107).
    """
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "cake_dsa_indexer_nondet_train_topk_with_candidates needs cu_seqlens_q and cu_seqlens_k"
        )
    if not return_scores and scores is not None:
        raise ValueError("scores given with return_scores=False")
    indices, cake_scores, blocks = (
        _backend().cake_dsa_indexer_nondet_train_topk_with_candidates(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            top_k=top_k,
            softmax_scale=softmax_scale,
            q_causal_offsets=q_causal_offsets,
            ratio=ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            workspace_buffer=workspace_buffer,
            indices=indices,
            scores=scores,
            candidate_blocks=candidate_blocks,
        )
    )
    return indices, cake_scores if return_scores else None, blocks


@cake_api
def cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = 512,
    ratio: int = 1,
    max_seqlen_k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> int:
    r"""Explicit workspace bound of
    :func:`cake_dsa_indexer_nondet_train_topk_with_candidates` in bytes for one
    call geometry.

    The scan's candidate buffers and work-unit tickets (the bound of the
    dispatched candidate-block program, which runs the key ranges unsplit) plus
    ``grid x queries_per_unit x blk_cap x 8`` bytes of block entries, where
    ``blk_cap = ceil(W / 8) - 1`` when ``ceil(W / 8) > 2048`` and 0 otherwise,
    ``W`` being ``max_seqlen_k`` when given, else ``num_keys``.  The ``[T,
    2048]`` int32 ``candidate_blocks`` output is not part of the bound.  Pass
    the call's ``max_seqlen_k``.  Raises ``NotImplementedError`` on an
    architecture without a registered candidate-block program.

    Parameters
    ----------
    num_queries, num_keys, num_segments : int
        ``T``, ``Tkv`` and ``S`` of the call.
    top_k : int
        Output slots per query, ``1 <= top_k <= 2047`` (default 512).
    ratio : int
        Must be 1.
    max_seqlen_k : Optional[int]
        The bound on every key segment that the call will pass.
    device : Optional[torch.device]
        The device whose SM count and architecture select the program; the
        current CUDA device when omitted.

    Returns
    -------
    int
        The workspace bound in bytes for this geometry.
    """
    return _backend().cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        max_seqlen_k=max_seqlen_k,
        device=device,
    )


@cake_api
def cake_dsa_indexer_nondet_train_topk_in_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    candidate_blocks: torch.Tensor,
    *,
    top_k: int = 512,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    return_scores: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r""":func:`cake_dsa_indexer_nondet_train_topk` restricted to given candidate
    blocks; SM100 / SM103 only (BF16 ``q`` / ``k``, ``ratio == 1``).

    Block ``b`` holds the segment-local keys ``8 b .. 8 b + 7``.  For query
    row ``t`` with ``n`` visible keys (the family's visibility rule) the pool
    ``P_t`` is the keys ``j < n`` whose block ``j // 8`` is named in
    ``candidate_blocks[t]``; the row selects the top ``min(top_k, |P_t|)``
    keys of ``P_t`` by score descending, then key id descending (``+0.0 ==
    -0.0``), with the family's scores.  The work of a row is bounded by its
    candidate keys (at most ``8 W``), not by its visible prefix: no key
    outside ``P_t`` is scored.

    Parameters
    ----------
    q, k, w, cu_seqlens_q, cu_seqlens_k, softmax_scale, q_causal_offsets, max_seqlen_q, max_seqlen_k
        As :func:`cake_dsa_indexer_nondet_train_topk` (BF16 ``q`` and ``k``;
        ``k`` as contiguous ``[Tkv, 128]`` rows).
    candidate_blocks : torch.Tensor
        int32 ``[T, W]`` contiguous segment-local block ids on the device of
        ``q``, ``W <= 2048`` (the ``candidate_blocks`` output of
        :func:`cake_dsa_indexer_nondet_train_topk_with_candidates` fits as is).
        Any order; ``-1`` and ids at or past the row's visible blocks name no
        key.  The non-negative ids of a row must be distinct.  Never read on
        the host.
    top_k : int
        Output slots per query, ``1 <= top_k <= 2048`` (default 512).
    ratio : int
        Must be 1 (blocks of 8 keys).
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least
        :func:`cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size`
        bytes (uint8, 8-byte aligned); allocated through the caching allocator
        when omitted.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    return_scores : bool
        ``False``: ``scores`` must be omitted and ``None`` is returned in its
        place (the scores are still computed internally).

    Returns
    -------
    indices, scores : torch.Tensor, Optional[torch.Tensor]
        As :func:`cake_dsa_indexer_nondet_train_topk`: the row's set in no
        specified slot order, ``-1`` / ``-inf`` padding anywhere.  Under
        ``torch.use_deterministic_algorithms(True)`` a call with queries, keys
        and candidate slots raises PyTorch's ``RuntimeError``
        (``warn_only=True``: a ``UserWarning``).
    The call never materializes a ``[T, Tkv]`` matrix and never synchronizes
    with the host.  Raises ``NotImplementedError`` on an architecture without
    a registered candidate-restricted program (SM107).
    """
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "cake_dsa_indexer_nondet_train_topk_in_candidates needs cu_seqlens_q and cu_seqlens_k"
        )
    if not return_scores and scores is not None:
        raise ValueError("scores given with return_scores=False")
    indices, cake_scores = _backend().cake_dsa_indexer_nondet_train_topk_in_candidates(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        candidate_blocks,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
    )
    return indices, cake_scores if return_scores else None


@cake_api
def cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = 512,
    ratio: int = 1,
    device: Optional[torch.device] = None,
) -> int:
    r"""Explicit workspace bound of
    :func:`cake_dsa_indexer_nondet_train_topk_in_candidates` in bytes.

    ``grid x 4 x cand_cap x 8`` bytes of candidate buffers (``cand_cap =
    max(4096, 4 top_k)``), ``grid x 64 KiB`` of block lists and a 16-byte work
    ticket, ``grid`` being the SM count of ``device``.  It depends on the
    device and ``top_k`` only, not on ``T``, ``Tkv``, the segments or the
    candidate content.  Raises ``NotImplementedError`` on an architecture
    without a registered candidate-restricted program.

    Parameters
    ----------
    num_queries, num_keys, num_segments : int
        ``T``, ``Tkv`` and ``S`` of the call.
    top_k : int
        Output slots per query, ``1 <= top_k <= 2048`` (default 512).
    ratio : int
        Must be 1.
    device : Optional[torch.device]
        The device whose SM count and architecture select the program; the
        current CUDA device when omitted.

    Returns
    -------
    int
        The workspace bound in bytes.
    """
    return _backend().cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
    )


__all__ = [
    "cake_dsa_indexer_nondet_train_topk",
    "cake_dsa_indexer_nondet_train_topk_in_candidates",
    "cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size",
    "cake_dsa_indexer_nondet_train_topk_with_candidates",
    "cake_dsa_indexer_nondet_train_topk_with_candidates_workspace_size",
    "cake_dsa_indexer_nondet_train_topk_workspace_size",
]
