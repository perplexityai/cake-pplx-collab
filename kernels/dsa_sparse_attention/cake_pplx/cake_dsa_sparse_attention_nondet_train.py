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

Cake DeepSeek Sparse Attention (top-k sparse MLA, absorbed queries) training
forward and backward on SM100 / SM103 / SM107, the non-deterministic package:
the stable entry points over the backend package
``cake_pplx._backends.cake_dsa_sparse_attention_nondet_train``.

Query heads: the rope entries take ``H = q_latent.shape[1]`` heads, ``H`` in
64, 96 and 128 (SM100 / SM103 / SM107); every head tensor below written
``[T, 64, ...]`` is ``[T, H, ...]`` for them.  The kernels compute 64 heads of
a token per work item: 96 and 128 heads run, where the program ships them for
the device (the forward builds on SM100 / SM103, the backward builds on all
three), the head-window builds -- one launch per stage over both 64-head
windows ``[0, 64)`` and ``[64, 128)`` of every token, the heads past ``H``
padding -- and otherwise the two-window bridge: the 64-head kernels on
``[H - 64, H)`` and then ``[0, 64)``, with the heads ``[H - 64, 64)`` both
windows cover taken from the second (see the backend README).  Their forward
runs, where the program ships it (SM100 / SM103 / SM107), the single-pass
forward instead: every head of a token in one 2-CTA cluster.  The NoPE entries
take 64 heads.

Numerical mode: the module name chooses the package, and no entry point takes a
determinism keyword; ``with_o_lo`` chooses whether the forward writes ``o_lo``
and how the backward forms ``delta`` (see below).  The backward adds the dK / dV contributions of every query row
into FP32 accumulators with unordered atomics (``dkv``: the ``kv_latent`` /
``k_rope`` or ``kv`` gradients, or the rows added into ``dkv_acc``), and the
package does not promise that any output repeats bit for bit.  Under
``torch.use_deterministic_algorithms(True)`` the forward runs and the backward
raises PyTorch's ``RuntimeError`` before its first backward kernel launch; with
``warn_only=True`` it warns
(``UserWarning``) and runs this package's backward.  Deterministic
preparation may run before the alert: a copy of a ``dout`` the TMA descriptor
cannot address; the plan of a new binding, which can JIT-load the backward
stages and, without a caller ``topk_length``, allocates and fills the
binding's row-length vector; the validation of ``dkv_acc`` /
``dkv_dst_map``.  It never falls back to another package.  The
deterministic package ``cake_pplx.cake_dsa_sparse_attention_det_train`` has
the same entry points (``det`` in place of ``nondet``); its outputs repeat bit
for bit run to run; the det entries take no ``with_o_lo``.

Entry points (each an autograd call: forward now, backward through autograd)::

    cake_dsa_sparse_attention_nondet               flat, rope + latent scores
    cake_dsa_sparse_attention_nondet_varlen        packed documents, rope + latent scores
    cake_dsa_sparse_attention_nondet_nope          flat, latent scores only (NoPE)
    cake_dsa_sparse_attention_nondet_nope_varlen   packed documents, NoPE

Explicit steps (no autograd; the forward returns what the backward reads), for
the flat rope form, the flat NoPE form and the packed NoPE form::

    cake_dsa_sparse_attention_nondet_train_forward
    cake_dsa_sparse_attention_nondet_train_backward
    cake_dsa_sparse_attention_nondet_train_nope_forward
    cake_dsa_sparse_attention_nondet_train_nope_backward
    cake_dsa_sparse_attention_nondet_train_nope_varlen_forward
    cake_dsa_sparse_attention_nondet_train_nope_varlen_backward

The packed forward also returns the global rows and row lengths that its
backward reads.

The flat entries and the step backwards of both forms take ``num_segments``,
the packed segment count of the key row (default 1, a one-document row).  A
caller that passes the global rows of a packed batch passes its count
(``len(cu_seqlens_k) - 1``): the backward then plans as the packed entry does
for the same rows (the rope form's key-range passes; on SM107 the ``dkv_direct``
rule of both forms).  The count selects the plan only.

The NoPE entries, flat and packed, take an optional trainable per-head sink,
``sink`` FP32 ``[64]`` (SM100 / SM103 / SM107): it adds ``exp(sink[h])`` to
every row's softmax denominator, ``lse`` includes it, and its gradient
``dsink`` flows back through autograd (the step backwards return it).

Every entry takes ``with_o_lo`` (``bool``, keyword-only, default ``True``).
With ``True`` the forward also writes ``o_lo``, the BF16 rounding residual of
``out``, and the backward forms ``delta = rowsum(dout * (out + o_lo))`` in
FP32, which keeps the gradients accurate on peaked rows (rows whose softmax
puts almost all weight on one key).  With ``False`` the forward neither
computes nor stores ``o_lo`` (``H`` KiB less written and saved per query
token, ``H`` the call's query head count: 64 on the NoPE entries; above 64
heads on the two-window bridge also no ``fwd_o_lo`` window scratch, 64 KiB per
token of workspace, and no ``o_lo`` window copies) and the
backward forms ``delta = rowsum(dout * out)`` from the BF16 ``out``,
the form cuDNN's DSA backward uses; on peaked rows the gradient error grows to
that class.  ``out`` and ``lse`` have the same accuracy in both modes.  The
step forwards then return ``None`` in the ``o_lo`` slot, and a step backward
takes ``o_lo=None`` with the same ``with_o_lo=False``: ``o_lo`` is ``None``
exactly when ``with_o_lo`` is ``False`` (a default call without query rows
also accepts ``None``), and a mismatch raises ``ValueError`` before anything
is planned or launched.

A call without query rows (``T = 0``) launches nothing: the forwards return
empty ``out`` / ``lse`` / ``o_lo`` (``None`` for ``o_lo`` with
``with_o_lo=False``); the NoPE step backwards return an empty
``dq``, a zero FP32 ``dkv`` (``None`` with a tensor ``dkv_acc``) and, with a
sink, a zero ``dsink``; the rope step backward returns empty ``dq_latent`` /
``dq_rope`` and zero ``dkv_latent`` / ``dk_rope``, FP32 with ``dkv_fp32=True``
and BF16 otherwise (``None`` with a tensor ``dkv_acc``).

Their autograd Functions, ``CakeDSASparseAttentionNondetFunction`` (both rope
entries), ``CakeDSASparseAttentionNondetNopeFunction`` and
``CakeDSASparseAttentionNondetNopeVarlenFunction``, resolve from the backend
package on first access.

Usage::

    from cake_pplx.cake_dsa_sparse_attention_nondet_train import (
        cake_dsa_sparse_attention_nondet,
    )

    out = cake_dsa_sparse_attention_nondet(q_latent, q_rope, kv_latent, k_rope, indices)
    out.float().square().sum().backward()  # dK / dV added with FP32 atomics
"""

import functools
import os
from typing import Any, Callable, Optional, Union

import torch

from ._cake_pplx_api_dsa_sparse_attention import cake_api

# Stable entry points (``@cake_api``) over the backend package; validation
# of the backend contract, the workspace, launch binding, autograd wrappers and
# JIT registration live in
# cake_pplx._backends.cake_dsa_sparse_attention_nondet_train, which this
# core module imports on first use only: importing this module loads nothing
# from the backend package.  The backend takes every switch as a keyword argument
# and reads no environment variable; the variables below are this module's A/B
# knobs, resolved here and passed down.

__all__ = [
    "BINDING_CACHE_CAPACITY_ENV",
    "CHECK_DST_MAP_ENV",
    "DKV_DIRECT_ENV",
    "CakeDSASparseAttentionNondetFunction",  # noqa: F822 (resolved by __getattr__)
    "CakeDSASparseAttentionNondetNopeFunction",  # noqa: F822
    "CakeDSASparseAttentionNondetNopeVarlenFunction",  # noqa: F822
    "cake_dsa_sparse_attention_nondet",
    "cake_dsa_sparse_attention_nondet_nope",
    "cake_dsa_sparse_attention_nondet_nope_varlen",
    "cake_dsa_sparse_attention_nondet_varlen",
    "cake_dsa_sparse_attention_nondet_train_backward",
    "cake_dsa_sparse_attention_nondet_train_forward",
    "cake_dsa_sparse_attention_nondet_train_nope_backward",
    "cake_dsa_sparse_attention_nondet_train_nope_forward",
    "cake_dsa_sparse_attention_nondet_train_nope_varlen_backward",
    "cake_dsa_sparse_attention_nondet_train_nope_varlen_forward",
]

# The backend's autograd Functions, re-exported on first access (module ``__getattr__``).
_FROM_BACKEND = (
    "CakeDSASparseAttentionNondetFunction",
    "CakeDSASparseAttentionNondetNopeFunction",
    "CakeDSASparseAttentionNondetNopeVarlenFunction",
)

# ``auto`` (default): the registered record's size rule of the device's
# architecture decides whether a backward with ``dkv_acc`` accumulates directly
# from the main stage; ``1`` forces the direct path, ``0`` the
# accumulate-and-cast path.
DKV_DIRECT_ENV = "CAKE_PPLX_DSA_DKV_DIRECT"
# Unset, empty or ``0`` (default): the values of ``dkv_dst_map`` are not
# checked; anything else validates them on every backward (one device sync).
CHECK_DST_MAP_ENV = "CAKE_PPLX_DSA_CHECK_DST_MAP"
# A positive integer: the capacity of the backend's binding cache (64 by
# default).  Read when this module is imported, applied when it first reaches
# the backend.
BINDING_CACHE_CAPACITY_ENV = "CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY"


def _dkv_direct_from_environment() -> Optional[bool]:
    """``CAKE_PPLX_DSA_DKV_DIRECT`` as the backend's ``dkv_direct`` keyword:
    ``auto`` (default) is ``None`` (the record's size rule), ``1`` is ``True``
    (direct whenever the program registers the natural stages), ``0`` is
    ``False`` (never).  Any other value raises ``ValueError``."""
    mode = os.environ.get(DKV_DIRECT_ENV, "auto").strip() or "auto"
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"{DKV_DIRECT_ENV} must be auto, 0 or 1, got {mode!r}")
    return None if mode == "auto" else mode == "1"


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
    from ._backends.cake_dsa_sparse_attention_nondet_train import cake_backend

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


def _check_with_o_lo(with_o_lo: Any) -> bool:
    """``with_o_lo`` as the backend takes it: a Python ``bool``.  ``0``, ``1``, ``None``, a string, a tensor or any
    other value raises ``ValueError`` before the backend is reached: the flag chooses the accuracy class of the
    gradients and is not read by truthiness."""
    if not isinstance(with_o_lo, bool):
        shown = (
            repr(with_o_lo)
            if with_o_lo is None or isinstance(with_o_lo, (int, float, str))
            else f"a {type(with_o_lo).__name__}"
        )
        raise ValueError(f"with_o_lo must be a Python bool, got {shown}")
    return with_o_lo


@cake_api
def cake_dsa_sparse_attention_nondet(
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
    with_o_lo: bool = True,
):
    r"""Differentiable top-k sparse MLA attention with absorbed queries (DeepSeek
    Sparse Attention) for training, ``H`` query heads (64, 96 or 128), SM100 /
    SM103 / SM107 only.

    ``O = softmax(softmax_scale * (q_rope . k_rope^T + q_latent . kv_latent^T)) . kv_latent``
    over the keys a row's ``indices`` select (K = V = the latent).

    Strided inputs are consumed in place (no host copy): the kernels read the
    query operands through TMA descriptors encoded from the tensor's own head
    and token strides and gather the key operands by row stride (above 64
    heads the backward copies a ``dout`` whose token stride is not ``H`` head
    strides, such as a head-major view, once on the device).  Admitted
    layouts: unit innermost stride; head / token / key-row strides that are
    positive multiples of 8 elements (16 bytes, the TMA descriptor stride
    granule); head stride and key row stride at least the slice width; a
    16-byte-aligned base address.  For a strided view, only descriptor fields
    and pointer offsets differ from a contiguous tensor.

    Parameters
    ----------
    q_latent : torch.Tensor
        BF16 ``[T, H, 512]`` absorbed query latents, ``H`` in 64, 96 or 128
        (the call's head count); contiguous or a view of a packed ``[T, H,
        576]`` query (head stride 576).  Another ``H`` raises ``ValueError``.
    q_rope : torch.Tensor
        BF16 ``[T, H, 64]`` query rope part; contiguous, a view of the packed
        ``[T, H, 576]`` query, or the ``192:256`` channel slice of the
        pre-absorption ``[T, H, 256]`` query (head stride 256, token stride
        ``H * 256``, storage offset 192 elements).
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
        Also return the natural-log logsumexp ``[T, H]`` FP32 of the scaled
        scores over the valid keys (``-inf`` for fully masked rows).
    key_passes : Optional[int]
        Number of key-range passes of the backward's main stage.  ``None``
        (default) applies the per-architecture policy of the registry record
        (the backend README's "Key-range passes for the DRAM regime"): on a
        one-segment row (``num_segments=1``) the key range is split into
        ``ceil(S * 2304 B / 100 MiB)`` passes, so that each pass's FP32 dK/dV
        accumulator slice stays L2-resident, when that exceeds one and the row
        is one token chunk (``T <= 4224`` at top-k 2048) or, on ``sm_103a`` /
        ``sm_107a``, a tail row (``2 T <= S``, at least 131,072 keys; at most 3
        passes on ``sm_103a``); ``sm_107a`` runs a fixed 2 passes with a
        131,072-key floor; a count above 4 runs one pass; otherwise a single
        pass runs.  Above 64 heads the head-window builds take no tail-row
        passes on ``sm_103a`` (one pass there; one-chunk rows as above).  A
        row of more than one segment (``num_segments > 1``)
        runs one pass unless ``key_passes`` overrides.  ``key_passes=1``
        forces the single pass; a larger value forces that many passes
        (``<= S``).  The passes carry dQ through
        an FP32 partial in the workspace (``T * 147,456`` B plus ``T * (4 *
        topk + 4)`` B, ~608 MiB at ``T = 4096``, top-k 2048); dQ is still
        written once per row at 64 heads and with the head-window builds
        (on the two-window bridge the overlap heads ``[H - 64, 64)`` are
        written by the first window, zeros, and then by the second) and the
        dK/dV accumulation is unchanged.
    backend : str
        Only ``"cake"``.
    dkv_acc : Optional[torch.Tensor | Callable]
        Caller-owned FP32 ``[S_dst, >= 576]`` buffer (row stride a multiple of
        4 elements, 16-byte aligned) into which the backward accumulates the
        key gradients in place (``+=``; the caller zeroes it): latent
        dimensions in columns ``0:512``, rope dimensions in ``512:576``.  When
        given, the ``kv_latent`` / ``k_rope`` gradients of the autograd call
        are ``None`` and no BF16 cast runs.  Instead of the tensor, a
        zero-argument callable that returns it -- or an ``(accumulator,
        dkv_dst_map)`` pair -- may be given; the backward calls it, so the
        accumulator need not exist at forward time (under activation
        checkpointing a tensor passed at forward time is alive for every
        layer at once).  The autograd entries save every tensor they produce
        for the backward through ``ctx.save_for_backward`` and retain nothing
        else of a checkpointed forward; a tensor ``dkv_acc`` stays caller-owned.
    dkv_dst_map : Optional[torch.Tensor]
        int32 ``[S]`` destination row of ``dkv_acc`` for every key row
        (default identity).  Repeated destinations are allowed (the rows of a
        context-parallel window that map onto one parameter row) and are
        combined with FP32 atomics.  Every value must lie in ``[0, S_dst)``;
        the kernel does not range-check the map (see ``check_dst_map``).
    dkv_direct : Optional[bool]
        How the backward accumulates dK / dV into ``dkv_acc`` when the
        registered program has the natural-layout main stages.  ``None``
        (default) reads ``CAKE_PPLX_DSA_DKV_DIRECT``: ``auto`` (its
        default) applies the record's size rule of the device's architecture
        -- the main stage adds into the caller's packed rows directly (no
        workspace accumulators, zero fill or cast launch) when the row has at
        least ``dkv_direct.min_keys_per_query`` (4) keys per query token and,
        on ``sm_100a`` / ``sm_103a``, whatever its work; on ``sm_107a``, whose
        natural-layout drain the rule models as costing extra in proportion to
        the main stage's work, only while ``T * K_eff < 112 * S + 2**20`` with ``K_eff =
        min(K_slots, ceil(S / num_segments))`` --, ``1``
        forces the direct path whenever the stages are registered, ``0`` always
        takes the accumulate-and-cast path; any other value raises
        ``ValueError``.  ``True`` / ``False`` select the path explicitly and
        ignore the variable.  Both paths add the same FP32 contributions into
        the same rows (another summation order); without ``dkv_acc`` or the
        natural stages the setting has no effect.
    check_dst_map : Optional[bool]
        Validate the values of ``dkv_dst_map`` (every destination row in
        ``[0, S_dst)``) on every backward, at the cost of one device
        synchronization, raising ``ValueError`` on a violation.  ``None``
        (default) reads ``CAKE_PPLX_DSA_CHECK_DST_MAP``: unset, empty or
        ``0`` (its default) means no check, anything else enables it.  ``True``
        / ``False`` decide explicitly and ignore the variable.  Leave it off in
        production.
    num_segments : int
        The packed segment count of the key row: ``len(cu_seqlens_k) - 1``
        when ``indices`` hold the global rows of a packed batch whose tokens
        select keys of their own segment only (a Python ``int``, read on the
        host: no device synchronization).  ``1`` (default) is a one-document
        row and keeps the key-range-pass policy above.  A count above 1 plans
        the backward as :func:`cake_dsa_sparse_attention_nondet_varlen` plans
        it for the same rows: one pass unless ``key_passes`` overrides, since
        whole-row key ranges would leave most passes empty for every token.
        On ``sm_107a`` the ``dkv_direct`` rule above computes its ``K_eff =
        min(K_slots, ceil(S / num_segments))`` from this count at every value,
        1 included (one segment gives ``K_eff = min(K_slots, S)``).  The count
        selects the plan only; every value gives a valid result.  ``bool``, a
        tensor or a value below 1 raises ``ValueError``.
    with_o_lo : bool
        With ``True`` (default) the forward also writes ``o_lo``, the BF16
        rounding residual of ``out``, and the backward forms ``delta =
        rowsum(dout * (out + o_lo))`` in FP32.  This keeps the gradients
        accurate on peaked rows (rows whose softmax puts almost all weight on
        one key).  With ``False`` the forward neither computes nor stores
        ``o_lo`` (``H`` KiB less written and saved per query token; above 64
        heads on the two-window bridge also no ``fwd_o_lo`` window scratch,
        64 KiB per token of workspace, and no ``o_lo`` window copies), and the
        backward forms
        ``delta = rowsum(dout * out)`` from the BF16 ``out``, the form
        cuDNN's DSA backward uses; on peaked rows the gradient error
        grows to that class.  ``out`` and ``lse`` have the same accuracy in
        both modes.  Any value other than a ``bool`` raises ``ValueError``.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, H, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.

    Gradients flow to ``q_latent``, ``q_rope``, ``kv_latent`` and ``k_rope``
    (``dq`` stored, not accumulated: written once per row at 64 heads and with
    the head-window builds, and on the two-window bridge the overlap heads
    ``[H - 64, 64)`` written by the first window, zeros, and then by the
    second; ``dkv`` accumulated in FP32 with atomics,
    then cast to BF16, or accumulated into ``dkv_acc``).  BF16 operands into the tensor cores, FP32
    accumulation; the backward recomputes the scores from the BF16 inputs and
    forms ``delta`` in FP32 from the BF16 output and its saved BF16 residual
    ``o_lo`` (with ``with_o_lo=False``, from the BF16 output alone).  The kernels
    apply no positional mask: the index rows define the key set.

    Under ``torch.use_deterministic_algorithms(True)`` the forward runs and the
    backward raises PyTorch's ``RuntimeError`` before its first backward
    kernel launch (its dK / dV atomics are not run-to-run reproducible);
    deterministic preparation may run first (see the module docstring).  With
    ``warn_only=True`` it issues a ``UserWarning`` and runs.  No deterministic
    package is chosen instead.

    Host side: the inputs are validated and bound on the first call for an
    input binding (``data_ptr``, shape, stride and dtype of every input plus
    the scale); later calls with the same binding launch from the remembered
    argument plans with freshly allocated outputs
    (``cake_pplx._backends.cake_dsa_sparse_attention_nondet_train.cake_backend.BINDING_CACHE``, 64
    bindings by default, least recently used first out;
    ``CAKE_PPLX_DSA_TRAIN_BINDING_CACHE_CAPACITY`` -- a positive integer,
    read when this module is imported -- sets the capacity through
    ``BINDING_CACHE.configure(capacity=...)``).
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    num_segments = _check_num_segments(num_segments)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_nondet(
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
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_varlen(
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
    with_o_lo: bool = True,
):
    r"""Packed multi-document form of :func:`cake_dsa_sparse_attention_nondet` (SM100 /
    SM103 / SM107).

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
    :func:`cake_dsa_sparse_attention_nondet` for the other arguments and the
    returns; the strided query slices and packed key rows described there are
    accepted here too.

    Under ``torch.use_deterministic_algorithms(True)`` the forward runs and the
    backward raises PyTorch's ``RuntimeError`` before its first backward
    kernel launch (its dK / dV atomics are not run-to-run reproducible);
    deterministic preparation may run first (see the module docstring).  With
    ``warn_only=True`` it issues a ``UserWarning`` and runs.  No deterministic
    package is chosen instead.

    Parameters
    ----------
    causal : bool
        With ``False`` (default) the index row is taken as is -- only ``-1`` /
        out-of-range slots are dropped -- exactly as in the first release.
        With ``True`` a slot that selects a key after the query's own position
        (``idx > (seqlen_k[d] - seqlen_q[d]) + local_q``) is invalid too, so a
        top-k selector need not enforce causality itself (the rule GLM-style
        trainers with query segments that are tails of their key prefix rely
        on).
    dkv_direct : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_nondet`: ``None`` (default) reads
        ``CAKE_PPLX_DSA_DKV_DIRECT`` (``auto`` = the record's size rule,
        ``1`` / ``0`` = force / disable the direct accumulation into
        ``dkv_acc``); ``True`` / ``False`` decide explicitly.
    check_dst_map : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_nondet`: ``None`` (default) reads
        ``CAKE_PPLX_DSA_CHECK_DST_MAP`` (unset, empty or ``0`` = no
        check); ``True`` validates the values of ``dkv_dst_map`` on every
        backward, ``False`` never does.
    with_o_lo : bool
        As in :func:`cake_dsa_sparse_attention_nondet`: ``True`` (default)
        writes ``o_lo`` and forms ``delta`` from ``out + o_lo``; ``False``
        writes no ``o_lo`` and forms ``delta`` from the BF16 ``out`` alone, on
        both index routes.  Any value other than a ``bool`` raises
        ``ValueError``.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_nondet_varlen(
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
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_nope(
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
    num_segments: int = 1,
    with_o_lo: bool = True,
):
    r"""Differentiable top-k sparse MLA attention without a rope part (NoPE:
    the whole score is the 512-dimensional latent product) for training, 64
    query heads, SM100 / SM103 / SM107 only.

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
        ``K_slots >= 1`` (the slot count is a per-row capacity, not a
        multiple of anything); ``-1`` or a value ``>= S`` marks an invalid
        slot, anywhere in the row.  A row whose every slot is invalid gives
        ``out = 0``, ``lse = -inf`` and zero gradients.
    topk_length : Optional[torch.Tensor]
        int32 ``[T]``; slots ``>= topk_length[t]`` are invalid regardless of
        their content.  Without it the route follows the table size
        (``cake_backend.NOPE_CONVERT_SLOT_LIMIT`` slots): below the limit the
        per-row lengths (last valid slot + 1) are derived by the forward kernel
        while it runs and saved for the backward with the rows as given -- one
        launch, a ``-1`` anywhere stays a masked slot; at or above it the rows
        are compacted on device first (valid keys first in slot order, then
        ``-1``, the valid count as the length) and the compacted pair is saved,
        so rows with holes cost their valid slots.  Hole-free rows give the
        same bytes on either route.
    softmax_scale : Optional[float]
        Defaults to ``1 / 16`` (``512 ** -0.5`` of the latent width is not
        used: the absorbed queries of this form carry their own scale).
    sink : Optional[torch.Tensor]
        FP32 ``[64]`` trainable per-head attention sink, contiguous, on the
        device of ``q``; ``None`` (default) runs without a sink, with
        unchanged results.  Head ``h`` adds ``exp(sink[h])`` -- not scaled by
        ``softmax_scale`` -- to the softmax denominator of every row and
        contributes no value: ``lse = log(sum_j exp(s_j) + exp(sink[h]))``,
        ``O = sum_j exp(s_j - lse) kv_j``, computed in FP32 and rounded once
        (``out`` is not a rescaled sink-free output).  A row whose every slot
        is invalid gives ``out = 0`` and ``lse = sink[h]``; a ``-inf`` head
        behaves as without a sink and gets ``dsink`` exactly 0.  The gradient
        reaches ``sink`` through autograd as ``dsink [64] = -sum_t exp(sink -
        lse[t]) * delta[t]`` in FP32, summed over all rows of the call
        (``delta = <dout, O>``, formed from ``out + o_lo``, or from ``out``
        alone with ``with_o_lo=False``), in a fixed order that does not depend on the
        SM count.  Only the metadata is checked (no device synchronization):
        ``+inf`` and NaN values are not supported.  SM100 / SM103 / SM107 (on
        SM107 the sink build of the forward's scanner-warp build).  A call
        without query rows (``T = 0``) launches nothing: it checks the sink's
        metadata and device, returns empty outputs and gives a zero
        ``dsink``.
    return_lse : bool
        Also return the natural-log logsumexp ``[T, 64]`` FP32 of the scaled
        scores over the valid keys (``-inf`` for fully masked rows), plus the
        sink when one is given (``sink[h]`` for fully masked rows).
    key_passes : Optional[int]
        Number of key-range passes of the backward's main stage.  ``None``
        (default) applies the registered policy of this program, which plans
        one pass on every row (one pass measured fastest on every shape of
        this form); ``1`` forces the single pass; a larger value forces that
        many passes (``<= S``) over slices of the key range, each pass's FP32
        accumulator slice sized at 2048 B per key.  The passes carry dQ
        through an FP32 partial in the workspace (``T * 131,072`` B plus
        ``T * (4 * K_slots + 4)`` B, ``T <= 4736`` tokens per chunk at
        ``K_slots`` 2048 / 2176); dQ is still written once per row and the
        dK + dV accumulation is unchanged.
    backend : str
        Only ``"cake"``.
    dkv_acc : Optional[torch.Tensor | Callable]
        Caller-owned FP32 ``[S_dst, >= 512]`` buffer (row stride a multiple of
        4 elements and at least 512, 16-byte-aligned base) into which the
        backward accumulates ``dK + dV`` in place (``+=`` into columns
        ``0:512``; the caller zeroes it).  When given, the ``kv`` gradient of
        the autograd call is ``None``.  Chunked backward passes over one long
        row accumulate into one such buffer in FP32 with no intermediate
        rounding.  Instead of the tensor, a zero-argument callable that
        returns it -- or an ``(accumulator, dkv_dst_map)`` pair -- may be
        given; the backward calls it, so the accumulator need not exist at
        forward time (under activation checkpointing a tensor passed at
        forward time is alive for every layer at once).  The autograd entries
        save every tensor they produce for the backward through
        ``ctx.save_for_backward`` and retain nothing else of a checkpointed
        forward; a tensor ``dkv_acc`` stays caller-owned.
    dkv_dst_map : Optional[torch.Tensor]
        int32 ``[S]`` destination row of ``dkv_acc`` for every key row
        (default identity).  Repeated destinations are allowed and combined
        with FP32 atomics.  Every value must lie in ``[0, S_dst)``; the kernel
        does not range-check the map (see ``check_dst_map``).
    dkv_direct : Optional[bool]
        How the backward accumulates ``dK + dV`` into its natural FP32 buffer
        (``dkv_acc`` or the fresh ``dkv``) when the registered program has the
        natural-layout main stages.  ``None`` (default) reads
        ``CAKE_PPLX_DSA_DKV_DIRECT``: ``auto`` (its default) applies the
        record's rule of the device's architecture -- the main stage adds
        into the buffer directly (no permuted accumulator, no cast launch)
        when the row has at least ``dkv_direct.min_keys_per_query`` keys per
        query token, 1 for this program: on ``sm_100a`` / ``sm_103a`` every
        row with at least as many keys as query tokens (the direct path
        measured faster than the permuted drain on every such row); on
        ``sm_107a``, whose natural-layout drain the rule models as costing extra
        in proportion to the main stage's work, only while ``T * K_eff < 112 * S + 2**20`` with
        ``K_eff = min(K_slots, ceil(S / num_segments))`` (the permuted
        accumulator plus cast was faster on the measured rows with more
        work) or ``S > 131072`` (``cast_max_keys``: on long single-segment
        rows the cast path's backward median was slower at 163,840 keys
        with ``T = 32768`` in both runs, and from 262,144 keys with
        ``T = 16384``); any other row takes
        the cast --, ``1`` forces the direct path whenever the stages are
        registered, ``0`` always
        takes the accumulate-and-cast path; any
        other value raises ``ValueError``.  ``True`` / ``False`` select the path explicitly and
        ignore the variable.  Both paths add the same FP32 contributions into
        the same rows (another summation order).
    check_dst_map : Optional[bool]
        Validate the values of ``dkv_dst_map`` on every backward (one device
        synchronization), raising ``ValueError`` on a violation.  ``None``
        (default) reads ``CAKE_PPLX_DSA_CHECK_DST_MAP`` (unset, empty
        or ``0`` = no check); ``True`` / ``False`` decide explicitly.
    num_segments : int
        The packed segment count of the key row: ``len(cu_seqlens_k) - 1``
        when ``indices`` hold the global rows of a packed batch whose tokens
        select keys of their own segment only (a Python ``int``, read on the
        host: no device synchronization; default 1, a one-document row).  The
        NoPE record plans one key-range pass on every row, so on this form
        the count reaches the ``dkv_direct`` rule alone: on SM107 its ``K_eff =
        min(K_slots, ceil(S / num_segments))``, as
        :func:`cake_dsa_sparse_attention_nondet_nope_varlen` computes it for
        the same rows; SM100 / SM103 do not read it.  The count selects the
        plan only; every value gives a valid result.  ``bool``, a tensor or a
        value below 1 raises ``ValueError``.
    with_o_lo : bool
        With ``True`` (default) the forward also writes ``o_lo``, the BF16
        rounding residual of ``out``, and the backward forms ``delta =
        rowsum(dout * (out + o_lo))`` in FP32.  This keeps the gradients
        accurate on peaked rows (rows whose softmax puts almost all weight on
        one key).  With ``False`` the forward neither computes nor stores
        ``o_lo`` (64 KiB less written and saved per query token), and the
        backward forms ``delta = rowsum(dout * out)`` from the BF16 ``out``,
        the form cuDNN's DSA backward uses; on peaked rows the gradient error
        grows to that class.  ``out`` and ``lse`` have the same accuracy in
        both modes.  Any value other than a ``bool`` raises ``ValueError``.
        With a ``sink``, ``dsink`` follows the ``delta`` of the mode.

    Returns
    -------
    out : torch.Tensor
        BF16 ``[T, 64, 512]``; zero for fully masked rows.
    lse : torch.Tensor
        Only with ``return_lse=True``.  Not differentiable.

    Gradients flow to ``q`` and ``kv`` (and ``sink`` when given).  ``dq`` is BF16, written once per row.
    ``dkv`` is accumulated in FP32 by the kernels' atomics (the summation order
    is not fixed) and handed over in FP32: through
    ``dkv_acc`` when given, otherwise as the ``kv`` gradient -- which the
    autograd engine casts to the dtype of the ``kv`` leaf (BF16 for a BF16
    leaf); call ``cake_pplx._backends.cake_dsa_sparse_attention_nondet_train.cake_backend.backward_nope``
    or pass ``dkv_acc`` to keep the FP32 values.  BF16 operands into the tensor
    cores, FP32 accumulation and statistics; the backward recomputes the scores
    from the BF16 inputs and forms ``delta`` in FP32 from the BF16 output and
    its saved BF16 residual ``o_lo`` (with ``with_o_lo=False``, from the BF16
    output alone).  The kernels apply no positional mask:
    the index rows define the key set.

    Under ``torch.use_deterministic_algorithms(True)`` the forward runs and the
    backward raises PyTorch's ``RuntimeError`` before its first backward
    kernel launch (its dK / dV atomics are not run-to-run reproducible);
    deterministic preparation may run first (see the module docstring).  With
    ``warn_only=True`` it issues a ``UserWarning`` and runs.  No deterministic
    package is chosen instead.

    The incoming ``dout`` may be a head-major view (strides ``[512, T * 512,
    1]``, the layout an output projection hands back) or any strided
    ``[T, 64, 512]`` view under the layout rules of ``q`` (unit channel
    stride, token and head strides multiples of 8 elements, 16-byte base): it
    is read in place -- the main stage through a stride-carrying tensor map,
    the delta stage through the token and head strides it receives -- with no
    host copy.

    Host side as in :func:`cake_dsa_sparse_attention_nondet`: the inputs are validated and
    bound on the first call for an input binding; later calls with the same
    binding launch from the remembered argument plans (the same
    ``BINDING_CACHE``, under keys disjoint from the rope entries).
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    num_segments = _check_num_segments(num_segments)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_nondet_nope(
        q,
        kv,
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
        sink=sink,
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_nope_varlen(
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
    with_o_lo: bool = True,
):
    r"""Packed multi-document form of :func:`cake_dsa_sparse_attention_nondet_nope` (SM100 /
    SM103 / SM107).

    Documents are packed along the first dimension of ``q`` (``T =
    cu_seqlens_q[-1]``) and of ``kv`` (``S = cu_seqlens_k[-1]``; ``[S, 512]``
    or ``[S, 1, 512]``); ``cu_seqlens_q`` and ``cu_seqlens_k`` are independent
    and query and key lengths may differ per document: the query segment is
    the tail of its key prefix, so query ``local_q`` of document ``d`` sits at
    key position ``(seqlen_k[d] - seqlen_q[d]) + local_q``.
    ``gather_kv_indices`` ``[T, K_slots]`` int32 hold key positions relative
    to the row's document (``-1`` or ``>= seqlen_k[d]`` invalid; any ``K_slots
    >= 1``) and are offset by ``cu_seqlens_k`` on device.  The route follows
    the table size (``cake_backend.NOPE_CONVERT_SLOT_LIMIT`` slots): below the
    limit, without caller lengths, the forward kernel converts the rows itself
    in its single launch, position-preserving (the global key or ``-1`` at the
    slot it came from; the row's length is its last valid slot + 1); at or
    above it, and always with a caller ``topk_length``, one launch right before
    the forward kernel compacts each row (valid keys first, in slot order, then
    ``-1``) so rows with holes cost their valid slots, and the caller's
    ``topk_length`` is a position cut applied before compaction.  Zero-length
    query segments contribute no rows; a row whose every slot is invalid gives
    ``out = 0``, ``lse = -inf`` and zero gradients.  The kernels apply no
    positional mask of their own; the offset index rows define the key sets.
    ``max_seqlen_q`` / ``max_seqlen_k`` are accepted for signature parity with
    other varlen attention entry points and are not read on the host.  See
    :func:`cake_dsa_sparse_attention_nondet_nope` for the other arguments, the
    ``1 / 16`` default scale, the FP32 ``dK + dV`` semantics and the returns.

    Under ``torch.use_deterministic_algorithms(True)`` the forward runs and the
    backward raises PyTorch's ``RuntimeError`` before its first backward
    kernel launch (its dK / dV atomics are not run-to-run reproducible);
    deterministic preparation may run first (see the module docstring).  With
    ``warn_only=True`` it issues a ``UserWarning`` and runs.  No deterministic
    package is chosen instead.

    ``cu_seqlens_q`` and ``cu_seqlens_k`` are caller-owned prefix sums with
    these invariants: int32 ``[num_docs + 1]`` tensors of one length on the
    device of ``q``, starting at 0 and non-decreasing, ``cu_seqlens_q[-1] ==
    T`` (the packed rows of ``q``), ``cu_seqlens_k[-1] == S`` (the packed rows
    of ``kv``; a shorter or longer key window is not accepted) and ``seqlen_q
    <= seqlen_k`` for every document.  Their structure (contiguous int32
    ``[num_docs + 1]`` of one length) is checked on every call; their values
    only with ``check_cu_seqlens=True``, since that check synchronizes with
    the device, and a violation is otherwise not detected on device: a
    document with ``seqlen_k < seqlen_q`` gives negative own positions
    (all-invalid rows under ``causal``), a non-monotonic prefix sum assigns
    rows to the wrong document.  The per-document rows are turned into global
    key rows on device: by the forward kernel itself below the slot limit
    (position-preserving, one launch), else compacted in one launch right
    before the forward kernel; a caller ``topk_length`` is a position cut
    applied before compaction and the kernels then read the compacted rows and
    their counts.

    Parameters
    ----------
    sink : Optional[torch.Tensor]
        FP32 ``[64]`` trainable per-head sink, as in
        :func:`cake_dsa_sparse_attention_nondet_nope`: the forward of every
        route (in-kernel conversion or index preparation) runs the sink build
        over the packed rows, a row whose every slot is invalid gives
        ``out = 0`` and ``lse = sink[h]``, a zero-length query segment adds no
        row, and the gradient ``dsink`` is summed over all packed rows of the
        call.  ``None`` (default) runs without a sink, with unchanged
        results.
    check_cu_seqlens : bool
        ``True`` checks the values of ``cu_seqlens_q`` / ``cu_seqlens_k``
        against the invariants above at the cost of one blocking device-to-host
        copy per call, before any launch (skipped during CUDA-graph capture,
        where no synchronization is possible; the eager call captured was
        checked).  ``False`` (default) keeps the structural checks (dtype,
        rank, length, contiguity, device) only.
    causal : bool
        With ``False`` (default) the index row is taken as is -- only ``-1`` /
        out-of-range slots are dropped.  With ``True`` a slot that selects a
        key after the query's own position (``idx > (seqlen_k[d] -
        seqlen_q[d]) + local_q``) is invalid too, so a top-k selector need not
        enforce causality itself.
    dkv_direct : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_nondet_nope`: ``None`` (default) reads
        ``CAKE_PPLX_DSA_DKV_DIRECT``; ``True`` / ``False`` decide
        explicitly.
    check_dst_map : Optional[bool]
        As in :func:`cake_dsa_sparse_attention_nondet_nope`: ``None`` (default) reads
        ``CAKE_PPLX_DSA_CHECK_DST_MAP``; ``True`` validates the values of
        ``dkv_dst_map`` on every backward, ``False`` never does.
    with_o_lo : bool
        As in :func:`cake_dsa_sparse_attention_nondet_nope`: ``True``
        (default) writes ``o_lo`` and forms ``delta`` from ``out + o_lo``;
        ``False`` writes no ``o_lo`` and forms ``delta`` (and, with a
        ``sink``, ``dsink``) from the BF16 ``out`` alone, on every index
        route.  Any value other than a ``bool`` raises ``ValueError``.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
    if check_dst_map is None:
        check_dst_map = _check_dst_map_from_environment()
    return _backend(backend).cake_dsa_sparse_attention_nondet_nope_varlen(
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
        sink=sink,
        with_o_lo=with_o_lo,
    )


# ---------------------------------------------------------------------------
# Explicit (non-autograd) steps: the forward and the backward as two calls.
# ---------------------------------------------------------------------------


@cake_api
def cake_dsa_sparse_attention_nondet_train_forward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    backend: str = "cake",
    with_o_lo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Training forward of the rope form without autograd: ``(out, lse,
    o_lo)``, or ``(out, lse, None)`` with ``with_o_lo=False``.

    The inputs are those of :func:`cake_dsa_sparse_attention_nondet`. ``out``
    is BF16 ``[T, H, 512]`` (zero for a fully masked row), ``lse`` FP32 ``[T,
    H]`` and ``o_lo`` BF16 ``[T, H, 512]``, the rounding residual of ``out``
    (``out + o_lo`` is the output before its BF16 rounding). Pass the three
    unchanged to :func:`cake_dsa_sparse_attention_nondet_train_backward`.
    ``out``, ``lse`` and ``o_lo`` of a head come from one forward kernel launch
    (with 96 or 128 heads, the single-pass or the head-window forward, or on
    the two-window bridge the launch of the window that delivers the head). A
    call without query rows returns empty outputs without launching.

    ``with_o_lo`` (``bool``, default ``True``): with ``True`` the forward also
    writes ``o_lo`` as above and the step backward forms ``delta =
    rowsum(dout * (out + o_lo))`` in FP32, which keeps the gradients accurate
    on peaked rows (rows whose softmax puts almost all weight on one key).
    With ``False`` the forward neither computes nor stores ``o_lo`` (``H`` KiB
    less written per query token; above 64 heads on the two-window bridge also
    no ``fwd_o_lo`` window scratch, 64 KiB per token of workspace, and no
    ``o_lo`` window copies) and
    returns ``None`` in its slot; pass
    ``None`` and ``with_o_lo=False`` to the step backward, which forms
    ``delta = rowsum(dout * out)`` from the BF16 ``out``, the form cuDNN's DSA
    backward uses (on peaked rows the gradient error grows to that class).
    ``out`` and ``lse`` have the same accuracy in both modes.  Any value
    other than a ``bool`` raises ``ValueError``.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    return _backend(backend).forward(
        q_latent,
        q_rope,
        kv_latent,
        k_rope,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_train_backward(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: Optional[torch.Tensor],
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
    with_o_lo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Training backward of the rope form without autograd: ``(dq_latent,
    dq_rope, dkv_latent, dk_rope)``.

    ``out``, ``o_lo`` and ``lse`` are the outputs of
    :func:`cake_dsa_sparse_attention_nondet_train_forward` on the same inputs
    and ``topk_length`` / ``softmax_scale``; ``dout`` is the BF16 ``[T, H,
    512]`` gradient of ``out``. The call leaves ``o_lo`` unchanged.  With 96 or
    128 heads a ``dout`` whose token stride is not ``H`` head strides (a
    head-major view) is copied once into the backward's workspace.
    ``dq_latent`` / ``dq_rope`` are BF16. ``dkv_latent`` / ``dk_rope`` are BF16
    in the layout of ``kv_latent`` / ``k_rope``, or FP32 with
    ``dkv_fp32=True``. With a tensor ``dkv_acc`` (FP32 ``[S_dst, >= 576]``) the
    call adds ``dK + dV`` into it instead (latent columns ``0:512``, rope
    ``512:576`` of row ``dkv_dst_map[s]``; the caller owns the zeroing) and
    returns ``None`` for both. ``key_passes``, ``check_dst_map`` and
    ``num_segments`` (the packed segment count of the key row, default 1) as
    in :func:`cake_dsa_sparse_attention_nondet`; ``dkv_direct``: ``None``
    (default) reads ``CAKE_PPLX_DSA_DKV_DIRECT``; ``True`` / ``False`` decide
    explicitly. ``dK`` / ``dV`` are added with FP32 atomics; no output is
    promised to repeat bit for bit. Under
    ``torch.use_deterministic_algorithms(True)`` it raises PyTorch's
    ``RuntimeError`` before its first backward kernel (with ``warn_only=True``
    a ``UserWarning``, then it runs).

    ``with_o_lo`` (``bool``, default ``True``) is the step forward's: ``o_lo``
    is the tensor that forward returned with ``True`` and ``None`` with
    ``False``, which forms ``delta = rowsum(dout * out)`` from the BF16
    ``out``.  A tensor with ``False``, or ``None`` with ``True`` on a call
    with query rows, raises ``ValueError`` before anything is planned or
    launched; any value of ``with_o_lo`` other than a ``bool`` raises
    ``ValueError`` too.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    num_segments = _check_num_segments(num_segments)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
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
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_train_nope_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sink: Optional[torch.Tensor] = None,
    backend: str = "cake",
    with_o_lo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Training forward of the NoPE form without autograd: ``(out, lse,
    o_lo)``, or ``(out, lse, None)`` with ``with_o_lo=False``.

    The inputs are those of :func:`cake_dsa_sparse_attention_nondet_nope`, and
    the outputs those of
    :func:`cake_dsa_sparse_attention_nondet_train_forward`. Pass them unchanged
    to :func:`cake_dsa_sparse_attention_nondet_train_nope_backward`. ``out``,
    ``lse`` and ``o_lo`` come from one forward kernel. With ``sink`` (FP32
    ``[64]``, as in :func:`cake_dsa_sparse_attention_nondet_nope`) ``lse``
    includes the sink (``sink[h]`` on a row without a valid slot); pass the
    same ``sink`` to the step backward. A call without query rows returns
    empty outputs without launching. ``with_o_lo`` as in
    :func:`cake_dsa_sparse_attention_nondet_train_forward`.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    return _backend(backend).forward_nope(
        q,
        kv,
        indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        sink=sink,
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_train_nope_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: Optional[torch.Tensor],
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
    num_segments: int = 1,
    with_o_lo: bool = True,
) -> tuple[Optional[torch.Tensor], ...]:
    """Training backward of the NoPE form without autograd: ``(dq, dkv)``,
    or ``(dq, dkv, dsink)`` with a ``sink``.

    ``out``, ``o_lo`` and ``lse`` are the outputs of
    :func:`cake_dsa_sparse_attention_nondet_train_nope_forward` on the same
    inputs. ``dq`` is BF16 ``[T, 64, 512]``; ``dkv = dK + dV`` is FP32 ``[S,
    512]``. With a tensor ``dkv_acc`` (FP32 ``[S_dst, >= 512]``) the call adds
    ``dkv`` into columns ``0:512`` of row ``dkv_dst_map[s]`` instead (the
    caller owns the zeroing) and returns ``(dq, None)``. The call leaves
    ``o_lo`` unchanged. ``key_passes`` and ``check_dst_map`` as in
    :func:`cake_dsa_sparse_attention_nondet_nope`; ``dkv_direct``: ``None``
    (default) reads ``CAKE_PPLX_DSA_DKV_DIRECT``; ``True`` / ``False`` decide
    explicitly. ``dK`` / ``dV`` are added with FP32 atomics; no output is
    promised to repeat bit for bit. Under
    ``torch.use_deterministic_algorithms(True)`` it raises PyTorch's
    ``RuntimeError`` before its first backward kernel (with ``warn_only=True``
    a ``UserWarning``, then it runs). ``sink``: the sink the step forward ran
    with; the call then also returns ``dsink``, FP32 ``[64]``, ``-sum_t
    exp(sink - lse[t]) * delta[t]`` summed over all rows of the call:
    ``(dq, dkv, dsink)``, or ``(dq, None, dsink)`` with a tensor ``dkv_acc``.
    With ``sink=None`` the return is ``(dq, dkv)`` or ``(dq, None)``.
    ``num_segments`` (the packed segment count of the key row, default 1) as
    in :func:`cake_dsa_sparse_attention_nondet_nope`. A call without query
    rows launches nothing and returns an empty ``dq``, a zero ``dkv``
    (``None`` with a tensor ``dkv_acc``) and, with a ``sink``, a zero
    ``dsink``.

    ``with_o_lo`` (``bool``, default ``True``) is the step forward's: ``o_lo``
    is the tensor that forward returned with ``True`` and ``None`` with
    ``False``, which forms ``delta = rowsum(dout * out)`` from the BF16
    ``out`` (and, with a ``sink``, ``dsink`` from that ``delta``).  A tensor
    with ``False``, or ``None`` with ``True`` on a call with query rows,
    raises ``ValueError`` before anything is planned or launched; any value
    of ``with_o_lo`` other than a ``bool`` raises ``ValueError`` too.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    num_segments = _check_num_segments(num_segments)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
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
        num_segments=num_segments,
        dkv_direct=dkv_direct,
        check_dst_map=check_dst_map,
        sink=sink,
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_train_nope_varlen_forward(
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
    with_o_lo: bool = True,
) -> tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Training forward of the packed NoPE form without autograd: ``(out, lse,
    o_lo, indices, topk_length)``; the ``o_lo`` slot is ``None`` with
    ``with_o_lo=False``.

    The inputs and their checks are those of
    :func:`cake_dsa_sparse_attention_nondet_nope_varlen`. ``out``, ``lse`` and
    ``o_lo`` as in :func:`cake_dsa_sparse_attention_nondet_train_forward`;
    ``indices`` (int32 ``[T, K_slots]``, global key rows) and ``topk_length``
    (int32 ``[T]``) are the rows and row lengths the forward read. Pass all
    five unchanged to
    :func:`cake_dsa_sparse_attention_nondet_train_nope_varlen_backward`.
    ``out``, ``lse`` and ``o_lo`` come from one forward kernel. With
    ``sink`` (FP32 ``[64]``, as in
    :func:`cake_dsa_sparse_attention_nondet_nope_varlen`) ``lse`` includes the
    sink (``sink[h]`` on a row without a valid slot); pass the same ``sink``
    to the step backward. ``with_o_lo`` as in
    :func:`cake_dsa_sparse_attention_nondet_train_forward`.
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
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
        sink=sink,
        with_o_lo=with_o_lo,
    )


@cake_api
def cake_dsa_sparse_attention_nondet_train_nope_varlen_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    o_lo: Optional[torch.Tensor],
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
    with_o_lo: bool = True,
) -> tuple[Optional[torch.Tensor], ...]:
    """Training backward of the packed NoPE form without autograd: ``(dq,
    dkv)``, or ``(dq, dkv, dsink)`` with a ``sink``.

    ``out``, ``o_lo``, ``lse``, ``indices`` and ``topk_length`` are the outputs
    of :func:`cake_dsa_sparse_attention_nondet_train_nope_varlen_forward`;
    ``cu_seqlens_k`` is the forward's. The outputs and the other arguments as
    in :func:`cake_dsa_sparse_attention_nondet_train_nope_backward`.
    ``dK`` / ``dV`` are added with FP32 atomics; no output is promised to
    repeat bit for bit. Under ``torch.use_deterministic_algorithms(True)``
    it raises PyTorch's ``RuntimeError`` before its first backward kernel
    (with ``warn_only=True`` a ``UserWarning``, then it runs). ``sink``: the
    sink the step forward ran with; the call then also returns ``dsink``,
    FP32 ``[64]``, summed over all packed rows of the call. ``with_o_lo`` as
    in :func:`cake_dsa_sparse_attention_nondet_train_nope_backward`: ``o_lo``
    is ``None`` exactly when ``with_o_lo`` is ``False`` (a default call
    without query rows also accepts ``None``).
    """
    with_o_lo = _check_with_o_lo(with_o_lo)
    if dkv_direct is None:
        dkv_direct = _dkv_direct_from_environment()
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
        sink=sink,
        with_o_lo=with_o_lo,
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
