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

Cake backend: fused DSA indexer scoring + deterministic exact top-k selection
for training on SM100 / SM103 / SM107 (target geometry: 32 indexer heads, head
dimension 128, top-k 2048).

Contract
--------
* ``q [T, 32, 128]`` BF16 indexer queries after RoPE (contiguous); ``k [Tkv,
  128]`` BF16 packed keys after normalization and RoPE -- a row-strided view
  (the trainer's ``[:, :128]`` view of a packed ``[Tkv, 704]`` tensor) is
  accepted without a copy; ``w [T, 32]`` FP32 signed head weights already
  scaled by ``32 ** -0.5``; ``cu_seqlens_q`` / ``cu_seqlens_k [S + 1]`` int32
  independent query / key segment boundaries on the device; optional
  ``q_causal_offsets [S]`` int64 on the device; ``top_k`` in ``[1, 4096]``
  (default 2048); ``softmax_scale > 0`` (default ``128 ** -0.5``); ``ratio
  >= 1``; ``max_seqlen_q`` is an accepted host mirror that no launch decision
  reads; ``max_seqlen_k`` (the caller's bound on every key segment, optional)
  sizes the rank finalize's bitmap pool and so selects its program variant
  (``rank_seg_window``), and where the record ships ``lsf_programs`` it runs a
  packed call whose key segments all have that length (``Tkv == S *
  max_seqlen_k``) without the ``:lsf`` copy -- never a result; a value above
  ``Tkv`` or below ``ceil(Tkv / S)`` is rejected.
* Outputs ``indices [T, top_k]`` int32 segment-local key ids and ``scores [T,
  top_k]`` FP32, both in ascending id order with the tail padded by ``-1`` /
  ``-inf``; an entirely empty row holds only padding.
* Score ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] *
  k[j, d])`` with FP32 products, accumulation and score arithmetic (no BF16
  intermediate rounding).  The FP32 reduction scheme of the generated program
  is documented in the package README and is fixed: the result does not depend
  on the internal query / key partition.
* Visibility: key ``j`` is visible to the query at segment-local position ``u``
  iff ``0 <= j < Lk`` and ``j < floor((offset + u + 1) / ratio)``; ``offset`` is
  ``q_causal_offsets[s]`` when supplied, else ``Lk - Lq`` for ``ratio == 1`` and
  ``0`` otherwise.  Keys of other segments or outside the visible prefix are
  never selected; negative offsets, empty segments and rows without visible
  keys are defined (all padding).
* Selection: the exact global top ``min(top_k, visible)`` keys per row, ranked
  by score descending then key id descending (``+0.0 == -0.0`` for ranking).
* Repeatable: identical inputs give identical ids and score bits; the program
  the host dispatches (unit geometry, key-range split, CTA pair, unroll, unit
  order, sampled threshold; the CUB block radix sort or the prefix-popcount
  rank scatter as the finalize) and the partition knobs (``grid_ctas``,
  ``candidate_multiplier``, ``check_period``, the sampled-threshold knobs, the
  finalize program) never change a result.
* Memory: no ``[T, Tkv]`` score matrix.  The workspace is
  :func:`cake_dsa_indexer_det_train_topk_workspace_size` bytes for the call's geometry -- the
  persistent candidate buffers ``grid x queries_per_unit x
  candidate_capacity(top_k) x entry_bytes`` plus, when the dispatch splits the
  key ranges, the staging ``T x n_split x top_k x entry_bytes`` -- separate
  from the inputs and the two outputs; no host synchronization inside the
  operator.
* Non-finite inputs (NaN / inf / overflow) are outside the normal domain: the
  operator returns with the output structure intact and no loop bound depends
  on a score value, so the kernels cannot hang.

FP8 form (opt-in, selected by the input dtype)
----------------------------------------------
* ``q [T, 32, 128]`` and ``k [Tkv, 128]`` float8_e4m3fn (the same layout
  rules; a row stride that is a multiple of 16 bytes), ``k_scale [Tkv]`` FP32
  per-key scales (contiguous, same device), ``w [T, 32]`` FP32 carrying every
  other scale (the query scale, the softmax scale and the head weights);
  ``softmax_scale`` must be ``None`` (the programs take 1.0).  Mixed q / k
  dtypes and float8_e5m2 are rejected; a BF16 call takes no ``k_scale``.
* Score ``s[t, j] = k_scale[j] * sum_h w[t, h] * relu(sum_d q[t, h, d] * k[j,
  d])``: the e4m3 products are exact and accumulate in FP32 on the tensor
  cores, four K = 32 steps on sm_100a / sm_103a (the order of DeepGEMM's
  ``fp8_mqa_logits``) and two K = 64 steps on sm_107a; the head sum and the
  ``k_scale[j]`` product are FP32.
* Selection, visibility, outputs and the workspace bound follow the BF16
  contract; every FP8 program of one architecture computes the same ids and
  score bits, so the dispatched program never changes a result.  FP8 results
  are not BF16 results, and sm_107a results can differ from sm_100a / sm_103a
  results in the last bits.

Candidate blocks (``cake_dsa_indexer_det_train_topk_with_candidates``)
-----------------------------------------------------------------------
* The same call (BF16 only, ``ratio == 1``, ``1 <= top_k <= 2047``, default
  512) also returns ``candidate_blocks [T, 2048]`` int32 computed from the
  same scores as ``indices``: block ``b`` holds the segment-local keys
  ``8 b .. 8 b + 7``; a row that sees ``n`` keys sees ``nb = ceil(n / 8)``
  blocks, a block's score is the largest score of its visible keys, and the
  row holds ``0 .. nb - 1`` when ``nb <= 2048``, else block ``nb - 1`` (the
  newest, always kept) and the 2047 older blocks of the largest block scores
  (ties: the larger block id; ``+0.0 == -0.0``); ``-1`` fills the other
  slots (all ``-1`` when ``n == 0``).  Every id of ``indices`` lies in a block
  of its row.  Rows are in ascending block id order with the ``-1`` tail and repeat
  bit for bit.
* A call whose longest key segment ``W`` (below) holds at most 2048 blocks
  keeps every block of every row: it runs the plain call's programs as
  dispatched, with the plain call's workspace, and writes each row's
  ``0 .. nb - 1`` and ``-1`` tail on the device before them (stage
  ``cb_iota``: a few small launches over the rows and one row copy, no host
  synchronization, graph-capturable; ``CandidateBlocksChoice.iota``).  A row
  past 2048 blocks (a ``max_seqlen_k`` below one of its key segments) stops
  that fill with a device-side assert, before the call's programs run.
* The scan of any other such call is the unsplit, unpaired narrow or l6 program of
  the decided unit (``cake_policy.select_cb_program``).  Its workspace is
  :func:`cake_dsa_indexer_det_train_topk_with_candidates_workspace_size` bytes:
  the scan's candidate buffers and ticket tail, then ``grid x block_q x
  blk_cap x 8`` bytes of block entries (``blk_cap = ceil(W / 8) - 1`` for the
  longest key segment ``W``, which holds more than 2048 blocks), separate
  from the caller's ``[T, 2048]`` output.  ``W`` is ``max_seqlen_k``
  when given, else ``Tkv``: with candidate blocks ``max_seqlen_k`` is a
  correctness bound, and a row with more older blocks than the workspace holds
  stops the kernel with a device trap.

Every allocation happens in :func:`prepare_dsa_indexer_det_topk`; the returned
runner launches with no CUDA allocation and no host synchronization, so a
runner (or a CUDA graph capturing it) replays for new values written into the
bound tensors.  The host decides the program from host-known integers only
(:mod:`cake_policy`, the registry's ``POLICY`` record of the architecture,
with its ``POLICY_FP8`` fields for the FP8 form) and fails closed when the
decided program is not registered for the device's architecture.  Kernels
are reached through the argument plans of the registry
(``cake_jit.ARG_PLANS``); ``cake_jit.ABI`` names the keyword set its kernels
expect (:data:`ABI_CONTRACT`).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    ABI,
    ARG_PLANS,
    FFI_ENTRY,
    NUMERICS,
    POLICY,
    POLICY_FP8,
    PROGRAM_KEYS,
    PROGRAMS,
    STAGES,
    TICKET_TAIL_ARCHES,
    load_program,
)
from .cake_policy import (
    BF16,
    CAND_MAX_TOP_K,
    CAND_ROLE,
    CAND_TICKET_BYTES,
    CAND_WIDTH_MAX,
    CB_IOTA_STAGE,
    CB_MAX_TOP_K,
    CB_ROLE,
    CB_SLOTS,
    FINALIZE_ROLE,
    FP8,
    MERGE_KEY,
    SAMPLE_SHIFT_PERMILLE_BOUNDS,
    TICKET_TAIL_BYTES,
    CandidateBlocksChoice,
    CandidateTopKChoice,
    DispatchPolicy,
    ProgramChoice,
    finalize_grid,
    fp8_policy_record,
    fp8_scan_programs,
    scan_key_form,
    select_cand_program,
    select_cb_program,
    select_fp8_program,
    select_program,
    ticket_static_path,
)

NUM_HEADS = 32
HEAD_DIM = 128
MAX_TOP_K = 4096
DEFAULT_TOP_K = 2048
PAD_ID = -1
PAD_SCORE = float("-inf")
TMA_ALIGNMENT_BYTES = 16
SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The q / k dtype of each input form (``cake_policy.FORMS``): BF16 is the default, FP8 the opt-in form.
FORM_DTYPES = {BF16: torch.bfloat16, FP8: torch.float8_e4m3fn}
_DTYPE_NAMES = {BF16: "bfloat16", FP8: "float8_e4m3fn"}

# Host binding profile: the keyword set of the generated kernels (the names
# below are what the host provides; a kernel's argument plan selects from
# them).  Tensor kinds in the argument plan are ``buffer`` / ``tma_buffer``
# (the binding encodes the tensor map itself from the view it receives),
# scalars are ``parameter``; ``grid_x/y/z`` are ``grid``; ``zero_region`` (kind
# ``zero``, scan only) is the byte range the scan binding zeroes on the launch
# stream right before its launch: the caller-owned ticket tail, else empty.
# ``TicketWord`` (scan only, right before ``zero_region``) is the 16-byte block
# a scan program draws its work units from where its registry record lists the
# architecture in ``ticket_arg_arches``; every other scan program takes it and
# never reads it.  ``K_scale`` (scan only, right before ``TicketWord``) is the
# per-key FP32 scale vector of the FP8 form: the caller's ``k_scale`` for an
# FP8 program; a BF16 program never reads it, so the host passes it a
# never-dereferenced placeholder.
ABI_CONTRACT = "dsa_indexer_v5"
SUPPORTED_ABIS = (ABI_CONTRACT,)
CONTRACT_TENSORS = (
    "Q",  # q viewed as [T * 32, 128] (TMA source; the FP8 form's e4m3 bytes as uint8)
    "K",  # k [Tkv, 128], any row stride that is a multiple of 16 bytes (TMA source; the FP8 form's e4m3 bytes as uint8)
    "W",  # w [T, 32] fp32 (TMA source)
    "cu_seqlens_q",
    "cu_seqlens_k",
    "q_offsets",  # q_causal_offsets, or a never-dereferenced int64 placeholder
    "Indices",
    "Scores",
    "Cand",  # the workspace viewed as packed int64 candidate entries (buffers, then the split staging)
    "Staging",  # the split staging [T, n_split, top_k] behind the candidate buffers (merge stage only)
    "zero_region",  # the scan's caller-owned ticket tail, zeroed by its binding before the launch, or an empty view
    "TicketWord",  # the scan's 16-byte ticket block (uint32 view): its own, or a zeroed placeholder it never reads
    "K_scale",  # the FP8 form's per-key FP32 scales k_scale [Tkv]; a never-dereferenced float32 placeholder for a BF16 program
    "CandBlk",  # scan_blocks: the block entries behind the workspace's ticket tail (int64), or a never-read placeholder
    "CandBlocks",  # scan_blocks: the candidate-block rows [T, 2048] int32 (the caller's candidate_blocks); cand: the
    # caller's candidate_blocks [T, W] int32 (W <= 2048)
    "Kraw",  # cand: k [Tkv, 128] as its plain rows, any row stride that is a multiple of 16 bytes (the binding encodes
    # the program's block tensor map from them; the newest partial block's copy reads the row stride)
    "Lists",  # cand: the per-CTA block lists in the workspace (int32)
)
SCAN_SCALARS = (
    "num_segments",
    "top_k",
    "ratio",
    "has_offsets",
    "cand_cap",
    "first_cap",
    "sample_tiles_max",
    "sample_shift_permille",
    "far_sample",
    "check_period",
    "grid_ctas",
    "softmax_scale",
    "n_split",
)
MERGE_SCALARS = ("top_k", "n_split")
# scan_blocks: the block entries per query slot, beside the scan's scalars
CB_SCALARS = ("blk_cap",)
# cand: the candidate-restricted top-k's own scalar beside the scan's names it shares
CAND_SCALARS = ("cand_width",)
FINALIZE_SCALARS = ("top_k", "key_bits")  # the CUB block radix sort programs
FINALIZE_RANK_SCALARS = (
    "top_k",
    "num_segments",
)  # the prefix-popcount rank finalize programs (+ cu_seqlens_q / cu_seqlens_k)
CONTRACT_SCALARS = SCAN_SCALARS + CB_SCALARS + CAND_SCALARS + ("key_bits",)
# Kernel-side spellings of the contract names (a renamed kernel argument
# resolves through this table; any other name fails closed at bind time).
CONTRACT_ALIASES = {
    "q": "Q",
    "k": "K",
    "w": "W",
    "indices": "Indices",
    "scores": "Scores",
    "cand": "Cand",
    "candidates": "Cand",
    "workspace": "Cand",
    "staging": "Staging",
    "q_causal_offsets": "q_offsets",
    "offsets": "q_offsets",
    "num_segs": "num_segments",
    "topk": "top_k",
    "scale": "softmax_scale",
    "sm_scale": "softmax_scale",
    "grid": "grid_ctas",
    "num_splits": "n_split",
    "candidate_blocks": "CandBlocks",
    "block_entries": "CandBlk",
}
# The candidate-block call (``*_with_candidates``): its default top_k (the issue's value) and bound (``CB_MAX_TOP_K``:
# with at most 2047 ids every returned id lies in a returned block).
CB_DEFAULT_TOP_K = 512

# Documented zero-sign behaviour of the head reduction (registry field
# ``NUMERICS["zero_sign_policy"]``): ``positive_accumulator`` -- the FMA chains
# start from ``+0.0``, so a row whose head terms are all zero scores ``+0.0``;
# ``ieee_sum`` -- the sum starts from the ``h = 0`` term, so an all-zero sum is
# ``-0.0`` iff every term is ``-0.0``.  Either way ``+0.0 == -0.0`` for
# ranking.
ZERO_SIGN_POLICIES = ("positive_accumulator", "ieee_sum")


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is not None and torch.device(device).type != "cuda":
        # the refusal torch.cuda.get_device_capability raised here before the facts were memoised
        raise ValueError(f"Expected a cuda device, but got: {device}")
    capability, _ = _device_facts(_device_index(device))
    return SUPPORTED_COMPUTE_CAPABILITIES.get(capability)


def default_softmax_scale() -> float:
    return HEAD_DIM**-0.5


def form_for(dtype: torch.dtype) -> str:
    """The input form (``FORM_DTYPES``) of a q / k dtype: ``"bf16"`` for bfloat16, ``"fp8"`` for float8_e4m3fn."""
    for form, form_dtype in FORM_DTYPES.items():
        if dtype == form_dtype:
            return form
    raise ValueError(
        f"dtype must be torch.bfloat16 or torch.float8_e4m3fn, got {dtype}"
    )


def registry_abi() -> str:
    if ABI not in SUPPORTED_ABIS:
        raise NotImplementedError(
            f"unsupported host binding profile {ABI!r} (this host binds {SUPPORTED_ABIS})"
        )
    return ABI


def policy_for(arch: str, form: str = BF16) -> DispatchPolicy:
    """The dispatch policy of the input form ``form`` on ``arch``: the registry's ``POLICY`` record (BF16), or for the
    FP8 form that record with the fields of the registry's ``POLICY_FP8`` record replaced (``fp8_policy_record``).
    The registry is immutable for the life of the process, so a record is validated once per architecture and form;
    an architecture without programs of the form raises on every call."""
    return _policy_for(arch, form)


@functools.cache
def _policy_for(arch: str, form: str) -> DispatchPolicy:
    record = POLICY.get(arch)
    if record is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered in this checkout "
            f"(registered: {sorted(POLICY)})"
        )
    if form == FP8:
        if not fp8_programs_for(arch) or arch not in POLICY_FP8:
            raise NotImplementedError(
                f"The generated float8_e4m3fn DSA indexer top-k programs for {arch} are not registered in this "
                "checkout"
            )
        record = fp8_policy_record(record, POLICY_FP8[arch])
    elif form != BF16:
        raise ValueError(f"form must be one of {tuple(FORM_DTYPES)}, got {form!r}")
    return DispatchPolicy.from_record(record)


@functools.cache
def fp8_programs_for(arch: str) -> dict:
    """The FP8 scan programs ``arch`` registers (``cake_policy.fp8_scan_programs``), indexed once per architecture;
    empty where the registry has none."""
    return fp8_scan_programs(PROGRAM_KEYS.get(arch, {}))


@functools.cache
def ticket_tail_skips(arch: str, form: str = BF16) -> Optional[frozenset]:
    """None when ``arch`` is not a ticket architecture (``TICKET_TAIL_ARCHES``); else the physical scan kinds of the
    ``static_scan_kinds`` of the form's record (``policy_for``), whose programs keep the static enumeration and get no
    ticket tail.  Fixed for the life of the process (like ``policy_for``), so a call decides its tail with one lookup
    and one set test."""
    if arch not in TICKET_TAIL_ARCHES:
        return None
    return frozenset(policy_for(arch, form).static_scan_kinds)


@functools.cache
def reads_ticket_word(program: str, arch: str) -> bool:
    """Whether the scan ``program`` draws its work units on ``arch`` from its ``TicketWord`` argument (the registry
    record's ``ticket_arg_arches``, which the exporter writes from the program's source build) instead of the
    workspace's ticket tail."""
    return arch in PROGRAMS[program].get("ticket_arg_arches", ())


def program_for(arch: str, key: str) -> str:
    """The registered program of ``key`` on ``arch``; fails closed by name when the dispatch reaches an unshipped form."""
    programs = PROGRAM_KEYS.get(arch)
    if programs is None:
        raise NotImplementedError(
            f"The generated DSA indexer top-k programs for {arch} are not registered in this checkout "
            f"(registered: {sorted(PROGRAM_KEYS)})"
        )
    name = programs.get(key)
    if name is None:
        raise NotImplementedError(
            f"The host dispatch selected program {key!r} on {arch}, which this checkout does not register "
            f"(registered on {arch}: {sorted(programs)})"
        )
    return name


def generated_program_available(
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.bfloat16
) -> bool:
    """True when this checkout registers programs for ``device`` under the host binding profile this module binds;
    ``dtype``: the q / k dtype of the input form (``torch.float8_e4m3fn``: the FP8 form's scan programs)."""
    form = form_for(dtype)
    arch = arch_for(device)
    if (
        arch is None
        or arch not in PROGRAM_KEYS
        or arch not in POLICY
        or ABI not in SUPPORTED_ABIS
    ):
        return False
    keys = PROGRAM_KEYS[arch]
    scan = (
        bool(fp8_programs_for(arch))
        if form == FP8
        else any(k.startswith("scan:") for k in keys)
    )
    return (
        scan
        and MERGE_KEY in keys
        and any(k.startswith(("finalize:", "finalize_rank:")) for k in keys)
    )


def record_zero_sign_policy() -> str:
    """The documented zero-sign policy of the program's head reduction."""
    policy = NUMERICS.get("zero_sign_policy") if isinstance(NUMERICS, dict) else None
    if policy not in ZERO_SIGN_POLICIES:
        raise ValueError(
            f"registry declares numerics {NUMERICS!r}; zero_sign_policy must be one of {ZERO_SIGN_POLICIES}"
        )
    return str(policy)


# ---------------------------------------------------------------------------
# Visibility (host helpers; the kernels evaluate the same rule on device)
# ---------------------------------------------------------------------------


def effective_offsets(
    seg_q_len: list[int],
    seg_k_len: list[int],
    q_causal_offsets: Optional[list[int]],
    ratio: int,
) -> list[int]:
    """Per-segment effective causal offset: the supplied value, else ``Lk - Lq`` for ``ratio == 1``, else ``0``."""
    if q_causal_offsets is not None:
        if len(q_causal_offsets) != len(seg_q_len):
            raise ValueError("q_causal_offsets must hold one entry per segment")
        return [int(o) for o in q_causal_offsets]
    if int(ratio) == 1:
        return [int(lk) - int(lq) for lq, lk in zip(seg_q_len, seg_k_len, strict=True)]
    return [0] * len(seg_q_len)


def visible_key_count(offset: int, position: int, ratio: int, num_keys: int) -> int:
    """Visible keys of one query: ``max(0, min(Lk, floor((offset + u + 1) / ratio)))`` (true floor)."""
    return max(0, min(int(num_keys), (int(offset) + int(position) + 1) // int(ratio)))


def visible_key_counts(
    seg_q_len: list[int],
    seg_k_len: list[int],
    *,
    ratio: int = 1,
    q_causal_offsets: Optional[list[int]] = None,
    device=None,
) -> torch.Tensor:
    """int64 ``[T]`` visible-key count of every query row from host segment metadata (floor division)."""
    offsets = effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, ratio)
    pieces = []
    for lq, lk, off in zip(seg_q_len, seg_k_len, offsets, strict=True):
        if lq == 0:
            continue
        u = torch.arange(int(lq), dtype=torch.int64, device=device)
        vis = torch.div(u + (int(off) + 1), int(ratio), rounding_mode="floor")
        pieces.append(vis.clamp_(min=0, max=int(lk)))
    if not pieces:
        return torch.zeros(0, dtype=torch.int64, device=device)
    return torch.cat(pieces)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_int(name: str, value: Any, lo: int, hi: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < lo or (hi is not None and value > hi):
        bound = f"in [{lo}, {hi}]" if hi is not None else f">= {lo}"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return int(value)


def _check_max_seqlen_k(value: int, num_keys: int, num_segments: int) -> int:
    """``max_seqlen_k`` as a host bound on every key segment: an int in ``[ceil(Tkv / S), Tkv]``."""
    v = _check_int("max_seqlen_k", value, 0)
    if v > int(num_keys) or v * max(1, int(num_segments)) < int(num_keys):
        raise ValueError(
            f"max_seqlen_k={v} is not an upper bound of every key segment: {int(num_keys)} keys in {int(num_segments)} segments"
        )
    return v


def validate_dsa_indexer_det_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_causal_offsets: Optional[torch.Tensor],
    top_k: int,
    softmax_scale: float,
    ratio: int,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    k_scale: Optional[torch.Tensor] = None,
) -> tuple[int, int, int]:
    """Check the contract's shapes, dtypes, strides, alignment and scalars; return ``(T, Tkv, S)``.

    The q dtype selects the input form (``form_for``): bfloat16 q / k, or
    float8_e4m3fn q / k with ``k_scale`` (FP32 ``[Tkv]``, contiguous).  Shape
    and dtype rules are checked before the device rule so that host tensors
    exercise them in CPU-only tests.  No host synchronization: the segment
    boundaries are not read.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (NUM_HEADS, HEAD_DIM):
        raise ValueError(
            f"q must be [T, {NUM_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}"
        )
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"q must be bfloat16 or float8_e4m3fn, got {q.dtype}")
    form = form_for(q.dtype)
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if k.ndim != 2 or k.shape[1] != HEAD_DIM:
        raise ValueError(f"k must be [Tkv, {HEAD_DIM}], got {tuple(k.shape)}")
    if k.dtype != q.dtype:
        raise ValueError(f"k must be {_DTYPE_NAMES[form]}, got {k.dtype}")
    if (
        k.stride(1) != 1
        or k.stride(0) < HEAD_DIM
        or (k.stride(0) * k.element_size()) % TMA_ALIGNMENT_BYTES
    ):
        raise ValueError(
            f"k needs a unit inner stride and a row stride >= {HEAD_DIM} elements that is a multiple of "
            f"{TMA_ALIGNMENT_BYTES // k.element_size()} elements, got strides {tuple(k.stride())}"
        )
    if form == BF16 and k_scale is not None:
        raise ValueError(
            "k_scale is the per-key scale of float8_e4m3fn keys; bfloat16 q / k take none"
        )
    if form == FP8:
        if k_scale is None:
            raise ValueError("float8_e4m3fn q / k need k_scale (float32 [Tkv])")
        if tuple(k_scale.shape) != (int(k.shape[0]),):
            raise ValueError(
                f"k_scale must be [Tkv = {int(k.shape[0])}], got {tuple(k_scale.shape)}"
            )
        if k_scale.dtype != torch.float32:
            raise ValueError(f"k_scale must be float32, got {k_scale.dtype}")
        if not k_scale.is_contiguous():
            raise ValueError("k_scale must be contiguous")
    T = int(q.shape[0])
    if tuple(w.shape) != (T, NUM_HEADS):
        raise ValueError(f"w must be [T = {T}, {NUM_HEADS}], got {tuple(w.shape)}")
    if w.dtype != torch.float32:
        raise ValueError(f"w must be float32, got {w.dtype}")
    if not w.is_contiguous():
        raise ValueError("w must be contiguous")
    for name, t in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if t.ndim != 1 or t.numel() < 1:
            raise ValueError(f"{name} must be a one-dimensional [S + 1] tensor")
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("cu_seqlens_q and cu_seqlens_k must both have S + 1 entries")
    S = int(cu_seqlens_q.numel()) - 1
    if q_causal_offsets is not None:
        if q_causal_offsets.ndim != 1 or q_causal_offsets.numel() != S:
            raise ValueError(
                f"q_causal_offsets must be [S = {S}], got {tuple(q_causal_offsets.shape)}"
            )
        if q_causal_offsets.dtype != torch.int64:
            raise ValueError(
                f"q_causal_offsets must be int64, got {q_causal_offsets.dtype}"
            )
        if not q_causal_offsets.is_contiguous():
            raise ValueError("q_causal_offsets must be contiguous")
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    _check_int("ratio", ratio, 1)
    if max_seqlen_q is not None:
        _check_int("max_seqlen_q", max_seqlen_q, 0)
    if (
        max_seqlen_k is not None
    ):  # a host bound on every key segment: validated before the device rule, read by the dispatch only
        _check_max_seqlen_k(max_seqlen_k, int(k.shape[0]), S)
    if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)):
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    if not math.isfinite(float(softmax_scale)) or float(softmax_scale) <= 0.0:
        raise ValueError(
            f"softmax_scale must be a positive finite number, got {softmax_scale!r}"
        )
    device = q.device
    if device.type != "cuda":
        raise ValueError("inputs must be CUDA tensors")
    tensors = (
        [k, w, cu_seqlens_q, cu_seqlens_k]
        + ([q_causal_offsets] if q_causal_offsets is not None else [])
        + ([k_scale] if k_scale is not None else [])
    )
    if any(t.device != device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if (
        (q.data_ptr() % TMA_ALIGNMENT_BYTES)
        or (k.data_ptr() % TMA_ALIGNMENT_BYTES)
        or (w.data_ptr() % TMA_ALIGNMENT_BYTES)
    ):
        raise ValueError(
            f"q, k and w must be {TMA_ALIGNMENT_BYTES}-byte aligned (TMA sources)"
        )
    return T, int(k.shape[0]), S


def _check_cb_call(
    top_k: int, ratio: int, q: torch.Tensor, k_scale: Optional[torch.Tensor]
) -> None:
    """The candidate-block call's own rules beside the family's: blocks of 8 keys (``ratio == 1``), ``1 <= top_k <=
    CB_MAX_TOP_K`` and the BF16 form (no ``k_scale``)."""
    _check_int("top_k", top_k, 1, CB_MAX_TOP_K)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio != 1:
        raise ValueError(
            f"candidate blocks need ratio == 1 (blocks of 8 keys), got {ratio!r}"
        )
    if (
        not isinstance(q, torch.Tensor)
        or q.dtype != torch.bfloat16
        or k_scale is not None
    ):
        raise ValueError("candidate blocks take bfloat16 q / k (no k_scale)")


def _check_output(
    t: Optional[torch.Tensor], name: str, shape: tuple, dtype, device
) -> None:
    if t is None:
        return
    if tuple(t.shape) != shape or t.dtype != dtype:
        raise ValueError(
            f"{name} must be a {dtype} tensor of shape {shape}, got {tuple(t.shape)} {t.dtype}"
        )
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if t.device != device:
        raise ValueError(f"{name} must be on {device}")


# ---------------------------------------------------------------------------
# Program selection and workspace bound
# ---------------------------------------------------------------------------

_DEVICE_FACTS: dict[int, tuple[tuple[int, int], int]] = {}


def _outputs_aligned(tensors: Iterable[Optional[torch.Tensor]], align: int) -> bool:
    """True when every given output tensor's address is a multiple of ``align`` bytes (None = allocated here, aligned by the
    caching allocator): the staged rank finalize's ``cp.async.bulk`` row I/O (lever FRB) needs ``rank_bulk_align_bytes``."""
    align = int(align)
    return all(t is None or int(t.data_ptr()) % align == 0 for t in tensors)


def _device_index(device: Optional[torch.device]) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    device = torch.device(device)
    return int(
        device.index if device.index is not None else torch.cuda.current_device()
    )


def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    """``((major, minor), multi_processor_count)`` of ``cuda:index``: immutable for the life of the process, so queried
    once per device index; the index itself (``None`` / an index-less device = torch's current device) is resolved on
    every call by ``_device_index``."""
    facts = _DEVICE_FACTS.get(index)
    if facts is None:
        properties = torch.cuda.get_device_properties(index)
        facts = _DEVICE_FACTS[index] = (
            (int(properties.major), int(properties.minor)),
            int(properties.multi_processor_count),
        )
    return facts


def _num_sms(device: Optional[torch.device]) -> int:
    return _device_facts(_device_index(device))[1]


def _dispatch_inputs(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DispatchPolicy, str, str, dict]:
    """The checked dispatch inputs of a call: its policy, architecture, input form and the keywords of
    ``cake_policy.select_program`` (no tensor is read)."""
    _check_int("num_queries", num_queries, 0)
    _check_int("num_keys", num_keys, 0)
    _check_int("num_segments", num_segments, 0)
    _check_int("top_k", top_k, 1, MAX_TOP_K)
    _check_int("ratio", ratio, 1)
    form = form_for(dtype)
    if arch is None:
        arch = arch_for(device)
        if arch is None:
            raise ValueError(
                "DSA indexer top-k requires compute capability 10.0, 10.3 or 10.7"
            )
    policy = policy_for(arch, form)
    if grid_ctas is not None:
        grid = _check_int("grid_ctas", grid_ctas, 1)
    elif num_sms is not None:
        grid = _check_int("num_sms", num_sms, 1)
    else:
        grid = _num_sms(device)
    if candidate_multiplier is not None:
        _check_int("candidate_multiplier", candidate_multiplier, 1)
    if sample_tiles_max is not None:
        _check_int("sample_tiles_max", sample_tiles_max, 0, 256)
    if sample_shift_permille is not None:
        _check_int(
            "sample_shift_permille",
            sample_shift_permille,
            *SAMPLE_SHIFT_PERMILLE_BOUNDS,
        )
    if far_sample is not None:
        _check_int("far_sample", far_sample, 0, 1)
    if check_period is not None:
        _check_int("check_period", check_period, 1)
    if finalize_threads is not None:
        _check_int("finalize_threads", finalize_threads, 1)
    if max_seqlen_k is not None:
        _check_max_seqlen_k(max_seqlen_k, int(num_keys), int(num_segments))
    call: dict[str, Any] = dict(
        num_queries=int(num_queries),
        num_keys=int(num_keys),
        num_segments=int(num_segments),
        ratio=int(ratio),
        top_k=int(top_k),
        grid=grid,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        max_seqlen_k=max_seqlen_k,
        outputs_aligned=bool(outputs_aligned),
        scan_keys=PROGRAM_KEYS.get(arch, {}),
    )
    return policy, arch, form, call


def plan_dsa_indexer_det_topk(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> ProgramChoice:
    """The host dispatch of a call from host-known integers (no tensor is read).

    ``arch`` defaults to the architecture of ``device`` (the current device) and
    ``num_sms`` to its SM count; both may be given explicitly for host-only use.
    The persistent grid is ``grid_ctas`` when given, else the SM count.
    ``outputs_aligned`` (default True: outputs the backend allocates) says whether
    the ``indices`` / ``scores`` addresses are multiples of the policy's
    ``rank_bulk_align_bytes``; the staged rank finalize's bulk row I/O (lever FRB)
    is dispatched only then, else its plain twin program (same results).
    ``dtype``: the q / k dtype (``torch.float8_e4m3fn``: the FP8 form, whose
    programs ``cake_policy.select_fp8_program`` picks under the architecture's
    FP8 record; ``NotImplementedError`` where none is registered).
    """
    policy, arch, form, call = _dispatch_inputs(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
        arch=arch,
        grid_ctas=grid_ctas,
        num_sms=num_sms,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        max_seqlen_k=max_seqlen_k,
        outputs_aligned=outputs_aligned,
        dtype=dtype,
    )
    if form == FP8:
        return select_fp8_program(policy, fp8_programs=fp8_programs_for(arch), **call)
    return select_program(policy, **call)


def plan_dsa_indexer_det_topk_with_candidates(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
) -> CandidateBlocksChoice:
    """The host dispatch of a call that also returns candidate blocks (BF16, ``ratio == 1``, ``1 <= top_k <= 2047``):
    the plan of :func:`plan_dsa_indexer_det_topk` run by the ``scan_blocks`` program of its unit
    (``cake_policy.select_cb_program``), with the block entries of ``cake_policy.cb_block_cap(num_keys,
    max_seqlen_k)``; where that is 0 (every row keeps all its blocks) the plan of :func:`plan_dsa_indexer_det_topk`
    itself with ``iota`` set (the backend writes the rows before its launches).  Raises ``NotImplementedError``
    where the architecture registers no ``scan_blocks`` program."""
    _check_int("top_k", top_k, 1, CB_MAX_TOP_K)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio != 1:
        raise ValueError(
            f"candidate blocks need ratio == 1 (blocks of 8 keys), got {ratio!r}"
        )
    policy, _arch, _form, call = _dispatch_inputs(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
        arch=arch,
        grid_ctas=grid_ctas,
        num_sms=num_sms,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        max_seqlen_k=max_seqlen_k,
        outputs_aligned=outputs_aligned,
        dtype=torch.bfloat16,
    )
    return select_cb_program(policy, **call)


def cake_dsa_indexer_det_train_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Explicit workspace bound in bytes of one call's geometry on ``device``.

    ``grid_ctas x queries_per_unit x candidate_capacity(top_k) x entry_bytes``
    for the dispatched unit geometry and capacity multiplier (``grid_ctas`` =
    the SM count, made even for the CTA-pair program), plus the split staging
    ``T x n_split x top_k x entry_bytes`` when the dispatch splits the key
    ranges.  The dispatch, and with it the bound, depends on ``T``, ``Tkv``,
    ``S``, ``ratio`` and ``top_k``; prepare a call's buffer for its own geometry
    (or for the largest bound over the geometries it will serve).  The bound
    covers both sampling modes: where the record's ``l6_solo_rule``,
    ``l6_solo_unsampled_max_queries``, ``split_wave_scope`` or ``wide_pair_snake_only`` gives a call with the
    sampled first threshold off (``sample_tiles_max=0``) another unit
    geometry, pair or split than the sampled default, the larger of the two
    plans.  ``dtype`` is the q / k dtype: the FP8 form
    (``torch.float8_e4m3fn``) dispatches its own programs, so its bound may
    differ from the BF16 one of the same geometry.
    """
    return max(
        int(
            plan_dsa_indexer_det_topk(
                num_queries,
                num_keys,
                num_segments,
                top_k=top_k,
                ratio=ratio,
                device=device,
                arch=arch,
                grid_ctas=grid_ctas,
                num_sms=num_sms,
                candidate_multiplier=candidate_multiplier,
                sample_tiles_max=sample_tiles_max,
                dtype=dtype,
            ).workspace_bytes
        )
        for sample_tiles_max in (None, 0)
    )


def cake_dsa_indexer_det_train_topk_with_candidates_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    ratio: int = 1,
    max_seqlen_k: Optional[int] = None,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
) -> int:
    """Explicit workspace bound in bytes of one candidate-block call's geometry on ``device``: the ``scan_blocks``
    program's candidate buffers and ticket tail plus its block entries ``grid x block_q x blk_cap x 8``
    (:func:`plan_dsa_indexer_det_topk_with_candidates`), or, where every row keeps all its blocks (``blk_cap`` 0),
    the workspace of the plain call's plan; the larger of the two sampling modes.  The ``[T, 2048]``
    int32 ``candidate_blocks`` output is not part of it.  Pass the call's ``max_seqlen_k``: without it the block
    entries are sized for ``Tkv``."""
    return max(
        int(
            plan_dsa_indexer_det_topk_with_candidates(
                num_queries,
                num_keys,
                num_segments,
                top_k=top_k,
                ratio=ratio,
                device=device,
                arch=arch,
                grid_ctas=grid_ctas,
                num_sms=num_sms,
                candidate_multiplier=candidate_multiplier,
                sample_tiles_max=sample_tiles_max,
                max_seqlen_k=max_seqlen_k,
            ).workspace_bytes
        )
        for sample_tiles_max in (None, 0)
    )


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    role: str
    program: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


def bind_program(
    program: str,
    arch: str,
    role: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``role`` and load ``program`` for ``arch``.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.
    """
    if role not in STAGES:
        raise ValueError(f"role must be one of {STAGES}, got {role!r}")
    if PROGRAMS[program]["role"] != role:
        raise ValueError(
            f"program {program!r} is a {PROGRAMS[program]['role']} program, not {role!r}"
        )
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in ARG_PLANS[role]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif key in values and values[key] is not None:
            arguments.append(values[key])
        else:
            raise KeyError(
                f"generated program {program!r} ({role}) expects argument {name!r} ({kind}); the host binding "
                f"provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    entry = getattr(load_program(program, arch), FFI_ENTRY)
    return _Launch(role, program, entry, tuple(arguments), grid)


_FFI_DEVICES: dict[int, Any] = {}
_EMPTY_REGIONS: dict[int, Any] = {}
_TICKET_PLACEHOLDERS: dict[int, Any] = {}
_QUIET_SCAN_ARGS: dict[int, dict[str, Any]] = {}
_UNREAD_PLACEHOLDERS: dict[tuple[int, torch.dtype], torch.Tensor] = {}
# the ticket block a ``TicketWord`` argument points at: the ticket word (the low u32 of its first 8 bytes) and padding
TICKET_BLOCK_BYTES = 16
# the bytes a binding's ``zero_region`` zeroes to restore the ticket word of a block
TICKET_WORD_ZERO_BYTES = 8


def _empty_region(index: int) -> Any:
    """The scan's empty ``zero_region`` on device ``index`` (its binding zeroes nothing): one ``tvm_ffi.Tensor``
    per device, made on first use, so a call that zeroes no tail neither makes a view nor converts one per call."""
    region = _EMPTY_REGIONS.get(index)
    if region is None:
        region = _EMPTY_REGIONS[index] = tvm_ffi.from_dlpack(
            torch.empty(0, dtype=torch.uint8, device=f"cuda:{index}")
        )
    return region


def _ticket_placeholder(index: int) -> Any:
    """The ``TicketWord`` of a scan launch that never reads it (a program drawing from the workspace's tail, a
    ``static_scan_kinds`` program, a static call): one zeroed 16-byte block per device, made on first use and never
    written (a program that reads its ``TicketWord`` never gets it)."""
    block = _TICKET_PLACEHOLDERS.get(index)
    if block is None:
        block = _TICKET_PLACEHOLDERS[index] = tvm_ffi.from_dlpack(
            torch.zeros(
                TICKET_BLOCK_BYTES, dtype=torch.uint8, device=f"cuda:{index}"
            ).view(torch.uint32)
        )
    return block


def _quiet_scan_args(index: int) -> dict[str, Any]:
    """The ticket arguments of a scan launch that zeroes nothing and draws from no block of its own (a static call, a
    ``static_scan_kinds`` program, a program drawing from a tail it does not share): the device's empty
    ``zero_region`` and ``TicketWord`` placeholder, one dict per device, so such a call adds them with one lookup."""
    args = _QUIET_SCAN_ARGS.get(index)
    if args is None:
        args = _QUIET_SCAN_ARGS[index] = {
            "zero_region": _empty_region(index),
            "TicketWord": _ticket_placeholder(index),
        }
    return args


def _unread_placeholder(index: int, dtype: torch.dtype) -> torch.Tensor:
    """A one-element ``dtype`` tensor on device ``index`` for an argument no launched program dereferences: one per
    device and dtype, allocated without a device operation and never written, so such a call launches nothing before
    its scan.  Made during a CUDA graph capture, it is the call's own (the capture's pool) and not kept."""
    holder = _UNREAD_PLACEHOLDERS.get((index, dtype))
    if holder is None:
        holder = torch.empty(1, dtype=dtype, device=f"cuda:{index}")
        if not torch.cuda.is_current_stream_capturing():
            _UNREAD_PLACEHOLDERS[(index, dtype)] = holder
    return holder


def _offsets_placeholder(index: int) -> torch.Tensor:
    """The ``q_offsets`` of a call without causal offsets (``has_offsets = 0``: no program dereferences it)."""
    return _unread_placeholder(index, torch.int64)


# row ``nb`` (``0 .. CB_SLOTS``) of the int32 ``[CB_SLOTS + 1, CB_SLOTS]`` table: ``0 .. nb - 1`` and a ``-1`` tail, per
# device, kept for the life of the process, with the event its creating stream recorded right after writing it
_IOTA_ROWS: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
# the row base and limit of the rows outside every query segment (before ``cu_seqlens_q[0]`` or from
# ``cu_seqlens_q[S]`` on): ``min(base + t + 8, limit)`` clamps to 7, so the row's visible count is 0
_IOTA_NO_SEGMENT = -(1 << 62)


def _capturing(index: int) -> bool:
    """True while torch's current stream on device ``index`` (not the current device's) captures a CUDA graph."""
    with torch.cuda.device(index):
        return torch.cuda.is_current_stream_capturing()


def _iota_rows(index: int) -> tuple[torch.Tensor, Optional[torch.cuda.Event]]:
    """The candidate-block rows of every block count on device ``index``: row ``nb`` of the int32 ``[CB_SLOTS + 1,
    CB_SLOTS]`` table holds ``0 .. nb - 1`` and a ``-1`` tail.  One table per device (16 MiB), made on first use on
    device ``index``'s current stream and kept for the life of the process, so an eager call never makes it again and no
    CUDA graph or stream that reads it can outlive it; an event recorded on that stream right after it lets a reader on
    another stream wait for it (:class:`_IotaBlocks`) before it reads the table.  Made during a CUDA graph capture, it is
    the call's own (the capture's pool; the capture orders its reads after its writes), not kept, and comes without an
    event."""
    entry = _IOTA_ROWS.get(index)
    if entry is not None:
        return entry
    with torch.cuda.device(index):
        device = f"cuda:{index}"
        slots = torch.arange(CB_SLOTS, dtype=torch.int32, device=device)
        counts = torch.arange(CB_SLOTS + 1, dtype=torch.int32, device=device)
        rows = torch.where(slots[None, :] < counts[:, None], slots[None, :], PAD_ID)
        if torch.cuda.is_current_stream_capturing():
            return rows, None
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(index))
        _IOTA_ROWS[index] = (rows, ready)
        return rows, ready


class _IotaBlocks:
    """The ``candidate_blocks`` rows of a call whose rows keep all their blocks (``CandidateBlocksChoice.iota``), on
    the device before the plain call's launches: row ``t`` holds ``0 .. nb - 1`` and a ``-1`` tail, ``nb = ceil(n /
    8)`` for its ``n`` visible keys (:func:`visible_key_count` at ``ratio == 1``, the rule of the ``scan_blocks``
    programs), all ``-1`` for a row outside every query segment (before ``cu_seqlens_q[0]`` or from
    ``cu_seqlens_q[S]`` on).  Every run reads the bound ``cu_seqlens_q`` / ``cu_seqlens_k`` / ``q_causal_offsets``
    (new values written into them are followed, also by a CUDA graph capturing the run), with no host synchronization
    and no allocation: the per-segment and per-row scratch is made here, by three small device launches (one ``full``,
    two ``arange``) on every prepare -- so on every call of the eager entry, which prepares per call.  Per segment
    ``s`` (``q0 / q1`` its query bounds, ``L`` its keys) ``base`` and ``lim`` give ``n + 7 = max(min(t + 8 + base,
    lim), 7)``: with offsets ``base = offset - q0`` and ``lim = L + 7``; without them ``base = L - q1`` (the default
    offset ``L - (q1 - q0)``), and no row of the segment reaches ``lim``, so it is skipped.  ``searchsorted`` over
    ``cu_seqlens_q`` maps row ``t`` to entry ``1 + s`` of ``base`` / ``lim`` for its segment ``s``, to entry 0 before
    the first segment and to entry ``S + 1`` after the last, the sentinels whose base and limit clamp the count to 0;
    ``nb`` indexes the rows of :func:`_iota_rows`.  A row with ``nb > CB_SLOTS`` (a ``max_seqlen_k`` below one of its
    key segments) indexes past the table and stops with ``index_select``'s device-side assert, where the
    ``scan_blocks`` programs trap; the runner launches this fill first, so such a call stops before its scan and
    finalize programs run.  The table may come from another stream's first call: the prepare and every run outside a
    capture make the current stream wait on the table's event while it has not completed at prepare."""

    def __init__(
        self,
        blocks: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        q_causal_offsets: Optional[torch.Tensor],
        num_segments: int,
        device_index: int,
    ):
        T, S = int(blocks.shape[0]), int(num_segments)
        device = blocks.device
        self.blocks = blocks
        self.num_segments = S
        self.offsets = q_causal_offsets
        self.cu_q = cu_seqlens_q
        self.q_lo, self.q_hi = cu_seqlens_q[:-1], cu_seqlens_q[1:]
        self.k_lo, self.k_hi = cu_seqlens_k[:-1], cu_seqlens_k[1:]
        self.index = device_index
        self.rows, ready = _iota_rows(device_index)
        # a table another stream is still writing: wait for it here and on every run (a run may use another stream)
        self.ready = None
        if ready is not None and not _capturing(device_index) and not ready.query():
            torch.cuda.current_stream(device).wait_event(ready)
            self.ready = ready
        # entries 0 and S + 1: the sentinels of the rows outside every segment; 1 .. S: the segments (each run)
        bounds = torch.full(
            (2, S + 2), _IOTA_NO_SEGMENT, dtype=torch.int64, device=device
        )
        self.base, self.lim = bounds[0], bounds[1]
        self.row_ids = torch.arange(T, dtype=cu_seqlens_q.dtype, device=device)
        self.row_c = torch.arange(8, T + 8, dtype=torch.int64, device=device)
        self.segment = torch.empty(T, dtype=torch.int64, device=device)
        self.count = torch.empty(T, dtype=torch.int64, device=device)
        self.row_lim = (
            None
            if q_causal_offsets is None
            else torch.empty(T, dtype=torch.int64, device=device)
        )

    def __call__(self) -> None:
        S = self.num_segments
        if self.ready is not None and not _capturing(self.index):
            torch.cuda.current_stream(self.index).wait_event(self.ready)
        if S == 0:
            self.blocks.fill_(PAD_ID)
            return
        base = self.base[1 : S + 1]
        if self.offsets is None:
            torch.sub(self.k_hi, self.k_lo, out=base)
            base.sub_(self.q_hi)
        else:
            torch.sub(self.offsets, self.q_lo, out=base)
            lim = self.lim[1 : S + 1]
            torch.sub(self.k_hi, self.k_lo, out=lim)
            lim.add_(7)
        torch.searchsorted(self.cu_q, self.row_ids, right=True, out=self.segment)
        torch.index_select(self.base, 0, self.segment, out=self.count)
        self.count.add_(self.row_c)
        if self.offsets is not None:
            torch.index_select(self.lim, 0, self.segment, out=self.row_lim)
            torch.minimum(self.count, self.row_lim, out=self.count)
        self.count.clamp_(min=7)
        self.count.bitwise_right_shift_(3)
        torch.index_select(self.rows, 0, self.count, out=self.blocks)


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )
    return tvm_ffi.use_raw_stream(device, raw)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class DSAIndexerDetRunner:
    """The prepared launches of one tensor binding.

    ``__call__()`` runs the dispatched scan program, the split merge when the
    dispatch splits the key ranges, and the finalize program (the CUB block
    radix sort, or the prefix-popcount rank scatter where the policy's
    ``finalize_rank_for`` names a bitmap window; stage role ``finalize`` /
    ``finalize_rank``) on torch's current stream and returns ``(indices,
    scores)``; a candidate-block runner of the plain plan (``iota``) first
    writes the ``candidate_blocks`` rows (stage ``cb_iota``).  No launch allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride or tensor binding changes; values
    may change freely.
    """

    arch: str
    abi: str
    choice: ProgramChoice
    programs: dict[str, str]  # launched role -> registered program name
    tensors: dict[str, torch.Tensor] = field(repr=False)
    scalars: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    stages: tuple[str, ...]  # launched roles in order
    device_index: int = 0
    workspace: torch.Tensor = field(default=None, repr=False)
    workspace_bytes: int = 0
    grid: tuple[int, int, int] = (1, 1, 1)
    # the workspace's ticket tail on the ticket architectures (``TICKET_TAIL_ARCHES``;
    # None for a program of the record's ``static_scan_kinds``).  Invariant: the ticket word reads 0 when a scan launches.  Every scan
    # launch that completes leaves the word at 0 (the kernel's end-of-launch reset, which
    # kernel-tests check as the output ``ticket_word`` on every ticket-drawing config).
    # Backend-owned workspace (no ``workspace_buffer`` passed to prepare): nothing else writes
    # the tail, so it is zeroed once, when the runner is prepared; ``run()`` issues no memset
    # and a CUDA graph capturing it carries none.  Caller-owned workspace (``_tail_shared``):
    # the buffer may be written between two runs by another runner of a different geometry
    # (its candidate region covers this tail) or by any other kernel the caller lends it to,
    # so the scan launch passes the tail as its ``zero_region``, which the scan binding zeroes
    # with ``cudaMemsetAsync`` on the launch stream right before every scan launch, in the same
    # host call (no host work between the two submissions); a graph capturing it carries that
    # memset node.  Static calls (``_tail_static``): where the
    # shape alone bounds the work units by the launched grid (``ticket_static_path``), every
    # scan launch takes the static fast path, which gives each CTA its unit by index and never
    # reads or writes the ticket word, so neither prepare nor a run zeroes the tail (no
    # memset on either workspace kind; the scan's ``zero_region`` is empty).  A ``run()`` that raised while launching may have
    # left the word non-zero: it marks the tail dirty and the next ``run()`` zeroes it
    # first (static or not).  The view (``ticket_tail``) is made on first use: a static call
    # that never zeroes its tail makes none on the launch path.
    # A scan program that reads its ``TicketWord`` argument on this architecture (``reads_ticket_word``) has no
    # ticket tail: a drawing call gets a 16-byte block of the runner's own (``ticket_block``), which no other work
    # writes, zeroed once here; every completed scan launch leaves it 0, so neither kind of workspace adds a memset
    # to ``run()``.  After a run that raised while launching, the next ``run()`` launches the scan through a binding
    # call whose ``zero_region`` is the block's ticket word (8 bytes), on the run's stream.  The eager entry draws from
    # the first 16 bytes of the workspace's tail instead, which its scan binding zeroes (8 bytes) right before the
    # launch.  Any other scan launch (static, or a program drawing from the tail) gets the device's zeroed
    # placeholder, which it never reads.  The :lsf copy of a TicketWord-reading program (sm_107a's l6 ticket
    # programs) still keeps its longest-first segment order in the tail's bytes behind the ticket word, which every
    # launch writes before it reads them, so stale bytes there never matter and nothing zeroes them.
    # byte offset of the tail in the workspace; -1: no tail
    _tail_at: int = field(default=-1, repr=False)
    _tail_view: Optional[torch.Tensor] = field(default=None, repr=False)
    _tail_shared: bool = field(default=False, repr=False)
    _tail_static: bool = field(default=False, repr=False)
    _tail_dirty: bool = field(default=False, repr=False)
    # the bytes a TicketWord-reading drawing scan draws from (uint8); None: the call draws from no ticket block
    _ticket_block: Optional[torch.Tensor] = field(default=None, repr=False)
    # the scan launch that zeroes the ticket block's word first (a prepared TicketWord runner after a failed run)
    _ticket_restore: Optional[_Launch] = field(default=None, repr=False)
    # the index of the scan launch in ``launches`` (1 behind an iota candidate-block runner's ``cb_iota`` fill)
    _scan_at: int = field(default=0, repr=False)
    # the call also returns its candidate blocks (``*_with_candidates``): ``run()`` returns them third
    _with_candidates: bool = field(default=False, repr=False)

    @property
    def ticket_tail(self) -> Optional[torch.Tensor]:
        if self._tail_at < 0:
            return None
        if self._tail_view is None:
            self._tail_view = self.workspace[
                self._tail_at : self._tail_at + TICKET_TAIL_BYTES
            ]
        return self._tail_view

    @property
    def ticket_block(self) -> Optional[torch.Tensor]:
        """The 16 bytes the scan draws its tickets from where its program reads ``TicketWord`` and the call draws."""
        return self._ticket_block

    @property
    def indices(self) -> torch.Tensor:
        return self.tensors["Indices"]

    @property
    def scores(self) -> torch.Tensor:
        return self.tensors["Scores"]

    @property
    def candidate_blocks(self) -> Optional[torch.Tensor]:
        """The ``[T, 2048]`` int32 candidate-block rows of a ``*_with_candidates`` runner (None otherwise)."""
        return self.tensors.get("CandBlocks")

    @property
    def num_queries(self) -> int:
        return self.choice.num_queries

    @property
    def num_keys(self) -> int:
        return self.choice.num_keys

    @property
    def top_k(self) -> int:
        return self.choice.top_k

    @property
    def form(self) -> str:
        """The input form of the bound q / k (``"bf16"`` or ``"fp8"``), read from the dispatched scan key."""
        return scan_key_form(self.choice.scan_key)

    def run(self) -> tuple[torch.Tensor, ...]:
        with _ffi_stream_context(self.device_index):
            launches = self.launches
            if self._tail_dirty:
                # a previous run raised while launching: restore the ticket word before the scan
                if self._ticket_restore is not None:
                    # the scan binding zeroes the block's word on this stream right before the scan launch
                    at = self._scan_at
                    launches = (
                        *launches[:at],
                        self._ticket_restore,
                        *launches[at + 1 :],
                    )
                elif self._tail_at >= 0:
                    # (a caller-owned tail is zeroed by the scan binding anyway)
                    self.ticket_tail.zero_()
                self._tail_dirty = False
            try:
                for launch in launches:
                    launch()
            except BaseException:
                self._tail_dirty = (
                    self._tail_at >= 0 or self._ticket_restore is not None
                )
                raise
        if self._with_candidates:
            return (
                self.tensors["Indices"],
                self.tensors["Scores"],
                self.tensors["CandBlocks"],
            )
        return self.tensors["Indices"], self.tensors["Scores"]

    __call__ = run


def _iota_launch(
    t: dict,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_causal_offsets: Optional[torch.Tensor],
    num_segments: int,
    device_index: int,
) -> _Launch:
    """The ``cb_iota`` stage of an iota candidate-block runner: :class:`_IotaBlocks` into its ``CandBlocks``."""
    blocks = t["CandBlocks"]
    fill = _IotaBlocks(
        blocks, cu_seqlens_q, cu_seqlens_k, q_causal_offsets, num_segments, device_index
    )
    return _Launch(CB_IOTA_STAGE, "torch", fill, (), (int(blocks.shape[0]), 1, 1))


def _padded_outputs(
    num_queries: int,
    top_k: int,
    device: torch.device,
    indices: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outputs of a call without keys: all padding (two fills, no synchronization)."""
    if indices is None:
        indices = torch.empty((num_queries, top_k), dtype=torch.int32, device=device)
    if scores is None:
        scores = torch.empty((num_queries, top_k), dtype=torch.float32, device=device)
    indices.fill_(PAD_ID)
    scores.fill_(PAD_SCORE)
    return indices, scores


def _prepare_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
    one_shot: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    with_candidates: bool = False,
    candidate_blocks: Optional[torch.Tensor] = None,
) -> DSAIndexerDetRunner:
    """:func:`prepare_dsa_indexer_det_topk`; ``one_shot``: the runner runs once (the eager entry), so a scan program
    that reads its ``TicketWord`` draws from the first bytes of the workspace's tail, which its scan binding zeroes,
    instead of a block of its own."""
    if backend != "cake":
        raise ValueError("DSA indexer top-k supports backend='cake'")
    if with_candidates:
        # before the family's checks, which would ask an FP8 call for the k_scale this call does not take
        _check_cb_call(top_k, ratio, q, k_scale)
    fp8 = q.dtype == torch.float8_e4m3fn
    if softmax_scale is None:
        # the FP8 programs take 1.0: every scale but k_scale is folded into w
        softmax_scale = 1.0 if fp8 else default_softmax_scale()
    elif fp8:
        raise ValueError(
            "float8_e4m3fn q / k take softmax_scale=None: fold the softmax scale into w"
        )
    num_queries, num_keys, num_segments = validate_dsa_indexer_det_inputs(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        q_causal_offsets,
        top_k,
        softmax_scale,
        ratio,
        max_seqlen_q,
        max_seqlen_k,
        k_scale,
    )
    form = form_for(q.dtype)
    device = q.device
    for name, tensor in (
        ("workspace_buffer", workspace_buffer),
        ("indices", indices),
        ("scores", scores),
        ("candidate_blocks", candidate_blocks),
    ):
        if tensor is not None and (not tensor.is_cuda or tensor.device != device):
            raise ValueError(f"{name} must be on {device}")
    _check_output(indices, "indices", (num_queries, int(top_k)), torch.int32, device)
    _check_output(scores, "scores", (num_queries, int(top_k)), torch.float32, device)
    if with_candidates:
        _check_output(
            candidate_blocks,
            "candidate_blocks",
            (num_queries, CB_SLOTS),
            torch.int32,
            device,
        )
    elif candidate_blocks is not None:
        raise ValueError(
            "candidate_blocks is an output of the *_with_candidates call only"
        )
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "DSA indexer top-k requires compute capability 10.0, 10.3 or 10.7"
        )
    abi = registry_abi()
    device_index = _device_index(device)
    # lever FRB: the staged rank finalize's bulk row I/O needs output addresses on the policy's 16-B granule; caller-owned
    # outputs may be arbitrary views (the plain twin program is registered alongside), the backend's own allocations are aligned
    outputs_aligned = _outputs_aligned(
        (indices, scores), policy_for(arch, form).rank_bulk_align_bytes
    )
    # a candidate-block call runs the scan_blocks program of its unit (the block entries behind the ticket tail)
    cb_choice: Optional[CandidateBlocksChoice] = None
    if with_candidates:
        cb_choice = plan_dsa_indexer_det_topk_with_candidates(
            num_queries,
            num_keys,
            num_segments,
            top_k=top_k,
            ratio=ratio,
            device=device,
            arch=arch,
            grid_ctas=grid_ctas,
            candidate_multiplier=candidate_multiplier,
            check_period=check_period,
            sample_tiles_max=sample_tiles_max,
            sample_shift_permille=sample_shift_permille,
            far_sample=far_sample,
            finalize_threads=finalize_threads,
            max_seqlen_k=max_seqlen_k,
            outputs_aligned=outputs_aligned,
        )
        choice = cb_choice.choice
    else:
        choice = plan_dsa_indexer_det_topk(
            num_queries,
            num_keys,
            num_segments,
            top_k=top_k,
            ratio=ratio,
            device=device,
            arch=arch,
            grid_ctas=grid_ctas,
            candidate_multiplier=candidate_multiplier,
            check_period=check_period,
            sample_tiles_max=sample_tiles_max,
            sample_shift_permille=sample_shift_permille,
            far_sample=far_sample,
            finalize_threads=finalize_threads,
            max_seqlen_k=max_seqlen_k,
            outputs_aligned=outputs_aligned,
            dtype=q.dtype,
        )
    # an iota candidate-block call runs the plain call's programs; the backend writes its rows before them
    cb_iota = cb_choice is not None and cb_choice.iota
    scan_role = CB_ROLE if with_candidates and not cb_iota else "scan"
    # every program of the call must be registered before anything is allocated (fail closed by name)
    # a merge that finalizes its rows itself (``choice.merge_rank``, key merge:rank:w*) leaves no finalize program to bind
    fin_role = choice.finalize_role
    programs = {scan_role: program_for(arch, choice.scan_key)}
    if not choice.merge_rank:
        programs[fin_role] = program_for(arch, choice.finalize_key)
    if choice.merge_key is not None:
        programs["merge"] = program_for(arch, choice.merge_key)
    workspace_bytes = int(
        choice.workspace_bytes if cb_choice is None else cb_choice.workspace_bytes
    )
    # the ticket tail ends the scan's own workspace; the block entries of a candidate-block call follow it
    tail_offset = int(choice.workspace_bytes) - TICKET_TAIL_BYTES
    # the ticket-tail decision (see the comment where the tail is passed) is host arithmetic only: it is made here,
    # before the call's first device operation, so an eager call does not hold the stream for it
    tail_at, tail_static, drawing, reads_word = -1, False, False, False
    tail_skips = ticket_tail_skips(arch, form)
    if (
        num_queries > 0
        and num_keys > 0
        and tail_skips is not None
        and choice.physical_kind not in tail_skips
    ):
        tail_static = ticket_static_path(choice)
        drawing = not tail_static
        # the scan program draws from its TicketWord argument instead of the tail (registry ``ticket_arg_arches``)
        reads_word = reads_ticket_word(programs[scan_role], arch)
        if not reads_word:
            tail_at = tail_offset
    caller_owned_workspace = workspace_buffer is not None
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < workspace_bytes:
        raise ValueError(
            f"workspace_buffer needs {workspace_bytes} bytes, got {flat.numel()}"
        )
    if flat.data_ptr() % 8:
        raise ValueError("workspace_buffer must be 8-byte aligned")
    cand = flat[: int(choice.workspace_bytes)].view(torch.int64)

    t: dict[str, torch.Tensor] = {
        # the FP8 form's e4m3 bytes reach the binding as uint8 views of the same storage
        "Q": (q.view(torch.uint8) if fp8 else q).view(
            num_queries * NUM_HEADS, HEAD_DIM
        ),
        "K": k.view(torch.uint8) if fp8 else k,
        "W": w,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        # the device's never-dereferenced placeholder when no offsets are supplied (has_offsets = 0): no fill per call
        "q_offsets": q_causal_offsets
        if q_causal_offsets is not None
        else _offsets_placeholder(device_index),
        # only the FP8 programs read the per-key scales; a BF16 program gets the device's never-dereferenced placeholder
        "K_scale": k_scale if fp8 else _unread_placeholder(device_index, torch.float32),
        "Indices": indices
        if indices is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.int32, device=device),
        "Scores": scores
        if scores is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.float32, device=device),
        "Cand": cand,
        # the split staging [T, n_split, top_k] lives behind the persistent candidate buffers (merge stage only)
        "Staging": cand[
            choice.scan_entries : choice.scan_entries + choice.staging_entries
        ]
        if choice.split
        else None,
    }
    if cb_choice is not None:
        # scan_blocks: the block entries behind the ticket tail (a never-read placeholder when no row stores any) and
        # the caller's (or a new) [T, 2048] rows; an iota call binds no scan_blocks program (no block entries)
        if not cb_iota:
            t["CandBlk"] = (
                flat[cb_choice.blocks_at : workspace_bytes].view(torch.int64)
                if cb_choice.blk_bytes
                else _unread_placeholder(device_index, torch.int64)
            )
        t["CandBlocks"] = (
            candidate_blocks
            if candidate_blocks is not None
            else torch.empty((num_queries, CB_SLOTS), dtype=torch.int32, device=device)
        )
    scalars: dict[str, Any] = dict(
        num_segments=num_segments,
        top_k=int(top_k),
        ratio=int(ratio),
        has_offsets=int(q_causal_offsets is not None),
        cand_cap=int(choice.cand_cap),
        first_cap=int(
            choice.first_cap
        ),  # regular trigger only: the first selection fires at the capacity rule
        sample_tiles_max=int(choice.sample_tiles_max),
        sample_shift_permille=int(choice.sample_shift_permille),
        far_sample=int(choice.far_sample),
        check_period=int(choice.check_period),
        grid_ctas=int(choice.grid_ctas),
        softmax_scale=float(softmax_scale),
        n_split=int(choice.n_split),
        key_bits=int(choice.key_bits),
        num_queries=num_queries,
        num_keys=num_keys,
    )
    if cb_choice is not None:
        scalars["blk_cap"] = int(cb_choice.blk_cap)
    grid = (int(choice.grid_ctas), 1, 1)
    row_grid = (max(1, num_queries), 1, 1)
    launches: list[_Launch] = []
    stages: list[str] = []
    # the ticket tail the scan binding zeroes right before every scan launch (``zero_region``): a caller-owned tail
    # the call draws tickets from; any other call passes the device's empty region (nothing zeroed, no per-call view)
    shared_tail = caller_owned_workspace and tail_at >= 0 and not tail_static
    # the scan's TicketWord: a program that reads it and draws gets a block no other work writes -- the runner's own
    # (zeroed here, once) or, for the eager entry's one launch, the first bytes of the workspace's tail, whose ticket
    # word the scan binding zeroes right before the launch; any other launch gets the device's placeholder.  A call
    # that neither shares its tail nor draws from a block (every static call) passes the device's quiet pair as is.
    ticket_block: Optional[torch.Tensor] = None
    scan_ticket = _quiet_scan_args(device_index)
    if shared_tail:
        scan_ticket = {
            "zero_region": flat[tail_at : tail_at + TICKET_TAIL_BYTES],
            "TicketWord": scan_ticket["TicketWord"],
        }
    elif reads_word and drawing:
        zero_region = scan_ticket["zero_region"]
        if one_shot:
            block_at = tail_offset
            ticket_block = flat[block_at : block_at + TICKET_BLOCK_BYTES]
            zero_region = ticket_block[:TICKET_WORD_ZERO_BYTES]
        else:
            ticket_block = torch.zeros(
                TICKET_BLOCK_BYTES, dtype=torch.uint8, device=device
            )
        ticket_word = ticket_block.view(torch.uint32)
        if ticket_word is _ticket_placeholder(device_index):
            raise RuntimeError(
                f"{programs[scan_role]} ({arch}) draws from its TicketWord argument but was given the shared placeholder"
            )
        scan_ticket = {"zero_region": zero_region, "TicketWord": ticket_word}
    ticket_restore: Optional[_Launch] = None
    scan_at = 0
    if num_queries > 0 and num_keys > 0:
        values: dict[str, Any] = dict(t)
        values.update(scalars)
        values.update(scan_ticket)
        if cb_iota:
            # first: a row past the table (a max_seqlen_k below a key segment) stops the call before its programs run,
            # as the scan_blocks programs' trap does
            launches.append(
                _iota_launch(
                    t,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    q_causal_offsets,
                    num_segments,
                    device_index,
                )
            )
            stages.append(CB_IOTA_STAGE)
        scan_at = len(launches)
        launches.append(
            bind_program(programs[scan_role], arch, scan_role, values, grid)
        )
        stages.append(scan_role)
        if ticket_block is not None and not one_shot:
            # the scan launch of the run after a failed one: its binding zeroes the block's ticket word first
            ticket_restore = bind_program(
                programs[scan_role],
                arch,
                scan_role,
                {**values, "zero_region": ticket_block[:TICKET_WORD_ZERO_BYTES]},
                grid,
            )
        if "merge" in programs:
            launches.append(
                bind_program(programs["merge"], arch, "merge", values, row_grid)
            )
            stages.append("merge")
        # lever FRP-K: the persistent rank finalize launches min(rows, CTAs per SM x (SMs - spare SMs)) CTAs, each walking rows bid, bid + grid, ...
        fin_grid = (finalize_grid(choice, _num_sms(device)), 1, 1)
        if fin_role in programs:
            launches.append(
                bind_program(programs[fin_role], arch, fin_role, values, fin_grid)
            )
            stages.append(fin_role)
    elif num_queries > 0:
        _padded_outputs(num_queries, int(top_k), device, t["Indices"], t["Scores"])
        if cb_choice is not None:
            # no keys: no row sees a block
            t["CandBlocks"].fill_(PAD_ID)
    # on the ticket architectures the scan program draws every work unit from the workspace's
    # ticket tail (behind the candidate buffers and the staging); every completed scan launch
    # resets the word to 0.  Backend-owned workspace: zeroed here once, the only memset, a run
    # adds none.  Caller-owned ``workspace_buffer``: the buffer may be written between runs by
    # others (another runner's candidate region, any kernel it is lent to), so nothing is zeroed
    # here; the scan launch passes the tail as its ``zero_region`` and the scan binding zeroes it
    # right before every scan launch (see ``DSAIndexerDetRunner.ticket_tail``).
    # A static call (``ticket_static_path``: the shape bounds the work units by the launched
    # grid, so every scan launch takes the static fast path and never touches the word) is
    # zeroed neither here nor per run.  A program of the ``static_scan_kinds`` of its form's record
    # (``policy_for(arch, form)``) keeps the static enumeration on every call: no tail view, nothing zeroed.
    # A program that reads its TicketWord argument (``reads_ticket_word``) has no tail here: it draws from its
    # ticket block (above), so neither a caller-owned nor a backend-owned workspace is zeroed for it (its :lsf copy
    # writes the segment order behind the tail's ticket word before it reads it).
    # (``tail_at`` / ``tail_static`` were decided above, before the first device operation; a call launches
    # iff it has queries and keys.)
    tail_view = scan_ticket["zero_region"] if shared_tail else None
    if tail_at >= 0 and not tail_static and not caller_owned_workspace:
        tail_view = flat[tail_at : tail_at + TICKET_TAIL_BYTES]
        tail_view.zero_()
    return DSAIndexerDetRunner(
        arch=arch,
        abi=abi,
        choice=choice,
        programs=programs,
        tensors=t,
        scalars=scalars,
        launches=tuple(launches),
        stages=tuple(stages),
        device_index=device_index,
        workspace=flat,
        workspace_bytes=workspace_bytes,
        grid=grid,
        _tail_at=tail_at,
        _tail_view=tail_view,
        _tail_shared=caller_owned_workspace and tail_at >= 0,
        _tail_static=tail_static,
        _ticket_block=ticket_block,
        _ticket_restore=ticket_restore,
        _scan_at=scan_at,
        _with_candidates=with_candidates,
    )


def prepare_dsa_indexer_det_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
    k_scale: Optional[torch.Tensor] = None,
) -> DSAIndexerDetRunner:
    """Validate one binding, decide its programs and prepare its launches.

    Missing outputs and the workspace are allocated here (the only allocations
    of the backend).  Pass ``workspace_buffer`` of at least
    :func:`cake_dsa_indexer_det_train_topk_workspace_size` bytes for the call's geometry (same
    ``grid_ctas`` / ``candidate_multiplier``) to reuse storage across calls.
    A caller-owned buffer may be written by other runners or kernels between
    two runs of the returned runner, so where the scan program draws from the
    buffer's ticket tail the scan binding zeroes it before every run (one small
    memset on the current stream, captured into a CUDA graph with the
    launches); the backend-owned workspace is zeroed once, here.  A scan
    program that reads its ``TicketWord`` argument draws from a 16-byte block
    of the runner's own instead, zeroed once, here, on either workspace.  ``indices`` / ``scores`` may
    be caller-owned ``[T, top_k]`` int32 / float32 tensors.

    The keyword-only knobs ``grid_ctas`` (persistent CTA count; default: the SM
    count), ``candidate_multiplier`` (candidate buffer per query in multiples
    of ``top_k``), ``check_period`` (key tiles between selection-trigger
    checks), ``sample_tiles_max`` / ``sample_shift_permille`` (sampled first
    threshold; ``0`` disables sampling and sizes the candidate buffers by the
    record's unsampled capacity values; the shift defaults to the record's
    value for the call's ``top_k``, ``sample_shift_rule`` where the record sets
    one), ``far_sample`` (``1``: the sampled first threshold of the
    ascending-tile-order scan programs ranks the far sample tiles only,
    leaving out the sample tile nearest the diagonal; ``0``: every sample
    tile; the descending-order and wide programs ignore it) and
    ``finalize_threads`` (an admissible registered finalize program) change
    only the internal partition and scheduling of the work: results are
    bitwise identical for every admissible value.  ``T == 0`` returns a runner
    whose launches are empty; a call without keys (``Tkv == 0``) fills the
    outputs with padding instead of launching.

    FP8 form: float8_e4m3fn ``q`` / ``k`` with ``k_scale`` (FP32 ``[Tkv]``)
    and ``softmax_scale=None``; the runner binds ``k_scale`` with ``q`` / ``k``
    (new scale values written into it are read by the next run).  The
    dispatched programs are the architecture's FP8 programs
    (``plan_dsa_indexer_det_topk(..., dtype=torch.float8_e4m3fn)``);
    ``NotImplementedError`` where none is registered.
    """
    return _prepare_runner(
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
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        backend=backend,
        k_scale=k_scale,
    )


def cake_dsa_indexer_det_train_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
    k_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager entry point: prepare a binding and run it once; returns ``(indices, scores)``.

    See :func:`prepare_dsa_indexer_det_topk` for the arguments.  The call allocates
    the outputs (unless given) and the workspace (unless given) through the
    caching allocator and performs no host synchronization, so it may be
    captured into a CUDA graph like any allocation-free launch sequence.  The
    ticket word the scan draws from is zeroed once per call by the scan
    binding, right before the scan launch: the ticket tail of a caller-owned
    ``workspace_buffer``, or the first 8 bytes of the tail of either workspace
    where the scan program reads its ``TicketWord`` argument.  Nothing else is
    launched before the scan: a call without ``q_causal_offsets`` passes the
    device's offsets placeholder, which no program reads.
    """
    runner = _prepare_runner(
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
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        backend=backend,
        one_shot=True,
        k_scale=k_scale,
    )
    return runner.run()


def prepare_dsa_indexer_det_topk_with_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    candidate_blocks: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    backend: str = "cake",
) -> DSAIndexerDetRunner:
    """:func:`prepare_dsa_indexer_det_topk` for a call that also returns its candidate blocks: the runner's
    ``run()`` returns ``(indices, scores, candidate_blocks)``.  BF16 only, ``ratio == 1``, ``1 <= top_k <= 2047``
    (default 512); ``candidate_blocks`` may be a caller-owned ``[T, 2048]`` int32 contiguous tensor on the device of
    ``q``.  A ``workspace_buffer`` holds at least
    :func:`cake_dsa_indexer_det_train_topk_with_candidates_workspace_size` bytes for the call's geometry and
    ``max_seqlen_k``, which must bound every key segment of the call (a row with more older blocks than the workspace
    holds stops the kernel with a device trap)."""
    return _prepare_runner(
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
        grid_ctas=grid_ctas,
        candidate_multiplier=candidate_multiplier,
        check_period=check_period,
        sample_tiles_max=sample_tiles_max,
        sample_shift_permille=sample_shift_permille,
        far_sample=far_sample,
        finalize_threads=finalize_threads,
        backend=backend,
        with_candidates=True,
        candidate_blocks=candidate_blocks,
    )


def cake_dsa_indexer_det_train_topk_with_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    candidate_blocks: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager entry point of the candidate-block call: prepare a binding and run it once; returns ``(indices, scores,
    candidate_blocks)``.  See :func:`prepare_dsa_indexer_det_topk_with_candidates` for the arguments.  No host
    synchronization; the outputs and the workspace are allocated through the caching allocator unless given."""
    runner = _prepare_runner(
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
        backend=backend,
        one_shot=True,
        with_candidates=True,
        candidate_blocks=candidate_blocks,
    )
    return runner.run()


# ---------------------------------------------------------------------------
# Candidate-restricted top-k (``*_topk_in_candidates``)
# ---------------------------------------------------------------------------


def _check_cand_form(top_k: int, ratio: int, q: torch.Tensor) -> None:
    """The candidate-restricted call's own rules checked before the family's: ``1 <= top_k <= CAND_MAX_TOP_K``,
    blocks of 8 keys (``ratio == 1``) and the BF16 form (the call takes no ``k_scale``)."""
    _check_int("top_k", top_k, 1, CAND_MAX_TOP_K)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio != 1:
        raise ValueError(
            f"the candidate-restricted top-k needs ratio == 1 (blocks of 8 keys), got {ratio!r}"
        )
    if not isinstance(q, torch.Tensor) or q.dtype != torch.bfloat16:
        raise ValueError("the candidate-restricted top-k takes bfloat16 q / k")


def _check_cand_call(
    candidate_blocks: Optional[torch.Tensor],
    num_queries: int,
    device: torch.device,
) -> int:
    """The candidate-restricted call's rules after the family's (which hold ``k`` to the family's rule: a unit inner
    stride and a row stride of at least 128 elements that is a multiple of 8, read in place):
    ``candidate_blocks`` an int32 ``[T, W]`` contiguous tensor on ``device`` with ``W <= CAND_WIDTH_MAX``; returns
    ``W``."""
    if not isinstance(candidate_blocks, torch.Tensor):
        raise ValueError("candidate_blocks must be an int32 [T, W] tensor")
    if (
        candidate_blocks.dtype != torch.int32
        or candidate_blocks.dim() != 2
        or candidate_blocks.shape[0] != num_queries
        or candidate_blocks.shape[1] > CAND_WIDTH_MAX
    ):
        raise ValueError(
            f"candidate_blocks must be an int32 [T, W] tensor with T = {num_queries} and W <= {CAND_WIDTH_MAX}, got "
            f"{tuple(candidate_blocks.shape)} {candidate_blocks.dtype}"
        )
    if not candidate_blocks.is_contiguous():
        raise ValueError("candidate_blocks must be contiguous")
    if not candidate_blocks.is_cuda or candidate_blocks.device != device:
        raise ValueError(f"candidate_blocks must be on {device}")
    return int(candidate_blocks.shape[1])


def plan_dsa_indexer_det_topk_in_candidates(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    ratio: int = 1,
    candidate_width: int = CAND_WIDTH_MAX,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
    cand_cap: Optional[int] = None,
    check_period: Optional[int] = None,
    finalize_threads: Optional[int] = None,
) -> CandidateTopKChoice:
    """The host plan of a candidate-restricted call from host-known integers (no tensor is read): the ``cand``
    program on ``grid_ctas`` (default: the SM count) persistent CTAs (``cake_policy.select_cand_program``).  BF16,
    ``ratio == 1``, ``1 <= top_k <= 2048``, ``0 <= candidate_width <= 2048``.  Raises ``NotImplementedError`` where
    the architecture registers no ``cand`` program."""
    _check_int("num_queries", num_queries, 0)
    _check_int("num_keys", num_keys, 0)
    _check_int("num_segments", num_segments, 0)
    _check_int("top_k", top_k, 1, CAND_MAX_TOP_K)
    _check_int("candidate_width", candidate_width, 0, CAND_WIDTH_MAX)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio != 1:
        raise ValueError(
            f"the candidate-restricted top-k needs ratio == 1 (blocks of 8 keys), got {ratio!r}"
        )
    for name, value in (
        ("grid_ctas", grid_ctas),
        ("num_sms", num_sms),
        ("cand_cap", cand_cap),
        ("check_period", check_period),
    ):
        if value is not None:
            _check_int(name, value, 1)
    if arch is None:
        arch = arch_for(device)
        if arch is None:
            raise ValueError(
                "the candidate-restricted top-k needs an sm_100a / sm_103a / sm_107a device (or an explicit arch)"
            )
    if arch not in PROGRAM_KEYS:
        raise NotImplementedError(
            f"no generated DSA indexer program is registered for {arch}"
        )
    if grid_ctas is None:
        grid_ctas = num_sms if num_sms is not None else _num_sms(device)
    return select_cand_program(
        num_queries=num_queries,
        num_keys=num_keys,
        num_segments=num_segments,
        top_k=top_k,
        grid=int(grid_ctas),
        program_keys=PROGRAM_KEYS.get(arch, {}),
        cand_width=candidate_width,
        cand_cap=cand_cap,
        check_period=check_period,
        policy=policy_for(arch),
        finalize_threads=finalize_threads,
    )


def cake_dsa_indexer_det_train_topk_in_candidates_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    arch: Optional[str] = None,
    grid_ctas: Optional[int] = None,
    num_sms: Optional[int] = None,
) -> int:
    """Explicit workspace bound in bytes of one candidate-restricted call on ``device``: ``grid x 4 x cand_cap x 8``
    bytes of candidate buffers (``cand_cap = max(4096, 4 top_k)``), ``grid x 64 KiB`` of block lists and a 16-byte
    ticket word (:func:`plan_dsa_indexer_det_topk_in_candidates`).  It depends on the grid and ``top_k`` alone, not
    on ``T``, ``Tkv``, the segments or the candidate content.  Never consults torch's deterministic algorithms
    setting."""
    return int(
        plan_dsa_indexer_det_topk_in_candidates(
            num_queries,
            num_keys,
            num_segments,
            top_k=top_k,
            ratio=ratio,
            device=device,
            arch=arch,
            grid_ctas=grid_ctas,
            num_sms=num_sms,
        ).workspace_bytes
    )


@dataclass
class DSAIndexerDetCandidateRunner:
    """The prepared launches of one candidate-restricted tensor binding: ``run()`` launches the ``cand`` program
    and the CUB finalize that writes each row in ascending id order on torch's current stream and returns ``(indices, scores)``.  The program draws its work units from a
    16-byte ticket word at the workspace's end, which every completed launch leaves at 0: a backend-owned workspace
    zeroes it once at prepare, a caller-owned one (which others may write between runs) before every run, and a run
    after one that raised zeroes it first."""

    arch: str
    abi: str
    choice: CandidateTopKChoice
    programs: dict[str, str]
    tensors: dict[str, torch.Tensor] = field(repr=False)
    scalars: dict[str, Any] = field(repr=False)
    launches: tuple[_Launch, ...] = field(repr=False)
    stages: tuple[str, ...]
    device_index: int = 0
    workspace: torch.Tensor = field(default=None, repr=False)
    workspace_bytes: int = 0
    _ticket: Optional[torch.Tensor] = field(default=None, repr=False)
    _ticket_each_run: bool = field(default=False, repr=False)
    _ticket_dirty: bool = field(default=False, repr=False)

    @property
    def indices(self) -> torch.Tensor:
        return self.tensors["Indices"]

    @property
    def scores(self) -> torch.Tensor:
        return self.tensors["Scores"]

    @property
    def candidate_blocks(self) -> torch.Tensor:
        return self.tensors["CandBlocks"]

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        with _ffi_stream_context(self.device_index):
            if self._ticket is not None and (
                self._ticket_each_run or self._ticket_dirty
            ):
                self._ticket.zero_()
                self._ticket_dirty = False
            try:
                for launch in self.launches:
                    launch()
            except BaseException:
                self._ticket_dirty = self._ticket is not None
                raise
        return self.tensors["Indices"], self.tensors["Scores"]

    __call__ = run


def _prepare_cand_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    candidate_blocks: torch.Tensor,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    cand_cap: Optional[int] = None,
    check_period: Optional[int] = None,
    backend: str = "cake",
    one_shot: bool = False,
) -> DSAIndexerDetCandidateRunner:
    """:func:`prepare_dsa_indexer_det_topk_in_candidates`; ``one_shot``: the eager entry's runner."""
    if backend != "cake":
        raise ValueError("DSA indexer top-k supports backend='cake'")
    _check_cand_form(top_k, ratio, q)
    if softmax_scale is None:
        softmax_scale = default_softmax_scale()
    num_queries, num_keys, num_segments = validate_dsa_indexer_det_inputs(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        q_causal_offsets,
        top_k,
        softmax_scale,
        ratio,
        max_seqlen_q,
        max_seqlen_k,
        None,
    )
    device = q.device
    width = _check_cand_call(candidate_blocks, num_queries, device)
    for name, tensor in (
        ("workspace_buffer", workspace_buffer),
        ("indices", indices),
        ("scores", scores),
    ):
        if tensor is not None and (not tensor.is_cuda or tensor.device != device):
            raise ValueError(f"{name} must be on {device}")
    _check_output(indices, "indices", (num_queries, int(top_k)), torch.int32, device)
    _check_output(scores, "scores", (num_queries, int(top_k)), torch.float32, device)
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            f"DSA indexer top-k needs compute capability 10.0 / 10.3 / 10.7, got {device}"
        )
    abi = registry_abi()
    device_index = _device_index(device)
    choice = plan_dsa_indexer_det_topk_in_candidates(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        candidate_width=width,
        device=device,
        arch=arch,
        grid_ctas=grid_ctas,
        cand_cap=cand_cap,
        check_period=check_period,
    )
    # every program of the call must be registered before anything is allocated (fail closed by name)
    programs = {CAND_ROLE: program_for(arch, choice.cand_key)}
    programs[FINALIZE_ROLE] = program_for(arch, choice.finalize_key)
    workspace_bytes = int(choice.workspace_bytes)
    caller_owned_workspace = workspace_buffer is not None
    if caller_owned_workspace:
        flat = workspace_buffer.view(-1).view(torch.uint8)
        if flat.numel() < workspace_bytes:
            raise ValueError(
                f"workspace_buffer needs {workspace_bytes} bytes, got {flat.numel()}"
            )
        if flat.data_ptr() % 8:
            raise ValueError("workspace_buffer must be 8-byte aligned")
    # a call without queries, keys or candidate slots launches nothing: every row's pool is empty
    launching = num_queries > 0 and num_keys > 0 and width > 0
    if not caller_owned_workspace:
        flat = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    ticket = flat[choice.ticket_at : choice.ticket_at + CAND_TICKET_BYTES]
    t: dict[str, torch.Tensor] = {
        "Q": q.view(num_queries * NUM_HEADS, HEAD_DIM),
        "Kraw": k,
        "W": w,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "q_offsets": q_causal_offsets
        if q_causal_offsets is not None
        else _offsets_placeholder(device_index),
        "CandBlocks": candidate_blocks,
        "Indices": indices
        if indices is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.int32, device=device),
        "Scores": scores
        if scores is not None
        else torch.empty((num_queries, int(top_k)), dtype=torch.float32, device=device),
        "Lists": flat[choice.lists_at : choice.lists_at + choice.lists_bytes].view(
            torch.int32
        ),
        "Cand": flat[: choice.cand_bytes].view(torch.int64),
        "TicketWord": ticket.view(torch.uint32),
    }
    scalars: dict[str, Any] = dict(
        num_segments=num_segments,
        top_k=int(top_k),
        has_offsets=int(q_causal_offsets is not None),
        cand_width=width,
        cand_cap=int(choice.cand_cap),
        check_period=int(choice.check_period),
        grid_ctas=int(choice.grid_ctas),
        softmax_scale=float(softmax_scale),
        num_queries=num_queries,
        num_keys=num_keys,
        key_bits=int(choice.key_bits),
    )
    launches: list[_Launch] = []
    stages: list[str] = []
    if launching:
        values: dict[str, Any] = dict(t)
        values.update(scalars)
        launches.append(
            bind_program(
                programs[CAND_ROLE],
                arch,
                CAND_ROLE,
                values,
                (int(choice.grid_ctas), 1, 1),
            )
        )
        stages.append(CAND_ROLE)
        launches.append(
            bind_program(
                programs[FINALIZE_ROLE],
                arch,
                FINALIZE_ROLE,
                values,
                (max(1, num_queries), 1, 1),
            )
        )
        stages.append(FINALIZE_ROLE)
        if not caller_owned_workspace:
            # backend-owned: nothing else writes the ticket word, and every completed launch leaves it 0
            ticket.zero_()
    elif num_queries > 0:
        _padded_outputs(num_queries, int(top_k), device, t["Indices"], t["Scores"])
    return DSAIndexerDetCandidateRunner(
        arch=arch,
        abi=abi,
        choice=choice,
        programs=programs,
        tensors=t,
        scalars=scalars,
        launches=tuple(launches),
        stages=tuple(stages),
        device_index=device_index,
        workspace=flat,
        workspace_bytes=workspace_bytes,
        _ticket=ticket if launching else None,
        _ticket_each_run=caller_owned_workspace,
    )


def prepare_dsa_indexer_det_topk_in_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    candidate_blocks: torch.Tensor,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    grid_ctas: Optional[int] = None,
    cand_cap: Optional[int] = None,
    check_period: Optional[int] = None,
    backend: str = "cake",
) -> DSAIndexerDetCandidateRunner:
    """Prepare a candidate-restricted top-k: per query row ``t`` the top ``min(top_k, |P_t|)`` of the pool ``P_t``
    of visible keys ``j`` with ``j // 8`` in ``candidate_blocks[t]`` (int32 ``[T, W]``, ``W <= 2048``, contiguous;
    any order, ``-1`` and ids at or past the visible blocks name no key; non-negative ids of a row distinct), ranked
    as the family ranks keys.  BF16 only, ``ratio == 1``, ``1 <= top_k <= 2048`` (default 512), ``k`` under the family's
    rule (row stride a multiple of 8 elements, read in place).  ``cand_cap`` / ``check_period`` are partition knobs that never change the selected set.
    Rows are written in ascending id order with a ``-1`` / ``-inf`` tail, and identical inputs give identical
    bits."""
    return _prepare_cand_runner(
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
        grid_ctas=grid_ctas,
        cand_cap=cand_cap,
        check_period=check_period,
        backend=backend,
    )


def cake_dsa_indexer_det_train_topk_in_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    candidate_blocks: torch.Tensor,
    *,
    top_k: int = CB_DEFAULT_TOP_K,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager entry point of the candidate-restricted top-k: prepare a binding and run it once; returns ``(indices,
    scores)``.  See :func:`prepare_dsa_indexer_det_topk_in_candidates`.  No host synchronization; the outputs and
    the workspace are allocated through the caching allocator unless given."""
    runner = _prepare_cand_runner(
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
        backend=backend,
        one_shot=True,
    )
    return runner.run()


__all__ = [
    "ABI_CONTRACT",
    "CB_DEFAULT_TOP_K",
    "CB_SCALARS",
    "CAND_MAX_TOP_K",
    "CAND_SCALARS",
    "CAND_WIDTH_MAX",
    "CB_SLOTS",
    "CONTRACT_ALIASES",
    "CONTRACT_SCALARS",
    "CONTRACT_TENSORS",
    "DEFAULT_TOP_K",
    "FINALIZE_RANK_SCALARS",
    "FINALIZE_SCALARS",
    "HEAD_DIM",
    "MAX_TOP_K",
    "MERGE_SCALARS",
    "NUM_HEADS",
    "PAD_ID",
    "PAD_SCORE",
    "SCAN_SCALARS",
    "SUPPORTED_ABIS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "ZERO_SIGN_POLICIES",
    "DSAIndexerDetCandidateRunner",
    "DSAIndexerDetRunner",
    "DispatchPolicy",
    "FORM_DTYPES",
    "ProgramChoice",
    "arch_for",
    "bind_program",
    "default_softmax_scale",
    "cake_dsa_indexer_det_train_topk",
    "cake_dsa_indexer_det_train_topk_with_candidates",
    "cake_dsa_indexer_det_train_topk_workspace_size",
    "cake_dsa_indexer_det_train_topk_with_candidates_workspace_size",
    "cake_dsa_indexer_det_train_topk_in_candidates",
    "cake_dsa_indexer_det_train_topk_in_candidates_workspace_size",
    "effective_offsets",
    "form_for",
    "fp8_programs_for",
    "generated_program_available",
    "plan_dsa_indexer_det_topk",
    "plan_dsa_indexer_det_topk_in_candidates",
    "plan_dsa_indexer_det_topk_with_candidates",
    "policy_for",
    "prepare_dsa_indexer_det_topk",
    "prepare_dsa_indexer_det_topk_in_candidates",
    "prepare_dsa_indexer_det_topk_with_candidates",
    "program_for",
    "reads_ticket_word",
    "record_zero_sign_policy",
    "registry_abi",
    "validate_dsa_indexer_det_inputs",
    "visible_key_count",
    "visible_key_counts",
]
