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

Host plan and autograd chain of the deterministic GDN chunked training
(head dimension 128, chunk 64): the chunk and segment tables, the scan
configuration and segment planner, the workspace pool, the forward and backward
launch order, the adjoint-scan and finalize selection and the autograd wrapper
with FLA's ``chunk_gated_delta_rule`` signature (no context parallelism).

The forward runs the prep (``prep_tma``, the q / k tiles staged by TMA and one
CTA per (chunk, key head), unless its dispatch rule :func:`use_prep_tma` keeps
the plain prep for contiguous q / k; both give the same bits), the tcgen05 WY stage
(:func:`wy_stage`: ``wy_tc2``, five CTAs per SM, on the architectures of
``WY_TC2_ARCHS``, else ``wy_tc``; the same bits) and the fused tcgen05 scan
and output (``fwd_ho``; a split plan first runs the pre-scan and the stitch
for the segment start states; on the architectures of ``FWD_HO_VH2_ARCHS`` a
scan of one launch whose CTAs fill at most half of the SMs runs as
``fwd_ho_vh2``, two CTAs per segment and value head, one value half each,
:func:`fwd_ho_vh2_engaged`, where the record is the split program,
:func:`fwd_ho_vh2_splits`; the same bits).  A call with FP32 ``beta``,
normalized q / k, a value-head group of at most three, no ``prep`` /
``scan_config`` / ``segment_chunks`` pin and a host-built table runs one of two
forwards on the WY A producer ``wy_a`` instead (other bits than the prep
forward, bitwise run to run): on the architectures of ``FWD_A_ARCHS``, where the
forward segment table is serial, ``wy_a`` and ``fwd_ho_a``, the fused scan and
output on its A (:func:`fwd_a_route`); on those of ``FWD_SPLIT_ARCHS``, where
the forward segment table splits a sequence and the host plan's piece table
is in the split's shape class, the one-wave split forward
(:func:`fwd_split_route`: the first pieces beside the middle pieces' affine
maps on a side stream, the stitch, the other pieces; on wy_a's A, or on the
WY stage's w / u below ``FWD_SPLIT_A_MIN_PIECE``).  Both keep the tape only
when the call records a backward (:func:`_forward_tape`).  The backward runs ``dv_local`` and
the register scan of the adjoint state (with the pre-scan and the stitch on a
split plan), or on a serial plan the tcgen05 adjoint scan with ``dv_local``
fused (``dhu_tc2``) when its selection rule (:func:`dhu_tc_pricing`) prices it
at most ``DHU_TC_GAIN`` of the register path; a split plan at ``SDT_CONFIG``
on a host-built table runs its pass 2 on ``dhu_tc2`` from the stitched segment
states when its rule (:func:`sdt_pricing`) prices that pass 2 below the
register one, and then launches no ``dv_local``: its pre-scan
(``dhu_pre_dvl_r64k1``) computes the ``dv_local`` tiles of the chunks it scans
(the same bits); where that pass 2 is ``dhu_tc2_dvl32`` at a value-head group
of three, on the architectures of ``ADJOINT_MAP_ARCHS``, the segment maps the
stitch composes come from ``dhu_tc2_dvl32_gm`` over every segment but each
sequence's first instead (:func:`split_map_stage`), on two CTAs per entry
(``dhu_tc2_dvl32_gm2``, the same bits) where that grid fits one wave of the
SMs (:func:`split_map_launch`); on the architectures of
``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`` (``sm_107a``) such a grid keeps the
pre-scan.  On ``sm_107a``, on a
host-built table of a value-head group of two, a split adjoint plan the
serial-scan rule (:func:`serial_scan_pricing`)
prices slower than ``dhu_tc2`` over whole sequences becomes that serial plan
(other bits of ``dk`` / ``dv`` / ``dg`` / ``dbeta``).  A host-built table
without plan pins then takes the adjoint table priced for ``dhu_tc2``
itself where that pricing (:func:`tc_plan_pricing`,
``TC_PLAN``) beats the planner's table, and on the architectures of
``CHAIN_PLAN`` deeper split tables priced for the ``fwd_ho`` and ``dhu_tc2``
chains (:func:`chain_plan_pricing`); a table whose adjoint scan runs as two
groups beside the fused tail keeps its form.  The backward runs such a table on
``dhu_tc2``; the bits follow the table.  Then, for a
value-head group of two or of three, the fused
tcgen05 tail (``tail_g2`` / ``tail_g3``); for any other group on the
architectures of ``PACKAGE_TAIL_ARCHS`` the package tail (``tail``: one CTA per
(chunk, value head), the group's FP32 ``dq`` / ``dk`` partials summed in
value-head order before one BF16 rounding, :func:`tail_segments`), at a group
of four or more only when the call's mean tokens per sequence reach the
architecture's ``PACKAGE_TAIL_MIN_TOKENS_PER_SEQ``; else ``dqkwg``, the WY
pullback and the finalize, which a ``finalize`` pin keeps for every group but
two.  On the architectures of ``F32X_TAIL_ARCHS`` the tail of a
group of three is the round-once ``tail_g3_f32x`` (the group's FP32 ``dq`` /
``dk`` partials summed in value-head order before one BF16 rounding, written
through token-row strides, so the head views of one packed row get one packed
gradient row, :class:`CakeGDNDetPackedFunction`), and on those of
``DVL32_ARCHS`` the tcgen05 adjoint scan of every group but two is
``dhu_tc2_dvl32`` (the FP32 ``dv_local`` values added to ``dv2`` before its one
rounding: other ``dk`` / ``dv`` / ``dg`` / ``dbeta`` / initial-state-gradient
bits than the register scan's);
:func:`fused_tail_stage`, :func:`dhu_tc2_stage`.  A host-built table lists
the sequences longest first (:func:`lpt_sequence_order`; every chunk keeps its
token offset, the state rows are gathered into that order and the final state
and its gradient scattered back) and ``fwd_ho`` walks the forward segments longest first
(:func:`lpt_walk`): each CTA computes what it computes in the caller's order.
Where a serial scan leaves its last round partly idle, a host-built table runs
it as two contiguous sequence groups (:func:`sequence_groups`), one on a
high-priority side stream forked from the caller's stream and joined back
before the stage returns: the forward's ``wy_tc`` and ``fwd_ho`` of the partial
group overlap the full rounds of the other, and the adjoint ``dhu_tc2`` of the
partial group of a serial plan overlaps ``tail_g2`` of the other (a group of
three's tail on the architectures of ``TWO_STREAM_SERIAL_G3_ARCHS``); beside a
group of three's tail the ``dhu_tc2`` pass 2 of a split plan runs as two such
groups of segments.

Every launch is a record of :mod:`.cake_jit` selected by its stage and the
device's architecture (:func:`.cake_backend.launch_stage`); the stages a plan
may use are the ones the registry delivers for that architecture.  The
planner's per-configuration facts (the register scans' shared-memory figure,
the chunks one pre-scan CTA covers) are the records' ``scan`` constants; the
``dhu_tc2`` selection constants (serial plans and the split pass 2) and the
``prep_tma`` dispatch window are those
records' ``select`` tables.

Where the ``dhu_tc2`` adjoint scan feeds ``tail_g2_bf16`` on one stream and
takes at most half the SMs (on ``sm_107a`` with the scan's release in batches
of eight chunks), or on ``sm_100a`` and ``sm_103a`` is the pass 2 of
a split plan with more CTAs than SMs on a call of enough tokens x value heads,
the backward runs them as the wavefront pair
``dhu_tc2_wavefront`` / ``tail_g2_bf16_wavefront`` (the split pass 2 on
``dhu_tc2_wavefront_split``; :func:`wavefront_engaged`,
:func:`wavefront_split_engaged`): the tail of a chunk starts once the scan has
stored it, counted per tile in a workspace; the same math and the same bits.

The chain is deterministic: no atomic on a value (the wavefront pair's ready
counts and the package tail's arrival counters are atomic integer counters whose
order no value depends on: the package tail's last CTA of a (chunk, key head)
tile sums the group's slots in value-head order whichever CTA it is), no
data-dependent ordering, and the two groups of a two-stream stage write disjoint outputs (as do
the first pieces and the middle pieces' maps of the split forward), so the
outputs and gradients are bitwise run to run for one layout and one plan, with
one or two streams.  Every route is chosen from host facts (shapes, dtypes, the
architecture, the SM count, the pins), never from a pointer or a value: an
operand whose address a stage cannot read is copied first.
The plan pins that tests and tuning use are explicit keyword arguments of
:func:`cake_gdn_det` (``scan_config``, ``segment_chunks``,
``finalize``, ``warmup_log2``, ``dhu_scan``, ``prep``, ``lpt``, ``streams``);
nothing here reads an environment variable.
"""

from __future__ import annotations

import math
import struct
import threading
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import cache, lru_cache
from operator import is_, itemgetter
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, cast

import torch

from . import cake_backend, cake_jit, cake_plan

__all__ = [
    "ADJOINT_MAP_ARCHS",
    "ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS",
    "ADJOINT_MAP_SPLIT_ARCHS",
    "ADJOINT_MAP_SPLIT_STAGE",
    "ADJOINT_MAP_STAGE",
    "CHAIN_PLAN",
    "CHUNK",
    "DVL32_ARCHS",
    "F32X_TAILS",
    "F32X_TAIL_ARCHS",
    "FINALIZE_STAGES",
    "FULL",
    "FWD_A_ARCHS",
    "FWD_A_SERIAL_KEEP",
    "FWD_SPLIT_ARCHS",
    "HEAD_DIM",
    "PACKAGE_TAILS",
    "PACKAGE_TAIL_ARCHS",
    "PACKAGE_TAIL_MIN_TOKENS_PER_SEQ",
    "PACKED_ROW",
    "PRESCAN_CONFIGS",
    "REG_SCAN_CONFIGS",
    "SPLIT_WARM",
    "TC_PLAN",
    "TC_PLAN_GAIN",
    "TC_PLAN_MIN_SAVING",
    "TC_PLAN_PRESCAN",
    "TC_PLAN_SELECT",
    "TC_PLAN_STEP",
    "WY_A_MAX_GROUP",
    "CakeGDNDetFunction",
    "CakeGDNDetPackedFunction",
    "ChainOptions",
    "ChunkMeta",
    "ScanPlan",
    "SegmentTable",
    "StageSet",
    "StreamGroups",
    "build_chunk_meta",
    "build_chunk_meta_from_cu_seqlens",
    "cake_gdn_det",
    "cake_gdn_det_train_backward",
    "cake_gdn_det_train_forward",
    "chain_plan_pricing",
    "choose_scan_plan",
    "dhu_tc2_stage",
    "dhu_tc_pricing",
    "finalize_stage",
    "fused_tail_stage",
    "fwd_a_route",
    "fwd_ho_vh2_engaged",
    "fwd_ho_vh2_splits",
    "fwd_serial_keep_pricing",
    "fwd_split_route",
    "host_plans",
    "lpt_sequence_order",
    "lpt_walk",
    "make_options",
    "scan_plans",
    "sdt_pricing",
    "serial_scan_pricing",
    "sequence_groups",
    "split_map_launch",
    "split_map_stage",
    "split_summary_entries",
    "stage_set",
    "tail_segments",
    "tc_plan_pricing",
    "tc_plan_thresholds",
    "two_stream_split",
    "use_prep_tma",
    "wavefront_engaged",
    "wavefront_order",
    "wavefront_split_engaged",
    "wy_stage",
]

CHUNK: int = int(cake_jit.CHUNK)
HEAD_DIM: int = int(cake_jit.HEAD_DIM)

# Register-scan configurations ``(value block, key split)`` in the planner's preference order, and the ones with a
# segment-parallel route (pre-scan + split scan for the forward, pre-scan + stitch for the adjoint).
REG_SCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (32, 1), (64, 1))
PRESCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (64, 1))
SCAN_KINDS = ("fwd_h", "dhu")
STITCH_VSPLIT = 4  # CTAs per (sequence, value head) of the stitch launch

# The tag of this package's forward tape: the backward reads only a tape that carries it (the non-deterministic
# package's tape carries ``"mode": "nondet"``, so it is refused).
_TAPE_MODE = "det"

# Kernel parameter names of every stage in call-site order; a delivered record whose parameters are these names in
# another order is launched with the values reordered.  The stages that read v (wy_tc2, wy_bwd, the tails) take its
# token pitch in heads last (``v_row_heads``, :func:`_v_rows`), but the round-once tail, which takes the token-row
# strides of v, dq, dk and dv in elements (:func:`_tail_row_strides`).
_PREP_ARGS = ("q", "k", "g", "beta", "beta32_out", "qn", "kn", "rstd_q", "rstd_k", "g_cs", "chunk_start", "chunk_len",
              "num_heads", "num_v_heads", "normalize_qk")  # fmt: skip
# prep_tma takes q / k (dhu_tc2 qn / kn / w / do) as the tensors its binding encodes the tensor maps from
_PREP_TMA_ARGS = ("q_tma", "k_tma") + _PREP_ARGS[2:]
_WY_ARGS = ("kn", "v", "g_cs", "beta", "A_out", "w_out", "u_out", "chunk_start", "chunk_len", "num_heads",
            "num_v_heads", "recompute")  # fmt: skip
_FWD_H_ARGS = ("kn", "w", "u", "g_cs", "h0", "h_out", "v_new", "final_state", "seg_state", "chunk_start",
               "chunk_len", "seg_chunk_start", "seg_seq", "seq_chunk_start", "num_heads", "num_v_heads",
               "use_initial_state", "store_final_state")  # fmt: skip
_FWD_H_SPLIT_ARGS = _FWD_H_ARGS + ("seg_flag",)
_FWD_H_PRE_ARGS = _FWD_H_SPLIT_ARGS + ("warmup_thresh_log2",)
_STITCH_ARGS = ("seg_state", "init_state", "state_out", "seq_seg_start", "seg_flag", "num_v_heads", "use_init",
                "reverse")  # fmt: skip
_FWD_HO_ARGS = ("qn_tma", "kn_tma", "w_tma", "u_tma", "g_cs", "h0", "seg_start", "h_tma", "v_new_tma", "v_new",
                "o_tma", "o_out", "final_state", "chunk_start", "chunk_len", "seg_chunk_start", "seg_seq",
                "seq_chunk_start", "walk", "num_heads", "num_v_heads", "use_initial_state", "store_final_state",
                "use_seg_start", "scale")  # fmt: skip
_DV_LOCAL_ARGS = ("qn", "kn", "do", "g_cs", "dv_local", "chunk_start", "chunk_len", "num_heads", "num_v_heads",
                  "scale")  # fmt: skip
_DHU_ARGS = ("qn", "kn", "w", "do", "dv_local", "g_cs", "dht", "dh_out", "dv2", "dh0", "seg_state", "chunk_start",
             "chunk_len", "seg_chunk_start", "seg_seq", "seq_chunk_start", "num_heads", "num_v_heads",
             "use_final_state_grad", "store_initial_state_grad", "scale")  # fmt: skip
_DHU_PRE_ARGS = _DHU_ARGS + ("seg_flag", "warmup_thresh_log2")
# the (64, 1) pre-scan that computes the dv_local tiles of its chunks in the CTA
_DHU_PRE_DVL_ARGS = tuple(a for a in _DHU_PRE_ARGS if a != "dv_local")
_DHU_TC2_ARGS = ("qn_tma", "kn_tma", "w_tma", "do_tma") + tuple(
    a for a in _DHU_ARGS[4:] if a != "dv_local"
)
# the segment maps of a split pass 2 on dhu_tc2_dvl32 (dhu_tc2_dvl32_gm): the scan's arguments, the entry table (segment
# index and mode bits per CTA row), the warm-up threshold and the segment flags
_DHU_TC2_GM_ARGS = (
    *_DHU_TC2_ARGS,
    "seg_index",
    "seg_mode",
    "warmup_thresh_log2",
    "seg_flag",
)
_DQKWG_ARGS = ("qn", "kn", "v_new", "do", "dv2", "h", "dh", "g_cs", "dq_out", "dk_out", "dw_out", "dg_out",
               "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale")  # fmt: skip
_WY_BWD_ARGS = ("kn", "v", "A", "dw", "du", "g_cs", "beta", "dk2_out", "dv_out", "dbeta_out", "dg2_out",
                "chunk_start", "chunk_len", "num_heads", "num_v_heads")  # fmt: skip
_FINALIZE_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "dq_hv", "dk_hv", "dk2", "dg1", "dg2", "dq_out", "dk_out", "dg_out",
                  "chunk_start", "chunk_len", "num_heads", "num_v_heads", "normalize_qk")  # fmt: skip
_TAIL_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "v_new", "do", "dv2", "h", "dh", "g_cs", "v", "A", "beta", "dq_out",
              "dk_out", "dg_out", "dv_out", "dbeta_out", "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale",
              "normalize_qk")  # fmt: skip
# the round-once tail of a value-head group of three: tail_g3's operands, then the token-row strides (elements) of v,
# dq, dk and dv
_TAIL_F32X_ARGS = (
    *_TAIL_ARGS,
    "v_row_stride",
    "dq_row_stride",
    "dk_row_stride",
    "dv_row_stride",
)
# the package tail of any other value-head group: the operands of tail_g2, then the workspace, its arrival counters and
# its tiles
_TAIL_WS_ARGS = (*_TAIL_ARGS[:18], "ws", "ws_count", *_TAIL_ARGS[18:], "ws_tiles")
# the wavefront pair: dhu_tc2 / tail_g2_bf16 with the ready counts (and the tail's chunk order)
_DHU_TC2_WAVEFRONT_ARGS = _DHU_TC2_ARGS + ("ready",)
_TAIL_WAVEFRONT_ARGS = _TAIL_ARGS + ("ready", "tile_order")
_META_ARGS = ("cu_seqlens", "chunk_start", "chunk_len", "seq_chunk_start", "fwd_seg_chunk_start", "fwd_seg_seq",
              "fwd_seq_seg_start", "bwd_seg_chunk_start", "bwd_seg_seq", "bwd_seq_seg_start", "num_seqs",
              "num_chunks_max", "total_tokens", "fwd_segment_len", "bwd_segment_len", "fwd_num_segments_max",
              "bwd_num_segments_max")  # fmt: skip
# the forwards on the WY stage's A: wy_a (k / q as the tensors its binding encodes the tensor maps from, k twice for
# its two boxes), fwd_ho_a (the scan and output on that A), fwd_fused (fwd_ho's scan and output with a tape word and
# the split forward's wave-1 outputs) and the summaries of the split forward's middle pieces
_WY_A_ARGS = ("k_tma64", "k_tma32", "q_tma", "glog", "beta", "a_out", "rk_out", "rq_out", "gcs_out", "qn_out", "kn_out",
              "chunk_start", "chunk_len", "num_chunks", "num_heads", "num_v_heads")  # fmt: skip
_FWD_A_ARGS = ("qn_tma", "kn_tma", "v_tma", "a_tma", "g_cs", "beta", "h0", "seg_start", "h_tma", "h_out", "v_new_tma",
               "v_new", "o_tma", "o_out", "final_state", "w_tma", "w_out", "chunk_start", "chunk_len",
               "seg_chunk_start", "seg_seq", "seq_chunk_start", "walk", "num_heads", "num_v_heads",
               "use_initial_state", "store_final_state", "use_seg_start", "scale", "tape")  # fmt: skip
_FWD_FUSED_ARGS = (*_FWD_HO_ARGS, "tape", "seg_state", "seg_flag", "split_out")
_SUMMARY_TAIL_ARGS = ("b_out", "m_out", "seg_flag", "chunk_start", "chunk_len", "seg_chunk_start", "walk", "m_slot",
                      "num_heads", "num_v_heads", "b_head", "b_row", "m_head", "m_row", "m_col")  # fmt: skip
_FWD_SUMMARY_ARGS = ("kn_tma", "w_tma", "u_tma", "g_cs", *_SUMMARY_TAIL_ARGS)
_FWD_SUMMARY_A_ARGS = ("kn_tma", "v_tma", "a_tma", "g_cs", "beta", *_SUMMARY_TAIL_ARGS)
LAUNCH_ARGS: dict[str, tuple[str, ...]] = {
    "meta": _META_ARGS,
    "prep": _PREP_ARGS,
    "prep_bf16beta": _PREP_ARGS,
    "prep_tma": _PREP_TMA_ARGS,
    "prep_tma_bf16beta": _PREP_TMA_ARGS,
    "wy_tc": _WY_ARGS,
    "wy_tc2": _WY_ARGS + ("v_row_heads",),
    "wy_a": _WY_A_ARGS,
    "fwd_ho": _FWD_HO_ARGS,
    "fwd_ho_vh2": _FWD_HO_ARGS,
    "fwd_ho_a": _FWD_A_ARGS,
    "fwd_fused": _FWD_FUSED_ARGS,
    "fwd_summary": _FWD_SUMMARY_ARGS,
    "fwd_summary_a": _FWD_SUMMARY_A_ARGS,
    "stitch": _STITCH_ARGS,
    "dv_local": _DV_LOCAL_ARGS,
    "dhu_tc2": _DHU_TC2_ARGS,
    "dhu_tc2_dvl32": _DHU_TC2_ARGS,
    "dhu_tc2_dvl32_gm": _DHU_TC2_GM_ARGS,
    "dhu_tc2_dvl32_gm2": _DHU_TC2_GM_ARGS,
    "dhu_tc2_wavefront": _DHU_TC2_WAVEFRONT_ARGS,
    "dhu_tc2_wavefront_split": _DHU_TC2_WAVEFRONT_ARGS,
    "dqkwg": _DQKWG_ARGS,
    "wy_bwd": _WY_BWD_ARGS + ("v_row_heads",),
    "wy_bwd_bf16": _WY_BWD_ARGS + ("v_row_heads",),
    "finalize": _FINALIZE_ARGS,
    "finalize_mlp": _FINALIZE_ARGS,
    "tail_g2": _TAIL_ARGS + ("v_row_heads",),
    "tail_g2_bf16": _TAIL_ARGS + ("v_row_heads",),
    "tail_g2_bf16_wavefront": _TAIL_WAVEFRONT_ARGS + ("v_row_heads",),
    "tail_g3": _TAIL_ARGS + ("v_row_heads",),
    "tail_g3_bf16": _TAIL_ARGS + ("v_row_heads",),
    "tail_g3_f32x": _TAIL_F32X_ARGS,
    "tail_g3_f32x_bf16": _TAIL_F32X_ARGS,
    "tail": _TAIL_WS_ARGS + ("v_row_heads",),
    "tail_bf16": _TAIL_WS_ARGS + ("v_row_heads",),
    "dhu_pre_dvl_r64k1": _DHU_PRE_DVL_ARGS,
    **{f"fwd_h_r{vb}k{ks}": _FWD_H_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_r{vb}k{ks}_split": _FWD_H_SPLIT_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"fwd_h_pre_r{vb}k{ks}": _FWD_H_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"dhu_r{vb}k{ks}": _DHU_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"dhu_pre_r{vb}k{ks}": _DHU_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
}


def _with_after(
    args: tuple[str, ...], name: str, extra: tuple[str, ...]
) -> tuple[str, ...]:
    """``args`` with ``extra`` right after ``name``."""
    i = args.index(name) + 1
    return (*args[:i], *extra, *args[i:])


# :data:`PV_TF32`: the call sites of the programs built with ``pv_tf32=True``.  The scan reads h and
# v_new and writes the FP32 dg state terms after ``dh_out``: ``hdh32_out`` [chunks, HV, 2] (sum h * dh at the chunk's
# end adjoint, then the chunk's sum of e) and ``e_out`` [T, HV] (each token's K1 . kn term); its tails read them after
# ``dh`` in place of their BF16 sum(h dh) and K1 . kn terms (the package tail's workspace operands then follow the
# first 20 operands instead of 18).  These records serve stages of their own (the stage + ``_pv_tf32``), so every
# other record keeps its call site.
_DHU_TC2_PV_ARGS = _with_after(
    _DHU_TC2_ARGS, "dh_out", ("h", "v_new", "hdh32_out", "e_out")
)
_DQKWG_PV_ARGS = _with_after(_DQKWG_ARGS, "dh", ("hdh32", "e32"))
_TAIL_PV_ARGS = _with_after(_TAIL_ARGS, "dh", ("hdh32", "e32"))
_TAIL_F32X_PV_ARGS = _with_after(_TAIL_F32X_ARGS, "dh", ("hdh32", "e32"))
_TAIL_WS_PV_ARGS = (
    *_TAIL_PV_ARGS[:20],
    "ws",
    "ws_count",
    *_TAIL_PV_ARGS[20:],
    "ws_tiles",
)
LAUNCH_ARGS.update(
    {
        "dhu_tc2_pv_tf32": _DHU_TC2_PV_ARGS,
        "dhu_tc2_dvl32_pv_tf32": _DHU_TC2_PV_ARGS,
        "dqkwg_pv_tf32": _DQKWG_PV_ARGS,
        "tail_g2_pv_tf32": _TAIL_PV_ARGS + ("v_row_heads",),
        "tail_g2_bf16_pv_tf32": _TAIL_PV_ARGS + ("v_row_heads",),
        "tail_g3_f32x_pv_tf32": _TAIL_F32X_PV_ARGS,
        "tail_g3_f32x_bf16_pv_tf32": _TAIL_F32X_PV_ARGS,
        "tail_pv_tf32": _TAIL_WS_PV_ARGS + ("v_row_heads",),
        "tail_bf16_pv_tf32": _TAIL_WS_PV_ARGS + ("v_row_heads",),
    }
)
# The low word of dv2 (see the pv_tf32 section): every pv_tf32 scan also writes ``dv2_lo`` [T, HV, 128] right after
# ``dv2`` (its FP32 dv2 minus the BF16 word of the tape, rounded to BF16) and every pv_tf32 tail but dqkwg_pv_tf32
# reads it right after ``dv2``; the WY pullback behind dqkwg_pv_tf32 reads it as ``du_lo`` right after ``du`` (stages
# of their own, the stage + ``_dvlo``: DV_LO_WY).
LAUNCH_ARGS.update(
    {
        stage: _with_after(LAUNCH_ARGS[stage], "dv2", ("dv2_lo",))
        for stage in LAUNCH_ARGS
        if stage.endswith("_pv_tf32") and stage != "dqkwg_pv_tf32"
    }
)
LAUNCH_ARGS.update(
    {
        "wy_bwd_dvlo": _with_after(LAUNCH_ARGS["wy_bwd"], "du", ("du_lo",)),
        "wy_bwd_bf16_dvlo": _with_after(LAUNCH_ARGS["wy_bwd_bf16"], "du", ("du_lo",)),
    }
)


# --------------------------------------------------------------------------- #
# the planner's per-configuration facts, read from the registry's scan records
# --------------------------------------------------------------------------- #
def _scan_facts() -> tuple[dict[tuple[str, int, int], int], int]:
    smem: dict[tuple[str, int, int], int] = {}
    chunks: set[int] = set()
    for name, found in cake_jit.MODULES.items():
        stage = found["stage"]
        scan = found.get("scan")
        for kind in SCAN_KINDS:
            for vb, ks in REG_SCAN_CONFIGS:
                if stage in (
                    f"{kind}_r{vb}k{ks}",
                    f"{kind}_pre_r{vb}k{ks}",
                    f"fwd_h_r{vb}k{ks}_split",
                ):
                    if scan is None or (scan.get("v_block"), scan.get("k_split")) != (
                        vb,
                        ks,
                    ):
                        raise ValueError(
                            f"record {name}: scan constants {scan} do not match stage {stage}"
                        )
        if scan is None:
            continue
        key = (stage.split("_r")[0], scan["v_block"], scan["k_split"])
        if "select_smem_bytes" in scan:
            if smem.setdefault(key, int(scan["select_smem_bytes"])) != int(
                scan["select_smem_bytes"]
            ):
                raise ValueError(f"records of {stage} disagree on select_smem_bytes")
        if "max_segment_chunks" in scan:
            chunks.add(int(scan["max_segment_chunks"]))
    missing = [
        (k, vb, ks)
        for k in SCAN_KINDS
        for vb, ks in REG_SCAN_CONFIGS
        if (k, vb, ks) not in smem
    ]
    if missing or len(chunks) != 1:
        raise ValueError(
            f"registry scan records incomplete: missing {missing}, pre-scan chunks {sorted(chunks)}"
        )
    return smem, chunks.pop()


REG_SCAN_SMEM, PRESCAN_BLOCK_THREADS = _scan_facts()


# --------------------------------------------------------------------------- #
# options (the plan pins)
# --------------------------------------------------------------------------- #
FINALIZE_MLP_GROUP = 3
FINALIZE_MLP_MIN_BLOCKS = 4
FINALIZE_STAGES = ("finalize", "finalize_mlp")
DHU_SCANS = ("reg", "tc")
PREP_FORMS = ("plain", "tma")
STREAM_COUNTS = (1, 2)


@dataclass(frozen=True)
class ChainOptions:
    """The explicit plan pins of one call.

    ``scan_config`` forces the register-scan configuration ``(value block, key split)`` of both scans;
    ``segment_chunks`` forces the segment length in chunks (0: the serial scan; above the chunks one pre-scan
    CTA covers, 128, a plan that splits a sequence raises); ``finalize`` forces the finalize stage of the unfused tail
    (value-head groups other than two, which then run the unfused tail instead of the fused tail of a group of three
    or the package tail of the other groups, :func:`fused_tail_stage`); ``warmup_log2`` overrides the pre-scans'
    warm-up threshold; ``dhu_scan`` forces the adjoint scan of a serial plan (``"reg"``: ``dv_local`` + the register
    scan, ``"tc"``: ``dhu_tc2``; ``None``: the selection rule) and the pass 2 of a split ``SDT_CONFIG`` plan on a
    host-built table (``"reg"``: ``dv_local`` + the register scan, ``"tc"``: ``dhu_tc2`` after the fused pre-scan).
    ``"tc"`` runs the tcgen05 scan of :func:`dhu_tc2_stage`: on ``dhu_tc2`` it gives the bits of ``"reg"``; on
    ``dhu_tc2_dvl32`` (every value-head group but two on ``DVL32_ARCHS``, dv2 rounded once) o and the final state keep
    the bits of ``"reg"`` and the gradients take others; ``prep``
    forces the prep (``"plain"``: grid (chunks, HV); ``"tma"``: ``prep_tma``,
    grid (chunks, H), where q and k are 16-byte aligned; ``None``: the dispatch rule) and with it the prep forward (a
    ``prep``, ``scan_config`` or ``segment_chunks`` pin keeps a call off the forwards on wy_a's A, :func:`fwd_a_route`,
    :func:`fwd_split_route`); ``save_state`` keeps the forward state of the architecture's tape level for the backward
    (:func:`cake_plan.tape_state`: ``w`` / ``h`` / ``v_new`` under ``T_full``; else they are recomputed, as ``u``
    is where ``h`` / ``v_new`` are); ``host_cache`` keeps the
    chunk tables, the selection results and the per-stream workspaces between calls; ``lpt`` dispatches the
    sequences of a host-built table longest first (the same bits; ``False``: the caller's order); ``streams`` is
    ``2`` to let a host-built table's serial scans (the adjoint one beside ``tail_g2``, and beside a value-head group
    of three's tail on the architectures of ``TWO_STREAM_SERIAL_G3_ARCHS``) and the ``dhu_tc2`` pass 2 of its split
    plans (beside a group of three's tail) run as two groups on the caller's stream and a side stream where their rule
    (:func:`two_stream_split`) fills the last round (the same bits), ``1`` to keep every launch on the caller's
    stream.
    """

    scan_config: tuple[int, int] | None = None
    segment_chunks: int | None = None
    finalize: str | None = None
    warmup_log2: float | None = None
    save_state: bool = True
    host_cache: bool = True
    dhu_scan: str | None = None
    prep: str | None = None
    lpt: bool = True
    streams: int = 2

    @property
    def pins(self) -> tuple:
        return (self.segment_chunks, self.scan_config)


def make_options(
    *,
    scan_config: tuple[int, int] | None = None,
    segment_chunks: int | None = None,
    finalize: str | None = None,
    warmup_log2: float | None = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: str | None = None,
    prep: str | None = None,
    lpt: bool = True,
    streams: int = 2,
) -> ChainOptions:
    """Validated :class:`ChainOptions`."""
    if scan_config is not None:
        if (
            not isinstance(scan_config, (tuple, list))
            or len(scan_config) != 2
            or not all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0
                for x in scan_config
            )
        ):
            raise ValueError(
                f"scan_config must be a pair of positive ints; got {scan_config!r}"
            )
        scan_config = (int(scan_config[0]), int(scan_config[1]))
    if segment_chunks is not None:
        if (
            not isinstance(segment_chunks, int)
            or isinstance(segment_chunks, bool)
            or segment_chunks < 0
        ):
            raise ValueError(
                f"segment_chunks must be a non-negative int; got {segment_chunks!r}"
            )
    if (
        not isinstance(streams, int)
        or isinstance(streams, bool)
        or streams not in STREAM_COUNTS
    ):
        raise ValueError(f"streams must be one of {STREAM_COUNTS}; got {streams!r}")
    if finalize is not None and finalize not in FINALIZE_STAGES:
        raise ValueError(f"finalize must be one of {FINALIZE_STAGES}; got {finalize!r}")
    if warmup_log2 is not None:
        warmup_log2 = float(warmup_log2)
        if not math.isfinite(warmup_log2):
            raise ValueError(f"warmup_log2 must be finite; got {warmup_log2!r}")
    if dhu_scan is not None and dhu_scan not in DHU_SCANS:
        raise ValueError(
            f"dhu_scan must be one of {DHU_SCANS} or None; got {dhu_scan!r}"
        )
    if prep is not None and prep not in PREP_FORMS:
        raise ValueError(f"prep must be one of {PREP_FORMS} or None; got {prep!r}")
    return ChainOptions(
        scan_config,
        segment_chunks,
        finalize,
        warmup_log2,
        bool(save_state),
        bool(host_cache),
        dhu_scan,
        prep,
        bool(lpt),
        int(streams),
    )


# --------------------------------------------------------------------------- #
# stage availability and launch
# --------------------------------------------------------------------------- #
@cache
def _available(stage: str, arch: str) -> bool:
    try:
        cake_backend.stage_record(stage, arch)
    except KeyError:
        return False
    return True


@cache
def _order(stage: str, arch: str) -> tuple[int, ...] | None:
    names = cake_backend.parameters(cake_backend.stage_record(stage, arch))
    wanted = LAUNCH_ARGS[stage]
    if names == wanted:
        return None
    if sorted(names) == sorted(wanted):
        return tuple(wanted.index(n) for n in names)
    raise ValueError(
        f"{stage}: record parameters {names} do not match the call site {wanted}"
    )


def _device_index(device) -> int:
    """The CUDA device index of ``device`` (the current device for a bare ``cuda``)."""
    index = torch.device(device).index
    return int(index) if index is not None else torch.cuda.current_device()


def _tensor_index(t) -> int:
    """``_device_index(t.device)`` from ``Tensor.get_device()``, without a ``torch.device`` object: a host or meta
    tensor (``-1``) maps to the current device too, so its launch reaches the binding's device checks."""
    index = t.get_device()
    return index if index >= 0 else torch.cuda.current_device()


def _launch(stage, gx, gy, gz, *values):
    """Launch ``stage`` on ``(gx, gy, gz)``; ``values`` in call-site order, the first a tensor on the device."""
    device_index = _tensor_index(values[0])
    order = _order(stage, cake_backend.device_arch(device_index))
    if order is not None:
        values = tuple(values[i] for i in order)
    cake_backend.launch_stage(stage, (gx, gy, gz), values, device_index=device_index)


def _prepared(stage, gx, gy, gz, *values):
    """The launch of ``stage`` on ``(gx, gy, gz)`` as :func:`_launch` makes it, resolved for :func:`_fire` (a forward's
    recording keeps it): ``values`` in call-site order (the first a tensor on the device) put in the record's
    parameter order."""
    device_index = _tensor_index(values[0])
    order = _order(stage, cake_backend.device_arch(device_index))
    if order is not None:
        values = tuple(values[i] for i in order)
    return stage, (gx, gy, gz), values, device_index


def _fire(launch) -> None:
    """Run one launch of :func:`_prepared` (a forward's recording)."""
    stage, grid, values, device_index = launch
    cake_backend.launch_stage(stage, grid, values, device_index=device_index)


# --------------------------------------------------------------------------- #
# chunk table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScanPlan:
    config: tuple[int, int]
    segment_len: int


@dataclass(frozen=True)
class SegmentTable:
    seg_chunk_start: torch.Tensor
    seg_seq: torch.Tensor
    seq_seg_start: torch.Tensor
    num_segments: int
    segment_len: int
    scan_config: tuple[int, int]
    # fwd_ho's segment order (CTA group b runs walk[b]); None: the identity
    walk: torch.Tensor | None = None
    # the entry table of the segment maps (``dhu_tc2_dvl32_gm``, :func:`split_map_stage`) of a host-built adjoint table
    # that splits a sequence: row 0 every segment but each sequence's first, in table order, row 1 its mode
    # (``SPLIT_WARM``); None: the other tables
    summary: torch.Tensor | None = None
    num_summary: int = 0
    # the two rows of ``summary``, the map launch's ``seg_index`` and ``seg_mode``, taken once with the table; None with
    # ``summary``
    seg_index: torch.Tensor | None = None
    seg_mode: torch.Tensor | None = None


@dataclass(frozen=True)
class TableReady:
    """The write of a cached device table (:func:`_table_ready`): the raw handle of the stream that wrote it and an
    event recorded on that stream after the write."""

    stream: int
    event: torch.cuda.Event


@dataclass(frozen=True)
class ChunkMeta:
    """The chunk and segment tables of one layout.  ``order`` (host route): table sequence ``vs`` is the caller's
    sequence ``order[vs]``, its chunks at their original token offsets (``None``: the caller's order);
    ``order_index`` holds it on the device as ``[order, inverse]`` for the state rows.  ``fwd_groups`` /
    ``bwd_groups``: the two-stream sequence groups of the forward / adjoint scan (``None``: one stream).
    ``bwd_wavefront``: the adjoint scan stage of the wavefront pair where the table's side of the wavefront rule
    holds (:func:`_wavefront_table`; ``None``: the separate launches); ``bwd_tc``: the adjoint table is priced for
    ``dhu_tc2`` (:func:`tc_plan_pricing`, :func:`chain_plan_pricing`), which runs it.  ``ready``: the write of a
    cached table (:func:`_use_table`; ``None``: a table built for its call).  ``fwd_split``: the tables of the
    one-wave split of the forward scan on the host plan's piece table (:class:`ForwardSplit`, with their own write),
    which the forward of :func:`fwd_split_route` runs instead of the pre-scan (``None``: the other plans)."""

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int
    seq_lens: tuple[int, ...] | None
    fwd_segments: SegmentTable
    bwd_segments: SegmentTable
    order: tuple[int, ...] | None = None
    order_index: torch.Tensor | None = None
    fwd_groups: StreamGroups | None = None
    bwd_groups: StreamGroups | None = None
    bwd_wavefront: str | None = None
    bwd_tc: bool = False
    ready: TableReady | None = None
    fwd_split: ForwardSplit | None = None


@dataclass(frozen=True)
class StreamGroups:
    """Two sequence groups of a serial scan: group A is the table's segments ``[0, split)`` (its chunks
    ``[0, split_chunk)``), group B the rest; one is launched on a side stream (:func:`sequence_groups`)."""

    split: int
    split_chunk: int


def segment_bounds(num_chunks: int, segment_len: int) -> list[int]:
    if segment_len <= 0 or num_chunks <= segment_len:
        return [0, num_chunks]
    count = -(-num_chunks // segment_len)
    return [(s * num_chunks) // count for s in range(count + 1)]


def _segment_lists(seq_chunk_start: list[int], segment_len: int):
    seg_chunk_start = [0]
    seg_seq: list[int] = []
    seq_seg_start = [0]
    for seq in range(len(seq_chunk_start) - 1):
        c0 = seq_chunk_start[seq]
        bounds = segment_bounds(seq_chunk_start[seq + 1] - c0, segment_len)
        seg_chunk_start.extend(c0 + b for b in bounds[1:])
        seg_seq.extend([seq] * (len(bounds) - 1))
        seq_seg_start.append(len(seg_seq))
    return seg_chunk_start, seg_seq, seq_seg_start


def split_summary_entries(seq_seg_start) -> list[int]:
    """The segments whose maps the stitch composes, over a segment table whose sequence ``vs`` owns the segments
    ``seq_seg_start[vs] .. seq_seg_start[vs + 1]``: every segment but each sequence's first, in table order (empty
    when no sequence splits)."""
    starts = [int(x) for x in seq_seg_start]
    return [
        seg
        for s0, s1 in zip(starts[:-1], starts[1:], strict=True)
        for seg in range(s0 + 1, s1)
    ]


def lpt_sequence_order(seq_lens) -> tuple[int, ...] | None:
    """The sequences longest first by chunk count, ties in the caller's order (``None`` when that is the caller's
    order)."""
    chunks = [-(-int(x) // CHUNK) for x in seq_lens]
    order = tuple(sorted(range(len(chunks)), key=lambda s: (-chunks[s], s)))
    return None if order == tuple(range(len(chunks))) else order


def lpt_walk(seg_chunk_start) -> list[int] | None:
    """The segments of a segment table longest first by chunk count, ties in table order (``None`` when that is the
    table order)."""
    lens = [
        b - a for a, b in zip(seg_chunk_start[:-1], seg_chunk_start[1:], strict=True)
    ]
    walk = sorted(range(len(lens)), key=lambda s: (-lens[s], s))
    return None if walk == list(range(len(lens))) else walk


def build_chunk_meta(
    seq_lens, device, *, scan_plans: tuple[ScanPlan, ScanPlan], lpt: bool = False
) -> ChunkMeta:
    """Host route: the chunk and segment tables of known sequence lengths, copied to ``device`` in one transfer.

    ``lpt`` dispatches longest first: the tables list the sequences in :func:`lpt_sequence_order` (each chunk keeps its
    token offset; the state rows follow ``order_index``) and ``fwd_ho`` walks the forward segments in
    :func:`lpt_walk` order.  Every CTA computes what it computes in the caller's order: the same bits.  An adjoint table
    that splits a sequence carries the entry table of its segment maps (``SegmentTable.summary``)."""
    lengths = [int(x) for x in seq_lens]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    order = lpt_sequence_order(lengths) if lpt else None
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    for seq in order if order is not None else range(len(lengths)):
        seq_len, token = lengths[seq], offsets[seq]
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
    parts: list[list[int]] = [starts, lens, seq_chunk_start]
    for plan in scan_plans:
        parts.extend(_segment_lists(seq_chunk_start, int(plan.segment_len)))
    walk = lpt_walk(parts[3]) if lpt else None
    extra: dict[str, int] = {}
    if walk is not None:
        extra["walk"] = len(parts)
        parts.append(walk)
    # the adjoint table's map entries (parts[8]: its seq_seg_start)
    summary = split_summary_entries(parts[8])
    if summary:
        extra["summary"] = len(parts)
        parts.append(summary + [SPLIT_WARM] * len(summary))
    if order is not None:
        inverse = [0] * len(order)
        for vs, seq in enumerate(order):
            inverse[seq] = vs
        extra["order_index"] = len(parts)
        parts.append(list(order) + inverse)
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device(device)
    table = (
        table.pin_memory().to(device, non_blocking=True)
        if device.type == "cuda"
        else table.to(device)
    )
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    summary_table = views[extra["summary"]].view(2, -1) if "summary" in extra else None
    seg_index = summary_table[0] if summary_table is not None else None
    seg_mode = summary_table[1] if summary_table is not None else None
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=len(parts[4 + 3 * i]),
            segment_len=int(scan_plans[i].segment_len),
            scan_config=cast("tuple[int, int]", tuple(scan_plans[i].config)),
            walk=views[extra["walk"]] if i == 0 and "walk" in extra else None,
            summary=summary_table if i == 1 else None,
            num_summary=len(summary) if i == 1 else 0,
            seg_index=seg_index if i == 1 else None,
            seg_mode=seg_mode if i == 1 else None,
        )
        for i in range(2)
    ]
    order_index = views[extra["order_index"]].view(2, -1) if order is not None else None
    return ChunkMeta(views[0], views[1], views[2], len(starts), len(lengths), tuple(lengths), segments[0],
                     segments[1], order, order_index)  # fmt: skip


# --------------------------------------------------------------------------- #
# cached device tables
# --------------------------------------------------------------------------- #
# A cached device table (a host-built layout's chunk and segment tables, the wavefront pair's tile order, the identity
# walk of fwd_ho) is written on the stream of the call that builds it, by a non-blocking copy or a launch, and read by
# later calls on any stream.  The build records an event after the write; a hit on another stream waits for that event
# and records the table's use on its stream, so the caching allocator reuses an evicted table only after that stream's
# launches.  A table built inside a CUDA graph capture is written only when the graph replays, so it is not cached.  A
# capture reads a cached table without waiting (a capturing stream cannot wait for an event recorded outside the
# capture), so the table's write must be done when the capture starts: torch.cuda.graph synchronizes the device first.
# That covers only work queued before the capture; a table another thread builds during the capture must have its
# stream synchronized before the capture reads it.  The table is then kept for the process, since every replay reads it.
# The split forward's tables (two allocations: the int32 tables and the int64 indices) carry one write of their own; a
# hit on a layout's ChunkMeta makes them safe too (:func:`_use_meta_tables`), since the forward reads them from it.
_HOST_META: dict = {}
_HOST_META_LIMIT = 256
_FORWARD_SPLITS: dict = {}
_FORWARD_SPLITS_LIMIT = 256
_GRAPH_TABLES: dict[int, torch.Tensor] = {}

# The handle of the current stream of a device without building a ``torch.cuda.Stream`` (that costs microseconds of
# host time per call); ``torch.cuda.current_stream`` where this PyTorch build lacks the getter.
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream_handle(device_index: int) -> int:
    return torch.cuda.current_stream(device_index).cuda_stream


# the handle of the current stream of a device: the C getter itself where PyTorch has it
_raw_stream = _RAW_STREAM if _RAW_STREAM is not None else _current_stream_handle


def _table_ready(device_index: int) -> TableReady:
    """The :class:`TableReady` of a table just written on the current stream."""
    stream = torch.cuda.current_stream(device_index)
    event = torch.cuda.Event()
    event.record(stream)
    return TableReady(stream.cuda_stream, event)


def _use_table(
    table: torch.Tensor, ready: TableReady | None, device_index: int
) -> None:
    """Make a cached table (``table``: any view of it) safe to read by the current stream's next launches: under a
    CUDA graph capture keep it for the process; on a stream other than the one that wrote it, wait for the write and
    record the use.  ``ready`` ``None``: a table built for its call, nothing to do."""
    if ready is None:
        return
    if torch.cuda.is_current_stream_capturing():
        _GRAPH_TABLES.setdefault(table.untyped_storage().data_ptr(), table)
    elif _raw_stream(device_index) != ready.stream:
        stream = torch.cuda.current_stream(device_index)
        stream.wait_event(ready.event)
        table.record_stream(stream)


def _use_forward_split(split: ForwardSplit, device_index: int) -> None:
    """:func:`_use_table` for both allocations of the split forward's tables (:func:`forward_split_tables`)."""
    _use_table(split.seg_chunk_start, split.ready, device_index)
    _use_table(split.first_index, split.ready, device_index)


def _use_meta_tables(meta: ChunkMeta, device_index: int) -> None:
    """:func:`_use_table` for every cached device table a :class:`ChunkMeta` carries: its chunk table and the split
    forward's tables."""
    _use_table(meta.chunk_start, meta.ready, device_index)
    if meta.fwd_split is not None:
        _use_forward_split(meta.fwd_split, device_index)


def _host_chunk_meta(
    seq_lens: tuple[int, ...],
    plans: tuple[ScanPlan, ScanPlan],
    device_index: int,
    lpt: bool,
) -> ChunkMeta:
    """:func:`build_chunk_meta` on the device, cached for the ``_HOST_META_LIMIT`` most recently used keys (outside a
    CUDA graph capture)."""
    key = (seq_lens, plans, device_index, lpt)
    meta = _HOST_META.pop(key, None)
    if meta is not None:
        _HOST_META[key] = meta
        _use_meta_tables(meta, device_index)
        return meta
    meta = build_chunk_meta(
        seq_lens, torch.device("cuda", device_index), scan_plans=plans, lpt=lpt
    )
    if torch.cuda.is_current_stream_capturing():
        return meta
    meta = replace(meta, ready=_table_ready(device_index))
    if len(_HOST_META) >= _HOST_META_LIMIT:
        _HOST_META.pop(next(iter(_HOST_META)), None)
    _HOST_META[key] = meta
    return meta


def segments_upper_bound(num_chunks_max: int, num_seqs: int, segment_len: int) -> int:
    return num_seqs + (
        num_chunks_max // int(segment_len) if int(segment_len) > 0 else 0
    )


def build_chunk_meta_from_cu_seqlens(
    cu_seqlens, total_tokens: int, *, scan_plans
) -> ChunkMeta:
    """Device route: the ``meta`` stage builds the tables on the GPU (padded launch extents)."""
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    seg_max = [
        segments_upper_bound(num_chunks_max, num_seqs, plan.segment_len)
        for plan in scan_plans
    ]
    cu32 = (
        cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    ).contiguous()
    sizes = [num_chunks_max, num_chunks_max, num_seqs + 1]
    for n in seg_max:
        sizes.extend((n + 1, n, num_seqs + 1))
    table = torch.empty(sum(sizes), dtype=torch.int32, device=cu_seqlens.device)
    views = []
    offset = 0
    for n in sizes:
        views.append(table[offset : offset + n])
        offset += n
    _launch("meta", 1, 1, 1, cu32, *views, num_seqs, num_chunks_max, int(total_tokens),
            int(scan_plans[0].segment_len), int(scan_plans[1].segment_len), seg_max[0], seg_max[1])  # fmt: skip
    segments = [
        SegmentTable(
            views[3 + 3 * i],
            views[4 + 3 * i],
            views[5 + 3 * i],
            seg_max[i],
            int(scan_plans[i].segment_len),
            tuple(scan_plans[i].config),
        )  # fmt: skip
        for i in range(2)
    ]
    return ChunkMeta(
        views[0],
        views[1],
        views[2],
        num_chunks_max,
        num_seqs,
        None,
        segments[0],
        segments[1],
    )


# --------------------------------------------------------------------------- #
# scan configuration and segment planner
# --------------------------------------------------------------------------- #
@cache
def _device_props(device_index: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return (
        int(props.multi_processor_count),
        int(props.shared_memory_per_block_optin),
        int(props.major),
    )


def finalize_stage(
    num_chunks: int,
    num_heads: int,
    num_v_heads: int,
    sm_count: int,
    forced: str | None = None,
) -> str:
    """The finalize stage of one backward: ``forced`` when given, else the three-head MLP finalize while its grid
    stays within ``FINALIZE_MLP_MIN_BLOCKS`` CTAs per SM, else the generic finalize."""
    if num_heads <= 0 or num_v_heads % num_heads != 0:
        raise ValueError(
            f"num_v_heads={num_v_heads} must be a positive multiple of num_heads={num_heads}"
        )
    group_ok = num_v_heads == FINALIZE_MLP_GROUP * num_heads
    if forced:
        if forced not in FINALIZE_STAGES:
            raise ValueError(f"finalize={forced!r}")
        if forced == "finalize_mlp" and not group_ok:
            raise ValueError("finalize_mlp is unrolled for a value-head group of 3")
        return forced
    if group_ok and int(num_chunks) * int(num_heads) <= FINALIZE_MLP_MIN_BLOCKS * int(
        sm_count
    ):
        return "finalize_mlp"
    return "finalize"


MIN_SEGMENT_CHUNKS = 4
SCAN_CO_RESIDENT_FACTOR = 1.60
PRESCAN_STEP_FACTOR = 1.16
PRESCAN_WARMUP_CHUNKS = 3.3
STITCH_FIXED_CHUNKS = 2.75
STITCH_COPY_CHUNKS = 2.3
SPLIT_GAIN_THRESHOLD = 0.95
MAX_SCAN_ROUNDS = 2
MAX_SCAN_WAVES_SERIAL = 4
SCAN_STEP_FACTOR = {(32, 2): 1.0, (32, 1): 1.18, (64, 1): 1.27}
WARMUP_THRESH_LOG2 = -10.0 / math.log(2.0)


def warmup_thresh_log2(options: ChainOptions) -> float:
    return (
        options.warmup_log2 if options.warmup_log2 is not None else WARMUP_THRESH_LOG2
    )


@dataclass(frozen=True)
class StageSet:
    """The scan stages a plan may use: register-scan configs per kind, and the configs whose split path (pre-scan +
    split scan for fwd_h, pre-scan + stitch for dhu) is complete.  ``FULL`` is every configuration."""

    reg: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    split: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    label: str

    def reg_of(self, kind):
        return dict(self.reg)[kind]

    def split_of(self, kind):
        return dict(self.split)[kind]


FULL = StageSet(
    reg=(("fwd_h", REG_SCAN_CONFIGS), ("dhu", REG_SCAN_CONFIGS)),
    split=(("fwd_h", PRESCAN_CONFIGS), ("dhu", PRESCAN_CONFIGS)),
    label="all register scans, pre-scan, split, stitch",
)


@cache
def stage_set(arch: str) -> StageSet:
    """The scan stages the registry delivers for ``arch``."""
    reg = []
    split = []
    for kind in SCAN_KINDS:
        reg.append(
            (
                kind,
                tuple(
                    c
                    for c in REG_SCAN_CONFIGS
                    if _available(f"{kind}_r{c[0]}k{c[1]}", arch)
                ),
            )
        )
        ok = []
        for vb, ks in PRESCAN_CONFIGS:
            pre = _available(f"{kind}_pre_r{vb}k{ks}", arch)
            tail = (
                _available(f"fwd_h_r{vb}k{ks}_split", arch)
                if kind == "fwd_h"
                else _available("stitch", arch)
            )
            if pre and tail and _available(f"{kind}_r{vb}k{ks}", arch):
                ok.append((vb, ks))
        split.append((kind, tuple(ok)))
    return StageSet(tuple(reg), tuple(split), f"registry for {arch}")


def _residency(kind, vb, ks, smem_optin) -> int:
    threads = 32 * (vb // 16) * ks
    smem = REG_SCAN_SMEM[(kind, vb, ks)]
    return min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)


def choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, sset: StageSet, pinned=True,
                                forced: tuple[int, int] | None = None):  # fmt: skip
    """The first configuration of ``sset`` in preference order whose grid is co-resident; else (64, 1) if available,
    else the last available configuration.  ``forced`` (honoured when ``pinned``) wins."""
    if pinned and forced is not None:
        return tuple(forced)
    sm_count, smem_optin, _major = _device_props(device_index)
    configs = sset.reg_of(kind)
    for vb, ks in REG_SCAN_CONFIGS:
        if (vb, ks) not in configs:
            continue
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        resident = _residency(kind, vb, ks, smem_optin)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    if (64, 1) in configs:
        return 64, 1
    return configs[-1] if configs else (64, 1)


def _scan_fits(kind, config, device_index, sset: StageSet) -> str | None:
    vb, ks = config
    _sm, smem_optin, major = _device_props(device_index)
    if config not in sset.reg_of(kind):
        return f"no {kind}_r{vb}k{ks} stage in {sset.label}"
    if major < 9:
        return f"compute capability {major}.x has no register scan"
    if REG_SCAN_SMEM[(kind, vb, ks)] > smem_optin:
        return (
            f"{kind} {config} needs {REG_SCAN_SMEM[(kind, vb, ks)]} B of shared memory"
        )
    return None


def _prescan_fits(kind, config, device_index, sset: StageSet) -> str | None:
    if config not in PRESCAN_CONFIGS:
        return f"{config} has no pre-scan stage"
    if config not in sset.split_of(kind):
        return f"no complete split path ({kind}_pre_r{config[0]}k{config[1]} + split/stitch) in {sset.label}"
    return _scan_fits(kind, config, device_index, sset)


def _scan_residency(kind, vb, ks, device_index) -> int:
    _sm, smem_optin, _major = _device_props(device_index)
    return max(1, _residency(kind, vb, ks, smem_optin))


def _wave_factor(ctas: int, sm_count: int, residency: int) -> float:
    co_resident = residency >= 2
    per_wave = sm_count * (2 if co_resident else 1)
    full_factor = SCAN_CO_RESIDENT_FACTOR if co_resident else 1.0
    full, tail = divmod(ctas, per_wave)
    cost = full * full_factor
    if tail:
        cost += full_factor if (full or tail > sm_count) else 1.0
    return cost


def _scan_makespan(chunks, splits, per_segment, sm_count, residency) -> float:
    runs = sorted(
        (-(-n // k), k * per_segment) for n, k in zip(chunks, splits, strict=False)
    )
    slots = sm_count * (2 if residency >= 2 else 1)
    longest = runs[-1][0]
    releases = []
    capacity, waiting = slots, 0
    for steps, ctas in runs:
        resident = min(ctas, capacity)
        capacity -= resident
        if resident:
            releases.append((steps, resident))
        waiting += ctas - resident
    makespan = longest
    while waiting > 0 and releases:
        next_releases = []
        for at, count in releases:
            if waiting <= 0:
                break
            started = min(count, waiting)
            waiting -= started
            makespan = max(makespan, at + longest)
            next_releases.append((at + longest, started))
        releases = next_releases
    factor = (
        SCAN_CO_RESIDENT_FACTOR
        if (residency >= 2 and sum(c for _s, c in runs) > sm_count)
        else 1.0
    )
    return factor * makespan


def bound_layout_chunks(chunks_bound: int, num_seqs: int) -> tuple[int, ...]:
    return (chunks_bound - (num_seqs - 1),) + (1,) * (num_seqs - 1)


def _serial_cost(chunks, per_segment, sm_count, residency, step_factor) -> float:
    return step_factor * _scan_makespan(
        chunks, (1,) * len(chunks), per_segment, sm_count, residency
    )


def split_cta_limit(sm_count: int, residency: int) -> int:
    return (
        MAX_SCAN_ROUNDS * 2 * sm_count
        if residency >= 2
        else MAX_SCAN_WAVES_SERIAL * sm_count
    )


@cache
def _prescan_capacity(kind: str, config: tuple[int, int]) -> int:
    """Chunks one pre-scan CTA of ``kind`` at ``config`` covers (the registry's ``scan.max_segment_chunks``): the
    longest segment the split route can run."""
    vb, ks = config
    for found in cake_jit.MODULES.values():
        if found["stage"] == f"{kind}_pre_r{vb}k{ks}":
            return int(found["scan"]["max_segment_chunks"])
    return 0


def _best_split(chunks, per_segment, sm_count, residency, step_factor, max_segment_len):
    """The cheapest split ``(segment_len, cost)``; a candidate longer than ``max_segment_len`` chunks (the pre-scan
    capacity) is not admissible, so the search keeps the best of the rest (``(0, inf)`` when none is left)."""
    longest = max(chunks)
    limit = split_cta_limit(sm_count, residency)
    best_len, best_cost = 0, math.inf
    for segments in range(2, longest // MIN_SEGMENT_CHUNKS + 1):
        seg_len = -(-longest // segments)
        if seg_len < MIN_SEGMENT_CHUNKS or seg_len >= longest:
            continue
        if seg_len > max_segment_len:
            continue
        splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
        if sum(splits) * per_segment > limit:
            continue
        scan = step_factor * _scan_makespan(
            chunks, splits, per_segment, sm_count, residency
        )
        pre = (
            PRESCAN_WARMUP_CHUNKS
            * PRESCAN_STEP_FACTOR
            * step_factor
            * _wave_factor(
                sum(k - 1 for k in splits) * per_segment, sm_count, residency
            )
        )
        cost = scan + pre + STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        if cost < best_cost:
            best_len, best_cost = seg_len, cost
    return best_len, best_cost


def choose_scan_plan(kind, total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset: StageSet,
                     pinned: bool = True, *, scan_config: tuple[int, int] | None = None,
                     segment_chunks: int | None = None) -> tuple[ScanPlan, dict]:  # fmt: skip
    """The scan plan of one scan kind over ``sset`` and its pricing record.  ``pinned`` honours the
    ``scan_config`` / ``segment_chunks`` pins."""
    forced_len = segment_chunks if pinned else None
    forced_config = scan_config if pinned else None
    baseline = choose_register_scan_config(
        kind, num_seqs, num_v_heads, device_index, sset, pinned, forced_config
    )
    rec: dict[str, Any] = {"baseline": list(baseline)}
    if forced_len is not None:
        unsplittable = _prescan_fits(kind, baseline, device_index, sset)
        if forced_len > 0 and unsplittable is not None:
            warnings.warn(
                f"forced segment length {forced_len} ignored: {unsplittable}",
                RuntimeWarning,
                stacklevel=2,
            )
            rec["forced_seg_ignored"] = unsplittable
            return ScanPlan(baseline, 0), rec
        rec["forced_seg"] = forced_len
        return ScanPlan(baseline, forced_len), rec
    unfit = _scan_fits(kind, baseline, device_index, sset)
    if unfit is not None:
        rec["baseline_unfit"] = unfit
        return ScanPlan(baseline, 0), rec
    is_pinned = forced_config is not None
    chunks_bound = total_tokens // CHUNK + num_seqs
    chunks = (
        tuple(-(-int(x) // CHUNK) for x in seq_lens)
        if seq_lens is not None
        else bound_layout_chunks(chunks_bound, num_seqs)
    )
    sm_count, _smem_optin, _major = _device_props(device_index)
    candidates = (
        [baseline]
        if is_pinned
        else [
            c
            for c in REG_SCAN_CONFIGS
            if _scan_fits(kind, c, device_index, sset) is None
        ]
    )
    baseline_cost = _serial_cost(
        chunks,
        num_v_heads * (HEAD_DIM // baseline[0]),
        sm_count,
        _scan_residency(kind, baseline[0], baseline[1], device_index),
        SCAN_STEP_FACTOR[baseline],
    )
    best = (baseline, 0, baseline_cost)
    priced = []
    for config in candidates:
        per_segment = num_v_heads * (HEAD_DIM // config[0])
        residency = _scan_residency(kind, config[0], config[1], device_index)
        options = [
            (
                0,
                _serial_cost(
                    chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[config]
                ),
            )
        ]
        if _prescan_fits(kind, config, device_index, sset) is None:
            options.append(
                _best_split(
                    chunks,
                    per_segment,
                    sm_count,
                    residency,
                    SCAN_STEP_FACTOR[config],
                    _prescan_capacity(kind, tuple(config)),
                )
            )
        for seg_len, cost in options:
            priced.append(
                [list(config), seg_len, round(cost, 3) if math.isfinite(cost) else None]
            )
            if cost < best[2]:
                best = (config, seg_len, cost)
    rec["priced"] = priced
    rec["baseline_cost"] = round(baseline_cost, 3)
    config, seg_len, cost = best
    if (config, seg_len) != (
        baseline,
        0,
    ) and cost >= SPLIT_GAIN_THRESHOLD * baseline_cost:
        return ScanPlan(baseline, 0), rec
    return ScanPlan(config, seg_len), rec


@lru_cache(maxsize=4096)
def _scan_plans_memo(
    total_tokens,
    num_seqs,
    num_v_heads,
    device_index,
    seq_lens,
    sset,
    pinned,
    props,
    pins,
):
    segment_chunks, scan_config = pins
    f, _ = choose_scan_plan("fwd_h", total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset, pinned,
                            scan_config=scan_config, segment_chunks=segment_chunks)  # fmt: skip
    b, _ = choose_scan_plan("dhu", total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset, pinned,
                            scan_config=scan_config, segment_chunks=segment_chunks)  # fmt: skip
    return f, b


def scan_plans(total_tokens, num_seqs, num_v_heads, device_index, seq_lens=None, sset: StageSet | None = None,
               pinned: bool = True, *, scan_config: tuple[int, int] | None = None,
               segment_chunks: int | None = None) -> tuple[ScanPlan, ScanPlan]:  # fmt: skip
    """The forward and adjoint scan plans of one layout on one device (``seq_lens`` ``None``: the device route's
    bound layout)."""
    sset = sset or stage_set(cake_backend.device_arch(int(device_index)))
    seq_key = None if seq_lens is None else tuple(int(x) for x in seq_lens)
    return _scan_plans_memo(int(total_tokens), int(num_seqs), int(num_v_heads), int(device_index), seq_key, sset,
                            bool(pinned), _device_props(int(device_index)), (segment_chunks, scan_config))  # fmt: skip


# --------------------------------------------------------------------------- #
# workspace pool
# --------------------------------------------------------------------------- #
WORKSPACE_SHAPES_PER_ROLE = 4


class Workspaces:
    def __init__(self):
        self.tables: dict = {}
        # self-cleaning buffers per (device, stream): zero when created, left zero by every launch that uses them (the
        # package tail's arrival counters)
        self.zeroed: dict = {}

    def stream_key(self, device_index: int) -> tuple[int, int] | None:
        """(device, stream): the key of the buffers of the current stream of ``device_index`` (:meth:`table`,
        :meth:`take_zeroed`).  The stream is its handle (:func:`_raw_stream`), the ``cuda_stream`` of
        ``torch.cuda.current_stream(device_index)``."""
        return (device_index, _raw_stream(device_index))

    def take_zeroed(
        self, key, role: str, rows: int, row_shape: tuple, dtype, device
    ) -> torch.Tensor:
        """The first ``rows`` rows of the ``role`` buffer of the stream ``key`` (:meth:`stream_key` of the call's
        stream, so launches that may run concurrently, on other streams, take distinct buffers): created (or grown) by
        ``torch.zeros`` on that stream, then reused without a memset, since every launch leaves it zero."""
        buffers = self.zeroed.setdefault(key, {})
        buffer = buffers.get(role)
        if (
            buffer is None
            or buffer.shape[0] < rows
            or tuple(buffer.shape[1:]) != tuple(row_shape)
        ):
            buffer = buffers[role] = _zeros((rows, *row_shape), dtype, device)
        return buffer if buffer.shape[0] == rows else buffer[:rows]

    def table(self, device_index: int, key: tuple[int, int] | None = None) -> dict:
        """The workspace table of the stream ``key`` (default: the current stream of ``device_index``, under the
        key of :meth:`stream_key`)."""
        if key is None:
            key = self.stream_key(device_index)
        return self.tables.setdefault(key, {})

    def take(self, table, role, shape, dtype, device):
        key = (tuple(shape), dtype)
        buffers = table.setdefault(role, {})
        buffer = buffers.get(key)
        if buffer is None:
            buffer = buffers[key] = torch.empty(shape, dtype=dtype, device=device)
            if len(buffers) > WORKSPACE_SHAPES_PER_ROLE:
                buffers.pop(next(iter(buffers)), None)
        return buffer

    def take_rows(self, table, role, rows, row_shape, dtype, device):
        """The first ``rows`` rows of the one ``role`` buffer of a stream's ``table`` (:meth:`table`): made without a
        fill (:func:`_empty_unfilled`) and grown to the largest call, for rows that every launch writes before it
        reads them (the package tail's FP32 workspace slots)."""
        key = (tuple(row_shape), dtype)
        buffers = table.setdefault(role, {})
        buffer = buffers.get(key)
        if buffer is None or buffer.shape[0] < rows:
            buffers.clear()  # one buffer per role: a larger one replaces it
            buffer = buffers[key] = _empty_unfilled((rows, *row_shape), dtype, device)
        return buffer if buffer.shape[0] == rows else buffer[:rows]


class FreshWorkspaces(Workspaces):
    def stream_key(self, device_index):
        return None  # fresh buffers belong to no stream's pool

    def table(self, device_index, key=None):
        return {}

    def take(self, table, role, shape, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device)

    def take_rows(self, table, role, rows, row_shape, dtype, device):
        return _empty_unfilled((rows, *row_shape), dtype, device)

    def take_zeroed(self, key, role, rows, row_shape, dtype, device):
        return _zeros((rows, *row_shape), dtype, device)


def _empty_unfilled(shape: tuple, dtype, device) -> torch.Tensor:
    """``torch.empty`` with no fill, also under ``torch.use_deterministic_algorithms(True)``, where PyTorch fills each
    new tensor (NaN for floating point) while ``torch.utils.deterministic.fill_uninitialized_memory`` is on, its
    default: for rows that every launch writes before it reads them, where that fill (up to 256 MiB of tail slots per
    uncached call and per CUDA graph replay) changes no value.  The setting is turned off around the allocation only
    and restored."""
    deterministic = torch.utils.deterministic
    if not (
        torch.are_deterministic_algorithms_enabled()
        and deterministic.fill_uninitialized_memory
    ):
        return torch.empty(shape, dtype=dtype, device=device)
    deterministic.fill_uninitialized_memory = False
    try:
        return torch.empty(shape, dtype=dtype, device=device)
    finally:
        deterministic.fill_uninitialized_memory = True


def _zeros(shape: tuple, dtype, device) -> torch.Tensor:
    """``torch.zeros`` on the current stream; an unsigned 32-bit buffer is zeroed as int32 and viewed."""
    if dtype == torch.uint32:
        return torch.zeros(shape, dtype=torch.int32, device=device).view(torch.uint32)
    return torch.zeros(shape, dtype=dtype, device=device)


_POOLED = Workspaces()
_FRESH = FreshWorkspaces()


def _workspaces(options: ChainOptions) -> Workspaces:
    """The pooled per-stream workspaces under ``host_cache``, else fresh buffers.  A CUDA graph capture takes fresh
    buffers (from the graph's memory pool): pooled ones would be shared with the other graphs captured on that stream
    and with eager calls on it, and an eviction would free them while the graph still writes them."""
    if options.host_cache and not torch.cuda.is_current_stream_capturing():
        return _POOLED
    return _FRESH


@cache
def _scan_stages(kind: str, config: tuple[int, int], arch: str):
    """(scan stage, value block, pre-scan stage or None, chunks the pre-scan covers)."""
    vb, ks = config
    stage = f"{kind}_r{vb}k{ks}"
    if not _available(stage, arch):
        raise NotImplementedError(
            f"the {kind} register scan {config} has no record for {arch}"
        )
    pre = f"{kind}_pre_r{vb}k{ks}"
    if not _available(pre, arch):
        return stage, vb, None, 0
    return stage, vb, pre, PRESCAN_BLOCK_THREADS


# --------------------------------------------------------------------------- #
# adjoint-scan selection: dhu_tc2 (dv_local fused) or dv_local + the register scan
# --------------------------------------------------------------------------- #
# The rule: T_tc = makespan steps x t(n), t(n) = t_lat + (t_full - t_lat) x clamp((n - 48) / (SMs - 48), 0, 1) with n
# the resident CTAs (one per (sequence, value head)); T_reg = the planner's serial cost of the register scan at the
# bound-layout plan's configuration x t_unit + tiles x c_dvl / SMs.  dhu_tc2 is selected iff T_tc <= DHU_TC_GAIN T_reg.
DHU_TC_GAIN = 0.95
DHU_TC_LOW_RESIDENT = 48  # resident CTAs up to which a tcgen05 scan step takes t_lat
DHU_TC_CONSTANTS = ("t_lat", "t_full", "t_unit", "c_dvl")
_DHU_AUTO: dict = {}
_DHU_AUTO_LIMIT = 256


@cache
def _dhu_tc_constants(arch: str) -> tuple[tuple[str, float], ...] | None:
    """The ``dhu_tc2`` record's ``select`` constants for ``arch`` (``None`` when the record carries none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "dhu_tc2" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in DHU_TC_CONSTANTS):
                return None
            return tuple((key, float(const[key])) for key in DHU_TC_CONSTANTS)
    return None


def dhu_tc_pricing(chunks, num_v_heads, device_index, sset: StageSet | None = None, *,
                   scan_config: tuple[int, int] | None = None, segment_chunks: int | None = None) -> dict:  # fmt: skip
    """The selection rule's two sides for a serial layout of ``chunks`` (chunks per sequence); ``select`` is the
    answer.  The register path is priced at the configuration of the bound layout's adjoint plan, under the same
    pins as the call."""
    arch = cake_backend.device_arch(int(device_index))
    found = _dhu_tc_constants(arch)
    sm_count, _smem, _major = _device_props(device_index)
    n = min(len(chunks) * num_v_heads, sm_count)
    steps_tc = _scan_makespan(chunks, (1,) * len(chunks), num_v_heads, sm_count, 1)
    rec: dict[str, Any] = {"arch": arch, "steps_tc": steps_tc, "resident": n}
    if found is None:
        rec["missing_constants"] = True
        return rec
    const = dict(found)
    low = DHU_TC_LOW_RESIDENT
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    total = sum(chunks) * CHUNK
    plan, _pricing = choose_scan_plan("dhu", total, len(chunks), num_v_heads, device_index, None,
                                      sset or stage_set(arch), True, scan_config=scan_config,
                                      segment_chunks=segment_chunks)  # fmt: skip
    cfg = cast("tuple[int, int]", tuple(plan.config))
    cost_units = SCAN_STEP_FACTOR[cfg] * _scan_makespan(
        chunks,
        (1,) * len(chunks),
        num_v_heads * (HEAD_DIM // cfg[0]),
        sm_count,
        _scan_residency("dhu", *cfg, device_index),
    )
    tiles = sum(chunks) * num_v_heads
    rec.update(t_n=t_n, T_tc=steps_tc * t_n, reg_config=list(cfg), reg_cost_units=cost_units,
               T_reg=cost_units * const["t_unit"] + tiles * const["c_dvl"] / sm_count)  # fmt: skip
    rec["select"] = rec["T_tc"] <= DHU_TC_GAIN * rec["T_reg"]
    return rec


def _dhu_tc_selected(
    meta: ChunkMeta, num_v_heads: int, device_index: int, options: ChainOptions
) -> bool:
    """Whether a serial plan's adjoint scan is ``dhu_tc2``: the ``dhu_scan`` pin, else the rule on a host-built table
    (a device-built table has no layout on the host: the register path)."""
    if options.dhu_scan is not None:
        return options.dhu_scan == "tc"
    if meta.seq_lens is None:
        return False
    key = (meta.seq_lens, int(num_v_heads), int(device_index), options.pins)
    hit = _DHU_AUTO.get(key) if options.host_cache else None
    if hit is None:
        chunks = tuple(-(-int(x) // CHUNK) for x in meta.seq_lens)
        hit = bool(dhu_tc_pricing(chunks, num_v_heads, device_index, scan_config=options.scan_config,
                                  segment_chunks=options.segment_chunks).get("select", False))  # fmt: skip
        if options.host_cache:
            if len(_DHU_AUTO) >= _DHU_AUTO_LIMIT:
                _DHU_AUTO.pop(next(iter(_DHU_AUTO)), None)
            _DHU_AUTO[key] = hit
    return hit


# --------------------------------------------------------------------------- #
# pass 2 of a split adjoint plan: dhu_tc2 from the stitched segment states, or the register scan
# --------------------------------------------------------------------------- #
# A split plan at ``SDT_CONFIG`` runs the pre-scan and the stitch, then pass 2 either as ``dv_local`` + the register
# scan or on ``dhu_tc2`` (one CTA per (segment, value head), ``dv_local`` fused) from the stitched per-segment ``dht``;
# on the ``dhu_tc2`` form the pre-scan is ``dhu_pre_dvl_r64k1``, which computes the ``dv_local`` tiles of the chunks it
# scans, so no ``dv_local`` stage runs (on ``dhu_tc2_dvl32`` the maps of ``dhu_tc2_dvl32_gm`` take its place where
# :func:`split_map_stage` names them, below).  Both forms give the same bits at the same segment table on
# ``dhu_tc2``; on ``dhu_tc2_dvl32`` (:func:`dhu_tc2_stage`: every value-head group but two on ``DVL32_ARCHS``) dv2
# rounds once, so dk / dv / dg / dbeta / dh0 take other bits than the register pass 2's, and this rule, the ``dhu_scan``
# pin and the table route (a device-built table keeps the register pass 2) choose between them.  The rule prices pass
# 2 the way :func:`dhu_tc_pricing` prices a serial scan: T_tc2 = steps x t(n), steps the makespan of the segment table
# at one CTA per (segment, value head) and one CTA per SM, n = min(segments x HV, SMs) and t(n) the ``dhu_tc2``
# interpolation above; T_reg2 = SCAN_STEP_FACTOR[SDT_CONFIG] x the planner makespan of the register pass 2 x t_unit +
# tiles x c_dvl / SMs (the ``dv_local`` stage that form needs, tiles = chunks x HV).  ``dhu_tc2`` runs pass 2 iff T_tc2
# <= sdt_gain x T_reg2 and T_reg2 - T_tc2 >= sdt_min_saving (microseconds), both per-architecture constants of the
# ``dhu_tc2`` record's ``select`` (an architecture without them keeps the register pass 2).  A device-built table has no
# layout on the host: the register pass 2.
SDT_CONFIG = (64, 1)
SDT_PRESCAN = "dhu_pre_dvl_r64k1"
SDT_CONSTANTS = ("sdt_gain", "sdt_min_saving")
_SDT_AUTO: dict = {}
_SDT_AUTO_LIMIT = 256


@cache
def _sdt_constants(arch: str) -> tuple[float, float] | None:
    """``(sdt_gain, sdt_min_saving)`` of the ``dhu_tc2`` record's ``select`` for ``arch`` (``None``: none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "dhu_tc2" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in SDT_CONSTANTS):
                return None
            return float(const["sdt_gain"]), float(const["sdt_min_saving"])
    return None


def sdt_pricing(chunks, segment_len: int, num_v_heads: int, device_index: int) -> dict:
    """The rule's two sides for pass 2 of a split ``SDT_CONFIG`` plan of ``segment_len`` chunks over ``chunks`` (chunks
    per sequence); ``select`` is the answer."""
    arch = cake_backend.device_arch(int(device_index))
    sm_count, _smem, _major = _device_props(device_index)
    splits = tuple(max(1, -(-int(c) // int(segment_len))) for c in chunks)
    n_tc = sum(splits) * int(num_v_heads)
    rec: dict[str, Any] = {
        "arch": arch,
        "segment_len": int(segment_len),
        "segments": sum(splits),
        "tc2_ctas": n_tc,
    }
    found, sdt = _dhu_tc_constants(arch), _sdt_constants(arch)
    if found is None or sdt is None:
        rec["missing_constants"] = True
        rec["select"] = False
        return rec
    const = dict(found)
    gain, min_saving = sdt
    vb, ks = SDT_CONFIG
    reg_units = SCAN_STEP_FACTOR[SDT_CONFIG] * _scan_makespan(chunks, splits, int(num_v_heads) * (HEAD_DIM // vb),
                                                              sm_count, _scan_residency("dhu", vb, ks, device_index))  # fmt: skip
    steps_tc = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    n = min(n_tc, sm_count)
    low = DHU_TC_LOW_RESIDENT
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    tiles = sum(chunks) * int(num_v_heads)
    t_dvl = tiles * const["c_dvl"] / sm_count
    t_tc2, t_reg2 = steps_tc * t_n, reg_units * const["t_unit"] + t_dvl
    rec.update(steps_tc=steps_tc, t_n=t_n, T_tc2=t_tc2, reg2_units=reg_units, T_dvl=t_dvl, T_reg2=t_reg2,
               saving=t_reg2 - t_tc2, gain=gain, min_saving=min_saving)  # fmt: skip
    rec["select"] = t_tc2 <= gain * t_reg2 and t_reg2 - t_tc2 >= min_saving
    return rec


def _sdt_selected(meta: ChunkMeta, segs: SegmentTable, num_v_heads: int, device_index: int, options: ChainOptions,
                  arch: str) -> bool:  # fmt: skip
    """Whether a split adjoint plan runs pass 2 on ``dhu_tc2``: a split ``SDT_CONFIG`` plan of a host-built table,
    the ``dhu_scan`` pin, else the rule (see above)."""
    if options.dhu_scan == "reg" or meta.seq_lens is None:
        return False
    if not (segs.segment_len > 0 and segs.num_segments > meta.num_seqs):
        return False
    if tuple(segs.scan_config) != SDT_CONFIG or not (
        _available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)
    ):
        return False
    if options.dhu_scan == "tc":
        return True
    key = (meta.seq_lens, int(num_v_heads), int(device_index), int(segs.segment_len))
    hit = _SDT_AUTO.get(key) if options.host_cache else None
    if hit is None:
        chunks = tuple(-(-int(x) // CHUNK) for x in meta.seq_lens)
        hit = bool(
            sdt_pricing(chunks, segs.segment_len, num_v_heads, device_index)["select"]
        )
        if options.host_cache:
            if len(_SDT_AUTO) >= _SDT_AUTO_LIMIT:
                _SDT_AUTO.pop(next(iter(_SDT_AUTO)), None)
            _SDT_AUTO[key] = hit
    return hit


# --------------------------------------------------------------------------- #
# the segment maps of a split pass 2 on dhu_tc2_dvl32: dhu_tc2_dvl32_gm in place of the register pre-scan
# --------------------------------------------------------------------------- #
# A split plan of a value-head group of three whose pass 2 runs on ``dhu_tc2_dvl32`` (:func:`_sdt_selected` on
# ``DVL32_ARCHS``) forms the segment maps the stitch composes with ``dhu_tc2_dvl32_gm`` on the architectures of
# ``ADJOINT_MAP_ARCHS`` (off value ``()``) instead of the register pre-scan ``SDT_PRESCAN``: one CTA per (entry, value
# head) over the table's ``summary`` entries (every segment but each sequence's first, whose map the stitch never
# reads; built on the host with the table, mode ``SPLIT_WARM``).  Each CTA walks two states in one pass with the
# tcgen05 scan of ``dhu_tc2_dvl32`` and stores the segment's map ``[B | M]`` into ``seg_state`` (``B`` from zero with
# ``do``, ``M`` from the identity) with the pre-scan's warm-up threshold (:func:`warmup_thresh_log2`): where the
# running sum of a segment's log2 chunk decays from its start falls below the threshold before its last chunk, the map
# keeps only the chunks up to there, ``B`` alone, and ``seg_flag`` is 0; elsewhere the whole segment and 1.  The rule
# is not the pre-scan's bound: where the sum first falls below the threshold in the last chunk the pre-scan keeps
# ``B`` alone (``seg_flag`` 0) and this map is whole, and from a segment's 65th chunk on the running sum adds the
# chunk decays in another order than the pre-scan's, so at a near tie the two can end a map at different chunks.  The
# workspaces, the stitch, pass 2 (one launch, or the two groups beside the fused tail), the tails and the tables are
# unchanged; the table prices of :func:`tc_plan_pricing` and :func:`chain_plan_pricing` still price the register
# pre-scan.  The maps add the FP32 ``dv_local`` values where the pre-scan rounds its tiles to BF16, so where a map
# carries state into the segment before it (slowly decaying gates) dk / dv / dg / dbeta / dh0 take other bits than
# with the pre-scan; o, dq and the final state keep theirs, and each form is bitwise run to run.  The stage follows
# host facts only (:func:`split_map_stage`, :func:`split_map_launch`); the maps' extent follows the gates through the
# warm-up threshold, as the pre-scan's does.
ADJOINT_MAP_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
ADJOINT_MAP_STAGE = "dhu_tc2_dvl32_gm"
# The same maps over two CTAs per (entry, value head) (``dhu_tc2_dvl32_gm2``, grid z 2: ``B`` and ``seg_flag`` on z 0,
# ``M`` on z 1, each CTA with gm's products, roundings and orders on its half: the same bits) on the architectures of
# ``ADJOINT_MAP_SPLIT_ARCHS`` (off value ``()``: gm for every grid that takes a map form; on the architectures of
# ``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`` a grid that fits one wave still keeps the pre-scan while that switch is on),
# where its ``2 x entries x HV`` CTAs fit one wave of the SMs (:func:`split_map_launch`).  The bound is measured: past
# one wave gm2 is still faster than gm where the gates decay slowly (whole maps), but slower where they decay fast (maps
# that end within their first chunks); the gates are on the device and the rule reads host facts only, so a grid of
# more than one wave keeps gm.
ADJOINT_MAP_SPLIT_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
ADJOINT_MAP_SPLIT_STAGE = "dhu_tc2_dvl32_gm2"
# On the architectures of ``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`` (off value ``()``: the maps on every grid) a grid
# that fits one wave keeps the fused pre-scan ``SDT_PRESCAN`` instead of either map form, the 0.2.7 route with its
# bits.  Measured on ``sm_107a`` at such a grid, gm2 ran faster than the pre-scan where the gates decay slowly (whole
# maps) and slower where they decay fast (maps that end within their first chunks); the gates are on the device and
# the rule reads host facts only, so such a grid keeps the pre-scan under every gate, and gives up gm2's gain where
# the gates decay slowly.
ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS: tuple[str, ...] = ("sm_107a",)
# the mode bit of a map entry: bound the map by the warm-up threshold and write its seg_flag
SPLIT_WARM = 16


def split_map_launch(
    arch: str, num_summary: int, num_v_heads: int, sm_count: int
) -> tuple[str, int] | None:
    """The stage and grid z of the segment maps that :func:`split_map_stage` names ``ADJOINT_MAP_STAGE``, over
    ``num_summary`` entries of ``num_v_heads`` value heads on ``sm_count`` SMs.  Where ``2 x num_summary x
    num_v_heads`` CTAs fit one wave (one CTA per SM): None (no map launch: the fused pre-scan ``SDT_PRESCAN`` forms
    the maps) on an architecture of ``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS``, else ``(ADJOINT_MAP_SPLIT_STAGE, 2)`` on
    one of ``ADJOINT_MAP_SPLIT_ARCHS`` whose registry delivers it; every other grid ``(ADJOINT_MAP_STAGE, 1)``.  Both
    map forms give the same bits.  Host facts only.  The most entries of the one-wave answer, and that answer, are
    decided once per (switches, architecture, value heads, SMs) and cached (:func:`_split_map_entries`), so a call
    compares two integers."""
    most, one_wave = _split_map_entries(
        ADJOINT_MAP_SPLIT_ARCHS,
        ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS,
        arch,
        num_v_heads,
        sm_count,
    )
    if num_summary <= most:
        return one_wave
    return ADJOINT_MAP_STAGE, 1


@cache
def _split_map_entries(
    split_archs: tuple[str, ...],
    prescan_archs: tuple[str, ...],
    arch: str,
    num_v_heads: int,
    sm_count: int,
) -> tuple[int, tuple[str, int] | None]:
    """The most map entries whose two-CTA grid (``2 x entries x num_v_heads`` CTAs) fits one wave of ``sm_count`` SMs
    and :func:`split_map_launch`'s answer there: None where ``prescan_archs`` names ``arch``, else
    ``(ADJOINT_MAP_SPLIT_STAGE, 2)`` where ``split_archs`` names ``arch`` and its registry delivers the stage; else
    ``(-1, None)`` (every grid keeps ``ADJOINT_MAP_STAGE``).  ``split_archs`` and ``prescan_archs`` are the
    switches' values."""
    most = int(sm_count) // (2 * int(num_v_heads))
    if arch in prescan_archs:
        return most, None
    if arch in split_archs and _available(ADJOINT_MAP_SPLIT_STAGE, arch):
        return most, (ADJOINT_MAP_SPLIT_STAGE, 2)
    return -1, None


def split_map_stage(arch: str, num_heads: int, num_v_heads: int) -> str:
    """The stage that forms the segment maps of a split plan whose pass 2 runs on the tcgen05 scan:
    ``ADJOINT_MAP_STAGE`` for a value-head group of three whose scan is ``dhu_tc2_dvl32`` (:func:`dhu_tc2_stage`) on an
    architecture of ``ADJOINT_MAP_ARCHS`` whose registry delivers it, else the fused pre-scan ``SDT_PRESCAN``.  The
    maps' grid then decides their launch (:func:`split_map_launch`: on ``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS`` a grid
    that fits one wave keeps the pre-scan)."""
    if (
        num_v_heads == 3 * num_heads
        and arch in ADJOINT_MAP_ARCHS
        and dhu_tc2_stage(arch, num_heads, num_v_heads) == "dhu_tc2_dvl32"
        and _available(ADJOINT_MAP_STAGE, arch)
    ):
        return ADJOINT_MAP_STAGE
    return SDT_PRESCAN


@lru_cache(maxsize=1024)
def _split_map_call(
    map_archs: tuple[str, ...],
    dvl32_archs: tuple[str, ...],
    split_archs: tuple[str, ...],
    prescan_archs: tuple[str, ...],
    arch: str,
    num_heads: int,
    num_v_heads: int,
    num_summary: int,
    sm_count: int,
) -> tuple[str, int] | None:
    """The map launch of a backward whose split pass 2 runs on the tcgen05 scan over a table of ``num_summary`` map
    entries: :func:`split_map_launch` where :func:`split_map_stage` names ``ADJOINT_MAP_STAGE``, else None (the fused
    pre-scan forms the maps).  ``map_archs``, ``dvl32_archs``, ``split_archs`` and ``prescan_archs`` are the values of
    the switches those rules read (``ADJOINT_MAP_ARCHS``, ``DVL32_ARCHS``, ``ADJOINT_MAP_SPLIT_ARCHS``,
    ``ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS``), so the answer is decided once per (switches, architecture, heads, entries,
    SMs) and a backward reads it with one lookup."""
    if split_map_stage(arch, num_heads, num_v_heads) != ADJOINT_MAP_STAGE:
        return None
    return split_map_launch(arch, num_summary, num_v_heads, sm_count)


# --------------------------------------------------------------------------- #
# the serial adjoint plan of a split layout: dhu_tc2 over whole sequences instead of the pre-scan, the stitch and pass 2
# --------------------------------------------------------------------------- #
# The segment planner prices the register scans only; with dhu_tc2 the serial scan can beat a split plan as it runs
# (the pre-scan, the stitch and pass 2, on dhu_tc2 where :func:`sdt_pricing` selects it).  The rule, on a host-built
# table of a value-head group of two (``HV = 2 H``: both plans feed ``tail_g2`` beside the scan) without plan pins,
# replaces a split adjoint plan by the serial one when T_ser <= T_cur and the serial plan's own rule
# (:func:`dhu_tc_pricing`) runs it on dhu_tc2.  T_ser = :func:`dhu_tc_pricing`'s ``T_tc``; T_cur = the
# split plan's pre-scan and stitch in planner units (:func:`_best_split`'s terms) x t_unit plus the dhu_tc2 pass 2's
# ``T_tc2`` where :func:`sdt_pricing` selects it, else the plan's planner units x t_unit + tiles x c_dvl / SMs.
# The rule is on for SERIAL_SCAN_ARCHS: on sm_100a and sm_103a the split pass 2 can run as the wavefront pair beside
# tail_g2, an overlap these prices do not see.  The forward plan is unchanged; the adjoint scan's dk / dv / dg / dbeta take other bits than the split plan's (bitwise run to run
# for the layout).  The decision is made with the chunk tables (:func:`_chunk_meta`) and cached with them.
SERIAL_SCAN_ARCHS = ("sm_107a",)


def serial_scan_pricing(
    chunks, plan: ScanPlan, num_v_heads: int, device_index: int
) -> dict:
    """The serial-scan rule's two sides for the split adjoint ``plan`` of a host layout of ``chunks`` (chunks per
    sequence); ``select`` is the answer."""
    arch = cake_backend.device_arch(int(device_index))
    sm_count, _smem, _major = _device_props(device_index)
    cfg = cast("tuple[int, int]", tuple(plan.config))
    seg_len = int(plan.segment_len)
    rec: dict[str, Any] = {
        "arch": arch,
        "config": list(cfg),
        "segment_len": seg_len,
        "select": False,
    }
    found = _dhu_tc_constants(arch)
    if found is None or arch not in SERIAL_SCAN_ARCHS or seg_len <= 0:
        rec["missing_constants"] = True
        return rec
    const = dict(found)
    vb, ks = cfg
    splits = tuple(max(1, -(-int(c) // seg_len)) for c in chunks)
    per_segment = int(num_v_heads) * (HEAD_DIM // vb)
    residency = _scan_residency("dhu", vb, ks, device_index)
    scan = SCAN_STEP_FACTOR[cfg] * _scan_makespan(
        chunks, splits, per_segment, sm_count, residency
    )
    pre = (
        PRESCAN_WARMUP_CHUNKS
        * PRESCAN_STEP_FACTOR
        * SCAN_STEP_FACTOR[cfg]
        * _wave_factor(sum(k - 1 for k in splits) * per_segment, sm_count, residency)
    )
    stitch = STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
    pass2 = (
        sdt_pricing(chunks, seg_len, num_v_heads, device_index)
        if cfg == SDT_CONFIG
        and _available("dhu_tc2", arch)
        and _available(SDT_PRESCAN, arch)
        else {"select": False}
    )
    sdt = bool(pass2["select"])
    if sdt:
        t_cur = (pre + stitch) * const["t_unit"] + pass2["T_tc2"]
    else:
        t_dvl = sum(chunks) * int(num_v_heads) * const["c_dvl"] / sm_count
        t_cur = (scan + pre + stitch) * const["t_unit"] + t_dvl
    serial = dhu_tc_pricing(chunks, num_v_heads, device_index)
    rec.update(segments=sum(splits), pass2_tc=sdt, T_cur=t_cur, T_ser=serial["T_tc"],
               serial_tc=bool(serial["select"]))  # fmt: skip
    rec["select"] = rec["serial_tc"] and serial["T_tc"] <= t_cur
    return rec


def _serial_adjoint_plan(
    seq_lens: tuple[int, ...],
    plans: tuple[ScanPlan, ScanPlan],
    num_heads: int | None,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions,
) -> tuple[ScanPlan, ScanPlan]:
    """``plans`` with the adjoint plan made serial where the serial-scan rule selects it (see above): a value-head
    group of two, no ``scan_config`` / ``segment_chunks`` / ``dhu_scan`` pin, a split adjoint plan."""
    bwd = plans[1]
    if (
        num_heads is None
        or int(num_v_heads) != 2 * int(num_heads)
        or options.scan_config is not None
        or options.segment_chunks is not None
        or options.dhu_scan is not None
        or bwd.segment_len <= 0
    ):
        return plans
    chunks = tuple(-(-int(x) // CHUNK) for x in seq_lens)
    if sum(max(1, -(-c // bwd.segment_len)) for c in chunks) <= len(chunks):
        return plans
    if not serial_scan_pricing(chunks, bwd, int(num_v_heads), device_index)["select"]:
        return plans
    return plans[0], ScanPlan(bwd.config, 0)


# --------------------------------------------------------------------------- #
# the adjoint table priced for dhu_tc2 (host route)
# --------------------------------------------------------------------------- #
# The register planner prices the adjoint table for the register scans (two CTAs per (segment, value head) at
# ``v_block`` 64, their step and co-residency factors); the pass-2 rules then run ``dhu_tc2`` on THAT table.  With
# ``TC_PLAN`` a host-built table without plan pins whose planner configuration is ``SDT_CONFIG`` may instead take the
# table priced for ``dhu_tc2`` itself.  Candidates: the serial table and the ``SDT_CONFIG`` split tables of ``L``
# chunks (``MIN_SEGMENT_CHUNKS`` to the pre-scan capacity), each priced (microseconds) T = makespan(segments x HV CTAs,
# one per SM) x t(n) (the ``dhu_tc2`` interpolation of :func:`sdt_pricing` over ``TC_PLAN_STEP``) + the register
# planner's pre-scan and stitch terms x ``t_unit`` + the ``dv_local`` tiles of the pre-scan's warm-up window (on an
# architecture of ``TC_PLAN_PRESCAN``: the stitch term + the pre-scan of a value head that scans its whole segment).  The
# planner's table (after the serial-scan rule) is priced in the form its rules run (tcgen05, or the register pass
# with ``dv_local``), and the cheapest candidate replaces it when it saves at least ``min_saving`` us and prices at
# most ``gain`` x the planner's table (:func:`tc_plan_thresholds`); the backward then runs the candidate on
# ``dhu_tc2``.  A table whose scan runs as two groups beside the fused tail keeps its form (:func:`_two_stream_kept`):
# its scan overlaps the tail, which the pricing does not credit.  Another table moves the warm-up truncation and the
# stitch points, so the bits follow the plan (bitwise run to run for the layout); plan pins keep the planner's table.
TC_PLAN = True
TC_PLAN_GAIN = 0.95
TC_PLAN_MIN_SAVING = 10.0
# (gain, min_saving) of an architecture in place of (TC_PLAN_GAIN, TC_PLAN_MIN_SAVING) (:func:`tc_plan_thresholds`).
# R200 measured the priced table against the planner's (backward): faster at 4.2-5.6 us priced savings (-0.9 % to
# -3.2 %), slower at 0.9 us (+0.35 %).
TC_PLAN_SELECT = {"sm_107a": (1.0, 2.0)}
# The dhu_tc2 step time (microseconds) at low and full residency that prices these tables, measured (CUPTI kernel
# durations of L = 64 tables, 3 to 288 CTAs: B200 2.32-2.44, GB300 2.15-2.26 per step); an architecture not listed takes
# the dhu_tc2 record's t_lat / t_full.
TC_PLAN_STEP = {"sm_100a": (2.35, 2.44), "sm_103a": (2.17, 2.25)}
# The pre-scan of a split table scans each segment but one of its sequence until the decay sum passes
# ``warmup_thresh_log2``: a value head that decays slowly scans its whole segment, and the launch lasts as long as its
# slowest CTA.  On an architecture listed here the candidates' pre-scan is priced for that head (microseconds):
# ``fixed`` + the longest pre-scanned segment x ``step`` x w, w = 1 while its CTAs (pre-scanned segments x HV x
# HEAD_DIM / v_block) fit one per SM, else ``co_resident`` x their rounds of two per SM.  Measured (CUPTI kernel sums of
# the pre-scan, dhu_tc2 and stitch of pinned tables of 11000 / 12000 / 9768 tokens over H = 1 to 8 value-head groups
# of three, per-head decay rates 1e-4 to 1 per token): the warm-up price above (a few chunks per segment) priced the
# B200 pre-scan of the 94-chunk table over 24 value heads at 20 us against 310 us measured, and took that table in
# place of the serial one at +8 % backward; this price picks the measured fastest table of each layout on B200 and
# GB300 when, as there, a split candidate is admissible only if its dhu_tc2 CTAs (segments x HV) fit one per SM (every
# measured table of two rounds was slower than the best table of one).  Fast decay in every head ends the pre-scan
# within a few chunks, so these prices then over-state the split tables.  An architecture not listed keeps the warm-up
# price and every candidate (R200 prices its split tables with ``CHAIN_PLAN``).  The planner's table in the register
# form is priced with the same pre-scan term.
TC_PLAN_PRESCAN = {"sm_100a": (10.0, 2.95, 1.55), "sm_103a": (25.0, 2.5, 1.55)}


def tc_plan_thresholds(arch: str) -> tuple[float, float]:
    """``(gain, min_saving)`` that select the tc-priced adjoint table on ``arch``."""
    return TC_PLAN_SELECT.get(arch, (TC_PLAN_GAIN, TC_PLAN_MIN_SAVING))


def _tc_table_price(
    chunks,
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    const: dict,
    pre_residency: int,
    prescan: tuple[float, float, float] | None = None,
) -> dict:
    """The priced ``dhu_tc2`` form of one adjoint table (``segment_len`` 0 or not below the longest sequence: serial);
    ``prescan`` the architecture's ``TC_PLAN_PRESCAN`` constants (``None``: the warm-up price)."""
    if segment_len <= 0 or max(chunks) <= segment_len:
        splits = (1,) * len(chunks)
    else:
        splits = tuple(max(1, -(-int(n) // int(segment_len))) for n in chunks)
    ctas = sum(splits) * int(num_v_heads)
    steps = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    low = DHU_TC_LOW_RESIDENT
    n = min(ctas, sm_count)
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    extra = 0.0
    if max(splits) > 1:
        tails = sum(k - 1 for k in splits) * int(num_v_heads)
        pre_ctas = tails * (HEAD_DIM // SDT_CONFIG[0])
        stitch = STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        if prescan is not None:
            fixed, step, co_resident = prescan
            pre_steps = max(
                -(-int(c) // k) for c, k in zip(chunks, splits, strict=True) if k > 1
            )
            rounds = (
                1.0
                if pre_ctas <= sm_count
                else co_resident * -(-pre_ctas // (2 * sm_count))
            )
            extra = fixed + pre_steps * step * rounds + stitch * const["t_unit"]
        else:
            units = (
                PRESCAN_WARMUP_CHUNKS * PRESCAN_STEP_FACTOR * SCAN_STEP_FACTOR[SDT_CONFIG]
                * _wave_factor(pre_ctas, sm_count, pre_residency) + stitch
            )  # fmt: skip
            extra = (
                units * const["t_unit"]
                + tails * PRESCAN_WARMUP_CHUNKS * const["c_dvl"] / sm_count
            )
    return {"segment_len": int(segment_len) if max(splits) > 1 else 0, "segments": sum(splits), "ctas": ctas,
            "steps": steps, "T_tc2": steps * t_n, "T_extra": extra, "T": steps * t_n + extra}  # fmt: skip


@lru_cache(maxsize=1024)
def _tc_plan_memo(
    chunks: tuple[int, ...],
    num_v_heads: int,
    device_index: int,
    plan: ScanPlan,
    props,
    arch: str,
    _gain: float,
    _min_saving: float,
    prescan: tuple[float, float, float] | None = None,
) -> dict:
    rec: dict[str, Any] = {
        "arch": arch,
        "select": False,
        "planner": [list(plan.config), plan.segment_len],
    }
    found = _dhu_tc_constants(arch)
    if found is None or tuple(plan.config) != SDT_CONFIG:
        return rec
    if not (_available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)):
        return rec
    const = dict(found)
    if arch in TC_PLAN_STEP:
        const["t_lat"], const["t_full"] = TC_PLAN_STEP[arch]
    sm_count = props[0]
    pre_res = _scan_residency("dhu", *SDT_CONFIG, device_index)
    serial = plan.segment_len <= 0 or max(chunks) <= plan.segment_len
    at = _tc_table_price(
        chunks,
        num_v_heads,
        0 if serial else plan.segment_len,
        sm_count,
        const,
        pre_res,
        prescan,
    )
    if serial:
        rule = dhu_tc_pricing(chunks, num_v_heads, device_index)
        plan_t = (
            at["T"] if rule.get("select") else min(at["T"], rule.get("T_reg", math.inf))
        )
    else:
        rule = sdt_pricing(chunks, plan.segment_len, num_v_heads, device_index)
        plan_t = (
            at["T"]
            if rule.get("select")
            else at["T_extra"] + rule.get("T_reg2", math.inf)
        )
    cap = _prescan_capacity("dhu", SDT_CONFIG)
    cands = [_tc_table_price(chunks, num_v_heads, 0, sm_count, const, pre_res, prescan)]
    for seg_len in range(MIN_SEGMENT_CHUNKS, min(cap, max(chunks) - 1) + 1):
        cand = _tc_table_price(
            chunks, num_v_heads, seg_len, sm_count, const, pre_res, prescan
        )
        if prescan is not None and cand["segment_len"] and cand["ctas"] > sm_count:
            continue  # TC_PLAN_PRESCAN: a split table runs its dhu_tc2 CTAs in one round
        cands.append(cand)
    best = min(cands, key=lambda c: (c["T"], -c["segment_len"]))
    rec.update(plan_T=plan_t, best=best, saving=plan_t - best["T"])
    rec["select"] = bool(
        best["segment_len"] != (0 if serial else plan.segment_len)
        and plan_t - best["T"] >= _min_saving
        and best["T"] <= _gain * plan_t
    )
    rec["segment_len"] = best["segment_len"]
    return rec


def tc_plan_pricing(
    chunks, num_v_heads: int, device_index: int, plan: ScanPlan
) -> dict:
    """The adjoint table priced for ``dhu_tc2`` against the planner's ``plan`` for a host layout of ``chunks`` (chunks
    per sequence); ``select`` is the answer and ``segment_len`` the table (0: serial).  See above."""
    arch = cake_backend.device_arch(int(device_index))
    gain, min_saving = tc_plan_thresholds(arch)
    return dict(
        _tc_plan_memo(
            tuple(int(c) for c in chunks),
            int(num_v_heads),
            int(device_index),
            plan,
            _device_props(int(device_index)),
            arch,
            gain,
            min_saving,
            TC_PLAN_PRESCAN.get(arch),
        )  # fmt: skip
    )


# --------------------------------------------------------------------------- #
# deeper split tables priced for the tcgen05 chains (fwd_ho, dhu_tc2), per architecture
# --------------------------------------------------------------------------- #
# The register planner prices the forward table for the register scans (``HEAD_DIM / v_block`` CTAs per (segment, value
# head), two per SM), but ``fwd_ho`` runs it with one CTA per (segment, value head) and one per SM; the adjoint table
# that ``TC_PLAN`` keeps can likewise stay shallower than the ``dhu_tc2`` chain needs on a device with many SMs.  With
# ``TC_PLAN`` (False keeps the planner's tables in both directions), on an architecture of ``CHAIN_PLAN``, a host-built
# table without plan pins whose scan does not run as two groups (:func:`_two_stream_kept`) may take a DEEPER
# ``SDT_CONFIG`` split table (more segments than the chosen table) for each direction, priced with the architecture's
# measured constants (microseconds): T = makespan(segments x HV CTAs, one per SM) x t(n), t(n) = t_lat + (t_full -
# t_lat) x clamp((n - 48) / (SMs - 48), 0, 1) for n = min(CTAs, SMs) resident, plus, for a split table, ``fixed`` +
# ``per_segment`` x (most segments of a sequence - 1) + ``per_wave`` x the pre-scan's waves (segments x HV x 2
# HEAD_DIM / v_block CTAs over 2 x SMs).  The cheapest deeper table replaces the chosen one when it saves at least
# ``min_saving`` and prices at most ``gain`` x the chosen table; the forward then runs the pre-scan, the stitch and
# ``fwd_ho`` on it, the backward the fused pre-scan, the stitch and ``dhu_tc2``.  The constants are R200 measurements
# (212 SMs, ``fwd_ho`` / ``dhu_tc2`` step times of serial tables of 48 to 208 CTAs, split costs fitted to pinned split
# tables of 12 rows; the fit's error per table: forward median 1.8 %, up to 11.7 %, adjoint 0.9 %, up to 4.7 %, hence
# ``gain`` 0.95); an architecture not listed keeps its tables.  Another table moves the segment boundaries, so the
# bits follow the plan (plan pins keep the planner's tables).  On an architecture of ``FWD_HO_VH2_ARCHS`` the forward
# prices a table whose launch runs the value-half split (:func:`fwd_ho_vh2_engaged`: twice its segments x HV CTAs fit
# one round) with its chain term scaled by ``FWD_HO_VH2_STEP``, the split's chain time over ``fwd_ho``'s on the same
# table (R200, the 96-CTA table of one 16384-token sequence over 16 value heads: 61.44 against 81.76 us, 0.7515); the
# deeper tables the split does not fit keep the ``fwd_ho`` price.
CHAIN_PLAN_CONSTANTS = ("t_lat", "t_full", "fixed", "per_segment", "per_wave")
CHAIN_PLAN: dict[str, dict[str, Any]] = {
    "sm_107a": {
        "fwd": (1.77, 2.31, 0.0, 1.7, 6.5),
        "bwd": (1.66, 1.88, 0.0, 0.6, 10.8),
        "gain": 0.95,
        "min_saving": 2.0,
    },
}
FWD_HO_VH2_STEP = 0.75


def _chain_table_price(
    chunks,
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    const: tuple,
    vh2: float = 1.0,
) -> float:
    """The priced tcgen05 chain of one table (``segment_len`` 0 or not below the longest sequence: serial); ``vh2``
    scales the chain term of a table whose launch runs the value-half split (:func:`fwd_ho_vh2_engaged`)."""
    t_lat, t_full, fixed, per_segment, per_wave = const
    if segment_len <= 0 or max(chunks) <= segment_len:
        splits = (1,) * len(chunks)
    else:
        splits = tuple(max(1, -(-int(n) // int(segment_len))) for n in chunks)
    ctas = sum(splits) * int(num_v_heads)
    steps = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    n = min(ctas, sm_count)
    low = DHU_TC_LOW_RESIDENT
    t_n = t_lat + (t_full - t_lat) * min(max((n - low) / (sm_count - low), 0.0), 1.0)
    extra = 0.0
    if max(splits) > 1:
        pre_ctas = ctas * (2 * HEAD_DIM // SDT_CONFIG[0])
        extra = (
            fixed
            + per_segment * (max(splits) - 1)
            + per_wave * -(-pre_ctas // (2 * sm_count))
        )
    chain = steps * t_n
    if vh2 != 1.0 and fwd_ho_vh2_engaged(sum(splits), num_v_heads, sm_count):
        chain *= vh2
    return chain + extra


@lru_cache(maxsize=1024)
def _chain_plan_memo(
    kind: str,
    chunks: tuple[int, ...],
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    arch: str,
    const: tuple,
    gain: float,
    min_saving: float,
    vh2: float,
) -> dict:
    current = 0 if segment_len <= 0 or max(chunks) <= segment_len else int(segment_len)
    segments = sum(max(1, -(-int(n) // current)) if current else 1 for n in chunks)
    plan_t = _chain_table_price(chunks, num_v_heads, current, sm_count, const, vh2)
    rec: dict[str, Any] = {
        "arch": arch,
        "kind": kind,
        "select": False,
        "segment_len": current,
        "plan_T": plan_t,
    }
    best = None
    for seg_len in range(MIN_SEGMENT_CHUNKS, min(_prescan_capacity("dhu" if kind == "bwd" else "fwd_h", SDT_CONFIG),
                                                    max(chunks) - 1) + 1):  # fmt: skip
        if sum(max(1, -(-int(n) // seg_len)) for n in chunks) <= segments:
            continue  # deeper tables only
        t = _chain_table_price(chunks, num_v_heads, seg_len, sm_count, const, vh2)
        if best is None or t < best[1]:
            best = (seg_len, t)
    if best is None:
        return rec
    rec.update(best={"segment_len": best[0], "T": best[1]}, saving=plan_t - best[1])
    if plan_t - best[1] >= min_saving and best[1] <= gain * plan_t:
        rec.update(select=True, segment_len=best[0])
    return rec


def chain_plan_pricing(
    kind: str, chunks, num_v_heads: int, device_index: int, plan: ScanPlan
) -> dict:
    """``kind`` ``"fwd"`` (the ``fwd_ho`` chain) or ``"bwd"`` (``dhu_tc2``): the deeper split table priced for that
    chain against ``plan`` for a host layout of ``chunks`` (chunks per sequence) on the device's architecture;
    ``select`` is the answer and ``segment_len`` the table; on an architecture of ``FWD_HO_VH2_ARCHS`` a forward table
    whose launch runs the value-half split is priced as that split (``FWD_HO_VH2_STEP``).  An architecture without
    ``CHAIN_PLAN`` constants never selects.  See above."""
    arch = cake_backend.device_arch(int(device_index))
    found = CHAIN_PLAN.get(arch)
    if found is None:
        return {
            "arch": arch,
            "kind": kind,
            "select": False,
            "segment_len": int(plan.segment_len),
        }
    return dict(
        _chain_plan_memo(
            kind,
            tuple(int(c) for c in chunks),
            int(num_v_heads),
            int(plan.segment_len),
            _device_props(int(device_index))[0],
            arch,
            tuple(float(x) for x in found[kind]),
            float(found["gain"]),
            float(found["min_saving"]),
            FWD_HO_VH2_STEP if kind == "fwd" and arch in FWD_HO_VH2_ARCHS else 1.0,
        )  # fmt: skip
    )


def _two_stream_kept(
    chunks,
    plan: ScanPlan,
    num_heads: int | None,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions,
) -> bool:
    """Whether the adjoint table ``plan`` of a host layout of ``chunks`` (chunks per sequence) keeps its form because
    its scan runs as two groups beside the fused tail (:func:`sequence_groups`), an overlap the pricing above does not
    credit: a serial table that :func:`two_stream_split` groups, or a split ``SDT_CONFIG`` table of a value-head group
    of three beside ``tail_g3`` (no ``finalize`` pin) whose pass 2 the ``sdt_pricing`` rule runs on ``dhu_tc2`` and
    whose segments :func:`two_stream_split` groups.  Both stream counts keep the same table (``streams=1`` runs it on
    one stream: the same bits)."""
    index = int(device_index)
    sm_count = _device_props(index)[0]
    seg_len = int(plan.segment_len)
    if seg_len <= 0 or max(chunks) <= seg_len:
        return two_stream_split(len(chunks), int(num_v_heads), sm_count) is not None
    arch = cake_backend.device_arch(index)
    if num_heads is None or int(num_v_heads) != 3 * int(num_heads):
        return False
    if options.finalize is not None or not _available("tail_g3", arch):
        return False
    if tuple(plan.config) != SDT_CONFIG:
        return False
    if not (_available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)):
        return False
    segments = sum(max(1, -(-int(n) // seg_len)) for n in chunks)
    if two_stream_split(segments, int(num_v_heads), sm_count) is None:
        return False
    return bool(sdt_pricing(tuple(chunks), seg_len, int(num_v_heads), index)["select"])


def _priced_plans(
    seq_lens: tuple[int, ...],
    plans: tuple[ScanPlan, ScanPlan],
    num_heads: int | None,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions,
    *,
    keep_serial: bool = False,
) -> tuple[tuple[ScanPlan, ScanPlan], bool]:
    """``plans`` with the adjoint table priced for ``dhu_tc2`` (``TC_PLAN``) and the deeper split tables of
    ``CHAIN_PLAN``, and whether the adjoint table runs on ``dhu_tc2`` for that pricing (``ChunkMeta.bwd_tc``); plan
    pins (and a ``dhu_scan`` pin) keep the planner's tables.  ``keep_serial``: the forward table is the serial table
    of ``FWD_A_SERIAL_KEEP``, which ``CHAIN_PLAN`` does not deepen."""
    if not TC_PLAN or options.pins != (None, None) or options.dhu_scan is not None:
        return plans, False
    index = int(device_index)
    chunks = tuple(-(-int(n) // CHUNK) for n in seq_lens)
    bwd_tc = False
    kept = _two_stream_kept(chunks, plans[1], num_heads, num_v_heads, index, options)
    if not kept:
        priced = tc_plan_pricing(chunks, int(num_v_heads), index, plans[1])
        if priced["select"]:
            plans, bwd_tc = (
                (plans[0], ScanPlan(SDT_CONFIG, priced["segment_len"])),
                True,
            )
    if cake_backend.device_arch(index) in CHAIN_PLAN:
        # deeper split tables priced for fwd_ho / dhu_tc2; a serial table that the two-stream rule groups keeps its
        # form in each direction (its partial group overlaps the neighbouring stage), as does a kept adjoint table
        stream_split = two_stream_split(
            len(chunks), int(num_v_heads), _device_props(index)[0]
        )
        for kind in ("fwd", "bwd"):
            plan = plans[0] if kind == "fwd" else plans[1]
            serial = plan.segment_len <= 0 or max(chunks) <= plan.segment_len
            if (
                (serial and stream_split is not None)
                or (kind == "bwd" and kept)
                or (kind == "fwd" and keep_serial)
            ):
                continue
            deeper = chain_plan_pricing(kind, chunks, int(num_v_heads), index, plan)
            if not deeper["select"]:
                continue
            table = ScanPlan(SDT_CONFIG, deeper["segment_len"])
            if kind == "fwd":
                plans = (table, plans[1])
            else:
                plans, bwd_tc = (plans[0], table), True
    return plans, bwd_tc


def host_plans(
    seq_lens,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions | None = None,
    *,
    total_tokens: int | None = None,
    num_heads: int | None = None,
    a_inputs: bool = False,
) -> tuple[tuple[ScanPlan, ScanPlan], bool]:
    """The forward and adjoint tables of a host layout (``seq_lens``, tokens per sequence) on device ``device_index``
    and whether the adjoint table runs on ``dhu_tc2`` for its pricing (``ChunkMeta.bwd_tc``): the planner's tables,
    the serial-scan rule's adjoint plan (``num_heads``, the key heads; ``None``: no value-head group), the serial
    forward table of ``FWD_A_SERIAL_KEEP`` (``a_inputs``: the call meets the input conditions of the forwards on
    wy_a's A, :func:`_wy_a_inputs`, so :func:`fwd_a_route` serves that table), then the tables priced for the tcgen05
    chains (``TC_PLAN``, ``CHAIN_PLAN``); plan pins (and a ``dhu_scan`` pin) keep the planner's tables.  Host work
    only."""
    options = options or make_options()
    seq_lens = tuple(int(n) for n in seq_lens)
    index = int(device_index)
    if total_tokens is None:
        total_tokens = sum(seq_lens)
    plans = scan_plans(int(total_tokens), len(seq_lens), int(num_v_heads), index, seq_lens,
                       scan_config=options.scan_config, segment_chunks=options.segment_chunks)  # fmt: skip
    plans = _serial_adjoint_plan(
        seq_lens, plans, num_heads, int(num_v_heads), index, options
    )
    keep_serial = False
    if a_inputs and options.pins == (None, None):
        arch = cake_backend.device_arch(index)
        if (
            arch in FWD_A_ARCHS
            and _available("wy_a", arch)
            and _available("fwd_ho_a", arch)
        ):
            # the serial forward table of FWD_A_SERIAL_KEEP (fwd_a_route serves it; CHAIN_PLAN deepens no forward
            # table)
            keep = fwd_serial_keep_pricing(
                seq_lens, int(num_v_heads), index, plans[0], total_tokens=total_tokens
            )
            if keep["select"]:
                plans, keep_serial = (keep["serial"], plans[1]), True
    return _priced_plans(seq_lens, plans, num_heads, int(num_v_heads), index, options,
                         keep_serial=keep_serial)  # fmt: skip


# --------------------------------------------------------------------------- #
# prep selection: prep_tma (TMA-staged q / k, one CTA per (chunk, key head)) or the plain prep
# --------------------------------------------------------------------------- #
# The rule: CTAs = chunks x H and slots = ctas_per_sm x SMs; the plain prep (grid (chunks, HV)) runs where prep_tma's
# grid needs a second, short wave, slots < CTAs <= window x slots, prep_tma elsewhere.  Both give the same bits.  The
# tensor maps need q and k 16-byte aligned; an unaligned q or k takes the plain prep.  The tensor maps read q and k
# through their strides, so prep_tma takes head views of a packed row (token stride (2 H + HV) x 128) in place; the
# plain prep reads contiguous rows only.
PREP_TMA_CONSTANTS = ("ctas_per_sm", "window")
TMA_ALIGN_BYTES = 16
# q / k views that prep_tma reads in place (row-major, not overlapping: :func:`_view_rows`) run prep_tma inside the
# rule's plain band too, so the forward copies no q / k; a ``prep`` pin still decides
VIEWS_TAKE_PREP_TMA = True


def _tma_rows(t: torch.Tensor) -> bool:
    """Whether the ``prep_tma`` tensor map of ``t`` (``[..., T, H, 128]``) reads it in place: a unit channel stride,
    16-byte aligned head and token strides and address, and leading axes that fold into the token axis (the binding's
    ``check_dense_leading_dims``)."""
    if t.dim() < 3 or t.stride(-1) != 1 or t.stride(-3) <= 0:
        return False
    size = t.element_size()
    head_bytes, token_bytes = t.stride(-2) * size, t.stride(-3) * size
    if head_bytes % TMA_ALIGN_BYTES or token_bytes % TMA_ALIGN_BYTES:
        return False
    if t.data_ptr() % TMA_ALIGN_BYTES:
        return False
    expected = t.stride(-3)
    for axis in range(t.dim() - 4, -1, -1):
        expected *= t.size(axis + 1)
        if t.size(axis) > 1 and t.stride(axis) != expected:
            return False
    return True


def _view_rows(t: torch.Tensor) -> bool:
    """Whether ``t`` is a strided view that :data:`VIEWS_TAKE_PREP_TMA` lets ``prep_tma`` read in place: not
    contiguous, read in place by the tensor map (:func:`_tma_rows`), and row-major without overlap (a head pitch of at
    least 128 elements, a token pitch of at least ``H`` heads), as the head views of a packed row are."""
    if t.is_contiguous() or not _tma_rows(t):
        return False
    return t.stride(-2) >= t.size(-1) and t.stride(-3) >= t.size(-2) * t.stride(-2)


def _v_rows(v: torch.Tensor) -> int | None:
    """The token pitch of ``v`` (``[..., T, HV, 128]``) in 128-element heads when the stages that read v take it in
    place (``wy_tc2``, ``wy_bwd``, the tails: a packed row's value heads, or contiguous rows), else ``None``: unit
    element stride, a head pitch of 128 elements, a token pitch of whole heads and at least ``HV`` of them, a
    16-byte aligned base (the 16-byte ``cp.async`` rows), and leading axes that flatten into the token axis."""
    if v.dim() < 3 or v.stride(-1) != 1 or v.stride(-2) != HEAD_DIM:
        return None
    pitch = v.stride(-3)
    if pitch % HEAD_DIM or pitch < v.size(-2) * HEAD_DIM or v.data_ptr() % 16:
        return None
    expected = pitch
    for axis in range(v.dim() - 4, -1, -1):
        expected *= v.size(axis + 1)
        if v.size(axis) > 1 and v.stride(axis) != expected:
            return None
    return pitch // HEAD_DIM


# The plain prep loads q / k 4 elements (8 bytes) at a time; the stages that read v and do load 16-byte rows.
PREP_ALIGN_BYTES = 8
ROW_ALIGN_BYTES = 16


def _aligned(t: torch.Tensor, align: int) -> torch.Tensor:
    """``t`` contiguous at an address that is a multiple of ``align`` bytes.  ``.contiguous()`` returns a contiguous
    view unchanged wherever it starts (a view a few elements into its storage), so such a view is copied."""
    t = t.contiguous()
    return t.clone() if t.data_ptr() % align else t


@cache
def _prep_tma_constants(arch: str) -> tuple[int, float] | None:
    """The ``prep_tma`` record's ``select`` window for ``arch`` (``None`` when the record carries none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "prep_tma" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in PREP_TMA_CONSTANTS):
                return None
            return int(const["ctas_per_sm"]), float(const["window"])
    return None


def use_prep_tma(num_chunks: int, num_heads: int, device_index: int) -> bool:
    """The dispatch rule: ``prep_tma`` for ``num_chunks`` x ``num_heads`` CTAs unless slots < CTAs <= window x slots
    (``False`` when the architecture's record carries no window)."""
    found = _prep_tma_constants(cake_backend.device_arch(int(device_index)))
    if found is None:
        return False
    ctas_per_sm, window = found
    sm_count, _smem, _major = _device_props(device_index)
    ctas, slots = int(num_chunks) * int(num_heads), ctas_per_sm * int(sm_count)
    return not (slots < ctas <= window * slots)


def _prep_stage(
    q,
    k,
    num_chunks: int,
    num_heads: int,
    arch: str,
    options: ChainOptions,
    *,
    views: bool = False,
) -> str:
    """``prep_tma`` or ``prep`` for one call: the ``prep`` pin, else the rule (``views``: ``prep_tma`` without the
    rule); the plain prep when q or k is not 16-byte aligned or the architecture has no ``prep_tma`` record."""
    if options.prep == "plain" or not _available("prep_tma", arch):
        return "prep"
    if any(t.data_ptr() % TMA_ALIGN_BYTES for t in (q, k)):
        return "prep"
    if (
        options.prep == "tma"
        or views
        or use_prep_tma(num_chunks, num_heads, _tensor_index(q))
    ):
        return "prep_tma"
    return "prep"


# --------------------------------------------------------------------------- #
# two-stream sequence groups
# --------------------------------------------------------------------------- #
# A serial scan launches one CTA per (segment, value head) and holds one CTA per SM, so its last round can leave SMs
# idle.  Split into two contiguous groups of segments, the group that fills whole rounds runs on one stream while the
# other group's scan overlaps the neighbouring stage (forward: wy_tc of the other group; adjoint: the fused tail of
# the first group) on a second stream.  The adjoint ``dhu_tc2`` pass 2 of a split plan (one CTA per segment, every
# segment starting from its stitched state) is grouped the same way over its segments: a sequence's segments are
# contiguous in the table, so group A covers the chunks before the first segment of group B.  A group launch is the
# full launch restricted to its segments (fwd_ho through a slice of its walk, dhu_tc2 through views of the segment
# table and of dht) or chunks (wy_tc and tail_g2 / tail_g3 through views of the chunk table and of h / dh): every CTA
# does the work of the CTA of the full launch that covers the same (segment or chunk, value head), so the bits are
# the same.


def _idle_slots(ctas: int, sm_count: int) -> int:
    """The SM slots the last round of ``ctas`` one-per-SM CTAs leaves idle."""
    return -(-ctas // sm_count) * sm_count - ctas


@lru_cache(maxsize=1024)
def two_stream_split(num_segments: int, num_v_heads: int, sm_count: int) -> int | None:
    """The segments ``k`` of a serial scan's full-round group: the fewest idle slots of ``k * num_v_heads`` CTAs over
    ``k`` in ``[1, num_segments)``, ties to the larger ``k``, when that leaves fewer idle slots than the whole scan;
    ``None`` (one stream) otherwise, for fewer than two segments and for a scan of one round.  The adjoint scan runs
    the first ``k`` segments first; the forward scan runs the last ``k`` on the caller's stream."""
    n = int(num_segments)
    if n < 2 or n * num_v_heads <= sm_count:
        return None
    best = min(range(1, n), key=lambda k: (_idle_slots(k * num_v_heads, sm_count), -k))
    if _idle_slots(best * num_v_heads, sm_count) >= _idle_slots(
        n * num_v_heads, sm_count
    ):
        return None
    return best


# the architectures where a serial tcgen05 adjoint scan beside tail_g3 runs as its two groups (the partial round's
# group B on the side stream beside the tail of group A); elsewhere it keeps one launch on the caller's stream
TWO_STREAM_SERIAL_G3_ARCHS = ("sm_100a", "sm_103a")


def sequence_groups(
    meta: ChunkMeta, num_v_heads: int, sm_count: int
) -> tuple[StreamGroups | None, StreamGroups | None]:
    """The two-stream groups of a host-built table's forward and adjoint scans (``None``: one stream).  A scan is
    grouped on a serial plan (one segment per sequence, the table's walk) when :func:`two_stream_split` splits it: the
    adjoint scan's group A is its full-round group, the forward scan's group A the rest.  The adjoint table of a split
    plan is grouped over its segments by the same rule: group A its first ``k`` segments, the chunks before segment
    ``k`` (used where its pass 2 runs on ``dhu_tc2`` beside ``tail_g3``; a serial adjoint scan's groups are used
    beside ``tail_g2``, and beside ``tail_g3`` on the architectures of ``TWO_STREAM_SERIAL_G3_ARCHS``)."""
    if meta.seq_lens is None:
        return None, None
    order = meta.order if meta.order is not None else range(meta.num_seqs)
    chunks = [-(-meta.seq_lens[seq] // CHUNK) for seq in order]
    groups: list[StreamGroups | None] = []
    for segs, adjoint in ((meta.fwd_segments, False), (meta.bwd_segments, True)):
        k = None
        if segs.num_segments == meta.num_seqs and segs.walk is None:
            k = two_stream_split(meta.num_seqs, int(num_v_heads), int(sm_count))
        elif adjoint and segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
            seq_chunk_start = [0]
            for n in chunks:
                seq_chunk_start.append(seq_chunk_start[-1] + n)
            seg_chunk_start = _segment_lists(seq_chunk_start, int(segs.segment_len))[0]
            k = None
            if len(seg_chunk_start) - 1 == segs.num_segments:
                k = two_stream_split(segs.num_segments, int(num_v_heads), int(sm_count))
            split_chunk = 0 if k is None else seg_chunk_start[k]
            groups.append(
                StreamGroups(k, split_chunk)
                if k is not None and 0 < split_chunk < meta.num_chunks
                else None
            )
            continue
        split = None if k is None else (k if adjoint else meta.num_seqs - k)
        split_chunk = 0 if split is None else sum(chunks[:split])
        groups.append(
            StreamGroups(split, split_chunk)
            if split is not None and 0 < split_chunk < meta.num_chunks
            else None
        )
    return groups[0], groups[1]


_SIDE_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _side_stream(device_index: int, main: torch.cuda.Stream) -> torch.cuda.Stream:
    """The side stream of the two-stream groups forked from the caller's stream ``main``: one per (device, caller
    stream), a stream of PyTorch's pool at the greatest priority the device offers, created once.  Callers on
    different streams (another thread, a CUDA graph capture on its capture stream) fork different side streams.

    Every tensor a side stream reads or writes is allocated on the caller's stream before the fork and stays
    referenced by the stage until the join (``main.wait_stream(side)``, also when a launch raises), so the caching
    allocator reuses its memory only after the side stream's work: no ``record_stream``.  A cached device table it
    reads was built on some stream; the caller's stream waited for its write and recorded its use before the fork
    (:func:`_use_table`)."""
    key = (device_index, main.cuda_stream)
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        _least, greatest = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device_index, priority=greatest)
        _SIDE_STREAMS[key] = stream
    return stream


# --------------------------------------------------------------------------- #
# the wavefront pair: dhu_tc2 -> tail_g2_bf16 overlapped through per-tile ready counts
# --------------------------------------------------------------------------- #
# Where the tcgen05 adjoint scan (a serial plan's, or a split plan's pass 2) feeds tail_g2_bf16 on one stream, the
# tail of a chunk can start as soon as the scan CTA of its segment has stored that chunk's dh and dv2.  The pair
# ``dhu_tc2_wavefront`` / ``tail_g2_bf16_wavefront`` does that with the same math: the scan adds one to a ready count per (chunk,
# value head) for each of its 64 store threads after their stores (one gpu fence per two chunks) and counts its CTAs
# in at entry (the split pass 2's scan ``dhu_tc2_wavefront_split``: the store warps meet at a barrier every four
# chunks and one elected thread adds 64 per chunk after one gpu fence); the tail visits the chunks in the order the
# scan finishes them (the k-th chunk from every segment's end, segments interleaved: :func:`wavefront_order`) and
# waits for the count of its tile to reach 64 before it reads dh and dv2.  The counts live in a workspace zeroed
# before the scan.  Every tile reads the values the plain tail reads, so the bits are the same.  The tail runs on the
# side stream after ``cuStreamWaitValue32`` on the entry count, so it
# starts only once every scan CTA has started, and the rule asks the scan to take at most half the SMs, so the waiting
# tail never holds the SMs the scan needs; the caller's stream joins the side stream before the backward returns.  A
# CUDA graph does not order its two branches (the side branch's wait could run ahead of the scan it waits for), so a
# capture records the plain launches.  The rule: a host-built table, an architecture of ``WAVEFRONT_ARCHS`` with the
# pair's records and at least ``WAVEFRONT_MIN_V_HEADS`` value heads, ``2 x segments x HV <= SMs``
# (:func:`wavefront_engaged`; this side is decided once per table, :func:`_wavefront_table`), and on the call the
# adjoint scan on ``dhu_tc2`` on the caller's stream (no two-stream groups) beside the ``tail_g2_bf16`` tail.
# On an architecture of ``WAVEFRONT_SPLIT_ARCHS`` the pass 2 of a split plan (the pre-scan and the stitch, then
# ``dhu_tc2`` from the stitched states) whose ``segments x HV`` CTAs outnumber the SMs runs the pair at any value-head
# count once the call's ``tokens x HV`` reach ``WAVEFRONT_SPLIT_MIN_TOKEN_HEADS`` (:func:`wavefront_split_engaged`):
# its GPU work then hides the pair's host cost, the pre-scan and the stitch run on the caller's stream before the
# fork, and the entry count reaches ``segments x HV`` only once the last CTA of the second round has started, after
# the first round has given back the SMs it needs, so again the waiting tail never holds an SM a scan CTA still needs.
# That pass 2 runs the scan ``dhu_tc2_wavefront_split`` (the elected release: with one release per store thread
# the slower scan ate most of the overlap on sm_103a); the scan of at most half the SMs runs ``dhu_tc2_wavefront``,
# whose sm_107a program releases in the elected form once per eight chunks (there the release per store thread
# slowed the scan by about a fifth); the split side stays off sm_107a (``WAVEFRONT_SPLIT_ARCHS``).
WAVEFRONT_ARCHS = ("sm_100a", "sm_103a", "sm_107a")
# sm_100a: the host path of a call with fewer value heads outlasts its GPU work, so the pair's host cost is not hidden
WAVEFRONT_MIN_V_HEADS = {"sm_100a": 32}
# the split pass 2 of more CTAs than SMs: the pair (scan ``dhu_tc2_wavefront_split``) at any value-head count, on a
# call of enough work
WAVEFRONT_SPLIT_ARCHS = ("sm_100a", "sm_103a")
# the split side's work floor, total tokens x value heads (393216): the pair's wait form adds host time to every
# backward (its counter workspace and reset, the fork and join events, the stream wait; about 75-115 us on B200),
# and below this much work the call is host-bound (its host path outlasts its GPU work, as 16384 tokens at 16 value
# heads), so the host cost is not hidden and the call slows down although the pair shortens its GPU span
WAVEFRONT_SPLIT_MIN_TOKEN_HEADS = 3 * 2**17
_WAVEFRONT_EVENTS: dict[tuple[int, int], tuple[torch.cuda.Event, torch.cuda.Event]] = {}


def wavefront_engaged(num_segments: int, num_v_heads: int, sm_count: int) -> bool:
    """The rule's occupancy side: the scan's ``segments x HV`` CTAs take at most half of the ``sm_count`` SMs."""
    return 2 * int(num_segments) * int(num_v_heads) <= int(sm_count)


def wavefront_split_engaged(
    num_segments: int, num_v_heads: int, sm_count: int, num_tokens: int
) -> bool:
    """The rule's split side: the split pass 2's ``segments x HV`` CTAs outnumber the ``sm_count`` SMs (two
    rounds), and the call's ``num_tokens x HV`` reach ``WAVEFRONT_SPLIT_MIN_TOKEN_HEADS`` (GPU work enough to hide
    the pair's host cost)."""
    return (
        int(num_segments) * int(num_v_heads) > int(sm_count)
        and int(num_tokens) * int(num_v_heads) >= WAVEFRONT_SPLIT_MIN_TOKEN_HEADS
    )


def wavefront_order(seg_chunk_start) -> list[int]:
    """The chunks in the order the adjoint scan finishes them: step ``k`` holds the ``k``-th chunk from the end of
    every segment (segments in table order) that has one."""
    sc = [int(x) for x in seg_chunk_start]
    lens = [b - a for a, b in zip(sc[:-1], sc[1:], strict=True)]
    return [
        sc[s + 1] - 1 - k
        for k in range(max(lens, default=0))
        for s in range(len(lens))
        if k < lens[s]
    ]


def _wavefront_table(
    meta: ChunkMeta, num_v_heads: int, device_index: int
) -> str | None:
    """The table's side of the wavefront rule, as the pair's adjoint scan stage (``None``: the separate launches): a
    host-built table on an architecture of ``WAVEFRONT_ARCHS`` with the pair's records, and either on an
    architecture of ``WAVEFRONT_SPLIT_ARCHS`` a split adjoint plan of more pass-2 CTAs than SMs and at least
    ``WAVEFRONT_SPLIT_MIN_TOKEN_HEADS`` tokens x value heads (``dhu_tc2_wavefront_split``), or at least
    ``WAVEFRONT_MIN_V_HEADS`` value heads and an adjoint scan of at most half the SMs (``dhu_tc2_wavefront``)."""
    if meta.seq_lens is None:
        return None
    arch = cake_backend.device_arch(device_index)
    if arch not in WAVEFRONT_ARCHS:
        return None
    if not (
        _available("dhu_tc2_wavefront", arch)
        and _available("tail_g2_bf16_wavefront", arch)
    ):
        return None
    segs = meta.bwd_segments
    sm_count = _device_props(device_index)[0]
    if (
        arch in WAVEFRONT_SPLIT_ARCHS
        and segs.segment_len > 0
        and segs.num_segments > meta.num_seqs
        and wavefront_split_engaged(
            segs.num_segments, num_v_heads, sm_count, sum(meta.seq_lens)
        )
        and _available("dhu_tc2_wavefront_split", arch)
    ):
        return "dhu_tc2_wavefront_split"
    if num_v_heads < WAVEFRONT_MIN_V_HEADS.get(arch, 0):
        return None
    if wavefront_engaged(segs.num_segments, num_v_heads, sm_count):
        return "dhu_tc2_wavefront"
    return None


def _wavefront_walk(
    seq_lens: tuple[int, ...], order: tuple[int, ...] | None, segment_len: int
) -> list[int]:
    """:func:`wavefront_order` of a host-built table's adjoint segments, from the host layout."""
    seq_chunk_start = [0]
    for seq in order if order is not None else range(len(seq_lens)):
        seq_chunk_start.append(seq_chunk_start[-1] - (-seq_lens[seq] // CHUNK))
    return wavefront_order(_segment_lists(seq_chunk_start, segment_len)[0])


@lru_cache(maxsize=256)
def _wavefront_order_tensor(
    seq_lens: tuple[int, ...],
    order: tuple[int, ...] | None,
    segment_len: int,
    device_index: int,
) -> tuple[torch.Tensor, TableReady]:
    """:func:`_wavefront_walk` on the device, built once per table (outside a CUDA graph capture: the wavefront pair
    is not captured), and its write."""
    table = torch.tensor(
        _wavefront_walk(seq_lens, order, segment_len), dtype=torch.int32
    )
    table = table.pin_memory().to(torch.device("cuda", device_index), non_blocking=True)
    return table, _table_ready(device_index)


def _wavefront_events(
    device_index: int, main: torch.cuda.Stream
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """The fork and join events of the pair on the caller's stream ``main``: one pair per (device, caller stream)."""
    key = (device_index, main.cuda_stream)
    events = _WAVEFRONT_EVENTS.get(key)
    if events is None:
        events = _WAVEFRONT_EVENTS[key] = (torch.cuda.Event(), torch.cuda.Event())
    return events


@cache
def _wait_entry(device_index: int):
    """``tail_g2_bf16_wavefront``'s ``wait_geq`` (``cuStreamWaitValue32`` GEQ on word 0 of the counts) on the device."""
    arch = cake_backend.device_arch(device_index)
    name = cake_backend.stage_record("tail_g2_bf16_wavefront", arch)
    return getattr(
        cake_backend.load(name, arch), cake_backend.record(name)["wait_entry"]
    )


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
# The forward WY stage: on an architecture of ``WY_TC2_ARCHS`` every ``wy_tc`` launch (the forward, both sequence
# groups of the P form and the backward's recompute) runs ``wy_tc2`` instead, the same per-element chains, grid and
# outputs (so the same bits) with a 64-column TMEM accumulator and packed shared-memory operands, so five CTAs fit
# on an SM where ``wy_tc`` fits four.
WY_TC2_ARCHS = ("sm_100a", "sm_103a", "sm_107a")


def wy_stage(arch: str) -> str:
    """The stage of the forward WY launches on ``arch``: ``wy_tc2`` on ``WY_TC2_ARCHS``, else ``wy_tc``."""
    return "wy_tc2" if arch in WY_TC2_ARCHS else "wy_tc"


def _wy_outputs(kn, v, total_tokens, num_v_heads, ex):
    w = ex.empty(
        (total_tokens, num_v_heads, HEAD_DIM), dtype=kn.dtype, device=kn.device
    )
    u = ex.empty(v.shape, dtype=v.dtype, device=v.device)
    return w, u


def _launch_wy(
    arch,
    kn,
    v,
    g_cs,
    beta32,
    A,
    meta,
    num_heads,
    num_v_heads,
    total_tokens,
    *,
    recompute,
    v_rows,
    ex,
):
    w, u = _wy_outputs(kn, v, total_tokens, num_v_heads, ex)
    stage = wy_stage(arch)
    ex.launch(stage, meta.num_chunks, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start, meta.chunk_len,
              num_heads, num_v_heads, 1 if recompute else 0, *((v_rows,) if stage == "wy_tc2" else ()))  # fmt: skip
    return w, u


def _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, *, store_final_state, table,
                  workspaces, options):  # fmt: skip
    device = kn.device
    h = torch.empty(
        (meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM),
        dtype=kn.dtype,
        device=device,
    )
    v_new = torch.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:
        final_state = torch.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        final_state = workspaces.take(table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32,
                                      device)  # fmt: skip
    if initial_state is not None:
        h0 = initial_state.contiguous()
        if h0.dtype != torch.float32:
            h0 = h0.float()
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    stage, vb, pre, pre_threads = _scan_stages("fwd_h", segs.scan_config, arch)
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        split_stage = f"{stage}_split"
        if pre is None or not _available(split_stage, arch):
            raise NotImplementedError(f"the segmented fwd_h scan needs {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and "
                                      f"{split_stage} for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
        seg_state = workspaces.take(table, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
                seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                num_heads, num_v_heads, use_init, 0, seg_flag, warmup_thresh_log2(options))  # fmt: skip
        _launch(split_stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new,
                final_state, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_init, 1 if store_final_state else 0,
                seg_flag)  # fmt: skip
        return h, v_new, final_state
    _launch(stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
            final_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
            num_heads, num_v_heads, use_init, 1 if store_final_state else 0)  # fmt: skip
    return h, v_new, final_state


# The value-half split of the fused forward scan and output: on an architecture of ``FWD_HO_VH2_ARCHS`` a forward
# scan of one launch (no two-stream groups) whose CTAs fill at most half of the SMs runs ``fwd_ho_vh2``, two CTAs per
# (segment, value head) on twice the grid, each the scan and output of one value half.  Its per-element math is
# fwd_ho's, so the bits are the same; the split shortens each CTA's per-chunk chain where the SMs would sit idle.
FWD_HO_VH2_ARCHS = ("sm_107a",)


@cache
def fwd_ho_vh2_splits(arch: str) -> bool:
    """Whether the ``fwd_ho_vh2`` record of ``arch`` is the split program (two CTAs per segment and value head).  On
    ``sm_100a`` and ``sm_103a`` the record is the ``fwd_ho`` program itself, which runs on ``fwd_ho``'s grid."""
    split = cake_backend.record(cake_backend.stage_record("fwd_ho_vh2", arch))
    plain = cake_backend.record(cake_backend.stage_record("fwd_ho", arch))
    return split["library"] != plain["library"]


def fwd_ho_vh2_engaged(num_segments: int, num_v_heads: int, sm_count: int) -> bool:
    """Whether a one-launch forward scan of ``num_segments`` segments runs as ``fwd_ho_vh2``: its ``2 x num_segments
    x num_v_heads`` CTAs fit one round of ``sm_count`` SMs (one CTA per SM)."""
    return 2 * int(num_segments) * int(num_v_heads) <= int(sm_count)


_WALKS: dict = {}


def _walk(device, segs: SegmentTable, options: ChainOptions, ex=None) -> torch.Tensor:
    """The walk of ``fwd_ho`` (CTA group ``b // HV`` runs segment ``walk[b]``): the table's own (longest first) when it
    has one, else the identity, a constant table per (device, segment count) kept between calls under
    ``host_cache`` (one built inside a CUDA graph capture is not kept).  ``ex``: the forward's executor
    (:data:`_DIRECT` when ``None``): a missing table is built as a cache fill (:meth:`_DirectExecutor.fill`), and the
    kept one is used through it (on the stream that built it, a use has nothing to wait for); the general path
    (:data:`_DIRECT`) uses a kept one in place (:func:`_use_table`), with no frame of an executor step."""
    if segs.walk is not None:
        return segs.walk
    num_segments = segs.num_segments
    key = (_device_index(device), int(num_segments))
    hit = _WALKS.get(key) if options.host_cache else None
    if hit is not None and (ex is _DIRECT or ex is None):
        _use_table(hit[0], hit[1], key[0])
        return hit[0]
    ex = ex or _DIRECT
    if hit is None:
        with ex.fill():
            walk = torch.arange(int(num_segments), dtype=torch.int32, device=device)
            if options.host_cache and not torch.cuda.is_current_stream_capturing():
                hit = _WALKS[key] = (walk, _table_ready(key[0]))
        if hit is None:
            return walk
    ex.use(key, hit)
    return hit[0]


def _table_rows(state, meta: ChunkMeta):
    """Per-sequence state rows (``initial_state``, ``dht``) in the table's sequence order."""
    if state is None or meta.order_index is None:
        return state
    return state.index_select(0, meta.order_index[0])


def _caller_rows(state, meta: ChunkMeta):
    """Per-sequence state rows (``final_state``, ``dh0``) back in the caller's sequence order."""
    if meta.order_index is None:
        return state
    return state.index_select(0, meta.order_index[1])


def _launch_fwd_ho(arch, qn, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, *, store_final_state, table,
                   workspaces, options, scale, ex, wy=None):  # fmt: skip
    """The fused forward scan and output on the plan's forward segment table: a split plan first runs the pre-scan
    and the stitch (``reverse = 0``) for the segments' start states.  ``wy`` (``(v, beta32, A)``, with
    ``meta.fwd_groups``): ``wy_tc`` has not run; it runs here with the scan as two sequence groups (P form).  ``ex``:
    the forward's executor."""
    device = kn.device
    h = ex.empty(
        (meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM),
        dtype=kn.dtype,
        device=device,
    )
    v_new = ex.empty(u.shape, dtype=u.dtype, device=device)
    o = ex.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:
        final_state = ex.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        final_state = workspaces.take(table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32,
                                      device)  # fmt: skip
    if initial_state is not None:
        h0 = ex.contiguous(initial_state)
        if h0.dtype != torch.float32:
            h0 = ex.float(h0)
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    seg_start, use_seg = final_state, 0
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        stage, vb, pre, pre_threads = _scan_stages("fwd_h", segs.scan_config, arch)
        if pre is None or not _available("stitch", arch):
            raise NotImplementedError(f"the segmented fwd_ho path needs {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and "
                                      f"stitch for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
        seg_state = workspaces.take(table, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        ex.launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
                  seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                  meta.seq_chunk_start, num_heads, num_v_heads, use_init, 0, seg_flag,
                  warmup_thresh_log2(options))  # fmt: skip
        seg_start = workspaces.take(table, "f_seg_start", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        ex.launch("stitch", meta.num_seqs, num_v_heads, STITCH_VSPLIT, seg_state, h0, seg_start, segs.seq_seg_start,
                  seg_flag, num_v_heads, use_init, 0)  # fmt: skip
        use_seg = 1
    walk = _walk(device, segs, options, ex)
    if wy is None:
        stage, ctas = "fwd_ho", segs.num_segments * num_v_heads
        if arch in FWD_HO_VH2_ARCHS and fwd_ho_vh2_engaged(
            segs.num_segments, num_v_heads, _device_props(_tensor_index(kn))[0]
        ):
            stage = "fwd_ho_vh2"
            if fwd_ho_vh2_splits(arch):
                ctas = 2 * ctas
        ex.launch(stage, ctas, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h, v_new, v_new, o, o, final_state,
                  meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, walk,
                  num_heads, num_v_heads, use_init, 1 if store_final_state else 0, use_seg, float(scale))  # fmt: skip
        return h, v_new, final_state, o
    # P form: the side stream runs wy_tc and fwd_ho of group A (the partial round); the caller's stream runs those of
    # group B (the full rounds) once wy_tc(A) is done, then joins the side stream.  fwd_ho(A) can read w / u rows past
    # the length of a partial chunk (its 64-row tiles) while wy_tc(B) writes them: such rows enter fwd_ho only
    # through selects (v_new rows past the chunk's length are 0), so they do not change a bit.  A kernel that used them
    # arithmetically (a multiply by a zero mask) would make this form nondeterministic.
    v, beta32, A, v_rows = wy
    groups = cast("StreamGroups", meta.fwd_groups)
    n, na, ca = segs.num_segments, groups.split, groups.split_chunk

    wy_name = wy_stage(arch)
    v_args = (v_rows,) if wy_name == "wy_tc2" else ()

    def wy_tc(c0, c1):
        ex.launch(wy_name, c1 - c0, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start[c0:c1],
                  meta.chunk_len[c0:c1], num_heads, num_v_heads, 0, *v_args)  # fmt: skip

    def fwd_ho(s0, s1):
        ex.launch("fwd_ho", (s1 - s0) * num_v_heads, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h, v_new, v_new, o, o,
                  final_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                  meta.seq_chunk_start, walk[s0:s1], num_heads, num_v_heads, use_init, 1 if store_final_state else 0,
                  use_seg, float(scale))  # fmt: skip

    # the general path (_DIRECT) makes the executor's fork, events and join in place, with no frame of an executor step
    if ex is _DIRECT:
        index = _tensor_index(kn)
        main = torch.cuda.current_stream(index)
        side = _side_stream(index, main)
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        try:
            with torch.cuda.stream(side):
                wy_tc(0, ca)
                wy_a = torch.cuda.Event()
                wy_a.record(side)
                fwd_ho(0, na)
            main.wait_event(wy_a)
            wy_tc(ca, meta.num_chunks)
            fwd_ho(na, n)
        finally:
            main.wait_stream(side)  # the join
        return h, v_new, final_state, o
    main, side = ex.fork(_tensor_index(kn))
    try:
        with ex.on(side):
            wy_tc(0, ca)
            wy_a = ex.signal(side)
            fwd_ho(0, na)
        ex.wait(main, wy_a)
        wy_tc(ca, meta.num_chunks)
        fwd_ho(na, n)
    finally:
        ex.join(main, side)  # the join
    return h, v_new, final_state, o


# --------------------------------------------------------------------------- #
# the forward on the WY stage's A (the rows whose forward scan is serial)
# --------------------------------------------------------------------------- #
# On an architecture of ``FWD_A_ARCHS`` a call on a host-built table whose forward segment table is serial (no sequence
# split over several scan CTAs, ``ChunkMeta.fwd_segments``) runs two launches instead of the prep, the WY stage and
# ``fwd_ho``: ``wy_a``, the WY A producer (``A = bf16(X)``, ``X = (I + L)^-1`` of every chunk and value head by a
# tcgen05 product and TF32x3 merges, within half a BF16 ulp of the FP64 ``X``, beside the prep's rstd_q / rstd_k / g_cs
# / qn / kn; one persistent CTA per SM over the (chunk, key head) items in a fixed order), and ``fwd_ho_a``, the fused
# scan and output on that A, qn and kn (``u = bf16(A bf16(beta v))`` and ``w = bf16(A bf16(beta 2^g kn))`` formed in
# the CTA; one CTA per (segment, value head) on ``fwd_ho``'s walk).  ``fwd_ho_a`` writes the tape's h / v_new / w
# (:data:`FWD_A_TAPE`) when the call keeps a tape and none when it keeps none (:func:`_forward_tape`); o and the final
# state are the same bits either way.  The tape is the ``T_full`` level's (:data:`cake_plan.LEVEL_STATE`) in the
# layouts the backward reads (u is never formed in memory).  Every reduction runs in a fixed order with no atomic, so
# the route is bitwise run to run; its bits differ from the prep forward's (another A producer).  The route serves
# FP32 beta, normalized q / k, value-head groups of at most :data:`WY_A_MAX_GROUP` and calls without a ``prep`` /
# ``scan_config`` / ``segment_chunks`` pin (:func:`_wy_a_inputs`); every other call, and every row whose forward
# segment table splits a sequence, runs the prep forward (the split forward keeps its own route).  The route is
# decided from those host facts alone: q / k are copied to 16-byte aligned rows where wy_a's tensor maps cannot read
# them in place, v where the stages that read it by rows cannot, g and beta where their base is not 16-byte aligned.
FWD_A_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
WY_A_MAX_GROUP = 3
# On an architecture of ``FWD_A_SERIAL_KEEP`` (price factor, fill) a host-built table without plan pins of a call the
# forward on wy_a's A serves keeps a SERIAL forward table where the register planner takes a split one, prices its
# cheapest serial table at most the factor above it (the planner record's ``priced`` entries,
# :func:`fwd_serial_keep_pricing`) and the serial table's CTAs (sequences x HV) fill at least the fill fraction of the
# SMs; ``CHAIN_PLAN`` then deepens no forward table there and :func:`fwd_a_route` serves the row.  A planner table
# that is serial already keeps every rule.  Measured with the non-deterministic package's programs on R200 (sm_107a,
# 212 SMs): the 3 sequences of 11000 / 12000 / 9768 tokens with HV 48 (144 serial CTAs, ratio 1.06) run fwd
# 0.71-0.85, fwd+bwd 0.89-0.95 and checkpoint 0.82-0.90 of the split table's time; with HV 24 (72 CTAs, ratio 1.51)
# the serial table is slower, so 1.10 keeps the measured win and the fill of 0.65 keeps it off layouts of fewer
# serial CTAs.
FWD_A_SERIAL_KEEP: dict[str, tuple[float, float]] = {"sm_107a": (1.10, 0.65)}
# fwd_ho_a's tape word per tape entry (the qn / kn tapes are wy_a's outputs)
FWD_A_TAPE = {"h": 1, "v_new": 2, "w": 8}


def _wy_a_inputs(
    num_heads: int,
    num_v_heads: int,
    beta_dtype: torch.dtype,
    normalize_qk: bool,
    options: ChainOptions,
) -> bool:
    """The input conditions of the forwards on wy_a's A (:func:`fwd_a_route`, :func:`fwd_split_route`) that a call
    fixes before its table: normalized q / k, FP32 beta, no ``prep`` / ``scan_config`` / ``segment_chunks`` pin and a
    value-head group of at most :data:`WY_A_MAX_GROUP`.  Shapes, dtypes and pins only (never an address)."""
    return (
        bool(normalize_qk)
        and beta_dtype == torch.float32
        and options.prep is None
        and options.pins == (None, None)
        and int(num_v_heads) <= WY_A_MAX_GROUP * int(num_heads)
    )


def _fwd_a_inputs(
    meta: ChunkMeta, q, k, v, beta, normalize_qk: bool, options: ChainOptions
) -> bool:
    """:func:`_wy_a_inputs` of a call on ``meta``, a host-built table."""
    return meta.seq_lens is not None and _wy_a_inputs(
        int(q.shape[-2]), int(v.shape[-2]), beta.dtype, normalize_qk, options
    )


def fwd_a_route(
    meta: ChunkMeta, arch: str, q, k, v, beta, normalize_qk: bool, options: ChainOptions
) -> bool:
    """Whether a forward on ``meta`` runs ``wy_a`` and ``fwd_ho_a`` (see :data:`FWD_A_ARCHS`): an architecture of
    ``FWD_A_ARCHS`` with both records, :func:`_fwd_a_inputs` and a serial forward segment table."""
    if arch not in FWD_A_ARCHS or not _fwd_a_inputs(
        meta, q, k, v, beta, normalize_qk, options
    ):
        return False
    segs = meta.fwd_segments
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        return False
    return _available("wy_a", arch) and _available("fwd_ho_a", arch)


@lru_cache(maxsize=4096)
def _fwd_scan_record(total_tokens, seq_lens, num_v_heads, device_index, sset, props):
    """The register planner's pricing record of a host layout's forward table without plan pins."""
    _plan, rec = choose_scan_plan("fwd_h", total_tokens, len(seq_lens), num_v_heads, device_index, seq_lens,
                                  sset)  # fmt: skip
    return rec


def fwd_serial_keep_pricing(
    seq_lens,
    num_v_heads: int,
    device_index: int,
    plan: ScanPlan,
    *,
    total_tokens: int | None = None,
) -> dict:
    """The forward table rule of :data:`FWD_A_SERIAL_KEEP` on a host layout whose planner forward table without plan
    pins is ``plan``: ``{"select", "ratio", "fill", "serial", "limit", "min_fill"}``, ``serial`` the planner's cheapest
    serial table, ``ratio`` its price over the price of ``plan`` and ``fill`` its CTAs over the SMs (``select``: ``plan``
    splits a sequence, ``ratio`` is at most the architecture's ``limit`` and ``fill`` at least its ``min_fill``).  Host
    work only."""
    index = int(device_index)
    arch = cake_backend.device_arch(index)
    seq_lens = tuple(int(n) for n in seq_lens)
    limit, min_fill = FWD_A_SERIAL_KEEP.get(arch, (None, None))
    rec: dict[str, Any] = {
        "select": False,
        "ratio": None,
        "fill": None,
        "serial": None,
        "limit": limit,
        "min_fill": min_fill,
    }
    chunks = tuple(-(-n // CHUNK) for n in seq_lens)
    if (
        limit is None
        or min_fill is None
        or not chunks
        or plan.segment_len <= 0
        or max(chunks) <= plan.segment_len
    ):
        return rec
    tokens = sum(seq_lens) if total_tokens is None else int(total_tokens)
    priced = _fwd_scan_record(tokens, seq_lens, int(num_v_heads), index, stage_set(arch),
                              _device_props(index)).get("priced") or []  # fmt: skip
    chosen = [
        cost
        for config, seg_len, cost in priced
        if tuple(config) == tuple(plan.config) and seg_len == plan.segment_len and cost
    ]
    serial = [
        (cost, tuple(config))
        for config, seg_len, cost in priced
        if seg_len == 0 and cost is not None
    ]
    if not chosen or not serial:
        return rec
    cost, config = min(serial)
    ratio = cost / chosen[0]
    fill = len(seq_lens) * int(num_v_heads) / _device_props(index)[0]
    rec.update(
        select=ratio <= limit and fill >= min_fill,
        ratio=round(ratio, 4),
        fill=round(fill, 4),
        serial=ScanPlan(config, 0),
    )
    return rec


# Whether the forward of the call in flight keeps a tape: :func:`cake_gdn_det` sets it off around a call that records
# no backward (grad mode off, or no input that requires grad); an autograd Function applied directly keeps one.
_RECORDS_BACKWARD: ContextVar[bool] = ContextVar(
    "cake_gdn_det_records_backward", default=True
)
_TOP_SAVED_TENSORS_HOOKS = getattr(
    torch._C._autograd, "_top_saved_tensors_default_hooks", None
)


def _checkpoint_first_pass() -> bool:
    """Whether the call runs in the first pass of a non-reentrant activation checkpoint (``torch.utils.checkpoint``,
    ``use_reentrant=False``): that pass's saved tensors are dropped and the forward runs again before the backward, so
    it keeps no tape.  Read from the innermost saved-tensor hooks (a private PyTorch hook, absent: keep the tape): the
    checkpoint's own pack hook; any other hooks, or none, keep the tape."""
    if _TOP_SAVED_TENSORS_HOOKS is None:
        return False
    hooks = _TOP_SAVED_TENSORS_HOOKS(False)
    if hooks is None:
        return False
    pack = hooks[0]
    return getattr(pack, "__module__", None) == "torch.utils.checkpoint" and getattr(
        pack, "__qualname__", ""
    ).startswith("_checkpoint_hook.")


def _forward_tape() -> bool:
    """Whether the forward in flight keeps a tape (:data:`_RECORDS_BACKWARD`, outside a checkpoint's first pass)."""
    return _RECORDS_BACKWARD.get() and not _checkpoint_first_pass()


def _tma_operands(q, k, ex):
    """q / k as wy_a's tensor maps read them: in place where both are (:func:`_tma_rows`, the head views of a packed
    row among them), else contiguous rows at a 16-byte aligned base (``ex``: the forward's executor)."""
    if _tma_rows(q) and _tma_rows(k):
        return q, k
    return ex.aligned(q, TMA_ALIGN_BYTES), ex.aligned(k, TMA_ALIGN_BYTES)


def _forward_state(meta: ChunkMeta, kn, v):
    """The (shape, dtype) of each state tensor of :data:`cake_plan.STATE_NAMES` (``w``, ``u``, ``h``, ``v_new``) of
    a forward over ``meta``, as the prep forward makes them."""
    num_v_heads = v.shape[-2]
    total_tokens = v.numel() // (num_v_heads * HEAD_DIM)
    return (
        ((total_tokens, num_v_heads, HEAD_DIM), kn.dtype),
        (tuple(v.shape), v.dtype),
        ((meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM), kn.dtype),
        (tuple(v.shape), v.dtype),
    )


def _start_state(initial_state, meta: ChunkMeta, output_final_state: bool, num_v_heads: int, table, workspaces,
                 device, ex):  # fmt: skip
    """``(final_state, h0, use_init)`` of a forward on wy_a's A: the final-state output (a zero-row workspace when the
    call outputs none) and the initial state rows in table order (the final-state buffer and 0 without one), FP32 and
    contiguous at a 16-byte aligned base: the split forward's stitch reads them with 16-byte loads, so rows the
    caller passes at another address are copied (the same bits)."""
    f32 = torch.float32
    if output_final_state:
        final_state = ex.empty((meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM), dtype=f32, device=device)  # fmt: skip
    else:
        final_state = workspaces.take(
            table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), f32, device
        )
    initial_rows = ex.table_rows(initial_state, meta)
    if initial_rows is None:
        return final_state, final_state, 0
    h0 = ex.contiguous(initial_rows)
    if h0.dtype != f32:
        h0 = ex.float(h0)
    return final_state, ex.aligned(h0, ROW_ALIGN_BYTES), 1


def _launch_wy_a(q, k, g32, beta32, meta: ChunkMeta, num_heads: int, num_v_heads: int, index: int,
                 ex):  # fmt: skip
    """``wy_a`` over ``meta``'s chunks (one persistent CTA per SM, at most one per (chunk, key head)): A, rstd_q,
    rstd_k, g_cs, qn and kn."""
    total_tokens = q.numel() // (num_heads * HEAD_DIM)
    device, f32, bf16 = q.device, torch.float32, q.dtype
    qn = ex.empty(q.shape, dtype=bf16, device=device)
    kn = ex.empty(k.shape, dtype=bf16, device=device)
    rstd_q = ex.empty((total_tokens, num_heads), dtype=f32, device=device)
    rstd_k = ex.empty((total_tokens, num_heads), dtype=f32, device=device)
    g_cs = ex.empty((total_tokens, num_v_heads), dtype=f32, device=device)
    A = ex.empty((total_tokens, num_v_heads, CHUNK), dtype=bf16, device=device)
    sm_count = _device_props(index)[0]
    ex.launch("wy_a", min(sm_count, meta.num_chunks * num_heads), 1, 1, k, k, q, g32, beta32,
              ex.view(A, torch.uint32), rstd_k, rstd_q, g_cs, ex.view(qn, torch.uint32), ex.view(kn, torch.uint32),
              meta.chunk_start, meta.chunk_len, meta.num_chunks, num_heads, num_v_heads)  # fmt: skip
    return qn, kn, rstd_q, rstd_k, g_cs, A


def _forward_tape_tensors(kept, state, device, *, ex, formed=()):
    """The state tensors of a forward on wy_a's A by :data:`cake_plan.STATE_NAMES`: an allocation for each name the
    tape keeps (``kept``) or the forward forms in memory (``formed``), ``None`` for the rest."""
    return {
        name: ex.empty(shape, dtype=dtype, device=device)
        if name in kept or name in formed
        else None
        for name, (shape, dtype) in zip(cake_plan.STATE_NAMES, state, strict=True)
    }


def _forward_result(fwd: dict[str, Any], kept, forward: str) -> dict[str, Any]:
    """The tape dict of a forward on wy_a's A: the state the tape keeps, ``None`` for the rest (the backward rebuilds
    it)."""
    for name in cake_plan.STATE_NAMES:
        if name not in kept:
            fwd[name] = None
    fwd["mode"] = _TAPE_MODE
    fwd["forward"] = forward
    return fwd


def _forward_a(q, k, v, g32, beta32, *, meta, arch, scale, initial_state, output_final_state, normalize_qk, options,
               tape, ex) -> dict[str, Any]:  # fmt: skip
    """The forward of :func:`fwd_a_route`'s calls: ``wy_a`` then ``fwd_ho_a``; the tape of
    :func:`cake_gdn_det_train_forward` (``None`` for the state it does not keep: u always)."""
    num_heads, num_v_heads = q.shape[-2], v.shape[-2]
    total_tokens = q.numel() // (num_heads * HEAD_DIM)
    device = q.device
    index = _tensor_index(q)
    bf16 = q.dtype
    q, k = _tma_operands(q, k, ex)
    # fwd_ho_a's tensor map reads v in place where the stages that read v by rows do (:func:`_v_rows`), else a copy
    if _v_rows(v) is None:
        v = ex.aligned(v, ROW_ALIGN_BYTES)
    kept = cake_plan.tape_state(arch, options.save_state) if tape else ()
    qn, kn, rstd_q, rstd_k, g_cs, A = _launch_wy_a(
        q, k, g32, beta32, meta, num_heads, num_v_heads, index, ex
    )
    state = _forward_tape_tensors(kept, _forward_state(meta, kn, v), device, ex=ex)
    o = ex.empty(v.shape, dtype=v.dtype, device=device)
    workspaces = ex.workspaces(options)
    table = workspaces.table(index)
    # fwd_ho_a binds every tape tensor it does not write to a one-row stand-in (its tensor map encodes, nothing stores)
    word = sum(FWD_A_TAPE[name] for name in kept if name in FWD_A_TAPE)
    rows = workspaces.take(table, "fa_rows_off", (1, 1, HEAD_DIM), bf16, device)
    h_t = state["h"]
    if h_t is None:
        h_t = workspaces.take(
            table, "fa_h_off", (1, 1, HEAD_DIM, HEAD_DIM), bf16, device
        )
    v_new_t = rows if state["v_new"] is None else state["v_new"]
    w_t = rows if state["w"] is None else state["w"]
    final_state, h0, use_init = _start_state(initial_state, meta, output_final_state, num_v_heads, table, workspaces,
                                             device, ex)  # fmt: skip
    segs = meta.fwd_segments
    walk = _walk(device, segs, options, ex)
    ex.launch("fwd_ho_a", segs.num_segments * num_v_heads, 1, 1, qn, kn, v, A, g_cs, beta32, h0, final_state, h_t,
              h_t, v_new_t, v_new_t, o, o, final_state, w_t, w_t, meta.chunk_start, meta.chunk_len,
              segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, walk, num_heads, num_v_heads, use_init,
              1 if output_final_state else 0, 0, float(scale), word)  # fmt: skip
    if output_final_state:
        final_state = ex.caller_rows(final_state, meta)
    fwd: dict[str, Any] = {
        "arch": arch, "normalize_qk": normalize_qk, "num_heads": num_heads, "num_v_heads": num_v_heads,
        "total_tokens": total_tokens, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "g_cs": g_cs, "A": A,
        "v": v, "beta32": beta32, "meta": meta, "scale": float(scale), "output": o,
        "final_state": final_state if output_final_state else None, "initial_state": initial_state, "options": options,
        **state,
    }  # fmt: skip
    return _forward_result(fwd, kept, "fwd_ho_a")


# --------------------------------------------------------------------------- #
# the split forward (the rows whose forward segment table splits a sequence)
# --------------------------------------------------------------------------- #
# On an architecture of ``FWD_SPLIT_ARCHS`` a call the forward on wy_a's A serves (:func:`_wy_a_inputs`) on a
# host-built table whose forward segment table splits a sequence runs the one-wave split of the forward scan on the
# host plan's piece table (:func:`cake_plan.pieces`) instead of the pre-scan, the stitch and ``fwd_ho``: the front
# ``wy_a`` (A, rstd_q / rstd_k / g_cs / qn / kn) and the WY stage with ``recompute = 1`` on that A (w and u), then
# wave 1, the first piece of every split sequence from its initial state (``fwd_fused``, its end state into the B half
# of the stitch's segment state by ``split_out``, its flag 0) beside the affine maps ``[B | M]`` of the middle pieces
# (``fwd_summary``, side stream, joined back before the stitch), the stitch (``reverse = 0``: the start state of every
# piece) and wave 2, every other piece from its stitched start state (``fwd_fused``; a sequence of one piece runs
# whole there from its initial state, which the stitch copies).  ``fwd_fused`` is ``fwd_ho``'s scan and output on
# the WY stage's w / u with the tape word of :data:`FWD_FUSED_TAPE`: h and v_new when the call keeps a tape, none
# without one.  Wave 1 and the summary write disjoint segments, every piece's state is composed by the stitch in a
# fixed order and no launch uses an atomic on a value, so the forward is bitwise run to run with one or two streams.
# The rows whose pieces are shorter than the architecture's :data:`cake_plan.FORWARD_SPLIT_MIN_PIECE` keep the prep
# forward (there the split's fixed costs exceed what it saves), as do the calls ``scan_config`` / ``segment_chunks``
# pin.  The piece table depends on the sequence lengths, HV and the SM count alone.
FWD_SPLIT_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a")
# fwd_fused's tape word per tape entry
FWD_FUSED_TAPE = {"h": 1, "v_new": 2}
# On an architecture of ``FWD_SPLIT_A_ARCHS`` (a subset of ``FWD_SPLIT_ARCHS``) the split forward runs on wy_a's A
# alone (no WY stage, u and w never formed in memory): both waves are ``fwd_ho_a`` (with the h / v_new / w tapes of
# :data:`FWD_A_TAPE`) on the split tables, and the middle pieces' maps are ``fwd_summary_a`` (the same maps and layout
# as ``fwd_summary``, u and w formed in the CTA).  ``fwd_ho_a`` has no ``split_out``: wave 1 stores the end state of
# each first piece as its sequence's final state, which is copied into the B half of the segment state (one row per
# first piece, distinct rows), the first pieces' flags zeroed, before the stitch.
FWD_SPLIT_A_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a")
# the shortest piece (chunks, the planned piece length) whose split forward runs on wy_a's A, per architecture (none
# listed: every split forward).  Measured with the non-deterministic package's programs on GB300 (sm_103a): the
# 63-chunk pieces of three sequences of about 11k tokens at HV 24 take 1.03 of the W form's time per checkpointed
# step, the pieces of 132 chunks and more 0.90-0.91 per forward; B200 (sm_100a) takes the A form on every split row.
FWD_SPLIT_A_MIN_PIECE: Mapping[str, int] = MappingProxyType({"sm_103a": 96})
# the stages of the split forward on wy_a's A, and of the split forward on the WY stage's w / u
FWD_SPLIT_A_STAGES = ("wy_a", "fwd_ho_a", "fwd_summary_a", "stitch")
FWD_SPLIT_W_STAGES = ("wy_a", "wy_tc2", "fwd_fused", "fwd_summary", "stitch")
# fwd_summary's output layout into the stitch's [B | M] FP32 [K][V + K] block: b_head, b_row, m_head, m_row, m_col
_SUMMARY_LAYOUT = (
    HEAD_DIM * 2 * HEAD_DIM,
    2 * HEAD_DIM,
    HEAD_DIM * 2 * HEAD_DIM,
    2 * HEAD_DIM,
    HEAD_DIM,
)


@dataclass(frozen=True)
class ForwardSplit:
    """The tables of the one-wave split forward over a piece table (:func:`forward_split_tables`): the pieces as a
    segment table (``seg_chunk_start``, ``seg_seq``, ``seq_seg_start``), ``first_end`` (the table's
    ``seq_chunk_start`` with each split sequence ending at the end of its first piece, so wave 1 stores that piece's
    end state), the walks of wave 1 (``first``: each split sequence's first piece), of the summary (``middle``) and of
    wave 2 (``rest``: every other piece, and the one piece of every unsplit sequence), and ``slot`` (the identity: the
    summary's map of piece ``s`` goes to segment ``s``); ``first_index`` (``first`` as int64, distinct segments) and
    ``first_seq`` (int64, the sequence of each first piece; ``None`` where every sequence splits, in table order) index
    the copy of wave 1's end states into the segment state when the waves run ``fwd_ho_a``
    (:data:`FWD_SPLIT_A_ARCHS`).  ``one_stream``: the summary then wave 1 on the caller's stream instead of the
    summary on a side stream (the same kernels and bits).  ``ready``: the write of cached tables (both copies;
    :func:`_use_forward_split`; ``None``: tables built for their call)."""

    seg_chunk_start: torch.Tensor
    seg_seq: torch.Tensor
    seq_seg_start: torch.Tensor
    first_end: torch.Tensor
    first: torch.Tensor
    middle: torch.Tensor
    rest: torch.Tensor
    slot: torch.Tensor
    first_index: torch.Tensor
    first_seq: torch.Tensor | None
    num_segments: int
    num_first: int
    num_middle: int
    num_rest: int
    length: int
    one_stream: bool = False
    ready: TableReady | None = None


def forward_split_lists(chunks, length: int) -> tuple[list[int], ...]:
    """The host lists of :class:`ForwardSplit` for sequences of ``chunks`` chunks (table order) at piece length
    ``length``: ``seg_chunk_start``, ``seg_seq``, ``seq_seg_start``, ``first_end``, ``first``, ``middle``, ``rest``
    (pieces by :func:`segment_bounds`: balanced to within one chunk, ``ceil(n / length)`` of them)."""
    seg_chunk_start: list[int] = [0]
    seg_seq: list[int] = []
    seq_seg_start: list[int] = [0]
    first_end: list[int] = [0]
    first: list[int] = []
    middle: list[int] = []
    rest: list[int] = []
    for seq, n in enumerate(int(c) for c in chunks):
        c0, s0 = seg_chunk_start[-1], len(seg_seq)
        bounds = segment_bounds(n, int(length))
        seg_chunk_start.extend(c0 + b for b in bounds[1:])
        seg_seq.extend([seq] * (len(bounds) - 1))
        seq_seg_start.append(len(seg_seq))
        count = len(bounds) - 1
        if count >= 2:
            first.append(s0)
            first_end.append(c0 + bounds[1])
            middle.extend(range(s0 + 1, s0 + count - 1))
            rest.extend(range(s0 + 1, s0 + count))
        else:
            first_end.append(c0 + n)
            rest.append(s0)
    return seg_chunk_start, seg_seq, seq_seg_start, first_end, first, middle, rest


def forward_split_tables(
    chunks: tuple[int, ...], length: int, device_index: int, one_stream: bool = False
) -> ForwardSplit:
    """:class:`ForwardSplit` of sequences of ``chunks`` chunks (table order) at piece length ``length``, copied to CUDA
    device ``device_index`` in two transfers (the int32 tables, the int64 indices).  Not cached here: the host route
    caches it with its write (:func:`_host_forward_split`)."""
    lists = forward_split_lists(chunks, length)
    num_segments = len(lists[1])
    parts = [*lists, list(range(num_segments))]
    # an empty walk binds a one-entry table (its launch is skipped)
    parts = [part if part else [0] for part in parts]
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device("cuda", int(device_index))
    table = table.pin_memory().to(device, non_blocking=True)
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    # the int64 indices of the copy of wave 1's end states (fwd_ho_a waves): first pieces and their sequences
    first = lists[4]
    first_seq = [lists[1][s] for s in first]
    index = torch.tensor(
        first + (first_seq if first_seq != list(range(len(chunks))) else []),
        dtype=torch.int64,
    )
    index = index.pin_memory().to(device, non_blocking=True)
    first_index = index[: len(first)]
    first_seq_index = index[len(first) :] if index.numel() > len(first) else None
    return ForwardSplit(seg_chunk_start=views[0], seg_seq=views[1], seq_seg_start=views[2], first_end=views[3],
                        first=views[4], middle=views[5], rest=views[6], slot=views[7],
                        first_index=first_index, first_seq=first_seq_index, num_segments=num_segments,
                        num_first=len(first), num_middle=len(lists[5]), num_rest=len(lists[6]), length=int(length),
                        one_stream=bool(one_stream))  # fmt: skip


def forward_split_length(
    meta: ChunkMeta, arch: str, num_v_heads: int, device_index: int
) -> int | None:
    """The piece length of the split forward of a host-built ``meta`` on ``arch``: the host plan's
    (:func:`cake_plan.host_plan` on the planner's SM count) where ``meta``'s forward segment table splits a sequence
    and the row is in the split's shape class (:func:`cake_plan.forward_split_length`), else ``None``."""
    segs = meta.fwd_segments
    if arch not in FWD_SPLIT_ARCHS or meta.seq_lens is None:
        return None
    if not (segs.segment_len > 0 and segs.num_segments > meta.num_seqs):
        return None
    index = int(device_index)
    plan = cake_plan.host_plan(
        tuple(meta.seq_lens), int(num_v_heads), arch, _device_props(index)[0]
    )
    return cake_plan.forward_split_length(arch, plan.pieces)


def _host_forward_split(
    meta: ChunkMeta, num_v_heads: int, device_index: int, options: ChainOptions
) -> ForwardSplit | None:
    """:func:`forward_split_tables` of a host-built ``meta`` (its sequences in table order; wave 1 on one stream under
    ``streams = 1``), cached in ``_FORWARD_SPLITS`` for the ``_FORWARD_SPLITS_LIMIT`` most recently used keys with
    their write (outside a CUDA graph capture; a hit is made safe to read, :func:`_use_forward_split`), or ``None``
    where :func:`forward_split_length` gives none or ``options`` pin the scan tables (``scan_config`` /
    ``segment_chunks``: the prep forward's split scan runs as pinned)."""
    if options.scan_config is not None or options.segment_chunks is not None:
        return None
    index = int(device_index)
    length = forward_split_length(
        meta, cake_backend.device_arch(index), num_v_heads, index
    )
    if length is None:
        return None
    chunks = cake_plan.chunk_counts(cast("tuple[int, ...]", meta.seq_lens))
    if meta.order is not None:
        chunks = tuple(chunks[seq] for seq in meta.order)
    key = (tuple(chunks), int(length), index, options.streams == 1)
    split = _FORWARD_SPLITS.pop(key, None)
    if split is not None:
        _FORWARD_SPLITS[key] = split
        _use_forward_split(split, index)
        return split
    split = forward_split_tables(*key)
    # a table built inside a CUDA graph capture (see "cached device tables") is not kept
    if torch.cuda.is_current_stream_capturing():
        return split
    split = replace(split, ready=_table_ready(index))
    if len(_FORWARD_SPLITS) >= _FORWARD_SPLITS_LIMIT:
        _FORWARD_SPLITS.pop(next(iter(_FORWARD_SPLITS)), None)
    _FORWARD_SPLITS[key] = split
    return split


def fwd_split_route(
    meta: ChunkMeta, arch: str, q, k, v, beta, normalize_qk: bool, options: ChainOptions
) -> bool:
    """Whether a forward on ``meta`` runs the split forward (see :data:`FWD_SPLIT_ARCHS`): ``meta`` carries its tables
    (:attr:`ChunkMeta.fwd_split`), the call meets :func:`_fwd_a_inputs` and the records of its form are delivered."""
    split = meta.fwd_split
    if split is None or arch not in FWD_SPLIT_ARCHS:
        return False
    if not _fwd_a_inputs(meta, q, k, v, beta, normalize_qk, options):
        return False
    return _fwd_split_a(arch, split.length) or all(
        _available(stage, arch) for stage in FWD_SPLIT_W_STAGES
    )


def _fwd_split_a(arch: str, length: int) -> bool:
    """Whether the split forward on ``arch`` at piece length ``length`` (chunks) runs on wy_a's A
    (:data:`FWD_SPLIT_A_ARCHS` within ``FWD_SPLIT_ARCHS``, pieces of at least :data:`FWD_SPLIT_A_MIN_PIECE` chunks,
    its stages delivered): a shape-only choice, from the host plan's piece length."""
    return (
        arch in FWD_SPLIT_ARCHS
        and arch in FWD_SPLIT_A_ARCHS
        and int(length) >= FWD_SPLIT_A_MIN_PIECE.get(arch, 0)
        and all(_available(stage, arch) for stage in FWD_SPLIT_A_STAGES)
    )


def _split_workspaces(split: ForwardSplit, num_v_heads: int, table, workspaces, device):
    """The stitch's segment state ``[B | M]``, its flags and the pieces' start states of a split forward."""
    f32 = torch.float32
    seg_state = workspaces.take(table, "f_seg_state", (split.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM), f32,
                                device)  # fmt: skip
    seg_flag = workspaces.take(
        table, "f_seg_flag", (split.num_segments, num_v_heads), torch.int32, device
    )
    seg_start = workspaces.take(table, "f_seg_start", (split.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM), f32,
                                device)  # fmt: skip
    return seg_state, seg_flag, seg_start


def _first_wave(split: ForwardSplit, index: int, summary, wave1, ex) -> None:
    """Wave 1 (``wave1()``) beside the middle pieces' maps (``summary()``, on a side stream joined back before this
    returns, also when a launch raises); both on the caller's stream, the summary first, under ``one_stream`` or where
    either has no piece.  The two write disjoint segments, so the bits are the same either way."""
    if not split.num_middle or not split.num_first or split.one_stream:
        if split.num_middle:
            summary()
        if split.num_first:
            wave1()
        return
    # the general path (_DIRECT) makes the executor's fork and join in place, with no frame of an executor step
    if ex is _DIRECT:
        main = torch.cuda.current_stream(index)
        side = _side_stream(index, main)
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        try:
            with torch.cuda.stream(side):
                summary()
            wave1()
        finally:
            main.wait_stream(side)  # the join
        return
    main, side = ex.fork(index)
    try:
        with ex.on(side):
            summary()
        wave1()
    finally:
        ex.join(main, side)  # the join


def _forward_split(q, k, v, g32, beta32, *, meta, arch, scale, initial_state, output_final_state, normalize_qk, options,
                   tape, ex) -> dict[str, Any]:  # fmt: skip
    """The forward of :func:`fwd_split_route`'s calls (:data:`FWD_SPLIT_ARCHS`); the tape of
    :func:`cake_gdn_det_train_forward` (``None`` for the state it does not keep)."""
    split = cast("ForwardSplit", meta.fwd_split)
    if _fwd_split_a(arch, split.length):
        return _forward_split_a(q, k, v, g32, beta32, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                                output_final_state=output_final_state, normalize_qk=normalize_qk, options=options,
                                tape=tape, ex=ex)  # fmt: skip
    num_heads, num_v_heads = q.shape[-2], v.shape[-2]
    total_tokens = q.numel() // (num_heads * HEAD_DIM)
    device = q.device
    index = _tensor_index(q)
    bf16 = q.dtype
    hv = num_v_heads
    q, k = _tma_operands(q, k, ex)
    # wy_tc2 reads v by rows at its token pitch (:func:`_v_rows`); any other layout, a contiguous v whose base is not
    # 16-byte aligned among them, is copied into contiguous rows
    v_rows = _v_rows(v)
    if v_rows is None:
        v, v_rows = ex.aligned(v, ROW_ALIGN_BYTES), num_v_heads
    kept = cake_plan.tape_state(arch, options.save_state) if tape else ()
    qn, kn, rstd_q, rstd_k, g_cs, A = _launch_wy_a(
        q, k, g32, beta32, meta, num_heads, num_v_heads, index, ex
    )
    # w and u are formed in memory either way (both waves and the summary read them); h and v_new only on the tape
    state = _forward_tape_tensors(
        kept, _forward_state(meta, kn, v), device, formed=("w", "u"), ex=ex
    )
    w, u = state["w"], state["u"]
    o = ex.empty(v.shape, dtype=v.dtype, device=device)
    workspaces = ex.workspaces(options)
    table = workspaces.table(index)
    word = sum(FWD_FUSED_TAPE[name] for name in kept if name in FWD_FUSED_TAPE)
    rows = workspaces.take(table, "fa_rows_off", (1, 1, HEAD_DIM), bf16, device)
    h_t = state["h"]
    if h_t is None:
        h_t = workspaces.take(
            table, "fa_h_off", (1, 1, HEAD_DIM, HEAD_DIM), bf16, device
        )
    v_new_t = rows if state["v_new"] is None else state["v_new"]
    final_state, h0, use_init = _start_state(initial_state, meta, output_final_state, num_v_heads, table, workspaces,
                                             device, ex)  # fmt: skip
    seg_state, seg_flag, seg_start = _split_workspaces(
        split, hv, table, workspaces, device
    )
    ex.launch("wy_tc2", meta.num_chunks, hv, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start, meta.chunk_len,
              num_heads, hv, 1, v_rows)  # fmt: skip

    def fused(walk, count, ends, use_seg, store_final, split_out):
        ex.launch("fwd_fused", count * hv, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h_t, v_new_t, v_new_t, o, o,
                  final_state, meta.chunk_start, meta.chunk_len, split.seg_chunk_start, split.seg_seq, ends, walk,
                  num_heads, hv, use_init, store_final, use_seg, float(scale), word, seg_state, seg_flag,
                  split_out)  # fmt: skip

    def summary():
        ex.launch("fwd_summary", split.num_middle * hv, 1, 1, kn, w, u, g_cs, seg_state, seg_state, seg_flag,
                  meta.chunk_start, meta.chunk_len, split.seg_chunk_start, split.middle, split.slot, num_heads, hv,
                  *_SUMMARY_LAYOUT)  # fmt: skip

    _first_wave(
        split,
        index,
        summary,
        lambda: fused(split.first, split.num_first, split.first_end, 0, 1, 1),
        ex,
    )
    ex.launch("stitch", meta.num_seqs, hv, STITCH_VSPLIT, seg_state, h0, seg_start, split.seq_seg_start, seg_flag, hv,
              use_init, 0)  # fmt: skip
    fused(
        split.rest,
        split.num_rest,
        meta.seq_chunk_start,
        1,
        1 if output_final_state else 0,
        0,
    )
    if output_final_state:
        final_state = ex.caller_rows(final_state, meta)
    fwd: dict[str, Any] = {
        "arch": arch, "normalize_qk": normalize_qk, "num_heads": num_heads, "num_v_heads": num_v_heads,
        "total_tokens": total_tokens, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "g_cs": g_cs, "A": A,
        "v": v, "beta32": beta32, "meta": meta, "scale": float(scale), "output": o,
        "final_state": final_state if output_final_state else None, "initial_state": initial_state, "options": options,
        **state,
    }  # fmt: skip
    return _forward_result(fwd, kept, "fwd_split")


def _index_copy_heads(state, index, rows) -> None:
    """``state[:, :, :, :HEAD_DIM].index_copy_(0, index, rows)``: the first pieces' end states into the stitch's
    segment rows (a view of a slot that a plan cannot hold as a dim-0 range, made inside one eager step)."""
    state[:, :, :, :HEAD_DIM].index_copy_(0, index, rows)


def _forward_split_a(q, k, v, g32, beta32, *, meta, arch, scale, initial_state, output_final_state, normalize_qk,
                     options, tape, ex) -> dict[str, Any]:  # fmt: skip
    """The split forward on wy_a's A (:data:`FWD_SPLIT_A_ARCHS`): ``wy_a``, then wave 1 (``fwd_ho_a`` on the first
    pieces, their end states as the final states) beside the middle pieces' maps (``fwd_summary_a``, side stream), the
    copy of those end states into the segment state, the stitch and wave 2 (``fwd_ho_a`` on every other piece from its
    stitched start state); the tape of :func:`cake_gdn_det_train_forward`, as the serial route's (u ``None``)."""
    split = cast("ForwardSplit", meta.fwd_split)
    num_heads, num_v_heads = q.shape[-2], v.shape[-2]
    total_tokens = q.numel() // (num_heads * HEAD_DIM)
    device = q.device
    index = _tensor_index(q)
    f32, bf16 = torch.float32, q.dtype
    hv = num_v_heads
    q, k = _tma_operands(q, k, ex)
    # fwd_ho_a's and fwd_summary_a's tensor maps read v in place where the stages that read v by rows do, else a copy
    if _v_rows(v) is None:
        v = ex.aligned(v, ROW_ALIGN_BYTES)
    kept = cake_plan.tape_state(arch, options.save_state) if tape else ()
    qn, kn, rstd_q, rstd_k, g_cs, A = _launch_wy_a(
        q, k, g32, beta32, meta, num_heads, num_v_heads, index, ex
    )
    state = _forward_tape_tensors(kept, _forward_state(meta, kn, v), device, ex=ex)
    o = ex.empty(v.shape, dtype=v.dtype, device=device)
    workspaces = ex.workspaces(options)
    table = workspaces.table(index)
    word = sum(FWD_A_TAPE[name] for name in kept if name in FWD_A_TAPE)
    rows = workspaces.take(table, "fa_rows_off", (1, 1, HEAD_DIM), bf16, device)
    h_t = state["h"]
    if h_t is None:
        h_t = workspaces.take(
            table, "fa_h_off", (1, 1, HEAD_DIM, HEAD_DIM), bf16, device
        )
    v_new_t = rows if state["v_new"] is None else state["v_new"]
    w_t = rows if state["w"] is None else state["w"]
    final_state, h0, use_init = _start_state(initial_state, meta, output_final_state, num_v_heads, table, workspaces,
                                             device, ex)  # fmt: skip
    state_shape = (meta.num_seqs, hv, HEAD_DIM, HEAD_DIM)
    # wave 1 stores the first pieces' end states as final states: the final-state output, or a buffer of every
    # sequence's rows
    first_final = (
        final_state
        if output_final_state
        else workspaces.take(table, "f_first_final", state_shape, f32, device)
    )
    seg_state, seg_flag, seg_start = _split_workspaces(
        split, hv, table, workspaces, device
    )

    def wave(walk, count, ends, init, use_seg, store_final, final):
        ex.launch("fwd_ho_a", count * hv, 1, 1, qn, kn, v, A, g_cs, beta32, h0, seg_start, h_t, h_t, v_new_t, v_new_t,
                  o, o, final, w_t, w_t, meta.chunk_start, meta.chunk_len, split.seg_chunk_start, split.seg_seq, ends,
                  walk, num_heads, hv, init, store_final, use_seg, float(scale), word)  # fmt: skip

    def summary():
        ex.launch("fwd_summary_a", split.num_middle * hv, 1, 1, kn, v, A, g_cs, beta32, seg_state, seg_state, seg_flag,
                  meta.chunk_start, meta.chunk_len, split.seg_chunk_start, split.middle, split.slot, num_heads, hv,
                  *_SUMMARY_LAYOUT)  # fmt: skip

    if split.num_first:
        # the first pieces' segments hold their end states (flag 0: a state, not a map) for the stitch; the segments
        # of first_index are distinct, so the fill is a plain store
        ex.index_fill_(seg_flag, 0, split.first_index, 0)
    _first_wave(
        split,
        index,
        summary,
        lambda: wave(
            split.first, split.num_first, split.first_end, use_init, 0, 1, first_final
        ),
        ex,
    )
    if split.num_first:
        ends = (
            first_final
            if split.first_seq is None
            else ex.index_select(first_final, 0, split.first_seq)
        )
        # one distinct segment row per first piece: a plain copy, the same bits under any order
        if ex is _DIRECT:  # the general path makes the eager step in place
            seg_state[:, :, :, :HEAD_DIM].index_copy_(0, split.first_index, ends)
        else:
            ex.index_copy_heads(seg_state, split.first_index, ends)
    ex.launch("stitch", meta.num_seqs, hv, STITCH_VSPLIT, seg_state, h0, seg_start, split.seq_seg_start, seg_flag, hv,
              use_init, 0)  # fmt: skip
    wave(
        split.rest,
        split.num_rest,
        meta.seq_chunk_start,
        0,
        1,
        1 if output_final_state else 0,
        final_state,
    )
    if output_final_state:
        final_state = ex.caller_rows(final_state, meta)
    fwd: dict[str, Any] = {
        "arch": arch, "normalize_qk": normalize_qk, "num_heads": num_heads, "num_v_heads": num_v_heads,
        "total_tokens": total_tokens, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "g_cs": g_cs, "A": A,
        "v": v, "beta32": beta32, "meta": meta, "scale": float(scale), "output": o,
        "final_state": final_state if output_final_state else None, "initial_state": initial_state, "options": options,
        **state,
    }  # fmt: skip
    return _forward_result(fwd, kept, "fwd_split")


def cake_gdn_det_train_forward(q, k, v, g, beta, *, meta, arch, scale=None, initial_state=None,
                               output_final_state=False, normalize_qk=True, options: ChainOptions | None = None,
                               tape: bool = True) -> dict[str, Any]:  # fmt: skip
    """The forward chain over flattened ``[T, H, 128]`` / ``[T, HV, 128]`` inputs and one chunk table; returns the
    output and the tape the backward reads (``"forward"``: ``"fwd_split"``, ``"fwd_ho_a"`` or ``"fwd_ho"``).  A call
    of :func:`fwd_split_route` runs the split forward, one of :func:`fwd_a_route` the forward on wy_a's A, every
    other call the prep forward.  The forwards on wy_a's A put in the tape only the state the architecture's tape level
    keeps (:func:`cake_plan.tape_state`; ``None`` for the rest, u always), and none when ``tape`` is ``False`` (a call
    that records no backward); the backward rebuilds what the tape lacks.  The prep forward's tape carries all four."""
    return _train_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                          output_final_state=output_final_state, normalize_qk=normalize_qk, options=options,
                          tape=tape, ex=_DIRECT)  # fmt: skip


def _train_forward(q, k, v, g, beta, *, meta, arch, scale, initial_state, output_final_state, normalize_qk, options,
                   tape, ex) -> dict[str, Any]:  # fmt: skip
    """:func:`cake_gdn_det_train_forward` with its allocations, views, eager copies, launches, stream switches and
    cached-table uses through the executor ``ex`` (:data:`_DIRECT`, or a :class:`_FwdRecorder` that also records
    them as a :class:`_ForwardPlan`), in the same order."""
    options = options or ChainOptions()
    if not q.is_cuda:
        # a host q reaches the first launch's operand checks, which refuse it; the buffers taken before them would
        # stay pooled, as host tensors, under the current device's key (_tensor_index), so such a call pools none
        options = replace(options, host_cache=False)
    num_heads, key_dim = q.shape[-2], q.shape[-1]
    num_v_heads, value_dim = v.shape[-2], v.shape[-1]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("cake_gdn_det_train_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    total_tokens = q.numel() // (num_heads * key_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    # the route is decided from host facts (the table, the architecture, shapes, dtypes, pins) before any operand is
    # copied; the forwards on wy_a's A then read g and beta as contiguous rows at a 16-byte aligned base
    split = fwd_split_route(meta, arch, q, k, v, beta, normalize_qk, options)
    if split or fwd_a_route(meta, arch, q, k, v, beta, normalize_qk, options):
        g32 = g if g.dtype == torch.float32 else ex.float(g)
        g_rows = ex.aligned(g32, ROW_ALIGN_BYTES)
        beta_rows = ex.aligned(beta, ROW_ALIGN_BYTES)
        run = _forward_split if split else _forward_a
        return run(q, k, v, g_rows, beta_rows, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                   output_final_state=output_final_state, normalize_qk=normalize_qk, options=options,
                   tape=tape, ex=ex)  # fmt: skip
    device = q.device
    # q and k stay views when prep_tma reads them in place (a packed row's head views; inside the rule's plain band
    # too, :data:`VIEWS_TAKE_PREP_TMA`); v stays a view when the stages that read it take its row pitch (wy_tc reads
    # contiguous rows only); an operand whose address the loads cannot take is copied
    in_place = _tma_rows(q) and _tma_rows(k)
    if not in_place:
        q, k = ex.aligned(q, PREP_ALIGN_BYTES), ex.aligned(k, PREP_ALIGN_BYTES)
    views = VIEWS_TAKE_PREP_TMA and _view_rows(q) and _view_rows(k)
    prep = _prep_stage(q, k, meta.num_chunks, num_heads, arch, options, views=views)
    if prep != "prep_tma" and in_place:
        q, k = ex.contiguous(q), ex.contiguous(k)
    v_rows = _v_rows(v) if wy_stage(arch) == "wy_tc2" else None
    if v_rows is None:
        v = ex.aligned(v, ROW_ALIGN_BYTES)
        v_rows = num_v_heads
    g32 = ex.contiguous(g)
    if g32.dtype != torch.float32:
        g32 = ex.float(g32)
    beta = ex.contiguous(beta)
    beta_bf16 = beta.dtype == torch.bfloat16
    if beta_bf16:
        beta32 = ex.empty(
            (total_tokens, num_v_heads), dtype=torch.float32, device=device
        )
    else:
        beta32 = beta if beta.dtype == torch.float32 else ex.float(beta)
    qn = ex.empty(q.shape, dtype=q.dtype, device=device)
    kn = ex.empty(k.shape, dtype=k.dtype, device=device)
    rstd_q = ex.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    rstd_k = ex.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    g_cs = ex.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    ex.launch(prep + ("_bf16beta" if beta_bf16 else ""), meta.num_chunks,
              num_heads if prep == "prep_tma" else num_v_heads, 1, q, k, g32, beta, beta32, qn, kn, rstd_q, rstd_k,
              g_cs, meta.chunk_start, meta.chunk_len, num_heads, num_v_heads, 1 if normalize_qk else 0)  # fmt: skip
    A = ex.empty((total_tokens, num_v_heads, CHUNK), dtype=q.dtype, device=device)
    if meta.fwd_groups is None:
        w, u = _launch_wy(
            arch,
            kn,
            v,
            g_cs,
            beta32,
            A,
            meta,
            num_heads,
            num_v_heads,
            total_tokens,
            recompute=False,
            v_rows=v_rows,
            ex=ex,
        )
        wy = None
    else:  # wy_tc runs with fwd_ho as two sequence groups
        w, u = _wy_outputs(kn, v, total_tokens, num_v_heads, ex)
        wy = (v, beta32, A, v_rows)
    workspaces = ex.workspaces(options)
    table = workspaces.table(_tensor_index(kn))
    h, v_new, final_state, o = _launch_fwd_ho(arch, qn, kn, w, u, g_cs, ex.table_rows(initial_state, meta),
                                              meta, num_heads, num_v_heads, store_final_state=output_final_state,
                                              table=table, workspaces=workspaces, options=options, scale=scale,
                                              wy=wy, ex=ex)  # fmt: skip
    if output_final_state:
        final_state = ex.caller_rows(final_state, meta)
    return {
        "arch": arch, "normalize_qk": normalize_qk, "num_heads": num_heads, "num_v_heads": num_v_heads,
        "total_tokens": total_tokens, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "g_cs": g_cs, "A": A,
        "w": w, "u": u, "v": v, "beta32": beta32, "meta": meta, "scale": float(scale), "output": o,
        "final_state": final_state if output_final_state else None, "initial_state": initial_state, "h": h,
        "v_new": v_new, "options": options, "mode": _TAPE_MODE, "forward": "fwd_ho",
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# the forward's executor and its recorded plan
# --------------------------------------------------------------------------- #
# The forward's host work (the route's checks, its allocations and copies, every launch's arguments in its record's
# parameter order) is the same for every call of one class: one layout on one stream, one tape mode (a checkpoint's
# first pass and its recompute are two), one set of arguments, and inputs of one shape, strides, dtype and 16-byte
# base alignment each, the same ones being one object.  The autograd functions record it once per class, as the
# straight step sequence of a :class:`_ForwardPlan` (the general path run through a :class:`_FwdRecorder`), and replay
# that sequence afterwards (:func:`_replay_forward`): the inputs and the tensors the call allocates, views and computes
# eagerly fill its slots, and so do the pooled workspaces of the stream, which each replay takes from the pool again as
# the general path takes them (a hit, or a miss that allocates and evicts); every other operand is a constant of the
# plan (a chunk table of the layout's ``ChunkMeta``, the cached walk, a scalar).  A replay makes the general path's
# allocations, workspace takes, eager calls, launches and event records and waits in its order on its streams, with the
# same arguments: the same kernels and the same bits.  The plan holds no tensor of a call and no pooled workspace (an
# eviction leaves it in place), a replay lets each tensor of the call go after its last step (:func:`_drop_steps`), no
# later than the general path does, so it needs no more device memory, and the plans of a layout leave with its
# ``ChunkMeta`` (:func:`_drop_meta_plans`).
# ``FWD_PLAN = False`` keeps every forward on the general path.
FWD_PLAN = True
# The record gate (``FWD_PLAN_GATE``; it picks among the three paths, which give the same bits, by host time only): a
# forward builds its plan key only from the ``FWD_PLAN_REPEAT``-th forward of its layout in its tape mode that the gate
# counts, and records only while the record credit holds one recording.  Each replay adds ``1 / FWD_PLAN_PAYBACK``
# recording to the credit, each gated call ``1 / FWD_PLAN_TRICKLE``, a recording takes one, and the credit starts at
# and holds at most ``FWD_PLAN_BUDGET``.  So for any sequence of calls of one thread, recordings <= BUDGET + replays /
# PAYBACK + calls / TRICKLE, and with PAYBACK at least the host time of a recording over the host time a replay saves,
# the recordings cost no more host time than the replays save (the additions take no lock: an addition of one thread
# that overlaps the spend of another can undo that spend, one recording more per such overlap).  While the credit holds
# less than one recording, a gated call of a layout with no plan kept on it (:data:`_PLAN_KEYS`) adds its share and
# takes the general path in :func:`_function_forward` without calling :func:`_fwd_gate` (it could neither replay nor
# record, and the gate's call, its code and entries cold in the caches, takes a few us of host time): such a call
# makes no entry and does not count toward its layout's ``FWD_PLAN_REPEAT`` forwards.  A layout used once never builds
# a key.  A key whose calls cannot be planned keeps its layout and tape mode on the general path without a key; the
# gate's entry is the layout and tape mode's, which all their call classes share.  ``FWD_PLAN_GATE = False``: every
# key records at its first call.
FWD_PLAN_GATE = True
FWD_PLAN_REPEAT = 2
FWD_PLAN_PAYBACK = 32
FWD_PLAN_BUDGET = 4
FWD_PLAN_TRICKLE = 4096
# (id(meta), tape) -> [weakref(meta), its forwards so far (at most FWD_PLAN_REPEAT), its state]; the oldest entry
# leaves first at _FWD_SEEN_LIMIT entries (two tape modes per cached layout).  A first sighting puts its entry in and
# lets the oldest go with one call each (OrderedDict.popitem), so it takes no lock: threads that sight layouts at once
# can let one entry more go, or count one forward of a layout twice
_FWD_SEEN: OrderedDict = OrderedDict()
_FWD_SEEN_LIMIT = 512
# a layout and tape mode with no plan yet, with one, and whose key cannot be planned
_UNPLANNED, _PLANNED, _REFUSED = range(3)
# the record credit in integer units: a gated call adds FWD_PLAN_PAYBACK, a replay FWD_PLAN_TRICKLE, a recording
# takes FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
_FWD_CREDIT = FWD_PLAN_BUDGET * FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
# plan key (:func:`_function_forward`) -> :class:`_ForwardPlan`; the oldest entry leaves first, and the plans of a
# layout leave with its ChunkMeta (:func:`_drop_meta_plans`)
_FWD_PLANS: dict = {}
_FWD_PLANS_LIMIT = 512  # two tape modes per cached layout (_META_FRONT_LIMIT = 256)
# id(meta) -> {plan key: None} of the plans recorded on that ChunkMeta
_PLAN_KEYS: dict = {}
# the insert and eviction of the plan cache (threads that record at once)
_PLANS_LOCK = threading.Lock()
# the steps of a plan: a launch, an allocation, a dim-0 range of a slot, the fork of the side stream, a stream switch,
# the join, a dtype view of a slot, an eager call, an event recorded on one stream of the fork and a wait for it, the
# use of a cached walk, the end of some slots (the replay lets them go), a workspace of the stream's pool (taken again
# by each replay)
_OP_LAUNCH, _OP_EMPTY, _OP_NARROW, _OP_FORK, _OP_STREAM, _OP_JOIN = range(6)
_OP_VIEW, _OP_CALL, _OP_SIGNAL, _OP_WAIT, _OP_USE, _OP_DROP, _OP_TAKE = range(6, 13)
# the forward functions a plan stands for (the direct executor's steps reach some of them only through ``_DIRECT``): a
# plan replays only while they and the module globals they read are the recorded objects (:func:`_fwd_plan_globals`)
_FWD_PLAN_ROOTS = ("_train_forward", "_forward_a", "_forward_split", "_forward_split_a", "_launch_wy", "_wy_outputs",
                   "_launch_wy_a", "_launch_fwd_ho", "_first_wave", "_start_state", "_walk", "_launch", "_prepared",
                   "_fire", "_side_stream", "_use_table", "_aligned", "_table_rows", "_caller_rows",
                   "_index_copy_heads", "_workspaces")  # fmt: skip
_FWD_PLAN_GLOBALS: Any = None  # their getter, made at the first call
# the entries of the forward's result that :func:`_function_forward` reads, the ones a plan returns
_FWD_OUTPUTS = ("qn", "kn", "rstd_q", "rstd_k", "v", "g_cs", "beta32", "A", "w", "u", "h", "v_new", "output",
                "final_state", "mode")  # fmt: skip


def _stream_key(device_index: int) -> tuple[int, int]:
    """(device, handle of its current stream): the key of the stream's pooled workspaces (:meth:`Workspaces.table`,
    :meth:`Workspaces.stream_key`) and of its side stream."""
    return device_index, _raw_stream(device_index)


# the cache fill context of the direct executor (:meth:`_DirectExecutor.fill`), one object for every call
_NO_FILL = nullcontext()


class _DirectExecutor:
    """The forward's launches, allocations, views, eager calls, pooled workspaces, stream switches and uses of cached
    tables as :func:`_train_forward` reaches them.  A step of :data:`_DIRECT`, the general path's executor, costs the
    host time of the call it stands for: it is that function or tensor method itself, held in an instance slot
    (``ex.empty(...)`` is one specialized slot load and the call of ``torch.empty``, as ``torch.empty(...)`` is one
    module attribute load and that call; a class-level ``staticmethod`` would be a descriptor that the interpreter
    does not specialize), with no Python frame of its own:

    - ``launch(stage, gx, gy, gz, *values)``: :func:`_launch`;
    - ``empty(shape, dtype=..., device=...)``: ``torch.empty``;
    - ``view(t, dtype)``: ``t`` viewed as ``dtype`` (the 32-bit words a kernel reads);
    - ``contiguous(t)``, ``float(t)``, ``index_fill_(t, ...)``, ``index_select(t, ...)``: those ``torch.Tensor``
      methods (``t.contiguous()``, ...), eager steps of the forward;
    - ``aligned(t, align)``, ``table_rows(state, meta)``, ``caller_rows(state, meta)``: :func:`_aligned`,
      :func:`_table_rows`, :func:`_caller_rows`;
    - ``workspaces(options)``: :func:`_workspaces`, the pool the forward takes its workspaces from
      (``workspaces.take(table, role, shape, dtype, device)``, the pool's own take).

    The forward's stream fork, side-stream switch, events and join and the use of a cached walk are methods (a step of
    several calls); the general path makes them in place (``if ex is _DIRECT:``), another executor through them.
    :data:`_DIRECT` holds its functions from the time the module loads: rebinding ``_launch`` afterwards does not
    reach the forward's general path (and, ``_launch`` being one of the forward's functions, sends the next call of a
    plan's key to record again).  A wrapper of ``cake_backend.launch_stage`` sees every launch of the forward: of the
    general path, of a recording and of a replay.  The recorders (:class:`_SlotRecorder`, :class:`_FwdRecorder`) have
    methods of these names in place of the slots."""

    __slots__ = ("launch", "empty", "view", "contiguous", "float", "index_fill_", "index_select", "aligned",
                 "table_rows", "caller_rows", "workspaces")  # fmt: skip

    def __init__(self):
        self.launch = _launch
        self.empty = torch.empty
        self.view = torch.Tensor.view
        self.contiguous = torch.Tensor.contiguous
        self.float = torch.Tensor.float
        self.index_fill_ = torch.Tensor.index_fill_
        self.index_select = torch.Tensor.index_select
        self.aligned = _aligned
        self.table_rows = _table_rows
        self.caller_rows = _caller_rows
        self.workspaces = _workspaces

    def fork(self, index):
        """(the caller's stream, its side stream), the side stream after the caller's work so far."""
        main = torch.cuda.current_stream(index)
        side = _side_stream(index, main)
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        return main, side

    def on(self, side):
        return torch.cuda.stream(side)

    def join(self, main, side):
        main.wait_stream(side)

    def signal(self, stream):
        """An event recorded on ``stream`` now, for a :meth:`wait` of the other stream of the fork."""
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait(self, stream, event) -> None:
        stream.wait_event(event)

    def use(self, key, entry) -> None:
        """Make the cached walk ``entry`` (``_WALKS[key]``: the walk and its write) safe to read by the current
        stream's next launches (:func:`_use_table`)."""
        _use_table(entry[0], entry[1], key[0])

    def fill(self):
        """The context of a cached table's build (its write and the event after it), which a later call does not
        repeat."""
        return _NO_FILL

    def refuse(self, reason: str) -> None:
        """A step a plan cannot replay (``reason``): the call runs it on the general path."""


_DIRECT = _DirectExecutor()


def _storages(obj, out: set, depth: int = 0) -> None:
    """Add the storage addresses of the tensors in ``obj`` (a tensor, a tuple / list or a dataclass of them, nested)."""
    if isinstance(obj, torch.Tensor):
        out.add(obj.untyped_storage().data_ptr())
    elif depth < 4 and isinstance(obj, (tuple, list)):
        for x in obj:
            _storages(x, out, depth + 1)
    elif depth < 4 and is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            _storages(getattr(obj, f.name), out, depth + 1)


# the field names of each dataclass type that :func:`_tensors` walked
_FIELD_NAMES: dict = {}


def _tensors(obj, out: list, depth: int = 0) -> None:
    """Append the tensors in ``obj`` (a tensor, a tuple / list or a dataclass of them, nested) to ``out``.  A tuple or
    list whose first entry is an int, float or str (``seq_lens``, ``order``, ``scan_config``) is passed over whole, so
    the walk takes no time per sequence of the layout (a step that read a tensor of such a tuple would refuse the
    recording)."""
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif depth < 4 and isinstance(obj, (tuple, list)):
        if obj and not isinstance(obj[0], (int, float, str)):
            for x in obj:
                _tensors(x, out, depth + 1)
    elif depth < 4 and is_dataclass(obj) and not isinstance(obj, type):
        names = _FIELD_NAMES.get(type(obj))
        if names is None:
            names = _FIELD_NAMES[type(obj)] = tuple(f.name for f in fields(obj))
        for name in names:
            _tensors(getattr(obj, name), out, depth + 1)


def _reached_globals(roots) -> set[str]:
    """The module globals that the functions ``roots`` read, themselves or through this module's functions they reach
    (found from their code)."""
    g = globals()
    names, seen = set(), set()
    todo = [g[name].__code__ for name in roots]
    while todo:
        code = todo.pop()
        if code in seen:
            continue
        seen.add(code)
        todo.extend(c for c in code.co_consts if isinstance(c, CodeType))
        for name in code.co_names:
            if name not in g:
                continue
            names.add(name)
            fn = getattr(g[name], "__wrapped__", g[name])
            if isinstance(fn, FunctionType) and fn.__globals__ is g:
                todo.append(fn.__code__)
    return names


def _fwd_plan_globals() -> tuple:
    """``cake_backend.device_arch``, the forward's functions (:data:`_FWD_PLAN_ROOTS`) and the module globals they read
    (through this module's functions, found once from their code), one flat tuple: a plan replays only while each is
    the object it was recorded with.  The plan compares them by identity: some are table caches (``_WALKS``: a dict of
    tensors, which ``==`` would read)."""
    global _FWD_PLAN_GLOBALS
    if _FWD_PLAN_GLOBALS is None:
        names = {*_FWD_PLAN_ROOTS, *_reached_globals(_FWD_PLAN_ROOTS)}
        _FWD_PLAN_GLOBALS = itemgetter(*sorted(names))
    try:
        return (cake_backend.device_arch, *_FWD_PLAN_GLOBALS(globals()))
    except KeyError:  # a global of the forward was deleted: a new object, which no plan holds (none replays)
        return (object(),)


def _keep_plan(plans: dict, limit: int, key, plan) -> None:
    """``plans[key] = plan``, the oldest entry leaving first at ``limit`` entries; under one lock, since threads that
    record at once would evict one entry twice."""
    with _PLANS_LOCK:
        plans.pop(key, None)
        if len(plans) >= limit:
            del plans[next(iter(plans))]
        plans[key] = plan


def _keep_fwd_plan(key, meta: ChunkMeta, plan) -> None:
    """Keep the forward plan ``plan`` of ``key``, recorded on the layout ``meta`` (:func:`_drop_meta_plans`)."""
    _keep_plan(_FWD_PLANS, _FWD_PLANS_LIMIT, key, plan)
    with _PLANS_LOCK:
        _PLAN_KEYS.setdefault(id(meta), {})[key] = None


def _drop_meta_plans(meta: ChunkMeta) -> None:
    """Drop the forward plans recorded on ``meta``, which left the layout cache: a plan keeps the tables of no layout
    longer than the cache does.  A key that left the plan cache earlier, or whose entry is a plan of another
    ChunkMeta under the same ``id`` (one that died), is passed over."""
    if id(meta) not in _PLAN_KEYS:
        return
    with _PLANS_LOCK:
        for key in _PLAN_KEYS.pop(id(meta), ()):
            plan = _FWD_PLANS.get(key)
            if plan is not None and plan.meta is meta:
                del _FWD_PLANS[key]


def _fwd_gate(meta: ChunkMeta, tape: bool):
    """The record gate's entry of the call's layout and tape mode when the call builds its plan key (to replay or to
    record), else ``None``: an early forward of the layout in that mode, a layout and mode whose key cannot be planned,
    or no plan yet and less credit than one recording.  Each call adds its share to the credit.  A first sighting of a
    layout and mode (a fresh layout) makes its entry and nothing else: one probe and one insert of :data:`_FWD_SEEN`
    (and one eviction once it is full), no lock and no plan key.  :func:`_function_forward` calls it only when the
    credit with the call's share holds one recording or the layout has a plan kept on it (:data:`FWD_PLAN_GATE`)."""
    global _FWD_CREDIT
    full = FWD_PLAN_BUDGET * FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
    credit = _FWD_CREDIT + FWD_PLAN_PAYBACK
    _FWD_CREDIT = credit = credit if credit < full else full
    key = (id(meta), tape)
    seen = _FWD_SEEN.get(key)
    if seen is not None and seen[0]() is meta:
        if seen[1] < FWD_PLAN_REPEAT:
            seen[1] += 1
            if seen[1] < FWD_PLAN_REPEAT:
                return None
    else:
        # a new layout, or the id of one that died (its entry goes): the newest entry, this forward its first.  A layout
        # whose entry left with a plan kept on it starts planned (its next forward builds its key once: a replay, or a
        # recording as an unplanned one would make, else unplanned)
        if seen is not None:
            _FWD_SEEN.pop(key, None)
        elif len(_FWD_SEEN) >= _FWD_SEEN_LIMIT:
            _FWD_SEEN.popitem(last=False)
        first = _PLANNED if _PLAN_KEYS and id(meta) in _PLAN_KEYS else _UNPLANNED
        count = 1 if FWD_PLAN_REPEAT > 0 else 0
        seen = _FWD_SEEN[key] = [weakref.ref(meta), count, first]
        if seen[1] < FWD_PLAN_REPEAT:
            return None
    state = seen[2]
    if state == _PLANNED or (
        state == _UNPLANNED and credit >= FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
    ):
        return seen
    return None


def _fwd_spend() -> bool:
    """Take one recording from the record credit (``False``: it holds less)."""
    global _FWD_CREDIT
    cost = FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
    with _PLANS_LOCK:
        if _FWD_CREDIT < cost:
            return False
        _FWD_CREDIT -= cost
    return True


def _fwd_replayed() -> None:
    """Add a replay's share to the record credit."""
    global _FWD_CREDIT
    full = FWD_PLAN_BUDGET * FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE
    credit = _FWD_CREDIT + FWD_PLAN_TRICKLE
    _FWD_CREDIT = credit if credit < full else full


def _slot_layout(slots) -> tuple:
    """The shape, strides, dtype and 16-byte base alignment of every tensor of ``slots`` (``None`` for another
    entry): what a recorded plan's steps assume of the inputs beyond their addresses."""
    return tuple(
        (t.shape, t.stride(), t.dtype, t.data_ptr() % 16)
        if isinstance(t, torch.Tensor)
        else None
        for t in slots
    )


def _float_bits(x) -> bytes | None:
    """The 8 bytes of ``x`` as a double (``None`` for ``None``): a key field that tells -0.0 from 0.0 and one NaN from
    another, which ``==`` does not (a launch argument of -0.0 is not one of 0.0)."""
    return None if x is None else struct.pack("<d", float(x))


def _no_slot() -> None:
    """The reference of an absent input (``initial_state`` ``None``)."""


def _drop_steps(ops, inputs: int, kept) -> tuple:
    """``ops`` with a :data:`_OP_DROP` step after the last step that reads each slot, which lets the replay's reference
    to it go: the general path's tensors go when nothing refers to them any more, which is no earlier.  The call's
    inputs (slots below ``inputs``) and the plan's results (``kept``) stay.  A slot that the side stream reads goes at
    the next join, not before: the caching allocator could hand its memory to the caller's stream while the side
    stream still reads it.  A slot made on the side stream stays until the replay returns."""
    last: dict[int, int] = {}
    stay = set(kept)
    side = False
    side_reads: list[int] = []
    for j, op in enumerate(ops):
        code = op[0]
        if code == _OP_STREAM:
            side = bool(op[1])
            continue
        if code == _OP_JOIN:
            last.update(dict.fromkeys(side_reads, j))
            side_reads.clear()
            continue
        if code == _OP_LAUNCH:
            reads, made = [i for _pos, i in op[5]], None
        elif code == _OP_CALL:
            reads, made = [i for _pos, i in op[3]], op[4]
        elif code in (_OP_VIEW, _OP_NARROW):
            reads, made = [op[2]], op[1]
        elif code in (_OP_EMPTY, _OP_TAKE):
            reads, made = [], op[1]
        else:
            continue
        for i in reads:
            last[i] = j
        if made is not None:
            last[made] = j  # a slot no step reads goes right away
        if side:
            side_reads += reads
            if made is not None:
                stay.add(made)
    stay.update(side_reads)  # read on the side stream with no join after it
    drops: dict[int, list[int]] = {}
    for i, j in last.items():
        if i >= inputs and i not in stay:
            drops.setdefault(j, []).append(i)
    steps = []
    for j, op in enumerate(ops):
        steps.append(op)
        if j in drops:
            steps.append((_OP_DROP, tuple(sorted(drops[j]))))
    return tuple(steps)


class _SlotRecorder(_DirectExecutor):
    """A general path that also records its steps.  ``slots``: the call's tensors the plan starts from.  Every tensor
    a recorded step reads is a slot (one of those, or one a recorded step made: an allocation, a workspace of the
    stream's pool, :meth:`take`, an eager result), a dim-0 range of a slot, or a constant of the plan or a view of one
    (:meth:`_constant_tensors`: a chunk table of ``meta``, the cached walk); anything else, or a :meth:`refuse`, leaves
    the call without a plan.  The recorder refers to the slots weakly: each goes when the general path lets it go, so
    a recording needs no more device memory than the general path.  It walks the constants once per recording, at the
    first operand that is not a slot, and their storage addresses once more only at the first view of one."""

    __slots__ = (
        "slots",
        "slot_of",
        "inputs",
        "ops",
        "why",
        "stream",
        "meta",
        "streams",
        "ranges",
        "const_ids",
        "const_ptrs",
        "const_tensors",
    )

    def __init__(self, slots, index: int, meta: ChunkMeta):
        self.slots = [_no_slot if t is None else weakref.ref(t) for t in slots]
        self.slot_of = {id(t): i for i, t in enumerate(slots) if t is not None}
        self.inputs = len(self.slots)
        self.ops: list = []
        self.why: str | None = None
        self.stream = _stream_key(index)
        self.meta = meta
        self.streams = None
        self.ranges: dict = {}
        # the constants (:meth:`_constant_tensors`), their ids and their storage addresses, each made when first read
        self.const_tensors: list | None = None
        self.const_ids: set | None = None
        self.const_ptrs: set | None = None

    def refuse(self, reason: str) -> None:
        if self.why is None:
            self.why = reason

    def _add(self, t) -> int:
        self.slots.append(weakref.ref(t))
        self.slot_of[id(t)] = len(self.slots) - 1
        return len(self.slots) - 1

    def _slot(self, t):
        i = self.slot_of.get(id(t))
        return i if i is not None and self.slots[i]() is t else None

    def _constant_tensors(self) -> list:
        """The plan's constants: the tables of ``meta`` (a pooled workspace is a slot, :meth:`take`)."""
        out: list = []
        _tensors(self.meta, out)
        return out

    def _ref(self, t):
        """The slot of ``t`` (a range of a slot gets a slot of its own, filled by a narrow step), ``None`` for a
        constant or a view of one, ``-1`` for a tensor the plan cannot rebuild.  A constant itself is found by its id
        before the ranges: it is no range of a slot (the slots are the call's inputs, its allocations, eager results
        and pooled workspaces, and views of those).  The recorder holds ``meta`` and the walks, so no id of theirs is
        reused during the recording."""
        i = self._slot(t)
        if i is not None:
            return i
        # once per recording, at its first operand that is not a slot
        if self.const_ids is None:
            self.const_tensors = self._constant_tensors()
            self.const_ids = {id(c) for c in self.const_tensors}
        if id(t) in self.const_ids:
            return None
        ptr = t.untyped_storage().data_ptr()
        for i, ref in enumerate(self.slots):
            s = ref()  # None for a slot that is gone
            if (
                not isinstance(s, torch.Tensor)
                or s.dim() == 0
                or s.stride(0) <= 0
                or s.dtype != t.dtype
                or s.stride() != t.stride()
                or s.shape[1:] != t.shape[1:]
                or s.untyped_storage().data_ptr() != ptr
            ):
                continue
            start, rest = divmod(t.storage_offset() - s.storage_offset(), s.stride(0))
            if rest == 0 and start >= 0 and start + t.shape[0] <= s.shape[0]:
                key = (i, start, t.shape[0])
                j = self.ranges.get(key)
                if j is None:
                    j = self.ranges[key] = self._add(t)
                    self.ops.append((_OP_NARROW, j, i, start, t.shape[0]))
                return j
        # once per recording, at its first view of a constant
        if self.const_ptrs is None:
            self.const_ptrs = {
                c.untyped_storage().data_ptr() for c in self.const_tensors
            }
        return None if ptr in self.const_ptrs else -1

    def _tables(self):
        return _POOLED.tables.get(self.stream)

    def _operands(self, values, what):
        """``(template, refs)`` of a step that reads ``values``: ``values`` with each slot or range of a slot as
        ``None`` and ``(position, slot)`` of each of those; ``None`` after a refusal (``what`` reads a tensor the
        plan cannot rebuild)."""
        template, refs = list(values), []
        for pos, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                i = self._ref(value)
                if i == -1:
                    self.refuse(
                        f"{what} reads a tensor that is neither a slot nor a constant"
                    )
                    return None
                if i is not None:
                    template[pos] = None
                    refs.append((pos, i))
        return tuple(template), tuple(refs)

    def launch(self, stage, gx, gy, gz, *values):
        """Run a launch of :func:`_prepared` and record it (in the order the launches run)."""
        launch = _prepared(stage, gx, gy, gz, *values)
        _fire(launch)
        self._record(launch)

    def _record(self, launch) -> None:
        if self.why is not None:
            return
        stage, grid, ordered, device_index = launch
        found = self._operands(ordered, stage)
        if found is not None:
            self.ops.append((_OP_LAUNCH, stage, grid, device_index, *found))

    def empty(self, shape, dtype, device):
        t = torch.empty(shape, dtype=dtype, device=device)
        self.ops.append((_OP_EMPTY, self._add(t), tuple(shape), dtype, device))
        return t

    def workspaces(self, options):
        """The call's workspace pool (:func:`_workspaces`), each take from it a :meth:`take` step."""
        return _RecordedPool(self, _workspaces(options))

    def take(self, pool, table, role, shape, dtype, device):
        """``pool.take(table, role, shape, dtype, device)`` of the stream's pool: a slot that each replay takes from the
        pool again, as the general path does (a hit, or a miss that allocates and evicts); a workspace of another pool
        or table refuses the call."""
        t = pool.take(table, role, shape, dtype, device)
        if pool is not _POOLED or table is not self._tables():
            self.refuse("a workspace outside the stream's pool")
        else:
            shape = tuple(shape)
            self.ops.append(
                (_OP_TAKE, self._add(t), role, shape, dtype, device, (shape, dtype))
            )
        return t

    def fork(self, index):
        main, side = super().fork(index)
        if self.streams is not None and self.streams != (main, side):
            self.refuse("two forks on different streams")
        self.streams = (main, side)
        self.ops.append((_OP_FORK,))
        return main, side

    def join(self, main, side):
        super().join(main, side)
        self.ops.append((_OP_JOIN,))


class _RecordedPool:
    """The workspace pool ``pool`` as a recording's forward takes from it (:meth:`_SlotRecorder.workspaces`): its
    tables, and takes that are the recorder's take steps (:meth:`_SlotRecorder.take`)."""

    __slots__ = ("recorder", "pool")

    def __init__(self, recorder, pool):
        self.recorder = recorder
        self.pool = pool

    def table(self, device_index: int) -> dict:
        return self.pool.table(device_index)

    def take(self, table, role, shape, dtype, device):
        return self.recorder.take(self.pool, table, role, shape, dtype, device)


# A forward's recording watches the stream and event operations of :data:`_WATCHED` (:func:`_watched`): one that the
# recording thread makes outside the recorder's own steps and cache fills would be missing from the plan, so it leaves
# the call without one.  The first recording in flight installs the hooks and the last one removes them; a call of
# another thread, or of this one outside a recording, passes through.
_WATCHED = ((torch.cuda, "set_stream"), (torch.cuda.Event, "record"), (torch.cuda.Event, "wait"),
            (torch.cuda.Stream, "wait_event"))  # fmt: skip
# the recorder of the recording in flight on this thread
_WATCHING: ContextVar[Any] = ContextVar("cake_gdn_det_watching", default=None)
_WATCH_LOCK = threading.Lock()
# the recordings in flight, and (owner, name, original, whether the owner itself held it, the hook) of every hooked
# attribute
_WATCH_STATE: dict[str, Any] = {"count": 0, "saved": ()}


def _watch_hook(name: str, original):
    def hook(*args, **kwargs):
        recorder = _WATCHING.get()
        if recorder is not None and not recorder.trusted:
            recorder.refuse(f"{name} outside the forward's executor")
        return original(*args, **kwargs)

    return hook


@contextmanager
def _watched(recorder):
    """Run a forward's recording by ``recorder`` with the operations of :data:`_WATCHED` hooked.  Another hook set in
    place of one of these during the recording (another package's recording) would not report this one's calls: the
    recording is refused for this call only (``recorder.again``: the key's next call records again), and the last exit
    leaves such an attribute as it is (it restores only its own hooks)."""
    with _WATCH_LOCK:
        if not _WATCH_STATE["count"]:
            saved = []
            for owner, name in _WATCHED:
                original = getattr(owner, name)
                hook = _watch_hook(name, original)
                saved.append((owner, name, original, name in vars(owner), hook))
                setattr(owner, name, hook)
            _WATCH_STATE["saved"] = tuple(saved)
        _WATCH_STATE["count"] += 1
    token = _WATCHING.set(recorder)
    try:
        yield
    finally:
        _WATCHING.reset(token)
        with _WATCH_LOCK:
            saved = _WATCH_STATE["saved"]
            if any(
                getattr(owner, name) is not hook
                for owner, name, _original, _own, hook in saved
            ):
                # not kept: the key's next call records again, most likely without the other hook in flight
                recorder.refuse("a watched function was replaced during the recording")
                recorder.again = True
            _WATCH_STATE["count"] -= 1
            if not _WATCH_STATE["count"]:
                for owner, name, original, own, hook in saved:
                    if getattr(owner, name) is not hook:
                        continue
                    if own:
                        setattr(owner, name, original)
                    else:
                        delattr(owner, name)
                _WATCH_STATE["saved"] = ()


class _ForwardPlan:
    """One recorded forward (:class:`_FwdRecorder`): its steps, the number of its slots, the stream it runs on (the
    caller's, ``main``, and its side stream), the layout's ``ChunkMeta``, the module globals, pool tables (whose
    workspaces its take steps take again) and cached walks it was recorded with, and the entries of its result
    (:data:`_FWD_OUTPUTS`, each ``(name, 0, slot)`` or ``(name, 2, value)``).  ``ops`` is ``None`` for a key whose
    forward cannot be planned: its calls run the general path without recording.  A plan leaves the plan cache with
    its ``ChunkMeta`` (:func:`_drop_meta_plans`); a pool eviction leaves it in place."""

    __slots__ = ("ops", "size", "stream", "main", "side", "meta", "names", "tables", "walks", "outputs",
                 "fork_event", "join_event", "events")  # fmt: skip

    def __init__(self, ops, size, stream, main, side, meta, names, tables, walks, signals, outputs):  # fmt: skip
        self.ops = ops
        self.size = size
        self.stream = stream
        self.main = main
        self.side = side
        self.meta = meta
        self.names = names
        self.tables = tables
        self.walks = walks
        self.outputs = outputs
        # the fork and the join of a plan with a side stream (recorded right before each wait), and the events one
        # stream records for the other (:meth:`_DirectExecutor.signal`)
        self.fork_event = torch.cuda.Event() if main is not None else None
        self.join_event = torch.cuda.Event() if main is not None else None
        self.events = tuple(torch.cuda.Event() for _ in range(signals))

    def ready(self, meta: ChunkMeta, stream) -> bool:
        """Whether this call (of the plan's key) may replay the plan: the layout's ``ChunkMeta``, the pool's tables of
        the stream, its side stream and the cached walks it uses are the ones of the recording (a cleared cache makes
        new ones), and so is every module global the forward reads (:func:`_fwd_plan_globals`, by identity)."""
        if self.meta is not meta or _POOLED.tables.get(stream) is not self.tables:
            return False
        if self.side is not None and _SIDE_STREAMS.get(stream) is not self.side:
            return False
        if not all(_WALKS.get(key) is entry for key, entry in self.walks):
            return False
        names = _fwd_plan_globals()
        return len(names) == len(self.names) and all(map(is_, names, self.names))


class _FwdRecorder(_SlotRecorder):
    """The forward's general path that also records its steps (:meth:`plan`).  ``slots``: the call's ``q``, ``k``,
    ``v``, ``g``, ``beta`` and ``initial_state``; the views, the allocations and the eager results of the call are
    slots too, the cached walks it uses (:meth:`use`) constants.  A cache it fills (:meth:`fill`: the walk's build and
    the event after it) is not recorded, since a later general-path call does not repeat it.  ``again``: a refusal of
    this call is not kept, so the key's next call records again (a call that filled a cache, or whose recording another
    hook of :data:`_WATCHED` overlapped: :func:`_watched`).  ``trusted``: inside one of the recorder's own stream or
    event steps or a fill, which :func:`_watched` lets pass."""

    __slots__ = ("uses", "events", "trusted", "again")

    def __init__(self, inputs, index: int, meta: ChunkMeta):
        super().__init__(inputs, index, meta)
        self.uses: list = []
        self.events: list = []
        self.trusted = 0
        self.again = False

    @contextmanager
    def _trust(self):
        self.trusted += 1
        try:
            yield
        finally:
            self.trusted -= 1

    def _constant_tensors(self) -> list:
        """The tables of ``meta`` and the cached walks the call used."""
        out = super()._constant_tensors()
        out += [entry[0] for _key, entry in self.uses]
        return out

    def view(self, t, dtype):
        out = t.view(dtype)
        i = self._slot(t)
        if i is not None:  # a view of a constant is a constant
            self.ops.append((_OP_VIEW, self._add(out), i, dtype))
        return out

    def call(self, fn, *args):
        out = fn(*args)
        self._call(fn, args, out)
        return out

    def rows(self, fn, t, *args):
        out = fn(t, *args)
        if out is not t:  # ``t`` itself: the layout in the key fixes that, no step
            self._call(fn, (t, *args), out)
        return out

    def contiguous(self, t):
        return self.rows(torch.Tensor.contiguous, t)

    def float(self, t):
        return self.call(torch.Tensor.float, t)

    def index_fill_(self, t, dim, index, value):
        return self.call(torch.Tensor.index_fill_, t, dim, index, value)

    def index_select(self, t, dim, index):
        return self.call(torch.Tensor.index_select, t, dim, index)

    def aligned(self, t, align):
        return self.rows(_aligned, t, align)

    def table_rows(self, state, meta):
        return self.rows(_table_rows, state, meta)

    def caller_rows(self, state, meta):
        return self.rows(_caller_rows, state, meta)

    def index_copy_heads(self, state, index, rows) -> None:
        self.call(_index_copy_heads, state, index, rows)

    def _call(self, fn, args, out) -> None:
        if self.why is not None:
            return
        found = self._operands(args, getattr(fn, "__name__", fn))
        if found is None:
            return
        # a new tensor is a slot; an in-place call returns one of its operands
        slot = None
        if isinstance(out, torch.Tensor) and not any(out is value for value in args):
            slot = self._add(out)
        self.ops.append((_OP_CALL, fn, *found, slot))

    def fork(self, index):
        with self._trust():
            return super().fork(index)

    @contextmanager
    def on(self, side):
        self.ops.append((_OP_STREAM, 1))
        context = torch.cuda.stream(side)
        with self._trust():
            context.__enter__()
        try:
            yield
        finally:
            with self._trust():
                context.__exit__(None, None, None)
        self.ops.append((_OP_STREAM, 0))

    def join(self, main, side):
        with self._trust():
            super().join(main, side)

    def _role(self, stream) -> int | None:
        """0 for the caller's stream of the fork, 1 for its side stream, ``None`` for another or before the fork."""
        for role, s in enumerate(self.streams or ()):
            if stream is s:
                return role
        return None

    def signal(self, stream):
        with self._trust():
            event = super().signal(stream)
        role = self._role(stream)
        if role is None:
            self.refuse("an event recorded outside the fork")
        else:
            self.ops.append((_OP_SIGNAL, len(self.events), role))
        self.events.append(event)
        return event

    def wait(self, stream, event) -> None:
        with self._trust():
            super().wait(stream, event)
        role = self._role(stream)
        k = next((k for k, e in enumerate(self.events) if e is event), None)
        if role is None or k is None:
            self.refuse("an event wait outside the fork")
        else:
            self.ops.append((_OP_WAIT, k, role))

    def use(self, key, entry) -> None:
        with self._trust():
            super().use(key, entry)
        self.uses.append((key, entry))
        self.ops.append((_OP_USE, entry, key[0]))
        # a walk used after the constants were first read is a constant too
        if self.const_ids is not None:
            self.const_tensors.append(entry[0])
            self.const_ids.add(id(entry[0]))
        if self.const_ptrs is not None:
            self.const_ptrs.add(entry[0].untyped_storage().data_ptr())

    @contextmanager
    def fill(self):
        self.again = True
        with self._trust():
            yield

    def plan(self, fwd: dict) -> _ForwardPlan | None:
        """The recorded plan of a call whose general path returned ``fwd``, ``None`` when the call refused one."""
        if self.why is not None:
            return None
        outputs = []
        for name in _FWD_OUTPUTS:
            value = fwd.get(name)
            if not isinstance(value, torch.Tensor):
                outputs.append((name, 2, value))
                continue
            i = self._slot(value)
            if i is None:
                return None  # an entry that is not a slot of the call
            outputs.append((name, 0, i))
        main, side = self.streams or (None, None)
        kept = [i for _name, kind, i in outputs if kind == 0]
        ops = _drop_steps(self.ops, self.inputs, kept)
        return _ForwardPlan(ops, len(self.slots), self.stream, main, side, self.meta, _fwd_plan_globals(),
                            self._tables(), tuple(self.uses), len(self.events), tuple(outputs))  # fmt: skip


def _replay_forward(plan: _ForwardPlan, inputs) -> dict[str, Any]:
    """Run ``plan`` on the call's ``q``, ``k``, ``v``, ``g``, ``beta`` and ``initial_state`` (``inputs``); returns the
    entries of the forward's result that :func:`_function_forward` reads (:data:`_FWD_OUTPUTS`).  A stream switch
    makes the stream's device current, which the general path's stream context restores on its exit: a plan with a
    side stream makes the caller's current device current again when it returns or raises."""
    slots = [*inputs]
    slots += [None] * (plan.size - len(slots))
    launch_stage = cake_backend.launch_stage
    tables, pool_take = plan.tables, _POOLED.take
    main, side = plan.main, plan.side
    caller = None if main is None else torch.cuda.current_device()
    forked = False
    try:
        for op in plan.ops:
            code = op[0]
            if code == _OP_LAUNCH:
                values = list(op[4])
                for pos, i in op[5]:
                    values[pos] = slots[i]
                launch_stage(op[1], op[2], tuple(values), device_index=op[3])
            elif code == _OP_EMPTY:
                slots[op[1]] = torch.empty(op[2], dtype=op[3], device=op[4])
            elif code == _OP_TAKE:
                # Workspaces.take inline on a hit (a hit keeps the role's order); a miss takes through the pool, which
                # allocates and evicts as the general path's take does
                buffers = tables.get(op[2])
                t = None if buffers is None else buffers.get(op[6])
                if t is None:
                    t = pool_take(tables, op[2], op[3], op[4], op[5])
                slots[op[1]] = t
            elif code == _OP_VIEW:
                slots[op[1]] = slots[op[2]].view(op[3])
            elif code == _OP_CALL:
                args = list(op[2])
                for pos, i in op[3]:
                    args[pos] = slots[i]
                out = op[1](*args)
                if op[4] is not None:
                    slots[op[4]] = out
            elif code == _OP_DROP:
                for i in op[1]:
                    slots[i] = None
                # the last launch's values and the last call's arguments and result may hold them too
                values = args = out = None
            elif code == _OP_USE:
                _use_table(op[1][0], op[1][1], op[2])
            elif code == _OP_FORK:
                plan.fork_event.record(main)
                side.wait_event(plan.fork_event)
                forked = True
            elif code == _OP_STREAM:
                torch.cuda.set_stream(side if op[1] else main)
            elif code == _OP_SIGNAL:
                plan.events[op[1]].record(side if op[2] else main)
            elif code == _OP_WAIT:
                (side if op[2] else main).wait_event(plan.events[op[1]])
            elif code == _OP_JOIN:
                plan.join_event.record(side)
                main.wait_event(plan.join_event)
                forked = False
            else:  # a dim-0 range of a slot
                slots[op[1]] = slots[op[2]].narrow(0, op[3], op[4])
    except BaseException:
        if forked:  # back on the caller's stream, after the side stream's work (as the general path's join)
            torch.cuda.set_stream(main)
            main.wait_stream(side)
        raise
    finally:
        if caller is not None and caller != plan.stream[0]:
            torch.cuda.set_device(caller)
    return {
        name: slots[value] if kind == 0 else value for name, kind, value in plan.outputs
    }


# --------------------------------------------------------------------------- #
# the round-once tail of a value-head group of three, the package tail and the FP32 dv_local adjoint
# --------------------------------------------------------------------------- #
# A value-head group of three runs its fused tail as tail_g3_f32x on the architectures of F32X_TAIL_ARCHS: the
# cluster kernel of tail_g3 whose three CTAs exchange the group's FP32 dq / dk partials over distributed shared memory
# and sum them in value-head order before one BF16 rounding (tail_g3 rounds each value head's partials before their
# sum).  One writer per element, no atomics, a fixed order: bitwise run to run.  dv, dg and dbeta are tail_g3's bits;
# dq / dk round once (the non-deterministic package's tail_g3_f32x, the same program in its plain build).  The tail
# reads v and writes dq / dk / dv through token-row strides in elements taken at launch (v, dq, dk, dv; the launcher
# requires dq and dk at one stride), so a call on the head views of one packed row writes one packed gradient row
# (:class:`CakeGDNDetPackedFunction`).  On the architectures of DVL32_ARCHS the tcgen05 adjoint scan of every group but
# two adds the FP32 dv_local values to dv2 before its one rounding (dhu_tc2_dvl32: dhu_tc2's arguments, tables and
# selection constants), so dv2 rounds once; dk / dv / dg / dbeta / dh0 take other bits than on dhu_tc2 and than on
# dv_local + the register scan (dhu_tc2's bits), so the selection rules and the ``dhu_scan`` pin choose between two sets
# of gradient bits (o and the final state keep theirs; each choice bitwise run to run).  B200 kernel
# times (CUPTI, cold L2, four processes): tail_g3_f32x 1.024-1.051 x tail_g3, dhu_tc2_dvl32 0.990-0.995 x dhu_tc2;
# GB300 1.039-1.045 x and 0.994-0.996 x.  Both switches are module constants whose off value is ``()``: with both off
# a group of three runs tail_g3 on dhu_tc2 again (the non-deterministic package's tests that compare its tail_g3 with
# this package switch both off by name).  Groups of two keep tail_g2 and dhu_tc2: the wavefront pair has dhu_tc2's bits
# and a CUDA graph capture runs the separate launches instead, so an FP32 dv_local adjoint there would give an eager
# call and its capture other bits.
#
# The other groups (one, and four or more) run the package tail on the architectures of PACKAGE_TAIL_ARCHS (``tail`` /
# ``tail_bf16``: the non-deterministic package's programs, one CTA per (chunk, value head), no cluster).  At a group of
# one each CTA keeps its dq / dk tile and writes it: no workspace slot is read and no counter is touched.  At a larger
# group each CTA writes its FP32 dq / dk partials to its own workspace slot and then takes a ticket from the arrival
# counter of its (chunk, key head) tile (an integer atomic increment that wraps to zero at the group size, so every
# complete launch leaves its counters zero); the CTA that draws the last ticket, G - 1, sums the group's slots in
# value-head order, its own included, and rounds dq / dk to BF16 once.  The operands and their order do not depend on
# which CTA draws the last ticket, and every other output (dv, dg, dbeta) has one writer with the unfused tail's bits,
# so the tail is bitwise run to run; dq / dk round once.  The workspace slots (FP32 rows of 64 dq rows then 64 dk rows,
# one slot per (chunk, value head) of a launch) are never zeroed or filled, also not under
# torch.use_deterministic_algorithms (:func:`_empty_unfilled`): the last CTA of a tile reads only the slot rows of the
# chunk's tokens, which each CTA of the tile writes in that launch before it takes its ticket
# (:meth:`Workspaces.take_rows`).  The u32 counters (one per (chunk, key head) tile) are created zero
# (:meth:`Workspaces.take_zeroed`).  Both are kept per (device, stream of the launch); uncached calls and a CUDA graph
# capture take buffers of their own (the counters zeroed on the caller's stream, inside the graph for a capture), and
# a backward launches its chunk table in segments of at most ``TAIL_WORKSPACE_TILES`` slots in stream order
# (:func:`tail_segments`; at least one chunk), so a workspace holds at most ``max(TAIL_WORKSPACE_TILES, num_v_heads)``
# slots.  A ``finalize`` pin keeps the unfused tail.  The switch is a module constant whose off value is ``()``: off,
# these groups run the unfused tail again.
#
# At a group of four or more the package tail serves a call only when its mean tokens per sequence, total tokens /
# sequences, reaches PACKAGE_TAIL_MIN_TOKENS_PER_SEQ[arch] (an architecture missing from the table, or 0: no
# threshold); below it the call runs the unfused tail on the same adjoint scan (dhu_tc2_dvl32 on DVL32_ARCHS).  A group
# of one is not affected.  Both counts are host facts on either table route (the tokens of q, the offsets of
# cu_seqlens less one or the batch, empty sequences included; never the device route's bound on the chunks), so the
# host and the device tables, an eager call and its CUDA graph capture take the same tail.  The package tail runs full
# 64-row tiles, so its cost per (chunk, value head) CTA stays about the same on short chunks, where the three unfused
# stages cost less.  Each value is the smallest mean tokens per sequence from which no backward on the tail the
# threshold picks ran slower than 0.2.3 (the unfused tail on dhu_tc2) beyond the noise band, in a sweep with the slots
# unfilled: G = 4 and 8, sequences of 4, 8, 16, 24, 32, 48 and 64 tokens over 4096, 16384 and 32768 (chunk, value head)
# CTAs and mixed layouts, the backward and the checkpointed step, eager and CUDA graph.  sm_100a (B200, 148 SMs) and
# sm_103a (GB300, 152 SMs): at G = 4 the package tail ran slower over sequences of 4 tokens (by up to 1.9 %, most at
# 32768 CTAs), and from 8 tokens on it was faster or within the band.  sm_107a (R200, 212 SMs): at G = 4 it ran slower
# up to 24 tokens (at 16384 and 32768 CTAs; at 24 tokens on each GPU's own band), and from 32 on it was faster or
# within the band.  Below each value the package tail still ran faster than the unfused tail on some layouts (at G = 8
# and at 4096 CTAs among them); one threshold cannot keep those gains and drop every loss.
F32X_TAIL_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
DVL32_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
F32X_TAILS: tuple[str, ...] = ("tail_g3_f32x", "tail_g3_f32x_bf16")
PACKAGE_TAIL_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
PACKAGE_TAIL_MIN_TOKENS_PER_SEQ: dict[str, int] = {
    "sm_100a": 8,
    "sm_103a": 8,
    "sm_107a": 32,
}
PACKAGE_TAILS: tuple[str, ...] = ("tail", "tail_bf16")
TAIL_TILE_FLOATS = 2 * CHUNK * HEAD_DIM
TAIL_WORKSPACE_TILES = 4096


def fused_tail_stage(
    arch: str,
    num_heads: int,
    num_v_heads: int,
    *,
    total_tokens: int,
    num_seqs: int,
    dbeta_bf16: bool,
    finalize: str | None = None,
) -> str | None:
    """The fused tail of a backward on ``arch`` over ``total_tokens`` tokens in ``num_seqs`` sequences (``_bf16``:
    the BF16-``dbeta`` form): ``tail_g2`` for a value-head group of two; ``None`` under a ``finalize`` pin for any
    other group; for a group of three ``tail_g3_f32x`` on an architecture of ``F32X_TAIL_ARCHS`` whose registry
    delivers it, else ``tail_g3`` where delivered; for any other group the package tail ``tail`` on an architecture of
    ``PACKAGE_TAIL_ARCHS`` whose registry delivers it, at a group of four or more only when ``total_tokens /
    num_seqs`` reaches ``PACKAGE_TAIL_MIN_TOKENS_PER_SEQ[arch]``; ``None`` otherwise (``dqkwg``, the WY pullback and
    the finalize)."""
    suffix = "_bf16" if dbeta_bf16 else ""
    if num_v_heads == 2 * num_heads:
        return "tail_g2" + suffix
    if finalize:
        return None
    if num_v_heads == 3 * num_heads:
        if arch in F32X_TAIL_ARCHS and _available("tail_g3_f32x" + suffix, arch):
            return "tail_g3_f32x" + suffix
        return "tail_g3" + suffix if _available("tail_g3" + suffix, arch) else None
    if arch not in PACKAGE_TAIL_ARCHS or not _available("tail" + suffix, arch):
        return None
    # a group of four or more below its mean tokens per sequence: the unfused tail
    threshold = PACKAGE_TAIL_MIN_TOKENS_PER_SEQ.get(arch, 0)
    if num_v_heads != num_heads and int(total_tokens) < threshold * int(num_seqs):
        return None
    return "tail" + suffix


def dhu_tc2_stage(arch: str, num_heads: int, num_v_heads: int) -> str:
    """The tcgen05 adjoint scan of a backward on ``arch``: ``dhu_tc2_dvl32`` for any value-head group but two on an
    architecture of ``DVL32_ARCHS`` whose registry delivers it, else ``dhu_tc2``."""
    if (
        num_v_heads != 2 * num_heads
        and arch in DVL32_ARCHS
        and _available("dhu_tc2_dvl32", arch)
    ):
        return "dhu_tc2_dvl32"
    return "dhu_tc2"


def tail_segments(num_chunks: int, num_heads: int, num_v_heads: int,
                  slots: int | None = None) -> tuple[tuple[tuple[int, int], ...], int]:  # fmt: skip
    """The chunk ranges ``[c0, c1)`` the package tail launches in stream order over a chunk table of ``num_chunks``
    chunks, and the (chunk, key head) tiles a launch may index (a tile is the group's ``num_v_heads // num_heads``
    workspace slots and one arrival counter): one launch and no tile at a value-head group of one, else ranges of
    ``max(1, slots // num_v_heads)`` chunks (``slots``: the workspace slots, default ``TAIL_WORKSPACE_TILES``, read at
    the call).  Kept per (chunks, heads, value heads, slots) for the last 64 of them (:func:`_tail_segments`): a
    varlen batch changes its chunk count from call to call, and a kept result holds one range per launch."""
    cap = TAIL_WORKSPACE_TILES if slots is None else slots
    return _tail_segments(int(num_chunks), int(num_heads), int(num_v_heads), int(cap))


@lru_cache(maxsize=64)
def _tail_segments(
    num_chunks: int, num_heads: int, num_v_heads: int, cap: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """:func:`tail_segments` of a workspace of ``cap`` slots."""
    if num_chunks <= 0:
        return (), 0
    if num_v_heads == num_heads:
        return ((0, num_chunks),), 0
    per = min(num_chunks, max(1, cap // num_v_heads))
    return tuple(
        (c, min(c + per, num_chunks)) for c in range(0, num_chunks, per)
    ), per * num_heads


# --------------------------------------------------------------------------- #
# The pv_tf32 adjoint scans and tails
# --------------------------------------------------------------------------- #
# A backward whose final state receives a gradient (a ``dht`` given to the low-level backward; under autograd, a
# gradient that autograd gives the final state, :func:`_grads_of`) runs, with PV_TF32 on (the default) and on an
# architecture of PV_TF32_ARCHS, the programs built with ``pv_tf32=True``: the tcgen05 adjoint scan of PV_TF32_SCANS
# (P_v on one TF32-rounded FP32 word of dh, the FP32 dg state terms) and the tail of PV_TF32_TAILS, with the FP32
# workspaces ``b_hdh32`` / ``b_e32`` between them (the scan writes every element the tail reads).  These calls get dv
# and dg within the BF16 naive tier where the 0.2.8 programs miss it; their programs cost more time.  Every other call
# (no state, an initial state alone, a final state that receives no gradient) keeps the 0.2.8 programs and bits, and
# so does a call whose route has no pv_tf32 pair (pv_tf32_stages): a register scan, the wavefront pair or tail_g3
# (such a call keeps the 0.2.8 accuracy as well).  A pv_tf32 call never takes the wavefront pair.  The selection rules
# still price the 0.2.8 scan.  The pv_tf32 scan also writes the low BF16 word of dv2 (``dv2_lo``, workspace
# ``b_dv2_lo``: the FP32 dv2 minus its BF16 word, rounded to BF16; its rows are written wherever the dv2 tape is) and
# its tail adds it on the diagonal of A^T du (A[i][i] = 1) before the beta scale and the dbeta sum; the unfused tail
# reads it in the WY pullback (DV_LO_WY, after dqkwg_pv_tf32).  So the diagonal term of dv and dbeta reads dv2 as
# two BF16 words instead of one; dq, dk, dg, dh0 and the final state keep the bits of the programs without that word.
# PV_TF32 off: every call keeps the 0.2.8 programs.
PV_TF32: bool = True
PV_TF32_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
PV_TF32_SCANS: dict[str, str] = {
    "dhu_tc2": "dhu_tc2_pv_tf32",
    "dhu_tc2_dvl32": "dhu_tc2_dvl32_pv_tf32",
}
PV_TF32_TAILS: dict[str, str] = {
    "dqkwg": "dqkwg_pv_tf32",
    "tail_g2": "tail_g2_pv_tf32",
    "tail_g2_bf16": "tail_g2_bf16_pv_tf32",
    "tail_g3_f32x": "tail_g3_f32x_pv_tf32",
    "tail_g3_f32x_bf16": "tail_g3_f32x_bf16_pv_tf32",
    "tail": "tail_pv_tf32",
    "tail_bf16": "tail_bf16_pv_tf32",
}
# the WY pullbacks that read dv2's low word (``du_lo``), behind dqkwg_pv_tf32 on the unfused tail of a pv_tf32 call
DV_LO_WY: dict[str, str] = {"wy_bwd": "wy_bwd_dvlo", "wy_bwd_bf16": "wy_bwd_bf16_dvlo"}


def pv_tf32_stages(arch: str, scan: str | None, tail: str) -> tuple[str, str] | None:
    """The pv_tf32 ``(scan, tail)`` stages of a backward on ``arch`` whose dh / dv2 come from the tcgen05 adjoint scan
    ``scan`` (``None``: another scan) and whose tail stage is ``tail`` (``dqkwg`` for the unfused tail); ``None`` when
    :data:`PV_TF32` is off, ``arch`` is not in ``PV_TF32_ARCHS`` or the scan or the tail has no pv_tf32 record there,
    nor, for the unfused tail, the WY pullbacks of :data:`DV_LO_WY` (the call keeps the 0.2.8 programs)."""
    if not PV_TF32 or arch not in PV_TF32_ARCHS or scan is None:
        return None
    pv_scan, pv_tail = PV_TF32_SCANS.get(scan), PV_TF32_TAILS.get(tail)
    if (
        pv_scan is None
        or pv_tail is None
        or not (_available(pv_scan, arch) and _available(pv_tail, arch))
        or (
            tail == "dqkwg"
            and not all(_available(wy, arch) for wy in DV_LO_WY.values())
        )
    ):
        return None
    return pv_scan, pv_tail


def _row_stride(t: torch.Tensor) -> int:
    """The token-row stride in elements of a ``[..., T, heads, 128]`` tensor that a kernel addresses through that
    stride (the round-once tail's v / dq / dk / dv): a unit channel stride, head stride 128, rows at least ``heads x
    128`` apart, and leading axes that fold into the token axis (``ValueError`` otherwise)."""
    shape = t.shape
    heads, dim = shape[-2], shape[-1]
    if t.is_contiguous():
        return heads * dim
    stride = t.stride()
    token = expected = stride[-3]
    for axis in range(len(shape) - 4, -1, -1):
        expected *= shape[axis + 1]
        if shape[axis] > 1 and stride[axis] != expected:
            raise ValueError(
                f"the leading axes of a {tuple(shape)} tensor of strides {stride} do not fold into its token axis"
            )
    if stride[-1] != 1 or stride[-2] != dim or token < heads * dim:
        raise ValueError(
            f"a {tuple(shape)} tensor of strides {stride} is not token rows of {heads} heads x {dim} "
            "contiguous elements"
        )
    return token


def _tail_row_strides(
    v, dq, dk, dv, common: int | None = None
) -> tuple[int, int, int, int]:
    """The token-row strides (elements) of ``v``, ``dq``, ``dk`` and ``dv`` that the round-once tail takes after its
    operands; ``common``: the one stride of ``dq`` / ``dk`` / ``dv`` a caller gives with them (their strides are not
    read).  ``dq`` and ``dk`` share one stride (the tail's launcher rule): ``ValueError`` otherwise, before the
    launch."""
    if common is not None:
        common = int(common)
        return _row_stride(v), common, common, common
    strides = (_row_stride(v), _row_stride(dq), _row_stride(dk), _row_stride(dv))
    if strides[1] != strides[2]:
        raise ValueError(
            f"the round-once tail writes dq and dk at one token-row stride; got dq {strides[1]}, dk {strides[2]} "
            "elements"
        )
    return strides


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def cake_gdn_det_train_backward(
    forward, do, dht=None, *, initial_state=None, dbeta_dtype=None, grad_qkv=None
) -> dict[str, Any]:
    """The backward chain over one forward tape of :func:`cake_gdn_det_train_forward`; returns ``dq``, ``dk``, ``dv``,
    ``dg``, ``dbeta`` and ``dinitial_state``.  The tape must carry this package's tag (``"mode": "det"``): any other
    tape, such as one of the non-deterministic package (``"mode": "nondet"``), is refused (``ValueError``).

    ``grad_qkv``: three tensors of the shapes of ``dq``, ``dk`` and ``dv`` with a unit channel stride, head stride 128
    and a token-row stride each, ``dq`` and ``dk`` at one stride (the head views of one packed gradient row), or a
    callable that returns them (optionally followed by their common token-row stride, which the tail then takes
    without reading their strides).  When the call's tail is the round-once tail (:data:`F32X_TAILS`) it writes
    ``dq`` / ``dk`` / ``dv`` into them in place (a callable is called then, right before the tail, and only then) and
    the returned ``dq`` / ``dk`` / ``dv`` are those tensors; any other tail returns tensors of its own.  The package
    tail (:data:`PACKAGE_TAILS`) runs on the workspace slots and the arrival counters of the caller's stream
    (:meth:`Workspaces.take_rows`, :meth:`Workspaces.take_zeroed`), so the backwards of one stream run one at a time
    on them, in stream order.

    A call with ``dht`` runs the pv_tf32 programs where its route has them (:data:`PV_TF32`) and a call without it the
    0.2.8 ones; the tape entry ``final_state_grad`` (set by the autograd functions, :func:`_grads_of`) overrides
    ``dht is not None``."""
    if forward.get("mode") != _TAPE_MODE:
        raise ValueError("cake_gdn_det_train_backward reads the tape of a deterministic forward "
                         "(cake_gdn_det_train_forward, tag 'mode': 'det'); this tape does not carry that tag "
                         "(a tape of the non-deterministic package, for example)")  # fmt: skip
    arch = forward["arch"]
    meta = forward["meta"]
    options = forward.get("options") or ChainOptions()
    workspaces = _workspaces(options)
    qn, kn, A, g_cs = forward["qn"], forward["kn"], forward["A"], forward["g_cs"]
    scale = float(forward["scale"])
    v, beta32 = forward["v"], forward["beta32"]
    num_heads, num_v_heads = qn.shape[-2], v.shape[-2]
    total_tokens = qn.numel() // (num_heads * qn.shape[-1])
    device = qn.device
    index = _tensor_index(qn)
    # the caller's stream, read once: the key of the workspace table and of the package tail's counters
    stream = workspaces.stream_key(index)
    table = workspaces.table(index, stream)
    v_rows = _v_rows(v)
    if v_rows is None:  # a tape whose v is not one the stages read in place
        v, v_rows = _aligned(v, ROW_ALIGN_BYTES), num_v_heads
    do = _aligned(do, ROW_ALIGN_BYTES)
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    # the tape of a forward on wy_a's A lacks u (and w / h / v_new where it keeps no tape): the WY stage runs again
    # (recompute = 1, on the tape's A) where w is missing, or h / v_new are missing with u, and the forward scan where
    # h / v_new are missing
    rebuild_state = forward.get("h") is None or forward.get("v_new") is None
    if forward.get("w") is None or (rebuild_state and forward.get("u") is None):
        w, u = _launch_wy(
            arch,
            kn,
            v,
            g_cs,
            beta32,
            A,
            meta,
            num_heads,
            num_v_heads,
            total_tokens,
            recompute=True,
            v_rows=v_rows,
            ex=_DIRECT,
        )
    else:
        w, u = forward["w"], forward.get("u")
    if rebuild_state:
        h, v_new, _ = _launch_fwd_h(arch, kn, w, u, g_cs, _table_rows(initial_state, meta), meta, num_heads,
                                    num_v_heads,
                                    store_final_state=False, table=table, workspaces=workspaces,
                                    options=options)  # fmt: skip
    else:
        h, v_new = forward["h"], forward["v_new"]
    token_hv = (total_tokens, num_v_heads, HEAD_DIM)
    segs = meta.bwd_segments
    serial = not (segs.segment_len > 0 and segs.num_segments > meta.num_seqs)
    use_tc = serial and (
        meta.bwd_tc or _dhu_tc_selected(meta, num_v_heads, index, options)
    )
    # a split plan's pass 2 on dhu_tc2: the fused pre-scan computes the dv_local tiles it reads, no dv_local stage
    split_tc = not serial and (
        meta.bwd_tc or _sdt_selected(meta, segs, num_v_heads, index, options, arch)
    )
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    # the fused tail (fused_tail_stage): tail_g2 for a value-head group of two; unless a finalize pin, for three the
    # round-once tail_g3_f32x on the architectures of F32X_TAIL_ARCHS, else tail_g3, and for any other group the
    # package tail on those of PACKAGE_TAIL_ARCHS (for four or more from PACKAGE_TAIL_MIN_TOKENS_PER_SEQ mean tokens
    # per sequence on: the tokens and the sequences of the call, host facts on either table route); else None
    fused_tail = fused_tail_stage(arch, num_heads, num_v_heads, total_tokens=total_tokens, num_seqs=meta.num_seqs,
                                  dbeta_bf16=dbeta_bf16,
                                  finalize=options.finalize or forward.get("finalize_stage"))  # fmt: skip
    # the tcgen05 adjoint scan (dhu_tc2_stage): dhu_tc2_dvl32 for any group but two on DVL32_ARCHS, else dhu_tc2
    tc_stage = dhu_tc2_stage(arch, num_heads, num_v_heads)
    # A call whose final state receives a gradient (the autograd functions' tape entry ``final_state_grad``, else
    # a dht given) runs the pv_tf32 scan and tail where its route has them (pv_tf32_stages); the tail's stage is the
    # fused tail's, or dqkwg for the unfused tail
    tail_stage = fused_tail or "dqkwg"
    pv = None
    if forward.get("final_state_grad", dht is not None):
        pv = pv_tf32_stages(arch, tc_stage if use_tc or split_tc else None, tail_stage)
    if pv is not None:
        tc_stage, tail_stage = pv
    # two groups: the tcgen05 adjoint scan feeding the fused tail, for a serial plan beside tail_g2 (beside a group of
    # three's tail on the architectures of TWO_STREAM_SERIAL_G3_ARCHS) and for the pass 2 of a split plan beside a
    # group of three's tail (a split plan beside tail_g2 runs one stream); either form of a group's tail is its kind;
    # the package tail runs on the caller's stream after the whole adjoint scan
    tail_kind = (
        None
        if fused_tail is None or fused_tail in PACKAGE_TAILS
        else f"tail_g{num_v_heads // num_heads}"
    )
    serial_kinds = (
        ("tail_g2", "tail_g3") if arch in TWO_STREAM_SERIAL_G3_ARCHS else ("tail_g2",)
    )
    two_stream = (use_tc and tail_kind in serial_kinds) or (
        split_tc and tail_kind == "tail_g3"
    )
    groups = meta.bwd_groups if two_stream else None
    # the wavefront pair (see its section): the table's side of the rule is decided with the table; a CUDA graph
    # capture records the plain launches
    wavefront = (
        pv is None
        and meta.bwd_wavefront is not None
        and fused_tail == "tail_g2_bf16"
        and (use_tc or split_tc)
        and groups is None
        and not torch.cuda.is_current_stream_capturing()
    )
    # the register scan's stages and the segmented scan's refusals (host checks): before the adjoint scan's launches
    # and before the package tail's outputs and workspace (made after those launches), so a refused call takes no
    # workspace
    scan_stage, vb, pre, pre_threads = _scan_stages("dhu", segs.scan_config, arch)
    if not serial:
        if pre is None or not _available("stitch", arch):
            raise NotImplementedError(f"the segmented dhu scan needs {scan_stage.replace('dhu_r', 'dhu_pre_r')} and "
                                      f"stitch for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
    package_tail = fused_tail in PACKAGE_TAILS
    if not (use_tc or split_tc):
        dv_local = workspaces.take(table, "b_dv_local", token_hv, v.dtype, device)
        _launch("dv_local", meta.num_chunks, num_v_heads, 1, qn, kn, do, g_cs, dv_local, meta.chunk_start,
                meta.chunk_len, num_heads, num_v_heads, scale)  # fmt: skip
    dh = workspaces.take(table, "b_dh", h.shape, h.dtype, device)
    dv2 = workspaces.take(table, "b_dv2", token_hv, v.dtype, device)
    if has_initial_state:
        dh0 = torch.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        dh0 = workspaces.take(
            table,
            "b_zero_state",
            (0, num_v_heads, HEAD_DIM, HEAD_DIM),
            torch.float32,
            device,
        )
    if dht is not None:
        dht32 = _table_rows(dht, meta).contiguous()
        if dht32.dtype != torch.float32:
            dht32 = dht32.float()
        use_dht = 1
    else:
        dht32, use_dht = dh0, 0
    hdh32 = e32 = None
    scan_x: tuple = ()
    tail_x: tuple = ()
    lo_x: tuple = ()
    if pv is not None:
        # the pv_tf32 scan's extra operands after dh_out (h, v_new, hdh32, e) and the tail's after dh (hdh32, e32; a
        # chunk range reads its rows of hdh32 and the whole e); dv2's low word after dv2 on every launch that writes
        # or reads the dv2 tape (the scan, the tail, or the WY pullback of the unfused tail; never zero filled: the
        # scan writes every row a tail reads)
        hdh32 = workspaces.take(
            table, "b_hdh32", (meta.num_chunks, num_v_heads, 2), torch.float32, device
        )
        e32 = workspaces.take(
            table, "b_e32", (total_tokens, num_v_heads), torch.float32, device
        )
        scan_x, tail_x = (h, v_new, hdh32, e32), (hdh32, e32)
        lo_x = (workspaces.take(table, "b_dv2_lo", token_hv, v.dtype, device),)
    if use_tc and groups is None and not wavefront:
        # dv_local fused into the tcgen05 serial scan (seg_state is not read): on dhu_tc2 the register scan's dh / dv2 /
        # dh0; dhu_tc2_dvl32 rounds dv2 once (other dh / dv2 / dh0 bits)
        _launch(tc_stage, segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, *scan_x, dv2, *lo_x, dh0,
                dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    elif segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        # (the refusals above checked the pre-scan, the stitch and the segment length)
        seg_state = workspaces.take(table, "b_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "b_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        # the launch of the maps that split_map_stage names (split_map_launch): on two CTAs per (entry, value head),
        # grid z 2, where that grid fits one wave (the same bits), else on one; None where split_map_stage names the
        # fused pre-scan or such a grid keeps it (ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS); read with one lookup of the
        # answer decided once per (switches, architecture, heads, entries, SMs) (_split_map_call)
        map_launch = (
            _split_map_call(ADJOINT_MAP_ARCHS, DVL32_ARCHS, ADJOINT_MAP_SPLIT_ARCHS, ADJOINT_MAP_ONE_WAVE_PRESCAN_ARCHS,
                            arch, num_heads, num_v_heads, segs.num_summary, _device_props(index)[0])
            if split_tc and segs.summary is not None
            else None
        )  # fmt: skip
        if map_launch is not None:
            # the maps [B | M] of every segment but each sequence's first on the tcgen05 walk of dhu_tc2_dvl32, with
            # the pre-scan's warm-up threshold (a bound in a segment's last chunk keeps the whole map; the same
            # seg_state / seg_flag; a sequence's first map is never read)
            gm_stage, gm_z = map_launch
            _launch(gm_stage, segs.num_summary, num_v_heads, gm_z, qn, kn, w, do, g_cs, dht32, dh, dv2, dh0,
                    seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                    meta.seq_chunk_start, num_heads, num_v_heads, 0, 0, scale, segs.seg_index, segs.seg_mode,
                    warmup_thresh_log2(options), seg_flag)  # fmt: skip
        elif split_tc:
            # the pre-scan with the dv_local tiles of its chunks computed in the CTA (the same seg_state / seg_flag)
            _launch(SDT_PRESCAN, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, qn, kn, w, do, g_cs, dht32, dh,
                    dv2, dh0, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                    meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 0, scale, seg_flag,
                    warmup_thresh_log2(options))  # fmt: skip
        else:
            _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32,
                    dh, dv2, dh0, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                    meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 0, scale, seg_flag,
                    warmup_thresh_log2(options))  # fmt: skip
        dh_end = workspaces.take(table, "b_dh_end", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM),
                                 torch.float32, device)  # fmt: skip
        _launch("stitch", meta.num_seqs, num_v_heads, STITCH_VSPLIT, seg_state, dht32, dh_end, segs.seq_seg_start,
                seg_flag, num_v_heads, use_dht, 1)  # fmt: skip
        dht32, use_dht = dh_end, 1
    if split_tc and groups is None and not wavefront:
        # pass 2 on the tcgen05 scan: segment ``seg`` starts from dht[seg], the stitched dh_end[seg]; the first segment
        # of a sequence stores its dh0 (the register pass 2's contract)
        _launch(tc_stage, segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, *scan_x, dv2, *lo_x, dh0,
                dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    elif not (use_tc or split_tc):
        _launch(scan_stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32, dh,
                dv2, dh0, dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    if fused_tail is not None:
        # a value-head group of two / three: dqkwg, the WY pullback and the finalize as one cluster kernel (the same
        # bits as the three stages; the round-once tail of a group of three sums the group's dq / dk partials in FP32
        # before one rounding: the three stages' dv / dg / dbeta); any other group: the package tail, one kernel per
        # (chunk, value head) that sums the group's dq / dk partials in FP32 before one rounding (the three stages'
        # dv / dg / dbeta); a finalize pin keeps the three stages of every group but two
        tail = tail_stage
        round_once = fused_tail in F32X_TAILS
        if groups is not None:
            # R form: group A's tcgen05 adjoint scan (tc_stage, the caller's stream) first, so the host work below that
            # makes the tail's outputs (the caller's gradient row among them) runs while it does
            na, ca = groups.split, groups.split_chunk

            def dhu_tc2(s0, s1):
                _launch(tc_stage, s1 - s0, num_v_heads, 1, qn, kn, w, do, g_cs, dht32[s0:s1] if use_dht else dht32,
                        dh, *scan_x, dv2, *lo_x, dh0, dh0, meta.chunk_start, meta.chunk_len,
                        segs.seg_chunk_start[s0:s1 + 1], segs.seg_seq[s0:s1], meta.seq_chunk_start, num_heads,
                        num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip

            dhu_tc2(0, na)
        stride = None
        if round_once and grad_qkv is not None:
            # the caller's dq / dk / dv (the head views of one packed gradient row), made right before the tail (after
            # the adjoint scan's launches; on the R form after group A's)
            qkv = grad_qkv() if callable(grad_qkv) else grad_qkv
            dq, dk, dv = qkv[0], qkv[1], qkv[2]
            if len(qkv) > 3:
                stride = qkv[3]
        else:
            # every tail's outputs after the adjoint scan's launches: the device runs the scan while the host makes them
            dv = torch.empty(v.shape, dtype=v.dtype, device=device)
            dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)
            dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
        dbeta = torch.empty(forward.get("beta_shape") or (total_tokens, num_v_heads),
                            dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)  # fmt: skip
        dg = torch.empty(
            forward.get("g_shape") or (total_tokens, num_v_heads),
            dtype=torch.float32,
            device=device,
        )
        normalize = 1 if forward.get("normalize_qk", True) else 0
        # the round-once tail takes the token-row strides of v, dq, dk and dv; the other tails v's token pitch in heads
        strides = _tail_row_strides(v, dq, dk, dv, stride) if round_once else (v_rows,)
        if wavefront:
            # the adjoint scan (serial, or the pass 2 from the stitched dht: the pre-scan and the stitch are already
            # on the caller's stream before the fork) on the caller's stream and the tail of its tiles as they are
            # stored on the side stream, after every scan CTA has started
            flags = workspaces.take(table, "b_wavefront_ready", (2 + meta.num_chunks * num_v_heads,), torch.int32,
                                    device)  # fmt: skip
            flags.zero_()
            flags = flags.view(torch.uint32)  # the pair's u32 counts
            order, ready = _wavefront_order_tensor(
                meta.seq_lens, meta.order, int(segs.segment_len), index
            )
            _use_table(order, ready, index)
            main = torch.cuda.current_stream(index)
            side = _side_stream(index, main)
            fork, join = _wavefront_events(index, main)
            fork.record(main)
            _launch(meta.bwd_wavefront, segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, dv2, dh0, dh0,
                    meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                    num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale, flags)  # fmt: skip
            try:
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    _wait_entry(index)(flags, segs.num_segments * num_v_heads)
                    _launch("tail_g2_bf16_wavefront", meta.num_chunks * num_v_heads, 1, 1, qn, kn, forward["rstd_q"],
                            forward["rstd_k"], v_new, do, dv2, h, dh, g_cs, v, A, beta32, dq, dk, dg, dv, dbeta,
                            meta.chunk_start, meta.chunk_len, num_heads, num_v_heads, scale, normalize, flags,
                            order, v_rows)  # fmt: skip
            finally:
                join.record(side)  # the join
                main.wait_event(join)
        elif package_tail:
            # the package tail on the caller's stream (never grouped, no wavefront pair): the chunk table in launches of
            # at most the workspace's slots, in stream order (tail_segments), on the workspace slots (written before
            # they are read, never zeroed) and the arrival counters (zero when created, left zero by every launch) of
            # the caller's stream, taken like the outputs after the adjoint scan's launches (the counters' zero fill,
            # when they are created, is queued between the scan and the tail)
            ranges, tiles = tail_segments(meta.num_chunks, num_heads, num_v_heads)
            ws = workspaces.take_rows(table, "tail_ws", max(tiles * (num_v_heads // num_heads), 1),
                                      (TAIL_TILE_FLOATS,), torch.float32, device)  # fmt: skip
            ws_count = workspaces.take_zeroed(
                stream, "tail_count", max(tiles, 1), (), torch.uint32, device
            )
            for a, b in ranges:
                # the chunk rows [a, b) of the per-chunk tensors: the tensors themselves for the whole table
                # (and the pv_tf32 tail's rows of hdh32 and the whole e)
                rows = (h, dh, meta.chunk_start, meta.chunk_len)
                x_r = tail_x
                if (a, b) != (0, meta.num_chunks):
                    rows = tuple(t[a:b] for t in rows)
                    x_r = () if pv is None else (hdh32[a:b], e32)
                h_r, dh_r, cs_r, cl_r = rows
                _launch(tail, (b - a) * num_v_heads, 1, 1, qn, kn, forward["rstd_q"], forward["rstd_k"], v_new, do,
                        dv2, *lo_x, h_r, dh_r, *x_r, g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, ws, ws_count, cs_r,
                        cl_r, num_heads, num_v_heads, scale, normalize, tiles, v_rows)  # fmt: skip
        elif groups is None:
            _launch(tail, meta.num_chunks * num_v_heads, 1, 1, qn, kn, forward["rstd_q"], forward["rstd_k"], v_new, do,
                    dv2, *lo_x, h, dh, *tail_x, g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, meta.chunk_start,
                    meta.chunk_len, num_heads, num_v_heads, scale, normalize, *strides)  # fmt: skip
        else:
            # R form: the caller's stream has run the tcgen05 adjoint scan (tc_stage) of group A (the full rounds,
            # launched above); it runs the tail of A while the side stream runs the scan of group B (the partial
            # round), then joins it and runs the tail of B

            def tail_chunks(c0, c1):
                _launch(tail, (c1 - c0) * num_v_heads, 1, 1, qn, kn, forward["rstd_q"], forward["rstd_k"], v_new, do,
                        dv2, *lo_x, h[c0:c1], dh[c0:c1], *(() if pv is None else (hdh32[c0:c1], e32)), g_cs, v, A,
                        beta32, dq, dk, dg, dv, dbeta, meta.chunk_start[c0:c1], meta.chunk_len[c0:c1], num_heads,
                        num_v_heads, scale, normalize, *strides)  # fmt: skip

            main = torch.cuda.current_stream(index)
            side = _side_stream(index, main)
            fork = torch.cuda.Event()
            fork.record(main)
            try:
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    dhu_tc2(na, segs.num_segments)
                tail_chunks(0, ca)
            finally:
                main.wait_stream(side)  # the join
            tail_chunks(ca, meta.num_chunks)
        return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta,
                "dinitial_state": _caller_rows(dh0, meta) if has_initial_state else None}  # fmt: skip
    dq_hv = workspaces.take(table, "b_dq_hv", token_hv, qn.dtype, device)
    dk_hv = workspaces.take(table, "b_dk_hv", token_hv, qn.dtype, device)
    dw = workspaces.take(table, "b_dw", w.shape, w.dtype, device)
    dg1 = workspaces.take(
        table, "b_dg1", (total_tokens, num_v_heads), torch.float32, device
    )
    _launch(tail_stage, meta.num_chunks, num_v_heads, 1, qn, kn, v_new, do, dv2, h, dh, *tail_x, g_cs, dq_hv, dk_hv, dw,
            dg1, meta.chunk_start, meta.chunk_len, num_heads, num_v_heads, scale)  # fmt: skip
    dk2 = workspaces.take(table, "b_dk2", token_hv, qn.dtype, device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=device)
    dbeta = torch.empty(forward.get("beta_shape") or (total_tokens, num_v_heads),
                        dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)  # fmt: skip
    dg2 = workspaces.take(
        table, "b_dg2", (total_tokens, num_v_heads), torch.float32, device
    )
    # the WY pullback; after dqkwg_pv_tf32 its DV_LO_WY twin, which reads dv2's low word
    wy = "wy_bwd_bf16" if dbeta_bf16 else "wy_bwd"
    _launch(DV_LO_WY[wy] if lo_x else wy, meta.num_chunks, num_v_heads, 1, kn, v, A, dw, dv2, *lo_x, g_cs, beta32, dk2,
            dv, dbeta, dg2, meta.chunk_start, meta.chunk_len, num_heads, num_v_heads, v_rows)  # fmt: skip
    dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)
    dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
    dg = torch.empty(
        forward.get("g_shape") or (total_tokens, num_v_heads),
        dtype=torch.float32,
        device=device,
    )
    sm_count, _smem, _major = _device_props(index)
    fin = forward.get("finalize_stage") or finalize_stage(meta.num_chunks, num_heads, num_v_heads, sm_count,
                                                          options.finalize)  # fmt: skip
    _launch(fin, meta.num_chunks, num_heads, 1, qn, kn,
            forward["rstd_q"], forward["rstd_k"], dq_hv, dk_hv, dk2, dg1, dg2, dq, dk, dg, meta.chunk_start,
            meta.chunk_len, num_heads, num_v_heads, 1 if forward.get("normalize_qk", True) else 0)  # fmt: skip
    return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta,
            "dinitial_state": _caller_rows(dh0, meta) if has_initial_state else None}  # fmt: skip


# --------------------------------------------------------------------------- #
# autograd wrapper with FLA's signature (no context parallelism)
# --------------------------------------------------------------------------- #
_META_FRONT: dict = {}
_META_FRONT_LIMIT = 256
_OFFSET_DTYPES = (torch.int32, torch.int64)
# the checked tensors in check order; the last two are optional (None skips them)
_INPUT_NAMES = ("q", "k", "v", "g", "beta", "initial_state", "cu_seqlens")
_OPTIONAL_INPUT = 5


def _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu) -> None:
    """The API-boundary checks of :func:`cake_gdn_det`: devices, shapes and the offsets' form, from tensor
    metadata only (no device synchronization; the values of ``cu_seqlens_cpu`` are checked when its layout is first
    planned, those of the device ``cu_seqlens`` not at all).  FLA's ``chunk_gated_delta_rule`` checks the head
    counts, the batch of a packed call and the rows of ``initial_state``; the rest is this backend's contract.

    Every call pays these checks, so they read the cheapest metadata: ``is_cuda`` and ``get_device()`` instead of
    ``torch.device`` objects (made only for an error message), each shape once."""
    index = -1
    for i, t in enumerate((q, k, v, g, beta, initial_state, cu_seqlens)):
        if t is None and i >= _OPTIONAL_INPUT:
            continue
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"{_INPUT_NAMES[i]} must be a CUDA tensor")
        if i == 0:
            index = t.get_device()
        elif t.get_device() != index:
            raise ValueError(f"{_INPUT_NAMES[i]} is on {t.device}, q on {q.device}")
    q_shape, v_shape = q.shape, v.shape
    if len(q_shape) != 4 or q_shape[3] != HEAD_DIM:
        raise ValueError(f"q must be [B, T, H, {HEAD_DIM}]; got {tuple(q_shape)}")
    if k.shape != q_shape:
        raise ValueError(
            f"k must have q's shape {tuple(q_shape)}; got {tuple(k.shape)}"
        )
    if (
        len(v_shape) != 4
        or v_shape[0] != q_shape[0]
        or v_shape[1] != q_shape[1]
        or v_shape[3] != HEAD_DIM
    ):
        raise ValueError(
            f"v must be [{q_shape[0]}, {q_shape[1]}, HV, {HEAD_DIM}]; got {tuple(v_shape)}"
        )
    num_heads, num_v_heads = q_shape[2], v_shape[2]
    if num_heads == 0 or num_v_heads == 0 or num_v_heads % num_heads != 0:
        raise ValueError(f"the value heads (HV={num_v_heads}) must be a positive multiple of the q / k heads "
                         f"(H={num_heads})")  # fmt: skip
    gate_shape = v_shape[:3]
    if g.shape != gate_shape:
        raise ValueError(f"g must be {tuple(gate_shape)}; got {tuple(g.shape)}")
    if beta.shape != gate_shape:
        raise ValueError(f"beta must be {tuple(gate_shape)}; got {tuple(beta.shape)}")
    if cu_seqlens is None:
        num_seqs = q_shape[0]
    else:
        offsets = cu_seqlens.numel()
        if (
            cu_seqlens.dim() != 1
            or offsets < 2
            or cu_seqlens.dtype not in _OFFSET_DTYPES
        ):
            raise ValueError(f"cu_seqlens must be a 1-D int32 / int64 tensor of at least two offsets; got "
                             f"{cu_seqlens.dtype} {tuple(cu_seqlens.shape)}")  # fmt: skip
        if q_shape[0] != 1:
            raise ValueError("cu_seqlens requires a leading batch dimension of 1")
        num_seqs = offsets - 1
        if cu_seqlens_cpu is not None:
            if isinstance(cu_seqlens_cpu, torch.Tensor):
                if (
                    cu_seqlens_cpu.dim() != 1
                    or cu_seqlens_cpu.dtype not in _OFFSET_DTYPES
                ):
                    raise ValueError(f"cu_seqlens_cpu must be a 1-D int32 / int64 tensor; got {cu_seqlens_cpu.dtype} "
                                     f"{tuple(cu_seqlens_cpu.shape)}")  # fmt: skip
                count = cu_seqlens_cpu.numel()
            else:
                count = len(cu_seqlens_cpu)
            if count != offsets:
                raise ValueError(
                    f"cu_seqlens_cpu has {count} offsets, cu_seqlens {offsets}"
                )
    if initial_state is not None:
        state_shape = (num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must be {state_shape} (one state per sequence); got "
                             f"{tuple(initial_state.shape)}")  # fmt: skip


def _host_offsets(cu_seqlens_cpu) -> tuple[int, ...]:
    """The host offsets as a tuple of ints (one ``tolist`` for a tensor, which may also live on the device)."""
    if isinstance(cu_seqlens_cpu, torch.Tensor):
        return tuple(cu_seqlens_cpu.tolist())
    return tuple(int(x) for x in cu_seqlens_cpu)


def _check_offsets(layout: tuple[int, ...], total_tokens: int) -> None:
    """The host offsets of a packed call start at 0, do not decrease and end at the tokens of ``q`` (an empty
    sequence is allowed)."""
    if (
        layout[0] != 0
        or layout[-1] != total_tokens
        or any(b < a for a, b in zip(layout[:-1], layout[1:], strict=True))
    ):
        raise ValueError(f"cu_seqlens_cpu must start at 0, not decrease and end at the {total_tokens} tokens of q; "
                         f"got {list(layout[:4])}{' ...' if len(layout) > 4 else ''} ending at {layout[-1]}")  # fmt: skip


def _chunk_meta(
    cu_seqlens,
    cu_seqlens_cpu,
    batch,
    seq_len,
    index: int,
    num_v_heads,
    options: ChainOptions,
    *,
    num_heads: int | None = None,
    a_inputs: bool = False,
) -> ChunkMeta:
    """The chunk and segment tables of one call on device ``index``: a ``host_cache`` hit on the layout's key, else
    planned (the layout's sequence lengths are derived only then).  ``num_heads`` (the key heads) lets a host-built
    table take the serial-scan rule's adjoint plan (:func:`_serial_adjoint_plan`; ``None``: the planner's plans);
    then the tables priced for the tcgen05 chains (:func:`host_plans`).  ``a_inputs`` (the call meets
    :func:`_wy_a_inputs`) lets a host-built table take the serial forward table of ``FWD_A_SERIAL_KEEP`` and carry
    the split forward's tables (:func:`_host_forward_split`)."""
    total_tokens = int(batch) * int(seq_len)
    if cu_seqlens is None:
        layout: tuple[Any, ...] = ("equal", int(batch), int(seq_len))
    elif batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    elif cu_seqlens_cpu is None:
        num_seqs = int(cu_seqlens.numel()) - 1
        return build_chunk_meta_from_cu_seqlens(
            cu_seqlens,
            total_tokens,
            scan_plans=scan_plans(total_tokens, num_seqs, int(num_v_heads), int(index),
                                  scan_config=options.scan_config, segment_chunks=options.segment_chunks),
        )  # fmt: skip
    else:
        layout = _host_offsets(cu_seqlens_cpu)
    # the key carries everything the tables depend on: the dhu_scan pin too, which keeps the planner's adjoint plan
    # where the serial-scan rule would make it serial (:func:`_serial_adjoint_plan`)
    key = (
        layout,
        int(num_v_heads),
        int(index),
        total_tokens,
        options.pins,
        options.dhu_scan,
        options.lpt,
        options.streams,
        num_heads,
        options.finalize,
        TC_PLAN,
        tuple(sorted(CHAIN_PLAN)),
        bool(a_inputs),
        FWD_A_ARCHS,
        FWD_SPLIT_ARCHS,
        tuple(sorted(cake_plan.FORWARD_SPLIT_MIN_PIECE.items())),
        tuple(sorted(FWD_A_SERIAL_KEEP.items())),
    )
    if options.host_cache:
        meta = _META_FRONT.get(key)
        if meta is not None:
            # the split forward's tables too: the forward reads them from the hit without a lookup of its own
            _use_meta_tables(meta, int(index))
            return meta
    if cu_seqlens is None:
        seq_lens = (int(seq_len),) * int(batch)
    else:
        # a layout reaches the table cache only after this check, so a cache hit is a checked layout
        _check_offsets(layout, total_tokens)
        seq_lens = tuple(b - a for a, b in zip(layout[:-1], layout[1:], strict=True))
    plans, bwd_tc = host_plans(seq_lens, int(num_v_heads), int(index), options, total_tokens=total_tokens,
                               num_heads=num_heads, a_inputs=a_inputs)  # fmt: skip
    meta = _host_chunk_meta(seq_lens, plans, int(index), options.lpt)
    if bwd_tc:
        meta = replace(meta, bwd_tc=True)
    if a_inputs:
        fwd_split = _host_forward_split(meta, int(num_v_heads), int(index), options)
        if fwd_split is not None:
            meta = replace(meta, fwd_split=fwd_split)
    if options.streams == 2:
        fwd_groups, bwd_groups = sequence_groups(
            meta, int(num_v_heads), _device_props(int(index))[0]
        )
        if fwd_groups is not None or bwd_groups is not None:
            meta = replace(meta, fwd_groups=fwd_groups, bwd_groups=bwd_groups)
    scan = _wavefront_table(meta, int(num_v_heads), int(index))
    if scan is not None:
        meta = replace(meta, bwd_wavefront=scan)
    # a table built inside a CUDA graph capture (see "cached device tables") is not kept
    capturing = meta.chunk_start.is_cuda and torch.cuda.is_current_stream_capturing()
    if options.host_cache and not capturing:
        if len(_META_FRONT) >= _META_FRONT_LIMIT:
            old = _META_FRONT.pop(next(iter(_META_FRONT)), None)
            if old is not None:  # its forward plans leave with it
                _drop_meta_plans(old)
        _META_FRONT[key] = meta
    return meta


def _planned_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel, options,
                     meta, arch, index, num_heads, num_v_heads, tape, seen) -> dict[str, Any]:  # fmt: skip
    """The forward's result (:func:`_train_forward`) of a :func:`_function_forward` call that builds its plan key
    (``seen``: the record gate's entry of its layout and tape mode, ``None`` with the gate off): a replay of the key's
    plan (:func:`_replay_forward`), else the general path, which records the key's plan when it has none ready and the
    record credit holds a recording.  Inside a CUDA graph capture the call runs the general path and builds no key."""
    gated = seen is not None
    plan = recorder = None
    stream = _stream_key(index)
    # a CUDA graph capture (never on the default stream) runs the general path on its own workspaces
    if not (stream[1] and torch.cuda.is_current_stream_capturing()):
        # besides meta's tables and the module globals (ready()), the key holds what the general path's steps
        # follow: the stream, the tape mode (a checkpoint's first pass and its recompute differ), the arguments,
        # the inputs' layouts and which inputs are one object (a call with ``k is q`` records q's slot for both,
        # so a plan of another pattern would read k for q)
        inputs = (q, k, v, g, beta, initial_state)
        alias = (None if len({id(t) for t in inputs}) == len(inputs)
                 else tuple(next(j for j, s in enumerate(inputs) if s is t) for t in inputs))  # fmt: skip
        # CakeGDNDetPackedFunction's forward sets row_shape
        packed = getattr(ctx, "row_shape", None) is not None
        level = cake_plan.tape_state(arch, options.save_state)
        # (a float argument by its bits: -0.0 == 0.0, yet a replay passes the recorded one to the launches)
        fkey = (id(meta), stream, arch, options, _float_bits(options.warmup_log2), tape, packed, num_heads,
                num_v_heads, _float_bits(scale), bool(use_qk_l2norm_in_kernel), bool(output_final_state),
                _slot_layout(inputs), alias, level)  # fmt: skip
        plan = _FWD_PLANS.get(fkey)
        if plan is not None and plan.ops is None:
            plan = None  # the key's calls keep the general path
            if gated:
                seen[2] = _REFUSED
        elif plan is None or not plan.ready(meta, stream):
            plan = None
            if not gated or _fwd_spend():
                recorder = _FwdRecorder(inputs, index, meta)
            else:  # no key until the credit holds a recording (for every call class of the entry)
                seen[2] = _UNPLANNED
    if plan is not None:
        fwd = _replay_forward(plan, inputs)
        if gated:
            seen[2] = _PLANNED
            _fwd_replayed()
        return fwd
    if recorder is None:
        return _train_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                              output_final_state=output_final_state, normalize_qk=use_qk_l2norm_in_kernel,
                              options=options, tape=tape, ex=_DIRECT)  # fmt: skip
    with _watched(recorder):
        fwd = _train_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                             output_final_state=output_final_state, normalize_qk=use_qk_l2norm_in_kernel,
                             options=options, tape=tape, ex=recorder)  # fmt: skip
    recorded = recorder.plan(fwd)
    # the recorder's (weak) references to the call's tensors
    recorder.slots.clear()
    # a refusal of a call that filled a table cache or met another hook is not kept: the key's next call records
    # again
    if recorded is not None or not recorder.again:
        if recorded is None:  # the key's calls keep the general path
            recorded = _ForwardPlan(
                None, 0, recorder.stream, None, None, meta, (), None, (), 0, ()
            )
        _keep_fwd_plan(fkey, meta, recorded)
        if gated:
            seen[2] = _PLANNED if recorded.ops is not None else _REFUSED
    elif gated:
        seen[2] = _UNPLANNED
    return fwd


def _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                      use_qk_l2norm_in_kernel, options):  # fmt: skip
    """The forward of :class:`CakeGDNDetFunction` and :class:`CakeGDNDetPackedFunction` over q / k / v (the inputs,
    or the head views of the packed row): the chain, the saved tape and ``ctx``'s fields.  A call that has a plan key
    (a layout of the host table cache with ``host_cache``: a plan's steps take the pooled workspaces of its stream)
    and that the record gate lets build it (:data:`FWD_PLAN_GATE`: from the ``FWD_PLAN_REPEAT``-th forward of the
    layout in its tape mode that the gate counts, while the layout has a plan or the record credit holds a recording)
    replays the key's plan or records it (:func:`_planned_forward`); any other runs the general path
    (:func:`_train_forward` through :data:`_DIRECT`)."""
    global _FWD_CREDIT
    batch, seq_len, num_heads, key_dim = q.shape
    num_v_heads = v.shape[2]
    index = _tensor_index(q)
    arch = cake_backend.device_arch(index)
    meta = _chunk_meta(
        cu_seqlens,
        cu_seqlens_cpu,
        batch,
        seq_len,
        index,
        num_v_heads,
        options,
        num_heads=num_heads,
        a_inputs=_wy_a_inputs(
            num_heads, num_v_heads, beta.dtype, use_qk_l2norm_in_kernel, options
        ),
    )
    tape = _forward_tape()
    seen = None
    # a layout of the host table cache (host-built or equal-length) with host_cache: the forward's plan, once the record
    # gate lets the call build its key (:func:`_fwd_gate`; an early forward of the layout in its tape mode builds none).
    # A host q runs the general path, which pools nothing and refuses it at the first launch (the key's layouts do not
    # name the device: such a call must not replay a plan of device inputs)
    planned = (
        FWD_PLAN
        and options.host_cache
        and (cu_seqlens is None or cu_seqlens_cpu is not None)
        and q.is_cuda
    )
    if planned and FWD_PLAN_GATE:
        # with less credit than one recording (with this call's share) and no plan kept on the layout, the gate could
        # only send the call to the general path: add the call's share here and skip the gate's call (its host time,
        # cold in the caches of such a call, is a few us).  The call makes no entry and is not counted among the
        # layout's forwards (FWD_PLAN_GATE)
        credit = _FWD_CREDIT + FWD_PLAN_PAYBACK
        if credit < FWD_PLAN_PAYBACK * FWD_PLAN_TRICKLE and id(meta) not in _PLAN_KEYS:
            _FWD_CREDIT = credit
            planned = False
        else:
            planned = (seen := _fwd_gate(meta, tape)) is not None
    if planned:
        fwd = _planned_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel,
                               options, meta, arch, index, num_heads, num_v_heads, tape, seen)  # fmt: skip
    else:
        fwd = _train_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                             output_final_state=output_final_state, normalize_qk=use_qk_l2norm_in_kernel,
                             options=options, tape=tape, ex=_DIRECT)  # fmt: skip
    # a fixed saved list of 13: the state the architecture's tape level keeps under save_state
    # (cake_plan.tape_state: w, h and v_new under T_full), a placeholder of the same shape, dtype and device for
    # the rest (a checkpoint's first pass compares the two passes' saved lists); the backward reads only the state
    # the level keeps (_function_backward).  A forward without a tape (a checkpoint's first pass, whose saved list is
    # dropped, or a call that records no backward) saves the state it formed anyway, which retains nothing, and
    # placeholders only for the state it did not form
    kept = cake_plan.tape_state(arch, options.save_state) if tape else ()
    state_tensors = [
        fwd.get(name) if not tape or name in kept else None
        for name in cake_plan.STATE_NAMES
    ]
    if any(t is None for t in state_tensors):
        state_tensors = [
            cake_plan.placeholder(shape, dtype, fwd["qn"].device) if t is None else t
            for t, (shape, dtype) in zip(
                state_tensors,
                _forward_state(meta, fwd["kn"], fwd["v"]),
                strict=True,
            )
        ]
    ctx.save_for_backward(fwd["qn"], fwd["kn"], fwd["rstd_q"], fwd["rstd_k"], fwd["v"], fwd["g_cs"],
                          fwd["beta32"], fwd["A"], initial_state, *state_tensors)  # fmt: skip
    ctx.meta, ctx.arch, ctx.scale = meta, arch, float(scale)
    ctx.grad_shapes = (g.shape, beta.shape)
    ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
    ctx.has_initial_state_input = initial_state is not None
    ctx.dtypes = (g.dtype, beta.dtype)
    ctx.options = options
    # the forward tape's tag; the backward reads only a tape of its own
    ctx.mode = fwd["mode"]
    output = fwd["output"]
    if output.dtype != q.dtype:
        output = output.to(q.dtype)
    final_state = fwd["final_state"] if output_final_state else None
    if output_final_state:
        # autograd passes None for an output that receives no gradient (:func:`_grads_of`), which tells a final state
        # without a gradient apart from a zero one; the zeros autograd would fill in: their shapes, dtypes and device.
        # A call without a final state keeps autograd's zero fill and sets no zero_fill (its backward is 0.2.8's)
        ctx.set_materialize_grads(False)
        ctx.zero_fill = (output.shape, output.dtype, output.device,
                         None if final_state is None else (final_state.shape, final_state.dtype))  # fmt: skip
    return output, final_state


def _grads_of(ctx, do, dht):
    """``(do, dht, final_state_grad)`` of an autograd backward: ``final_state_grad`` is whether autograd gave the final
    state a gradient (the forward turned autograd's zero fill off, so ``dht`` is None for a final state that receives
    none, also when the caller asked for it), and ``do`` / ``dht`` are the gradients with the zeros that autograd
    would fill in for an output without one, the chain's 0.2.8 inputs and bits (:data:`PV_TF32` runs the pv_tf32
    programs only with ``final_state_grad``).  A call without a final state (no ``zero_fill``: the forward kept
    autograd's zero fill) has no final-state gradient and takes ``do`` / ``dht`` as autograd gives them."""
    zero_fill = getattr(ctx, "zero_fill", None)
    if zero_fill is None:
        return do, dht, False
    o_shape, o_dtype, device, state = zero_fill
    final_state_grad = dht is not None
    if do is None:
        do = torch.zeros(o_shape, dtype=o_dtype, device=device)
    if dht is None and state is not None:
        dht = torch.zeros(state[0], dtype=state[1], device=device)
    return do, dht, final_state_grad


def _function_backward(ctx, do, dht, grad_qkv=None):
    """The backward of both autograd functions: the gradients of q, k, v, g, beta and the initial state, in the
    dtypes of g and beta (``grad_qkv``: see :func:`cake_gdn_det_train_backward`)."""
    do, dht, final_state_grad = _grads_of(ctx, do, dht)
    qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, *state_tensors = (
        ctx.saved_tensors
    )
    g_shape, beta_shape = ctx.grad_shapes
    g_dtype, beta_dtype = ctx.dtypes
    tape = {"arch": ctx.arch, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "v": v_flat, "g_cs": g_cs,
            "beta32": beta32, "A": A, "meta": ctx.meta, "scale": ctx.scale, "normalize_qk": ctx.normalize_qk,
            "g_shape": g_shape, "beta_shape": beta_shape, "options": ctx.options, "mode": ctx.mode,
            "final_state_grad": final_state_grad}  # fmt: skip
    # the state of the architecture's tape level under save_state (a checkpoint's recomputed forward keeps it even
    # where its first pass kept none); every other entry is a placeholder, never read: the backward rebuilds that
    # state (as it does for a level entry that is a placeholder, judged from its strides alone)
    kept = cake_plan.tape_state(ctx.arch, ctx.options.save_state)
    tape.update(
        (name, saved)
        for name, saved in zip(cake_plan.STATE_NAMES, state_tensors, strict=True)
        if name in kept and saved is not None and not cake_plan.is_placeholder(saved)
    )
    grads = cake_gdn_det_train_backward(
        tape,
        do.contiguous(),
        dht,
        initial_state=saved_state,
        dbeta_dtype=beta_dtype,
        grad_qkv=grad_qkv,
    )
    dg = grads["dg"]
    if dg.dtype != g_dtype:
        dg = dg.to(g_dtype)
    dbeta = grads["dbeta"]
    if dbeta.dtype != beta_dtype:
        dbeta = dbeta.to(beta_dtype)
    dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
    return grads["dq"], grads["dk"], grads["dv"], dg, dbeta, dh0


class CakeGDNDetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                use_qk_l2norm_in_kernel, options):  # fmt: skip
        return _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
                                 cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip

    @staticmethod
    def backward(ctx, do, dht):
        dq, dk, dv, dg, dbeta, dh0 = _function_backward(ctx, do, dht)
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None


# --------------------------------------------------------------------------- #
# q / k / v as the head views of one packed row: one autograd input, one packed gradient row
# --------------------------------------------------------------------------- #
# The customer layout: one BF16 row [..., T, (2 H + HV) x 128] per token, q / k / v its head views
# (``row.split((H * 128, H * 128, HV * 128), -1)``, each viewed as [B, T, heads, 128]).  Run on the views, autograd
# returns dq / dk / dv as three tensors and the split's backward copies them into a new row gradient.  With the packed
# row as the one input instead, the backward allocates the row gradient and the round-once tail writes dq / dk / dv
# straight into its head views through their token-row strides: no concatenation, the same bits.  The views are
# recognized from their metadata alone (:func:`_packed_row`); any other layout takes :class:`CakeGDNDetFunction`, as
# does a call whose tail is not the round-once tail (:data:`F32X_TAILS`) or that records no gradient for the row.
# ``PACKED_ROW`` off: every call takes :class:`CakeGDNDetFunction`.
PACKED_ROW = True


def _packed_row(q, k, v) -> torch.Tensor | None:
    """The packed row whose head views ``q``, ``k`` and ``v`` are, or ``None``: their common base, a contiguous BF16
    tensor of ``B x T`` rows of ``(2 H + HV) x 128`` elements that requires grad, q / k / v at its row offsets 0,
    ``H x 128`` and ``2 H x 128`` with its row stride, head stride 128 and unit channel stride, each requiring grad
    through a view of it.  It reads each tensor's shape, strides and offset once (tuples compared whole; the strides
    of size-one axes, which any value may take, only when the whole compare fails), after :func:`_check_inputs`,
    whose shape checks (q ``[B, T, H, 128]``, k of q's shape, v ``[B, T, HV, 128]``) it does not repeat."""
    base = q._base
    if base is None or k._base is not base or v._base is not base:
        return None
    if not (
        base.requires_grad and q.requires_grad and k.requires_grad and v.requires_grad
    ):
        return None
    if base.dtype != torch.bfloat16:
        return None
    batch, seq_len, num_heads, dim = q.shape
    width = (2 * num_heads + v.shape[2]) * dim
    base_shape = base.shape
    if (
        not base_shape
        or base_shape[-1] != width
        or base.numel() != batch * seq_len * width
        or not base.is_contiguous()
    ):
        return None
    offset = base.storage_offset()
    if (
        q.storage_offset() != offset
        or k.storage_offset() != offset + num_heads * dim
        or v.storage_offset() != offset + 2 * num_heads * dim
    ):
        return None
    strides = (seq_len * width, width, dim, 1)
    for t in (q, k, v):
        stride = t.stride()
        if stride != strides and any(
            size > 1 and got != want
            for size, got, want in zip(t.shape, stride, strides, strict=True)
        ):
            return None
    return base


class CakeGDNDetPackedFunction(torch.autograd.Function):
    """:class:`CakeGDNDetFunction` with the packed row as its one q / k / v input (see :func:`_packed_row`): the same
    forward on the caller's head views of the row, and a backward that returns the row's gradient, written by the
    round-once tail.  ``views`` is the tuple of those views: a container, so autograd does not take them as inputs,
    and the forward reads them as they are instead of viewing the row again."""

    @staticmethod
    def forward(ctx, row, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                use_qk_l2norm_in_kernel, options, views):  # fmt: skip
        q, k, v = views
        ctx.row_shape = row.shape
        ctx.view_shapes = (q.shape, v.shape)
        return _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
                                 cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip

    @staticmethod
    def backward(ctx, do, dht):
        # the gradient row and its q / k / v head views (one strided view each, at the row offsets of the forward's
        # views), which the tail writes through their token-row strides; made when the chain reaches the tail (on the
        # two-stream R form after group A's adjoint scan is launched), so the host work runs while the adjoint scan's
        # kernels do instead of before the first of them
        made = []

        def row_views():
            q_shape, v_shape = ctx.view_shapes
            seq_len, num_heads, dim = q_shape[1], q_shape[2], q_shape[3]
            width = ctx.row_shape[-1]
            strides = (seq_len * width, width, dim, 1)
            # on the output's device (do is None when the output receives no gradient, :func:`_grads_of`; never
            # for a call without a final state, which sets no zero_fill)
            d_row = torch.empty(
                ctx.row_shape,
                dtype=torch.bfloat16,
                device=ctx.zero_fill[2] if do is None else do.device,
            )
            views = (
                d_row.as_strided(q_shape, strides, 0),
                d_row.as_strided(q_shape, strides, num_heads * dim),
                d_row.as_strided(v_shape, strides, 2 * num_heads * dim),
            )
            made.append((d_row, views))
            return (*views, width)

        dq, dk, dv, dg, dbeta, dh0 = _function_backward(ctx, do, dht, row_views)
        if not made:
            # a tail that did not write the row's views (not the round-once tail)
            row_views()
        d_row, views = made[0]
        for view, grad in zip(views, (dq, dk, dv), strict=True):
            if grad is not view:
                view.copy_(grad)
        return d_row, dg, dbeta, None, dh0, None, None, None, None, None, None


def _packed_route(
    q, k, v, beta, cu_seqlens, options: ChainOptions
) -> torch.Tensor | None:
    """The packed row a call takes :class:`CakeGDNDetPackedFunction` on, or ``None``: under ``PACKED_ROW``, when ``q``,
    ``k`` and ``v`` are the head views of a packed row (:func:`_packed_row`) and the backward's tail is the round-once
    tail (:func:`fused_tail_stage`; the BF16-``dbeta`` form for BF16 ``beta``).  Only a value-head group of three
    takes the round-once tail, so any other group is refused before the row's checks and the rule."""
    if not PACKED_ROW or v.shape[2] != 3 * q.shape[2]:
        return None
    row = _packed_row(q, k, v)
    if row is None:
        return None
    arch = cake_backend.device_arch(_tensor_index(q))
    num_seqs = q.shape[0] if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    tail = fused_tail_stage(arch, q.shape[2], v.shape[2], total_tokens=q.shape[0] * q.shape[1], num_seqs=num_seqs,
                            dbeta_bf16=beta.dtype == torch.bfloat16, finalize=options.finalize)  # fmt: skip
    return row if tail in F32X_TAILS else None


def cake_gdn_det(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    use_qk_l2norm_in_kernel=False,
    *,
    scan_config: tuple[int, int] | None = None,
    segment_chunks: int | None = None,
    finalize: str | None = None,
    warmup_log2: float | None = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: str | None = None,
    prep: str | None = None,
    lpt: bool = True,
    streams: int = 2,
):
    """Deterministic chunked gated delta rule with FLA's signature (head dimension 128, ``HV`` a multiple of ``H``).

    ``q`` / ``k`` are ``[B, T, H, 128]`` and ``v`` ``[B, T, HV, 128]`` bfloat16; ``g`` is ``[B, T, HV]`` (computed in
    float32); ``beta`` is ``[B, T, HV]`` float32 or bfloat16 (its gradient comes back in its dtype).  Packed
    variable-length batches pass ``B = 1`` and ``cu_seqlens`` on the device; ``cu_seqlens_cpu`` (the same offsets on
    the host) selects the host-table route, without it the ``meta`` kernel builds the tables on the device.  Returns
    ``(o, final_state)``; ``final_state`` is ``None`` unless ``output_final_state``.  The keyword-only arguments are
    the plan pins of :class:`ChainOptions`.

    The inputs are checked before the first launch (``ValueError``): every tensor on q's CUDA device, ``k`` of q's
    shape, ``v`` / ``g`` / ``beta`` of q's tokens, ``HV`` a positive multiple of ``H``, ``initial_state`` ``[N, HV,
    128, 128]``, ``cu_seqlens`` 1-D int32 / int64 with a batch of 1, ``cu_seqlens_cpu`` as many offsets, from 0, not
    decreasing, to the tokens of ``q``.  The device route trusts the values of ``cu_seqlens`` (reading them would
    synchronize): the same contract, unchecked.

    When q, k and v are the head views of one packed BF16 row that requires grad (``row.split((H * 128, H * 128,
    HV * 128), -1)``, each viewed as ``[B, T, heads, 128]``; :func:`_packed_row`) and the backward tail is the
    round-once tail of a value-head group of three (:func:`fused_tail_stage`), the row is the autograd input
    (:class:`CakeGDNDetPackedFunction`): the backward writes dq / dk / dv into one packed gradient row, the same bits
    as on the views.
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise NotImplementedError("q/k/v must be bfloat16")
    _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
    options = make_options(scan_config=scan_config, segment_chunks=segment_chunks, finalize=finalize,
                           warmup_log2=warmup_log2, save_state=save_state, host_cache=host_cache,
                           dhu_scan=dhu_scan, prep=prep, lpt=lpt, streams=streams)  # fmt: skip
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    # a call that records no backward (grad mode off, or no input that requires grad) keeps no forward tape
    records = torch.is_grad_enabled() and (
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or g.requires_grad
        or beta.requires_grad
        or (initial_state is not None and initial_state.requires_grad)
    )
    if records:
        # the head views of one packed row under the round-once tail: the row is the autograd input
        row = _packed_route(q, k, v, beta, cu_seqlens, options)
        if row is not None:
            return CakeGDNDetPackedFunction.apply(row, g, beta, float(scale), initial_state, output_final_state,
                                                  cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, options,
                                                  (q, k, v))  # fmt: skip
    token = None if records else _RECORDS_BACKWARD.set(False)
    try:
        return CakeGDNDetFunction.apply(q, k, v, g, beta, float(scale), initial_state, output_final_state,
                                        cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip
    finally:
        if token is not None:
            _RECORDS_BACKWARD.reset(token)
