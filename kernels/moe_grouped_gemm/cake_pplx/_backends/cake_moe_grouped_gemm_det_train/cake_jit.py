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
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from . import cake_build

# Registry of the generated ragged grouped GEMM programs.  ``PROGRAMS`` and
# ``ROUTES`` are written by the Cake generated-program export; the host plan
# constants and the functions below are this package's.
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by every architecture in ``arches`` (genuine
# per-architecture lowering differences live inside the source under exact
# ``__CUDA_ARCH__`` regions) -- with its architecture-free ``library`` stem, FFI
# entry, argument plan and launch shape (a lead twin's record also names, in
# ``twin_of``, the program it twins; below).  What to compile (the translation
# units, the compile flags, the per-architecture closure digest) is recorded in
# ``build.json`` (see ``cake_build``), whose entry for a program is
# ``<library>[<specialization tag>]_<arch>``; a program is built once per
# (architecture, compile-line specialization), prebuilt under ``aot/`` or by
# the package's JIT shim.
#
# ``ROUTES``: one record per route -- the public routes ``fwd``, ``dgrad`` and
# ``wgrad_{bf16,f32}_k{256,512}`` and the lead twin of each (below) -- naming
# its architectures and ordered stages.  Each stage binds a program and the
# compile-line specializations the loader defines for it (the weight-gradient tail-reduce
# program takes the k width of the partials it sums, ``WGRAD_TILE``, from the
# route's tile instead of carrying one source per tile; the weight-gradient
# programs take their output element width, ``WGRAD_OUT`` = 16 for bf16 or 32
# for fp32, from the route's output dtype instead of carrying one source per
# output dtype).  A stage serves the architectures its program is built for
# among the route's: a route may name two stages of one name whose programs
# serve disjoint architectures (the forward and data-gradient ``main`` and
# ``main_wide`` stages: the domain-scheduled program on sm_100a / sm_103a, the
# static program on sm_107a, whose data-gradient ``main_wide`` program keeps
# each G tile in the tensor core's A collector for both 256-column halves of
# the output tile; the 512-wide k tile's weight-gradient ``main`` stage: one
# build on sm_100a / sm_103a, another on sm_107a), so a stage is looked up per
# architecture (``stage_binding``).  The host chooses among the stages a route
# serves by ``HOST_PLAN_CONSTANTS``: the forward and data-gradient
# ``main_wide`` stage is the program with the 512-column output tile
# (``wide_block_n``), launched on the architectures of ``wide_fwd_archs`` /
# ``wide_dgrad_archs`` when the output columns are a multiple of it; the
# data-gradient ``main_wide_ef`` stage is its twin that loads G and stores dX
# with the L2 evict-first policy, so the group's weights stay resident in L2,
# launched in its place on the architectures of ``dgrad_evict_first_archs``
# when the groups average at least the architecture's
# ``dgrad_evict_first_min_rows`` rows and the output has at least
# ``dgrad_evict_first_min_k`` columns; the forward ``main_wide_ef`` stage is
# its twin that stores Y with the L2 evict-first policy, launched in its place
# on the architectures of ``fwd_lwh_ef_archs`` (the A-collector program that
# also releases the last wave of tiles in halves) when the reduction K is at
# most ``fwd_lwh_ef_max_k`` (the down projection), and on those of
# ``fwd_evict_first_archs`` (the domain-scheduled program that also loads X
# with that policy) when the reduction K is at least ``fwd_evict_first_min_k``
# (the gate/up projection); the sm_107a ``main_wide_ef`` programs split two
# head and two tail k steps of an item around the release of the
# accumulator's halves instead of four and three; the data-gradient
# ``main_wide_ts33`` stage is the twin of ``main_wide`` that splits three head
# and three tail k steps around that release, launched in its place on the
# architectures of ``dgrad_wide_ts33_archs`` when the groups average at least
# the architecture's ``dgrad_wide_ts33_min_rows`` rows and the output has at
# least ``dgrad_wide_ts33_min_k`` columns, and the forward ``main_wide_ts33``
# stage on the architectures of ``fwd_wide_ts33_archs`` when the reduction K is
# at most ``fwd_wide_ts33_max_k`` with at most ``fwd_wide_ts33_max_rows`` rows
# per group; the weight gradient
# takes the 512-wide k tile on the architectures of ``k512_archs`` when the
# groups average at least the architecture's ``k512_min_rows_per_group`` rows;
# the weight-gradient ``main_wave`` stage is the main program with wave
# synchronisation, launched from ``wave_sync_min_k`` on the architectures
# ``wave_sync_archs`` lists for the k tile, with the architecture's
# ``wave_sync_mode``; the 512-wide k tile's ``main_halves`` and
# ``main_wave_halves`` stages are the twins of ``main`` and ``main_wave`` that
# release the accumulator in two halves, launched on the architectures of
# ``k512_release_archs``; its ``main_halves_k256t`` stage is the twin of
# ``main_halves`` that runs the tiles left after the full waves as 256-column
# half tiles, whose split tail the tail-reduce stage of the 256-wide k tile's
# route sums, launched in its place on the architectures of
# ``k256_tail_archs`` when K is at most ``k256_tail_max_k`` and the groups
# average at least ``k256_tail_min_rows`` rows; on sm_100a and sm_103a its
# ``main_halves_queue`` stage (and, on the fp32 route,
# ``main_halves_queue_acc``) is the wave-free A-collector half-release build
# whose tiles before the split tail are claimed from one global counter in tile
# order instead of strided over the clusters, with the split tail and tail
# reduce of ``main_halves``: each tile keeps its k-step order, so it gives the
# bits of the strided walk.  It takes one more workspace, ``sched``: two u32
# words, the counter and the retired-cluster count, at zero before and after a
# launch (the last cluster to retire sets them back to zero).  The host binds
# it in place of ``main_halves`` / ``main_wave_halves`` (and their accumulate
# stages) where ``queue_walk`` names the call, with ``queue_sched_words``
# sizing its counters.  On the architectures of
# ``domain_split_archs`` a launch of an op in ``domain_split_ops`` (the data
# gradient) whose weight has at most ``domain_split_max_weight_elems`` elements
# per group (``N * K``, the down projection) splits its tile queues by the
# probed SM pairs of each locality domain instead of evenly (``cake_backend``,
# stream workspaces).  On the architectures of ``wide_rows_archs`` a forward
# or data gradient whose output span passes ``int32_elems`` elements (the
# programs' 32-bit element addressing of C) launches its stage once per row
# chunk that fits (``cake_backend``, wide outputs).
#
# The lead routes: beside each public route ``ROUTES`` holds its lead twin
# ``<route>_lead`` (``LEAD_SUFFIX``, ``lead_route``; not a public route) with
# the route's op, architectures, stage names and specializations, whose every
# stage binds the ``offs_lead`` twin of the program the route binds there
# (``lead_twin``: the ``PROGRAMS`` record ``<program>_lead`` whose ``twin_of``
# names the program).  A twin takes its program's argument
# plan with the ``m_begin`` buffer after ``offs``: the first entry of an
# ``m_indptr[E + 1]`` row table, read on the device, so that group 0 starts at
# row ``m_indptr[0]`` instead of row 0; on the rows ``[m_begin, sum_m)`` it
# gives the bits of its program launched over ``offs - m_begin`` with the same
# grid and scalars.  The host binds the lead route for an ``m_indptr`` row
# table and the route itself for ``[E]`` end offsets; it selects the stages and
# plans the launch on the route either way (``cake_backend``, row tables).
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_det_gemm_09a336207c15c131f5dc": {
        "library": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["parameter", "lda"],
            ["parameter", "ldb"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_09a336207c15c131f5dc_lead": {
        "library": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["parameter", "lda"],
            ["parameter", "ldb"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc",
    },
    "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08": {
        "library": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08_lead": {
        "library": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08",
    },
    "cake_moe_grouped_det_gemm_1edcad995dcc790600cb": {
        "library": "cake_moe_grouped_det_gemm_1edcad995dcc790600cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_1edcad995dcc790600cb_lead": {
        "library": "cake_moe_grouped_det_gemm_1edcad995dcc790600cb_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_1edcad995dcc790600cb",
    },
    "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb": {
        "library": "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb_lead": {
        "library": "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb",
    },
    "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3": {
        "library": "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3_lead": {
        "library": "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3",
    },
    "cake_moe_grouped_det_gemm_3193de3faf82702b98c4": {
        "library": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_3193de3faf82702b98c4_lead": {
        "library": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4",
    },
    "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7": {
        "library": "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7_lead": {
        "library": "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7",
    },
    "cake_moe_grouped_det_gemm_412177509e084ace65e7": {
        "library": "cake_moe_grouped_det_gemm_412177509e084ace65e7",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_412177509e084ace65e7_lead": {
        "library": "cake_moe_grouped_det_gemm_412177509e084ace65e7_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_412177509e084ace65e7",
    },
    "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd": {
        "library": "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd_lead": {
        "library": "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd",
    },
    "cake_moe_grouped_det_gemm_5064b19de3716266ed03": {
        "library": "cake_moe_grouped_det_gemm_5064b19de3716266ed03",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_5064b19de3716266ed03_lead": {
        "library": "cake_moe_grouped_det_gemm_5064b19de3716266ed03_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_5064b19de3716266ed03",
    },
    "cake_moe_grouped_det_gemm_55833be10c8515562245": {
        "library": "cake_moe_grouped_det_gemm_55833be10c8515562245",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_55833be10c8515562245_lead": {
        "library": "cake_moe_grouped_det_gemm_55833be10c8515562245_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_55833be10c8515562245",
    },
    "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0": {
        "library": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0_lead": {
        "library": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0",
    },
    "cake_moe_grouped_det_gemm_5c7372a761f608717143": {
        "library": "cake_moe_grouped_det_gemm_5c7372a761f608717143",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_5c7372a761f608717143_lead": {
        "library": "cake_moe_grouped_det_gemm_5c7372a761f608717143_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_5c7372a761f608717143",
    },
    "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c": {
        "library": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c_lead": {
        "library": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c",
    },
    "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b": {
        "library": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b_lead": {
        "library": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b",
    },
    "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1": {
        "library": "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1_lead": {
        "library": "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1",
    },
    "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df": {
        "library": "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df_lead": {
        "library": "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df",
    },
    "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e": {
        "library": "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e_lead": {
        "library": "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e",
    },
    "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9": {
        "library": "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9_lead": {
        "library": "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9",
    },
    "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f": {
        "library": "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a"],
    },
    "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f_lead": {
        "library": "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a"],
        "twin_of": "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f",
    },
    "cake_moe_grouped_det_gemm_994f65a456ae68a92e84": {
        "library": "cake_moe_grouped_det_gemm_994f65a456ae68a92e84",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_det_gemm_994f65a456ae68a92e84_lead": {
        "library": "cake_moe_grouped_det_gemm_994f65a456ae68a92e84_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_994f65a456ae68a92e84",
    },
    "cake_moe_grouped_det_gemm_9df3c426f0afac83124c": {
        "library": "cake_moe_grouped_det_gemm_9df3c426f0afac83124c",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_9df3c426f0afac83124c_lead": {
        "library": "cake_moe_grouped_det_gemm_9df3c426f0afac83124c_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_9df3c426f0afac83124c",
    },
    "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f": {
        "library": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f_lead": {
        "library": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f",
    },
    "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a": {
        "library": "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a_lead": {
        "library": "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a",
    },
    "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef": {
        "library": "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef_lead": {
        "library": "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef",
    },
    "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91": {
        "library": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91_lead": {
        "library": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91",
    },
    "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c": {
        "library": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead": {
        "library": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
    },
    "cake_moe_grouped_det_gemm_bcd90000cc752d22a392": {
        "library": "cake_moe_grouped_det_gemm_bcd90000cc752d22a392",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_bcd90000cc752d22a392_lead": {
        "library": "cake_moe_grouped_det_gemm_bcd90000cc752d22a392_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_bcd90000cc752d22a392",
    },
    "cake_moe_grouped_det_gemm_ce655879551e583539e2": {
        "library": "cake_moe_grouped_det_gemm_ce655879551e583539e2",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_det_gemm_ce655879551e583539e2_lead": {
        "library": "cake_moe_grouped_det_gemm_ce655879551e583539e2_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_ce655879551e583539e2",
    },
    "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d": {
        "library": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d_lead": {
        "library": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d",
    },
    "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2": {
        "library": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2_lead": {
        "library": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
        "twin_of": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2",
    },
    "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb": {
        "library": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb_lead": {
        "library": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb_lead",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "m_begin"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
        "twin_of": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb",
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_det_gemm_dgrad": {
        "op": "dgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_1edcad995dcc790600cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_5064b19de3716266ed03",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_det_gemm_994f65a456ae68a92e84",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_dgrad_lead": {
        "op": "dgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_7e39db00fc7c99693f3e_lead",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_1edcad995dcc790600cb_lead",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_6bd667b99e236d0c69df_lead",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_aba5ff762eb9355f540a_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_5064b19de3716266ed03_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_det_gemm_994f65a456ae68a92e84_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_932cb8a0d39a950ac28f_lead",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_fwd": {
        "op": "fwd",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_9df3c426f0afac83124c",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_55833be10c8515562245",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_det_gemm_ce655879551e583539e2",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_fwd_lead": {
        "op": "fwd",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_9df3c426f0afac83124c_lead",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_276cec1f1d2c8ba1e4cb_lead",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_55833be10c8515562245_lead",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_det_gemm_3edd5c53a1269a6095a7_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_4f40b5f07baf6e086fdd_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_det_gemm_ce655879551e583539e2_lead",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_det_gemm_27a75f32c7a0e39f50f3_lead",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_bf16_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_bf16_k256_lead": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d_lead",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc_lead",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb_lead",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_bf16_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91",
                "specializations": {},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f",
                "specializations": {},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c",
                "specializations": {},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_bf16_k512_lead": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b_lead",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0_lead",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08_lead",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2_lead",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91_lead",
                "specializations": {},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f_lead",
                "specializations": {},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c_lead",
                "specializations": {},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4_lead",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead",
                "specializations": {"WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_f32_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
            {
                "name": "tail_reduce_acc",
                "program": "cake_moe_grouped_det_gemm_5c7372a761f608717143",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_f32_k256_lead": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_ed9a74f36ba13a52183d_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_09a336207c15c131f5dc_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_fc0a5c2da4d4782f42eb_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
            {
                "name": "tail_reduce_acc",
                "program": "cake_moe_grouped_det_gemm_5c7372a761f608717143_lead",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_f32_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_acc",
                "program": "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_acc",
                "program": "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves_acc",
                "program": "cake_moe_grouped_det_gemm_bcd90000cc752d22a392",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t_acc",
                "program": "cake_moe_grouped_det_gemm_412177509e084ace65e7",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue_acc",
                "program": "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 512},
            },
            {
                "name": "tail_reduce_acc",
                "program": "cake_moe_grouped_det_gemm_5c7372a761f608717143",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_det_gemm_wgrad_f32_k512_lead": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_611af0ca12365b0e7c3b_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_det_gemm_5b71b8844ca822afa8c0_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_det_gemm_0b2f2facc57c8bb9de08_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_f95641c4d3138b139ce2_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_det_gemm_b0fc259b54d6672efb91_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_det_gemm_9f489f95d8e138f00f9f_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_det_gemm_5c9fb14b887cc4228d8c_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_det_gemm_3193de3faf82702b98c4_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_acc",
                "program": "cake_moe_grouped_det_gemm_afaf3525ef60281e3aef_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_acc",
                "program": "cake_moe_grouped_det_gemm_6b795cefd960a34a58a1_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves_acc",
                "program": "cake_moe_grouped_det_gemm_bcd90000cc752d22a392_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t_acc",
                "program": "cake_moe_grouped_det_gemm_412177509e084ace65e7_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue_acc",
                "program": "cake_moe_grouped_det_gemm_8ca0902085eec3278fe9_lead",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_det_gemm_bc774d802e4f0b3ac27c_lead",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 512},
            },
            {
                "name": "tail_reduce_acc",
                "program": "cake_moe_grouped_det_gemm_5c7372a761f608717143_lead",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 512},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
}
HOST_PLAN_CONSTANTS: dict[str, Any] = {
    "block_n": 256,
    "wide_block_n": 512,
    "wide_fwd_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "wide_dgrad_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "dgrad_evict_first_archs": ["sm_100a", "sm_107a"],
    "dgrad_evict_first_min_rows": {"sm_100a": 4096, "sm_103a": 16384, "sm_107a": 16384},
    "dgrad_evict_first_min_k": 4096,
    "fwd_lwh_ef_archs": ["sm_107a"],
    "fwd_lwh_ef_max_k": 2048,
    "fwd_evict_first_archs": ["sm_100a", "sm_103a"],
    "fwd_evict_first_min_k": 6144,
    "dgrad_wide_ts33_archs": ["sm_103a"],
    "dgrad_wide_ts33_min_rows": {"sm_103a": 8192},
    "dgrad_wide_ts33_min_k": 4096,
    "fwd_wide_ts33_archs": ["sm_103a"],
    "fwd_wide_ts33_max_k": 2048,
    "fwd_wide_ts33_max_rows": 4096,
    "block_k": 64,
    "cta_group": 2,
    "cluster_m": 256,
    "epi_chunk": 16,
    "tmap_slot_bytes": 128,
    "tmap_slots_per_cta": 18,
    "k512_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k512_release_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k256_tail_archs": ["sm_107a"],
    "k256_tail_max_k": 2048,
    "k256_tail_min_rows": 8192,
    "k512_min_rows_per_group": {"sm_100a": 3072, "sm_103a": 3072, "sm_107a": 4096},
    "wave_sync_min_k": 4096,
    "wave_sync_archs": {
        256: ["sm_100a", "sm_103a"],
        512: ["sm_100a", "sm_103a"],
    },
    "wave_sync_mode": {"sm_100a": 2, "sm_103a": 2},
    "wave_sync_words": 2,
    "sched_words": 4224,
    "sched_rest": [0, 2147483648, 0],
    "domain_split_archs": ["sm_100a", "sm_103a"],
    "domain_split_ops": ["dgrad"],
    "domain_split_max_weight_elems": 12582912,
    "wgrad_tail_split": 1,
    "tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    # Where a 512-wide k tile weight gradient binds the queue walk
    # (``cake_backend.wgrad_queue_walk``): the ``main_halves_queue`` stage, or
    # ``main_halves_queue_acc`` for a fused FP32 accumulate, in place of the
    # stage the host selects.  A call takes it when one rule names the device's
    # architecture and SM count, the selected stage, the kind of output
    # (``"bfloat16"`` / ``"float32"`` overwrite, ``"accumulate"`` for the
    # fused FP32 accumulate) and a range ``[min, max]`` holding the group count;
    # every other call keeps its stage.  The rules are the parts of the
    # non-deterministic package's rules where this package's queue walk was
    # timed against the stage it replaces on that part (148-SM sm_100a, 152-SM
    # sm_103a) and no row was confirmed slower across several GPUs, except one
    # disclosed row: the 8-group gate/up weight gradient with a composed
    # bfloat16 accumulate, lead route, eager, slower on one of three 148-SM
    # sm_100a GPUs and faster on the other two.  On the 148-SM sm_100a part the
    # ``main_halves`` stage keeps the queue walk only for the fused accumulate
    # (a bfloat16 row there was slower than its stage), and the
    # ``main_wave_halves`` stage keeps it up to 16 groups (a 32-group bfloat16
    # row there was slower).  A part with no rule here keeps its stage.
    "queue_walk": [
        {
            "arch": "sm_100a",
            "sm_counts": [148],
            "stage": "main_wave_halves",
            "outputs": ["bfloat16", "accumulate"],
            "groups": [4, 16],
        },
        {
            "arch": "sm_100a",
            "sm_counts": [148],
            "stage": "main_halves",
            "outputs": ["accumulate"],
            "groups": [8, 32],
        },
        {
            "arch": "sm_103a",
            "sm_counts": [152],
            "stage": "main_wave_halves",
            "outputs": ["bfloat16", "accumulate"],
            "groups": [4, 32],
        },
        {
            "arch": "sm_103a",
            "sm_counts": [152],
            "stage": "main_halves",
            "outputs": ["accumulate"],
            "groups": [8, 16],
        },
    ],
    # The u32 words of the queue walk's tile counters (``queue_sched``).
    "queue_sched_words": 2,
    # The architectures where a forward or data gradient whose output span passes ``int32_elems``
    # elements launches its stage once per row chunk that fits (``cake_backend``, wide outputs); on the
    # others such a call is refused.
    "wide_rows_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "int32_elems": 2147483647,
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")
# The name suffix of a route's lead twin (the lead routes, above).
LEAD_SUFFIX = "_lead"


def lead_route(name: str) -> str:
    """The lead twin of route ``name``: the route whose stages bind the ``offs_lead`` twins of its
    programs (``lead_twin``), which the host binds for an ``m_indptr[E + 1]`` row table."""
    if is_lead_route(name):
        raise ValueError(f"route {name!r} is a lead route; it has no lead twin")
    return name + LEAD_SUFFIX


def is_lead_route(name: str) -> bool:
    """True for the name of a lead twin route (``lead_route``)."""
    return name.endswith(LEAD_SUFFIX)


def lead_route_served(name: str, arch: str) -> bool:
    """True when this checkout registers the lead twin of route ``name`` for ``arch``."""
    route = ROUTES.get(lead_route(name))
    return route is not None and arch in route["arches"]


def lead_twin(program: str) -> str:
    """The ``offs_lead`` twin of ``program``: the ``PROGRAMS`` record ``<program>_lead``
    whose ``twin_of`` names ``program`` (``KeyError`` for a program without one)."""
    twin = program + LEAD_SUFFIX
    if PROGRAMS.get(twin, {}).get("twin_of") != program:
        raise KeyError(f"program {program!r} has no lead twin in PROGRAMS")
    return twin


def record_name(
    op: str, *, out_dtype: Optional[str] = None, tile_k: Optional[int] = None
) -> str:
    """Route key of ``op``; weight-gradient routes are further keyed by output dtype and k tile."""
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    base = f"cake_moe_grouped_det_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient routes are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return base


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
    lead: bool = False,
) -> bool:
    """True when this checkout registers a route of ``op`` for ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.  With ``lead=True`` a
    route counts only when its lead twin (``lead_route``) is registered for
    ``arch`` too: then both row-table forms, ``[E]`` and ``m_indptr[E + 1]``, run.
    """
    if arch not in ARCHES:
        return False
    if op != "wgrad":
        names = [record_name(op)]
    else:
        dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
        tiles = (tile_k,) if tile_k else (256, 512)
        names = [
            record_name(op, out_dtype=dtype, tile_k=tile)
            for dtype in dtypes
            for tile in tiles
        ]
    for name in names:
        route = ROUTES.get(name)
        if route is None or arch not in route["arches"]:
            continue
        if not lead or lead_route_served(name, arch):
            return True
    return False


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered route name serving ``op`` on ``arch`` or raise."""
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    name = record_name(op, out_dtype=out_dtype, tile_k=tile_k)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM route {name!r} is not registered for "
            f"{arch} in this checkout (the registry of "
            "cake_pplx._backends.cake_moe_grouped_gemm_det_train.cake_jit is filled by the "
            "generated-program export)"
        )
    if route["op"] != op:
        raise RuntimeError(
            f"registered route {name!r} is a {route['op']} route, bound to {op}"
        )
    return name


def stage_binding(name: str, stage: str, arch: str) -> dict[str, Any]:
    """The ``{"name", "program", "specializations"}`` record of ``stage`` of route ``name`` on ``arch``."""
    route = ROUTES[name]
    for item in route["stages"]:
        if item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]:
            return item
    raise KeyError(f"route {name!r} has no {stage!r} stage on {arch}")


def stage_served(name: str, stage: str, arch: str) -> bool:
    """True when route ``name`` has a ``stage`` stage whose program is built for ``arch``."""
    return any(
        item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]
        for item in ROUTES[name]["stages"]
    )


def _specialization_items(
    specializations: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(specializations.items()))


def specialization_tag(specializations: tuple[tuple[str, Any], ...]) -> str:
    """Library-name suffix of a compile-line specialization set: ``_wgrad_tile256`` for
    ``{"WGRAD_TILE": 256}``, empty for none.  ``build.json`` names a specialized build
    ``<library><tag>_<arch>`` and carries its ``-D<KEY>=<value>`` defines in that entry's ``nvcc_flags``."""
    return "".join(f"_{key.lower()}{value}" for key, value in specializations)


def library_name(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
) -> str:
    """``build.json`` entry of ``program`` built for ``arch`` under ``specializations``."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    tag = specialization_tag(_specialization_items(specializations or {}))
    return f"{record['library']}{tag}_{arch}"


@functools.cache
def _load_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    return cake_build.load_library(library_name(program, arch, dict(specializations)))


def load_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """Load (once per program, exact target and specialization) one physical program: the prebuilt
    library under ``aot/`` when the package ships it, else the JIT shim's build of its ``build.json``
    entry."""
    return _load_module(program, arch, _specialization_items(specializations or {}))
